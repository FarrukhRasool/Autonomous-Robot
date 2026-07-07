"""FastSLAM-2.0-style particle filter: each particle carries its own pose AND
its own occupancy log-odds map.  Before a scan is scored/integrated, each
particle's pose is first refined with a small local correlative search against
its OWN map (the "improved proposal distribution" from FastSLAM 2.0) -- this is
what keeps each particle's own map crisp.  The refined pose's residual
likelihood is used directly as the particle's importance weight, particles are
resampled when needed, and the scan is rasterized into every surviving
particle's own (now-corrected) map.  A PoseGraphSLAM backend tracks keyframes
and detects loop closures, correcting accumulated drift.

Faithful port of the reference project's slam.py (Hieu Tran et al.).  The only
change is decoupling from their `MyRobot`: `self.robot.map_object.*` calls are
rewired to this project's `mapping` module (M2 primitives) and `pose_graph`
(M1) -- no algorithm change.  A module-level singleton wrapper (init/predict/
observe/estimated_pose/reset/particle_positions_map) exposes the system the way
this project's other modules are consumed.

Not wired into the control loop yet (that is a later milestone).  Pure
NumPy / OpenCV / SciPy(via pose_graph) -- no Webots imports.
"""
import threading
import time

import numpy as np
import cv2

from config import (
    MAP_SIZE, MAP_RES_M as RESOLUTION, LOGODDS_INIT as INITIAL_LOG_ODD,
    SLAM_NUM_PARTICLES, SLAM_ALPHA1, SLAM_ALPHA2, SLAM_ALPHA3, SLAM_ALPHA4,
    SLAM_SCAN_MAX_BEAMS, SLAM_RESAMPLE_NEFF_RATIO, SLAM_LIKELIHOOD_SIGMA_M,
    SLAM_REFINE_RADIUS_PX, SLAM_REFINE_WINDOW_DEG, SLAM_REFINE_STEP_DEG,
    SLAM_OBSERVE_HZ,
)
import mapping
from pose_graph import PoseGraphSLAM, wrap_angle, apply_rigid_correction

# The shared writer lock lives in mapping (mapping.LOCK, reentrant).  predict()
# (main thread) and observe() (background mapping thread) both acquire it so they
# never mutate the particle set / map concurrently.  observe() re-acquires it
# transitively via mapping.sync_from_log_odds (hence RLock).
_LOCK = mapping.LOCK

# Matches the P > 0.7 obstacle threshold used by mapping's grid thresholding, in log-odds.
LOGIT_OBSTACLE_THRESHOLD = float(np.log(0.7 / 0.3))


class Particle:
    __slots__ = ('x', 'y', 'theta', 'weight', 'log_odds')

    def __init__(self, x, y, theta, weight, log_odds):
        self.x = x
        self.y = y
        self.theta = theta
        self.weight = weight
        self.log_odds = log_odds


class SlamSystem:
    """FastSLAM 2.0 particle filter (scan-matched proposal) + pose-graph loop closure."""

    def __init__(self, num_particles=SLAM_NUM_PARTICLES, map_size=MAP_SIZE, resolution=RESOLUTION):
        self.map_size = map_size
        self.resolution = resolution
        self.num_particles = num_particles

        initial_log_odds = np.full((map_size, map_size), INITIAL_LOG_ODD, dtype=np.float32)
        self.particles = [
            Particle(0.0, 0.0, 0.0, 1.0 / num_particles, initial_log_odds.copy())
            for _ in range(num_particles)
        ]

        self.pose_graph = PoseGraphSLAM()
        self._estimated_pose = np.array([0.0, 0.0, 0.0])

    # ------------------------------------------------------------------
    # Motion update (called every simulation step)
    # ------------------------------------------------------------------
    def predict(self, delta_trans, delta_rot):
        """Sample odometry motion model.  `delta_trans`/`delta_rot` are the
        already slip-gated per-step deltas computed by the odometry layer
        (zero translation while wheel slip is detected).
        """
        n = self.num_particles
        trans_noise_std = max(SLAM_ALPHA3 * abs(delta_trans) + SLAM_ALPHA4 * abs(delta_rot), 1e-9)
        rot_noise_std = max(SLAM_ALPHA1 * abs(delta_trans) + SLAM_ALPHA2 * abs(delta_rot), 1e-9)

        noisy_trans = delta_trans + np.random.normal(0.0, trans_noise_std, n)
        noisy_rot = delta_rot + np.random.normal(0.0, rot_noise_std, n)

        for particle, dtr, dro in zip(self.particles, noisy_trans, noisy_rot):
            particle.theta = wrap_angle(particle.theta + dro)
            particle.x += dtr * np.cos(particle.theta)
            particle.y += dtr * np.sin(particle.theta)

        self._update_estimated_pose()

    # ------------------------------------------------------------------
    # Measurement update (called from the mapping cadence, ~10 Hz)
    # ------------------------------------------------------------------
    def observe(self, local_lidar_points):
        """Refine each particle's pose against its own map (FastSLAM 2.0
        proposal), weight particles by the refined residual, resample if needed,
        fold the scan into every surviving particle's (refined) map, then refresh
        the estimated pose, the canonical map, and the loop-closure pose graph.
        """
        if local_lidar_points is None or len(local_lidar_points) == 0:
            return

        scan = self._downsample_scan(local_lidar_points)

        log_weights = np.empty(self.num_particles)
        for i, particle in enumerate(self.particles):
            dist_field = self._obstacle_distance_field(particle)
            log_weights[i] = self._refine_pose(particle, scan, dist_field)

        log_weights -= log_weights.max()
        weights = np.exp(log_weights)
        weights_sum = weights.sum()
        if weights_sum <= 0 or not np.isfinite(weights_sum):
            weights = np.full(self.num_particles, 1.0 / self.num_particles)
        else:
            weights = weights / weights_sum

        for particle, w in zip(self.particles, weights):
            particle.weight = float(w)

        neff = 1.0 / np.sum(weights ** 2)
        if neff < SLAM_RESAMPLE_NEFF_RATIO * self.num_particles:
            self._resample(weights)

        for particle in self.particles:
            world_points = self._transform_points(scan, particle.x, particle.y, particle.theta)
            map_points = mapping.world_points_to_map(world_points)
            robot_map_pos = mapping.world_points_to_map(
                np.array([[particle.x, particle.y]])
            )[0]
            mapping.rasterize_scan(particle.log_odds, robot_map_pos, map_points, self.map_size)

        self._update_estimated_pose()
        self._sync_canonical_map()
        self._update_pose_graph(scan)

    def _obstacle_distance_field(self, particle):
        """Distance (in pixels) from every cell to the nearest obstacle cell in
        `particle`'s own map -- the likelihood-field lookup table, shared by both
        the pose-refinement search and the (implicit) importance weight.
        """
        obstacle_mask = particle.log_odds > LOGIT_OBSTACLE_THRESHOLD
        free_src = np.where(obstacle_mask, 0, 255).astype(np.uint8)
        return cv2.distanceTransform(free_src, cv2.DIST_L2, 3)

    def _refine_pose(self, particle, scan_local, dist_field):
        """Snap `particle`'s pose onto its own map with a small local correlative
        search (likelihood-field / beam-endpoint model) before this scan is
        scored/rasterized.

        Mutates particle.x/y/theta in place.  Returns the best log-likelihood
        found, used directly as the particle's (unnormalized) importance weight.
        """
        px_vals = np.arange(-SLAM_REFINE_RADIUS_PX, SLAM_REFINE_RADIUS_PX + 1) * self.resolution
        dth_vals = np.radians(np.arange(-SLAM_REFINE_WINDOW_DEG, SLAM_REFINE_WINDOW_DEG + 1e-9, SLAM_REFINE_STEP_DEG))

        dxx, dyy = np.meshgrid(px_vals, px_vals, indexing='ij')
        offsets = np.stack([dxx.ravel(), dyy.ravel()], axis=1)  # (num_offsets, 2)
        num_offsets = offsets.shape[0]
        num_points = max(len(scan_local), 1)

        best_score = -float('inf')
        best_pose = (particle.x, particle.y, particle.theta)

        for dth in dth_vals:
            theta = particle.theta + dth
            world_points = self._transform_points(scan_local, particle.x, particle.y, theta)
            if len(world_points) == 0:
                continue

            candidate_points = world_points[None, :, :] + offsets[:, None, :]  # (num_offsets, num_points, 2)
            flat = candidate_points.reshape(-1, 2)
            map_points = mapping.world_points_to_map(flat)
            xs, ys = map_points[:, 0], map_points[:, 1]
            valid = (xs >= 0) & (xs < self.map_size) & (ys >= 0) & (ys < self.map_size)

            # Out-of-bounds beams get a large fixed distance penalty rather than
            # being dropped, so every candidate is scored on equal footing.
            dist_m = np.full(flat.shape[0], 5.0, dtype=np.float64)
            dist_m[valid] = dist_field[ys[valid], xs[valid]] * self.resolution

            log_lik = -0.5 * (dist_m / SLAM_LIKELIHOOD_SIGMA_M) ** 2
            scores = log_lik.reshape(num_offsets, num_points).sum(axis=1)

            max_idx = int(np.argmax(scores))
            if scores[max_idx] > best_score:
                best_score = float(scores[max_idx])
                best_pose = (
                    particle.x + float(offsets[max_idx, 0]),
                    particle.y + float(offsets[max_idx, 1]),
                    theta,
                )

        particle.x, particle.y, particle.theta = best_pose
        return best_score

    def _resample(self, weights):
        """Systematic resampling; deep-copies surviving particles' maps."""
        n = self.num_particles
        positions = (np.arange(n) + np.random.uniform()) / n
        cumulative = np.cumsum(weights)
        cumulative[-1] = 1.0
        indices = np.searchsorted(cumulative, positions)

        self.particles = [
            Particle(self.particles[i].x, self.particles[i].y, self.particles[i].theta,
                     1.0 / n, self.particles[i].log_odds.copy())
            for i in indices
        ]

    # ------------------------------------------------------------------
    # Loop closure
    # ------------------------------------------------------------------
    def _update_pose_graph(self, scan_local):
        pose = self._estimated_pose
        new_index = self.pose_graph.add_keyframe(pose, scan_local)
        if new_index is None:
            return

        if not self.pose_graph.try_loop_closure(new_index):
            return

        old_last_pose = self.pose_graph.nodes[new_index].copy()
        corrected_poses = self.pose_graph.optimize()
        new_last_pose = corrected_poses[new_index]
        self.pose_graph.nodes = [p.copy() for p in corrected_poses]

        particle_poses = np.array([[p.x, p.y, p.theta] for p in self.particles])
        corrected_particle_poses = apply_rigid_correction(particle_poses, old_last_pose, new_last_pose)
        for particle, corrected in zip(self.particles, corrected_particle_poses):
            particle.x, particle.y, particle.theta = corrected

        rebuilt_log_odds = self.pose_graph.rebuild_log_odds(
            corrected_poses, self.map_size, self.resolution,
            mapping.world_points_to_map,
        )
        for particle in self.particles:
            particle.log_odds = rebuilt_log_odds.copy()

        self._update_estimated_pose()
        self._sync_canonical_map()
        print(f"[SLAM] Loop closure applied: map rebuilt from {len(corrected_poses)} keyframes")

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def estimated_pose(self):
        return self._estimated_pose.copy()

    def particle_positions_map(self):
        """(map_x, map_y) for every particle, for the debug visualizer."""
        return [mapping.world_to_map(p.x, p.y) for p in self.particles]

    def _best_particle(self):
        return max(self.particles, key=lambda p: p.weight)

    def _sync_canonical_map(self):
        mapping.sync_from_log_odds(self._best_particle().log_odds)

    def _update_estimated_pose(self):
        weights = np.array([p.weight for p in self.particles])
        if weights.sum() <= 0:
            weights = np.full(self.num_particles, 1.0 / self.num_particles)
        weights = weights / weights.sum()

        xs = np.array([p.x for p in self.particles])
        ys = np.array([p.y for p in self.particles])
        thetas = np.array([p.theta for p in self.particles])

        mean_x = float(np.sum(weights * xs))
        mean_y = float(np.sum(weights * ys))
        mean_theta = float(np.arctan2(np.sum(weights * np.sin(thetas)), np.sum(weights * np.cos(thetas))))
        self._estimated_pose = np.array([mean_x, mean_y, mean_theta])

    @staticmethod
    def _downsample_scan(points):
        points = np.asarray(points)
        if len(points) <= SLAM_SCAN_MAX_BEAMS:
            return points
        idx = np.linspace(0, len(points) - 1, SLAM_SCAN_MAX_BEAMS).astype(int)
        return points[idx]

    @staticmethod
    def _transform_points(points_local, x, y, theta):
        if len(points_local) == 0:
            return np.asarray(points_local).reshape(0, 2)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        return np.asarray(points_local) @ R.T + np.array([x, y])


# ── Module-level singleton (the interface the control loop will consume) ──────
_system = None


def system():
    """Return the singleton SlamSystem, creating it on first use."""
    global _system
    if _system is None:
        _system = SlamSystem()
    return _system


def reset():
    """Discard all particles/maps and start a fresh SLAM system (called on pose reset)."""
    global _system
    with _LOCK:
        _system = SlamSystem()


def predict(delta_trans, delta_rot):
    """Motion update (main thread).  Locked against the background observe()."""
    with _LOCK:
        system().predict(delta_trans, delta_rot)


def observe(local_lidar_points):
    """Measurement update (background mapping thread, or a forced main-thread
    refresh).  Locked against predict() and other map writers."""
    with _LOCK:
        system().observe(local_lidar_points)


def estimated_pose():
    # Lockless: _estimated_pose is replaced by whole-array rebind, so a reader
    # always gets a complete (possibly one-cycle-stale) pose.  Keeps get_pose fast.
    return system().estimated_pose()


def particle_positions_map():
    return system().particle_positions_map()


# ── Background mapping thread (ports the reference lidar_update_loop) ──────────
_map_thread = None
_map_thread_running = False


def start_mapping_thread(read_cloud, is_turning, hz=SLAM_OBSERVE_HZ):
    """Start the continuous background SLAM measurement thread.

    read_cloud() -> Nx2 body-frame lidar points (this project's
    sensors.read_lidar_pointcloud_2d); is_turning() -> bool (motion.is_turning).
    Mirrors the reference's daemon lidar thread: run observe() at ~hz, only when
    not turning (rotation smears the per-scan match).  Runs continuously for the
    whole controller lifetime so both teleop and the blocking explore loop map
    without stalling — the GIL is released during robot.step() waits and during
    numpy/cv2 work, so observe overlaps the control loop instead of blocking it.
    """
    global _map_thread, _map_thread_running
    if _map_thread is not None and _map_thread.is_alive():
        return
    _map_thread_running = True
    period = 1.0 / max(1.0, hz)

    def _loop():
        while _map_thread_running:
            try:
                if not is_turning():
                    cloud = read_cloud()
                    if cloud is not None and len(cloud) > 0:
                        observe(cloud)
            except Exception as e:
                print(f"[SLAM] mapping thread error: {e}")
            time.sleep(period)

    _map_thread = threading.Thread(target=_loop, daemon=True)
    _map_thread.start()
    print(f"[SLAM] Started background mapping thread (~{hz:.0f} Hz)")


def stop_mapping_thread():
    global _map_thread_running
    _map_thread_running = False
