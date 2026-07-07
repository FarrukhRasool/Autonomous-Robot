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
    SLAM_OBSERVE_HZ, LOOP_CLOSURE_COOLDOWN_KEYFRAMES,
    SLAM_OBSERVE_MIN_TRANS_M, SLAM_OBSERVE_MIN_ROT_RAD,
    LOOP_CLOSURE_ATTEMPT_INTERVAL,
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
        self._last_closure_kf = -LOOP_CLOSURE_COOLDOWN_KEYFRAMES  # allow the first closure freely
        self._last_attempt_kf = -LOOP_CLOSURE_ATTEMPT_INTERVAL     # throttle failed-attempt searches

        # Motion accumulated by predict() since the last scan folded by observe().
        # observe() only runs once these cross the SLAM_OBSERVE_MIN_* thresholds,
        # then resets them — so the filter updates on motion, not on the clock.
        self._acc_trans = 0.0
        self._acc_rot = 0.0

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

        # Bank the motion so observe() can gate on distance travelled, not a timer.
        self._acc_trans += abs(delta_trans)
        self._acc_rot += abs(delta_rot)

        self._update_estimated_pose()

    # ------------------------------------------------------------------
    # Measurement update (called from the mapping cadence, ~10 Hz)
    # ------------------------------------------------------------------
    def observe(self, local_lidar_points):
        """Refine each particle's pose against its own map (FastSLAM 2.0
        proposal), weight particles by the refined residual, resample if needed,
        fold the scan into every surviving particle's (refined) map, then refresh
        the estimated pose, the canonical map, and the loop-closure pose graph.

        Lock discipline (this is the ONLY writer of particle.log_odds, and runs on
        the single background mapping thread).  predict() runs every control tick
        and shares _LOCK with us, so it FREEZES the sim while we hold it.  Only the
        pose state (particle x/y/theta, weights, the particle list, _estimated_pose,
        _acc_*) is shared with predict(); particle.log_odds is not.  So the two
        heavy loops -- the per-particle distance fields and the per-particle
        rasterize -- run OFF-lock (they only read/write log_odds), and the lock is
        held only around the short pose sections.  This keeps predict() (hence
        robot.step()) from stalling on the 30-particle map grind.  Same algorithm,
        same work -- just not under the lock.
        """
        if local_lidar_points is None or len(local_lidar_points) == 0:
            return

        # Motion gate: skip if the robot has barely moved since the last update.
        # _acc_* are shared with predict(), so gate + reset under the lock.  The
        # accumulators keep summing across skipped cycles, so a slow crawl still
        # triggers an update once it adds up.
        with _LOCK:
            if (self._acc_trans < SLAM_OBSERVE_MIN_TRANS_M
                    and self._acc_rot < SLAM_OBSERVE_MIN_ROT_RAD):
                return
            self._acc_trans = 0.0
            self._acc_rot = 0.0
            particles = list(self.particles)   # stable object list for this cycle

        _t0 = time.perf_counter()   # [PROF] temporary stall diagnostic

        scan = self._downsample_scan(local_lidar_points)

        # OFF-lock: distance fields read only each particle's own map (never
        # touched by predict), and are pose-independent -- the heavy
        # cv2.distanceTransform work must not block the control loop.
        dist_fields = [self._obstacle_distance_field(p) for p in particles]
        _t_df = time.perf_counter()   # [PROF]

        # LOCKED: refine / reweight / resample all mutate pose state shared with
        # predict.  Short + vectorised.  No resample has run yet this cycle, so
        # self.particles is still `particles` (same objects) and aligns with
        # dist_fields by index.
        with _LOCK:
            log_weights = np.empty(self.num_particles)
            for i, particle in enumerate(self.particles):
                log_weights[i] = self._refine_pose(particle, scan, dist_fields[i])

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

            self._update_estimated_pose()
            # Snapshot the committed (refined/resampled) poses + map refs so the
            # off-lock rasterize uses exactly these poses even if predict() nudges
            # the live particles before it runs.
            snap = [(p.x, p.y, p.theta, p.log_odds) for p in self.particles]

        _t_ref = time.perf_counter()   # [PROF]

        # OFF-lock: fold the scan into each particle's own map.  Writes only
        # particle.log_odds (single-writer, not shared with predict).
        for px, py, pth, log_odds in snap:
            world_points = self._transform_points(scan, px, py, pth)
            map_points = mapping.world_points_to_map(world_points)
            robot_map_pos = mapping.world_points_to_map(np.array([[px, py]]))[0]
            mapping.rasterize_scan(log_odds, robot_map_pos, map_points, self.map_size)

        _t_rast = time.perf_counter()   # [PROF]

        # LOCKED (brief): publish the canonical map for the control loop.
        with _LOCK:
            self._sync_canonical_map()

        # OFF-lock: keyframe bookkeeping + loop-closure DETECTION only read the
        # pose graph (single-writer, not shared with predict), so the growing
        # candidate scan-match search must not hold the lock.  _update_pose_graph
        # re-acquires the lock itself only for the rare closure COMMIT (which
        # re-anchors the particle poses).
        self._update_pose_graph(scan)

        # [PROF] temporary stall diagnostic — remove once the stall is located.
        # Reports where observe() spends its GIL-holding wall-clock time.
        _t_end = time.perf_counter()
        _tot = (_t_end - _t0) * 1000.0
        if _tot > 8.0:
            print(f"[PROF] observe {_tot:5.0f}ms | "
                  f"distfield {(_t_df - _t0) * 1000:4.0f} "
                  f"refine {(_t_ref - _t_df) * 1000:4.0f} "
                  f"rasterize {(_t_rast - _t_ref) * 1000:4.0f} "
                  f"publish+graph {(_t_end - _t_rast) * 1000:4.0f}")

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
        # Runs OFF the shared lock: everything here reads/writes only the pose
        # graph (mutated solely by this single background thread, never by
        # predict), EXCEPT the closure commit at the end, which re-anchors the
        # particle poses and so takes the lock itself.
        pose = self._estimated_pose.copy()
        new_index = self.pose_graph.add_keyframe(pose, scan_local)
        if new_index is None:
            return

        # Cooldown: don't even ATTEMPT a closure until enough keyframes have
        # passed since the last SUCCESSFUL one.
        if new_index - self._last_closure_kf < LOOP_CLOSURE_COOLDOWN_KEYFRAMES:
            return

        # Attempt throttle: the candidate scan-match search runs even when it
        # FAILS, and its cost grows with keyframe count -> only search every
        # LOOP_CLOSURE_ATTEMPT_INTERVAL keyframes so it can't dominate the run as
        # the map fills.  (A real revisit is still caught within the interval.)
        if new_index - self._last_attempt_kf < LOOP_CLOSURE_ATTEMPT_INTERVAL:
            return
        self._last_attempt_kf = new_index

        if not self.pose_graph.try_loop_closure(new_index):
            return

        # Closure found: optimize + rebuild the map OFF-lock (both only read the
        # pose graph / build a fresh array), then COMMIT under the lock.
        self._last_closure_kf = new_index
        old_last_pose = self.pose_graph.nodes[new_index].copy()
        corrected_poses = self.pose_graph.optimize()
        new_last_pose = corrected_poses[new_index]
        rebuilt_log_odds = self.pose_graph.rebuild_log_odds(
            corrected_poses, self.map_size, self.resolution,
            mapping.world_points_to_map,
        )

        with _LOCK:
            self.pose_graph.nodes = [p.copy() for p in corrected_poses]
            particle_poses = np.array([[p.x, p.y, p.theta] for p in self.particles])
            corrected_particle_poses = apply_rigid_correction(
                particle_poses, old_last_pose, new_last_pose
            )
            for particle, corrected in zip(self.particles, corrected_particle_poses):
                particle.x, particle.y, particle.theta = corrected
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

    def _representative_particle(self):
        """The particle nearest the (weighted-mean) estimated pose — a stable,
        representative choice for publishing the canonical map.

        Publishing the raw argmax-weight particle makes the map flicker:
        FastSLAM 2.0's improved proposal refines every particle onto its own map,
        driving weights to near-equal, so argmax flip-flops between spatially
        different particles frame to frame.  The particle closest to the mean
        pose sits in the centre of the cloud and shifts smoothly, and it keeps
        the published map consistent with the pose get_pose() reports (also the
        weighted mean) instead of a different particle's pose.
        """
        ex, ey = self._estimated_pose[0], self._estimated_pose[1]
        return min(self.particles, key=lambda p: (p.x - ex) ** 2 + (p.y - ey) ** 2)

    def _sync_canonical_map(self):
        # Called right after _update_estimated_pose(), so _estimated_pose is fresh.
        mapping.sync_from_log_odds(self._representative_particle().log_odds)

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
    """Measurement update (background mapping thread).  SlamSystem.observe()
    manages _LOCK itself with a narrow scope -- it holds the lock only around the
    short pose sections and runs the heavy per-particle map work off-lock, so
    predict() (every control tick) never stalls on the full update.  Do NOT wrap
    the whole call in _LOCK here, or that decoupling is lost."""
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
