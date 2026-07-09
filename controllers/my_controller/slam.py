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

_LOCK = mapping.LOCK


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
        self._last_closure_kf = -LOOP_CLOSURE_COOLDOWN_KEYFRAMES 
        self._last_attempt_kf = -LOOP_CLOSURE_ATTEMPT_INTERVAL    

        self._acc_trans = 0.0
        self._acc_rot = 0.0

    def predict(self, delta_trans, delta_rot):
        n = self.num_particles
        trans_noise_std = max(SLAM_ALPHA3 * abs(delta_trans) + SLAM_ALPHA4 * abs(delta_rot), 1e-9)
        rot_noise_std = max(SLAM_ALPHA1 * abs(delta_trans) + SLAM_ALPHA2 * abs(delta_rot), 1e-9)

        noisy_trans = delta_trans + np.random.normal(0.0, trans_noise_std, n)
        noisy_rot = delta_rot + np.random.normal(0.0, rot_noise_std, n)

        for particle, dtr, dro in zip(self.particles, noisy_trans, noisy_rot):
            particle.theta = wrap_angle(particle.theta + dro)
            particle.x += dtr * np.cos(particle.theta)
            particle.y += dtr * np.sin(particle.theta)

        self._acc_trans += abs(delta_trans)
        self._acc_rot += abs(delta_rot)

        self._update_estimated_pose()

    def observe(self, local_lidar_points):
        if local_lidar_points is None or len(local_lidar_points) == 0:
            return
        with _LOCK:
            if (self._acc_trans < SLAM_OBSERVE_MIN_TRANS_M
                    and self._acc_rot < SLAM_OBSERVE_MIN_ROT_RAD):
                return
            self._acc_trans = 0.0
            self._acc_rot = 0.0
            particles = list(self.particles)  

        scan = self._downsample_scan(local_lidar_points)

        dist_fields = [self._obstacle_distance_field(p) for p in particles]

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
            snap = [(p.x, p.y, p.theta, p.log_odds) for p in self.particles]

        for px, py, pth, log_odds in snap:
            world_points = self._transform_points(scan, px, py, pth)
            map_points = mapping.world_points_to_map(world_points)
            robot_map_pos = mapping.world_points_to_map(np.array([[px, py]]))[0]
            mapping.rasterize_scan(log_odds, robot_map_pos, map_points, self.map_size)

        with _LOCK:
            self._sync_canonical_map()

        self._update_pose_graph(scan)

    def _obstacle_distance_field(self, particle):
        obstacle_mask = particle.log_odds > LOGIT_OBSTACLE_THRESHOLD
        free_src = np.where(obstacle_mask, 0, 255).astype(np.uint8)
        return cv2.distanceTransform(free_src, cv2.DIST_L2, 3)

    def _refine_pose(self, particle, scan_local, dist_field):
        px_vals = np.arange(-SLAM_REFINE_RADIUS_PX, SLAM_REFINE_RADIUS_PX + 1) * self.resolution
        dth_vals = np.radians(np.arange(-SLAM_REFINE_WINDOW_DEG, SLAM_REFINE_WINDOW_DEG + 1e-9, SLAM_REFINE_STEP_DEG))

        dxx, dyy = np.meshgrid(px_vals, px_vals, indexing='ij')
        offsets = np.stack([dxx.ravel(), dyy.ravel()], axis=1) 
        num_offsets = offsets.shape[0]
        num_points = max(len(scan_local), 1)

        best_score = -float('inf')
        best_pose = (particle.x, particle.y, particle.theta)

        for dth in dth_vals:
            theta = particle.theta + dth
            world_points = self._transform_points(scan_local, particle.x, particle.y, theta)
            if len(world_points) == 0:
                continue

            candidate_points = world_points[None, :, :] + offsets[:, None, :] 
            flat = candidate_points.reshape(-1, 2)
            map_points = mapping.world_points_to_map(flat)
            xs, ys = map_points[:, 0], map_points[:, 1]
            valid = (xs >= 0) & (xs < self.map_size) & (ys >= 0) & (ys < self.map_size)

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

    def _update_pose_graph(self, scan_local):
        pose = self._estimated_pose.copy()
        new_index = self.pose_graph.add_keyframe(pose, scan_local)
        if new_index is None:
            return

        if new_index - self._last_closure_kf < LOOP_CLOSURE_COOLDOWN_KEYFRAMES:
            return

        if new_index - self._last_attempt_kf < LOOP_CLOSURE_ATTEMPT_INTERVAL:
            return
        self._last_attempt_kf = new_index

        if not self.pose_graph.try_loop_closure(new_index):
            return

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

    def estimated_pose(self):
        return self._estimated_pose.copy()

    def particle_positions_map(self):
        return [mapping.world_to_map(p.x, p.y) for p in self.particles]

    def _representative_particle(self):
        ex, ey = self._estimated_pose[0], self._estimated_pose[1]
        return min(self.particles, key=lambda p: (p.x - ex) ** 2 + (p.y - ey) ** 2)

    def _sync_canonical_map(self):
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

_system = None


def system():
    global _system
    if _system is None:
        _system = SlamSystem()
    return _system


def reset():
    global _system
    with _LOCK:
        _system = SlamSystem()


def predict(delta_trans, delta_rot):
    with _LOCK:
        system().predict(delta_trans, delta_rot)


def observe(local_lidar_points):
    system().observe(local_lidar_points)


def estimated_pose():
    return system().estimated_pose()


def particle_positions_map():
    return system().particle_positions_map()


_map_thread = None
_map_thread_running = False


def start_mapping_thread(read_cloud, is_turning, hz=SLAM_OBSERVE_HZ):
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
