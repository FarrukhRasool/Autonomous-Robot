import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from config import (
    LOGODDS_INIT as INITIAL_LOG_ODD,
    KEYFRAME_DIST_THRESHOLD_M, KEYFRAME_ANGLE_THRESHOLD_DEG,
    LOOP_CLOSURE_SEARCH_RADIUS_M, LOOP_CLOSURE_MIN_KEYFRAME_GAP,
    LOOP_CLOSURE_SCORE_THRESHOLD, LOOP_CLOSURE_SEARCH_WINDOW_M,
    LOOP_CLOSURE_SEARCH_STEP_M, LOOP_CLOSURE_SEARCH_WINDOW_DEG,
    LOOP_CLOSURE_SEARCH_STEP_DEG,
)


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def relative_pose(pose_a, pose_b):
    ax, ay, ath = pose_a
    bx, by, bth = pose_b
    dx, dy = bx - ax, by - ay
    c, s = np.cos(-ath), np.sin(-ath)
    local_dx = c * dx - s * dy
    local_dy = s * dx + c * dy
    local_dth = wrap_angle(bth - ath)
    return np.array([local_dx, local_dy, local_dth])


def transform_points(points, pose):
    if len(points) == 0:
        return np.asarray(points).reshape(0, 2)
    x, y, theta = pose
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return np.asarray(points) @ R.T + np.array([x, y])


def apply_rigid_correction(poses, old_anchor, new_anchor):
    poses = np.asarray(poses, dtype=np.float64)
    dtheta = wrap_angle(new_anchor[2] - old_anchor[2])
    c, s = np.cos(dtheta), np.sin(dtheta)
    R = np.array([[c, -s], [s, c]])
    rel_xy = poses[:, :2] - old_anchor[:2]
    rotated_xy = rel_xy @ R.T
    corrected = np.empty_like(poses)
    corrected[:, :2] = new_anchor[:2] + rotated_xy
    corrected[:, 2] = wrap_angle(poses[:, 2] + dtheta)
    return corrected


def _correlative_match(current_scan_local, cand_scan_local, init_rel):
    tree = cKDTree(cand_scan_local)

    dx_vals = np.arange(-LOOP_CLOSURE_SEARCH_WINDOW_M, LOOP_CLOSURE_SEARCH_WINDOW_M + 1e-9, LOOP_CLOSURE_SEARCH_STEP_M)
    dy_vals = np.arange(-LOOP_CLOSURE_SEARCH_WINDOW_M, LOOP_CLOSURE_SEARCH_WINDOW_M + 1e-9, LOOP_CLOSURE_SEARCH_STEP_M)
    dth_vals = np.radians(np.arange(-LOOP_CLOSURE_SEARCH_WINDOW_DEG, LOOP_CLOSURE_SEARCH_WINDOW_DEG + 1e-9, LOOP_CLOSURE_SEARCH_STEP_DEG))

    dxx, dyy = np.meshgrid(dx_vals, dy_vals, indexing='ij')
    offsets = np.stack([dxx.ravel(), dyy.ravel()], axis=1)  # (M, 2)
    num_offsets = offsets.shape[0]
    base_xy = offsets + init_rel[:2]

    best_score = float('inf')
    best_rel = init_rel

    for dth in dth_vals:
        theta = init_rel[2] + dth
        c, s = np.cos(theta), np.sin(theta)
        rotated = current_scan_local @ np.array([[c, -s], [s, c]]).T 

        translated = rotated[None, :, :] + base_xy[:, None, :]
        flat = translated.reshape(-1, 2)
        dists, _ = tree.query(flat, k=1)
        scores = dists.reshape(num_offsets, -1).mean(axis=1)

        min_idx = int(np.argmin(scores))
        if scores[min_idx] < best_score:
            best_score = float(scores[min_idx])
            best_rel = np.array([base_xy[min_idx, 0], base_xy[min_idx, 1], theta])

    return best_score, best_rel


class PoseGraphSLAM:

    def __init__(self):
        self.nodes = []  
        self.scans = []  
        self.edges = [] 
        self.loop_closure_count = 0

    def add_keyframe(self, pose, local_scan):
        pose = np.array(pose, dtype=np.float64)
        local_scan = np.asarray(local_scan, dtype=np.float64)

        if not self.nodes:
            self.nodes.append(pose)
            self.scans.append(local_scan)
            return 0

        last = self.nodes[-1]
        rel = relative_pose(last, pose)
        moved = float(np.hypot(rel[0], rel[1]))
        turned_deg = float(np.degrees(abs(rel[2])))
        if moved < KEYFRAME_DIST_THRESHOLD_M and turned_deg < KEYFRAME_ANGLE_THRESHOLD_DEG:
            return None

        new_index = len(self.nodes)
        self.nodes.append(pose)
        self.scans.append(local_scan)
        self.edges.append((new_index - 1, new_index, rel))
        return new_index

    def try_loop_closure(self, node_index):
        if node_index is None or node_index < LOOP_CLOSURE_MIN_KEYFRAME_GAP:
            return False

        current_pose = self.nodes[node_index]
        current_scan_local = self.scans[node_index]
        if len(current_scan_local) == 0:
            return False

        search_limit = node_index - LOOP_CLOSURE_MIN_KEYFRAME_GAP
        candidate_indices = [
            i for i in range(search_limit)
            if np.hypot(*(self.nodes[i][:2] - current_pose[:2])) < LOOP_CLOSURE_SEARCH_RADIUS_M
            and len(self.scans[i]) > 0
        ]
        if not candidate_indices:
            return False

        best_score = float('inf')
        best_rel = None
        best_index = None

        for cand_index in candidate_indices:
            cand_pose = self.nodes[cand_index]
            cand_scan_local = self.scans[cand_index]

            init_rel = relative_pose(cand_pose, current_pose)
            score, refined_rel = _correlative_match(current_scan_local, cand_scan_local, init_rel)
            if score < best_score:
                best_score = score
                best_rel = refined_rel
                best_index = cand_index

        if best_index is None or best_score > LOOP_CLOSURE_SCORE_THRESHOLD:
            return False

        self.edges.append((best_index, node_index, best_rel))
        self.loop_closure_count += 1
        print(f"[SLAM] Loop closure accepted: keyframe {node_index} <-> {best_index} "
              f"(score={best_score:.4f}m, total closures={self.loop_closure_count})")
        return True

    def optimize(self):
        if len(self.nodes) < 2:
            return [n.copy() for n in self.nodes]

        x0 = np.concatenate(self.nodes)
        anchor = self.nodes[0]

        def residuals(x):
            poses = x.reshape(-1, 3)
            res = []
            for i, j, rel in self.edges:
                predicted = relative_pose(poses[i], poses[j])
                err = predicted - rel
                err[2] = wrap_angle(err[2])
                res.append(err)
            anchor_err = poses[0] - anchor
            anchor_err[2] = wrap_angle(anchor_err[2])
            res.append(anchor_err * 10.0)
            return np.concatenate(res)

        result = least_squares(residuals, x0, loss='soft_l1', method='trf')
        corrected = result.x.reshape(-1, 3)
        return [pose.copy() for pose in corrected]

    def rebuild_log_odds(self, corrected_poses, map_size, resolution, world_to_map_fn):
        from mapping import rasterize_scan 

        log_odds = np.full((map_size, map_size), INITIAL_LOG_ODD, dtype=np.float32)
        for pose, scan_local in zip(corrected_poses, self.scans):
            if len(scan_local) == 0:
                continue
            world_points = transform_points(scan_local, pose)
            map_points = world_to_map_fn(world_points)
            robot_map_pos = world_to_map_fn(np.array([[pose[0], pose[1]]]))[0]
            rasterize_scan(log_odds, robot_map_pos, map_points, map_size)
        return log_odds
