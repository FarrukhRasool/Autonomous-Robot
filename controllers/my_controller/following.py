
import math

import numpy as np
import cv2

import mapping
from kinematics import WHEEL_RADIUS_M, MAX_WHEEL_SPEED_RAD_S
from config import (
    DWA_VELOCITY_SAMPLES, DWA_ANGULAR_SAMPLES, DWA_ROLLOUT_STEPS,
    DWA_HEADING_WEIGHT, DWA_DISTANCE_WEIGHT, DWA_SPEED_WEIGHT, DWA_CLEARANCE_WEIGHT,
    DWA_ROBOT_CLEAR_PX, DWA_CLEAR_PENALTY,
    PATH_FOLLOWING_TARGET_REACH_DIST_PX, FOLLOW_WAYPOINT_STRIDE,
    FOLLOW_RECOVER_STEPS, FOLLOW_RECOVER_VEL, FOLLOW_RECOVER_OMEGA, FOLLOW_MAX_RECOVERS,
    FOLLOW_PROGRESS_WIN, FOLLOW_MIN_PROGRESS_CELLS,
    REAR_SAFE_DIST,
    CELL_OCC, CELL_GREEN, CELL_CLOSED, CELL_BLUE, CELL_YELLOW,
)


_MAX_SPEED = MAX_WHEEL_SPEED_RAD_S * WHEEL_RADIUS_M


_path = None
_target_index = 0
_recover_ticks = 0    
_recover_count = 0   
_prog_ref = None      
_prog_ticks = 0      


def _wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _map_dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _obstacle_distance_field():
    
    grid = mapping.get_grid()
    obstacle = ((grid == CELL_OCC) | (grid == CELL_GREEN) | (grid == CELL_CLOSED)
                | (grid == CELL_BLUE) | (grid == CELL_YELLOW))
    free_src = np.where(obstacle, 0, 255).astype(np.uint8)
    return cv2.distanceTransform(free_src, cv2.DIST_L2, 3)


def dwa_velocity(pose, world_target, dt):
    
    x, y, theta = pose
    tx, ty = world_target
    current_distance = math.hypot(tx - x, ty - y)

    dist_field = _obstacle_distance_field()
    map_h, map_w = dist_field.shape

    best_score = -float("inf")
    best_v = 0.0
    best_w = 0.0

    for v in DWA_VELOCITY_SAMPLES:
        for w in DWA_ANGULAR_SAMPLES:
            cx, cy, ct = x, y, theta
            good = True
            min_clear = float("inf")
            end_clear = float("inf")  

            
            translating = v > 1e-3

            for _ in range(DWA_ROLLOUT_STEPS):
                cx += v * math.cos(ct) * dt
                cy += v * math.sin(ct) * dt
                ct += w * dt
                if not translating:
                    continue
                mx, my = mapping.world_to_map(cx, cy)

               
                if not (0 <= mx < map_w and 0 <= my < map_h):
                    good = False
                    break
                clear_px = float(dist_field[my, mx])
                if clear_px < 1.0:
                    good = False
                    break
                if clear_px < min_clear:
                    min_clear = clear_px
                end_clear = clear_px    

            if not good:
                continue

            pred_dist = math.hypot(tx - cx, ty - cy)
            pred_angle = math.atan2(ty - cy, tx - cx)
            heading_score = math.cos(_wrap(pred_angle - ct))
            distance_score = 1.0 - pred_dist / max(current_distance, 1e-3)
            speed_score = (v / _MAX_SPEED) if _MAX_SPEED > 0 else 0.0
            clearance_score = (min(min_clear, 4.0) / 4.0) if min_clear != float("inf") else 1.0

            score = (DWA_HEADING_WEIGHT * heading_score
                     + DWA_DISTANCE_WEIGHT * distance_score
                     + DWA_SPEED_WEIGHT * speed_score
                     + DWA_CLEARANCE_WEIGHT * clearance_score)

            
            if end_clear != float("inf") and end_clear < DWA_ROBOT_CLEAR_PX:
                score -= DWA_CLEAR_PENALTY * (DWA_ROBOT_CLEAR_PX - end_clear)

            if score > best_score:
                best_score = score
                best_v = v
                best_w = w

    return best_v, best_w




def _reset_recovery():
    global _recover_ticks, _recover_count, _prog_ref, _prog_ticks
    _recover_ticks = 0
    _recover_count = 0
    _prog_ref = None
    _prog_ticks = 0


def set_path(path):
    global _path, _target_index
    _path = list(path) if path else None
    _target_index = min(FOLLOW_WAYPOINT_STRIDE, len(_path) - 1) if _path else 0
    _reset_recovery()


def reset():
    global _path, _target_index
    _path = None
    _target_index = 0
    _reset_recovery()


def _rear_clear(pose):
    x, y, theta = pose
    for d in (0.06, 0.12, 0.18):
        bx = x - d * math.cos(theta)
        by = y - d * math.sin(theta)
        if mapping.there_is_obstacle(mapping.world_to_map(bx, by)):
            return False
    return True


def has_path():
    return _path is not None


def current_path():
    return _path


def step(pose, dt):
    global _target_index, _recover_ticks, _recover_count, _prog_ref, _prog_ticks

    if not _path:
        return 0.0, 0.0, "idle"

    x, y, theta = pose
    robot_cell = mapping.world_to_map(x, y)

    goal_dist = _map_dist(robot_cell, _path[-1])
    if goal_dist < PATH_FOLLOWING_TARGET_REACH_DIST_PX:
        return 0.0, 0.0, "done"

    
    if _prog_ref is None:
        _prog_ref = goal_dist
    _prog_ticks += 1
    if _prog_ticks >= FOLLOW_PROGRESS_WIN:
        closed = _prog_ref - goal_dist
        _prog_ref = goal_dist
        _prog_ticks = 0
        if closed >= FOLLOW_MIN_PROGRESS_CELLS:
            _recover_count = 0
        else:
            _recover_count += 1
            if _recover_count > FOLLOW_MAX_RECOVERS:
                return 0.0, 0.0, "stuck"          
            _recover_ticks = FOLLOW_RECOVER_STEPS  

    
    if _recover_ticks > 0:
        _recover_ticks -= 1
        if _rear_clear(pose):
            return FOLLOW_RECOVER_VEL, FOLLOW_RECOVER_OMEGA, "recovering"
        return 0.0, FOLLOW_RECOVER_OMEGA, "recovering"

    
    while (_target_index < len(_path) - 1
           and _map_dist(robot_cell, _path[_target_index]) < PATH_FOLLOWING_TARGET_REACH_DIST_PX):
        _target_index += FOLLOW_WAYPOINT_STRIDE
    _target_index = min(_target_index, len(_path) - 1)

    target = _path[_target_index]
    world_target = mapping.map_to_world(target[0], target[1])

    
    v, w = dwa_velocity(pose, world_target, dt)

    
    if abs(v) < 1e-3 and abs(w) < 1e-3:
        _recover_ticks = FOLLOW_RECOVER_STEPS
        return (FOLLOW_RECOVER_VEL if _rear_clear(pose) else 0.0), FOLLOW_RECOVER_OMEGA, "recovering"

    return v, w, "following"
