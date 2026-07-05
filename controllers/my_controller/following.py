"""DWA local path follower for the Husarion RosBot.

Ported from AURE (dwa_planner + follow_local_target), adapted from AURE's
blocking while-loop into a step-wise controller that fits this project's single
non-blocking control loop: load a path with set_path(), then call step() once
per control tick to get one (v, omega) twist.

Pure NumPy / mapping / config — no Webots imports.  Twist -> wheels is left to
motion.drive_twist() in the caller (AURE's velocity_to_wheel_speeds is the same
differential-drive map, already in kinematics.py).

Conventions: pose is (x_m, y_m, theta_rad) world frame; paths are lists of
(x, y) = (col, row) map cells (as produced by planning.plan()).
"""

import math

import mapping
from kinematics import WHEEL_RADIUS_M, MAX_WHEEL_SPEED_RAD_S
from config import (
    DWA_VELOCITY_SAMPLES, DWA_ANGULAR_SAMPLES, DWA_ROLLOUT_STEPS, DWA_ROBOT_RADIUS_PX,
    DWA_HEADING_WEIGHT, DWA_DISTANCE_WEIGHT, DWA_SPEED_WEIGHT, DWA_CLEARANCE_WEIGHT,
    PATH_FOLLOWING_TARGET_REACH_DIST_PX, FOLLOW_WAYPOINT_STRIDE,
    FOLLOW_STUCK_MOVE_M, FOLLOW_STUCK_TURN_RAD, FOLLOW_STUCK_STEPS,
)

# Forward-speed cap used to normalise the DWA speed reward (m/s).
_MAX_SPEED = MAX_WHEEL_SPEED_RAD_S * WHEEL_RADIUS_M

# ── Follower state ────────────────────────────────────────────────────────────
_path = None
_target_index = 0
_stuck_last_pos = None
_stuck_count = 0


def _wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _map_dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def dwa_velocity(pose, world_target, dt):
    """AURE Dynamic Window: pick the (v, omega) that best drives toward
    world_target while its short forward rollout stays clear of mapped
    obstacles.  Score = heading + distance-progress + speed + clearance.
    """
    x, y, theta = pose
    tx, ty = world_target
    current_distance = math.hypot(tx - x, ty - y)

    best_score = -float("inf")
    best_v = 0.0
    best_w = 0.0

    for v in DWA_VELOCITY_SAMPLES:
        for w in DWA_ANGULAR_SAMPLES:
            cx, cy, ct = x, y, theta
            good = True
            min_clear = float("inf")

            # Pure in-place rotation (v == 0) never translates the robot into a
            # wall, so it is always allowed — this is the robot's escape hatch
            # when it is boxed in and every forward trajectory is blocked.
            translating = v > 1e-3

            for _ in range(DWA_ROLLOUT_STEPS):
                cx += v * math.cos(ct) * dt
                cy += v * math.sin(ct) * dt
                ct += w * dt
                if not translating:
                    continue
                mx, my = mapping.world_to_map(cx, cy)

                # Hard reject: rollout enters an obstacle cell.
                if mapping.there_is_obstacle((mx, my)):
                    good = False
                    break

                # Nearest obstacle in a 5x5 neighbourhood (Manhattan) — reaches
                # the 3-cell body radius used for the hard-reject below.
                local_min = float("inf")
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        if mapping.there_is_obstacle((mx + dx, my + dy)):
                            local_min = min(local_min, abs(dx) + abs(dy))

                # Hard reject: the robot body would clip a wall (keep a radius buffer).
                if local_min <= DWA_ROBOT_RADIUS_PX:
                    good = False
                    break
                if local_min < min_clear:
                    min_clear = local_min

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

            if score > best_score:
                best_score = score
                best_v = v
                best_w = w

    return best_v, best_w


# ── Step-wise path follower ───────────────────────────────────────────────────

def set_path(path):
    """Load a path (list of (x, y) map cells) and reset follow state."""
    global _path, _target_index, _stuck_last_pos, _stuck_count
    _path = list(path) if path else None
    _target_index = min(FOLLOW_WAYPOINT_STRIDE, len(_path) - 1) if _path else 0
    _stuck_last_pos = None
    _stuck_count = 0


def reset():
    """Clear the follower state (no active path)."""
    global _path, _target_index, _stuck_last_pos, _stuck_count
    _path = None
    _target_index = 0
    _stuck_last_pos = None
    _stuck_count = 0


def has_path():
    return _path is not None


def current_path():
    """Return the path currently being followed (list of cells), or None."""
    return _path


def step(pose, dt):
    """Advance one control tick.  Returns (v, omega, status).

    status is one of: "following", "done", "stuck", "idle".
    """
    global _target_index, _stuck_last_pos, _stuck_count

    if not _path:
        return 0.0, 0.0, "idle"

    x, y, theta = pose
    robot_cell = mapping.world_to_map(x, y)

    # Reached the final goal?
    if _map_dist(robot_cell, _path[-1]) < PATH_FOLLOWING_TARGET_REACH_DIST_PX:
        return 0.0, 0.0, "done"

    # AURE follower: lock onto a FIXED waypoint until it is reached, then step
    # ahead by the stride.  A stable target (not re-picked every tick) is what
    # lets DWA drive straight at the path instead of hunting a moving carrot.
    while (_target_index < len(_path) - 1
           and _map_dist(robot_cell, _path[_target_index]) < PATH_FOLLOWING_TARGET_REACH_DIST_PX):
        _target_index += FOLLOW_WAYPOINT_STRIDE
    _target_index = min(_target_index, len(_path) - 1)

    # Stuck detection: no meaningful progress for N ticks.  Rotating in place
    # counts as progress (the robot is actively turning to escape / re-aim),
    # so a long turn isn't mistaken for being stuck.
    if _stuck_last_pos is not None:
        moved = math.hypot(x - _stuck_last_pos[0], y - _stuck_last_pos[1])
        turned = abs(_wrap(theta - _stuck_last_pos[2]))
        if moved < FOLLOW_STUCK_MOVE_M and turned < FOLLOW_STUCK_TURN_RAD:
            _stuck_count += 1
            if _stuck_count >= FOLLOW_STUCK_STEPS:
                return 0.0, 0.0, "stuck"
        else:
            _stuck_count = 0
    _stuck_last_pos = (x, y, theta)

    target = _path[_target_index]
    world_target = mapping.map_to_world(target[0], target[1])
    v, w = dwa_velocity(pose, world_target, dt)
    return v, w, "following"
