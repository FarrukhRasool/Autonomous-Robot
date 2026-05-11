"""World-frame perception transforms for the Husarion RosBot.

Pure Python — no Webots imports.  Composes pose, body-frame measurements,
and a uniform body→world transform into world-frame estimates.

Frame convention (matches localization.py and mapping.py):
    Body x forward, y left, z up; +theta yaws CCW (left).
    World origin (0, 0) at reset_pose() location.
"""

import math


def target_world_position(pose, bearing_rad, distance_m):
    """Project a target observation into the world frame.

    Parameters
    ----------
    pose : (x_m, y_m, theta_rad) or None
        Current robot pose in the (reset-anchored) world frame.
    bearing_rad : float or None
        Body-frame bearing of the target (0 = forward, +left), radians.
    distance_m : float or None
        Range from robot to target in metres.  Must be finite and > 0.

    Returns
    -------
    (wx, wy) : tuple of float, or None
        World-frame target position, or None when any input is missing
        or degenerate (None, NaN, inf, non-positive distance).

    Body→world transform (identical to mapping.update_from_laser):
        bx = d * cos(bearing)                body forward component
        by = d * sin(bearing)                body left component
        wx = px + bx*cos(theta) - by*sin(theta)
        wy = py + bx*sin(theta) + by*cos(theta)
    """
    if pose is None or bearing_rad is None or distance_m is None:
        return None
    if not math.isfinite(distance_m) or distance_m <= 0.0:
        return None
    if not math.isfinite(bearing_rad):
        return None

    px, py, ptheta = pose
    if not (math.isfinite(px) and math.isfinite(py) and math.isfinite(ptheta)):
        return None

    bx = distance_m * math.cos(bearing_rad)
    by = distance_m * math.sin(bearing_rad)

    cos_t = math.cos(ptheta)
    sin_t = math.sin(ptheta)

    wx = px + bx * cos_t - by * sin_t
    wy = py + bx * sin_t + by * cos_t
    return wx, wy


def bearing_distance_from_pose(pose, world_position):
    """Algebraic inverse of target_world_position.

    Given the robot's pose and a world-frame target point, return the
    body-frame (bearing_rad, distance_m) the robot would observe if
    looking at that point — exactly what target_world_position consumed
    in the forward direction.

    Returns (None, None) for missing or non-finite inputs.

    World→body rotation is the transpose of the body→world rotation:
        bx =  dx*cos(theta) + dy*sin(theta)
        by = -dx*sin(theta) + dy*cos(theta)
    """
    if pose is None or world_position is None:
        return None, None
    px, py, ptheta = pose
    if not (math.isfinite(px) and math.isfinite(py) and math.isfinite(ptheta)):
        return None, None
    wx, wy = world_position
    if not (math.isfinite(wx) and math.isfinite(wy)):
        return None, None

    dx = wx - px
    dy = wy - py
    cos_t = math.cos(ptheta)
    sin_t = math.sin(ptheta)

    bx =  dx * cos_t + dy * sin_t
    by = -dx * sin_t + dy * cos_t

    bearing  = math.atan2(by, bx)
    distance = math.hypot(bx, by)
    return bearing, distance


# ── Persistent target memory ─────────────────────────────────────────────────
# Latest world-frame sighting per target color, or None when unseen / cleared.
_target_memory = {"blue": None, "yellow": None}


def update_target_memory(color, pose, bearing_rad, distance_m):
    """Project the current observation and store it as the latest sighting.

    Silently no-op if the projection fails (any None / non-finite input)
    or if `color` is not a tracked target.
    """
    if color not in _target_memory:
        return
    wp = target_world_position(pose, bearing_rad, distance_m)
    if wp is None:
        return
    _target_memory[color] = wp


def get_target_memory(color):
    """Return the latest world-frame sighting for `color`, or None."""
    return _target_memory.get(color)


def reset_target_memory():
    """Clear all stored sightings (called by the R keypress)."""
    for key in _target_memory:
        _target_memory[key] = None
