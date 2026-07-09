import math


def target_world_position(pose, bearing_rad, distance_m):
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

_target_memory = {"blue": None, "yellow": None}


def update_target_memory(color, pose, bearing_rad, distance_m):
    if color not in _target_memory:
        return
    wp = target_world_position(pose, bearing_rad, distance_m)
    if wp is None:
        return
    _target_memory[color] = wp


def get_target_memory(color):
    return _target_memory.get(color)


def forget_target(color):
    if color in _target_memory:
        _target_memory[color] = None


def reset_target_memory():
    for key in _target_memory:
        _target_memory[key] = None
