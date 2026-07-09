import math

import devices
import sensors
import perception
from reactive import laser_5_sectors, wall_follow_twist
from config import (
    TARGET_LIN_VEL, TARGET_ANG_VEL,
    FRONT_STOP_DIST, FRONT_BLOCK_DIST, FRONT_BLOCK_CLEAR_DIST, FRONT_CAUTION_DIST,
    CAUTION_LIN_VEL, CAUTION_ANG_VEL,
    WALL_TARGET_DIST, WALL_CLOSE_BAND, WALL_LOST_DIST, SIDE_DANGER_DIST,
    OMEGA_SMALL, BLOCK_TIMEOUT, REAR_SAFE_DIST,
    GREEN_STOP_DIST, GREEN_CAUTION_DIST,
    OVERHEAD_DETECT_DIST,
    SEEK_LIN_VEL, SEEK_OMEGA, SEEK_BEARING_DEADBAND_RAD,
    TARGET_REACHED_DIST_M,
)


SEEKING_BLUE   = 0
SEEKING_YELLOW = 1
DONE           = 2

_MISSION_NAMES = {
    SEEKING_BLUE:   "SEEKING_BLUE",
    SEEKING_YELLOW: "SEEKING_YELLOW",
    DONE:           "DONE",
}

_mission_state = SEEKING_BLUE


_block_latched = False
_block_open_side = None


def _choose_open_side(left_min, right_min):
    return "left" if left_min >= right_min else "right"


def reset_autonomous_state():
    global _block_latched, _block_open_side
    _block_latched = False
    _block_open_side = None


def reset_mission_state():
    global _mission_state
    _mission_state = SEEKING_BLUE
    print("[MISSION] reset to SEEKING_BLUE")


def mission_state_name():
    return _MISSION_NAMES.get(_mission_state, "?")


def _fuse_front_clearance(laser_front):
    _, _, overhead_front = sensors.overhead_depth_regions(OVERHEAD_DETECT_DIST)
    if laser_front > FRONT_CAUTION_DIST:
        return min(laser_front, overhead_front), overhead_front
    return laser_front, overhead_front


def _active_target_bearing(colors, pose):
    if _mission_state == SEEKING_BLUE:
        color = "blue"
    elif _mission_state == SEEKING_YELLOW:
        color = "yellow"
    else:
        return None, None, None

    visible = (
        colors.get(color)
        and colors.get(f"{color}_bearing_rad") is not None
    )
    if visible:
        bearing  = colors[f"{color}_bearing_rad"]
        distance = colors.get(f"{color}_distance_m", float('inf'))
        perception.update_target_memory(color, pose, bearing, distance)
        return color, bearing, distance

    mem = perception.get_target_memory(color)
    if mem is not None:
        m_bearing, m_distance = perception.bearing_distance_from_pose(pose, mem)
        if m_bearing is not None:
            return color, m_bearing, m_distance

    return None, None, None


def _is_target_reached(active_color, distance_m, bearing_rad, center_min):
    if active_color is None:
        return False
    if (distance_m is not None
            and math.isfinite(distance_m)
            and distance_m < TARGET_REACHED_DIST_M):
        return True
    if (bearing_rad is not None
            and abs(bearing_rad) < SEEK_BEARING_DEADBAND_RAD
            and center_min < FRONT_BLOCK_DIST):
        return True
    return False


def _advance_mission_state(reached_color, distance_m):
    global _mission_state
    new_state = _mission_state
    if _mission_state == SEEKING_BLUE and reached_color == "blue":
        new_state = SEEKING_YELLOW
    elif _mission_state == SEEKING_YELLOW and reached_color == "yellow":
        new_state = DONE
    if new_state != _mission_state:
        d_str = (
            f"{distance_m:.2f}"
            if (distance_m is not None and math.isfinite(distance_m))
            else "inf"
        )
        print(
            f"[MISSION] {_MISSION_NAMES[_mission_state]} → "
            f"{_MISSION_NAMES[new_state]} ({reached_color} reached, distance={d_str})"
        )
        _mission_state = new_state


def _seek_twist(bearing_rad):
    if abs(bearing_rad) > SEEK_BEARING_DEADBAND_RAD:
        if bearing_rad > 0:
            return 0.0, +SEEK_OMEGA, "seek_turn_left"
        return 0.0, -SEEK_OMEGA, "seek_turn_right"
    return SEEK_LIN_VEL, 0.0, "seek_forward"


def autonomous_step(block_timer, pose=None):
    raw_ranges = []
    if devices.laser:
        try:
            raw_ranges = devices.laser.getRangeImage() or []
        except Exception:
            pass
    far_left, left_min, center_min, right_min, far_right = laser_5_sectors(raw_ranges)

    center_min, overhead_min = _fuse_front_clearance(center_min)

    fl_val = devices.fl_range.getValue() if devices.fl_range else float('inf')
    fr_val = devices.fr_range.getValue() if devices.fr_range else float('inf')
    rl_val = devices.rl_range.getValue() if devices.rl_range else float('inf')
    rr_val = devices.rr_range.getValue() if devices.rr_range else float('inf')

    rear_safe = rl_val >= REAR_SAFE_DIST and rr_val >= REAR_SAFE_DIST

    colors = sensors.read_color_detections()
    global _block_latched, _block_open_side
    if center_min <= FRONT_BLOCK_DIST:
        if not _block_latched:
            _block_open_side = _choose_open_side(left_min, right_min)
        _block_latched = True
    elif _block_latched and center_min < FRONT_BLOCK_CLEAR_DIST:
        pass
    else:
        _block_latched = False
        _block_open_side = None

    if _block_latched:
        block_timer += 1
    else:
        block_timer = 0

    reactive_front_min = min(center_min, FRONT_BLOCK_DIST) if _block_latched else center_min
    v_cmd, omega_cmd, label = wall_follow_twist(
        reactive_front_min, right_min, left_min,
        block_timer, rear_safe,
        FRONT_BLOCK_DIST, FRONT_CAUTION_DIST,
        WALL_TARGET_DIST, WALL_CLOSE_BAND, WALL_LOST_DIST, SIDE_DANGER_DIST,
        BLOCK_TIMEOUT,
        TARGET_LIN_VEL, CAUTION_LIN_VEL,
        OMEGA_SMALL, CAUTION_ANG_VEL, TARGET_ANG_VEL,
        open_side_hint=_block_open_side,
    )


    active_color, active_bearing, active_distance = _active_target_bearing(colors, pose)
    if active_color is not None:
        sv, somega, slabel = _seek_twist(active_bearing)
        
        if sv == 0.0:
            v_cmd, omega_cmd, label = sv, somega, f"{slabel}_{active_color}"
            block_timer = 0
        elif center_min > FRONT_BLOCK_DIST:
            v_cmd = sv
            label = f"{slabel}_{active_color}"



  
    if _is_target_reached(active_color, active_distance, active_bearing, center_min):
        _advance_mission_state(active_color, active_distance)

    laser_front_emg = center_min < FRONT_STOP_DIST
    fl_emg    = fl_val < FRONT_STOP_DIST
    fr_emg    = fr_val < FRONT_STOP_DIST
    front_emg = laser_front_emg or fl_emg or fr_emg

    if v_cmd > 0 and front_emg:
        v_cmd = 0.0
        if fl_emg and not fr_emg:
            omega_cmd = -TARGET_ANG_VEL
            label     = "front_left_emg_turn_right"
        elif fr_emg and not fl_emg:
            omega_cmd = TARGET_ANG_VEL
            label     = "front_right_emg_turn_left"
        else:
            omega_cmd = TARGET_ANG_VEL
            label     = "front_emg_turn_left"

    if v_cmd < 0 and (rl_val < REAR_SAFE_DIST or rr_val < REAR_SAFE_DIST):
        v_cmd     = 0.0
        omega_cmd = TARGET_ANG_VEL
        label     = "rear_blocked_turn_left"

    if colors["green"] and v_cmd > 0:
        open_sign = 1.0 if left_min >= right_min else -1.0
        green_dist = colors["green_distance"]
        if green_dist < GREEN_STOP_DIST:
            v_cmd     = 0.0
            omega_cmd = open_sign * TARGET_ANG_VEL
            label     = "green_ground_stop_turn_left" if open_sign > 0 else "green_ground_stop_turn_right"
        elif green_dist < GREEN_CAUTION_DIST:
            v_cmd     = min(v_cmd, CAUTION_LIN_VEL)
            omega_cmd = open_sign * OMEGA_SMALL
            label     = "green_ground_slow_left" if open_sign > 0 else "green_ground_slow_right"
        elif green_dist == float('inf'):
            v_cmd     = 0.0
            omega_cmd = open_sign * TARGET_ANG_VEL
            label     = "green_ground_unknown_turn_left" if open_sign > 0 else "green_ground_unknown_turn_right"

    if _mission_state == DONE:
        v_cmd     = 0.0
        omega_cmd = 0.0
        label     = "mission_done"

    debug = {
        "green":          colors["green"],
        "green_ratio":    colors["green_ratio"],
        "green_distance": colors["green_distance"],
        "blue":           colors["blue"],
        "blue_ratio":     colors["blue_ratio"],
        "yellow":         colors["yellow"],
        "yellow_ratio":   colors["yellow_ratio"],
        "overhead_front": overhead_min,
        "block_timer":    block_timer,
        "active_target":  active_color,
        "mission_state":  _MISSION_NAMES[_mission_state],
    }
    return v_cmd, omega_cmd, label, block_timer, debug
