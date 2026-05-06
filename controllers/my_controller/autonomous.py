"""Autonomous navigation step for the Husarion RosBot.

Call autonomous_step(block_timer) once per simulation step.
Returns (v_cmd, omega_cmd, label, new_block_timer, debug).

All sensor reads happen here; no Webots imports elsewhere in this file.
"""

import devices
import sensors
from reactive import laser_5_sectors, wall_follow_twist
from config import (
    TARGET_LIN_VEL, TARGET_ANG_VEL,
    FRONT_STOP_DIST, FRONT_BLOCK_DIST, FRONT_BLOCK_CLEAR_DIST, FRONT_CAUTION_DIST,
    CAUTION_LIN_VEL, CAUTION_ANG_VEL,
    WALL_TARGET_DIST, WALL_CLOSE_BAND, WALL_LOST_DIST, SIDE_DANGER_DIST,
    OMEGA_SMALL, BLOCK_TIMEOUT, REAR_SAFE_DIST,
    GREEN_STOP_DIST, GREEN_CAUTION_DIST,
    OVERHEAD_DETECT_DIST,
)


_block_latched = False
_block_open_side = None


def _choose_open_side(left_min, right_min):
    """Choose a turn direction from current laser side clearance."""
    return "left" if left_min >= right_min else "right"


def reset_autonomous_state():
    """Clear short-term navigation memory when autonomous mode is reset."""
    global _block_latched, _block_open_side
    _block_latched = False
    _block_open_side = None


def _fuse_front_clearance(laser_front):
    """Fuse laser front range with overhead depth without hiding close laser blocks."""
    _, _, overhead_front = sensors.overhead_depth_regions(OVERHEAD_DETECT_DIST)
    if laser_front > FRONT_CAUTION_DIST:
        return min(laser_front, overhead_front), overhead_front
    return laser_front, overhead_front


def autonomous_step(block_timer):
    """Run one step of autonomous right-hand wall-following.

    Parameters
    ----------
    block_timer : int
        Consecutive steps the front has been blocked (carried across calls).

    Returns
    -------
    (v_cmd, omega_cmd, label, new_block_timer, debug) : tuple
    """
    # ── Laser sectors ──────────────────────────────────────────────────────────
    raw_ranges = []
    if devices.laser:
        try:
            raw_ranges = devices.laser.getRangeImage() or []
        except Exception:
            pass
    far_left, left_min, center_min, right_min, far_right = laser_5_sectors(raw_ranges)

    center_min, overhead_min = _fuse_front_clearance(center_min)

    # ── Short-range sensors ────────────────────────────────────────────────────
    fl_val = devices.fl_range.getValue() if devices.fl_range else float('inf')
    fr_val = devices.fr_range.getValue() if devices.fr_range else float('inf')
    rl_val = devices.rl_range.getValue() if devices.rl_range else float('inf')
    rr_val = devices.rr_range.getValue() if devices.rr_range else float('inf')

    rear_safe = rl_val >= REAR_SAFE_DIST and rr_val >= REAR_SAFE_DIST

    colors = sensors.read_color_detections()

    # ── Block-timer with hysteresis ────────────────────────────────────────────
    # Enter blocked at FRONT_BLOCK_DIST, but do not release until the front
    # clears FRONT_BLOCK_CLEAR_DIST. This prevents 0.249/0.250 chatter.
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

    # ── Core twist selection ───────────────────────────────────────────────────
    # Let the reactive layer handle blocked/caution/recovery decisions so its
    # block-timeout backup behavior can actually run.
    reactive_front_min = min(center_min, FRONT_BLOCK_DIST) if _block_latched else center_min
    v_cmd, omega_cmd, label = wall_follow_twist(
        reactive_front_min, right_min, left_min,
        block_timer, rear_safe,
        FRONT_BLOCK_DIST, FRONT_CAUTION_DIST,
        WALL_TARGET_DIST, WALL_CLOSE_BAND, WALL_LOST_DIST, SIDE_DANGER_DIST,
        BLOCK_TIMEOUT,
        TARGET_LIN_VEL, CAUTION_LIN_VEL,
        # wall_follow_twist signature: omega_small, omega_caution, omega_large
        OMEGA_SMALL, CAUTION_ANG_VEL, TARGET_ANG_VEL,
        open_side_hint=_block_open_side,
    )

    # ── Hard emergency overrides (applied after wall-follow decision) ──────────
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

    # ── Rear safety: block reverse when rear sensors are close ─────────────────
    if v_cmd < 0 and (rl_val < REAR_SAFE_DIST or rr_val < REAR_SAFE_DIST):
        v_cmd     = 0.0
        omega_cmd = TARGET_ANG_VEL
        label     = "rear_blocked_turn_left"

    # ── Semantic safety: never drive onto green forbidden ground ──────────────
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
            # If green distance is unavailable, keep the conservative stop.
            v_cmd     = 0.0
            omega_cmd = open_sign * TARGET_ANG_VEL
            label     = "green_ground_unknown_turn_left" if open_sign > 0 else "green_ground_unknown_turn_right"

    debug = {
        "green":         colors["green"],
        "green_ratio":   colors["green_ratio"],
        "green_distance": colors["green_distance"],
        "blue":          colors["blue"],
        "blue_ratio":    colors["blue_ratio"],
        "yellow":        colors["yellow"],
        "yellow_ratio":  colors["yellow_ratio"],
        "overhead_front": overhead_min,
        "block_timer":   block_timer,
    }
    return v_cmd, omega_cmd, label, block_timer, debug
