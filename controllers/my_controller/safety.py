import devices
import sensors
from reactive import laser_5_sectors
from config import (
    FRONT_STOP_DIST, FRONT_BLOCK_DIST, REAR_SAFE_DIST,
    TARGET_ANG_VEL, OMEGA_SMALL, CAUTION_LIN_VEL,
    GREEN_STOP_DIST, GREEN_CAUTION_DIST,
    FRONT_BLOCK_TIMEOUT_STEPS, FRONT_ESCAPE_REVERSE_STEPS, FRONT_ESCAPE_REVERSE_VEL,
    FRONT_ESCAPE_TURN_OMEGA,
)

INF = float("inf")

_front_latched = False
_front_side = 1.0       
_block_count = 0      
_reverse_count = 0     


def _range(sensor):
    try:
        return sensor.getValue() if sensor is not None else INF
    except Exception:
        return INF


def apply(v_cmd, omega_cmd, colors=None):
    global _front_latched, _front_side, _block_count, _reverse_count
    label = ""

    raw = []
    if devices.laser is not None:
        try:
            raw = devices.laser.getRangeImage() or []
        except Exception:
            raw = []
    _, left_min, center_min, right_min, _ = laser_5_sectors(raw)

    fl = _range(devices.fl_range)
    fr = _range(devices.fr_range)
    rl = _range(devices.rl_range)
    rr = _range(devices.rr_range)
    if colors is None:
        colors = sensors.read_color_detections()

    reversing_cmd = v_cmd < -1e-3
    rear_clear = (rl > REAR_SAFE_DIST) and (rr > REAR_SAFE_DIST)
    block_now = (center_min < FRONT_STOP_DIST) or (fl < FRONT_STOP_DIST) or (fr < FRONT_STOP_DIST)
    clear_now = (center_min > FRONT_BLOCK_DIST) and (fl > FRONT_BLOCK_DIST) and (fr > FRONT_BLOCK_DIST)

    if not reversing_cmd:
        if block_now:
            if not _front_latched:
                _front_side = 1.0 if left_min >= right_min else -1.0
                _block_count = 0
            _front_latched = True
            _block_count += 1
        elif _front_latched and not clear_now:
            _block_count += 1
        else:
            _front_latched = False
            _block_count = 0

        if _reverse_count > 0:
            _reverse_count -= 1
            if rear_clear:
                v_cmd, omega_cmd = FRONT_ESCAPE_REVERSE_VEL, _front_side * FRONT_ESCAPE_TURN_OMEGA
                label = "block_reverse"
            else:
                v_cmd, omega_cmd = 0.0, _front_side * TARGET_ANG_VEL
                label = "block_reverse_turn"
        elif _front_latched:
            if _block_count >= FRONT_BLOCK_TIMEOUT_STEPS and rear_clear:
                _reverse_count = FRONT_ESCAPE_REVERSE_STEPS
                _front_latched = False
                _block_count = 0
                _front_side = -_front_side
                v_cmd, omega_cmd = FRONT_ESCAPE_REVERSE_VEL, 0.0
                label = "block_reverse"
            else:
                v_cmd = 0.0
                omega_cmd = _front_side * TARGET_ANG_VEL
                label = "front_block_turn_left" if _front_side > 0 else "front_block_turn_right"

    if v_cmd < 0 and not rear_clear:
        v_cmd = 0.0
        omega_cmd = _front_side * TARGET_ANG_VEL
        label = "rear_blocked"

    if colors["green"] and v_cmd > 0:
        open_sign = 1.0 if left_min >= right_min else -1.0
        green_dist = colors["green_distance"]
        if green_dist < GREEN_STOP_DIST:
            v_cmd = 0.0
            omega_cmd = open_sign * TARGET_ANG_VEL
            label = "green_stop"
        elif green_dist < GREEN_CAUTION_DIST:
            v_cmd = min(v_cmd, CAUTION_LIN_VEL)
            omega_cmd = open_sign * OMEGA_SMALL
            label = "green_slow"
        elif green_dist == INF:
            v_cmd = 0.0
            omega_cmd = open_sign * TARGET_ANG_VEL
            label = "green_unknown"

    return v_cmd, omega_cmd, label
