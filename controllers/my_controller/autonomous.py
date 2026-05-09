"""Autonomous navigation step for the Husarion RosBot.

Call autonomous_step(block_timer) once per simulation step.
Returns (v_cmd, omega_cmd, label, new_block_timer, debug).

All sensor reads happen here; no Webots imports elsewhere in this file.
"""

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


# ── Mission state ────────────────────────────────────────────────────────────
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
    """Choose a turn direction from current laser side clearance."""
    return "left" if left_min >= right_min else "right"


def reset_autonomous_state():
    """Clear short-term navigation memory when autonomous mode is reset."""
    global _block_latched, _block_open_side
    _block_latched = False
    _block_open_side = None


def reset_mission_state():
    """Reset the mission to SEEKING_BLUE.  Called by the R keypress."""
    global _mission_state
    _mission_state = SEEKING_BLUE
    print("[MISSION] reset to SEEKING_BLUE")


def mission_state_name():
    """Return the current mission state as a human-readable string."""
    return _MISSION_NAMES.get(_mission_state, "?")


def _fuse_front_clearance(laser_front):
    """Fuse laser front range with overhead depth without hiding close laser blocks."""
    _, _, overhead_front = sensors.overhead_depth_regions(OVERHEAD_DETECT_DIST)
    if laser_front > FRONT_CAUTION_DIST:
        return min(laser_front, overhead_front), overhead_front
    return laser_front, overhead_front


def _active_target_bearing(colors, pose):
    """Pick the active target governed by the mission state, with memory fallback.

    SEEKING_BLUE   → seek blue (yellow ignored even if visible).
    SEEKING_YELLOW → seek yellow (blue ignored even if visible).
    DONE           → no active target.

    When the active color is currently visible:
        - update perception's persistent memory with the world-frame projection
        - return the LIVE bearing + distance from the camera

    When the active color is NOT visible but a previous sighting is memorized:
        - synthesize bearing + distance from the current pose to the stored
          world position via perception.bearing_distance_from_pose

    Returns (color_name, bearing_rad, distance_m) or (None, None, None).
    """
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
        # Update memory with the projected world position (no-op if invalid).
        perception.update_target_memory(color, pose, bearing, distance)
        return color, bearing, distance

    mem = perception.get_target_memory(color)
    if mem is not None:
        m_bearing, m_distance = perception.bearing_distance_from_pose(pose, mem)
        if m_bearing is not None:
            return color, m_bearing, m_distance

    return None, None, None


def _is_target_reached(active_color, distance_m, bearing_rad, center_min):
    """Decide if the active target has been reached.

    Primary signal:
        depth-at-centroid below TARGET_REACHED_DIST_M.

    Fallback signal (depth saturates near the target):
        target is centred (|bearing| within seek deadband) AND the front
        laser is blocked (center_min < FRONT_BLOCK_DIST).  The bearing
        gate prevents a generic wall in front from triggering reached.
    """
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
    """Transition the mission state when the active target is reached."""
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
    """Body twist that orients toward the target then drives forward.

    Bearing convention (matches sensors._pixel_to_bearing):
        bearing > 0 → target on robot's left  → omega > 0 (rotate left)
        bearing < 0 → target on robot's right → omega < 0 (rotate right)
    Inside the deadband, drive forward at SEEK_LIN_VEL with omega = 0.
    """
    if abs(bearing_rad) > SEEK_BEARING_DEADBAND_RAD:
        if bearing_rad > 0:
            return 0.0, +SEEK_OMEGA, "seek_turn_left"
        return 0.0, -SEEK_OMEGA, "seek_turn_right"
    return SEEK_LIN_VEL, 0.0, "seek_forward"


def autonomous_step(block_timer, pose=None):
    """Run one step of autonomous right-hand wall-following.

    Parameters
    ----------
    block_timer : int
        Consecutive steps the front has been blocked (carried across calls).
    pose : (x_m, y_m, theta_rad), optional
        Current robot pose in the (reset-anchored) world frame.  When
        provided, target sightings are memorised and re-acquisition uses
        memory when the target leaves the field of view.  When None,
        memory is neither read nor written and seek behavior degrades to
        live-only (M4.0c equivalent).

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

    # ── Reactive target seek (FR5: blue first, then yellow) ──────────────────
    # Substitute the seek twist for the wall-follow output only when the
    # front is clear; otherwise let wall-follow handle the obstruction so
    # its block-recovery logic still runs.  All safety overrides below
    # still apply on top of whichever twist was chosen.
    active_color, active_bearing, active_distance = _active_target_bearing(colors, pose)
    if active_color is not None and center_min > FRONT_BLOCK_DIST:
        sv, somega, slabel = _seek_twist(active_bearing)
        v_cmd, omega_cmd, label = sv, somega, f"{slabel}_{active_color}"

    # ── Mission state advancement: check if active target is reached ──────────
    # active_distance is the live depth-at-centroid when the target is
    # visible, or the synthesised pose-to-memory distance when it isn't.
    # The reached predicate also falls back to "centred AND front blocked"
    # when depth is unmeasurable.
    if _is_target_reached(active_color, active_distance, active_bearing, center_min):
        _advance_mission_state(active_color, active_distance)

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

    # ── DONE: mission complete, hold still regardless of any other signal ────
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
