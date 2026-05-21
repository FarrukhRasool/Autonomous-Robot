"""Autonomous navigation step for the Husarion RosBot.

Call autonomous_step(block_timer, pose) once per simulation step.
Returns (v_cmd, omega_cmd, label, new_block_timer, debug).

Navigation pipeline
-------------------
Each mission target (blue pillar, then yellow) passes through three
sub-modes controlled by _nav_mode:

  EXPLORE  Wall-follow + camera-based reactive seek.  The occupancy map
           grows while the robot explores.  When the active target's
           world position is recorded in perception memory, the mode
           transitions to PLAN.

  PLAN     A* is computed from the robot's current cell to the target's
           last known world position (one step; may be slow in Python).
           On success → FOLLOW.  On failure (no path yet) → EXPLORE.

  FOLLOW   The waypoint controller steers along the A* path.  Replanning
           is triggered automatically when the robot gets stuck or after
           REPLAN_PERIOD_STEPS steps so the path stays accurate as new
           walls are discovered.

Safety overrides (front/rear emergency, green-ground avoidance) are applied
on top of whichever mode produced the twist.  They always win.
"""

import math

import devices
import sensors
import perception
import planner
import waypoint_controller
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
    REPLAN_BLOCK_TIMEOUT, REPLAN_PERIOD_STEPS,
)


# ── Mission states ────────────────────────────────────────────────────────────
SEEKING_BLUE   = 0
SEEKING_YELLOW = 1
DONE           = 2

_MISSION_NAMES = {
    SEEKING_BLUE:   "SEEKING_BLUE",
    SEEKING_YELLOW: "SEEKING_YELLOW",
    DONE:           "DONE",
}

_mission_state = SEEKING_BLUE

# ── Navigation sub-modes ──────────────────────────────────────────────────────
_NAV_EXPLORE = 0    # wall-follow + reactive seek; build map
_NAV_PLAN    = 1    # compute A* path to target (one-step transient)
_NAV_FOLLOW  = 2    # follow A* waypoints via waypoint_controller

_NAV_NAMES = {_NAV_EXPLORE: "EXPLORE", _NAV_PLAN: "PLAN", _NAV_FOLLOW: "FOLLOW"}

_nav_mode       = _NAV_EXPLORE
_replan_counter = 0     # steps spent in FOLLOW (drives periodic replan)

# ── Reactive state ────────────────────────────────────────────────────────────
_block_latched  = False
_block_open_side = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _choose_open_side(left_min, right_min):
    return "left" if left_min >= right_min else "right"


def _current_target_color():
    if _mission_state == SEEKING_BLUE:
        return "blue"
    if _mission_state == SEEKING_YELLOW:
        return "yellow"
    return None


def reset_autonomous_state():
    """Clear short-term navigation memory (called when autonomous mode toggled)."""
    global _block_latched, _block_open_side, _nav_mode, _replan_counter
    _block_latched   = False
    _block_open_side = None
    _nav_mode        = _NAV_EXPLORE
    _replan_counter  = 0
    waypoint_controller.clear_path()


def reset_mission_state():
    """Reset the mission to SEEKING_BLUE.  Called by the R keypress."""
    global _mission_state
    _mission_state = SEEKING_BLUE
    reset_autonomous_state()
    print("[MISSION] reset to SEEKING_BLUE")


def mission_state_name():
    return _MISSION_NAMES.get(_mission_state, "?")


# ── Sensor fusion ─────────────────────────────────────────────────────────────

def _fuse_front_clearance(laser_front):
    _, _, overhead_front = sensors.overhead_depth_regions(OVERHEAD_DETECT_DIST)
    if laser_front > FRONT_CAUTION_DIST:
        return min(laser_front, overhead_front), overhead_front
    return laser_front, overhead_front


# ── Target bearing helpers ────────────────────────────────────────────────────

def _live_camera_target(colors):
    """Return (color, bearing_rad, distance_m) from the LIVE camera only.

    Returns (None, None, None) when the active target is not currently
    visible in the camera.  Never uses perception memory.

    Used exclusively for reached-detection so that memory drift or a noisy
    depth reading never causes a premature mission advance.
    """
    color = _current_target_color()
    if color is None or not colors.get(color):
        return None, None, None
    bearing = colors.get(f"{color}_bearing_rad")
    if bearing is None:
        return None, None, None
    distance = colors.get(f"{color}_distance_m", float('inf'))
    return color, bearing, distance


def _active_target_bearing(colors, pose):
    """Return (color, bearing_rad, distance_m) for navigation.

    Live camera takes priority; falls back to perception memory when the
    target is out of view so the robot keeps heading toward the last known
    world position even between camera sightings.
    Returns (None, None, None) when DONE or target has never been seen.
    """
    color = _current_target_color()
    if color is None:
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


# ── Target reached predicate ──────────────────────────────────────────────────

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


# ── Mission advancement ───────────────────────────────────────────────────────

def _advance_mission_state(reached_color, distance_m):
    global _mission_state, _nav_mode, _replan_counter
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
            f"{_MISSION_NAMES[new_state]} ({reached_color} reached, dist={d_str} m)"
        )
        _mission_state  = new_state
        _nav_mode       = _NAV_EXPLORE
        _replan_counter = 0
        waypoint_controller.clear_path()


# ── Reactive seek twist (used in EXPLORE mode when target visible) ─────────────

def _seek_twist(bearing_rad):
    if abs(bearing_rad) > SEEK_BEARING_DEADBAND_RAD:
        if bearing_rad > 0:
            return 0.0, +SEEK_OMEGA, "seek_turn_left"
        return 0.0, -SEEK_OMEGA, "seek_turn_right"
    return SEEK_LIN_VEL, 0.0, "seek_forward"


# ── Navigation mode transitions ───────────────────────────────────────────────

def _update_nav_mode(pose, active_color, block_timer):
    """Advance _nav_mode according to target memory availability and stuck state.

    Returns the updated _nav_mode (module global is also mutated).
    """
    global _nav_mode, _replan_counter

    if _mission_state == DONE:
        return _nav_mode

    color = _current_target_color()
    mem   = perception.get_target_memory(color) if color else None

    # ── EXPLORE → PLAN: target world position is now known ───────────────────
    if _nav_mode == _NAV_EXPLORE and mem is not None and pose is not None:
        _nav_mode = _NAV_PLAN
        print(f"[NAV] {color} position known → planning path")

    # ── PLAN: compute A* path (transient; runs for one step) ─────────────────
    if _nav_mode == _NAV_PLAN:
        if mem is None or pose is None:
            _nav_mode = _NAV_EXPLORE
        else:
            path = planner.plan(pose, mem)
            if path and len(path) >= 2:
                waypoint_controller.set_path(path)
                _replan_counter = 0
                _nav_mode = _NAV_FOLLOW
                print(f"[NAV] Path found ({len(path)} waypoints) → FOLLOW")
            else:
                _nav_mode = _NAV_EXPLORE
                print(f"[NAV] No path to {color} yet → EXPLORE")

    # ── FOLLOW: check for replan triggers ────────────────────────────────────
    if _nav_mode == _NAV_FOLLOW:
        _replan_counter += 1
        if block_timer >= REPLAN_BLOCK_TIMEOUT:
            _nav_mode = _NAV_PLAN
            print("[NAV] Stuck in FOLLOW → replanning")
        elif _replan_counter >= REPLAN_PERIOD_STEPS:
            _nav_mode = _NAV_PLAN
            print("[NAV] Periodic replan")

    return _nav_mode


# ── Main step ─────────────────────────────────────────────────────────────────

def autonomous_step(block_timer, pose=None):
    """Run one step of autonomous navigation.

    Parameters
    ----------
    block_timer : int
        Consecutive steps the front has been blocked (carried across calls).
    pose : (x_m, y_m, theta_rad), optional
        Current robot pose.  Without pose, planning is disabled and the
        controller falls back to EXPLORE mode only.

    Returns
    -------
    (v_cmd, omega_cmd, label, new_block_timer, debug) : tuple
    """
    global _block_latched, _block_open_side

    # ── Laser sectors ─────────────────────────────────────────────────────────
    raw_ranges = []
    if devices.laser:
        try:
            raw_ranges = devices.laser.getRangeImage() or []
        except Exception:
            pass
    far_left, left_min, center_min, right_min, far_right = laser_5_sectors(raw_ranges)
    center_min, overhead_min = _fuse_front_clearance(center_min)

    # ── Short-range sensors ───────────────────────────────────────────────────
    fl_val = devices.fl_range.getValue() if devices.fl_range else float('inf')
    fr_val = devices.fr_range.getValue() if devices.fr_range else float('inf')
    rl_val = devices.rl_range.getValue() if devices.rl_range else float('inf')
    rr_val = devices.rr_range.getValue() if devices.rr_range else float('inf')

    rear_safe = rl_val >= REAR_SAFE_DIST and rr_val >= REAR_SAFE_DIST

    colors = sensors.read_color_detections()

    # Memory is updated only for the ACTIVE target inside _active_target_bearing.

    # ── Block-timer with hysteresis ───────────────────────────────────────────
    if center_min <= FRONT_BLOCK_DIST:
        if not _block_latched:
            _block_open_side = _choose_open_side(left_min, right_min)
        _block_latched = True
    elif _block_latched and center_min < FRONT_BLOCK_CLEAR_DIST:
        pass  # hysteresis: stay latched until fully clear
    else:
        _block_latched   = False
        _block_open_side = None

    if _block_latched:
        block_timer += 1
    else:
        block_timer = 0

    # ── Active target: two views of the same target ───────────────────────────
    # Navigation  → may fall back to memory so the robot keeps heading toward
    #               the last known position even when the target is off-camera.
    # Reached det → LIVE camera only.  Memory drift or a noisy close-range
    #               depth pixel must never cause a premature mission advance.
    active_color, active_bearing, active_distance = _active_target_bearing(
        colors, pose
    )
    live_color, live_bearing, live_distance = _live_camera_target(colors)

    # ── Navigation mode transition ─────────────────────────────────────────────
    nav = _update_nav_mode(pose, active_color, block_timer)

    # ── Core twist selection ───────────────────────────────────────────────────
    if nav == _NAV_FOLLOW and pose is not None:
        # Waypoint controller drives; block_timer is zeroed while following.
        v_cmd, omega_cmd, label, path_done = waypoint_controller.follow_step(pose)
        block_timer = 0
        if path_done:
            # Path exhausted — switch to EXPLORE to let mission check handle it.
            global _nav_mode
            _nav_mode = _NAV_EXPLORE

    else:
        # EXPLORE mode: wall-follow as base, seek override when target visible.
        reactive_front_min = (
            min(center_min, FRONT_BLOCK_DIST) if _block_latched else center_min
        )
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

        # Reactive seek override (only in EXPLORE — in FOLLOW the path handles it).
        if active_color is not None:
            sv, somega, slabel = _seek_twist(active_bearing)
            if sv == 0.0:
                # Rotation always allowed.
                v_cmd, omega_cmd, label = sv, somega, f"{slabel}_{active_color}"
                block_timer = 0
            elif center_min > FRONT_BLOCK_DIST:
                # Forward only when front is clear.
                v_cmd = sv
                label = f"{slabel}_{active_color}"

    # ── Mission advancement ───────────────────────────────────────────────────
    # Reached check uses LIVE camera data only — never memory-estimated distance.
    # active_distance may be a memory-based estimate used only for navigation;
    # using it here would let drift or a noisy depth pixel stop the robot early.
    if _is_target_reached(live_color, live_distance, live_bearing, center_min):
        _advance_mission_state(live_color, live_distance)

    # ── Hard emergency overrides ──────────────────────────────────────────────
    laser_front_emg = center_min < FRONT_STOP_DIST
    fl_emg          = fl_val < FRONT_STOP_DIST
    fr_emg          = fr_val < FRONT_STOP_DIST
    front_emg       = laser_front_emg or fl_emg or fr_emg

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

    # ── Rear safety ───────────────────────────────────────────────────────────
    if v_cmd < 0 and (rl_val < REAR_SAFE_DIST or rr_val < REAR_SAFE_DIST):
        v_cmd     = 0.0
        omega_cmd = TARGET_ANG_VEL
        label     = "rear_blocked_turn_left"

    # ── Green-ground avoidance ────────────────────────────────────────────────
    if colors["green"] and v_cmd > 0:
        open_sign  = 1.0 if left_min >= right_min else -1.0
        green_dist = colors["green_distance"]
        if green_dist < GREEN_STOP_DIST:
            v_cmd     = 0.0
            omega_cmd = open_sign * TARGET_ANG_VEL
            label     = "green_stop_" + ("left" if open_sign > 0 else "right")
        elif green_dist < GREEN_CAUTION_DIST:
            v_cmd     = min(v_cmd, CAUTION_LIN_VEL)
            omega_cmd = open_sign * OMEGA_SMALL
            label     = "green_slow_" + ("left" if open_sign > 0 else "right")
        elif green_dist == float('inf'):
            v_cmd     = 0.0
            omega_cmd = open_sign * TARGET_ANG_VEL
            label     = "green_unknown_" + ("left" if open_sign > 0 else "right")

    # ── DONE: hold still unconditionally ─────────────────────────────────────
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
        "live_color":     live_color,
        "live_dist":      live_distance,
        "overhead_front": overhead_min,
        "block_timer":    block_timer,
        "active_target":  active_color,
        "mission_state":  _MISSION_NAMES[_mission_state],
        "nav_mode":       _NAV_NAMES[_nav_mode],
        "wp_remaining":   len(waypoint_controller.remaining_waypoints()),
    }
    return v_cmd, omega_cmd, label, block_timer, debug
