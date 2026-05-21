"""Proportional waypoint-following controller for the Husarion RosBot.

Usage
-----
    waypoint_controller.set_path(waypoints)   # load a new planned path
    v, omega, label, done = waypoint_controller.follow_step(pose)

The controller turns in place until aligned with the next waypoint, then
drives forward.  It advances automatically when within WP_ACCEPT_DIST_M of
the current waypoint.

Pure Python — no Webots imports.
"""

import perception
from config import (
    WP_ACCEPT_DIST_M,
    WP_BEARING_DEADBAND_RAD,
    WP_LIN_VEL,
    WP_ANG_VEL,
)

_waypoints   = []
_current_idx = 0


def set_path(waypoints):
    """Load a new list of (wx, wy) world-frame waypoints and reset the index."""
    global _waypoints, _current_idx
    _waypoints   = list(waypoints)
    _current_idx = 0


def clear_path():
    global _waypoints, _current_idx
    _waypoints   = []
    _current_idx = 0


def has_path():
    """True when there are unvisited waypoints remaining."""
    return bool(_waypoints) and _current_idx < len(_waypoints)


def current_waypoint():
    """Return the current target (wx, wy), or None."""
    return _waypoints[_current_idx] if has_path() else None


def remaining_waypoints():
    """Return the unvisited suffix of the path (for visualisation)."""
    return _waypoints[_current_idx:] if _waypoints else []


def follow_step(pose):
    """One simulation step of waypoint following.

    Parameters
    ----------
    pose : (x_m, y_m, theta_rad)

    Returns
    -------
    (v_cmd, omega_cmd, label, done) : tuple
        done=True when the final waypoint is reached or no path is loaded.
    """
    if not has_path():
        return 0.0, 0.0, "wp_no_path", True

    global _current_idx

    # Advance past any waypoints already within acceptance radius.
    while _current_idx < len(_waypoints):
        bearing, distance = perception.bearing_distance_from_pose(
            pose, _waypoints[_current_idx]
        )
        if bearing is None:
            return 0.0, 0.0, "wp_invalid", False
        if distance <= WP_ACCEPT_DIST_M:
            _current_idx += 1
        else:
            break
    else:
        return 0.0, 0.0, "wp_done", True

    bearing, distance = perception.bearing_distance_from_pose(
        pose, _waypoints[_current_idx]
    )
    if bearing is None:
        return 0.0, 0.0, "wp_invalid", False

    # Rotate in place until aligned, then drive forward.
    if abs(bearing) > WP_BEARING_DEADBAND_RAD:
        sign = 1.0 if bearing > 0 else -1.0
        side = "left" if sign > 0 else "right"
        return 0.0, sign * WP_ANG_VEL, f"wp_turn_{side}", False

    return WP_LIN_VEL, 0.0, "wp_forward", False
