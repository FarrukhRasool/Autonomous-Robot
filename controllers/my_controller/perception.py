"""World-frame perception transforms for the Husarion RosBot.

Pure Python — no Webots imports.  Composes pose, body-frame measurements,
and a uniform body→world transform into world-frame estimates.

Frame convention (matches localization.py and mapping.py):
    Body x forward, y left, z up; +theta yaws CCW (left).
    World origin (0, 0) at reset_pose() location.
"""

import math

import sensors
import localization
import mapping
from config import (
    MISSION_MARK_RATIO, MISSION_MARK_LASER_M, MISSION_MARK_BEARING_RAD,
    MISSION_MARK_MAX_STAMP_M,
)


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

# True once a color has been CONFIRMED at least once (pillar_confirmed) — after
# that, tag_pillars_on_map freezes _target_memory at the reliable stamped
# position instead of continuing to overwrite it with noisy live depth
# sightings, since re-planning/backtracking to a pillar relies on the memory
# matching the position actually stamped on the map.
_target_confirmed = {"blue": False, "yellow": False}


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


def forget_target(color):
    """Drop the stored sighting for one color (e.g. a stale/wrong memory).

    Also clears the confirmed flag, so tag_pillars_on_map resumes updating
    the memory from live sightings again (otherwise a forgotten pillar could
    never be re-acquired, since a confirmed color's memory is normally frozen).
    """
    if color in _target_memory:
        _target_memory[color] = None
    if color in _target_confirmed:
        _target_confirmed[color] = False


def reset_target_memory():
    """Clear all stored sightings (called by the R keypress)."""
    for key in _target_memory:
        _target_memory[key] = None
    for key in _target_confirmed:
        _target_confirmed[key] = False


# ── Pillar recognition + map colour-tagging (FR4) ────────────────────────────
# Mode-independent: safe to call from teleop, G/Y/E, or mission.run() alike, so
# blue/yellow always show up as their own colour on the map (mapping.CELL_BLUE/
# CELL_YELLOW) instead of a generic obstacle, regardless of which mode mapped
# them.  mission.py layers its own SEEKING_BLUE/YELLOW sequencing state on top
# of pillar_confirmed(); it does not duplicate this confirm logic.

def pillar_confirmed(color, colors):
    """True only if `color` is CONFIRMED genuinely at the pillar — via signals
    that don't depend on the unreliable depth-Pythagoras distance:
      1. the mask fills MISSION_MARK_RATIO of the frame (pillar dominates view), OR
      2. the front laser is at contact range (< MISSION_MARK_LASER_M) while the
         pillar is centred (|bearing| < MISSION_MARK_BEARING_RAD) — a wall can't
         trigger it because the pillar must be visible AND pointing forward.
    (The depth distance is deliberately NOT used here: it under-estimates far
    pillars badly, which caused false "reached" marks from across the map.)
    """
    if not colors.get(color):
        return False
    ratio = colors.get(f"{color}_ratio", 0.0)
    bearing = colors.get(f"{color}_bearing_rad")
    if ratio >= MISSION_MARK_RATIO:
        return True
    if (bearing is not None and abs(bearing) < MISSION_MARK_BEARING_RAD
            and sensors.get_front_laser_min() < MISSION_MARK_LASER_M):
        return True
    return False


def tag_pillars_on_map(colors=None):
    """Recognize blue/yellow pillars and stamp the confirmed one's cell on the
    occupancy grid in its true colour (mapping.mark_pillar).

    Updates the world-frame sighting memory for any visible pillar (used to
    navigate toward it) with the live, noisy depth sighting only UNTIL that
    pillar is confirmed for the first time; once confirmed, the memory is
    frozen at the accurate stamped (laser) position below instead, so a
    still-noisy live glimpse can't drag an already-registered pillar's
    remembered position off the spot actually marked on the map (which
    backtracking / re-planning rely on).  The map itself is stamped only once
    CONFIRMED (pillar_confirmed) — a distant glimpse is never coloured on the
    map.
    """
    if colors is None:
        colors = sensors.read_color_detections()
    pose = localization.get_pose()
    for color in ("blue", "yellow"):
        if not colors.get(color):
            continue
        bearing = colors.get(f"{color}_bearing_rad")
        dist = colors.get(f"{color}_distance_m", float("inf"))
        if bearing is None or not math.isfinite(dist):
            continue
        if not _target_confirmed[color]:
            update_target_memory(color, pose, bearing, dist)
        if pillar_confirmed(color, colors):
            # Stamp at the reliable close range (front laser, clamped),
            # directly ahead — the pillar is centred at this moment — NOT the
            # bad depth distance.
            fd = sensors.get_front_laser_min()
            stamp_d = fd if math.isfinite(fd) and fd <= MISSION_MARK_MAX_STAMP_M else MISSION_MARK_MAX_STAMP_M
            world = target_world_position(pose, bearing, stamp_d)
            if world is not None:
                mapping.mark_pillar(mapping.world_to_map(world[0], world[1]), color)
                # Freeze memory at this reliable stamped position so it matches
                # the map stamp (and the drawn circle) from now on.
                update_target_memory(color, pose, bearing, stamp_d)
            _target_confirmed[color] = True
