"""Blue-then-yellow mission coordinator for the Husarion RosBot.

Ties perception into the explore -> plan -> follow -> safety stack to fulfil the
assignment goal (FR5): reach the BLUE pillar first, then the YELLOW pillar.

Pure — no Webots imports; the caller passes in the camera colour detections
(sensors.read_color_detections()).  Reuses exploration (to find a pillar when
it is not visible) and planning + following (to drive to it once seen).

State machine: SEEKING_BLUE -> SEEKING_YELLOW -> DONE.  Yellow is never sought
or accepted until blue has been reached.
"""

import math

import mapping
import planning
import following
import exploration
import perception
from config import (
    TARGET_REACHED_DIST_M, APPROACH_OFFSET_M, MISSION_REACHED_RATIO,
)

SEEKING_BLUE, SEEKING_YELLOW, DONE = 0, 1, 2
_NAMES = {SEEKING_BLUE: "SEEKING_BLUE", SEEKING_YELLOW: "SEEKING_YELLOW", DONE: "DONE"}

_state = SEEKING_BLUE
_goal_cell = None       # current approach cell for the active pillar


def state_name():
    return _NAMES.get(_state, "?")


def current_goal():
    return _goal_cell


def pillar_cells():
    """Remembered blue/yellow pillar map cells, for colored map overlays.

    Returns {"blue": (x, y) | None, "yellow": (x, y) | None}.
    """
    out = {}
    for c in ("blue", "yellow"):
        mem = perception.get_target_memory(c)
        out[c] = mapping.world_to_map(mem[0], mem[1]) if mem is not None else None
    return out


def reset():
    """Restart the mission at SEEKING_BLUE and clear all navigation state."""
    global _state, _goal_cell
    _state = SEEKING_BLUE
    _goal_cell = None
    following.reset()
    exploration.reset()
    print("[MISSION] reset -> SEEKING_BLUE")


def _advance(active):
    global _state, _goal_cell
    if _state == SEEKING_BLUE:
        _state = SEEKING_YELLOW
        print("[MISSION] BLUE reached -> SEEKING_YELLOW")
    elif _state == SEEKING_YELLOW:
        _state = DONE
        print("[MISSION] YELLOW reached -> DONE")
    _goal_cell = None
    following.reset()
    exploration.reset()      # fresh exploration memory for the next target


def _approach_cell(pose, world):
    """Map cell APPROACH_OFFSET_M in front of the pillar.

    The pillar's own cell is an obstacle in the grid, so we aim for a free cell
    a short distance in front of it along the robot->pillar line.
    """
    px, py = pose[0], pose[1]
    wx, wy = world
    dx, dy = wx - px, wy - py
    d = math.hypot(dx, dy)
    if d < 1e-6:
        return mapping.world_to_map(wx, wy)
    ax = wx - APPROACH_OFFSET_M * dx / d
    ay = wy - APPROACH_OFFSET_M * dy / d
    return mapping.world_to_map(ax, ay)


def mission_step(pose, dt, colors):
    """Advance one mission tick.  Returns (v, omega, status, debug)."""
    global _goal_cell

    debug = {"state": _NAMES[_state], "goal": _goal_cell,
             "active": None, "visible": False, "dist": float("inf")}

    # Remember BOTH pillars whenever either is seen — so a yellow spotted while
    # seeking blue is available for free later (and vice-versa).
    for c in ("blue", "yellow"):
        if colors.get(c) and colors.get(f"{c}_bearing_rad") is not None:
            perception.update_target_memory(
                c, pose, colors[f"{c}_bearing_rad"], colors.get(f"{c}_distance_m", float("inf"))
            )

    if _state == DONE:
        return 0.0, 0.0, "done", debug

    active = "blue" if _state == SEEKING_BLUE else "yellow"
    visible = bool(colors.get(active)) and colors.get(f"{active}_bearing_rad") is not None
    dist = colors.get(f"{active}_distance_m", float("inf"))
    ratio = colors.get(f"{active}_ratio", 0.0)
    debug.update({"active": active, "visible": visible, "dist": dist})

    # Reached only when we actually SEE the pillar up close (confirm, don't trust
    # a possibly-stale memory alone).
    reached = visible and (
        (math.isfinite(dist) and dist < TARGET_REACHED_DIST_M) or ratio > MISSION_REACHED_RATIO
    )
    if reached:
        _advance(active)
        return 0.0, 0.0, f"reached_{active}", debug

    # World position of the active target: live projection if visible, else the
    # remembered sighting — this is what lets us head straight to a known pillar.
    active_world = None
    if visible:
        active_world = perception.target_world_position(pose, colors[f"{active}_bearing_rad"], dist)
    if active_world is None:
        active_world = perception.get_target_memory(active)

    if active_world is not None:
        new_goal = _approach_cell(pose, active_world)
        if new_goal != _goal_cell:
            _goal_cell = new_goal
            following.reset()              # re-plan to the updated approach cell

    robot_cell = mapping.world_to_map(pose[0], pose[1])

    # Drive to the known approach cell via plan + follow.
    if _goal_cell is not None:
        if not following.has_path():
            route = planning.plan(robot_cell, _goal_cell)
            if route:
                following.set_path(route)
            else:
                _goal_cell = None          # unreachable -> fall back to exploring
        if following.has_path():
            v, w, st = following.step(pose, dt)
            if st in ("done", "stuck", "idle"):
                following.reset()
                _goal_cell = None
                # Arrived at a remembered spot but the pillar isn't actually
                # there -> the memory was stale; forget it and go explore.
                if not visible:
                    perception.forget_target(active)
                return 0.0, 0.0, f"approach_{st}_{active}", debug
            return v, w, f"going_to_{active}", debug

    # Target not visible / not reachable yet -> explore to discover it.
    v, w, ex = exploration.explore_step(pose, dt)
    if ex == "complete":
        exploration.reset()                # keep searching for the pillar
    return v, w, f"seeking_{active}", debug
