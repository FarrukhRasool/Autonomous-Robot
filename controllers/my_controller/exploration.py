"""Frontier-based exploration for the Husarion RosBot.

Ported from AURE (GridMap.detect_frontiers / _cluster_frontiers_bfs +
select_frontier_target), adapted into a step-wise coordinator that fits this
project's single control loop: explore_step() is called once per tick and
manages goal selection, planning (planning.plan), following (following.step),
and a visited-frontier memory to avoid re-picking the same region.

Pure — no Webots imports.  Frontiers are FREE cells bordering UNKNOWN space;
the best is chosen by information-gain utility size / (distance + bias).
"""

import math
import random
from collections import deque

import numpy as np

import mapping
import planning
import following
from config import (
    CELL_FREE, CELL_UNKNOWN,
    FRONTIER_MIN_CLUSTER, FRONTIER_MIN_SIZE, FRONTIER_MIN_DIST_PX,
    FRONTIER_VISITED_RADIUS_PX, FRONTIER_SCORE_BIAS,
    RECOVERY_REVERSE_TICKS, RECOVERY_TURN_TICKS,
    RECOVERY_REVERSE_VEL, RECOVERY_TURN_OMEGA,
    FOLLOW_PROGRESS_WINDOW, FOLLOW_MIN_PROGRESS_PX,
)

# ── Coordinator state ─────────────────────────────────────────────────────────
_visited = []          # centroids of frontiers already attempted (np.array cells)
_current_goal = None   # (x, y) frontier currently being driven to
_recovery_ticks = 0    # >0 while running a reverse+turn un-wedge maneuver
_recovery_sign = 1.0   # turn direction chosen for the current recovery
_progress_counter = 0  # ticks since the last progress check
_window_start_dist = None   # goal distance at the start of the current window


# ── Frontier detection ────────────────────────────────────────────────────────

def detect_frontiers(grid):
    """Return clustered frontier regions (lists of (x, y) cells).

    A frontier cell is FREE and 4-adjacent to UNKNOWN.  The boundary mask is
    found vectorised (fast), then grouped into 8-connected clusters of at least
    FRONTIER_MIN_CLUSTER cells (BFS, as in AURE).
    """
    free = (grid == CELL_FREE)
    unknown = (grid == CELL_UNKNOWN)

    adj_unknown = np.zeros_like(unknown)
    adj_unknown[1:, :] |= unknown[:-1, :]    # neighbour above is unknown
    adj_unknown[:-1, :] |= unknown[1:, :]    # below
    adj_unknown[:, 1:] |= unknown[:, :-1]    # left
    adj_unknown[:, :-1] |= unknown[:, 1:]    # right

    ys, xs = np.where(free & adj_unknown)
    cells = list(zip(xs.tolist(), ys.tolist()))
    return _cluster_bfs(cells, FRONTIER_MIN_CLUSTER)


def _cluster_bfs(cells, min_size):
    """Group frontier cells into 8-connected clusters; drop clusters < min_size."""
    if not cells:
        return []
    cell_set = set(cells)
    visited = set()
    clusters = []
    n8 = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    for start in cells:
        if start in visited:
            continue
        cluster = []
        queue = deque([start])
        visited.add(start)
        while queue:
            x, y = queue.popleft()
            cluster.append((x, y))
            for dx, dy in n8:
                nb = (x + dx, y + dy)
                if nb in cell_set and nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        if len(cluster) >= min_size:
            clusters.append(cluster)
    return clusters


# ── Frontier selection (information-gain utility) ─────────────────────────────

def _near_visited(centroid):
    return any(np.linalg.norm(centroid - v) < FRONTIER_VISITED_RADIUS_PX
               for v in _visited)


def select_frontier(grid, robot_cell):
    """Pick the best unvisited frontier centroid, or None if none remain.

    Primary: maximise size / (distance + bias), skipping small / too-close /
    recently-visited regions.  Fallback: nearest unvisited region centroid.
    """
    regions = detect_frontiers(grid)
    if not regions:
        return None

    robot = np.asarray(robot_cell, dtype=float)
    best = None
    best_score = -float("inf")

    for region in regions:
        cells = np.array(region)
        centroid = cells.mean(axis=0)
        dist = np.linalg.norm(centroid - robot)
        size = len(region)
        if size < FRONTIER_MIN_SIZE or dist < FRONTIER_MIN_DIST_PX:
            continue
        if _near_visited(centroid):
            continue
        score = size / (dist + FRONTIER_SCORE_BIAS)
        if score > best_score:
            best_score = score
            best = (int(centroid[0]), int(centroid[1]))

    if best is not None:
        return best

    # Fallback: nearest region we haven't visited yet (ignore size/dist filters).
    fallback = None
    fallback_dist = float("inf")
    for region in regions:
        cells = np.array(region)
        centroid = cells.mean(axis=0)
        if _near_visited(centroid):
            continue
        dist = np.linalg.norm(centroid - robot)
        if dist < fallback_dist:
            fallback_dist = dist
            fallback = (int(centroid[0]), int(centroid[1]))
    return fallback


# ── Step-wise coordinator ─────────────────────────────────────────────────────

def current_goal():
    return _current_goal


def reset():
    """Clear exploration memory and stop any active follow."""
    global _visited, _current_goal, _recovery_ticks
    global _progress_counter, _window_start_dist
    _visited = []
    _current_goal = None
    _recovery_ticks = 0
    _progress_counter = 0
    _window_start_dist = None
    following.reset()


def _rear_clear(pose):
    """True if the map shows no obstacle just behind the robot (safe to reverse)."""
    x, y, theta = pose
    for d in (0.06, 0.12, 0.18):
        bx = x - d * math.cos(theta)
        by = y - d * math.sin(theta)
        if mapping.there_is_obstacle(mapping.world_to_map(bx, by)):
            return False
    return True


def explore_step(pose, dt):
    """Advance one exploration tick.  Returns (v, omega, status).

    status: "exploring" (driving to a frontier), "replanning" (selecting the
    next frontier this tick), or "complete" (no reachable frontiers left).
    """
    global _current_goal, _recovery_ticks, _recovery_sign
    global _progress_counter, _window_start_dist

    # ── Active recovery maneuver: reverse (if rear clear) then turn ────────────
    if _recovery_ticks > 0:
        _recovery_ticks -= 1
        reversing = _recovery_ticks >= RECOVERY_TURN_TICKS
        if reversing and _rear_clear(pose):
            return RECOVERY_REVERSE_VEL, 0.0, "recovering"
        return 0.0, _recovery_sign * RECOVERY_TURN_OMEGA, "recovering"

    x, y, _theta = pose
    robot_cell = mapping.world_to_map(x, y)

    # Need a new frontier + path?
    if not following.has_path():
        goal = select_frontier(mapping.get_grid(), robot_cell)
        if goal is None and _visited:
            # The visited list may be over-suppressing still-open frontiers;
            # forget it and try once more before declaring the map explored.
            _visited.clear()
            goal = select_frontier(mapping.get_grid(), robot_cell)
        if goal is None:
            return 0.0, 0.0, "complete"
        route = planning.plan(robot_cell, goal)
        if not route:
            # Unreachable this attempt: remember it and try another next tick.
            _visited.append(np.asarray(goal, dtype=float))
            _current_goal = None
            return 0.0, 0.0, "replanning"
        _current_goal = goal
        following.set_path(route)
        _progress_counter = 0
        _window_start_dist = None

    v, w, st = following.step(pose, dt)
    if st == "stuck":
        # Wedged: remember this goal, drop the path, and start a recovery maneuver.
        if _current_goal is not None:
            _visited.append(np.asarray(_current_goal, dtype=float))
        _current_goal = None
        following.reset()
        _recovery_sign = random.choice([-1.0, 1.0])
        _recovery_ticks = RECOVERY_REVERSE_TICKS + RECOVERY_TURN_TICKS
        return (RECOVERY_REVERSE_VEL if _rear_clear(pose) else 0.0), 0.0, "recovering"

    if st in ("done", "idle"):
        if _current_goal is not None:
            _visited.append(np.asarray(_current_goal, dtype=float))
        _current_goal = None
        following.reset()
        return 0.0, 0.0, "replanning"

    # Progress watchdog: if we haven't gotten meaningfully closer to the goal
    # over a window, the path is stale (a wall was discovered on it) — drop it
    # and replan against the now-updated map (do NOT blacklist the frontier).
    if _current_goal is not None:
        d = math.hypot(robot_cell[0] - _current_goal[0], robot_cell[1] - _current_goal[1])
        _progress_counter += 1
        if _window_start_dist is None:
            _window_start_dist = d
        elif _progress_counter >= FOLLOW_PROGRESS_WINDOW:
            if d > _window_start_dist - FOLLOW_MIN_PROGRESS_PX:
                following.reset()
                _current_goal = None
                _progress_counter = 0
                _window_start_dist = None
                return 0.0, 0.0, "replanning"
            _window_start_dist = d
            _progress_counter = 0

    return v, w, "exploring"
