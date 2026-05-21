"""A* path planner on the occupancy grid for the Husarion RosBot.

plan(robot_pose, goal_world) returns a simplified list of world-frame
(x, y) waypoints from the robot's current cell to the goal, or None when
no path exists (goal unreachable or both start/goal out of bounds).

Occupied cells are inflated by PLANNER_INFLATE_CELLS before the search so
the robot keeps a safety margin from walls.  The goal cell itself is always
treated as passable so the robot can reach targets adjacent to walls.

Pure Python — no Webots imports.
"""

import math
import heapq

import mapping
from config import PLANNER_INFLATE_CELLS


def _build_blocked(n):
    """Return a flat bytearray of length n*n: 1 = blocked, 0 = passable.

    Indexing: blocked[ix * n + iy]
    Pass 1 collects all OCCUPIED cells.
    Pass 2 stamps a square of radius PLANNER_INFLATE_CELLS around each.
    """
    blocked = bytearray(n * n)
    occupied = []

    for ix in range(n):
        for iy in range(n):
            if mapping.get_cell(ix, iy) == mapping.OCCUPIED:
                blocked[ix * n + iy] = 1
                occupied.append((ix, iy))

    r = PLANNER_INFLATE_CELLS
    for ox, oy in occupied:
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                nx, ny = ox + dx, oy + dy
                if mapping.in_bounds(nx, ny):
                    blocked[nx * n + ny] = 1

    return blocked


def plan(robot_pose, goal_world):
    """A* from robot_pose to goal_world on the inflated occupancy grid.

    Parameters
    ----------
    robot_pose : (x_m, y_m, theta_rad)
    goal_world : (x_m, y_m)

    Returns
    -------
    list of (x_m, y_m) waypoints (world metres), or None.
    """
    if robot_pose is None or goal_world is None:
        return None

    rx, ry, _ = robot_pose
    gx_w, gy_w = goal_world

    six, siy = mapping.world_to_grid(rx, ry)
    gix, giy = mapping.world_to_grid(gx_w, gy_w)

    n, _, _ = mapping.grid_size()

    if not mapping.in_bounds(six, siy) or not mapping.in_bounds(gix, giy):
        return None

    blocked = _build_blocked(n)

    # Always allow the goal cell — target may sit next to a wall.
    blocked[gix * n + giy] = 0

    # 8-connected neighbours: (dx, dy, cost)
    NEIGHBORS = (
        ( 1,  0, 1.000), (-1,  0, 1.000), ( 0,  1, 1.000), ( 0, -1, 1.000),
        ( 1,  1, 1.414), ( 1, -1, 1.414), (-1,  1, 1.414), (-1, -1, 1.414),
    )

    open_set = [(0.0, six, siy)]
    came_from = {}
    g_score = {(six, siy): 0.0}

    while open_set:
        _, cx, cy = heapq.heappop(open_set)

        if cx == gix and cy == giy:
            return _reconstruct(came_from, cx, cy)

        cur_g = g_score.get((cx, cy), float('inf'))

        for dx, dy, step_cost in NEIGHBORS:
            nx, ny = cx + dx, cy + dy
            if not mapping.in_bounds(nx, ny):
                continue
            if blocked[nx * n + ny]:
                continue
            ng = cur_g + step_cost
            if ng < g_score.get((nx, ny), float('inf')):
                came_from[(nx, ny)] = (cx, cy)
                g_score[(nx, ny)] = ng
                h = math.hypot(nx - gix, ny - giy)
                heapq.heappush(open_set, (ng + h, nx, ny))

    return None  # No path found


def _reconstruct(came_from, cx, cy):
    """Trace came_from back to the start and return a simplified waypoint list."""
    cells = []
    node = (cx, cy)
    while node in came_from:
        cells.append(node)
        node = came_from[node]
    cells.append(node)
    cells.reverse()
    waypoints = [mapping.grid_to_world(ix, iy) for ix, iy in cells]
    return _simplify(waypoints)


def _simplify(pts, tol_rad=0.20):
    """Remove nearly-collinear intermediate waypoints to reduce tracking jitter."""
    if len(pts) <= 2:
        return pts
    result = [pts[0]]
    for i in range(1, len(pts) - 1):
        p0 = result[-1]
        p1 = pts[i]
        p2 = pts[i + 1]
        a1 = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        a2 = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
        delta = abs((a2 - a1 + math.pi) % (2.0 * math.pi) - math.pi)
        if delta > tol_rad:
            result.append(p1)
    result.append(pts[-1])
    return result
