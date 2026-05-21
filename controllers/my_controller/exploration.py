"""Frontier-based exploration for the Husarion RosBot.

A frontier is a FREE cell with at least one UNKNOWN 4-connected neighbour.
nearest_frontier() returns the world-frame centre of the closest such cell
so the navigation layer can treat it as a temporary goal while the map is
being built.

Pure Python — no Webots imports.
"""

import math
import mapping

# Ignore frontiers closer than this — the robot is already there.
_MIN_FRONTIER_DIST_M = 0.40


def _iter_frontiers():
    """Yield (ix, iy) for every frontier cell in the current grid."""
    n, _, _ = mapping.grid_size()
    for ix in range(1, n - 1):
        for iy in range(1, n - 1):
            if mapping.get_cell(ix, iy) != mapping.FREE:
                continue
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if mapping.get_cell(ix + dx, iy + dy) == mapping.UNKNOWN:
                    yield ix, iy
                    break


def nearest_frontier(robot_x, robot_y):
    """Return world-frame (wx, wy) of the nearest frontier, or None.

    Returns None when the map is fully explored (no frontiers remain)
    or when all frontiers are closer than _MIN_FRONTIER_DIST_M.
    """
    best_dist = float('inf')
    best_world = None

    for ix, iy in _iter_frontiers():
        wx, wy = mapping.grid_to_world(ix, iy)
        d = math.hypot(wx - robot_x, wy - robot_y)
        if _MIN_FRONTIER_DIST_M <= d < best_dist:
            best_dist = d
            best_world = (wx, wy)

    return best_world


def frontier_count():
    """Return the total number of frontier cells (for diagnostics)."""
    return sum(1 for _ in _iter_frontiers())
