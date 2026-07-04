"""Log-odds occupancy grid for the Husarion RosBot.

Ported from the AURE `GridMap` occupancy core.  Pure NumPy — no Webots
imports; the robot pose and the lidar point cloud are passed in.  A single
finalised mapping approach: probabilistic log-odds accumulation from lidar
endpoints, thresholded into a discrete grid the downstream planner/DWA read.

Frame convention (matches localization.py):
    World x forward, y left, +theta CCW.  World origin (0, 0) is the robot
    pose at reset.  The grid is MAP_SIZE x MAP_SIZE cells, origin at the grid
    centre, indexed [row = map_y, col = map_x] with the y-axis flipped:
        map_x = HALF + int(x / res)
        map_y = HALF - ceil(y / res)

Log-odds model (per lidar scan, Bresenham from the robot cell to each hit):
    cells the ray passes through  += LOGODDS_FREE   (negative — free evidence)
    the ray's endpoint cell       += LOGODDS_OCC    (positive — hit evidence)
    cells at/above LOGODDS_LOCK are frozen (sticky walls): free updates skip
    them so a confirmed wall is never eroded by later rays.

Discrete grid (P = sigmoid(clip(log_odds))):
    P > P_OCC  -> CELL_OCC
    P < P_FREE -> CELL_FREE
    log_odds still at LOGODDS_INIT (never touched) -> CELL_UNKNOWN
    CELL_CLOSED / CELL_GREEN cells are protected from sensor overwrites
    (reserved for later milestones; inert until those cells are ever written).
"""

import math
import struct
import zlib

import numpy as np

from config import (
    MAP_SIZE, MAP_RES_M,
    LOGODDS_INIT, LOGODDS_FREE, LOGODDS_OCC, LOGODDS_LOCK, LOGODDS_CLIP,
    P_OCC, P_FREE,
    CELL_FREE, CELL_OCC, CELL_UNKNOWN, CELL_CLOSED, CELL_GREEN,
)


_HALF = MAP_SIZE // 2

# World->map linear part (scale + y-flip) for vectorised point-cloud projection.
_R_MAP = np.array([[1.0 / MAP_RES_M, 0.0],
                   [0.0, -1.0 / MAP_RES_M]], dtype=np.float64)
_T_MAP = np.array([_HALF, _HALF], dtype=np.float64)

_log_odds = np.full((MAP_SIZE, MAP_SIZE), LOGODDS_INIT, dtype=np.float32)
_grid = np.full((MAP_SIZE, MAP_SIZE), CELL_UNKNOWN, dtype=np.uint8)


# ── Coordinate helpers ───────────────────────────────────────────────────────

def world_to_map(x, y):
    """World metres -> integer (map_x, map_y) cell (matches AURE get_map_position)."""
    map_x = _HALF + int(x / MAP_RES_M)
    map_y = _HALF - int(math.ceil(y / MAP_RES_M))
    return map_x, map_y


def map_to_world(map_x, map_y):
    """Integer cell -> world-frame metres at the cell origin."""
    x = (map_x - _HALF) * MAP_RES_M
    y = (_HALF - map_y) * MAP_RES_M
    return float(x), float(y)


def robot_map_pos(pose):
    """(x, y, theta) world pose -> the robot's (map_x, map_y) cell."""
    x, y, _ = pose
    return world_to_map(x, y)


def _transform_to_world(points_local, pose):
    """Rotate/translate Nx2 body-frame points [x_fwd, y_left] into the world frame."""
    x, y, theta = pose
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    return points_local @ R.T + np.array([x, y], dtype=np.float64)


def _world_points_to_map(points_world):
    """Vectorised world-metre points -> Nx2 int32 map cells (scale + y-flip)."""
    pts = points_world @ _R_MAP.T + _T_MAP
    return pts.astype(np.int32)


def _bresenham(x0, y0, x1, y1):
    """Integer grid cells from (x0,y0) to (x1,y1) inclusive (AURE bresenham_line)."""
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0
    while True:
        points.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    return points


# ── Occupancy update ─────────────────────────────────────────────────────────

def _update_log_odds(robot_map, map_points):
    """Accumulate log-odds along each ray: free on the path, occupied at the hit."""
    rx, ry = int(robot_map[0]), int(robot_map[1])
    for mx, my in map_points:
        cells = _bresenham(rx, ry, int(mx), int(my))
        # Free evidence for every cell the ray passes through (all but the hit).
        for (cx, cy) in cells[:-1]:
            if 0 <= cx < MAP_SIZE and 0 <= cy < MAP_SIZE:
                if _log_odds[cy, cx] < LOGODDS_LOCK:   # sticky-wall lock
                    _log_odds[cy, cx] += LOGODDS_FREE
        # Occupied evidence for the endpoint cell (unconditional, as in AURE).
        ex, ey = cells[-1]
        if 0 <= ex < MAP_SIZE and 0 <= ey < MAP_SIZE:
            _log_odds[ey, ex] += LOGODDS_OCC


def _update_grid_map():
    """Threshold the log-odds field into the discrete grid, protecting special cells."""
    limited = np.clip(_log_odds, -LOGODDS_CLIP, LOGODDS_CLIP)
    P = 1.0 / (1.0 + np.exp(-limited))

    closed_mask = (_grid == CELL_CLOSED)
    green_mask = (_grid == CELL_GREEN)
    protected = closed_mask | green_mask

    unknown_mask = (_log_odds == LOGODDS_INIT) & (~protected)
    obstacle_mask = (P > P_OCC) & (~protected)
    free_mask = (P < P_FREE) & (~protected)

    _grid[obstacle_mask] = CELL_OCC
    _grid[free_mask] = CELL_FREE
    _grid[unknown_mask] = CELL_UNKNOWN

    # Restore protected cells so sensor updates never erase them.
    _grid[green_mask] = CELL_GREEN
    _grid[closed_mask] = CELL_CLOSED


def lidar_update(pose, points_local):
    """Update the occupancy grid from one lidar scan.

    Parameters
    ----------
    pose : (x_m, y_m, theta_rad)
        Robot pose in the reset-anchored world frame at scan time.
    points_local : array-like, shape (N, 2)
        Lidar points in the robot body frame as [x_forward, y_left], metres,
        already inf/nan-filtered.  Empty input is a no-op.
    """
    if pose is None or points_local is None:
        return
    pts = np.asarray(points_local, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] != 2:
        return

    points_world = _transform_to_world(pts, pose)
    map_points = _world_points_to_map(points_world)
    _update_log_odds(robot_map_pos(pose), map_points)
    _update_grid_map()


# ── Queries ──────────────────────────────────────────────────────────────────

def mark_green(pose, points_local):
    """Stamp projected green-ground body points as CELL_GREEN (forbidden terrain).

    Parameters mirror lidar_update: pose is (x, y, theta); points_local is Nx2
    body-frame [x_forward, y_left].  Green cells are protected from later sensor
    overwrites by update_grid_map(), so once marked they persist.
    """
    if pose is None or points_local is None:
        return
    pts = np.asarray(points_local, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] != 2:
        return
    map_points = _world_points_to_map(_transform_to_world(pts, pose))
    for mx, my in map_points:
        if 0 <= mx < MAP_SIZE and 0 <= my < MAP_SIZE:
            _grid[my, mx] = CELL_GREEN


def mark_robot_free(pose, radius_cells=2):
    """Force the robot's own footprint to FREE, correcting phantom walls.

    The robot is physically at this pose, so any OCCUPIED cell under its body
    must be spurious (e.g. from odometry slip while wedged).  We reset those
    cells' log-odds to strongly-free — this also clears the sticky-wall lock, so
    driving back through a distorted area now repairs it.  GREEN / CLOSED
    semantic markers are preserved.
    """
    if pose is None:
        return
    mx, my = world_to_map(pose[0], pose[1])
    r2 = radius_cells * radius_cells
    for dy in range(-radius_cells, radius_cells + 1):
        for dx in range(-radius_cells, radius_cells + 1):
            if dx * dx + dy * dy > r2:
                continue
            cx, cy = mx + dx, my + dy
            if not (0 <= cx < MAP_SIZE and 0 <= cy < MAP_SIZE):
                continue
            if _grid[cy, cx] == CELL_GREEN or _grid[cy, cx] == CELL_CLOSED:
                continue
            _log_odds[cy, cx] = -LOGODDS_CLIP      # strongly free; clears any lock
            _grid[cy, cx] = CELL_FREE


def there_is_obstacle(map_cell):
    """True if the given (map_x, map_y) cell is OCCUPIED / GREEN / CLOSED.

    Out-of-bounds cells are treated as blocked so callers never plan off-grid.
    """
    x, y = int(map_cell[0]), int(map_cell[1])
    if not (0 <= x < MAP_SIZE and 0 <= y < MAP_SIZE):
        return True
    cell = _grid[y, x]
    return cell == CELL_OCC or cell == CELL_GREEN or cell == CELL_CLOSED


def get_grid():
    """Return the live discrete occupancy grid (uint8, indexed [row=y, col=x])."""
    return _grid


def clear():
    """Reset the log-odds field and discrete grid back to all-UNKNOWN."""
    global _log_odds, _grid
    _log_odds = np.full((MAP_SIZE, MAP_SIZE), LOGODDS_INIT, dtype=np.float32)
    _grid = np.full((MAP_SIZE, MAP_SIZE), CELL_UNKNOWN, dtype=np.uint8)


# ── Reporting ────────────────────────────────────────────────────────────────

def summary(robot_xy=None):
    """Format a one/two-line summary of the grid state for the Webots console."""
    occ = int(np.count_nonzero(_grid == CELL_OCC))
    free = int(np.count_nonzero(_grid == CELL_FREE))
    unknown = int(np.count_nonzero(_grid == CELL_UNKNOWN))

    lines = [
        f"[MAP] free={free} occupied={occ} unknown={unknown} "
        f"(grid={MAP_SIZE}x{MAP_SIZE}, res={MAP_RES_M:.4f} m/cell)",
    ]
    if robot_xy is not None:
        rx, ry = robot_xy
        mx, my = world_to_map(rx, ry)
        inb = 0 <= mx < MAP_SIZE and 0 <= my < MAP_SIZE
        lines.append(
            f"      robot map=({mx}, {my}) world=({rx:+.3f}, {ry:+.3f}) "
            f"[{'in' if inb else 'OUT-OF-BOUNDS'}]"
        )
    return "\n".join(lines)


def _png_chunk(tag, payload):
    return (struct.pack(">I", len(payload)) + tag + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xffffffff))


def save_png(path="map.png", path_cells=None, blue_cell=None, yellow_cell=None):
    """Dump the grid as a viewable RGB PNG (pure stdlib, no OpenCV).

    OBSTACLE->black, FREESPACE->white, UNKNOWN->gray, GREEN->green, CLOSED->purple.
    The route (if given) is drawn red; the blue/yellow pillars in their own
    colours.  PNG opens in Preview/QuickLook.  Returns the path.
    """
    img = np.zeros((MAP_SIZE, MAP_SIZE, 3), dtype=np.uint8)
    img[_grid == CELL_UNKNOWN] = (128, 128, 128)
    img[_grid == CELL_FREE] = (255, 255, 255)
    img[_grid == CELL_OCC] = (0, 0, 0)
    img[_grid == CELL_CLOSED] = (128, 0, 128)
    img[_grid == CELL_GREEN] = (0, 180, 0)

    if path_cells:
        for (x, y) in path_cells:
            if 0 <= x < MAP_SIZE and 0 <= y < MAP_SIZE:
                img[y, x] = (255, 0, 0)       # red route

    def _dot(cell, rgb):
        if cell is None:
            return
        cx, cy = int(cell[0]), int(cell[1])
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < MAP_SIZE and 0 <= ny < MAP_SIZE:
                    img[ny, nx] = rgb
    _dot(blue_cell, (0, 0, 255))              # blue pillar
    _dot(yellow_cell, (255, 255, 0))          # yellow pillar

    h, w, _ = img.shape
    raw = bytearray()
    for y in range(h):
        raw.append(0)                          # PNG per-row filter type 0 (None)
        raw += img[y].tobytes()                # RGB row (R,G,B per pixel)

    png = (b"\x89PNG\r\n\x1a\n"
           + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))  # colour type 2 = RGB
           + _png_chunk(b"IDAT", zlib.compress(bytes(raw), 6))
           + _png_chunk(b"IEND", b""))
    with open(path, "wb") as f:
        f.write(png)
    return path
