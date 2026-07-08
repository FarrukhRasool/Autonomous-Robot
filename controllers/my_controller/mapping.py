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
import threading
import zlib

import numpy as np

from config import (
    MAP_SIZE, MAP_RES_M,
    LOGODDS_INIT, LOGODDS_FREE, LOGODDS_OCC, LOGODDS_LOCK, LOGODDS_CLIP,
    P_OCC, P_FREE,
    CELL_FREE, CELL_OCC, CELL_UNKNOWN, CELL_CLOSED, CELL_GREEN,
    CELL_BLUE, CELL_YELLOW,
)


# Guards every WRITER of the shared map/particle state so the background SLAM
# mapping thread (slam.observe -> sync_from_log_odds) can't tear a concurrent
# write by a main-thread writer (mark_green / clear) or by slam.predict.  Readers
# (get_grid / there_is_obstacle) run unlocked: the grid is only ever swapped in
# atomically (rebind), so a reader always sees a complete grid, just possibly one
# cycle stale — matching the reference's unlocked-reader design.
LOCK = threading.RLock()

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


def world_points_to_map(points_world):
    """Public form of the internal Nx2 world->map projection.

    Identical formula to the reference GridMap.convert_to_map_coordinate_matrix
    (points @ [[1/res,0],[0,-1/res]].T + [HALF, HALF], int32).  Used by
    slam.py / pose_graph.py to rasterize per-particle / per-keyframe scans into
    the same cell frame as the canonical map.
    """
    return _world_points_to_map(np.asarray(points_world, dtype=np.float64))


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

def rasterize_scan(log_odds_array, robot_map_pos, map_points, map_size=MAP_SIZE):
    """Bresenham log-odds raycast update on an ARBITRARY log-odds array.

    Ported from the reference map.rasterize_scan so the raycasting math lives in
    exactly one place, shared by SLAM (slam.py / pose_graph.py, per-particle and
    per-keyframe maps).  The canonical grid is published from the best particle
    via sync_from_log_odds().

    Each ray: every cell it passes through gets FREE evidence (+= LOGODDS_FREE,
    negative), unless the cell is locked at/above LOGODDS_LOCK (a sticky wall,
    never eroded); the ray's endpoint cell gets OCC evidence (+= LOGODDS_OCC).

    Vectorised DDA form of the per-beam Bresenham loop: ray i has Chebyshev-
    distance nsteps[i] cells between robot and endpoint, so sampling s=0..nsteps[i]
    at fraction s/nsteps[i] reproduces the Bresenham cells (the dominant axis
    advances exactly one cell per step, so there are no intra-ray duplicates).
    Done as NumPy array ops it runs ~50x faster and releases the GIL — essential
    because this runs on the background SLAM thread (30x per update, and once per
    keyframe during loop-closure rebuild); a Python loop here froze the control
    loop.  Cross-ray order at a cell that is both a hit and a pass-through was
    already ambiguous in the per-beam loop (it depended on beam order), so
    applying all OCC then all lock-gated FREE stays within that same ambiguity.
    """
    pts = np.asarray(map_points, dtype=np.int64).reshape(-1, 2)
    if pts.shape[0] == 0:
        return
    rx, ry = int(robot_map_pos[0]), int(robot_map_pos[1])
    ex, ey = pts[:, 0], pts[:, 1]          # endpoints (x=col, y=row)

    # Endpoint OCC evidence (unconditional, as in AURE).
    ok = (ex >= 0) & (ex < map_size) & (ey >= 0) & (ey < map_size)
    np.add.at(log_odds_array, (ey[ok], ex[ok]), LOGODDS_OCC)

    # Free cells along each ray: s = 0 .. nsteps-1 (endpoint excluded above).
    dx = ex - rx
    dy = ey - ry
    nsteps = np.maximum(np.abs(dx), np.abs(dy))       # cells per ray
    max_steps = int(nsteps.max())
    if max_steps == 0:
        return                                        # every endpoint on the robot cell

    s = np.arange(max_steps)                          # free-sample indices
    denom = np.maximum(nsteps, 1)[:, None].astype(np.float64)
    frac = s[None, :] / denom                         # (rays, samples)
    fx = np.rint(rx + dx[:, None] * frac).astype(np.int64)
    fy = np.rint(ry + dy[:, None] * frac).astype(np.int64)

    free = s[None, :] < nsteps[:, None]               # samples belonging to each ray
    fx = fx[free]
    fy = fy[free]

    inb = (fx >= 0) & (fx < map_size) & (fy >= 0) & (fy < map_size)
    fx = fx[inb]
    fy = fy[inb]

    # Sticky-wall lock: FREE is negative, so it only ever lowers a cell — a cell
    # already at/above LOCK is never eroded.  Masking on the current value (== the
    # pre-FREE value, since FREE can't lift a cell to LOCK) matches the per-cell
    # check; duplicate pass-throughs across rays each subtract, as in the loop.
    below = log_odds_array[fy, fx] < LOGODDS_LOCK
    np.add.at(log_odds_array, (fy[below], fx[below]), LOGODDS_FREE)


def _grid_from_log_odds(log_odds, prev_grid):
    """Threshold a log-odds field into a NEW discrete grid, preserving the
    protected (GREEN / CLOSED) cells of prev_grid.  Returns a fresh array so the
    caller can swap it into place atomically (never mutating a grid a reader may
    be holding)."""
    limited = np.clip(log_odds, -LOGODDS_CLIP, LOGODDS_CLIP)
    P = 1.0 / (1.0 + np.exp(-limited))

    closed_mask = (prev_grid == CELL_CLOSED)
    green_mask = (prev_grid == CELL_GREEN)
    blue_mask = (prev_grid == CELL_BLUE)
    yellow_mask = (prev_grid == CELL_YELLOW)

    # Lidar override: a CELL_CLOSED (overhead-marked "floating wall") cell is
    # normally protected from sensor updates -- but lidar physically CANNOT see
    # a genuine floating wall at all (it sits outside the scan plane), so if
    # lidar-derived log-odds independently confirms occupancy here (P > P_OCC),
    # this was never a floating wall -- it's an ordinary wall the overhead band
    # also happened to touch, mismarked before lidar had accumulated enough
    # evidence.  Let lidar win and demote it to a normal CELL_OCC cell, whether
    # that confirmation lands before OR after the overhead mark.  A REAL
    # floating wall's footprint stays FREE under lidar (the beam passes clean
    # through underneath it), so it never crosses P_OCC and is unaffected.
    closed_lidar_confirmed = closed_mask & (P > P_OCC)
    closed_still_protected = closed_mask & ~closed_lidar_confirmed

    protected = closed_still_protected | green_mask | blue_mask | yellow_mask

    unknown_mask = (log_odds == LOGODDS_INIT) & (~protected)
    obstacle_mask = (P > P_OCC) & (~protected)   # includes closed_lidar_confirmed
    free_mask = (P < P_FREE) & (~protected)

    new_grid = np.empty_like(prev_grid)
    new_grid[obstacle_mask] = CELL_OCC
    new_grid[free_mask] = CELL_FREE
    new_grid[unknown_mask] = CELL_UNKNOWN
    # Cells that are none of the above keep their prior value (rare transitional band).
    other = ~(obstacle_mask | free_mask | unknown_mask)
    new_grid[other] = prev_grid[other]
    # Restore protected cells so sensor updates never erase them.
    new_grid[green_mask] = CELL_GREEN
    new_grid[closed_still_protected] = CELL_CLOSED
    new_grid[blue_mask] = CELL_BLUE
    new_grid[yellow_mask] = CELL_YELLOW
    return new_grid


def _update_grid_map():
    """Rebuild the canonical grid from _log_odds and swap it in atomically."""
    global _grid
    _grid = _grid_from_log_odds(_log_odds, _grid)


# ── Queries ──────────────────────────────────────────────────────────────────

def mark_green(pose, points_local):
    """Stamp projected green-ground body points as CELL_GREEN (forbidden terrain).

    pose is (x, y, theta); points_local is Nx2 body-frame [x_forward, y_left].
    Green cells are protected from later sensor overwrites by _update_grid_map(),
    so once marked they persist.
    """
    if pose is None or points_local is None:
        return
    pts = np.asarray(points_local, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] != 2:
        return
    map_points = _world_points_to_map(_transform_to_world(pts, pose))
    # Build on a copy and swap atomically under the lock, so the background SLAM
    # thread's grid swap and this main-thread write never clobber each other.
    global _grid
    with LOCK:
        new_grid = _grid.copy()
        for mx, my in map_points:
            if 0 <= mx < MAP_SIZE and 0 <= my < MAP_SIZE:
                new_grid[my, mx] = CELL_GREEN
        _grid = new_grid


def mark_overhead(pose, points_local):
    """Stamp projected floating/overhead-obstacle points as CELL_CLOSED.

    pose is (x, y, theta); points_local is Nx2 body-frame [x_forward, y_left]
    (from sensors.overhead_obstacle_points_body).  CELL_CLOSED is already
    protected from sensor-driven erasure by _grid_from_log_odds and already
    treated as a hard obstacle by there_is_obstacle / planning / following, so
    a floating wall the lidar's scan plane passes under stays mapped even as
    later lidar rays sweep the (lidar-height) free space beneath it.

    A cell already confirmed as a pillar (CELL_BLUE / CELL_YELLOW — pillars are
    tall enough to also trigger the overhead depth band) is left alone: the
    dedicated pillar colour must win over a generic obstacle mark so FR4's
    blue/yellow targets stay visible on the map instead of blending into the
    walls.

    Lidar cross-check: a cell the lidar has already confirmed as CELL_OCC is
    ALSO left alone.  Lidar cannot see a genuine floating wall at all (it sits
    outside the scan plane), so if lidar independently sees an obstacle at the
    same spot, this is an ordinary floor-to-ceiling wall the overhead band just
    happened to also pick up -- stamping CELL_CLOSED over it would be a
    redundant, uninflated overlay on an already-correct wall (CELL_CLOSED is
    deliberately excluded from planning's inflation pass, so overlapping it
    onto a CELL_OCC wall can leave a thin gap in that wall's safety margin
    right at the overlap, which is worse than just leaving CELL_OCC alone).
    """
    if pose is None or points_local is None:
        return
    pts = np.asarray(points_local, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] != 2:
        return
    map_points = _world_points_to_map(_transform_to_world(pts, pose))
    global _grid
    with LOCK:
        new_grid = _grid.copy()
        for mx, my in map_points:
            if (0 <= mx < MAP_SIZE and 0 <= my < MAP_SIZE
                    and new_grid[my, mx] not in (CELL_BLUE, CELL_YELLOW, CELL_OCC)):
                new_grid[my, mx] = CELL_CLOSED
        _grid = new_grid


def mark_pillar(cell, color, radius_cells=1):
    """Stamp a small disk around `cell` with the pillar colour code (CELL_BLUE /
    CELL_YELLOW).  Called by the mission once a pillar is confirmed within the
    registration distance.  Pillar cells are protected from sensor overwrites by
    _grid_from_log_odds and remain physical obstacles (there_is_obstacle)."""
    code = CELL_BLUE if color == "blue" else CELL_YELLOW if color == "yellow" else None
    if code is None or cell is None:
        return
    cx, cy = int(cell[0]), int(cell[1])
    global _grid
    with LOCK:
        new_grid = _grid.copy()
        for dy in range(-radius_cells, radius_cells + 1):
            for dx in range(-radius_cells, radius_cells + 1):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < MAP_SIZE and 0 <= ny < MAP_SIZE:
                    new_grid[ny, nx] = code
        _grid = new_grid


def there_is_obstacle(map_cell):
    """True if the given (map_x, map_y) cell is OCCUPIED / GREEN / CLOSED / a pillar.

    Out-of-bounds cells are treated as blocked so callers never plan off-grid.
    """
    x, y = int(map_cell[0]), int(map_cell[1])
    if not (0 <= x < MAP_SIZE and 0 <= y < MAP_SIZE):
        return True
    cell = _grid[y, x]
    return (cell == CELL_OCC or cell == CELL_GREEN or cell == CELL_CLOSED
            or cell == CELL_BLUE or cell == CELL_YELLOW)


def get_grid():
    """Return the live discrete occupancy grid (uint8, indexed [row=y, col=x])."""
    return _grid


def sync_from_log_odds(log_odds_array):
    """Replace the canonical log-odds field + discrete grid from an externally
    owned log-odds array (ported from the reference GridMap.sync_from_log_odds).

    SlamSystem publishes its best particle's map through this each observe().
    GREEN / CLOSED cells already in the grid are preserved because
    _update_grid_map() protects them.
    """
    global _log_odds
    with LOCK:
        _log_odds = np.asarray(log_odds_array, dtype=np.float32).copy()
        _update_grid_map()          # rebuilds + atomically swaps _grid


def clear():
    """Reset the log-odds field and discrete grid back to all-UNKNOWN."""
    global _log_odds, _grid
    with LOCK:
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
    img[_grid == CELL_BLUE] = (0, 0, 255)         # blue pillar (registered)
    img[_grid == CELL_YELLOW] = (255, 255, 0)     # yellow pillar (registered)

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
