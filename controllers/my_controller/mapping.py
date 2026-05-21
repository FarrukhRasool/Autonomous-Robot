"""Occupancy-grid foundation for the Husarion RosBot.

Pure Python — no Webots imports.  Defines a 2-D world-frame grid plus
coordinate-conversion helpers and two write paths:
    - mark_visited_from_pose(x, y)   stamps the robot's cell as FREE
    - update_from_laser(pose, ...)   updates cells using probabilistic votes

Probabilistic update model
--------------------------
Each cell stores a hit_count (laser-endpoint hits) and a miss_count
(free-ray passes-through).  The derived state is:

    OCCUPIED  : hit_count >= MAP_OCC_HIT_THRESHOLD
    FREE      : any observation but hit_count < threshold
    UNKNOWN   : never observed (both counts == 0)

A cell reverts from OCCUPIED to UNKNOWN when
    miss_count >= MAP_FREE_VOTE_CLEAR * hit_count,
so noise / sensor-body reflections self-correct as the robot moves and
later rays consistently pass through an incorrectly-occupied cell.

Minimum laser range filtering (LASER_MIN_VALID_RANGE_M) prevents the
robot's own chassis from being mapped as a wall.

Frame convention (matches localization.py):
    World x forward, y left.  World origin (0, 0) at reset_pose().
    Cell (half_cells, half_cells) maps to the origin.
    grid_to_world() returns each cell's *centre* in world metres.
"""

import math

from config import (
    GRID_RES_M, GRID_HALF_EXTENT_M, MAX_FREE_RAY_LENGTH,
    LASER_MIN_VALID_RANGE_M,
    MAP_OCC_HIT_THRESHOLD, MAP_FREE_VOTE_CLEAR,
    LASER_RAY_STRIDE, MAP_MIN_SCAN_TRANS_M, MAP_MIN_SCAN_ROT_RAD,
)


UNKNOWN  = 0
FREE     = 1
OCCUPIED = 2


_n_cells    = int(round(2.0 * GRID_HALF_EXTENT_M / GRID_RES_M))
_half_cells = _n_cells // 2

# Derived state (UNKNOWN/FREE/OCCUPIED) — kept in sync with vote counts.
_grid = [bytearray(_n_cells) for _ in range(_n_cells)]

# Vote accumulators — capped at 127 to fit a signed byte.
_hit_count  = [bytearray(_n_cells) for _ in range(_n_cells)]
_miss_count = [bytearray(_n_cells) for _ in range(_n_cells)]

_free_count     = 0
_occupied_count = 0

# Bounding boxes [xmin, xmax, ymin, ymax]; None when no cell of that state.
_free_bbox     = [None, None, None, None]
_occupied_bbox = [None, None, None, None]

# Pose at which the last scan was integrated; None after clear().
_last_scan_pose = None


# ── Geometry helpers ──────────────────────────────────────────────────────────

def grid_size():
    """Return (n_cells_x, n_cells_y, half_cells)."""
    return _n_cells, _n_cells, _half_cells


def world_to_grid(x, y):
    """Convert world coords (m) to grid indices (ix, iy).  Not bounds-checked."""
    ix = int(round(x / GRID_RES_M)) + _half_cells
    iy = int(round(y / GRID_RES_M)) + _half_cells
    return ix, iy


def grid_to_world(ix, iy):
    """Return the world-frame centre (m) of grid cell (ix, iy)."""
    x = (ix - _half_cells) * GRID_RES_M
    y = (iy - _half_cells) * GRID_RES_M
    return x, y


def in_bounds(ix, iy):
    return 0 <= ix < _n_cells and 0 <= iy < _n_cells


def get_cell(ix, iy):
    """Return UNKNOWN/FREE/OCCUPIED for cell (ix, iy).

    Out-of-bounds indices return UNKNOWN; callers need no bounds check.
    """
    if not in_bounds(ix, iy):
        return UNKNOWN
    return _grid[ix][iy]


# ── Internal state updater ────────────────────────────────────────────────────

def _expand_bbox(bbox, x, y):
    if bbox[0] is None:
        bbox[0] = bbox[1] = x
        bbox[2] = bbox[3] = y
        return
    if x < bbox[0]: bbox[0] = x
    if x > bbox[1]: bbox[1] = x
    if y < bbox[2]: bbox[2] = y
    if y > bbox[3]: bbox[3] = y


def _sync_state(ix, iy):
    """Recompute _grid[ix][iy] from vote counts and update stats."""
    global _free_count, _occupied_count

    h = _hit_count[ix][iy]
    m = _miss_count[ix][iy]

    # Determine new state from votes.
    if h >= MAP_OCC_HIT_THRESHOLD:
        # Revert to UNKNOWN if free votes dominate (self-correcting noise).
        if m >= MAP_FREE_VOTE_CLEAR * h:
            new_state = UNKNOWN
        else:
            new_state = OCCUPIED
    elif h > 0 or m > 0:
        new_state = FREE
    else:
        new_state = UNKNOWN

    old_state = _grid[ix][iy]
    if old_state == new_state:
        return

    # Update counters.
    if old_state == FREE:
        _free_count -= 1
    elif old_state == OCCUPIED:
        _occupied_count -= 1

    _grid[ix][iy] = new_state
    wx, wy = grid_to_world(ix, iy)

    if new_state == FREE:
        _free_count += 1
        _expand_bbox(_free_bbox, wx, wy)
    elif new_state == OCCUPIED:
        _occupied_count += 1
        _expand_bbox(_occupied_bbox, wx, wy)


# ── Vote writers (internal) ───────────────────────────────────────────────────

def _vote_hit(ix, iy):
    """Register one laser-endpoint hit at cell (ix, iy)."""
    if not in_bounds(ix, iy):
        return
    if _hit_count[ix][iy] < 127:
        _hit_count[ix][iy] += 1
    _sync_state(ix, iy)


def _vote_free(ix, iy):
    """Register one free-ray pass-through at cell (ix, iy).

    Increments miss_count.  Also decrements hit_count so that noise hits
    self-correct when the robot later scans through the same cell cleanly.
    """
    if not in_bounds(ix, iy):
        return
    if _miss_count[ix][iy] < 127:
        _miss_count[ix][iy] += 1
    if _hit_count[ix][iy] > 0:
        _hit_count[ix][iy] -= 1
    _sync_state(ix, iy)


# ── Public state writers ──────────────────────────────────────────────────────

def mark_free(ix, iy):
    """Mark cell (ix, iy) as FREE (one miss vote).

    Out-of-bounds calls are silently ignored.
    """
    _vote_free(ix, iy)


def mark_occupied(ix, iy):
    """Mark cell (ix, iy) as OCCUPIED (one hit vote).

    The cell becomes OCCUPIED only after MAP_OCC_HIT_THRESHOLD votes.
    Out-of-bounds calls are silently ignored.
    """
    _vote_hit(ix, iy)


def mark_visited_from_pose(x, y):
    """Mark the robot's cell and immediate neighbours as FREE.

    Casts two free votes per cell so the robot's known-free footprint
    overcomes occasional noise hits near the robot's path.
    """
    ix, iy = world_to_grid(x, y)
    for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
        nx, ny = ix + dx, iy + dy
        if in_bounds(nx, ny):
            _vote_free(nx, ny)
            _vote_free(nx, ny)   # double vote: beats a single noise hit


# ── Bresenham line rasteriser ─────────────────────────────────────────────────

def _bresenham(ix0, iy0, ix1, iy1):
    """Yield integer (ix, iy) cells along the line from start to end, inclusive."""
    dx =  abs(ix1 - ix0)
    dy = -abs(iy1 - iy0)
    sx = 1 if ix0 < ix1 else -1
    sy = 1 if iy0 < iy1 else -1
    err = dx + dy
    x, y = ix0, iy0
    while True:
        yield x, y
        if x == ix1 and y == iy1:
            return
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x   += sx
        if e2 <= dx:
            err += dx
            y   += sy


# ── Laser update ──────────────────────────────────────────────────────────────

def update_from_laser(pose, ranges, fov, max_range, reject_margin=0.0):
    """Update the grid from one laser scan using probabilistic voting.

    Parameters
    ----------
    pose : (x_m, y_m, theta_rad)
        Robot pose in the (reset-anchored) world frame at scan time.
    ranges : sequence of float
        Laser ray ranges; index 0 = leftmost ray (matches reactive.py).
    fov : float
        Horizontal field of view, radians.
    max_range : float
        Sensor maximum range, metres.
    reject_margin : float
        Rays with r > (max_range - reject_margin) are no-hit: free votes
        are cast along the ray up to MAX_FREE_RAY_LENGTH.

    Filtering
    ---------
    Rays with r < LASER_MIN_VALID_RANGE_M are skipped entirely — these
    hit the robot's own chassis and would create false occupied cells.

    Motion gate
    -----------
    Skips integration when the robot has not moved at least MAP_MIN_SCAN_TRANS_M
    or rotated at least MAP_MIN_SCAN_ROT_RAD since the last integrated scan.
    This prevents noise accumulation while the robot is stationary.

    Ray stride
    ----------
    Only every LASER_RAY_STRIDE-th ray is processed.  The angular spacing
    between processed rays is still fine enough for wall detection.

    Two-pass scan
    -------------
    Pass 1: HIT rays — cast one hit vote at the endpoint; cast free votes
            along the Bresenham path from robot to wall.
    Pass 2: NO-HIT rays — cast free votes along the ray, stopping at any
            OCCUPIED cell to avoid leaking FREE past confirmed walls.

    Body→world transform per ray (body x forward, y left, +theta CCW):
        alpha_i = +fov/2 - i * fov/(n - 1)
        bx = r * cos(alpha_i),  by = r * sin(alpha_i)
        wx = px + bx*cos(theta) - by*sin(theta)
        wy = py + bx*sin(theta) + by*cos(theta)
    """
    global _last_scan_pose

    if pose is None or ranges is None or fov is None or max_range is None:
        return
    n = len(ranges)
    if n < 2:
        return

    # Motion gate: skip if robot hasn't moved enough since last integration.
    if _last_scan_pose is not None:
        lx, ly, lt = _last_scan_pose
        px, py, ptheta = pose
        trans = math.hypot(px - lx, py - ly)
        drot  = abs(ptheta - lt)
        if drot > math.pi:
            drot = 2.0 * math.pi - drot
        if trans < MAP_MIN_SCAN_TRANS_M and drot < MAP_MIN_SCAN_ROT_RAD:
            return
    _last_scan_pose = pose

    px, py, ptheta = pose
    cos_t = math.cos(ptheta)
    sin_t = math.sin(ptheta)

    half_fov   = 0.5 * fov
    angle_step = fov / (n - 1)
    threshold  = max_range - reject_margin
    min_r      = LASER_MIN_VALID_RANGE_M          # ← skip robot-body hits

    ix_robot, iy_robot = world_to_grid(px, py)

    hit_endpoints    = []
    no_hit_endpoints = []

    for i in range(0, n, LASER_RAY_STRIDE):
        r = ranges[i]
        if r is None or not math.isfinite(r) or r >= threshold:
            r_eff  = min(max_range, MAX_FREE_RAY_LENGTH)
            is_hit = False
        elif r < min_r:
            continue          # skip — almost certainly robot chassis
        else:
            r_eff  = r
            is_hit = True

        alpha = half_fov - i * angle_step
        bx = r_eff * math.cos(alpha)
        by = r_eff * math.sin(alpha)

        wx = px + bx * cos_t - by * sin_t
        wy = py + bx * sin_t + by * cos_t
        ix_end, iy_end = world_to_grid(wx, wy)

        if is_hit:
            hit_endpoints.append((ix_end, iy_end))
        else:
            no_hit_endpoints.append((ix_end, iy_end))

    # Pass 1 — HIT rays: free votes along path, hit vote at endpoint.
    for ix_end, iy_end in hit_endpoints:
        last_x = last_y = None
        for cx, cy in _bresenham(ix_robot, iy_robot, ix_end, iy_end):
            if last_x is not None:
                _vote_free(last_x, last_y)
            last_x, last_y = cx, cy
        if last_x is not None:
            _vote_hit(last_x, last_y)

    # Pass 2 — NO-HIT rays: free votes along path, stop early.
    # Stop at any cell with hit_count > 0, not just fully-OCCUPIED cells.
    # This prevents free-space from leaking through wall cells that have
    # accumulated one hit vote but haven't yet reached the OCCUPIED threshold.
    for ix_end, iy_end in no_hit_endpoints:
        last_x = last_y = None
        blocked = False
        for cx, cy in _bresenham(ix_robot, iy_robot, ix_end, iy_end):
            if in_bounds(cx, cy) and (_grid[cx][cy] == OCCUPIED
                                      or _hit_count[cx][cy] > 0):
                blocked = True
                break
            if last_x is not None:
                _vote_free(last_x, last_y)
            last_x, last_y = cx, cy
        if last_x is not None and not blocked:
            _vote_free(last_x, last_y)


# ── Grid reset ────────────────────────────────────────────────────────────────

def clear():
    """Wipe the grid back to UNKNOWN and reset all stats."""
    global _grid, _hit_count, _miss_count, _free_count, _occupied_count, _last_scan_pose
    _grid       = [bytearray(_n_cells) for _ in range(_n_cells)]
    _hit_count  = [bytearray(_n_cells) for _ in range(_n_cells)]
    _miss_count = [bytearray(_n_cells) for _ in range(_n_cells)]
    _free_count     = 0
    _occupied_count = 0
    _free_bbox[:]     = [None, None, None, None]
    _occupied_bbox[:] = [None, None, None, None]
    _last_scan_pose   = None


# ── Reporting ─────────────────────────────────────────────────────────────────

def _format_bbox_line(label, bbox):
    if bbox[0] is None:
        return f"      {label} bbox: empty"
    return (
        f"      {label} bbox: "
        f"x=[{bbox[0]:+.2f}, {bbox[1]:+.2f}] "
        f"y=[{bbox[2]:+.2f}, {bbox[3]:+.2f}]"
    )


def summary(robot_xy=None):
    """Format a multi-line summary of the grid state."""
    total   = _n_cells * _n_cells
    unknown = total - _free_count - _occupied_count

    lines = [
        f"[MAP] free={_free_count} occupied={_occupied_count} unknown={unknown}"
        f" (grid={_n_cells}x{_n_cells}, res={GRID_RES_M:.2f} m/cell)",
    ]

    if robot_xy is not None:
        rx, ry = robot_xy
        ix, iy = world_to_grid(rx, ry)
        loc = "in" if in_bounds(ix, iy) else "OUT-OF-BOUNDS"
        lines.append(
            f"      robot grid=({ix}, {iy}) world=({rx:+.3f}, {ry:+.3f}) [{loc}]"
        )

    lines.append(_format_bbox_line("free    ", _free_bbox))
    lines.append(_format_bbox_line("occupied", _occupied_bbox))
    return "\n".join(lines)
