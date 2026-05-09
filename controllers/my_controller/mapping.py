"""Occupancy-grid foundation for the Husarion RosBot.

Pure Python — no Webots imports.  Defines a 2-D world-frame grid plus
coordinate-conversion helpers and two write paths:
    - mark_visited_from_pose(x, y)   stamps the robot's cell as FREE
    - update_from_laser(pose, ...)   stamps each laser hit's cell as OCCUPIED

Frame convention (matches localization.py):
    World x forward, y left.  The world origin (0, 0) — the location of
    the robot at reset_pose() — maps to the grid centre.

Cell layout:
    n_cells     = round(2 * GRID_HALF_EXTENT_M / GRID_RES_M)   square grid
    half_cells  = n_cells // 2
    ix          = round(x / res) + half_cells
    iy          = round(y / res) + half_cells
    Cell (half_cells, half_cells) is the world origin.
    grid_to_world() returns the cell *centre* in world metres.

State precedence:
    OCCUPIED beats FREE — a laser hit overrides a previous breadcrumb.
    FREE never overrides OCCUPIED — walls are sticky.
    Both override UNKNOWN.

Free-space ray-casting (FREE along the entire ray, not just at the
endpoint) is intentionally deferred to a later milestone.
"""

import math

from config import GRID_RES_M, GRID_HALF_EXTENT_M, MAX_FREE_RAY_LENGTH


UNKNOWN  = 0
FREE     = 1
OCCUPIED = 2


_n_cells    = int(round(2.0 * GRID_HALF_EXTENT_M / GRID_RES_M))
_half_cells = _n_cells // 2

_grid = [bytearray(_n_cells) for _ in range(_n_cells)]

_free_count     = 0
_occupied_count = 0

# Bounding boxes of cells in world metres, [xmin, xmax, ymin, ymax].
# Stored as a list so we can mutate in place without `global` declarations.
# Entries are None when no cell of that state has been marked.
_free_bbox     = [None, None, None, None]
_occupied_bbox = [None, None, None, None]


# ── Geometry helpers ─────────────────────────────────────────────────────────

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


# ── State writers ────────────────────────────────────────────────────────────

def _expand_bbox(bbox, x, y):
    """In-place expansion of [xmin, xmax, ymin, ymax] to include (x, y)."""
    if bbox[0] is None:
        bbox[0] = bbox[1] = x
        bbox[2] = bbox[3] = y
        return
    if x < bbox[0]: bbox[0] = x
    if x > bbox[1]: bbox[1] = x
    if y < bbox[2]: bbox[2] = y
    if y > bbox[3]: bbox[3] = y


def mark_free(ix, iy):
    """Mark cell (ix, iy) as FREE if currently UNKNOWN.

    Out-of-bounds calls are silently ignored.  FREE is idempotent.
    OCCUPIED cells are *not* downgraded — walls are sticky.
    """
    global _free_count
    if not in_bounds(ix, iy):
        return
    if _grid[ix][iy] != UNKNOWN:
        return
    _grid[ix][iy] = FREE
    _free_count += 1
    wx, wy = grid_to_world(ix, iy)
    _expand_bbox(_free_bbox, wx, wy)


def mark_occupied(ix, iy):
    """Mark cell (ix, iy) as OCCUPIED.

    Out-of-bounds calls are silently ignored.  OCCUPIED is idempotent and
    overrides FREE (laser hit beats breadcrumb).
    """
    global _free_count, _occupied_count
    if not in_bounds(ix, iy):
        return
    state = _grid[ix][iy]
    if state == OCCUPIED:
        return
    if state == FREE:
        _free_count -= 1
    _grid[ix][iy] = OCCUPIED
    _occupied_count += 1
    wx, wy = grid_to_world(ix, iy)
    _expand_bbox(_occupied_bbox, wx, wy)


def mark_visited_from_pose(x, y):
    """Mark the cell containing world position (x, y) as FREE."""
    ix, iy = world_to_grid(x, y)
    mark_free(ix, iy)


def _bresenham(ix0, iy0, ix1, iy1):
    """Yield integer (ix, iy) cells along the line from start to end, inclusive.

    Standard 2-D Bresenham; handles all 8 octants.
    """
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


def update_from_laser(pose, ranges, fov, max_range, reject_margin=0.0):
    """Project each laser ray into the grid: FREE along the path, OCCUPIED at hit.

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
        Rays with r > (max_range - reject_margin) are treated as no-hit:
        FREE is marked along the virtual ray to max_range, but no
        OCCUPIED endpoint is stamped.  Non-finite ranges are also no-hit.
        Rays with r <= GRID_RES_M are skipped entirely (noise / self-hit).

    Body→world transform per ray (frame: body x forward, y left, +theta CCW):
        alpha_i = +fov/2 - i * fov/(n - 1)        body-frame ray angle
        bx, by  = r_eff * cos(alpha_i), r_eff * sin(alpha_i)
        wx = px + bx*cos(theta) - by*sin(theta)
        wy = py + bx*sin(theta) + by*cos(theta)

    Per-ray cell walk:
        Bresenham from the robot's grid cell to the endpoint cell.
        Intermediate cells -> mark_free.
        Endpoint cell -> mark_occupied (hit) or mark_free (no-hit).
        mark_free never overrides OCCUPIED, so sticky walls survive
        re-walks of the same line on subsequent scans.

    The laser's mounting offset from robot centre is ignored: at
    GRID_RES_M = 0.10 m the worst-case bias is one cell.
    """
    if pose is None or ranges is None or fov is None or max_range is None:
        return
    n = len(ranges)
    if n < 2:
        return

    px, py, ptheta = pose
    cos_t = math.cos(ptheta)
    sin_t = math.sin(ptheta)

    half_fov   = 0.5 * fov
    angle_step = fov / (n - 1)
    threshold  = max_range - reject_margin
    min_range  = GRID_RES_M

    ix_robot, iy_robot = world_to_grid(px, py)

    # Two-pass scan to prevent NO-HIT rays from leaking FREE past walls.
    #   Pass 1: HIT rays — stamp every wall this scan sees.
    #   Pass 2: NO-HIT rays — walk Bresenham, but stop at any OCCUPIED cell.
    # Without two passes, a NO-HIT ray processed before its corresponding HIT
    # ray would walk through a not-yet-stamped wall and write FREE past it.
    hit_endpoints    = []
    no_hit_endpoints = []

    for i, r in enumerate(ranges):
        if r is None or not math.isfinite(r) or r >= threshold:
            r_eff = min(max_range, MAX_FREE_RAY_LENGTH)
            is_hit = False
        elif r <= min_range:
            continue
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

    # Pass 1 — HIT rays.  Endpoint is the wall, so Bresenham terminates
    # exactly there; intermediates FREE, endpoint OCCUPIED.
    for ix_end, iy_end in hit_endpoints:
        last_x = last_y = None
        for cx, cy in _bresenham(ix_robot, iy_robot, ix_end, iy_end):
            if last_x is not None:
                mark_free(last_x, last_y)
            last_x, last_y = cx, cy
        if last_x is not None:
            mark_occupied(last_x, last_y)

    # Pass 2 — NO-HIT rays.  Virtual endpoint is at max_range and may be
    # past actual walls; we stop the walk at the first OCCUPIED cell so
    # FREE never leaks beyond a wall.
    for ix_end, iy_end in no_hit_endpoints:
        last_x = last_y = None
        blocked = False
        for cx, cy in _bresenham(ix_robot, iy_robot, ix_end, iy_end):
            if in_bounds(cx, cy) and _grid[cx][cy] == OCCUPIED:
                blocked = True
                break
            if last_x is not None:
                mark_free(last_x, last_y)
            last_x, last_y = cx, cy
        if last_x is not None and not blocked:
            mark_free(last_x, last_y)


def clear():
    """Wipe the grid back to UNKNOWN and reset all stats."""
    global _grid, _free_count, _occupied_count
    _grid = [bytearray(_n_cells) for _ in range(_n_cells)]
    _free_count     = 0
    _occupied_count = 0
    _free_bbox[:]     = [None, None, None, None]
    _occupied_bbox[:] = [None, None, None, None]


# ── Reporting ────────────────────────────────────────────────────────────────

def _format_bbox_line(label, bbox):
    if bbox[0] is None:
        return f"      {label} bbox: empty"
    return (
        f"      {label} bbox: "
        f"x=[{bbox[0]:+.2f}, {bbox[1]:+.2f}] "
        f"y=[{bbox[2]:+.2f}, {bbox[3]:+.2f}]"
    )


def summary(robot_xy=None):
    """Format a multi-line summary of the grid state.

    Parameters
    ----------
    robot_xy : (x, y), optional
        Current robot pose in world metres.  When provided, the summary
        also reports the robot's grid index.
    """
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
