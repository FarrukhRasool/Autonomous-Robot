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


LOCK = threading.RLock()

_HALF = MAP_SIZE // 2


_R_MAP = np.array([[1.0 / MAP_RES_M, 0.0],
                   [0.0, -1.0 / MAP_RES_M]], dtype=np.float64)
_T_MAP = np.array([_HALF, _HALF], dtype=np.float64)

_log_odds = np.full((MAP_SIZE, MAP_SIZE), LOGODDS_INIT, dtype=np.float32)
_grid = np.full((MAP_SIZE, MAP_SIZE), CELL_UNKNOWN, dtype=np.uint8)



def world_to_map(x, y):
    map_x = _HALF + int(x / MAP_RES_M)
    map_y = _HALF - int(math.ceil(y / MAP_RES_M))
    return map_x, map_y


def map_to_world(map_x, map_y):
    x = (map_x - _HALF) * MAP_RES_M
    y = (_HALF - map_y) * MAP_RES_M
    return float(x), float(y)


def robot_map_pos(pose):
    x, y, _ = pose
    return world_to_map(x, y)


def _transform_to_world(points_local, pose):
    x, y, theta = pose
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    return points_local @ R.T + np.array([x, y], dtype=np.float64)


def _world_points_to_map(points_world):
    pts = points_world @ _R_MAP.T + _T_MAP
    return pts.astype(np.int32)


def world_points_to_map(points_world):
    return _world_points_to_map(np.asarray(points_world, dtype=np.float64))


def _bresenham(x0, y0, x1, y1):
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



def rasterize_scan(log_odds_array, robot_map_pos, map_points, map_size=MAP_SIZE):
    pts = np.asarray(map_points, dtype=np.int64).reshape(-1, 2)
    if pts.shape[0] == 0:
        return
    rx, ry = int(robot_map_pos[0]), int(robot_map_pos[1])
    ex, ey = pts[:, 0], pts[:, 1]          

   
    ok = (ex >= 0) & (ex < map_size) & (ey >= 0) & (ey < map_size)
    np.add.at(log_odds_array, (ey[ok], ex[ok]), LOGODDS_OCC)

    dx = ex - rx
    dy = ey - ry
    nsteps = np.maximum(np.abs(dx), np.abs(dy))     
    max_steps = int(nsteps.max())
    if max_steps == 0:
        return                                        

    s = np.arange(max_steps)                         
    denom = np.maximum(nsteps, 1)[:, None].astype(np.float64)
    frac = s[None, :] / denom                        
    fx = np.rint(rx + dx[:, None] * frac).astype(np.int64)
    fy = np.rint(ry + dy[:, None] * frac).astype(np.int64)

    free = s[None, :] < nsteps[:, None]              
    fx = fx[free]
    fy = fy[free]

    inb = (fx >= 0) & (fx < map_size) & (fy >= 0) & (fy < map_size)
    fx = fx[inb]
    fy = fy[inb]

    below = log_odds_array[fy, fx] < LOGODDS_LOCK
    np.add.at(log_odds_array, (fy[below], fx[below]), LOGODDS_FREE)


def _grid_from_log_odds(log_odds, prev_grid):
    limited = np.clip(log_odds, -LOGODDS_CLIP, LOGODDS_CLIP)
    P = 1.0 / (1.0 + np.exp(-limited))

    closed_mask = (prev_grid == CELL_CLOSED)
    green_mask = (prev_grid == CELL_GREEN)
    blue_mask = (prev_grid == CELL_BLUE)
    yellow_mask = (prev_grid == CELL_YELLOW)

    closed_lidar_confirmed = closed_mask & (P > P_OCC)
    closed_still_protected = closed_mask & ~closed_lidar_confirmed

    protected = closed_still_protected | green_mask | blue_mask | yellow_mask

    unknown_mask = (log_odds == LOGODDS_INIT) & (~protected)
    obstacle_mask = (P > P_OCC) & (~protected)  
    free_mask = (P < P_FREE) & (~protected)

    new_grid = np.empty_like(prev_grid)
    new_grid[obstacle_mask] = CELL_OCC
    new_grid[free_mask] = CELL_FREE
    new_grid[unknown_mask] = CELL_UNKNOWN
    other = ~(obstacle_mask | free_mask | unknown_mask)
    new_grid[other] = prev_grid[other]

    new_grid[green_mask] = CELL_GREEN
    new_grid[closed_still_protected] = CELL_CLOSED
    new_grid[blue_mask] = CELL_BLUE
    new_grid[yellow_mask] = CELL_YELLOW
    return new_grid


def _update_grid_map():
    global _grid
    _grid = _grid_from_log_odds(_log_odds, _grid)


# ── Queries ──────────────────────────────────────────────────────────────────

def mark_green(pose, points_local):
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
            if 0 <= mx < MAP_SIZE and 0 <= my < MAP_SIZE:
                new_grid[my, mx] = CELL_GREEN
        _grid = new_grid


def mark_overhead(pose, points_local):
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
    x, y = int(map_cell[0]), int(map_cell[1])
    if not (0 <= x < MAP_SIZE and 0 <= y < MAP_SIZE):
        return True
    cell = _grid[y, x]
    return (cell == CELL_OCC or cell == CELL_GREEN or cell == CELL_CLOSED
            or cell == CELL_BLUE or cell == CELL_YELLOW)


def get_grid():
    return _grid


def sync_from_log_odds(log_odds_array):
    global _log_odds
    with LOCK:
        _log_odds = np.asarray(log_odds_array, dtype=np.float32).copy()
        _update_grid_map()          # rebuilds + atomically swaps _grid


def clear():
    global _log_odds, _grid
    with LOCK:
        _log_odds = np.full((MAP_SIZE, MAP_SIZE), LOGODDS_INIT, dtype=np.float32)
        _grid = np.full((MAP_SIZE, MAP_SIZE), CELL_UNKNOWN, dtype=np.uint8)


# ── Reporting ────────────────────────────────────────────────────────────────

def summary(robot_xy=None):
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
    img = np.zeros((MAP_SIZE, MAP_SIZE, 3), dtype=np.uint8)
    img[_grid == CELL_UNKNOWN] = (128, 128, 128)
    img[_grid == CELL_FREE] = (255, 255, 255)
    img[_grid == CELL_OCC] = (0, 0, 0)
    img[_grid == CELL_CLOSED] = (128, 0, 128)
    img[_grid == CELL_GREEN] = (0, 180, 0)
    img[_grid == CELL_BLUE] = (0, 0, 255)       
    img[_grid == CELL_YELLOW] = (255, 255, 0)    

    if path_cells:
        for (x, y) in path_cells:
            if 0 <= x < MAP_SIZE and 0 <= y < MAP_SIZE:
                img[y, x] = (255, 0, 0)    

    def _dot(cell, rgb):
        if cell is None:
            return
        cx, cy = int(cell[0]), int(cell[1])
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < MAP_SIZE and 0 <= ny < MAP_SIZE:
                    img[ny, nx] = rgb
    _dot(blue_cell, (0, 0, 255))            
    _dot(yellow_cell, (255, 255, 0))        

    h, w, _ = img.shape
    raw = bytearray()
    for y in range(h):
        raw.append(0)                         
        raw += img[y].tobytes()                

    png = (b"\x89PNG\r\n\x1a\n"
           + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))  
           + _png_chunk(b"IDAT", zlib.compress(bytes(raw), 6))
           + _png_chunk(b"IEND", b""))
    with open(path, "wb") as f:
        f.write(png)
    return path
