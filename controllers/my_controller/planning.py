import heapq
import math

import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import splprep, splev

import mapping
from config import (
    CELL_FREE, CELL_OCC, CELL_CLOSED, CELL_GREEN, CELL_UNKNOWN,
    ASTAR_INFLATION_LEVELS, ASTAR_EXPANSION_PIXELS, PATH_MIN_LENGTH_M,
    ASTAR_SAFE_DISTANCE_PX, ASTAR_PENALTY_STRENGTH, ASTAR_HEURISTIC_WEIGHT,
    PLAN_BLOCK_UNKNOWN, ASTAR_FRONTIER_INFLATION,
)


def _clean_small_components(grid, min_size=6, connectivity=4):
    binary = np.where(grid == CELL_OCC, 255, 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=connectivity)
    keep = np.zeros_like(binary)
    for lbl in range(1, n):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_size:
            keep[labels == lbl] = 255
    out = grid.copy()
    out[(keep == 0) & (grid == CELL_OCC)] = CELL_FREE
    return out


def _remove_noisy_pixels(grid, connectivity=4):
    binary = np.where(grid == CELL_OCC, 255, 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=connectivity)
    cleaned = np.zeros_like(binary)
    for lbl in range(1, n):
        if stats[lbl, cv2.CC_STAT_AREA] > 1:
            cleaned[labels == lbl] = 255
    return np.where(cleaned == 255, CELL_OCC, CELL_FREE).astype(np.uint8)


def _inflate_obstacles(binary, inflation_px):
    k = int(2 * inflation_px + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    src = np.where(binary == CELL_OCC, 255, 0).astype(np.uint8)
    inflated = cv2.dilate(src, kernel, iterations=1)
    return (inflated > 0).astype(np.uint8)


def _expand_free_disk(grid, cell, radius_px):
    h, w = grid.shape
    x, y = int(cell[0]), int(cell[1])
    r2 = radius_px * radius_px
    for dy in range(-radius_px, radius_px + 1):
        for dx in range(-radius_px, radius_px + 1):
            if dx * dx + dy * dy <= r2:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    grid[ny, nx] = 0


def _reconstruct(parent, cy, cx):
    path = []
    cur = [cy, cx]
    while cur[0] != -1:
        path.append((cur[1], cur[0]))  
        cur = parent[cur[0], cur[1]]
    return path[::-1]


def _smooth_path(path, smoothness=0.1):
    if len(path) < 4:
        return path
    arr = np.array(path, dtype=float)
    x, y = arr[:, 0], arr[:, 1]
    s_val = max(0.0, smoothness * len(path))
    try:
        tck, _ = splprep([x, y], s=s_val, k=3)
        u_fine = np.linspace(0, 1, max(10, len(path) * 8))
        sx, sy = splev(u_fine, tck)
    except Exception:
        return path
    out, seen = [], set()
    for px, py in zip(sx, sy):
        p = (int(round(px)), int(round(py)))
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _run_astar(binary_map, start, goal):
    rows, cols = binary_map.shape
    sx, sy = int(start[0]), int(start[1])
    gx, gy = int(goal[0]), int(goal[1])

    dist_to_obs = distance_transform_edt(binary_map == 0)
    safe = ASTAR_SAFE_DISTANCE_PX
    penalty = np.where(
        dist_to_obs < safe,
        ASTAR_PENALTY_STRENGTH * (1.0 - dist_to_obs / safe),
        0.0,
    ).astype(np.float32)

    g_costs = np.full((rows, cols), np.inf, dtype=np.float32)
    parent = np.full((rows, cols, 2), -1, dtype=np.int32)
    g_costs[sy, sx] = 0.0

    W = ASTAR_HEURISTIC_WEIGHT
    pq = [(math.hypot(sx - gx, sy - gy) * W, sy, sx)]

    neighbors = [
        (0, 2, 2.0), (0, -2, 2.0), (2, 0, 2.0), (-2, 0, 2.0),
        (2, 2, 2.828), (2, -2, 2.828), (-2, 2, 2.828), (-2, -2, 2.828),
    ]

    while pq:
        f, cy, cx = heapq.heappop(pq)
        if abs(cx - gx) <= 2 and abs(cy - gy) <= 2:
            return _smooth_path(_reconstruct(parent, cy, cx))
        if f > g_costs[cy, cx] + math.hypot(cx - gx, cy - gy) * W:
            continue
        for dy, dx, step in neighbors:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < rows and 0 <= nx < cols and binary_map[ny, nx] == 0:
                ng = g_costs[cy, cx] + step + penalty[ny, nx]
                if ng < g_costs[ny, nx]:
                    g_costs[ny, nx] = ng
                    parent[ny, nx] = (cy, cx)
                    h = math.hypot(nx - gx, ny - gy) * W
                    heapq.heappush(pq, (ng + h, ny, nx))
    return []


def _path_length_m(path):
    total = 0.0
    prev = mapping.map_to_world(path[0][0], path[0][1])
    for p in path[1:]:
        cur = mapping.map_to_world(p[0], p[1])
        total += math.hypot(cur[0] - prev[0], cur[1] - prev[1])
        prev = cur
    return total


def plan(start_cell, goal_cell, grid=None, block_unknown=None, inflation_levels=None):
    if grid is None:
        grid = mapping.get_grid()
    if block_unknown is None:
        block_unknown = PLAN_BLOCK_UNKNOWN
    if inflation_levels is None:
        inflation_levels = ASTAR_INFLATION_LEVELS
    base = grid.copy()

    best_path = []
    best_len = 0.0

    for inflation in inflation_levels:
        work = base.astype(np.int32)
        closed_mask = (work == CELL_CLOSED)
        green_mask = (work == CELL_GREEN)
        unknown_mask = (work == CELL_UNKNOWN)

        work[closed_mask] = CELL_OCC
        work[green_mask] = CELL_OCC

        work = _clean_small_components(work.astype(np.uint8), min_size=6, connectivity=4)
        work = _remove_noisy_pixels(work, connectivity=4)   
        work = _inflate_obstacles(work, inflation)        

        work[closed_mask] = CELL_OCC
        work[green_mask] = CELL_OCC

        if block_unknown:
            work[unknown_mask] = CELL_OCC

        _expand_free_disk(work, goal_cell, ASTAR_EXPANSION_PIXELS)
        _expand_free_disk(work, start_cell, ASTAR_EXPANSION_PIXELS)

        path = _run_astar(work, start_cell, goal_cell)
        if not path or len(path) <= 1:
            continue

        length_m = _path_length_m(path)
        if length_m >= PATH_MIN_LENGTH_M:
            return path
        if length_m > best_len:
            best_len = length_m
            best_path = path

    return best_path


def plan_frontier(start_cell, goal_cell, grid=None):
    return plan(start_cell, goal_cell, grid=grid,
                block_unknown=False, inflation_levels=[ASTAR_FRONTIER_INFLATION])
