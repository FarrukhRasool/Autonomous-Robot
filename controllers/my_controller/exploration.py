import math
import random
from collections import deque

import numpy as np

import devices
import sensors
import motion
import localization
import mapping
import planning
import following
from config import (
    CELL_FREE, CELL_UNKNOWN,
    FRONTIER_MIN_CLUSTER, FRONTIER_MIN_SIZE, FRONTIER_MIN_DIST_PX,
    FRONTIER_VISITED_RADIUS_PX, FRONTIER_SCORE_BIAS, FRONTIER_SELF_EXCLUDE_PX,
    PATH_FOLLOWING_TARGET_REACH_DIST_PX,
    EXPLORE_FREESPACE_RADIUS_PX, EXPLORE_FREESPACE_TRIES,
    EXPLORE_SCAN_TURN_TICKS, EXPLORE_SCAN_TURN_WHEEL, EXPLORE_FORGET_VISITED_EVERY,
    SLAM_GREEN_PERIOD_STEPS, GREEN_MARK_ENABLED, OVERHEAD_MARK_ENABLED,
    PILLAR_BIAS_WEIGHT, OBSTACLE_RECOVER_TURN_AFTER,
    FOLLOW_GOVERNOR_FULL_M, FOLLOW_GOVERNOR_MIN_M, FOLLOW_GOVERNOR_ARC_DEG,
)

_visited = []                
_current_goal = None         
_current_path = None         
_should_continue = None       
_tick_count = 0              
_no_path_streak = 0          

_target_bias = None


_follow_ref_dist = None            
_follow_ref_pos = None              
_follow_stuck_count = 0            
FOLLOW_MIN_PROGRESS_PX = 3         
FOLLOW_STUCK_THRESHOLD = 25        

FOLLOW_STUCK_MIN_MOVE_M = 0.01


_obstacle_recover_streak = 0


_NAV_DEBUG = True
_nav_dbg_tick = 0  


def _tick(ms=None):
    global _tick_count
    if ms is None:
        ms = devices.timestep
    result = devices.robot.step(ms)
    if result == -1:
        return -1

    left_rad, right_rad = sensors.read_wheel_angles()
    localization.update_from_encoders(
        left_rad, right_rad, sensors.read_imu_yaw(), sensors.read_gyro_z()
    )

    _tick_count += 1
    if (GREEN_MARK_ENABLED
            and _tick_count % SLAM_GREEN_PERIOD_STEPS == 0
            and not motion.is_turning()):
        green_pts = sensors.green_ground_points_body()
        if len(green_pts) > 0:
            mapping.mark_green(localization.get_pose(), green_pts)

    if (OVERHEAD_MARK_ENABLED
            and _tick_count % SLAM_GREEN_PERIOD_STEPS == 0
            and not motion.is_turning()):
        overhead_pts = sensors.overhead_obstacle_points_body()
        if len(overhead_pts) > 0:
            mapping.mark_overhead(localization.get_pose(), overhead_pts)

    if _should_continue is not None and not _should_continue():
        return -1
    return result



def _stop_motor():
    motion.stop_robot()


def _set_robot_velocity(left_speed, right_speed):
    motion.set_wheel_speeds(left_speed, right_speed)


def _get_position():
    return np.array(localization.get_position(), dtype=float)


def _get_map_position():
    x, y = localization.get_position()
    return np.array(mapping.world_to_map(x, y), dtype=int)


def _get_map_distance(map_target):
    return float(np.linalg.norm(_get_map_position() - np.array(map_target)))


def _convert_to_world_coordinates(map_x, map_y):
    return mapping.map_to_world(map_x, map_y)


def _get_distances():
    return [
        devices.fl_range.getValue(),
        devices.rl_range.getValue(),
        devices.fr_range.getValue(),
        devices.rr_range.getValue(),
    ]


def _there_is_obstacle(cell):
    return mapping.there_is_obstacle(cell)


def _turn_left_milisecond(s=200):
    _set_robot_velocity(-4, 4)
    _tick(s)
    _stop_motor()


def _turn_right_milisecond(s=200):
    _set_robot_velocity(4, -4)
    _tick(s)
    _stop_motor()


def _lidar_update_map():
    print("updated")
    return


def _get_lidar_front_min_dist(angle_range_deg=30):
    points_local = np.asarray(sensors.read_lidar_pointcloud_2d())
    if len(points_local) == 0:
        return float("inf")
    angles = np.arctan2(points_local[:, 1], points_local[:, 0])
    distances = np.linalg.norm(points_local, axis=1)
    limit = np.radians(angle_range_deg)
    front_mask = (angles > -limit) & (angles < limit)
    front_distances = distances[front_mask]
    if len(front_distances) == 0:
        return float("inf")
    return float(np.min(front_distances))


def _govern_speed(v):
    if v <= 0.0:
        return v
    front = _get_lidar_front_min_dist(angle_range_deg=FOLLOW_GOVERNOR_ARC_DEG)
    if not math.isfinite(front) or front >= FOLLOW_GOVERNOR_FULL_M:
        return v
    if front <= FOLLOW_GOVERNOR_MIN_M:
        if _NAV_DEBUG:
            print(f"[NAV/gov] STOP  front={front:.3f}m <= MIN {FOLLOW_GOVERNOR_MIN_M}m "
                  f"(arc ±{FOLLOW_GOVERNOR_ARC_DEG}°) -> v {v:.3f}->0.000")
        return 0.0
    scale = (front - FOLLOW_GOVERNOR_MIN_M) / (FOLLOW_GOVERNOR_FULL_M - FOLLOW_GOVERNOR_MIN_M)
    return v * scale


def _nav_debug_snapshot(v_raw, v_governed, w):
    global _nav_dbg_tick
    _nav_dbg_tick += 1
    if _nav_dbg_tick % 3 != 0:
        return
    front_gov = _get_lidar_front_min_dist(angle_range_deg=FOLLOW_GOVERNOR_ARC_DEG)
    front_bump = _get_lidar_front_min_dist(angle_range_deg=35)
    ds = _get_distances()                       
    gov_limited = v_governed < v_raw - 1e-4
    ds_trig = min(ds[0], ds[2]) < 0.05         
    lidar_trig = front_bump < 0.10             
    who = (" [DS]" if ds_trig else "") + (" [LIDAR]" if lidar_trig else "")
    print(f"[NAV] v_raw={v_raw:.3f} v_gov={v_governed:.3f} w={w:+.2f} "
          f"gov_limited={gov_limited} front{int(FOLLOW_GOVERNOR_ARC_DEG)}={front_gov:.3f} "
          f"front35={front_bump:.3f} ds[fl={ds[0]:.3f} rl={ds[1]:.3f} fr={ds[2]:.3f} rr={ds[3]:.3f}] "
          f"bumper={'TRUE' if (ds_trig or lidar_trig) else 'false'}{who}")


def _obstacle_in_front():
    ds_distances = _get_distances()
    ds_obstacle = min(ds_distances[0], ds_distances[2]) < 0.05
    if ds_obstacle:
        print(f"[Virtual Bumper] Distance sensors detected obstacle at {min(ds_distances[0], ds_distances[2]):.2f}m")

    lidar_dist = _get_lidar_front_min_dist(angle_range_deg=35)
    lidar_obstacle = lidar_dist < 0.10
    if lidar_obstacle:
        print(f"[Virtual Bumper] Lidar detected obstacle at {lidar_dist:.2f}m")

    return ds_obstacle or lidar_obstacle


RECOVER_REAR_BLOCKED_M = 0.08


def _side_clearances():
    pts = np.asarray(sensors.read_lidar_pointcloud_2d())
    if len(pts) == 0:
        return float("inf"), float("inf")
    angles = np.arctan2(pts[:, 1], pts[:, 0])
    dists = np.linalg.norm(pts, axis=1)
    lo, hi = math.radians(30), math.radians(90)
    left = dists[(angles > lo) & (angles < hi)]
    right = dists[(angles < -lo) & (angles > -hi)]
    left_m = float(np.min(left)) if left.size else float("inf")
    right_m = float(np.min(right)) if right.size else float("inf")
    return left_m, right_m


def _rear_blocked(distances):
    return distances[1] < RECOVER_REAR_BLOCKED_M and distances[3] < RECOVER_REAR_BLOCKED_M


def _pivot_to_open_side(turn_duration, tag):
    left_clear, right_clear = _side_clearances()
    duration = random.randint(turn_duration[0], turn_duration[1])
    if left_clear >= right_clear:
        print(f"[{tag}] rear blocked -> pivot LEFT toward clearance "
              f"(L={left_clear:.2f} R={right_clear:.2f})")
        _turn_left_milisecond(duration)
    else:
        print(f"[{tag}] rear blocked -> pivot RIGHT toward clearance "
              f"(L={left_clear:.2f} R={right_clear:.2f})")
        _turn_right_milisecond(duration)
    _stop_motor()


def _recover_from_stuck(turn_duration=(800, 1200)):
    distances = _get_distances() 

    if _rear_blocked(distances):
        print(f"[recover from stuck] rear against wall (rl={distances[1]:.3f} "
              f"rr={distances[3]:.3f}) — not reversing")
        _pivot_to_open_side(turn_duration, "recover from stuck")
        for _ in range(40):
            if _tick(devices.timestep) == -1:
                _stop_motor()
                return False
        return

    _set_robot_velocity(-5, -5)
    if distances[1] > 0.25 and distances[3] > 0.25:
        print("reversing longer")
        _tick(500)
    else:
        print("reversing shorter")
        _tick(250)

    _stop_motor()
    for _ in range(40):
        if _tick(devices.timestep) == -1:
            _stop_motor()
            return False

    random_duration = random.randint(turn_duration[0], turn_duration[1])
    if random.random() < 0.5:
        _turn_left_milisecond(random_duration)
    else:
        _turn_right_milisecond(random_duration)
    _set_robot_velocity(5, 5)
    _tick(250)

    _stop_motor()
    for _ in range(40):
        if _tick(devices.timestep) == -1:
            _stop_motor()
            return False


def _recover_from_obstacle(turn_duration=(800, 1200)):
    global _obstacle_recover_streak
    distances = _get_distances()   
    if _rear_blocked(distances):
        print(f"[recover from obstacle] rear against wall (rl={distances[1]:.3f} "
              f"rr={distances[3]:.3f}) — not reversing")
        _pivot_to_open_side(turn_duration, "recover from obstacle")
        for _ in range(80):
            if _tick(devices.timestep) == -1:
                _stop_motor()
                return False
        return

    _set_robot_velocity(-5, -5)
    if distances[1] > 0.25 and distances[3] > 0.25:  
        print("reversing longer [recover from obstacle]")
        _tick(500)
    else:
        print("reversing shorter [recover from obstacle]")
        _tick(250)

    _stop_motor()
    for _ in range(80):
        if _tick(devices.timestep) == -1:
            _stop_motor()
            return False
    _tick(250)


    _obstacle_recover_streak += 1
    if _obstacle_recover_streak >= OBSTACLE_RECOVER_TURN_AFTER:
        _obstacle_recover_streak = 0
        duration = random.randint(turn_duration[0], turn_duration[1])
        if distances[0] >= distances[2]:      
            print("[recover from obstacle] wedged -> turning left to change approach")
            _turn_left_milisecond(duration)
        else:                                 
            print("[recover from obstacle] wedged -> turning right to change approach")
            _turn_right_milisecond(duration)




def _slowly_360():
    print("[Scan] Starting 360 rotation...")
    _set_robot_velocity(3, -3)
    steps_taken = 0
    target_steps = 8500 // devices.timestep
    while steps_taken < target_steps:
        if _tick(devices.timestep) == -1:
            break
        steps_taken += 1
    _stop_motor()


def _rotate_in_place(ticks):
    _set_robot_velocity(EXPLORE_SCAN_TURN_WHEEL, -EXPLORE_SCAN_TURN_WHEEL)
    for _ in range(ticks):
        if _tick(devices.timestep) == -1:
            break
    _stop_motor()


def detect_frontiers(grid):
    free = (grid == CELL_FREE)
    unknown = (grid == CELL_UNKNOWN)

    adj_unknown = np.zeros_like(unknown)
    adj_unknown[1:, :] |= unknown[:-1, :]
    adj_unknown[:-1, :] |= unknown[1:, :]
    adj_unknown[:, 1:] |= unknown[:, :-1]
    adj_unknown[:, :-1] |= unknown[:, 1:]

    ys, xs = np.where(free & adj_unknown)

    rx, ry = _get_map_position()
    far = (xs - rx) ** 2 + (ys - ry) ** 2 > FRONTIER_SELF_EXCLUDE_PX ** 2
    xs, ys = xs[far], ys[far]

    cells = list(zip(xs.tolist(), ys.tolist()))
    return _cluster_frontiers_bfs(cells, FRONTIER_MIN_CLUSTER)


def _cluster_frontiers_bfs(frontier_cells, min_cluster_size=15):
    if not frontier_cells:
        print("No frontier cells found.")
        return []

    frontier_set = set(frontier_cells)
    visited = set()
    clusters = []
    n8 = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    for start_cell in frontier_cells:
        if start_cell in visited:
            continue
        cluster = []
        queue = deque([start_cell])
        visited.add(start_cell)
        while queue:
            x, y = queue.popleft()
            cluster.append((x, y))
            for dx, dy in n8:
                nb = (x + dx, y + dy)
                if nb in frontier_set and nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        if len(cluster) >= min_cluster_size:
            clusters.append(cluster)
    return clusters


def _bias_map_cell():
    if _target_bias is None:
        return None
    return np.array(mapping.world_to_map(_target_bias[0], _target_bias[1]), dtype=float)


def _bias_multiplier(robot_pos, goal_cell, bias_cell):
    v_goal = np.asarray(goal_cell, dtype=float) - robot_pos
    v_bias = bias_cell - robot_pos
    ng = np.linalg.norm(v_goal)
    nb = np.linalg.norm(v_bias)
    if ng < 1e-6 or nb < 1e-6:
        return 1.0
    alignment = max(0.0, float(np.dot(v_goal, v_bias) / (ng * nb)))
    return 1.0 + PILLAR_BIAS_WEIGHT * alignment


def select_frontier_target(frontier_regions):
    if not frontier_regions:
        return None

    robot_map_pos = _get_map_position()
    bias_cell = _bias_map_cell()
    best_score = -float("inf")
    best_region = None

    for region in frontier_regions:
        cells = np.array(region)
        centroid = np.mean(cells, axis=0)
        dist = np.linalg.norm(centroid - robot_map_pos)
        size = len(region)

        if size < FRONTIER_MIN_SIZE or dist < FRONTIER_MIN_DIST_PX:
            continue

        too_close = any(
            np.linalg.norm(np.array([centroid[0], centroid[1]]) - np.array(v)) < FRONTIER_VISITED_RADIUS_PX
            for v in _visited if v is not None
        )
        if too_close:
            continue

        score = size / (dist + FRONTIER_SCORE_BIAS)
        if bias_cell is not None:
            score *= _bias_multiplier(robot_map_pos, centroid, bias_cell)
        if score > best_score:
            best_score = score
            best_region = region

    if best_region is None:
        return select_frontier_target2(frontier_regions)

    cells = np.array(best_region)
    cx = int(np.mean(cells[:, 0]))
    cy = int(np.mean(cells[:, 1]))
    return (cx, cy)


def select_frontier_target2(frontier_regions):
    if not frontier_regions:
        return None
    best_region = random.choice(frontier_regions)
    region_cells = np.array(best_region)
    centroid_x = int(np.mean(region_cells[:, 0]) + random.randint(-5, 5))
    centroid_y = int(np.mean(region_cells[:, 1]) + random.randint(-5, 5))
    return (centroid_x, centroid_y)


def select_random_freespace_near_robot(radius=EXPLORE_FREESPACE_RADIUS_PX, max_tries=EXPLORE_FREESPACE_TRIES):
    grid = mapping.get_grid()
    if grid is None:
        return None
    map_h, map_w = grid.shape
    rx, ry = _get_map_position()
    bias_cell = _bias_map_cell()

    best = None
    best_bias_dist = float("inf")
    for _ in range(max_tries):
        dx = random.randint(-radius, radius)
        dy = random.randint(-radius, radius)
        x = int(rx + dx)
        y = int(ry + dy)
        if x < 0 or x >= map_w or y < 0 or y >= map_h:
            continue
        if grid[y, x] != CELL_FREE:
            continue
        if _there_is_obstacle((x, y)):
            continue
        if bias_cell is None:
            return (x, y)                      
        d = (x - bias_cell[0]) ** 2 + (y - bias_cell[1]) ** 2
        if d < best_bias_dist:                 
            best_bias_dist = d
            best = (x, y)
    return best




def _follow_local_target(map_target):
    global _follow_ref_dist, _follow_ref_pos, _follow_stuck_count

    goal_dist = _get_map_distance(map_target)          
    if goal_dist < PATH_FOLLOWING_TARGET_REACH_DIST_PX:
        _follow_ref_dist = None
        _follow_ref_pos = None
        _follow_stuck_count = 0
        return True, False

    if _follow_ref_dist is None:
        _follow_ref_dist = goal_dist
        _follow_ref_pos = _get_position()      
        _follow_stuck_count = 0
    _follow_stuck_count += 1
    if _follow_stuck_count >= FOLLOW_STUCK_THRESHOLD:
        progressed = _follow_ref_dist - goal_dist                      
        moved_m = float(np.linalg.norm(_get_position() - _follow_ref_pos))  
        _follow_ref_dist = goal_dist
        _follow_ref_pos = _get_position()
        _follow_stuck_count = 0
        low_goal_progress = progressed < FOLLOW_MIN_PROGRESS_PX
        barely_moved = moved_m < FOLLOW_STUCK_MIN_MOVE_M
        if low_goal_progress and barely_moved:
            if _NAV_DEBUG:
                print(f"[NAV/stuck] TRUE stuck: closed {progressed:.1f}px "
                      f"< {FOLLOW_MIN_PROGRESS_PX}px AND moved {moved_m:.3f}m "
                      f"< {FOLLOW_STUCK_MIN_MOVE_M}m over {FOLLOW_STUCK_THRESHOLD} ticks "
                      f"-> is_stuck (reverse+replan)")
            return False, True
        if _NAV_DEBUG and low_goal_progress:
            print(f"[NAV/stuck] slow-but-moving (NOT stuck): closed {progressed:.1f}px "
                  f"< {FOLLOW_MIN_PROGRESS_PX}px but moved {moved_m:.3f}m "
                  f">= {FOLLOW_STUCK_MIN_MOVE_M}m")

    world_target = np.array(_convert_to_world_coordinates(map_target[0], map_target[1]))
    pose = localization.get_pose()
    v_raw, w = following.dwa_velocity(pose, world_target, devices.timestep / 1000.0)
    v = _govern_speed(v_raw)      
    if _NAV_DEBUG:
        _nav_debug_snapshot(v_raw, v, w)  
    motion.drive_twist(v, w)
    return False, False


def _path_blocked(path):
    for p in path:
        if _there_is_obstacle(p):
            return True
    return False


def frontier_following(path, replan_interval=20):
    global _current_path, _obstacle_recover_streak

    if path is None or len(path) == 0:
        return False

    _obstacle_recover_streak = 0       
    frontier_goal = path[-1]
    current_path = list(path)
    _current_path = current_path
    timestep_counter = 0
    stuck_attempt_count = 0
    replan_attempts = 0
    target_index = min(5, len(current_path) - 1)
    MAX_STUCK_ATTEMPTS = 3

    while target_index < len(current_path):
        target = current_path[target_index]

        while _tick(devices.timestep) != -1:
            timestep_counter += 1

           
            if _obstacle_in_front():
                replan_attempts += 1
                print(f"[Frontier] Obstacle detected! Attempt {replan_attempts}/2")

                _recover_from_obstacle()
                _lidar_update_map()

                if replan_attempts >= 2:
                    print("[Frontier] Failed 2 times. Giving up on this frontier for now.")
                    _visited.append(tuple(frontier_goal))  
                    return False

                current_start = _get_map_position()
                new_path = planning.plan_frontier(tuple(current_start), tuple(frontier_goal))
                if new_path and len(new_path) > 5:
                    print("[Frontier] Replanned successfully! Trying new route...")
                    current_path = new_path
                    _current_path = current_path
                    target_index = 5
                    break
                else:
                    print("[Frontier] No path found after obstacle. Dropping.")
                    _visited.append(tuple(frontier_goal))  
                    return False

            if timestep_counter % replan_interval == 0 and _path_blocked(current_path):
                current_start = _get_map_position()
                new_path = planning.plan_frontier(tuple(current_start), tuple(frontier_goal))
                if new_path and len(new_path) > 5:
                    print("[Replan] Current path blocked by new obstacle, replanning")
                    current_path = new_path
                    _current_path = current_path
                    target_index = 5
                    break

           
            reached, is_stuck = _follow_local_target(target)
            if is_stuck:
                stuck_attempt_count += 1
                if stuck_attempt_count >= MAX_STUCK_ATTEMPTS:
                    _visited.append(tuple(frontier_goal))
                    _stop_motor()
                    return False

                _stop_motor()
                _lidar_update_map()
                _recover_from_stuck()

                current_start = _get_map_position()
                new_path = planning.plan_frontier(tuple(current_start), tuple(frontier_goal))
                if new_path and len(new_path) > 5:
                    current_path = list(new_path)
                    _current_path = current_path
                    target_index = 5
                    break
                else:
                    _visited.append(tuple(frontier_goal))
                    return False

            if reached:
                break

        target_index += 5

    _stop_motor()
    return True




def handle_frontier_exploration(count):
    global _current_goal
    frontier_regions = detect_frontiers(mapping.get_grid())
    chosen_frontier = select_frontier_target(frontier_regions)
    path_to_frontier = None
    if chosen_frontier:
        if chosen_frontier != _current_goal:
            print(f"[EXPLORE] scored frontier {chosen_frontier} (iter {count})")
        _current_goal = chosen_frontier
        path_to_frontier = planning.plan_frontier(tuple(_get_map_position()), chosen_frontier)
    return frontier_regions, chosen_frontier, path_to_frontier


def explore():
    global _current_goal, _current_path, _no_path_streak

    count = 0
    _slowly_360()

    while _tick(devices.timestep) != -1:
        _regions, _chosen, path_to_target = handle_frontier_exploration(count)
        if not path_to_target:
            fallback_cell = select_random_freespace_near_robot()
            if fallback_cell is not None:
                _current_goal = fallback_cell
                path_to_target = planning.plan_frontier(
                    tuple(_get_map_position()), fallback_cell
                )

        if path_to_target:
            frontier_following(path_to_target)
            _no_path_streak = 0
        else:
            _no_path_streak += 1
            if _no_path_streak % EXPLORE_FORGET_VISITED_EVERY == 0 and _visited:
                _visited.clear()
                print("[Explore] no path — cleared visited-frontier blacklist")
            _rotate_in_place(EXPLORE_SCAN_TURN_TICKS)

        count += 1

    _stop_motor()
    print("Exploration completed.")




def run(should_continue):
    global _should_continue
    reset()
    _should_continue = should_continue
    try:
        explore()
    finally:
        _should_continue = None
        _stop_motor()


def current_goal():
    return _current_goal


def current_path():
    return _current_path


def set_target_bias(world_xy):
    global _target_bias
    _target_bias = (float(world_xy[0]), float(world_xy[1])) if world_xy is not None else None


def clear_target_bias():
    global _target_bias
    _target_bias = None


def reset():
    global _visited, _current_goal, _current_path, _tick_count, _no_path_streak
    global _follow_ref_dist, _follow_ref_pos, _follow_stuck_count, _target_bias
    global _obstacle_recover_streak
    _visited = []
    _current_goal = None
    _current_path = None
    _tick_count = 0
    _no_path_streak = 0
    _follow_ref_dist = None
    _follow_ref_pos = None
    _follow_stuck_count = 0
    _target_bias = None
    _obstacle_recover_streak = 0
