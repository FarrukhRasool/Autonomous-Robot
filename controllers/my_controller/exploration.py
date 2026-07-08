"""Frontier-based exploration — faithful port of the reference project's
exploration (Hieu Tran et al., src/controllers/main/my_robot.py).

The reference exploration is a BLOCKING controller: its methods run their own
`while self.step()` loops.  This module reproduces that control flow verbatim,
with the reference's `self.<method>()` calls mapped 1:1 onto this project's
modules (devices / sensors / motion / localization / mapping / planning /
following).  The reference method each function mirrors is named in its
docstring, and the reference's print statements are preserved so the console
output matches.

Not ported (belong to other layers / not applicable here):
  * FastSLAM pose — we use this project's odometry pose instead.
  * Red-wall dead-end closure — maze-specific to their worlds (violates the
    generalization constraint); their mazes mark dead-ends with red walls.
  * Column / green-carpet interrupts inside frontier_following — those are the
    perception/mission (FR4/FR5) layer, added separately; here we run pure
    exploration (FR9), so those interrupt branches are omitted.

Their `self.step(ms)` override advances the sim then updates odometry every call;
their lidar mapping runs in a background thread.  `_tick()` below reproduces both
(step + odometry + map) so the map stays fresh throughout the blocking loops.
"""

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

# ── Reference MyRobot state (the fields exploration touches) ──────────────────
_visited = []                 # map_object.visited_frontiers
_current_goal = None          # for the live-map overlay
_current_path = None          # for the live-map overlay
_should_continue = None       # stop hook; when it returns False, _tick() aborts like sim-end
_tick_count = 0               # sim ticks since run start (paces the SLAM observe cadence)
_no_path_streak = 0           # consecutive iterations with no frontier/freespace path (anti-idle)
# World (x, y) of a SIGHTED active target pillar.  When set (by the mission via
# set_target_bias), frontier and free-cell selection are steered toward it so
# exploration heads for the pillar instead of wandering agnostically; None = pure
# frontier exploration (e.g. the standalone E-key mode never sets it).
_target_bias = None

# follow_local_target stuck-detection state (reference __init__ values)
_follow_ref_dist = None             # goal distance (map cells) at the progress-window start
_follow_stuck_count = 0             # ticks elapsed in the current progress window
FOLLOW_MIN_PROGRESS_PX = 3          # cells the goal distance must shrink per window, else stuck
FOLLOW_STUCK_THRESHOLD = 25         # window length (ticks) over which progress is measured

# Consecutive obstacle recoveries without escaping — escalates to a turn (see
# _recover_from_obstacle).  Reset when a new frontier follow starts / on reset.
_obstacle_recover_streak = 0


# ── self.step() override + odometry/mapping (reference step() + lidar thread) ─

def _tick(ms=None):
    """Reference self.step(ms): advance the sim, then update odometry every
    advance (-> slam.predict via localization).  The SLAM measurement update
    (slam.observe) runs on the background mapping thread, exactly like the
    reference's lidar thread, so it is NOT called here.  Green marking (camera,
    main-thread) runs on its cadence.  Returns -1 on sim-end OR when the stop
    hook fires, so the reference's `while self.step()!=-1` guards double as the
    exploration's stop mechanism."""
    global _tick_count
    if ms is None:
        ms = devices.timestep
    result = devices.robot.step(ms)
    if result == -1:
        return -1

    # Motion model every advance -> slam.predict (via localization).
    left_rad, right_rad = sensors.read_wheel_angles()
    localization.update_from_encoders(
        left_rad, right_rad, sensors.read_imu_yaw(), sensors.read_gyro_z()
    )

    # Green marking (camera-based) ~10 Hz, only when not turning.
    _tick_count += 1
    if (GREEN_MARK_ENABLED
            and _tick_count % SLAM_GREEN_PERIOD_STEPS == 0
            and not motion.is_turning()):
        green_pts = sensors.green_ground_points_body()
        if len(green_pts) > 0:
            mapping.mark_green(localization.get_pose(), green_pts)

    # Overhead/floating-wall marking (depth-camera based) ~10 Hz, only when not
    # turning.  A floating wall sits above the lidar's scan plane, so the depth
    # ROI is the only evidence of it; project it to the floor and stamp it as a
    # hard obstacle (CELL_CLOSED) so the planner/DWA route around it.
    if (OVERHEAD_MARK_ENABLED
            and _tick_count % SLAM_GREEN_PERIOD_STEPS == 0
            and not motion.is_turning()):
        overhead_pts = sensors.overhead_obstacle_points_body()
        if len(overhead_pts) > 0:
            mapping.mark_overhead(localization.get_pose(), overhead_pts)

    if _should_continue is not None and not _should_continue():
        return -1
    return result


# ── Low-level interface (reference MyRobot helpers -> this project's modules) ──

def _stop_motor():
    motion.stop_robot()


def _set_robot_velocity(left_speed, right_speed):
    """Reference set_robot_velocity: left -> fl/rl, right -> fr/rr."""
    motion.set_wheel_speeds(left_speed, right_speed)


def _get_position():
    return np.array(localization.get_position(), dtype=float)


def _get_map_position():
    """Reference get_map_position -> (map_x, map_y) as np.array."""
    x, y = localization.get_position()
    return np.array(mapping.world_to_map(x, y), dtype=int)


def _get_map_distance(map_target):
    return float(np.linalg.norm(_get_map_position() - np.array(map_target)))


def _convert_to_world_coordinates(map_x, map_y):
    return mapping.map_to_world(map_x, map_y)


def _get_distances():
    """Reference get_distances: [fl, rl, fr, rr] (order per their setup.py)."""
    return [
        devices.fl_range.getValue(),
        devices.rl_range.getValue(),
        devices.fr_range.getValue(),
        devices.rr_range.getValue(),
    ]


def _there_is_obstacle(cell):
    return mapping.there_is_obstacle(cell)


def _turn_left_milisecond(s=200):
    """Reference turn_left_milisecond."""
    _set_robot_velocity(-4, 4)
    _tick(s)
    _stop_motor()


def _turn_right_milisecond(s=200):
    """Reference turn_right_milisecond."""
    _set_robot_velocity(4, -4)
    _tick(s)
    _stop_motor()


def _lidar_update_map():
    """Reference lidar_update_map (recovery hook): originally forced a map update
    from the current scan.  The background mapping thread now owns the lidar and
    maps continuously (~10 Hz), including throughout the multi-tick recovery
    maneuvers that call this — so the map is already refreshed by the time
    recovery ends.  Kept as a no-op to preserve the reference call sites and to
    keep lidar reads on a single thread (avoids a same-device read race)."""
    return


def _get_lidar_front_min_dist(angle_range_deg=30):
    """Reference get_lidar_front_min_dist: min range in a front-facing arc."""
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
    """Scale a commanded forward speed by live front-lidar clearance.

    DWA only avoids MAPPED obstacles, so in unknown space it commands full speed
    straight at unmapped walls until the binary bumper trips too late.  This caps
    v by what the lidar actually sees ahead: full above FOLLOW_GOVERNOR_FULL_M,
    linearly down to 0 at FOLLOW_GOVERNOR_MIN_M (the bumper/recover own the last
    ~0.10 m).  Only forward speed is governed — the caller's w is untouched, so the
    robot can still rotate toward an opening while slowed.
    """
    if v <= 0.0:
        return v
    front = _get_lidar_front_min_dist(angle_range_deg=FOLLOW_GOVERNOR_ARC_DEG)
    if not math.isfinite(front) or front >= FOLLOW_GOVERNOR_FULL_M:
        return v
    if front <= FOLLOW_GOVERNOR_MIN_M:
        return 0.0
    scale = (front - FOLLOW_GOVERNOR_MIN_M) / (FOLLOW_GOVERNOR_FULL_M - FOLLOW_GOVERNOR_MIN_M)
    return v * scale


def _obstacle_in_front():
    """Reference obstacle_in_front: distance-sensor V-shape OR lidar bumper."""
    ds_distances = _get_distances()
    ds_obstacle = min(ds_distances[0], ds_distances[2]) < 0.05
    if ds_obstacle:
        print(f"[Virtual Bumper] Distance sensors detected obstacle at {min(ds_distances[0], ds_distances[2]):.2f}m")

    lidar_dist = _get_lidar_front_min_dist(angle_range_deg=35)
    lidar_obstacle = lidar_dist < 0.10
    if lidar_obstacle:
        print(f"[Virtual Bumper] Lidar detected obstacle at {lidar_dist:.2f}m")

    return ds_obstacle or lidar_obstacle


# ── Recovery (reference recover_from_stuck / recover_from_obstacle) ────────────

def _recover_from_stuck(turn_duration=(800, 1200)):
    """Reference recover_from_stuck: reverse (length gated on rear clearance),
    settle, then turn a random direction/duration."""
    _set_robot_velocity(-5, -5)
    distances = _get_distances()
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
    """Reverse to back off a front obstacle, then settle.

    Two changes from the reference recover_from_obstacle (which reversed straight,
    no turn, and gated on the wrong sensor):
      * The reverse-length gate reads REAR clearance — rl (index 1) and rr (index
        3) — instead of the reference's fr (index 2).  fr is a FRONT sensor, so a
        robot nosed into a corner always saw it blocked and only ever reversed
        'shorter' (250 ms), never far enough to escape -> the wedge loop.  Matches
        _recover_from_stuck, which correctly gates on rl+rr.
      * Escalation: after OBSTACLE_RECOVER_TURN_AFTER consecutive recoveries the
        straight reverse clearly isn't breaking the wedge (the replan keeps driving
        back in at the same angle), so also TURN toward the more-open front side to
        change the re-approach heading.
    """
    global _obstacle_recover_streak
    _set_robot_velocity(-5, -5)
    distances = _get_distances()   # [fl, rl, fr, rr]
    if distances[1] > 0.25 and distances[3] > 0.25:   # rear-left AND rear-right clear
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

    # Escalate on repeated wedges: turn toward whichever front side is more open
    # (fl vs fr) so the replanned path re-approaches at a fresh angle.
    _obstacle_recover_streak += 1
    if _obstacle_recover_streak >= OBSTACLE_RECOVER_TURN_AFTER:
        _obstacle_recover_streak = 0
        duration = random.randint(turn_duration[0], turn_duration[1])
        if distances[0] >= distances[2]:      # front-left clearer -> turn left
            print("[recover from obstacle] wedged -> turning left to change approach")
            _turn_left_milisecond(duration)
        else:                                 # front-right clearer -> turn right
            print("[recover from obstacle] wedged -> turning right to change approach")
            _turn_right_milisecond(duration)


# ── Initial scan (reference slowly_360) ───────────────────────────────────────

def _slowly_360():
    """Reference slowly_360: spin in place to seed the map.  (The reference also
    breaks early on a camera column/red/green signal; omitted in pure exploration.)"""
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
    """Turn in place for `ticks` sim steps to reveal new space.  Anti-idle escape
    when no frontier/freespace path is available — rotating never translates the
    robot into a wall and always changes what the sensors (and thus the next
    frontier selection) can see."""
    _set_robot_velocity(EXPLORE_SCAN_TURN_WHEEL, -EXPLORE_SCAN_TURN_WHEEL)
    for _ in range(ticks):
        if _tick(devices.timestep) == -1:
            break
    _stop_motor()


# ── Frontier detection (reference detect_frontiers + _cluster_frontiers_bfs) ──

def detect_frontiers(grid):
    """Reference detect_frontiers: FREE cells 4-adjacent to UNKNOWN, BFS-clustered."""
    free = (grid == CELL_FREE)
    unknown = (grid == CELL_UNKNOWN)

    adj_unknown = np.zeros_like(unknown)
    adj_unknown[1:, :] |= unknown[:-1, :]
    adj_unknown[:-1, :] |= unknown[1:, :]
    adj_unknown[:, 1:] |= unknown[:, :-1]
    adj_unknown[:, :-1] |= unknown[:, 1:]

    ys, xs = np.where(free & adj_unknown)

    # Drop the phantom near-field frontier ring: the unmapped disk under the robot
    # (below lidar min range) is never cleared, so its boundary is a frontier ring
    # that moves with the robot and can never be resolved.  Excluding cells within
    # FRONTIER_SELF_EXCLUDE_PX of the robot keeps exploration from getting trapped
    # selecting it in place.
    rx, ry = _get_map_position()
    far = (xs - rx) ** 2 + (ys - ry) ** 2 > FRONTIER_SELF_EXCLUDE_PX ** 2
    xs, ys = xs[far], ys[far]

    cells = list(zip(xs.tolist(), ys.tolist()))
    return _cluster_frontiers_bfs(cells, FRONTIER_MIN_CLUSTER)


def _cluster_frontiers_bfs(frontier_cells, min_cluster_size=15):
    """Reference _cluster_frontiers_bfs: 8-connected BFS, drop clusters < min."""
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


# ── Frontier selection (reference select_frontier_target / target2 / freespace) ──

def _bias_map_cell():
    """The sighted target pillar as a (float) map cell, or None if unbiased."""
    if _target_bias is None:
        return None
    return np.array(mapping.world_to_map(_target_bias[0], _target_bias[1]), dtype=float)


def _bias_multiplier(robot_pos, goal_cell, bias_cell):
    """Utility multiplier 1 + PILLAR_BIAS_WEIGHT * alignment, where alignment in
    [0, 1] is how well the robot->goal direction points toward the robot->pillar
    direction (cosine, clamped at 0 so goals pointing away are just neutral, not
    penalised)."""
    v_goal = np.asarray(goal_cell, dtype=float) - robot_pos
    v_bias = bias_cell - robot_pos
    ng = np.linalg.norm(v_goal)
    nb = np.linalg.norm(v_bias)
    if ng < 1e-6 or nb < 1e-6:
        return 1.0
    alignment = max(0.0, float(np.dot(v_goal, v_bias) / (ng * nb)))
    return 1.0 + PILLAR_BIAS_WEIGHT * alignment


def select_frontier_target(frontier_regions):
    """Reference select_frontier_target: score size/(dist+bias); skip small/close/
    visited; return best cluster's CENTROID, else select_frontier_target2().

    When a target pillar has been sighted (_target_bias set), each cluster's score
    is multiplied by _bias_multiplier so clusters in the pillar's direction win —
    steering exploration toward it (FR5) instead of purely maximising new area."""
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
    """Reference select_frontier_target2: a random cluster centroid with jitter."""
    if not frontier_regions:
        return None
    best_region = random.choice(frontier_regions)
    region_cells = np.array(best_region)
    centroid_x = int(np.mean(region_cells[:, 0]) + random.randint(-5, 5))
    centroid_y = int(np.mean(region_cells[:, 1]) + random.randint(-5, 5))
    return (centroid_x, centroid_y)


def select_random_freespace_near_robot(radius=EXPLORE_FREESPACE_RADIUS_PX, max_tries=EXPLORE_FREESPACE_TRIES):
    """Reference select_random_freespace_near_robot: a random FREESPACE cell near
    the robot.

    When a pillar is sighted (_target_bias set), return the sampled free cell
    NEAREST the pillar instead of the first random hit — a greedy step toward it.
    This matters in open rooms: once the walls are mapped there are few frontiers,
    so this anti-idle fallback does most of the driving, and it must advance toward
    the target rather than wander."""
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
            return (x, y)                       # unbiased: first valid hit (as before)
        d = (x - bias_cell[0]) ** 2 + (y - bias_cell[1]) ** 2
        if d < best_bias_dist:                  # biased: keep the one nearest the pillar
            best_bias_dist = d
            best = (x, y)
    return best


# ── DWA follow (reference dwa_planner via following.dwa_velocity + follow_local_target) ──

def _follow_local_target(map_target):
    """Reference follow_local_target -> (reached, is_stuck).  Drives DWA toward
    map_target; flags stuck when it stops making PROGRESS toward the target.

    Progress (goal distance shrinking), not per-tick movement, is the stuck test:
    at a wall the governor zeroes forward speed while DWA keeps a turn, so the robot
    wiggles/rotates in place — that per-tick motion kept resetting the old
    'hasn't-moved' counter, so stuck never fired and the robot sat there forever.
    Measuring goal-distance progress over a window catches that oscillation."""
    global _follow_ref_dist, _follow_stuck_count

    goal_dist = _get_map_distance(map_target)          # map cells
    if goal_dist < PATH_FOLLOWING_TARGET_REACH_DIST_PX:
        _follow_ref_dist = None
        _follow_stuck_count = 0
        return True, False

    # Progress watchdog: over FOLLOW_STUCK_THRESHOLD ticks the goal distance must
    # shrink by at least FOLLOW_MIN_PROGRESS_PX cells, else the follow is stuck.
    if _follow_ref_dist is None:
        _follow_ref_dist = goal_dist
        _follow_stuck_count = 0
    _follow_stuck_count += 1
    if _follow_stuck_count >= FOLLOW_STUCK_THRESHOLD:
        progressed = _follow_ref_dist - goal_dist
        _follow_ref_dist = goal_dist
        _follow_stuck_count = 0
        if progressed < FOLLOW_MIN_PROGRESS_PX:
            return False, True

    world_target = np.array(_convert_to_world_coordinates(map_target[0], map_target[1]))
    pose = localization.get_pose()
    v, w = following.dwa_velocity(pose, world_target, devices.timestep / 1000.0)
    v = _govern_speed(v)          # live-lidar brake so we don't charge unmapped walls
    motion.drive_twist(v, w)
    return False, False


def _path_blocked(path):
    """Reference _path_blocked: True if any waypoint is now a known obstacle."""
    for p in path:
        if _there_is_obstacle(p):
            return True
    return False


def frontier_following(path, replan_interval=20):
    """Reference frontier_following (exploration subset): follow `path`'s waypoints
    (stride 5) to its fixed goal, with front-obstacle reverse+replan (<=2), a
    periodic blocked-path replan, and stuck reverse-turn+replan (<=3) before
    blacklisting.  Red-wall / column / green interrupts omitted (see module doc)."""
    global _current_path, _obstacle_recover_streak

    if path is None or len(path) == 0:
        return False

    _obstacle_recover_streak = 0        # fresh frontier -> reset the wedge counter
    frontier_goal = path[-1]
    current_path = list(path)
    _current_path = current_path
    timestep_counter = 0
    stuck_attempt_count = 0
    replan_attempts = 0
    # Start at the stride waypoint, but clamp to the last waypoint for short paths
    # — otherwise `while target_index < len(path)` is false on entry and the whole
    # follow no-ops (no drive, no sim step), which trapped the robot on close goals.
    target_index = min(5, len(current_path) - 1)
    MAX_STUCK_ATTEMPTS = 3

    while target_index < len(current_path):
        target = current_path[target_index]

        while _tick(devices.timestep) != -1:
            timestep_counter += 1

            # FRONT OBSTACLE -> REVERSE -> REPLAN (max 2)
            if _obstacle_in_front():
                replan_attempts += 1
                print(f"[Frontier] Obstacle detected! Attempt {replan_attempts}/2")

                _recover_from_obstacle()
                _lidar_update_map()

                if replan_attempts >= 2:
                    print("[Frontier] Failed 2 times. Giving up on this frontier for now.")
                    _visited.append(tuple(frontier_goal))   # blacklist so it isn't re-picked
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
                    _visited.append(tuple(frontier_goal))   # blacklist so it isn't re-picked
                    return False

            # PERIODIC BLOCKED-PATH CHECK (only replan if a waypoint is now an obstacle)
            if timestep_counter % replan_interval == 0 and _path_blocked(current_path):
                current_start = _get_map_position()
                new_path = planning.plan_frontier(tuple(current_start), tuple(frontier_goal))
                if new_path and len(new_path) > 5:
                    print("[Replan] Current path blocked by new obstacle, replanning")
                    current_path = new_path
                    _current_path = current_path
                    target_index = 5
                    break

            # FOLLOW LOCAL TARGET
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


# ── Orchestration (reference handle_frontier_exploration + explore) ───────────

def handle_frontier_exploration(count):
    """Detect frontiers, score-select the best (biased toward a sighted pillar when
    set), plan a path to it, and RETURN (regions, chosen, path) — the caller drives.

    Selection runs EVERY iteration (not every-Nth-after-a-warm-up as in the
    reference).  Each explore() iteration drives a whole path, so the old cadence
    meant ~4 of every 5 drives fell back to a random ALREADY-EXPLORED free cell
    instead of a frontier; running selection every iteration keeps the robot
    heading for the unexplored boundary.  (No internal follow here — that produced
    a double-drive of the same path with explore().)"""
    global _current_goal
    frontier_regions = detect_frontiers(mapping.get_grid())
    chosen_frontier = select_frontier_target(frontier_regions)
    path_to_frontier = None
    if chosen_frontier:
        # Log only when the target changes (selection runs every iteration).
        if chosen_frontier != _current_goal:
            print(f"[EXPLORE] scored frontier {chosen_frontier} (iter {count})")
        _current_goal = chosen_frontier
        path_to_frontier = planning.plan_frontier(tuple(_get_map_position()), chosen_frontier)
    return frontier_regions, chosen_frontier, path_to_frontier


def explore():
    """Exploration loop: initial 360 scan, then EVERY iteration head for the best
    frontier (unexplored boundary); only when no reachable frontier exists fall
    back to a nearby free cell (biased toward the pillar/unknown when set) or, if
    even that fails, rotate in place to reveal new space.  Runs until the stop hook
    fires (via _tick).  Frontier-first is the fix for the robot re-visiting
    already-explored space: the old every-Nth cadence sent most drives to random
    explored cells."""
    global _current_goal, _current_path, _no_path_streak

    count = 0
    _slowly_360()

    while _tick(devices.timestep) != -1:
        # Frontier-first, every iteration.
        _regions, _chosen, path_to_target = handle_frontier_exploration(count)

        # Fallback ONLY when there is no reachable frontier (rare now): a nearby
        # free cell, biased toward the pillar/unknown when a bias is set.
        # NOTE: plan_frontier returns an EMPTY list [] (not None) for an unreachable
        # frontier, so test falsiness (None OR []).  Testing `is None` let an
        # unreachable frontier's [] pass as a valid path -> frontier_following([])
        # no-ops -> the robot re-selected the same unreachable cluster forever
        # without ever falling back or rotating.
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
            # Neither a frontier NOR a reachable free cell: local frontiers are
            # exhausted or the visited blacklist has suppressed them.  Never idle —
            # rotate to reveal new space, and periodically forget the blacklist so
            # previously-dropped frontiers become selectable again.
            _no_path_streak += 1
            if _no_path_streak % EXPLORE_FORGET_VISITED_EVERY == 0 and _visited:
                _visited.clear()
                print("[Explore] no path — cleared visited-frontier blacklist")
            _rotate_in_place(EXPLORE_SCAN_TURN_TICKS)

        count += 1

    _stop_motor()
    print("Exploration completed.")


# ── Entry point + accessors for the controller ───────────────────────────────

def run(should_continue):
    """Run blocking frontier exploration until `should_continue()` returns False
    (or the simulation ends).  `should_continue` is polled every sim tick."""
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
    """Steer frontier/free-cell selection toward this world (x, y) — the mission
    calls this each tick once the active pillar has been sighted.  None restores
    pure (unbiased) frontier exploration."""
    global _target_bias
    _target_bias = (float(world_xy[0]), float(world_xy[1])) if world_xy is not None else None


def clear_target_bias():
    global _target_bias
    _target_bias = None


def reset():
    """Clear exploration state (visited frontiers, follow state, overlays, bias)."""
    global _visited, _current_goal, _current_path, _tick_count, _no_path_streak
    global _follow_ref_dist, _follow_stuck_count, _target_bias
    global _obstacle_recover_streak
    _visited = []
    _current_goal = None
    _current_path = None
    _tick_count = 0
    _no_path_streak = 0
    _follow_ref_dist = None
    _follow_stuck_count = 0
    _target_bias = None
    _obstacle_recover_streak = 0
