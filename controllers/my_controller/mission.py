import math

import devices
import sensors
import motion
import localization
import mapping
import planning
import perception
import exploration
from config import (
    APPROACH_OFFSET_M, MISSION_MARK_RATIO,
    MISSION_MARK_LASER_M, MISSION_MARK_BEARING_RAD, MISSION_MARK_MAX_STAMP_M,
    GREEN_MARK_ENABLED, OVERHEAD_MARK_ENABLED, SLAM_GREEN_PERIOD_STEPS,
    FOLLOW_WAYPOINT_STRIDE,
    SEEK_LIN_VEL, SEEK_OMEGA, SEEK_BEARING_DEADBAND_RAD,
    PILLAR_COMMIT_DIST_M,
)

SEEKING_BLUE, SEEKING_YELLOW, DONE = 0, 1, 2
_NAMES = {SEEKING_BLUE: "SEEKING_BLUE", SEEKING_YELLOW: "SEEKING_YELLOW", DONE: "DONE"}

_seen = {"blue": False, "yellow": False}
_registered = {"blue": False, "yellow": False}
_state = SEEKING_BLUE
_tick_count = 0
_current_route = None  

_MISSION_DEBUG = True
_last_seen_time = {"blue": None, "yellow": None}  


def reset():
    global _seen, _registered, _state, _tick_count, _current_route, _last_seen_time
    _seen = {"blue": False, "yellow": False}
    _registered = {"blue": False, "yellow": False}
    _state = SEEKING_BLUE
    _tick_count = 0
    _current_route = None
    _last_seen_time = {"blue": None, "yellow": None}
    perception.reset_target_memory()


def current_path():
    return _current_route


def state_name():
    return _NAMES.get(_state, "?")


def pillar_cells():
    out = {}
    for c in ("blue", "yellow"):
        mem = perception.get_target_memory(c)
        out[c] = mapping.world_to_map(mem[0], mem[1]) if mem is not None else None
    return out


def _within_mark_dist(color, colors):
    if not colors.get(color):
        return False
    ratio = colors.get(f"{color}_ratio", 0.0)
    bearing = colors.get(f"{color}_bearing_rad")
    if ratio >= MISSION_MARK_RATIO:
        return True
    if (bearing is not None and abs(bearing) < MISSION_MARK_BEARING_RAD
            and exploration._get_lidar_front_min_dist(angle_range_deg=15) < MISSION_MARK_LASER_M):
        return True
    return False


def _perceive(colors=None):
    if colors is None:
        colors = sensors.read_color_detections()
    pose = localization.get_pose()
    for color in ("blue", "yellow"):
        if not colors.get(color):
            continue
        if _MISSION_DEBUG:
            _last_seen_time[color] = devices.robot.getTime()   
        bearing = colors.get(f"{color}_bearing_rad")
        dist = colors.get(f"{color}_distance_m", float("inf"))
        if bearing is None or not math.isfinite(dist):
            continue
        if not _registered[color]:
            perception.update_target_memory(color, pose, bearing, dist)
        if not _seen[color]:
            print(f"[MISSION] {color} column spotted (dist~{dist:.2f} m) — approaching to confirm")
            _seen[color] = True
        if _within_mark_dist(color, colors) and not _registered[color]:
            fd = exploration._get_lidar_front_min_dist(angle_range_deg=15)
            stamp_d = fd if math.isfinite(fd) and fd <= MISSION_MARK_MAX_STAMP_M else MISSION_MARK_MAX_STAMP_M
            world = perception.target_world_position(pose, bearing, stamp_d)
            if world is not None:
                mapping.mark_pillar(mapping.world_to_map(world[0], world[1]), color)
                perception.update_target_memory(color, pose, bearing, stamp_d)
            _registered[color] = True
            print(f"[MISSION] {color} pillar reached & marked on map "
                  f"(laser={fd:.2f} m, ratio={colors.get(f'{color}_ratio', 0):.3f})")


def _close_to_mem(color):
    mem = perception.get_target_memory(color)
    if mem is None:
        return False
    px, py = localization.get_position()
    return math.hypot(mem[0] - px, mem[1] - py) <= PILLAR_COMMIT_DIST_M


def _should_commit(color, colors):
    return _within_mark_dist(color, colors) or _close_to_mem(color)



def _approach_cell(pose, world):
    px, py = pose[0], pose[1]
    wx, wy = world
    dx, dy = wx - px, wy - py
    d = math.hypot(dx, dy)
    if d < 1e-6:
        return mapping.world_to_map(wx, wy)
    ax = wx - APPROACH_OFFSET_M * dx / d
    ay = wy - APPROACH_OFFSET_M * dy / d
    return mapping.world_to_map(ax, ay)


def _tick():
    global _tick_count
    result = devices.robot.step(devices.timestep)
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
    return result


RETRY_SAME_ATTEMPTS = 2
MAX_REJOIN_ATTEMPTS = 4


def _nearest_remaining_index(path, from_index, robot_cell):
    start = min(int(from_index), len(path) - 1)
    best_i = start
    best_d = float("inf")
    for i in range(start, len(path)):
        dx = path[i][0] - robot_cell[0]
        dy = path[i][1] - robot_cell[1]
        d = dx * dx + dy * dy
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


def _resume_after_recovery(path, target_index, recover_count):
    if recover_count <= RETRY_SAME_ATTEMPTS:
        return target_index, False                     
    if recover_count <= MAX_REJOIN_ATTEMPTS:
        if exploration._path_blocked(path[target_index:]):
            return target_index, True                 
        return _nearest_remaining_index(
            path, target_index, exploration._get_map_position()), False
    return target_index, True                         


def _drive_log_return(color, reason, value):
    if _MISSION_DEBUG:
        x, y, th = localization.get_pose()
        last = _last_seen_time.get(color)
        now = devices.robot.getTime()
        seen = f"{now - last:.1f}s ago" if last is not None else "never"
        print(f"[DRIVE/exit] {color} reason={reason} -> {value}  "
              f"pose=({x:+.2f}, {y:+.2f}, {math.degrees(th):+.0f}deg)  last_seen={seen}")
    return value


def _drive_to(color, should_continue):
    global _current_route
    replans = 0
    while should_continue():
        world = perception.get_target_memory(color)
        if world is None:
            return _drive_log_return(color, "no_memory", False)

        goal_cell = _approach_cell(localization.get_pose(), world)
        route = planning.plan(tuple(exploration._get_map_position()), goal_cell,
                              block_unknown=False)
        if _MISSION_DEBUG:
            _rx, _ry, _rth = localization.get_pose()
            _dist = math.hypot(world[0] - _rx, world[1] - _ry)
            _herr = ((math.atan2(world[1] - _ry, world[0] - _rx) - _rth + math.pi)
                     % (2.0 * math.pi) - math.pi)
            _rlen = len(route) if route else 0
            print(f"[DRIVE/plan] {color} pose=({_rx:+.2f}, {_ry:+.2f}, "
                  f"{math.degrees(_rth):+.0f}deg) mem=({world[0]:+.2f}, {world[1]:+.2f}) "
                  f"goal_cell={goal_cell} dist={_dist:.2f}m "
                  f"head_err={math.degrees(_herr):+.0f}deg route_len={_rlen} "
                  f"valid={bool(route) and _rlen >= 2}")
        if not route or len(route) < 2:
            colors = sensors.read_color_detections()
            _perceive(colors)
            if _within_mark_dist(color, colors):
                exploration._stop_motor()
                return _drive_log_return(color, "already_marked", True)
            return _drive_log_return(color, "planning_failed", False)

        current_path = list(route)
        _current_route = current_path      
        target_index = FOLLOW_WAYPOINT_STRIDE
        need_replan = False
        recover_count = 0       

        while target_index < len(current_path) and not need_replan:
            target = current_path[target_index]
            while _tick() != -1:
                if not should_continue():
                    exploration._stop_motor()
                    return _drive_log_return(color, "stopped", False)

                colors = sensors.read_color_detections()
                _perceive(colors)                       
                if _within_mark_dist(color, colors):    
                    exploration._stop_motor()
                    return _drive_log_return(color, "mark_confirmed", True)

                bearing = colors.get(f"{color}_bearing_rad")
                if colors.get(color) and bearing is not None and not exploration._obstacle_in_front():
                    if abs(bearing) > SEEK_BEARING_DEADBAND_RAD:
                        motion.drive_twist(0.0, math.copysign(SEEK_OMEGA, bearing))
                    else:
                        motion.drive_twist(SEEK_LIN_VEL, 0.0)
                    continue

                if exploration._obstacle_in_front():
                    exploration._recover_from_obstacle()
                    recover_count += 1
                    target_index, need_replan = _resume_after_recovery(
                        current_path, target_index, recover_count)
                    break

                reached, is_stuck = exploration._follow_local_target(target)
                if is_stuck:
                    exploration._recover_from_stuck()
                    recover_count += 1
                    target_index, need_replan = _resume_after_recovery(
                        current_path, target_index, recover_count)
                    break
                if reached:
                    recover_count = 0                      
                    target_index += FOLLOW_WAYPOINT_STRIDE  
                    break
            else:
                break       

        replans += 1
        if replans > 12:            
            exploration._stop_motor()
            return _drive_log_return(color, "mark_not_confirmed", False)

    exploration._stop_motor()
    return _drive_log_return(color, "stopped", False)


def _seek_and_reach(color, should_continue):
    attempts = 0
    while should_continue():
        if perception.get_target_memory(color) is None:
            def _hook():
                colors = sensors.read_color_detections()
                _perceive(colors)                         
                mem = perception.get_target_memory(color)
                if mem is not None:
                    exploration.set_target_bias(mem)      
                    if _should_commit(color, colors):
                        return False                     
                return should_continue()

            print(f"[MISSION] exploring to find {color} pillar...")
            exploration.reset()
            exploration.run(_hook)
            exploration.clear_target_bias()
            if perception.get_target_memory(color) is None:
                return False                              

        print(f"[MISSION] {color} localized -> driving to it...")
        if _drive_to(color, should_continue):
            return True

        if _registered[color]:
            print(f"[MISSION] {color} not reached -> retrying drive (keeping marked position)")
        else:
            print(f"[MISSION] {color} not reached at remembered spot -> re-acquiring")
            perception.forget_target(color)
        attempts += 1
        if attempts > 6:
            return False
    return False


# ── Entry point ───────────────────────────────────────────────────────────────

def run(should_continue):
    global _state
    reset()
    exploration.reset()

    _state = SEEKING_BLUE
    print("[MISSION] SEEKING_BLUE")
    if not _seek_and_reach("blue", should_continue):
        print("[MISSION] stopped before reaching blue")
        return
    print("[MISSION] BLUE reached")

    _state = SEEKING_YELLOW
    if perception.get_target_memory("yellow") is not None:
        print("[MISSION] SEEKING_YELLOW (yellow already seen while seeking blue -> traceback)")
    else:
        print("[MISSION] SEEKING_YELLOW")
    if not _seek_and_reach("yellow", should_continue):
        print("[MISSION] stopped before reaching yellow")
        return
    print("[MISSION] YELLOW reached")

    _state = DONE
    motion.stop_robot()
    print("[MISSION] DONE — reached blue then yellow.")
