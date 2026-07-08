"""Blue-then-yellow mission (FR5) for the Husarion RosBot.

Sequential state machine SEEKING_BLUE -> SEEKING_YELLOW -> DONE, the most
time-efficient ordering that still satisfies the spec's hard blue-first rule
(CLAUDE.md / FR5: reach blue first, then yellow; minimize simulation time):

  * SEEKING_BLUE  — explore only until BLUE is seen (localized), then drive to it.
                    Any YELLOW glimpsed along the way is opportunistically
                    remembered (but not chased).
  * SEEKING_YELLOW— if yellow was already seen while seeking blue, head straight
                    there (traceback, no re-exploration); otherwise explore until
                    yellow is seen, then drive to it.
  * DONE          — both reached, stop.

This is strictly better than a "find both, then backtrack to blue" scheme (which
traverses the blue<->yellow gap twice) and than the reference's approach (which
ends at yellow without guaranteeing blue-first).

Ideas reused from the reference (Hieu Tran et al.): the column reached heuristic
(frame-ratio / depth / front-lidar close) and the plan->follow->recover drive
structure of follow_final_path.  Column PERCEPTION is this project's synchronous
sensors.read_color_detections (our FR4 port of their estimate_column_distance /
HSV) + perception world-projection/memory, not their background camera thread.
Not ported: red-wall dead-end closure (maze-specific -> violates generalization).

Blocking, like exploration.run(): call mission.run(should_continue); it drives
until DONE or should_continue() returns False.  The background SLAM thread keeps
mapping throughout; this module only reads the SLAM pose/map.
"""

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

# ── State ─────────────────────────────────────────────────────────────────────
# _seen: printed-once flag when a column is first spotted (any distance) — a
#        sighting sets perception memory so the robot can navigate toward it.
# _registered: the pillar has been CONFIRMED at contact range and stamped
#        on the map in its colour (the "visited/seen" mark the min-distance gate
#        controls).  A distant glimpse never registers a pillar.
_seen = {"blue": False, "yellow": False}
_registered = {"blue": False, "yellow": False}
_state = SEEKING_BLUE
_tick_count = 0
_current_route = None   # the live A* route being driven (for the map overlay)

# ── TEMPORARY mission diagnostics (Phase 1) ───────────────────────────────────
# Instrumentation ONLY — no logic/behaviour change.  Traces each _drive_to
# planning attempt and every _drive_to return reason, so the "blue not reached"
# failure can be split into planning-failure vs mark-acceptance-failure.  Set
# _MISSION_DEBUG = False to silence.
_MISSION_DEBUG = True
_last_seen_time = {"blue": None, "yellow": None}   # sim time (s) each pillar was last visible


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
    """The complete route the mission is currently driving (list of (x, y) map
    cells), e.g. the blue->yellow traceback path — for the live-map overlay."""
    return _current_route


def state_name():
    return _NAMES.get(_state, "?")


def pillar_cells():
    """Remembered blue/yellow pillar map cells (for the live-map overlay)."""
    out = {}
    for c in ("blue", "yellow"):
        mem = perception.get_target_memory(c)
        out[c] = mapping.world_to_map(mem[0], mem[1]) if mem is not None else None
    return out


# ── Perception: localize columns (any distance) + register when close ─────────

def _within_mark_dist(color, colors):
    """True only if `color` is CONFIRMED genuinely at the pillar — via signals that
    don't depend on the unreliable depth-Pythagoras distance:
      1. the mask fills MISSION_MARK_RATIO of the frame (pillar dominates view), OR
      2. the front LASER is at contact range (< MISSION_MARK_LASER_M) while the
         pillar is centred (|bearing| < MISSION_MARK_BEARING_RAD) — a wall can't
         trigger it because the pillar must be visible AND pointing forward.
    (The depth distance is deliberately NOT used here: it underestimates far
    pillars badly, which caused false 'reached' marks from across the map.)
    """
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
    """Per-tick perception for the mission.  For each visible column:
      * update perception memory with its projected world position (used to
        navigate toward it — a distant sighting is enough for this), and
      * once CONFIRMED at the pillar (_within_mark_dist), stamp its cell on the
        map in its colour and flag it registered/visited (the mark gate).
    """
    if colors is None:
        colors = sensors.read_color_detections()
    pose = localization.get_pose()
    for color in ("blue", "yellow"):
        if not colors.get(color):
            continue
        if _MISSION_DEBUG:
            _last_seen_time[color] = devices.robot.getTime()   # Phase-1 diagnostic only
        bearing = colors.get(f"{color}_bearing_rad")
        dist = colors.get(f"{color}_distance_m", float("inf"))
        if bearing is None or not math.isfinite(dist):
            continue
        # Localize (for navigation direction) with the camera-depth sighting only
        # UNTIL the pillar is registered.  Once it's marked, the memory is frozen at
        # the accurate stamped (laser) position below, so stop letting noisy depth
        # sightings drift it — that frozen position is what backtracking uses.
        if not _registered[color]:
            perception.update_target_memory(color, pose, bearing, dist)
        if not _seen[color]:
            print(f"[MISSION] {color} column spotted (dist~{dist:.2f} m) — approaching to confirm")
            _seen[color] = True
        # Register + colour-stamp only once CONFIRMED at the pillar.
        if _within_mark_dist(color, colors) and not _registered[color]:
            # Stamp at the reliable close range (front laser, clamped), directly
            # ahead — the pillar is centred at this moment — NOT the bad depth dist.
            fd = exploration._get_lidar_front_min_dist(angle_range_deg=15)
            stamp_d = fd if math.isfinite(fd) and fd <= MISSION_MARK_MAX_STAMP_M else MISSION_MARK_MAX_STAMP_M
            world = perception.target_world_position(pose, bearing, stamp_d)
            if world is not None:
                mapping.mark_pillar(mapping.world_to_map(world[0], world[1]), color)
                # Remember THIS stamped position for backtracking — freeze the memory
                # at the laser-based mark so it matches the map stamp (and the drawn
                # circle) instead of the drifting depth sighting.
                perception.update_target_memory(color, pose, bearing, stamp_d)
            _registered[color] = True
            print(f"[MISSION] {color} pillar reached & marked on map "
                  f"(laser={fd:.2f} m, ratio={colors.get(f'{color}_ratio', 0):.3f})")


# ── Commit gate: biased-explore -> final-drive handoff ────────────────────────

def _close_to_mem(color):
    """True if the robot is within PILLAR_COMMIT_DIST_M of the remembered pillar."""
    mem = perception.get_target_memory(color)
    if mem is None:
        return False
    px, py = localization.get_position()
    return math.hypot(mem[0] - px, mem[1] - py) <= PILLAR_COMMIT_DIST_M


def _should_commit(color, colors):
    """Stop biased exploration and hand off to the precise final drive once the
    pillar is confirmed at contact range OR the robot is within the commit
    distance of the remembered sighting.  Approaching under bias first (rather
    than committing from a far, depth-noisy glimpse) avoids arrive-at-wrong-spot
    retries."""
    return _within_mark_dist(color, colors) or _close_to_mem(color)


# ── Final drive helpers ───────────────────────────────────────────────────────

def _approach_cell(pose, world):
    """Map cell APPROACH_OFFSET_M in front of a pillar (the pillar cell itself is
    an obstacle), along the robot->pillar line."""
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
    """Advance the sim one step and feed odometry to SLAM (predict); the
    background thread keeps folding scans into the map.  Green marking runs on its
    cadence.  Returns the raw step result (-1 on sim end)."""
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
    # Overhead/floating-wall marking (depth-camera based): stamp floating walls
    # above the lidar's scan plane as hard obstacles (CELL_CLOSED) so the final
    # drive's planner/DWA route around them.
    if (OVERHEAD_MARK_ENABLED
            and _tick_count % SLAM_GREEN_PERIOD_STEPS == 0
            and not motion.is_turning()):
        overhead_pts = sensors.overhead_obstacle_points_body()
        if len(overhead_pts) > 0:
            mapping.mark_overhead(localization.get_pose(), overhead_pts)
    return result


# Local-recovery escalation for the final drive.  After a recovery the robot
# resumes the EXISTING optimal A* route rather than discarding it and replanning,
# escalating in tiers keyed on the count of consecutive recoveries WITHOUT reaching
# a waypoint (reset to 0 whenever one is reached):
#   1 .. RETRY_SAME_ATTEMPTS   -> retry the SAME waypoint (most recoveries move only
#                                 a few cm, so the original target is usually still valid)
#   .. MAX_REJOIN_ATTEMPTS     -> rejoin from the nearest remaining waypoint; and only
#                                 now (map settled) test whether the route is blocked
#   beyond MAX_REJOIN_ATTEMPTS -> global replan (last resort)
RETRY_SAME_ATTEMPTS = 2
MAX_REJOIN_ATTEMPTS = 4


def _nearest_remaining_index(path, from_index, robot_cell):
    """Index of the waypoint AT OR AHEAD OF from_index that is closest to
    robot_cell ((map_x, map_y)).  Scans only forward from from_index, so a
    recovery can never rewind the follow cursor onto an already-passed waypoint.
    Used to rejoin the existing global path after a local recovery instead of
    discarding it and running a fresh A* search."""
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
    """Decide how to resume the drive after a LOCAL recovery; returns
    (new_target_index, need_replan).  Preserves the existing global path as long
    as possible via the tiered escalation described in the module constants:
      * retry the SAME waypoint first (target unchanged — most recoveries move
        only a few cm, so the original target is usually still reachable),
      * then rejoin the nearest remaining waypoint — and only at this point, once
        the map has settled past the first recoveries, test whether the remaining
        route is genuinely blocked (deferred path-invalidation),
      * then, as a last resort, request a global replan."""
    if recover_count <= RETRY_SAME_ATTEMPTS:
        return target_index, False                      # retry the same waypoint
    if recover_count <= MAX_REJOIN_ATTEMPTS:
        if exploration._path_blocked(path[target_index:]):
            return target_index, True                   # route genuinely blocked -> replan
        return _nearest_remaining_index(
            path, target_index, exploration._get_map_position()), False
    return target_index, True                           # repeated failure -> global replan


def _drive_log_return(color, reason, value):
    """Phase-1 diagnostic: log the exact _drive_to exit reason, final pose, and how
    long ago the pillar was last visible, then return `value` UNCHANGED (no
    behaviour change — every _drive_to return is wrapped in this)."""
    if _MISSION_DEBUG:
        x, y, th = localization.get_pose()
        last = _last_seen_time.get(color)
        now = devices.robot.getTime()
        seen = f"{now - last:.1f}s ago" if last is not None else "never"
        print(f"[DRIVE/exit] {color} reason={reason} -> {value}  "
              f"pose=({x:+.2f}, {y:+.2f}, {math.degrees(th):+.0f}deg)  last_seen={seen}")
    return value


def _drive_to(color, should_continue):
    """Drive to the remembered `color` pillar: known-free A* to a stand-off cell
    in front of it + DWA following, continuously re-planning toward the (refined)
    remembered position after each finished path or recovery.  Returns True once
    the pillar is CONFIRMED within the registration distance (_within_mark_dist),
    False if aborted/unreachable.
    """
    global _current_route
    replans = 0
    while should_continue():
        world = perception.get_target_memory(color)
        if world is None:
            return _drive_log_return(color, "no_memory", False)

        goal_cell = _approach_cell(localization.get_pose(), world)
        # Route THROUGH unknown space (block_unknown=False), like the frontier
        # planner.  The traceback to a KNOWN/marked pillar (blue->yellow) crosses
        # cells the robot never mapped on its winding way out; the default goal-run
        # mode makes every unknown cell a wall, so A* finds no route and the drive
        # gives up instantly (the "retrying drive" spam that ends in "stopped before
        # reaching yellow").  Real obstacles in the unmapped stretch are handled by
        # the governor / bumper / recovery as the robot advances.
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
            # Can't plan a path -- if we're already confirmed at the pillar, win.
            colors = sensors.read_color_detections()
            _perceive(colors)
            if _within_mark_dist(color, colors):
                exploration._stop_motor()
                return _drive_log_return(color, "already_marked", True)
            return _drive_log_return(color, "planning_failed", False)

        current_path = list(route)
        _current_route = current_path        # expose the full route for the live map
        target_index = FOLLOW_WAYPOINT_STRIDE
        need_replan = False
        recover_count = 0        # consecutive recoveries since the last waypoint reached

        while target_index < len(current_path) and not need_replan:
            target = current_path[target_index]
            while _tick() != -1:
                if not should_continue():
                    exploration._stop_motor()
                    return _drive_log_return(color, "stopped", False)

                colors = sensors.read_color_detections()
                _perceive(colors)                        # localize + register both pillars
                if _within_mark_dist(color, colors):     # confirmed within mark distance
                    exploration._stop_motor()
                    return _drive_log_return(color, "mark_confirmed", True)

                # ── Reactive visual approach (autonomous.py seek) ──────────────
                # When the pillar is IN SIGHT, drive straight at it (centre the
                # bearing, then creep forward) instead of following the A* path to
                # the depth-estimated cell.  Line-of-sight so it reliably closes
                # the final gap and marks — this is what stops the earlier
                # "drive -> not reached -> re-explore" loop.  A* below still runs
                # when the pillar is NOT visible (to bring it into view).
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
                    recover_count = 0                       # progress -> reset rejoin budget
                    target_index += FOLLOW_WAYPOINT_STRIDE   # advance ONLY on a real reach
                    break
            else:
                break        # inner loop ended via _tick()==-1 (sim end) -> stop following

        replans += 1
        if replans > 12:            # give up after persistent failure to make progress
            exploration._stop_motor()
            return _drive_log_return(color, "mark_not_confirmed", False)

    exploration._stop_motor()
    return _drive_log_return(color, "stopped", False)


def _seek_and_reach(color, should_continue):
    """Reach the `color` pillar: explore until it is SEEN (localized) if we don't
    already know where it is, then drive to it.  While exploring for it, any OTHER
    pillar seen is opportunistically remembered (so a yellow spotted while seeking
    blue is available for free later).  Retries (forget stale memory + re-explore)
    if a drive arrives at the remembered spot but the pillar isn't actually there.
    Returns True when the pillar is reached, False if aborted.
    """
    attempts = 0
    while should_continue():
        # ── If the pillar's location is UNKNOWN, explore to find it ───────────
        # (biased toward it once sighted; a sighting steers exploration but does
        # NOT stop it — only the commit distance / contact range does).  If we
        # ALREADY know where it is (e.g. yellow spotted while seeking blue), skip
        # exploration entirely and go straight to the direct traceback drive,
        # which plans and follows the full blue->yellow path.
        if perception.get_target_memory(color) is None:
            def _hook():
                colors = sensors.read_color_detections()
                _perceive(colors)                         # localize + register whatever is visible
                mem = perception.get_target_memory(color)
                if mem is not None:
                    exploration.set_target_bias(mem)      # steer exploration toward the active pillar
                    if _should_commit(color, colors):
                        return False                      # close/confirmed -> hand off to the drive
                return should_continue()

            print(f"[MISSION] exploring to find {color} pillar...")
            exploration.reset()
            exploration.run(_hook)
            exploration.clear_target_bias()
            if perception.get_target_memory(color) is None:
                return False                              # aborted before ever seeing it

        # ── Drive to it ───────────────────────────────────────────────────────
        print(f"[MISSION] {color} localized -> driving to it...")
        if _drive_to(color, should_continue):
            return True

        # Don't discard a pillar we've already MARKED (reached + colour-stamped on
        # the map): its position is known-good, so forgetting it and re-exploring is
        # the robot wandering with the pillar already on the map.  Keep the position
        # and retry the drive.  Only a never-confirmed (far-glimpsed, unregistered)
        # pillar is forgotten as a stale sighting and re-acquired by exploration.
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
    """Sequential blue-then-yellow mission (FR5), most time-efficient ordering:
    seek+reach BLUE, then seek+reach YELLOW.  Yellow glimpsed while seeking blue
    is remembered, so the yellow phase heads straight there (traceback) instead of
    re-exploring.  Runs until DONE or should_continue() returns False.
    """
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
