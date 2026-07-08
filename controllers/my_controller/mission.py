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
    APPROACH_OFFSET_M,
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


def reset():
    global _seen, _registered, _state, _tick_count, _current_route
    _seen = {"blue": False, "yellow": False}
    _registered = {"blue": False, "yellow": False}
    _state = SEEKING_BLUE
    _tick_count = 0
    _current_route = None
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

def _perceive(colors=None):
    """Per-tick perception for the mission.  Recognition + map colour-tagging
    is delegated to perception.tag_pillars_on_map (shared with every other
    mode via my_controller.py / exploration.py, so a pillar is coloured on the
    map no matter which mode drove past it).  This function only layers the
    mission's own one-time "spotted" / "registered" console messages on top,
    using perception.pillar_confirmed as the same confirm gate.
    """
    if colors is None:
        colors = sensors.read_color_detections()
    perception.tag_pillars_on_map(colors)
    for color in ("blue", "yellow"):
        if not colors.get(color):
            continue
        bearing = colors.get(f"{color}_bearing_rad")
        dist = colors.get(f"{color}_distance_m", float("inf"))
        if bearing is None or not math.isfinite(dist):
            continue
        if not _seen[color]:
            print(f"[MISSION] {color} column spotted (dist~{dist:.2f} m) — approaching to confirm")
            _seen[color] = True
        if perception.pillar_confirmed(color, colors) and not _registered[color]:
            _registered[color] = True
            print(f"[MISSION] {color} pillar reached & marked on map "
                  f"(ratio={colors.get(f'{color}_ratio', 0):.3f})")


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
    return perception.pillar_confirmed(color, colors) or _close_to_mem(color)


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
    if (OVERHEAD_MARK_ENABLED
            and _tick_count % SLAM_GREEN_PERIOD_STEPS == 0
            and not motion.is_turning()):
        overhead_pts = sensors.overhead_obstacle_points_body()
        if len(overhead_pts) > 0:
            mapping.mark_overhead(localization.get_pose(), overhead_pts)
    return result


def _drive_to(color, should_continue):
    """Drive to the remembered `color` pillar: known-free A* to a stand-off cell
    in front of it + DWA following, continuously re-planning toward the (refined)
    remembered position after each finished path or recovery.  Returns True once
    the pillar is CONFIRMED within the registration distance
    (perception.pillar_confirmed), False if aborted/unreachable.
    """
    global _current_route
    replans = 0
    while should_continue():
        world = perception.get_target_memory(color)
        if world is None:
            return False

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
        if not route or len(route) < 2:
            # Can't plan a path -- if we're already confirmed at the pillar, win.
            colors = sensors.read_color_detections()
            _perceive(colors)
            if perception.pillar_confirmed(color, colors):
                exploration._stop_motor()
                return True
            return False

        current_path = list(route)
        _current_route = current_path        # expose the full route for the live map
        target_index = FOLLOW_WAYPOINT_STRIDE
        need_replan = False

        while target_index < len(current_path) and not need_replan:
            target = current_path[target_index]
            while _tick() != -1:
                if not should_continue():
                    exploration._stop_motor()
                    return False

                colors = sensors.read_color_detections()
                _perceive(colors)                        # localize + register both pillars
                if perception.pillar_confirmed(color, colors):  # confirmed within mark distance
                    exploration._stop_motor()
                    return True

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
                    need_replan = True
                    break

                reached, is_stuck = exploration._follow_local_target(target)
                if is_stuck:
                    exploration._recover_from_stuck()
                    need_replan = True
                    break
                if reached:
                    break
            target_index += FOLLOW_WAYPOINT_STRIDE

        replans += 1
        if replans > 12:            # give up after persistent failure to make progress
            exploration._stop_motor()
            return False

    exploration._stop_motor()
    return False


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
