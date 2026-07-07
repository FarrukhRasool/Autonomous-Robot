"""Husarion RosBot controller — main entry point.

Imports all subsystems and runs the Webots control loop.
Contains no sensor reads, kinematics, or navigation logic.

Keys: F/S/A/D drive | Space stop | T self-test | G autonomous |
      I snapshot | L log | O pose | R reset pose & map | M map summary |
      P target bearings
"""

import math

import devices                          # hardware init (must import first)
import sensors
import motion
import localization
import mapping
import perception
import planning
import following
import exploration
import mission
import slam
import visualizer
from autonomous import autonomous_step, reset_autonomous_state, reset_mission_state
from sensor_debug import format_compact_sensors, format_sensor_snapshot
from config import (
    TARGET_LIN_VEL, TARGET_ANG_VEL, FRONT_STOP_DIST,
    POSE_LOG_PERIOD_STEPS,
    SLAM_GREEN_PERIOD_STEPS,
    VIZ_PERIOD_STEPS, GREEN_MARK_ENABLED, FOLLOW_MAX_RETRIES,
)

# ── Controller state ───────────────────────────────────────────────────────────
step_count       = 0
previous_pressed = set()
test_timer       = 0
sensor_log       = False
auto_mode        = False
block_timer      = 0
plan_targets     = []    # up to 2 goal cells captured by B; V plans between them
last_route       = None  # most recently planned path, shown in the live view
viz_on           = False # live cv2 map window toggle (C key)
follow_mode      = False # DWA path-follow mode toggle (Y key)
follow_retries   = 0     # replan-on-stuck counter for manual Follow

print(
    "Controller ready.  Keys: F/S/A/D drive | Space stop | T self-test | "
    "G autonomous mode | I sensor snapshot | L toggle sensor log | "
    "O pose snapshot | R reset pose & map | M map summary | "
    "P target bearings | B set goal | V plan to goal | C live map view | "
    "Y follow path | E explore | X mission (blue->yellow)"
)


OVERHEAD_CENTER_BLOCK_DIST = 0.55
OVERHEAD_SIDE_WARN_DIST = 0.40
OVERHEAD_SLOW_VEL = 0.04
OVERHEAD_STEER_OMEGA = 0.12

# The E/X start key is held for many sim steps and Webots AUTO-REPEATS it with
# 1-tick gaps.  A blocking mode (explore/mission) must only arm its stop-listener
# after the start key has been RELEASED for this many consecutive ticks, so an
# auto-repeat gap can't arm it early (which would let the next repeat self-cancel
# the mode).  ~10 ticks (~0.3 s) is well above the auto-repeat gap.
KEY_ARM_RELEASE_TICKS = 10


# Start the background SLAM mapping thread (folds lidar scans into the map at
# ~10 Hz off the control loop; localization feeds it motion deltas every step).
slam.start_mapping_thread(sensors.read_lidar_pointcloud_2d, motion.is_turning)


# ── Main loop ──────────────────────────────────────────────────────────────────
while devices.robot.step(devices.timestep) != -1:
    step_count += 1
    reset_this_step = False
    # ── Odometry update (runs every step, before any control logic) ───────────
    left_rad, right_rad = sensors.read_wheel_angles()
    imu_yaw             = sensors.read_imu_yaw()
    gyro_z              = sensors.read_gyro_z()
    localization.update_from_encoders(left_rad, right_rad, imu_yaw, gyro_z)

    # ── Pose + green marking ───────────────────────────────────────────────────
    # SLAM runs on its own background thread (started before this loop): it folds
    # lidar scans into the map (slam.observe) at ~10 Hz and localization feeds it
    # motion deltas via slam.predict every step.  This loop only reads the pose
    # and stamps green ground (camera-based, so it stays on the main thread; the
    # grid write is done under mapping.LOCK inside mark_green).
    robot_x, robot_y, robot_theta = localization.get_pose()   # SLAM estimate
    if (not reset_this_step
            and GREEN_MARK_ENABLED
            and step_count % SLAM_GREEN_PERIOD_STEPS == 0
            and not motion.is_turning()):
        green_pts = sensors.green_ground_points_body()
        if len(green_pts) > 0:
            mapping.mark_green((robot_x, robot_y, robot_theta), green_pts)

    # Keyboard edge detection: new_keys fires only on the step a key first appears
    pressed_now = set()
    k = devices.keyboard.getKey()
    while k != -1:
        pressed_now.add(k)
        k = devices.keyboard.getKey()
    new_keys = pressed_now - previous_pressed

    v_cmd     = 0.0
    omega_cmd = 0.0
    sel_label = ''

    # ── One-shot key actions ───────────────────────────────────────────────────
    if ord('I') in new_keys or ord('i') in new_keys:
        print(format_sensor_snapshot(sensors.read_sensor_snapshot()))

    if ord('L') in new_keys or ord('l') in new_keys:
        sensor_log = not sensor_log
        print(f"Sensor logging {'ON' if sensor_log else 'OFF'}")

    if ord('O') in new_keys or ord('o') in new_keys:
        print(f"[POSE] {localization.format_pose()}")

    if ord('R') in new_keys or ord('r') in new_keys:
        localization.reset_pose()
        mapping.clear()
        reset_autonomous_state()
        reset_mission_state()
        perception.reset_target_memory()
        # Clear planning/following debug state too.
        plan_targets = []
        last_route = None
        follow_mode = False
        following.reset()
        exploration.reset()
        mission.reset()
        reset_this_step = True
        print("[POSE] reset to (0, 0, 0); map cleared; targets & path cleared")

    if ord('M') in new_keys or ord('m') in new_keys:
        rx, ry, _ = localization.get_pose()
        print(mapping.summary(robot_xy=(rx, ry)))
        print(f"[MAP] saved {mapping.save_png('map.png')}")

    if ord('B') in new_keys or ord('b') in new_keys:
        if len(plan_targets) >= 2:
            plan_targets = []           # third press starts a fresh A/B pair
        cell = mapping.robot_map_pos(localization.get_pose())
        plan_targets.append(cell)
        label = "A" if len(plan_targets) == 1 else "B"
        print(f"[PLAN] target {label} set to map cell {cell}  ({len(plan_targets)}/2)")

    if ord('V') in new_keys or ord('v') in new_keys:
        if not plan_targets:
            print("[PLAN] no target — press B to set one (B again for A->B)")
        else:
            if len(plan_targets) >= 2:
                start_cell, goal_cell = plan_targets[0], plan_targets[1]   # A -> B
            else:
                start_cell = mapping.robot_map_pos(localization.get_pose())
                goal_cell = plan_targets[0]                                # robot -> A
            route = planning.plan(start_cell, goal_cell)
            last_route = route
            if route:
                length_m = 0.0
                pw = mapping.map_to_world(*route[0])
                for c in route[1:]:
                    cw = mapping.map_to_world(*c)
                    length_m += math.hypot(cw[0] - pw[0], cw[1] - pw[1])
                    pw = cw
                saved = mapping.save_png('path.png', path_cells=route)
                print(f"[PLAN] {start_cell} -> {goal_cell}: {len(route)} waypoints, "
                      f"{length_m:.2f} m; saved {saved}")
            else:
                print(f"[PLAN] {start_cell} -> {goal_cell}: UNREACHABLE")

    if ord('C') in new_keys or ord('c') in new_keys:
        viz_on = not viz_on
        if not viz_on:
            visualizer.close()
        print(f"Live map view {'ON' if viz_on else 'OFF'} (cv2 window)")

    if ord('P') in new_keys or ord('p') in new_keys:
        _colors = sensors.read_color_detections()
        _pose   = localization.get_pose()

        def _fmt_target(label, bearing, distance):
            world = perception.target_world_position(_pose, bearing, distance)
            b_s = (
                f"{bearing:+.3f} rad ({math.degrees(bearing):+.1f}°)"
                if bearing is not None else "None"
            )
            d_s = (
                f"{distance:.3f} m"
                if (distance is not None and math.isfinite(distance)) else "inf"
            )
            w_s = f"({world[0]:+.3f}, {world[1]:+.3f}) m" if world is not None else "None"
            return f"{label}: bearing={b_s}  distance={d_s}  world={w_s}"

        print("[PERCEPT] " + _fmt_target("blue",
                                         _colors.get("blue_bearing_rad"),
                                         _colors.get("blue_distance_m")))
        print("          " + _fmt_target("yellow",
                                         _colors.get("yellow_bearing_rad"),
                                         _colors.get("yellow_distance_m")))

    if ord('G') in new_keys or ord('g') in new_keys:
        auto_mode = not auto_mode
        if not auto_mode:
            block_timer = 0
            reset_autonomous_state()
        print(f"Autonomous mode {'ON' if auto_mode else 'OFF'}")

    # ── Y: follow the last planned path with the DWA follower ─────────────────
    if ord('Y') in new_keys or ord('y') in new_keys:
        follow_mode = not follow_mode
        if follow_mode:
            goal_cell = plan_targets[-1] if plan_targets else None
            if goal_cell is None:
                follow_mode = False
                print("Follow mode: no goal — press B at the target first")
            else:
                # Always plan from the robot's CURRENT cell to the goal, so the
                # path starts where the robot is (no navigating to a distant A).
                start_cell = mapping.robot_map_pos(localization.get_pose())
                route = planning.plan(start_cell, goal_cell)
                if route:
                    last_route = route
                    following.set_path(route)
                    follow_retries = 0
                    print(f"Follow mode ON — driving {len(route)}-waypoint path "
                          f"robot->goal (DWA)")
                else:
                    follow_mode = False
                    print(f"Follow mode: no path from robot to goal {goal_cell}")
        else:
            following.reset()
            print("Follow mode OFF")

    # ── E: frontier exploration (faithful blocking port of the reference).
    #     Runs its own sim-step loop until E or Space is pressed again. ─────────
    if ord('E') in new_keys or ord('e') in new_keys:
        follow_mode = False
        auto_mode = False
        following.reset()
        reset_autonomous_state()
        # Fresh SLAM + map + pose for this run — mirrors the reference, where every
        # `main()` builds a brand-new MyRobot() (empty SlamSystem, empty grid, pose
        # at origin).  Prevents the pose graph accumulating across runs (which grows
        # loop-closure cost every session -> progressive slowdown).
        localization.reset_pose()
        mapping.clear()

        # Arm the stop-listener only after E/Space has been released for a
        # sustained run of ticks (KEY_ARM_RELEASE_TICKS), so the held/auto-repeated
        # start key can't arm it early and then self-cancel exploration.
        explore_stop = {"armed": False, "release": 0}
        _STOP_KEYS = {ord('E'), ord('e'), ord(' ')}

        def _explore_should_continue():
            # Polled every sim tick inside exploration: refresh the live map and
            # stop on a FRESH E/Space press (once armed).
            if viz_on:
                visualizer.render(
                    mapping.get_grid(),
                    robot_cell=mapping.robot_map_pos(localization.get_pose()),
                    goals=[exploration.current_goal()] if exploration.current_goal() else [],
                    path=exploration.current_path(),
                )
            keys = set()
            kk = devices.keyboard.getKey()
            while kk != -1:
                keys.add(kk)
                kk = devices.keyboard.getKey()
            stop_pressed = bool(keys & _STOP_KEYS)
            if not explore_stop["armed"]:
                explore_stop["release"] = 0 if stop_pressed else explore_stop["release"] + 1
                if explore_stop["release"] >= KEY_ARM_RELEASE_TICKS:
                    explore_stop["armed"] = True
                return True
            return not stop_pressed

        print("Explore mode ON — frontier exploration (E/Space to stop)")
        exploration.run(_explore_should_continue)
        motion.stop_robot()
        v_cmd = omega_cmd = 0.0
        print("Explore mode OFF")

    # ── X: blue-then-yellow mission (FR5) — faithful blocking port. ────────────
    #     Explores to find both columns, then drives blue->yellow. ──────────────
    if ord('X') in new_keys or ord('x') in new_keys:
        follow_mode = False
        auto_mode = False
        following.reset()
        reset_autonomous_state()
        # Fresh SLAM + map + pose for this run (see E handler) — every mission
        # starts clean, exactly like the reference's per-process MyRobot().
        localization.reset_pose()
        mapping.clear()

        mission_stop = {"armed": False, "release": 0}
        _MSTOP_KEYS = {ord('X'), ord('x'), ord(' ')}

        def _mission_should_continue():
            if viz_on:
                pcs = mission.pillar_cells()
                goals = [c for c in (pcs["blue"], pcs["yellow"]) if c is not None]
                if not goals and exploration.current_goal() is not None:
                    goals = [exploration.current_goal()]
                visualizer.render(
                    mapping.get_grid(),
                    robot_cell=mapping.robot_map_pos(localization.get_pose()),
                    goals=goals,
                    path=following.current_path() or exploration.current_path(),
                )
            keys = set()
            kk = devices.keyboard.getKey()
            while kk != -1:
                keys.add(kk)
                kk = devices.keyboard.getKey()
            stop_pressed = bool(keys & _MSTOP_KEYS)
            if not mission_stop["armed"]:
                mission_stop["release"] = 0 if stop_pressed else mission_stop["release"] + 1
                if mission_stop["release"] >= KEY_ARM_RELEASE_TICKS:
                    mission_stop["armed"] = True
                return True
            return not stop_pressed

        print("Mission ON — blue-then-yellow (X/Space to stop)")
        mission.run(_mission_should_continue)
        motion.stop_robot()
        v_cmd = omega_cmd = 0.0
        print("Mission OFF")

    # ── Hard stop: Space exits autonomous/follow mode and zeroes twist ────────
    if ord(' ') in new_keys:
        v_cmd = omega_cmd = 0.0
        if auto_mode:
            auto_mode   = False
            block_timer = 0
            reset_autonomous_state()
            print("Autonomous mode OFF (Space pressed)")
        if follow_mode:
            follow_mode = False
            following.reset()
            print("Follow mode OFF (Space pressed)")

    # ── Follow / autonomous / teleop (mutually exclusive) ─────────────────────
    elif follow_mode:
        # DWA drives the planned path directly (the planned path is already
        # collision- and green-free, and DWA rejects obstacle trajectories).
        v_cmd, omega_cmd, follow_status = following.step(
            (robot_x, robot_y, robot_theta), devices.timestep / 1000.0
        )
        sel_label = f"follow_{follow_status}"
        if step_count % 15 == 0:
            print(f"[FOLLOW] {follow_status} v={v_cmd:+.2f} w={omega_cmd:+.2f} "
                  f"retries={follow_retries}")
        if follow_status == "done":
            follow_mode = False
            following.reset()
            v_cmd = omega_cmd = 0.0
            print("Follow mode OFF (done)")
        elif follow_status in ("stuck", "idle"):
            # Replan robot->goal against the now-updated map and keep going,
            # instead of giving up on the first obstacle (like Mission/Explore).
            goal_cell = plan_targets[-1] if plan_targets else None
            follow_retries += 1
            route = (planning.plan(mapping.robot_map_pos((robot_x, robot_y, robot_theta)),
                                   goal_cell)
                     if goal_cell is not None and follow_retries <= FOLLOW_MAX_RETRIES else None)
            if route:
                following.set_path(route)
                print(f"[FOLLOW] {follow_status} -> replanned ({len(route)} wp, retry {follow_retries})")
            else:
                follow_mode = False
                following.reset()
                v_cmd = omega_cmd = 0.0
                print(f"Follow mode OFF ({follow_status}; retries={follow_retries})")

    elif auto_mode:
        v_cmd, omega_cmd, sel_label, block_timer, dbg = autonomous_step(
            block_timer, localization.get_pose()
        )

        if step_count % 10 == 0:
            print(
                f"[COLOR] green={dbg['green']}:{dbg['green_ratio']:.3f} "
                f"dist={dbg['green_distance']:.3f} "
                f"blue={dbg['blue']}:{dbg['blue_ratio']:.3f} "
                f"yellow={dbg['yellow']}:{dbg['yellow_ratio']:.3f} | "
                f"target={dbg['active_target']} mission={dbg['mission_state']} | "
                f"overhead={dbg['overhead_front']:.3f} "
                f"block_timer={dbg['block_timer']} | "
                f"{sel_label} v={v_cmd:+.2f} omega={omega_cmd:+.2f}"
            )

    else:
        front_min = sensors.get_front_laser_min()
        if ord('F') in pressed_now or ord('f') in pressed_now:
            if front_min >= FRONT_STOP_DIST:
                v_cmd = TARGET_LIN_VEL
        elif ord('S') in pressed_now or ord('s') in pressed_now:
            v_cmd = -TARGET_LIN_VEL
        elif ord('A') in pressed_now or ord('a') in pressed_now:
            omega_cmd = TARGET_ANG_VEL
        elif ord('D') in pressed_now or ord('d') in pressed_now:
            omega_cmd = -TARGET_ANG_VEL
    
    if test_timer > 0:
        v_cmd, omega_cmd = TARGET_LIN_VEL, 0.0
        test_timer -= 1
        if test_timer == 0:
            v_cmd = omega_cmd = 0.0
            print("Motor test complete; stopping motors")

    # ── Advance edge-detection state ───────────────────────────────────────────
    previous_pressed = pressed_now

    # ── Apply twist ────────────────────────────────────────────────────────────
    v_real, omega_real, wL, wR = motion.drive_twist(v_cmd, omega_cmd)

    # ── Live map view (inline cv2 window, throttled) ──────────────────────────
    if viz_on and step_count % VIZ_PERIOD_STEPS == 0:
        viz_path = following.current_path() if follow_mode else last_route
        visualizer.render(
            mapping.get_grid(),
            robot_cell=mapping.robot_map_pos((robot_x, robot_y, robot_theta)),
            goals=plan_targets,
            path=viz_path,
        )

    # ── Periodic pose log ─────────────────────────────────────────────────────
    if step_count % POSE_LOG_PERIOD_STEPS == 0:
        print(f"[POSE] {localization.format_pose()}")
