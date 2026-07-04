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
import safety
import visualizer
from autonomous import autonomous_step, reset_autonomous_state, reset_mission_state
from sensor_debug import format_compact_sensors, format_sensor_snapshot
from config import (
    TARGET_LIN_VEL, TARGET_ANG_VEL, FRONT_STOP_DIST,
    POSE_LOG_PERIOD_STEPS,
    MAP_UPDATE_PERIOD_STEPS, MAP_UPDATE_MAX_OMEGA,
    VIZ_PERIOD_STEPS, GREEN_MARK_ENABLED,
)

# ── Controller state ───────────────────────────────────────────────────────────
step_count       = 0
previous_pressed = set()
test_timer       = 0
sensor_log       = False
auto_mode        = False
block_timer      = 0
prev_omega       = 0.0   # last step's applied yaw rate — gates mapping while turning
prev_blocked     = False # last step hit a wall-block/reverse — pose unreliable, pause mapping
plan_targets     = []    # up to 2 goal cells captured by B; V plans between them
last_route       = None  # most recently planned path, shown in the live view
viz_on           = False # live cv2 map window toggle (C key)
follow_mode      = False # DWA path-follow mode toggle (Y key)
explore_mode     = False # frontier exploration mode toggle (E key)
mission_mode     = False # blue-then-yellow mission toggle (X key)

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


# ── Main loop ──────────────────────────────────────────────────────────────────
while devices.robot.step(devices.timestep) != -1:
    step_count += 1
    reset_this_step = False
    # ── Odometry update (runs every step, before any control logic) ───────────
    left_rad, right_rad = sensors.read_wheel_angles()
    imu_yaw             = sensors.read_imu_yaw()
    gyro_z              = sensors.read_gyro_z()
    localization.update_from_encoders(left_rad, right_rad, imu_yaw, gyro_z)

    # ── Mapping update (AURE log-odds occupancy from lidar point cloud) ────────
    # Throttled to bound per-step cost, and skipped while turning fast (rotation
    # smears the scan) — a single-loop analog of AURE's "update when not turning"
    # background mapper.  prev_omega is last step's applied yaw rate.
    robot_x, robot_y, robot_theta = localization.get_pose()
    # The robot is physically here -> its footprint is free.  Runs every step and
    # repairs any phantom walls left by odometry slip while wedged.
    mapping.mark_robot_free((robot_x, robot_y, robot_theta))
    if (not reset_this_step
            and step_count % MAP_UPDATE_PERIOD_STEPS == 0
            and abs(prev_omega) < MAP_UPDATE_MAX_OMEGA
            and not prev_blocked):        # skip while wedged/escaping (pose unreliable)
        cloud = sensors.read_lidar_pointcloud_2d()
        if len(cloud) > 0:
            mapping.lidar_update((robot_x, robot_y, robot_theta), cloud)
        # Stamp detected green ground as forbidden so planning/DWA avoid it.
        if GREEN_MARK_ENABLED:
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
    safe_label = ''   # set by safety.apply in autonomous modes; drives the mapping pause

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
        # Clear planning/following/exploration/mission debug state too.
        plan_targets = []
        last_route = None
        follow_mode = False
        explore_mode = False
        mission_mode = False
        following.reset()
        exploration.reset()
        mission.reset()
        reset_this_step = True
        print("[POSE] reset to (0, 0, 0); map cleared; targets & path cleared")

    if ord('M') in new_keys or ord('m') in new_keys:
        rx, ry, _ = localization.get_pose()
        print(mapping.summary(robot_xy=(rx, ry)))
        _pc = mission.pillar_cells()
        print(f"[MAP] saved {mapping.save_png('map.png', blue_cell=_pc['blue'], yellow_cell=_pc['yellow'])}")

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
                _pc = mission.pillar_cells()
                saved = mapping.save_png('path.png', path_cells=route,
                                         blue_cell=_pc['blue'], yellow_cell=_pc['yellow'])
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
            explore_mode = False           # mutually exclusive with explore
            if last_route:
                following.set_path(last_route)
                print(f"Follow mode ON — driving {len(last_route)}-waypoint path (DWA)")
            else:
                follow_mode = False
                print("Follow mode: no planned path — press V to plan one first")
        else:
            following.reset()
            print("Follow mode OFF")

    # ── E: autonomous frontier exploration ────────────────────────────────────
    if ord('E') in new_keys or ord('e') in new_keys:
        explore_mode = not explore_mode
        if explore_mode:
            follow_mode = False
            auto_mode = False
            mission_mode = False
            exploration.reset()
            print("Explore mode ON — frontier exploration (DWA)")
        else:
            exploration.reset()
            print("Explore mode OFF")

    # ── X: full blue-then-yellow mission ──────────────────────────────────────
    if ord('X') in new_keys or ord('x') in new_keys:
        mission_mode = not mission_mode
        if mission_mode:
            explore_mode = False
            follow_mode = False
            auto_mode = False
            mission.reset()
            print("Mission mode ON — reach BLUE then YELLOW")
        else:
            mission.reset()
            print("Mission mode OFF")

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
        if explore_mode:
            explore_mode = False
            exploration.reset()
            print("Explore mode OFF (Space pressed)")
        if mission_mode:
            mission_mode = False
            mission.reset()
            print("Mission mode OFF (Space pressed)")

    # ── Mission / explore / follow / autonomous / teleop (mutually exclusive) ──
    elif mission_mode:
        colors = sensors.read_color_detections()
        v_cmd, omega_cmd, m_status, m_dbg = mission.mission_step(
            (robot_x, robot_y, robot_theta), devices.timestep / 1000.0, colors
        )
        v_cmd, omega_cmd, safe_label = safety.apply(v_cmd, omega_cmd, colors=colors)
        sel_label = f"mission_{m_status}" + (f"|{safe_label}" if safe_label else "")
        if step_count % 15 == 0:
            print(f"[MISSION] {mission.state_name()} status={m_status} "
                  f"active={m_dbg['active']} visible={m_dbg['visible']} "
                  f"dist={m_dbg['dist'] if m_dbg['dist'] != float('inf') else 'inf'} "
                  f"safety={safe_label or '-'}")
        if m_status == "done":
            mission_mode = False
            v_cmd = omega_cmd = 0.0
            print("Mission mode OFF (MISSION COMPLETE — blue then yellow reached)")

    elif explore_mode:
        v_cmd, omega_cmd, ex_status = exploration.explore_step(
            (robot_x, robot_y, robot_theta), devices.timestep / 1000.0
        )
        v_cmd, omega_cmd, safe_label = safety.apply(v_cmd, omega_cmd)
        sel_label = f"explore_{ex_status}" + (f"|{safe_label}" if safe_label else "")
        if step_count % 15 == 0:
            print(f"[EXPLORE] {ex_status} safety={safe_label or '-'} "
                  f"v={v_cmd:+.2f} w={omega_cmd:+.2f} goal={exploration.current_goal()} "
                  f"| {mapping.summary()}")
        if ex_status == "complete":
            explore_mode = False
            exploration.reset()
            v_cmd = omega_cmd = 0.0
            print("Explore mode OFF (exploration complete)")

    elif follow_mode:
        v_cmd, omega_cmd, follow_status = following.step(
            (robot_x, robot_y, robot_theta), devices.timestep / 1000.0
        )
        v_cmd, omega_cmd, safe_label = safety.apply(v_cmd, omega_cmd)
        sel_label = f"follow_{follow_status}" + (f"|{safe_label}" if safe_label else "")
        if follow_status in ("done", "stuck", "idle"):
            follow_mode = False
            following.reset()
            v_cmd = omega_cmd = 0.0
            print(f"Follow mode OFF ({follow_status})")

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
    prev_omega = omega_real
    # Pose is unreliable during a wall-block/reverse (wheels may slip) — pause
    # mapping next step so we don't stamp walls from a drifting pose.
    prev_blocked = ("block" in safe_label) or ("reverse" in safe_label)

    # ── Live map view (inline cv2 window, throttled) ──────────────────────────
    if viz_on and step_count % VIZ_PERIOD_STEPS == 0:
        # Show the active route (mission/explore/follow) if any, else the last plan.
        if mission_mode or explore_mode or follow_mode:
            viz_path = following.current_path()
        else:
            viz_path = last_route
        # Colored pillar markers (mission mode) + the current goal/frontier.
        viz_blue = viz_yellow = None
        if mission_mode:
            _pc = mission.pillar_cells()
            viz_blue, viz_yellow = _pc["blue"], _pc["yellow"]
            viz_goals = [mission.current_goal()] if mission.current_goal() is not None else []
        elif explore_mode and exploration.current_goal() is not None:
            viz_goals = [exploration.current_goal()]
        else:
            viz_goals = plan_targets
        visualizer.render(
            mapping.get_grid(),
            robot_cell=mapping.robot_map_pos((robot_x, robot_y, robot_theta)),
            goals=viz_goals,
            path=viz_path,
            blue_cell=viz_blue,
            yellow_cell=viz_yellow,
        )

    # ── Periodic pose log ─────────────────────────────────────────────────────
    if step_count % POSE_LOG_PERIOD_STEPS == 0:
        print(f"[POSE] {localization.format_pose()}")
