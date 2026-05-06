"""Husarion RosBot controller — main entry point.

Imports all subsystems and runs the Webots control loop.
Contains no sensor reads, kinematics, or navigation logic.

Keys: F/S/A/D drive | Space stop | T self-test | G autonomous | I snapshot | L log
"""

import devices                          # hardware init (must import first)
import sensors
import motion
from autonomous import autonomous_step, reset_autonomous_state
from sensor_debug import format_compact_sensors, format_sensor_snapshot
from config import TARGET_LIN_VEL, TARGET_ANG_VEL, FRONT_STOP_DIST

# ── Controller state ───────────────────────────────────────────────────────────
step_count       = 0
previous_pressed = set()
test_timer       = 0
sensor_log       = False
auto_mode        = False
block_timer      = 0

print(
    "Controller ready.  Keys: F/S/A/D drive | Space stop | T self-test | "
    "G autonomous mode | I sensor snapshot | L toggle sensor log"
)


OVERHEAD_CENTER_BLOCK_DIST = 0.55
OVERHEAD_SIDE_WARN_DIST = 0.40
OVERHEAD_SLOW_VEL = 0.04
OVERHEAD_STEER_OMEGA = 0.12


# ── Main loop ──────────────────────────────────────────────────────────────────
while devices.robot.step(devices.timestep) != -1:
    step_count += 1

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

    if ord('G') in new_keys or ord('g') in new_keys:
        auto_mode = not auto_mode
        if not auto_mode:
            block_timer = 0
            reset_autonomous_state()
        print(f"Autonomous mode {'ON' if auto_mode else 'OFF'}")

    # ── Hard stop: Space exits autonomous mode and zeroes twist ───────────────
    if ord(' ') in new_keys:
        v_cmd = omega_cmd = 0.0
        if auto_mode:
            auto_mode   = False
            block_timer = 0
            reset_autonomous_state()
            print("Autonomous mode OFF (Space pressed)")

    # ── Autonomous or teleop (mutually exclusive) ─────────────────────────────
    elif auto_mode:
        v_cmd, omega_cmd, sel_label, block_timer, dbg = autonomous_step(block_timer)

        if step_count % 10 == 0:
            print(
                f"[COLOR] green={dbg['green']}:{dbg['green_ratio']:.3f} "
                f"dist={dbg['green_distance']:.3f} "
                f"blue={dbg['blue']}:{dbg['blue_ratio']:.3f} "
                f"yellow={dbg['yellow']}:{dbg['yellow_ratio']:.3f} | "
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
