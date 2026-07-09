import devices
from kinematics import MAX_WHEEL_SPEED_RAD_S, clamp_twist, wheel_speeds_from_twist


def set_wheel_speeds(left, right):
    left  = max(-MAX_WHEEL_SPEED_RAD_S, min(MAX_WHEEL_SPEED_RAD_S, left))
    right = max(-MAX_WHEEL_SPEED_RAD_S, min(MAX_WHEEL_SPEED_RAD_S, right))
    devices.fl_motor.setVelocity(left)
    devices.rl_motor.setVelocity(left)
    devices.fr_motor.setVelocity(right)
    devices.rr_motor.setVelocity(right)


def drive_twist(v, omega):
    v_c, w_c = clamp_twist(v, omega)
    left, right = wheel_speeds_from_twist(v_c, w_c)
    set_wheel_speeds(left, right)
    return v_c, w_c, left, right


def stop_robot():
    set_wheel_speeds(0.0, 0.0)


def is_turning():
    return abs(devices.fl_motor.getVelocity() - devices.fr_motor.getVelocity()) > 0.02
