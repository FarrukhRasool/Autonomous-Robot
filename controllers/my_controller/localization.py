
import math

import slam
from kinematics import WHEEL_RADIUS_M, WHEEL_TRACK_M
from config import SLIP_ENC_ROT_RAD, SLIP_GYRO_RATE_RADS

_x = 0.0
_y = 0.0
_theta = 0.0
_prev_left = None
_prev_right = None
_imu_yaw_baseline = None


def _wrap_angle(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def reset_pose():
    global _x, _y, _theta, _prev_left, _prev_right, _imu_yaw_baseline
    _x = 0.0
    _y = 0.0
    _theta = 0.0
    _prev_left = None
    _prev_right = None
    _imu_yaw_baseline = None
    slam.reset()         


def get_pose():
    p = slam.estimated_pose()
    return float(p[0]), float(p[1]), float(p[2])


def get_position():
    p = slam.estimated_pose()
    return float(p[0]), float(p[1])


def get_heading(unit='rad'):
    theta = float(slam.estimated_pose()[2])
    return math.degrees(theta) if unit == 'deg' else theta


def update_from_encoders(left_rad, right_rad, imu_yaw=None, gyro_z=None):
    global _x, _y, _theta, _prev_left, _prev_right, _imu_yaw_baseline

    if left_rad is None or right_rad is None:
        return get_pose()

    
    if _prev_left is None or _prev_right is None:
        _prev_left = left_rad
        _prev_right = right_rad
        if imu_yaw is not None and _imu_yaw_baseline is None:
            _imu_yaw_baseline = imu_yaw
        return get_pose()

    
    if imu_yaw is not None and _imu_yaw_baseline is None:
        _imu_yaw_baseline = imu_yaw

    d_left_rad = left_rad - _prev_left
    d_right_rad = right_rad - _prev_right
    _prev_left = left_rad
    _prev_right = right_rad

    d_left = d_left_rad * WHEEL_RADIUS_M
    d_right = d_right_rad * WHEEL_RADIUS_M

    d_s = 0.5 * (d_right + d_left)

    if imu_yaw is not None and _imu_yaw_baseline is not None:
        theta_new = _wrap_angle(imu_yaw - _imu_yaw_baseline)
    else:
        
        d_theta_wheels = (d_right - d_left) / WHEEL_TRACK_M
        theta_new = _wrap_angle(_theta + d_theta_wheels)

   
    delta_theta = _wrap_angle(theta_new - _theta)

    encoder_dtheta = (d_right - d_left) / WHEEL_TRACK_M
    if (gyro_z is not None
            and abs(encoder_dtheta) > SLIP_ENC_ROT_RAD
            and abs(gyro_z) < SLIP_GYRO_RATE_RADS):
        _theta = theta_new
        slam.predict(0.0, delta_theta)
        return get_pose()

    theta_mid = _wrap_angle(_theta + 0.5 * delta_theta)
    _x += d_s * math.cos(theta_mid)
    _y += d_s * math.sin(theta_mid)
    _theta = theta_new

    
    slam.predict(d_s, delta_theta)

    return get_pose()


def format_pose():
    x, y, theta = get_pose()
    return (
        f"x={x:+.3f} m  y={y:+.3f} m  "
        f"theta={theta:+.3f} rad ({math.degrees(theta):+.1f} deg)"
    )
