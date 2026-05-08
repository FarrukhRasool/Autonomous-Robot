"""IMU-fused wheel-encoder odometry for the Husarion RosBot.

Pure Python — no Webots imports.  Maintains a pose (x, y, theta) in a
fixed world frame.  (x, y) are integrated from differential-drive forward
kinematics applied to wheel-encoder deltas.  Theta comes directly from
the IMU yaw (minus a baseline captured at reset), which is drift-free in
the simulator and immune to skid-steer slip.

Frame convention (matches kinematics.py):
    Body x forward, y left, z up.
    Positive theta yaws left (about +z).
    Pose at startup / reset is (0, 0, 0) regardless of the robot's
    actual world-frame heading at that moment.

Per-step:
    d_left  = d_left_rad  * r          arc length covered by the left side
    d_right = d_right_rad * r          arc length covered by the right side
    d_s     = (d_right + d_left) / 2   forward distance from the wheels
    theta_new = wrap(imu_yaw - baseline)        when IMU available
    theta_new = wrap(theta + d_theta_wheels)    fallback (wheel-only)
    theta_mid = wrap(theta + 0.5 * wrap(theta_new - theta))
    x += d_s * cos(theta_mid)
    y += d_s * sin(theta_mid)
    theta = theta_new
"""

import math

from kinematics import WHEEL_RADIUS_M, WHEEL_TRACK_M


_x = 0.0
_y = 0.0
_theta = 0.0
_prev_left = None
_prev_right = None
_imu_yaw_baseline = None


def _wrap_angle(a):
    """Wrap an angle to (-pi, pi]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def reset_pose():
    """Zero the pose and forget all sensor baselines.

    The next call to update_from_encoders() will re-seed the encoder
    baseline from whatever the wheels read, and the IMU yaw baseline
    from whatever the inertial_unit reads, so the pose remains (0, 0, 0)
    regardless of the robot's actual world-frame heading at reset time.
    """
    global _x, _y, _theta, _prev_left, _prev_right, _imu_yaw_baseline
    _x = 0.0
    _y = 0.0
    _theta = 0.0
    _prev_left = None
    _prev_right = None
    _imu_yaw_baseline = None


def get_pose():
    """Return the current pose (x_m, y_m, theta_rad)."""
    return _x, _y, _theta


def update_from_encoders(left_rad, right_rad, imu_yaw=None):
    """Integrate one step of IMU-fused wheel odometry.

    Parameters
    ----------
    left_rad, right_rad : float
        Cumulative wheel angles for the left/right side, in radians.
        Pass averaged front+rear values — both wheels on a side share a motor.
    imu_yaw : float, optional
        World-frame yaw from the inertial_unit (radians).  When provided
        and finite, replaces wheel-derived theta.  When None, the function
        falls back to wheel-only integration.

    On the first call (or first call after reset), the baseline values are
    captured and no pose integration happens — pose stays (0, 0, 0).

    Returns
    -------
    (x, y, theta) after the update.
    """
    global _x, _y, _theta, _prev_left, _prev_right, _imu_yaw_baseline

    if left_rad is None or right_rad is None:
        return _x, _y, _theta

    # First-step seeding: capture both encoder and IMU baselines, no integration.
    if _prev_left is None or _prev_right is None:
        _prev_left = left_rad
        _prev_right = right_rad
        if imu_yaw is not None and _imu_yaw_baseline is None:
            _imu_yaw_baseline = imu_yaw
        return _x, _y, _theta

    # Late IMU baseline capture: covers the case where IMU returned None on
    # the first call but is now available.
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
        # Wheel-only fallback: derive d_theta from the encoder difference.
        d_theta_wheels = (d_right - d_left) / WHEEL_TRACK_M
        theta_new = _wrap_angle(_theta + d_theta_wheels)

    # Midpoint heading using the signed shortest delta — robust across the
    # +pi/-pi wrap boundary.
    delta_theta = _wrap_angle(theta_new - _theta)
    theta_mid = _wrap_angle(_theta + 0.5 * delta_theta)

    _x += d_s * math.cos(theta_mid)
    _y += d_s * math.sin(theta_mid)
    _theta = theta_new

    return _x, _y, _theta


def format_pose():
    """Format the current pose for console logging."""
    x, y, theta = get_pose()
    return (
        f"x={x:+.3f} m  y={y:+.3f} m  "
        f"theta={theta:+.3f} rad ({math.degrees(theta):+.1f} deg)"
    )
