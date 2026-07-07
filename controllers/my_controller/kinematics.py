"""Differential-drive forward kinematics for the Husarion RosBot.

The 4-wheel RosBot is treated as a differential drive: the two left
wheels share one angular velocity, the two right wheels share another.
This is an approximation -- skid-steer slip means the achieved (v, omega)
will not exactly match the commanded (v, omega), especially during
in-place rotation. That residual is handled later in the localization
layer; this module only solves the geometric mapping.

Frame convention: body x forward, y left, z up; positive omega yaws
left (about +z). Units are SI throughout: v in m/s, omega in rad/s,
wheel angular velocities in rad/s.
"""

# Husarion RosBot mechanical constants — matched to the reference project's
# CONSTANTS.py (WHEEL_RADIUS=0.043, AXLE_LENGTH=0.18) for parity: the wheel radius
# sets the encoder->distance scale (SLAM translation input) and the track sets the
# twist->wheel-speed differential; using the reference's values keeps odometry and
# actuation consistent with its tuning.
WHEEL_RADIUS_M = 0.043
WHEEL_TRACK_M = 0.18

# Per-wheel angular velocity cap sent to the motors (reference MAX_VELOCITY=36;
# also normalises the DWA speed reward via MAX_SPEED = this * WHEEL_RADIUS_M).
MAX_WHEEL_SPEED_RAD_S = 36.0


def wheel_speeds_from_twist(v, omega,
                            wheel_radius=WHEEL_RADIUS_M,
                            wheel_track=WHEEL_TRACK_M):
    """Map a body twist (v, omega) -> (left_rad_s, right_rad_s).

    Differential-drive inverse kinematics:
        omega_left  = (v - omega * L/2) / r
        omega_right = (v + omega * L/2) / r
    Each returned value applies to BOTH wheels on that side.
    """
    half_track = wheel_track / 2.0
    left = (v - omega * half_track) / wheel_radius
    right = (v + omega * half_track) / wheel_radius
    return left, right


def clamp_twist(v, omega,
                max_wheel_speed=MAX_WHEEL_SPEED_RAD_S,
                wheel_radius=WHEEL_RADIUS_M,
                wheel_track=WHEEL_TRACK_M):
    """Scale (v, omega) down uniformly if either wheel would saturate.

    Preserves the v / omega ratio so the trajectory shape is unchanged --
    the robot simply executes the same arc more slowly.
    """
    left, right = wheel_speeds_from_twist(v, omega, wheel_radius, wheel_track)
    peak = max(abs(left), abs(right))
    if peak <= max_wheel_speed or peak == 0.0:
        return v, omega
    scale = max_wheel_speed / peak
    return v * scale, omega * scale
