
WHEEL_RADIUS_M = 0.043
WHEEL_TRACK_M = 0.18


MAX_WHEEL_SPEED_RAD_S = 36.0


def wheel_speeds_from_twist(v, omega,
                            wheel_radius=WHEEL_RADIUS_M,
                            wheel_track=WHEEL_TRACK_M):
    half_track = wheel_track / 2.0
    left = (v - omega * half_track) / wheel_radius
    right = (v + omega * half_track) / wheel_radius
    return left, right


def clamp_twist(v, omega,
                max_wheel_speed=MAX_WHEEL_SPEED_RAD_S,
                wheel_radius=WHEEL_RADIUS_M,
                wheel_track=WHEEL_TRACK_M):
    left, right = wheel_speeds_from_twist(v, omega, wheel_radius, wheel_track)
    peak = max(abs(left), abs(right))
    if peak <= max_wheel_speed or peak == 0.0:
        return v, omega
    scale = max_wheel_speed / peak
    return v * scale, omega * scale
