import math
from config import TARGET_REV_VEL


def laser_5_sectors(ranges):
    if not ranges:
        return (float('inf'),) * 5

    n = len(ranges)

    def _min(start, end):
        vals = [r for r in ranges[start:end] if math.isfinite(r) and r > 0.0]
        return min(vals) if vals else float('inf')

    s = max(1, n // 5)
    return (
        _min(0,     s  ),   
        _min(s,   2*s  ),   
        _min(2*s, 3*s  ),  
        _min(3*s, 4*s  ),  
        _min(4*s,   n  ),   
    )


def wall_follow_twist(front_min, right_min, left_min,
                      block_timer, rear_safe,
                      front_block, front_caution,
                      wall_target, wall_band, wall_lost, side_danger,
                      block_timeout,
                      v_fwd, v_caution,
                      omega_small, omega_caution, omega_large,
                      open_side_hint=None):
    

    def _announce(action, v, omega):
        return v, omega

    if open_side_hint == "left":
        open_side_omega = omega_large
        open_side_omega_caution = omega_caution
        open_side_label = "left"
    elif open_side_hint == "right":
        open_side_omega = -omega_large
        open_side_omega_caution = -omega_caution
        open_side_label = "right"
    elif left_min > right_min:
        open_side_omega = omega_large
        open_side_omega_caution = omega_caution
        open_side_label = "left"
    else:
        open_side_omega = -omega_large
        open_side_omega_caution = -omega_caution
        open_side_label = "right"
    print(
        f"[REACTIVE] open side: {open_side_label} | "
        f"left={left_min:.3f} right={right_min:.3f}"
    )

    if front_min <= front_block:
        if block_timer >= block_timeout:
            if rear_safe:
                v, omega = _announce("front blocked too long: backing up", -TARGET_REV_VEL, 0.0)
                return v, omega, "recovery_backup"
            else:
                v, omega = _announce(f"front blocked too long: rotate {open_side_label}", 0.0, open_side_omega)
                return v, omega, f"recovery_rotate_{open_side_label}"

        v, omega = _announce(f"front blocked: turn {open_side_label}", 0.0, open_side_omega)
        return v, omega, f"front_block_turn_{open_side_label}"

    if front_min <= front_caution:
        v, omega = _announce(f"front caution: slow slight {open_side_label}", v_caution, open_side_omega_caution)
        return v, omega, f"front_caution_turn_{open_side_label}"

    if left_min < side_danger:
        v, omega = _announce("left side too close: slow slight right", v_fwd * 0.5, -omega_small)
        return v, omega, "avoid_left"

    if right_min < side_danger:
        v, omega = _announce("right side too close: slow slight left", v_fwd * 0.5, omega_small)
        return v, omega, "avoid_right"

    if right_min > wall_lost:
        v, omega = _announce("right wall lost: slight right", v_fwd, -omega_small)
        return v, omega, "seek_wall"

    if right_min > wall_target + wall_band:
        v, omega = _announce("right wall too far: slight right", v_fwd, -omega_small)
        return v, omega, "correct_right"

    if right_min < wall_target - wall_band:
        v, omega = _announce("right wall too close: slight left", v_fwd, omega_small)
        return v, omega, "correct_left"

    v, omega = _announce("corridor good: forward", v_fwd, 0.0)
    return v, omega, "forward"
