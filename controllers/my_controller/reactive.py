"""Reactive navigation helpers for the Husarion RosBot.

Pure Python — no Webots imports.  Receives already-read sensor data and
returns a body twist (v, omega).

Laser indexing convention (verified working in M3 / M3.0b):
  Ray index 0 = leftmost ray; index n-1 = rightmost ray (robot's view).
  Five equal sectors of n//5 rays:

    far_left   [0,      n//5)     broad left arc
    left       [n//5,   2*n//5)   front-left quadrant
    center     [2*n//5, 3*n//5)   straight ahead      <- matches get_front_laser_min()
    right      [3*n//5, 4*n//5)   front-right quadrant
    far_right  [4*n//5, n)        broad right arc

  The center sector [2n/5, 3n/5] equals [0.4n, 0.6n], which is the same
  window used by get_front_laser_min() — the empirically validated front
  direction.  Only the outer two fifths on each side are new.
"""

import math
from config import TARGET_REV_VEL


def laser_5_sectors(ranges):
    """Split a laser range image into five directional sectors.

    Returns (far_left_min, left_min, center_min, right_min, far_right_min).
    Any sector with no finite positive ray returns inf.
    Gracefully handles None or empty ranges.
    """
    if not ranges:
        return (float('inf'),) * 5

    n = len(ranges)

    def _min(start, end):
        vals = [r for r in ranges[start:end] if math.isfinite(r) and r > 0.0]
        return min(vals) if vals else float('inf')

    s = max(1, n // 5)
    return (
        _min(0,     s  ),   # far_left
        _min(s,   2*s  ),   # left
        _min(2*s, 3*s  ),   # center  (= [0.4n, 0.6n], same as get_front_laser_min)
        _min(3*s, 4*s  ),   # right
        _min(4*s,   n  ),   # far_right (may be slightly wider when n % 5 != 0)
    )


def wall_follow_twist(front_min, right_min, left_min,
                      block_timer, rear_safe,
                      front_block, front_caution,
                      wall_target, wall_band, wall_lost, side_danger,
                      block_timeout,
                      v_fwd, v_caution,
                      omega_small, omega_caution, omega_large,
                      open_side_hint=None):
    """Right-hand wall-following with front caution and open-side turning."""

    def _announce(action, v, omega):
        print(
            f"[REACTIVE] {action} | "
            f"front={front_min:.3f} left={left_min:.3f} right={right_min:.3f} | "
            f"v={v:+.2f} omega={omega:+.2f}"
        )
        return v, omega

    # Decide which side is more open
    # Positive omega = turn left
    # Negative omega = turn right
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

    # 1. Front truly blocked
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

    # 2. Front caution zone: narrow but maybe passable
    if front_min <= front_caution:
        v, omega = _announce(f"front caution: slow slight {open_side_label}", v_caution, open_side_omega_caution)
        return v, omega, f"front_caution_turn_{open_side_label}"

    # 3. Left side danger
    if left_min < side_danger:
        v, omega = _announce("left side too close: slow slight right", v_fwd * 0.5, -omega_small)
        return v, omega, "avoid_left"

    # 4. Right side danger
    if right_min < side_danger:
        v, omega = _announce("right side too close: slow slight left", v_fwd * 0.5, omega_small)
        return v, omega, "avoid_right"

    # 5. Right wall lost
    if right_min > wall_lost:
        v, omega = _announce("right wall lost: slight right", v_fwd, -omega_small)
        return v, omega, "seek_wall"

    # 6. Right wall too far
    if right_min > wall_target + wall_band:
        v, omega = _announce("right wall too far: slight right", v_fwd, -omega_small)
        return v, omega, "correct_right"

    # 7. Right wall too close
    if right_min < wall_target - wall_band:
        v, omega = _announce("right wall too close: slight left", v_fwd, omega_small)
        return v, omega, "correct_left"

    # 8. Good corridor
    v, omega = _announce("corridor good: forward", v_fwd, 0.0)
    return v, omega, "forward"
