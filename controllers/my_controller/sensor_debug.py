"""Sensor snapshot formatting and plausibility checks for the Husarion RosBot.

Pure Python — no Webots imports.  Receives already-read values from the
controller and returns formatted strings for the Webots console.

Plausibility rules are intentionally generous: we flag [WARN] only for
values that are outright broken (NaN, inf, out of physical range), not
for values that are merely unexpected.  Calibration belongs in later
milestones.
"""

import math


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ok(flag):
    return "[OK]  " if flag else "[WARN]"


def _fin(v):
    """True if v is a finite scalar."""
    return v is not None and isinstance(v, (int, float)) and math.isfinite(v)


def _vecfin(v):
    """True if v is a 3-tuple/list of finite floats."""
    return v is not None and len(v) == 3 and all(math.isfinite(x) for x in v)


def _mag(v):
    return math.sqrt(sum(x * x for x in v))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_plausible(name, value):
    """Return True if value is within expected range for sensor key name.

    name  — readings-dict key, e.g. 'enc_fl', 'accel', 'range_fl'.
    value — already-read Python object (float, tuple, etc.).
    Returns False for None or non-finite values.
    """
    if value is None:
        return False
    if name.startswith("enc_"):
        return _fin(value)
    if name == "accel":
        # Gravity magnitude at rest: ~9.81 m/s².  Allow generous band.
        return _vecfin(value) and 5.0 < _mag(value) < 15.0
    if name == "gyro":
        return _vecfin(value)
    if name == "compass":
        # Webots compass returns a unit-normalised magnetic field vector.
        return _vecfin(value) and 0.8 < _mag(value) < 1.2
    if name == "imu_rpy":
        return _vecfin(value)
    if name.startswith("range_"):
        return _fin(value) and 0.0 < value < 2.0
    if name == "laser_count":
        return isinstance(value, int) and value > 0
    if name in ("laser_min", "laser_max", "laser_mean"):
        return _fin(value) and value > 0
    if name == "cam_rgb":
        if not value:
            return False
        w, h, px = value
        return w > 0 and h > 0 and px is not None and all(0 <= c <= 255 for c in px)
    if name == "colors":
        if not isinstance(value, dict):
            return False
        return all(
            key in value for key in
            (
                "green", "green_ratio", "green_distance",
                "blue", "blue_ratio", "yellow", "yellow_ratio",
            )
        )
    
    if name in ("overhead_left", "overhead_center", "overhead_right"):
        return value is not None and isinstance(value, (int, float))

    if name == "cam_depth":
        if not value:
            return False
        w, h, ctr = value
        return w > 0 and h > 0 and ctr is not None and _fin(ctr) and ctr > 0
    return False


def format_sensor_snapshot(readings):
    """Return a multi-line sensor-snapshot string with OK / WARN / MISS tags.

    Expected keys in readings (all optional — missing keys get [MISS]):
      enc_fl, enc_fr, enc_rl, enc_rr         float (rad, cumulative wheel angle)
      accel, gyro, compass, imu_rpy           tuple[3] of float
      range_fl, range_fr, range_rl, range_rr  float (m)
      laser_count                             int   (number of rays)
      laser_min, laser_max, laser_mean        float (m)
      cam_rgb    (width, height, (R, G, B))   or None
      cam_depth  (width, height, center_m)    or None
    """
    lines = ["=" * 46, "         SENSOR SNAPSHOT", "=" * 46]

    # -- Encoders --
    lines.append("ENCODERS  (cumulative wheel angle, rad)")
    for key in ("enc_fl", "enc_fr", "enc_rl", "enc_rr"):
        v = readings.get(key)
        if v is None:
            lines.append(f"  {key:10s}: [MISS]")
        else:
            lines.append(
                f"  {key:10s}: {v:+11.4f} rad  {_ok(is_plausible(key, v))}"
            )

    # -- IMU --
    lines.append("IMU")
    v = readings.get("accel")
    if v is None:
        lines.append("  accel     : [MISS]")
    else:
        mag = _mag(v) if _vecfin(v) else float("nan")
        lines.append(
            f"  accel     :  ax={v[0]:+.3f}  ay={v[1]:+.3f}  az={v[2]:+.3f}"
            f"  |g|={mag:.2f}  {_ok(is_plausible('accel', v))}"
        )

    v = readings.get("gyro")
    if v is None:
        lines.append("  gyro      : [MISS]")
    else:
        lines.append(
            f"  gyro      :  gx={v[0]:+.4f}  gy={v[1]:+.4f}  gz={v[2]:+.4f}"
            f"  {_ok(is_plausible('gyro', v))}"
        )

    v = readings.get("compass")
    if v is None:
        lines.append("  compass   : [MISS]")
    else:
        mag = _mag(v) if _vecfin(v) else float("nan")
        lines.append(
            f"  compass   :  cx={v[0]:+.3f}  cy={v[1]:+.3f}  cz={v[2]:+.3f}"
            f"  |B|={mag:.3f}  {_ok(is_plausible('compass', v))}"
        )

    v = readings.get("imu_rpy")
    if v is None:
        lines.append("  imu_rpy   : [MISS]")
    else:
        lines.append(
            f"  imu_rpy   :  roll={v[0]:+.4f}  pitch={v[1]:+.4f}"
            f"  yaw={v[2]:+.4f} rad  {_ok(is_plausible('imu_rpy', v))}"
        )

    # -- Range sensors --
    lines.append("RANGE SENSORS  (m)")
    for key in ("range_fl", "range_fr", "range_rl", "range_rr"):
        v = readings.get(key)
        if v is None:
            lines.append(f"  {key:12s}: [MISS]")
        else:
            val_str = f"{v:.4f} m" if _fin(v) else str(v)
            lines.append(
                f"  {key:12s}: {val_str:14s}  {_ok(is_plausible(key, v))}"
            )

    # -- Laser --
    lines.append("LASER")
    count = readings.get("laser_count")
    if count is None:
        lines.append("  laser: [MISS]")
    else:
        lmin  = readings.get("laser_min",  float("inf"))
        lmax  = readings.get("laser_max",  0.0)
        lmean = readings.get("laser_mean", float("nan"))
        ok = is_plausible("laser_count", count) and is_plausible("laser_min", lmin)
        lines.append(
            f"  rays={count}  min={lmin:.3f}m  max={lmax:.3f}m"
            f"  mean={lmean:.3f}m  {_ok(ok)}"
        )

    # -- Cameras --
    lines.append("CAMERAS")
    v = readings.get("cam_rgb")
    if v is None:
        lines.append("  rgb       : [MISS]")
    else:
        w, h, px = v
        px_str = f"R={px[0]},G={px[1]},B={px[2]}" if px is not None else "?"
        lines.append(
            f"  rgb       : {w}x{h}  center=({px_str})"
            f"  {_ok(is_plausible('cam_rgb', v))}"
        )

    colors = readings.get("colors")
    if colors is None:
        lines.append("  colors    : [MISS]")
    else:
        green_dist = colors["green_distance"]
        green_dist_str = f"{green_dist:.3f}m" if _fin(green_dist) else "?"
        lines.append(
            f"  colors    : green={colors['green']}:{colors['green_ratio']:.3f}"
            f" dist={green_dist_str}"
            f"  blue={colors['blue']}:{colors['blue_ratio']:.3f}"
            f"  yellow={colors['yellow']}:{colors['yellow_ratio']:.3f}"
            f"  {_ok(is_plausible('colors', colors))}"
        )

    v = readings.get("cam_depth")
    if v is None:
        lines.append("  depth     : [MISS]")
    else:
        w, h, ctr = v
        ctr_str = f"{ctr:.3f}m" if (ctr is not None and _fin(ctr)) else str(ctr)
        lines.append(
            f"  depth     : {w}x{h}  center={ctr_str}"
            f"  {_ok(is_plausible('cam_depth', v))}"
        )

        overhead_left = readings.get("overhead_left")
        overhead_center = readings.get("overhead_center")
        overhead_right = readings.get("overhead_right")

        def fmt_overhead(v):
            return f"{v:.3f}m" if _fin(v) else str(v)

        lines.append(
            f"  overhead  : left={fmt_overhead(overhead_left)}"
            f"  center={fmt_overhead(overhead_center)}"
            f"  right={fmt_overhead(overhead_right)}"
        )

    lines.append("=" * 46)
    return "\n".join(lines)


def format_compact_sensors(readings):
    """Return a single-line compact sensor summary for per-step logging.

    Covers one value from each sensor group so a quick eye-scan confirms
    all groups are live.
    """
    enc = readings.get("enc_fl")
    enc_str = f"{enc:+.2f}" if _fin(enc) else "?"

    rpy = readings.get("imu_rpy")
    yaw_str = f"{math.degrees(rpy[2]):+.1f}°" if (_vecfin(rpy) if rpy else False) else "?"

    lmin = readings.get("laser_min")
    lmin_str = f"{lmin:.3f}m" if _fin(lmin) else "?"

    rfl = readings.get("range_fl")
    rfr = readings.get("range_fr")
    rfl_str = f"{rfl:.3f}m" if _fin(rfl) else "?"
    rfr_str = f"{rfr:.3f}m" if _fin(rfr) else "?"

    colors = readings.get("colors") or {}
    green_dist = colors.get("green_distance")
    green_dist_str = f"{green_dist:.2f}m" if _fin(green_dist) else "?"
    color_str = (
        f"  green={colors.get('green', '?')}:{green_dist_str}"
        f"  blue={colors.get('blue', '?')}"
        f"  yellow={colors.get('yellow', '?')}"
    )

    return (
        f"[SENS] enc_fl={enc_str}rad"
        f"  yaw={yaw_str}"
        f"  laser_min={lmin_str}"
        f"  rng_fl={rfl_str}  rng_fr={rfr_str}"
        f"{color_str}"
    )
