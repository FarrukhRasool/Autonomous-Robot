"""Sensor-reading helpers for the Husarion RosBot.

All functions read from the hardware handles in devices.py.
Pure data functions — no control logic.
"""

import math

import devices
from config import (
    GREEN_DEPTH_MIN_VALID, GREEN_PIXEL_RATIO, TARGET_PIXEL_RATIO, FRONT_BLOCK_DIST
)


INF = float('inf')


def _is_valid_range(v, min_valid):
    return math.isfinite(v) and v > min_valid


def _percentile(values, percentile_index):
    if not values:
        return INF
    values.sort()
    idx = max(0, min(len(values) - 1, len(values) // percentile_index))
    return values[idx]


def _safe_call(sensor, method, default=None):
    if sensor is None:
        return default
    try:
        return getattr(sensor, method)()
    except Exception:
        return default


def _read_camera_image(camera):
    if camera is None:
        return None, 0, 0
    try:
        w = camera.getWidth()
        h = camera.getHeight()
        img = camera.getImage()
    except Exception:
        return None, 0, 0
    if not img or w == 0 or h == 0:
        return None, 0, 0
    return img, w, h


def _read_depth_image():
    if devices.camera_depth is None:
        return None, 0, 0
    try:
        w = devices.camera_depth.getWidth()
        h = devices.camera_depth.getHeight()
        data = devices.camera_depth.getRangeImage()
    except Exception:
        return None, 0, 0
    if not data or w == 0 or h == 0:
        return None, 0, 0
    return data, w, h


def _rgb_at(img, w, row, col):
    # Webots RGB camera image is BGRA byte order.
    idx = 4 * (row * w + col)
    b, g, r = img[idx], img[idx + 1], img[idx + 2]
    return r, g, b


def _depth_at_rgb_pixel(depth_data, depth_w, depth_h, rgb_w, rgb_h, row, col):
    if not depth_data or depth_w == 0 or depth_h == 0:
        return None

    # RGB/depth camera resolutions may differ; scale image coordinates.
    depth_col = min(depth_w - 1, max(0, int(col * depth_w / rgb_w)))
    depth_row = min(depth_h - 1, max(0, int(row * depth_h / rgb_h)))
    v = depth_data[depth_row * depth_w + depth_col]
    return v if _is_valid_range(v, GREEN_DEPTH_MIN_VALID) else None


def _green_pixel(r, g, b):
    return g > 80 and g > 1.35 * r and g > 1.25 * b


def _blue_pixel(r, g, b):
    return b > 90 and b > 1.35 * r and b > 1.25 * g


def _yellow_pixel(r, g, b):
    return r > 110 and g > 95 and b < 0.75 * min(r, g)


def _green_ground_roi(w, h):
    # Bottom-center ROI: the next ground patch likely to pass under the robot.
    return int(0.20 * w), int(0.80 * w), int(0.62 * h), int(0.95 * h)


def _target_roi(w, h):
    # Central ROI: forward objects/pillars without ground-dominated bottom rows.
    return int(0.20 * w), int(0.80 * w), int(0.12 * h), int(0.72 * h)


def _sample_color_ratio(img, w, h, roi, predicate,
                        depth_data=None, depth_w=0, depth_h=0):
    col0, col1, row0, row1 = roi
    total = 0
    hits = 0
    depths = []

    for row in range(row0, row1, 3):
        for col in range(col0, col1, 3):
            r, g, b = _rgb_at(img, w, row, col)
            total += 1
            if not predicate(r, g, b):
                continue

            hits += 1
            depth = _depth_at_rgb_pixel(depth_data, depth_w, depth_h, w, h, row, col)
            if depth is not None:
                depths.append(depth)

    ratio = hits / total if total else 0.0
    return ratio, depths


def _read_scalar(sensor):
    if sensor is None:
        return None
    try:
        return sensor.getValue()
    except Exception:
        return None


def _read_vector(sensor, method):
    value = _safe_call(sensor, method)
    return tuple(value) if value is not None else None


def _read_laser_summary():
    if devices.laser is None:
        return {"laser_count": None}
    try:
        data = devices.laser.getRangeImage()
    except Exception:
        return {"laser_count": None}
    if not data:
        return {"laser_count": 0}

    finite = [v for v in data if _is_valid_range(v, 0.0)]
    return {
        "laser_count": len(data),
        "laser_min": min(finite) if finite else INF,
        "laser_max": max(finite) if finite else 0.0,
        "laser_mean": sum(finite) / len(finite) if finite else float("nan"),
    }


def _read_rgb_center():
    img, w, h = _read_camera_image(devices.camera_rgb)
    if img is None:
        return None
    return w, h, _rgb_at(img, w, h // 2, w // 2)


def _read_depth_center():
    data, w, h = _read_depth_image()
    if data is None:
        return None
    return w, h, data[h // 2 * w + w // 2]

def overhead_depth_regions(max_distance):
    """Detect overhead/floating obstacles in upper-left, upper-center, upper-right depth ROIs.

    Returns:
        detected: bool
        region_name: "left", "center", "right", or "none"
        depth_value: robust nearest depth in detected region, or inf
    """
    data, w, h = _read_depth_image()
    if data is None:
        return False, "none", INF

    rois = {
        "left":   (0.05, 0.40, 0.02, 0.55),
        "center": (0.25, 0.75, 0.02, 0.55),
        "right":  (0.60, 0.95, 0.02, 0.55),
    }

    best_region = "none"
    best_depth = INF

    for name, (x0f, x1f, y0f, y1f) in rois.items():
        col_start = int(x0f * w)
        col_end   = int(x1f * w)
        row_start = int(y0f * h)
        row_end   = int(y1f * h)

        vals = []

        for row in range(row_start, row_end, 3):
            base = row * w
            for col in range(col_start, col_end, 3):
                v = data[base + col]
                if math.isfinite(v) and v > GREEN_DEPTH_MIN_VALID:
                    vals.append(v)

        if not vals:
            continue

        vals.sort()
        depth_near = vals[max(0, len(vals) // 20)]  # 5th percentile

        if depth_near < best_depth:
            best_depth = depth_near
            best_region = name

    detected = best_depth < max_distance
    return detected, best_region, best_depth

def overhead_depth_region_values():
    """Return robust depth values for overhead left/center/right ROIs.

    These are the same ROIs used by overhead_depth_regions().
    Returns inf for a region if no valid depth pixels exist.
    """
    data, w, h = _read_depth_image()
    if data is None:
        return {
            "overhead_left": INF,
            "overhead_center": INF,
            "overhead_right": INF,
        }

    rois = {
        "overhead_left":   (0.05, 0.40, 0.02, 0.55),
        "overhead_center": (0.25, 0.75, 0.02, 0.55),
        "overhead_right":  (0.60, 0.95, 0.02, 0.55),
    }

    result = {}

    for name, (x0f, x1f, y0f, y1f) in rois.items():
        col_start = int(x0f * w)
        col_end   = int(x1f * w)
        row_start = int(y0f * h)
        row_end   = int(y1f * h)

        vals = []

        for row in range(row_start, row_end, 3):
            base = row * w
            for col in range(col_start, col_end, 3):
                v = data[base + col]
                if math.isfinite(v) and v > GREEN_DEPTH_MIN_VALID:
                    vals.append(v)

        if not vals:
            result[name] = INF
            continue

        vals.sort()
        result[name] = vals[max(0, len(vals) // 20)]  # 5th percentile

    return result


def get_front_laser_min():
    """Minimum range in the laser's front-centre band (40% to 60% of rays)."""
    if devices.laser is None:
        return INF
    try:
        ranges = devices.laser.getRangeImage()
    except Exception:
        return INF
    if not ranges:
        return INF

    n = len(ranges)
    front = [
        r for r in ranges[int(0.4 * n):int(0.6 * n)]
        if _is_valid_range(r, 0.0)
    ]
    return min(front) if front else INF


def read_color_detections():
    """Detect semantic colors from the RGB camera.

    Returns ratios for green ground in the lower image and blue/yellow pillars in
    the central image. Green distance is estimated by reading depth pixels that
    correspond to green RGB pixels.
    """
    result = {
        "green": False, "green_ratio": 0.0, "green_distance": INF,
        "blue": False, "blue_ratio": 0.0,
        "yellow": False, "yellow_ratio": 0.0,
    }

    img, w, h = _read_camera_image(devices.camera_rgb)
    if img is None:
        return result

    depth_data, depth_w, depth_h = _read_depth_image()
    green_ratio, green_depths = _sample_color_ratio(
        img, w, h, _green_ground_roi(w, h), _green_pixel,
        depth_data, depth_w, depth_h,
    )
    blue_ratio, _ = _sample_color_ratio(img, w, h, _target_roi(w, h), _blue_pixel)
    yellow_ratio, _ = _sample_color_ratio(img, w, h, _target_roi(w, h), _yellow_pixel)

    result["green_ratio"] = green_ratio
    result["blue_ratio"] = blue_ratio
    result["yellow_ratio"] = yellow_ratio

    if green_depths:
        result["green_distance"] = _percentile(green_depths, 10)

    result["green"] = green_ratio >= GREEN_PIXEL_RATIO
    result["blue"] = blue_ratio >= TARGET_PIXEL_RATIO
    result["yellow"] = yellow_ratio >= TARGET_PIXEL_RATIO
    return result


def read_sensor_snapshot():
    """Read every enabled sensor and return a readings dict for sensor_debug."""
    r = {}

    r["enc_fl"] = _read_scalar(devices.fl_wheel_sensor)
    r["enc_fr"] = _read_scalar(devices.fr_wheel_sensor)
    r["enc_rl"] = _read_scalar(devices.rl_wheel_sensor)
    r["enc_rr"] = _read_scalar(devices.rr_wheel_sensor)

    r["accel"] = _read_vector(devices.accelerometer, "getValues")
    r["gyro"] = _read_vector(devices.gyro, "getValues")
    r["compass"] = _read_vector(devices.compass, "getValues")
    r["imu_rpy"] = _read_vector(devices.inertial_unit, "getRollPitchYaw")

    r["range_fl"] = _read_scalar(devices.fl_range)
    r["range_fr"] = _read_scalar(devices.fr_range)
    r["range_rl"] = _read_scalar(devices.rl_range)
    r["range_rr"] = _read_scalar(devices.rr_range)

    r.update(_read_laser_summary())
    r["cam_rgb"] = _read_rgb_center()
    r["colors"] = read_color_detections()
    r["cam_depth"] = _read_depth_center()
    r.update(overhead_depth_region_values())
    return r
