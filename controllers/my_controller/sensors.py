"""Sensor-reading helpers for the Husarion RosBot.

All functions read from the hardware handles in devices.py.
Pure data functions — no control logic.
"""

import math

import numpy as np
import cv2

import devices
from config import (
    GREEN_DEPTH_MIN_VALID, GREEN_PIXEL_RATIO, TARGET_PIXEL_RATIO, FRONT_BLOCK_DIST,
    GREEN_ROI_TOP_FRAC,
    BLUE_HSV_LOWER, BLUE_HSV_UPPER,
    YELLOW_HSV_LOWER, YELLOW_HSV_UPPER,
    GREEN_HSV_LOWER, GREEN_HSV_UPPER,
    COLUMN_HEIGHT_CM, COLUMN_DIST_OFFSET_CM, COLUMN_TOP_STRIP_PX,
    GREEN_CAM_HEIGHT_M, GREEN_CAM_X_OFFSET, GREEN_MAX_PROJ_DIST, GREEN_MARK_MIN_PIXELS,
)


INF = float('inf')


def _is_valid_range(v, min_valid):
    return math.isfinite(v) and v > min_valid


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


# HSV colour bands (OpenCV hue 0-179), built once from config.
_HSV_BANDS = {
    "blue":   (np.array(BLUE_HSV_LOWER,   np.uint8), np.array(BLUE_HSV_UPPER,   np.uint8)),
    "yellow": (np.array(YELLOW_HSV_LOWER, np.uint8), np.array(YELLOW_HSV_UPPER, np.uint8)),
    "green":  (np.array(GREEN_HSV_LOWER,  np.uint8), np.array(GREEN_HSV_UPPER,  np.uint8)),
}


def _get_hsv_image():
    """RGB camera frame as an HSV numpy image (h, w, 3), or None if unavailable."""
    img, w, h = _read_camera_image(devices.camera_rgb)
    if img is None:
        return None
    arr = np.frombuffer(img, np.uint8).reshape((h, w, 4))       # Webots frame is BGRA
    bgr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)


def _get_depth_cm():
    """Depth range image as an int16 numpy array in centimetres (-1 = invalid)."""
    data, w, h = _read_depth_image()
    if data is None:
        return None
    depth = np.asarray(data, dtype=np.float32).reshape((h, w)) * 100.0  # m -> cm
    depth = np.where(np.isinf(depth), -1.0, depth)
    return depth.astype(np.int16)


def _segment_color(hsv, color):
    """Binary mask (uint8 0/255) of pixels within the HSV band for `color`."""
    band = _HSV_BANDS.get(color)
    if band is None or hsv is None:
        return None
    lo, hi = band
    return cv2.inRange(hsv, lo, hi)


def _resize_mask_to(mask, shape_hw):
    """Nearest-neighbour resize a mask to (h, w) so it aligns with the depth grid."""
    if mask.shape == shape_hw:
        return mask
    return cv2.resize(mask, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)


def _column_fully_in_frame(mask, top_strip=COLUMN_TOP_STRIP_PX):
    """True if no column pixels touch the top strip (column not clipped at top)."""
    return int(np.count_nonzero(mask[:top_strip, :])) == 0


def _estimate_column_distance_cm(mask, depth_cm):
    """Horizontal distance (cm) to a colour column from depth + known height.

    Ported from AURE estimate_column_distance: take the far depth over the
    column mask, correct for a column clipped at the top of frame, then remove
    the known vertical component (COLUMN_HEIGHT_CM) via Pythagoras.  Returns
    None when unmeasurable, or a small "close" value when the column fills the
    frame but depth is invalid (very near — depth saturates on contact).
    """
    if mask is None or depth_cm is None or not np.any(mask):
        return None

    mask_d = _resize_mask_to(mask, depth_cm.shape)
    depth_values = depth_cm[mask_d != 0]
    valid = depth_values[depth_values > 0]

    if valid.size == 0:
        area_ratio = float(np.count_nonzero(mask)) / float(mask.size)
        return 50.0 if area_ratio > 0.20 else None

    max_depth_cm = float(np.max(valid))
    if not _column_fully_in_frame(mask):
        max_depth_cm *= 1.25 if max_depth_cm < 110.0 else 1.1

    if max_depth_cm <= COLUMN_HEIGHT_CM:
        return float(np.mean(valid))   # too close for the height correction
    horizontal = math.sqrt(max_depth_cm ** 2 - COLUMN_HEIGHT_CM ** 2)
    return horizontal + COLUMN_DIST_OFFSET_CM


def _pixel_to_bearing(col, w, fov):
    """Convert image column to body-frame bearing in radians.

    Sign convention (matches reactive.py laser sectors):
        col = 0       → +fov/2  (target on robot's left, +y body)
        col = w - 1   → -fov/2  (target on robot's right, -y body)
    Returns None for degenerate inputs.
    """
    if w is None or w <= 1 or fov is None:
        return None
    return 0.5 * fov * (1.0 - 2.0 * col / (w - 1))


def _read_rgb_fov():
    if devices.camera_rgb is None:
        return None
    try:
        fov = devices.camera_rgb.getFov()
    except Exception:
        return None
    return fov if fov is not None and math.isfinite(fov) else None


# Cached at import time — RGB camera FOV does not change at runtime.
_rgb_fov = _read_rgb_fov()


def _read_scalar(sensor):
    if sensor is None:
        return None
    try:
        return sensor.getValue()
    except Exception:
        return None


def read_wheel_angles():
    """Return (left_rad, right_rad) — averaged cumulative wheel angles.

    Front and rear wheels on the same side share a motor command, so their
    encoders should agree closely; the average is robust to small jitter.
    Returns (None, None) if any encoder is unavailable.
    """
    fl = _read_scalar(devices.fl_wheel_sensor)
    fr = _read_scalar(devices.fr_wheel_sensor)
    rl = _read_scalar(devices.rl_wheel_sensor)
    rr = _read_scalar(devices.rr_wheel_sensor)
    if fl is None or fr is None or rl is None or rr is None:
        return None, None
    return 0.5 * (fl + rl), 0.5 * (fr + rr)


def read_imu_yaw():
    """Return the inertial-unit yaw in radians, or None if unavailable.

    Webots' InertialUnit returns absolute world-frame roll/pitch/yaw; we
    take element [2].  No baseline subtraction here — that belongs to the
    pose layer, which captures a baseline on reset.
    """
    rpy = _read_vector(devices.inertial_unit, "getRollPitchYaw")
    if rpy is None:
        return None
    yaw = rpy[2]
    return yaw if math.isfinite(yaw) else None


def read_gyro_z():
    """Return the IMU gyro yaw rate (rad/s about +z), or None if unavailable.

    Element [2] is the z-axis (yaw) angular rate in the robot's Z-up frame.
    Unlike wheel encoders, the gyro measures true inertial body rotation, so
    wheel slip cannot fake it — this feeds the localization slip gate.
    """
    vals = _read_vector(devices.gyro, "getValues")
    if vals is None:
        return None
    gz = vals[2]
    return gz if math.isfinite(gz) else None


def read_laser_scan():
    """Return (ranges, fov_rad, max_range_m) for the laser, or (None, None, None).

    `ranges` is the raw range image (index 0 = leftmost ray, matching
    reactive.py's sector convention).  Caller is responsible for
    rejecting inf, near-zero, and near-max-range readings.
    """
    laser = devices.laser
    if laser is None:
        return None, None, None
    try:
        ranges = laser.getRangeImage()
        fov = laser.getFov()
        max_range = laser.getMaxRange()
    except Exception:
        return None, None, None
    if not ranges or fov is None or max_range is None:
        return None, None, None
    return ranges, fov, max_range


def green_ground_points_body():
    """Project detected green-ground pixels onto the floor plane.

    Returns Nx2 body-frame points [x_forward, y_left] (metres) via an inverse
    pinhole model (camera intrinsics from FOV + mounting height).  Only the
    lower image is trusted (ground), and points are subsampled for cost.
    Empty (0, 2) array when no green is visible or the camera is unavailable.
    """
    empty = np.empty((0, 2), dtype=np.float64)
    hsv = _get_hsv_image()
    if hsv is None or _rgb_fov is None:
        return empty
    mask = _segment_color(hsv, "green")
    if mask is None:
        return empty
    h, w = mask.shape
    mask[:int(GREEN_ROI_TOP_FRAC * h), :] = 0     # trust only the lower image (floor)
    vs, us = np.where(mask == 255)                # rows (v), cols (u)
    if us.size < GREEN_MARK_MIN_PIXELS:           # reject stray green (noise -> streaks)
        return empty
    if us.size > 400:                             # subsample dense masks
        idx = np.linspace(0, us.size - 1, 400).astype(int)
        us, vs = us[idx], vs[idx]

    fx = w / (2.0 * math.tan(_rgb_fov / 2.0))
    cx, cy = w / 2.0, h / 2.0
    x_norm = (us - cx) / fx
    y_norm = (vs - cy) / fx                        # square pixels: fy == fx

    # Forward distance on the floor: D = camera_height / y_norm (y_norm below horizon).
    D = GREEN_CAM_HEIGHT_M / (y_norm + 1e-6)
    valid = (y_norm > 0.001) & (D > 0.1) & (D < GREEN_MAX_PROJ_DIST)
    D, x_norm = D[valid], x_norm[valid]
    if D.size == 0:
        return empty

    bx = D + GREEN_CAM_X_OFFSET                     # forward
    by = -D * x_norm                                # left (+) / right (-)
    return np.stack([bx, by], axis=1)


def read_lidar_pointcloud_2d():
    """Return Nx2 lidar points [x_forward, y_left] in the robot body frame.

    Reads the laser point cloud (requires laser.enablePointCloud(), done in
    devices.py), drops non-finite points, and returns an (N, 2) float array.
    Returns an empty (0, 2) array when the lidar is unavailable or the cloud
    is empty.  This is the map layer's sole lidar input.
    """
    laser = devices.laser
    empty = np.empty((0, 2), dtype=np.float64)
    if laser is None:
        return empty
    try:
        cloud = laser.getPointCloud()
    except Exception:
        return empty
    if not cloud:
        return empty
    pts = np.array([[p.x, p.y] for p in cloud], dtype=np.float64)
    if pts.size == 0:
        return empty
    finite = ~np.isinf(pts).any(axis=1) & ~np.isnan(pts).any(axis=1)
    pts = pts[finite]
    if pts.shape[0] == 0:
        return empty
    # Full-range point cloud (matches the reference get_pointcloud_2d): no distance
    # cap.  With 30 particles the pose is accurate enough that far returns don't
    # smear; SLAM downsamples the scan (SLAM_SCAN_MAX_BEAMS) itself.
    return pts


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
    """Detect blue/yellow columns and green ground via HSV segmentation.

    AURE's cv2 approach: blue/yellow report a bearing (from the mask centroid)
    and a depth+height distance estimate; green reports a lower-image ratio and
    a depth-sampled distance.  The return schema is kept stable for the reactive
    stack (autonomous.py, the P key) — this swaps only the detection engine.
    """
    result = {
        "green":  False, "green_ratio":  0.0, "green_distance": INF,
        "blue":   False, "blue_ratio":   0.0,
        "blue_bearing_rad":   None, "blue_distance_m":   INF,
        "yellow": False, "yellow_ratio": 0.0,
        "yellow_bearing_rad": None, "yellow_distance_m": INF,
    }

    hsv = _get_hsv_image()
    if hsv is None:
        return result
    h, w = hsv.shape[:2]
    depth_cm = _get_depth_cm()

    # ── Green ground: trusted only in the lower image ─────────────────────────
    green_mask = _segment_color(hsv, "green")
    if green_mask is not None:
        top = int(GREEN_ROI_TOP_FRAC * h)
        green_mask[:top, :] = 0
        roi_area = w * (h - top)
        green_ratio = int(cv2.countNonZero(green_mask)) / float(roi_area) if roi_area > 0 else 0.0
        result["green_ratio"] = green_ratio
        result["green"] = green_ratio >= GREEN_PIXEL_RATIO
        if result["green"] and depth_cm is not None:
            gm = _resize_mask_to(green_mask, depth_cm.shape)
            gd = depth_cm[gm != 0]
            gd = gd[gd > 0]
            if gd.size:
                result["green_distance"] = float(np.min(gd)) / 100.0   # cm -> m

    # ── Blue / yellow columns ─────────────────────────────────────────────────
    for color in ("blue", "yellow"):
        mask = _segment_color(hsv, color)
        if mask is None:
            continue
        ratio = int(cv2.countNonZero(mask)) / float(w * h)
        result[f"{color}_ratio"] = ratio
        if ratio < TARGET_PIXEL_RATIO:
            continue
        result[color] = True

        moments = cv2.moments(mask)
        if moments["m00"] > 0:
            centroid_col = moments["m10"] / moments["m00"]
            result[f"{color}_bearing_rad"] = _pixel_to_bearing(centroid_col, w, _rgb_fov)

        dist_cm = _estimate_column_distance_cm(mask, depth_cm)
        if dist_cm is not None:
            result[f"{color}_distance_m"] = dist_cm / 100.0   # cm -> m

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
