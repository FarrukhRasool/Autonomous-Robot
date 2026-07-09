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
    OVERHEAD_DETECT_DIST, OVERHEAD_MARK_MIN_PIXELS,
    OVERHEAD_CAM_HEIGHT_M, OVERHEAD_ROBOT_CLEARANCE_M,
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
    idx = 4 * (row * w + col)
    b, g, r = img[idx], img[idx + 1], img[idx + 2]
    return r, g, b


_HSV_BANDS = {
    "blue":   (np.array(BLUE_HSV_LOWER,   np.uint8), np.array(BLUE_HSV_UPPER,   np.uint8)),
    "yellow": (np.array(YELLOW_HSV_LOWER, np.uint8), np.array(YELLOW_HSV_UPPER, np.uint8)),
    "green":  (np.array(GREEN_HSV_LOWER,  np.uint8), np.array(GREEN_HSV_UPPER,  np.uint8)),
}


def _get_hsv_image():
    img, w, h = _read_camera_image(devices.camera_rgb)
    if img is None:
        return None
    arr = np.frombuffer(img, np.uint8).reshape((h, w, 4))  
    bgr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)


def _get_depth_cm():
    data, w, h = _read_depth_image()
    if data is None:
        return None
    depth = np.asarray(data, dtype=np.float32).reshape((h, w)) * 100.0 
    depth = np.where(np.isinf(depth), -1.0, depth)
    return depth.astype(np.int16)


def _segment_color(hsv, color):
    band = _HSV_BANDS.get(color)
    if band is None or hsv is None:
        return None
    lo, hi = band
    return cv2.inRange(hsv, lo, hi)


def _resize_mask_to(mask, shape_hw):
    if mask.shape == shape_hw:
        return mask
    return cv2.resize(mask, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)


def _column_fully_in_frame(mask, top_strip=COLUMN_TOP_STRIP_PX):
    return int(np.count_nonzero(mask[:top_strip, :])) == 0


def _estimate_column_distance_cm(mask, depth_cm):
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
        return float(np.mean(valid))  
    horizontal = math.sqrt(max_depth_cm ** 2 - COLUMN_HEIGHT_CM ** 2)
    return horizontal + COLUMN_DIST_OFFSET_CM


def _pixel_to_bearing(col, w, fov):
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


_rgb_fov = _read_rgb_fov()


def _read_depth_fov():
    if devices.camera_depth is None:
        return None
    try:
        fov = devices.camera_depth.getFov()
    except Exception:
        return None
    return fov if fov is not None and math.isfinite(fov) else None


_depth_fov = _read_depth_fov()


def _read_scalar(sensor):
    if sensor is None:
        return None
    try:
        return sensor.getValue()
    except Exception:
        return None


def read_wheel_angles():
    fl = _read_scalar(devices.fl_wheel_sensor)
    fr = _read_scalar(devices.fr_wheel_sensor)
    rl = _read_scalar(devices.rl_wheel_sensor)
    rr = _read_scalar(devices.rr_wheel_sensor)
    if fl is None or fr is None or rl is None or rr is None:
        return None, None
    return 0.5 * (fl + rl), 0.5 * (fr + rr)


def read_imu_yaw():
    rpy = _read_vector(devices.inertial_unit, "getRollPitchYaw")
    if rpy is None:
        return None
    yaw = rpy[2]
    return yaw if math.isfinite(yaw) else None


def read_gyro_z():
    vals = _read_vector(devices.gyro, "getValues")
    if vals is None:
        return None
    gz = vals[2]
    return gz if math.isfinite(gz) else None


def read_laser_scan():
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
    empty = np.empty((0, 2), dtype=np.float64)
    hsv = _get_hsv_image()
    if hsv is None or _rgb_fov is None:
        return empty
    mask = _segment_color(hsv, "green")
    if mask is None:
        return empty
    h, w = mask.shape
    mask[:int(GREEN_ROI_TOP_FRAC * h), :] = 0    
    vs, us = np.where(mask == 255)               
    if us.size < GREEN_MARK_MIN_PIXELS:          
        return empty
    if us.size > 400:                            
        idx = np.linspace(0, us.size - 1, 400).astype(int)
        us, vs = us[idx], vs[idx]

    fx = w / (2.0 * math.tan(_rgb_fov / 2.0))
    cx, cy = w / 2.0, h / 2.0
    x_norm = (us - cx) / fx
    y_norm = (vs - cy) / fx                       

    D = GREEN_CAM_HEIGHT_M / (y_norm + 1e-6)
    valid = (y_norm > 0.001) & (D > 0.1) & (D < GREEN_MAX_PROJ_DIST)
    D, x_norm = D[valid], x_norm[valid]
    if D.size == 0:
        return empty

    bx = D + GREEN_CAM_X_OFFSET                    
    by = -D * x_norm                             
    return np.stack([bx, by], axis=1)


def read_lidar_pointcloud_2d():
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


_OVERHEAD_ROIS = {
    "left":   (0.05, 0.40, 0.02, 0.55),
    "center": (0.25, 0.75, 0.02, 0.55),
    "right":  (0.60, 0.95, 0.02, 0.55),
}

_OVERHEAD_BAND_COL_FRAC = (0.05, 0.95)
_OVERHEAD_BAND_ROW_FRAC = (0.02, 0.63)


def _overhead_roi_near_depth(data, w, h, x0f, x1f, y0f, y1f):
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
        return INF
    vals.sort()
    return vals[max(0, len(vals) // 20)]  # 5th percentile


def overhead_depth_regions(max_distance):
    data, w, h = _read_depth_image()
    if data is None:
        return False, "none", INF

    best_region = "none"
    best_depth = INF

    for name, roi in _OVERHEAD_ROIS.items():
        depth_near = _overhead_roi_near_depth(data, w, h, *roi)
        if depth_near < best_depth:
            best_depth = depth_near
            best_region = name

    detected = best_depth < max_distance
    return detected, best_region, best_depth


def overhead_depth_region_values():
    data, w, h = _read_depth_image()
    if data is None:
        return {
            "overhead_left": INF,
            "overhead_center": INF,
            "overhead_right": INF,
        }

    return {
        f"overhead_{name}": _overhead_roi_near_depth(data, w, h, *roi)
        for name, roi in _OVERHEAD_ROIS.items()
    }


def overhead_obstacle_points_body():
    empty = np.empty((0, 2), dtype=np.float64)
    data, w, h = _read_depth_image()
    if data is None or _depth_fov is None:
        return empty

    depth = np.asarray(data, dtype=np.float64).reshape((h, w))
    col0 = int(_OVERHEAD_BAND_COL_FRAC[0] * w)
    col1 = int(_OVERHEAD_BAND_COL_FRAC[1] * w)
    row0 = int(_OVERHEAD_BAND_ROW_FRAC[0] * h)
    row1 = int(_OVERHEAD_BAND_ROW_FRAC[1] * h)

    band = depth[row0:row1, col0:col1]
    valid = np.isfinite(band) & (band > GREEN_DEPTH_MIN_VALID) & (band < OVERHEAD_DETECT_DIST)
    rows, cols = np.where(valid)
    if rows.size < OVERHEAD_MARK_MIN_PIXELS:
        return empty
    if rows.size > 400:
        idx = np.linspace(0, rows.size - 1, 400).astype(int)
        rows, cols = rows[idx], cols[idx]

    d = band[rows, cols]
    cols_full = cols + col0                     
    rows_full = rows + row0

    fx = w / (2.0 * math.tan(_depth_fov / 2.0))
    cx, cy = w / 2.0, h / 2.0
    x_norm = (cols_full - cx) / fx
    y_norm = (rows_full - cy) / fx
    ray_norm = np.sqrt(x_norm ** 2 + y_norm ** 2 + 1.0)

    bx = d / ray_norm              
    by = -d * x_norm / ray_norm    

    height_above_floor = OVERHEAD_CAM_HEIGHT_M - d * y_norm / ray_norm
    blocking = height_above_floor <= OVERHEAD_ROBOT_CLEARANCE_M
    if not np.any(blocking):
        return empty
    bx = bx[blocking]
    by = by[blocking]
    return np.stack([bx, by], axis=1)


def get_front_laser_min():
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
                result["green_distance"] = float(np.min(gd)) / 100.0  

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
            result[f"{color}_distance_m"] = dist_cm / 100.0  

    return result


def read_sensor_snapshot():
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
