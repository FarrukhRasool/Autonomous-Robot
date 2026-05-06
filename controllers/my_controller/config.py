"""Tunable control constants for the Husarion RosBot controller.

All physical values are in SI units (metres, radians, seconds).
Adjust these to tune robot behaviour without touching logic files.
"""

# ── Motion speeds ────────────────────────────────────────────────────────────
TARGET_LIN_VEL = 0.35   # m/s   — nominal forward speed
TARGET_ANG_VEL = 0.50   # rad/s — nominal rotation rate
TARGET_REV_VEL = 0.20
# ── Laser front safety thresholds ────────────────────────────────────────────
FRONT_STOP_DIST    = 0.15  # m — hard emergency stop (laser + range sensors)
FRONT_BLOCK_DIST   = 0.25  # m — robot is blocked; stop forward and rotate
FRONT_BLOCK_CLEAR_DIST = 0.30  # m — blocked state releases only after this clearance
FRONT_CAUTION_DIST = 0.35  # m — caution zone entry; slow down and steer

# ── Overhead/floating obstacle depth safety ───────────────────────────────────
OVERHEAD_DETECT_DIST = 1.00  # m — depth ROI max range for floating obstacles

# ── Caution zone speeds ───────────────────────────────────────────────────────
CAUTION_LIN_VEL = 0.20  # m/s   — reduced forward speed in caution zone
CAUTION_ANG_VEL = 0.12  # rad/s — steering rate in caution zone

# ── Wall following ────────────────────────────────────────────────────────────
WALL_TARGET_DIST = 0.35  # m     — desired right-wall following distance
WALL_CLOSE_BAND  = 0.12  # m     — dead-band half-width around target
WALL_LOST_DIST   = 0.80  # m     — right distance above which wall is considered lost
SIDE_DANGER_DIST = 0.03  # m     — side clearance below which danger steering activates
OMEGA_SMALL      = 0.22  # rad/s — gentle wall-correction angular rate

# ── Recovery ─────────────────────────────────────────────────────────────────
BLOCK_TIMEOUT  = 40    # steps — front-blocked steps before timeout recovery activates
REAR_SAFE_DIST = 0.25  # m     — rear range below which reverse is forbidden

# ── RGB semantic detection ────────────────────────────────────────────────────
GREEN_PIXEL_RATIO  = 0.08  # bottom ROI fraction that means forbidden ground
TARGET_PIXEL_RATIO = 0.03  # center ROI fraction that means pillar visible
GREEN_DEPTH_MIN_VALID = 0.03  # m — reject only zero/near-zero green depth pixels
GREEN_STOP_DIST    = 0.35  # m — green is too close; stop and turn away
GREEN_CAUTION_DIST = 0.75  # m — green ahead; slow down and steer away
