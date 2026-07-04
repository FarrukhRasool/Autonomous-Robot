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

# Safety-net wall-block escape: turn decisively; if the front stays blocked past
# the timeout, reverse out (dead-end / tight corner) then try the other side.
FRONT_BLOCK_TIMEOUT_STEPS  = 20    # decisive-turn ticks before escalating to a reverse
FRONT_ESCAPE_REVERSE_STEPS = 25    # ticks spent reversing to clear the block
FRONT_ESCAPE_REVERSE_VEL   = -0.12 # m/s reverse speed during the escape
FRONT_ESCAPE_TURN_OMEGA    = 0.8   # rad/s — arc while reversing to back out and re-orient

# ── HSV semantic detection (AURE cv2 approach) ────────────────────────────────
# HSV colour bands (OpenCV hue 0-179).  Ported from AURE CONSTANTS.
BLUE_HSV_LOWER   = [100, 150,  50]
BLUE_HSV_UPPER   = [140, 255, 255]
YELLOW_HSV_LOWER = [ 20, 100, 100]
YELLOW_HSV_UPPER = [ 35, 255, 255]
GREEN_HSV_LOWER  = [ 36, 100, 100]
GREEN_HSV_UPPER  = [ 86, 255, 255]

GREEN_PIXEL_RATIO  = 0.08  # lower-image green fraction that means forbidden ground
TARGET_PIXEL_RATIO = 0.03  # full-frame colour fraction that means a pillar is visible
GREEN_ROI_TOP_FRAC = 0.5   # green trusted only below this image fraction (bottom half)

# Column ranging: horizontal distance from far depth + known column height.
COLUMN_HEIGHT_CM      = 125.0  # physical column height (Pythagoras vertical leg)
COLUMN_DIST_OFFSET_CM = 10.0   # additive bias matching AURE estimate_column_distance
COLUMN_TOP_STRIP_PX   = 50     # top rows checked to detect a column clipped at frame top

GREEN_DEPTH_MIN_VALID = 0.03  # m — reject zero/near-zero depth pixels (overhead depth helper)
GREEN_STOP_DIST    = 0.35  # m — green is too close; stop and turn away
GREEN_CAUTION_DIST = 0.75  # m — green ahead; slow down and steer away

# Green-ground map marking (inverse-pinhole projection of green pixels to floor,
# stamped as forbidden CELL_GREEN so the planner/DWA route around it):
GREEN_MARK_ENABLED  = True   # set False to rely on the reactive net only
GREEN_CAM_HEIGHT_M  = 0.17   # RGB camera height above the floor (tune per robot)
GREEN_CAM_X_OFFSET  = 0.03   # camera forward offset from robot centre (m)
GREEN_MAX_PROJ_DIST = 1.2    # m — only mark nearby green; far projection smears into streaks
GREEN_MARK_MIN_PIXELS = 60   # min green pixels in the lower image before marking (reject noise)

# ── Localization ──────────────────────────────────────────────────────────────
POSE_LOG_PERIOD_STEPS = 50  # steps between automatic pose log lines

# Rotational wheel-slip gate: when the encoders claim the robot is turning
# (|d_theta_enc| > SLIP_ENC_ROT_RAD per step) but the gyro says the body is NOT
# rotating (|gyro_z| < SLIP_GYRO_RATE_RADS), the wheels are slipping in place —
# suppress the position update so phantom motion doesn't corrupt the map.
SLIP_ENC_ROT_RAD    = 0.01  # rad/step — encoder-differential rotation to call a "claimed turn"
SLIP_GYRO_RATE_RADS = 0.05  # rad/s    — gyro yaw rate below this = body not actually rotating

# ── Mapping (AURE log-odds occupancy grid) ────────────────────────────────────
MAP_SIZE  = 300           # cells per side (square grid)
MAP_RES_M = 10.0 / 300.0  # m/cell — 300 cells span 10 m (~0.0333 m)

# Log-odds occupancy update (Bresenham from robot cell to each lidar hit):
LOGODDS_INIT = 1.0    # initial value; also the UNKNOWN sentinel (cell never touched)
LOGODDS_FREE = -0.36  # added to cells the ray passes through (free evidence)
LOGODDS_OCC  = 0.85   # added to the ray's endpoint cell (hit evidence)
LOGODDS_LOCK = 3.5    # cells at/above this are frozen (sticky walls); free updates skip them
LOGODDS_CLIP = 5.0    # clamp log-odds to +/- this before the sigmoid

# Discrete grid thresholds (P = sigmoid(log_odds)):
P_OCC  = 0.7  # P above this -> OBSTACLE
P_FREE = 0.5  # P below this -> FREESPACE

# Cell codes (match AURE GridMap so the downstream planner/DWA read the same grid):
CELL_FREE    = 0
CELL_OCC     = 1
CELL_UNKNOWN = 255
CELL_CLOSED  = 200  # reserved — closure marking (later milestone)
CELL_GREEN   = 190  # reserved — green carpet (later milestone)

# Mapping update scheduling (single-loop port of AURE's threaded lidar mapper):
MAP_UPDATE_PERIOD_STEPS = 1     # rebuild the grid every control step (freshest map for DWA)
MAP_UPDATE_MAX_OMEGA    = 5.0   # rad/s — only skip mapping during very fast spins
                                # (DWA turns up to ~2.5 rad/s must still map, or it drives blind)

# ── Path planning (AURE A* + clearance + spline) ──────────────────────────────
ASTAR_INFLATION_LEVELS = [4, 3, 2]  # px — escalating obstacle inflation (safest first)
ASTAR_EXPANSION_PIXELS = 3          # px — free-disk radius around start/goal endpoints
PATH_MIN_LENGTH_M      = 0.8        # m — shorter A* results are retried at thinner inflation
ASTAR_SAFE_DISTANCE_PX = 5.0        # px — clearance band within which the wall penalty applies
ASTAR_PENALTY_STRENGTH = 2.0        # cost weight pushing paths away from walls
ASTAR_HEURISTIC_WEIGHT = 1.2        # A* heuristic multiplier (matches AURE)

# ── DWA local path follower (AURE) ────────────────────────────────────────────
DWA_VELOCITY_SAMPLES = [0.0, 0.1, 0.2, 0.3]   # m/s; 0.0 lets DWA rotate in place to escape
DWA_ANGULAR_SAMPLES  = [0.0, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0, 2.5, -2.5]  # rad/s (gentler)
DWA_ROLLOUT_STEPS    = 15    # forward-prediction horizon (control steps)
DWA_ROBOT_RADIUS_PX  = 3     # px — true robot half-width; keeps the body clear so it can't wedge
DWA_HEADING_WEIGHT   = 4.0   # reward facing the target
DWA_DISTANCE_WEIGHT  = 3.5   # reward closing distance to the target
DWA_SPEED_WEIGHT     = 0.5   # reward higher speed
DWA_CLEARANCE_WEIGHT = 2.0   # reward staying away from obstacles

PATH_FOLLOWING_TARGET_REACH_DIST_PX = 4   # px — waypoint considered reached within this
FOLLOW_WAYPOINT_STRIDE = 5                # advance this many path cells per waypoint
FOLLOW_STUCK_MOVE_M = 0.005               # m/tick below which progress counts as stalled
FOLLOW_STUCK_TURN_RAD = 0.03              # rad/tick — rotating this much still counts as progress
FOLLOW_STUCK_STEPS  = 25                  # stalled ticks before declaring "stuck"

# Progress watchdog: if the robot doesn't get meaningfully closer to its goal
# over a window, the path is stale (a wall was discovered on it) -> replan.
FOLLOW_PROGRESS_WINDOW = 40   # control ticks between progress checks
FOLLOW_MIN_PROGRESS_PX = 3    # min cells closer to goal per window, else replan

# ── Frontier exploration (AURE) ───────────────────────────────────────────────
FRONTIER_MIN_CLUSTER       = 15   # min frontier cells to keep a cluster
FRONTIER_MIN_SIZE          = 20   # min cluster size to score as a primary target
FRONTIER_MIN_DIST_PX       = 5    # ignore frontiers closer than this (already here)
FRONTIER_VISITED_RADIUS_PX = 8    # a frontier within this of a visited one is skipped
FRONTIER_SCORE_BIAS        = 15   # utility = size / (distance + bias)

# Recovery maneuver when the follower reports "stuck" (wedged against a wall):
# reverse (if the rear is clear), then turn a fixed spell, then re-select.
RECOVERY_REVERSE_TICKS = 15    # control ticks spent reversing
RECOVERY_TURN_TICKS    = 20    # control ticks spent turning after reversing
RECOVERY_REVERSE_VEL   = -0.1  # m/s reverse speed during recovery
RECOVERY_TURN_OMEGA    = 1.5   # rad/s turn rate during recovery

# ── Live visualizer ───────────────────────────────────────────────────────────
VIZ_PERIOD_STEPS = 5   # redraw the live cv2 map window every N control steps

# ── Reactive target seeking ───────────────────────────────────────────────────
SEEK_LIN_VEL              = 0.20  # m/s   — forward speed when target is centred
SEEK_OMEGA                = 0.40  # rad/s — yaw rate while orienting toward target
SEEK_BEARING_DEADBAND_RAD = 0.10  # rad   — |bearing| at or below this counts as centred

# ── Mission ───────────────────────────────────────────────────────────────────
TARGET_REACHED_DIST_M     = 0.50  # m     — depth-at-centroid below this counts as reached
APPROACH_OFFSET_M         = 0.40  # m     — aim this far in front of a pillar (its cell is an obstacle)
MISSION_REACHED_RATIO     = 0.25  # frame fraction a pillar fills that also counts as reached