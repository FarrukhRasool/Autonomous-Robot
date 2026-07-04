"""Hardware initialisation for the Husarion RosBot.

Importing this module creates the Robot instance and enables every device.
All device handles are module-level names so other modules can do:

    import devices
    devices.laser.getRangeImage()
"""

from controller import Robot

robot    = Robot()
timestep = int(robot.getBasicTimeStep())


def _get(name):
    dev = robot.getDevice(name)
    if dev is None:
        print(f"WARNING: device '{name}' not found on robot model")
    return dev


def _enable(dev, label=""):
    if dev is None:
        return
    try:
        dev.enable(timestep)
    except Exception:
        print(f"Could not enable {label or dev}")


# ── Motors ────────────────────────────────────────────────────────────────────
fl_motor = _get("fl_wheel_joint")
fr_motor = _get("fr_wheel_joint")
rl_motor = _get("rl_wheel_joint")
rr_motor = _get("rr_wheel_joint")

for _m in (fl_motor, fr_motor, rl_motor, rr_motor):
    if _m is None:
        continue
    try:
        _m.setPosition(float('inf'))
        _m.setVelocity(0.0)
    except Exception as _e:
        print(f"Motor config error: {_e}")

# ── Wheel encoders ────────────────────────────────────────────────────────────
fl_wheel_sensor = _get("front left wheel motor sensor")
fr_wheel_sensor = _get("front right wheel motor sensor")
rl_wheel_sensor = _get("rear left wheel motor sensor")
rr_wheel_sensor = _get("rear right wheel motor sensor")

for _s in (fl_wheel_sensor, fr_wheel_sensor, rl_wheel_sensor, rr_wheel_sensor):
    _enable(_s, "wheel encoder")

# ── IMU ───────────────────────────────────────────────────────────────────────
accelerometer = _get("imu accelerometer")
gyro          = _get("imu gyro")
compass       = _get("imu compass")
inertial_unit = _get("imu inertial_unit")

for _s in (accelerometer, gyro, compass, inertial_unit):
    _enable(_s, "IMU sensor")

# ── Short-range sensors ───────────────────────────────────────────────────────
fl_range = _get("fl_range")
fr_range = _get("fr_range")
rl_range = _get("rl_range")
rr_range = _get("rr_range")

for _s in (fl_range, fr_range, rl_range, rr_range):
    _enable(_s, "range sensor")

# ── Cameras ───────────────────────────────────────────────────────────────────
camera_rgb   = _get("camera rgb")
camera_depth = _get("camera depth")
_enable(camera_rgb,   "camera rgb")
_enable(camera_depth, "camera depth")

# ── Laser ─────────────────────────────────────────────────────────────────────
laser = _get("laser")
_enable(laser, "laser")
if laser is not None:
    try:
        laser.enablePointCloud()   # required by mapping.read_lidar_pointcloud_2d
    except Exception:
        print("Could not enable laser point cloud")

# ── Keyboard ──────────────────────────────────────────────────────────────────
keyboard = robot.getKeyboard()
if keyboard:
    keyboard.enable(timestep)
    print("Keyboard enabled")
else:
    print("WARNING: keyboard device not available")
