# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Goal

Autonomous navigation for a Husarion RosBot in a Webots maze world, written in Python. The robot must:

1. Reach the **blue pillar** first.
2. Then reach the **yellow pillar**.
3. Minimize simulation time.
4. Never drive over **green ground** (forbidden surface).
5. Generalize across maze layouts — solutions must not be hand-fitted to `worlds/Maze1.wbt`.

## Hard Constraints (non-negotiable)

Never suggest or implement:

- Use of the Webots **Supervisor API**.
- Modifying the robot proto or the world (`worlds/*.wbt`, `protos/*.proto`).
- Hardcoded maze-specific paths or waypoints.
- Direct access to hidden global object positions (cheating via simulator state).
- Solutions that only work in `Maze1`.

## Repository Layout

- `controllers/my_controller/my_controller.py` — the only controller. Its name must stay `my_controller` because `worlds/Maze1.wbt` references it via `controller "my_controller"`.
- `worlds/Maze1.wbt` — the maze world. Spawns `Rosbot` proto plus `Solid` walls built from `WallShort` / `WallMedium` / `WallLong`.
- `protos/Wall*.proto` — wall geometry (boxes); not robot-related.
- `plugins/`, `libraries/` — Webots scaffold; currently unused.
- `venv/` — local Python 3.13 venv (gitignored content); used for any host-side tooling. The Webots controller itself runs under Webots' Python, not this venv.
- `docs/specs/001-rosbot-navigation.md` — Spec-Driven Development spec for this assignment.
- `.Codex/commands/sdd-*.md` — SDD slash commands (see below).

## Running and Testing

There is no build system. The workflow is:

1. Open `worlds/Maze1.wbt` in Webots (R2025a).
2. Webots launches `controllers/my_controller/my_controller.py` automatically.
3. To regain keyboard focus, **click the 3D view** before pressing keys.
4. Reset the simulation (`Ctrl+Shift+T` in Webots) to re-run after code changes.

There are no unit tests; every milestone must be **observable in Webots** (logs to the Webots console, robot behavior in the 3D view).

## Current State

The controller is a **keyboard teleop baseline**, not autonomous yet:

- Initializes wheel motors, wheel encoders, IMU (accel/gyro/compass/inertial_unit), 4 short-range range sensors, RGB + depth cameras, and the laser.
- `F/S/A/D` drive forward / back / rotate-left / rotate-right; space stops; `T` runs a 2 s motor self-test.
- Single safety check: forward laser min < `FRONT_STOP_DIST` (0.25 m) blocks forward motion.
- No odometry, mapping, perception, planning, or mission state machine yet.

Build the autonomous stack on top of this — do not rewrite the device-init block from scratch.

## Available Robot Devices

- **Wheels (motors + encoders):** `fl_wheel_joint`, `fr_wheel_joint`, `rl_wheel_joint`, `rr_wheel_joint`. Encoder names use the verbose form: `front left wheel motor sensor`, etc.
- **IMU:** `imu accelerometer`, `imu gyro`, `imu compass`, `imu inertial_unit`.
- **Range sensors (short):** `fl_range`, `fr_range`, `rl_range`, `rr_range`.
- **Vision:** `camera rgb`, `camera depth`.
- **Laser:** `laser`.

## Syllabus-Aligned Build Order

The assignment is graded against the *Autonomous-Robots* syllabus. Implement progressively, in this order:

1. Mobile robot kinematics — linear/angular velocity → wheel speeds.
2. Localization — pose from wheel odometry + IMU.
3. Mapping — laser/depth/range → occupancy grid.
4. Semantic perception — RGB/depth detection of blue, yellow, green.
5. Frontier-based exploration when targets aren't visible.
6. Path planning — A*/Dijkstra on inflated occupancy grid.
7. Path-following control — waypoint tracking.
8. Safety layer — laser/range collision avoidance + green-ground avoidance.

Each layer must be testable in Webots before moving to the next.

## Spec-Driven Development (mandatory)

Every feature follows: **Requirements → Design → Tasks → Implementation → Validation**. Do not implement until requirements, design, and tasks are written and approved.

Custom slash commands enforce this loop — use them instead of free-form coding:

- `/sdd-understand` — read `AGENTS.md` + the spec, summarize, identify status. Read-only.
- `/sdd-plan-next` — propose the smallest useful next milestone with files-touched and acceptance criteria. No code yet.
- `/sdd-implement-approved` — implement the most recently approved milestone.
- `/sdd-review` — audit code against constraints (Supervisor API, hardcoding, safety, modularity).
- `/sdd-debug` — minimal-fix debugging; do not refactor unrelated code.

## Coding Rules

- Python only.
- Keep the controller modular — split into helpers / files when it grows; do not let `my_controller.py` become a monolith.
- Small, testable functions over deep call stacks.
- Comment robotics/math logic (frame conventions, kinematic equations, units) — these are non-obvious. Skip comments on plain Python.
- Prefer simple, observable behavior in early milestones over clever globally-optimal solutions.

## Response Style for This Project

Before writing code: inspect the relevant files, restate what you understood, propose a small next step, and **wait for approval**. After implementing, explain how to test it in Webots (which key to press, what to watch in the console, what behavior in the 3D view confirms success).
