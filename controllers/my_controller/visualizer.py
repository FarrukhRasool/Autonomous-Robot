"""Live occupancy-grid viewer for the Husarion RosBot (OpenCV window).

Pure NumPy/OpenCV — no Webots imports.  Rendered inline from the control loop
(no background thread, unlike AURE's pygame visualizer): each call colourises
the grid, overlays the robot / goal / current path, and shows it in a single
cv2 window.  If the Webots OpenCV build is headless (no highgui), the first
render fails gracefully, prints a one-time notice, and disables itself.
"""

import numpy as np
import cv2

from config import (
    CELL_FREE, CELL_OCC, CELL_UNKNOWN, CELL_CLOSED, CELL_GREEN,
    CELL_BLUE, CELL_YELLOW,
)

_WINDOW = "RosBot Map (live)"
_failed = False   # set True if the OpenCV build cannot open a window (headless)


def render(grid, robot_cell=None, goals=None, path=None, scale=2,
           blue_cell=None, yellow_cell=None):
    """Draw the grid + overlays into the live window.  Returns True if shown.

    grid : uint8 occupancy grid [row=y, col=x].
    robot_cell : (x, y) robot map cell (or None).
    goals : list of (x, y) target map cells (or None) — drawn green.
    path : list of (x, y) map cells (or None) — the current planned route.
    blue_cell / yellow_cell : remembered pillar cells, drawn in their colours.
    scale : integer upscale factor for visibility.
    """
    global _failed
    if _failed:
        return False

    h, w = grid.shape
    # Colourise (BGR): unknown gray, free white, obstacle black, green, closed purple.
    img = np.full((h, w, 3), 128, dtype=np.uint8)
    img[grid == CELL_FREE] = (255, 255, 255)
    img[grid == CELL_OCC] = (0, 0, 0)
    img[grid == CELL_UNKNOWN] = (128, 128, 128)
    img[grid == CELL_GREEN] = (0, 200, 0)
    img[grid == CELL_CLOSED] = (128, 0, 128)
    img[grid == CELL_BLUE] = (255, 0, 0)          # blue pillar (BGR)
    img[grid == CELL_YELLOW] = (0, 255, 255)      # yellow pillar (BGR)

    if scale != 1:
        img = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)

    def _s(cell):
        return (int(cell[0] * scale), int(cell[1] * scale))

    if path and len(path) > 1:
        pts = np.array([_s(c) for c in path], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], False, (0, 0, 255), 1)   # red route
    if goals:
        for gcell in goals:
            cv2.circle(img, _s(gcell), 4, (0, 255, 0), -1)  # green target(s)
    # Pillars drawn in their own colours (BGR).
    if blue_cell is not None:
        cv2.circle(img, _s(blue_cell), 6, (255, 0, 0), -1)     # blue pillar
    if yellow_cell is not None:
        cv2.circle(img, _s(yellow_cell), 6, (0, 255, 255), -1)  # yellow pillar
    if robot_cell is not None:
        cv2.circle(img, _s(robot_cell), 4, (0, 0, 255), -1)  # robot (red, on top)

    try:
        cv2.imshow(_WINDOW, img)
        cv2.waitKey(1)
        return True
    except Exception as e:
        _failed = True
        print(f"[VIZ] cv2 window unavailable ({e}); live view disabled "
              f"(headless OpenCV build?). Ask to switch the visualizer to pygame.")
        return False


def close():
    """Close the live window (no-op if it was never opened / is unavailable)."""
    try:
        cv2.destroyWindow(_WINDOW)
        cv2.waitKey(1)
    except Exception:
        pass
