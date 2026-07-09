import numpy as np
import cv2

from config import (
    CELL_FREE, CELL_OCC, CELL_UNKNOWN, CELL_CLOSED, CELL_GREEN,
    CELL_BLUE, CELL_YELLOW,
)

_WINDOW = "RosBot Map (live)"
_failed = False 


def render(grid, robot_cell=None, goals=None, path=None, scale=2):
    global _failed
    if _failed:
        return False

    h, w = grid.shape
    img = np.full((h, w, 3), 128, dtype=np.uint8)
    img[grid == CELL_FREE] = (255, 255, 255)
    img[grid == CELL_OCC] = (0, 0, 0)
    img[grid == CELL_UNKNOWN] = (128, 128, 128)
    img[grid == CELL_GREEN] = (0, 200, 0)
    img[grid == CELL_CLOSED] = (128, 0, 128)
    img[grid == CELL_BLUE] = (255, 0, 0)        
    img[grid == CELL_YELLOW] = (0, 255, 255)   

    if scale != 1:
        img = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)

    def _s(cell):
        return (int(cell[0] * scale), int(cell[1] * scale))

    if path and len(path) > 1:
        pts = np.array([_s(c) for c in path], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], False, (0, 0, 255), 1)   
    if goals:
        for gcell in goals:
            cv2.circle(img, _s(gcell), 4, (0, 255, 0), -1) 

    for _code, _colour in ((CELL_BLUE, (255, 0, 0)), (CELL_YELLOW, (0, 255, 255))):
        _ys, _xs = np.where(grid == _code)
        if len(_xs) > 0:
            cv2.circle(img, _s((int(_xs.mean()), int(_ys.mean()))), 7, _colour, -1)
    if robot_cell is not None:
        cv2.circle(img, _s(robot_cell), 4, (0, 0, 255), -1) 

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
