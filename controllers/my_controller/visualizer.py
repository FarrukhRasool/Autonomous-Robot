"""Occupancy-grid and path visualiser for the Husarion RosBot.

Saves PNG snapshots using matplotlib (Agg backend — no display required).
Call save_map() at any time; the file appears in the my_controller/ directory
and can be opened with any image viewer while Webots is running.

Pure Python — no Webots imports.
"""

import os
import math

import mapping
from config import GRID_RES_M, GRID_HALF_EXTENT_M

_OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

_MPL_OK = False
try:
    import matplotlib
    matplotlib.use("Agg")          # non-interactive; safe inside Webots subprocess
    import matplotlib.pyplot as plt
    _MPL_OK = True
except ImportError:
    print(
        "[VIZ] matplotlib not found — run:  pip install matplotlib\n"
        "      Map plots will be skipped until it is installed."
    )


# ── Internal helpers ─────────────────────────────────────────────────────────

def _build_rgb_image():
    """Convert the occupancy grid to a 2-D list of [R, G, B] floats for imshow.

    Row index = iy (y axis), column index = ix (x axis).
    origin='lower' in imshow keeps world +y pointing up.
    """
    n, _, _ = mapping.grid_size()
    OCC = mapping.OCCUPIED

    # Snapshot states once (avoids repeated get_cell overhead).
    states = [[mapping.get_cell(ix, iy) for iy in range(n)] for ix in range(n)]

    # Morphological erosion: remove OCCUPIED pixels with fewer than 2
    # occupied neighbours in the 8-connected sense.  Single stray pixels
    # (0 neighbours) and diagonal-only pairs (1 neighbour) are both noise;
    # real wall cells always sit next to at least two other wall cells.
    def _occ_neighbour_count(ix, iy):
        count = 0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = ix + dx, iy + dy
                if 0 <= nx < n and 0 <= ny < n and states[nx][ny] == OCC:
                    count += 1
        return count

    _PALETTE = {
        mapping.UNKNOWN:  [0.70, 0.70, 0.70],
        mapping.FREE:     [1.00, 1.00, 1.00],
        mapping.OCCUPIED: [0.10, 0.10, 0.10],
    }
    default = [0.50, 0.50, 0.50]

    img = []
    for iy in range(n):
        row = []
        for ix in range(n):
            s = states[ix][iy]
            if s == OCC and _occ_neighbour_count(ix, iy) < 2:
                s = mapping.FREE   # sparse noise pixel → treat as free
            row.append(_PALETTE.get(s, default)[:])
        img.append(row)
    return img


# ── Public API ───────────────────────────────────────────────────────────────

def save_map(filename="map.png",
             robot_pose=None,
             path=None,
             blue_pos=None,
             yellow_pos=None):
    """Save the occupancy grid as a PNG with optional overlays.

    Parameters
    ----------
    filename : str
        Output filename, saved to the my_controller/ directory.
    robot_pose : (x_m, y_m, theta_rad), optional
        Current robot pose drawn as a green dot with heading arrow.
    path : list of (x_m, y_m), optional
        Planned path drawn as a blue polyline.
    blue_pos : (x_m, y_m), optional
        Known blue-pillar world position drawn as a blue triangle.
    yellow_pos : (x_m, y_m), optional
        Known yellow-pillar world position drawn as a gold triangle.
    """
    if not _MPL_OK:
        return

    extent = GRID_HALF_EXTENT_M      # world half-size in metres

    img = _build_rgb_image()

    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
    ax.imshow(
        img,
        origin="lower",
        extent=[-extent, extent, -extent, extent],
        interpolation="nearest",
    )

    # ── Planned path ─────────────────────────────────────────────────────────
    if path and len(path) >= 2:
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        ax.plot(xs, ys, color="royalblue", linewidth=2.2,
                label="Planned path", zorder=3)
        ax.scatter(xs[1:-1], ys[1:-1],
                   color="royalblue", s=18, zorder=4)
        ax.plot(xs[0],  ys[0],  marker="^", color="royalblue",
                markersize=9,  zorder=5, label="Path start")
        ax.plot(xs[-1], ys[-1], marker="*", color="royalblue",
                markersize=14, zorder=5, label="Path end")

    # ── Robot pose ───────────────────────────────────────────────────────────
    if robot_pose is not None:
        rx, ry, rtheta = robot_pose
        ax.plot(rx, ry, "go", markersize=12, label="Robot", zorder=6)
        arrow_len = 0.35
        ax.annotate(
            "",
            xy=(rx + arrow_len * math.cos(rtheta),
                ry + arrow_len * math.sin(rtheta)),
            xytext=(rx, ry),
            arrowprops=dict(arrowstyle="->", color="green", lw=2.2),
            zorder=7,
        )

    # ── Pillars ───────────────────────────────────────────────────────────────
    if blue_pos is not None:
        ax.plot(blue_pos[0], blue_pos[1],
                marker="^", color="blue", markersize=16,
                label="Blue pillar", zorder=6)
    if yellow_pos is not None:
        ax.plot(yellow_pos[0], yellow_pos[1],
                marker="^", color="goldenrod", markersize=16,
                label="Yellow pillar", zorder=6)

    # ── Decoration ───────────────────────────────────────────────────────────
    ax.set_xlabel("World X (m)", fontsize=11)
    ax.set_ylabel("World Y (m)", fontsize=11)
    ax.set_title("RosBot Occupancy Grid Map", fontsize=13)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.18, color="gray")
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)

    out_path = os.path.join(_OUTPUT_DIR, filename)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[VIZ] Map saved → {out_path}")
