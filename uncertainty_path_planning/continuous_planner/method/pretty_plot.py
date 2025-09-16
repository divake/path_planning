#!/usr/bin/env python3
"""
pretty_plot.py — publication-ready 2D map & path rendering for MRPB
Clean, minimal style for ICRA/IEEE conferences
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon

# ---- Global style (ICRA/IEEE friendly) ----
def set_pub_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.linewidth": 0.9,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "savefig.transparent": False,
        "pdf.fonttype": 42,   # editable text in Illustrator
        "ps.fonttype": 42,
        "figure.dpi": 300,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
    })

# Inches for ICRA/IEEE
SIZE_SINGLE = (3.35, 3.35)   # single column
SIZE_DOUBLE = (7.00, 3.35)   # double column short

def make_fig(size="single"):
    set_pub_style()
    if isinstance(size, (tuple, list)):
        fig = plt.figure(figsize=size)
    else:
        fig = plt.figure(figsize=SIZE_SINGLE if size=="single" else SIZE_DOUBLE)
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])  # tight, but leaves room for legend
    return fig, ax

# ---- Core drawing helpers ----
def draw_map(ax, occupancy_grid, wall_thresh=50, wall_px=None):
    """
    Render walls in dark gray, free space white.
    If wall_px is given, we fatten walls by using imshow with interpolation='nearest'
    and a small extent padding (works well for line-art maps).
    """
    occ = (occupancy_grid > wall_thresh).astype(np.uint8)
    # occ==1 means wall; invert for grayscale mapping
    show = 1 - occ  # 1=free(white), 0=wall(black)
    # Use a 2-color grayscale where walls are ~20% gray (not pure black)
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "wallmap", [(0.20, 0.20, 0.20, 1.0), (1.0, 1.0, 1.0, 1.0)], N=256
    )
    # Nearest keeps walls crisp; wall_px can slightly expand by resampling up
    if wall_px is None:
        ax.imshow(show, cmap=cmap, origin="lower", interpolation="nearest")
    else:
        # Upsample to thicken visual walls
        scale = max(1, int(wall_px))
        # Use scipy zoom for smoother upsampling
        from scipy.ndimage import zoom
        upsampled = zoom(show, scale, order=0)  # order=0 for nearest neighbor
        ax.imshow(upsampled, cmap=cmap, origin="lower", interpolation="nearest",
                  extent=[0, occupancy_grid.shape[1], 0, occupancy_grid.shape[0]])

    # Clean axes
    ax.set_aspect("equal")
    ax.set_xlim(0, occupancy_grid.shape[1])
    ax.set_ylim(0, occupancy_grid.shape[0])
    # Don't set empty ticks - will be set by add_axes_labels
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_color("0.15")

def draw_tree(ax, nodes, color=(0.16, 0.60, 0.16, 0.45), lw=0.6, every=1):
    """Darker, more visible tree; set every>1 to sparsify."""
    if nodes is None: return
    c = 0
    for n in nodes:
        if n.parent is None: continue
        c += 1
        if c % every != 0: continue  # Fixed modulo logic
        ax.plot([n.parent.x, n.x], [n.parent.y, n.y], "-", linewidth=lw, color=color, zorder=2)

def draw_path(ax, path_xy, lw=1.6, outline=True):
    """High-contrast red path with optional white outline."""
    if not path_xy: return
    xs = [p[0] for p in path_xy]; ys = [p[1] for p in path_xy]
    if outline:
        ax.plot(xs, ys, "-", linewidth=lw+0.8, color="white", alpha=0.95, zorder=4)
    ax.plot(xs, ys, "-", linewidth=lw, color=(0.82, 0.10, 0.10), zorder=5)

def draw_start_goal(ax, start, goal, s=7.0, edge=1.2):
    """Large, outlined markers that remain visible when shrunk."""
    # Start: filled blue circle with black edge
    start_pt = Circle((start[0], start[1]), radius=s, facecolor=(0.15,0.35,0.85),
                      edgecolor="black", linewidth=edge, zorder=6)
    ax.add_patch(start_pt)

    # Goal: red 5-point star with black edge
    goal_pt = RegularPolygon((goal[0], goal[1]), numVertices=5, radius=s*1.4, orientation=np.pi/2,
                             facecolor=(0.90,0.15,0.15), edgecolor="black", linewidth=edge, zorder=6)
    ax.add_patch(goal_pt)

def add_axes_labels(ax, grid_shape, resolution=0.05, origin=(-7.0, -7.0)):
    """Add axis labels in meters"""
    # Calculate world coordinates
    x_max_m = grid_shape[1] * resolution + origin[0]
    y_max_m = grid_shape[0] * resolution + origin[1]

    # Set ticks in pixels but label in meters
    # X-axis: 5 ticks
    x_ticks_px = np.linspace(0, grid_shape[1], 5)
    x_labels = [f"{(px * resolution + origin[0]):.0f}" for px in x_ticks_px]
    ax.set_xticks(x_ticks_px)
    ax.set_xticklabels(x_labels, fontsize=12, fontweight='bold')

    # Y-axis: 5 ticks
    y_ticks_px = np.linspace(0, grid_shape[0], 5)
    y_labels = [f"{(py * resolution + origin[1]):.0f}" for py in y_ticks_px]
    ax.set_yticks(y_ticks_px)
    ax.set_yticklabels(y_labels, fontsize=12, fontweight='bold')

    # Make tick marks visible and bold
    ax.tick_params(axis='both', which='major', labelsize=12, length=4, width=1.2)

def save_fig(fig, out_base, tight=True):
    bbox = "tight" if tight else None
    fig.savefig(f"{out_base}.pdf", bbox_inches=bbox)
    fig.savefig(f"{out_base}.png", dpi=400, bbox_inches=bbox)
    print(f"Saved {out_base}.pdf and {out_base}.png")

# ---- One-shot convenience ----
def render_rrt_figure(occupancy_grid, start, goal, nodes, path,
                      size="single", wall_px=2, show_tree=True,
                      out_base="rrt_pub", resolution=0.05, origin=(-7.0, -7.0)):
    """
    Complete RRT* figure rendering

    Args:
        occupancy_grid: numpy array of occupancy values
        start: (x, y) start position in pixels
        goal: (x, y) goal position in pixels
        nodes: list of Node objects with parent references
        path: list of [x, y] waypoints
        size: "single", "double", or custom (width, height) tuple
        wall_px: visual thickness of walls (1-3)
        show_tree: whether to show RRT tree
        out_base: base filename for output
        resolution: map resolution in meters/pixel
        origin: map origin in meters
    """
    fig, ax = make_fig(size)
    draw_map(ax, occupancy_grid, wall_px=wall_px)
    if show_tree:
        draw_tree(ax, nodes, lw=0.5, every=1)
    draw_path(ax, path, lw=1.6, outline=True)
    draw_start_goal(ax, start, goal, s=6.0)

    # Add axis labels in meters
    add_axes_labels(ax, occupancy_grid.shape, resolution, origin)

    save_fig(fig, out_base)
    plt.close(fig)