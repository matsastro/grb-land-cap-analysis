#!/usr/bin/env python3
"""
Compute minimal spherical caps that contain X% of land points (0.9 and 1.0) and
produce AEQD maps. The AEQD panel is clipped to a circular disk so that the
visible "ocean"/background takes the correct circular shape for this azimuthal-
equidistant projection.

This script:
 - finds minimal caps for coverage fractions (0.9 and 1.0)
 - draws a single AEQD panel centered on the 100% cap center for each file
 - shows inner reference circles (30°, 60°, 90°(Hem.), 120°) where they fit,
   labels them, and labels the percent-of-surface covered by each outer cap.
"""
import os
import math
import gc
# import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as path_effects
from matplotlib.transforms import Bbox
from sklearn.neighbors import BallTree
from PIL import Image

# --- Settings (adjust these) ---
input_folder = "C:/Users/Matan/OneDrive/Documents/GRB Code/PaleoDEMS"
output_csv = "SurfaceCoverage_90PercentLand.csv"
output_map_dir = "Maps_90_100_2D"
os.makedirs(output_map_dir, exist_ok=True)

# Multiplies every font size in the figure (and the outline stroke behind the
# labels, so the outlines stay proportional to the glyphs). One knob for all text.
FONT_SCALE = 2.0

# Figure size in inches (unchanged from the original 6x6). Kept as a name here
# because the label offsets below are in data units, so text size relative to the
# canvas is what decides whether labels collide.
FIG_SIZE = 6.0

# Output canvas in inches. With bbox_inches='tight' every PNG is cropped to its own
# contents, and since the enlarged title is wider than the axes, a short title ("0 Mya")
# yields a narrower PNG than a long one ("445 Mya"). Sized to the same width in LaTeX,
# that short-title map comes out visibly bigger. Cropping every figure to one fixed box
# centred on the same point keeps the globe the same size across time slices.
# Set to None to go back to the plain 'tight' behaviour.
OUTPUT_CANVAS_IN = (6.6, 6.1)

# Multiplies the linewidth of every cap/reference circle: the 90% and 100% outer
# boundaries, the dashed hemisphere circle, and the dotted 30/60/120 circles.
# Matplotlib scales dash patterns by linewidth, so the dotted circles keep their
# rhythm -- the dots themselves get thicker rather than the gaps closing up.
LINE_SCALE = 1.5

# At 2x the legend is tall enough to sit on top of the "Hem." / "120" / outer-%
# labels at the bottom of the disk, so it is placed under the map instead.
# Set to False to put it back inside the axes at lower right.
LEGEND_BELOW = True

# Only produce output for these times, in Mya. Files whose parsed time is not in
# this list are skipped before anything is read, so nothing else is computed for
# them. Set to None to go back to processing every CSV in the folder.
TARGET_TIMES_MA = None
TIME_MATCH_TOL = 0.5   # Mya; filename times are matched to the list within this tolerance

# Fractions we want to compute (order matters for plotting; 0.9 first)
fractions = [0.9, 1.0]

# Target resolution for the local refinement (0.5 deg)
resolution_deg = 1

# Coarse grid step used to find initial candidate centers (faster than brute force)
coarse_step = 1

# How many land-point candidates to sample for the 100% search
# sample_size_full = 1200

# Plotting options
ENABLE_3D = False   # set True if you want to enable 3D (not implemented here)



# Geometry utilities
def sph_to_cart_vec(lats_deg, lons_deg, r=1.0):
    """Convert latitude/longitude arrays (deg) to unit sphere Cartesian vectors shape (N,3)."""
    lat = np.radians(lats_deg)
    lon = np.radians(lons_deg)
    x = r * np.cos(lat) * np.cos(lon)
    y = r * np.cos(lat) * np.sin(lon)
    z = r * np.sin(lat)
    return np.vstack((x, y, z)).T


def compute_circle_boundary(center_lat, center_lon, radius_deg, n_points=720):
    """
    Robust 3D-rotation method to compute boundary points of a spherical cap
    centered at (center_lat, center_lon) with angular radius radius_deg.
    Returns (lats, lons) arrays in degrees.
    """
    cx, cy, cz = sph_to_cart_vec([center_lat], [center_lon])[0]
    center_vec = np.array([cx, cy, cz], dtype=float)

    # Build an orthonormal tangent basis at center_vec
    if np.allclose(center_vec, [0.0, 0.0, 1.0]):
        ortho1 = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        ortho1 = np.cross(center_vec, [0.0, 0.0, 1.0])
        nrm = np.linalg.norm(ortho1)
        if nrm == 0:
            ortho1 = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            ortho1 /= nrm
    ortho2 = np.cross(center_vec, ortho1)
    ortho2 /= np.linalg.norm(ortho2)

    r = np.radians(radius_deg)
    thetas = np.linspace(0.0, 2.0 * math.pi, n_points, endpoint=False)

    lats = np.empty_like(thetas)
    lons = np.empty_like(thetas)
    for i, theta in enumerate(thetas):
        v = np.cos(r) * center_vec + np.sin(r) * (np.cos(theta) * ortho1 + np.sin(theta) * ortho2)
        v = v / np.linalg.norm(v)
        x, y, z = v
        lats[i] = np.degrees(np.arcsin(z))
        lons[i] = np.degrees(np.arctan2(y, x))
    # normalize longitudes to [-180, 180)
    lons = (lons + 180) % 360 - 180
    return lats, lons


def compute_multiple_circle_boundaries(center_lat, center_lon, radius_deg, n_circles=4):
    """Return list of inner circle boundaries (not including the outermost)."""
    circles = []
    for i in range(1, n_circles):
        frac_radius = radius_deg * i / n_circles
        lats, lons = compute_circle_boundary(center_lat, center_lon, frac_radius)
        circles.append((lats, lons))
    return circles


def haversine_cap_area(radius_rad):
    """Fraction of sphere area inside cap (radius_rad in radians)."""
    return 2 * math.pi * (1 - math.cos(radius_rad)) / (4 * math.pi)


# AEQD projection (pure math — radial distance equals central angle c in radians)
def aeqd_project(lons_deg, lats_deg, center_lon_deg, center_lat_deg):
    """
    Azimuthal Equidistant projection centered on (center_lat_deg, center_lon_deg).
    Returns x,y arrays whose radial distance is the central angle in radians.
    """
    lon = np.radians(np.asarray(lons_deg))
    lat = np.radians(np.asarray(lats_deg))
    lon0 = np.radians(center_lon_deg)
    lat0 = np.radians(center_lat_deg)

    cosc = np.sin(lat0) * np.sin(lat) + np.cos(lat0) * np.cos(lat) * np.cos(lon - lon0)
    cosc = np.clip(cosc, -1.0, 1.0)
    c = np.arccos(cosc)

    # handle near-antipodal / numerical issues
    eps = 1e-8
    k = np.ones_like(c)
    mask = c > eps
    safe = mask.copy()
    # avoid division by tiny sin(c)
    sinc = np.sin(c[mask])
    tiny = 1e-12
    good = np.abs(sinc) > tiny
    idxs = np.nonzero(mask)[0]
    if idxs.size > 0:
        good_idxs = idxs[good]
        if good_idxs.size > 0:
            k[good_idxs] = c[good_idxs] / sinc[good]
        bad_idxs = idxs[~good]
        if bad_idxs.size > 0:
            # mark bad as invalid
            k[bad_idxs] = np.nan
            safe[bad_idxs] = False

    x = np.full_like(c, np.nan, dtype=float)
    y = np.full_like(c, np.nan, dtype=float)
    valid = (~np.isnan(k)) & np.isfinite(k)
    valid = valid & np.isfinite(lat) & np.isfinite(lon)
    if np.any(valid):
        x[valid] = k[valid] * np.cos(lat[valid]) * np.sin(lon[valid] - lon0)
        y[valid] = k[valid] * (np.cos(lat0) * np.sin(lat[valid]) - np.sin(lat0) * np.cos(lat[valid]) * np.cos(lon[valid] - lon0))

    return x, y



# Search / optimization helpers
def evaluate_radius_for_center_via_tree(center_lat, center_lon, tree, k):
    """Return angular radius (radians) to k-th nearest land point using BallTree."""
    dists, _ = tree.query(np.radians([[center_lat, center_lon]]), k=k)
    return float(np.max(dists))


def evaluate_radius_for_center_vector(center_lat, center_lon, land_vecs):
    """Vectorized max angular distance (used for k == N). land_vecs shape: (N,3)"""
    cx, cy, cz = sph_to_cart_vec([center_lat], [center_lon])[0]
    cvec = np.array([cx, cy, cz], dtype=float)
    dots = land_vecs.dot(cvec)
    dots = np.clip(dots, -1.0, 1.0)
    angs = np.arccos(dots)
    return float(np.max(angs))


def evaluate_radius_for_center_mixed(center_lat, center_lon, tree, land_vecs, k):
    """Choose efficient evaluation depending on k relative to N."""
    N = land_vecs.shape[0]
    if k < N:
        return evaluate_radius_for_center_via_tree(center_lat, center_lon, tree, k)
    else:
        return evaluate_radius_for_center_vector(center_lat, center_lon, land_vecs)


def refine_center_local(best_center, tree, land_vecs, k, step_init_deg=2.0, n_iters=10):
    """
    Local greedy refinement around best_center. Halves step each iter.
    Returns (refined_center(lat,lon), radius_radians).
    """
    if best_center is None:
        return None, None
    best_lat, best_lon = float(best_center[0]), float(best_center[1])
    best_r = evaluate_radius_for_center_mixed(best_lat, best_lon, tree, land_vecs, k)
    step = float(step_init_deg)
    for _ in range(n_iters):
        improved = False
        for dlat in (-1, 0, 1):
            for dlon in (-1, 0, 1):
                cand_lat = best_lat + dlat * step
                cand_lon = best_lon + dlon * step
                cand_lat = max(-90.0, min(90.0, cand_lat))
                cand_lon = ((cand_lon + 180.0) % 360.0) - 180.0
                rc = evaluate_radius_for_center_mixed(cand_lat, cand_lon, tree, land_vecs, k)
                if rc + 1e-12 < best_r:
                    best_r = rc
                    best_lat = cand_lat
                    best_lon = cand_lon
                    improved = True
        step /= 2.0
    return (best_lat, best_lon), best_r


def find_best_center_for_full_vectorized(land_vecs, sample_size=1200, seed=12345):
    """
    For k == N (100% coverage) sample candidate land points and compute their
    maximal angular distance to all land points in blocks. Return best candidate
    (lat, lon) and its maximal angular distance (radians).
    """
    N = land_vecs.shape[0]
    rng = np.random.RandomState(seed)
    if N <= sample_size:
        cand_idx = np.arange(N)
    else:
        cand_idx = rng.choice(np.arange(N), size=sample_size, replace=False)
    cand_vecs = land_vecs[cand_idx]
    M = cand_vecs.shape[0]
    K = land_vecs.shape[0]

    maxangs = np.empty(M, dtype=float)
    block = 2000
    for i in range(M):
        cv = cand_vecs[i]
        max_block = 0.0
        for j in range(0, K, block):
            block_vecs = land_vecs[j:j+block]
            dots = block_vecs.dot(cv)
            dots = np.clip(dots, -1.0, 1.0)
            angs = np.arccos(dots)
            mb = float(np.max(angs))
            if mb > max_block:
                max_block = mb
        maxangs[i] = max_block
    bidx = int(np.argmin(maxangs))
    best_vec = cand_vecs[bidx]
    x, y, z = best_vec
    lat = float(np.degrees(np.arcsin(z)))
    lon = float(np.degrees(np.arctan2(y, x)))
    return (lat, lon), float(maxangs[bidx])



# AEQD plotting (single panel, centered @ 100% cap)

def save_aeqd_single_centered(land_lats, land_lons, caps_info, outpath, time_ma=None):
    """
    Draw AEQD panel centered on 100% circle center, with:
      - 90% (red) and 100% (orange) outer boundaries,
      - center dots,
      - inner dotted circles (30°, 60°, 120°) that fit inside each cap,
      - dashed brown hemisphere circle (90°) for each cap (light brown for 100%, darker for 90%),
      - percent-of-surface labels for outer circles (bold),
      - inner-angle labels ("30°","60°","Hem.","120°") placed:
          * UNDER the bottommost part of the circles (in white, with black outline) for the 100% cap
          * ABOVE the topmost part of the circles (in black, with white outline) for the 90% cap
    """
    cap100 = next((c for c in caps_info if abs(c['frac'] - 1.0) < 1e-9), None)
    if cap100 is None:
        raise RuntimeError("No 100% cap found in caps_info!")
    cx_lat, cx_lon = cap100['center']

    fig, ax = plt.subplots(figsize=(FIG_SIZE, FIG_SIZE))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_aspect('equal', adjustable='box')

    # Draw Earth disk (radius pi in AEQD units)
    earth_disk = mpatches.Circle((0.0, 0.0), math.pi, facecolor='lightblue', edgecolor=None, zorder=0)
    ax.add_patch(earth_disk)

    # Use limits that show the full globe disk
    ax.set_xlim(-math.pi, math.pi)
    ax.set_ylim(-math.pi, math.pi)

    # Project and plot land points
    xs_land, ys_land = aeqd_project(np.asarray(land_lons), np.asarray(land_lats), cx_lon, cx_lat)
    mask_land = np.isfinite(xs_land) & np.isfinite(ys_land)
    if np.any(mask_land):
        ax.scatter(xs_land[mask_land], ys_land[mask_land], s=1.0, color='darkgreen', zorder=3, label="Land points")

    handles = []
    labels = []

    ref_angles = [30, 60, 90, 120]

    ordering = sorted(caps_info, key=lambda x: (0 if abs(x['frac'] - 0.9) < 1e-9 else 1))
    for entry in ordering:
        frac = entry['frac']
        entry_center_lat, entry_center_lon = entry['center'][0], entry['center'][1]
        color = 'red' if abs(frac - 0.9) < 1e-9 else 'orange'
        label = "90% land" if abs(frac - 0.9) < 1e-9 else "100% land"

        # Outer cap boundary
        outer_lats, outer_lons = compute_circle_boundary(entry_center_lat, entry_center_lon, entry['radius_deg'], n_points=720)
        xs_outer, ys_outer = aeqd_project(outer_lons, outer_lats, cx_lon, cx_lat)
        finite_outer = np.isfinite(xs_outer) & np.isfinite(ys_outer)
        if np.count_nonzero(finite_outer) > 1:
            line, = ax.plot(xs_outer[finite_outer], ys_outer[finite_outer], color=color, linewidth=2.5 * LINE_SCALE, zorder=5)
            handles.append(line)
            labels.append(label)

            try:
                idxs = np.where(finite_outer)[0]
                if abs(frac - 1.0) < 1e-9:
                    idx_label = idxs[np.argmin(ys_outer[idxs])]
                else:
                    idx_label = idxs[np.argmax(ys_outer[idxs])]
                xt, yt = xs_outer[idx_label], ys_outer[idx_label]
            except Exception:
                xt, yt = None, None

            frac_surface = haversine_cap_area(np.radians(entry['radius_deg']))
            label_text = f"{frac_surface*100:.1f}%, θ={entry['radius_deg']:.1f}°"

            # offsets scale with the text: at a fixed offset the taller labels
            # would run into the next circle's label
            offset_outer = 0.09 * FONT_SCALE
            if xt is not None and yt is not None and np.isfinite(xt) and np.isfinite(yt):
                if abs(frac - 1.0) < 1e-9:
                    txt = ax.text(xt, yt - offset_outer, label_text, color='white', fontsize=10 * FONT_SCALE,
                                  ha='center', va='top', fontweight='bold', zorder=10)  # outer labels on top
                    txt.set_path_effects([path_effects.Stroke(linewidth=1.5 * FONT_SCALE, foreground='black'),
                                          path_effects.Normal()])
                else:
                    txt = ax.text(xt, yt + offset_outer, label_text, color='black', fontsize=10 * FONT_SCALE,
                                  ha='center', va='bottom', fontweight='bold', zorder=10)  # outer labels on top
                    txt.set_path_effects([path_effects.Stroke(linewidth=1.5 * FONT_SCALE, foreground='white'),
                                          path_effects.Normal()])

        # inner reference circles
        hemi_color = "red" if abs(frac - 0.9) < 1e-9 else "orange"

        for a in ref_angles:
            if a >= entry['radius_deg'] - 1e-9:
                continue

            if a == 90:
                lat_h, lon_h = compute_circle_boundary(entry_center_lat, entry_center_lon, 90.0, n_points=720)
                xs_h, ys_h = aeqd_project(lon_h, lat_h, cx_lon, cx_lat)
                finite_h = np.isfinite(xs_h) & np.isfinite(ys_h)
                if np.count_nonzero(finite_h) > 1:
                    ax.plot(xs_h[finite_h], ys_h[finite_h], linestyle='--', color=hemi_color, linewidth=1.5 * LINE_SCALE, zorder=4)
                    try:
                        idxs_h = np.where(finite_h)[0]
                        if abs(frac - 1.0) < 1e-9:
                            idx_label_h = idxs_h[np.argmin(ys_h[idxs_h])]
                        else:
                            idx_label_h = idxs_h[np.argmax(ys_h[idxs_h])]
                        xt_h, yt_h = xs_h[idx_label_h], ys_h[idx_label_h]
                    except Exception:
                        xt_h, yt_h = None, None

                    offset_hemi = 0.035 * FONT_SCALE
                    if xt_h is not None and yt_h is not None and np.isfinite(xt_h) and np.isfinite(yt_h):
                        if abs(frac - 1.0) < 1e-9:
                            txt = ax.text(xt_h, yt_h - offset_hemi, "Hem.", color='white', fontsize=9 * FONT_SCALE,
                                          ha='center', va='top', fontweight='bold', zorder=7)  # inner labels below outer
                            txt.set_path_effects([path_effects.Stroke(linewidth=1.5 * FONT_SCALE, foreground='black'),
                                                  path_effects.Normal()])
                        else:
                            txt = ax.text(xt_h, yt_h + offset_hemi, "Hem.", color='black', fontsize=9 * FONT_SCALE,
                                          ha='center', va='bottom', fontweight='bold', zorder=7)  # inner labels below outer
                            txt.set_path_effects([path_effects.Stroke(linewidth=1.5 * FONT_SCALE, foreground='white'),
                                                  path_effects.Normal()])
            else:
                lat_i, lon_i = compute_circle_boundary(entry_center_lat, entry_center_lon, float(a), n_points=360)
                xs_i, ys_i = aeqd_project(lon_i, lat_i, cx_lon, cx_lat)
                finite_i = np.isfinite(xs_i) & np.isfinite(ys_i)
                if np.count_nonzero(finite_i) > 1:
                    spread_factor = {30: 1, 60: 2, 120: 4}.get(a, 1)
                    dot_line, = ax.plot(xs_i[finite_i], ys_i[finite_i], linestyle=':', color=color, linewidth=1.0 * LINE_SCALE, zorder=4)
                    dot_line.set_dashes([1, 3 * spread_factor])
                    dot_line.set_dash_capstyle('round')
                    try:
                        idxs_i = np.where(finite_i)[0]
                        if abs(frac - 1.0) < 1e-9:
                            idx_label_i = idxs_i[np.argmin(ys_i[idxs_i])]
                        else:
                            idx_label_i = idxs_i[np.argmax(ys_i[idxs_i])]
                        xt_i, yt_i = xs_i[idx_label_i], ys_i[idx_label_i]
                    except Exception:
                        xt_i, yt_i = None, None

                    offset_inner = 0.03 * FONT_SCALE
                    if xt_i is not None and yt_i is not None and np.isfinite(xt_i) and np.isfinite(yt_i):
                        if abs(frac - 1.0) < 1e-9:
                            txt = ax.text(xt_i, yt_i - offset_inner, f"{a}°", color='white', fontsize=9 * FONT_SCALE,
                                          ha='center', va='top', zorder=7)  # inner labels below outer
                            txt.set_path_effects([path_effects.Stroke(linewidth=1.5 * FONT_SCALE, foreground='black'),
                                                  path_effects.Normal()])
                        else:
                            txt = ax.text(xt_i, yt_i + offset_inner, f"{a}°", color='black', fontsize=9 * FONT_SCALE,
                                          ha='center', va='bottom', zorder=7)  # inner labels below outer
                            txt.set_path_effects([path_effects.Stroke(linewidth=1.5 * FONT_SCALE, foreground='white'),
                                                  path_effects.Normal()])

        # center dot
        xc, yc = aeqd_project([entry_center_lon], [entry_center_lat], cx_lon, cx_lat)
        if np.isfinite(xc[0]) and np.isfinite(yc[0]):
            if abs(frac - 0.9) < 1e-9:
                ax.plot(xc, yc, 'o', color=color, markersize=7, zorder=6)
            else:
                ax.plot(xc, yc, 'o', color=color, markersize=5, zorder=6)

    if time_ma is not None:
        ax.set_title(f"Earth Land Clustering at {int(time_ma)} Mya", fontsize=13 * FONT_SCALE)
    else:
        ax.set_title("Earth Land Clustering", fontsize=13 * FONT_SCALE)

    if handles and labels:
        uniq = {}
        for h, l in zip(handles, labels):
            if l not in uniq:
                uniq[l] = h
        # the legend never had a fontsize, so it was on the 10 pt rcParams default
        if LEGEND_BELOW:
            ax.legend(list(uniq.values()), list(uniq.keys()), loc='upper center',
                      bbox_to_anchor=(0.5, -0.01), ncol=2, frameon=True,
                      fontsize=10 * FONT_SCALE)
        else:
            ax.legend(list(uniq.values()), list(uniq.keys()), loc='lower right', frameon=True,
                      fontsize=10 * FONT_SCALE)

    try:
        if OUTPUT_CANVAS_IN is None:
            fig.savefig(outpath, dpi=220, bbox_inches='tight', pad_inches=0.1)
        else:
            fig.canvas.draw()
            tb = fig.get_tightbbox(fig.canvas.get_renderer())
            W, H = OUTPUT_CANVAS_IN
            if tb.width > W or tb.height > H:
                print(f"    [WARN] contents are {tb.width:.2f}x{tb.height:.2f} in, "
                      f"larger than OUTPUT_CANVAS_IN={OUTPUT_CANVAS_IN} -- edges will be cropped")
            # centre the fixed box on the tight box, which is itself centred on the axes
            cx, cy = 0.5 * (tb.x0 + tb.x1), 0.5 * (tb.y0 + tb.y1)
            fig.savefig(outpath, dpi=220,
                        bbox_inches=Bbox.from_bounds(cx - W / 2, cy - H / 2, W, H))
        print(f"    [OK] AEQD image saved: {outpath}")
    except Exception as e:
        print(f"    [FAIL] Could not save AEQD image {outpath}: {type(e).__name__}: {e}")
    finally:
        plt.close(fig)




# (Optional) Plotly 3D placeholder (disabled by default)
if ENABLE_3D:
    import plotly.graph_objects as go
    def create_plotly_figure(land_lats, land_lons, center_lat, center_lon, radius_deg, view):
        raise NotImplementedError("3D mode not enabled in this release. Set ENABLE_3D=True and implement as needed.")



# Main loop
def main():
    results = []
    files = sorted(Path(input_folder).glob("*.csv"))
    if len(files) == 0:
        print("No CSVs found in", input_folder)
        return
    if TARGET_TIMES_MA is not None:
        print("Restricted to times (Mya):", TARGET_TIMES_MA)

    for file in files:
        # parse time from filename (your convention) -- done before anything is read
        # so that files outside TARGET_TIMES_MA cost nothing
        time_str = file.stem.split("_")[-1].replace("Ma", "")
        try:
            time_ma = float(time_str)
        except Exception:
            time_ma = None

        if TARGET_TIMES_MA is not None:
            if time_ma is None or not any(abs(time_ma - t) <= TIME_MATCH_TOL for t in TARGET_TIMES_MA):
                continue

        print(f"\nProcessing: {file.name}")
        try:
            # read file format: skip first header row, columns lon, lat, elev
            df = pd.read_csv(file, skiprows=1, names=["lon", "lat", "elev"])
            df = df[["lat", "lon", "elev"]].dropna()
            df = df.apply(pd.to_numeric, errors="coerce").dropna()
            land_df = df[df["elev"] > 0]
            if len(land_df) < 10:
                print(" Too little land data, skipping.")
                continue

            # prepare BallTree and unit vectors
            land_rad = np.radians(land_df[["lat", "lon"]].values)
            tree = BallTree(land_rad, metric='haversine')
            land_vecs = sph_to_cart_vec(land_df["lat"].values, land_df["lon"].values)

            # build grid once per file (same for all fractions)
            glats = np.arange(-90, 90 + coarse_step, coarse_step)
            glons = np.arange(-180, 180, coarse_step)
            grid = np.array([(lat, lon) for lat in glats for lon in glons])

            caps_info = []

            # compute each fraction
            for frac in fractions:
                print(f" computing frac={frac} ...", end=" ", flush=True)
                N = len(land_vecs)
                k = int(np.ceil(N * frac))

                # chunk the grid to avoid materializing a huge distance matrix
                bytes_per_float = 8
                target_bytes = 1.5e9  # 1.5 GB per chunk; lower if still crashing
                chunk_size = min(500, max(1, int(target_bytes / (k * bytes_per_float))))

                best_r = np.inf
                best_center = None
                for start in range(0, len(grid), chunk_size):
                    chunk = grid[start:start + chunk_size]
                    dists, _ = tree.query(np.radians(chunk), k=k)
                    maxd = np.max(dists, axis=1)
                    idx = int(np.argmin(maxd))
                    if maxd[idx] < best_r:
                        best_r = float(maxd[idx])
                        best_center = tuple(chunk[idx])

                # refine locally until desired resolution (outside chunk loop)
                n_iters = max(4, int(math.ceil(math.log2(coarse_step / max(0.1, resolution_deg))) + 2))
                refined_center, refined_r = refine_center_local(best_center, tree, land_vecs, k, step_init_deg=coarse_step, n_iters=n_iters)
                if refined_center is None:
                    refined_center = best_center
                    refined_r = best_r

                radius_deg = float(np.degrees(refined_r))
                caps_info.append({
                    "frac": frac,
                    "center": (float(refined_center[0]), float(refined_center[1])),
                    "radius_deg": radius_deg
                })
                print(f"done. center={refined_center}, radius_deg={radius_deg:.4f}")

            # Save AEQD map centered on 100% cap
            out_img = f"{output_map_dir}/Map_{int(time_ma) if time_ma is not None else file.stem}_AEQD.png"
            save_aeqd_single_centered(
                land_df["lat"].values, land_df["lon"].values, caps_info, out_img, time_ma=time_ma
            )
            print(f" saved AEQD image: {out_img}")

            # Append to results
            for entry in caps_info:
                results.append({
                    "Time_Ma": time_ma,
                    "Coverage_Fraction": entry['frac'],
                    "Center_Lat": entry['center'][0],
                    "Center_Lon": entry['center'][1],
                    "Radius_deg": entry['radius_deg'],
                    "Fraction_Surface_Covered": haversine_cap_area(np.radians(entry['radius_deg']))
                })

            gc.collect()

        except Exception as e:
            print(f" Error processing {file.name}: {type(e).__name__}: {e}")

    # save CSV results
    if results:
        pd.DataFrame(results).sort_values(["Time_Ma", "Coverage_Fraction"]).to_csv(output_csv, index=False)
        print(f"\nDone. Results saved to {output_csv}")
    else:
        print("\nNo results generated.")
        if TARGET_TIMES_MA is not None:
            print("(Nothing matched TARGET_TIMES_MA =", TARGET_TIMES_MA,
                  "-- check that a filename ends in e.g. '_445Ma')")


if __name__ == "__main__":
    main()