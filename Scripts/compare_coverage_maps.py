#!/usr/bin/env python3
# cite as: M. Sade, A. Tsarfati and O. Birnholtz (2026), "Gamma Ray Bursts Effects on
# Extinction and Survivability in the Galaxy", arXiv:2603.18250
"""caps, spherical hulls and the sampled GRB direction that lights the most land, side by side.

place in PaleoDEMS and run:

    python compare_coverage_maps.py

reads the MapX_YMa.csv time slices and HullCenterSearch/HullComparison_all.csv (read only)
and writes the figures and example_directions.csv to CoverageMapComparison. red and orange
mean 90% and 100% in panels A and B, and illuminated and unilluminated land in panel C.
with --example-mode gap, panel C shows instead a direction that lights at least 90% of the
land but misses cells of both fixed 90% groups.
"""

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from scipy.optimize import linprog
from scipy.spatial import ConvexHull


TARGET_TIMES_MA = [505, 475, 450, 430, 425, 390, 345]
OUTPUT_FOLDER = "CoverageMapComparison"
RANDOM_SEED = 1
MAX_EXAMPLE_DIRECTIONS = 200_000
EXAMPLE_CHUNK_SIZE = 64
OPEN_HEMISPHERE_TOL = 1e-10
CAP_ROUNDING_TOL_DEG = 1e-6
HULL_PROBABILITY_TOL = 2e-8
RED = "red"
ORANGE = "darkorange"
LAND_COLOR = "darkgreen"
OCEAN_COLOR = "lightblue"
FONT_SCALE = 1.0
LINE_SCALE = 1.0
DPI = 220


def unit_vectors(lat_deg, lon_deg):
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    return np.column_stack((np.cos(lat) * np.cos(lon),
                            np.cos(lat) * np.sin(lon), np.sin(lat)))


def lat_lon(vector):
    v = np.asarray(vector, dtype=float)
    v = v / np.linalg.norm(v)
    return float(np.degrees(np.arcsin(np.clip(v[2], -1, 1)))), \
        float(np.degrees(np.arctan2(v[1], v[0])))


def tangent_basis(center):
    reference = np.eye(3)[np.argmin(np.abs(center))]
    east = np.cross(reference, center)
    east /= np.linalg.norm(east)
    north = np.cross(center, east)
    return east, north


def circle_vectors(center, radius_deg, count=1440):
    center = np.asarray(center, dtype=float)
    center = center / np.linalg.norm(center)
    east, north = tangent_basis(center)
    phase = np.linspace(0, 2 * np.pi, count, endpoint=False)
    radius = np.radians(radius_deg)
    points = (np.cos(radius) * center +
              np.sin(radius) * (np.cos(phase)[:, None] * east +
                                np.sin(phase)[:, None] * north))
    return np.vstack((points, points[0]))


def aeqd_project_vectors(points, view_lat, view_lon):
    """azimuthal equidistant projection, north up."""
    points = np.asarray(points, dtype=float)
    center = unit_vectors([view_lat], [view_lon])[0]
    longitude = np.radians(view_lon)
    east = np.array([-np.sin(longitude), np.cos(longitude), 0.0])
    north = np.cross(center, east)
    cosine = np.clip(points @ center, -1.0, 1.0)
    angle = np.arctan2(np.hypot(points @ east, points @ north), cosine)
    sine = np.sin(angle)
    scale = np.ones_like(angle)
    valid = (np.pi - angle) > 1e-9
    away = (angle > 1e-12) & valid
    scale[away] = angle[away] / sine[away]
    xy = scale[:, None] * np.column_stack((points @ east, points @ north))
    xy[~valid] = np.nan  # the projection's antipode has no unique position
    return xy


def spherical_hull(points, hint=None):
    """spherical convex hull of unit vectors: vertices, perimeter L, area, and the probability
    (2*pi-L)/(4*pi) that a random hemisphere contains it (0 if no open hemisphere does).
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError("A hull needs a nonempty (N, 3) array.")
    lengths = np.linalg.norm(points, axis=1)
    if not np.isfinite(points).all() or np.any(lengths == 0):
        raise ValueError("Invalid land vectors.")
    points = points / lengths[:, None]
    center = None
    if hint is not None and np.linalg.norm(hint) > 0:
        candidate = np.asarray(hint, dtype=float) / np.linalg.norm(hint)
        if np.min(points @ candidate) > OPEN_HEMISPHERE_TOL:
            center = candidate
    if center is None:
        solution = linprog(
            [0.0, 0.0, 0.0, -1.0],
            A_ub=np.column_stack((-points, np.ones(len(points)))),
            b_ub=np.zeros(len(points)), bounds=[(-1, 1)] * 3 + [(0, 1)],
            method="highs",
            options={"primal_feasibility_tolerance": 1e-10,
                     "dual_feasibility_tolerance": 1e-10},
        )
        if not solution.success:
            raise ValueError("Hemisphere feasibility solver failed: " + solution.message)
        if solution.x[3] <= OPEN_HEMISPHERE_TOL:
            return dict(probability=0.0, area=np.nan, perimeter=np.nan,
                        vertices=np.empty((0, 3)),
                        status="no_open_hemisphere_at_numerical_tolerance")
        center = solution.x[:3] / np.linalg.norm(solution.x[:3])
        if np.min(points @ center) <= OPEN_HEMISPHERE_TOL:
            raise ValueError("Hull is too close to the hemisphere tolerance.")

    east, north = tangent_basis(center)
    xy = np.column_stack((points @ east, points @ north)) / (points @ center)[:, None]
    shifted = xy - xy.mean(axis=0)
    _, singular, axes = np.linalg.svd(shifted, full_matrices=False)
    if singular[0] < 1e-13:
        vertices = points[:1]
    elif len(singular) < 2 or singular[1] <= 1e-12 * singular[0]:
        coordinate = shifted @ axes[0]
        vertices = points[[np.argmin(coordinate), np.argmax(coordinate)]]
    else:
        vertices = points[ConvexHull(xy).vertices]
    following = np.roll(vertices, -1, axis=0)
    perimeter = float(np.sum(np.arctan2(
        np.linalg.norm(np.cross(vertices, following), axis=1),
        np.sum(vertices * following, axis=1))))
    probability = (2 * np.pi - perimeter) / (4 * np.pi)
    if probability < -1e-10 or probability > 0.5 + 1e-10:
        raise ValueError("Invalid spherical-hull probability.")
    area = 0.0
    if len(vertices) >= 3:
        anchor = vertices.sum(axis=0)
        anchor /= np.linalg.norm(anchor)
        determinants = np.abs(np.cross(vertices, following) @ anchor)
        denominators = (1 + vertices @ anchor + following @ anchor +
                        np.sum(vertices * following, axis=1))
        area = float(np.sum(2 * np.arctan2(determinants, denominators)))
    return dict(probability=float(np.clip(probability, 0, 0.5)), area=area,
                perimeter=perimeter, vertices=vertices, status="open_hemisphere")


def hull_boundary(vertices, max_step_deg=0.35):
    """points along the short great-circle edges between consecutive vertices."""
    if len(vertices) <= 1:
        return vertices
    pieces = []
    for first, second in zip(vertices, np.roll(vertices, -1, axis=0)):
        angle = np.arctan2(np.linalg.norm(np.cross(first, second)),
                           np.clip(first @ second, -1, 1))
        if angle < 1e-12:
            pieces.append(first[None, :])
            continue
        count = max(2, int(np.ceil(np.degrees(angle) / max_step_deg)))
        fraction = np.linspace(0, 1, count, endpoint=False)
        piece = (np.sin((1 - fraction) * angle)[:, None] * first +
                 np.sin(fraction * angle)[:, None] * second) / np.sin(angle)
        pieces.append(piece / np.linalg.norm(piece, axis=1)[:, None])
    boundary = np.vstack(pieces)
    return np.vstack((boundary, boundary[0]))


def time_of(path):
    match = re.search(r"(?:^|_)(-?\d+(?:\.\d+)?)Ma$", path.stem, re.IGNORECASE)
    return float(match.group(1)) if match else None


def read_land(path):
    frame = pd.read_csv(path, skiprows=1, names=["lon", "lat", "elev"])
    frame = frame.apply(pd.to_numeric, errors="coerce").dropna()
    frame = frame[frame["elev"] > 0]
    if frame.empty:
        raise ValueError("No positive-elevation land cells in " + str(path))
    if (frame["lat"].abs() > 90).any():
        raise ValueError("Latitude outside [-90, 90] in " + str(path))
    return unit_vectors(frame["lat"].to_numpy(), frame["lon"].to_numpy())


def matching_row(frame, time_ma, coverage):
    rows = frame[(np.abs(frame["Time_Ma"] - time_ma) < 0.5) &
                 (frame["Coverage_Percent"] == coverage)]
    if len(rows) != 1:
        raise ValueError(f"Expected one saved {coverage}% row for {time_ma:g} Mya; "
                         f"found {len(rows)}.")
    return rows.iloc[0]


def saved_cap(row):
    center = unit_vectors([float(row["Original_cap_center_lat"])],
                          [float(row["Original_cap_center_lon"])])[0]
    radius = float(row["Radius_deg"])
    if not 0 <= radius <= 180:
        raise ValueError("Saved cap radius is outside [0, 180].")
    return dict(center=center, radius=radius)


def cap_hit(direction, cap):
    return bool(cap["radius"] < 90 and
                direction @ cap["center"] >= np.sin(np.radians(cap["radius"])))


def probability_text(probability):
    percent = 100 * probability
    if percent == 0:
        return "0%"
    if percent < 0.001:
        return f"{percent:.2g}%"
    if percent < 0.1:
        return f"{percent:.4f}%"
    return f"{percent:.2f}%"


def prepare_epoch(land, row90, row100):
    """rebuild both 90% groups and check them against the saved results."""
    count = len(land)
    k = int(np.ceil(0.9 * count))
    if int(row90["N_land_cells"]) != count or int(row100["N_land_cells"]) != count:
        raise ValueError("Land-cell count differs from the saved comparison. "
                         "Use the same raw time-slice CSV as the probability calculation.")
    cap90, cap100 = saved_cap(row90), saved_cap(row100)
    original = np.argsort(-(land @ cap90["center"]), kind="stable")[:k]
    required90 = np.degrees(np.arccos(np.clip(
        np.min(land[original] @ cap90["center"]), -1, 1)))
    required100 = np.degrees(np.arccos(np.clip(
        np.min(land @ cap100["center"]), -1, 1)))
    for coverage, required, cap in ((90, required90, cap90), (100, required100, cap100)):
        if required > cap["radius"] + CAP_ROUNDING_TOL_DEG:
            raise ValueError(f"The saved {coverage}% cap does not enclose the required "
                             "cells from this raw file. Check the time slice and CSV versions.")

    original_hull = spherical_hull(land[original], hint=cap90["center"])
    best_lat, best_lon = (float(row90["Search_best_center_lat"]),
                          float(row90["Search_best_center_lon"]))
    if not np.isfinite([best_lat, best_lon]).all():
        raise ValueError("A saved best 90% selection center is missing for this epoch.")
    best_center = unit_vectors([best_lat], [best_lon])[0]
    best = np.argsort(-(land @ best_center), kind="stable")[:k]
    best_hull = spherical_hull(land[best], hint=best_center)
    full_hull = spherical_hull(land, hint=cap100["center"])
    for label, hull, expected in (
        ("original 90%", original_hull, float(row90["P_hull_fixed_pct"]) / 100),
        ("best-found 90%", best_hull, float(row90["P_hull_best_found_pct"]) / 100),
        ("100%", full_hull, float(row100["P_hull_fixed_pct"]) / 100),
    ):
        if abs(hull["probability"] - expected) > HULL_PROBABILITY_TOL:
            raise ValueError(f"Reconstructed {label} hull probability differs from the "
                             "saved result. Check raw-file ordering and saved selection centers.")
    return dict(land=land, k=k, cap90=cap90, cap100=cap100,
                original=original, best=best, original_hull=original_hull,
                best_hull=best_hull, full_hull=full_hull,
                row90=row90, row100=row100,
                view_lat=float(row100["Original_cap_center_lat"]),
                view_lon=float(row100["Original_cap_center_lon"]))


def find_gap_examples(epoch, directions, number=1):
    """directions that light at least 90% of the land but miss cells of both saved fixed 90% groups.
    takes the first ones found; it does not estimate p.
    """
    land, k = epoch["land"], epoch["k"]
    selected = []
    for start in range(0, len(directions), EXAMPLE_CHUNK_SIZE):
        chunk = directions[start:start + EXAMPLE_CHUNK_SIZE]
        lit = chunk @ land.T > 0
        counts = lit.sum(axis=1)
        missed_original = k - lit[:, epoch["original"]].sum(axis=1)
        missed_best = k - lit[:, epoch["best"]].sum(axis=1)
        if epoch["cap90"]["radius"] < 90:
            cap_fully_lit = (chunk @ epoch["cap90"]["center"] >=
                             np.sin(np.radians(epoch["cap90"]["radius"])))
        else:
            cap_fully_lit = np.zeros(len(chunk), dtype=bool)
        eligible = np.flatnonzero((counts >= k) & (missed_original > 0) &
                                  (missed_best > 0) & ~cap_fully_lit)
        for offset in eligible:
            direction, mask = chunk[offset], lit[offset].copy()
            if any(np.array_equal(mask, example["lit"]) for example in selected):
                continue
            # keep later examples distinct from the first one
            if any(direction @ example["direction"] > np.cos(np.radians(10))
                   for example in selected):
                continue
            selected.append(dict(
                direction=direction.copy(), lit=mask,
                direction_index=start + int(offset),
                lit_count=int(counts[offset]),
                missed_original=int(missed_original[offset]),
                missed_best=int(missed_best[offset]),
            ))
            if len(selected) == number:
                return selected
    raise ValueError(f"Found only {len(selected)} of {number} requested illustrative "
                     f"directions in {len(directions)} draws. Increase --max-directions "
                     "or request fewer examples.")


def find_examples(epoch, directions, number=1, mode="maximum"):
    """the sampled direction that lights the most land cells (mode 'maximum'), or gap examples
    (mode 'gap'). further examples are alternatives at least 10 degrees apart.
    """
    if number < 1 or len(directions) < 1:
        raise ValueError("At least one direction and one example are required.")
    if mode == "gap":
        selected = find_gap_examples(epoch, directions, number)
        scanned = min(len(directions), EXAMPLE_CHUNK_SIZE *
                      (max(item["direction_index"] for item in selected) //
                       EXAMPLE_CHUNK_SIZE + 1))
        for item in selected:
            item.update(selection_mode="gap", directions_checked=scanned,
                        maximum_sampled_count=None, coverage_rank=None)
        return selected
    if mode != "maximum":
        raise ValueError("Unknown direction-selection mode: " + str(mode))

    land = epoch["land"]
    counts = np.empty(len(directions), dtype=np.int64)
    report_every = max(EXAMPLE_CHUNK_SIZE, min(5000, len(directions) // 10))
    next_report = report_every
    highest = -1
    for start in range(0, len(directions), EXAMPLE_CHUNK_SIZE):
        stop = min(start + EXAMPLE_CHUNK_SIZE, len(directions))
        counts[start:stop] = (directions[start:stop] @ land.T > 0).sum(axis=1)
        highest = max(highest, int(counts[start:stop].max()))
        if stop >= next_report or stop == len(directions):
            print(f"  Checked {stop:,}/{len(directions):,} directions; "
                  f"highest coverage so far: {100 * highest / len(land):.3f}%",
                  flush=True)
            next_report = stop + report_every

    order = np.argsort(-counts, kind="stable")
    selected = []
    for rank, index in enumerate(order, 1):
        direction = directions[index]
        if any(direction @ item["direction"] > np.cos(np.radians(10))
               for item in selected):
            continue
        # re-evaluate with the same block shape as above, so rounding at the boundary is identical
        start = (int(index) // EXAMPLE_CHUNK_SIZE) * EXAMPLE_CHUNK_SIZE
        block_lit = directions[start:start + EXAMPLE_CHUNK_SIZE] @ land.T > 0
        mask = block_lit[int(index) - start].copy()
        if any(np.array_equal(mask, item["lit"]) for item in selected):
            continue
        lit_count = int(mask.sum())
        if lit_count != int(counts[index]):
            raise ValueError("Illuminated-cell count was not reproducible.")
        selected.append(dict(
            direction=direction.copy(), lit=mask, direction_index=int(index),
            lit_count=lit_count,
            missed_original=int(np.count_nonzero(~mask[epoch["original"]])),
            missed_best=int(np.count_nonzero(~mask[epoch["best"]])),
            selection_mode="maximum", directions_checked=len(directions),
            maximum_sampled_count=highest, coverage_rank=rank,
        ))
        if len(selected) == number:
            return selected
    raise ValueError(f"Only {len(selected)} distinct high-coverage examples were "
                     f"available; request fewer than {number} examples.")


def plot_projected_curve(ax, points, epoch, **style):
    xy = aeqd_project_vectors(points, epoch["view_lat"], epoch["view_lon"])
    # break the curve at the antipode cut instead of drawing a false chord
    separated = np.any(~np.isfinite(xy), axis=1)
    if len(xy) > 1:
        separated[1:] |= np.linalg.norm(np.diff(xy, axis=0), axis=1) > 0.45
    xy = xy.copy()
    xy[separated] = np.nan
    line, = ax.plot(xy[:, 0], xy[:, 1], **style)
    return line


def draw_ocean_and_shadow(ax, epoch, direction):
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-3.68, 3.68)
    ax.set_ylim(-3.68, 3.68)
    ax.axis("off")
    disk = Circle((0, 0), np.pi, facecolor=OCEAN_COLOR,
                  edgecolor="0.65", linewidth=0.6, zorder=0)
    ax.add_patch(disk)
    # inverse AEQD: shade the hemisphere left dark by this direction
    grid = np.linspace(-np.pi, np.pi, 500)
    xx, yy = np.meshgrid(grid, grid)
    radius = np.hypot(xx, yy)
    center = unit_vectors([epoch["view_lat"]], [epoch["view_lon"]])[0]
    longitude = np.radians(epoch["view_lon"])
    east = np.array([-np.sin(longitude), np.cos(longitude), 0.0])
    north = np.cross(center, east)
    scale = np.ones_like(radius)
    nonzero = radius > 1e-12
    scale[nonzero] = np.sin(radius[nonzero]) / radius[nonzero]
    dot = (np.cos(radius) * (center @ direction) +
           scale * (xx * (east @ direction) + yy * (north @ direction)))
    rgba = np.zeros((*radius.shape, 4))
    rgba[:, :, :3] = 0.05
    rgba[:, :, 3] = ((radius <= np.pi) & (dot <= 0)) * 0.17
    shade = ax.imshow(rgba, origin="lower", extent=(-np.pi, np.pi, -np.pi, np.pi),
                      interpolation="nearest", zorder=1)
    shade.set_clip_path(disk)
    plot_projected_curve(ax, circle_vectors(direction, 90), epoch,
                         color="black", linestyle="--", linewidth=1.0 * LINE_SCALE,
                         zorder=5)


def corner(ax, location, text, color):
    positions = {
        "tl": (0.015, 0.985, "left", "top"),
        "tr": (0.985, 0.985, "right", "top"),
        "bl": (0.015, 0.015, "left", "bottom"),
        "br": (0.985, 0.015, "right", "bottom"),
    }
    x, y, ha, va = positions[location]
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va,
            fontsize=10 * FONT_SCALE, color=color, fontweight="bold", zorder=10)


def panel_legend(ax, handles, columns=2):
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncol=columns, frameon=False, fontsize=8.5 * FONT_SCALE,
              handlelength=2.3, columnspacing=1.0, borderaxespad=0)


def render_comparison(epoch, example, time_ma, output_path, example_number):
    land, lit = epoch["land"], example["lit"]
    xy = aeqd_project_vectors(land, epoch["view_lat"], epoch["view_lon"])
    finite = np.isfinite(xy).all(axis=1)
    fig, axes = plt.subplots(1, 3, figsize=(19, 7.8))
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.26, top=0.89, wspace=0.06)
    fig.suptitle(f"{time_ma:g} Mya: caps, hulls and GRB land coverage",
                 fontsize=17 * FONT_SCALE, y=0.98)
    for ax in axes:
        draw_ocean_and_shadow(ax, epoch, example["direction"])

    # panel A: fitted caps
    ax = axes[0]
    ax.set_title("A. Fitted circular caps", fontsize=13 * FONT_SCALE, pad=10)
    ax.scatter(xy[finite, 0], xy[finite, 1], s=1.4,
               color=LAND_COLOR, zorder=3, rasterized=True)
    for coverage, color, cap, top, bottom in (
        (90, RED, epoch["cap90"], "tl", "bl"),
        (100, ORANGE, epoch["cap100"], "tr", "br"),
    ):
        plot_projected_curve(ax, circle_vectors(cap["center"], cap["radius"]), epoch,
                             color=color, linewidth=2.1 * LINE_SCALE, zorder=6)
        center_xy = aeqd_project_vectors(cap["center"][None, :],
                                         epoch["view_lat"], epoch["view_lon"])
        ax.plot(center_xy[:, 0], center_xy[:, 1], "o", color=color, markersize=4, zorder=7)
        area = 50 * (1 - np.cos(np.radians(cap["radius"])))
        corner(ax, top, f"A{coverage}={area:.1f}%", color)
        corner(ax, bottom, f"\u03b8{coverage}={cap['radius']:.1f}\N{DEGREE SIGN}", color)
    row90, row100 = epoch["row90"], epoch["row100"]
    ax.text(0.5, -0.025,
            "p(whole 90% cap): " + probability_text(float(row90["P_cap_pct"]) / 100) +
            "\np(whole 100% cap): " + probability_text(float(row100["P_cap_pct"]) / 100),
            ha="center", va="top", transform=ax.transAxes, fontsize=10 * FONT_SCALE)
    panel_legend(ax, [
        Line2D([], [], color=RED, lw=2, label="90% cap"),
        Line2D([], [], color=ORANGE, lw=2, label="100% cap"),
    ])

    # panel B: hull edges are great-circle arcs
    ax = axes[1]
    ax.set_title("B. Spherical convex hulls", fontsize=13 * FONT_SCALE, pad=10)
    ax.scatter(xy[finite, 0], xy[finite, 1], s=1.4,
               color=LAND_COLOR, zorder=3, rasterized=True)
    original_hull, best_hull, full_hull = (
        epoch["original_hull"], epoch["best_hull"], epoch["full_hull"])
    plot_projected_curve(ax, hull_boundary(original_hull["vertices"]), epoch,
                         color=RED, linestyle="--", linewidth=1.3 * LINE_SCALE, zorder=6)
    plot_projected_curve(ax, hull_boundary(best_hull["vertices"]), epoch,
                         color=RED, linewidth=2.2 * LINE_SCALE, zorder=7)
    missed = epoch["best"][~lit[epoch["best"]]]
    missed = missed[finite[missed]]
    ax.scatter(xy[missed, 0], xy[missed, 1], marker="x", s=50,
               color="black", linewidths=1.5, zorder=20, rasterized=True)
    corner(ax, "tl", f"A90={100 * best_hull['area'] / (4*np.pi):.1f}%", RED)
    corner(ax, "bl", f"V90={len(best_hull['vertices'])}", RED)
    handles = [
        Line2D([], [], color=RED, lw=2.2, label="Best fixed 90% group"),
        Line2D([], [], color=RED, lw=1.3, ls="--", label="Original fixed 90% group"),
    ]
    if len(missed):
        handles.append(Line2D([], [], color="black", marker="x", ls="",
                              label="Unlit cells in best 90% group"))
    if len(full_hull["vertices"]):
        plot_projected_curve(ax, hull_boundary(full_hull["vertices"]), epoch,
                             color=ORANGE, linewidth=1.7 * LINE_SCALE, zorder=6)
        corner(ax, "tr", f"A100={100 * full_hull['area'] / (4*np.pi):.1f}%", ORANGE)
        corner(ax, "br", f"V100={len(full_hull['vertices'])}", ORANGE)
        handles.append(Line2D([], [], color=ORANGE, lw=1.7, label="100% hull"))
    else:
        # no 100% hull when the land fits in no open hemisphere (p = 0)
        corner(ax, "tr", "100%: p=0", ORANGE)
        corner(ax, "br", "No open\nhemisphere", ORANGE)
    ax.text(0.5, -0.025,
            "p(original 90%): " + probability_text(original_hull["probability"]) +
            "   best: " + probability_text(best_hull["probability"]) +
            "\np(100%): " + probability_text(full_hull["probability"]) +
            f"   Unlit cells in best 90% group: {example['missed_best']}",
            ha="center", va="top", transform=ax.transAxes, fontsize=10 * FONT_SCALE)
    panel_legend(ax, handles, columns=2)

    # panel C: all illuminated land is red
    ax = axes[2]
    if example["selection_mode"] == "maximum":
        title = ("C. Highest land coverage found" if example_number == 1
                 else "C. Alternative high-coverage direction")
        title += f"\nAmong {example['directions_checked']:,} sampled directions"
    else:
        title = "C. Example illustrating the gap"
    ax.set_title(title, fontsize=13 * FONT_SCALE, pad=10)
    for mask, color in ((~lit & finite, ORANGE), (lit & finite, RED)):
        ax.scatter(xy[mask, 0], xy[mask, 1], s=1.7, color=color,
                   zorder=4, rasterized=True)
    illuminated = 100 * example["lit_count"] / len(land)
    corner(ax, "tl", f"Lit: {illuminated:.2f}%", RED)
    corner(ax, "tr", f"Unlit: {100-illuminated:.2f}%", ORANGE)
    corner(ax, "bl", f"{example['lit_count']:,} cells", RED)
    corner(ax, "br", f"{len(land)-example['lit_count']:,} cells", ORANGE)
    ax.text(0.5, -0.025,
            "Saved MC p(any 90%): " + probability_text(float(row90["P_land_MC_pct"]) / 100) +
            "\nSaved MC p(100%): " + probability_text(float(row100["P_land_MC_pct"]) / 100),
            ha="center", va="top", transform=ax.transAxes, fontsize=10 * FONT_SCALE)
    panel_legend(ax, [
        Line2D([], [], color=RED, marker="o", markersize=4, ls="", label="Illuminated land"),
        Line2D([], [], color=ORANGE, marker="o", markersize=4, ls="", label="Unilluminated land"),
    ])

    latitude, longitude = lat_lon(example["direction"])
    fig.text(0.5, 0.085,
             f"Same sampled hemisphere in all panels: dashed black boundary; dark shading = unilluminated."
             f"   GRB center: {latitude:.1f}\N{DEGREE SIGN}, {longitude:.1f}\N{DEGREE SIGN}.",
             ha="center", fontsize=10 * FONT_SCALE)
    fig.text(0.5, 0.050,
             "A = fraction of Earth's surface; V = hull vertices. "
             "The 90% criterion counts land cells equally, as in the probability calculations.",
             ha="center", fontsize=9 * FONT_SCALE)
    if example["selection_mode"] == "maximum":
        if example_number > 1:
            highest = 100 * example["maximum_sampled_count"] / len(land)
            note = (f"Alternative high-coverage direction; the highest sampled coverage is {highest:.2f}%. "
                    "Saved MC probabilities summarize all directions.")
        elif example["lit_count"] == len(land):
            note = ("This direction illuminates all land cells, reaching the 100% upper limit. "
                    "Saved MC probabilities still summarize all sampled directions.")
        else:
            note = ("Direction selected by maximum illuminated-cell count among the samples. "
                    "Saved MC probabilities summarize all directions.")
    else:
        note = ("This direction was selected to illustrate the gap, not to maximize land coverage. "
                "Saved MC probabilities summarize all sampled directions.")
    fig.text(0.5, 0.018, note, ha="center", fontsize=9 * FONT_SCALE)
    fig.savefig(output_path, dpi=DPI, facecolor="white")
    plt.close(fig)
    return dict(
        Time_Ma=time_ma, Example_Number=example_number,
        Seed=RANDOM_SEED, Direction_index_zero_based=example["direction_index"],
        Selection_mode=example["selection_mode"],
        Directions_checked=example["directions_checked"],
        Coverage_rank_among_samples=example["coverage_rank"],
        Maximum_sampled_illuminated_cells=example["maximum_sampled_count"],
        GRB_center_lat=latitude, GRB_center_lon=longitude,
        N_land_cells=len(land), N_illuminated=example["lit_count"],
        Land_illuminated_pct=illuminated,
        Original_fixed_90_cells_left_dark=example["missed_original"],
        Best_fixed_90_cells_left_dark=example["missed_best"],
        Whole_90_cap_illuminated=cap_hit(example["direction"], epoch["cap90"]),
        Whole_100_cap_illuminated=cap_hit(example["direction"], epoch["cap100"]),
        P_cap_90_pct=float(row90["P_cap_pct"]),
        P_original_hull_90_pct=100 * original_hull["probability"],
        P_best_hull_90_pct=100 * best_hull["probability"],
        P_MC_any_90_pct=float(row90["P_land_MC_pct"]),
        P_MC_100_pct=float(row100["P_land_MC_pct"]),
        Output_image=output_path.name,
    )


def geometry_checks():
    octant = spherical_hull(np.eye(3))
    np.testing.assert_allclose(octant["probability"], 1 / 8, atol=1e-12)
    np.testing.assert_allclose(octant["area"], np.pi / 2, atol=1e-12)
    np.testing.assert_allclose(
        spherical_hull(np.array([[1., 0, 0]]))["probability"], 0.5, atol=1e-12)
    pair = unit_vectors([0, 0], [0, 120])
    np.testing.assert_allclose(spherical_hull(pair)["probability"], 1 / 6, atol=1e-12)
    tetrahedron = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1.]])
    assert spherical_hull(tetrahedron)["probability"] == 0
    ring = circle_vectors([1., 0, 0], 60, count=1440)[:-1]
    expected = (1 - np.sin(np.radians(60))) / 2
    np.testing.assert_allclose(spherical_hull(ring)["probability"], expected, atol=3e-7)
    positions = aeqd_project_vectors(unit_vectors([0, 0, 30], [0, 45, 0]), 0, 0)
    np.testing.assert_allclose(positions, [[0, 0], [np.pi/4, 0], [0, np.pi/6]], atol=1e-12)
    edge_points = hull_boundary(octant["vertices"])
    np.testing.assert_allclose(np.linalg.norm(edge_points, axis=1), 1, atol=1e-12)
    assert time_of(Path("MapX_505Ma.csv")) == 505
    assert time_of(Path("top10_probability_90.csv")) is None
    print("Geometry checks passed: point, pair, octant, tetrahedron, cap limit, "
          "AEQD projection and great-circle edges.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-dir", type=Path,
                        default=Path(__file__).resolve().parent,
                        help="PaleoDEMS folder; default is beside this script")
    parser.add_argument("--results", type=Path,
                        help="Saved center-search CSV; default: HullCenterSearch/HullComparison_all.csv")
    parser.add_argument("--times", nargs="+", type=float, default=TARGET_TIMES_MA)
    parser.add_argument("--examples", type=int, default=1,
                        help="Maximum direction plus distinct high-coverage alternatives; default 1")
    parser.add_argument("--example-mode", choices=("maximum", "gap"), default="maximum",
                        help="maximum: best of all sampled directions (default); gap: earlier counterexample")
    parser.add_argument("--max-directions", type=int, default=MAX_EXAMPLE_DIRECTIONS,
                        help="Number of sampled directions searched; all are checked in maximum mode")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        geometry_checks()
        return
    if args.examples < 1 or args.max_directions < 1:
        raise ValueError("--examples and --max-directions must be positive.")
    if not np.isfinite(args.times).all() or len(set(args.times)) != len(args.times):
        raise ValueError("--times must contain distinct, finite dates.")

    root = args.base_dir.expanduser().resolve()
    # the script may sit in PaleoDEMS or beside it
    if (root / "PaleoDEMS").is_dir() and not (root / "HullCenterSearch").is_dir():
        root = root / "PaleoDEMS"
    if not root.is_dir():
        raise ValueError("PaleoDEMS folder does not exist: " + str(root))
    if args.results is not None:
        source = args.results.expanduser().resolve()
    else:
        choices = [root / "HullCenterSearch" / "HullComparison_all.csv",
                   root / "HullComparison_all.csv"]
        source = next((path for path in choices if path.is_file()), None)
        if source is None:
            raise ValueError("Saved center-search results not found. Expected "
                             "HullCenterSearch/HullComparison_all.csv inside " + str(root) +
                             ". Alternatively supply --results with the CSV's exact path.")
    if not source.is_file():
        raise ValueError("Saved results file does not exist: " + str(source))
    saved = pd.read_csv(source)
    required = ["Time_Ma", "Coverage_Percent", "N_land_cells", "Radius_deg",
                "P_cap_pct", "P_hull_fixed_pct", "P_hull_best_found_pct", "P_land_MC_pct",
                "Original_cap_center_lat", "Original_cap_center_lon",
                "Search_best_center_lat", "Search_best_center_lon"]
    missing = sorted(set(required) - set(saved.columns))
    if missing:
        raise ValueError("The results CSV is missing columns: " + ", ".join(missing) +
                         ". Use the output from compare_land_hulls_center_search.py.")
    for name in required:
        saved[name] = pd.to_numeric(saved[name], errors="raise")
    always_finite = [name for name in required
                     if name not in ("Search_best_center_lat", "Search_best_center_lon")]
    if not np.isfinite(saved[always_finite].to_numpy(dtype=float)).all():
        raise ValueError("The saved comparison contains invalid numeric values.")

    files = [path for path in root.rglob("*")
             if path.is_file() and path.suffix.lower() == ".csv"
             and time_of(path) is not None]
    selected_files = {}
    for time_ma in args.times:
        candidates = [path for path in files if abs(time_of(path) - time_ma) < 0.5]
        if len(candidates) != 1:
            raise ValueError(f"Expected exactly one raw MapX_YMa.csv for {time_ma:g} Mya "
                             f"inside {root}; found {len(candidates)}: {candidates}")
        selected_files[time_ma] = candidates[0]
        matching_row(saved, time_ma, 90)
        matching_row(saved, time_ma, 100)
    destination = (args.output_dir.expanduser().resolve()
                   if args.output_dir is not None else root / OUTPUT_FOLDER)
    destination.mkdir(parents=True, exist_ok=True)
    print("Raw-map folder:", root, flush=True)
    print("Saved probabilities:", source, flush=True)
    print("Output folder:", destination, flush=True)
    print("Times (Mya):", args.times, flush=True)
    print("Reusing fitted caps and saved selection centers; no cap or center search.", flush=True)
    if args.example_mode == "maximum":
        print(f"Checking all {args.max_directions:,} sampled directions per date "
              "to find the highest land coverage. Saved probabilities are reused.", flush=True)
    else:
        print("Selecting illustrative gap examples. Saved probabilities are reused.", flush=True)

    rng = np.random.default_rng(RANDOM_SEED)
    directions = rng.normal(size=(args.max_directions, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    summary = []
    for time_ma in args.times:
        print(f"\n{time_ma:g} Mya: reading {selected_files[time_ma].name}", flush=True)
        land = read_land(selected_files[time_ma])
        epoch = prepare_epoch(land, matching_row(saved, time_ma, 90),
                               matching_row(saved, time_ma, 100))
        examples = find_examples(epoch, directions, args.examples, mode=args.example_mode)
        for number, example in enumerate(examples, 1):
            suffix = "" if args.example_mode == "maximum" else "_gap"
            if number > 1:
                suffix += f"_example{number}"
            path = destination / f"comparison_{time_ma:g}Ma{suffix}.png"
            summary.append(render_comparison(epoch, example, time_ma, path, number))
            percentage = 100 * example["lit_count"] / len(land)
            print(f"  Example {number}: {percentage:.3f}% of land lit; "
                  f"{example['missed_best']} best-group cells and "
                  f"{example['missed_original']} original-group cells left dark.", flush=True)
            print("  Updated:", path, flush=True)
    summary_name = ("example_directions.csv" if args.example_mode == "maximum"
                    else "example_directions_gap.csv")
    summary_path = destination / summary_name
    pd.DataFrame(summary).to_csv(summary_path, index=False)
    print("\nSaved reproducible example directions:", summary_path, flush=True)
    print(f"Done: {len(summary)} comparison figures.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as error:
        raise SystemExit("ERROR: " + str(error))
