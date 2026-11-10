#!/usr/bin/env python3
# cite as: M. Sade, A. Tsarfati and O. Birnholtz (2026), "Gamma Ray Bursts Effects on
# Extinction and Survivability in the Galaxy", arXiv:2603.18250
"""compare the saved cap and sampled probabilities with spherical hulls of the land cells.

for 90% the hull belongs to a fixed group of ceil(0.9*N) cells, the ones nearest a
selection center: the saved cap center, or the best center found by a finite search over
the whole sphere (ocean included). for 100% it is the hull of all land cells. a hull
probability is a lower bound for the chance of covering any 90% of the land.

for a spherical convex hull K inside an open hemisphere, the probability that a uniformly
random hemisphere contains K is (2*pi - perimeter(K))/(4*pi), with the perimeter taken
from great-circle edge lengths. reference: Nagy and Vigh, Random Spherical Disc-Polygons
and a Duality, Lemma 4(ii) with r=pi/2,
https://www.math.u-szeged.hu/~vigvik/preprints/spherical_duality.pdf

place the script in PaleoDEMS, next to the time slices, SurfaceCoverage_90PercentLand.csv
and LandHitProbability.csv (probabilities there are fractions), and run

    python compare_land_hulls_center_search.py

the inputs are only read; results go to HullCenterSearch/. with --self-test only the
geometry checks run.
"""

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.spatial import ConvexHull


# must match the number of directions used for LandHitProbability.csv
N_DIRECTIONS = 200000
INPUT_FOLDER = "."
CAPS_CSV = "SurfaceCoverage_90PercentLand.csv"
MC_CSV = "LandHitProbability.csv"
OUTPUT_FOLDER = "HullCenterSearch"
TOP_N = 10

# finite search for better fixed 90% groups; raise these for a more thorough search
SEARCH_GLOBAL_CENTERS = 768
SEARCH_LOCAL_STARTS = 2
SEARCH_LOCAL_ANGLES_DEG = (8, 4, 2, 1, 0.5)
SEARCH_LOCAL_DIRECTIONS = 24

# floating-point tolerances
CAP_ROUNDING_TOL_DEG = 1e-6
OPEN_HEMISPHERE_TOL = 1e-10


def unit_vectors(lat_deg, lon_deg):
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    return np.column_stack((np.cos(lat) * np.cos(lon),
                            np.cos(lat) * np.sin(lon), np.sin(lat)))


def time_of(path):
    import re
    # the Ma suffix is required, so the generated top10_..._90.csv tables are skipped
    match = re.search(r"(?:^|_)(-?\d+(?:\.\d+)?)Ma$", path.stem, re.IGNORECASE)
    return float(match.group(1)) if match else None


def read_numeric_csv(path, columns):
    if not path.is_file():
        raise ValueError("File not found: {}".format(path))
    table = pd.read_csv(path)
    table.columns = table.columns.str.strip()
    missing = sorted(set(columns) - set(table.columns))
    if missing:
        raise ValueError("{} is missing columns {}. Found: {}".format(
            path, missing, list(table.columns)))
    for col in columns:
        table[col] = pd.to_numeric(table[col], errors="raise")
        if not np.isfinite(table[col]).all():
            raise ValueError("Non-finite values in {}: {}".format(path, col))
    return table


def read_land(path):
    # same reading and land criterion as land_hit_all.py
    table = pd.read_csv(path, skiprows=1, names=["lon", "lat", "elev"])
    table = table.apply(pd.to_numeric, errors="coerce").dropna()
    table = table[table["elev"] > 0]
    if table.empty:
        raise ValueError("No positive-elevation land cells in {}".format(path))
    return unit_vectors(table["lat"].values, table["lon"].values)


def cap_probability(radius_deg):
    if not 0 <= radius_deg <= 180:
        raise ValueError("Invalid cap radius: {}".format(radius_deg))
    return (1 - np.sin(np.radians(radius_deg))) / 2 if radius_deg < 90 else 0.0


def enclosing_hemisphere(points, hint=None):
    """return a center with all dot products positive, or None if there is none.

    linear program: maximize t subject to points @ v >= t and -1 <= v_i <= 1. a positive
    optimum exists exactly when an open hemisphere contains the points.
    """
    if hint is not None:
        hint = np.asarray(hint, dtype=float)
        hint = hint / np.linalg.norm(hint)
        margin = float(np.min(points @ hint))
        if margin > OPEN_HEMISPHERE_TOL:
            return hint, margin
    objective = np.array([0.0, 0.0, 0.0, -1.0])
    constraints = np.column_stack((-points, np.ones(len(points))))
    result = linprog(objective, A_ub=constraints, b_ub=np.zeros(len(points)),
                     bounds=[(-1, 1)] * 3 + [(0, 1)], method="highs",
                     options={"primal_feasibility_tolerance": 1e-10,
                              "dual_feasibility_tolerance": 1e-10})
    if not result.success:
        raise ValueError("Hemisphere feasibility solver failed: " + result.message)
    if result.x[3] <= OPEN_HEMISPHERE_TOL:
        return None, float(result.x[3])
    center = result.x[:3] / np.linalg.norm(result.x[:3])
    margin = float(np.min(points @ center))
    if margin <= OPEN_HEMISPHERE_TOL:
        raise ValueError("Hemisphere solution is too close to the numerical boundary.")
    return center, margin


def spherical_hull(points, hint=None):
    """containment probability, angular perimeter and spherical area of the hull."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError("Expected a nonempty array of unit vectors, shape (N, 3).")
    if not np.isfinite(points).all() or np.any(np.linalg.norm(points, axis=1) == 0):
        raise ValueError("Invalid land vectors.")
    points = points / np.linalg.norm(points, axis=1)[:, None]
    center, margin = enclosing_hemisphere(points, hint)
    if center is None:
        # no hull in an open hemisphere, so no perimeter or area
        return dict(probability=0.0, perimeter=np.nan, area=np.nan,
                    vertices=0, margin=margin,
                    status="no_open_hemisphere_at_numerical_tolerance")

    reference = np.eye(3)[np.argmin(np.abs(center))]
    east = np.cross(reference, center)
    east /= np.linalg.norm(east)
    north = np.cross(center, east)
    denominator = points @ center
    # gnomonic projection: great-circle edges become straight lines
    xy = np.column_stack((points @ east, points @ north)) / denominator[:, None]
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
        raise ValueError("Invalid hull probability; inspect spherical geometry.")
    probability = float(np.clip(probability, 0, 0.5))
    area = 0.0
    if len(vertices) >= 3:
        anchor = vertices.sum(axis=0)
        anchor /= np.linalg.norm(anchor)  # inside the hull
        determinants = np.abs(np.cross(vertices, following) @ anchor)
        denominators = (1 + vertices @ anchor + following @ anchor
                        + np.sum(vertices * following, axis=1))
        area = float(np.sum(2 * np.arctan2(determinants, denominators)))
    return dict(probability=probability, perimeter=perimeter, area=area,
                vertices=len(vertices), margin=margin, status="open_hemisphere")


def mc_interval(probability, n_directions):
    """Wilson 95% interval; zero hits still give a nonzero upper limit."""
    hits = int(round(probability * n_directions))
    if abs(probability * n_directions - hits) > 1e-5:
        raise ValueError("MC probability does not match an integer count for "
                         "N_DIRECTIONS={}. Check that setting and use the original, "
                         "unrounded LandHitProbability.csv.".format(n_directions))
    z = 1.959963984540054
    divisor = 1 + z * z / n_directions
    midpoint = (probability + z * z / (2 * n_directions)) / divisor
    half_width = z * np.sqrt(probability * (1 - probability) / n_directions
                            + z * z / (4 * n_directions ** 2)) / divisor
    return (max(0.0, midpoint - half_width), min(1.0, midpoint + half_width), hits)


def matching_row(table, time_ma, fraction=None):
    mask = np.abs(table["Time_Ma"] - time_ma) < 0.5
    if fraction is not None:
        mask &= np.isclose(table["Coverage_Fraction"], fraction)
    rows = table.loc[mask]
    if len(rows) != 1:
        raise ValueError("Expected exactly one cap row for {} Mya, fraction {}; "
                         "found {}.".format(time_ma, fraction, len(rows)))
    return rows.iloc[0]


def independent_center_search(land, k, cap_center, baseline_probability):
    """search other fixed k-cell groups for a larger hull probability.

    candidate centers cover the whole sphere, with no land mask. each candidate takes its
    nearest k land cells, and the hull probability of that group is evaluated. the group
    of the saved cap center is the baseline. the result is not a global optimum.
    """
    import hashlib

    center0 = np.asarray(cap_center, dtype=float)
    center0 /= np.linalg.norm(center0)
    best_probability = float(baseline_probability)
    best_center = center0.copy()
    tested = 0
    cache = {}
    seeds = []

    def key_for(indices):
        ordered = np.sort(indices).astype(np.int32)
        return hashlib.blake2b(ordered.tobytes(), digest_size=16).digest()

    original_indices = np.argsort(-(land @ center0), kind="stable")[:k]
    cache[key_for(original_indices)] = best_probability
    if best_probability > 0:
        seeds.append((best_probability, center0.copy()))

    def evaluate(center):
        nonlocal tested, best_probability, best_center
        tested += 1
        center = np.asarray(center, dtype=float)
        length = np.linalg.norm(center)
        if length < 1e-14:
            return None
        center = center / length
        dots = land @ center
        # the center must light at least k cells, so it illuminates its own group
        if np.count_nonzero(dots > OPEN_HEMISPHERE_TOL) < k:
            return None
        # stable sort keeps the choice reproducible when distances tie
        indices = np.argsort(-dots, kind="stable")[:k]
        key = key_for(indices)
        if key not in cache:
            cache[key] = spherical_hull(land[indices], hint=center)["probability"]
            seeds.append((cache[key], center.copy()))
        probability = cache[key]
        if probability > best_probability + 1e-12:
            best_probability = probability
            best_center = center.copy()
        return probability, center

    evaluate(center0)
    evaluate(land.mean(axis=0))
    # fibonacci points: deterministic, roughly uniform over the sphere
    i = np.arange(SEARCH_GLOBAL_CENTERS, dtype=float)
    z = 1 - 2 * (i + 0.5) / SEARCH_GLOBAL_CENTERS
    longitude = i * np.pi * (3 - np.sqrt(5))
    radius = np.sqrt(np.maximum(0, 1 - z * z))
    global_centers = np.column_stack((radius * np.cos(longitude),
                                      radius * np.sin(longitude), z))
    for center in global_centers:
        evaluate(center)

    # refine around the best few groups, not only the original one
    starts = sorted(seeds, key=lambda item: item[0], reverse=True)[:SEARCH_LOCAL_STARTS]
    for local_score, local_center in starts:
        for angle_deg in SEARCH_LOCAL_ANGLES_DEG:
            reference = np.eye(3)[np.argmin(np.abs(local_center))]
            east = np.cross(reference, local_center)
            east /= np.linalg.norm(east)
            north = np.cross(local_center, east)
            angle = np.radians(angle_deg)
            phase = np.linspace(0, 2 * np.pi, SEARCH_LOCAL_DIRECTIONS, endpoint=False)
            ring = (np.cos(angle) * local_center
                    + np.sin(angle) * (np.cos(phase)[:, None] * east
                                       + np.sin(phase)[:, None] * north))
            next_score, next_center = local_score, local_center
            for center in ring:
                candidate = evaluate(center)
                if candidate is not None and candidate[0] > next_score + 1e-12:
                    next_score, next_center = candidate
            local_score, local_center = next_score, next_center

    shift = np.degrees(np.arccos(np.clip(np.dot(center0, best_center), -1, 1)))
    return {
        "P_hull_best_found": best_probability,
        "P_hull_best_found_pct": 100 * best_probability,
        "Search_best_center_lat": float(np.degrees(np.arcsin(np.clip(best_center[2], -1, 1)))),
        "Search_best_center_lon": float(np.degrees(np.arctan2(best_center[1], best_center[0]))),
        "Search_center_shift_deg": float(shift),
        "Search_centers_tested": tested,
        "Search_unique_subsets": len(cache),
        "Search_status": "positive_fixed_group_found" if best_probability > 0 else "no_positive_group_found_in_finite_search",
    }


def comparison_row(land, time_ma, fraction, cap, mc, n_directions):
    name = str(int(round(100 * fraction)))
    center = unit_vectors([float(cap["Center_Lat"])],
                          [float(cap["Center_Lon"])])[0]
    k = int(np.ceil(fraction * len(land)))
    if fraction == 1.0:
        selected = land
    else:
        # fixed subset of exactly k cells (ties kept in file order)
        selected = land[np.argsort(-(land @ center), kind="stable")[:k]]
    required = float(np.degrees(np.arccos(np.clip(np.min(selected @ center), -1, 1))))
    radius = float(cap["Radius_deg"])
    if required > radius + CAP_ROUNDING_TOL_DEG:
        raise ValueError(
            "{} Mya, {}%: saved cap radius {:.9f} deg, but these {} selected cells "
            "need {:.9f} deg around the saved center. Check the PaleoDEMS files, "
            "cap CSV, cell criterion and time/fraction rows."
            .format(time_ma, name, radius, k, required))
    p_cap = cap_probability(radius)
    if not np.isclose(p_cap, float(mc["P_cap_" + name]), rtol=1e-8, atol=1e-10):
        raise ValueError("{} Mya, {}%: cap formula disagrees with the saved MC CSV. "
                         "Use the same cap CSV that created it.".format(time_ma, name))
    hull = spherical_hull(selected, hint=center)
    if p_cap > hull["probability"] + 1e-8:
        raise ValueError("Cap probability exceeds hull probability at {} Mya, {}%; "
                         "check the cap and data files.".format(time_ma, name))
    p_mc = float(mc["P_land_" + name])
    lo, hi, hits = mc_interval(p_mc, n_directions)
    p_hull = hull["probability"]
    return {
        "Time_Ma": time_ma, "Coverage_Percent": int(name),
        "N_land_cells": len(land), "N_fixed_subset_cells": k,
        "Radius_deg": radius, "Required_fixed_subset_radius_deg": required,
        "P_cap": p_cap, "P_hull_fixed": p_hull, "P_land_MC": p_mc,
        "P_cap_pct": 100 * p_cap, "P_hull_fixed_pct": 100 * p_hull,
        "P_land_MC_pct": 100 * p_mc,
        "MC_SE_pct": 100 * np.sqrt(p_mc * (1 - p_mc) / n_directions),
        "MC_CI95_low_pct": 100 * lo, "MC_CI95_high_pct": 100 * hi,
        "MC_successes": hits, "MC_directions": n_directions,
        "Gap_MC_minus_cap_pp": 100 * (p_mc - p_cap),
        "Gap_hull_minus_cap_pp": 100 * (p_hull - p_cap),
        "Gap_MC_minus_hull_pp": 100 * (p_mc - p_hull),
        "MC_to_cap_ratio": p_mc / p_cap if p_cap > 0 else np.nan,
        "Hull_perimeter_rad": hull["perimeter"],
        "Hull_area_sr": hull["area"],
        "Hull_area_fraction": hull["area"] / (4 * np.pi),
        "Hull_vertices": hull["vertices"], "Hemisphere_margin": hull["margin"],
        "Hull_status": hull["status"],
        "Hull_probability_inside_MC_CI95": lo <= p_hull <= hi,
    }


def write_outputs(result, destination, top_n):
    destination.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination / "HullComparison_all.csv", index=False)
    overlaps = []
    display = ["Time_Ma", "Radius_deg", "P_cap_pct", "P_hull_fixed_pct",
               "P_land_MC_pct", "MC_CI95_low_pct", "MC_CI95_high_pct",
               "Gap_MC_minus_cap_pp", "Gap_MC_minus_hull_pp"]
    notes = [
        "cap, spherical hull and direct land comparison",
        "p columns are fractions unless the name ends in _pct; _pp columns are "
        "percentage-point differences.",
        "90% hull: the fixed group of the ceil(0.9*N) cells nearest the saved cap center. "
        "It is not the probability of covering any 90%.",
        "100% hull: the same event as direct sampling of all land cells.",
        "the hull probability uses the spherical perimeter, not the hull area.",
        "best-found 90% group: finite search over selection centers on the whole sphere, "
        "not restricted to land. Each center takes its nearest ceil(0.9*N) cells, and the "
        "group with the largest hull probability is kept. It is not a global optimum, and "
        "it is still a lower bound for covering any 90%.",
        "Search_best_center_lat/lon are selection centers, not hull centroids; "
        "Search_center_shift_deg is the distance from the saved cap center.",
        "there is no search for 100%, since the hull of all land does not depend on a center.",
        "zero MC hits do not prove a zero probability. Intervals are pointwise Wilson "
        "95% intervals.",
        "rows with no open hemisphere have no hull area or perimeter.",
        "top lists only contain epochs with positive probability.",
    ]
    for coverage in (90, 100):
        table = result[result["Coverage_Percent"] == coverage].copy()
        ranks = {
            "cap_clustering": table.sort_values(["Radius_deg", "Time_Ma"]),
            "hull_area": table.dropna(subset=["Hull_area_sr"]).sort_values(
                ["Hull_area_sr", "Time_Ma"]),
            "hull_probability": table[table["P_hull_fixed"] > 0].sort_values(
                ["P_hull_fixed", "Time_Ma"], ascending=[False, True]),
            "land_probability": table[table["P_land_MC"] > 0].sort_values(
                ["P_land_MC", "Time_Ma"], ascending=[False, True]),
            "largest_gap": table[table["Gap_MC_minus_cap_pp"] > 0].sort_values(
                ["Gap_MC_minus_cap_pp", "Time_Ma"], ascending=[False, True]),
        }
        if coverage == 90:
            ranks["best_group_probability"] = table[table["P_hull_best_found"] > 0].sort_values(
                ["P_hull_best_found", "Time_Ma"], ascending=[False, True])
        lists = {name: ranked.head(top_n) for name, ranked in ranks.items()}
        for name, ranked in lists.items():
            ranked.to_csv(destination / ("top{}_{}_{}.csv".format(top_n, name, coverage)),
                          index=False)
        for (name_a, rows_a), (name_b, rows_b) in combinations(lists.items(), 2):
            times_a, times_b = set(rows_a["Time_Ma"]), set(rows_b["Time_Ma"])
            shared = sorted(times_a & times_b)
            overlaps.append(dict(Coverage_Percent=coverage, List_A=name_a, List_B=name_b,
                                 N_A=len(times_a), N_B=len(times_b), N_shared=len(shared),
                                 Shared_times_Ma="; ".join("{:g}".format(t) for t in shared)))
        print("\n{}%: top {} epochs by direct land probability".format(coverage, top_n))
        print(lists["land_probability"][display].to_string(index=False, float_format="%.5f"))
        outside = int((~table["Hull_probability_inside_MC_CI95"]).sum())
        notes += ["", "{}%: {} epochs, {} with nonzero MC hits.".format(
            coverage, len(table), int((table["MC_successes"] > 0).sum()))]
        if coverage == 100:
            notes.append("100% hull outside the pointwise MC 95% interval: {} / {}.".format(
                outside, len(table)))
        else:
            notes.append("90%: the sampled probability also counts other subsets, so it can "
                         "exceed the hull probability of one fixed group.")
    pd.DataFrame(overlaps).to_csv(destination / "Top10_overlap.csv", index=False)
    (destination / "READ_ME.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")


def plot_comparisons(result, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)

    for coverage in (90, 100):
        t = result[
            result["Coverage_Percent"] == coverage
        ].sort_values("Time_Ma")

        x = t["Time_Ma"].to_numpy()

        if coverage == 90:
            hull_label = (
                "Every cell in one fixed 90% group (spherical hull)"
            )
            land_label = (
                "Any group covering at least 90% of land cells (Monte Carlo)"
            )
            title = (
                "90% coverage: original group, independent group search and direct sampling"
            )
            note = (
                "Green: group from the original cap. Orange: best group found among tested centers, "
                "including local refinement. Both remain fixed as GRB direction changes."
            )
        else:
            hull_label = "All land cells (spherical hull calculation)"
            land_label = "All land cells (Monte Carlo estimate)"
            title = (
                "100% land coverage: hull calculation versus Monte Carlo"
            )
            note = (
                "The hull calculation and Monte Carlo estimate concern the "
                "same land cells. The cap formula also requires surrounding "
                "ocean to be lit."
            )

        fig, axes = plt.subplots(
            2, 1,
            figsize=(12, 8),
            sharex=True,
            gridspec_kw={"height_ratios": [2.2, 1]},
        )

        # upper panel: probabilities
        axes[0].plot(
            x, t["P_cap_pct"],
            "--",
            color="0.4",
            linewidth=1.7,
            label=(
                "Entire enclosing circular cap, including ocean (formula)"
            ),
        )

        axes[0].plot(
            x, t["P_hull_fixed_pct"],
            color="seagreen",
            linewidth=1.8,
            label=hull_label,
        )

        axes[0].plot(
            x, t["P_land_MC_pct"],
            ".",
            color="royalblue",
            markersize=4,
            label=land_label,
        )

        if coverage == 90:
            axes[0].plot(
                x, t["P_hull_best_found_pct"], color="darkorange", linewidth=1.8,
                label="Best fixed 90% group found (independent center search)",
            )

        axes[0].fill_between(
            x,
            t["MC_CI95_low_pct"].to_numpy(),
            t["MC_CI95_high_pct"].to_numpy(),
            color="royalblue",
            alpha=0.16,
            label="Monte Carlo 95% sampling interval",
        )

        axes[0].set_ylabel("Probability per GRB (%)")
        axes[0].tick_params(axis="x", labelbottom=True)
        axes[0].set_xlabel("Time before present (Mya)")
        axes[0].set_title(title, fontsize=13, pad=12)
        axes[0].legend(
            loc="upper right",
            fontsize=9,
            framealpha=0.96,
        )

        # lower panel: differences in percentage points
        axes[1].plot(
            x, t["Gap_MC_minus_cap_pp"],
            color="0.4",
            linewidth=1.6,
            label="Direct land estimate minus cap formula",
        )

        axes[1].plot(
            x, t["Gap_MC_minus_hull_pp"],
            color="seagreen",
            linewidth=1.6,
            label="Direct land estimate minus hull calculation",
        )

        if coverage == 90:
            axes[1].plot(
                x, t["Gap_MC_minus_best_hull_pp"], color="darkorange", linewidth=1.6,
                label="Direct land estimate minus best-found group calculation",
            )
        axes[1].axhline(0, color="black", linewidth=0.7)
        axes[1].set_ylabel("Difference\n(percentage points)")
        axes[1].set_xlabel("Time before present (Mya)")
        axes[1].set_title(
            "Differences: above zero means the direct estimate is higher",
            fontsize=10,
            pad=8,
        )
        axes[1].legend(
            loc="upper right",
            fontsize=9,
            framealpha=0.96,
        )

        if x.max() > x.min():
            axes[1].set_xlim(x.max(), x.min())

        for ax in axes:
            ax.grid(alpha=0.2)
            ax.tick_params(labelsize=10)

        fig.text(0.075, 0.035, note, fontsize=9)
        fig.tight_layout(rect=(0, 0.065, 1, 1))

        output_path = destination / f"comparison_{coverage}.png"
        fig.savefig(output_path, dpi=220)
        print(f"Updated: {output_path.resolve()}", flush=True)
        plt.close(fig)


def geometry_checks():
    """known-geometry checks; no data files needed."""
    triangle = spherical_hull(np.eye(3))
    np.testing.assert_allclose(triangle["probability"], 1 / 8, atol=1e-12)
    np.testing.assert_allclose(triangle["area"], np.pi / 2, atol=1e-12)
    angle = np.radians(120)
    pair = np.array([[1, 0, 0], [np.cos(angle), np.sin(angle), 0]])
    np.testing.assert_allclose(spherical_hull(pair)["probability"], 1 / 6, atol=1e-12)
    tetrahedron = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])
    assert spherical_hull(tetrahedron)["probability"] == 0
    assert spherical_hull(np.array([[1, 0, 0], [-1, 0, 0]]))["probability"] == 0
    assert spherical_hull(np.array([[1, 0, 0]]))["probability"] == 0.5
    phi = np.linspace(0, 2 * np.pi, 1440, endpoint=False)
    radius = np.radians(60)
    ring = np.column_stack((np.sin(radius) * np.cos(phi),
                            np.sin(radius) * np.sin(phi), np.full(len(phi), np.cos(radius))))
    np.testing.assert_allclose(spherical_hull(ring)["probability"], cap_probability(60), atol=2e-7)
    q, _ = np.linalg.qr(np.random.default_rng(4).normal(size=(3, 3)))
    np.testing.assert_allclose(spherical_hull(np.eye(3) @ q)["probability"], 1 / 8, atol=1e-12)
    lo, hi, hits = mc_interval(0, 200000)
    assert lo == 0 and hi > 0 and hits == 0
    print("Geometry checks passed: octant, rotation, point, pair, antipodes, "
          "tetrahedron, circular-cap limit and zero-hit interval.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--n-directions", type=int, default=N_DIRECTIONS)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    if args.plot_only:
        destination = args.base_dir.resolve() / OUTPUT_FOLDER
        saved_csv = destination / "HullComparison_all.csv"
        if not saved_csv.is_file():
            raise ValueError(f"Saved results not found: {saved_csv}")
        plot_comparisons(pd.read_csv(saved_csv), destination)
        return

    geometry_checks()
    if args.self_test:
        return
    if args.n_directions <= 0:
        raise ValueError("N_DIRECTIONS must be positive.")
    root = args.base_dir.resolve()
    caps = read_numeric_csv(root / CAPS_CSV, ["Time_Ma", "Coverage_Fraction",
                                           "Center_Lat", "Center_Lon", "Radius_deg"])
    mc = read_numeric_csv(root / MC_CSV, ["Time_Ma", "P_land_90", "P_land_100", "P_cap_90", "P_cap_100"])
    if mc.empty or mc["Time_Ma"].duplicated().any():
        raise ValueError("MC CSV must contain one row per time step.")
    if ((mc[["P_land_90", "P_land_100", "P_cap_90", "P_cap_100"]] < 0).any().any()
            or (mc[["P_land_90", "P_land_100", "P_cap_90", "P_cap_100"]] > 1).any().any()):
        raise ValueError("Input probabilities must be fractions between 0 and 1, not percentages.")
    if (mc["P_land_100"] > mc["P_land_90"] + 1e-12).any():
        raise ValueError("100% MC probability exceeds 90% MC probability.")
    input_dir = (root / INPUT_FOLDER).resolve()
    csv_files = sorted(
        path for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".csv"
    )
    files = [
        (time_of(path), path)
        for path in csv_files
        if time_of(path) is not None
    ]

    print("Searching folder:", input_dir, flush=True)
    print("CSV files found:", len(csv_files), flush=True)
    print("Recognized time-slice files:", len(files), flush=True)
    print(
        "Recognized times:",
        sorted(set(t for t, path in files)),
        flush=True,
    )
    print(
        "First five CSV filenames:",
        [str(path.relative_to(input_dir)) for path in csv_files[:5]],
        flush=True,
    )
    results = []
    mc = mc.sort_values("Time_Ma")
    print("Reading {} saved time steps; no Monte Carlo resampling. "
          "Independent selection-center search enabled.".format(len(mc)))
    for index, (_, saved) in enumerate(mc.iterrows(), 1):
        time_ma = float(saved["Time_Ma"])
        paths = [path for file_time, path in files if abs(file_time - time_ma) < 0.5]
        if len(paths) != 1:
            raise ValueError("Expected exactly one PaleoDEMS time-slice file for {} Mya; "
                             "found {}: {}".format(time_ma, len(paths), paths))
        land = read_land(paths[0])
        descriptions = []
        for fraction in (0.9, 1.0):
            cap = matching_row(caps, time_ma, fraction)
            row = comparison_row(land, time_ma, fraction, cap, saved, args.n_directions)
            row["Original_cap_center_lat"] = float(cap["Center_Lat"])
            row["Original_cap_center_lon"] = float(cap["Center_Lon"])
            if fraction == 0.9:
                old_center = unit_vectors([float(cap["Center_Lat"])],
                                          [float(cap["Center_Lon"])])[0]
                searched = independent_center_search(
                    land, int(np.ceil(0.9 * len(land))), old_center, row["P_hull_fixed"])
                row.update(searched)
            else:
                # the hull of all land cells is fixed, so there is nothing to search
                row.update({"P_hull_best_found": row["P_hull_fixed"],
                            "P_hull_best_found_pct": row["P_hull_fixed_pct"],
                            "Search_best_center_lat": np.nan,
                            "Search_best_center_lon": np.nan,
                            "Search_center_shift_deg": np.nan,
                            "Search_centers_tested": 0,
                            "Search_unique_subsets": 0,
                            "Search_status": "not_applicable_all_cells_fixed"})
            row["Gap_MC_minus_best_hull_pp"] = 100 * (row["P_land_MC"] - row["P_hull_best_found"])
            row["Gain_best_hull_minus_original_hull_pp"] = 100 * (row["P_hull_best_found"] - row["P_hull_fixed"])
            results.append(row)
            descriptions.append("{}% cap {:.4f}%, hull {:.4f}%, MC {:.4f}%".format(
                int(100 * fraction), row["P_cap_pct"], row["P_hull_fixed_pct"], row["P_land_MC_pct"]))
        descriptions.append("best 90% hull {:.4f}% (center shift {:.2f} deg)".format(
            results[-2]["P_hull_best_found_pct"], results[-2]["Search_center_shift_deg"]))
        print("[{}/{}] {:g} Mya: {}".format(index, len(mc), time_ma, " | ".join(descriptions)), flush=True)
    result = pd.DataFrame(results).sort_values(["Time_Ma", "Coverage_Percent"])
    destination = root / OUTPUT_FOLDER
    write_outputs(result, destination, TOP_N)
    if not args.no_plots:
        plot_comparisons(result, destination)
    print("\nSaved comparison tables{} to {}".format(" and plots" if not args.no_plots else "", destination))
    print("Start with HullComparison_all.csv and comparison_90.png.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as error:
        raise SystemExit("ERROR: " + str(error))
