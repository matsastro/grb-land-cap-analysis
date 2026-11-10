#!/usr/bin/env python3
# cite as: M. Sade, A. Tsarfati and O. Birnholtz (2026), "Gamma Ray Bursts Effects on
# Extinction and Survivability in the Galaxy", arXiv:2603.18250
"""land coverage by a randomly oriented GRB hemisphere.

requires compare_coverage_maps.py in the same folder.
land fractions count grid cells equally.

all epochs, 4K frames and both MP4 layouts:
    python plot_coverage_regions.py --all-times --video --mp4 --frame-ms 750

row-layout movie from existing frames:
    python plot_coverage_regions.py --row-mp4-only --frame-ms 750

caption, probability definitions and references are written to
coverage_regions_notes.md.
"""

import argparse
import base64
import hashlib
import html
from io import BytesIO
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, QhullError, cKDTree

try:
    import compare_coverage_maps as geometry
except ModuleNotFoundError as error:
    if error.name == "compare_coverage_maps":
        raise SystemExit("Put compare_coverage_maps.py beside this script.") from error
    raise

CITATION = ('M. Sade, A. Tsarfati & O. Birnholtz (2026), '
            'Gamma Ray Bursts Effects on Extinction and Survivability in the Galaxy, '
            'arXiv:2603.18250')
CACHE_VERSION = 1
TOL = 1e-10
LAND_MARKER_SIZE = 0.6  # points squared
FONT_SCALE = 2.0
LIGHT_PURPLE = "#bd7ac8"
DARK_PURPLE = "#641579"
RING_COLOR = "#64646d"
OUTPUT_SIZE = (3840, 2160)
FIGURE_SIZE = (24.0, 13.5)
FIGURE_DPI = 160
VIDEO_SIZE = OUTPUT_SIZE
FRAME_MS = 750


def write_output_bytes(path, payload):
    """write atomically and return the saved path, including any fallback filename.

    close the temporary file before os.replace to avoid locked-file errors.
    """
    path = Path(path)
    redirected = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent, prefix=path.stem+"_new_", suffix=path.suffix)
    except OSError as error:
        redirected = error
        recovery = Path(tempfile.gettempdir())/"GRBCoverageRegions_Recovery"
        recovery.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=recovery, prefix=path.stem+"_new_", suffix=path.suffix)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        try:
            temporary.unlink()
        except OSError:
            pass
        raise
    if redirected is not None:
        print(f"  Could not write in {path.parent}: {redirected}\n"
              f"  Saved complete output instead: {temporary}", flush=True)
        return temporary
    try:
        os.replace(temporary, path)
    except OSError as error:
        print(f"  Could not replace {path}: {error}\n"
              f"  Saved complete output instead: {temporary}", flush=True)
        return temporary
    return path


def save_figure(fig, path):
    try:
        with BytesIO() as buffer:
            with plt.rc_context({"savefig.bbox": None}):
                fig.savefig(buffer, format="png", dpi=FIGURE_DPI, facecolor="white")
            payload = buffer.getvalue()
            size = (int.from_bytes(payload[16:20], "big"),
                    int.from_bytes(payload[20:24], "big"))
            if size != OUTPUT_SIZE:
                raise ValueError(f"Rendered image is {size}, expected {OUTPUT_SIZE}: {path}")
            return write_output_bytes(path, payload)
    finally:
        plt.close(fig)


def raw_time(path):
    import re
    found = re.fullmatch(r"Map.*_(-?\d+(?:\.\d+)?)Ma", path.stem, re.I)
    return float(found.group(1)) if found else None


def count_illuminated(land, directions, chunk_size=256, progress=False):
    counts = np.empty(len(directions), dtype=np.int32)
    report_at = 0
    for start in range(0, len(directions), chunk_size):
        stop = min(start + chunk_size, len(directions))
        counts[start:stop] = (directions[start:stop] @ land.T > 0).sum(axis=1)
        if progress and (stop >= report_at or stop == len(directions)):
            print(f"  Counted {stop:,}/{len(directions):,} directions", flush=True)
            report_at = stop + 20_000
    return counts


def cached_counts(land, directions, seed, path, chunk_size, recount):
    signature = json.dumps({
        "version": CACHE_VERSION, "directions": len(directions), "seed": seed,
        "rule": "positive elevation; equal cells; dot > 0",
        "land_sha256": hashlib.sha256(np.ascontiguousarray(land).tobytes()).hexdigest(),
    }, sort_keys=True)
    if path.is_file() and not recount:
        with np.load(path, allow_pickle=False) as old:
            if str(old["signature"].item()) == signature:
                counts = old["counts"]
                if (counts.shape != (len(directions),) or
                        np.any((counts < 0) | (counts > len(land)))):
                    raise ValueError("Invalid cached counts: " + str(path))
                print("  Reusing cached direction counts", flush=True)
                return counts
    counts = count_illuminated(land, directions, chunk_size, progress=True)
    np.savez_compressed(path, signature=np.array(signature), counts=counts)
    return counts


def wilson_interval(successes, total):
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z*z / total
    center = (p + z*z / (2*total)) / denominator
    half = z * np.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return max(0.0, center-half), min(1.0, center+half)


def containing_center(points, hint):
    hint = np.asarray(hint, dtype=float)
    hint /= np.linalg.norm(hint)
    if np.min(points @ hint) > TOL:
        return hint
    result = linprog([0, 0, 0, -1],
                     A_ub=np.column_stack((-points, np.ones(len(points)))),
                     b_ub=np.zeros(len(points)), bounds=[(-1, 1)]*3+[(0, 1)],
                     method="highs", options={"primal_feasibility_tolerance": 1e-10,
                                             "dual_feasibility_tolerance": 1e-10})
    if not result.success:
        raise ValueError("Containing-hemisphere solver failed: " + result.message)
    if result.x[3] <= TOL:
        return None
    center = result.x[:3] / np.linalg.norm(result.x[:3])
    if np.min(points @ center) <= TOL:
        raise ValueError("Containing hemisphere is too close to the numerical tolerance.")
    return center


def all_land_hull(land, hint):
    hull = geometry.spherical_hull(land, hint=hint)
    hull["whole_sphere"] = False
    if hull["status"] == "open_hemisphere":
        hull["center"] = containing_center(land, hint)
        if hull["center"] is None:
            raise ValueError("Hull and hemisphere tests disagree.")
    else:
        hull["center"] = None
        # no containing hemisphere exists when the origin is inside the 3D hull.
        try:
            ordinary = ConvexHull(land)
            hull["whole_sphere"] = bool(np.max(ordinary.equations[:, 3]) < -TOL)
        except QhullError:
            # leave degenerate hull areas undefined.
            pass
        # a whole-sphere hull can still have a cap smaller than the whole sphere.
        hull["area"] = 4*np.pi if hull["whole_sphere"] else np.nan
    return hull


def nearest_segment_point(starts, ends):
    """closest points to the origin on the supplied 3D segments."""
    delta = ends - starts
    squared = np.einsum("ij,ij->i", delta, delta)
    fraction = np.divide(-np.einsum("ij,ij->i", starts, delta), squared,
                         out=np.zeros_like(squared), where=squared > 0)
    points = starts + np.clip(fraction, 0, 1)[:, None]*delta
    return points[np.argmin(np.einsum("ij,ij->i", points, points))]


def minimum_enclosing_cap(points):
    """smallest all-point spherical cap, allowing a continuous center.

    for the ordinary 3D hull P, a closest point q gives center q/|q|
    when the origin is outside P. when it is inside, use the opposite
    outward normal of the nearest facet. check the radius against all points.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError("A cap needs a nonempty (N, 3) array.")
    lengths = np.linalg.norm(points, axis=1)
    if not np.isfinite(points).all() or np.any(lengths == 0):
        raise ValueError("Invalid vectors in minimal-cap calculation.")
    points = points/lengths[:, None]
    unique = np.unique(points, axis=0)
    mean = unique.mean(axis=0)
    _, singular, axes = np.linalg.svd(unique-mean, full_matrices=False)
    rank = int(np.sum(singular > 1e-12*max(float(singular[0]), 1.0)))
    expected_tau = None

    if rank == 0:
        center = unique[0]
        method = "point"
    elif rank == 1:
        coordinate = (unique-mean) @ axes[0]
        first, last = unique[np.argmin(coordinate)], unique[np.argmax(coordinate)]
        closest = nearest_segment_point(first[None, :], last[None, :])
        distance = np.linalg.norm(closest)
        center = (closest/distance if distance > 1e-12 else
                  geometry.tangent_basis(first)[0])
        method = "segment"
    elif rank == 2:
        basis = axes[:2]
        xy = (unique-mean) @ basis.T
        planar = ConvexHull(xy)
        origin_xy = -mean @ basis.T
        projection_inside = np.all(
            planar.equations[:, :2] @ origin_xy + planar.equations[:, 2] <= 1e-12)
        if projection_inside:
            closest = mean + origin_xy @ basis
        else:
            boundary = unique[planar.vertices]
            closest = nearest_segment_point(boundary, np.roll(boundary, -1, axis=0))
        distance = np.linalg.norm(closest)
        center = closest/distance if distance > 1e-12 else axes[2]
        method = "planar_hull"
    else:
        ordinary = ConvexHull(unique)
        normal = ordinary.equations[:, :3].copy()
        norm = np.linalg.norm(normal, axis=1)
        normal /= norm[:, None]
        offset = ordinary.equations[:, 3]/norm
        if np.max(offset) <= 64*np.finfo(float).eps:
            index = int(np.argmax(offset))
            center = -normal[index]
            expected_tau = float(offset[index])
            method = "nearest_supporting_facet"
        else:
            triangle = unique[ordinary.simplices]
            next_vertex = np.roll(triangle, -1, axis=1)
            closest = nearest_segment_point(triangle.reshape(-1, 3),
                                            next_vertex.reshape(-1, 3))
            best_squared = float(closest @ closest)
            projection = -offset[:, None]*normal
            edge = next_vertex-triangle
            relative = projection[:, None, :]-triangle
            oriented = np.sign(np.einsum("ij,ij->i",
                np.cross(triangle[:, 1]-triangle[:, 0], triangle[:, 2]-triangle[:, 0]),
                normal))
            side = np.einsum("ijk,ik->ij", np.cross(edge, relative), normal)
            tolerance = (64*np.finfo(float).eps*np.linalg.norm(edge, axis=2)*
                         np.linalg.norm(relative, axis=2))
            inside = ((oriented != 0) &
                      np.all(oriented[:, None]*side >= -tolerance, axis=1))
            if np.any(inside):
                face_points = projection[inside]
                squared = np.einsum("ij,ij->i", face_points, face_points)
                index = int(np.argmin(squared))
                if squared[index] < best_squared:
                    closest = face_points[index]
            distance = np.linalg.norm(closest)
            if distance <= 1e-13:
                raise ValueError("Minimal-cap geometry is too close to the hemisphere boundary.")
            center = closest/distance
            expected_tau = float(distance)
            method = "nearest_point_on_3d_hull"

    center = center/np.linalg.norm(center)
    tau = float(np.min(points @ center))
    if expected_tau is not None and abs(tau-expected_tau) > 2e-9:
        raise ValueError("Minimal-cap solution failed its all-point support check.")
    radius = float(np.degrees(np.arccos(np.clip(tau, -1, 1))))
    return dict(center=center, radius_deg=radius,
                area_fraction=(1-np.clip(tau, -1, 1))/2,
                cos_radius=tau, method=method,
                color="green" if radius < 90-1e-9 else "red")


def sampled_union(successful_directions):
    """area of the finite union of hemisphere footprints, not p90.

    its complement is the intersection of the opposite hemispheres.
    for the spherical hull K of the footprint centers,
    area(union)/(4*pi) = 1 - (2*pi-L(K))/(4*pi).
    use extremal hull vertices for footprint membership tests.
    """
    if not len(successful_directions):
        return dict(area_fraction=0.0, vertices=np.empty((0, 3)), empty=True)
    mean = successful_directions.mean(axis=0)
    hull = geometry.spherical_hull(successful_directions, hint=mean)
    if hull["status"] == "open_hemisphere":
        vertices = hull["vertices"]
        entire_sphere = False
    else:
        # a zero-area complement does not affect the displayed area.
        vertices = np.empty((0, 3))
        entire_sphere = True
    return dict(area_fraction=1-hull["probability"], vertices=vertices,
                entire_sphere=entire_sphere, empty=False)


def union_field(points, union):
    """signed max-dot field for the sampled footprint union."""
    if union["empty"]:
        return np.full(len(points), -1.0)
    if union["entire_sphere"]:
        return np.ones(len(points))
    answer = np.empty(len(points))
    for start in range(0, len(points), 2048):
        answer[start:start+2048] = np.max(
            points[start:start+2048] @ union["vertices"].T, axis=1)
    return answer


def in_union(points, union):
    return union_field(points, union) > 0


def surface_grid(view, resolution=420):
    axis = np.linspace(-np.pi, np.pi, resolution)
    x, y = np.meshgrid(axis, axis)
    radius = np.hypot(x, y)
    center = geometry.unit_vectors([view[0]], [view[1]])[0]
    lon = np.radians(view[1])
    east = np.array([-np.sin(lon), np.cos(lon), 0.0])
    north = np.cross(center, east)
    scale = np.ones_like(radius)
    away = radius > 1e-12
    scale[away] = np.sin(radius[away]) / radius[away]
    points = (np.cos(radius)[..., None] * center + scale[..., None] *
              (x[..., None]*east+y[..., None]*north))
    inside = radius <= np.pi
    return points[inside], inside


def base_map(ax, land, view, title):
    ax.set_aspect("equal")
    ax.set_xlim(-3.27, 3.27)
    ax.set_ylim(-3.27, 3.27)
    ax.axis("off")
    disk = Circle((0, 0), np.pi, facecolor="lightblue", edgecolor="0.6", lw=0.6, zorder=0)
    ax.add_patch(disk)
    xy = geometry.aeqd_project_vectors(land, *view)
    ax.scatter(xy[:, 0], xy[:, 1], s=LAND_MARKER_SIZE, c="darkgreen", zorder=5,
               linewidths=0, edgecolors="none", rasterized=True)
    ax.set_title(title, fontsize=12 * FONT_SCALE, pad=10)
    return disk


def shade_mask(ax, disk, flat_mask, inside, color, outline=False, alpha=0.42, zorder=2):
    from matplotlib.colors import to_rgba
    mask = np.zeros(inside.shape, dtype=bool)
    mask[inside] = flat_mask
    rgba = np.zeros((*inside.shape, 4))
    rgba[mask] = to_rgba(color, alpha=alpha)
    image = ax.imshow(rgba, extent=(-np.pi, np.pi, -np.pi, np.pi),
                      origin="lower", interpolation="nearest", zorder=zorder)
    image.set_clip_path(disk)
    if outline and mask.any() and np.any(~flat_mask):
        grid = np.linspace(-np.pi, np.pi, len(mask))
        masked = np.ma.masked_where(~inside, mask.astype(float))
        ax.contour(grid, grid, masked, levels=[0.5], colors=[color],
                   linewidths=1.1, zorder=5)


def draw_union(ax, disk, field, inside, alpha=0.34):
    """contour the max-dot field rather than the raster mask."""
    if not np.any(field > 0):
        return
    raster = np.ma.masked_all(inside.shape)
    raster[inside] = field
    grid = np.linspace(-np.pi, np.pi, len(inside))
    ax.contourf(grid, grid, raster, levels=[0, 2], colors=[LIGHT_PURPLE],
                alpha=alpha, zorder=1)
    if np.any(field < 0):
        ax.contour(grid, grid, raster, levels=[0], colors=[LIGHT_PURPLE],
                   linewidths=1.3, zorder=6)


def plot_curve(ax, points, view, **style):
    geometry.plot_projected_curve(ax, points,
                                 {"view_lat": view[0], "view_lon": view[1]}, **style)


def hull_inside(points, hull):
    if hull["whole_sphere"]:
        return np.ones(len(points), dtype=bool)
    if hull["status"] != "open_hemisphere":
        return np.zeros(len(points), dtype=bool)
    vertices = hull["vertices"]
    if len(vertices) < 3:
        return np.zeros(len(points), dtype=bool)
    normals = np.cross(vertices, np.roll(vertices, -1, axis=0))
    # the sum of hull vertices lies inside and fixes edge orientation.
    interior = vertices.sum(axis=0)
    normals[(normals @ interior) < 0] *= -1
    answer = np.empty(len(points), dtype=bool)
    for start in range(0, len(points), 2048):
        answer[start:start+2048] = np.all(points[start:start+2048] @ normals.T >= -TOL,
                                         axis=1)
    return answer


def note(ax, text, color="black"):
    ax.text(0.5, -0.04, text, transform=ax.transAxes,
            ha="center", va="top", fontsize=10, linespacing=1.4, color=color)


def hull_message(hull):
    if hull["whole_sphere"]:
        return "No hemisphere contains all land"
    return "No all-land hull inside an open hemisphere"


def hull_display_color(hull):
    """color by planetary surface fraction, not enclosed land fraction."""
    if hull["whole_sphere"]:
        return "darkorange"
    if np.isfinite(hull["area"]):
        return "darkblue" if hull["area"] < 2*np.pi else "darkorange"
    # degenerate hulls with undefined area have no area color.
    return "dimgray"


def projected_enclosing_outline(land, view):
    """planar convex outline in the AEQD projection; used only for display.

    its edges need not follow great circles. keep its area and perimeter
    separate from the spherical probability calculation.
    """
    xy = geometry.aeqd_project_vectors(land, *view)
    if not np.isfinite(xy).all():
        raise ValueError("The projected outline crosses the map's antipode. "
                         "Choose a projection center away from the land.")
    unique = np.unique(xy, axis=0)
    planar = ConvexHull(unique)
    outline = unique[planar.vertices]

    if np.max(xy @ planar.equations[:, :2].T + planar.equations[:, 2]) > 1e-9:
        raise ValueError("Projected enclosing outline failed its land-point check.")
    return np.vstack((outline, outline[0]))


def projected_polygon_surface_area(outline):
    """spherical area enclosed by the straight edges of an AEQD-map polygon.

    on the unit sphere, the area element is sin(r) dr dphi. integrate
    (1-cos(r)) dphi around the boundary. for each edge a+t*d, use
    cross(a,d)*(1-cos(|a+t*d|))/|a+t*d|**2, 0 <= t <= 1.
    evaluate the quotient as 0.5*sinc(r/(2*pi))**2 near r=0.
    """
    points = np.asarray(outline, dtype=float)
    if len(points) > 1 and np.allclose(points[0], points[-1], atol=1e-14, rtol=0):
        points = points[:-1]
    if len(points) < 3:
        return 0.0
    if (points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all()
            or np.max(np.linalg.norm(points, axis=1)) > np.pi+1e-10):
        raise ValueError("Projected polygon must lie inside the full-globe AEQD disk.")
    edge = np.roll(points, -1, axis=0)-points
    cross = points[:, 0]*edge[:, 1]-points[:, 1]*edge[:, 0]

    def integrate(order):
        nodes, weights = np.polynomial.legendre.leggauss(order)
        fraction = (nodes+1)/2
        positions = points[:, None, :]+fraction[None, :, None]*edge[:, None, :]
        radius = np.linalg.norm(positions, axis=2)
        quotient = 0.5*np.sinc(radius/(2*np.pi))**2
        return float(abs(cross @ (quotient @ (weights/2))))

    coarse, area = integrate(16), integrate(32)
    if abs(area-coarse) > 1e-10:
        fine = integrate(64)
        if abs(fine-area) > 1e-9:
            raise ValueError("Projected-region surface-area quadrature did not converge.")
        area = fine
    if not 0 <= area <= 4*np.pi+1e-9:
        raise ValueError("Invalid projected-region surface area.")
    return float(np.clip(area, 0, 4*np.pi))


def draw_projected_outline(ax, disk, land, view, color, linewidth):
    outline = projected_enclosing_outline(land, view)
    line, = ax.plot(outline[:, 0], outline[:, 1], c=color,
                    lw=linewidth, zorder=8)
    line.set_clip_path(disk)
    return projected_polygon_surface_area(outline)


def font(base):
    return base * FONT_SCALE


def short_probability(value):
    return "0%" if value == 0 else f"{100*value:.3g}%"


def citation_footer(fig):
    fig.text(0.5, 0.037,
             "Cite as: M. Sade, A. Tsarfati & O. Birnholtz (2026), arXiv:2603.18250",
             ha="center", va="center", fontsize=font(7))
    fig.text(0.5, 0.014,
             "Gamma Ray Bursts Effects on Extinction and Survivability in the Galaxy",
             ha="center", va="center", fontsize=font(7))


def dual_area_probability(hull):
    """check p100 by triangulating the dual polygon instead of using L.

    inward unit edge normals of the land hull form the vertices of D100.
    """
    if hull["status"] != "open_hemisphere":
        return 0.0
    vertices = hull["vertices"]
    if len(vertices) < 3:
        return np.nan
    normals = np.cross(vertices, np.roll(vertices, -1, axis=0))
    length = np.linalg.norm(normals, axis=1)
    normals = normals[length > 1e-14]
    if len(normals) < 3:
        return np.nan
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    normals[(normals @ vertices.sum(axis=0)) < 0] *= -1
    dual = geometry.spherical_hull(normals, hint=vertices.mean(axis=0))
    if dual["status"] != "open_hemisphere":
        raise ValueError("The all-land successful-direction polygon is degenerate.")
    value = dual["area"]/(4*np.pi)
    if abs(value-hull["probability"]) > 2e-9:
        raise ValueError("p100 perimeter and independent dual-area checks disagree.")
    return float(value)


def scene_data(land, view, hull, cap, union, qualifying, direction_tree,
               resolution=1040):
    """render D90 with nearest-sample labels; compute p90 from counts only.

    do not convexify the successful directions or estimate p90 from pixels.
    """
    surface, inside = surface_grid(view, resolution)
    field = union_field(surface, union)
    if np.any(qualifying):
        nearest = direction_tree.query(surface, k=1)[1]
        arrival_mask = qualifying[nearest] & (field > 0)
    else:
        arrival_mask = np.zeros(len(surface), dtype=bool)
    return dict(land=land, view=view, hull=hull, cap=cap, union=union,
                surface=surface, inside=inside, union_field=field,
                arrival_mask=arrival_mask)


def draw_arrival_domain(ax, disk, scene):
    """draw the nearest-sample approximation to D90."""
    mask = scene["arrival_mask"]
    inside = scene["inside"]
    if not np.any(mask):
        return
    shade_mask(ax, disk, mask, inside, DARK_PURPLE, alpha=0.47, zorder=6)
    if np.any(~mask):
        raster = np.ma.masked_all(inside.shape)
        raster[inside] = mask.astype(float)
        grid = np.linspace(-np.pi, np.pi, len(inside))
        lines = ax.contour(grid, grid, raster, levels=[0.5],
                           colors=[DARK_PURPLE], linewidths=0.95, zorder=7)
        lines.set_clip_path(disk)


def draw_reference_rings(ax, disk, scene):
    """angular reference rings about the cap-center marker."""
    center, view = scene["cap"]["center"], scene["view"]
    longitude = np.radians(geometry.lat_lon(center)[1])
    east = np.array([-np.sin(longitude), np.cos(longitude), 0.])
    north = np.cross(center, east)
    for radius in (30, 60, 90):
        plot_curve(ax, geometry.circle_vectors(center, radius), view,
                   color=RING_COLOR, lw=0.85, linestyle=(0, (4, 3)),
                   alpha=0.9, zorder=9)
        angle = np.radians(radius)
        point = (np.cos(angle)*center +
                 np.sin(angle)*(east+north)/np.sqrt(2))
        xy = geometry.aeqd_project_vectors(point[None, :], *view)[0]
        if np.isfinite(xy).all():
            label = ax.text(xy[0], xy[1], f"{radius}\N{DEGREE SIGN}",
                            color="0.12", fontsize=font(8), ha="left",
                            va="bottom", zorder=11)
            label.set_clip_path(disk)
    xy = geometry.aeqd_project_vectors(center[None, :], *view)[0]
    ax.plot(xy[0], xy[1], marker="+", color="red", ls="", ms=8,
            mew=1.7, zorder=12)


def draw_map(ax, scene, kind, title):
    land, view = scene["land"], scene["view"]
    hull, cap = scene["hull"], scene["cap"]
    surface, inside = scene["surface"], scene["inside"]
    disk = base_map(ax, land, view, title)
    color = hull_display_color(hull)
    if kind in ("C", "all"):
        draw_union(ax, disk, scene["union_field"], inside, alpha=0.36)
    if kind in ("B", "all"):
        shade_mask(ax, disk, surface @ cap["center"] >= cap["cos_radius"],
                   inside, cap["color"], alpha=0.12 if kind == "all" else 0.20,
                   zorder=2)
        plot_curve(ax, geometry.circle_vectors(cap["center"], cap["radius_deg"]),
                   view, color=cap["color"], lw=2, zorder=8)
    if kind in ("A", "all"):
        if hull["status"] == "open_hemisphere":
            shade_mask(ax, disk, hull_inside(surface, hull), inside, color,
                       alpha=0.16 if kind == "all" else 0.28, zorder=3)
            plot_curve(ax, geometry.hull_boundary(hull["vertices"], max_step_deg=0.1),
                       view, color=color, lw=2, zorder=8)
        elif hull["whole_sphere"]:
            draw_projected_outline(ax, disk, land, view, color, 2)
    if kind in ("C", "all"):
        draw_arrival_domain(ax, disk, scene)
    draw_reference_rings(ax, disk, scene)
    if kind == "A":
        ax.title.set_color(color)
    return disk


def area_for_display(scene):
    hull = scene["hull"]
    if hull["whole_sphere"]:
        outline = projected_enclosing_outline(scene["land"], scene["view"])
        return projected_polygon_surface_area(outline)
    return hull["area"]


def metrics(scene, p90):
    hull, cap, union = scene["hull"], scene["cap"], scene["union"]
    area = area_for_display(scene)
    a = f"A = {100*area/(4*np.pi):.2f}%" if np.isfinite(area) else "A: boundary case"
    return [a + "\np100 = " + short_probability(hull["probability"]),
            f"\u03b8 = {cap['radius_deg']:.2f}\N{DEGREE SIGN}\n"
            f"A = {100*cap['area_fraction']:.2f}%",
            f"U = {100*union['area_fraction']:.2f}%\n"
            "p90 (MC) = " + short_probability(p90)]


def map_legend(fig, scene, anchor, columns=3, base_font=9):
    handles = [
        Line2D([], [], marker="o", color="darkgreen", ls="", label="Land"),
        Line2D([], [], color=hull_display_color(scene["hull"]), lw=2, label="Land outline"),
        Line2D([], [], color=scene["cap"]["color"], lw=2, label="Minimal cap"),
        Patch(facecolor=LIGHT_PURPLE, alpha=0.55, label="Footprint union"),
        Patch(facecolor=DARK_PURPLE, alpha=0.70, label="Successful centers"),
        Line2D([], [], color=RING_COLOR, ls="--", lw=1, label="30 / 60 / 90\N{DEGREE SIGN}")]
    return fig.legend(handles=handles, loc="lower center", bbox_to_anchor=anchor,
                      ncol=columns, frameon=False, fontsize=font(base_font),
                      columnspacing=1.2, handlelength=1.7)


def render_regions(time_ma, scene, p90, path):
    fig, axes = plt.subplots(1, 3, figsize=FIGURE_SIZE)
    fig.subplots_adjust(left=0.015, right=0.985, top=0.84, bottom=0.31, wspace=0.07)
    fig.suptitle(f"{time_ma:g} Mya", fontsize=font(17), y=0.975)
    titles = ["A. Land outline", "B. Minimal cap", "C. 90% coverage"]
    for ax, kind, title, value in zip(axes, "ABC", titles, metrics(scene, p90)):
        draw_map(ax, scene, kind, title)
        ax.text(0.5, -0.055, value, transform=ax.transAxes,
                ha="center", va="top", fontsize=font(10), linespacing=1.35)
    map_legend(fig, scene, (0.5, 0.10), columns=3)
    citation_footer(fig)
    return save_figure(fig, path)


def render_overlay(time_ma, scene, p90, path):
    fig = plt.figure(figsize=FIGURE_SIZE)
    ax = fig.add_axes((0.015, 0.11, 0.60, 0.79))
    draw_map(ax, scene, "all", "")
    fig.suptitle(f"{time_ma:g} Mya", fontsize=font(17), y=0.975)
    titles = ["A. Land outline", "B. Minimal cap", "C. 90% coverage"]
    for index, (title, value) in enumerate(zip(titles, metrics(scene, p90))):
        y = 0.80-index*0.24
        fig.text(0.655, y, title, ha="left", va="top", fontsize=font(12))
        fig.text(0.655, y-0.065, value, ha="left", va="top",
                 fontsize=font(10), linespacing=1.45)
    map_legend(fig, scene, (0.5, 0.062), columns=6)
    citation_footer(fig)
    return save_figure(fig, path)


def render_video_frame(time_ma, scene, p90, path):
    """three maps stacked at left and their combined overlay at right."""
    fig = plt.figure(figsize=FIGURE_SIZE)
    fig.suptitle(f"{time_ma:g} Mya", x=0.725, y=0.982,
                 fontsize=font(17), ha="center")
    values = metrics(scene, p90)
    titles = ["A. Land outline", "B. Minimal cap", "C. 90% coverage"]
    for index, (kind, title, text) in enumerate(zip("ABC", titles, values)):
        bottom = 0.67-index*0.29
        ax = fig.add_axes((0.005, bottom, 0.195, 0.255))
        draw_map(ax, scene, kind, title)
        info = fig.add_axes((0.208, bottom+0.025, 0.225, 0.20))
        info.axis("off")
        info.text(0.0, 0.58, text, va="center", ha="left",
                  fontsize=font(10), linespacing=1.5,
                  color=hull_display_color(scene["hull"]) if kind == "A" else "black")
    large = fig.add_axes((0.445, 0.11, 0.55, 0.79))
    draw_map(large, scene, "all", "All regions")
    map_legend(fig, scene, (0.5, 0.062), columns=6)
    citation_footer(fig)
    return save_figure(fig, path)


def render_directions(time_ma, directions, qualifying, interval, path):
    """equal-area map of the original sampled arrival directions."""
    fig = plt.figure(figsize=FIGURE_SIZE)
    ax = fig.add_subplot(111, projection="mollweide")
    fig.subplots_adjust(left=0.07, right=0.93, top=0.79, bottom=0.16)
    lon = np.arctan2(directions[:, 1], directions[:, 0])
    lat = np.arcsin(np.clip(directions[:, 2], -1, 1))
    ax.scatter(lon[~qualifying], lat[~qualifying], s=0.15, c="0.85",
               linewidths=0, edgecolors="none", rasterized=True)
    ax.scatter(lon[qualifying], lat[qualifying], s=0.25, c=DARK_PURPLE,
               linewidths=0, edgecolors="none", rasterized=True)
    ax.grid(alpha=0.35)
    ax.tick_params(labelsize=font(10))
    hits, total = int(qualifying.sum()), len(directions)
    fig.suptitle(f"{time_ma:g} Mya: successful directions\n"
                 f"p90 (MC) = {short_probability(hits/total)}",
                 fontsize=font(12), y=0.97)
    fig.text(0.5, 0.105,
             f"95% MC interval: {100*interval[0]:.3f}% to {100*interval[1]:.3f}%",
             ha="center", fontsize=font(12))
    citation_footer(fig)
    return save_figure(fig, path)


def save_animation(paths, destination, frame_ms=FRAME_MS):
    from PIL import Image
    frames = []
    try:
        for path in paths:
            with Image.open(path) as original:
                frame = original.convert("RGB")
                if frame.size != OUTPUT_SIZE:
                    raise ValueError(f"GIF input must be native 3840x2160: {path}")
                frames.append(frame.quantize(colors=128))
        with BytesIO() as buffer:
            frames[0].save(buffer, format="GIF", save_all=True, append_images=frames[1:],
                           duration=frame_ms, loop=0, disposal=2, optimize=False)
            return write_output_bytes(destination, buffer.getvalue())
    finally:
        for frame in frames:
            frame.close()


def save_html_player(paths, times, destination, frame_ms=FRAME_MS):
    """embed frames in an offline player with pause, stepping and speed control."""
    from PIL import Image
    payloads = []
    for path in paths:
        with Image.open(path) as source:
            frame = source.convert("RGB")
            if frame.size != OUTPUT_SIZE:
                raise ValueError(f"Player input must be native 3840x2160: {path}")
            with BytesIO() as encoded:
                frame.save(encoded, "JPEG", quality=94, subsampling=0)
                payloads.append("data:image/jpeg;base64," +
                                base64.b64encode(encoded.getvalue()).decode("ascii"))
    data = json.dumps(dict(frames=payloads, times=[float(t) for t in times],
                           frame_ms=frame_ms), separators=(",", ":"))
    page = """<!doctype html>
<html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GRB land coverage</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f3f4f7;color:#182031;font:16px system-ui,sans-serif}
main{max-width:3840px;margin:auto}#map{display:block;width:100%;height:calc(100vh - 135px);object-fit:contain;background:white}
nav{display:flex;align-items:center;gap:12px;flex-wrap:wrap;padding:12px 18px;background:#fff;
border-top:1px solid #d9dce5;position:sticky;bottom:0}
button,input{font:inherit}button{padding:8px 15px;border:1px solid #aeb7c8;border-radius:6px;background:#f8f9fc;cursor:pointer}
#seek{flex:1;min-width:180px}#speed{width:90px;padding:7px}#status{min-width:190px;font-variant-numeric:tabular-nums}
small{display:block;padding:8px 18px;color:#525e70}
</style>
<main><img id="map" alt="Land coverage map">
<nav aria-label="Animation controls">
<button id="previous" aria-label="Previous frame">Previous</button>
<button id="play">Play</button>
<button id="next" aria-label="Next frame">Next</button>
<input id="seek" type="range" min="0" step="1" aria-label="Time frame">
<span id="status"></span>
<label>ms/frame <input id="speed" type="number" min="10" step="10"></label>
<label><input id="loop" type="checkbox" checked> Loop</label>
</nav><small>Space: play/pause. Left/right arrows: step through frames.
Original PNGs retain full resolution. Notes and definitions: coverage_regions_notes.md.</small></main>
<script id="frames-data" type="application/json">__DATA__</script>
<script>
const data=JSON.parse(document.getElementById("frames-data").textContent);
const picture=document.getElementById("map"),seek=document.getElementById("seek");
const play=document.getElementById("play"),speed=document.getElementById("speed");
const status=document.getElementById("status"),loop=document.getElementById("loop");
let index=0,timer=null;
seek.max=data.frames.length-1;speed.value=data.frame_ms;
function show(i){index=Math.max(0,Math.min(data.frames.length-1,i));
picture.src=data.frames[index];seek.value=index;
status.textContent=data.times[index]+" Mya | "+(index+1)+" / "+data.frames.length;}
function pause(){if(timer!==null)clearInterval(timer);timer=null;play.textContent="Play";}
function start(){pause();play.textContent="Pause";timer=setInterval(function(){
if(index===data.frames.length-1){if(loop.checked)show(0);else pause();}
else show(index+1);},Math.max(10,Number(speed.value)||data.frame_ms));}
play.addEventListener("click",function(){if(timer===null)start();else pause();});
document.getElementById("previous").addEventListener("click",function(){pause();show(index-1);});
document.getElementById("next").addEventListener("click",function(){pause();show(index+1);});
seek.addEventListener("input",function(){pause();show(Number(seek.value));});
speed.addEventListener("change",function(){if(timer!==null)start();});
document.addEventListener("keydown",function(event){
if(["INPUT","BUTTON"].includes(document.activeElement.tagName))return;
if(event.code==="Space"){event.preventDefault();if(timer===null)start();else pause();}
if(event.code==="ArrowLeft"){event.preventDefault();pause();show(index-1);}
if(event.code==="ArrowRight"){event.preventDefault();pause();show(index+1);}
});
show(0);
</script></html>"""
    return write_output_bytes(destination, page.replace("__DATA__", data).encode("utf-8"))


def video_encoder():
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError) as error:
        raise RuntimeError("MP4 export needs ffmpeg. Install it with the same Python "
                           "interpreter: python -m pip install imageio-ffmpeg. "
                           "The offline HTML player does not need it.") from error


def save_mp4(paths, destination, frame_ms=FRAME_MS):
    """stream 3840x2160 RGB frames to the encoder without storing raw video."""
    from PIL import Image
    encoder = video_encoder()
    descriptor, name = tempfile.mkstemp(suffix=".mp4")
    os.close(descriptor)
    temporary = Path(name)
    process = None
    try:
        with tempfile.TemporaryFile() as errors:
            process = subprocess.Popen(
                [encoder, "-y", "-loglevel", "error", "-f", "rawvideo",
                 "-pix_fmt", "rgb24", "-s", f"{OUTPUT_SIZE[0]}x{OUTPUT_SIZE[1]}",
                 "-framerate", f"1000/{frame_ms}", "-i", "pipe:0",
                 "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary)],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors)
            try:
                for index, path in enumerate(paths):
                    with Image.open(path) as original:
                        frame = original.convert("RGB")
                        if frame.size != OUTPUT_SIZE:
                            raise ValueError(f"MP4 input must be native 3840x2160: {path}")
                        process.stdin.write(frame.tobytes())
                    if (index+1) % 20 == 0 or index+1 == len(paths):
                        print(f"  Encoded {index+1}/{len(paths)} video frames", flush=True)
            except BrokenPipeError:
                pass
            finally:
                process.stdin.close()
            status = process.wait()
            if status:
                errors.seek(0)
                raise RuntimeError("MP4 encoder failed: " +
                                   errors.read().decode("utf-8", errors="replace")[-2000:])
        return write_output_bytes(destination, temporary.read_bytes())
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        temporary.unlink(missing_ok=True)


def existing_row_frames(folder, times=None):
    """find existing row frames, oldest first; require all 109 unless times is set.

    use the summary to resolve filenames redirected after write failures.
    """
    if not folder.is_dir():
        raise ValueError("Image folder does not exist: " + str(folder))
    wanted = sorted(times if times is not None else range(0, 545, 5), reverse=True)
    if not wanted or len(set(wanted)) != len(wanted) or not np.isfinite(wanted).all():
        raise ValueError("Row-video times must be distinct finite values.")
    found = {}
    for path in folder.glob("regions_*Ma.png"):
        match = re.fullmatch(r"regions_(-?\d+(?:\.\d+)?)Ma", path.stem)
        if match:
            time = float(match.group(1))
            if time in found:
                raise ValueError(f"Two row images found for {time:g} Mya.")
            found[time] = path
    summary = folder/"coverage_regions_summary.csv"
    if summary.is_file():
        table = pd.read_csv(summary)
        if {"Time_Ma", "Regions_image_file"} <= set(table.columns):
            for row in table.itertuples():
                if not isinstance(row.Regions_image_file, str):
                    continue
                original = Path(row.Regions_image_file)
                for candidate in (folder/original.name, original):
                    if candidate.is_file():
                        found[float(row.Time_Ma)] = candidate
                        break
    missing = [t for t in wanted if t not in found]
    if missing:
        examples = ", ".join(f"{t:g}" for t in missing[:12])
        raise ValueError(f"Missing row PNGs for {len(missing)} epochs (Mya): {examples}. "
                         "Generate the images first, or use --times for a chosen subset.")
    from PIL import Image
    paths = [found[t] for t in wanted]
    for path in paths:
        with Image.open(path) as image:
            if image.size != OUTPUT_SIZE:
                raise ValueError(f"{path.name} is {image.size}, expected {OUTPUT_SIZE}. "
                                 "Regenerate this image with the 4K script first.")
    return paths


def write_caption_notes(destination):
    notes = """# Land-coverage graphics: notes

All frames are 3840x2160. regions_YMa.png is the three-panel figure used in the
thesis; video_YMa.png stacks A, B and C on the left with their overlay on the right.

## Caption

Land geometry and hemisphere irradiation at one epoch. Green points are land
cells. A: the spherical convex hull of all land (dark blue) where it fits inside
an open hemisphere; otherwise an orange convex outline in the map projection,
which is only a visual aid and is not used for p100. B: the smallest cap that
encloses all land (green if its radius is below 90 degrees, red otherwise), with
radius theta and area A. Dashed circles mark 30, 60 and 90 degrees about the red
plus sign, the center of that cap. C: light purple is the union U of the
hemispheres lit by the sampled directions that cover at least 90% of the land
cells, as a fraction of the whole surface (not a probability). Dark purple marks
those directions (D90), classified by the nearest sampled direction, so it is a
finite-sample picture of the set.

p90 is the fraction of the sampled directions that light at least 90% of the land
cells, an estimate of Area(D90)/(4*pi). Areas and probabilities are computed on
the sphere, not measured on the distorted map. A zero p90 does not prove the
event is impossible; Wilson 95% intervals are in the CSV.

## D90 in code

D90 = {n : sum_i 1[n . x_i > 0] >= ceil(0.9*N_land)} is a set of directions.
qualifying is the boolean membership array for the sampled directions,
D90_samples = directions[qualifying] holds the successful ones, and p90 is a number.

## Exact p100

For hull vertices v_1, ..., v_m in order (v_(m+1) = v_1), the edge lengths on the
unit sphere are l_i = atan2(|v_i x v_(i+1)|, v_i . v_(i+1)) and L = sum_i l_i.
The directions that light all land form D100 = {n : n . v_i >= 0 for all i}, the
polar polygon of the hull, whose interior angles are pi - l_i. The spherical
excess formula (Gauss-Bonnet) gives

    Area(D100) = sum_i (pi - l_i) - (m-2)*pi = 2*pi - L
    p100 = (2*pi - L)/(4*pi)

If no open hemisphere contains all land, p100 = 0. The script also triangulates
D100 (column P_land_100_dual_area_pct) and compares p100 with the sampled count
(column P100_inside_MC_CI95).

Reference: K. Nagy and V. Vigh, Random Spherical Disc-Polygons and a Duality,
Lemma 4(ii) with r = pi/2,
https://www.math.u-szeged.hu/~vigvik/preprints/spherical_duality.pdf

## Animations

--video writes coverage_regions_player.html (offline, with pause and a time
slider). --mp4 writes coverage_regions_video.mp4 and coverage_regions_row.mp4 and
needs ffmpeg or the imageio-ffmpeg package. --gif writes GIF copies. Frames run
from the oldest epoch to the present; --frame-ms sets the time per frame.

Cite as: M. Sade, A. Tsarfati and O. Birnholtz (2026),
Gamma Ray Bursts Effects on Extinction and Survivability in the Galaxy,
arXiv:2603.18250.
"""
    return write_output_bytes(destination, notes.encode("utf-8"))


def self_test():
    geometry.geometry_checks()
    octant = all_land_hull(np.eye(3), np.ones(3))
    np.testing.assert_allclose(octant["probability"], 1/8, atol=1e-12)
    np.testing.assert_allclose(octant["area"]/(4*np.pi), 1/8, atol=1e-12)
    np.testing.assert_allclose(dual_area_probability(octant), 1/8, atol=1e-12)
    offset_land = geometry.unit_vectors([60, 60, 80], [0, 20, 10])
    outside_hint = geometry.unit_vectors([60], [180])[0]
    offset_hull = all_land_hull(offset_land, outside_hint)
    assert np.all(hull_inside(offset_land, offset_hull))
    np.testing.assert_allclose(dual_area_probability(offset_hull),
                               offset_hull["probability"], atol=1e-12)
    union = sampled_union(np.eye(3))
    np.testing.assert_allclose(union["area_fraction"], 7/8, atol=1e-12)
    pair = geometry.unit_vectors([0, 0], [0, 60])
    np.testing.assert_allclose(sampled_union(pair)["area_fraction"], 2/3, atol=1e-12)
    point = np.array([[0., 0, 1]])
    np.testing.assert_allclose(sampled_union(point)["area_fraction"], 0.5, atol=1e-12)
    rng = np.random.default_rng(3)
    points = rng.normal(size=(3000, 3)); points /= np.linalg.norm(points, axis=1)[:, None]
    np.testing.assert_array_equal(in_union(points, union), np.any(points @ np.eye(3) > 0, axis=1))
    interior_centers = np.vstack((np.eye(3), np.ones(3)/np.sqrt(3)))
    np.testing.assert_array_equal(in_union(points, sampled_union(interior_centers)),
                                  np.any(points @ interior_centers.T > 0, axis=1))
    counts = count_illuminated(np.eye(3), points, 43)
    np.testing.assert_array_equal(counts, (points > 0).sum(axis=1))
    tetra = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1.]])
    full = all_land_hull(tetra/np.sqrt(3), np.array([0., 0, 1]))
    assert full["whole_sphere"] and full["probability"] == 0
    np.testing.assert_allclose(full["area"], 4*np.pi, atol=1e-12)
    cap = minimum_enclosing_cap(tetra/np.sqrt(3))
    np.testing.assert_allclose(cap["radius_deg"], np.degrees(np.arccos(-1/3)), atol=1e-8)
    assert cap["color"] == "red"
    np.testing.assert_allclose(minimum_enclosing_cap(np.eye(3))["radius_deg"],
                               np.degrees(np.arccos(1/np.sqrt(3))), atol=1e-8)
    np.testing.assert_allclose(minimum_enclosing_cap(pair)["radius_deg"], 30, atol=1e-8)
    np.testing.assert_allclose(minimum_enclosing_cap(point)["radius_deg"], 0, atol=1e-8)
    antipodes = np.array([[1., 0, 0], [-1., 0, 0]])
    np.testing.assert_allclose(minimum_enclosing_cap(antipodes)["radius_deg"], 90, atol=1e-8)
    assert raw_time(Path("Map81.5_475Ma.csv")) == 475
    assert raw_time(Path("coverage_475Ma_40_directions.csv")) is None
    print("Region checks passed: exact all-land probability, minimal caps below/at/above "
          "90 degrees, hemisphere unions, union membership, direction counts and raw filenames.", flush=True)


def main():
    global FONT_SCALE
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--results", type=Path)
    chosen = parser.add_mutually_exclusive_group()
    chosen.add_argument("--times", nargs="+", type=float)
    chosen.add_argument("--all-times", action="store_true")
    parser.add_argument("--directions", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--recount", action="store_true")
    parser.add_argument("--gif", action="store_true")
    parser.add_argument("--video", action="store_true", help="Offline player; combined PNGs are always created")
    parser.add_argument("--html", action="store_true", help="Same as --video; no encoder required")
    movie_mode = parser.add_mutually_exclusive_group()
    movie_mode.add_argument("--mp4", action="store_true", help="Export combined-layout and row-layout MP4s")
    movie_mode.add_argument("--row-mp4-only", action="store_true",
                            help="Encode existing row PNGs only; no maps or sampling are regenerated")
    parser.add_argument("--frame-ms", type=int, default=FRAME_MS)
    parser.add_argument("--font-scale", type=float, default=2.0)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    FONT_SCALE = args.font_scale
    if args.self_test:
        self_test(); return
    if args.directions < 1 or args.chunk_size < 1:
        raise ValueError("Direction count and chunk size must be positive.")
    if args.frame_ms < 10 or not np.isfinite(args.font_scale) or args.font_scale <= 0:
        raise ValueError("Use frame-ms >= 10 and a positive finite font-scale.")
    make_player = args.video or args.html or args.mp4
    root = args.base_dir.expanduser().resolve()
    if not list(root.glob("Map*Ma.csv")) and (root/"PaleoDEMS").is_dir():
        root = root/"PaleoDEMS"
    if args.row_mp4_only:
        output = args.output_dir.expanduser().resolve() if args.output_dir else root/"CoverageRegions"
        paths = existing_row_frames(output, args.times)
        print(f"Encoding {len(paths)} existing row images, oldest to present; "
              f"{args.frame_ms} ms per frame. No maps will be regenerated.", flush=True)
        movie_path = save_mp4(paths, output/"coverage_regions_row.mp4", args.frame_ms)
        print("Saved row-layout MP4:", movie_path, flush=True)
        return
    choices = [root/"HullCenterSearch"/"HullComparison_all.csv", root/"HullComparison_all.csv"]
    source = args.results.expanduser().resolve() if args.results else next((p for p in choices if p.is_file()), None)
    if source is None or not source.is_file():
        raise ValueError("Missing HullCenterSearch/HullComparison_all.csv; use --results for another path.")
    saved = pd.read_csv(source)
    needed = {"Time_Ma", "Coverage_Percent", "N_land_cells", "Original_cap_center_lat",
              "Original_cap_center_lon", "P_hull_fixed_pct"}
    if needed - set(saved.columns):
        raise ValueError("Saved comparison is missing: " + ", ".join(sorted(needed-set(saved.columns))))
    times = (sorted(saved.Time_Ma.unique(), reverse=True) if args.all_times else
             sorted(args.times if args.times is not None else [475, 360], reverse=True))
    if len(set(times)) != len(times) or not np.isfinite(times).all():
        raise ValueError("Times must be distinct finite values.")
    files = [p for p in root.glob("*.csv") if raw_time(p) is not None]
    selected = {}
    for time_ma in times:
        candidates = [p for p in files if abs(raw_time(p)-time_ma) < 0.5]
        if len(candidates) != 1:
            raise ValueError(f"Expected one raw MapX_YMa.csv for {time_ma:g} Mya in {root}; found {candidates}")
        selected[time_ma] = candidates[0]
        geometry.matching_row(saved, time_ma, 100)
    output = args.output_dir.expanduser().resolve() if args.output_dir else root/"CoverageRegions"
    output.mkdir(parents=True, exist_ok=True)
    cache = output/"_cache"; cache.mkdir(exist_ok=True)
    rng = np.random.default_rng(args.seed)
    directions = rng.normal(size=(args.directions, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    direction_tree = cKDTree(directions)
    print("Raw map folder:", root, "\nOutput folder:", output, flush=True)
    print("4K outputs: 3840x2160 pixels. video_XMa.png always contains "
          "three vertical maps at left + their combined overlay at right.", flush=True)
    summaries, images, overlays, video_images = [], [], [], []
    for time_ma in times:
        print(f"\n{time_ma:g} Mya: {selected[time_ma].name}", flush=True)
        land = geometry.read_land(selected[time_ma])
        row100 = geometry.matching_row(saved, time_ma, 100)
        if len(land) != int(row100.N_land_cells):
            raise ValueError("Raw land count does not match saved results.")
        view = (float(row100.Original_cap_center_lat), float(row100.Original_cap_center_lon))
        hint = geometry.unit_vectors([view[0]], [view[1]])[0]
        hull = all_land_hull(land, hint)
        p100_dual = dual_area_probability(hull)
        if hull["status"] == "open_hemisphere" and not np.all(hull_inside(land, hull)):
            raise ValueError("The spherical hull does not enclose every land-point center.")
        if hull["whole_sphere"]:
            display_area = projected_polygon_surface_area(projected_enclosing_outline(land, view))
        else:
            display_area = hull["area"]
        cap = minimum_enclosing_cap(land)
        if "Radius_deg" in row100 and cap["radius_deg"] > float(row100.Radius_deg)+1e-5:
            raise ValueError("The new minimal cap is larger than the previously fitted cap; check inputs.")
        if abs(100*hull["probability"]-float(row100.P_hull_fixed_pct)) > 2e-6:
            raise ValueError("Reconstructed all-land hull differs from saved results; check input versions.")
        counts = cached_counts(land, directions, args.seed,
                               cache/f"counts_{time_ma:g}Ma.npz", args.chunk_size, args.recount)
        k90 = int(np.ceil(0.9*len(land)))
        qualifying = counts >= k90
        successes = int(qualifying.sum())
        p90 = successes/len(directions)
        interval = wilson_interval(successes, len(directions))
        D90_samples = directions[qualifying]
        union = sampled_union(D90_samples)
        scene = scene_data(land, view, hull, cap, union, qualifying, direction_tree)
        region_path = output/f"regions_{time_ma:g}Ma.png"
        overlay_path = output/f"overlay_{time_ma:g}Ma.png"
        direction_path = output/f"successful_directions_{time_ma:g}Ma.png"
        region_path = render_regions(time_ma, scene, p90, region_path)
        overlay_path = render_overlay(time_ma, scene, p90, overlay_path)
        direction_path = render_directions(time_ma, directions, qualifying, interval, direction_path)
        video_path = render_video_frame(time_ma, scene, p90, output/f"video_{time_ma:g}Ma.png")
        video_images.append(video_path)
        images.append(region_path)
        overlays.append(overlay_path)
        hits100 = int(np.sum(counts == len(land)))
        ci100 = wilson_interval(hits100, len(directions))
        summaries.append(dict(Time_Ma=time_ma, N_land_cells=len(land), MC_directions=len(directions),
                              MC_seed=args.seed, MC_successes_90=successes, P_land_90_MC_pct=100*p90,
                              MC_CI95_90_low_pct=100*interval[0], MC_CI95_90_high_pct=100*interval[1],
                              P_land_100_exact_pct=100*hull["probability"],
                              P_land_100_MC_pct=100*np.mean(counts == len(land)),
                              P_land_100_dual_area_pct=100*p100_dual,
                              Hull_perimeter_rad=hull.get("perimeter", np.nan),
                              MC_successes_100=hits100,
                              MC_CI95_100_low_pct=100*ci100[0],
                              MC_CI95_100_high_pct=100*ci100[1],
                              P100_inside_MC_CI95=bool(ci100[0]-1e-12 <= hull["probability"] <= ci100[1]+1e-12),
                              All_land_hull_area_pct=100*hull["area"]/(4*np.pi),
                              All_land_hull_area_displayed=hull["status"] == "open_hemisphere",
                              Plot_A_geometry=("projected_planar_envelope" if hull["whole_sphere"] else
                                               "spherical_hull" if hull["status"] == "open_hemisphere" else
                                               "boundary_degenerate"),
                              Projected_outline_is_used_for_probability=False,
                              Plot_A_enclosed_surface_area_pct=100*display_area/(4*np.pi),
                              Plot_A_area_method=("inverse_AEQD_boundary_quadrature" if hull["whole_sphere"] else
                                                  "spherical_hull_area"),
                              All_land_point_centers_enclosed=(True if hull["status"] == "open_hemisphere" or
                                                               hull["whole_sphere"] else np.nan),
                              All_land_hull_whole_sphere=hull["whole_sphere"],
                              All_land_hull_status=hull["status"],
                              All_land_hull_color=hull_display_color(hull),
                              Minimum_all_land_cap_radius_deg=cap["radius_deg"],
                              Minimum_all_land_cap_area_pct=100*cap["area_fraction"],
                              Minimum_all_land_cap_center_lat=geometry.lat_lon(cap["center"])[0],
                              Minimum_all_land_cap_center_lon=geometry.lat_lon(cap["center"])[1],
                              Minimum_all_land_cap_method=cap["method"],
                              Minimum_all_land_cap_color=cap["color"],
                              Sampled_successful_footprint_union_area_pct=100*union["area_fraction"],
                              Sampled_union_is_empty=union["empty"],
                              Union_area_is_full_continuum_area=False,
                              D90_display_method="nearest_MC_sample_classification",
                              D90_display_grid_resolution=scene["inside"].shape[0],
                              D90_is_rendered_as_convex_hull=False,
                              D90_display_area_used_for_probability=False,
                              Image_width_px=OUTPUT_SIZE[0], Image_height_px=OUTPUT_SIZE[1],
                              Image_aspect_ratio="16:9", Font_scale=FONT_SCALE,
                              Angle_label_background=False,
                              Combined_layout="three_vertical_maps_left_overlay_right",
                              Combined_image_file=str(video_path),
                              Video_image_file=str(video_path),
                              Regions_image_file=str(region_path),
                              Overlay_image_file=str(overlay_path),
                              Successful_directions_image_file=str(direction_path)))
        print(f"  p90 MC={100*p90:.4f}%; p100 exact={100*hull['probability']:.6f}%; "
              f"sampled footprint union area={100*union['area_fraction']:.3f}%", flush=True)
        print(f"  Minimal all-land cap: {cap['radius_deg']:.4f} degrees; "
              f"area {100*cap['area_fraction']:.4f}%; {cap['color']}", flush=True)
        print("  Saved:", region_path.name, overlay_path.name,
              direction_path.name, "and", video_path.name, flush=True)
    summary_path = output/"coverage_regions_summary.csv"
    table = pd.DataFrame(summaries)
    summary_path = write_output_bytes(summary_path, table.to_csv(index=False).encode("utf-8"))
    if args.gif:
        animation_path = save_animation(images, output/"coverage_regions.gif", args.frame_ms)
        overlay_animation_path = save_animation(overlays, output/"coverage_regions_overlay.gif", args.frame_ms)
        combined_animation_path = save_animation(video_images, output/"coverage_regions_combined.gif", args.frame_ms)
        print("Saved animation:", animation_path, flush=True)
        print("Saved overlay animation:", overlay_animation_path, flush=True)
        print("Saved combined-layout animation:", combined_animation_path, flush=True)
    if make_player:
        player_path = save_html_player(video_images, times,
                                      output/"coverage_regions_player.html", args.frame_ms)
        print("Saved pauseable offline player:", player_path, flush=True)
    notes_path = write_caption_notes(output/"coverage_regions_notes.md")
    print("Saved caption and method notes:", notes_path, flush=True)
    if args.mp4:
        mp4_path = save_mp4(video_images, output/"coverage_regions_video.mp4", args.frame_ms)
        print("Saved combined-layout MP4:", mp4_path, flush=True)
        row_mp4_path = save_mp4(images, output/"coverage_regions_row.mp4", args.frame_ms)
        print("Saved row-layout MP4:", row_mp4_path, flush=True)
    print(f"p100 lies inside its pointwise MC 95% interval at "
          f"{int(table.P100_inside_MC_CI95.sum())}/{len(table)} processed epochs.", flush=True)
    for column, label in (("P_land_90_MC_pct", "p90 Monte Carlo"),
                          ("P_land_100_exact_pct", "p100 exact model")):
        top = table.loc[table[column].idxmax()]
        print(f"Maximum {label} across the {len(table)} processed epochs: "
              f"{top[column]:.6f}% at {top.Time_Ma:g} Mya", flush=True)
    print("Saved summary:", summary_path, flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError) as error:
        raise SystemExit("ERROR: " + str(error))
