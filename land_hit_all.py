# Probability that a single GRB hemisphere lights 90%, 100% of the land cells,
# by direct sampling of GRB directions, at all 109 time steps. Also gives the
# cap-formula lower bound for comparison and the numbers for Table 2.
# Cite as: M. Sade, A. Tsarfati and O. Birnholtz (2026), "Gamma Ray Bursts Effects on
# Extinction and Survivability in the Galaxy", arXiv:2603.18250
import numpy as np
import pandas as pd
from pathlib import Path

INPUT_FOLDER = "PaleoDEMS"
CAPS_CSV = "PaleoDEMS/SurfaceCoverage_90PercentLand.csv"  # cap search output (both fractions)
OUTPUT_CSV = "LandHitProbability.csv"
N_DIRECTIONS = 200000
rng = np.random.default_rng(1)


def unit_vectors(lat_deg, lon_deg):
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    return np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def time_of(f):
    try:
        return float(f.stem.split("_")[-1].replace("Ma", ""))
    except ValueError:
        return None  # not a PaleoDEMS time slice (e.g the caps CSV)


caps = pd.read_csv(CAPS_CSV)

dirs = rng.normal(size=(N_DIRECTIONS, 3))
dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)  # random GRB directions, uniform on the sphere

results = []
files = sorted((f for f in Path(INPUT_FOLDER).glob("*.csv") if time_of(f) is not None), key=time_of)
for f in files:
    t = time_of(f)
    df = pd.read_csv(f, skiprows=1, names=["lon", "lat", "elev"])
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    df = df[df["elev"] > 0]
    land = unit_vectors(df["lat"].values, df["lon"].values)
    k90, k100 = int(np.ceil(0.9 * len(land))), len(land)

    hits90 = hits100 = 0
    for start in range(0, N_DIRECTIONS, 1000):
        n_lit = (dirs[start:start + 1000] @ land.T > 0).sum(axis=1)  # land cells lit by each GRB
        hits90 += (n_lit >= k90).sum()
        hits100 += (n_lit >= k100).sum()

    row = {"Time_Ma": t,
           "P_land_90": hits90 / N_DIRECTIONS,
           "P_land_100": hits100 / N_DIRECTIONS}
    for frac, name in ((0.9, "90"), (1.0, "100")):
        c = caps[(abs(caps["Time_Ma"] - t) < 0.5) & np.isclose(caps["Coverage_Fraction"], frac)]
        theta = np.radians(float(c["Radius_deg"].iloc[0]))
        row[f"P_cap_{name}"] = (1 - np.sin(theta)) / 2 if theta < np.pi / 2 else 0.0
    results.append(row)
    print(f"{t:5.0f} Mya: land 90% {100*row['P_land_90']:6.3f}%  (cap {100*row['P_cap_90']:6.3f}%)   "
          f"land 100% {100*row['P_land_100']:6.3f}%  (cap {100*row['P_cap_100']:6.3f}%)")

res = pd.DataFrame(results).sort_values("Time_Ma")
res.to_csv(OUTPUT_CSV, index=False)

# numbers for Table 2 and the text
print(f"\nsaved {OUTPUT_CSV} ({len(res)} time steps)")
for name in ("90", "100"):
    col = res[f"P_land_{name}"]
    imax = col.idxmax()
    print(f"\n{name}% of land:")
    print(f"  max: {100*col.max():.3f}% at {res.loc[imax, 'Time_Ma']:.0f} Mya")
    print(f"  avg Phanerozoic (540-0): {100*col.mean():.3f}%")
    print(f"  avg since Permian (<=300): {100*col[res['Time_Ma'] <= 300].mean():.3f}%")
    print(f"  avg since Jurassic (<=200): {100*col[res['Time_Ma'] <= 200].mean():.3f}%")
    print(f"  present: {100*col[res['Time_Ma'] == 0].iloc[0]:.3f}%")
    pang = col[(res["Time_Ma"] <= 300) & (res["Time_Ma"] >= 155)]
    print(f"  Pangaea era (300-155): {100*pang.min():.3f}-{100*pang.max():.3f}%")
    nonzero = res.loc[col > 0, "Time_Ma"]
    print(f"  youngest time with a nonzero chance: {nonzero.min():.0f} Mya" if len(nonzero) else "  zero at every time step")
    cap_at_max = res.loc[imax, f"P_cap_{name}"]
    if cap_at_max > 0:
        print(f"  sampled / formula at that time: {col.max() / cap_at_max:.1f}")