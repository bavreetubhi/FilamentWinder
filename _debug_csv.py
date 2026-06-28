"""Analyze the actual CSV output."""
from __future__ import annotations
import csv
import numpy as np

rows = list(csv.DictReader(open("exports/demo_domed_pressure_vessel/path.csv", newline="")))

geo = [r for r in rows if r["layer_id"] == "02-geodesic-dome-to-dome-1"]
print(f"Geo layer points: {len(geo)}")
zs = np.array([float(r["z_mm"]) for r in geo])
rs = np.array([float(r["r_mm"]) for r in geo])
print(f"Z range: {float(np.min(zs)):.3f} - {float(np.max(zs)):.3f}")
print(f"R range: {float(np.min(rs)):.3f} - {float(np.max(rs)):.3f}")

max_i = int(np.argmax(zs))
min_i = int(np.argmin(zs))
for idx, label in [(max_i, "Max Z"), (min_i, "Min Z")]:
    r = geo[idx]
    print(f"{label}: z={r['z_mm']}, r={r['r_mm']}, theta={r['theta_deg']}, seg_type={r['segment_type']}")

# Check non-geodesic too
ngeo = [r for r in rows if r["layer_id"] == "03-non-geodesic-controlled-1"]
print(f"\nNon-geo layer points: {len(ngeo)}")
nzs = np.array([float(r["z_mm"]) for r in ngeo])
nrs = np.array([float(r["r_mm"]) for r in ngeo])
print(f"Z range: {float(np.min(nzs)):.3f} - {float(np.max(nzs)):.3f}")
print(f"R range: {float(np.min(nrs)):.3f} - {float(np.max(nrs)):.3f}")
