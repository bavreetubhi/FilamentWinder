"""Debug: analyze dome winding in generated CSV."""
from __future__ import annotations
import csv
import sys
from pathlib import Path

path = Path("exports/demo_domed_pressure_vessel/path.csv")
with path.open(newline="") as f:
    rows = list(csv.DictReader(f))

geo = [r for r in rows if r["layer_id"] == "02-geodesic-dome-to-dome-1"]
print(f"Geodesic layer: {len(geo)} rows")

prev = None
segments = []
start_idx = 0
for i, r in enumerate(geo):
    seg = (r["segment_id"], r["segment_type"])
    if seg != prev and i > 0:
        segments.append((prev[0], prev[1], start_idx, i - 1))
        start_idx = i
    prev = seg
segments.append((prev[0], prev[1], start_idx, len(geo) - 1))

for sid, stype, si, ei in segments:
    sz = float(geo[si]["z_mm"])
    sr = float(geo[si]["r_mm"])
    sa = float(geo[si]["local_winding_angle_deg"])
    ez = float(geo[ei]["z_mm"])
    er = float(geo[ei]["r_mm"])
    ea = float(geo[ei]["local_winding_angle_deg"])
    marker = " <TURN>" if "turn" in stype.lower() else ""
    print(f"{sid:12s} {stype:20s} pts={ei-si+1:6d}  z={sz:8.1f}->{ez:8.1f}  r={sr:8.1f}->{er:8.1f}  angle={sa:6.1f}->{ea:6.1f}{marker}")

# Also check non-geodesic layer
ngeo = [r for r in rows if r["layer_id"] == "03-non-geodesic-controlled-1"]
print(f"\nNon-geodesic layer: {len(ngeo)} rows")
zs = [float(r["z_mm"]) for r in ngeo]
rs = [float(r["r_mm"]) for r in ngeo]
angles = [float(r["local_winding_angle_deg"]) for r in ngeo]
print(f"  Z range: {min(zs):.1f} - {max(zs):.1f}")
print(f"  R range: {min(rs):.1f} - {max(rs):.1f}")
print(f"  Angle range: {min(angles):.1f} - {max(angles):.1f}")
print(f"  First: z={ngeo[0]['z_mm']}, r={ngeo[0]['r_mm']}, angle={ngeo[0]['local_winding_angle_deg']}")
print(f"  Last:  z={ngeo[-1]['z_mm']}, r={ngeo[-1]['r_mm']}, angle={ngeo[-1]['local_winding_angle_deg']}")
