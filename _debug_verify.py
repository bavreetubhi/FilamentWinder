"""Verify dome path z-ranges are within mandrel bounds."""
import csv
import numpy as np

rows = list(csv.DictReader(open("exports/demo_domed_pressure_vessel/path.csv", newline="")))

for lid in ["01-hoop-cylinder-1", "02-geodesic-dome-to-dome-1", "03-non-geodesic-controlled-1"]:
    layer = [r for r in rows if r["layer_id"] == lid]
    zs = np.array([float(r["z_mm"]) for r in layer])
    rs = np.array([float(r["r_mm"]) for r in layer])
    angles = np.array([float(r["local_winding_angle_deg"]) for r in layer])
    print(f"{lid}:")
    print(f"  z=[{float(np.min(zs)):.3f}, {float(np.max(zs)):.3f}]")
    print(f"  r=[{float(np.min(rs)):.3f}, {float(np.max(rs)):.3f}]")
    print(f"  angle=[{float(np.min(angles)):.2f}, {float(np.max(angles)):.2f}]")
    print()
