"""Debug: trace the full dome winding pipeline."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path("src").resolve()))

import numpy as np

from filament_winder.core.geometry.axisymmetric import cylinder_with_domes_profile
from filament_winder.core.path_planning.profile import (
    ProfileDomePathConfig,
    ProfileDomePathGenerator,
    find_profile_safe_zone,
)

# Build the same mandrel as _build_mandrel in winding_job.py
mandrel = cylinder_with_domes_profile(
    cylinder_length_mm=1000,
    cylinder_radius_mm=101.6,
    left_dome_length_mm=120,
    right_dome_length_mm=120,
    polar_opening_radius_mm=25,
    samples_per_region=max(16, 360 // 3),
    dome_shape="isotensoid",
    name="demo_domed_pressure_vessel",
)

print(f"Mandrel: z=[{mandrel.start_z_mm:.3f}, {mandrel.end_z_mm:.3f}] r=[{float(np.min(mandrel.r_mm)):.3f}, {mandrel.max_radius_mm:.3f}]")
eps = 1e-3
cyl_idx = np.where(mandrel.r_mm >= mandrel.max_radius_mm - eps)[0]
if len(cyl_idx) > 0:
    print(f"Cylinder z-range: {mandrel.z_mm[cyl_idx[0]]:.3f} - {mandrel.z_mm[cyl_idx[-1]]:.3f}")
print(f"Mandrel start/end r: {mandrel.r_mm[0]:.3f}, {mandrel.r_mm[-1]:.3f}")
print()

# Build ProfileDomePathConfig matching the geodesic layer
pconf = ProfileDomePathConfig(
    winding_angle_deg=45,
    tow_width_mm=6.0,
    turnaround_radius_mm=28,
    turnaround_angle_deg=180.0,
    points_per_span=140,
    turnaround_points=20,
    circuits=1,
    start_theta_rad=0.0,
)

print(f"Config clairaut_radius_mm (R*sin(alpha)): {pconf.clairaut_radius_mm(mandrel):.6f}")
turn_r = pconf.resolved_turnaround_radius_mm(mandrel)
print(f"Config resolved_turnaround_radius_mm: {turn_r:.3f}")

# What the ProfileDomePathGenerator does in __init__:
K_original = pconf.clairaut_radius_mm(mandrel)
polar_r = float(np.min(mandrel.r_mm[[0, -1]]))
clamped_K = min(K_original, polar_r) if polar_r > 0 and polar_r < K_original else K_original
print(f"\nClamping: K_orig={K_original:.6f}, polar_r={polar_r:.6f}, clamped_K={clamped_K:.6f}")

# find_profile_safe_zone with clamped_K
if clamped_K > 0:
    try:
        sz = find_profile_safe_zone(mandrel, min_radius_mm=clamped_K)
        print(f"Safe zone at r>={clamped_K:.6f}: z=[{sz.start_z_mm:.3f}, {sz.end_z_mm:.3f}] len={sz.end_z_mm - sz.start_z_mm:.3f}")
        sz_start = sz.start_z_mm
        sz_end = sz.end_z_mm
    except ValueError as e:
        print(f"Safe zone error: {e}")
        sz_start = None
        sz_end = None
else:
    print(f"Invalid clamped_K={clamped_K}")

# Now build the generator and see what happens
gen = ProfileDomePathGenerator(mandrel, pconf)
print(f"\nGenerator:")
print(f"  clairaut_radius_mm: {gen.clairaut_radius_mm:.6f}")
print(f"  turnaround_radius_mm: {gen.turnaround_radius_mm:.6f}")
print(f"  dome_start_z: {gen.dome_start_z:.3f}")
print(f"  dome_end_z: {gen.dome_end_z:.3f}")
print(f"  actual_winding_angle_deg: {gen.actual_winding_angle_deg:.6f}")

# Generate
path = gen.generate()
print(f"\nGenerated path: z=[{float(np.min(path.z_mm)):.3f}, {float(np.max(path.z_mm)):.3f}]")
print(f"  theta=[{float(np.min(path.theta_rad)):.3f}, {float(np.max(path.theta_rad)):.3f}]")
