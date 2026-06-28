"""Debug: find_profile_safe_zone with isotensoid dome."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path("src").resolve()))

import numpy as np
from filament_winder.core.geometry.axisymmetric import (
    cylinder_with_domes_profile,
)
from filament_winder.core.path_planning.profile import (
    ProfileDomePathConfig,
    ProfileDomePathGenerator,
    find_profile_safe_zone,
)

# Build the same mandrel as the demo
mandrel = cylinder_with_domes_profile(
    cylinder_length_mm=1000,
    cylinder_radius_mm=101.6,
    left_dome_length_mm=120,
    right_dome_length_mm=120,
    polar_opening_radius_mm=25,
    samples_per_region=120,
    dome_shape="isotensoid",
)

print(f"Mandrel z range: {mandrel.start_z_mm:.1f} - {mandrel.end_z_mm:.1f}")
print(f"Mandrel r range: {float(np.min(mandrel.r_mm)):.3f} - {mandrel.max_radius_mm:.3f}")
print(f"Mandrel points: {mandrel.z_mm.size}")
print(f"First 3 points: z={mandrel.z_mm[:3]}, r={mandrel.r_mm[:3]}")
print(f"Last 3 points: z={mandrel.z_mm[-3:]}, r={mandrel.r_mm[-3:]}")
print(f"Min(endpoints): {float(np.min(mandrel.r_mm[[0, -1]])):.6f}")
print()

# Check find_profile_safe_zone at r_min = 25
try:
    sz = find_profile_safe_zone(mandrel, min_radius_mm=25.0)
    print(f"Safe zone at r>=25: z=[{sz.start_z_mm:.3f}, {sz.end_z_mm:.3f}]")
except ValueError as e:
    print(f"Safe zone error at 25: {e}")

# Check find_profile_safe_zone at the actual Clairaut radius
R = 101.6
alpha = 45
K = R * np.sin(np.deg2rad(alpha))
print(f"\nClairaut K=R*sin(alpha) = {K:.6f}")

# What does the clamp produce?
polar_r = float(np.min(mandrel.r_mm[[0, -1]]))
clamped_K = min(K, polar_r)
print(f"polar_r = {polar_r:.6f}, clamped K = {clamped_K:.6f}")

try:
    sz = find_profile_safe_zone(mandrel, min_radius_mm=clamped_K)
    print(f"Safe zone at r>={clamped_K}: z=[{sz.start_z_mm:.3f}, {sz.end_z_mm:.3f}]")
except ValueError as e:
    print(f"Safe zone error at {clamped_K}: {e}")

# What radius at z=0?
idx0 = np.argmin(np.abs(mandrel.z_mm - 0))
print(f"\nAt z≈0: z={mandrel.z_mm[idx0]:.6f}, r={mandrel.r_mm[idx0]:.6f}")

# Find all z where r < 25.001
below = mandrel.z_mm[mandrel.r_mm < 25.001]
print(f"z points with r < 25.001: {below}")
