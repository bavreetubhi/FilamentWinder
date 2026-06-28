"""Debug: trace the geodesic dome winding pipeline."""
from __future__ import annotations
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path("src").resolve()))

import numpy as np

from filament_winder.core.geometry.axisymmetric import cylinder_with_domes_profile
from filament_winder.core.path_planning.profile import find_profile_safe_zone
from filament_winder.core.path_planning.geodesic import (
    GeodesicPathConfig, generate_geodesic_path
)

# Build the same mandrel as the demo
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

print(f"Mandrel z-range: {mandrel.start_z_mm:.3f} - {mandrel.end_z_mm:.3f}")
print(f"Mandrel r-range: {float(np.min(mandrel.r_mm)):.3f} - {mandrel.max_radius_mm:.3f}")

# Reproduce _axisymmetric_safe_turnaround_radius logic
# spec.turnaround_radius_mm = 28, winding_type = "geodesic", target_angle = 45
# _tow_edge_clearance_mm = tow_width * 0.5 + thickness * 0.5 = 6*0.5 + 0.25*0.5 = 3.125
min_radius = 28 + 3.125  # = 31.125
clairaut_radius = mandrel.max_radius_mm * math.sin(math.radians(45))
safe_radius = max(min_radius, clairaut_radius)
print(f"\nmir=28, clearance=3.125, min_radius={min_radius:.3f}")
print(f"clairaut_radius=R*sin(45)={clairaut_radius:.3f}")
print(f"safe_radius={safe_radius:.3f}")

# find_profile_safe_zone at safe_radius
try:
    sz = find_profile_safe_zone(mandrel, min_radius_mm=safe_radius)
    print(f"\nSafe zone at r>={safe_radius:.3f}: z=[{sz.start_z_mm:.3f}, {sz.end_z_mm:.3f}]")
    print(f"Safe zone length: {sz.end_z_mm - sz.start_z_mm:.3f}")
    
    # Apply edge_ease_mm
    edge_ease = min(max(6.0 * 2.0, 3.0), (sz.end_z_mm - sz.start_z_mm) * 0.12)
    print(f"edge_ease_mm = {edge_ease:.3f}")
    start_z = sz.start_z_mm + edge_ease
    end_z = sz.end_z_mm - edge_ease
    print(f"After edge_ease: z=[{start_z:.3f}, {end_z:.3f}]")
    print(f"  (matches CSV: z=[32.369, 1207.631])")
except ValueError as e:
    print(f"Safe zone error: {e}")

# Now generate the geodesic path
print(f"\nGenerating geodesic path...")
config = GeodesicPathConfig(
    initial_angle_deg=45,
    tow_width_mm=6.0,
    start_z_mm=start_z if 'start_z' in dir() else 0,
    end_z_mm=end_z if 'end_z' in dir() else 0,
    start_theta_rad=0.0,
    direction="positive",
    turnaround_radius_mm=safe_radius if 'safe_radius' in dir() else 0,
    reference_radius_mm=mandrel.max_radius_mm,
    point_count=140,
)
path, diag = generate_geodesic_path(mandrel, config)
print(f"Generated path: z=[{float(np.min(path.z_mm)):.3f}, {float(np.max(path.z_mm)):.3f}]")
print(f"Diagnostics: theta_total={np.rad2deg(float(diag.theta_total_rad)):.1f} deg")
print(f"Max angle: {float(np.max(diag.winding_angle_deg)):.1f}, Min angle: {float(np.min(diag.winding_angle_deg)):.1f}")
