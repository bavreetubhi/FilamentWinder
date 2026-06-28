"""Debug the generate error."""
from __future__ import annotations
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path("src").resolve()))

import numpy as np

from filament_winder.core.geometry.axisymmetric import cylinder_with_domes_profile
from filament_winder.core.path_planning.profile import find_profile_safe_zone
from filament_winder.core.path_planning.geodesic import (
    GeodesicPathConfig, generate_geodesic_path,
    ControlledAnglePathConfig, generate_controlled_angle_path,
)
from filament_winder.core.path_planning.schedule import (
    _axisymmetric_safe_turnaround_radius,
    _axisymmetric_turnaround_z_bounds,
    WindingLayerSpec,
    _plan_axisymmetric_geodesic_layer,
    _plan_axisymmetric_non_geodesic_layer,
)

# Build isotensoid mandrel
mandrel = cylinder_with_domes_profile(
    cylinder_length_mm=1000,
    cylinder_radius_mm=101.6,
    left_dome_length_mm=120,
    right_dome_length_mm=120,
    polar_opening_radius_mm=25,
    samples_per_region=max(16, 360 // 3),
    dome_shape="isotensoid",
    name="demo",
)
print(f"Mandrel: z=[{mandrel.start_z_mm:.3f}, {mandrel.end_z_mm:.3f}] r=[{float(np.min(mandrel.r_mm)):.3f}, {mandrel.max_radius_mm:.3f}]")

# Test geodesic layer
spec_g = WindingLayerSpec(
    name="geodesic",
    winding_type="geodesic",
    target_angle_deg=45,
    tow_width_mm=6.0,
    layer_thickness_mm=0.25,
    turnaround_radius_mm=28,
    turnaround_points=20,
    turnaround_angle_deg=180,
    point_count=140,
    coverage_target=1.0,
    max_angle_error_deg=5.0,
)

safe_r = _axisymmetric_safe_turnaround_radius(mandrel, spec_g)
print(f"Geodesic safe radius: {safe_r:.3f}")
start_z, end_z, safe_start_z, safe_end_z = _axisymmetric_turnaround_z_bounds(mandrel, spec_g)
print(f"Geodesic bounds: start={start_z:.3f}, end={end_z:.3f}, safe=[{safe_start_z:.3f}, {safe_end_z:.3f}]")
print(f"  in mandrel: {start_z >= mandrel.start_z_mm} and {end_z <= mandrel.end_z_mm}")

# Try generate
try:
    path, report, motion = _plan_axisymmetric_geodesic_layer(mandrel, spec_g, 0.0)
    print(f"Geodesic layer OK: z=[{float(np.min(path.z_mm)):.3f}, {float(np.max(path.z_mm)):.3f}]")
except Exception as e:
    print(f"Geodesic layer FAILED: {e}")

# Test non-geodesic layer
spec_ng = WindingLayerSpec(
    name="non_geodesic",
    winding_type="non_geodesic",
    target_angle_deg=35,
    tow_width_mm=6.0,
    layer_thickness_mm=0.25,
    turnaround_points=20,
    turnaround_angle_deg=180,
    point_count=140,
    coverage_target=1.0,
    max_angle_error_deg=5.0,
)

safe_r = _axisymmetric_safe_turnaround_radius(mandrel, spec_ng)
print(f"\nNon-geodesic safe radius: {safe_r:.3f}")
start_z, end_z, safe_start_z, safe_end_z = _axisymmetric_turnaround_z_bounds(mandrel, spec_ng)
print(f"Non-geodesic bounds: start={start_z:.3f}, end={end_z:.3f}, safe=[{safe_start_z:.3f}, {safe_end_z:.3f}]")
print(f"  in mandrel: {start_z >= mandrel.start_z_mm} and {end_z <= mandrel.end_z_mm}")

# Try generate
try:
    path, report, motion = _plan_axisymmetric_non_geodesic_layer(mandrel, spec_ng, 0.0)
    print(f"Non-geodesic layer OK: z=[{float(np.min(path.z_mm)):.3f}, {float(np.max(path.z_mm)):.3f}]")
except Exception as e:
    print(f"Non-geodesic layer FAILED: {e}")
