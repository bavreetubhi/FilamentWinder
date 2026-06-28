"""Full geodesic layer generation debug."""
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
from filament_winder.core.path_planning.schedule import (
    _plan_axisymmetric_geodesic_layer,
    WindingLayerSpec,
)

# Build mandrel
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

print(f"Mandrel: z=[{mandrel.start_z_mm:.3f}, {mandrel.end_z_mm:.3f}]")

# Create a spec exactly matching geodesic_dome_to_dome_1 layer
spec = WindingLayerSpec(
    name="geodesic_dome_to_dome_1",
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
    start_z_mm=None,
    end_z_mm=None,
    start_offset_deg=0.0,
    phase_offset_deg=None,
    direction="forward",
    number_of_passes=None,
)

# Call the actual planner
path, report, motion = _plan_axisymmetric_geodesic_layer(mandrel, spec, 0.0)

print(f"Path: z=[{float(np.min(path.z_mm)):.3f}, {float(np.max(path.z_mm)):.3f}]")
print(f"Path points: {len(path.z_mm)}")
print(f"Report: actual_angle={report.actual_angle_deg:.3f}")

# Check z-range of each motion type
motion_types = {}
for i, mt in enumerate(motion):
    if mt not in motion_types:
        motion_types[mt] = []
    motion_types[mt].append(i)

for mt, indices in motion_types.items():
    zs = path.z_mm[indices]
    print(f"  {mt}: z=[{float(np.min(zs)):.3f}, {float(np.max(zs)):.3f}] pts={len(indices)}")
