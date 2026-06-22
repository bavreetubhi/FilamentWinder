"""Fast mandrel/layer geometry debug runner.

Edit the constants below, then run:
    python examples/rapid_geometry_debug.py
"""

from __future__ import annotations

from filament_winder.core.geometry import CylinderMandrel, cylinder_with_domes_profile
from filament_winder.core.path_planning import (
    WindingLayerSpec,
    WindingSchedule,
    plan_winding_schedule,
)

USE_DOMED_PROFILE = False

LENGTH_MM = 1000.0
RADIUS_MM = 100.0
TOW_WIDTH_MM = 6.0
POINTS_PER_PASS = 500

LAYERS = (
    WindingLayerSpec(
        name="helical-debug",
        winding_type="helical",
        target_angle_deg=45.0,
        tow_width_mm=TOW_WIDTH_MM,
        coverage_target=1.0,
        point_count=POINTS_PER_PASS,
        number_of_passes=None,
    ),

)


def build_mandrel():
    if not USE_DOMED_PROFILE:
        return CylinderMandrel(length_mm=LENGTH_MM, radius_mm=RADIUS_MM)
    return cylinder_with_domes_profile(
        cylinder_length_mm=LENGTH_MM,
        cylinder_radius_mm=RADIUS_MM,
        left_dome_length_mm=120.0,
        right_dome_length_mm=120.0,
        polar_opening_radius_mm=25.0,
        samples_per_region=32,
    )


def main() -> int:
    mandrel = build_mandrel()
    program = plan_winding_schedule(
        mandrel,
        WindingSchedule(layers=LAYERS, radial_clearance_mm=25.0),
    )
    print(f"mandrel={type(mandrel).__name__} length={mandrel.length_mm:.3f} mm")
    print(f"program points={program.point_count} layers={len(program.layers)}")
    for report in program.reports:
        print(
            f"{report.layer_name}: angle={report.actual_angle_deg:.3f} deg "
            f"passes={report.circuits} coverage={report.coverage_percent:.3f}% "
            f"gap={report.gap_mm:.4f} mm overlap={report.overlap_mm:.4f} mm"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
