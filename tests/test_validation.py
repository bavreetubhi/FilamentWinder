from __future__ import annotations

from filament_winder.core.geometry import CylinderMandrel
from filament_winder.core.kinematics import machine_path_from_surface_path
from filament_winder.core.path_planning import HelicalPathConfig, HelicalPathGenerator
from filament_winder.core.validation import AxisLimitConfig, NoGoZone, validate_motion_table


def test_validation_passes_when_path_stays_inside_limits() -> None:
    mandrel = CylinderMandrel(length_mm=100.0, radius_mm=20.0)
    config = HelicalPathConfig(winding_angle_deg=45.0, tow_width_mm=3.0, point_count=5)
    path = HelicalPathGenerator(mandrel, config).generate()
    motion = machine_path_from_surface_path(path, radial_clearance_mm=5.0)

    report = validate_motion_table(
        motion,
        limits=AxisLimitConfig(x_min_mm=0.0, x_max_mm=30.0, z_min_mm=0.0, z_max_mm=100.0),
    )

    assert not report.has_errors
    assert report.error_count == 0


def test_validation_reports_axis_limits_and_no_go_zones() -> None:
    mandrel = CylinderMandrel(length_mm=100.0, radius_mm=20.0)
    config = HelicalPathConfig(winding_angle_deg=45.0, tow_width_mm=3.0, point_count=5)
    path = HelicalPathGenerator(mandrel, config).generate()
    motion = machine_path_from_surface_path(path, radial_clearance_mm=5.0)

    report = validate_motion_table(
        motion,
        limits=AxisLimitConfig(x_max_mm=10.0),
        no_go_zones=(
            NoGoZone(
                "fixture",
                x_min_mm=20.0,
                x_max_mm=30.0,
                z_min_mm=0.0,
                z_max_mm=5.0,
            ),
        ),
    )

    assert report.has_errors
    assert {issue.code for issue in report.issues} == {"X_LIMIT", "NO_GO_ZONE"}
