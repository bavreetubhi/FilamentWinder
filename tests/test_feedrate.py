from __future__ import annotations

import numpy as np
import pytest

from filament_winder.core.feedrate import FeedrateConfig, plan_feedrate
from filament_winder.core.geometry import CylinderMandrel
from filament_winder.core.kinematics import machine_path_from_surface_path
from filament_winder.core.path_planning import HelicalPathConfig, HelicalPathGenerator, SurfacePath
from filament_winder.io import GCodeOptions, export_gcode


def test_feedrate_plan_slows_tight_curvature_and_reports_slip_risk() -> None:
    theta = np.linspace(0.0, np.pi, 25)
    radius_mm = 5.0
    path = SurfacePath(
        z_mm=np.zeros(theta.shape),
        theta_rad=theta,
        x_mm=radius_mm * np.cos(theta),
        y_mm=radius_mm * np.sin(theta),
        winding_angle_deg=45.0,
        tow_width_mm=3.0,
    )

    schedule = plan_feedrate(
        path,
        FeedrateConfig(
            nominal_feedrate_mm_min=1000.0,
            minimum_feedrate_mm_min=250.0,
            curvature_slowdown_radius_mm=10.0,
            slip_slowdown_threshold=0.2,
            slip_max_risk=0.6,
        ),
    )

    assert schedule.min_feedrate_mm_min < schedule.max_feedrate_mm_min
    assert schedule.min_feedrate_mm_min == pytest.approx(250.0)
    assert schedule.max_slip_risk > 0.2
    assert schedule.min_curvature_radius_mm == pytest.approx(radius_mm, rel=0.2)


def test_gcode_export_can_use_per_point_feed_schedule(tmp_path) -> None:
    mandrel = CylinderMandrel(length_mm=100.0, radius_mm=20.0)
    config = HelicalPathConfig(winding_angle_deg=45.0, tow_width_mm=3.0, point_count=8)
    path = HelicalPathGenerator(mandrel, config).generate()
    motion = machine_path_from_surface_path(path, radial_clearance_mm=5.0)
    schedule = plan_feedrate(
        path,
        FeedrateConfig(
            nominal_feedrate_mm_min=500.0,
            minimum_feedrate_mm_min=125.0,
            curvature_slowdown_radius_mm=50.0,
        ),
    )

    output_path = export_gcode(
        motion,
        tmp_path / "path.gcode",
        options=GCodeOptions(feedrate_mm_min=500.0, feed_schedule=schedule),
    )

    text = output_path.read_text(encoding="utf-8")
    assert f"F{schedule.min_feedrate_mm_min:.3f}" in text
    assert f"F{schedule.max_feedrate_mm_min:.3f}" in text
