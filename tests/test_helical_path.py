from __future__ import annotations

import math

import numpy as np
import pytest

from filament_winder.core.geometry import CylinderMandrel
from filament_winder.core.kinematics import machine_path_from_surface_path
from filament_winder.core.path_planning import (
    HelicalPathConfig,
    HelicalPathGenerator,
    estimate_cylinder_pattern_closure,
)


def test_cylinder_helix_matches_expected_final_rotation() -> None:
    mandrel = CylinderMandrel(length_mm=1000.0, radius_mm=100.0)
    config = HelicalPathConfig(winding_angle_deg=45.0, tow_width_mm=6.0, point_count=101)

    path = HelicalPathGenerator(mandrel, config).generate()

    expected_theta_rad = math.tan(math.radians(45.0)) / 100.0 * 1000.0
    assert path.theta_rad[-1] == pytest.approx(expected_theta_rad)
    assert path.final_rotation_deg == pytest.approx(math.degrees(expected_theta_rad))
    assert path.final_turns == pytest.approx(math.degrees(expected_theta_rad) / 360.0)


def test_generated_surface_points_stay_on_cylinder() -> None:
    mandrel = CylinderMandrel(length_mm=750.0, radius_mm=80.0)
    config = HelicalPathConfig(winding_angle_deg=35.0, tow_width_mm=4.0, point_count=50)

    path = HelicalPathGenerator(mandrel, config).generate()

    assert path.z_mm[0] == pytest.approx(0.0)
    assert path.z_mm[-1] == pytest.approx(750.0)
    assert np.allclose(path.surface_radius_mm, 80.0)
    assert path.points_mm.shape == (50, 3)


def test_machine_mapping_uses_a_x_z_b_convention() -> None:
    mandrel = CylinderMandrel(length_mm=500.0, radius_mm=60.0)
    config = HelicalPathConfig(winding_angle_deg=30.0, tow_width_mm=6.0, point_count=25)
    path = HelicalPathGenerator(mandrel, config).generate()

    motion = machine_path_from_surface_path(path, radial_clearance_mm=12.5)

    assert np.allclose(motion.a_deg, path.theta_deg)
    assert np.allclose(motion.x_mm, 72.5)
    assert np.allclose(motion.z_mm, path.z_mm)
    assert np.allclose(motion.b_deg, 30.0)


def test_multi_pass_helix_alternates_direction_and_phase() -> None:
    mandrel = CylinderMandrel(length_mm=100.0, radius_mm=20.0)
    config = HelicalPathConfig(
        winding_angle_deg=45.0,
        tow_width_mm=3.0,
        point_count=5,
        passes=3,
    )

    path = HelicalPathGenerator(mandrel, config).generate()

    assert path.point_count == 15
    assert path.pass_count == 3
    assert path.pass_index.tolist() == [0] * 5 + [1] * 5 + [2] * 5
    assert path.z_mm[:5].tolist() == pytest.approx([0.0, 25.0, 50.0, 75.0, 100.0])
    assert path.z_mm[5:10].tolist() == pytest.approx([100.0, 75.0, 50.0, 25.0, 0.0])
    assert np.rad2deg(path.theta_rad[5]) == pytest.approx(120.0)


def test_pattern_closure_estimate_reports_turn_error_and_band_spacing() -> None:
    mandrel = CylinderMandrel(length_mm=1000.0, radius_mm=100.0)
    config = HelicalPathConfig(
        winding_angle_deg=45.0,
        tow_width_mm=6.0,
        point_count=100,
        passes=4,
    )

    closure = estimate_cylinder_pattern_closure(mandrel, config)

    assert closure.rotations_per_pass == pytest.approx(10.0 / (2.0 * math.pi))
    assert closure.nearest_integer_turns == 2
    assert closure.phase_offset_deg == pytest.approx(90.0)
    assert closure.band_spacing_mm == pytest.approx(
        2.0 * math.pi * 100.0 * math.cos(math.radians(45.0)) / 4.0
    )
