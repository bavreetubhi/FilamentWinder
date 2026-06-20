from __future__ import annotations

import numpy as np

from filament_winder.core.coverage import cylinder_coverage_map
from filament_winder.core.geometry import CylinderMandrel
from filament_winder.core.path_planning import HelicalPathConfig, HelicalPathGenerator
from filament_winder.core.tow import generate_cylinder_tow_band


def test_tow_band_generates_two_surface_edges() -> None:
    mandrel = CylinderMandrel(length_mm=1000.0, radius_mm=100.0)
    config = HelicalPathConfig(winding_angle_deg=45.0, tow_width_mm=6.0, point_count=50)
    path = HelicalPathGenerator(mandrel, config).generate()

    tow_band = generate_cylinder_tow_band(mandrel, path)

    assert tow_band.point_count == path.point_count
    assert tow_band.vertices_mm.shape == (100, 3)
    assert tow_band.quad_indices.shape == (49, 4)
    assert np.allclose(np.linalg.norm(tow_band.left_points_mm[:, :2], axis=1), 100.0)
    assert np.allclose(np.linalg.norm(tow_band.right_points_mm[:, :2], axis=1), 100.0)


def test_cylinder_coverage_map_marks_gaps_and_covered_cells() -> None:
    mandrel = CylinderMandrel(length_mm=1000.0, radius_mm=100.0)
    config = HelicalPathConfig(winding_angle_deg=45.0, tow_width_mm=6.0, point_count=200)
    path = HelicalPathGenerator(mandrel, config).generate()

    coverage = cylinder_coverage_map(mandrel, path, z_samples=80, theta_samples=120)

    assert coverage.coverage_count.shape == (80, 120)
    assert 0.0 < coverage.covered_fraction < 0.05
    assert 0.95 < coverage.gap_fraction < 1.0
    assert coverage.overlap_fraction == 0.0


def test_multi_pass_coverage_counts_multiple_passes() -> None:
    mandrel = CylinderMandrel(length_mm=1000.0, radius_mm=100.0)
    config = HelicalPathConfig(
        winding_angle_deg=45.0,
        tow_width_mm=20.0,
        point_count=100,
        passes=4,
    )
    path = HelicalPathGenerator(mandrel, config).generate()

    coverage = cylinder_coverage_map(mandrel, path, z_samples=40, theta_samples=72)
    summary = coverage.summary()

    assert summary.covered_fraction > 0.05
    assert summary.max_coverage_count >= 1
