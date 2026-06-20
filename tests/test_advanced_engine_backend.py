from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from filament_winder.config import load_winding_config
from filament_winder.core.geometry import (
    CylinderMandrel,
    classify_regions,
    cylinder_with_domes_profile,
)
from filament_winder.core.path_planning import (
    ControlledAnglePathConfig,
    GeodesicPathConfig,
    build_path_segments,
    generate_controlled_angle_path,
    generate_geodesic_path,
)
from filament_winder.services import generate_winding_job


def test_axisymmetric_cylinder_surface_utilities() -> None:
    mandrel = CylinderMandrel(length_mm=100.0, radius_mm=25.0)
    z = np.asarray([0.0, 50.0, 100.0])
    theta = np.asarray([0.0, np.pi / 2.0, np.pi])

    assert np.allclose(mandrel.dr_dz_at(z), 0.0)
    assert np.allclose(mandrel.meridional_arc_length_at(z), z)
    assert np.allclose(np.linalg.norm(mandrel.surface_normal(z, theta), axis=1), 1.0)
    assert np.allclose(mandrel.circumferential_curvature_at(z), 1.0 / 25.0)


def test_cylinder_with_domes_profile_regions_and_normals() -> None:
    mandrel = cylinder_with_domes_profile(
        cylinder_length_mm=200.0,
        cylinder_radius_mm=50.0,
        left_dome_length_mm=60.0,
        right_dome_length_mm=60.0,
        polar_opening_radius_mm=8.0,
        samples_per_region=32,
    )
    regions = classify_regions(mandrel, polar_opening_radius_mm=8.0)

    assert "left_dome" in regions
    assert "cylinder" in regions
    assert "right_dome" in regions
    assert mandrel.max_radius_mm == pytest.approx(50.0)
    normals = mandrel.surface_normal(mandrel.z_mm[::10], np.zeros(mandrel.z_mm[::10].shape))
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0)
    assert np.all(np.diff(mandrel.meridional_arc_length_at(mandrel.z_mm)) >= 0.0)


def test_geodesic_clairaut_constant_on_domed_profile() -> None:
    mandrel = cylinder_with_domes_profile(
        cylinder_length_mm=200.0,
        cylinder_radius_mm=50.0,
        left_dome_length_mm=60.0,
        right_dome_length_mm=60.0,
        polar_opening_radius_mm=10.0,
        samples_per_region=48,
    )
    path, diagnostics = generate_geodesic_path(
        mandrel,
        GeodesicPathConfig(
            initial_angle_deg=30.0,
            tow_width_mm=3.0,
            start_z_mm=60.0,
            end_z_mm=260.0,
            point_count=120,
            turnaround_radius_mm=12.0,
        ),
    )
    clairaut = mandrel.radius_at(path.z_mm) * np.sin(np.deg2rad(path.tow_eye_angle_deg))

    assert path.point_count == 120
    assert diagnostics.warning_flags == ()
    assert float(np.max(np.abs(clairaut - diagnostics.clairaut_constant_mm))) < 1e-6


def test_controlled_angle_reports_slip_risk() -> None:
    mandrel = cylinder_with_domes_profile(
        cylinder_length_mm=80.0,
        cylinder_radius_mm=40.0,
        left_dome_length_mm=60.0,
        right_dome_length_mm=60.0,
        polar_opening_radius_mm=5.0,
        samples_per_region=48,
    )
    _, diagnostics = generate_controlled_angle_path(
        mandrel,
        ControlledAnglePathConfig(
            target_angle_deg=55.0,
            tow_width_mm=3.0,
            start_z_mm=60.0,
            end_z_mm=180.0,
            point_count=100,
            high_slip_risk_deg=10.0,
        ),
    )

    assert diagnostics.max_slip_risk_deg > 0.0
    assert diagnostics.warning_flags


def test_config_job_exports_segments_validation_and_coverage(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    result = generate_winding_job(load_winding_config(config_path))

    assert result.segments_path is not None and result.segments_path.exists()
    assert result.validation_report_path is not None and result.validation_report_path.exists()
    assert result.coverage_grid_path is not None and result.coverage_grid_path.exists()
    assert result.summary["total_segments"] > 0
    assert result.summary["output_files"]["segments"]

    segments = json.loads(result.segments_path.read_text(encoding="utf-8"))
    assert segments[0]["segment_id"].startswith("seg-")
    assert {segment["segment_type"] for segment in segments} >= {
        "hoop_pass",
        "helical_pass",
    }
    assert build_path_segments(result.program)


def test_domed_pressure_vessel_config_generates() -> None:
    config = load_winding_config("examples/demo_domed_pressure_vessel.yaml")

    result = generate_winding_job(config, make_plots=False)

    assert result.csv_path is not None and result.csv_path.exists()
    assert result.gcode_path is not None and result.gcode_path.exists()
    assert result.segments_path is not None and result.segments_path.exists()
    assert result.validation_report_path is not None and result.validation_report_path.exists()
    assert result.coverage_grid_path is not None and result.coverage_grid_path.exists()
    assert result.summary["mandrel"]["type"] == "cylinder_with_elliptical_domes"
    assert result.summary["continuity"]["continuous_machine_path"] is True


def test_unwrapped_a_axis_no_reset_on_domed_job() -> None:
    result = generate_winding_job(
        load_winding_config("examples/demo_domed_pressure_vessel.yaml"),
        make_plots=False,
    )
    a_delta = np.diff(result.program.motion_table.a_deg)

    assert np.count_nonzero(a_delta < -180.0) == 0
    assert np.max(np.abs(a_delta)) < 180.0


def test_path_segments_have_tow_and_process_state() -> None:
    result = generate_winding_job(
        load_winding_config("examples/demo_domed_pressure_vessel.yaml"),
        make_plots=False,
    )
    segments = json.loads(result.segments_path.read_text(encoding="utf-8"))

    assert all(segment["tow_state"] == "on" for segment in segments)
    assert {segment["process_state"] for segment in segments} >= {"winding", "transition"}


def test_dome_turnaround_segment_exists_and_validation_checks_limits() -> None:
    result = generate_winding_job(
        load_winding_config("examples/demo_domed_pressure_vessel.yaml"),
        make_plots=False,
    )
    report = json.loads(result.validation_report_path.read_text(encoding="utf-8"))
    segment_types = {segment.segment_type for segment in build_path_segments(result.program)}

    assert "dome_turnaround" in segment_types
    assert report["machine_validation_summary"]["axis_velocity_limits_checked"] is True
    assert report["machine_validation_summary"]["axis_acceleration_limits_checked"] is True
    assert report["turnaround_summary"]["turnaround_segment_count"] > 0


def _write_config(tmp_path: Path) -> Path:
    output_dir = (tmp_path / "out").as_posix()
    config_path = tmp_path / "job.yaml"
    config_path.write_text(
        f"""project:
  name: advanced_backend
  units: mm

machine:
  clearance_mm: 15

mandrel:
  type: cylinder
  length_mm: 120
  radius_mm: 30

tow:
  width_mm: 4.0
  thickness_mm: 0.2

layers:
  - name: hoop
    type: hoop
    winding_angle_deg: 90
    passes: 1
    points: 18
  - name: helical
    type: helical
    winding_angle_deg: 45
    passes: 3
    points: 18

coverage:
  z_cells: 24
  theta_cells: 36

output:
  directory: "{output_dir}"
  csv: true
  summary_json: true
  segments_json: true
  validation_report_json: true
  coverage_grid: true

plot:
  enabled: false
""",
        encoding="utf-8",
    )
    return config_path
