from __future__ import annotations

import csv
import json
from dataclasses import replace
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
    assert result.summary["mandrel"]["min_wind_radius_mm"] == pytest.approx(28.0)
    assert result.summary["continuity"]["continuous_machine_path"] is True
    assert result.summary["path_validation"]["csv_summary_row_count_match"] is True
    assert result.summary["stack_uniformity_status"]["stack_uniformity_passed"] is False
    assert result.summary["stack_uniformity_status"]["covered_percent"] == pytest.approx(100.0)
    assert result.summary["stack_uniformity_status"]["gap_percent"] == pytest.approx(0.0)
    assert (
        result.summary["pattern_optimisation_status"]["pattern_optimisation_passed"]
        is False
    )
    assert (
        result.summary["pattern_optimisation_status"]["invalid_selected_candidate_count"]
        == 0
    )
    assert result.summary["backend_ready"] is False
    assert result.summary["machine_ready"] is False
    assert result.summary["calibration_status"]["calibration_passed"] is False
    assert result.summary["friction_margin_status"]["friction_margin_passed"] is False


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


def test_dome_paths_do_not_insert_hoop_like_turnaround_rings() -> None:
    result = generate_winding_job(
        load_winding_config("examples/demo_domed_pressure_vessel.yaml"),
        make_plots=False,
    )
    report = json.loads(result.validation_report_path.read_text(encoding="utf-8"))
    segment_types = {segment.segment_type for segment in build_path_segments(result.program)}

    assert "dome_turnaround" not in segment_types
    assert "DomeTurnaround" in segment_types
    assert report["machine_validation_summary"]["axis_velocity_limits_checked"] is True
    assert report["machine_validation_summary"]["axis_acceleration_limits_checked"] is True
    assert report["turnaround_summary"]["turnaround_segment_count"] > 0


def test_domed_paths_stay_on_mandrel_surface_and_export_phase_debug() -> None:
    result = generate_winding_job(
        load_winding_config("examples/demo_domed_pressure_vessel.yaml"),
        make_plots=False,
    )

    radius_from_xy = np.hypot(result.program.path.x_mm, result.program.path.y_mm)
    expected_radius = result.mandrel.radius_at(result.program.path.z_mm)
    radial_offset = radius_from_xy - expected_radius
    assert float(np.min(radial_offset)) >= -1e-6
    assert float(np.max(radial_offset)) <= 0.75 + 1e-6

    assert result.csv_path is not None
    with result.csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        assert reader.fieldnames is not None
        assert {"winding_index", "phase_angle_deg", "circuit_index"} <= set(
            reader.fieldnames
        )
    assert len(rows) == result.program.point_count
    assert all(None not in row for row in rows)
    assert result.summary["path_validation"]["csv_summary_row_count_match"] is True


def test_dome_coverage_and_polar_reports_must_pass_dedicated_gates() -> None:
    result = generate_winding_job(
        load_winding_config("examples/demo_domed_pressure_vessel.yaml"),
        make_plots=False,
    )

    assert result.dome_coverage_report_path is not None
    dome = json.loads(result.dome_coverage_report_path.read_text(encoding="utf-8"))
    assert dome["summary"]["pin_layout_enabled"] is False
    assert dome["summary"]["source"] == "axisymmetric_dome_surface_path"
    assert dome["summary"]["configured_min_wind_radius_mm"] == pytest.approx(28.0)
    assert dome["summary"]["detected_dome_surface_point_count"] > 0
    assert result.summary["coverage_summary"]["gap_percent"] == pytest.approx(0.0)
    assert dome["summary"]["max_gap_mm"] == pytest.approx(0.0)
    assert dome["summary"]["dome_coverage_passed"] is True
    assert dome["summary"]["left_dome_coverage_passed"] is True
    assert dome["summary"]["right_dome_coverage_passed"] is True
    assert dome["summary"]["invalid_zero_or_infinite_coverage_metrics"] is False
    assert result.summary["backend_ready"] is False
    assert result.summary["machine_ready"] is False
    non_geodesic = next(
        layer for layer in result.program.layers if layer.spec.winding_type == "non_geodesic"
    )
    assert non_geodesic.report.gap_mm <= 0.25
    assert non_geodesic.report.coverage_percent >= 99.0

    assert result.left_dome_coverage_report_path is not None
    assert result.right_dome_coverage_report_path is not None
    left = json.loads(result.left_dome_coverage_report_path.read_text(encoding="utf-8"))
    right = json.loads(result.right_dome_coverage_report_path.read_text(encoding="utf-8"))
    for report in (left, right):
        summary = report["summary"]
        assert summary["local_winding_angle_mean_deg"] == pytest.approx(
            summary["measured_shell_winding_angle_mean_deg"]
        )
        assert summary["boss_transition_validation"]["passed"] is True
        assert summary["surface_band_conformance_validation"]["passed"] is True
        assert summary["boss_contact_point_count"] >= 0
        assert summary["deposited_shell_point_count"] > 0
        assert summary["covered_area_percentage"] == pytest.approx(100.0)
        assert summary["maximum_uncovered_gap_mm"] == pytest.approx(0.0)
        assert summary["minimum_shell_radius_mm"] >= 28.0 - 1e-6
        assert abs(
            summary["local_winding_angle_mean_deg"]
            - summary["target_winding_angle_deg"]
        ) <= 15.0

    assert result.polar_overbuild_report_path is not None
    polar = json.loads(result.polar_overbuild_report_path.read_text(encoding="utf-8"))
    assert polar["summary"]["polar_overbuild_passed"] is True
    assert polar["summary"]["physical_boss_excluded_from_required_shell"] is True
    assert polar["summary"]["physical_boss_buildup_mm"] > polar["summary"]["polar_buildup_mm"]


def test_domed_job_exports_boss_specific_diagnostics(tmp_path: Path) -> None:
    config = load_winding_config("examples/demo_domed_pressure_vessel.yaml")
    config = replace(
        config,
        output=replace(config.output, directory=tmp_path / "out"),
        plot=replace(
            config.plot,
            enabled=True,
            save=True,
            show=False,
            modes=("unwrapped", "three_d"),
        ),
    )

    result = generate_winding_job(config)

    assert any(path.name == "dome_shell_only_unwrapped.png" for path in result.plot_paths)
    assert any(path.name == "dome_boss_contact_unwrapped.png" for path in result.plot_paths)
    assert any(path.name == "dome_transition_moves_unwrapped.png" for path in result.plot_paths)
    assert any(path.name == "dome_boss_closeup_left.png" for path in result.plot_paths)
    assert any(path.name == "dome_boss_closeup_right.png" for path in result.plot_paths)
    assert result.dome_coverage_report_path is not None
    dome = json.loads(result.dome_coverage_report_path.read_text(encoding="utf-8"))
    assert dome["summary"]["boss_transition_validation_passed"] is True
    assert dome["summary"]["surface_band_conformance_passed"] is True
    assert dome["summary"]["boss_transition_validation_by_side"]["left"]["passed"] is True
    assert dome["summary"]["boss_transition_validation_by_side"]["right"]["passed"] is True
    assert dome["summary"]["surface_band_conformance_by_side"]["left"]["passed"] is True
    assert dome["summary"]["surface_band_conformance_by_side"]["right"]["passed"] is True
    assert dome["summary"]["boss_contact_point_count"] >= 0


def test_invalid_min_wind_radius_is_rejected(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    text = config_path.read_text(encoding="utf-8").replace(
        "radius_mm: 30\n",
        "radius_mm: 30\n  min_wind_radius_mm: 30\n",
        1,
    )
    config_path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="min_wind_radius_mm"):
        generate_winding_job(load_winding_config(config_path))


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
