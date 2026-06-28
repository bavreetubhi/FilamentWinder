from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from filament_winder.cli import main
from filament_winder.config import load_winding_config
from filament_winder.core.geometry import CylinderMandrel, cylinder_with_domes_profile
from filament_winder.core.path_planning import (
    GeodesicPathConfig,
    PatternSearchRequest,
    effective_roving_width_mm,
    generate_geodesic_path,
    required_number_of_windings,
    select_winding_pattern,
)
from filament_winder.services import generate_winding_job
from filament_winder.services.winding_job import _pattern_request_for_layer


def test_leading_pattern_integer_condition() -> None:
    result = select_winding_pattern(
        _request(delta_phi_total_deg=(360.0 + 360.0 / 7.0) / 3.0, radius_mm=2.0),
        CylinderMandrel(length_mm=50.0, radius_mm=2.0),
    )

    candidate = next(item for item in result.candidates if item.pattern_type == "leading")
    assert (candidate.p + 1) * candidate.k * candidate.d - candidate.nd == 1
    assert candidate.closure_error_deg <= 0.5


def test_lagging_pattern_integer_condition() -> None:
    result = select_winding_pattern(
        _request(delta_phi_total_deg=(360.0 - 360.0 / 7.0) / 3.0, radius_mm=2.0),
        CylinderMandrel(length_mm=40.0, radius_mm=2.0),
    )

    candidate = next(item for item in result.candidates if item.pattern_type == "lagging")
    assert candidate.pattern_type == "lagging"
    assert candidate.p * candidate.k * candidate.d - candidate.nd == -1


def test_reject_repeated_gcd_pattern() -> None:
    result = select_winding_pattern(
        _request(delta_phi_total_deg=(360.0 + 18.0) / 6.0, radius_mm=5.0),
        CylinderMandrel(length_mm=50.0, radius_mm=5.0),
    )

    assert any(
        candidate.p == 6
        and candidate.k == 3
        and "repeated_gcd_pattern" in candidate.rejection_reasons
        for candidate in result.rejected
    )


def test_effective_roving_width_changes_with_angle() -> None:
    assert effective_roving_width_mm(6.0, 45.0) > effective_roving_width_mm(6.0, 10.0)
    assert effective_roving_width_mm(6.0, 90.0) == pytest.approx(6.0)


def test_required_number_of_windings_from_roving_dimensions() -> None:
    required = required_number_of_windings(
        equatorial_radius_mm=10.0,
        effective_roving_width_mm=5.0,
        target_coverage=1.0,
        roving_thickness_mm=0.25,
        target_layer_thickness_mm=0.5,
    )

    assert required == math.ceil(2.0 * math.pi * 10.0 / 5.0 * 2.0)


def test_pattern_candidates_sorted_by_score() -> None:
    result = select_winding_pattern(
        _request(delta_phi_total_deg=202.5, radius_mm=5.0),
        CylinderMandrel(length_mm=50.0, radius_mm=5.0),
    )

    scores = [candidate.score for candidate in result.candidates]
    assert scores == sorted(scores)


def test_pattern_rejects_large_closure_error() -> None:
    result = select_winding_pattern(
        _request(delta_phi_total_deg=17.3, radius_mm=5.0, angle_tolerance_deg=0.001),
        CylinderMandrel(length_mm=50.0, radius_mm=5.0),
    )

    assert result.selected is not None
    assert not result.selected.valid
    assert result.rejection_counts["closure_error"] > 0


def test_pattern_rejects_excessive_coverage_candidate() -> None:
    result = select_winding_pattern(
        PatternSearchRequest(
            layer_id="layer",
            layer_name="layer",
            winding_type="helical",
            winding_angle_deg=45.0,
            delta_phi_total_deg=202.5,
            equatorial_radius_mm=12.0,
            trajectory_length_mm=100.0,
            roving_width_mm=3.0,
            roving_thickness_mm=0.25,
            target_coverage=0.2,
            feedrate_mm_min=500.0,
            max_p=12,
            max_k=6,
            max_d=3,
            max_coverage_estimate=0.21,
            candidate_count=10,
        ),
        CylinderMandrel(length_mm=50.0, radius_mm=12.0),
    )

    assert result.rejection_counts["excessive_coverage"] > 0


def test_pattern_score_penalises_high_overlap_and_time() -> None:
    low_overlap = select_winding_pattern(
        _request(delta_phi_total_deg=202.5, radius_mm=8.0),
        CylinderMandrel(length_mm=50.0, radius_mm=8.0),
    )
    high_overlap_request = replace(
        _request(delta_phi_total_deg=202.5, radius_mm=8.0),
        target_coverage=0.25,
        max_coverage_estimate=2.0,
        feedrate_mm_min=100.0,
    )
    high_overlap = select_winding_pattern(
        high_overlap_request,
        CylinderMandrel(length_mm=50.0, radius_mm=8.0),
    )

    assert low_overlap.selected is not None
    assert high_overlap.selected is not None
    assert high_overlap.selected.score > low_overlap.selected.score


def test_cylinder_helical_layers_bias_pattern_search_toward_cylinder_coverage(
    tmp_path: Path,
) -> None:
    config = load_winding_config(_write_textbook_config(tmp_path))
    mandrel = CylinderMandrel(
        length_mm=config.mandrel.length_mm,
        radius_mm=config.mandrel.radius_mm,
    )
    layer = next(item for item in config.layers if item.type == "helical")

    request = _pattern_request_for_layer(
        config,
        mandrel,
        layer,
        1,
        stack_pair_count=1,
        coverage_share=1.0,
    )

    assert request.target_coverage >= 0.95
    assert request.prefer_full_cylinder_coverage is True
    assert request.undercoverage_weight > request.overlap_weight
    assert request.thickness_weight < 1.0
    assert request.polar_buildup_weight < 1.0


def test_geodesic_angle_changes_on_dome() -> None:
    mandrel = cylinder_with_domes_profile(
        cylinder_length_mm=80.0,
        cylinder_radius_mm=30.0,
        left_dome_length_mm=40.0,
        right_dome_length_mm=40.0,
        polar_opening_radius_mm=8.0,
        samples_per_region=36,
        dome_shape="spherical",
    )
    path, _diagnostics = generate_geodesic_path(
        mandrel,
        GeodesicPathConfig(
            initial_angle_deg=25.0,
            tow_width_mm=3.0,
            start_z_mm=20.0,
            end_z_mm=140.0,
            point_count=120,
            turnaround_radius_mm=10.0,
        ),
    )

    assert np.ptp(path.tow_eye_angle_deg) > 2.0


def test_selected_pattern_generates_closed_layer_and_exports(tmp_path: Path) -> None:
    config_path = _write_textbook_config(tmp_path)

    result = generate_winding_job(load_winding_config(config_path), make_plots=False)

    selected = result.summary["textbook_pattern_selection"]["selected_patterns"][0]
    helical_layer = next(
        layer for layer in result.program.layers if layer.spec.winding_type == "helical"
    )
    assert helical_layer.report.circuits == selected["nd"]
    assert helical_layer.report.angular_shift_deg == pytest.approx(360.0 / selected["nd"])
    assert result.pattern_candidates_path is not None and result.pattern_candidates_path.exists()
    assert result.selected_pattern_path is not None and result.selected_pattern_path.exists()
    assert (
        result.pattern_rejection_report_path is not None
        and result.pattern_rejection_report_path.exists()
    )
    assert (
        result.thickness_distribution_path is not None
        and result.thickness_distribution_path.exists()
    )
    assert (
        result.layer_completion_report_path is not None
        and result.layer_completion_report_path.exists()
    )
    assert (
        result.stack_coverage_report_path is not None
        and result.stack_coverage_report_path.exists()
    )
    assert (
        result.machine_smoothing_report_path is not None
        and result.machine_smoothing_report_path.exists()
    )
    assert (
        result.pattern_optimisation_report_path is not None
        and result.pattern_optimisation_report_path.exists()
    )
    assert (
        result.candidate_pair_report_path is not None
        and result.candidate_pair_report_path.exists()
    )
    assert (
        result.actual_thickness_report_path is not None
        and result.actual_thickness_report_path.exists()
    )
    assert (
        result.region_quality_report_path is not None
        and result.region_quality_report_path.exists()
    )
    assert (
        result.optimisation_repair_suggestions_path is not None
        and result.optimisation_repair_suggestions_path.exists()
    )
    assert "thickness_summary" in selected
    assert "manufacturing_report" in result.summary


def test_coverage_is_validation_not_primary_generation(tmp_path: Path) -> None:
    config_path = _write_textbook_config(tmp_path)
    result = generate_winding_job(load_winding_config(config_path), make_plots=False)

    selected = result.summary["textbook_pattern_selection"]["selected_patterns"][0]
    assert selected["nd"] != math.ceil(
        2.0 * math.pi * result.mandrel.radius_mm / result.config.tow.width_mm
    )
    assert result.summary["output_files"]["coverage_grid"]


def test_patterns_cli_outputs_candidates(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _write_textbook_config(tmp_path)

    assert main(["patterns", "--config", str(config_path)]) == 0

    output = capsys.readouterr().out
    assert "Pattern Candidates" in output
    assert "Best for helical" in output


def test_inspect_pattern_cli_outputs_candidate_details(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _write_textbook_config(tmp_path)
    result = generate_winding_job(load_winding_config(config_path), make_plots=False)
    selected_path = result.selected_pattern_path
    assert selected_path is not None
    candidate_id = json.loads(selected_path.read_text(encoding="utf-8"))[0]["pattern_id"]

    assert main(["inspect-pattern", "--config", str(config_path), "--candidate", candidate_id]) == 0

    assert "Closure error" in capsys.readouterr().out


def test_inspect_layer_cli_outputs_completion(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _write_textbook_config(tmp_path)

    assert main(["inspect-layer", "--config", str(config_path), "--layer", "helical"]) == 0

    output = capsys.readouterr().out
    assert "Layer:" in output
    assert "Completion:" in output


def test_plot_layers_cli_writes_manifest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _write_textbook_config(tmp_path)

    assert main(["plot-layers", "--config", str(config_path)]) == 0

    output = capsys.readouterr().out
    assert "plot_manifest.json" in output
    assert (tmp_path / "out" / "layer_01_helical_unwrapped.png").exists()


def test_repair_suggestions_created_when_no_valid_pattern(tmp_path: Path) -> None:
    config_path = _write_textbook_config(tmp_path)
    text = config_path.read_text(encoding="utf-8").replace(
        "angle_tolerance_deg: 0.5",
        "angle_tolerance_deg: 0.0001",
    )
    config_path.write_text(text, encoding="utf-8")

    result = generate_winding_job(load_winding_config(config_path), make_plots=False)

    assert result.optimisation_repair_suggestions_path is not None
    suggestions = json.loads(
        result.optimisation_repair_suggestions_path.read_text(encoding="utf-8")
    )
    assert suggestions["summary"]["suggestion_count"] > 0


def _request(
    *,
    delta_phi_total_deg: float,
    radius_mm: float,
    angle_tolerance_deg: float = 0.5,
) -> PatternSearchRequest:
    return PatternSearchRequest(
        layer_id="layer",
        layer_name="layer",
        winding_type="helical",
        winding_angle_deg=45.0,
        delta_phi_total_deg=delta_phi_total_deg,
        equatorial_radius_mm=radius_mm,
        trajectory_length_mm=100.0,
        roving_width_mm=3.0,
        roving_thickness_mm=0.25,
        target_coverage=1.0,
        feedrate_mm_min=500.0,
        max_p=12,
        max_k=6,
        max_d=3,
        angle_tolerance_deg=angle_tolerance_deg,
        candidate_count=10,
    )


def _write_textbook_config(tmp_path: Path) -> Path:
    output_dir = (tmp_path / "out").as_posix()
    length = math.radians(202.5) * 4.0
    config_path = tmp_path / "textbook.yaml"
    config_path.write_text(
        f"""project:
  name: textbook_test
  units: mm

machine:
  clearance_mm: 10

mandrel:
  type: cylinder
  length_mm: {length}
  radius_mm: 4.0

tow:
  width_mm: 3.0
  thickness_mm: 0.25

roving:
  width_mm: 3.0
  thickness_mm: 0.25
  fiber_volume_fraction: 0.5
  resin_factor: auto

pattern_selection:
  method: textbook_integer_closure
  max_p: 12
  max_k: 6
  max_d: 3
  angle_tolerance_deg: 0.5
  require_gcd_clean_pattern: true
  candidate_count: 10

laminate_targets:
  mode: simplified
  target_layer_thickness_mm: 0.25

layers:
  - name: helical
    type: helical
    winding_angle_deg: 45
    passes: auto
    coverage_target: 1.0
    feedrate_mm_min: 500
    points: 24

coverage:
  z_cells: 12
  theta_cells: 18

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
