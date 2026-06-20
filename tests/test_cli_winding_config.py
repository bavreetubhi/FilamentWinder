from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

from filament_winder.cli import main
from filament_winder.config import load_winding_config
from filament_winder.services import (
    generate_winding_job,
    validate_path_csv,
    validate_winding_job_config,
)


def test_parse_config(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)

    config = load_winding_config(config_path)

    assert config.project.name == "test_stack"
    assert config.mandrel.radius_mm == 30.0
    assert len(config.layers) == 3
    assert config.layers[1].passes == "auto"


def test_validate_good_config(tmp_path: Path) -> None:
    config = load_winding_config(_write_config(tmp_path))

    warnings = validate_winding_job_config(config)

    assert warnings == ()


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ("radius_mm: -30", "mandrel.radius_mm"),
        ("width_mm: 0", "tow.width_mm"),
        ("winding_angle_deg: 90\n    direction: forward\n    passes: auto", "helical"),
        ("start_z_mm: 90\n    end_z_mm: 10", "end_z_mm"),
    ],
)
def test_reject_invalid_config_values(
    tmp_path: Path,
    replacement: str,
    message: str,
) -> None:
    config_path = _write_config(tmp_path)
    text = config_path.read_text(encoding="utf-8")
    if replacement.startswith("radius"):
        text = text.replace("radius_mm: 30", replacement)
    elif replacement.startswith("width"):
        text = text.replace("width_mm: 4.0", replacement)
    elif replacement.startswith("winding"):
        text = text.replace(
            "winding_angle_deg: 45\n    direction: forward\n    passes: auto",
            replacement,
        )
    else:
        text = text.replace("start_z_mm: 0\n    end_z_mm: 120", replacement)
    config_path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        validate_winding_job_config(load_winding_config(config_path))


def test_generate_writes_csv_summary_and_plots(tmp_path: Path) -> None:
    config = load_winding_config(_write_config(tmp_path))

    result = generate_winding_job(config)

    assert result.csv_path is not None and result.csv_path.exists()
    assert result.summary_path is not None and result.summary_path.exists()
    assert {
        "path_unwrapped.png",
        "path_3d.png",
        "path_debug_passes.png",
        "path_debug_transitions.png",
    } <= {path.name for path in result.plot_paths}
    assert all(path.read_bytes().startswith(b"\x89PNG") for path in result.plot_paths)
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["enabled_layer_count"] == 3
    assert summary["continuity"]["continuous_machine_path"] is True
    assert summary["continuity"]["large_boundary_jump_count"] == 0
    assert summary["transition_summary"]["transition_count"] > 0
    assert summary["path_validation"]["plot_files_non_empty"] is True
    assert Path(summary["output_files"]["summary"]).exists()
    assert summary["output_files"]["plots"]
    with result.csv_path.open(newline="", encoding="utf-8") as handle:
        first_row = next(csv.DictReader(handle))
    assert first_row["layer_id"]
    assert "time_s" in first_row
    assert "motion_type" in first_row


def test_generated_path_is_continuous_machine_motion(tmp_path: Path) -> None:
    result = generate_winding_job(load_winding_config(_write_config(tmp_path)))

    assert result.summary["continuity"]["large_boundary_jump_count"] == 0
    assert result.summary["continuity"]["continuous_machine_path"] is True


def test_hoop_layer_generates_full_coverage(tmp_path: Path) -> None:
    result = generate_winding_job(load_winding_config(_write_config(tmp_path)))
    hoop_layer = next(layer for layer in result.program.layers if layer.spec.winding_type == "hoop")

    expected_bands = math.ceil(result.mandrel.length_mm / hoop_layer.spec.tow_width_mm)
    assert hoop_layer.report.circuits >= expected_bands
    assert hoop_layer.report.coverage_percent >= 99.0
    assert hoop_layer.report.gap_mm <= 0.25
    assert result.layer_completion_report_path is not None
    layer_completion = json.loads(
        result.layer_completion_report_path.read_text(encoding="utf-8")
    )
    hoop_report = next(
        layer
        for layer in layer_completion["layers"]
        if layer["winding_mode"] == "hoop"
    )
    assert hoop_report["hoop_continuity"]["passed"] is True
    assert hoop_report["hoop_continuity"]["exact_pure_hoop_angle"] is False
    assert hoop_report["continuous_traverse_passed"] is True
    assert hoop_report["turnaround_quality"]["validator"] == "validate_continuous_hoop_traverse"
    assert hoop_report["overlap_percent"] < 5.0


def test_transition_rows_are_reported(tmp_path: Path) -> None:
    result = generate_winding_job(load_winding_config(_write_config(tmp_path)))

    assert result.csv_path is not None
    with result.csv_path.open(newline="", encoding="utf-8") as handle:
        motion_types = {row["motion_type"] for row in csv.DictReader(handle)}
    assert "transition" in motion_types
    assert result.summary["transition_summary"]["transition_points"] > 0


def test_machine_ready_false_when_strict_quality_limit_fails(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    text = config_path.read_text(encoding="utf-8").replace(
        "output:\n  directory:",
        "quality_limits:\n  max_estimated_winding_time_min: 0.001\n\noutput:\n  directory:",
    )
    config_path.write_text(text, encoding="utf-8")

    result = generate_winding_job(load_winding_config(config_path))

    assert result.summary["machine_ready"] is False
    assert (
        result.summary["stack_uniformity_status"]["winding_time_limit_passed"]
        is False
    )
    assert (
        result.summary["manufacturing_report"]["strict_quality"]["stack_uniformity"][
            "winding_time_limit_passed"
        ]
        is False
    )


def test_validate_path_command_reports_machine_valid_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = generate_winding_job(load_winding_config(_write_config(tmp_path)))

    assert result.csv_path is not None
    assert result.summary_path is not None
    validation = validate_path_csv(result.csv_path, summary_path=result.summary_path)

    assert validation.ok
    assert main(
        [
            "validate-path",
            "--path",
            str(result.csv_path),
            "--summary",
            str(result.summary_path),
        ]
    ) == 0
    assert "Result: PASS" in capsys.readouterr().out


def test_plot_can_be_disabled(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    text = config_path.read_text(encoding="utf-8").replace(
        "plot:\n  enabled: true",
        "plot:\n  enabled: false",
    )
    config_path.write_text(text, encoding="utf-8")

    result = generate_winding_job(load_winding_config(config_path))

    assert result.plot_paths == ()


def test_disabled_layer_excludes_from_generation(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    text = config_path.read_text(encoding="utf-8").replace(
        "name: helical_minus_45\n    enabled: true",
        "name: helical_minus_45\n    enabled: false",
    )
    config_path.write_text(text, encoding="utf-8")

    result = generate_winding_job(load_winding_config(config_path))

    assert len(result.program.layers) == 2
    assert all(layer.spec.name != "helical_minus_45" for layer in result.program.layers)


def test_polar_layer_placeholder_does_not_crash(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, include_polar=True)
    config = load_winding_config(config_path)

    warnings = validate_winding_job_config(config)
    result = generate_winding_job(config)

    assert any("limited cylinder polar support" in warning for warning in warnings)
    assert any(layer.spec.winding_type == "polar" for layer in result.program.layers)


def test_cli_validate_and_generate_commands_run(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _write_config(tmp_path)

    assert main(["validate", "--config", str(config_path)]) == 0
    assert "Config valid" in capsys.readouterr().out
    assert main(["generate", "--config", str(config_path)]) == 0
    assert "Project: test_stack" in capsys.readouterr().out


def test_backend_check_command_runs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _write_config(tmp_path)

    status = main(["backend-check", "--config", str(config_path)])
    output = capsys.readouterr().out

    assert status in {0, 1}
    assert "Backend Check" in output
    assert "Region quality:" in output
    assert "Exports:" in output


def test_cli_inspect_coverage_and_export_gcode_run(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _write_config(tmp_path)
    gcode_path = tmp_path / "out" / "path.gcode"

    assert main(["inspect", "--config", str(config_path)]) == 0
    assert "Coverage grid" in capsys.readouterr().out
    assert main(["coverage", "--config", str(config_path)]) == 0
    assert "Coverage Summary" in capsys.readouterr().out
    assert main(
        [
            "export-gcode",
            "--config",
            str(config_path),
            "--output",
            str(gcode_path),
        ]
    ) == 0
    assert gcode_path.exists()
    assert "Wrote G-code" in capsys.readouterr().out


def _write_config(tmp_path: Path, *, include_polar: bool = False) -> Path:
    output_dir = (tmp_path / "out").as_posix()
    polar_block = (
        """
  - name: polar_check
    enabled: true
    type: polar
    winding_angle_deg: 70
    direction: reverse
    passes: 2
    coverage_target: 0.4
    feedrate_mm_min: 250
    start_z_mm: 0
    end_z_mm: 120
    colour: "#22aa66"
    points: 18
"""
        if include_polar
        else ""
    )
    config_path = tmp_path / "stack.yaml"
    config_path.write_text(
        f"""project:
  name: test_stack
  units: mm

machine:
  axis_order: [A, X, Z, B]
  clearance_mm: 15

mandrel:
  type: cylinder
  length_mm: 120
  radius_mm: 30

tow:
  tow_id: carbon
  name: carbon
  width_mm: 4.0
  thickness_mm: 0.2

layers:
  - name: hoop
    enabled: true
    type: hoop
    winding_angle_deg: 90
    direction: forward
    passes: 2
    coverage_target: 1.0
    feedrate_mm_min: 300
    start_z_mm: 0
    end_z_mm: 120
    colour: "#888888"
    points: 24

  - name: helical_plus_45
    enabled: true
    type: helical
    winding_angle_deg: 45
    direction: forward
    passes: auto
    coverage_target: 0.5
    feedrate_mm_min: 300
    start_z_mm: 0
    end_z_mm: 120
    colour: "#cc4444"
    points: 24

  - name: helical_minus_45
    enabled: true
    type: helical
    winding_angle_deg: -45
    direction: reverse
    passes: 3
    phase_offset_deg: 90
    coverage_target: 0.5
    feedrate_mm_min: 300
    start_z_mm: 0
    end_z_mm: 120
    colour: "#4466cc"
    points: 24
{polar_block}
output:
  directory: "{output_dir}"
  csv: true
  summary_json: true
  gcode: false

plot:
  enabled: true
  show: false
  save: true
  formats: [png]
  modes: [unwrapped, three_d, debug_passes, debug_transitions]
  include_2d_unwrapped: true
  include_3d_path: true
""",
        encoding="utf-8",
    )
    return config_path
