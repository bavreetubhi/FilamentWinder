from __future__ import annotations

from pathlib import Path

from filament_winder.cli import main
from filament_winder.project import load_project


def test_cylinder_cli_can_write_all_optional_outputs(tmp_path: Path) -> None:
    csv_path = tmp_path / "path.csv"
    gcode_path = tmp_path / "path.gcode"
    coverage_path = tmp_path / "coverage.csv"
    coverage_summary_path = tmp_path / "coverage_summary.csv"
    preview_path = tmp_path / "preview.obj"
    project_path = tmp_path / "project.fwp.json"

    result = main(
        [
            "cylinder",
            "--length",
            "100",
            "--radius",
            "20",
            "--tow-width",
            "3",
            "--angle",
            "45",
            "--points",
            "5",
            "--clearance",
            "5",
            "--csv",
            str(csv_path),
            "--gcode",
            str(gcode_path),
            "--coverage-csv",
            str(coverage_path),
            "--coverage-summary-csv",
            str(coverage_summary_path),
            "--coverage-z-samples",
            "10",
            "--coverage-theta-samples",
            "12",
            "--preview-obj",
            str(preview_path),
            "--project",
            str(project_path),
            "--passes",
            "2",
            "--validate",
            "--x-min",
            "0",
            "--x-max",
            "30",
        ]
    )

    assert result == 0
    assert csv_path.exists()
    assert gcode_path.exists()
    assert coverage_path.exists()
    assert coverage_summary_path.exists()
    assert preview_path.exists()
    assert project_path.exists()
    assert load_project(project_path).outputs.coverage_csv_path == str(coverage_path)
    assert load_project(project_path).winding.passes == 2
