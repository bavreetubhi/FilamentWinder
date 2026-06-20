from __future__ import annotations

import pytest

from filament_winder.cli import main
from filament_winder.core.path_planning import (
    CylinderPatternOptimizationRequest,
    optimize_cylinder_pattern,
)


def test_optimize_cylinder_pattern_finds_near_full_coverage_closed_candidate() -> None:
    result = optimize_cylinder_pattern(
        CylinderPatternOptimizationRequest(
            length_mm=1000.0,
            radius_mm=100.0,
            tow_width_mm=6.0,
            target_coverage_fraction=1.0,
            min_angle_deg=10.0,
            max_angle_deg=85.0,
            preferred_angle_deg=45.0,
            min_passes=1,
            max_passes=120,
            max_results=5,
        )
    )

    best = result.best

    assert best.passes == 49
    assert best.turns_per_pass == 3
    assert best.phase_offset_deg == pytest.approx(360.0 / 49.0)
    assert best.estimated_coverage_percent == pytest.approx(99.843324, rel=1e-6)
    assert abs(best.estimated_gap_overlap_mm) < 0.02


def test_optimize_cylinder_pattern_returns_ranked_candidates() -> None:
    result = optimize_cylinder_pattern(
        CylinderPatternOptimizationRequest(
            length_mm=500.0,
            radius_mm=50.0,
            tow_width_mm=10.0,
            target_coverage_fraction=0.5,
            max_passes=50,
            max_results=3,
        )
    )

    assert len(result.candidates) == 3
    assert result.candidates[0].score <= result.candidates[1].score
    assert result.candidates[1].score <= result.candidates[2].score


def test_optimize_cylinder_cli_prints_candidates(capsys) -> None:
    result = main(
        [
            "optimize-cylinder",
            "--length",
            "1000",
            "--radius",
            "100",
            "--tow-width",
            "6",
            "--max-passes",
            "120",
            "--results",
            "2",
        ]
    )

    output = capsys.readouterr().out

    assert result == 0
    assert "rank angle_deg passes turns/pass" in output
    assert "49" in output
