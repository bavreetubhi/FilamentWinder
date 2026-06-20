"""Structured backend program exports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from filament_winder.core.coverage import CoverageMap
from filament_winder.core.path_planning import PathSegment


def export_segments_json(
    segments: tuple[PathSegment, ...],
    output_path: str | Path,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([segment.to_dict() for segment in segments], indent=2),
        encoding="utf-8",
    )
    return path


def export_validation_report_json(
    report: dict[str, Any],
    output_path: str | Path,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def export_coverage_grid_npz(
    coverage: CoverageMap,
    output_path: str | Path,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        z_mm=coverage.z_mm,
        theta_rad=coverage.theta_rad,
        coverage_count=coverage.coverage_count,
        tow_width_mm=np.asarray([coverage.tow_width_mm], dtype=float),
        winding_angle_deg=np.asarray([coverage.winding_angle_deg], dtype=float),
    )
    return path
