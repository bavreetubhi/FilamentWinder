"""Path continuity and generated-output validation."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from filament_winder.core.geometry import AxisymmetricProfileMandrel, CylinderMandrel
from filament_winder.core.path_planning import PlannedWindingProgram


@dataclass(frozen=True, slots=True)
class PathValidationResult:
    csv_rows: int
    total_points: int | None
    z_min_mm: float | None
    z_max_mm: float | None
    radius_min_mm: float | None
    radius_max_mm: float | None
    continuity: dict[str, Any]
    path_validation: dict[str, bool]
    transition_summary: dict[str, Any]
    result: str
    reasons: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return self.result == "PASS"


def program_continuity_summary(program: PlannedWindingProgram) -> dict[str, Any]:
    metrics = _continuity_from_arrays(
        program.path.points_mm,
        layer_ids=program.metadata.layer_id,
        pass_ids=tuple(str(value) for value in program.metadata.pass_index),
        motion_types=program.metadata.motion_type,
        tow_width_mm=program.path.tow_width_mm,
    )
    return metrics


def program_transition_summary(program: PlannedWindingProgram) -> dict[str, Any]:
    return _transition_summary_from_arrays(
        program.path.points_mm,
        program.metadata.motion_type,
        tow_width_mm=program.path.tow_width_mm,
    )


def program_path_validation_summary(
    mandrel: CylinderMandrel | AxisymmetricProfileMandrel,
    program: PlannedWindingProgram,
    *,
    csv_path: Path | None = None,
    summary_path: Path | None = None,
    plot_paths: tuple[Path, ...] = (),
) -> dict[str, bool]:
    points = program.path.points_mm
    finite = np.isfinite(points)
    radius = program.path.surface_radius_mm
    csv_row_count_match = True
    if csv_path is not None and csv_path.exists():
        csv_row_count_match = _count_csv_rows(csv_path) == program.point_count
    elif csv_path is not None:
        csv_row_count_match = False
    summary_file_ok = True if summary_path is None else bool(summary_path.parent.exists())
    return {
        "no_nan_values": not bool(np.any(np.isnan(points))),
        "no_infinite_values": not bool(np.any(np.isinf(points))),
        "path_is_not_empty": program.point_count > 0,
        "z_bounds_ok": (
            float(np.min(program.path.z_mm)) >= -1e-6
            and float(np.max(program.path.z_mm)) <= _mandrel_end_z(mandrel) + 1e-6
        ),
        "radius_bounds_ok": (
            bool(np.all(finite))
            and float(np.min(radius)) > 0.0
            and float(np.max(radius)) >= _mandrel_radius(mandrel) - 1e-6
        ),
        "csv_summary_row_count_match": csv_row_count_match,
        "summary_path_writable": summary_file_ok,
        "plot_files_exist": all(path.exists() for path in plot_paths),
        "plot_files_non_empty": all(
            path.exists() and path.stat().st_size > 0 for path in plot_paths
        ),
    }


def validate_path_csv(
    csv_path: Path,
    *,
    summary_path: Path | None = None,
) -> PathValidationResult:
    rows = _read_csv_rows(csv_path)
    summary = _read_summary(summary_path)
    points = _csv_points(rows)
    z_values = _csv_float_column(rows, "z_mm")
    radius = _csv_radius(rows)
    motion_types = tuple(row.get("motion_type", "wind") for row in rows)
    layer_ids = tuple(row.get("layer_id", "") for row in rows)
    pass_ids = tuple(row.get("pass_id", row.get("pass_index", "")) for row in rows)
    tow_width = _summary_tow_width(summary)
    continuity = _continuity_from_arrays(
        points,
        layer_ids=layer_ids,
        pass_ids=pass_ids,
        motion_types=motion_types,
        tow_width_mm=tow_width,
    )
    transition_summary = _transition_summary_from_arrays(
        points,
        motion_types,
        tow_width_mm=tow_width,
    )
    total_points = _summary_total_points(summary)
    plot_paths = _summary_plot_paths(summary, summary_path)
    path_validation = {
        "no_nan_values": _no_nan(rows),
        "no_infinite_values": _no_infinite(rows),
        "path_is_not_empty": len(rows) > 0,
        "z_bounds_ok": _z_bounds_ok(z_values, summary),
        "radius_bounds_ok": _radius_bounds_ok(radius, summary),
        "csv_summary_row_count_match": total_points is None or total_points == len(rows),
        "plot_files_exist": all(path.exists() for path in plot_paths),
        "plot_files_non_empty": all(
            path.exists() and path.stat().st_size > 0 for path in plot_paths
        ),
    }
    reasons = _validation_reasons(path_validation, continuity)
    return PathValidationResult(
        csv_rows=len(rows),
        total_points=total_points,
        z_min_mm=None if z_values.size == 0 else float(np.min(z_values)),
        z_max_mm=None if z_values.size == 0 else float(np.max(z_values)),
        radius_min_mm=None if radius.size == 0 else float(np.min(radius)),
        radius_max_mm=None if radius.size == 0 else float(np.max(radius)),
        continuity=continuity,
        path_validation=path_validation,
        transition_summary=transition_summary,
        result="PASS" if not reasons else "FAIL",
        reasons=tuple(reasons),
    )


def format_path_validation_report(result: PathValidationResult) -> str:
    reason_text = "\n".join(f"Reason: {reason}" for reason in result.reasons)
    bounds = (
        f"  Z min/max: {_fmt(result.z_min_mm)} / {_fmt(result.z_max_mm)} mm\n"
        f"  Radius min/max: {_fmt(result.radius_min_mm)} / {_fmt(result.radius_max_mm)} mm"
    )
    report = (
        "Path Validation Report\n"
        "----------------------\n"
        f"CSV rows: {result.csv_rows}\n"
        f"Summary total points: {_summary_points_text(result)}\n\n"
        "Bounds:\n"
        f"{bounds}\n\n"
        "Continuity:\n"
        f"  Max within-pass step: {result.continuity['max_within_pass_step_mm']:.3f} mm\n"
        f"  Max boundary step: {result.continuity['max_boundary_step_mm']:.3f} mm\n"
        f"  Large boundary jumps: {result.continuity['large_boundary_jump_count']}\n\n"
        f"Transitions: {result.transition_summary['transition_count']} segments, "
        f"{result.transition_summary['transition_points']} points\n\n"
        f"Result: {result.result}"
    )
    if reason_text:
        report += "\n" + reason_text
    return report


def _continuity_from_arrays(
    points: np.ndarray,
    *,
    layer_ids: tuple[str, ...],
    pass_ids: tuple[str, ...],
    motion_types: tuple[str, ...],
    tow_width_mm: float,
) -> dict[str, Any]:
    if points.shape[0] < 2:
        return {
            "max_within_pass_step_mm": 0.0,
            "max_boundary_step_mm": 0.0,
            "large_boundary_jump_count": 0,
            "large_step_threshold_mm": _large_step_threshold(tow_width_mm),
            "continuous_machine_path": points.shape[0] > 0,
        }
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    boundary = np.zeros(steps.shape, dtype=bool)
    for index in range(1, len(layer_ids)):
        boundary[index - 1] = (
            layer_ids[index] != layer_ids[index - 1]
            or pass_ids[index] != pass_ids[index - 1]
            or motion_types[index] != motion_types[index - 1]
        )
    threshold = _large_step_threshold(tow_width_mm)
    boundary_steps = steps[boundary]
    within_steps = steps[~boundary]
    large_boundary_jump_count = int(np.count_nonzero(boundary_steps > threshold))
    return {
        "max_within_pass_step_mm": float(np.max(within_steps)) if within_steps.size else 0.0,
        "max_boundary_step_mm": float(np.max(boundary_steps)) if boundary_steps.size else 0.0,
        "large_boundary_jump_count": large_boundary_jump_count,
        "large_step_threshold_mm": threshold,
        "continuous_machine_path": large_boundary_jump_count == 0,
    }


def _transition_summary_from_arrays(
    points: np.ndarray,
    motion_types: tuple[str, ...],
    *,
    tow_width_mm: float,
) -> dict[str, Any]:
    transition_mask = np.asarray([value == "transition" for value in motion_types], dtype=bool)
    transition_points = int(np.count_nonzero(transition_mask))
    transition_count = 0
    previous = False
    for value in transition_mask:
        if bool(value) and not previous:
            transition_count += 1
        previous = bool(value)
    transition_steps = []
    for index in range(1, points.shape[0]):
        if transition_mask[index] or transition_mask[index - 1]:
            transition_steps.append(float(np.linalg.norm(points[index] - points[index - 1])))
    max_step = max(transition_steps, default=0.0)
    return {
        "transition_count": transition_count,
        "transition_points": transition_points,
        "max_transition_step_mm": max_step,
        "transitions_are_continuous": max_step <= _large_step_threshold(tow_width_mm),
    }


def _large_step_threshold(tow_width_mm: float) -> float:
    return max(25.0, float(tow_width_mm) * 6.0)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _csv_points(rows: list[dict[str, str]]) -> np.ndarray:
    if not rows:
        return np.empty((0, 3), dtype=float)
    return np.asarray(
        [
            (
                _float(row.get("surface_x_mm", row.get("x_mm", "nan"))),
                _float(row.get("surface_y_mm", row.get("y_mm", "nan"))),
                _float(row.get("surface_z_mm", row.get("z_mm", "nan"))),
            )
            for row in rows
        ],
        dtype=float,
    )


def _csv_float_column(rows: list[dict[str, str]], column: str) -> np.ndarray:
    return np.asarray([_float(row.get(column, "nan")) for row in rows], dtype=float)


def _csv_radius(rows: list[dict[str, str]]) -> np.ndarray:
    if not rows:
        return np.asarray([], dtype=float)
    values = []
    for row in rows:
        if row.get("local_radius_mm"):
            values.append(_float(row["local_radius_mm"]))
            continue
        x_value = _float(row.get("surface_x_mm", "nan"))
        y_value = _float(row.get("surface_y_mm", "nan"))
        values.append(math.hypot(x_value, y_value))
    return np.asarray(values, dtype=float)


def _read_summary(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _summary_total_points(summary: dict[str, Any]) -> int | None:
    value = summary.get("total_path_points")
    return None if value is None else int(value)


def _summary_tow_width(summary: dict[str, Any]) -> float:
    tow = summary.get("tow", {})
    return float(tow.get("width_mm", 6.0))


def _summary_plot_paths(summary: dict[str, Any], summary_path: Path | None) -> tuple[Path, ...]:
    output_files = summary.get("output_files", {})
    paths = []
    base_dir = None if summary_path is None else summary_path.parent
    for raw_path in output_files.get("plots", ()):
        path = Path(str(raw_path))
        if not path.is_absolute() and not path.exists() and base_dir is not None:
            path = base_dir / path
        paths.append(path)
    return tuple(paths)


def _z_bounds_ok(values: np.ndarray, summary: dict[str, Any]) -> bool:
    if values.size == 0 or not np.all(np.isfinite(values)):
        return False
    mandrel = summary.get("mandrel", {})
    length = mandrel.get("length_mm")
    if length is None:
        return True
    return float(np.min(values)) >= -1e-6 and float(np.max(values)) <= float(length) + 1e-6


def _radius_bounds_ok(values: np.ndarray, summary: dict[str, Any]) -> bool:
    if values.size == 0 or not np.all(np.isfinite(values)):
        return False
    mandrel = summary.get("mandrel", {})
    radius = mandrel.get("radius_mm")
    if radius is None:
        return float(np.min(values)) > 0.0
    return float(np.min(values)) > 0.0 and float(np.max(values)) >= float(radius) - 1e-6


def _no_nan(rows: list[dict[str, str]]) -> bool:
    return all(
        not math.isnan(_float(value))
        for row in rows
        for value in row.values()
        if _is_number(value)
    )


def _no_infinite(rows: list[dict[str, str]]) -> bool:
    return all(
        not math.isinf(_float(value))
        for row in rows
        for value in row.values()
        if _is_number(value)
    )


def _count_csv_rows(path: Path) -> int:
    return len(_read_csv_rows(path))


def _validation_reasons(
    path_validation: dict[str, bool],
    continuity: dict[str, Any],
) -> list[str]:
    reasons = [key for key, value in path_validation.items() if not value]
    if not continuity["continuous_machine_path"]:
        reasons.append("path is not continuous between pass or layer boundaries")
    return reasons


def _float(value: str | None) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def _is_number(value: str | None) -> bool:
    if value is None or value == "":
        return False
    try:
        float(value)
    except ValueError:
        return False
    return True


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _summary_points_text(result: PathValidationResult) -> str:
    return "n/a" if result.total_points is None else str(result.total_points)


def _mandrel_end_z(mandrel: CylinderMandrel | AxisymmetricProfileMandrel) -> float:
    return mandrel.length_mm if isinstance(mandrel, CylinderMandrel) else mandrel.end_z_mm


def _mandrel_radius(mandrel: CylinderMandrel | AxisymmetricProfileMandrel) -> float:
    return mandrel.radius_mm if isinstance(mandrel, CylinderMandrel) else mandrel.max_radius_mm
