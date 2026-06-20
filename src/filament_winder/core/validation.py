"""Machine path validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from filament_winder.core.kinematics import MachineMotionTable

Severity = Literal["error", "warning"]


@dataclass(frozen=True, slots=True)
class AxisLimitConfig:
    a_min_deg: float | None = None
    a_max_deg: float | None = None
    x_min_mm: float | None = None
    x_max_mm: float | None = None
    z_min_mm: float | None = None
    z_max_mm: float | None = None
    b_min_deg: float | None = None
    b_max_deg: float | None = None


@dataclass(frozen=True, slots=True)
class NoGoZone:
    name: str
    x_min_mm: float
    x_max_mm: float
    z_min_mm: float
    z_max_mm: float

    def __post_init__(self) -> None:
        values = [self.x_min_mm, self.x_max_mm, self.z_min_mm, self.z_max_mm]
        if not all(np.isfinite(value) for value in values):
            raise ValueError("no-go zone bounds must be finite")
        if self.x_max_mm < self.x_min_mm:
            raise ValueError("x_max_mm must be greater than or equal to x_min_mm")
        if self.z_max_mm < self.z_min_mm:
            raise ValueError("z_max_mm must be greater than or equal to z_min_mm")


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    severity: Severity
    code: str
    message: str
    point_index: int | None = None


@dataclass(frozen=True, slots=True)
class ValidationReport:
    issues: tuple[ValidationIssue, ...]

    @property
    def has_errors(self) -> bool:
        return any(issue.severity == "error" for issue in self.issues)

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "warning")


def validate_motion_table(
    motion_table: MachineMotionTable,
    *,
    limits: AxisLimitConfig | None = None,
    no_go_zones: tuple[NoGoZone, ...] = (),
) -> ValidationReport:
    """Validate machine positions against axis limits and rectangular X-Z no-go zones."""

    resolved_limits = AxisLimitConfig() if limits is None else limits
    issues: list[ValidationIssue] = []
    _check_axis(
        issues,
        code="A_LIMIT",
        axis_name="A",
        units="deg",
        values=motion_table.a_deg,
        minimum=resolved_limits.a_min_deg,
        maximum=resolved_limits.a_max_deg,
    )
    _check_axis(
        issues,
        code="X_LIMIT",
        axis_name="X",
        units="mm",
        values=motion_table.x_mm,
        minimum=resolved_limits.x_min_mm,
        maximum=resolved_limits.x_max_mm,
    )
    _check_axis(
        issues,
        code="Z_LIMIT",
        axis_name="Z",
        units="mm",
        values=motion_table.z_mm,
        minimum=resolved_limits.z_min_mm,
        maximum=resolved_limits.z_max_mm,
    )
    _check_axis(
        issues,
        code="B_LIMIT",
        axis_name="B",
        units="deg",
        values=motion_table.b_deg,
        minimum=resolved_limits.b_min_deg,
        maximum=resolved_limits.b_max_deg,
    )
    for zone in no_go_zones:
        inside = (
            (motion_table.x_mm >= zone.x_min_mm)
            & (motion_table.x_mm <= zone.x_max_mm)
            & (motion_table.z_mm >= zone.z_min_mm)
            & (motion_table.z_mm <= zone.z_max_mm)
        )
        if np.any(inside):
            first_index = int(np.flatnonzero(inside)[0])
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="NO_GO_ZONE",
                    message=f"Path enters no-go zone '{zone.name}' at point {first_index}",
                    point_index=first_index,
                )
            )
    return ValidationReport(tuple(issues))


def _check_axis(
    issues: list[ValidationIssue],
    *,
    code: str,
    axis_name: str,
    units: str,
    values: np.ndarray,
    minimum: float | None,
    maximum: float | None,
) -> None:
    if minimum is not None and np.any(values < minimum):
        first_index = int(np.flatnonzero(values < minimum)[0])
        issues.append(
            ValidationIssue(
                severity="error",
                code=code,
                message=(
                    f"{axis_name} axis is below minimum {minimum:g} {units} "
                    f"at point {first_index}: {values[first_index]:g} {units}"
                ),
                point_index=first_index,
            )
        )
    if maximum is not None and np.any(values > maximum):
        first_index = int(np.flatnonzero(values > maximum)[0])
        issues.append(
            ValidationIssue(
                severity="error",
                code=code,
                message=(
                    f"{axis_name} axis is above maximum {maximum:g} {units} "
                    f"at point {first_index}: {values[first_index]:g} {units}"
                ),
                point_index=first_index,
            )
        )
