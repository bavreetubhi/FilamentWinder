"""Cylinder pattern optimization."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from filament_winder.core.geometry import CylinderMandrel
from filament_winder.core.path_planning.helical import HelicalPathConfig


@dataclass(frozen=True, slots=True)
class CylinderPatternOptimizationRequest:
    length_mm: float
    radius_mm: float
    tow_width_mm: float
    point_count: int = 500
    target_coverage_fraction: float = 1.0
    min_angle_deg: float = 10.0
    max_angle_deg: float = 85.0
    min_passes: int = 1
    max_passes: int = 200
    preferred_angle_deg: float | None = 45.0
    max_results: int = 10

    def validate(self) -> None:
        if not np.isfinite(self.length_mm) or self.length_mm <= 0.0:
            raise ValueError("length_mm must be a positive finite value")
        if not np.isfinite(self.radius_mm) or self.radius_mm <= 0.0:
            raise ValueError("radius_mm must be a positive finite value")
        if not np.isfinite(self.tow_width_mm) or self.tow_width_mm <= 0.0:
            raise ValueError("tow_width_mm must be a positive finite value")
        if self.point_count < 2:
            raise ValueError("point_count must be at least 2")
        if not np.isfinite(self.target_coverage_fraction) or self.target_coverage_fraction <= 0.0:
            raise ValueError("target_coverage_fraction must be a positive finite value")
        if not 0.0 < self.min_angle_deg < self.max_angle_deg < 90.0:
            raise ValueError("angle bounds must satisfy 0 < min_angle_deg < max_angle_deg < 90")
        if self.min_passes < 1 or self.max_passes < self.min_passes:
            raise ValueError("pass bounds must satisfy 1 <= min_passes <= max_passes")
        if self.preferred_angle_deg is not None and not (
            self.min_angle_deg <= self.preferred_angle_deg <= self.max_angle_deg
        ):
            raise ValueError("preferred_angle_deg must be inside the angle search range")
        if self.max_results < 1:
            raise ValueError("max_results must be at least 1")


@dataclass(frozen=True, slots=True)
class CylinderPatternCandidate:
    winding_angle_deg: float
    passes: int
    turns_per_pass: int
    phase_offset_deg: float
    band_spacing_mm: float
    estimated_coverage_fraction: float
    estimated_gap_overlap_mm: float
    score: float

    @property
    def estimated_coverage_percent(self) -> float:
        return self.estimated_coverage_fraction * 100.0

    @property
    def has_gap(self) -> bool:
        return self.estimated_gap_overlap_mm > 0.0

    @property
    def has_overlap(self) -> bool:
        return self.estimated_gap_overlap_mm < 0.0

    def to_helical_config(self, *, tow_width_mm: float, point_count: int) -> HelicalPathConfig:
        return HelicalPathConfig(
            winding_angle_deg=self.winding_angle_deg,
            tow_width_mm=tow_width_mm,
            point_count=point_count,
            passes=self.passes,
            phase_offset_deg=None,
        )


@dataclass(frozen=True, slots=True)
class CylinderPatternOptimizationResult:
    request: CylinderPatternOptimizationRequest
    candidates: tuple[CylinderPatternCandidate, ...]

    @property
    def best(self) -> CylinderPatternCandidate:
        if not self.candidates:
            raise ValueError("no pattern candidates were found")
        return self.candidates[0]


def optimize_cylinder_pattern(
    request: CylinderPatternOptimizationRequest,
) -> CylinderPatternOptimizationResult:
    """Find closed integer-turn cylinder winding candidates.

    The optimizer keeps each pass closed by forcing `turns_per_pass` to an
    integer. Pass count is then ranked against target coverage using the
    pass-to-pass circumferential band spacing.
    """

    request.validate()
    mandrel = CylinderMandrel(length_mm=request.length_mm, radius_mm=request.radius_mm)
    circumference_mm = 2.0 * math.pi * mandrel.radius_mm
    candidates: list[CylinderPatternCandidate] = []

    for turns_per_pass in _closed_turn_range(request):
        angle_deg = math.degrees(
            math.atan(circumference_mm * turns_per_pass / request.length_mm)
        )
        if not request.min_angle_deg <= angle_deg <= request.max_angle_deg:
            continue
        for passes in range(request.min_passes, request.max_passes + 1):
            band_spacing_mm = (
                circumference_mm * math.cos(math.radians(angle_deg)) / passes
            )
            estimated_coverage = request.tow_width_mm / band_spacing_mm
            gap_overlap_mm = band_spacing_mm - request.tow_width_mm
            score = _score_candidate(
                request,
                angle_deg=angle_deg,
                passes=passes,
                estimated_coverage=estimated_coverage,
            )
            candidates.append(
                CylinderPatternCandidate(
                    winding_angle_deg=angle_deg,
                    passes=passes,
                    turns_per_pass=turns_per_pass,
                    phase_offset_deg=360.0 / passes,
                    band_spacing_mm=band_spacing_mm,
                    estimated_coverage_fraction=estimated_coverage,
                    estimated_gap_overlap_mm=gap_overlap_mm,
                    score=score,
                )
            )

    candidates.sort(
        key=lambda candidate: (
            candidate.score,
            abs(candidate.estimated_coverage_fraction - request.target_coverage_fraction),
            abs(_angle_preference(request) - candidate.winding_angle_deg),
            candidate.passes,
            candidate.turns_per_pass,
        )
    )
    return CylinderPatternOptimizationResult(
        request=request,
        candidates=tuple(candidates[: request.max_results]),
    )


def _closed_turn_range(request: CylinderPatternOptimizationRequest) -> range:
    min_turns = math.ceil(
        math.tan(math.radians(request.min_angle_deg)) * request.length_mm
        / (2.0 * math.pi * request.radius_mm)
    )
    max_turns = math.floor(
        math.tan(math.radians(request.max_angle_deg)) * request.length_mm
        / (2.0 * math.pi * request.radius_mm)
    )
    return range(max(1, min_turns), max(1, max_turns) + 1)


def _score_candidate(
    request: CylinderPatternOptimizationRequest,
    *,
    angle_deg: float,
    passes: int,
    estimated_coverage: float,
) -> float:
    coverage_error = abs(estimated_coverage - request.target_coverage_fraction)
    angle_error = abs(_angle_preference(request) - angle_deg) / 90.0
    pass_penalty = passes / max(1, request.max_passes) * 0.01
    return coverage_error * 100.0 + angle_error + pass_penalty


def _angle_preference(request: CylinderPatternOptimizationRequest) -> float:
    if request.preferred_angle_deg is not None:
        return request.preferred_angle_deg
    return (request.min_angle_deg + request.max_angle_deg) / 2.0
