"""Textbook-style integer closure pattern selection.

The implementation here keeps the integer pattern logic separate from path
generation. A caller supplies the angular propagation of one physically valid
trajectory; this module searches leading and lagging closed integer patterns
and ranks manufacturable candidates.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from filament_winder.core.geometry import AxisymmetricProfileMandrel, CylinderMandrel

FloatArray = NDArray[np.float64]
PatternType = Literal["leading", "lagging"]
ThicknessModel = Literal[
    "classic_smeared_thickness",
    "flat_polar_approximation",
    "polynomial_smoothed_polar_approximation",
]


@dataclass(frozen=True, slots=True)
class ThicknessSummary:
    minimum_thickness_mm: float
    maximum_thickness_mm: float
    mean_thickness_mm: float
    thickness_variation_percent: float
    polar_buildup_mm: float
    dome_buildup_mm: float
    cylinder_buildup_mm: float

    def to_dict(self) -> dict[str, float]:
        return {
            "minimum_thickness_mm": self.minimum_thickness_mm,
            "maximum_thickness_mm": self.maximum_thickness_mm,
            "mean_thickness_mm": self.mean_thickness_mm,
            "thickness_variation_percent": self.thickness_variation_percent,
            "polar_buildup_mm": self.polar_buildup_mm,
            "dome_buildup_mm": self.dome_buildup_mm,
            "cylinder_buildup_mm": self.cylinder_buildup_mm,
        }


@dataclass(frozen=True, slots=True)
class ThicknessDistribution:
    z_mm: tuple[float, ...]
    radius_mm: tuple[float, ...]
    thickness_mm: tuple[float, ...]
    region: tuple[str, ...]
    summary: ThicknessSummary
    model: ThicknessModel

    def rows(self) -> list[dict[str, float | str]]:
        return [
            {
                "z_mm": z,
                "radius_mm": radius,
                "thickness_mm": thickness,
                "region": region,
            }
            for z, radius, thickness, region in zip(
                self.z_mm,
                self.radius_mm,
                self.thickness_mm,
                self.region,
                strict=True,
            )
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "summary": self.summary.to_dict(),
            "samples": self.rows(),
        }


@dataclass(frozen=True, slots=True)
class WindingPatternCandidate:
    pattern_id: str
    pattern_type: PatternType
    layer_id: str
    layer_name: str
    p: int
    k: int
    d: int
    nd: int
    delta_phi_total_deg: float
    delta_phi_pattern_deg: float
    delta_phi_error_deg: float
    roving_width_mm: float
    roving_thickness_mm: float
    effective_roving_width_mm: float
    layer_thickness_mm: float
    number_of_windings: int
    number_of_closed_layers: int
    gcd_check: bool
    closure_error_deg: float
    estimated_winding_time_min: float
    coverage_estimate: float
    thickness_distribution: ThicknessDistribution
    warnings: tuple[str, ...]
    rejection_reasons: tuple[str, ...]
    score: float

    @property
    def valid(self) -> bool:
        return not self.rejection_reasons

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type,
            "layer_id": self.layer_id,
            "layer_name": self.layer_name,
            "p": self.p,
            "k": self.k,
            "d": self.d,
            "nd": self.nd,
            "delta_phi_total_deg": self.delta_phi_total_deg,
            "delta_phi_pattern_deg": self.delta_phi_pattern_deg,
            "delta_phi_error_deg": self.delta_phi_error_deg,
            "roving_width_mm": self.roving_width_mm,
            "roving_thickness_mm": self.roving_thickness_mm,
            "effective_roving_width_mm": self.effective_roving_width_mm,
            "layer_thickness_mm": self.layer_thickness_mm,
            "number_of_windings": self.number_of_windings,
            "number_of_closed_layers": self.number_of_closed_layers,
            "gcd_check": self.gcd_check,
            "closure_error_deg": self.closure_error_deg,
            "estimated_winding_time_min": self.estimated_winding_time_min,
            "coverage_estimate": self.coverage_estimate,
            "thickness_distribution": self.thickness_distribution.to_dict(),
            "warnings": list(self.warnings),
            "rejection_reasons": list(self.rejection_reasons),
            "score": self.score,
            "valid": self.valid,
        }


@dataclass(frozen=True, slots=True)
class PatternSearchRequest:
    layer_id: str
    layer_name: str
    winding_type: str
    winding_angle_deg: float
    delta_phi_total_deg: float
    equatorial_radius_mm: float
    trajectory_length_mm: float
    roving_width_mm: float
    roving_thickness_mm: float
    target_coverage: float = 1.0
    target_layer_thickness_mm: float | None = None
    target_number_of_closed_layers: int | None = None
    feedrate_mm_min: float = 500.0
    max_p: int = 500
    max_k: int = 500
    max_d: int = 20
    angle_tolerance_deg: float = 0.5
    require_gcd_clean_pattern: bool = True
    candidate_count: int = 10
    thickness_model: ThicknessModel = "classic_smeared_thickness"
    max_coverage_estimate: float = 1.35
    max_winding_time_min: float = 600.0
    max_thickness_variation_percent: float = 75.0
    max_polar_buildup_mm: float = 0.75
    prefer_full_cylinder_coverage: bool = False
    undercoverage_weight: float = 1.0
    overlap_weight: float = 1.0
    thickness_weight: float = 1.0
    polar_buildup_weight: float = 1.0
    time_weight: float = 1.0


@dataclass(frozen=True, slots=True)
class PatternSelectionResult:
    request: PatternSearchRequest
    selected: WindingPatternCandidate | None
    candidates: tuple[WindingPatternCandidate, ...]
    rejected: tuple[WindingPatternCandidate, ...]
    rejection_counts: dict[str, int]

    @property
    def all_candidates(self) -> tuple[WindingPatternCandidate, ...]:
        return self.candidates + self.rejected

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": {
                "layer_id": self.request.layer_id,
                "layer_name": self.request.layer_name,
                "winding_type": self.request.winding_type,
                "winding_angle_deg": self.request.winding_angle_deg,
                "delta_phi_total_deg": self.request.delta_phi_total_deg,
                "equatorial_radius_mm": self.request.equatorial_radius_mm,
                "trajectory_length_mm": self.request.trajectory_length_mm,
                "angle_tolerance_deg": self.request.angle_tolerance_deg,
                "target_coverage": self.request.target_coverage,
                "max_coverage_estimate": self.request.max_coverage_estimate,
                "max_winding_time_min": self.request.max_winding_time_min,
                "max_thickness_variation_percent": (
                    self.request.max_thickness_variation_percent
                ),
                "max_polar_buildup_mm": self.request.max_polar_buildup_mm,
            },
            "selected": None if self.selected is None else self.selected.to_dict(),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "rejected": [candidate.to_dict() for candidate in self.rejected],
            "rejection_counts": self.rejection_counts,
        }


@dataclass(frozen=True, slots=True)
class MultiLayerPatternResult:
    layer_results: tuple[PatternSelectionResult, ...]

    @property
    def selected_candidates(self) -> tuple[WindingPatternCandidate, ...]:
        return tuple(
            result.selected for result in self.layer_results if result.selected is not None
        )

    @property
    def candidates(self) -> tuple[WindingPatternCandidate, ...]:
        candidates = tuple(
            candidate
            for result in self.layer_results
            for candidate in result.candidates
        )
        return tuple(sorted(candidates, key=lambda candidate: candidate.score))

    @property
    def rejected(self) -> tuple[WindingPatternCandidate, ...]:
        return tuple(
            candidate
            for result in self.layer_results
            for candidate in result.rejected
        )

    @property
    def rejection_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for result in self.layer_results:
            for reason, count in result.rejection_counts.items():
                counts[reason] = counts.get(reason, 0) + count
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_patterns": [candidate.to_dict() for candidate in self.selected_candidates],
            "layer_results": [result.to_dict() for result in self.layer_results],
            "rejection_counts": self.rejection_counts,
        }


def effective_roving_width_mm(roving_width_mm: float, winding_angle_deg: float) -> float:
    """Project finite roving width into the equatorial spacing direction."""

    if roving_width_mm <= 0.0:
        raise ValueError("roving_width_mm must be positive")
    angle = abs(winding_angle_deg)
    if angle >= 89.999:
        return roving_width_mm
    projection = max(math.sin(math.radians(angle)), 0.1)
    return roving_width_mm * projection * 0.85


def required_number_of_windings(
    *,
    equatorial_radius_mm: float,
    effective_roving_width_mm: float,
    target_coverage: float,
    roving_thickness_mm: float,
    target_layer_thickness_mm: float | None = None,
    target_number_of_closed_layers: int | None = None,
) -> int:
    if equatorial_radius_mm <= 0.0:
        raise ValueError("equatorial_radius_mm must be positive")
    if effective_roving_width_mm <= 0.0:
        raise ValueError("effective_roving_width_mm must be positive")
    if target_coverage <= 0.0:
        raise ValueError("target_coverage must be positive")
    closed_layers = _target_closed_layers(
        roving_thickness_mm=roving_thickness_mm,
        target_layer_thickness_mm=target_layer_thickness_mm,
        target_number_of_closed_layers=target_number_of_closed_layers,
    )
    circumference = 2.0 * math.pi * equatorial_radius_mm
    return max(
        1,
        math.ceil(circumference * target_coverage * closed_layers / effective_roving_width_mm),
    )


def select_winding_pattern(
    request: PatternSearchRequest,
    mandrel: CylinderMandrel | AxisymmetricProfileMandrel,
) -> PatternSelectionResult:
    effective_width = effective_roving_width_mm(
        request.roving_width_mm,
        request.winding_angle_deg,
    )
    required_nd = required_number_of_windings(
        equatorial_radius_mm=request.equatorial_radius_mm,
        effective_roving_width_mm=effective_width,
        target_coverage=request.target_coverage,
        roving_thickness_mm=request.roving_thickness_mm,
        target_layer_thickness_mm=request.target_layer_thickness_mm,
        target_number_of_closed_layers=request.target_number_of_closed_layers,
    )
    candidates: list[WindingPatternCandidate] = []
    rejected: list[WindingPatternCandidate] = []
    rejection_counts: dict[str, int] = {}
    min_nd = max(1, int(math.floor(required_nd * 0.45)))
    max_nd = max(min_nd, int(math.ceil(required_nd * 1.25)) + max(12, request.candidate_count))
    actual_step = _normalise_angle_0_360(request.delta_phi_total_deg)
    pattern_index = 0
    for pattern_type in ("leading", "lagging"):
        for d in range(1, request.max_d + 1):
            for k in range(1, request.max_k + 1):
                for p in range(1, request.max_p + 1):
                    nd = _candidate_nd(pattern_type, p, k, d)
                    if nd < min_nd:
                        continue
                    if nd > max_nd:
                        break
                    pattern_index += 1
                    candidate = _make_candidate(
                        request=request,
                        mandrel=mandrel,
                        pattern_type=pattern_type,
                        p=p,
                        k=k,
                        d=d,
                        nd=nd,
                        actual_step=actual_step,
                        required_nd=required_nd,
                        effective_width=effective_width,
                        pattern_index=pattern_index,
                    )
                    if candidate.rejection_reasons:
                        rejected.append(candidate)
                        for reason in candidate.rejection_reasons:
                            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                    else:
                        candidates.append(candidate)
    candidates.sort(key=lambda candidate: candidate.score)
    rejected.sort(key=lambda candidate: candidate.score)
    selected = candidates[0] if candidates else (rejected[0] if rejected else None)
    return PatternSelectionResult(
        request=request,
        selected=selected,
        candidates=tuple(candidates[: request.candidate_count]),
        rejected=tuple(rejected[: max(request.candidate_count, 20)]),
        rejection_counts=rejection_counts,
    )


def export_pattern_candidates_json(
    result: MultiLayerPatternResult,
    output_path: str | Path,
) -> Path:
    rows = [candidate.to_dict() for candidate in result.candidates]
    return _write_json(rows, output_path)


def export_selected_pattern_json(
    result: MultiLayerPatternResult,
    output_path: str | Path,
) -> Path:
    rows = [candidate.to_dict() for candidate in result.selected_candidates]
    return _write_json(rows, output_path)


def export_pattern_rejection_report_json(
    result: MultiLayerPatternResult,
    output_path: str | Path,
) -> Path:
    return _write_json(
        {
            "rejection_counts": result.rejection_counts,
            "rejected": [candidate.to_dict() for candidate in result.rejected],
        },
        output_path,
    )


def export_thickness_distribution_csv(
    result: MultiLayerPatternResult,
    output_path: str | Path,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "layer_id",
        "pattern_id",
        "z_mm",
        "radius_mm",
        "thickness_mm",
        "region",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in result.selected_candidates:
            for row in candidate.thickness_distribution.rows():
                writer.writerow(
                    {
                        "layer_id": candidate.layer_id,
                        "pattern_id": candidate.pattern_id,
                        **row,
                    }
                )
    return path


def _make_candidate(
    *,
    request: PatternSearchRequest,
    mandrel: CylinderMandrel | AxisymmetricProfileMandrel,
    pattern_type: PatternType,
    p: int,
    k: int,
    d: int,
    nd: int,
    actual_step: float,
    required_nd: int,
    effective_width: float,
    pattern_index: int,
) -> WindingPatternCandidate:
    residual = 360.0 / max(nd, 1)
    if pattern_type == "lagging":
        residual = -residual
    revolutions = round((actual_step * p - residual) / 360.0)
    delta_phi_pattern = (revolutions * 360.0 + residual) / max(p, 1)
    closure_error = _angular_error_deg(actual_step, delta_phi_pattern)
    gcd_check = _gcd_clean(p, k, d, nd)
    coverage_estimate = nd * effective_width / (2.0 * math.pi * request.equatorial_radius_mm)
    layer_thickness = request.roving_thickness_mm * d
    distribution = estimate_thickness_distribution(
        mandrel,
        roving_thickness_mm=request.roving_thickness_mm,
        coverage_estimate=coverage_estimate,
        number_of_closed_layers=d,
        model=request.thickness_model,
    )
    warnings: list[str] = []
    rejection_reasons: list[str] = []
    if closure_error > request.angle_tolerance_deg:
        rejection_reasons.append("closure_error")
    if request.require_gcd_clean_pattern and not gcd_check:
        rejection_reasons.append("repeated_gcd_pattern")
    if coverage_estimate < request.target_coverage * 0.92:
        rejection_reasons.append("insufficient_coverage")
    if coverage_estimate > request.max_coverage_estimate:
        rejection_reasons.append("excessive_coverage")
    elif coverage_estimate > request.target_coverage * 1.15:
        warnings.append("high overlap estimate")
    if layer_thickness < _target_thickness(request) * 0.9:
        rejection_reasons.append("insufficient_thickness")
    winding_time_min = request.trajectory_length_mm * nd / max(request.feedrate_mm_min, 1e-9)
    if winding_time_min > request.max_winding_time_min:
        rejection_reasons.append("excessive_winding_time")
    if distribution.summary.thickness_variation_percent > request.max_thickness_variation_percent:
        rejection_reasons.append("excessive_thickness_variation")
    if distribution.summary.polar_buildup_mm > request.max_polar_buildup_mm:
        rejection_reasons.append("excessive_polar_buildup")
    score = _candidate_score(
        closure_error=closure_error,
        coverage_estimate=coverage_estimate,
        target_coverage=request.target_coverage,
        thickness_summary=distribution.summary,
        windings=nd,
        required_nd=required_nd,
        winding_time_min=winding_time_min,
        request=request,
    )
    return WindingPatternCandidate(
        pattern_id=f"{request.layer_id}-{pattern_type}-{pattern_index:04d}",
        pattern_type=pattern_type,
        layer_id=request.layer_id,
        layer_name=request.layer_name,
        p=p,
        k=k,
        d=d,
        nd=nd,
        delta_phi_total_deg=request.delta_phi_total_deg,
        delta_phi_pattern_deg=delta_phi_pattern,
        delta_phi_error_deg=closure_error,
        roving_width_mm=request.roving_width_mm,
        roving_thickness_mm=request.roving_thickness_mm,
        effective_roving_width_mm=effective_width,
        layer_thickness_mm=layer_thickness,
        number_of_windings=nd,
        number_of_closed_layers=d,
        gcd_check=gcd_check,
        closure_error_deg=closure_error,
        estimated_winding_time_min=winding_time_min,
        coverage_estimate=coverage_estimate,
        thickness_distribution=distribution,
        warnings=tuple(warnings),
        rejection_reasons=tuple(rejection_reasons),
        score=score,
    )


def estimate_thickness_distribution(
    mandrel: CylinderMandrel | AxisymmetricProfileMandrel,
    *,
    roving_thickness_mm: float,
    coverage_estimate: float,
    number_of_closed_layers: int,
    model: ThicknessModel,
) -> ThicknessDistribution:
    z_mm, radius = _profile_samples(mandrel)
    max_radius = max(float(np.max(radius)), 1e-9)
    safe_radius = np.maximum(radius, 1e-9)
    base_thickness = max(roving_thickness_mm, 0.0) * max(number_of_closed_layers, 1)
    if model == "flat_polar_approximation":
        crowding = np.ones_like(radius)
    elif model == "polynomial_smoothed_polar_approximation":
        ratio = np.clip(max_radius / safe_radius, 1.0, 8.0)
        crowding = 1.0 + 0.65 * (ratio - 1.0) / (1.0 + (ratio - 1.0) ** 2)
    else:
        crowding = np.clip(max_radius / safe_radius, 1.0, 8.0)
    thickness = base_thickness * coverage_estimate * crowding
    region = _regions(z_mm, radius)
    summary = _thickness_summary(thickness, region)
    return ThicknessDistribution(
        z_mm=tuple(float(value) for value in z_mm),
        radius_mm=tuple(float(value) for value in radius),
        thickness_mm=tuple(float(value) for value in thickness),
        region=tuple(region),
        summary=summary,
        model=model,
    )


def _candidate_nd(pattern_type: PatternType, p: int, k: int, d: int) -> int:
    if pattern_type == "leading":
        return (p + 1) * k * d - 1
    return p * k * d + 1


def _gcd_clean(p: int, k: int, d: int, nd: int) -> bool:
    return math.gcd(p, nd) == 1 and math.gcd(k, nd) == 1 and math.gcd(d, nd) == 1


def _normalise_angle_0_360(value: float) -> float:
    return value % 360.0


def _angular_error_deg(first: float, second: float) -> float:
    return abs(((first - second + 180.0) % 360.0) - 180.0)


def _target_closed_layers(
    *,
    roving_thickness_mm: float,
    target_layer_thickness_mm: float | None,
    target_number_of_closed_layers: int | None,
) -> int:
    if target_number_of_closed_layers is not None:
        return max(1, target_number_of_closed_layers)
    if target_layer_thickness_mm is None or roving_thickness_mm <= 0.0:
        return 1
    return max(1, math.ceil(target_layer_thickness_mm / roving_thickness_mm))


def _target_thickness(request: PatternSearchRequest) -> float:
    if request.target_layer_thickness_mm is not None:
        return request.target_layer_thickness_mm
    layers = _target_closed_layers(
        roving_thickness_mm=request.roving_thickness_mm,
        target_layer_thickness_mm=request.target_layer_thickness_mm,
        target_number_of_closed_layers=request.target_number_of_closed_layers,
    )
    return request.roving_thickness_mm * layers


def _candidate_score(
    *,
    closure_error: float,
    coverage_estimate: float,
    target_coverage: float,
    thickness_summary: ThicknessSummary,
    windings: int,
    required_nd: int,
    winding_time_min: float,
    request: PatternSearchRequest,
) -> float:
    if not request.prefer_full_cylinder_coverage:
        coverage_error = abs(coverage_estimate - target_coverage)
        overlap_penalty = max(coverage_estimate - target_coverage, 0.0)
        winding_penalty = abs(windings - required_nd) / max(required_nd, 1)
        excessive_time_penalty = max(winding_time_min - 60.0, 0.0)
        return (
            closure_error * 250.0
            + coverage_error * 160.0
            + overlap_penalty * 320.0
            + thickness_summary.thickness_variation_percent * 0.2
            + thickness_summary.polar_buildup_mm * 80.0
            + winding_penalty * 15.0
            + winding_time_min * 0.25
            + excessive_time_penalty * 0.75
        )
    undercoverage_penalty = max(target_coverage - coverage_estimate, 0.0)
    overlap_penalty = max(coverage_estimate - target_coverage, 0.0)
    coverage_error = undercoverage_penalty + overlap_penalty * 0.35
    winding_penalty = abs(windings - required_nd) / max(required_nd, 1)
    excessive_time_penalty = max(winding_time_min - 60.0, 0.0)
    undercoverage_penalty *= 2.4
    overlap_penalty *= 0.45
    return (
        closure_error * 250.0
        + coverage_error * 150.0 * max(request.undercoverage_weight, 0.05)
        + undercoverage_penalty * 340.0 * max(request.undercoverage_weight, 0.05)
        + overlap_penalty * 120.0 * max(request.overlap_weight, 0.05)
        + thickness_summary.thickness_variation_percent
        * 0.2
        * max(request.thickness_weight, 0.0)
        + thickness_summary.polar_buildup_mm * 80.0 * max(request.polar_buildup_weight, 0.0)
        + winding_penalty * 15.0
        + winding_time_min * 0.25 * max(request.time_weight, 0.0)
        + excessive_time_penalty * 0.75 * max(request.time_weight, 0.0)
    )


def _profile_samples(
    mandrel: CylinderMandrel | AxisymmetricProfileMandrel,
) -> tuple[FloatArray, FloatArray]:
    if isinstance(mandrel, CylinderMandrel):
        z_mm = np.linspace(0.0, mandrel.length_mm, 160)
        radius = np.full(z_mm.shape, mandrel.radius_mm, dtype=float)
        return z_mm, radius
    return np.asarray(mandrel.z_mm, dtype=float), np.asarray(mandrel.r_mm, dtype=float)


def _regions(z_mm: FloatArray, radius: FloatArray) -> list[str]:
    max_radius = float(np.max(radius))
    midpoint = (float(z_mm[0]) + float(z_mm[-1])) / 2.0
    regions = []
    for z_value, radius_value in zip(z_mm, radius, strict=True):
        if radius_value < max_radius * 0.35:
            regions.append("polar")
        elif radius_value >= max_radius * 0.98:
            regions.append("cylinder")
        elif z_value < midpoint:
            regions.append("left_dome")
        else:
            regions.append("right_dome")
    return regions


def _thickness_summary(thickness: FloatArray, region: list[str]) -> ThicknessSummary:
    minimum = float(np.min(thickness))
    maximum = float(np.max(thickness))
    mean = float(np.mean(thickness))
    variation = 0.0 if mean <= 1e-9 else (maximum - minimum) / mean * 100.0
    polar = _region_mean(thickness, region, {"polar"})
    dome = _region_mean(thickness, region, {"left_dome", "right_dome"})
    cylinder = _region_mean(thickness, region, {"cylinder"})
    return ThicknessSummary(
        minimum_thickness_mm=minimum,
        maximum_thickness_mm=maximum,
        mean_thickness_mm=mean,
        thickness_variation_percent=variation,
        polar_buildup_mm=max(0.0, polar - cylinder),
        dome_buildup_mm=max(0.0, dome - cylinder),
        cylinder_buildup_mm=cylinder,
    )


def _region_mean(thickness: FloatArray, region: list[str], names: set[str]) -> float:
    values = [value for value, label in zip(thickness, region, strict=True) if label in names]
    if not values:
        return float(np.mean(thickness))
    return float(np.mean(np.asarray(values, dtype=float)))


def _write_json(data: Any, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path
