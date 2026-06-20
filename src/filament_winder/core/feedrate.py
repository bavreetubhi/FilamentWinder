"""Feedrate planning from local path curvature and slip risk."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from filament_winder.core.path_planning import SurfacePath

FloatArray = NDArray[np.float64]

DEFAULT_MINIMUM_FEED_FRACTION = 0.25
DEFAULT_CURVATURE_SLOWDOWN_RADIUS_MM = 50.0
DEFAULT_SLIP_SLOWDOWN_THRESHOLD = 0.25
DEFAULT_SLIP_MAX_RISK = 1.0


@dataclass(frozen=True, slots=True)
class FeedrateConfig:
    """Adaptive feedrate settings.

    Slip risk is a first-order dimensionless value: tow width divided by local
    centreline radius of curvature. Larger values mean the tow is being asked
    to conform to a tighter bend.
    """

    nominal_feedrate_mm_min: float
    minimum_feedrate_mm_min: float | None = None
    curvature_slowdown_radius_mm: float | None = DEFAULT_CURVATURE_SLOWDOWN_RADIUS_MM
    slip_slowdown_threshold: float | None = DEFAULT_SLIP_SLOWDOWN_THRESHOLD
    slip_max_risk: float = DEFAULT_SLIP_MAX_RISK

    def __post_init__(self) -> None:
        if not np.isfinite(self.nominal_feedrate_mm_min) or self.nominal_feedrate_mm_min <= 0.0:
            raise ValueError("nominal_feedrate_mm_min must be a positive finite value")
        minimum_feedrate = self.resolved_minimum_feedrate_mm_min
        if not np.isfinite(minimum_feedrate) or minimum_feedrate <= 0.0:
            raise ValueError("minimum_feedrate_mm_min must be a positive finite value")
        if minimum_feedrate > self.nominal_feedrate_mm_min:
            raise ValueError("minimum_feedrate_mm_min cannot exceed nominal_feedrate_mm_min")
        if self.curvature_slowdown_radius_mm is not None and (
            not np.isfinite(self.curvature_slowdown_radius_mm)
            or self.curvature_slowdown_radius_mm <= 0.0
        ):
            raise ValueError("curvature_slowdown_radius_mm must be positive when provided")
        if self.slip_slowdown_threshold is not None:
            if not np.isfinite(self.slip_slowdown_threshold) or self.slip_slowdown_threshold < 0.0:
                raise ValueError("slip_slowdown_threshold must be non-negative when provided")
            if not np.isfinite(self.slip_max_risk) or self.slip_max_risk <= 0.0:
                raise ValueError("slip_max_risk must be a positive finite value")
            if self.slip_max_risk <= self.slip_slowdown_threshold:
                raise ValueError("slip_max_risk must be greater than slip_slowdown_threshold")

    @property
    def resolved_minimum_feedrate_mm_min(self) -> float:
        if self.minimum_feedrate_mm_min is None:
            return self.nominal_feedrate_mm_min * DEFAULT_MINIMUM_FEED_FRACTION
        return self.minimum_feedrate_mm_min


@dataclass(frozen=True, slots=True)
class FeedSchedule:
    """Per-point feedrate and diagnostic values for a surface path."""

    feedrate_mm_min: FloatArray
    curvature_1_per_mm: FloatArray
    curvature_radius_mm: FloatArray
    slip_risk: FloatArray

    def __post_init__(self) -> None:
        arrays = {
            "feedrate_mm_min": np.asarray(self.feedrate_mm_min, dtype=float),
            "curvature_1_per_mm": np.asarray(self.curvature_1_per_mm, dtype=float),
            "curvature_radius_mm": np.asarray(self.curvature_radius_mm, dtype=float),
            "slip_risk": np.asarray(self.slip_risk, dtype=float),
        }
        shapes = {values.shape for values in arrays.values()}
        if len(shapes) != 1:
            raise ValueError("feed schedule arrays must all have the same shape")
        for name, values in arrays.items():
            if values.ndim != 1:
                raise ValueError(f"{name} must be one-dimensional")
            if np.any(np.isnan(values)):
                raise ValueError(f"{name} cannot contain NaN values")
            object.__setattr__(self, name, values)
        if self.feedrate_mm_min.size < 2:
            raise ValueError("a feed schedule needs at least two points")
        if np.any(~np.isfinite(self.feedrate_mm_min)) or np.any(self.feedrate_mm_min <= 0.0):
            raise ValueError("feedrate_mm_min must contain positive finite values")
        if np.any(~np.isfinite(self.curvature_1_per_mm)) or np.any(
            self.curvature_1_per_mm < 0.0
        ):
            raise ValueError("curvature_1_per_mm must contain non-negative finite values")
        if np.any(self.curvature_radius_mm < 0.0):
            raise ValueError("curvature_radius_mm must contain non-negative values")
        if np.any(~np.isfinite(self.slip_risk)) or np.any(self.slip_risk < 0.0):
            raise ValueError("slip_risk must contain non-negative finite values")

    @property
    def point_count(self) -> int:
        return int(self.feedrate_mm_min.size)

    @property
    def min_feedrate_mm_min(self) -> float:
        return float(np.min(self.feedrate_mm_min))

    @property
    def max_feedrate_mm_min(self) -> float:
        return float(np.max(self.feedrate_mm_min))

    @property
    def max_slip_risk(self) -> float:
        return float(np.max(self.slip_risk))

    @property
    def max_curvature_1_per_mm(self) -> float:
        return float(np.max(self.curvature_1_per_mm))

    @property
    def min_curvature_radius_mm(self) -> float:
        finite_radii = self.curvature_radius_mm[np.isfinite(self.curvature_radius_mm)]
        if finite_radii.size == 0:
            return float("inf")
        return float(np.min(finite_radii))


def plan_feedrate(surface_path: SurfacePath, config: FeedrateConfig) -> FeedSchedule:
    """Build an adaptive per-point feedrate schedule for a surface path."""

    points = surface_path.points_mm
    arc_length_mm = _strict_arc_lengths(points)
    curvature = _local_curvature(points, arc_length_mm)
    curvature_radius = np.full_like(curvature, np.inf)
    curved = curvature > 1e-12
    curvature_radius[curved] = 1.0 / curvature[curved]
    slip_risk = surface_path.tow_width_mm * curvature

    minimum_feedrate = config.resolved_minimum_feedrate_mm_min
    minimum_fraction = minimum_feedrate / config.nominal_feedrate_mm_min
    feed_factor = np.ones(surface_path.point_count, dtype=float)

    if config.curvature_slowdown_radius_mm is not None:
        curvature_factor = np.ones(surface_path.point_count, dtype=float)
        tight = curvature_radius < config.curvature_slowdown_radius_mm
        curvature_factor[tight] = np.clip(
            curvature_radius[tight] / config.curvature_slowdown_radius_mm,
            minimum_fraction,
            1.0,
        )
        feed_factor = np.minimum(feed_factor, curvature_factor)

    if config.slip_slowdown_threshold is not None:
        slip_fraction = np.clip(
            (slip_risk - config.slip_slowdown_threshold)
            / (config.slip_max_risk - config.slip_slowdown_threshold),
            0.0,
            1.0,
        )
        slip_factor = 1.0 - slip_fraction * (1.0 - minimum_fraction)
        feed_factor = np.minimum(feed_factor, slip_factor)

    feedrate = np.clip(
        config.nominal_feedrate_mm_min * feed_factor,
        minimum_feedrate,
        config.nominal_feedrate_mm_min,
    )
    feedrate = _smooth_feedrate(feedrate, minimum_feedrate, config.nominal_feedrate_mm_min)
    return FeedSchedule(
        feedrate_mm_min=feedrate,
        curvature_1_per_mm=curvature,
        curvature_radius_mm=curvature_radius,
        slip_risk=slip_risk,
    )


def _strict_arc_lengths(points_mm: FloatArray) -> FloatArray:
    segment_lengths = np.linalg.norm(np.diff(points_mm, axis=0), axis=1)
    arc_length = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    if arc_length[-1] <= 0.0:
        raise ValueError("surface path has zero geometric length")
    for index in range(1, arc_length.size):
        if arc_length[index] <= arc_length[index - 1]:
            arc_length[index] = arc_length[index - 1] + 1e-9
    return arc_length.astype(float, copy=False)


def _local_curvature(points_mm: FloatArray, arc_length_mm: FloatArray) -> FloatArray:
    first_derivative = _differentiate(points_mm, arc_length_mm)
    second_derivative = _differentiate(first_derivative, arc_length_mm)
    numerator = np.linalg.norm(np.cross(first_derivative, second_derivative), axis=1)
    speed = np.linalg.norm(first_derivative, axis=1)
    denominator = speed**3
    return np.divide(
        numerator,
        denominator,
        out=np.zeros(points_mm.shape[0], dtype=float),
        where=denominator > 1e-12,
    )


def _differentiate(values: FloatArray, positions: FloatArray) -> FloatArray:
    derivative = np.empty_like(values)
    if values.shape[0] == 2:
        derivative[:] = (values[1] - values[0]) / (positions[1] - positions[0])
        return derivative
    derivative[0] = (values[1] - values[0]) / (positions[1] - positions[0])
    derivative[-1] = (values[-1] - values[-2]) / (positions[-1] - positions[-2])
    derivative[1:-1] = (
        (values[2:] - values[:-2]) / (positions[2:] - positions[:-2])[:, np.newaxis]
    )
    return derivative


def _smooth_feedrate(
    feedrate_mm_min: FloatArray,
    minimum_feedrate_mm_min: float,
    nominal_feedrate_mm_min: float,
) -> FloatArray:
    if feedrate_mm_min.size < 5:
        return feedrate_mm_min
    smoothed = feedrate_mm_min.astype(float, copy=True)
    kernel = np.asarray([1.0, 2.0, 3.0, 2.0, 1.0], dtype=float)
    kernel /= float(np.sum(kernel))
    for _ in range(2):
        padded = np.pad(smoothed, (2, 2), mode="edge")
        smoothed = np.convolve(padded, kernel, mode="valid")
    smoothed = np.clip(smoothed, minimum_feedrate_mm_min, nominal_feedrate_mm_min)
    return smoothed
