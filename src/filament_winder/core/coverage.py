"""Approximate cylinder coverage maps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from filament_winder.core.geometry import CylinderMandrel
from filament_winder.core.path_planning import SurfacePath

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]


@dataclass(frozen=True, slots=True)
class CoverageMap:
    """Coverage count on a regular Z-theta grid."""

    z_mm: FloatArray
    theta_rad: FloatArray
    coverage_count: IntArray
    tow_width_mm: float
    winding_angle_deg: float

    def __post_init__(self) -> None:
        z_values = np.asarray(self.z_mm, dtype=float)
        theta_values = np.asarray(self.theta_rad, dtype=float)
        coverage = np.asarray(self.coverage_count, dtype=int)
        if z_values.ndim != 1 or theta_values.ndim != 1:
            raise ValueError("z_mm and theta_rad must be one-dimensional")
        if coverage.shape != (z_values.size, theta_values.size):
            raise ValueError("coverage_count must have shape (len(z_mm), len(theta_rad))")
        if not np.all(np.isfinite(z_values)) or not np.all(np.isfinite(theta_values)):
            raise ValueError("coverage coordinates must be finite")
        if np.any(coverage < 0):
            raise ValueError("coverage_count cannot be negative")
        object.__setattr__(self, "z_mm", z_values)
        object.__setattr__(self, "theta_rad", theta_values)
        object.__setattr__(self, "coverage_count", coverage)

    @property
    def covered(self) -> NDArray[np.bool_]:
        return self.coverage_count > 0

    @property
    def gap_fraction(self) -> float:
        return float(np.mean(self.coverage_count == 0))

    @property
    def covered_fraction(self) -> float:
        return float(np.mean(self.coverage_count > 0))

    @property
    def overlap_fraction(self) -> float:
        return float(np.mean(self.coverage_count > 1))

    def summary(self) -> CoverageSummary:
        return CoverageSummary(
            covered_fraction=self.covered_fraction,
            gap_fraction=self.gap_fraction,
            overlap_fraction=self.overlap_fraction,
            max_coverage_count=int(np.max(self.coverage_count)),
            mean_coverage_count=float(np.mean(self.coverage_count)),
        )


@dataclass(frozen=True, slots=True)
class CoverageSummary:
    covered_fraction: float
    gap_fraction: float
    overlap_fraction: float
    max_coverage_count: int
    mean_coverage_count: float

    @property
    def covered_percent(self) -> float:
        return self.covered_fraction * 100.0

    @property
    def gap_percent(self) -> float:
        return self.gap_fraction * 100.0

    @property
    def overlap_percent(self) -> float:
        return self.overlap_fraction * 100.0


def cylinder_coverage_map(
    mandrel: CylinderMandrel,
    surface_path: SurfacePath,
    *,
    z_samples: int = 120,
    theta_samples: int = 180,
) -> CoverageMap:
    """Approximate tow coverage on a cylinder using an unwrapped Z-S plane."""

    if z_samples < 2:
        raise ValueError("z_samples must be at least 2")
    if theta_samples < 4:
        raise ValueError("theta_samples must be at least 4")

    z_mm = np.linspace(0.0, mandrel.length_mm, z_samples)
    theta_rad = np.linspace(0.0, 2.0 * np.pi, theta_samples, endpoint=False)
    z_grid, theta_grid = np.meshgrid(z_mm, theta_rad, indexing="ij")

    actual_s_mm = mandrel.radius_mm * theta_grid
    circumference_mm = 2.0 * np.pi * mandrel.radius_mm
    coverage_count = np.zeros(z_grid.shape, dtype=int)
    pass_index = (
        np.zeros(surface_path.point_count, dtype=int)
        if surface_path.pass_index is None
        else surface_path.pass_index
    )

    for start_index, end_index in _contiguous_pass_spans(pass_index):
        pass_z = surface_path.z_mm[start_index:end_index]
        pass_theta = surface_path.theta_rad[start_index:end_index]
        if pass_z.size < 2:
            continue
        if np.isclose(pass_z[-1], pass_z[0]):
            coverage_count += (
                np.abs(z_grid - float(pass_z[0])) <= surface_path.tow_width_mm / 2.0
            ).astype(int)
            continue
        slope_ds_dz = mandrel.radius_mm * (pass_theta[-1] - pass_theta[0])
        slope_ds_dz /= pass_z[-1] - pass_z[0]
        start_s_mm = mandrel.radius_mm * pass_theta[0]
        start_z_mm = pass_z[0]
        expected_s_mm = start_s_mm + slope_ds_dz * (z_grid - start_z_mm)
        signed_periodic_offset_mm = _wrap_periodic(
            actual_s_mm - expected_s_mm,
            period=circumference_mm,
        )
        perpendicular_distance_mm = np.abs(signed_periodic_offset_mm) / np.sqrt(
            slope_ds_dz**2 + 1.0
        )
        coverage_count += (
            perpendicular_distance_mm <= surface_path.tow_width_mm / 2.0
        ).astype(int)
    return CoverageMap(
        z_mm=z_mm,
        theta_rad=theta_rad,
        coverage_count=coverage_count,
        tow_width_mm=surface_path.tow_width_mm,
        winding_angle_deg=surface_path.winding_angle_deg,
    )


def _wrap_periodic(values: FloatArray, *, period: float) -> FloatArray:
    return ((values + period / 2.0) % period) - period / 2.0


def _contiguous_pass_spans(pass_index: IntArray) -> tuple[tuple[int, int], ...]:
    if pass_index.size == 0:
        return ()
    spans = []
    start = 0
    for index in range(1, pass_index.size):
        if pass_index[index] != pass_index[index - 1]:
            spans.append((start, index))
            start = index
    spans.append((start, pass_index.size))
    return tuple(spans)
