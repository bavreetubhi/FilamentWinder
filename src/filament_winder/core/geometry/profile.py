"""Axisymmetric Z-R profile mandrel geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _as_float_array(values: Any, *, name: str) -> FloatArray:
    array = np.atleast_1d(np.asarray(values, dtype=float))
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite numeric values")
    return array


@dataclass(frozen=True, slots=True)
class AxisymmetricProfileMandrel:
    """Mandrel described by a longitudinal Z-R side profile."""

    z_mm: FloatArray
    r_mm: FloatArray
    name: str = "axisymmetric-profile"

    def __post_init__(self) -> None:
        z_values = _as_float_array(self.z_mm, name="z_mm")
        r_values = _as_float_array(self.r_mm, name="r_mm")
        if z_values.ndim != 1 or r_values.ndim != 1:
            raise ValueError("z_mm and r_mm must be one-dimensional")
        if z_values.shape != r_values.shape:
            raise ValueError("z_mm and r_mm must have the same shape")
        if z_values.size < 2:
            raise ValueError("a profile mandrel needs at least two points")
        if np.any(np.diff(z_values) <= 0.0):
            raise ValueError("z_mm values must be strictly increasing")
        if np.any(r_values < 0.0):
            raise ValueError("r_mm values must be non-negative")
        object.__setattr__(self, "z_mm", z_values)
        object.__setattr__(self, "r_mm", r_values)

    @property
    def start_z_mm(self) -> float:
        return float(self.z_mm[0])

    @property
    def end_z_mm(self) -> float:
        return float(self.z_mm[-1])

    @property
    def length_mm(self) -> float:
        return self.end_z_mm - self.start_z_mm

    @property
    def max_radius_mm(self) -> float:
        return float(np.max(self.r_mm))

    def validate_z_range(self, z_mm: Any, *, tolerance_mm: float = 1e-9) -> FloatArray:
        z_values = _as_float_array(z_mm, name="z_mm")
        too_low = z_values < self.start_z_mm - tolerance_mm
        too_high = z_values > self.end_z_mm + tolerance_mm
        if np.any(too_low | too_high):
            raise ValueError(
                f"z_mm values must stay within {self.start_z_mm:g} and "
                f"{self.end_z_mm:g} mm for this profile"
            )
        return z_values

    def radius_at(self, z_mm: Any) -> FloatArray:
        z_values = self.validate_z_range(z_mm)
        return np.interp(z_values, self.z_mm, self.r_mm).astype(float, copy=False)

    def dr_dz_at(self, z_mm: Any) -> FloatArray:
        z_values = self.validate_z_range(z_mm)
        slopes = np.gradient(self.r_mm, self.z_mm)
        return np.interp(z_values, self.z_mm, slopes).astype(float, copy=False)

    def d2r_dz2_at(self, z_mm: Any) -> FloatArray:
        z_values = self.validate_z_range(z_mm)
        slopes = np.gradient(self.r_mm, self.z_mm)
        curvature = np.gradient(slopes, self.z_mm)
        return np.interp(z_values, self.z_mm, curvature).astype(float, copy=False)

    def meridional_arc_length_at(self, z_mm: Any) -> FloatArray:
        z_values = self.validate_z_range(z_mm)
        slopes = np.gradient(self.r_mm, self.z_mm)
        segment_lengths = np.sqrt(1.0 + slopes[:-1] ** 2) * np.diff(self.z_mm)
        arc = np.concatenate(([0.0], np.cumsum(segment_lengths)))
        return np.interp(z_values, self.z_mm, arc).astype(float, copy=False)

    def surface_tangent_z(self, z_mm: Any, theta_rad: Any) -> FloatArray:
        z_values = self.validate_z_range(z_mm)
        theta_values = _as_float_array(theta_rad, name="theta_rad")
        z_broadcast, theta_broadcast = np.broadcast_arrays(z_values, theta_values)
        slope = self.dr_dz_at(z_broadcast)
        tangent = np.stack(
            (
                slope * np.cos(theta_broadcast),
                slope * np.sin(theta_broadcast),
                np.ones(z_broadcast.shape, dtype=float),
            ),
            axis=-1,
        )
        return _normalize_rows(tangent)

    def surface_tangent_theta(self, z_mm: Any, theta_rad: Any) -> FloatArray:
        z_values = self.validate_z_range(z_mm)
        theta_values = _as_float_array(theta_rad, name="theta_rad")
        z_broadcast, theta_broadcast = np.broadcast_arrays(z_values, theta_values)
        return np.stack(
            (
                -np.sin(theta_broadcast),
                np.cos(theta_broadcast),
                np.zeros(z_broadcast.shape, dtype=float),
            ),
            axis=-1,
        )

    def surface_normal(self, z_mm: Any, theta_rad: Any) -> FloatArray:
        z_values = self.validate_z_range(z_mm)
        theta_values = _as_float_array(theta_rad, name="theta_rad")
        z_broadcast, theta_broadcast = np.broadcast_arrays(z_values, theta_values)
        slope = self.dr_dz_at(z_broadcast)
        normal = np.stack(
            (
                np.cos(theta_broadcast),
                np.sin(theta_broadcast),
                -slope,
            ),
            axis=-1,
        )
        return _normalize_rows(normal)

    def meridional_curvature_at(self, z_mm: Any) -> FloatArray:
        z_values = self.validate_z_range(z_mm)
        first = self.dr_dz_at(z_values)
        second = self.d2r_dz2_at(z_values)
        return np.abs(second) / np.maximum((1.0 + first**2) ** 1.5, 1e-12)

    def circumferential_curvature_at(self, z_mm: Any) -> FloatArray:
        z_values = self.validate_z_range(z_mm)
        radius = np.maximum(self.radius_at(z_values), 1e-9)
        slope = self.dr_dz_at(z_values)
        return 1.0 / (radius * np.sqrt(1.0 + slope**2))

    def surface_points(self, z_mm: Any, theta_rad: Any) -> FloatArray:
        """Return surface points P(z, theta) as [x, y, z] rows."""

        z_values = self.validate_z_range(z_mm)
        theta_values = _as_float_array(theta_rad, name="theta_rad")
        z_broadcast, theta_broadcast = np.broadcast_arrays(z_values, theta_values)
        radii = self.radius_at(z_broadcast)
        x_mm = radii * np.cos(theta_broadcast)
        y_mm = radii * np.sin(theta_broadcast)
        return np.stack((x_mm, y_mm, z_broadcast), axis=-1).astype(float, copy=False)


def _normalize_rows(values: FloatArray) -> FloatArray:
    norms = np.linalg.norm(values, axis=-1)
    return values / np.maximum(norms[..., np.newaxis], 1e-12)
