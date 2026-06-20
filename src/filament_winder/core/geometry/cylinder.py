"""Cylinder mandrel geometry."""

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
class CylinderMandrel:
    """Constant-radius cylindrical mandrel.

    The cylinder follows the project convention where Z is the longitudinal
    machine axis and R is radial distance from the mandrel centreline.
    """

    length_mm: float
    radius_mm: float
    name: str = "cylinder"

    def __post_init__(self) -> None:
        if not np.isfinite(self.length_mm) or self.length_mm <= 0:
            raise ValueError("length_mm must be a positive finite value")
        if not np.isfinite(self.radius_mm) or self.radius_mm <= 0:
            raise ValueError("radius_mm must be a positive finite value")

    @property
    def diameter_mm(self) -> float:
        return self.radius_mm * 2.0

    def validate_z_range(self, z_mm: Any, *, tolerance_mm: float = 1e-9) -> FloatArray:
        z_values = _as_float_array(z_mm, name="z_mm")
        too_low = z_values < -tolerance_mm
        too_high = z_values > self.length_mm + tolerance_mm
        if np.any(too_low | too_high):
            raise ValueError(
                f"z_mm values must stay within 0 and {self.length_mm:g} mm for this mandrel"
            )
        return z_values

    def radius_at(self, z_mm: Any) -> FloatArray:
        z_values = self.validate_z_range(z_mm)
        return np.full(z_values.shape, self.radius_mm, dtype=float)

    def dr_dz_at(self, z_mm: Any) -> FloatArray:
        z_values = self.validate_z_range(z_mm)
        return np.zeros(z_values.shape, dtype=float)

    def meridional_arc_length_at(self, z_mm: Any) -> FloatArray:
        z_values = self.validate_z_range(z_mm)
        return z_values.astype(float, copy=False)

    def surface_tangent_z(self, z_mm: Any, theta_rad: Any) -> FloatArray:
        z_values = self.validate_z_range(z_mm)
        theta_values = _as_float_array(theta_rad, name="theta_rad")
        z_broadcast, theta_broadcast = np.broadcast_arrays(z_values, theta_values)
        return np.stack(
            (
                np.zeros(z_broadcast.shape, dtype=float),
                np.zeros(theta_broadcast.shape, dtype=float),
                np.ones(z_broadcast.shape, dtype=float),
            ),
            axis=-1,
        )

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
        return np.stack(
            (
                np.cos(theta_broadcast),
                np.sin(theta_broadcast),
                np.zeros(z_broadcast.shape, dtype=float),
            ),
            axis=-1,
        )

    def meridional_curvature_at(self, z_mm: Any) -> FloatArray:
        z_values = self.validate_z_range(z_mm)
        return np.zeros(z_values.shape, dtype=float)

    def circumferential_curvature_at(self, z_mm: Any) -> FloatArray:
        z_values = self.validate_z_range(z_mm)
        return np.full(z_values.shape, 1.0 / self.radius_mm, dtype=float)

    def surface_points(self, z_mm: Any, theta_rad: Any) -> FloatArray:
        """Return surface points P(z, theta) as [x, y, z] rows."""

        z_values = self.validate_z_range(z_mm)
        theta_values = _as_float_array(theta_rad, name="theta_rad")
        z_broadcast, theta_broadcast = np.broadcast_arrays(z_values, theta_values)
        x_mm = self.radius_mm * np.cos(theta_broadcast)
        y_mm = self.radius_mm * np.sin(theta_broadcast)
        return np.stack((x_mm, y_mm, z_broadcast), axis=-1).astype(float, copy=False)
