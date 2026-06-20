"""Tow band generation for cylinder paths."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from filament_winder.core.geometry import CylinderMandrel
from filament_winder.core.path_planning import SurfacePath

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]


@dataclass(frozen=True, slots=True)
class TowBand:
    """Full-width tow strip around a centreline path."""

    width_mm: float
    left_z_mm: FloatArray
    left_theta_rad: FloatArray
    right_z_mm: FloatArray
    right_theta_rad: FloatArray
    left_points_mm: FloatArray
    right_points_mm: FloatArray

    def __post_init__(self) -> None:
        arrays = {
            "left_z_mm": np.asarray(self.left_z_mm, dtype=float),
            "left_theta_rad": np.asarray(self.left_theta_rad, dtype=float),
            "right_z_mm": np.asarray(self.right_z_mm, dtype=float),
            "right_theta_rad": np.asarray(self.right_theta_rad, dtype=float),
        }
        point_arrays = {
            "left_points_mm": np.asarray(self.left_points_mm, dtype=float),
            "right_points_mm": np.asarray(self.right_points_mm, dtype=float),
        }
        shapes = {values.shape for values in arrays.values()}
        if len(shapes) != 1:
            raise ValueError("tow band edge arrays must have the same shape")
        for name, values in arrays.items():
            if values.ndim != 1:
                raise ValueError(f"{name} must be one-dimensional")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain only finite values")
            object.__setattr__(self, name, values)
        for name, values in point_arrays.items():
            if values.shape != (self.point_count, 3):
                raise ValueError(f"{name} must have shape (point_count, 3)")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain only finite values")
            object.__setattr__(self, name, values)
        if self.width_mm < 0.0 or not np.isfinite(self.width_mm):
            raise ValueError("width_mm must be a non-negative finite value")

    @property
    def point_count(self) -> int:
        return int(np.asarray(self.left_z_mm).size)

    @property
    def vertices_mm(self) -> FloatArray:
        vertices = np.empty((self.point_count * 2, 3), dtype=float)
        vertices[0::2] = self.left_points_mm
        vertices[1::2] = self.right_points_mm
        return vertices

    @property
    def quad_indices(self) -> IntArray:
        if self.point_count < 2:
            return np.empty((0, 4), dtype=int)
        quads = []
        for index in range(self.point_count - 1):
            quads.append((2 * index, 2 * index + 1, 2 * index + 3, 2 * index + 2))
        return np.asarray(quads, dtype=int)


def generate_cylinder_tow_band(mandrel: CylinderMandrel, surface_path: SurfacePath) -> TowBand:
    """Generate a full-width tow band on a cylinder surface.

    The edge offsets are calculated in the unwrapped cylinder plane. The strip
    is perpendicular to the local helical tangent, then projected back onto the
    cylinder.
    """

    half_width_mm = surface_path.tow_width_mm / 2.0
    alpha_rad = np.deg2rad(surface_path.winding_angle_deg)
    z_offset_mm = -np.sin(alpha_rad) * half_width_mm
    circumferential_offset_mm = np.cos(alpha_rad) * half_width_mm
    centre_s_mm = mandrel.radius_mm * surface_path.theta_rad

    left_z_mm = np.clip(surface_path.z_mm + z_offset_mm, 0.0, mandrel.length_mm)
    right_z_mm = np.clip(surface_path.z_mm - z_offset_mm, 0.0, mandrel.length_mm)
    left_theta_rad = (centre_s_mm + circumferential_offset_mm) / mandrel.radius_mm
    right_theta_rad = (centre_s_mm - circumferential_offset_mm) / mandrel.radius_mm

    return TowBand(
        width_mm=surface_path.tow_width_mm,
        left_z_mm=left_z_mm,
        left_theta_rad=left_theta_rad,
        right_z_mm=right_z_mm,
        right_theta_rad=right_theta_rad,
        left_points_mm=mandrel.surface_points(left_z_mm, left_theta_rad),
        right_points_mm=mandrel.surface_points(right_z_mm, right_theta_rad),
    )
