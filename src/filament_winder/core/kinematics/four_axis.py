"""First-order A/X/Z/B machine mapping."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from filament_winder.core.path_planning import SurfacePath

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class MachineMotionTable:
    """Vectorised machine axis positions for the same points as a surface path."""

    a_deg: FloatArray
    x_mm: FloatArray
    z_mm: FloatArray
    b_deg: FloatArray

    def __post_init__(self) -> None:
        arrays = {
            "a_deg": np.asarray(self.a_deg, dtype=float),
            "x_mm": np.asarray(self.x_mm, dtype=float),
            "z_mm": np.asarray(self.z_mm, dtype=float),
            "b_deg": np.asarray(self.b_deg, dtype=float),
        }
        shapes = {values.shape for values in arrays.values()}
        if len(shapes) != 1:
            raise ValueError("machine motion arrays must all have the same shape")
        for name, values in arrays.items():
            if values.ndim != 1:
                raise ValueError(f"{name} must be one-dimensional")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain only finite values")
            object.__setattr__(self, name, values)
        if len(self.a_deg) < 2:
            raise ValueError("a motion table needs at least two points")

    @property
    def point_count(self) -> int:
        return int(self.a_deg.size)

    def rows(self) -> Iterator[dict[str, float | int]]:
        for index in range(self.point_count):
            yield {
                "index": index,
                "A_deg": float(self.a_deg[index]),
                "X_mm": float(self.x_mm[index]),
                "Z_mm": float(self.z_mm[index]),
                "B_deg": float(self.b_deg[index]),
            }


def machine_path_from_surface_path(
    surface_path: SurfacePath,
    *,
    radial_clearance_mm: float,
    tow_eye_angle_deg: float | FloatArray | None = None,
) -> MachineMotionTable:
    """Map surface path points to A/X/Z/B machine positions.

    Version 0.1 uses the direct cylinder mapping:
    A = theta in degrees, Z = longitudinal position, X = radius + clearance,
    B = requested winding angle.
    """

    if not np.isfinite(radial_clearance_mm) or radial_clearance_mm < 0:
        raise ValueError("radial_clearance_mm must be a non-negative finite value")
    if tow_eye_angle_deg is None:
        if surface_path.tow_eye_angle_deg is None:
            b_deg = np.full(surface_path.point_count, surface_path.winding_angle_deg, dtype=float)
        else:
            b_deg = surface_path.tow_eye_angle_deg
    elif np.isscalar(tow_eye_angle_deg):
        b_angle = cast(float, tow_eye_angle_deg)
        if not np.isfinite(b_angle):
            raise ValueError("tow_eye_angle_deg must be finite")
        b_deg = np.full(surface_path.point_count, b_angle, dtype=float)
    else:
        b_deg = np.asarray(tow_eye_angle_deg, dtype=float)
        if b_deg.shape != surface_path.z_mm.shape:
            raise ValueError("tow_eye_angle_deg must have the same shape as the surface path")
        if b_deg.ndim != 1 or not np.all(np.isfinite(b_deg)):
            raise ValueError("tow_eye_angle_deg must be a finite one-dimensional array")
    theta_unwrapped_rad = np.unwrap(surface_path.theta_rad)
    return MachineMotionTable(
        a_deg=np.rad2deg(theta_unwrapped_rad),
        x_mm=surface_path.surface_radius_mm + radial_clearance_mm,
        z_mm=surface_path.z_mm,
        b_deg=b_deg,
    )
