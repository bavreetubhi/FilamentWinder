"""Axisymmetric mandrel helpers and generated profiles."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from filament_winder.core.geometry.profile import AxisymmetricProfileMandrel

FloatArray = NDArray[np.float64]
MandrelRegion = Literal["left_dome", "cylinder", "right_dome", "polar_opening"]


def cylinder_with_domes_profile(
    *,
    cylinder_length_mm: float,
    cylinder_radius_mm: float,
    left_dome_length_mm: float,
    right_dome_length_mm: float,
    polar_opening_radius_mm: float = 0.0,
    samples_per_region: int = 120,
    name: str = "cylinder-with-domes",
) -> AxisymmetricProfileMandrel:
    """Create a smooth axisymmetric pressure-vessel style Z-R profile.

    The dome profile is a rounded spherical-cap style section that starts at
    the polar opening radius and reaches the cylinder radius tangent to the
    straight cylinder. It is intentionally deterministic and simple enough for
    path validation tests; imported DXF profiles remain the route for exact
    tooling.
    """

    if cylinder_length_mm <= 0.0:
        raise ValueError("cylinder_length_mm must be positive")
    if cylinder_radius_mm <= 0.0:
        raise ValueError("cylinder_radius_mm must be positive")
    if left_dome_length_mm < 0.0 or right_dome_length_mm < 0.0:
        raise ValueError("dome lengths must be non-negative")
    if polar_opening_radius_mm < 0.0 or polar_opening_radius_mm >= cylinder_radius_mm:
        raise ValueError("polar_opening_radius_mm must be less than cylinder_radius_mm")
    if samples_per_region < 8:
        raise ValueError("samples_per_region must be at least 8")

    z_chunks: list[FloatArray] = []
    r_chunks: list[FloatArray] = []
    current_z = 0.0

    if left_dome_length_mm > 0.0:
        t = np.linspace(0.0, 1.0, samples_per_region, endpoint=False)
        z_chunks.append(current_z + t * left_dome_length_mm)
        r_chunks.append(_rounded_dome_radius(t, polar_opening_radius_mm, cylinder_radius_mm))
        current_z += left_dome_length_mm

    cylinder_samples = max(2, samples_per_region)
    z_chunks.append(np.linspace(current_z, current_z + cylinder_length_mm, cylinder_samples))
    r_chunks.append(np.full(cylinder_samples, cylinder_radius_mm, dtype=float))
    current_z += cylinder_length_mm

    if right_dome_length_mm > 0.0:
        t = np.linspace(0.0, 1.0, samples_per_region + 1)[1:]
        z_chunks.append(current_z + t * right_dome_length_mm)
        r_chunks.append(
            _rounded_dome_radius(1.0 - t, polar_opening_radius_mm, cylinder_radius_mm)
        )

    return AxisymmetricProfileMandrel(
        z_mm=np.concatenate(z_chunks),
        r_mm=np.concatenate(r_chunks),
        name=name,
    )


def classify_regions(
    mandrel: AxisymmetricProfileMandrel,
    *,
    cylinder_radius_mm: float | None = None,
    polar_opening_radius_mm: float = 0.0,
    tolerance_mm: float = 1e-6,
) -> tuple[MandrelRegion, ...]:
    """Classify each profile sample into dome/cylinder/polar regions."""

    max_radius = mandrel.max_radius_mm if cylinder_radius_mm is None else cylinder_radius_mm
    regions: list[MandrelRegion] = []
    midpoint = (mandrel.start_z_mm + mandrel.end_z_mm) / 2.0
    for z_value, radius in zip(mandrel.z_mm, mandrel.r_mm, strict=True):
        if radius <= polar_opening_radius_mm + tolerance_mm:
            regions.append("polar_opening")
        elif radius >= max_radius - tolerance_mm:
            regions.append("cylinder")
        elif z_value < midpoint:
            regions.append("left_dome")
        else:
            regions.append("right_dome")
    return tuple(regions)


def _rounded_dome_radius(
    t: FloatArray,
    polar_opening_radius_mm: float,
    cylinder_radius_mm: float,
) -> FloatArray:
    t_clipped = np.clip(t, 0.0, 1.0)
    cap_depth_mm = np.sqrt(
        max(cylinder_radius_mm**2 - polar_opening_radius_mm**2, 0.0)
    )
    axial_from_pole_mm = cap_depth_mm * t_clipped
    radius = np.sqrt(
        np.clip(
            cylinder_radius_mm**2 - (cap_depth_mm - axial_from_pole_mm) ** 2,
            0.0,
            None,
        )
    )
    return np.maximum(radius, polar_opening_radius_mm)
