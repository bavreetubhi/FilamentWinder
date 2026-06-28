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
    dome_shape: str = "isotensoid",
) -> AxisymmetricProfileMandrel:
    """Create an axisymmetric pressure-vessel Z-R profile with domes.

    Parameters
    ----------
    dome_shape : str
        ``"isotensoid"`` (default) — geodesic-equilibrium contour from textbook
        (Clairaut r·sin(α)=constant). The dome axial length is determined by
        the cylinder radius and polar opening radius; the corresponding
        ``left_dome_length_mm`` / ``right_dome_length_mm`` are overridden.
        ``"spherical"`` — original simple spherical-cap shape; dome lengths
        are explicit parameters.
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
        if dome_shape == "isotensoid" and polar_opening_radius_mm > 0.0:
            dome_z, dome_r = _isotensoid_dome_contour(
                samples_per_region, polar_opening_radius_mm, cylinder_radius_mm,
            )
            z_chunks.append(current_z + dome_z[:-1])  # exclude cylinder junction
            r_chunks.append(dome_r[:-1])
            current_z += float(dome_z[-1])
        else:
            t = np.linspace(0.0, 1.0, samples_per_region, endpoint=False)
            z_chunks.append(current_z + t * left_dome_length_mm)
            r_chunks.append(_rounded_dome_radius(t, polar_opening_radius_mm, cylinder_radius_mm))
            current_z += left_dome_length_mm

    cylinder_samples = max(2, samples_per_region)
    z_chunks.append(np.linspace(current_z, current_z + cylinder_length_mm, cylinder_samples))
    r_chunks.append(np.full(cylinder_samples, cylinder_radius_mm, dtype=float))
    current_z += cylinder_length_mm

    if right_dome_length_mm > 0.0:
        if dome_shape == "isotensoid" and polar_opening_radius_mm > 0.0:
            dome_z, dome_r = _isotensoid_dome_contour(
                samples_per_region, polar_opening_radius_mm, cylinder_radius_mm,
            )
            z_chunks.append(current_z + dome_z[1:])   # exclude cylinder junction
            r_chunks.append(dome_r[:0:-1])             # reversed, exclude cylinder junction
        else:
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


def _isotensoid_dome_contour(
    num_samples: int,
    polar_opening_radius_mm: float,
    cylinder_radius_mm: float,
) -> tuple[FloatArray, FloatArray]:
    """Compute the textbook isotensoid (geodesic-equilibrium) dome contour.

    Returns (z_mm, r_mm) for a *left* dome starting at the polar opening
    (z=0, r=r_p) and ending at the cylinder-dome junction (z=H, r=R).

    The meridian satisfies
        z(r) = ∫_{r}^{R} √((u² - r_p²) / (R² - u²)) du
    which is the shape where a geodesic fibre under constant tension is in
    equilibrium (Clairaut r·sin(α)=r_p).  The dome axial length H is
    determined entirely by R and r_p, not by a free parameter.
    """
    R = cylinder_radius_mm
    r_p = max(polar_opening_radius_mm, 1e-9)

    if r_p >= R:
        raise ValueError("polar_opening_radius_mm must be less than cylinder_radius_mm")

    # Sample radius from r_p to R, with extra density near R where the
    # integrand has an integrable singularity.
    n_near = max(8, num_samples // 4)
    n_mid = num_samples - n_near
    r_mid = np.linspace(r_p, R * 0.92, n_mid)
    r_near = R - (R - R * 0.92) * np.linspace(1.0, 0.0, n_near + 1) ** 2
    r_near = r_near[r_near < R]
    r_vals = np.concatenate([r_mid, r_near])
    r_vals = np.unique(r_vals)

    # Cumulative integral z(r) = ∫_{r}^{R} f(u) du via simple midpoint rule
    z_vals = np.zeros_like(r_vals)
    max_f = 50.0 * R  # clip to handle the integrable singularity at u=R
    for i in range(len(r_vals) - 2, -1, -1):
        u_mid = 0.5 * (r_vals[i] + r_vals[i + 1])
        du = r_vals[i + 1] - r_vals[i]
        f = np.sqrt((u_mid ** 2 - r_p ** 2) / max(R ** 2 - u_mid ** 2, 1e-12))
        f = min(f, max_f)
        z_vals[i] = z_vals[i + 1] + f * du

    # z_vals[0] ≈ total dome height H  (at r=r_p)
    # z_vals[-1] = 0                    (at r=R)
    dome_height = z_vals[0]
    z_mm = dome_height - z_vals  # z=0 at polar opening, z=H at cylinder
    r_mm = r_vals                # r=r_p at z=0, r=R at z=H

    return z_mm, r_mm


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
