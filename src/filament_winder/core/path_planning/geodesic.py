"""Axisymmetric geodesic and controlled-angle path generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from filament_winder.core.geometry import AxisymmetricProfileMandrel
from filament_winder.core.path_planning.helical import SurfacePath

Direction = Literal["positive", "negative"]


@dataclass(frozen=True, slots=True)
class GeodesicPathConfig:
    initial_angle_deg: float
    tow_width_mm: float
    start_z_mm: float
    end_z_mm: float
    start_theta_rad: float = 0.0
    direction: Direction = "positive"
    turnaround_radius_mm: float | None = None
    point_count: int = 400


@dataclass(frozen=True, slots=True)
class ControlledAnglePathConfig:
    target_angle_deg: float
    tow_width_mm: float
    start_z_mm: float
    end_z_mm: float
    start_theta_rad: float = 0.0
    direction: Direction = "positive"
    allowed_angle_error_deg: float = 5.0
    high_slip_risk_deg: float = 15.0
    smoothing_strength: float = 1.0
    point_count: int = 400


@dataclass(frozen=True, slots=True)
class AxisymmetricPathDiagnostics:
    clairaut_constant_mm: float
    min_radius_mm: float
    max_slip_risk_deg: float
    warning_flags: tuple[str, ...]


def generate_geodesic_path(
    mandrel: AxisymmetricProfileMandrel,
    config: GeodesicPathConfig,
) -> tuple[SurfacePath, AxisymmetricPathDiagnostics]:
    _validate_angle(config.initial_angle_deg)
    z_mm = _z_samples(config.start_z_mm, config.end_z_mm, config.point_count, config.direction)
    radius = mandrel.radius_at(z_mm)
    turnaround_radius = config.turnaround_radius_mm
    if turnaround_radius is not None:
        valid = radius >= turnaround_radius
        z_mm = z_mm[valid]
        radius = radius[valid]
    if z_mm.size < 2:
        raise ValueError("geodesic path has fewer than two valid points")
    start_radius = float(radius[0])
    constant = start_radius * np.sin(np.deg2rad(abs(config.initial_angle_deg)))
    sin_alpha = np.clip(constant / np.maximum(radius, 1e-9), -0.999999, 0.999999)
    alpha = np.arcsin(sin_alpha)
    theta_rad = _integrate_theta(
        mandrel,
        z_mm,
        alpha,
        start_theta_rad=config.start_theta_rad,
        direction=config.direction,
    )
    points = mandrel.surface_points(z_mm, theta_rad)
    warnings = _geodesic_warnings(radius, constant, turnaround_radius)
    path = SurfacePath(
        z_mm=z_mm,
        theta_rad=theta_rad,
        x_mm=points[:, 0],
        y_mm=points[:, 1],
        winding_angle_deg=float(np.rad2deg(np.mean(np.abs(alpha)))),
        tow_width_mm=config.tow_width_mm,
        pass_index=np.zeros(z_mm.shape, dtype=int),
        tow_eye_angle_deg=np.rad2deg(alpha),
    )
    return path, AxisymmetricPathDiagnostics(
        clairaut_constant_mm=float(constant),
        min_radius_mm=float(np.min(radius)),
        max_slip_risk_deg=0.0,
        warning_flags=warnings,
    )


def generate_controlled_angle_path(
    mandrel: AxisymmetricProfileMandrel,
    config: ControlledAnglePathConfig,
) -> tuple[SurfacePath, AxisymmetricPathDiagnostics]:
    _validate_angle(config.target_angle_deg)
    z_mm = _z_samples(config.start_z_mm, config.end_z_mm, config.point_count, config.direction)
    radius = mandrel.radius_at(z_mm)
    target = np.full(z_mm.shape, np.deg2rad(abs(config.target_angle_deg)), dtype=float)
    start_constant = float(radius[0]) * np.sin(target[0])
    geodesic_angle = np.arcsin(
        np.clip(start_constant / np.maximum(radius, 1e-9), -0.999999, 0.999999)
    )
    allowed_deviation = np.deg2rad(config.allowed_angle_error_deg)
    requested = geodesic_angle + np.clip(
        target - geodesic_angle,
        -allowed_deviation,
        allowed_deviation,
    )
    blend = np.clip(config.smoothing_strength, 0.0, 1.0)
    requested = geodesic_angle * (1.0 - blend) + requested * blend
    slip_risk_deg = np.abs(np.rad2deg(requested - geodesic_angle))
    theta_rad = _integrate_theta(
        mandrel,
        z_mm,
        requested,
        start_theta_rad=config.start_theta_rad,
        direction=config.direction,
    )
    points = mandrel.surface_points(z_mm, theta_rad)
    warnings = []
    max_slip = float(np.max(slip_risk_deg))
    if max_slip > config.high_slip_risk_deg:
        warnings.append(f"high slip risk {max_slip:.3f} deg")
    elif max_slip > config.allowed_angle_error_deg:
        warnings.append(f"medium slip risk {max_slip:.3f} deg")
    if float(np.max(np.abs(np.rad2deg(target - requested)))) > 1e-6:
        warnings.append("target angle blended toward geodesic to reduce slip")
    path = SurfacePath(
        z_mm=z_mm,
        theta_rad=theta_rad,
        x_mm=points[:, 0],
        y_mm=points[:, 1],
        winding_angle_deg=abs(config.target_angle_deg),
        tow_width_mm=config.tow_width_mm,
        pass_index=np.zeros(z_mm.shape, dtype=int),
        tow_eye_angle_deg=np.rad2deg(requested),
    )
    return path, AxisymmetricPathDiagnostics(
        clairaut_constant_mm=start_constant,
        min_radius_mm=float(np.min(radius)),
        max_slip_risk_deg=max_slip,
        warning_flags=tuple(warnings),
    )


def _integrate_theta(
    mandrel: AxisymmetricProfileMandrel,
    z_mm: np.ndarray,
    alpha_rad: np.ndarray,
    *,
    start_theta_rad: float,
    direction: Direction,
) -> np.ndarray:
    radius = np.maximum(mandrel.radius_at(z_mm), 1e-9)
    meridian_scale = np.sqrt(1.0 + mandrel.dr_dz_at(z_mm) ** 2)
    dtheta_dz = np.tan(alpha_rad) * meridian_scale / radius
    if direction == "negative":
        dtheta_dz = -dtheta_dz
    dz = np.diff(z_mm)
    increments = 0.5 * (dtheta_dz[:-1] + dtheta_dz[1:]) * dz
    return start_theta_rad + np.concatenate(([0.0], np.cumsum(increments)))


def _z_samples(
    start_z_mm: float,
    end_z_mm: float,
    point_count: int,
    direction: Direction,
) -> np.ndarray:
    if point_count < 2:
        raise ValueError("point_count must be at least 2")
    if end_z_mm <= start_z_mm:
        raise ValueError("end_z_mm must be greater than start_z_mm")
    z_mm = np.linspace(start_z_mm, end_z_mm, point_count)
    return z_mm if direction == "positive" else z_mm[::-1]


def _validate_angle(value: float) -> None:
    if not np.isfinite(value) or not 0.0 < abs(value) < 90.0:
        raise ValueError("winding angle must be greater than 0 and less than 90 degrees")


def _geodesic_warnings(
    radius: np.ndarray,
    constant: float,
    turnaround_radius: float | None,
) -> tuple[str, ...]:
    warnings = []
    if np.any(radius <= constant):
        warnings.append("geodesic approaches invalid Clairaut radius")
    if turnaround_radius is not None and float(np.min(radius)) < turnaround_radius - 1e-6:
        warnings.append("turnaround radius violated")
    if float(np.min(radius)) < 1.0:
        warnings.append("near-pole singularity risk")
    return tuple(warnings)
