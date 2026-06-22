"""Profile-aware helical surface path generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from filament_winder.core.geometry import AxisymmetricProfileMandrel
from filament_winder.core.path_planning.helical import SurfacePath


@dataclass(frozen=True, slots=True)
class ProfileHelicalPathConfig:
    winding_angle_deg: float
    tow_width_mm: float
    point_count: int
    start_z_mm: float | None = None
    end_z_mm: float | None = None
    start_theta_rad: float = 0.0
    min_radius_mm: float = 1.0

    def resolved_start_z(self, mandrel: AxisymmetricProfileMandrel) -> float:
        return mandrel.start_z_mm if self.start_z_mm is None else self.start_z_mm

    def resolved_end_z(self, mandrel: AxisymmetricProfileMandrel) -> float:
        return mandrel.end_z_mm if self.end_z_mm is None else self.end_z_mm

    def validate(self, mandrel: AxisymmetricProfileMandrel) -> None:
        if not np.isfinite(self.winding_angle_deg) or not 0.0 < self.winding_angle_deg < 90.0:
            raise ValueError("winding_angle_deg must be greater than 0 and less than 90")
        if not np.isfinite(self.tow_width_mm) or self.tow_width_mm < 0.0:
            raise ValueError("tow_width_mm must be a non-negative finite value")
        if self.point_count < 2:
            raise ValueError("point_count must be at least 2")
        if not np.isfinite(self.min_radius_mm) or self.min_radius_mm <= 0.0:
            raise ValueError("min_radius_mm must be a positive finite value")
        start_z = self.resolved_start_z(mandrel)
        end_z = self.resolved_end_z(mandrel)
        mandrel.validate_z_range([start_z, end_z])
        if end_z <= start_z:
            raise ValueError("end_z_mm must be greater than start_z_mm")


class ProfileHelicalPathGenerator:
    """Generate a first-order helical path over an imported Z-R profile."""

    def __init__(
        self,
        mandrel: AxisymmetricProfileMandrel,
        config: ProfileHelicalPathConfig,
    ) -> None:
        self.mandrel = mandrel
        self.config = config
        self.config.validate(mandrel)

    def generate(self) -> SurfacePath:
        start_z = self.config.resolved_start_z(self.mandrel)
        end_z = self.config.resolved_end_z(self.mandrel)
        z_mm = np.linspace(start_z, end_z, self.config.point_count)
        radius_mm = self.mandrel.radius_at(z_mm)
        if np.any(radius_mm < self.config.min_radius_mm):
            raise ValueError(
                "profile path crosses a radius below min_radius_mm; add turnaround logic "
                "or avoid the singular region"
            )

        dr_dz = np.gradient(radius_mm, z_mm)
        meridian_scale = np.sqrt(1.0 + dr_dz**2)
        dtheta_dz = (
            np.tan(np.deg2rad(self.config.winding_angle_deg))
            * meridian_scale
            / radius_mm
        )
        theta_rad = self.config.start_theta_rad + _cumulative_trapezoid(dtheta_dz, z_mm)
        points = self.mandrel.surface_points(z_mm, theta_rad)
        return SurfacePath(
            z_mm=z_mm,
            theta_rad=theta_rad,
            x_mm=points[:, 0],
            y_mm=points[:, 1],
            winding_angle_deg=self.config.winding_angle_deg,
            tow_width_mm=self.config.tow_width_mm,
        )


@dataclass(frozen=True, slots=True)
class ProfileSafeZone:
    """Safe winding interval on a Z-R profile."""

    start_z_mm: float
    end_z_mm: float
    min_radius_mm: float
    start_radius_mm: float
    end_radius_mm: float

    @property
    def length_mm(self) -> float:
        return self.end_z_mm - self.start_z_mm


@dataclass(frozen=True, slots=True)
class ProfileTurnaroundPathConfig:
    winding_angle_deg: float
    tow_width_mm: float
    points_per_span: int
    turnaround_points: int = 25
    min_radius_mm: float = 5.0
    turnaround_angle_deg: float = 180.0
    circuits: int = 1
    start_theta_rad: float = 0.0

    def validate(self, mandrel: AxisymmetricProfileMandrel) -> None:
        if not np.isfinite(self.winding_angle_deg) or not 0.0 < self.winding_angle_deg < 90.0:
            raise ValueError("winding_angle_deg must be greater than 0 and less than 90")
        if not np.isfinite(self.tow_width_mm) or self.tow_width_mm < 0.0:
            raise ValueError("tow_width_mm must be a non-negative finite value")
        if self.points_per_span < 2:
            raise ValueError("points_per_span must be at least 2")
        if self.turnaround_points < 2:
            raise ValueError("turnaround_points must be at least 2")
        if not np.isfinite(self.min_radius_mm) or self.min_radius_mm <= 0.0:
            raise ValueError("min_radius_mm must be a positive finite value")
        if not np.isfinite(self.turnaround_angle_deg) or self.turnaround_angle_deg <= 0.0:
            raise ValueError("turnaround_angle_deg must be a positive finite value")
        if self.circuits < 1:
            raise ValueError("circuits must be at least 1")
        if not np.isfinite(self.start_theta_rad):
            raise ValueError("start_theta_rad must be finite")
        find_profile_safe_zone(mandrel, min_radius_mm=self.min_radius_mm)


class ProfileTurnaroundPathGenerator:
    """Generate profile paths that turn around before pole/opening singularities."""

    def __init__(
        self,
        mandrel: AxisymmetricProfileMandrel,
        config: ProfileTurnaroundPathConfig,
    ) -> None:
        self.mandrel = mandrel
        self.config = config
        self.config.validate(mandrel)
        self.safe_zone = find_profile_safe_zone(
            mandrel,
            min_radius_mm=config.min_radius_mm,
        )

    def generate(self) -> SurfacePath:
        z_chunks: list[np.ndarray] = []
        theta_chunks: list[np.ndarray] = []
        pass_chunks: list[np.ndarray] = []
        theta_start = self.config.start_theta_rad
        pass_number = 0

        for _circuit in range(self.config.circuits):
            forward_z = np.linspace(
                self.safe_zone.start_z_mm,
                self.safe_zone.end_z_mm,
                self.config.points_per_span,
            )
            forward_theta = theta_start + _theta_increment_along_profile(
                self.mandrel,
                forward_z,
                winding_angle_deg=self.config.winding_angle_deg,
                min_radius_mm=self.config.min_radius_mm,
            )
            _append_segment(
                z_chunks,
                theta_chunks,
                pass_chunks,
                forward_z,
                forward_theta,
                pass_number,
            )

            end_turn_theta = np.linspace(
                forward_theta[-1],
                forward_theta[-1] + np.deg2rad(self.config.turnaround_angle_deg),
                self.config.turnaround_points,
            )
            end_turn_z = np.full(end_turn_theta.shape, self.safe_zone.end_z_mm, dtype=float)
            _append_segment(
                z_chunks,
                theta_chunks,
                pass_chunks,
                end_turn_z,
                end_turn_theta,
                pass_number,
                drop_first=True,
            )

            pass_number += 1
            return_z = np.linspace(
                self.safe_zone.end_z_mm,
                self.safe_zone.start_z_mm,
                self.config.points_per_span,
            )
            return_theta = end_turn_theta[-1] + _theta_increment_along_profile(
                self.mandrel,
                return_z,
                winding_angle_deg=self.config.winding_angle_deg,
                min_radius_mm=self.config.min_radius_mm,
            )
            _append_segment(
                z_chunks,
                theta_chunks,
                pass_chunks,
                return_z,
                return_theta,
                pass_number,
                drop_first=True,
            )

            start_turn_theta = np.linspace(
                return_theta[-1],
                return_theta[-1] + np.deg2rad(self.config.turnaround_angle_deg),
                self.config.turnaround_points,
            )
            start_turn_z = np.full(start_turn_theta.shape, self.safe_zone.start_z_mm, dtype=float)
            _append_segment(
                z_chunks,
                theta_chunks,
                pass_chunks,
                start_turn_z,
                start_turn_theta,
                pass_number,
                drop_first=True,
            )
            pass_number += 1
            theta_start = start_turn_theta[-1]

        z_mm = np.concatenate(z_chunks)
        theta_rad = np.concatenate(theta_chunks)
        pass_index = np.concatenate(pass_chunks)
        points = self.mandrel.surface_points(z_mm, theta_rad)
        return SurfacePath(
            z_mm=z_mm,
            theta_rad=theta_rad,
            x_mm=points[:, 0],
            y_mm=points[:, 1],
            winding_angle_deg=self.config.winding_angle_deg,
            tow_width_mm=self.config.tow_width_mm,
            pass_index=pass_index,
        )


@dataclass(frozen=True, slots=True)
class ProfileDomePathConfig:
    """Dome-aware profile winding settings.

    The winding angle is held constant across the dome spans (same as the
    cylinder helix angle). Turnaround segments blend from the helix angle
    to 90° (hoop) at constant Z.
    """

    winding_angle_deg: float
    tow_width_mm: float
    points_per_span: int
    turnaround_points: int = 25
    turnaround_angle_deg: float = 180.0
    circuits: int = 1
    start_theta_rad: float = 0.0
    turnaround_radius_mm: float | None = None
    phase_offset_deg: float = 0.0

    def clairaut_radius_mm(self, mandrel: AxisymmetricProfileMandrel) -> float:
        return float(mandrel.max_radius_mm * np.sin(np.deg2rad(self.winding_angle_deg)))

    def resolved_turnaround_radius_mm(self, mandrel: AxisymmetricProfileMandrel) -> float:
        clairaut_radius = self.clairaut_radius_mm(mandrel)
        if self.turnaround_radius_mm is None:
            return clairaut_radius
        return self.turnaround_radius_mm

    def validate(self, mandrel: AxisymmetricProfileMandrel) -> None:
        if not np.isfinite(self.winding_angle_deg) or not 0.0 < self.winding_angle_deg < 90.0:
            raise ValueError("winding_angle_deg must be greater than 0 and less than 90")
        if not np.isfinite(self.tow_width_mm) or self.tow_width_mm < 0.0:
            raise ValueError("tow_width_mm must be a non-negative finite value")
        if self.points_per_span < 2:
            raise ValueError("points_per_span must be at least 2")
        if self.turnaround_points < 2:
            raise ValueError("turnaround_points must be at least 2")
        if not np.isfinite(self.turnaround_angle_deg) or self.turnaround_angle_deg <= 0.0:
            raise ValueError("turnaround_angle_deg must be a positive finite value")
        if self.circuits < 1:
            raise ValueError("circuits must be at least 1")
        if not np.isfinite(self.start_theta_rad):
            raise ValueError("start_theta_rad must be finite")
        clairaut_radius = self.clairaut_radius_mm(mandrel)
        turnaround_radius = self.resolved_turnaround_radius_mm(mandrel)
        if not np.isfinite(turnaround_radius) or turnaround_radius <= 0.0:
            raise ValueError("turnaround_radius_mm must be a positive finite value")
        if turnaround_radius < clairaut_radius - 1e-9:
            raise ValueError(
                "turnaround_radius_mm cannot be smaller than the geodesic turnaround radius"
            )
        if turnaround_radius >= mandrel.max_radius_mm:
            raise ValueError("turnaround_radius_mm must be smaller than the profile max radius")
        find_profile_safe_zone(mandrel, min_radius_mm=turnaround_radius)


class ProfileDomePathGenerator:
    """Generate a profile winding path with constant-angle dome spans and helix spans."""

    def __init__(
        self,
        mandrel: AxisymmetricProfileMandrel,
        config: ProfileDomePathConfig,
    ) -> None:
        self.mandrel = mandrel
        self.config = config
        self.config.validate(mandrel)
        self.clairaut_radius_mm = config.clairaut_radius_mm(mandrel)
        self.turnaround_radius_mm = config.resolved_turnaround_radius_mm(mandrel)
        self.safe_zone = find_profile_safe_zone(
            mandrel,
            min_radius_mm=self.turnaround_radius_mm,
        )

    def generate(self) -> SurfacePath:
        z_chunks: list[np.ndarray] = []
        theta_chunks: list[np.ndarray] = []
        angle_chunks: list[np.ndarray] = []
        pass_chunks: list[np.ndarray] = []
        theta_start = self.config.start_theta_rad
        pass_number = 0
        phase_offset_rad = np.deg2rad(self.config.phase_offset_deg)
        helix_angle_deg = self.config.winding_angle_deg

        for circuit_idx in range(self.config.circuits):
            circuit_phase_offset = circuit_idx * phase_offset_rad

            forward_z = np.linspace(
                self.safe_zone.start_z_mm,
                self.safe_zone.end_z_mm,
                self.config.points_per_span,
            )
            forward_theta = theta_start + circuit_phase_offset + _theta_increment_along_profile(
                self.mandrel,
                forward_z,
                winding_angle_deg=helix_angle_deg,
                min_radius_mm=self.turnaround_radius_mm,
            )
            forward_angles = np.full_like(forward_z, helix_angle_deg)
            _append_dome_segment(
                z_chunks,
                theta_chunks,
                angle_chunks,
                pass_chunks,
                forward_z,
                forward_theta,
                forward_angles,
                pass_number,
            )

            end_turn_theta = np.linspace(
                forward_theta[-1],
                forward_theta[-1] + np.deg2rad(self.config.turnaround_angle_deg),
                self.config.turnaround_points,
            )
            end_turn_z = np.full(end_turn_theta.shape, self.safe_zone.end_z_mm, dtype=float)
            end_turn_angles = np.linspace(helix_angle_deg, 90.0, self.config.turnaround_points)
            _append_dome_segment(
                z_chunks,
                theta_chunks,
                angle_chunks,
                pass_chunks,
                end_turn_z,
                end_turn_theta,
                end_turn_angles,
                pass_number,
                drop_first=True,
            )

            pass_number += 1
            return_z = np.linspace(
                self.safe_zone.end_z_mm,
                self.safe_zone.start_z_mm,
                self.config.points_per_span,
            )
            return_theta = end_turn_theta[-1] + _theta_increment_along_profile(
                self.mandrel,
                return_z,
                winding_angle_deg=helix_angle_deg,
                min_radius_mm=self.turnaround_radius_mm,
            )
            return_angles = np.full_like(return_z, helix_angle_deg)
            _append_dome_segment(
                z_chunks,
                theta_chunks,
                angle_chunks,
                pass_chunks,
                return_z,
                return_theta,
                return_angles,
                pass_number,
                drop_first=True,
            )

            start_turn_theta = np.linspace(
                return_theta[-1],
                return_theta[-1] + np.deg2rad(self.config.turnaround_angle_deg),
                self.config.turnaround_points,
            )
            start_turn_z = np.full(
                start_turn_theta.shape,
                self.safe_zone.start_z_mm,
                dtype=float,
            )
            start_turn_angles = np.linspace(helix_angle_deg, 90.0, self.config.turnaround_points)
            _append_dome_segment(
                z_chunks,
                theta_chunks,
                angle_chunks,
                pass_chunks,
                start_turn_z,
                start_turn_theta,
                start_turn_angles,
                pass_number,
                drop_first=True,
            )
            pass_number += 1
            theta_start = start_turn_theta[-1]

        z_mm = np.concatenate(z_chunks)
        theta_rad = np.concatenate(theta_chunks)
        tow_eye_angle_deg = np.concatenate(angle_chunks)
        pass_index = np.concatenate(pass_chunks)
        points = self.mandrel.surface_points(z_mm, theta_rad)
        return SurfacePath(
            z_mm=z_mm,
            theta_rad=theta_rad,
            x_mm=points[:, 0],
            y_mm=points[:, 1],
            winding_angle_deg=self.config.winding_angle_deg,
            tow_width_mm=self.config.tow_width_mm,
            pass_index=pass_index,
            tow_eye_angle_deg=tow_eye_angle_deg,
        )


def find_profile_safe_zone(
    mandrel: AxisymmetricProfileMandrel,
    *,
    min_radius_mm: float,
) -> ProfileSafeZone:
    """Find the longest profile interval with radius above the minimum."""

    if not np.isfinite(min_radius_mm) or min_radius_mm <= 0.0:
        raise ValueError("min_radius_mm must be a positive finite value")
    intervals: list[tuple[float, float]] = []
    for index in range(mandrel.z_mm.size - 1):
        z0 = float(mandrel.z_mm[index])
        z1 = float(mandrel.z_mm[index + 1])
        r0 = float(mandrel.r_mm[index])
        r1 = float(mandrel.r_mm[index + 1])
        start: float | None = None
        end: float | None = None
        if r0 >= min_radius_mm and r1 >= min_radius_mm:
            start, end = z0, z1
        elif r0 < min_radius_mm <= r1:
            start, end = _interpolate_radius_crossing(z0, r0, z1, r1, min_radius_mm), z1
        elif r0 >= min_radius_mm > r1:
            start, end = z0, _interpolate_radius_crossing(z0, r0, z1, r1, min_radius_mm)
        if start is not None and end is not None and end > start:
            intervals.append((start, end))

    merged = _merge_intervals(intervals)
    if not merged:
        raise ValueError("profile has no winding zone above min_radius_mm")
    start_z, end_z = max(merged, key=lambda interval: interval[1] - interval[0])
    if end_z <= start_z:
        raise ValueError("profile safe winding zone has zero length")
    return ProfileSafeZone(
        start_z_mm=start_z,
        end_z_mm=end_z,
        min_radius_mm=min_radius_mm,
        start_radius_mm=float(mandrel.radius_at([start_z])[0]),
        end_radius_mm=float(mandrel.radius_at([end_z])[0]),
    )


def _cumulative_trapezoid(values: np.ndarray, positions: np.ndarray) -> np.ndarray:
    deltas = np.diff(positions)
    areas = 0.5 * (values[:-1] + values[1:]) * deltas
    return np.concatenate(([0.0], np.cumsum(areas)))


def _theta_increment_along_profile(
    mandrel: AxisymmetricProfileMandrel,
    z_mm: np.ndarray,
    *,
    winding_angle_deg: float,
    min_radius_mm: float,
) -> np.ndarray:
    radius_mm = mandrel.radius_at(z_mm)
    if np.any(radius_mm < min_radius_mm - 1e-9):
        raise ValueError("profile segment crosses a radius below min_radius_mm")
    dr_dz = np.gradient(radius_mm, z_mm)
    meridian_scale = np.sqrt(1.0 + dr_dz**2)
    dtheta_dz_magnitude = np.tan(np.deg2rad(winding_angle_deg)) * meridian_scale / radius_mm
    deltas = np.abs(np.diff(z_mm))
    areas = 0.5 * (dtheta_dz_magnitude[:-1] + dtheta_dz_magnitude[1:]) * deltas
    return np.concatenate(([0.0], np.cumsum(areas)))


def _theta_increment_along_dome_geodesic(
    mandrel: AxisymmetricProfileMandrel,
    z_mm: np.ndarray,
    *,
    clairaut_radius_mm: float,
) -> np.ndarray:
    radius_mm = mandrel.radius_at(z_mm)
    if np.any(radius_mm < clairaut_radius_mm - 1e-9):
        raise ValueError("dome segment crosses below the geodesic turnaround radius")
    z_start = z_mm[:-1]
    z_end = z_mm[1:]
    z_mid = 0.5 * (z_start + z_end)
    radius_mid = mandrel.radius_at(z_mid)
    dr_dz = (radius_mm[1:] - radius_mm[:-1]) / (z_end - z_start)
    meridian_scale = np.sqrt(1.0 + dr_dz**2)
    radius_margin = radius_mid**2 - clairaut_radius_mm**2
    denominator = np.sqrt(np.maximum(radius_margin, 0.0))
    tan_alpha = np.divide(
        clairaut_radius_mm,
        denominator,
        out=np.full(radius_mid.shape, np.inf, dtype=float),
        where=denominator > 1e-9,
    )
    max_tan = np.tan(np.deg2rad(80.0))
    tan_alpha = np.minimum(tan_alpha, max_tan)
    dtheta_dz_magnitude = tan_alpha * meridian_scale / radius_mid
    dtheta_dz_magnitude = _fill_nonfinite_with_nearest(dtheta_dz_magnitude)
    areas = dtheta_dz_magnitude * np.abs(z_end - z_start)
    return np.concatenate(([0.0], np.cumsum(areas)))


def _dome_winding_angles_deg(
    mandrel: AxisymmetricProfileMandrel,
    z_mm: np.ndarray,
    *,
    clairaut_radius_mm: float,
) -> np.ndarray:
    radius_mm = mandrel.radius_at(z_mm)
    ratio = np.clip(clairaut_radius_mm / radius_mm, 0.0, 1.0)
    return np.rad2deg(np.arcsin(ratio))


def _append_segment(
    z_chunks: list[np.ndarray],
    theta_chunks: list[np.ndarray],
    pass_chunks: list[np.ndarray],
    z_mm: np.ndarray,
    theta_rad: np.ndarray,
    pass_number: int,
    *,
    drop_first: bool = False,
) -> None:
    if drop_first:
        z_mm = z_mm[1:]
        theta_rad = theta_rad[1:]
    z_chunks.append(z_mm)
    theta_chunks.append(theta_rad)
    pass_chunks.append(np.full(z_mm.shape, pass_number, dtype=int))


def _append_dome_segment(
    z_chunks: list[np.ndarray],
    theta_chunks: list[np.ndarray],
    angle_chunks: list[np.ndarray],
    pass_chunks: list[np.ndarray],
    z_mm: np.ndarray,
    theta_rad: np.ndarray,
    tow_eye_angle_deg: np.ndarray,
    pass_number: int,
    *,
    drop_first: bool = False,
) -> None:
    if drop_first:
        z_mm = z_mm[1:]
        theta_rad = theta_rad[1:]
        tow_eye_angle_deg = tow_eye_angle_deg[1:]
    z_chunks.append(z_mm)
    theta_chunks.append(theta_rad)
    angle_chunks.append(tow_eye_angle_deg)
    pass_chunks.append(np.full(z_mm.shape, pass_number, dtype=int))


def _interpolate_radius_crossing(
    z0: float,
    r0: float,
    z1: float,
    r1: float,
    target_radius: float,
) -> float:
    if np.isclose(r0, r1):
        return z0
    fraction = (target_radius - r0) / (r1 - r0)
    return z0 + fraction * (z1 - z0)


def _fill_nonfinite_with_nearest(values: np.ndarray) -> np.ndarray:
    output = values.copy()
    finite = np.isfinite(output)
    if not np.any(finite):
        raise ValueError("dome winding derivative has no finite samples")
    if np.all(finite):
        return output
    indices = np.arange(output.size)
    output[~finite] = np.interp(indices[~finite], indices[finite], output[finite])
    return output


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    sorted_intervals = sorted(intervals)
    merged = [sorted_intervals[0]]
    for start, end in sorted_intervals[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end + 1e-9:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged
