"""Cylinder-only helical winding path generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from filament_winder.core.geometry import CylinderMandrel

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]


@dataclass(frozen=True, slots=True)
class SurfacePath:
    """Generated centreline path on a mandrel surface."""

    z_mm: FloatArray
    theta_rad: FloatArray
    x_mm: FloatArray
    y_mm: FloatArray
    winding_angle_deg: float
    tow_width_mm: float
    pass_index: IntArray | None = None
    tow_eye_angle_deg: FloatArray | None = None

    def __post_init__(self) -> None:
        arrays = {
            "z_mm": np.asarray(self.z_mm, dtype=float),
            "theta_rad": np.asarray(self.theta_rad, dtype=float),
            "x_mm": np.asarray(self.x_mm, dtype=float),
            "y_mm": np.asarray(self.y_mm, dtype=float),
        }
        lengths = {values.shape for values in arrays.values()}
        if len(lengths) != 1:
            raise ValueError("surface path arrays must all have the same shape")
        for name, values in arrays.items():
            if values.ndim != 1:
                raise ValueError(f"{name} must be one-dimensional")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain only finite values")
            object.__setattr__(self, name, values)
        if len(self.z_mm) < 2:
            raise ValueError("a surface path needs at least two points")
        if self.tow_width_mm < 0 or not np.isfinite(self.tow_width_mm):
            raise ValueError("tow_width_mm must be a non-negative finite value")
        if self.pass_index is None:
            pass_index = np.zeros(self.point_count, dtype=int)
        else:
            pass_index = np.asarray(self.pass_index, dtype=int)
        if pass_index.shape != self.z_mm.shape:
            raise ValueError("pass_index must have the same shape as the path arrays")
        if np.any(pass_index < 0):
            raise ValueError("pass_index cannot contain negative values")
        object.__setattr__(self, "pass_index", pass_index)
        if self.tow_eye_angle_deg is not None:
            tow_eye_angle = np.asarray(self.tow_eye_angle_deg, dtype=float)
            if tow_eye_angle.shape != self.z_mm.shape:
                raise ValueError("tow_eye_angle_deg must have the same shape as the path arrays")
            if tow_eye_angle.ndim != 1:
                raise ValueError("tow_eye_angle_deg must be one-dimensional")
            if not np.all(np.isfinite(tow_eye_angle)):
                raise ValueError("tow_eye_angle_deg must contain only finite values")
            object.__setattr__(self, "tow_eye_angle_deg", tow_eye_angle)

    @property
    def point_count(self) -> int:
        return int(self.z_mm.size)

    @property
    def theta_deg(self) -> FloatArray:
        return np.rad2deg(self.theta_rad)

    @property
    def surface_radius_mm(self) -> FloatArray:
        return np.sqrt(self.x_mm**2 + self.y_mm**2)

    @property
    def points_mm(self) -> FloatArray:
        return np.column_stack((self.x_mm, self.y_mm, self.z_mm))

    @property
    def final_rotation_deg(self) -> float:
        return float(self.theta_deg[-1] - self.theta_deg[0])

    @property
    def final_turns(self) -> float:
        return self.final_rotation_deg / 360.0

    @property
    def pass_count(self) -> int:
        if self.pass_index is None:
            return 1
        return int(np.max(self.pass_index)) + 1


@dataclass(frozen=True, slots=True)
class HelicalPathConfig:
    winding_angle_deg: float
    tow_width_mm: float
    point_count: int
    start_z_mm: float = 0.0
    end_z_mm: float | None = None
    start_theta_rad: float = 0.0
    passes: int = 1
    phase_offset_deg: float | None = None
    alternate_direction: bool = True

    def resolved_end_z(self, mandrel: CylinderMandrel) -> float:
        return mandrel.length_mm if self.end_z_mm is None else self.end_z_mm

    def validate(self, mandrel: CylinderMandrel) -> None:
        if not np.isfinite(self.winding_angle_deg) or not 0.0 < self.winding_angle_deg < 90.0:
            raise ValueError("winding_angle_deg must be greater than 0 and less than 90")
        if not np.isfinite(self.tow_width_mm) or self.tow_width_mm < 0:
            raise ValueError("tow_width_mm must be a non-negative finite value")
        if self.point_count < 2:
            raise ValueError("point_count must be at least 2")
        if self.passes < 1:
            raise ValueError("passes must be at least 1")
        if self.phase_offset_deg is not None and not np.isfinite(self.phase_offset_deg):
            raise ValueError("phase_offset_deg must be finite when provided")
        if not np.isfinite(self.start_theta_rad):
            raise ValueError("start_theta_rad must be finite")
        end_z = self.resolved_end_z(mandrel)
        mandrel.validate_z_range([self.start_z_mm, end_z])
        if end_z <= self.start_z_mm:
            raise ValueError("end_z_mm must be greater than start_z_mm")


class HelicalPathGenerator:
    """Generate a helical centreline on a constant-radius cylinder."""

    def __init__(self, mandrel: CylinderMandrel, config: HelicalPathConfig) -> None:
        self.mandrel = mandrel
        self.config = config
        self.config.validate(mandrel)

    def generate(self) -> SurfacePath:
        end_z = self.config.resolved_end_z(self.mandrel)
        alpha_rad = np.deg2rad(self.config.winding_angle_deg)
        slope_theta_per_mm = np.tan(alpha_rad) / self.mandrel.radius_mm
        phase_offset_rad = np.deg2rad(
            360.0 / self.config.passes
            if self.config.phase_offset_deg is None
            else self.config.phase_offset_deg
        )
        z_chunks = []
        theta_chunks = []
        pass_chunks = []
        for pass_number in range(self.config.passes):
            pass_is_reverse = self.config.alternate_direction and pass_number % 2 == 1
            pass_start_z = end_z if pass_is_reverse else self.config.start_z_mm
            pass_end_z = self.config.start_z_mm if pass_is_reverse else end_z
            z_pass = np.linspace(pass_start_z, pass_end_z, self.config.point_count)
            travel_mm = np.abs(z_pass - pass_start_z)
            theta_pass = (
                self.config.start_theta_rad
                + pass_number * phase_offset_rad
                + slope_theta_per_mm * travel_mm
            )
            z_chunks.append(z_pass)
            theta_chunks.append(theta_pass)
            pass_chunks.append(np.full(self.config.point_count, pass_number, dtype=int))

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
class PatternClosureEstimate:
    """Simple cylinder pattern-closure estimate."""

    rotations_per_pass: float
    rotation_per_pass_deg: float
    nearest_integer_turns: int
    closure_error_deg: float
    passes: int
    phase_offset_deg: float
    band_spacing_mm: float
    approximate_gap_overlap_mm: float

    @property
    def closes_on_integer_turn(self) -> bool:
        return abs(self.closure_error_deg) < 1e-6


def estimate_cylinder_pattern_closure(
    mandrel: CylinderMandrel,
    config: HelicalPathConfig,
) -> PatternClosureEstimate:
    """Estimate cylinder turn closure and pass-to-pass band spacing."""

    config.validate(mandrel)
    end_z = config.resolved_end_z(mandrel)
    axial_distance_mm = end_z - config.start_z_mm
    rotation_per_pass_rad = np.tan(np.deg2rad(config.winding_angle_deg))
    rotation_per_pass_rad *= axial_distance_mm / mandrel.radius_mm
    rotations_per_pass = float(rotation_per_pass_rad / (2.0 * np.pi))
    nearest_integer_turns = int(round(rotations_per_pass))
    closure_error_deg = float((rotations_per_pass - nearest_integer_turns) * 360.0)
    phase_offset_deg = (
        360.0 / config.passes
        if config.phase_offset_deg is None
        else config.phase_offset_deg
    )
    band_spacing_mm = (
        2.0
        * np.pi
        * mandrel.radius_mm
        * np.cos(np.deg2rad(config.winding_angle_deg))
        / config.passes
    )
    return PatternClosureEstimate(
        rotations_per_pass=rotations_per_pass,
        rotation_per_pass_deg=float(np.rad2deg(rotation_per_pass_rad)),
        nearest_integer_turns=nearest_integer_turns,
        closure_error_deg=closure_error_deg,
        passes=config.passes,
        phase_offset_deg=phase_offset_deg,
        band_spacing_mm=float(band_spacing_mm),
        approximate_gap_overlap_mm=float(band_spacing_mm - config.tow_width_mm),
    )
