"""Dependency-free preview scene construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from filament_winder.core.coverage import CoverageSummary, cylinder_coverage_map
from filament_winder.core.geometry import AxisymmetricProfileMandrel, CylinderMandrel
from filament_winder.core.kinematics import MachineMotionTable, machine_path_from_surface_path
from filament_winder.core.path_planning import (
    HelicalPathConfig,
    HelicalPathGenerator,
    PatternClosureEstimate,
    PlannedWindingProgram,
    ProfileDomePathConfig,
    ProfileDomePathGenerator,
    SurfacePath,
    WindingLayerSpec,
    WindingSchedule,
    estimate_cylinder_pattern_closure,
    plan_winding_schedule,
)
from filament_winder.core.tow import TowBand, generate_cylinder_tow_band
from filament_winder.io import import_dxf_zr_profile

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]
ProfilePathMode = Literal["dome"]


@dataclass(frozen=True, slots=True)
class CylinderPreviewConfig:
    length_mm: float = 1000.0
    radius_mm: float = 100.0
    tow_width_mm: float = 6.0
    winding_angle_deg: float = 45.0
    points_per_pass: int = 500
    passes: int = 1
    radial_clearance_mm: float = 25.0
    phase_offset_deg: float | None = None
    alternate_direction: bool = True
    coverage_z_samples: int = 120
    coverage_theta_samples: int = 180
    mesh_theta_segments: int = 64
    mesh_z_segments: int = 24
    display_surface_offset_mm: float = 0.75


@dataclass(frozen=True, slots=True)
class CylinderPreviewScene:
    config: CylinderPreviewConfig
    mandrel: CylinderMandrel
    path: SurfacePath
    tow_band: TowBand
    motion_table: MachineMotionTable
    closure: PatternClosureEstimate
    coverage_summary: CoverageSummary
    cylinder_vertices_mm: FloatArray
    cylinder_faces: IntArray
    tow_vertices_mm: FloatArray
    tow_faces: IntArray
    display_cylinder_vertices_mm: FloatArray
    display_path_points_mm: FloatArray
    display_tow_vertices_mm: FloatArray


@dataclass(frozen=True, slots=True)
class ProfileDomePreviewConfig:
    profile_path: Path = Path("mandrels/profile.dxf")
    samples: int | None = None
    path_mode: ProfilePathMode = "dome"
    tow_width_mm: float = 3.0
    winding_angle_deg: float = 35.0
    points_per_span: int = 500
    min_radius_mm: float = 5.0
    turnaround_points: int = 25
    turnaround_angle_deg: float = 180.0
    circuits: int = 1
    turnaround_radius_mm: float | None = None
    radial_clearance_mm: float = 25.0
    mesh_theta_segments: int = 64
    mesh_z_segments: int = 48
    display_surface_offset_mm: float = 0.75


@dataclass(frozen=True, slots=True)
class ProfileDomePreviewScene:
    config: ProfileDomePreviewConfig
    profile: AxisymmetricProfileMandrel
    path: SurfacePath
    motion_table: MachineMotionTable
    geodesic_radius_mm: float
    turnaround_radius_mm: float
    safe_start_z_mm: float
    safe_end_z_mm: float
    profile_vertices_mm: FloatArray
    profile_faces: IntArray
    display_profile_vertices_mm: FloatArray
    display_path_points_mm: FloatArray


@dataclass(frozen=True, slots=True)
class _ProfilePathBuild:
    path: SurfacePath
    motion_table: MachineMotionTable
    geodesic_radius_mm: float
    turnaround_radius_mm: float
    safe_start_z_mm: float
    safe_end_z_mm: float


@dataclass(frozen=True, slots=True)
class PatternPlannerConfig:
    coverage_target: float = 1.0
    include_hoop_layer: bool = False
    balanced_pm_layers: bool = True
    max_angle_error_deg: float = 5.0

    def validate(self) -> None:
        if not np.isfinite(self.coverage_target) or self.coverage_target <= 0.0:
            raise ValueError("coverage_target must be a positive finite value")
        if not np.isfinite(self.max_angle_error_deg) or self.max_angle_error_deg < 0.0:
            raise ValueError("max_angle_error_deg must be a non-negative finite value")


@dataclass(frozen=True, slots=True)
class PatternPreviewScene:
    config: PatternPlannerConfig
    mandrel: CylinderMandrel | AxisymmetricProfileMandrel
    program: PlannedWindingProgram
    mandrel_vertices_mm: FloatArray
    mandrel_faces: IntArray
    display_mandrel_vertices_mm: FloatArray
    display_layer_path_points_mm: tuple[FloatArray, ...]
    display_transition_path_points_mm: tuple[FloatArray, ...]


def build_cylinder_preview_scene(config: CylinderPreviewConfig) -> CylinderPreviewScene:
    """Build all geometry needed by a live or exported cylinder preview."""

    mandrel = CylinderMandrel(length_mm=config.length_mm, radius_mm=config.radius_mm)
    path_config = HelicalPathConfig(
        winding_angle_deg=config.winding_angle_deg,
        tow_width_mm=config.tow_width_mm,
        point_count=config.points_per_pass,
        passes=config.passes,
        phase_offset_deg=config.phase_offset_deg,
        alternate_direction=config.alternate_direction,
    )
    path = HelicalPathGenerator(mandrel, path_config).generate()
    motion_table = machine_path_from_surface_path(
        path,
        radial_clearance_mm=config.radial_clearance_mm,
    )
    tow_band = generate_cylinder_tow_band(mandrel, path)
    coverage_summary = cylinder_coverage_map(
        mandrel,
        path,
        z_samples=config.coverage_z_samples,
        theta_samples=config.coverage_theta_samples,
    ).summary()
    cylinder_vertices, cylinder_faces = cylinder_mesh_arrays(
        mandrel,
        theta_segments=config.mesh_theta_segments,
        z_segments=config.mesh_z_segments,
    )
    return CylinderPreviewScene(
        config=config,
        mandrel=mandrel,
        path=path,
        tow_band=tow_band,
        motion_table=motion_table,
        closure=estimate_cylinder_pattern_closure(mandrel, path_config),
        coverage_summary=coverage_summary,
        cylinder_vertices_mm=cylinder_vertices,
        cylinder_faces=cylinder_faces,
        tow_vertices_mm=tow_band.vertices_mm,
        tow_faces=_quads_to_triangles(tow_band.quad_indices),
        display_cylinder_vertices_mm=orient_points_for_horizontal_view(
            cylinder_vertices,
            length_mm=mandrel.length_mm,
        ),
        display_path_points_mm=offset_display_surface(
            orient_points_for_horizontal_view(
                path.points_mm,
                length_mm=mandrel.length_mm,
            ),
            offset_mm=config.display_surface_offset_mm,
        ),
        display_tow_vertices_mm=offset_display_surface(
            orient_points_for_horizontal_view(
                tow_band.vertices_mm,
                length_mm=mandrel.length_mm,
            ),
            offset_mm=config.display_surface_offset_mm,
        ),
    )


def build_profile_dome_preview_scene(config: ProfileDomePreviewConfig) -> ProfileDomePreviewScene:
    """Build all geometry needed by the live profile-dome preview."""

    profile = import_dxf_zr_profile(config.profile_path, samples=config.samples)
    built_path = _build_profile_path(profile, config)
    profile_vertices, profile_faces = profile_mesh_arrays(
        profile,
        theta_segments=config.mesh_theta_segments,
        z_segments=config.mesh_z_segments,
    )
    center_z = 0.5 * (profile.start_z_mm + profile.end_z_mm)
    return ProfileDomePreviewScene(
        config=config,
        profile=profile,
        path=built_path.path,
        motion_table=built_path.motion_table,
        geodesic_radius_mm=built_path.geodesic_radius_mm,
        turnaround_radius_mm=built_path.turnaround_radius_mm,
        safe_start_z_mm=built_path.safe_start_z_mm,
        safe_end_z_mm=built_path.safe_end_z_mm,
        profile_vertices_mm=profile_vertices,
        profile_faces=profile_faces,
        display_profile_vertices_mm=orient_points_for_horizontal_view(
            profile_vertices,
            length_mm=profile.length_mm,
            center_z_mm=center_z,
        ),
        display_path_points_mm=offset_display_surface(
            orient_points_for_horizontal_view(
                built_path.path.points_mm,
                length_mm=profile.length_mm,
                center_z_mm=center_z,
            ),
            offset_mm=config.display_surface_offset_mm,
        ),
    )


def _build_profile_path(
    profile: AxisymmetricProfileMandrel,
    config: ProfileDomePreviewConfig,
) -> _ProfilePathBuild:
    if config.path_mode == "dome":
        dome_config = ProfileDomePathConfig(
            winding_angle_deg=config.winding_angle_deg,
            tow_width_mm=config.tow_width_mm,
            points_per_span=config.points_per_span,
            turnaround_points=config.turnaround_points,
            turnaround_angle_deg=config.turnaround_angle_deg,
            circuits=config.circuits,
            turnaround_radius_mm=config.turnaround_radius_mm,
        )
        dome_generator = ProfileDomePathGenerator(profile, dome_config)
        path = dome_generator.generate()
        return _profile_path_build(
            path,
            radial_clearance_mm=config.radial_clearance_mm,
            geodesic_radius_mm=dome_generator.clairaut_radius_mm,
            turnaround_radius_mm=dome_generator.turnaround_radius_mm,
            safe_start_z_mm=dome_generator.dome_start_z,
            safe_end_z_mm=dome_generator.dome_end_z,
        )

    raise ValueError(f"unsupported profile path mode: {config.path_mode}")


def _profile_path_build(
    path: SurfacePath,
    *,
    radial_clearance_mm: float,
    geodesic_radius_mm: float,
    turnaround_radius_mm: float,
    safe_start_z_mm: float,
    safe_end_z_mm: float,
) -> _ProfilePathBuild:
    motion_table = machine_path_from_surface_path(
        path,
        radial_clearance_mm=radial_clearance_mm,
    )
    return _ProfilePathBuild(
        path=path,
        motion_table=motion_table,
        geodesic_radius_mm=geodesic_radius_mm,
        turnaround_radius_mm=turnaround_radius_mm,
        safe_start_z_mm=safe_start_z_mm,
        safe_end_z_mm=safe_end_z_mm,
    )


def build_cylinder_pattern_preview_scene(
    config: CylinderPreviewConfig,
    pattern_config: PatternPlannerConfig,
    *,
    feedrate_mm_min: float = 500.0,
) -> PatternPreviewScene:
    """Build a full cylinder layer schedule preview."""

    pattern_config.validate()
    mandrel = CylinderMandrel(length_mm=config.length_mm, radius_mm=config.radius_mm)
    schedule = WindingSchedule(
        layers=_cylinder_pattern_layers(config, pattern_config),
        radial_clearance_mm=config.radial_clearance_mm,
        nominal_feedrate_mm_min=feedrate_mm_min,
    )
    program = plan_winding_schedule(mandrel, schedule)
    cylinder_vertices, cylinder_faces = cylinder_mesh_arrays(
        mandrel,
        theta_segments=config.mesh_theta_segments,
        z_segments=config.mesh_z_segments,
    )
    return PatternPreviewScene(
        config=pattern_config,
        mandrel=mandrel,
        program=program,
        mandrel_vertices_mm=cylinder_vertices,
        mandrel_faces=cylinder_faces,
        display_mandrel_vertices_mm=orient_points_for_horizontal_view(
            cylinder_vertices,
            length_mm=mandrel.length_mm,
        ),
        display_layer_path_points_mm=_display_layer_paths(
            program,
            length_mm=mandrel.length_mm,
            offset_mm=config.display_surface_offset_mm,
        ),
        display_transition_path_points_mm=_display_transition_paths(
            program,
            length_mm=mandrel.length_mm,
            offset_mm=config.display_surface_offset_mm,
        ),
    )


def build_profile_dome_pattern_preview_scene(
    config: ProfileDomePreviewConfig,
    pattern_config: PatternPlannerConfig,
    *,
    feedrate_mm_min: float = 500.0,
) -> PatternPreviewScene:
    """Build a full geodesic dome layer schedule preview from a DXF profile."""

    pattern_config.validate()
    profile = import_dxf_zr_profile(config.profile_path, samples=config.samples)
    schedule = WindingSchedule(
        layers=_profile_dome_pattern_layers(config, pattern_config),
        radial_clearance_mm=config.radial_clearance_mm,
        nominal_feedrate_mm_min=feedrate_mm_min,
    )
    program = plan_winding_schedule(profile, schedule)
    profile_vertices, profile_faces = profile_mesh_arrays(
        profile,
        theta_segments=config.mesh_theta_segments,
        z_segments=config.mesh_z_segments,
    )
    center_z = 0.5 * (profile.start_z_mm + profile.end_z_mm)
    return PatternPreviewScene(
        config=pattern_config,
        mandrel=profile,
        program=program,
        mandrel_vertices_mm=profile_vertices,
        mandrel_faces=profile_faces,
        display_mandrel_vertices_mm=orient_points_for_horizontal_view(
            profile_vertices,
            length_mm=profile.length_mm,
            center_z_mm=center_z,
        ),
        display_layer_path_points_mm=_display_layer_paths(
            program,
            length_mm=profile.length_mm,
            center_z_mm=center_z,
            offset_mm=config.display_surface_offset_mm,
        ),
        display_transition_path_points_mm=_display_transition_paths(
            program,
            length_mm=profile.length_mm,
            center_z_mm=center_z,
            offset_mm=config.display_surface_offset_mm,
        ),
    )


def cylinder_mesh_arrays(
    mandrel: CylinderMandrel,
    *,
    theta_segments: int,
    z_segments: int,
) -> tuple[FloatArray, IntArray]:
    if theta_segments < 8:
        raise ValueError("theta_segments must be at least 8")
    if z_segments < 2:
        raise ValueError("z_segments must be at least 2")

    z_mm = np.linspace(0.0, mandrel.length_mm, z_segments + 1)
    theta_rad = np.linspace(0.0, 2.0 * np.pi, theta_segments, endpoint=False)
    vertices = []
    for z_value in z_mm:
        for theta_value in theta_rad:
            vertices.append(mandrel.surface_points([z_value], [theta_value])[0])

    faces = []
    for z_index in range(z_segments):
        ring_start = z_index * theta_segments
        next_ring_start = (z_index + 1) * theta_segments
        for theta_index in range(theta_segments):
            a = ring_start + theta_index
            b = ring_start + ((theta_index + 1) % theta_segments)
            c = next_ring_start + ((theta_index + 1) % theta_segments)
            d = next_ring_start + theta_index
            faces.append((a, b, c))
            faces.append((a, c, d))

    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=int)


def profile_mesh_arrays(
    profile: AxisymmetricProfileMandrel,
    *,
    theta_segments: int,
    z_segments: int,
) -> tuple[FloatArray, IntArray]:
    if theta_segments < 8:
        raise ValueError("theta_segments must be at least 8")
    if z_segments < 2:
        raise ValueError("z_segments must be at least 2")

    z_mm = np.linspace(profile.start_z_mm, profile.end_z_mm, z_segments + 1)
    theta_rad = np.linspace(0.0, 2.0 * np.pi, theta_segments, endpoint=False)
    vertices = []
    for z_value in z_mm:
        for theta_value in theta_rad:
            vertices.append(profile.surface_points([z_value], [theta_value])[0])

    faces = []
    for z_index in range(z_segments):
        ring_start = z_index * theta_segments
        next_ring_start = (z_index + 1) * theta_segments
        for theta_index in range(theta_segments):
            a = ring_start + theta_index
            b = ring_start + ((theta_index + 1) % theta_segments)
            c = next_ring_start + ((theta_index + 1) % theta_segments)
            d = next_ring_start + theta_index
            faces.append((a, b, c))
            faces.append((a, c, d))

    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=int)


def _cylinder_pattern_layers(
    config: CylinderPreviewConfig,
    pattern_config: PatternPlannerConfig,
) -> tuple[WindingLayerSpec, ...]:
    layers: list[WindingLayerSpec] = []
    if pattern_config.include_hoop_layer:
        layers.append(
            WindingLayerSpec(
                name="hoop",
                winding_type="hoop",
                target_angle_deg=90.0,
                tow_width_mm=config.tow_width_mm,
                coverage_target=pattern_config.coverage_target,
                direction="hoop",
                point_count=max(12, config.points_per_pass),
            )
        )

    if pattern_config.balanced_pm_layers:
        layers.append(_helical_layer_spec(config, pattern_config, "+helical", "positive"))
        layers.append(_helical_layer_spec(config, pattern_config, "-helical", "negative"))
    else:
        layers.append(_helical_layer_spec(config, pattern_config, "helical", "positive"))
    return tuple(layers)


def _profile_dome_pattern_layers(
    config: ProfileDomePreviewConfig,
    pattern_config: PatternPlannerConfig,
) -> tuple[WindingLayerSpec, ...]:
    winding_type = _pattern_winding_type_for_profile_mode(config.path_mode)
    layer_prefix = _pattern_layer_prefix_for_profile_mode(config.path_mode)
    if pattern_config.balanced_pm_layers:
        return (
            _profile_layer_spec(
                config,
                pattern_config,
                f"+{layer_prefix}",
                "positive",
                winding_type,
            ),
            _profile_layer_spec(
                config,
                pattern_config,
                f"-{layer_prefix}",
                "negative",
                winding_type,
            ),
        )
    return (
        _profile_layer_spec(
            config,
            pattern_config,
            layer_prefix,
            "positive",
            winding_type,
        ),
    )


def _helical_layer_spec(
    config: CylinderPreviewConfig,
    pattern_config: PatternPlannerConfig,
    name: str,
    direction: Literal["positive", "negative"],
) -> WindingLayerSpec:
    return WindingLayerSpec(
        name=name,
        winding_type="helical",
        target_angle_deg=config.winding_angle_deg,
        tow_width_mm=config.tow_width_mm,
        coverage_target=pattern_config.coverage_target,
        direction=direction,
        point_count=config.points_per_pass,
        max_angle_error_deg=pattern_config.max_angle_error_deg,
    )


def _profile_layer_spec(
    config: ProfileDomePreviewConfig,
    pattern_config: PatternPlannerConfig,
    name: str,
    direction: Literal["positive", "negative"],
    winding_type: Literal["dome"],
) -> WindingLayerSpec:
    return WindingLayerSpec(
        name=name,
        winding_type=winding_type,
        target_angle_deg=config.winding_angle_deg,
        tow_width_mm=config.tow_width_mm,
        coverage_target=pattern_config.coverage_target,
        direction=direction,
        point_count=config.points_per_span,
        max_angle_error_deg=pattern_config.max_angle_error_deg,
        turnaround_radius_mm=config.turnaround_radius_mm,
        turnaround_points=config.turnaround_points,
        turnaround_angle_deg=config.turnaround_angle_deg,
    )


def _pattern_winding_type_for_profile_mode(
    path_mode: ProfilePathMode,
) -> Literal["dome"]:
    return "dome"


def _pattern_layer_prefix_for_profile_mode(path_mode: ProfilePathMode) -> str:
    return "dome"


def _display_layer_paths(
    program: PlannedWindingProgram,
    *,
    length_mm: float,
    offset_mm: float,
    center_z_mm: float | None = None,
) -> tuple[FloatArray, ...]:
    return tuple(
        offset_display_surface(
            orient_points_for_horizontal_view(
                layer.path.points_mm,
                length_mm=length_mm,
                center_z_mm=center_z_mm,
            ),
            offset_mm=offset_mm,
        )
        for layer in program.layers
    )


def _display_transition_paths(
    program: PlannedWindingProgram,
    *,
    length_mm: float,
    offset_mm: float,
    center_z_mm: float | None = None,
) -> tuple[FloatArray, ...]:
    winding_type = np.asarray(program.metadata.winding_type)
    transition_indices = np.flatnonzero(winding_type == "transition")
    if transition_indices.size == 0:
        return ()

    groups = np.split(
        transition_indices,
        np.flatnonzero(np.diff(transition_indices) > 1) + 1,
    )
    display_paths = []
    program_points = program.path.points_mm
    for group in groups:
        if group.size < 2:
            continue
        display_paths.append(
            offset_display_surface(
                orient_points_for_horizontal_view(
                    program_points[group],
                    length_mm=length_mm,
                    center_z_mm=center_z_mm,
                ),
                offset_mm=offset_mm,
            )
        )
    return tuple(display_paths)


def _quads_to_triangles(quads: IntArray) -> IntArray:
    triangles = []
    for quad in quads:
        a, b, c, d = (int(index) for index in quad)
        triangles.append((a, b, c))
        triangles.append((a, c, d))
    return np.asarray(triangles, dtype=int)


def orient_points_for_horizontal_view(
    points_mm: FloatArray,
    *,
    length_mm: float,
    center_z_mm: float | None = None,
) -> FloatArray:
    """Map engineering coordinates to viewport coordinates.

    Core geometry uses [radial X, radial Y, longitudinal Z]. The live viewport is
    easier to read when longitudinal Z is drawn horizontally, so display
    coordinates are [Z centered about zero, radial X, radial Y].
    """

    points = np.asarray(points_mm, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_mm must have shape (n, 3)")
    if not np.isfinite(length_mm) or length_mm <= 0.0:
        raise ValueError("length_mm must be positive and finite")
    z_center = length_mm / 2.0 if center_z_mm is None else center_z_mm
    if not np.isfinite(z_center):
        raise ValueError("center_z_mm must be finite")
    return np.column_stack(
        (
            points[:, 2] - z_center,
            points[:, 0],
            points[:, 1],
        )
    ).astype(float, copy=False)


def offset_display_surface(points_mm: FloatArray, *, offset_mm: float) -> FloatArray:
    """Offset display points away from the horizontal mandrel axis."""

    points = np.asarray(points_mm, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_mm must have shape (n, 3)")
    if not np.isfinite(offset_mm) or offset_mm < 0.0:
        raise ValueError("offset_mm must be non-negative and finite")
    output = points.copy()
    radial = output[:, 1:3]
    radius = np.linalg.norm(radial, axis=1)
    scale = np.ones_like(radius)
    nonzero = radius > 0.0
    scale[nonzero] = (radius[nonzero] + offset_mm) / radius[nonzero]
    output[:, 1:3] = radial * scale[:, np.newaxis]
    return output
