"""Layer-level winding pattern planning."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

from filament_winder.core.feedrate import FeedrateConfig, FeedSchedule, plan_feedrate
from filament_winder.core.geometry import AxisymmetricProfileMandrel, CylinderMandrel
from filament_winder.core.kinematics.four_axis import (
    MachineMotionTable,
    machine_path_from_surface_path,
)
from filament_winder.core.path_planning.geodesic import (
    ControlledAnglePathConfig,
    GeodesicPathConfig,
    generate_controlled_angle_path,
    generate_geodesic_path,
)
from filament_winder.core.path_planning.helical import (
    HelicalPathConfig,
    HelicalPathGenerator,
    SurfacePath,
)
from filament_winder.core.path_planning.optimization import (
    CylinderPatternOptimizationRequest,
    optimize_cylinder_pattern,
)
from filament_winder.core.path_planning.profile import (
    ProfileDomePathConfig,
    ProfileDomePathGenerator,
    ProfileTurnaroundPathConfig,
    ProfileTurnaroundPathGenerator,
    find_profile_safe_zone,
)
from filament_winder.core.validation import ValidationIssue, ValidationReport

if TYPE_CHECKING:
    from filament_winder.core.coverage import CoverageMap

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]

WindingType = Literal[
    "helical",
    "hoop",
    "local_reinforcement_band",
    "polar",
    "geodesic",
    "non_geodesic",
    "dome",
    "nosecone",
    "axisymmetric",
    "transition",
]
LayerDirection = Literal["positive", "negative", "alternating", "hoop", "polar"]
TransitionMode = Literal["continuous", "cut_restart"]
MandrelLike = CylinderMandrel | AxisymmetricProfileMandrel


@dataclass(frozen=True, slots=True)
class WindingLayerSpec:
    """High-level definition for one winding layer."""

    name: str
    winding_type: WindingType
    target_angle_deg: float
    tow_width_mm: float
    layer_thickness_mm: float = 0.0
    coverage_target: float = 1.0
    direction: LayerDirection = "positive"
    point_count: int = 500
    layer_id: str = ""
    enabled: bool = True
    number_of_passes: int | None = None
    start_z_mm: float | None = None
    end_z_mm: float | None = None
    feedrate_mm_min: float | None = None
    mandrel_clearance_mm: float | None = None
    colour: str = "#1e90ff"
    notes: str = ""
    phase_offset_deg: float | None = None
    max_pattern_candidates: int = 20
    max_angle_error_deg: float = 5.0
    start_offset_deg: float = 0.0
    turnaround_radius_mm: float | None = None
    turnaround_points: int = 25
    turnaround_angle_deg: float = 180.0
    transition_points: int = 20
    transition_mode: TransitionMode = "continuous"
    hoop_mode: str = "continuous_traverse"
    hoop_nominal_angle_deg: float = 89.0
    hoop_min_angle_offset_deg: float = 0.25
    allow_exact_pure_hoop: bool = False
    tow_state_during_traverse: str = "on"

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("layer name cannot be empty")
        if (
            self.winding_type not in {"hoop", "local_reinforcement_band"}
            and not 0.0 < abs(self.target_angle_deg) < 90.0
        ):
            raise ValueError("target_angle_deg must be between 0 and 90 for non-hoop layers")
        if (
            self.winding_type in {"hoop", "local_reinforcement_band"}
            and abs(self.target_angle_deg) > 90.0
        ):
            raise ValueError("hoop target_angle_deg cannot exceed 90")
        if not np.isfinite(self.tow_width_mm) or self.tow_width_mm <= 0.0:
            raise ValueError("tow_width_mm must be a positive finite value")
        if not np.isfinite(self.layer_thickness_mm) or self.layer_thickness_mm < 0.0:
            raise ValueError("layer_thickness_mm must be a non-negative finite value")
        if not np.isfinite(self.coverage_target) or self.coverage_target <= 0.0:
            raise ValueError("coverage_target must be a positive finite value")
        if self.point_count < 2:
            raise ValueError("point_count must be at least 2")
        if self.number_of_passes is not None and self.number_of_passes < 1:
            raise ValueError("number_of_passes must be at least 1 when provided")
        if self.start_z_mm is not None and not np.isfinite(self.start_z_mm):
            raise ValueError("start_z_mm must be finite when provided")
        if self.end_z_mm is not None and not np.isfinite(self.end_z_mm):
            raise ValueError("end_z_mm must be finite when provided")
        if (
            self.start_z_mm is not None
            and self.end_z_mm is not None
            and self.end_z_mm <= self.start_z_mm
        ):
            raise ValueError("end_z_mm must be greater than start_z_mm")
        if self.feedrate_mm_min is not None and (
            not np.isfinite(self.feedrate_mm_min) or self.feedrate_mm_min <= 0.0
        ):
            raise ValueError("feedrate_mm_min must be positive when provided")
        if self.mandrel_clearance_mm is not None and (
            not np.isfinite(self.mandrel_clearance_mm) or self.mandrel_clearance_mm < 0.0
        ):
            raise ValueError("mandrel_clearance_mm must be non-negative when provided")
        if self.max_pattern_candidates < 1:
            raise ValueError("max_pattern_candidates must be at least 1")
        if not np.isfinite(self.max_angle_error_deg) or self.max_angle_error_deg < 0.0:
            raise ValueError("max_angle_error_deg must be a non-negative finite value")
        if not np.isfinite(self.start_offset_deg):
            raise ValueError("start_offset_deg must be finite")
        if self.phase_offset_deg is not None and not np.isfinite(self.phase_offset_deg):
            raise ValueError("phase_offset_deg must be finite when provided")
        if self.turnaround_points < 2:
            raise ValueError("turnaround_points must be at least 2")
        if not np.isfinite(self.turnaround_angle_deg) or self.turnaround_angle_deg <= 0.0:
            raise ValueError("turnaround_angle_deg must be a positive finite value")
        if self.transition_points < 2:
            raise ValueError("transition_points must be at least 2")


@dataclass(frozen=True, slots=True)
class WindingSchedule:
    layers: tuple[WindingLayerSpec, ...]
    radial_clearance_mm: float = 25.0
    nominal_feedrate_mm_min: float = 500.0
    minimum_feedrate_mm_min: float | None = None

    def validate(self) -> None:
        if not self.layers:
            raise ValueError("schedule must contain at least one layer")
        if not np.isfinite(self.radial_clearance_mm) or self.radial_clearance_mm < 0.0:
            raise ValueError("radial_clearance_mm must be a non-negative finite value")
        if not np.isfinite(self.nominal_feedrate_mm_min) or self.nominal_feedrate_mm_min <= 0.0:
            raise ValueError("nominal_feedrate_mm_min must be a positive finite value")
        for layer in self.layers:
            if not layer.enabled:
                continue
            layer.validate()


@dataclass(frozen=True, slots=True)
class WindingPatternReport:
    layer_id: str
    layer_name: str
    winding_type: WindingType
    target_angle_deg: float
    actual_angle_deg: float
    angle_error_deg: float
    circuits: int
    starts: int
    angular_shift_deg: float
    tow_spacing_mm: float
    coverage_percent: float
    gap_mm: float
    overlap_mm: float
    layer_completion_z_mm: float
    pattern_repeat_length_mm: float
    closes: bool
    acceptable: bool
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class WindingPointMetadata:
    layer_id: tuple[str, ...]
    layer_index: IntArray
    circuit_index: IntArray
    pass_index: IntArray
    local_radius_mm: FloatArray
    local_winding_angle_deg: FloatArray
    layer_name: tuple[str, ...]
    winding_type: tuple[str, ...]
    motion_type: tuple[str, ...]
    warning_flags: tuple[str, ...]

    def __post_init__(self) -> None:
        arrays = {
            "layer_index": np.asarray(self.layer_index, dtype=int),
            "circuit_index": np.asarray(self.circuit_index, dtype=int),
            "pass_index": np.asarray(self.pass_index, dtype=int),
            "local_radius_mm": np.asarray(self.local_radius_mm, dtype=float),
            "local_winding_angle_deg": np.asarray(self.local_winding_angle_deg, dtype=float),
        }
        shapes = {values.shape for values in arrays.values()}
        if len(shapes) != 1:
            raise ValueError("metadata arrays must all have the same shape")
        point_count = int(next(iter(arrays.values())).size)
        label_lengths = {
            len(self.layer_id),
            len(self.layer_name),
            len(self.winding_type),
            len(self.motion_type),
            len(self.warning_flags),
        }
        if label_lengths != {point_count}:
            raise ValueError("metadata labels must match metadata array length")
        for name, values in arrays.items():
            if values.ndim != 1:
                raise ValueError(f"{name} must be one-dimensional")
            if name.startswith("local") and not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain only finite values")
            object.__setattr__(self, name, values)

    @property
    def point_count(self) -> int:
        return int(self.layer_index.size)


@dataclass(frozen=True, slots=True)
class PlannedLayer:
    spec: WindingLayerSpec
    path: SurfacePath
    motion_table: MachineMotionTable
    feed_schedule: FeedSchedule
    metadata: WindingPointMetadata
    report: WindingPatternReport
    effective_radius_mm: float
    accumulated_thickness_before_mm: float


@dataclass(frozen=True, slots=True)
class PlannedWindingProgram:
    layers: tuple[PlannedLayer, ...]
    path: SurfacePath
    motion_table: MachineMotionTable
    feed_schedule: FeedSchedule
    metadata: WindingPointMetadata
    reports: tuple[WindingPatternReport, ...]

    @property
    def point_count(self) -> int:
        return self.path.point_count


def plan_winding_schedule(mandrel: MandrelLike, schedule: WindingSchedule) -> PlannedWindingProgram:
    """Plan a complete multi-layer winding schedule on the existing mandrel geometry."""

    schedule.validate()
    layers: list[PlannedLayer] = []
    planned_chunks: list[PlannedLayer] = []
    path_chunks: list[SurfacePath] = []
    metadata_chunks: list[WindingPointMetadata] = []
    theta_offset_rad = 0.0
    previous_path: SurfacePath | None = None
    accumulated_thickness_mm = 0.0

    for source_layer_index, spec in enumerate(schedule.layers):
        if not spec.enabled:
            continue
        layer_index = len(layers)
        effective_spec = _resolve_alternating_direction(
            _with_generated_layer_id(spec, source_layer_index),
            layer_index,
        )
        layer_mandrel = _mandrel_with_radius_offset(mandrel, accumulated_thickness_mm)
        layer = _plan_single_layer(
            layer_mandrel,
            effective_spec,
            layer_index=layer_index,
            theta_offset_rad=theta_offset_rad,
            schedule=schedule,
            accumulated_thickness_before_mm=accumulated_thickness_mm,
        )
        if previous_path is not None and effective_spec.transition_mode == "continuous":
            transition = _transition_between_paths(
                layer_mandrel,
                previous_path,
                layer.path,
                layer_index=layer_index,
                point_count=effective_spec.transition_points,
                radial_clearance_mm=_layer_clearance_mm(effective_spec, schedule),
                nominal_feedrate_mm_min=_layer_feedrate_mm_min(effective_spec, schedule),
                minimum_feedrate_mm_min=schedule.minimum_feedrate_mm_min,
                effective_radius_mm=_max_radius(layer_mandrel),
                accumulated_thickness_before_mm=accumulated_thickness_mm,
            )
            planned_chunks.append(transition)
            path_chunks.append(transition.path)
            metadata_chunks.append(transition.metadata)
        layers.append(layer)
        planned_chunks.append(layer)
        path_chunks.append(layer.path)
        metadata_chunks.append(layer.metadata)
        previous_path = layer.path
        theta_offset_rad = float(layer.path.theta_rad[-1]) + math.radians(
            effective_spec.tow_width_mm / max(_max_radius(layer_mandrel), 1e-9) * 180.0 / math.pi
        )
        accumulated_thickness_mm += effective_spec.layer_thickness_mm

    if not layers:
        raise ValueError("schedule must contain at least one enabled layer")

    program_path = _concatenate_surface_paths(path_chunks)
    metadata = _concatenate_metadata(metadata_chunks)
    motion_table = _concatenate_motion_tables([chunk.motion_table for chunk in planned_chunks])
    feed_schedule = _concatenate_feed_schedules([chunk.feed_schedule for chunk in planned_chunks])
    return PlannedWindingProgram(
        layers=tuple(layers),
        path=program_path,
        motion_table=motion_table,
        feed_schedule=feed_schedule,
        metadata=metadata,
        reports=tuple(layer.report for layer in layers),
    )


def validate_winding_program(
    program: PlannedWindingProgram,
    *,
    max_angle_error_deg: float = 5.0,
    max_gap_mm: float = 0.25,
    max_overlap_mm: float | None = None,
    max_slip_risk: float = 1.0,
) -> ValidationReport:
    """Validate closure, coverage, slip risk, and machine continuity for a planned program."""

    issues: list[ValidationIssue] = []
    for layer_index, report in enumerate(program.reports):
        if not report.closes:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="PATTERN_OPEN",
                    message=f"Layer '{report.layer_name}' does not close on the selected pattern",
                )
            )
        if report.angle_error_deg > max_angle_error_deg:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    code="ANGLE_ERROR",
                    message=(
                        f"Layer '{report.layer_name}' angle error is "
                        f"{report.angle_error_deg:.3f} deg"
                    ),
                )
            )
        if report.gap_mm > max_gap_mm:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    code="COVERAGE_GAP",
                    message=f"Layer '{report.layer_name}' has {report.gap_mm:.3f} mm tow gap",
                )
            )
        if max_overlap_mm is not None and report.overlap_mm > max_overlap_mm:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    code="COVERAGE_OVERLAP",
                    message=(
                        f"Layer '{report.layer_name}' has {report.overlap_mm:.3f} mm tow overlap"
                    ),
                )
            )
        if not report.acceptable:
            warning_text = ", ".join(report.warnings)
            issues.append(
                ValidationIssue(
                    severity="warning",
                    code="PATTERN_REVIEW",
                    message=f"Layer '{report.layer_name}' needs review: {warning_text}",
                )
            )
        layer_mask = program.metadata.layer_index == layer_index
        if np.any(layer_mask):
            layer_slip = program.feed_schedule.slip_risk[layer_mask]
            if float(np.max(layer_slip)) > max_slip_risk:
                first_index = int(np.flatnonzero(layer_mask)[0])
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        code="SLIP_RISK",
                        message=(
                            f"Layer '{report.layer_name}' exceeds slip risk "
                            f"{max_slip_risk:.3f}"
                        ),
                        point_index=first_index,
                    )
                )

    axis_steps = np.column_stack(
        (
            np.diff(program.motion_table.a_deg),
            np.diff(program.motion_table.x_mm),
            np.diff(program.motion_table.z_mm),
            np.diff(program.motion_table.b_deg),
        )
    )
    if np.any(~np.isfinite(axis_steps)):
        issues.append(
            ValidationIssue(
                severity="error",
                code="MOTION_DISCONTINUITY",
                message="Machine motion contains non-finite axis deltas",
            )
        )
    return ValidationReport(tuple(issues))


def axisymmetric_surface_coverage_map(
    mandrel: MandrelLike,
    surface_path: SurfacePath,
    *,
    z_samples: int = 120,
    theta_samples: int = 180,
) -> CoverageMap:
    """Approximate coverage on an axisymmetric surface using local radius at each Z."""

    from filament_winder.core.coverage import CoverageMap

    if z_samples < 2:
        raise ValueError("z_samples must be at least 2")
    if theta_samples < 4:
        raise ValueError("theta_samples must be at least 4")
    start_z, end_z = _z_bounds(mandrel)
    z_mm = np.linspace(start_z, end_z, z_samples)
    theta_rad = np.linspace(0.0, 2.0 * np.pi, theta_samples, endpoint=False)
    z_grid, theta_grid = np.meshgrid(z_mm, theta_rad, indexing="ij")
    radius_grid = mandrel.radius_at(z_grid)
    meridional_mm = mandrel.meridional_arc_length_at(z_mm)
    meridional_grid = meridional_mm[:, np.newaxis]
    coverage_count = np.zeros(z_grid.shape, dtype=int)
    pass_index = np.asarray(surface_path.pass_index, dtype=int)

    for start_index, end_index in _contiguous_pass_spans(pass_index):
        pass_z = surface_path.z_mm[start_index:end_index]
        pass_theta = surface_path.theta_rad[start_index:end_index]
        if pass_z.size < 2:
            continue
        if np.isclose(pass_z[-1], pass_z[0]):
            pass_m = float(mandrel.meridional_arc_length_at(np.asarray([pass_z[0]]))[0])
            coverage_count += (
                np.abs(meridional_grid - pass_m) <= surface_path.tow_width_mm / 2.0
            ).astype(int)
            continue
        pass_theta_unwrapped = np.unwrap(pass_theta)
        pass_m = mandrel.meridional_arc_length_at(pass_z)
        order = np.argsort(pass_m)
        unique_m, unique_indices = np.unique(pass_m[order], return_index=True)
        if unique_m.size < 2:
            continue
        unique_theta = pass_theta_unwrapped[order][unique_indices]
        theta_expected = np.interp(meridional_mm, unique_m, unique_theta)
        theta_expected_grid = theta_expected[:, np.newaxis]
        dtheta_dm = np.gradient(unique_theta, unique_m, edge_order=1)
        local_theta_slope = np.interp(meridional_mm, unique_m, dtheta_dm)[:, np.newaxis]
        local_surface_slope = radius_grid * local_theta_slope
        circumference_grid = 2.0 * np.pi * radius_grid
        actual_s_mm = radius_grid * theta_grid
        expected_s_mm = radius_grid * theta_expected_grid
        wrapped_offset_mm = _wrap_periodic(actual_s_mm - expected_s_mm, circumference_grid)
        perpendicular_distance_mm = np.abs(wrapped_offset_mm) / np.sqrt(
            local_surface_slope**2 + 1.0
        )
        coverage_count += (
            perpendicular_distance_mm <= surface_path.tow_width_mm / 2.0
        ).astype(int)

    return CoverageMap(
        z_mm=z_mm,
        theta_rad=theta_rad,
        coverage_count=coverage_count,
        tow_width_mm=surface_path.tow_width_mm,
        winding_angle_deg=surface_path.winding_angle_deg,
    )


def _plan_single_layer(
    mandrel: MandrelLike,
    spec: WindingLayerSpec,
    *,
    layer_index: int,
    theta_offset_rad: float,
    schedule: WindingSchedule,
    accumulated_thickness_before_mm: float,
) -> PlannedLayer:
    if spec.winding_type == "hoop":
        if isinstance(mandrel, CylinderMandrel):
            path, report, motion_type = _plan_cylinder_hoop_layer(mandrel, spec, theta_offset_rad)
        elif isinstance(mandrel, AxisymmetricProfileMandrel):
            path, report, motion_type = _plan_axisymmetric_hoop_layer(
                mandrel,
                spec,
                theta_offset_rad,
            )
        else:
            raise ValueError("unsupported mandrel for hoop planning")
    elif spec.winding_type == "local_reinforcement_band":
        if isinstance(mandrel, CylinderMandrel):
            path, report, motion_type = _plan_cylinder_local_reinforcement_layer(
                mandrel,
                spec,
                theta_offset_rad,
            )
        elif isinstance(mandrel, AxisymmetricProfileMandrel):
            path, report, motion_type = _plan_axisymmetric_local_reinforcement_layer(
                mandrel,
                spec,
                theta_offset_rad,
            )
        else:
            raise ValueError("unsupported mandrel for local reinforcement planning")
    elif spec.winding_type in {"helical", "transition"}:
        if not isinstance(mandrel, CylinderMandrel):
            raise ValueError(f"{spec.winding_type} planning currently requires a cylinder mandrel")
        path, report, motion_type = _plan_cylinder_helical_layer(mandrel, spec, theta_offset_rad)
    elif spec.winding_type == "polar":
        if isinstance(mandrel, CylinderMandrel):
            path, report, motion_type = _plan_cylinder_helical_layer(
                mandrel,
                spec,
                theta_offset_rad,
            )
        elif isinstance(mandrel, AxisymmetricProfileMandrel):
            path, report, motion_type = _plan_axisymmetric_geodesic_layer(
                mandrel,
                spec,
                theta_offset_rad,
            )
        else:
            raise ValueError("unsupported mandrel for polar planning")
    elif spec.winding_type == "dome":
        if not isinstance(mandrel, AxisymmetricProfileMandrel):
            raise ValueError(f"{spec.winding_type} planning requires an axisymmetric profile")
        path, report = _plan_profile_dome_layer(mandrel, spec, theta_offset_rad)
        motion_type = tuple("wind" for _ in range(path.point_count))
    elif spec.winding_type == "geodesic":
        if not isinstance(mandrel, AxisymmetricProfileMandrel):
            raise ValueError("geodesic planning requires an axisymmetric profile")
        path, report, motion_type = _plan_axisymmetric_geodesic_layer(
            mandrel,
            spec,
            theta_offset_rad,
        )
    elif spec.winding_type == "non_geodesic":
        if not isinstance(mandrel, AxisymmetricProfileMandrel):
            raise ValueError("non_geodesic planning requires an axisymmetric profile")
        path, report, motion_type = _plan_axisymmetric_non_geodesic_layer(
            mandrel,
            spec,
            theta_offset_rad,
        )
    elif spec.winding_type in {"nosecone", "axisymmetric"}:
        if not isinstance(mandrel, AxisymmetricProfileMandrel):
            raise ValueError(f"{spec.winding_type} planning requires an axisymmetric profile")
        path, report = _plan_profile_turnaround_layer(mandrel, spec, theta_offset_rad)
        motion_type = tuple("wind" for _ in range(path.point_count))
    else:
        raise ValueError(f"unsupported winding type: {spec.winding_type}")

    motion_table = machine_path_from_surface_path(
        path,
        radial_clearance_mm=_layer_clearance_mm(spec, schedule),
    )
    feed_schedule = plan_feedrate(
        path,
        FeedrateConfig(
            nominal_feedrate_mm_min=_layer_feedrate_mm_min(spec, schedule),
            minimum_feedrate_mm_min=schedule.minimum_feedrate_mm_min,
        ),
    )
    motion_table = _smooth_motion_b_axis(motion_table, path)
    feed_schedule = _slow_for_b_axis_changes(feed_schedule, motion_table, path)
    metadata = _metadata_for_path(
        mandrel,
        path,
        layer_id=_resolved_layer_id(spec, layer_index),
        layer_index=layer_index,
        layer_name=spec.name,
        winding_type=spec.winding_type,
        motion_type=motion_type,
        warning_flags=report.warnings,
    )
    return PlannedLayer(
        spec=spec,
        path=path,
        motion_table=motion_table,
        feed_schedule=feed_schedule,
        metadata=metadata,
        report=report,
        effective_radius_mm=_max_radius(mandrel),
        accumulated_thickness_before_mm=accumulated_thickness_before_mm,
    )


def _plan_cylinder_helical_layer(
    mandrel: CylinderMandrel,
    spec: WindingLayerSpec,
    theta_offset_rad: float,
) -> tuple[SurfacePath, WindingPatternReport, tuple[str, ...]]:
    circumference = 2.0 * math.pi * mandrel.radius_mm
    start_z = 0.0 if spec.start_z_mm is None else spec.start_z_mm
    end_z = mandrel.length_mm if spec.end_z_mm is None else spec.end_z_mm
    axial_length = end_z - start_z
    if spec.number_of_passes is not None:
        actual_angle = abs(spec.target_angle_deg)
        passes = spec.number_of_passes
        if passes % 2:
            passes += 1
        lanes_per_direction = max(1, passes // 2)
        phase_offset = 360.0 / passes if spec.phase_offset_deg is None else spec.phase_offset_deg
        perpendicular_circumference = circumference * math.cos(math.radians(actual_angle))
        tow_spacing = perpendicular_circumference / lanes_per_direction
        coverage_percent = spec.tow_width_mm / tow_spacing * 100.0
        gap_mm = max(tow_spacing - spec.tow_width_mm, 0.0)
        overlap_mm = max(spec.tow_width_mm - tow_spacing, 0.0)
        closes = True
    else:
        min_angle = max(1.0, abs(spec.target_angle_deg) - spec.max_angle_error_deg)
        max_angle = min(89.0, abs(spec.target_angle_deg) + spec.max_angle_error_deg)
        result = optimize_cylinder_pattern(
            CylinderPatternOptimizationRequest(
                length_mm=axial_length,
                radius_mm=mandrel.radius_mm,
                tow_width_mm=spec.tow_width_mm,
                point_count=spec.point_count,
                target_coverage_fraction=spec.coverage_target,
                min_angle_deg=min_angle,
                max_angle_deg=max_angle,
                preferred_angle_deg=abs(spec.target_angle_deg),
                max_results=spec.max_pattern_candidates,
                max_passes=max(20, math.ceil(circumference / spec.tow_width_mm * 3.0)),
            )
        )
        if result.candidates:
            candidate = result.best
            actual_angle = candidate.winding_angle_deg
            lanes_per_direction = candidate.passes
            passes = lanes_per_direction * 2
            phase_offset = 360.0 / passes
            tow_spacing = candidate.band_spacing_mm
            coverage_percent = candidate.estimated_coverage_percent
            gap_mm = max(candidate.estimated_gap_overlap_mm, 0.0)
            overlap_mm = max(-candidate.estimated_gap_overlap_mm, 0.0)
            closes = True
            minimum_coverage_passes = max(
                1,
                math.ceil(
                    circumference
                    * math.cos(math.radians(actual_angle))
                    * spec.coverage_target
                    / spec.tow_width_mm
                ),
            )
            if lanes_per_direction < minimum_coverage_passes:
                lanes_per_direction = minimum_coverage_passes
                passes = lanes_per_direction * 2
                phase_offset = 360.0 / passes
                tow_spacing = (
                    circumference * math.cos(math.radians(actual_angle))
                    / lanes_per_direction
                )
                coverage_percent = spec.tow_width_mm / tow_spacing * 100.0
                gap_mm = max(tow_spacing - spec.tow_width_mm, 0.0)
                overlap_mm = max(spec.tow_width_mm - tow_spacing, 0.0)
        else:
            actual_angle = abs(spec.target_angle_deg)
            perpendicular_circumference = circumference * math.cos(math.radians(actual_angle))
            lanes_per_direction = max(
                1,
                math.ceil(perpendicular_circumference * spec.coverage_target / spec.tow_width_mm),
            )
            passes = lanes_per_direction * 2
            phase_offset = 360.0 / passes
            tow_spacing = perpendicular_circumference / lanes_per_direction
            coverage_percent = spec.tow_width_mm / tow_spacing * 100.0
            gap_mm = max(tow_spacing - spec.tow_width_mm, 0.0)
            overlap_mm = max(spec.tow_width_mm - tow_spacing, 0.0)
            closes = False

    config = HelicalPathConfig(
        winding_angle_deg=actual_angle,
        tow_width_mm=spec.tow_width_mm,
        point_count=spec.point_count,
        start_z_mm=start_z,
        end_z_mm=end_z,
        start_theta_rad=theta_offset_rad + math.radians(spec.start_offset_deg),
        passes=passes,
        phase_offset_deg=phase_offset,
        alternate_direction=True,
    )
    path = HelicalPathGenerator(mandrel, config).generate()
    path = _apply_direction(mandrel, path, spec.direction, actual_angle)
    path, motion_type = _insert_pass_transitions(
        mandrel,
        path,
        point_count=spec.transition_points,
    )
    warnings = _pattern_warnings(
        target_angle=abs(spec.target_angle_deg),
        actual_angle=actual_angle,
        gap_mm=gap_mm,
        overlap_mm=overlap_mm,
        max_angle_error=spec.max_angle_error_deg,
    )
    report = WindingPatternReport(
        layer_id=_resolved_layer_id(spec, 0),
        layer_name=spec.name,
        winding_type=spec.winding_type,
        target_angle_deg=spec.target_angle_deg,
        actual_angle_deg=_signed_angle(spec.direction, actual_angle),
        angle_error_deg=abs(abs(spec.target_angle_deg) - actual_angle),
        circuits=passes,
        starts=passes,
        angular_shift_deg=phase_offset,
        tow_spacing_mm=tow_spacing,
        coverage_percent=coverage_percent,
        gap_mm=gap_mm,
        overlap_mm=overlap_mm,
        layer_completion_z_mm=float(path.z_mm[-1]),
        pattern_repeat_length_mm=axial_length,
        closes=closes,
        acceptable=closes and not warnings,
        warnings=warnings,
    )
    return path, report, motion_type


def _plan_cylinder_hoop_layer(
    mandrel: CylinderMandrel,
    spec: WindingLayerSpec,
    theta_offset_rad: float,
) -> tuple[SurfacePath, WindingPatternReport, tuple[str, ...]]:
    return _plan_continuous_hoop_layer(mandrel, spec, theta_offset_rad)


def _plan_continuous_hoop_layer(
    mandrel: MandrelLike,
    spec: WindingLayerSpec,
    theta_offset_rad: float,
) -> tuple[SurfacePath, WindingPatternReport, tuple[str, ...]]:
    start_z, end_z = _z_bounds(mandrel)
    if spec.start_z_mm is not None:
        start_z = spec.start_z_mm
    if spec.end_z_mm is not None:
        end_z = spec.end_z_mm
    axial_length = end_z - start_z
    if axial_length <= 0.0:
        raise ValueError("continuous hoop layer requires a positive axial travel")

    target_pitch = max(spec.tow_width_mm / max(spec.coverage_target, 1e-9), 1e-9)
    radius = max(_max_radius(mandrel), 1e-9)
    nominal_angle = min(
        90.0 - max(spec.hoop_min_angle_offset_deg, 1e-6),
        abs(spec.hoop_nominal_angle_deg),
    )
    pitch_from_nominal = 2.0 * math.pi * radius / math.tan(math.radians(nominal_angle))
    if spec.hoop_mode == "nominal_angle":
        target_pitch = pitch_from_nominal
    turns_per_sweep = max(1, int(math.ceil(axial_length / target_pitch)))
    repeat_count = spec.number_of_passes or 1
    circuits = turns_per_sweep * repeat_count
    actual_pitch = axial_length / turns_per_sweep
    actual_angle = math.degrees(math.atan2(2.0 * math.pi * radius, actual_pitch))
    max_angle = 90.0 - (0.0 if spec.allow_exact_pure_hoop else spec.hoop_min_angle_offset_deg)
    if actual_angle > max_angle:
        actual_angle = max_angle
        actual_pitch = 2.0 * math.pi * radius / math.tan(math.radians(actual_angle))
        turns_per_sweep = max(1, int(math.ceil(axial_length / actual_pitch)))
        circuits = turns_per_sweep * repeat_count
        actual_pitch = axial_length / turns_per_sweep

    total_turns = max(1, circuits)
    points_per_turn = max(16, spec.point_count)
    point_count = max(2, total_turns * points_per_turn + 1)
    t = np.linspace(0.0, 1.0, point_count)
    theta_start = theta_offset_rad + math.radians(spec.start_offset_deg)
    theta_rad = theta_start + 2.0 * math.pi * total_turns * t
    if repeat_count == 1:
        z_mm = start_z + axial_length * t
    else:
        phase = t * repeat_count
        sweep_index = np.floor(phase).astype(int)
        sweep_fraction = phase - sweep_index
        sweep_index[-1] = repeat_count - 1
        sweep_fraction[-1] = 1.0
        forward_fraction = np.where(
            sweep_index % 2 == 0,
            sweep_fraction,
            1.0 - sweep_fraction,
        )
        z_mm = start_z + axial_length * forward_fraction
    points = mandrel.surface_points(z_mm, theta_rad)
    pass_index = np.minimum((t * total_turns).astype(int), total_turns - 1)
    warnings = []
    if not spec.allow_exact_pure_hoop and abs(abs(spec.target_angle_deg) - 90.0) <= 1e-9:
        warnings.append(
            f"pure hoop request converted to {actual_angle:.3f} deg continuous traverse"
        )
    path = SurfacePath(
        z_mm=z_mm,
        theta_rad=theta_rad,
        x_mm=points[:, 0],
        y_mm=points[:, 1],
        winding_angle_deg=actual_angle,
        tow_width_mm=spec.tow_width_mm,
        pass_index=pass_index,
        tow_eye_angle_deg=np.full(z_mm.shape, _signed_angle(spec.direction, actual_angle)),
    )
    gap_mm = max(actual_pitch - spec.tow_width_mm, 0.0)
    overlap_mm = max(spec.tow_width_mm - actual_pitch, 0.0)
    if gap_mm > 0.25:
        warnings.append(f"hoop traverse pitch gap {gap_mm:.3f} mm")
    if overlap_mm > spec.tow_width_mm * 0.35:
        warnings.append(f"hoop traverse overlap {overlap_mm:.3f} mm")
    return path, WindingPatternReport(
        layer_id=_resolved_layer_id(spec, 0),
        layer_name=spec.name,
        winding_type=spec.winding_type,
        target_angle_deg=spec.target_angle_deg,
        actual_angle_deg=_signed_angle(spec.direction, actual_angle),
        angle_error_deg=abs(abs(spec.target_angle_deg) - actual_angle),
        circuits=circuits,
        starts=1,
        angular_shift_deg=360.0,
        tow_spacing_mm=actual_pitch,
        coverage_percent=spec.tow_width_mm / actual_pitch * 100.0,
        gap_mm=gap_mm,
        overlap_mm=overlap_mm,
        layer_completion_z_mm=float(z_mm[-1]),
        pattern_repeat_length_mm=actual_pitch,
        closes=True,
        acceptable=not warnings,
        warnings=tuple(warnings),
    ), tuple("wind" for _ in range(path.point_count))


def _plan_cylinder_local_reinforcement_layer(
    mandrel: CylinderMandrel,
    spec: WindingLayerSpec,
    theta_offset_rad: float,
) -> tuple[SurfacePath, WindingPatternReport, tuple[str, ...]]:
    start_z = 0.0 if spec.start_z_mm is None else spec.start_z_mm
    end_z = mandrel.length_mm if spec.end_z_mm is None else spec.end_z_mm
    axial_length = end_z - start_z
    effective_width = spec.tow_width_mm / spec.coverage_target
    coverage_bands = max(1, math.ceil(axial_length / effective_width))
    repeat_count = spec.number_of_passes or 1
    circuits = coverage_bands * repeat_count
    if circuits == 1:
        z_positions = np.asarray([start_z + axial_length / 2.0], dtype=float)
    else:
        z_positions = np.linspace(
            start_z + effective_width / 2.0,
            end_z - effective_width / 2.0,
            coverage_bands,
        )
        z_positions = np.clip(z_positions, start_z, end_z)
        if repeat_count > 1:
            z_positions = np.tile(z_positions, repeat_count)
    points_per_ring = max(12, spec.point_count)
    z_chunks = []
    theta_chunks = []
    pass_chunks = []
    for circuit, z_value in enumerate(z_positions):
        theta = np.linspace(
            theta_offset_rad + math.radians(spec.start_offset_deg),
            theta_offset_rad + math.radians(spec.start_offset_deg) + 2.0 * math.pi,
            points_per_ring,
        )
        z_chunks.append(np.full(theta.shape, z_value, dtype=float))
        theta_chunks.append(theta)
        pass_chunks.append(np.full(theta.shape, circuit, dtype=int))
    z_mm = np.concatenate(z_chunks)
    theta_rad = np.concatenate(theta_chunks)
    points = mandrel.surface_points(z_mm, theta_rad)
    path = SurfacePath(
        z_mm=z_mm,
        theta_rad=theta_rad,
        x_mm=points[:, 0],
        y_mm=points[:, 1],
        winding_angle_deg=90.0,
        tow_width_mm=spec.tow_width_mm,
        pass_index=np.concatenate(pass_chunks),
        tow_eye_angle_deg=np.full(z_mm.shape, 90.0, dtype=float),
    )
    path, motion_type = _insert_pass_transitions(
        mandrel,
        path,
        point_count=spec.transition_points,
    )
    tow_spacing = axial_length / coverage_bands
    gap_mm = max(tow_spacing - spec.tow_width_mm, 0.0)
    overlap_mm = max(spec.tow_width_mm - tow_spacing, 0.0)
    warnings = () if gap_mm <= 0.25 else (f"hoop gap {gap_mm:.3f} mm",)
    return path, WindingPatternReport(
        layer_id=_resolved_layer_id(spec, 0),
        layer_name=spec.name,
        winding_type=spec.winding_type,
        target_angle_deg=90.0,
        actual_angle_deg=90.0,
        angle_error_deg=0.0,
        circuits=circuits,
        starts=circuits,
        angular_shift_deg=360.0,
        tow_spacing_mm=tow_spacing,
        coverage_percent=spec.tow_width_mm / tow_spacing * 100.0,
        gap_mm=gap_mm,
        overlap_mm=overlap_mm,
        layer_completion_z_mm=float(z_mm[-1]),
        pattern_repeat_length_mm=tow_spacing,
        closes=True,
        acceptable=not warnings,
        warnings=warnings,
    ), motion_type


def _plan_axisymmetric_hoop_layer(
    mandrel: AxisymmetricProfileMandrel,
    spec: WindingLayerSpec,
    theta_offset_rad: float,
) -> tuple[SurfacePath, WindingPatternReport, tuple[str, ...]]:
    return _plan_continuous_hoop_layer(mandrel, spec, theta_offset_rad)


def _plan_axisymmetric_local_reinforcement_layer(
    mandrel: AxisymmetricProfileMandrel,
    spec: WindingLayerSpec,
    theta_offset_rad: float,
) -> tuple[SurfacePath, WindingPatternReport, tuple[str, ...]]:
    start_z = mandrel.start_z_mm if spec.start_z_mm is None else spec.start_z_mm
    end_z = mandrel.end_z_mm if spec.end_z_mm is None else spec.end_z_mm
    axial_length = end_z - start_z
    effective_width = spec.tow_width_mm / spec.coverage_target
    coverage_bands = max(1, math.ceil(axial_length / effective_width))
    repeat_count = spec.number_of_passes or 1
    circuits = coverage_bands * repeat_count
    if coverage_bands == 1:
        z_positions = np.asarray([start_z + axial_length / 2.0], dtype=float)
    else:
        z_positions = np.linspace(
            start_z + effective_width / 2.0,
            end_z - effective_width / 2.0,
            coverage_bands,
        )
        z_positions = np.clip(z_positions, start_z, end_z)
        if repeat_count > 1:
            z_positions = np.tile(z_positions, repeat_count)
    points_per_ring = max(12, spec.point_count)
    z_chunks = []
    theta_chunks = []
    pass_chunks = []
    for circuit, z_value in enumerate(z_positions):
        theta = np.linspace(
            theta_offset_rad + math.radians(spec.start_offset_deg),
            theta_offset_rad + math.radians(spec.start_offset_deg) + 2.0 * math.pi,
            points_per_ring,
        )
        z_chunks.append(np.full(theta.shape, z_value, dtype=float))
        theta_chunks.append(theta)
        pass_chunks.append(np.full(theta.shape, circuit, dtype=int))
    z_mm = np.concatenate(z_chunks)
    theta_rad = np.concatenate(theta_chunks)
    points = mandrel.surface_points(z_mm, theta_rad)
    path = SurfacePath(
        z_mm=z_mm,
        theta_rad=theta_rad,
        x_mm=points[:, 0],
        y_mm=points[:, 1],
        winding_angle_deg=90.0,
        tow_width_mm=spec.tow_width_mm,
        pass_index=np.concatenate(pass_chunks),
        tow_eye_angle_deg=np.full(z_mm.shape, 90.0, dtype=float),
    )
    path, motion_type = _insert_pass_transitions(
        mandrel,
        path,
        point_count=spec.transition_points,
    )
    tow_spacing = axial_length / coverage_bands
    gap_mm = max(tow_spacing - spec.tow_width_mm, 0.0)
    overlap_mm = max(spec.tow_width_mm - tow_spacing, 0.0)
    warnings = () if gap_mm <= 0.25 else (f"hoop gap {gap_mm:.3f} mm",)
    return path, WindingPatternReport(
        layer_id=_resolved_layer_id(spec, 0),
        layer_name=spec.name,
        winding_type=spec.winding_type,
        target_angle_deg=90.0,
        actual_angle_deg=90.0,
        angle_error_deg=0.0,
        circuits=circuits,
        starts=circuits,
        angular_shift_deg=360.0,
        tow_spacing_mm=tow_spacing,
        coverage_percent=spec.tow_width_mm / tow_spacing * 100.0,
        gap_mm=gap_mm,
        overlap_mm=overlap_mm,
        layer_completion_z_mm=float(z_mm[-1]),
        pattern_repeat_length_mm=tow_spacing,
        closes=True,
        acceptable=not warnings,
        warnings=warnings,
    ), motion_type


def _plan_profile_dome_layer(
    mandrel: AxisymmetricProfileMandrel,
    spec: WindingLayerSpec,
    theta_offset_rad: float,
) -> tuple[SurfacePath, WindingPatternReport]:
    circumference = 2.0 * math.pi * mandrel.max_radius_mm
    circuits = spec.number_of_passes or max(
        1,
        math.ceil(circumference * spec.coverage_target / spec.tow_width_mm),
    )
    total_passes = max(1, math.ceil(circuits / 2.0))
    config = ProfileDomePathConfig(
        winding_angle_deg=abs(spec.target_angle_deg),
        tow_width_mm=spec.tow_width_mm,
        points_per_span=spec.point_count,
        turnaround_points=spec.turnaround_points,
        turnaround_angle_deg=spec.turnaround_angle_deg,
        circuits=total_passes,
        start_theta_rad=theta_offset_rad + math.radians(spec.start_offset_deg),
        turnaround_radius_mm=spec.turnaround_radius_mm,
        phase_offset_deg=360.0 / total_passes if total_passes > 1 else 0.0,
    )
    generator = ProfileDomePathGenerator(mandrel, config)
    path = generator.generate()
    path = _apply_direction(mandrel, path, spec.direction, abs(spec.target_angle_deg))
    tow_spacing = circumference / circuits
    gap_mm = max(tow_spacing - spec.tow_width_mm, 0.0)
    overlap_mm = max(spec.tow_width_mm - tow_spacing, 0.0)
    warnings = _pattern_warnings(
        target_angle=abs(spec.target_angle_deg),
        actual_angle=abs(spec.target_angle_deg),
        gap_mm=gap_mm,
        overlap_mm=overlap_mm,
        max_angle_error=spec.max_angle_error_deg,
    )
    return path, WindingPatternReport(
        layer_id=_resolved_layer_id(spec, 0),
        layer_name=spec.name,
        winding_type=spec.winding_type,
        target_angle_deg=spec.target_angle_deg,
        actual_angle_deg=_signed_angle(spec.direction, abs(spec.target_angle_deg)),
        angle_error_deg=0.0,
        circuits=circuits,
        starts=circuits,
        angular_shift_deg=360.0 / circuits,
        tow_spacing_mm=tow_spacing,
        coverage_percent=spec.tow_width_mm / tow_spacing * 100.0,
        gap_mm=gap_mm,
        overlap_mm=overlap_mm,
        layer_completion_z_mm=float(path.z_mm[-1]),
        pattern_repeat_length_mm=generator.safe_zone.length_mm,
        closes=True,
        acceptable=not warnings,
        warnings=warnings,
    )


def _plan_axisymmetric_geodesic_layer(
    mandrel: AxisymmetricProfileMandrel,
    spec: WindingLayerSpec,
    theta_offset_rad: float,
) -> tuple[SurfacePath, WindingPatternReport, tuple[str, ...]]:
    safe_radius = _axisymmetric_safe_turnaround_radius(mandrel, spec)
    start_z, end_z, safe_start_z, safe_end_z = _axisymmetric_turnaround_z_bounds(
        mandrel,
        spec,
    )
    selected_legs = _even_axisymmetric_lane_count(mandrel, spec)
    circuits = max(1, selected_legs // 2)
    phase_offset = (
        360.0 / selected_legs if spec.phase_offset_deg is None else spec.phase_offset_deg
    )
    phase_offset_rad = math.radians(phase_offset)
    reference_z = _axisymmetric_reference_z(mandrel, start_z, end_z)
    forward_reference_path, _forward_reference_diagnostics = generate_geodesic_path(
        mandrel,
        GeodesicPathConfig(
            initial_angle_deg=abs(spec.target_angle_deg),
            tow_width_mm=spec.tow_width_mm,
            start_z_mm=start_z,
            end_z_mm=end_z,
            start_theta_rad=0.0,
            direction="positive",
            turnaround_radius_mm=safe_radius,
            reference_radius_mm=mandrel.max_radius_mm,
            point_count=spec.point_count,
        ),
    )
    return_reference_path, _return_reference_diagnostics = generate_geodesic_path(
        mandrel,
        GeodesicPathConfig(
            initial_angle_deg=abs(spec.target_angle_deg),
            tow_width_mm=spec.tow_width_mm,
            start_z_mm=start_z,
            end_z_mm=end_z,
            start_theta_rad=0.0,
            direction="negative",
            turnaround_radius_mm=safe_radius,
            reference_radius_mm=mandrel.max_radius_mm,
            point_count=spec.point_count,
        ),
    )
    forward_reference_theta = _path_theta_at_z(forward_reference_path, reference_z)
    return_reference_theta = _path_theta_at_z(return_reference_path, reference_z)
    chunks = []
    warnings: list[str] = []
    motion_chunks: list[tuple[str, ...]] = []
    next_forward_alignment_theta: float | None = None
    for circuit_number in range(circuits):
        forward_lane = circuit_number * 2
        return_lane = forward_lane + 1
        start_theta = (
            theta_offset_rad
            + forward_lane * phase_offset_rad
            - forward_reference_theta
        )
        forward_path, forward_diagnostics = generate_geodesic_path(
            mandrel,
            GeodesicPathConfig(
                initial_angle_deg=abs(spec.target_angle_deg),
                tow_width_mm=spec.tow_width_mm,
                start_z_mm=start_z,
                end_z_mm=end_z,
                start_theta_rad=start_theta,
                direction="positive",
                turnaround_radius_mm=safe_radius,
                reference_radius_mm=mandrel.max_radius_mm,
                point_count=spec.point_count,
            ),
        )
        forward_path = _densify_surface_path(
            mandrel,
            forward_path,
            max_segment_length_mm=_axisymmetric_max_segment_length(spec),
        )
        if next_forward_alignment_theta is not None:
            forward_path = _align_path_start_theta(
                mandrel,
                forward_path,
                next_forward_alignment_theta,
            )
            next_forward_alignment_theta = None
        return_start_theta = (
            theta_offset_rad
            + return_lane * phase_offset_rad
            - return_reference_theta
        )
        return_path, return_diagnostics = generate_geodesic_path(
            mandrel,
            GeodesicPathConfig(
                initial_angle_deg=abs(spec.target_angle_deg),
                tow_width_mm=spec.tow_width_mm,
                start_z_mm=start_z,
                end_z_mm=end_z,
                start_theta_rad=return_start_theta,
                direction="negative",
                turnaround_radius_mm=safe_radius,
                reference_radius_mm=mandrel.max_radius_mm,
                point_count=spec.point_count,
            ),
        )
        return_path = _densify_surface_path(
            mandrel,
            return_path,
            max_segment_length_mm=_axisymmetric_max_segment_length(spec),
        )
        end_turnaround = _smooth_dome_turnaround_path(
            mandrel,
            previous_path=forward_path,
            next_path=return_path,
            safe_start_z_mm=safe_start_z,
            safe_end_z_mm=safe_end_z,
            point_count=spec.turnaround_points,
            wrap_angle_deg=spec.turnaround_angle_deg,
        )
        return_path = _align_path_start_theta(
            mandrel,
            return_path,
            float(end_turnaround.theta_rad[-1]),
        )
        forward_output = (
            _drop_first_point(forward_path) if circuit_number > 0 else forward_path
        )
        chunks.append(_with_pass_index(forward_output, circuit_number * 2))
        chunks.append(_with_pass_index(_drop_first_point(end_turnaround), circuit_number * 2))
        return_path = _drop_first_point(return_path)
        chunks.append(_with_pass_index(return_path, circuit_number * 2 + 1))
        motion_chunks.append(tuple("wind" for _ in range(forward_output.point_count)))
        motion_chunks.append(
            tuple(
                "DomeTurnaround"
                for _ in range(max(end_turnaround.point_count - 1, 0))
            )
        )
        motion_chunks.append(tuple("wind" for _ in range(return_path.point_count)))
        if circuit_number < circuits - 1:
            next_start_theta = (
                theta_offset_rad
                + (forward_lane + 2) * phase_offset_rad
                - forward_reference_theta
            )
            next_forward_path, _next_forward_diagnostics = generate_geodesic_path(
                mandrel,
                GeodesicPathConfig(
                    initial_angle_deg=abs(spec.target_angle_deg),
                    tow_width_mm=spec.tow_width_mm,
                    start_z_mm=start_z,
                    end_z_mm=end_z,
                    start_theta_rad=next_start_theta,
                    direction="positive",
                    turnaround_radius_mm=safe_radius,
                    reference_radius_mm=mandrel.max_radius_mm,
                    point_count=spec.point_count,
                ),
            )
            next_forward_path = _densify_surface_path(
                mandrel,
                next_forward_path,
                max_segment_length_mm=_axisymmetric_max_segment_length(spec),
            )
            start_turnaround = _smooth_dome_turnaround_path(
                mandrel,
                previous_path=return_path,
                next_path=next_forward_path,
                safe_start_z_mm=safe_start_z,
                safe_end_z_mm=safe_end_z,
                point_count=spec.turnaround_points,
                wrap_angle_deg=spec.turnaround_angle_deg,
            )
            next_forward_alignment_theta = float(start_turnaround.theta_rad[-1])
            chunks.append(
                _with_pass_index(_drop_first_point(start_turnaround), circuit_number * 2 + 1)
            )
            motion_chunks.append(
                tuple(
                    "DomeTurnaround"
                    for _ in range(max(start_turnaround.point_count - 1, 0))
                )
            )
        warnings.extend(forward_diagnostics.warning_flags)
        warnings.extend(return_diagnostics.warning_flags)
    path = _concatenate_surface_paths(chunks)
    path = _apply_direction(mandrel, path, spec.direction, abs(spec.target_angle_deg))
    motion_type = tuple(label for chunk in motion_chunks for label in chunk)
    report = _axisymmetric_report(
        mandrel,
        spec,
        path,
        circuits=circuits,
        winding_lanes=selected_legs,
        warnings=tuple(sorted(set(warnings))),
    )
    return path, report, motion_type


def _plan_axisymmetric_non_geodesic_layer(
    mandrel: AxisymmetricProfileMandrel,
    spec: WindingLayerSpec,
    theta_offset_rad: float,
) -> tuple[SurfacePath, WindingPatternReport, tuple[str, ...]]:
    start_z, end_z, safe_start_z, safe_end_z = _axisymmetric_turnaround_z_bounds(
        mandrel,
        spec,
    )
    selected_legs = _even_axisymmetric_lane_count(mandrel, spec)
    circuits = max(1, selected_legs // 2)
    phase_offset = (
        360.0 / selected_legs if spec.phase_offset_deg is None else spec.phase_offset_deg
    )
    phase_offset_rad = math.radians(phase_offset)
    reference_z = _axisymmetric_reference_z(mandrel, start_z, end_z)
    forward_reference_path, _forward_reference_diagnostics = generate_controlled_angle_path(
        mandrel,
        ControlledAnglePathConfig(
            target_angle_deg=abs(spec.target_angle_deg),
            tow_width_mm=spec.tow_width_mm,
            start_z_mm=start_z,
            end_z_mm=end_z,
            start_theta_rad=0.0,
            direction="positive",
            reference_radius_mm=mandrel.max_radius_mm,
            point_count=spec.point_count,
        ),
    )
    return_reference_path, _return_reference_diagnostics = generate_controlled_angle_path(
        mandrel,
        ControlledAnglePathConfig(
            target_angle_deg=abs(spec.target_angle_deg),
            tow_width_mm=spec.tow_width_mm,
            start_z_mm=start_z,
            end_z_mm=end_z,
            start_theta_rad=0.0,
            direction="negative",
            reference_radius_mm=mandrel.max_radius_mm,
            point_count=spec.point_count,
        ),
    )
    forward_reference_theta = _path_theta_at_z(forward_reference_path, reference_z)
    return_reference_theta = _path_theta_at_z(return_reference_path, reference_z)
    chunks = []
    warnings: list[str] = []
    motion_chunks: list[tuple[str, ...]] = []
    next_forward_alignment_theta: float | None = None
    for circuit_number in range(circuits):
        forward_lane = circuit_number * 2
        return_lane = forward_lane + 1
        start_theta = (
            theta_offset_rad
            + forward_lane * phase_offset_rad
            - forward_reference_theta
        )
        forward_path, forward_diagnostics = generate_controlled_angle_path(
            mandrel,
            ControlledAnglePathConfig(
                target_angle_deg=abs(spec.target_angle_deg),
                tow_width_mm=spec.tow_width_mm,
                start_z_mm=start_z,
                end_z_mm=end_z,
                start_theta_rad=start_theta,
                direction="positive",
                reference_radius_mm=mandrel.max_radius_mm,
                point_count=spec.point_count,
            ),
        )
        forward_path = _densify_surface_path(
            mandrel,
            forward_path,
            max_segment_length_mm=_axisymmetric_max_segment_length(spec),
        )
        if next_forward_alignment_theta is not None:
            forward_path = _align_path_start_theta(
                mandrel,
                forward_path,
                next_forward_alignment_theta,
            )
            next_forward_alignment_theta = None
        return_start_theta = (
            theta_offset_rad
            + return_lane * phase_offset_rad
            - return_reference_theta
        )
        return_path, return_diagnostics = generate_controlled_angle_path(
            mandrel,
            ControlledAnglePathConfig(
                target_angle_deg=abs(spec.target_angle_deg),
                tow_width_mm=spec.tow_width_mm,
                start_z_mm=start_z,
                end_z_mm=end_z,
                start_theta_rad=return_start_theta,
                direction="negative",
                reference_radius_mm=mandrel.max_radius_mm,
                point_count=spec.point_count,
            ),
        )
        return_path = _densify_surface_path(
            mandrel,
            return_path,
            max_segment_length_mm=_axisymmetric_max_segment_length(spec),
        )
        end_turnaround = _smooth_dome_turnaround_path(
            mandrel,
            previous_path=forward_path,
            next_path=return_path,
            safe_start_z_mm=safe_start_z,
            safe_end_z_mm=safe_end_z,
            point_count=spec.turnaround_points,
            wrap_angle_deg=spec.turnaround_angle_deg,
        )
        return_path = _align_path_start_theta(
            mandrel,
            return_path,
            float(end_turnaround.theta_rad[-1]),
        )
        forward_output = (
            _drop_first_point(forward_path) if circuit_number > 0 else forward_path
        )
        chunks.append(_with_pass_index(forward_output, circuit_number * 2))
        chunks.append(_with_pass_index(_drop_first_point(end_turnaround), circuit_number * 2))
        return_path = _drop_first_point(return_path)
        chunks.append(_with_pass_index(return_path, circuit_number * 2 + 1))
        motion_chunks.append(tuple("wind" for _ in range(forward_output.point_count)))
        motion_chunks.append(
            tuple(
                "DomeTurnaround"
                for _ in range(max(end_turnaround.point_count - 1, 0))
            )
        )
        motion_chunks.append(tuple("wind" for _ in range(return_path.point_count)))
        if circuit_number < circuits - 1:
            next_start_theta = (
                theta_offset_rad
                + (forward_lane + 2) * phase_offset_rad
                - forward_reference_theta
            )
            next_forward_path, _next_forward_diagnostics = generate_controlled_angle_path(
                mandrel,
                ControlledAnglePathConfig(
                    target_angle_deg=abs(spec.target_angle_deg),
                    tow_width_mm=spec.tow_width_mm,
                    start_z_mm=start_z,
                    end_z_mm=end_z,
                    start_theta_rad=next_start_theta,
                    direction="positive",
                    reference_radius_mm=mandrel.max_radius_mm,
                    point_count=spec.point_count,
                ),
            )
            next_forward_path = _densify_surface_path(
                mandrel,
                next_forward_path,
                max_segment_length_mm=_axisymmetric_max_segment_length(spec),
            )
            start_turnaround = _smooth_dome_turnaround_path(
                mandrel,
                previous_path=return_path,
                next_path=next_forward_path,
                safe_start_z_mm=safe_start_z,
                safe_end_z_mm=safe_end_z,
                point_count=spec.turnaround_points,
                wrap_angle_deg=spec.turnaround_angle_deg,
            )
            next_forward_alignment_theta = float(start_turnaround.theta_rad[-1])
            chunks.append(
                _with_pass_index(_drop_first_point(start_turnaround), circuit_number * 2 + 1)
            )
            motion_chunks.append(
                tuple(
                    "DomeTurnaround"
                    for _ in range(max(start_turnaround.point_count - 1, 0))
                )
            )
        warnings.extend(forward_diagnostics.warning_flags)
        warnings.extend(return_diagnostics.warning_flags)
    path = _concatenate_surface_paths(chunks)
    path = _apply_direction(mandrel, path, spec.direction, abs(spec.target_angle_deg))
    motion_type = tuple(label for chunk in motion_chunks for label in chunk)
    report = _axisymmetric_report(
        mandrel,
        spec,
        path,
        circuits=circuits,
        winding_lanes=selected_legs,
        warnings=tuple(sorted(set(warnings))),
    )
    return path, report, motion_type


def _plan_profile_turnaround_layer(
    mandrel: AxisymmetricProfileMandrel,
    spec: WindingLayerSpec,
    theta_offset_rad: float,
) -> tuple[SurfacePath, WindingPatternReport]:
    circumference = 2.0 * math.pi * mandrel.max_radius_mm
    circuits = spec.number_of_passes or max(
        1,
        math.ceil(circumference * spec.coverage_target / spec.tow_width_mm),
    )
    min_radius = _profile_turnaround_min_radius(mandrel, spec)
    config = ProfileTurnaroundPathConfig(
        winding_angle_deg=abs(spec.target_angle_deg),
        tow_width_mm=spec.tow_width_mm,
        points_per_span=spec.point_count,
        turnaround_points=spec.turnaround_points,
        min_radius_mm=min_radius,
        turnaround_angle_deg=spec.turnaround_angle_deg,
        circuits=max(1, math.ceil(circuits / 2.0)),
        start_theta_rad=theta_offset_rad + math.radians(spec.start_offset_deg),
    )
    generator = ProfileTurnaroundPathGenerator(mandrel, config)
    path = generator.generate()
    path = _apply_direction(mandrel, path, spec.direction, abs(spec.target_angle_deg))
    tow_spacing = circumference / circuits
    gap_mm = max(tow_spacing - spec.tow_width_mm, 0.0)
    overlap_mm = max(spec.tow_width_mm - tow_spacing, 0.0)
    warnings = _pattern_warnings(
        target_angle=abs(spec.target_angle_deg),
        actual_angle=abs(spec.target_angle_deg),
        gap_mm=gap_mm,
        overlap_mm=overlap_mm,
        max_angle_error=spec.max_angle_error_deg,
    )
    return path, WindingPatternReport(
        layer_id=_resolved_layer_id(spec, 0),
        layer_name=spec.name,
        winding_type=spec.winding_type,
        target_angle_deg=spec.target_angle_deg,
        actual_angle_deg=_signed_angle(spec.direction, abs(spec.target_angle_deg)),
        angle_error_deg=0.0,
        circuits=circuits,
        starts=circuits,
        angular_shift_deg=360.0 / circuits,
        tow_spacing_mm=tow_spacing,
        coverage_percent=spec.tow_width_mm / tow_spacing * 100.0,
        gap_mm=gap_mm,
        overlap_mm=overlap_mm,
        layer_completion_z_mm=float(path.z_mm[-1]),
        pattern_repeat_length_mm=generator.safe_zone.length_mm,
        closes=True,
        acceptable=not warnings,
        warnings=warnings,
    )


def _transition_between_paths(
    mandrel: MandrelLike,
    previous_path: SurfacePath,
    next_path: SurfacePath,
    *,
    layer_index: int,
    point_count: int,
    radial_clearance_mm: float,
    nominal_feedrate_mm_min: float,
    minimum_feedrate_mm_min: float | None,
    effective_radius_mm: float,
    accumulated_thickness_before_mm: float,
) -> PlannedLayer:
    start_z = float(previous_path.z_mm[-1])
    end_z = float(next_path.z_mm[0])
    start_theta = float(previous_path.theta_rad[-1])
    end_theta = float(next_path.theta_rad[0])
    theta_delta = _unwrap_delta(start_theta, end_theta)
    point_count = _resolved_transition_point_count(
        mandrel,
        start_z=start_z,
        end_z=end_z,
        start_theta=start_theta,
        theta_delta=theta_delta,
        tow_width_mm=next_path.tow_width_mm,
        minimum_point_count=point_count,
    )
    z_t = np.linspace(0.0, 1.0, point_count)
    eased = _smoothstep(z_t)
    z_mm = start_z + (end_z - start_z) * eased
    theta_rad = start_theta + theta_delta * eased
    points = mandrel.surface_points(z_mm, theta_rad)
    b_start = _path_b_angle(previous_path)[-1]
    b_end = _path_b_angle(next_path)[0]
    path = SurfacePath(
        z_mm=z_mm,
        theta_rad=theta_rad,
        x_mm=points[:, 0],
        y_mm=points[:, 1],
        winding_angle_deg=0.0,
        tow_width_mm=next_path.tow_width_mm,
        pass_index=np.full(point_count, 0, dtype=int),
        tow_eye_angle_deg=np.linspace(b_start, b_end, point_count),
    )
    motion_table = machine_path_from_surface_path(path, radial_clearance_mm=radial_clearance_mm)
    feed_schedule = plan_feedrate(
        path,
        FeedrateConfig(
            nominal_feedrate_mm_min=nominal_feedrate_mm_min * 0.25,
            minimum_feedrate_mm_min=minimum_feedrate_mm_min,
        ),
    )
    motion_table = _smooth_motion_b_axis(motion_table, path)
    feed_schedule = _slow_for_b_axis_changes(feed_schedule, motion_table, path)
    metadata = _metadata_for_path(
        mandrel,
        path,
        layer_id="transition",
        layer_index=layer_index,
        layer_name="transition",
        winding_type="transition",
        motion_type=tuple("transition" for _ in range(path.point_count)),
        warning_flags=(),
    )
    report = WindingPatternReport(
        layer_id="transition",
        layer_name="transition",
        winding_type="transition",
        target_angle_deg=0.0,
        actual_angle_deg=0.0,
        angle_error_deg=0.0,
        circuits=1,
        starts=1,
        angular_shift_deg=0.0,
        tow_spacing_mm=0.0,
        coverage_percent=0.0,
        gap_mm=0.0,
        overlap_mm=0.0,
        layer_completion_z_mm=float(path.z_mm[-1]),
        pattern_repeat_length_mm=0.0,
        closes=True,
        acceptable=True,
    )
    return PlannedLayer(
        spec=WindingLayerSpec(
            name="transition",
            winding_type="transition",
            target_angle_deg=1.0,
            tow_width_mm=next_path.tow_width_mm,
        ),
        path=path,
        motion_table=motion_table,
        feed_schedule=feed_schedule,
        metadata=metadata,
        report=report,
        effective_radius_mm=effective_radius_mm,
        accumulated_thickness_before_mm=accumulated_thickness_before_mm,
    )


def _smoothstep(values: FloatArray) -> FloatArray:
    values = np.asarray(values, dtype=float)
    return values * values * (3.0 - 2.0 * values)


def _resolved_transition_point_count(
    mandrel: MandrelLike,
    *,
    start_z: float,
    end_z: float,
    start_theta: float,
    theta_delta: float,
    tow_width_mm: float,
    minimum_point_count: int,
) -> int:
    start_point = mandrel.surface_points(
        np.asarray([start_z], dtype=float),
        np.asarray([start_theta], dtype=float),
    )[0]
    end_point = mandrel.surface_points(
        np.asarray([end_z], dtype=float),
        np.asarray([start_theta + theta_delta], dtype=float),
    )[0]
    distance = float(np.linalg.norm(end_point - start_point))
    max_step = max(25.0, tow_width_mm * 6.0)
    eased_step_allowance = max_step / 1.6
    return max(minimum_point_count, int(math.ceil(distance / eased_step_allowance)) + 1)


def _unwrap_delta(start_value: float, end_value: float) -> float:
    delta = end_value - start_value
    return ((delta + math.pi) % (2.0 * math.pi)) - math.pi


def _insert_pass_transitions(
    mandrel: MandrelLike,
    path: SurfacePath,
    *,
    point_count: int,
) -> tuple[SurfacePath, tuple[str, ...]]:
    """Insert followable transition samples between pass chunks in one layer."""

    pass_index = np.asarray(path.pass_index, dtype=int)
    pass_numbers = tuple(int(value) for value in np.unique(pass_index))
    if len(pass_numbers) <= 1:
        return path, tuple("wind" for _ in range(path.point_count))

    z_chunks: list[FloatArray] = []
    theta_chunks: list[FloatArray] = []
    pass_chunks: list[IntArray] = []
    b_chunks: list[FloatArray] = []
    motion_chunks: list[tuple[str, ...]] = []
    b_angles = _path_b_angle(path)
    transition_t = _smoothstep(np.linspace(0.0, 1.0, max(2, point_count) + 2)[1:-1])

    for chunk_index, pass_number in enumerate(pass_numbers):
        mask = pass_index == pass_number
        z_pass = path.z_mm[mask]
        theta_pass = path.theta_rad[mask]
        b_pass = b_angles[mask]
        z_chunks.append(z_pass)
        theta_chunks.append(theta_pass)
        pass_chunks.append(np.full(z_pass.shape, pass_number, dtype=int))
        b_chunks.append(b_pass)
        motion_chunks.append(tuple("wind" for _ in range(z_pass.size)))

        if chunk_index == len(pass_numbers) - 1:
            continue
        next_mask = pass_index == pass_numbers[chunk_index + 1]
        start_z = float(z_pass[-1])
        end_z = float(path.z_mm[next_mask][0])
        start_theta = float(theta_pass[-1])
        end_theta = float(path.theta_rad[next_mask][0])
        theta_delta = _unwrap_delta(start_theta, end_theta)
        transition_z = start_z + (end_z - start_z) * transition_t
        transition_theta = start_theta + theta_delta * transition_t
        start_b = float(b_pass[-1])
        end_b = float(b_angles[next_mask][0])
        transition_b = start_b + (end_b - start_b) * transition_t
        z_chunks.append(transition_z)
        theta_chunks.append(transition_theta)
        pass_chunks.append(np.full(transition_z.shape, pass_number, dtype=int))
        b_chunks.append(transition_b)
        motion_chunks.append(tuple("transition" for _ in range(transition_z.size)))

    z_mm = np.concatenate(z_chunks)
    theta_rad = np.concatenate(theta_chunks)
    points = mandrel.surface_points(z_mm, theta_rad)
    transition_path = SurfacePath(
        z_mm=z_mm,
        theta_rad=theta_rad,
        x_mm=points[:, 0],
        y_mm=points[:, 1],
        winding_angle_deg=path.winding_angle_deg,
        tow_width_mm=path.tow_width_mm,
        pass_index=np.concatenate(pass_chunks),
        tow_eye_angle_deg=np.concatenate(b_chunks),
    )
    motion_type = tuple(label for chunk in motion_chunks for label in chunk)
    return transition_path, motion_type


def _metadata_for_path(
    mandrel: MandrelLike,
    path: SurfacePath,
    *,
    layer_id: str,
    layer_index: int,
    layer_name: str,
    winding_type: WindingType,
    motion_type: tuple[str, ...] | None = None,
    warning_flags: tuple[str, ...] = (),
) -> WindingPointMetadata:
    point_count = path.point_count
    motion_labels = (
        tuple("wind" for _ in range(point_count))
        if motion_type is None
        else motion_type
    )
    if len(motion_labels) != point_count:
        raise ValueError("motion_type length must match path point count")
    local_radius = mandrel.radius_at(path.z_mm)
    local_angle = _local_winding_angle(path)
    pass_index = np.asarray(path.pass_index, dtype=int)
    warning_text = "; ".join(warning_flags)
    return WindingPointMetadata(
        layer_id=tuple(layer_id for _ in range(point_count)),
        layer_index=np.full(point_count, layer_index, dtype=int),
        circuit_index=pass_index,
        pass_index=pass_index.copy(),
        local_radius_mm=local_radius,
        local_winding_angle_deg=local_angle,
        layer_name=tuple(layer_name for _ in range(point_count)),
        winding_type=tuple(winding_type for _ in range(point_count)),
        motion_type=motion_labels,
        warning_flags=tuple(warning_text for _ in range(point_count)),
    )


def _local_winding_angle(path: SurfacePath) -> FloatArray:
    radius = path.surface_radius_mm
    dz = np.gradient(path.z_mm)
    dtheta = np.gradient(path.theta_rad)
    dr = np.gradient(radius)
    meridian = np.sqrt(dz**2 + dr**2)
    circumferential = radius * dtheta
    return np.rad2deg(np.arctan2(np.abs(circumferential), np.maximum(np.abs(meridian), 1e-12)))


def _apply_direction(
    mandrel: MandrelLike,
    path: SurfacePath,
    direction: LayerDirection,
    actual_angle_deg: float,
) -> SurfacePath:
    if direction not in {"negative", "polar"}:
        return path
    theta_rad = 2.0 * path.theta_rad[0] - path.theta_rad
    points = mandrel.surface_points(path.z_mm, theta_rad)
    b_angle = -abs(actual_angle_deg)
    if path.tow_eye_angle_deg is not None:
        b_angles = -np.abs(path.tow_eye_angle_deg)
    else:
        b_angles = np.full(path.point_count, b_angle, dtype=float)
    return SurfacePath(
        z_mm=path.z_mm,
        theta_rad=theta_rad,
        x_mm=points[:, 0],
        y_mm=points[:, 1],
        winding_angle_deg=abs(actual_angle_deg),
        tow_width_mm=path.tow_width_mm,
        pass_index=path.pass_index,
        tow_eye_angle_deg=b_angles,
    )


def _resolve_alternating_direction(
    spec: WindingLayerSpec,
    layer_index: int,
) -> WindingLayerSpec:
    if spec.direction != "alternating":
        return spec
    direction: LayerDirection = "positive" if layer_index % 2 == 0 else "negative"
    return replace(spec, direction=direction)


def _concatenate_surface_paths(paths: list[SurfacePath]) -> SurfacePath:
    if not paths:
        raise ValueError("cannot concatenate an empty path list")
    z_mm = np.concatenate([path.z_mm for path in paths])
    theta_rad = np.concatenate([path.theta_rad for path in paths])
    x_mm = np.concatenate([path.x_mm for path in paths])
    y_mm = np.concatenate([path.y_mm for path in paths])
    pass_index = np.concatenate([np.asarray(path.pass_index, dtype=int) for path in paths])
    tow_eye_angle = np.concatenate([_path_b_angle(path) for path in paths])
    return SurfacePath(
        z_mm=z_mm,
        theta_rad=theta_rad,
        x_mm=x_mm,
        y_mm=y_mm,
        winding_angle_deg=paths[0].winding_angle_deg,
        tow_width_mm=paths[0].tow_width_mm,
        pass_index=pass_index,
        tow_eye_angle_deg=tow_eye_angle,
    )


def _drop_first_point(path: SurfacePath) -> SurfacePath:
    if path.point_count <= 1:
        return path
    return SurfacePath(
        z_mm=path.z_mm[1:],
        theta_rad=path.theta_rad[1:],
        x_mm=path.x_mm[1:],
        y_mm=path.y_mm[1:],
        winding_angle_deg=path.winding_angle_deg,
        tow_width_mm=path.tow_width_mm,
        pass_index=(
            np.asarray(path.pass_index, dtype=int)[1:]
            if path.pass_index is not None
            else None
        ),
        tow_eye_angle_deg=(
            path.tow_eye_angle_deg[1:] if path.tow_eye_angle_deg is not None else None
        ),
    )


def _align_path_start_theta(
    mandrel: MandrelLike,
    path: SurfacePath,
    target_start_theta_rad: float,
) -> SurfacePath:
    theta_rad = np.asarray(path.theta_rad, dtype=float)
    if theta_rad.size == 0:
        return path
    offset = target_start_theta_rad - theta_rad[0]
    aligned_theta = theta_rad + offset
    points = mandrel.surface_points(path.z_mm, aligned_theta)
    return SurfacePath(
        z_mm=path.z_mm,
        theta_rad=aligned_theta,
        x_mm=points[:, 0],
        y_mm=points[:, 1],
        winding_angle_deg=path.winding_angle_deg,
        tow_width_mm=path.tow_width_mm,
        pass_index=path.pass_index,
        tow_eye_angle_deg=path.tow_eye_angle_deg,
    )


def _constant_z_turnaround_arc(
    mandrel: MandrelLike,
    *,
    z_mm: float,
    start_theta_rad: float,
    end_theta_rad: float,
    tow_width_mm: float,
    winding_angle_deg: float,
    point_count: int,
) -> SurfacePath:
    count = max(3, min(int(point_count), 8))
    end_theta_rad = start_theta_rad + _unwrap_delta(start_theta_rad, end_theta_rad)
    theta_rad = np.linspace(start_theta_rad, end_theta_rad, count)
    z_values = np.full(theta_rad.shape, z_mm, dtype=float)
    points = mandrel.surface_points(z_values, theta_rad)
    return SurfacePath(
        z_mm=z_values,
        theta_rad=theta_rad,
        x_mm=points[:, 0],
        y_mm=points[:, 1],
        winding_angle_deg=winding_angle_deg,
        tow_width_mm=tow_width_mm,
        pass_index=np.zeros(theta_rad.shape, dtype=int),
        tow_eye_angle_deg=np.full(theta_rad.shape, 90.0, dtype=float),
    )


def _smooth_dome_turnaround_path(
    mandrel: AxisymmetricProfileMandrel,
    *,
    previous_path: SurfacePath,
    next_path: SurfacePath,
    safe_start_z_mm: float,
    safe_end_z_mm: float,
    point_count: int,
    wrap_angle_deg: float,
) -> SurfacePath:
    count = max(31, int(point_count))
    start_z = float(previous_path.z_mm[-1])
    end_z = float(next_path.z_mm[0])
    start_theta = float(previous_path.theta_rad[-1])
    boundary_z = _turnaround_boundary_z(
        start_z=start_z,
        safe_start_z_mm=safe_start_z_mm,
        safe_end_z_mm=safe_end_z_mm,
    )
    z_mm, theta_rad = _wrapped_dome_turnaround_curve(
        previous_path,
        next_path,
        start_z=start_z,
        end_z=end_z,
        boundary_z=boundary_z,
        start_theta=start_theta,
        point_count=count,
        min_z=min(safe_start_z_mm, safe_end_z_mm),
        max_z=max(safe_start_z_mm, safe_end_z_mm),
        wrap_angle_deg=wrap_angle_deg,
    )
    sample_count = int(z_mm.size)
    points = mandrel.surface_points(z_mm, theta_rad)
    tow_eye_angle = _local_winding_angle(
        SurfacePath(
            z_mm=z_mm,
            theta_rad=theta_rad,
            x_mm=points[:, 0],
            y_mm=points[:, 1],
            winding_angle_deg=previous_path.winding_angle_deg,
            tow_width_mm=previous_path.tow_width_mm,
            pass_index=np.zeros(sample_count, dtype=int),
        )
    )
    if tow_eye_angle.size:
        tow_eye_angle[0] = _path_b_angle(previous_path)[-1]
        tow_eye_angle[-1] = _path_b_angle(next_path)[0]
    return SurfacePath(
        z_mm=z_mm,
        theta_rad=theta_rad,
        x_mm=points[:, 0],
        y_mm=points[:, 1],
        winding_angle_deg=previous_path.winding_angle_deg,
        tow_width_mm=previous_path.tow_width_mm,
        pass_index=np.zeros(sample_count, dtype=int),
        tow_eye_angle_deg=tow_eye_angle,
    )


def _turnaround_boundary_z(
    *,
    start_z: float,
    safe_start_z_mm: float,
    safe_end_z_mm: float,
) -> float:
    if abs(start_z - safe_start_z_mm) <= abs(start_z - safe_end_z_mm):
        return safe_start_z_mm
    return safe_end_z_mm


def _wrapped_dome_turnaround_curve(
    previous_path: SurfacePath,
    next_path: SurfacePath,
    *,
    start_z: float,
    end_z: float,
    boundary_z: float,
    start_theta: float,
    point_count: int,
    min_z: float,
    max_z: float,
    wrap_angle_deg: float,
) -> tuple[FloatArray, FloatArray]:
    slope_start = _path_endpoint_dz_dtheta(previous_path, at_start=False)
    slope_end = _path_endpoint_dz_dtheta(next_path, at_start=True)
    target_end_theta = float(next_path.theta_rad[0])
    target_delta = _unwrap_delta(start_theta, target_end_theta)
    wrap_sign = _turnaround_wrap_sign(
        start_z=start_z,
        end_z=end_z,
        boundary_z=boundary_z,
        start_slope=slope_start,
        end_slope=slope_end,
        fallback_delta=target_delta,
    )
    wrap_angle_rad = math.radians(max(abs(wrap_angle_deg), 90.0)) * wrap_sign
    blend_angle_abs = max(
        math.radians(12.0),
        min(abs(wrap_angle_rad) * 0.25, math.radians(45.0)),
    )
    total_theta_delta = _turnaround_total_theta_delta(
        target_delta=target_delta,
        direction=wrap_sign,
        minimum_magnitude=abs(wrap_angle_rad) + 2.0 * blend_angle_abs,
    )
    blend_total = total_theta_delta - wrap_angle_rad
    first_theta_delta = blend_total * 0.5
    second_theta_delta = blend_total - first_theta_delta
    first_count = max(12, point_count // 4 + 1)
    wrap_count = max(9, point_count // 2 + 1)
    second_count = max(12, point_count - first_count - wrap_count + 4)
    t_first = np.linspace(0.0, 1.0, first_count)
    t_wrap = np.linspace(0.0, 1.0, wrap_count)
    t_second = np.linspace(0.0, 1.0, second_count)
    contact_start_theta = start_theta + first_theta_delta
    contact_end_theta = contact_start_theta + wrap_angle_rad
    end_theta = start_theta + total_theta_delta
    m0 = slope_start * first_theta_delta
    m1 = slope_end * second_theta_delta
    first_z = _cubic_hermite(start_z, boundary_z, m0, 0.0, t_first)
    second_z = _cubic_hermite(boundary_z, end_z, 0.0, m1, t_second)
    for _ in range(12):
        if (
            float(np.min(first_z)) >= min_z - 1e-9
            and float(np.max(first_z)) <= max_z + 1e-9
            and float(np.min(second_z)) >= min_z - 1e-9
            and float(np.max(second_z)) <= max_z + 1e-9
        ):
            break
        m0 *= 0.5
        m1 *= 0.5
        first_z = _cubic_hermite(start_z, boundary_z, m0, 0.0, t_first)
        second_z = _cubic_hermite(boundary_z, end_z, 0.0, m1, t_second)
    first_theta = np.linspace(start_theta, contact_start_theta, first_count)
    wrap_theta = contact_start_theta + wrap_angle_rad * t_wrap
    wrap_z = np.full(wrap_theta.shape, boundary_z, dtype=float)
    second_theta = np.linspace(contact_end_theta, end_theta, second_count)
    z_mm = np.concatenate((first_z, wrap_z[1:], second_z[1:]))
    theta_rad = np.concatenate((first_theta, wrap_theta[1:], second_theta[1:]))
    return np.clip(z_mm, min_z, max_z), theta_rad


def _turnaround_wrap_sign(
    *,
    start_z: float,
    end_z: float,
    boundary_z: float,
    start_slope: float,
    end_slope: float,
    fallback_delta: float,
) -> float:
    candidates = []
    if abs(start_slope) > 1e-9 and abs(boundary_z - start_z) > 1e-9:
        candidates.append(math.copysign(1.0, (boundary_z - start_z) / start_slope))
    if abs(end_slope) > 1e-9 and abs(end_z - boundary_z) > 1e-9:
        candidates.append(math.copysign(1.0, (end_z - boundary_z) / end_slope))
    if candidates:
        score = sum(candidates)
        if abs(score) > 1e-9:
            return math.copysign(1.0, score)
    if abs(fallback_delta) > 1e-9:
        return math.copysign(1.0, fallback_delta)
    return 1.0


def _turnaround_total_theta_delta(
    *,
    target_delta: float,
    direction: float,
    minimum_magnitude: float,
) -> float:
    direction = math.copysign(1.0, direction)
    best_delta: float | None = None
    best_score: tuple[float, float] | None = None
    for turns in range(-6, 7):
        candidate = target_delta + turns * 2.0 * math.pi
        if math.copysign(1.0, candidate if abs(candidate) > 1e-12 else direction) != direction:
            continue
        if abs(candidate) < minimum_magnitude:
            continue
        score = (abs(abs(candidate) - minimum_magnitude), abs(candidate))
        if best_score is None or score < best_score:
            best_delta = candidate
            best_score = score
    if best_delta is not None:
        return best_delta
    turns_needed = math.ceil((minimum_magnitude - abs(target_delta)) / (2.0 * math.pi))
    return target_delta + direction * max(1, turns_needed) * 2.0 * math.pi


def _cubic_hermite(
    start_value: float,
    end_value: float,
    start_tangent: float,
    end_tangent: float,
    t: FloatArray,
) -> FloatArray:
    h00 = 2.0 * t**3 - 3.0 * t**2 + 1.0
    h10 = t**3 - 2.0 * t**2 + t
    h01 = -2.0 * t**3 + 3.0 * t**2
    h11 = t**3 - t**2
    return (
        h00 * start_value
        + h10 * start_tangent
        + h01 * end_value
        + h11 * end_tangent
    )


def _path_endpoint_dz_dtheta(path: SurfacePath, *, at_start: bool) -> float:
    if path.point_count < 2:
        return 0.0
    if at_start:
        dz = float(path.z_mm[1] - path.z_mm[0])
        dtheta = _unwrap_delta(float(path.theta_rad[0]), float(path.theta_rad[1]))
    else:
        dz = float(path.z_mm[-1] - path.z_mm[-2])
        dtheta = _unwrap_delta(float(path.theta_rad[-2]), float(path.theta_rad[-1]))
    if abs(dtheta) <= 1e-9:
        return 0.0
    return dz / dtheta


def _densify_surface_path(
    mandrel: MandrelLike,
    path: SurfacePath,
    *,
    max_segment_length_mm: float,
) -> SurfacePath:
    if path.point_count < 2 or max_segment_length_mm <= 0.0:
        return path
    segment_lengths = np.linalg.norm(np.diff(path.points_mm, axis=0), axis=1)
    if not np.any(segment_lengths > max_segment_length_mm):
        return path
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total = float(cumulative[-1])
    if total <= 1e-9:
        return path
    sample_count = int(math.ceil(total / max_segment_length_mm)) + 1
    target = np.linspace(0.0, total, sample_count)
    theta_unwrapped = np.unwrap(path.theta_rad)
    z_mm = np.interp(target, cumulative, path.z_mm)
    theta_rad = np.interp(target, cumulative, theta_unwrapped)
    points = mandrel.surface_points(z_mm, theta_rad)
    tow_eye = (
        None
        if path.tow_eye_angle_deg is None
        else np.interp(target, cumulative, path.tow_eye_angle_deg)
    )
    pass_index = (
        None
        if path.pass_index is None
        else np.rint(np.interp(target, cumulative, path.pass_index)).astype(int)
    )
    return SurfacePath(
        z_mm=z_mm,
        theta_rad=theta_rad,
        x_mm=points[:, 0],
        y_mm=points[:, 1],
        winding_angle_deg=path.winding_angle_deg,
        tow_width_mm=path.tow_width_mm,
        pass_index=pass_index,
        tow_eye_angle_deg=tow_eye,
    )


def _concatenate_metadata(metadata: list[WindingPointMetadata]) -> WindingPointMetadata:
    if not metadata:
        raise ValueError("cannot concatenate empty metadata")
    return WindingPointMetadata(
        layer_id=tuple(label for item in metadata for label in item.layer_id),
        layer_index=np.concatenate([item.layer_index for item in metadata]),
        circuit_index=np.concatenate([item.circuit_index for item in metadata]),
        pass_index=np.concatenate([item.pass_index for item in metadata]),
        local_radius_mm=np.concatenate([item.local_radius_mm for item in metadata]),
        local_winding_angle_deg=np.concatenate([item.local_winding_angle_deg for item in metadata]),
        layer_name=tuple(label for item in metadata for label in item.layer_name),
        winding_type=tuple(label for item in metadata for label in item.winding_type),
        motion_type=tuple(label for item in metadata for label in item.motion_type),
        warning_flags=tuple(label for item in metadata for label in item.warning_flags),
    )


def _concatenate_motion_tables(tables: list[MachineMotionTable]) -> MachineMotionTable:
    if not tables:
        raise ValueError("cannot concatenate empty motion table list")
    a_chunks = []
    previous_end: float | None = None
    for table in tables:
        a_values = table.a_deg.copy()
        if previous_end is not None:
            delta = math.radians(float(a_values[0] - previous_end))
            offset = -math.degrees(((delta + math.pi) % (2.0 * math.pi)) - math.pi)
            a_values = a_values + previous_end - a_values[0] - offset
        previous_end = float(a_values[-1])
        a_chunks.append(a_values)
    return MachineMotionTable(
        a_deg=np.concatenate(a_chunks),
        x_mm=np.concatenate([table.x_mm for table in tables]),
        z_mm=np.concatenate([table.z_mm for table in tables]),
        b_deg=np.concatenate([table.b_deg for table in tables]),
    )


def _concatenate_feed_schedules(schedules: list[FeedSchedule]) -> FeedSchedule:
    if not schedules:
        raise ValueError("cannot concatenate empty feed schedule list")
    return FeedSchedule(
        feedrate_mm_min=np.concatenate([schedule.feedrate_mm_min for schedule in schedules]),
        curvature_1_per_mm=np.concatenate(
            [schedule.curvature_1_per_mm for schedule in schedules]
        ),
        curvature_radius_mm=np.concatenate(
            [schedule.curvature_radius_mm for schedule in schedules]
        ),
        slip_risk=np.concatenate([schedule.slip_risk for schedule in schedules]),
    )


def _smooth_motion_b_axis(
    motion_table: MachineMotionTable,
    path: SurfacePath,
) -> MachineMotionTable:
    b_deg = np.rad2deg(np.unwrap(np.deg2rad(motion_table.b_deg)))
    if path.point_count < 7 or np.allclose(b_deg, b_deg[0]):
        return motion_table
    smoothed = b_deg.copy()
    kernel = np.asarray([1.0, 2.0, 3.0, 2.0, 1.0], dtype=float)
    kernel /= float(np.sum(kernel))
    for _ in range(2):
        padded = np.pad(smoothed, (2, 2), mode="edge")
        smoothed = np.convolve(padded, kernel, mode="valid")
    smoothed[0] = b_deg[0]
    smoothed[-1] = b_deg[-1]
    return MachineMotionTable(
        a_deg=motion_table.a_deg,
        x_mm=motion_table.x_mm,
        z_mm=motion_table.z_mm,
        b_deg=smoothed,
    )


def _slow_for_b_axis_changes(
    feed_schedule: FeedSchedule,
    motion_table: MachineMotionTable,
    path: SurfacePath,
) -> FeedSchedule:
    if path.point_count < 2:
        return feed_schedule
    b_step = np.zeros(path.point_count, dtype=float)
    b_step[1:] = np.abs(np.diff(motion_table.b_deg))
    b_step[0] = b_step[1]
    transition_factor = np.ones(path.point_count, dtype=float)
    sharp = b_step > 1.0
    transition_factor[sharp] = np.clip(1.0 / b_step[sharp], 0.02, 1.0)
    feedrate = feed_schedule.feedrate_mm_min * transition_factor
    return FeedSchedule(
        feedrate_mm_min=feedrate,
        curvature_1_per_mm=feed_schedule.curvature_1_per_mm,
        curvature_radius_mm=feed_schedule.curvature_radius_mm,
        slip_risk=feed_schedule.slip_risk,
    )


def _with_generated_layer_id(spec: WindingLayerSpec, source_layer_index: int) -> WindingLayerSpec:
    if spec.layer_id.strip():
        return spec
    clean_name = "".join(
        char.lower() if char.isalnum() else "-" for char in spec.name.strip()
    ).strip("-")
    layer_id = clean_name or f"layer-{source_layer_index + 1}"
    return replace(spec, layer_id=f"{source_layer_index + 1:02d}-{layer_id}")


def _resolved_layer_id(spec: WindingLayerSpec, layer_index: int) -> str:
    if spec.layer_id.strip():
        return spec.layer_id
    clean_name = "".join(
        char.lower() if char.isalnum() else "-" for char in spec.name.strip()
    ).strip("-")
    return clean_name or f"layer-{layer_index + 1}"


def _mandrel_with_radius_offset(mandrel: MandrelLike, offset_mm: float) -> MandrelLike:
    if offset_mm <= 0.0:
        return mandrel
    if isinstance(mandrel, CylinderMandrel):
        return CylinderMandrel(
            length_mm=mandrel.length_mm,
            radius_mm=mandrel.radius_mm + offset_mm,
            name=mandrel.name,
        )
    return AxisymmetricProfileMandrel(
        z_mm=mandrel.z_mm,
        r_mm=mandrel.r_mm + offset_mm,
        name=mandrel.name,
    )


def _layer_feedrate_mm_min(spec: WindingLayerSpec, schedule: WindingSchedule) -> float:
    return (
        schedule.nominal_feedrate_mm_min
        if spec.feedrate_mm_min is None
        else spec.feedrate_mm_min
    )


def _layer_clearance_mm(spec: WindingLayerSpec, schedule: WindingSchedule) -> float:
    return (
        schedule.radial_clearance_mm
        if spec.mandrel_clearance_mm is None
        else spec.mandrel_clearance_mm
    )


def _path_b_angle(path: SurfacePath) -> FloatArray:
    if path.tow_eye_angle_deg is not None:
        return path.tow_eye_angle_deg
    return np.full(path.point_count, path.winding_angle_deg, dtype=float)


def _pattern_warnings(
    *,
    target_angle: float,
    actual_angle: float,
    gap_mm: float,
    overlap_mm: float,
    max_angle_error: float,
) -> tuple[str, ...]:
    warnings = []
    angle_error = abs(target_angle - actual_angle)
    if angle_error > max_angle_error:
        warnings.append(f"angle error {angle_error:.3f} deg")
    if gap_mm > 0.25:
        warnings.append(f"tow gap {gap_mm:.3f} mm")
    if overlap_mm > 0.25:
        warnings.append(f"tow overlap {overlap_mm:.3f} mm")
    return tuple(warnings)


def _signed_angle(direction: LayerDirection, angle: float) -> float:
    if direction in {"negative", "polar"}:
        return -abs(angle)
    return abs(angle)


def _z_bounds(mandrel: MandrelLike) -> tuple[float, float]:
    if isinstance(mandrel, CylinderMandrel):
        return 0.0, mandrel.length_mm
    return mandrel.start_z_mm, mandrel.end_z_mm


def _max_radius(mandrel: MandrelLike) -> float:
    if isinstance(mandrel, CylinderMandrel):
        return mandrel.radius_mm
    return mandrel.max_radius_mm


def _profile_turnaround_min_radius(
    mandrel: AxisymmetricProfileMandrel,
    spec: WindingLayerSpec,
) -> float:
    if spec.turnaround_radius_mm is not None:
        return spec.turnaround_radius_mm
    positive_radius = mandrel.r_mm[mandrel.r_mm > 1e-9]
    if positive_radius.size == 0:
        raise ValueError("axisymmetric profile has no positive-radius winding zone")
    return max(1.0, float(np.min(positive_radius)))


def _axisymmetric_turnaround_z_bounds(
    mandrel: AxisymmetricProfileMandrel,
    spec: WindingLayerSpec,
) -> tuple[float, float, float, float]:
    min_radius = _axisymmetric_safe_turnaround_radius(mandrel, spec)
    safe_zone = find_profile_safe_zone(mandrel, min_radius_mm=min_radius)
    requested_start_z = safe_zone.start_z_mm if spec.start_z_mm is None else spec.start_z_mm
    requested_end_z = safe_zone.end_z_mm if spec.end_z_mm is None else spec.end_z_mm
    start_z = max(requested_start_z, safe_zone.start_z_mm)
    end_z = min(requested_end_z, safe_zone.end_z_mm)
    edge_ease_mm = min(
        max(spec.tow_width_mm * 2.0, 3.0),
        max((safe_zone.end_z_mm - safe_zone.start_z_mm) * 0.12, 0.0),
    )
    if spec.start_z_mm is None:
        start_z += edge_ease_mm
    if spec.end_z_mm is None:
        end_z -= edge_ease_mm
    if end_z <= start_z:
        raise ValueError("axisymmetric winding zone has invalid z bounds")
    return start_z, end_z, safe_zone.start_z_mm, safe_zone.end_z_mm


def _tow_edge_clearance_mm(spec: WindingLayerSpec) -> float:
    return spec.tow_width_mm * 0.5 + spec.layer_thickness_mm * 0.5


def _axisymmetric_safe_turnaround_radius(
    mandrel: AxisymmetricProfileMandrel,
    spec: WindingLayerSpec,
) -> float:
    min_radius = _profile_turnaround_min_radius(mandrel, spec) + _tow_edge_clearance_mm(spec)
    if spec.winding_type != "geodesic":
        return min_radius
    clairaut_radius = mandrel.max_radius_mm * math.sin(math.radians(abs(spec.target_angle_deg)))
    return max(min_radius, clairaut_radius)


def _axisymmetric_reference_z(
    mandrel: AxisymmetricProfileMandrel,
    start_z: float,
    end_z: float,
) -> float:
    z_values = np.asarray(mandrel.z_mm, dtype=float)
    radius_values = np.asarray(mandrel.r_mm, dtype=float)
    mask = (z_values >= start_z) & (z_values <= end_z)
    if not np.any(mask):
        return (start_z + end_z) * 0.5
    local_z = z_values[mask]
    local_radius = radius_values[mask]
    max_radius = float(np.max(local_radius))
    plateau = local_z[local_radius >= max_radius - max(1e-6, max_radius * 1e-5)]
    if plateau.size:
        return float((np.min(plateau) + np.max(plateau)) * 0.5)
    return float(local_z[int(np.argmax(local_radius))])


def _path_theta_at_z(path: SurfacePath, z_mm: float) -> float:
    theta_unwrapped = np.unwrap(np.asarray(path.theta_rad, dtype=float))
    order = np.argsort(path.z_mm)
    z_sorted = np.asarray(path.z_mm[order], dtype=float)
    theta_sorted = theta_unwrapped[order]
    unique_z, unique_indices = np.unique(z_sorted, return_index=True)
    if unique_z.size < 2:
        return float(theta_sorted[0])
    unique_theta = theta_sorted[unique_indices]
    return float(np.interp(z_mm, unique_z, unique_theta))


def _axisymmetric_pass_count(
    mandrel: AxisymmetricProfileMandrel,
    spec: WindingLayerSpec,
) -> int:
    circumference = 2.0 * math.pi * max(mandrel.max_radius_mm, 1e-9)
    lanes_per_direction = max(
        1,
        math.ceil(circumference * spec.coverage_target / spec.tow_width_mm),
    )
    coverage_passes = lanes_per_direction * 2
    if spec.number_of_passes is not None:
        return spec.number_of_passes
    return coverage_passes


def _axisymmetric_max_segment_length(spec: WindingLayerSpec) -> float:
    return max(spec.tow_width_mm * 1.25, 8.0)


def _even_axisymmetric_lane_count(
    mandrel: AxisymmetricProfileMandrel,
    spec: WindingLayerSpec,
) -> int:
    lane_count = max(1, _axisymmetric_pass_count(mandrel, spec))
    if lane_count % 2:
        lane_count += 1
    return max(2, lane_count)


def _axisymmetric_report(
    mandrel: AxisymmetricProfileMandrel,
    spec: WindingLayerSpec,
    path: SurfacePath,
    *,
    circuits: int,
    winding_lanes: int | None = None,
    warnings: tuple[str, ...],
) -> WindingPatternReport:
    effective_lanes = max(winding_lanes or circuits, 1)
    circumference = 2.0 * math.pi * max(mandrel.max_radius_mm, 1e-9)
    tow_spacing = circumference / effective_lanes
    gap_mm = max(tow_spacing - spec.tow_width_mm, 0.0)
    overlap_mm = max(spec.tow_width_mm - tow_spacing, 0.0)
    all_warnings = tuple(sorted(set(warnings + _pattern_warnings(
        target_angle=abs(spec.target_angle_deg),
        actual_angle=abs(spec.target_angle_deg),
        gap_mm=gap_mm,
        overlap_mm=overlap_mm,
        max_angle_error=spec.max_angle_error_deg,
    ))))
    return WindingPatternReport(
        layer_id=_resolved_layer_id(spec, 0),
        layer_name=spec.name,
        winding_type=spec.winding_type,
        target_angle_deg=spec.target_angle_deg,
        actual_angle_deg=_signed_angle(spec.direction, abs(spec.target_angle_deg)),
        angle_error_deg=0.0,
        circuits=effective_lanes,
        starts=circuits,
        angular_shift_deg=360.0 / effective_lanes,
        tow_spacing_mm=tow_spacing,
        coverage_percent=spec.tow_width_mm / tow_spacing * 100.0,
        gap_mm=gap_mm,
        overlap_mm=overlap_mm,
        layer_completion_z_mm=float(path.z_mm[-1]),
        pattern_repeat_length_mm=mandrel.length_mm,
        closes=True,
        acceptable=not all_warnings,
        warnings=all_warnings,
    )


def _with_pass_index(path: SurfacePath, pass_number: int) -> SurfacePath:
    return SurfacePath(
        z_mm=path.z_mm,
        theta_rad=path.theta_rad,
        x_mm=path.x_mm,
        y_mm=path.y_mm,
        winding_angle_deg=path.winding_angle_deg,
        tow_width_mm=path.tow_width_mm,
        pass_index=np.full(path.z_mm.shape, pass_number, dtype=int),
        tow_eye_angle_deg=_path_b_angle(path),
    )


def _wrap_periodic(values: FloatArray, period: FloatArray) -> FloatArray:
    safe_period = np.maximum(period, 1e-9)
    return ((values + safe_period / 2.0) % safe_period) - safe_period / 2.0


def _contiguous_pass_spans(pass_index: IntArray) -> tuple[tuple[int, int], ...]:
    if pass_index.size == 0:
        return ()
    spans = []
    start = 0
    for index in range(1, pass_index.size):
        if pass_index[index] != pass_index[index - 1]:
            spans.append((start, index))
            start = index
    spans.append((start, pass_index.size))
    return tuple(spans)
