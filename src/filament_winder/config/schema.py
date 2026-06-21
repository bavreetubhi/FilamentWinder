"""Typed config objects for CLI-first winding generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    name: str = "winding_project"
    units: str = "mm"


@dataclass(frozen=True, slots=True)
class MachineConfig:
    clearance_mm: float = 25.0
    axis_order: tuple[str, ...] = ("A", "X", "Z", "B")
    mandrel_orientation: str = "horizontal"
    controller: str = "generic"
    max_a_rpm: float | None = None
    max_x_mm: float | None = None
    max_z_mm: float | None = None
    max_b_deg: float | None = None
    max_b_velocity_deg_s: float | None = None
    max_segment_length_mm: float | None = None
    max_a_accel_deg_s2: float | None = None
    max_x_accel_mm_s2: float | None = None
    max_z_accel_mm_s2: float | None = None
    max_b_accel_deg_s2: float | None = None


@dataclass(frozen=True, slots=True)
class MandrelConfig:
    type: str = "cylinder"
    length_mm: float = 1000.0
    radius_mm: float = 100.0
    cylinder_length_mm: float | None = None
    cylinder_radius_mm: float | None = None
    left_dome_length_mm: float = 0.0
    right_dome_length_mm: float = 0.0
    polar_opening_radius_mm: float = 0.0
    profile_path: Path | None = None
    samples: int | None = None
    mesh_points_z: int = 240
    mesh_points_theta: int = 180


@dataclass(frozen=True, slots=True)
class TowConfig:
    tow_id: str = "tow"
    name: str = "tow"
    width_mm: float = 6.0
    thickness_mm: float = 0.0
    effective_width_mm: float | None = None
    calibrated_effective_width: bool = False
    min_bend_radius_mm: float | None = None
    tension_N: float | None = None
    friction_coefficient: float | None = None
    calibrated_friction: bool = False
    fibre_type: str = ""
    resin_system: str = ""
    notes: str = ""


@dataclass(frozen=True, slots=True)
class RovingConfig:
    width_mm: float = 6.0
    thickness_mm: float = 0.25
    fiber_volume_fraction: float = 0.5
    resin_factor: float | str = "auto"


@dataclass(frozen=True, slots=True)
class PatternSelectionConfig:
    method: str = "legacy_coverage"
    max_p: int = 500
    max_k: int = 500
    max_d: int = 20
    angle_tolerance_deg: float = 0.5
    require_gcd_clean_pattern: bool = True
    candidate_count: int = 10


@dataclass(frozen=True, slots=True)
class LaminateTargetsConfig:
    mode: str = "simplified"
    target_layer_thickness_mm: float | None = None
    target_number_of_closed_layers: int | None = None
    target_total_thickness_mm: float | None = None
    required_thickness_profile: str | None = None


@dataclass(frozen=True, slots=True)
class PatternObjectivesConfig:
    minimise_winding_time: bool = True
    minimise_overlap: bool = True
    minimise_polar_buildup: bool = True
    require_closed_pattern: bool = True


@dataclass(frozen=True, slots=True)
class HoopWindingConfig:
    mode: str = "continuous_traverse"
    nominal_angle_deg: float = 89.0
    min_angle_offset_from_pure_hoop_deg: float = 0.25
    axial_pitch_mode: str = "tow_width"
    allow_exact_pure_hoop: bool = False
    tow_state_during_traverse: str = "on"


@dataclass(frozen=True, slots=True)
class QualityLimitsConfig:
    max_layer_overlap_percent: float = 35.0
    max_stack_overlap_percent: float = 45.0
    max_thickness_variation_percent: float = 75.0
    max_polar_buildup_mm: float = 0.75
    max_coverage_count: int = 20
    allow_min_thickness_zero: bool = False
    max_estimated_winding_time_min: float = 600.0


@dataclass(frozen=True, slots=True)
class CoverageModeConfig:
    individual_layer_full_coverage: bool = False
    stack_level_full_coverage: bool = True
    paired_layer_coverage: bool = True


@dataclass(frozen=True, slots=True)
class PinLayoutConfig:
    enabled: bool = False
    layout_type: str = "shoulder_cross"
    shoulders: str = "both"
    count_per_shoulder: int = 4
    angular_offset_deg: float = 0.0
    left_shoulder_z_mm: float | None = None
    right_shoulder_z_mm: float | None = None
    shoulder_zone_width_mm: float = 60.0
    pin_radius_mm: float = 4.0
    pin_height_mm: float = 25.0
    pin_standoff_mm: float = 2.0
    pin_clearance_mm: float = 0.5
    min_wrap_deg: float = 120.0
    max_wrap_deg: float = 270.0
    max_buildup_height_mm: float = 8.0
    max_contact_balance_ratio: float = 1.25
    friction_coefficient: float | None = None
    min_bend_radius_mm: float | None = None
    route_family: str = "shoulder_cross_reinforcement"
    routing_mode: str = "deterministic"
    candidate_count: int = 192
    route_step_size: int = 0
    wrap_direction: str = "both"
    target_dome_angle_min_deg: float = 25.0
    target_dome_angle_max_deg: float = 55.0
    coverage_tolerance_mm: float = 6.0


@dataclass(frozen=True, slots=True)
class LayerConfig:
    name: str
    type: str
    winding_angle_deg: float
    enabled: bool = True
    ply_order: int | None = None
    material: str = ""
    region: str = "full_mandrel"
    winding_mode: str | None = None
    initial_angle_deg: float | None = None
    target_angle_deg: float | None = None
    angle_tolerance_deg: float = 0.5
    direction: str = "forward"
    passes: int | str | None = "auto"
    coverage_target: float = 1.0
    turnaround_radius_mm: float | None = None
    polar_opening_radius_mm: float | None = None
    tow_width_mm: float | None = None
    tow_thickness_mm: float | None = None
    feedrate_mm_min: float | None = None
    start_z_mm: float | None = None
    end_z_mm: float | None = None
    transition_before: bool = True
    transition_after: bool = True
    phase_offset_deg: float | None = None
    colour: str = "#1e90ff"
    notes: str = ""
    points: int = 500


@dataclass(frozen=True, slots=True)
class OutputConfig:
    directory: Path = Path("exports/winding_job")
    csv: bool = True
    summary_json: bool = True
    segments_json: bool = True
    validation_report_json: bool = True
    coverage_grid: bool = True
    gcode: bool = False


@dataclass(frozen=True, slots=True)
class CoverageConfig:
    z_cells: int = 120
    theta_cells: int = 180
    tow_band_model: str = "centerline_projected"


@dataclass(frozen=True, slots=True)
class PlotConfig:
    enabled: bool = True
    show: bool = False
    save: bool = True
    formats: tuple[str, ...] = ("png",)
    modes: tuple[str, ...] = ("unwrapped", "three_d")
    include_2d_unwrapped: bool = True
    include_3d_path: bool = True


@dataclass(frozen=True, slots=True)
class WindingJobConfig:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    machine: MachineConfig = field(default_factory=MachineConfig)
    mandrel: MandrelConfig = field(default_factory=MandrelConfig)
    tow: TowConfig = field(default_factory=TowConfig)
    roving: RovingConfig = field(default_factory=RovingConfig)
    layers: tuple[LayerConfig, ...] = ()
    pattern_selection: PatternSelectionConfig = field(default_factory=PatternSelectionConfig)
    laminate_targets: LaminateTargetsConfig = field(default_factory=LaminateTargetsConfig)
    pattern_objectives: PatternObjectivesConfig = field(default_factory=PatternObjectivesConfig)
    hoop_winding: HoopWindingConfig = field(default_factory=HoopWindingConfig)
    quality_limits: QualityLimitsConfig = field(default_factory=QualityLimitsConfig)
    coverage_mode: CoverageModeConfig = field(default_factory=CoverageModeConfig)
    pin_layout: PinLayoutConfig = field(default_factory=PinLayoutConfig)
    coverage: CoverageConfig = field(default_factory=CoverageConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    plot: PlotConfig = field(default_factory=PlotConfig)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> WindingJobConfig:
        layers_raw = data.get("layers", ())
        if not isinstance(layers_raw, list):
            raise ValueError("layers must be a list")
        return cls(
            project=_project_config(data.get("project", {})),
            machine=_machine_config(data.get("machine", {})),
            mandrel=_mandrel_config(data.get("mandrel", {})),
            tow=_tow_config(data.get("tow", {})),
            roving=_roving_config(data.get("roving", data.get("tow", {}))),
            layers=tuple(_layer_config(index, item) for index, item in enumerate(layers_raw)),
            pattern_selection=_pattern_selection_config(data.get("pattern_selection", {})),
            laminate_targets=_laminate_targets_config(data.get("laminate_targets", {})),
            pattern_objectives=_pattern_objectives_config(data.get("pattern_objectives", {})),
            hoop_winding=_hoop_winding_config(data.get("hoop_winding", {})),
            quality_limits=_quality_limits_config(data.get("quality_limits", {})),
            coverage_mode=_coverage_mode_config(data.get("coverage_mode", {})),
            pin_layout=_pin_layout_config(data.get("pin_layout", {})),
            coverage=_coverage_config(data.get("coverage", {})),
            output=_output_config(data.get("output", {})),
            plot=_plot_config(data.get("plot", {})),
        )


def _project_config(raw: object) -> ProjectConfig:
    data = _mapping(raw, "project")
    return ProjectConfig(
        name=str(data.get("name", "winding_project")),
        units=str(data.get("units", "mm")),
    )


def _machine_config(raw: object) -> MachineConfig:
    data = _mapping(raw, "machine")
    return MachineConfig(
        clearance_mm=float(data.get("clearance_mm", data.get("clearance", 25.0))),
        axis_order=tuple(str(item) for item in data.get("axis_order", ("A", "X", "Z", "B"))),
        mandrel_orientation=str(data.get("mandrel_orientation", "horizontal")),
        controller=str(data.get("controller", "generic")),
        max_a_rpm=_optional_float(data.get("max_a_rpm")),
        max_x_mm=_optional_float(data.get("max_x_mm")),
        max_z_mm=_optional_float(data.get("max_z_mm")),
        max_b_deg=_optional_float(data.get("max_b_deg")),
        max_b_velocity_deg_s=_optional_float(data.get("max_b_velocity_deg_s")),
        max_segment_length_mm=_optional_float(data.get("max_segment_length_mm")),
        max_a_accel_deg_s2=_optional_float(data.get("max_a_accel_deg_s2")),
        max_x_accel_mm_s2=_optional_float(data.get("max_x_accel_mm_s2")),
        max_z_accel_mm_s2=_optional_float(data.get("max_z_accel_mm_s2")),
        max_b_accel_deg_s2=_optional_float(data.get("max_b_accel_deg_s2")),
    )


def _mandrel_config(raw: object) -> MandrelConfig:
    data = _mapping(raw, "mandrel")
    cylinder_length = _optional_float(data.get("cylinder_length_mm"))
    cylinder_radius = _optional_float(data.get("cylinder_radius_mm"))
    return MandrelConfig(
        type=str(data.get("type", "cylinder")),
        length_mm=float(data.get("length_mm", cylinder_length or 1000.0)),
        radius_mm=float(data.get("radius_mm", cylinder_radius or 100.0)),
        cylinder_length_mm=cylinder_length,
        cylinder_radius_mm=cylinder_radius,
        left_dome_length_mm=float(data.get("left_dome_length_mm", 0.0)),
        right_dome_length_mm=float(data.get("right_dome_length_mm", 0.0)),
        polar_opening_radius_mm=float(data.get("polar_opening_radius_mm", 0.0)),
        profile_path=(
            None
            if data.get("profile_path") in {None, ""}
            else Path(str(data.get("profile_path")))
        ),
        samples=None if data.get("samples") in {None, "", 0} else int(data.get("samples", 0)),
        mesh_points_z=int(data.get("mesh_points_z", 240)),
        mesh_points_theta=int(data.get("mesh_points_theta", 180)),
    )


def _tow_config(raw: object) -> TowConfig:
    data = _mapping(raw, "tow")
    return TowConfig(
        tow_id=str(data.get("tow_id", data.get("id", "tow"))),
        name=str(data.get("name", "tow")),
        width_mm=float(data.get("width_mm", 6.0)),
        thickness_mm=float(data.get("thickness_mm", 0.0)),
        effective_width_mm=_optional_float(data.get("effective_width_mm")),
        calibrated_effective_width=bool(data.get("calibrated_effective_width", False)),
        min_bend_radius_mm=_optional_float(data.get("min_bend_radius_mm")),
        tension_N=_optional_float(data.get("tension_N", data.get("tension_n"))),
        friction_coefficient=_optional_float(data.get("friction_coefficient")),
        calibrated_friction=bool(data.get("calibrated_friction", False)),
        fibre_type=str(data.get("fibre_type", "")),
        resin_system=str(data.get("resin_system", "")),
        notes=str(data.get("notes", "")),
    )


def _roving_config(raw: object) -> RovingConfig:
    data = _mapping(raw, "roving")
    resin_factor = data.get("resin_factor", "auto")
    return RovingConfig(
        width_mm=float(data.get("width_mm", 6.0)),
        thickness_mm=float(data.get("thickness_mm", 0.25)),
        fiber_volume_fraction=float(data.get("fiber_volume_fraction", 0.5)),
        resin_factor=(
            "auto"
            if resin_factor in {None, "", "auto"}
            else float(str(resin_factor))
        ),
    )


def _pattern_selection_config(raw: object) -> PatternSelectionConfig:
    data = _mapping(raw, "pattern_selection")
    return PatternSelectionConfig(
        method=str(data.get("method", "legacy_coverage")),
        max_p=int(data.get("max_p", 500)),
        max_k=int(data.get("max_k", 500)),
        max_d=int(data.get("max_d", 20)),
        angle_tolerance_deg=float(data.get("angle_tolerance_deg", 0.5)),
        require_gcd_clean_pattern=bool(data.get("require_gcd_clean_pattern", True)),
        candidate_count=int(data.get("candidate_count", 10)),
    )


def _laminate_targets_config(raw: object) -> LaminateTargetsConfig:
    data = _mapping(raw, "laminate_targets")
    return LaminateTargetsConfig(
        mode=str(data.get("mode", "simplified")),
        target_layer_thickness_mm=_optional_float(data.get("target_layer_thickness_mm")),
        target_number_of_closed_layers=(
            None
            if data.get("target_number_of_closed_layers") in {None, "", "auto"}
            else int(str(data["target_number_of_closed_layers"]))
        ),
        target_total_thickness_mm=_optional_float(data.get("target_total_thickness_mm")),
        required_thickness_profile=(
            None
            if data.get("required_thickness_profile") in {None, ""}
            else str(data.get("required_thickness_profile"))
        ),
    )


def _pattern_objectives_config(raw: object) -> PatternObjectivesConfig:
    data = _mapping(raw, "pattern_objectives")
    return PatternObjectivesConfig(
        minimise_winding_time=bool(data.get("minimise_winding_time", True)),
        minimise_overlap=bool(data.get("minimise_overlap", True)),
        minimise_polar_buildup=bool(data.get("minimise_polar_buildup", True)),
        require_closed_pattern=bool(data.get("require_closed_pattern", True)),
    )


def _hoop_winding_config(raw: object) -> HoopWindingConfig:
    data = _mapping(raw, "hoop_winding")
    return HoopWindingConfig(
        mode=str(data.get("mode", "continuous_traverse")),
        nominal_angle_deg=float(data.get("nominal_angle_deg", 89.0)),
        min_angle_offset_from_pure_hoop_deg=float(
            data.get("min_angle_offset_from_pure_hoop_deg", 0.25)
        ),
        axial_pitch_mode=str(data.get("axial_pitch_mode", "tow_width")),
        allow_exact_pure_hoop=bool(data.get("allow_exact_pure_hoop", False)),
        tow_state_during_traverse=str(data.get("tow_state_during_traverse", "on")),
    )


def _quality_limits_config(raw: object) -> QualityLimitsConfig:
    data = _mapping(raw, "quality_limits")
    return QualityLimitsConfig(
        max_layer_overlap_percent=float(data.get("max_layer_overlap_percent", 35.0)),
        max_stack_overlap_percent=float(data.get("max_stack_overlap_percent", 45.0)),
        max_thickness_variation_percent=float(
            data.get("max_thickness_variation_percent", 75.0)
        ),
        max_polar_buildup_mm=float(data.get("max_polar_buildup_mm", 0.75)),
        max_coverage_count=int(data.get("max_coverage_count", 20)),
        allow_min_thickness_zero=bool(data.get("allow_min_thickness_zero", False)),
        max_estimated_winding_time_min=float(
            data.get("max_estimated_winding_time_min", 600.0)
        ),
    )


def _coverage_mode_config(raw: object) -> CoverageModeConfig:
    data = _mapping(raw, "coverage_mode")
    return CoverageModeConfig(
        individual_layer_full_coverage=bool(
            data.get("individual_layer_full_coverage", False)
        ),
        stack_level_full_coverage=bool(data.get("stack_level_full_coverage", True)),
        paired_layer_coverage=bool(data.get("paired_layer_coverage", True)),
    )


def _pin_layout_config(raw: object) -> PinLayoutConfig:
    data = _mapping(raw, "pin_layout")
    return PinLayoutConfig(
        enabled=bool(data.get("enabled", False)),
        layout_type=str(data.get("layout_type", data.get("type", "shoulder_cross"))),
        shoulders=str(data.get("shoulders", "both")),
        count_per_shoulder=int(data.get("count_per_shoulder", data.get("pin_count", 4))),
        angular_offset_deg=float(data.get("angular_offset_deg", 0.0)),
        left_shoulder_z_mm=_optional_float(data.get("left_shoulder_z_mm")),
        right_shoulder_z_mm=_optional_float(data.get("right_shoulder_z_mm")),
        shoulder_zone_width_mm=float(data.get("shoulder_zone_width_mm", 60.0)),
        pin_radius_mm=float(data.get("pin_radius_mm", data.get("radius_mm", 4.0))),
        pin_height_mm=float(data.get("pin_height_mm", data.get("height_mm", 25.0))),
        pin_standoff_mm=float(data.get("pin_standoff_mm", data.get("standoff_mm", 2.0))),
        pin_clearance_mm=float(data.get("pin_clearance_mm", data.get("clearance_mm", 0.5))),
        min_wrap_deg=float(data.get("min_wrap_deg", 120.0)),
        max_wrap_deg=float(data.get("max_wrap_deg", 270.0)),
        max_buildup_height_mm=float(data.get("max_buildup_height_mm", 8.0)),
        max_contact_balance_ratio=float(data.get("max_contact_balance_ratio", 1.25)),
        friction_coefficient=_optional_float(
            data.get("friction_coefficient", data.get("friction_mu_pin"))
        ),
        min_bend_radius_mm=_optional_float(data.get("min_bend_radius_mm")),
        route_family=str(data.get("route_family", "shoulder_cross_reinforcement")),
        routing_mode=str(data.get("routing_mode", "deterministic")),
        candidate_count=int(data.get("candidate_count", data.get("candidate_limit", 192))),
        route_step_size=int(data.get("route_step_size", data.get("step_size", 0))),
        wrap_direction=str(data.get("wrap_direction", "both")),
        target_dome_angle_min_deg=float(data.get("target_dome_angle_min_deg", 25.0)),
        target_dome_angle_max_deg=float(data.get("target_dome_angle_max_deg", 55.0)),
        coverage_tolerance_mm=float(data.get("coverage_tolerance_mm", 6.0)),
    )


def _layer_config(index: int, raw: object) -> LayerConfig:
    data = _mapping(raw, f"layers[{index}]")
    ply_order_raw = data.get("ply_order")
    ply_order = None if ply_order_raw in {None, ""} else int(float(str(ply_order_raw)))
    return LayerConfig(
        name=str(data.get("name", f"layer_{index + 1}")),
        type=str(data.get("type", data.get("winding_mode", data.get("winding_type", "helical")))),
        winding_angle_deg=float(
            data.get("winding_angle_deg", data.get("angle_deg", data.get("target_angle_deg", 45.0)))
        ),
        enabled=bool(data.get("enabled", True)),
        ply_order=ply_order,
        material=str(data.get("material", data.get("material_name", ""))),
        region=str(data.get("region", "full_mandrel")),
        winding_mode=(
            None if data.get("winding_mode") is None else str(data.get("winding_mode"))
        ),
        initial_angle_deg=_optional_float(data.get("initial_angle_deg")),
        target_angle_deg=_optional_float(data.get("target_angle_deg")),
        angle_tolerance_deg=float(data.get("angle_tolerance_deg", 0.5)),
        direction=str(data.get("direction", "forward")),
        passes=data.get("passes", "auto"),
        coverage_target=float(data.get("coverage_target", 1.0)),
        turnaround_radius_mm=_optional_float(data.get("turnaround_radius_mm")),
        polar_opening_radius_mm=_optional_float(data.get("polar_opening_radius_mm")),
        tow_width_mm=_optional_float(data.get("tow_width_mm")),
        tow_thickness_mm=_optional_float(data.get("tow_thickness_mm")),
        feedrate_mm_min=_optional_float(data.get("feedrate_mm_min")),
        start_z_mm=_optional_float(data.get("start_z_mm")),
        end_z_mm=_optional_float(data.get("end_z_mm")),
        transition_before=_transition_enabled(data.get("transition_before", True)),
        transition_after=_transition_enabled(data.get("transition_after", True)),
        phase_offset_deg=_optional_float(data.get("phase_offset_deg")),
        colour=str(data.get("colour", data.get("color", "#1e90ff"))),
        notes=str(data.get("notes", "")),
        points=int(data.get("points", data.get("point_count", 500))),
    )


def _output_config(raw: object) -> OutputConfig:
    data = _mapping(raw, "output")
    return OutputConfig(
        directory=Path(str(data.get("directory", "exports/winding_job"))),
        csv=bool(data.get("csv", True)),
        summary_json=bool(data.get("summary_json", True)),
        segments_json=bool(data.get("segments_json", data.get("segments", True))),
        validation_report_json=bool(
            data.get("validation_report_json", data.get("validation_report", True))
        ),
        coverage_grid=bool(data.get("coverage_grid", True)),
        gcode=bool(data.get("gcode", False)),
    )


def _coverage_config(raw: object) -> CoverageConfig:
    data = _mapping(raw, "coverage")
    return CoverageConfig(
        z_cells=int(data.get("z_cells", 120)),
        theta_cells=int(data.get("theta_cells", 180)),
        tow_band_model=str(data.get("tow_band_model", "centerline_projected")),
    )


def _plot_config(raw: object) -> PlotConfig:
    data = _mapping(raw, "plot")
    modes_raw = data.get("modes")
    if modes_raw is None:
        modes = tuple(
            mode
            for mode, enabled in (
                ("unwrapped", bool(data.get("include_2d_unwrapped", True))),
                ("three_d", bool(data.get("include_3d_path", True))),
            )
            if enabled
        )
    else:
        modes = tuple(str(item) for item in modes_raw)
    return PlotConfig(
        enabled=bool(data.get("enabled", True)),
        show=bool(data.get("show", False)),
        save=bool(data.get("save", True)),
        formats=tuple(str(item) for item in data.get("formats", ("png",))),
        modes=modes,
        include_2d_unwrapped=bool(data.get("include_2d_unwrapped", True)),
        include_3d_path=bool(data.get("include_3d_path", True)),
    )


def _mapping(raw: object, name: str) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"{name} must be a mapping")
    return raw


def _optional_float(raw: object) -> float | None:
    if raw in {None, "", "auto"}:
        return None
    return float(str(raw))


def _transition_enabled(raw: object) -> bool:
    if isinstance(raw, str):
        return raw.lower() not in {"", "none", "false", "off", "no"}
    return bool(raw)
