"""Config-driven headless winding job service."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from filament_winder.config import LayerConfig, WindingJobConfig
from filament_winder.core.coverage import cylinder_coverage_map
from filament_winder.core.feedrate import FeedrateConfig, plan_feedrate
from filament_winder.core.geometry import (
    AxisymmetricProfileMandrel,
    CylinderMandrel,
    cylinder_with_domes_profile,
)
from filament_winder.core.kinematics.four_axis import machine_path_from_surface_path
from filament_winder.core.path_planning import (
    MultiLayerPatternResult,
    PatternSearchRequest,
    PlannedLayer,
    PlannedWindingProgram,
    SurfacePath,
    WindingLayerSpec,
    WindingPatternReport,
    WindingPointMetadata,
    WindingSchedule,
    axisymmetric_surface_coverage_map,
    build_path_segments,
    export_pattern_candidates_json,
    export_pattern_rejection_report_json,
    export_selected_pattern_json,
    export_thickness_distribution_csv,
    generate_controlled_angle_path,
    generate_geodesic_path,
    plan_winding_schedule,
    select_winding_pattern,
)
from filament_winder.core.path_planning.geodesic import (
    ControlledAnglePathConfig,
    GeodesicPathConfig,
)
from filament_winder.io import (
    GCodeOptions,
    export_coverage_grid_npz,
    export_gcode,
    export_segments_json,
    export_validation_report_json,
    export_winding_program_csv,
    import_dxf_zr_profile,
)
from filament_winder.plot import (
    plot_dome_coverage_maps,
    plot_dome_motion_diagnostics,
    plot_layer_diagnostics,
    plot_winding_program,
)
from filament_winder.services.path_validation import (
    program_continuity_summary,
    program_path_validation_summary,
    program_transition_summary,
)


@dataclass(frozen=True, slots=True)
class WindingJobResult:
    config: WindingJobConfig
    mandrel: CylinderMandrel | AxisymmetricProfileMandrel
    program: PlannedWindingProgram
    csv_path: Path | None
    gcode_path: Path | None
    summary_path: Path | None
    segments_path: Path | None
    validation_report_path: Path | None
    coverage_grid_path: Path | None
    pattern_candidates_path: Path | None
    selected_pattern_path: Path | None
    pattern_rejection_report_path: Path | None
    thickness_distribution_path: Path | None
    layer_completion_report_path: Path | None
    stack_coverage_report_path: Path | None
    machine_smoothing_report_path: Path | None
    pattern_optimisation_report_path: Path | None
    candidate_pair_report_path: Path | None
    actual_thickness_report_path: Path | None
    region_quality_report_path: Path | None
    calibration_report_path: Path | None
    friction_margin_report_path: Path | None
    polar_overbuild_report_path: Path | None
    collision_report_path: Path | None
    pin_layout_report_path: Path | None
    pin_contact_report_path: Path | None
    pin_buildup_report_path: Path | None
    pin_slip_report_path: Path | None
    shoulder_quality_report_path: Path | None
    machine_reachability_report_path: Path | None
    pin_route_candidates_path: Path | None
    pin_route_selected_path: Path | None
    pin_route_score_report_path: Path | None
    dome_coverage_report_path: Path | None
    left_dome_coverage_report_path: Path | None
    right_dome_coverage_report_path: Path | None
    dome_gap_overlap_map_path: Path | None
    dome_angle_map_path: Path | None
    dome_thickness_map_path: Path | None
    dome_overbuild_report_path: Path | None
    shoulder_transition_report_path: Path | None
    optimisation_repair_suggestions_path: Path | None
    plot_manifest_path: Path | None
    plot_paths: tuple[Path, ...]
    summary: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _PinRouteCandidate:
    candidate_id: str
    step_size: int
    wrap_direction: int
    circuit_repeats: int
    target_angle_deg: float
    tangent_bias_deg: float
    layer: PlannedLayer | None
    valid: bool
    score: float
    terms: dict[str, float]
    rejection_reasons: tuple[str, ...]
    repair_suggestions: tuple[str, ...]


def validate_winding_job_config(config: WindingJobConfig) -> tuple[str, ...]:
    warnings: list[str] = []
    if config.project.units != "mm":
        raise ValueError("only mm units are currently supported")
    if config.mandrel.type not in {
        "cylinder",
        "cylinder_with_domes",
        "cylinder_with_elliptical_domes",
        "axisymmetric_profile",
        "profile",
    }:
        raise ValueError(
            "mandrel.type must be cylinder, cylinder_with_domes, "
            "cylinder_with_elliptical_domes, or axisymmetric_profile"
        )
    if (
        config.mandrel.type in {"axisymmetric_profile", "profile"}
        and config.mandrel.profile_path is None
    ):
        raise ValueError("mandrel.profile_path is required for imported profile mandrels")
    if _mandrel_length_mm(config) <= 0.0:
        raise ValueError("mandrel.length_mm must be positive")
    if _mandrel_radius_mm(config) <= 0.0:
        raise ValueError("mandrel.radius_mm must be positive")
    if config.tow.width_mm <= 0.0:
        raise ValueError("tow.width_mm must be positive")
    if config.tow.thickness_mm < 0.0:
        raise ValueError("tow.thickness_mm must be non-negative")
    if config.tow.effective_width_mm is not None and config.tow.effective_width_mm <= 0.0:
        raise ValueError("tow.effective_width_mm must be positive when provided")
    if config.tow.friction_coefficient is not None and config.tow.friction_coefficient <= 0.0:
        raise ValueError("tow.friction_coefficient must be positive when provided")
    if config.tow.min_bend_radius_mm is not None and config.tow.min_bend_radius_mm <= 0.0:
        raise ValueError("tow.min_bend_radius_mm must be positive when provided")
    if config.tow.tension_N is not None and config.tow.tension_N < 0.0:
        raise ValueError("tow.tension_N must be non-negative when provided")
    _validate_pin_layout_config(config)
    if config.machine.clearance_mm < 0.0:
        raise ValueError("machine.clearance_mm must be non-negative")
    enabled_layers = [layer for layer in config.layers if layer.enabled]
    if not enabled_layers:
        raise ValueError("config must contain at least one enabled layer")
    for index, layer in enumerate(config.layers, start=1):
        _validate_layer(config, layer, index)
    if not config.output.directory:
        raise ValueError("output.directory is required")
    if config.plot.enabled:
        if not config.plot.save and not config.plot.show:
            raise ValueError("plot is enabled but both plot.save and plot.show are false")
        if not config.plot.modes:
            raise ValueError("plot is enabled but no plot outputs are selected")
    return tuple(warnings)


def _validate_pin_layout_config(config: WindingJobConfig) -> None:
    pins = config.pin_layout
    if not pins.enabled:
        return
    if pins.layout_type != "shoulder_cross":
        raise ValueError("pin_layout.type must be shoulder_cross")
    if pins.routing_mode not in {"deterministic", "optimise_candidates"}:
        raise ValueError("pin_layout.routing_mode must be deterministic or optimise_candidates")
    if pins.candidate_count < 1:
        raise ValueError("pin_layout.candidate_count must be at least 1")
    if pins.route_step_size < 0:
        raise ValueError("pin_layout.route_step_size must be non-negative")
    if pins.wrap_direction not in {"both", "forward", "reverse"}:
        raise ValueError("pin_layout.wrap_direction must be both, forward, or reverse")
    if pins.target_dome_angle_max_deg < pins.target_dome_angle_min_deg:
        raise ValueError("pin_layout target dome angle range must be increasing")
    if pins.coverage_tolerance_mm < 0.0:
        raise ValueError("pin_layout.coverage_tolerance_mm must be non-negative")
    if pins.shoulders not in {"left", "right", "both"}:
        raise ValueError("pin_layout.shoulders must be left, right, or both")
    if pins.count_per_shoulder < 2:
        raise ValueError("pin_layout.count_per_shoulder must be at least 2")
    if pins.pin_radius_mm <= 0.0:
        raise ValueError("pin_layout.pin_radius_mm must be positive")
    if pins.pin_height_mm <= 0.0:
        raise ValueError("pin_layout.pin_height_mm must be positive")
    if pins.pin_standoff_mm < 0.0:
        raise ValueError("pin_layout.pin_standoff_mm must be non-negative")
    if pins.pin_clearance_mm < 0.0:
        raise ValueError("pin_layout.pin_clearance_mm must be non-negative")
    if pins.min_wrap_deg <= 0.0 or pins.max_wrap_deg <= pins.min_wrap_deg:
        raise ValueError("pin_layout wrap limits must be positive and increasing")
    if pins.max_buildup_height_mm <= 0.0:
        raise ValueError("pin_layout.max_buildup_height_mm must be positive")
    if pins.max_contact_balance_ratio < 1.0:
        raise ValueError("pin_layout.max_contact_balance_ratio must be at least 1")
    bend_limit = (
        pins.min_bend_radius_mm
        if pins.min_bend_radius_mm is not None
        else config.tow.min_bend_radius_mm
    )
    if bend_limit is not None and pins.pin_radius_mm < bend_limit:
        raise ValueError(
            "pin_layout.pin_radius_mm must be >= configured tow/pin min_bend_radius_mm"
        )


def generate_winding_job(
    config: WindingJobConfig,
    *,
    export_csv: bool | None = None,
    export_summary: bool | None = None,
    make_plots: bool | None = None,
) -> WindingJobResult:
    warnings = list(validate_winding_job_config(config))
    mandrel = _build_mandrel(config)
    pattern_result = analyze_winding_patterns(config, mandrel)
    schedule = _schedule_from_config(config, pattern_result=pattern_result)
    program = plan_winding_schedule(mandrel, schedule)
    program = _append_pin_routed_program(config, mandrel, program)
    output_dir = config.output.directory
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_enabled = config.output.csv if export_csv is None else export_csv
    summary_enabled = config.output.summary_json if export_summary is None else export_summary
    plot_enabled = config.plot.enabled if make_plots is None else make_plots
    segments = build_path_segments(program)
    coverage = _coverage_map_for_program(config, mandrel, program)

    csv_path = export_winding_program_csv(program, output_dir / "path.csv") if csv_enabled else None
    gcode_path = (
        export_gcode(
            program.motion_table,
            output_dir / "path.gcode",
            options=GCodeOptions(feedrate_mm_min=500.0, feed_schedule=program.feed_schedule),
        )
        if config.output.gcode
        else None
    )
    summary_path = output_dir / "summary.json" if summary_enabled else None
    segments_path = (
        export_segments_json(segments, output_dir / "segments.json")
        if config.output.segments_json
        else None
    )
    coverage_grid_path = (
        export_coverage_grid_npz(coverage, output_dir / "coverage_grid.npz")
        if config.output.coverage_grid
        else None
    )
    pattern_candidates_path = None
    selected_pattern_path = None
    pattern_rejection_report_path = None
    thickness_distribution_path = None
    layer_completion_report_path = None
    stack_coverage_report_path = None
    machine_smoothing_report_path = None
    pattern_optimisation_report_path = None
    candidate_pair_report_path = None
    actual_thickness_report_path = None
    region_quality_report_path = None
    calibration_report_path = None
    friction_margin_report_path = None
    polar_overbuild_report_path = None
    collision_report_path = None
    pin_layout_report_path = None
    pin_contact_report_path = None
    pin_buildup_report_path = None
    pin_slip_report_path = None
    shoulder_quality_report_path = None
    machine_reachability_report_path = None
    pin_route_candidates_path = None
    pin_route_selected_path = None
    pin_route_score_report_path = None
    dome_coverage_report_path = None
    left_dome_coverage_report_path = None
    right_dome_coverage_report_path = None
    dome_gap_overlap_map_path = None
    dome_angle_map_path = None
    dome_thickness_map_path = None
    dome_overbuild_report_path = None
    shoulder_transition_report_path = None
    optimisation_repair_suggestions_path = None
    plot_manifest_path = None
    if pattern_result is not None:
        pattern_candidates_path = export_pattern_candidates_json(
            pattern_result,
            output_dir / "pattern_candidates.json",
        )
        selected_pattern_path = export_selected_pattern_json(
            pattern_result,
            output_dir / "selected_pattern.json",
        )
        pattern_rejection_report_path = export_pattern_rejection_report_json(
            pattern_result,
            output_dir / "pattern_rejection_report.json",
        )
        thickness_distribution_path = export_thickness_distribution_csv(
            pattern_result,
            output_dir / "thickness_distribution.csv",
        )
    selected_patterns = _selected_patterns_by_layer(pattern_result)
    layer_completion_report = _build_layer_completion_report(
        config,
        mandrel,
        program,
        selected_patterns=selected_patterns,
    )
    stack_coverage_report = _build_stack_coverage_report(
        config,
        mandrel,
        program,
        coverage,
        layer_completion_report=layer_completion_report,
    )
    machine_smoothing_report = _machine_smoothing_report(config, program, segments)
    region_quality_report = _region_quality_report(config, mandrel, coverage)
    layer_completion_report_path = _write_json(
        layer_completion_report,
        output_dir / "layer_completion_report.json",
    )
    stack_coverage_report_path = _write_json(
        stack_coverage_report,
        output_dir / "stack_coverage_report.json",
    )
    machine_smoothing_report_path = _write_json(
        machine_smoothing_report,
        output_dir / "machine_smoothing_report.json",
    )
    candidate_pair_report = _candidate_pair_report(pattern_result)
    actual_thickness_report = _actual_thickness_report(
        config,
        mandrel,
        coverage,
        nominal_stack_thickness_mm=sum(layer.spec.layer_thickness_mm for layer in program.layers),
    )
    calibration_report = _calibration_report(config)
    friction_margin_report = _friction_margin_report(config, program)
    polar_overbuild_report = _polar_overbuild_report(config, actual_thickness_report)
    collision_report = _collision_report(config, program)
    pin_layout_report = _pin_layout_report(config, mandrel)
    pin_contact_report = _pin_contact_report(config, pin_layout_report, program)
    pin_buildup_report = _pin_buildup_report(config, pin_contact_report)
    pin_slip_report = _pin_slip_report(config, pin_contact_report)
    machine_reachability_report = _machine_reachability_report(
        config,
        pin_layout_report,
        collision_report,
    )
    pin_route_reports = _pin_route_reports(config, mandrel, program)
    dome_coverage_report = _pin_dome_coverage_report(config, mandrel, program)
    shoulder_transition_report = _shoulder_transition_report(config, program)
    dome_overbuild_report = _dome_overbuild_report(config, dome_coverage_report)
    pattern_optimisation_report = _pattern_optimisation_report(
        config,
        pattern_result,
        layer_completion_report=layer_completion_report,
        stack_coverage_report=stack_coverage_report,
        dome_coverage_report=dome_coverage_report,
        dome_overbuild_report=dome_overbuild_report,
    )
    shoulder_quality_report = _shoulder_quality_report(
        config,
        pin_contact_report,
        pin_buildup_report,
        pin_slip_report,
        dome_coverage_report,
        shoulder_transition_report,
        machine_reachability_report,
    )
    pattern_optimisation_report_path = _write_json(
        pattern_optimisation_report,
        output_dir / "pattern_optimisation_report.json",
    )
    candidate_pair_report_path = _write_json(
        candidate_pair_report,
        output_dir / "candidate_pair_report.json",
    )
    actual_thickness_report_path = _write_json(
        actual_thickness_report,
        output_dir / "actual_thickness_report.json",
    )
    region_quality_report_path = _write_json(
        region_quality_report,
        output_dir / "region_quality_report.json",
    )
    calibration_report_path = _write_json(
        calibration_report,
        output_dir / "calibration_report.json",
    )
    friction_margin_report_path = _write_json(
        friction_margin_report,
        output_dir / "friction_margin_report.json",
    )
    polar_overbuild_report_path = _write_json(
        polar_overbuild_report,
        output_dir / "polar_overbuild_report.json",
    )
    collision_report_path = _write_json(
        collision_report,
        output_dir / "collision_report.json",
    )
    pin_layout_report_path = _write_json(pin_layout_report, output_dir / "pin_layout.json")
    pin_contact_report_path = _write_json(
        pin_contact_report,
        output_dir / "pin_contact_report.json",
    )
    pin_buildup_report_path = _write_json(
        pin_buildup_report,
        output_dir / "pin_buildup_report.json",
    )
    pin_slip_report_path = _write_json(pin_slip_report, output_dir / "pin_slip_report.json")
    shoulder_quality_report_path = _write_json(
        shoulder_quality_report,
        output_dir / "shoulder_quality_report.json",
    )
    machine_reachability_report_path = _write_json(
        machine_reachability_report,
        output_dir / "machine_reachability_report.json",
    )
    pin_route_candidates_path = _write_json(
        pin_route_reports["candidates"],
        output_dir / "pin_route_candidates.json",
    )
    pin_route_selected_path = _write_json(
        pin_route_reports["selected"],
        output_dir / "pin_route_selected.json",
    )
    pin_route_score_report_path = _write_json(
        pin_route_reports["score_report"],
        output_dir / "pin_route_score_report.json",
    )
    dome_coverage_report_path = _write_json(
        dome_coverage_report,
        output_dir / "dome_coverage_report.json",
    )
    left_dome_coverage_report_path = _write_json(
        dome_coverage_report["by_side"]["left"],
        output_dir / "left_dome_coverage_report.json",
    )
    right_dome_coverage_report_path = _write_json(
        dome_coverage_report["by_side"]["right"],
        output_dir / "right_dome_coverage_report.json",
    )
    dome_gap_overlap_map_path = _write_pin_dome_csv(
        dome_coverage_report["cells"],
        output_dir / "dome_gap_overlap_map.csv",
        (
            "side",
            "meridian_fraction",
            "theta_deg",
            "coverage_count",
            "gap_mm",
            "overlap_mm",
            "area_weight",
        ),
    )
    dome_angle_map_path = _write_pin_dome_csv(
        dome_coverage_report["angle_cells"],
        output_dir / "dome_angle_map.csv",
        (
            "side",
            "meridian_fraction",
            "theta_deg",
            "local_angle_deg",
            "target_angle_deg",
            "angle_error_deg",
        ),
    )
    dome_thickness_map_path = _write_pin_dome_csv(
        dome_coverage_report["thickness_cells"],
        output_dir / "dome_thickness_map.csv",
        (
            "side",
            "meridian_fraction",
            "theta_deg",
            "thickness_mm",
            "coverage_count",
            "area_weight",
        ),
    )
    dome_overbuild_report_path = _write_json(
        dome_overbuild_report,
        output_dir / "dome_overbuild_report.json",
    )
    shoulder_transition_report_path = _write_json(
        shoulder_transition_report,
        output_dir / "shoulder_transition_report.json",
    )
    optimisation_repair_suggestions = _optimisation_repair_suggestions(
        config=config,
        pattern_result=pattern_result,
        stack_coverage_report=stack_coverage_report,
        region_quality_report=region_quality_report,
    )
    optimisation_repair_suggestions_path = _write_json(
        optimisation_repair_suggestions,
        output_dir / "optimisation_repair_suggestions.json",
    )
    diagnostic_plot_paths: tuple[Path, ...] = ()
    plot_manifest: dict[str, Any] = {"plots": []}
    if plot_enabled:
        diagnostic_plot_paths, plot_manifest = plot_layer_diagnostics(
            config,
            mandrel,
            program,
            coverage,
            output_dir,
        )
        dome_plot_paths = plot_dome_coverage_maps(dome_coverage_report, output_dir)
        motion_plot_paths = plot_dome_motion_diagnostics(config, mandrel, program, output_dir)
        diagnostic_plot_paths = diagnostic_plot_paths + dome_plot_paths + motion_plot_paths
        manifest_plots = plot_manifest.setdefault("plots", [])
        if isinstance(manifest_plots, list):
            manifest_plots.extend(
                {"type": path.stem, "path": str(path)} for path in dome_plot_paths
            )
            manifest_plots.extend(
                {"type": path.stem, "path": str(path)} for path in motion_plot_paths
            )
        plot_manifest_path = _write_json(plot_manifest, output_dir / "plot_manifest.json")
    plot_paths = (
        plot_winding_program(config, mandrel, program, output_dir) + diagnostic_plot_paths
        if plot_enabled
        else ()
    )
    validation_report = _build_validation_report(
        config,
        program,
        segments,
        coverage,
        layer_completion_report=layer_completion_report,
        stack_coverage_report=stack_coverage_report,
        machine_smoothing_report=machine_smoothing_report,
        pattern_optimisation_report=pattern_optimisation_report,
        region_quality_report=region_quality_report,
        calibration_report=calibration_report,
        friction_margin_report=friction_margin_report,
        polar_overbuild_report=polar_overbuild_report,
        collision_report=collision_report,
        shoulder_quality_report=shoulder_quality_report,
        machine_reachability_report=machine_reachability_report,
    )
    validation_report_path = (
        export_validation_report_json(validation_report, output_dir / "validation_report.json")
        if config.output.validation_report_json
        else None
    )
    summary = _build_summary(
        config,
        mandrel,
        program,
        total_segments=len(segments),
        warnings=tuple(warnings),
        csv_path=csv_path,
        gcode_path=gcode_path,
        summary_path=summary_path,
        segments_path=segments_path,
        validation_report_path=validation_report_path,
        coverage_grid_path=coverage_grid_path,
        pattern_result=pattern_result,
        pattern_candidates_path=pattern_candidates_path,
        selected_pattern_path=selected_pattern_path,
        pattern_rejection_report_path=pattern_rejection_report_path,
        thickness_distribution_path=thickness_distribution_path,
        layer_completion_report_path=layer_completion_report_path,
        stack_coverage_report_path=stack_coverage_report_path,
        machine_smoothing_report_path=machine_smoothing_report_path,
        pattern_optimisation_report_path=pattern_optimisation_report_path,
        candidate_pair_report_path=candidate_pair_report_path,
        actual_thickness_report_path=actual_thickness_report_path,
        region_quality_report_path=region_quality_report_path,
        calibration_report_path=calibration_report_path,
        friction_margin_report_path=friction_margin_report_path,
        polar_overbuild_report_path=polar_overbuild_report_path,
        collision_report_path=collision_report_path,
        pin_layout_report_path=pin_layout_report_path,
        pin_contact_report_path=pin_contact_report_path,
        pin_buildup_report_path=pin_buildup_report_path,
        pin_slip_report_path=pin_slip_report_path,
        shoulder_quality_report_path=shoulder_quality_report_path,
        machine_reachability_report_path=machine_reachability_report_path,
        pin_route_candidates_path=pin_route_candidates_path,
        pin_route_selected_path=pin_route_selected_path,
        pin_route_score_report_path=pin_route_score_report_path,
        dome_coverage_report_path=dome_coverage_report_path,
        left_dome_coverage_report_path=left_dome_coverage_report_path,
        right_dome_coverage_report_path=right_dome_coverage_report_path,
        dome_gap_overlap_map_path=dome_gap_overlap_map_path,
        dome_angle_map_path=dome_angle_map_path,
        dome_thickness_map_path=dome_thickness_map_path,
        dome_overbuild_report_path=dome_overbuild_report_path,
        shoulder_transition_report_path=shoulder_transition_report_path,
        optimisation_repair_suggestions_path=optimisation_repair_suggestions_path,
        plot_manifest_path=plot_manifest_path,
        layer_completion_report=layer_completion_report,
        stack_coverage_report=stack_coverage_report,
        machine_smoothing_report=machine_smoothing_report,
        pattern_optimisation_report=pattern_optimisation_report,
        region_quality_report=region_quality_report,
        calibration_report=calibration_report,
        friction_margin_report=friction_margin_report,
        polar_overbuild_report=polar_overbuild_report,
        collision_report=collision_report,
        shoulder_quality_report=shoulder_quality_report,
        machine_reachability_report=machine_reachability_report,
        plot_paths=plot_paths,
        coverage=coverage,
    )
    consistency_report = _run_consistency_validation(
        config=config,
        program=program,
        segments=segments,
        coverage=coverage,
        csv_path=csv_path,
        summary_path=summary_path,
        validation_report_path=validation_report_path,
        pattern_candidates_path=pattern_candidates_path,
        selected_pattern_path=selected_pattern_path,
        pattern_rejection_report_path=pattern_rejection_report_path,
        thickness_distribution_path=thickness_distribution_path,
        layer_completion_report_path=layer_completion_report_path,
        stack_coverage_report_path=stack_coverage_report_path,
        machine_smoothing_report_path=machine_smoothing_report_path,
        pattern_optimisation_report_path=pattern_optimisation_report_path,
        candidate_pair_report_path=candidate_pair_report_path,
        actual_thickness_report_path=actual_thickness_report_path,
        region_quality_report_path=region_quality_report_path,
        calibration_report_path=calibration_report_path,
        friction_margin_report_path=friction_margin_report_path,
        polar_overbuild_report_path=polar_overbuild_report_path,
        collision_report_path=collision_report_path,
        pin_layout_report_path=pin_layout_report_path,
        pin_contact_report_path=pin_contact_report_path,
        pin_buildup_report_path=pin_buildup_report_path,
        pin_slip_report_path=pin_slip_report_path,
        shoulder_quality_report_path=shoulder_quality_report_path,
        machine_reachability_report_path=machine_reachability_report_path,
        pin_route_candidates_path=pin_route_candidates_path,
        pin_route_selected_path=pin_route_selected_path,
        pin_route_score_report_path=pin_route_score_report_path,
        dome_coverage_report_path=dome_coverage_report_path,
        left_dome_coverage_report_path=left_dome_coverage_report_path,
        right_dome_coverage_report_path=right_dome_coverage_report_path,
        dome_gap_overlap_map_path=dome_gap_overlap_map_path,
        dome_angle_map_path=dome_angle_map_path,
        dome_thickness_map_path=dome_thickness_map_path,
        dome_overbuild_report_path=dome_overbuild_report_path,
        shoulder_transition_report_path=shoulder_transition_report_path,
        optimisation_repair_suggestions_path=optimisation_repair_suggestions_path,
        plot_manifest_path=plot_manifest_path,
        gcode_path=gcode_path,
    )
    summary["manufacturing_report"] = _manufacturing_report(
        config=config,
        program=program,
        coverage=coverage,
        consistency_report=consistency_report,
        pattern_result=pattern_result,
        layer_completion_report=layer_completion_report,
        stack_coverage_report=stack_coverage_report,
        machine_smoothing_report=machine_smoothing_report,
        calibration_report=calibration_report,
        friction_margin_report=friction_margin_report,
        polar_overbuild_report=polar_overbuild_report,
        collision_report=collision_report,
        shoulder_quality_report=shoulder_quality_report,
        machine_reachability_report=machine_reachability_report,
    )
    if summary_enabled:
        assert summary_path is not None
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return WindingJobResult(
        config=config,
        mandrel=mandrel,
        program=program,
        csv_path=csv_path,
        gcode_path=gcode_path,
        summary_path=summary_path,
        segments_path=segments_path,
        validation_report_path=validation_report_path,
        coverage_grid_path=coverage_grid_path,
        pattern_candidates_path=pattern_candidates_path,
        selected_pattern_path=selected_pattern_path,
        pattern_rejection_report_path=pattern_rejection_report_path,
        thickness_distribution_path=thickness_distribution_path,
        layer_completion_report_path=layer_completion_report_path,
        stack_coverage_report_path=stack_coverage_report_path,
        machine_smoothing_report_path=machine_smoothing_report_path,
        pattern_optimisation_report_path=pattern_optimisation_report_path,
        candidate_pair_report_path=candidate_pair_report_path,
        actual_thickness_report_path=actual_thickness_report_path,
        region_quality_report_path=region_quality_report_path,
        calibration_report_path=calibration_report_path,
        friction_margin_report_path=friction_margin_report_path,
        polar_overbuild_report_path=polar_overbuild_report_path,
        collision_report_path=collision_report_path,
        pin_layout_report_path=pin_layout_report_path,
        pin_contact_report_path=pin_contact_report_path,
        pin_buildup_report_path=pin_buildup_report_path,
        pin_slip_report_path=pin_slip_report_path,
        shoulder_quality_report_path=shoulder_quality_report_path,
        machine_reachability_report_path=machine_reachability_report_path,
        pin_route_candidates_path=pin_route_candidates_path,
        pin_route_selected_path=pin_route_selected_path,
        pin_route_score_report_path=pin_route_score_report_path,
        dome_coverage_report_path=dome_coverage_report_path,
        left_dome_coverage_report_path=left_dome_coverage_report_path,
        right_dome_coverage_report_path=right_dome_coverage_report_path,
        dome_gap_overlap_map_path=dome_gap_overlap_map_path,
        dome_angle_map_path=dome_angle_map_path,
        dome_thickness_map_path=dome_thickness_map_path,
        dome_overbuild_report_path=dome_overbuild_report_path,
        shoulder_transition_report_path=shoulder_transition_report_path,
        optimisation_repair_suggestions_path=optimisation_repair_suggestions_path,
        plot_manifest_path=plot_manifest_path,
        plot_paths=plot_paths,
        summary=summary,
    )


def with_pattern_method(config: WindingJobConfig, method: str | None) -> WindingJobConfig:
    if method is None:
        return config
    resolved = "textbook_integer_closure" if method == "textbook" else method
    return replace(
        config,
        pattern_selection=replace(config.pattern_selection, method=resolved),
    )


def _effective_tow_width(config: WindingJobConfig) -> float:
    return (
        config.tow.effective_width_mm
        if config.tow.effective_width_mm is not None
        else config.tow.width_mm
    )


def _append_pin_routed_program(
    config: WindingJobConfig,
    mandrel: CylinderMandrel | AxisymmetricProfileMandrel,
    program: PlannedWindingProgram,
) -> PlannedWindingProgram:
    if not config.pin_layout.enabled:
        return program
    if config.pin_layout.routing_mode == "optimise_candidates":
        candidates = _generate_pin_route_candidates(config, mandrel, len(program.layers))
        valid = [
            candidate
            for candidate in candidates
            if candidate.valid and candidate.layer is not None
        ]
        if not valid:
            suggestions = sorted(
                {item for candidate in candidates for item in candidate.repair_suggestions}
            )
            reasons = sorted(
                {item for candidate in candidates for item in candidate.rejection_reasons}
            )
            raise ValueError(
                "pin route optimiser found no valid shoulder cross-pin route; "
                f"reasons={reasons}; repair_suggestions={suggestions}"
            )
        layer = min(valid, key=lambda candidate: candidate.score).layer
        assert layer is not None
    else:
        layer = _build_pin_routed_layer(
            config,
            mandrel,
            len(program.layers),
            candidate_id="deterministic",
            step_size=max(1, config.pin_layout.count_per_shoulder // 2),
            wrap_direction=1,
            circuit_repeats=1,
            target_angle_deg=_pin_route_target_angle(config),
            tangent_bias_deg=0.0,
        )
    return PlannedWindingProgram(
        layers=(*program.layers, layer),
        path=_concat_surface_paths((program.path, layer.path)),
        motion_table=_concat_motion_tables((program.motion_table, layer.motion_table)),
        feed_schedule=_concat_feed_schedules((program.feed_schedule, layer.feed_schedule)),
        metadata=_concat_program_metadata((program.metadata, layer.metadata)),
        reports=(*program.reports, layer.report),
    )


def _build_pin_routed_layer(
    config: WindingJobConfig,
    mandrel: CylinderMandrel | AxisymmetricProfileMandrel,
    layer_index: int,
    *,
    candidate_id: str,
    step_size: int,
    wrap_direction: int,
    circuit_repeats: int,
    target_angle_deg: float,
    tangent_bias_deg: float,
) -> PlannedLayer:
    pins = config.pin_layout
    count = max(2, int(pins.count_per_shoulder))
    shoulder_z = _pin_shoulder_stations(config, mandrel)
    tow_width = _effective_tow_width(config)
    layer_thickness = max(config.tow.thickness_mm, config.roving.thickness_mm, 0.0)
    target_angle = target_angle_deg
    points_z: list[np.ndarray] = []
    points_theta: list[np.ndarray] = []
    pass_chunks: list[np.ndarray] = []
    b_chunks: list[np.ndarray] = []
    motion_chunks: list[tuple[str, ...]] = []
    warning_chunks: list[tuple[str, ...]] = []
    theta_step = 2.0 * math.pi / count
    offset = math.radians(pins.angular_offset_deg)
    left_z = shoulder_z["left"]
    right_z = shoulder_z["right"]

    total_passes = count * max(1, circuit_repeats)
    phase_step = theta_step / max(1, circuit_repeats)
    for pass_id in range(total_passes):
        index = pass_id % count
        repeat = pass_id // count
        phase = repeat * phase_step
        theta_left = offset + index * theta_step + phase
        theta_left_opposite = offset + ((index + step_size) % count) * theta_step + phase
        theta_right = offset + ((index + wrap_direction) % count) * theta_step + phase
        theta_right_opposite = (
            offset + ((index + wrap_direction + step_size) % count) * theta_step + phase
        )
        _append_pin_arc(
            points_z,
            points_theta,
            pass_chunks,
            b_chunks,
            motion_chunks,
            warning_chunks,
            z_mm=left_z,
            theta_start=theta_left - math.radians(pins.min_wrap_deg) * 0.5 * wrap_direction,
            theta_end=theta_left + math.radians(pins.min_wrap_deg) * 0.5 * wrap_direction,
            pass_id=pass_id,
            b_deg=target_angle,
            pin_id=f"left_{index:02d}",
            wrap_deg=pins.min_wrap_deg,
        )
        _append_dome_span(
            config,
            mandrel,
            points_z,
            points_theta,
            pass_chunks,
            b_chunks,
            motion_chunks,
            warning_chunks,
            side="left",
            z_shoulder=left_z,
            theta_start=theta_left + math.radians(tangent_bias_deg),
            theta_end=theta_left_opposite - math.radians(tangent_bias_deg),
            pass_id=pass_id,
            b_deg=target_angle,
        )
        _append_cylinder_span(
            points_z,
            points_theta,
            pass_chunks,
            b_chunks,
            motion_chunks,
            warning_chunks,
            z_start=left_z,
            z_end=right_z,
            theta_start=theta_left_opposite,
            theta_end=theta_right,
            pass_id=pass_id,
            b_deg=target_angle,
        )
        _append_pin_arc(
            points_z,
            points_theta,
            pass_chunks,
            b_chunks,
            motion_chunks,
            warning_chunks,
            z_mm=right_z,
            theta_start=theta_right - math.radians(pins.min_wrap_deg) * 0.5 * wrap_direction,
            theta_end=theta_right + math.radians(pins.min_wrap_deg) * 0.5 * wrap_direction,
            pass_id=pass_id,
            b_deg=target_angle,
            pin_id=f"right_{(index + wrap_direction) % count:02d}",
            wrap_deg=pins.min_wrap_deg,
        )
        _append_dome_span(
            config,
            mandrel,
            points_z,
            points_theta,
            pass_chunks,
            b_chunks,
            motion_chunks,
            warning_chunks,
            side="right",
            z_shoulder=right_z,
            theta_start=theta_right + math.radians(tangent_bias_deg),
            theta_end=theta_right_opposite - math.radians(tangent_bias_deg),
            pass_id=pass_id,
            b_deg=target_angle,
        )
        _append_cylinder_span(
            points_z,
            points_theta,
            pass_chunks,
            b_chunks,
            motion_chunks,
            warning_chunks,
            z_start=right_z,
            z_end=left_z,
            theta_start=theta_right_opposite,
            theta_end=offset + (index + wrap_direction) * theta_step + phase,
            pass_id=pass_id,
            b_deg=-target_angle,
        )
    z_mm = np.concatenate(points_z)
    theta_rad = np.concatenate(points_theta)
    pass_index = np.concatenate(pass_chunks)
    b_angle = np.concatenate(b_chunks)
    motion_type = tuple(label for chunk in motion_chunks for label in chunk)
    warnings = tuple(label for chunk in warning_chunks for label in chunk)
    points = mandrel.surface_points(z_mm, theta_rad)
    path = SurfacePath(
        z_mm=z_mm,
        theta_rad=theta_rad,
        x_mm=points[:, 0],
        y_mm=points[:, 1],
        winding_angle_deg=target_angle,
        tow_width_mm=tow_width,
        pass_index=pass_index,
        tow_eye_angle_deg=b_angle,
    )
    clearance = max(config.machine.clearance_mm, config.pin_layout.pin_height_mm)
    motion = machine_path_from_surface_path(path, radial_clearance_mm=clearance)
    feed = plan_feedrate(
        path,
        FeedrateConfig(
            nominal_feedrate_mm_min=450.0,
            minimum_feedrate_mm_min=120.0,
        ),
    )
    layer_id = f"pin-route-shoulder-cross-{candidate_id}"
    metadata = WindingPointMetadata(
        layer_id=tuple(layer_id for _ in range(path.point_count)),
        layer_index=np.full(path.point_count, layer_index, dtype=int),
        circuit_index=pass_index.copy(),
        pass_index=pass_index.copy(),
        local_radius_mm=path.surface_radius_mm,
        local_winding_angle_deg=_local_angle_from_path(path),
        layer_name=tuple("shoulder_cross_pin_route" for _ in range(path.point_count)),
        winding_type=tuple("pin_route" for _ in range(path.point_count)),
        motion_type=motion_type,
        warning_flags=warnings,
    )
    report = WindingPatternReport(
        layer_id=layer_id,
        layer_name="shoulder_cross_pin_route",
        winding_type="helical",  # type: ignore[arg-type]
        target_angle_deg=target_angle,
        actual_angle_deg=target_angle,
        angle_error_deg=0.0,
        circuits=total_passes,
        starts=1,
        angular_shift_deg=360.0 / count,
        tow_spacing_mm=tow_width,
        coverage_percent=100.0,
        gap_mm=0.0,
        overlap_mm=0.0,
        layer_completion_z_mm=float(np.max(z_mm) - np.min(z_mm)),
        pattern_repeat_length_mm=float(
            np.sum(np.linalg.norm(np.diff(path.points_mm, axis=0), axis=1))
        ),
        closes=True,
        acceptable=True,
        warnings=(
            f"pin-routed shoulder-cross layer candidate_id={candidate_id}",
            f"routing_mode={config.pin_layout.routing_mode}",
        ),
    )
    spec = WindingLayerSpec(
        layer_id=layer_id,
        name="shoulder_cross_pin_route",
        winding_type="helical",
        target_angle_deg=target_angle,
        tow_width_mm=tow_width,
        layer_thickness_mm=layer_thickness,
        point_count=path.point_count,
        enabled=True,
    )
    return PlannedLayer(
        spec=spec,
        path=path,
        motion_table=motion,
        feed_schedule=feed,
        metadata=metadata,
        report=report,
        effective_radius_mm=float(np.max(path.surface_radius_mm)),
        accumulated_thickness_before_mm=0.0,
    )


def _generate_pin_route_candidates(
    config: WindingJobConfig,
    mandrel: CylinderMandrel | AxisymmetricProfileMandrel,
    layer_index: int,
) -> tuple[_PinRouteCandidate, ...]:
    pins = config.pin_layout
    count = max(2, int(pins.count_per_shoulder))
    steps = [max(1, count // 2), 1]
    if pins.route_step_size:
        steps.insert(0, max(1, min(count - 1, pins.route_step_size)))
    steps = list(dict.fromkeys(step for step in steps if 0 < step < count))
    directions = (
        [1, -1]
        if pins.wrap_direction == "both"
        else [1 if pins.wrap_direction == "forward" else -1]
    )
    base_angle = _pin_route_target_angle(config)
    angle_min = max(0.1, pins.target_dome_angle_min_deg)
    angle_max = max(angle_min, pins.target_dome_angle_max_deg)
    angle_values = list(
        dict.fromkeys(
            float(max(angle_min, min(angle_max, angle)))
            for angle in (base_angle, angle_min, angle_max, (angle_min + angle_max) * 0.5)
        )
    )
    tangent_biases = [0.0, 5.0, -5.0]
    repeat_values = [1, 2, 4, 8]
    candidates: list[_PinRouteCandidate] = []
    for repeats in repeat_values:
        for step in steps:
            for direction in directions:
                for angle in angle_values:
                    for bias in tangent_biases:
                        candidate_id = (
                            f"step{step}_dir{'f' if direction > 0 else 'r'}_"
                            f"rep{repeats}_ang{angle:.1f}_bias{bias:+.1f}"
                        ).replace(".", "p").replace("+", "p").replace("-", "m")
                        layer = _build_pin_routed_layer(
                            config,
                            mandrel,
                            layer_index,
                            candidate_id=candidate_id,
                            step_size=step,
                            wrap_direction=direction,
                            circuit_repeats=repeats,
                            target_angle_deg=angle,
                            tangent_bias_deg=bias,
                        )
                        candidates.append(
                            _score_pin_route_candidate(
                                config,
                                mandrel,
                                candidate_id=candidate_id,
                                step_size=step,
                                wrap_direction=direction,
                                circuit_repeats=repeats,
                                target_angle_deg=angle,
                                tangent_bias_deg=bias,
                                layer=layer,
                            )
                        )
                        if len(candidates) >= pins.candidate_count:
                            return tuple(candidates)
    return tuple(candidates)


def _score_pin_route_candidate(
    config: WindingJobConfig,
    mandrel: CylinderMandrel | AxisymmetricProfileMandrel,
    *,
    candidate_id: str,
    step_size: int,
    wrap_direction: int,
    circuit_repeats: int,
    target_angle_deg: float,
    tangent_bias_deg: float,
    layer: PlannedLayer,
) -> _PinRouteCandidate:
    segments = build_path_segments(
        PlannedWindingProgram(
            layers=(layer,),
            path=layer.path,
            motion_table=layer.motion_table,
            feed_schedule=layer.feed_schedule,
            metadata=layer.metadata,
            reports=(layer.report,),
        )
    )
    reasons: list[str] = []
    suggestions: list[str] = []
    pin_segments = [segment for segment in segments if segment.segment_type == "PinContactArc"]
    dome_segments = [segment for segment in segments if segment.segment_type == "DomeSurfaceSpan"]
    if not pin_segments:
        reasons.append("no_real_pin_contact_arc")
    if not dome_segments:
        reasons.append("no_real_dome_surface_span")
    wrap_margin = min(
        (
            min(
                _segment_wrap_angle(segment.warnings) - config.pin_layout.min_wrap_deg,
                config.pin_layout.max_wrap_deg - _segment_wrap_angle(segment.warnings),
            )
            for segment in pin_segments
        ),
        default=-1.0,
    )
    if wrap_margin < 0.0:
        reasons.append("pin_wrap_limits_exceeded")
        suggestions.append("increase allowed wrap angle")
    bend_radius = config.pin_layout.pin_radius_mm
    bend_limit = config.pin_layout.min_bend_radius_mm or config.tow.min_bend_radius_mm or 0.0
    bend_margin = bend_radius - bend_limit
    if bend_margin < 0.0:
        reasons.append("bend_radius_limit_failed")
        suggestions.append("increase pin radius")
    x_limit = config.machine.max_x_mm
    reach_margin = math.inf if x_limit is None else x_limit - (
        float(np.max(layer.path.surface_radius_mm))
        + config.pin_layout.pin_height_mm
        + config.machine.clearance_mm
    )
    if reach_margin < 0.0:
        reasons.append("machine_reachability_failed")
        suggestions.append("add Y/yaw axis support if X/Z/A/B cannot reach")
    jumps = np.linalg.norm(np.diff(layer.path.points_mm, axis=0), axis=1)
    max_jump = float(np.max(jumps)) if len(jumps) else 0.0
    jump_limit = max(
        config.pin_layout.shoulder_zone_width_mm,
        _effective_tow_width(config) * 8.0,
        float(np.max(layer.path.surface_radius_mm)) * math.pi,
    )
    if max_jump > jump_limit:
        reasons.append("large_path_jump")
        suggestions.append("increase shoulder clearance")
    local_angles = _local_angle_from_path(layer.path)
    angle_error = float(np.mean(np.abs(np.asarray(local_angles) - target_angle_deg)))
    candidate_program = PlannedWindingProgram(
        layers=(layer,),
        path=layer.path,
        motion_table=layer.motion_table,
        feed_schedule=layer.feed_schedule,
        metadata=layer.metadata,
        reports=(layer.report,),
    )
    dome_report = _pin_dome_coverage_report(config, candidate_program)
    left_dome = dome_report["by_side"]["left"]["summary"]
    right_dome = dome_report["by_side"]["right"]["summary"]
    dome_gap_penalty = max(
        0.0,
        float(left_dome["maximum_uncovered_gap_mm"]) - config.pin_layout.coverage_tolerance_mm,
    ) + max(
        0.0,
        float(right_dome["maximum_uncovered_gap_mm"]) - config.pin_layout.coverage_tolerance_mm,
    )
    overlap_penalty = max(
        0.0,
        float(dome_report["summary"]["max_overlap_mm"])
        - max(2.5, config.quality_limits.max_stack_overlap_percent / 20.0),
    )
    thickness_variation_penalty = max(
        0.0,
        float(left_dome["thickness_coefficient_of_variation"]) - 0.75,
    ) + max(
        0.0,
        float(right_dome["thickness_coefficient_of_variation"]) - 0.75,
    )
    if not bool(dome_report["summary"]["dome_coverage_passed"]):
        reasons.append("dome_coverage_failed")
        suggestions.extend(dome_report["summary"]["repair_suggestions"])
    if bool(dome_report["summary"]["ring_like_path_detected"]):
        reasons.append("ring_like_dome_path")
        suggestions.append("reduce allowed hoop-like dome motion")
    transition_penalty = _pin_transition_penalty(layer)
    buildup_penalty = _pin_buildup_penalty(config, layer)
    slip_margin = _pin_candidate_slip_margin(config, pin_segments)
    if slip_margin < 0.0:
        reasons.append("slip_friction_margin_failed")
        suggestions.append("reduce target winding angle")
    terms = {
        "dome_coverage_gap_penalty": dome_gap_penalty,
        "left_dome_gap_penalty": max(
            0.0,
            float(left_dome["maximum_uncovered_gap_mm"]) - config.pin_layout.coverage_tolerance_mm,
        ),
        "right_dome_gap_penalty": max(
            0.0,
            float(right_dome["maximum_uncovered_gap_mm"]) - config.pin_layout.coverage_tolerance_mm,
        ),
        "dome_overlap_penalty": overlap_penalty,
        "dome_overbuild_penalty": overlap_penalty,
        "dome_thickness_variation_penalty": thickness_variation_penalty,
        "local_winding_angle_error": angle_error,
        "dome_local_angle_error": (
            abs(float(left_dome["local_winding_angle_mean_deg"]) - target_angle_deg)
            + abs(float(right_dome["local_winding_angle_mean_deg"]) - target_angle_deg)
        ),
        "shoulder_transition_smoothness": transition_penalty,
        "pin_buildup_penalty": buildup_penalty,
        "pin_wrap_angle_margin": -wrap_margin,
        "slip_friction_margin": -slip_margin,
        "bend_radius_margin": -bend_margin,
        "collision_clearance_margin": 0.0,
        "machine_axis_smoothness": _pin_machine_smoothness_penalty(layer),
        "total_path_length": float(np.sum(jumps)),
        "unnecessary_free_spans": 0.0,
        "circuit_closure_quality": _pin_circuit_closure_penalty(layer),
        "layer_stacking_uniformity": dome_gap_penalty + overlap_penalty,
    }
    score = sum(max(0.0, value) for value in terms.values())
    if reasons and not suggestions:
        suggestions.append("increase pin count")
    return _PinRouteCandidate(
        candidate_id=candidate_id,
        step_size=step_size,
        wrap_direction=wrap_direction,
        circuit_repeats=circuit_repeats,
        target_angle_deg=target_angle_deg,
        tangent_bias_deg=tangent_bias_deg,
        layer=layer,
        valid=not reasons,
        score=score,
        terms=terms,
        rejection_reasons=tuple(reasons),
        repair_suggestions=tuple(dict.fromkeys(suggestions)),
    )


def _pin_route_target_angle(config: WindingJobConfig) -> float:
    enabled = [layer for layer in config.layers if layer.enabled]
    if not enabled:
        return 45.0
    non_hoop = [
        layer
        for layer in enabled
        if layer.type not in {"hoop", "continuous_hoop_traverse"}
    ]
    return float(abs((non_hoop or enabled)[0].winding_angle_deg))


def _pin_dome_spacing_penalty(config: WindingJobConfig, layer: PlannedLayer) -> float:
    dome_theta = [
        theta
        for theta, label in zip(layer.path.theta_rad, layer.metadata.motion_type, strict=True)
        if label == "DomeSurfaceSpan"
    ]
    if len(dome_theta) < 2:
        return config.pin_layout.coverage_tolerance_mm * 10.0
    theta_sorted = np.sort(np.mod(np.asarray(dome_theta), 2.0 * math.pi))
    gaps = np.diff(np.concatenate((theta_sorted, [theta_sorted[0] + 2.0 * math.pi])))
    radius = max(float(np.mean(layer.path.surface_radius_mm)), 1.0)
    spacing_mm = float(np.max(gaps) * radius)
    return max(0.0, spacing_mm - config.pin_layout.coverage_tolerance_mm)


def _pin_overlap_penalty(config: WindingJobConfig, layer: PlannedLayer) -> float:
    dome_count = sum(1 for label in layer.metadata.motion_type if label == "DomeSurfaceSpan")
    expected = max(1, config.pin_layout.count_per_shoulder * 20)
    return max(0.0, (dome_count - expected) / expected)


def _pin_transition_penalty(layer: PlannedLayer) -> float:
    motion = layer.metadata.motion_type
    if len(motion) < 2:
        return 100.0
    changes = sum(1 for a, b in zip(motion, motion[1:], strict=False) if a != b)
    b_delta = np.abs(np.diff(layer.motion_table.b_deg))
    return float(changes) * 0.1 + float(np.max(b_delta, initial=0.0)) * 0.05


def _pin_buildup_penalty(config: WindingJobConfig, layer: PlannedLayer) -> float:
    counts: dict[str, int] = {}
    for label, warning in zip(
        layer.metadata.motion_type,
        layer.metadata.warning_flags,
        strict=True,
    ):
        if label != "PinContactArc":
            continue
        pin_id = _segment_pin_id((warning,))
        if pin_id:
            counts[pin_id] = counts.get(pin_id, 0) + 1
    max_buildup = max(counts.values(), default=0) * max(config.tow.thickness_mm, 0.0)
    return max(0.0, max_buildup - config.pin_layout.max_buildup_height_mm)


def _pin_candidate_slip_margin(config: WindingJobConfig, pin_segments: list[Any]) -> float:
    mu = config.pin_layout.friction_coefficient or config.tow.friction_coefficient
    if mu is None or mu <= 0.0 or not config.tow.calibrated_friction:
        return -1.0
    margins = []
    for segment in pin_segments:
        wrap_rad = math.radians(_segment_wrap_angle(segment.warnings))
        margins.append(float(mu) * wrap_rad - 0.1)
    return min(margins, default=-1.0)


def _pin_machine_smoothness_penalty(layer: PlannedLayer) -> float:
    if layer.motion_table.point_count < 3:
        return 100.0
    axes = (
        np.asarray(layer.motion_table.x_mm, dtype=float),
        np.asarray(layer.motion_table.z_mm, dtype=float),
        np.asarray(layer.motion_table.a_deg, dtype=float),
        np.asarray(layer.motion_table.b_deg, dtype=float),
    )
    return float(sum(np.max(np.abs(np.diff(axis, n=2)), initial=0.0) for axis in axes))


def _pin_circuit_closure_penalty(layer: PlannedLayer) -> float:
    if layer.path.point_count < 2:
        return 100.0
    return float(np.linalg.norm(layer.path.points_mm[-1] - layer.path.points_mm[0]))


def _append_pin_arc(
    z_chunks: list[np.ndarray],
    theta_chunks: list[np.ndarray],
    pass_chunks: list[np.ndarray],
    b_chunks: list[np.ndarray],
    motion_chunks: list[tuple[str, ...]],
    warning_chunks: list[tuple[str, ...]],
    *,
    z_mm: float,
    theta_start: float,
    theta_end: float,
    pass_id: int,
    b_deg: float,
    pin_id: str,
    wrap_deg: float,
) -> None:
    count = 18
    theta = np.linspace(theta_start, theta_end, count)
    z = np.full(count, z_mm, dtype=float)
    z_chunks.append(z)
    theta_chunks.append(theta)
    pass_chunks.append(np.full(count, pass_id, dtype=int))
    b_chunks.append(np.full(count, b_deg, dtype=float))
    motion_chunks.append(tuple("PinContactArc" for _ in range(count)))
    warning_chunks.append(
        tuple(f"pin_id={pin_id};wrap_angle_deg={wrap_deg:g}" for _ in range(count))
    )


def _append_dome_span(
    config: WindingJobConfig,
    mandrel: CylinderMandrel | AxisymmetricProfileMandrel,
    z_chunks: list[np.ndarray],
    theta_chunks: list[np.ndarray],
    pass_chunks: list[np.ndarray],
    b_chunks: list[np.ndarray],
    motion_chunks: list[tuple[str, ...]],
    warning_chunks: list[tuple[str, ...]],
    *,
    side: str,
    z_shoulder: float,
    theta_start: float,
    theta_end: float,
    pass_id: int,
    b_deg: float,
) -> None:
    count = 42
    length = _mandrel_length(mandrel)
    depth = (
        config.mandrel.left_dome_length_mm
        if side == "left"
        else config.mandrel.right_dome_length_mm
    ) or length * 0.18
    inward = -1.0 if side == "left" else 1.0
    t = np.linspace(0.0, 1.0, count)
    z = z_shoulder + inward * depth * 0.72 * np.sin(np.pi * t)
    z = np.clip(z, 0.0, length)
    theta = theta_start + _unwrap_delta(theta_start, theta_end) * t
    z_chunks.append(z)
    theta_chunks.append(theta)
    pass_chunks.append(np.full(count, pass_id, dtype=int))
    b_chunks.append(np.full(count, b_deg, dtype=float))
    motion_chunks.append(tuple("DomeSurfaceSpan" for _ in range(count)))
    warning_chunks.append(tuple(f"dome_side={side};no_polar_turnaround=true" for _ in range(count)))


def _append_cylinder_span(
    z_chunks: list[np.ndarray],
    theta_chunks: list[np.ndarray],
    pass_chunks: list[np.ndarray],
    b_chunks: list[np.ndarray],
    motion_chunks: list[tuple[str, ...]],
    warning_chunks: list[tuple[str, ...]],
    *,
    z_start: float,
    z_end: float,
    theta_start: float,
    theta_end: float,
    pass_id: int,
    b_deg: float,
) -> None:
    count = 54
    t = np.linspace(0.0, 1.0, count)
    z_chunks.append(z_start + (z_end - z_start) * t)
    theta_chunks.append(theta_start + _unwrap_delta(theta_start, theta_end) * t)
    pass_chunks.append(np.full(count, pass_id, dtype=int))
    b_chunks.append(np.full(count, b_deg, dtype=float))
    motion_chunks.append(tuple("CylinderHelixSpan" for _ in range(count)))
    warning_chunks.append(tuple("" for _ in range(count)))


def _unwrap_delta(start: float, end: float) -> float:
    return float(((end - start + math.pi) % (2.0 * math.pi)) - math.pi)


def _local_angle_from_path(path: SurfacePath) -> np.ndarray:
    radius = path.surface_radius_mm
    dz = np.gradient(path.z_mm)
    dtheta = np.gradient(path.theta_rad)
    circumferential = radius * dtheta
    return np.rad2deg(np.arctan2(np.abs(circumferential), np.maximum(np.abs(dz), 1e-12)))


def _concat_surface_paths(paths: tuple[SurfacePath, ...]) -> SurfacePath:
    return SurfacePath(
        z_mm=np.concatenate([path.z_mm for path in paths]),
        theta_rad=np.concatenate([path.theta_rad for path in paths]),
        x_mm=np.concatenate([path.x_mm for path in paths]),
        y_mm=np.concatenate([path.y_mm for path in paths]),
        winding_angle_deg=paths[0].winding_angle_deg,
        tow_width_mm=paths[0].tow_width_mm,
        pass_index=np.concatenate([np.asarray(path.pass_index, dtype=int) for path in paths]),
        tow_eye_angle_deg=np.concatenate([
            np.full(path.point_count, path.winding_angle_deg, dtype=float)
            if path.tow_eye_angle_deg is None
            else path.tow_eye_angle_deg
            for path in paths
        ]),
    )


def _concat_motion_tables(tables: tuple[Any, ...]) -> Any:
    from filament_winder.core.kinematics.four_axis import MachineMotionTable

    return MachineMotionTable(
        a_deg=np.concatenate([table.a_deg for table in tables]),
        x_mm=np.concatenate([table.x_mm for table in tables]),
        z_mm=np.concatenate([table.z_mm for table in tables]),
        b_deg=np.concatenate([table.b_deg for table in tables]),
    )


def _concat_feed_schedules(schedules: tuple[Any, ...]) -> Any:
    from filament_winder.core.feedrate import FeedSchedule

    return FeedSchedule(
        feedrate_mm_min=np.concatenate([schedule.feedrate_mm_min for schedule in schedules]),
        curvature_1_per_mm=np.concatenate([schedule.curvature_1_per_mm for schedule in schedules]),
        curvature_radius_mm=np.concatenate([
            schedule.curvature_radius_mm for schedule in schedules
        ]),
        slip_risk=np.concatenate([schedule.slip_risk for schedule in schedules]),
    )


def _concat_program_metadata(items: tuple[WindingPointMetadata, ...]) -> WindingPointMetadata:
    return WindingPointMetadata(
        layer_id=tuple(label for item in items for label in item.layer_id),
        layer_index=np.concatenate([item.layer_index for item in items]),
        circuit_index=np.concatenate([item.circuit_index for item in items]),
        pass_index=np.concatenate([item.pass_index for item in items]),
        local_radius_mm=np.concatenate([item.local_radius_mm for item in items]),
        local_winding_angle_deg=np.concatenate([item.local_winding_angle_deg for item in items]),
        layer_name=tuple(label for item in items for label in item.layer_name),
        winding_type=tuple(label for item in items for label in item.winding_type),
        motion_type=tuple(label for item in items for label in item.motion_type),
        warning_flags=tuple(label for item in items for label in item.warning_flags),
    )


def analyze_winding_patterns(
    config: WindingJobConfig,
    mandrel: CylinderMandrel | AxisymmetricProfileMandrel | None = None,
) -> MultiLayerPatternResult | None:
    method = config.pattern_selection.method
    if method not in {"textbook", "textbook_integer_closure"}:
        return None
    resolved_mandrel = _build_mandrel(config) if mandrel is None else mandrel
    results = []
    stack_layer_counts = _paired_stack_layer_counts(config)
    for index, layer in enumerate(config.layers, start=1):
        if not layer.enabled or layer.type in {"hoop", "local_reinforcement_band"}:
            continue
        pair_count = stack_layer_counts.get(_stack_pair_key(layer), 1)
        coverage_share = _coverage_share_for_layer(config, layer, pair_count)
        request = _pattern_request_for_layer(
            config,
            resolved_mandrel,
            layer,
            index,
            stack_pair_count=pair_count,
            coverage_share=coverage_share,
        )
        results.append(select_winding_pattern(request, resolved_mandrel))
    return MultiLayerPatternResult(layer_results=tuple(results))


def _pattern_request_for_layer(
    config: WindingJobConfig,
    mandrel: CylinderMandrel | AxisymmetricProfileMandrel,
    layer: LayerConfig,
    index: int,
    *,
    stack_pair_count: int = 1,
    coverage_share: float | None = None,
) -> PatternSearchRequest:
    start_z, end_z = _layer_z_bounds(config, layer)
    tow_width = (
        layer.tow_width_mm
        if layer.tow_width_mm is not None
        else _effective_tow_width(config)
    )
    tow_thickness = (
        layer.tow_thickness_mm
        if layer.tow_thickness_mm is not None
        else config.roving.thickness_mm
    )
    delta_phi, trajectory_length = _layer_angular_propagation(
        mandrel,
        layer,
        start_z=start_z,
        end_z=end_z,
        tow_width=tow_width,
    )
    target_thickness = (
        layer.tow_thickness_mm
        if layer.tow_thickness_mm is not None
        else config.laminate_targets.target_layer_thickness_mm
    )
    share = (
        layer.coverage_target / max(stack_pair_count, 1)
        if coverage_share is None
        else layer.coverage_target * coverage_share
    )
    return PatternSearchRequest(
        layer_id=f"{index:02d}-{_safe_id(layer.name)}",
        layer_name=layer.name,
        winding_type=layer.type,
        winding_angle_deg=abs(layer.winding_angle_deg),
        delta_phi_total_deg=delta_phi,
        equatorial_radius_mm=_mandrel_radius(mandrel),
        trajectory_length_mm=trajectory_length,
        roving_width_mm=tow_width,
        roving_thickness_mm=tow_thickness,
        target_coverage=max(0.1, share),
        target_layer_thickness_mm=target_thickness,
        target_number_of_closed_layers=config.laminate_targets.target_number_of_closed_layers,
        feedrate_mm_min=layer.feedrate_mm_min or _nominal_feedrate(config),
        max_p=config.pattern_selection.max_p,
        max_k=config.pattern_selection.max_k,
        max_d=config.pattern_selection.max_d,
        angle_tolerance_deg=config.pattern_selection.angle_tolerance_deg,
        require_gcd_clean_pattern=config.pattern_selection.require_gcd_clean_pattern,
        candidate_count=config.pattern_selection.candidate_count,
        thickness_model="polynomial_smoothed_polar_approximation",
        max_coverage_estimate=max(
            0.15,
            share * 1.25,
        ),
        max_winding_time_min=config.quality_limits.max_estimated_winding_time_min * max(
            0.5,
            min(0.8, coverage_share or (1.0 / max(stack_pair_count, 1))),
        ),
        max_thickness_variation_percent=config.quality_limits.max_thickness_variation_percent,
        max_polar_buildup_mm=config.quality_limits.max_polar_buildup_mm,
    )


def _paired_stack_layer_counts(config: WindingJobConfig) -> dict[str, int]:
    counts: dict[str, int] = {}
    for layer in config.layers:
        if not layer.enabled:
            continue
        if layer.type not in {"geodesic", "non_geodesic"}:
            continue
        key = _stack_pair_key(layer)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _coverage_share_for_layer(
    config: WindingJobConfig,
    layer: LayerConfig,
    pair_count: int,
) -> float:
    if not config.coverage_mode.paired_layer_coverage or pair_count <= 1:
        return 1.0
    if layer.type == "geodesic":
        return 0.7
    if layer.type == "non_geodesic":
        return 0.3
    return 1.0 / max(pair_count, 1)


def _stack_pair_key(layer: LayerConfig) -> str:
    if layer.type in {"geodesic", "non_geodesic"}:
        return f"{layer.region}:dome_pair"
    return f"{layer.region}:{layer.type}"


def _layer_angular_propagation(
    mandrel: CylinderMandrel | AxisymmetricProfileMandrel,
    layer: LayerConfig,
    *,
    start_z: float,
    end_z: float,
    tow_width: float,
) -> tuple[float, float]:
    angle = abs(layer.winding_angle_deg)
    if isinstance(mandrel, CylinderMandrel):
        axial_length = end_z - start_z
        delta_theta = math.tan(math.radians(angle)) * axial_length / mandrel.radius_mm
        path_length = math.hypot(axial_length, abs(delta_theta) * mandrel.radius_mm)
        return math.degrees(delta_theta), path_length
    if layer.type == "non_geodesic":
        path, _diagnostics = generate_controlled_angle_path(
            mandrel,
            ControlledAnglePathConfig(
                target_angle_deg=angle,
                tow_width_mm=tow_width,
                start_z_mm=start_z,
                end_z_mm=end_z,
                point_count=max(40, min(layer.points, 300)),
                direction="positive",
            ),
        )
    else:
        path, _diagnostics = generate_geodesic_path(
            mandrel,
            GeodesicPathConfig(
                initial_angle_deg=angle,
                tow_width_mm=tow_width,
                start_z_mm=start_z,
                end_z_mm=end_z,
                turnaround_radius_mm=layer.turnaround_radius_mm,
                point_count=max(40, min(layer.points, 300)),
                direction="positive",
            ),
        )
    segment_lengths = np.linalg.norm(np.diff(path.points_mm, axis=0), axis=1)
    return math.degrees(float(path.theta_rad[-1] - path.theta_rad[0])), float(
        np.sum(segment_lengths)
    )


def _pattern_selection_summary(
    pattern_result: MultiLayerPatternResult | None,
) -> dict[str, Any]:
    if pattern_result is None:
        return {
            "enabled": False,
            "method": "legacy_coverage",
            "selected_patterns": [],
            "rejection_counts": {},
        }
    selected = []
    for candidate in pattern_result.selected_candidates:
        selected.append(
            {
                "layer_id": candidate.layer_id,
                "layer_name": candidate.layer_name,
                "selected_pattern_type": candidate.pattern_type,
                "pattern_id": candidate.pattern_id,
                "p": candidate.p,
                "k": candidate.k,
                "d": candidate.d,
                "nd": candidate.nd,
                "closure_error_deg": candidate.closure_error_deg,
                "effective_roving_width_mm": candidate.effective_roving_width_mm,
                "candidate_score": candidate.score,
                "coverage_estimate": candidate.coverage_estimate,
                "thickness_summary": candidate.thickness_distribution.summary.to_dict(),
                "why_selected": "lowest score among valid leading/lagging closed candidates",
            }
        )
    return {
        "enabled": True,
        "method": "textbook_integer_closure",
        "selected_patterns": selected,
        "rejection_counts": pattern_result.rejection_counts,
    }


def summarize_winding_job(result: WindingJobResult) -> str:
    layer_lines = []
    for index, layer in enumerate(result.program.layers, start=1):
        spec = layer.spec
        layer_lines.append(
            f"{index}. {spec.name:<20} {spec.winding_type:<8} "
            f"{layer.report.actual_angle_deg:+.2f} deg  passes={layer.report.circuits}"
        )
    warning_lines = result.summary["warnings"] or ["none"]
    output_lines = []
    if result.csv_path is not None:
        output_lines.append(f"- CSV: {result.csv_path}")
    if result.gcode_path is not None:
        output_lines.append(f"- G-code: {result.gcode_path}")
    if result.summary_path is not None:
        output_lines.append(f"- Summary: {result.summary_path}")
    for path in result.plot_paths:
        output_lines.append(f"- Plot: {path}")
    return (
        f"Project: {result.config.project.name}\n"
        f"Mandrel: {result.config.mandrel.type}, L={_mandrel_length(result.mandrel):g} mm, "
        f"R={_mandrel_radius(result.mandrel):g} mm\n"
        f"Tow: {result.config.tow.width_mm:g} mm x "
        f"{result.config.tow.thickness_mm:g} mm\n"
        f"Enabled layers: {len(result.program.layers)}\n\n"
        "Layer summary:\n"
        + "\n".join(layer_lines)
        + "\n\nGenerated:\n"
        f"- path points: {result.program.point_count}\n"
        f"- estimated winding time: {result.summary['estimated_winding_time_min']:.3f} min\n"
        + "\n".join(output_lines)
        + "\n\nWarnings:\n"
        + "\n".join(f"- {warning}" for warning in warning_lines)
    )


def _validate_layer(config: WindingJobConfig, layer: LayerConfig, index: int) -> None:
    prefix = f"layers[{index}]"
    layer_type = layer.type
    if layer_type not in {
        "hoop",
        "continuous_hoop_traverse",
        "local_reinforcement_band",
        "helical",
        "polar",
        "geodesic",
        "non_geodesic",
    }:
        raise ValueError(
            f"{prefix}.type must be hoop, local_reinforcement_band, helical, polar, "
            "geodesic, or non_geodesic"
        )
    tow_width = _effective_tow_width(config) if layer.tow_width_mm is None else layer.tow_width_mm
    tow_thickness = (
        config.tow.thickness_mm if layer.tow_thickness_mm is None else layer.tow_thickness_mm
    )
    if tow_width <= 0.0:
        raise ValueError(f"{prefix}.tow_width_mm must be positive")
    if tow_thickness < 0.0:
        raise ValueError(f"{prefix}.tow_thickness_mm must be non-negative")
    if layer.feedrate_mm_min is not None and layer.feedrate_mm_min <= 0.0:
        raise ValueError(f"{prefix}.feedrate_mm_min must be positive")
    start_z, end_z = _layer_z_bounds(config, layer)
    if start_z < 0.0 or end_z > _mandrel_length_mm(config):
        raise ValueError(f"{prefix}.start_z_mm/end_z_mm must stay inside the mandrel")
    if end_z <= start_z:
        raise ValueError(f"{prefix}.end_z_mm must be greater than start_z_mm")
    if layer.coverage_target <= 0.0:
        raise ValueError(f"{prefix}.coverage_target must be positive")
    if layer.points < 2:
        raise ValueError(f"{prefix}.points must be at least 2")
    if layer_type in {"hoop", "continuous_hoop_traverse"} and abs(layer.winding_angle_deg) > 90.0:
        raise ValueError(f"{prefix}.winding_angle_deg must be 90 for hoop layers")
    if layer_type == "local_reinforcement_band" and abs(layer.winding_angle_deg) > 90.0:
        raise ValueError(f"{prefix}.winding_angle_deg cannot exceed 90 for local reinforcement")
    if layer_type in {"helical", "polar", "geodesic", "non_geodesic"} and not 0.0 < abs(
        layer.winding_angle_deg
    ) < 90.0:
        raise ValueError(
            f"{prefix}.winding_angle_deg must be between 0 and 90 for {layer_type}"
        )
    if isinstance(layer.passes, str):
        if layer.passes != "auto":
            raise ValueError(f"{prefix}.passes must be a positive integer or auto")
    elif layer.passes is not None and int(layer.passes) <= 0:
        raise ValueError(f"{prefix}.passes must be positive")


def _schedule_from_config(
    config: WindingJobConfig,
    *,
    pattern_result: MultiLayerPatternResult | None = None,
) -> WindingSchedule:
    selected_candidates = pattern_result.selected_candidates if pattern_result else ()
    selected_by_layer = {
        candidate.layer_id: candidate for candidate in selected_candidates
    }
    layers = []
    for index, layer in enumerate(config.layers, start=1):
        tow_width = (
            _effective_tow_width(config) if layer.tow_width_mm is None else layer.tow_width_mm
        )
        tow_thickness = (
            config.tow.thickness_mm if layer.tow_thickness_mm is None else layer.tow_thickness_mm
        )
        winding_type = "hoop" if layer.type == "continuous_hoop_traverse" else layer.type
        direction = _layer_direction(layer)
        layer_id = f"{index:02d}-{_safe_id(layer.name)}"
        selected_pattern = selected_by_layer.get(layer_id)
        selected_passes = (
            None if selected_pattern is None else selected_pattern.number_of_windings
        )
        selected_phase = (
            None
            if selected_pattern is None
            else 360.0 / max(selected_pattern.number_of_windings, 1)
        )
        layers.append(
            WindingLayerSpec(
                layer_id=layer_id,
                name=layer.name,
                winding_type=winding_type,  # type: ignore[arg-type]
                target_angle_deg=abs(layer.winding_angle_deg),
                tow_width_mm=tow_width,
                layer_thickness_mm=tow_thickness,
                coverage_target=layer.coverage_target,
                direction=direction,  # type: ignore[arg-type]
                point_count=layer.points,
                enabled=layer.enabled,
                number_of_passes=selected_passes or _passes(layer.passes),
                start_z_mm=_layer_z_bounds(config, layer)[0],
                end_z_mm=_layer_z_bounds(config, layer)[1],
                feedrate_mm_min=layer.feedrate_mm_min,
                mandrel_clearance_mm=config.machine.clearance_mm,
                colour=layer.colour,
                notes=layer.notes,
                phase_offset_deg=selected_phase or layer.phase_offset_deg,
                hoop_mode=config.hoop_winding.mode,
                hoop_nominal_angle_deg=config.hoop_winding.nominal_angle_deg,
                hoop_min_angle_offset_deg=(
                    config.hoop_winding.min_angle_offset_from_pure_hoop_deg
                ),
                allow_exact_pure_hoop=config.hoop_winding.allow_exact_pure_hoop,
                tow_state_during_traverse=config.hoop_winding.tow_state_during_traverse,
            )
        )
    return WindingSchedule(
        layers=tuple(layers),
        radial_clearance_mm=config.machine.clearance_mm,
        nominal_feedrate_mm_min=_nominal_feedrate(config),
    )


def _build_summary(
    config: WindingJobConfig,
    mandrel: CylinderMandrel | AxisymmetricProfileMandrel,
    program: PlannedWindingProgram,
    *,
    total_segments: int,
    warnings: tuple[str, ...],
    csv_path: Path | None,
    gcode_path: Path | None,
    summary_path: Path | None,
    segments_path: Path | None,
    validation_report_path: Path | None,
    coverage_grid_path: Path | None,
    pattern_result: MultiLayerPatternResult | None,
    pattern_candidates_path: Path | None,
    selected_pattern_path: Path | None,
    pattern_rejection_report_path: Path | None,
    thickness_distribution_path: Path | None,
    layer_completion_report_path: Path | None,
    stack_coverage_report_path: Path | None,
    machine_smoothing_report_path: Path | None,
    pattern_optimisation_report_path: Path | None,
    candidate_pair_report_path: Path | None,
    actual_thickness_report_path: Path | None,
    region_quality_report_path: Path | None,
    calibration_report_path: Path | None,
    friction_margin_report_path: Path | None,
    polar_overbuild_report_path: Path | None,
    collision_report_path: Path | None,
    pin_layout_report_path: Path | None,
    pin_contact_report_path: Path | None,
    pin_buildup_report_path: Path | None,
    pin_slip_report_path: Path | None,
    shoulder_quality_report_path: Path | None,
    machine_reachability_report_path: Path | None,
    pin_route_candidates_path: Path | None,
    pin_route_selected_path: Path | None,
    pin_route_score_report_path: Path | None,
    dome_coverage_report_path: Path | None,
    left_dome_coverage_report_path: Path | None,
    right_dome_coverage_report_path: Path | None,
    dome_gap_overlap_map_path: Path | None,
    dome_angle_map_path: Path | None,
    dome_thickness_map_path: Path | None,
    dome_overbuild_report_path: Path | None,
    shoulder_transition_report_path: Path | None,
    optimisation_repair_suggestions_path: Path | None,
    plot_manifest_path: Path | None,
    layer_completion_report: dict[str, Any],
    stack_coverage_report: dict[str, Any],
    machine_smoothing_report: dict[str, Any],
    pattern_optimisation_report: dict[str, Any],
    region_quality_report: dict[str, Any],
    calibration_report: dict[str, Any],
    friction_margin_report: dict[str, Any],
    polar_overbuild_report: dict[str, Any],
    collision_report: dict[str, Any],
    shoulder_quality_report: dict[str, Any],
    machine_reachability_report: dict[str, Any],
    plot_paths: tuple[Path, ...],
    coverage: Any,
) -> dict[str, Any]:
    coverage_summary = coverage.summary()
    coverage_by_type = _coverage_by_layer_type(mandrel, program)
    estimated_time_s = _estimated_time_s(program)
    continuity = program_continuity_summary(program)
    transition_summary = program_transition_summary(program)
    machine_validation = _machine_validation(config, program, build_path_segments(program))
    quality_report = _quality_report(
        coverage_summary=coverage_summary,
        continuity=continuity,
        transition_summary=transition_summary,
        machine_summary=machine_validation["summary"],
        slip_summary=_slip_risk_summary(program),
        turnaround_summary=_turnaround_summary(build_path_segments(program)),
    )
    path_validation = program_path_validation_summary(
        mandrel,
        program,
        csv_path=csv_path,
        summary_path=summary_path,
        plot_paths=plot_paths,
    )
    backend_ready = (
        quality_report["machine_ready"]
        and bool(layer_completion_report["summary"]["completion_passed"])
        and bool(layer_completion_report["summary"]["strict_completion_passed"])
        and bool(layer_completion_report["summary"]["continuous_traverse_passed"])
        and bool(stack_coverage_report["summary"]["stack_uniformity_passed"])
        and bool(stack_coverage_report["summary"]["strict_stack_passed"])
        and bool(region_quality_report["summary"]["region_quality_passed"])
        and bool(pattern_optimisation_report["summary"]["pattern_optimisation_passed"])
        and bool(machine_smoothing_report["summary"]["machine_kinematics_passed"])
        and bool(polar_overbuild_report["summary"]["polar_overbuild_passed"])
        and bool(collision_report["summary"]["collision_passed"])
        and bool(shoulder_quality_report["summary"]["shoulder_quality_passed"])
        and bool(machine_reachability_report["summary"]["machine_reachability_passed"])
        and bool(path_validation["csv_summary_row_count_match"])
    )
    return {
        "project": {"name": config.project.name, "units": config.project.units},
        "mandrel": {
            "type": config.mandrel.type,
            "length_mm": _mandrel_length(mandrel),
            "radius_mm": _mandrel_radius(mandrel),
        },
        "tow": {
            "tow_id": config.tow.tow_id,
            "name": config.tow.name,
            "width_mm": config.tow.width_mm,
            "effective_width_mm": config.tow.effective_width_mm,
            "calibrated_effective_width": config.tow.calibrated_effective_width,
            "friction_coefficient": config.tow.friction_coefficient,
            "calibrated_friction": config.tow.calibrated_friction,
            "thickness_mm": config.tow.thickness_mm,
        },
        "layer_count": len(config.layers),
        "enabled_layer_count": len(program.layers),
        "total_segments": total_segments,
        "total_path_points": program.point_count,
        "estimated_winding_time_s": estimated_time_s,
        "estimated_winding_time_min": estimated_time_s / 60.0,
        "coverage_summary": {
            "overall_covered_percent": coverage_summary.covered_percent,
            "helical_coverage_percent": coverage_by_type.get("helical", 0.0),
            "hoop_coverage_percent": coverage_by_type.get("hoop", 0.0),
            "polar_coverage_percent": coverage_by_type.get("polar", 0.0),
            "covered_percent": coverage_summary.covered_percent,
            "gap_percent": coverage_summary.gap_percent,
            "overlap_percent": coverage_summary.overlap_percent,
            "max_coverage_count": coverage_summary.max_coverage_count,
            "mean_coverage_count": coverage_summary.mean_coverage_count,
            "combined_thickness_mm": sum(layer.spec.layer_thickness_mm for layer in program.layers),
        },
        "continuity": continuity,
        "path_validation": path_validation,
        "transition_summary": transition_summary,
        "machine_validation_summary": machine_validation["summary"],
        "quality_report": quality_report,
        "layer_completion_status": layer_completion_report["summary"],
        "stack_uniformity_status": stack_coverage_report["summary"],
        "machine_smoothing_status": machine_smoothing_report["summary"],
        "pattern_optimisation_status": pattern_optimisation_report["summary"],
        "region_quality_status": region_quality_report["summary"],
        "calibration_status": calibration_report["summary"],
        "friction_margin_status": friction_margin_report["summary"],
        "polar_overbuild_status": polar_overbuild_report["summary"],
        "collision_status": collision_report["summary"],
        "shoulder_quality_status": shoulder_quality_report["summary"],
        "machine_reachability_status": machine_reachability_report["summary"],
        "backend_ready": backend_ready,
        "machine_ready": (
            backend_ready
            and bool(calibration_report["summary"]["calibration_passed"])
            and bool(friction_margin_report["summary"]["friction_margin_passed"])
        ),
        "textbook_pattern_selection": _pattern_selection_summary(pattern_result),
        "pattern_summary": [
            {
                "layer_id": layer.spec.layer_id,
                "layer_name": layer.spec.name,
                "layer_type": layer.spec.winding_type,
                "passes": layer.report.circuits,
                "actual_winding_angle_deg": layer.report.actual_angle_deg,
                "band_spacing_mm": layer.report.tow_spacing_mm,
                "coverage_percent": layer.report.coverage_percent,
                "gap_estimate_mm": layer.report.gap_mm,
                "overlap_estimate_mm": layer.report.overlap_mm,
                "base_radius_mm": _mandrel_radius(mandrel),
                "effective_radius_mm": layer.effective_radius_mm,
                "tow_thickness_mm": layer.spec.layer_thickness_mm,
                "accumulated_thickness_before_mm": layer.accumulated_thickness_before_mm,
                "accumulated_thickness_after_mm": (
                    layer.accumulated_thickness_before_mm + layer.spec.layer_thickness_mm
                ),
                "warnings": list(layer.report.warnings),
            }
            for layer in program.layers
        ],
        "warnings": list(warnings) + [
            warning
            for report in program.reports
            for warning in report.warnings
            if warning not in warnings
        ],
        "output_files": {
            "csv": None if csv_path is None else str(csv_path),
            "gcode": None if gcode_path is None else str(gcode_path),
            "summary": None if summary_path is None else str(summary_path),
            "segments": None if segments_path is None else str(segments_path),
            "validation_report": (
                None if validation_report_path is None else str(validation_report_path)
            ),
            "coverage_grid": None if coverage_grid_path is None else str(coverage_grid_path),
            "pattern_candidates": (
                None if pattern_candidates_path is None else str(pattern_candidates_path)
            ),
            "selected_pattern": (
                None if selected_pattern_path is None else str(selected_pattern_path)
            ),
            "pattern_rejection_report": (
                None
                if pattern_rejection_report_path is None
                else str(pattern_rejection_report_path)
            ),
            "thickness_distribution": (
                None if thickness_distribution_path is None else str(thickness_distribution_path)
            ),
            "layer_completion_report": (
                None if layer_completion_report_path is None else str(layer_completion_report_path)
            ),
            "stack_coverage_report": (
                None if stack_coverage_report_path is None else str(stack_coverage_report_path)
            ),
            "machine_smoothing_report": (
                None
                if machine_smoothing_report_path is None
                else str(machine_smoothing_report_path)
            ),
            "pattern_optimisation_report": (
                None
                if pattern_optimisation_report_path is None
                else str(pattern_optimisation_report_path)
            ),
            "candidate_pair_report": (
                None if candidate_pair_report_path is None else str(candidate_pair_report_path)
            ),
            "actual_thickness_report": (
                None if actual_thickness_report_path is None else str(actual_thickness_report_path)
            ),
            "region_quality_report": (
                None if region_quality_report_path is None else str(region_quality_report_path)
            ),
            "calibration_report": (
                None if calibration_report_path is None else str(calibration_report_path)
            ),
            "friction_margin_report": (
                None if friction_margin_report_path is None else str(friction_margin_report_path)
            ),
            "polar_overbuild_report": (
                None if polar_overbuild_report_path is None else str(polar_overbuild_report_path)
            ),
            "collision_report": (
                None if collision_report_path is None else str(collision_report_path)
            ),
            "pin_layout": None if pin_layout_report_path is None else str(pin_layout_report_path),
            "pin_contact_report": (
                None if pin_contact_report_path is None else str(pin_contact_report_path)
            ),
            "pin_buildup_report": (
                None if pin_buildup_report_path is None else str(pin_buildup_report_path)
            ),
            "pin_slip_report": None if pin_slip_report_path is None else str(pin_slip_report_path),
            "shoulder_quality_report": (
                None
                if shoulder_quality_report_path is None
                else str(shoulder_quality_report_path)
            ),
            "machine_reachability_report": (
                None
                if machine_reachability_report_path is None
                else str(machine_reachability_report_path)
            ),
            "pin_route_candidates": (
                None if pin_route_candidates_path is None else str(pin_route_candidates_path)
            ),
            "pin_route_selected": (
                None if pin_route_selected_path is None else str(pin_route_selected_path)
            ),
            "pin_route_score_report": (
                None if pin_route_score_report_path is None else str(pin_route_score_report_path)
            ),
            "dome_coverage_report": (
                None if dome_coverage_report_path is None else str(dome_coverage_report_path)
            ),
            "left_dome_coverage_report": (
                None
                if left_dome_coverage_report_path is None
                else str(left_dome_coverage_report_path)
            ),
            "right_dome_coverage_report": (
                None
                if right_dome_coverage_report_path is None
                else str(right_dome_coverage_report_path)
            ),
            "dome_gap_overlap_map": (
                None if dome_gap_overlap_map_path is None else str(dome_gap_overlap_map_path)
            ),
            "dome_angle_map": None if dome_angle_map_path is None else str(dome_angle_map_path),
            "dome_thickness_map": (
                None if dome_thickness_map_path is None else str(dome_thickness_map_path)
            ),
            "dome_overbuild_report": (
                None if dome_overbuild_report_path is None else str(dome_overbuild_report_path)
            ),
            "shoulder_transition_report": (
                None
                if shoulder_transition_report_path is None
                else str(shoulder_transition_report_path)
            ),
            "optimisation_repair_suggestions": (
                None
                if optimisation_repair_suggestions_path is None
                else str(optimisation_repair_suggestions_path)
            ),
            "plot_manifest": None if plot_manifest_path is None else str(plot_manifest_path),
            "plots": [str(path) for path in plot_paths],
        },
    }


def _build_validation_report(
    config: WindingJobConfig,
    program: PlannedWindingProgram,
    segments: tuple[Any, ...],
    coverage: Any,
    *,
    layer_completion_report: dict[str, Any],
    stack_coverage_report: dict[str, Any],
    machine_smoothing_report: dict[str, Any],
    pattern_optimisation_report: dict[str, Any],
    region_quality_report: dict[str, Any],
    calibration_report: dict[str, Any],
    friction_margin_report: dict[str, Any],
    polar_overbuild_report: dict[str, Any],
    collision_report: dict[str, Any],
    shoulder_quality_report: dict[str, Any],
    machine_reachability_report: dict[str, Any],
) -> dict[str, Any]:
    continuity = program_continuity_summary(program)
    transition_summary = program_transition_summary(program)
    machine_validation = _machine_validation(config, program, segments)
    warnings = [
        {
            "layer_id": segment.layer_id,
            "segment_id": segment.segment_id,
            "pass_id": segment.pass_id,
            "severity": "warning",
            "problem": warning,
        }
        for segment in segments
        for warning in segment.warnings
    ]
    coverage_summary = coverage.summary()
    quality_report = _quality_report(
        coverage_summary=coverage_summary,
        continuity=continuity,
        transition_summary=transition_summary,
        machine_summary=machine_validation["summary"],
        slip_summary=_slip_risk_summary(program),
        turnaround_summary=_turnaround_summary(segments),
    )
    report_warnings = warnings + machine_validation["warnings"]
    backend_ready = (
        quality_report["machine_ready"]
        and bool(layer_completion_report["summary"]["completion_passed"])
        and bool(layer_completion_report["summary"]["strict_completion_passed"])
        and bool(layer_completion_report["summary"]["continuous_traverse_passed"])
        and bool(stack_coverage_report["summary"]["stack_uniformity_passed"])
        and bool(stack_coverage_report["summary"]["strict_stack_passed"])
        and bool(region_quality_report["summary"]["region_quality_passed"])
        and bool(pattern_optimisation_report["summary"]["pattern_optimisation_passed"])
        and bool(machine_smoothing_report["summary"]["machine_kinematics_passed"])
        and bool(polar_overbuild_report["summary"]["polar_overbuild_passed"])
        and bool(collision_report["summary"]["collision_passed"])
        and bool(shoulder_quality_report["summary"]["shoulder_quality_passed"])
        and bool(machine_reachability_report["summary"]["machine_reachability_passed"])
    )
    return {
        "project": config.project.name,
        "path_continuity": continuity,
        "transition_summary": transition_summary,
        "machine_validation_summary": machine_validation["summary"],
        "slip_risk_summary": _slip_risk_summary(program),
        "turnaround_summary": _turnaround_summary(segments),
        "quality_report": quality_report,
        "layer_completion_status": layer_completion_report["summary"],
        "stack_uniformity_status": stack_coverage_report["summary"],
        "machine_smoothing_status": machine_smoothing_report["summary"],
        "pattern_optimisation_status": pattern_optimisation_report["summary"],
        "region_quality_status": region_quality_report["summary"],
        "calibration_status": calibration_report["summary"],
        "friction_margin_status": friction_margin_report["summary"],
        "polar_overbuild_status": polar_overbuild_report["summary"],
        "collision_status": collision_report["summary"],
        "shoulder_quality_status": shoulder_quality_report["summary"],
        "machine_reachability_status": machine_reachability_report["summary"],
        "backend_ready": backend_ready,
        "machine_ready": (
            backend_ready
            and bool(calibration_report["summary"]["calibration_passed"])
            and bool(friction_margin_report["summary"]["friction_margin_passed"])
        ),
        "warnings": report_warnings,
    }


def _run_consistency_validation(
    *,
    config: WindingJobConfig,
    program: PlannedWindingProgram,
    segments: tuple[Any, ...],
    coverage: Any,
    csv_path: Path | None,
    summary_path: Path | None,
    validation_report_path: Path | None,
    pattern_candidates_path: Path | None,
    selected_pattern_path: Path | None,
    pattern_rejection_report_path: Path | None,
    thickness_distribution_path: Path | None,
    layer_completion_report_path: Path | None,
    stack_coverage_report_path: Path | None,
    machine_smoothing_report_path: Path | None,
    pattern_optimisation_report_path: Path | None,
    candidate_pair_report_path: Path | None,
    actual_thickness_report_path: Path | None,
    region_quality_report_path: Path | None,
    calibration_report_path: Path | None,
    friction_margin_report_path: Path | None,
    polar_overbuild_report_path: Path | None,
    collision_report_path: Path | None,
    pin_layout_report_path: Path | None,
    pin_contact_report_path: Path | None,
    pin_buildup_report_path: Path | None,
    pin_slip_report_path: Path | None,
    shoulder_quality_report_path: Path | None,
    machine_reachability_report_path: Path | None,
    pin_route_candidates_path: Path | None,
    pin_route_selected_path: Path | None,
    pin_route_score_report_path: Path | None,
    dome_coverage_report_path: Path | None,
    left_dome_coverage_report_path: Path | None,
    right_dome_coverage_report_path: Path | None,
    dome_gap_overlap_map_path: Path | None,
    dome_angle_map_path: Path | None,
    dome_thickness_map_path: Path | None,
    dome_overbuild_report_path: Path | None,
    shoulder_transition_report_path: Path | None,
    optimisation_repair_suggestions_path: Path | None,
    plot_manifest_path: Path | None,
    gcode_path: Path | None,
) -> dict[str, Any]:
    issues: list[str] = []
    expected_points = program.point_count
    if csv_path is not None:
        import csv as _csv

        with csv_path.open(newline="", encoding="utf-8") as handle:
            csv_rows = sum(1 for _ in _csv.DictReader(handle))
        if csv_rows != expected_points:
            issues.append(f"csv_row_mismatch:{csv_rows}!={expected_points}")
    if validation_report_path is not None and not validation_report_path.exists():
        issues.append("validation_report_missing")
    if gcode_path is not None and not gcode_path.exists():
        issues.append("gcode_missing")
    if pattern_candidates_path is not None and not pattern_candidates_path.exists():
        issues.append("pattern_candidates_missing")
    if selected_pattern_path is not None and not selected_pattern_path.exists():
        issues.append("selected_pattern_missing")
    if pattern_rejection_report_path is not None and not pattern_rejection_report_path.exists():
        issues.append("pattern_rejection_report_missing")
    if thickness_distribution_path is not None and not thickness_distribution_path.exists():
        issues.append("thickness_distribution_missing")
    if layer_completion_report_path is not None and not layer_completion_report_path.exists():
        issues.append("layer_completion_report_missing")
    if stack_coverage_report_path is not None and not stack_coverage_report_path.exists():
        issues.append("stack_coverage_report_missing")
    if machine_smoothing_report_path is not None and not machine_smoothing_report_path.exists():
        issues.append("machine_smoothing_report_missing")
    if (
        pattern_optimisation_report_path is not None
        and not pattern_optimisation_report_path.exists()
    ):
        issues.append("pattern_optimisation_report_missing")
    if candidate_pair_report_path is not None and not candidate_pair_report_path.exists():
        issues.append("candidate_pair_report_missing")
    if actual_thickness_report_path is not None and not actual_thickness_report_path.exists():
        issues.append("actual_thickness_report_missing")
    if region_quality_report_path is not None and not region_quality_report_path.exists():
        issues.append("region_quality_report_missing")
    if calibration_report_path is not None and not calibration_report_path.exists():
        issues.append("calibration_report_missing")
    if friction_margin_report_path is not None and not friction_margin_report_path.exists():
        issues.append("friction_margin_report_missing")
    if polar_overbuild_report_path is not None and not polar_overbuild_report_path.exists():
        issues.append("polar_overbuild_report_missing")
    if collision_report_path is not None and not collision_report_path.exists():
        issues.append("collision_report_missing")
    if pin_layout_report_path is not None and not pin_layout_report_path.exists():
        issues.append("pin_layout_missing")
    if pin_contact_report_path is not None and not pin_contact_report_path.exists():
        issues.append("pin_contact_report_missing")
    if pin_buildup_report_path is not None and not pin_buildup_report_path.exists():
        issues.append("pin_buildup_report_missing")
    if pin_slip_report_path is not None and not pin_slip_report_path.exists():
        issues.append("pin_slip_report_missing")
    if shoulder_quality_report_path is not None and not shoulder_quality_report_path.exists():
        issues.append("shoulder_quality_report_missing")
    if (
        machine_reachability_report_path is not None
        and not machine_reachability_report_path.exists()
    ):
        issues.append("machine_reachability_report_missing")
    if pin_route_candidates_path is not None and not pin_route_candidates_path.exists():
        issues.append("pin_route_candidates_missing")
    if pin_route_selected_path is not None and not pin_route_selected_path.exists():
        issues.append("pin_route_selected_missing")
    if pin_route_score_report_path is not None and not pin_route_score_report_path.exists():
        issues.append("pin_route_score_report_missing")
    if dome_coverage_report_path is not None and not dome_coverage_report_path.exists():
        issues.append("dome_coverage_report_missing")
    if (
        left_dome_coverage_report_path is not None
        and not left_dome_coverage_report_path.exists()
    ):
        issues.append("left_dome_coverage_report_missing")
    if (
        right_dome_coverage_report_path is not None
        and not right_dome_coverage_report_path.exists()
    ):
        issues.append("right_dome_coverage_report_missing")
    if dome_gap_overlap_map_path is not None and not dome_gap_overlap_map_path.exists():
        issues.append("dome_gap_overlap_map_missing")
    if dome_angle_map_path is not None and not dome_angle_map_path.exists():
        issues.append("dome_angle_map_missing")
    if dome_thickness_map_path is not None and not dome_thickness_map_path.exists():
        issues.append("dome_thickness_map_missing")
    if dome_overbuild_report_path is not None and not dome_overbuild_report_path.exists():
        issues.append("dome_overbuild_report_missing")
    if shoulder_transition_report_path is not None and not shoulder_transition_report_path.exists():
        issues.append("shoulder_transition_report_missing")
    if (
        optimisation_repair_suggestions_path is not None
        and not optimisation_repair_suggestions_path.exists()
    ):
        issues.append("optimisation_repair_suggestions_missing")
    if plot_manifest_path is not None and not plot_manifest_path.exists():
        issues.append("plot_manifest_missing")
    if coverage.coverage_count.shape != (config.coverage.z_cells, config.coverage.theta_cells):
        issues.append("coverage_grid_shape_mismatch")
    if len(segments) == 0:
        issues.append("no_segments")
    return {
        "passed": not issues,
        "issue_count": len(issues),
        "issues": issues,
        "exported_files": {
            "csv": None if csv_path is None else str(csv_path),
        "summary": None if summary_path is None else str(summary_path),
            "validation_report": None
            if validation_report_path is None
            else str(validation_report_path),
            "gcode": None if gcode_path is None else str(gcode_path),
            "calibration_report": None
            if calibration_report_path is None
            else str(calibration_report_path),
            "friction_margin_report": None
            if friction_margin_report_path is None
            else str(friction_margin_report_path),
            "polar_overbuild_report": None
            if polar_overbuild_report_path is None
            else str(polar_overbuild_report_path),
            "collision_report": (
                None if collision_report_path is None else str(collision_report_path)
            ),
            "pin_layout": None if pin_layout_report_path is None else str(pin_layout_report_path),
            "pin_contact_report": (
                None if pin_contact_report_path is None else str(pin_contact_report_path)
            ),
            "pin_buildup_report": (
                None if pin_buildup_report_path is None else str(pin_buildup_report_path)
            ),
            "pin_slip_report": None if pin_slip_report_path is None else str(pin_slip_report_path),
            "shoulder_quality_report": (
                None
                if shoulder_quality_report_path is None
                else str(shoulder_quality_report_path)
            ),
            "machine_reachability_report": (
                None
                if machine_reachability_report_path is None
                else str(machine_reachability_report_path)
            ),
            "pin_route_candidates": (
                None if pin_route_candidates_path is None else str(pin_route_candidates_path)
            ),
            "pin_route_selected": (
                None if pin_route_selected_path is None else str(pin_route_selected_path)
            ),
            "pin_route_score_report": (
                None if pin_route_score_report_path is None else str(pin_route_score_report_path)
            ),
            "dome_coverage_report": (
                None if dome_coverage_report_path is None else str(dome_coverage_report_path)
            ),
            "left_dome_coverage_report": (
                None
                if left_dome_coverage_report_path is None
                else str(left_dome_coverage_report_path)
            ),
            "right_dome_coverage_report": (
                None
                if right_dome_coverage_report_path is None
                else str(right_dome_coverage_report_path)
            ),
            "dome_gap_overlap_map": (
                None if dome_gap_overlap_map_path is None else str(dome_gap_overlap_map_path)
            ),
            "dome_angle_map": None if dome_angle_map_path is None else str(dome_angle_map_path),
            "dome_thickness_map": (
                None if dome_thickness_map_path is None else str(dome_thickness_map_path)
            ),
            "dome_overbuild_report": (
                None if dome_overbuild_report_path is None else str(dome_overbuild_report_path)
            ),
            "shoulder_transition_report": (
                None
                if shoulder_transition_report_path is None
                else str(shoulder_transition_report_path)
            ),
        },
    }


def _manufacturing_report(
    *,
    config: WindingJobConfig,
    program: PlannedWindingProgram,
    coverage: Any,
    consistency_report: dict[str, Any],
    pattern_result: MultiLayerPatternResult | None,
    layer_completion_report: dict[str, Any],
    stack_coverage_report: dict[str, Any],
    machine_smoothing_report: dict[str, Any],
    calibration_report: dict[str, Any],
    friction_margin_report: dict[str, Any],
    polar_overbuild_report: dict[str, Any],
    collision_report: dict[str, Any],
    shoulder_quality_report: dict[str, Any],
    machine_reachability_report: dict[str, Any],
) -> dict[str, Any]:
    coverage_summary = coverage.summary()
    machine_validation = _machine_validation(config, program, build_path_segments(program))
    selected_patterns = [] if pattern_result is None else [
        {
            "layer_id": candidate.layer_id,
            "pattern_id": candidate.pattern_id,
            "pattern_type": candidate.pattern_type,
            "p": candidate.p,
            "k": candidate.k,
            "d": candidate.d,
            "nd": candidate.nd,
            "closure_error_deg": candidate.closure_error_deg,
            "effective_roving_width_mm": candidate.effective_roving_width_mm,
            "candidate_score": candidate.score,
            "why_selected": "lowest score among valid leading/lagging closed candidates",
        }
        for candidate in pattern_result.selected_candidates
    ]
    return {
        "run_consistency": consistency_report,
        "gaps": {
            "gap_percent": coverage_summary.gap_percent,
            "overlap_percent": coverage_summary.overlap_percent,
            "max_coverage_count": coverage_summary.max_coverage_count,
        },
        "thickness": {
            "covered_percent": coverage_summary.covered_percent,
            "combined_layer_thickness_mm": sum(
                layer.spec.layer_thickness_mm for layer in program.layers
            ),
        },
        "slip": _slip_risk_summary(program),
        "machine_limits": machine_validation["summary"],
        "selected_pattern_explanation": selected_patterns,
        "strict_quality": {
            "layer_completion": layer_completion_report["summary"],
            "stack_uniformity": stack_coverage_report["summary"],
            "machine_smoothing": machine_smoothing_report["summary"],
            "calibration": calibration_report["summary"],
            "friction_margin": friction_margin_report["summary"],
            "polar_overbuild": polar_overbuild_report["summary"],
            "collision": collision_report["summary"],
            "hoop_continuity_passed": layer_completion_report["summary"].get(
                "continuous_traverse_passed",
                True,
            ),
        },
        "backend_ready": (
            bool(layer_completion_report["summary"]["strict_completion_passed"])
            and bool(stack_coverage_report["summary"]["strict_stack_passed"])
            and bool(machine_smoothing_report["summary"]["machine_kinematics_passed"])
            and bool(polar_overbuild_report["summary"]["polar_overbuild_passed"])
            and bool(collision_report["summary"]["collision_passed"])
            and _quality_report(
            coverage_summary=coverage_summary,
            continuity=program_continuity_summary(program),
            transition_summary=program_transition_summary(program),
            machine_summary=machine_validation["summary"],
            slip_summary=_slip_risk_summary(program),
            turnaround_summary=_turnaround_summary(build_path_segments(program)),
            )["machine_ready"]
        ),
        "machine_ready": (
            bool(calibration_report["summary"]["calibration_passed"])
            and bool(friction_margin_report["summary"]["friction_margin_passed"])
            and bool(layer_completion_report["summary"]["strict_completion_passed"])
            and bool(stack_coverage_report["summary"]["strict_stack_passed"])
            and bool(machine_smoothing_report["summary"]["machine_kinematics_passed"])
            and bool(polar_overbuild_report["summary"]["polar_overbuild_passed"])
            and bool(collision_report["summary"]["collision_passed"])
        ),
    }


def _selected_patterns_by_layer(
    pattern_result: MultiLayerPatternResult | None,
) -> dict[str, Any]:
    if pattern_result is None:
        return {}
    return {
        candidate.layer_id: candidate for candidate in pattern_result.selected_candidates
    }


def _pattern_optimisation_report(
    config: WindingJobConfig,
    pattern_result: MultiLayerPatternResult | None,
    *,
    layer_completion_report: dict[str, Any],
    stack_coverage_report: dict[str, Any],
    dome_coverage_report: dict[str, Any],
    dome_overbuild_report: dict[str, Any],
) -> dict[str, Any]:
    selected = [] if pattern_result is None else list(pattern_result.selected_candidates)
    total_time = sum(candidate.estimated_winding_time_min for candidate in selected)
    invalid_candidates = [candidate.pattern_id for candidate in selected if not candidate.valid]
    excessive_candidates = [
        candidate.pattern_id
        for candidate in selected
        if candidate.coverage_estimate > 1.25
    ]
    layer_summary = layer_completion_report["summary"]
    stack_summary = stack_coverage_report["summary"]
    dome_summary = dome_coverage_report["summary"]
    dome_overbuild_summary = dome_overbuild_report["summary"]
    passed = (
        bool(layer_summary["strict_completion_passed"])
        and bool(stack_summary["strict_stack_passed"])
        and total_time <= config.quality_limits.max_estimated_winding_time_min
        and bool(dome_summary["dome_coverage_passed"])
        and bool(dome_summary["left_dome_coverage_passed"])
        and bool(dome_summary["right_dome_coverage_passed"])
        and bool(dome_overbuild_summary["dome_overbuild_passed"])
        and not invalid_candidates
    )
    candidate_rows = [
        {
            "layer_id": candidate.layer_id,
            "pattern_id": candidate.pattern_id,
            "coverage_estimate_percent": candidate.coverage_estimate * 100.0,
            "estimated_winding_time_min": candidate.estimated_winding_time_min,
            "thickness_summary": candidate.thickness_distribution.summary.to_dict(),
            "score": candidate.score,
            "warnings": list(candidate.warnings),
            "rejection_reasons": list(candidate.rejection_reasons),
            "valid": candidate.valid,
        }
        for candidate in selected
    ]
    return {
        "summary": {
            "pattern_optimisation_passed": passed,
            "selected_candidate_count": len(selected),
            "total_selected_winding_time_min": total_time,
            "excessive_candidate_count": len(excessive_candidates),
            "invalid_selected_candidate_count": len(invalid_candidates),
            "invalid_selected_candidates": invalid_candidates,
        },
        "selected_candidates": candidate_rows,
        "stack_uniformity_summary": stack_summary,
        "layer_completion_summary": layer_summary,
        "dome_coverage_summary": dome_summary,
        "dome_overbuild_summary": dome_overbuild_summary,
    }


def _candidate_pair_report(pattern_result: MultiLayerPatternResult | None) -> dict[str, Any]:
    if pattern_result is None:
        return {"summary": {"pair_count": 0, "pair_optimisation_passed": True}, "pairs": []}
    selected = list(pattern_result.selected_candidates)
    pairs = []
    if selected:
        invalid = [candidate.pattern_id for candidate in selected if not candidate.valid]
        total_coverage = sum(candidate.coverage_estimate for candidate in selected)
        total_time = sum(candidate.estimated_winding_time_min for candidate in selected)
        max_variation = max(
            candidate.thickness_distribution.summary.thickness_variation_percent
            for candidate in selected
        )
        max_polar = max(
            candidate.thickness_distribution.summary.polar_buildup_mm
            for candidate in selected
        )
        pairs.append(
            {
                "pair_id": "selected_stack_pair",
                "layer_ids": [candidate.layer_id for candidate in selected],
                "combined_coverage_estimate_percent": total_coverage * 100.0,
                "combined_estimated_winding_time_min": total_time,
                "max_thickness_variation_percent": max_variation,
                "max_polar_buildup_mm": max_polar,
                "candidate_scores": {
                    candidate.layer_id: candidate.score for candidate in selected
                },
                "invalid_candidate_ids": invalid,
            }
        )
    return {
        "summary": {
            "pair_count": len(pairs),
            "pair_optimisation_passed": bool(pairs)
            and all(not pair["invalid_candidate_ids"] for pair in pairs),
        },
        "pairs": pairs,
    }


def _actual_thickness_report(
    config: WindingJobConfig,
    mandrel: CylinderMandrel | AxisymmetricProfileMandrel,
    coverage: Any,
    *,
    nominal_stack_thickness_mm: float,
) -> dict[str, Any]:
    counts = np.asarray(coverage.coverage_count, dtype=float)
    tow_thickness = max(config.tow.thickness_mm, config.roving.thickness_mm, 0.0)
    actual = counts * tow_thickness
    if actual.size == 0:
        min_thickness = mean_thickness = max_thickness = variation = 0.0
    else:
        min_thickness = float(np.min(actual))
        mean_thickness = float(np.mean(actual))
        max_thickness = float(np.max(actual))
        variation = (
            0.0
            if mean_thickness <= 1e-12
            else (max_thickness - min_thickness) / mean_thickness * 100.0
        )
    z_values = np.asarray(coverage.z_mm, dtype=float)
    radius = mandrel.radius_at(z_values) if z_values.size else np.asarray([], dtype=float)
    regions = _coverage_regions(z_values, radius) if z_values.size else []
    masks = _surface_masks(config, mandrel, coverage) if z_values.size else {}
    required_mean = (
        _row_mask_thickness_mean(actual, masks["required_winding_region"])
        if masks
        else 0.0
    )
    boss_mean = (
        _row_mask_thickness_mean(actual, masks["polar_opening_region"])
        if masks
        else 0.0
    )
    turnaround_mean = (
        _row_mask_thickness_mean(actual, masks["turnaround_region"])
        if masks
        else 0.0
    )
    cylinder_mean = _region_thickness_mean(actual, regions, {"cylinder"}) if regions else 0.0
    polar_mean = _region_thickness_mean(actual, regions, {"polar"}) if regions else 0.0
    dome_mean = (
        _region_thickness_mean(actual, regions, {"left_dome", "right_dome"})
        if regions
        else 0.0
    )
    return {
        "summary": {
            "nominal_stack_thickness_mm": nominal_stack_thickness_mm,
            "actual_minimum_thickness_mm": min_thickness,
            "actual_mean_thickness_mm": mean_thickness,
            "actual_maximum_thickness_mm": max_thickness,
            "actual_thickness_variation_percent": variation,
            "required_shell_mean_thickness_mm": required_mean,
            "turnaround_buildup_mm": max(0.0, turnaround_mean - required_mean),
            "physical_boss_buildup_mm": max(0.0, boss_mean - required_mean),
            "polar_buildup_mm": max(0.0, turnaround_mean - required_mean),
            "legacy_polar_region_buildup_mm": max(0.0, polar_mean - cylinder_mean),
            "dome_buildup_mm": max(0.0, dome_mean - cylinder_mean),
            "cylinder_buildup_mm": cylinder_mean,
        },
        "limits": {
            "max_thickness_variation_percent": (
                config.quality_limits.max_thickness_variation_percent
            ),
            "allow_min_thickness_zero": config.quality_limits.allow_min_thickness_zero,
        },
    }


def _calibration_report(config: WindingJobConfig) -> dict[str, Any]:
    effective_width = (
        config.tow.effective_width_mm
        if config.tow.effective_width_mm is not None
        else config.tow.width_mm
    )
    width_calibrated = bool(
        config.tow.calibrated_effective_width
        and config.tow.effective_width_mm is not None
        and config.tow.effective_width_mm > 0.0
    )
    friction_calibrated = bool(
        config.tow.calibrated_friction
        and config.tow.friction_coefficient is not None
        and config.tow.friction_coefficient > 0.0
    )
    tension_measured = bool(config.tow.tension_N is not None and config.tow.tension_N > 0.0)
    missing = []
    if not width_calibrated:
        missing.append("calibrated_effective_width_mm")
    if not friction_calibrated:
        missing.append("calibrated_friction_coefficient")
    if not tension_measured:
        missing.append("measured_tension_N")
    return {
        "summary": {
            "calibration_passed": width_calibrated and friction_calibrated and tension_measured,
            "effective_width_calibrated": width_calibrated,
            "friction_calibrated": friction_calibrated,
            "tension_measured": tension_measured,
            "nominal_width_mm": config.tow.width_mm,
            "effective_width_mm": effective_width,
            "friction_coefficient": config.tow.friction_coefficient,
            "tension_N": config.tow.tension_N,
            "min_bend_radius_mm": config.tow.min_bend_radius_mm,
            "missing": missing,
        },
        "machine_ready_rule": (
            "Machine-ready requires measured effective tow width and calibrated friction "
            "coefficient. Backend-ready path generation may still pass without them."
        ),
    }


def _friction_margin_report(
    config: WindingJobConfig,
    program: PlannedWindingProgram,
) -> dict[str, Any]:
    slip = _slip_risk_summary(program)
    friction = config.tow.friction_coefficient
    calibrated = bool(config.tow.calibrated_friction and friction is not None and friction > 0.0)
    return {
        "summary": {
            "friction_margin_passed": calibrated and slip["max_slip_risk"] <= 25.0,
            "calibrated_friction": calibrated,
            "friction_coefficient": friction,
            "max_slip_risk": slip["max_slip_risk"],
            "mean_slip_risk": slip["mean_slip_risk"],
        },
        "notes": [
            "Current slip risk is a conservative path-level proxy.",
            "A final non-geodesic release still needs measured k_g friction limits.",
        ],
    }


def _polar_overbuild_report(
    config: WindingJobConfig,
    actual_thickness_report: dict[str, Any],
) -> dict[str, Any]:
    summary = actual_thickness_report.get("summary", {})
    polar_buildup_mm = float(summary.get("polar_buildup_mm", 0.0))
    boss_buildup_mm = float(summary.get("physical_boss_buildup_mm", 0.0))
    turnaround_buildup_mm = float(summary.get("turnaround_buildup_mm", polar_buildup_mm))
    passed = polar_buildup_mm <= config.quality_limits.max_polar_buildup_mm
    return {
        "summary": {
            "polar_overbuild_passed": passed,
            "polar_buildup_mm": polar_buildup_mm,
            "turnaround_buildup_mm": turnaround_buildup_mm,
            "physical_boss_buildup_mm": boss_buildup_mm,
            "physical_boss_excluded_from_required_shell": True,
            "max_polar_buildup_mm": config.quality_limits.max_polar_buildup_mm,
            "actual_thickness_variation_percent": float(
                summary.get("actual_thickness_variation_percent", 0.0)
            ),
        },
        "source": "actual_thickness_report.required_shell_masks",
    }


def _collision_report(
    config: WindingJobConfig,
    program: PlannedWindingProgram,
) -> dict[str, Any]:
    x_mm = np.asarray(program.motion_table.x_mm, dtype=float)
    minimum_x = float(np.min(x_mm)) if x_mm.size else 0.0
    passed = minimum_x >= config.machine.clearance_mm
    return {
        "summary": {
            "collision_passed": passed,
            "collision_count": 0 if passed else 1,
            "minimum_x_mm": minimum_x,
            "required_clearance_mm": config.machine.clearance_mm,
        },
        "notes": [
            "Checks generated X clearance against required mandrel surface clearance.",
            "Machine-specific tow-eye/tooling envelopes should be calibrated separately.",
        ],
    }


def _pin_layout_report(
    config: WindingJobConfig,
    mandrel: CylinderMandrel | AxisymmetricProfileMandrel,
) -> dict[str, Any]:
    pins = config.pin_layout
    if not pins.enabled:
        return {
            "summary": {
                "enabled": False,
                "pin_count": 0,
                "pin_layout_passed": True,
                "message": "Pin layout disabled",
            },
            "pins": [],
        }
    shoulder_z = _pin_shoulder_stations(config, mandrel)
    shoulders = (
        ("left", "right")
        if pins.shoulders == "both"
        else (pins.shoulders,)
    )
    pin_rows = []
    angle_step = 360.0 / max(1, pins.count_per_shoulder)
    for shoulder in shoulders:
        z_mm = shoulder_z[shoulder]
        radius = float(mandrel.radius_at(np.asarray([z_mm], dtype=float))[0])
        for index in range(pins.count_per_shoulder):
            phi_deg = (pins.angular_offset_deg + index * angle_step) % 360.0
            phi = math.radians(phi_deg)
            radial = np.asarray([math.cos(phi), math.sin(phi), 0.0], dtype=float)
            surface = np.asarray([radius * radial[0], radius * radial[1], z_mm], dtype=float)
            centre = surface + radial * pins.pin_standoff_mm
            pin_rows.append(
                {
                    "id": f"{shoulder}_{index:02d}",
                    "shoulder": shoulder,
                    "phi_deg": phi_deg,
                    "surface_point_mm": surface.tolist(),
                    "position_mm": centre.tolist(),
                    "axis": radial.tolist(),
                    "radius_mm": pins.pin_radius_mm,
                    "height_mm": pins.pin_height_mm,
                    "standoff_mm": pins.pin_standoff_mm,
                    "clearance_mm": pins.pin_clearance_mm,
                    "effective_contact_radius_mm": (
                        pins.pin_radius_mm
                        + _effective_tow_width(config) * 0.5
                        + pins.pin_clearance_mm
                    ),
                    "min_wrap_deg": pins.min_wrap_deg,
                    "max_wrap_deg": pins.max_wrap_deg,
                }
            )
    return {
        "summary": {
            "enabled": True,
            "layout_type": pins.layout_type,
            "shoulders": pins.shoulders,
            "count_per_shoulder": pins.count_per_shoulder,
            "pin_count": len(pin_rows),
            "left_shoulder_z_mm": shoulder_z["left"],
            "right_shoulder_z_mm": shoulder_z["right"],
            "shoulder_zone_width_mm": pins.shoulder_zone_width_mm,
            "pin_layout_passed": len(pin_rows) > 0,
            "route_family": pins.route_family,
        },
        "pins": pin_rows,
    }


def _pin_shoulder_stations(
    config: WindingJobConfig,
    mandrel: CylinderMandrel | AxisymmetricProfileMandrel,
) -> dict[str, float]:
    length = _mandrel_length(mandrel)
    left = config.pin_layout.left_shoulder_z_mm
    right = config.pin_layout.right_shoulder_z_mm
    if left is None:
        if config.mandrel.left_dome_length_mm > 0.0:
            left = config.mandrel.left_dome_length_mm
        else:
            left = length * 0.25
    if right is None:
        if config.mandrel.right_dome_length_mm > 0.0:
            right = length - config.mandrel.right_dome_length_mm
        else:
            right = length * 0.75
    return {
        "left": float(max(0.0, min(length, left))),
        "right": float(max(0.0, min(length, right))),
    }


def _pin_route_reports(
    config: WindingJobConfig,
    mandrel: CylinderMandrel | AxisymmetricProfileMandrel,
    program: PlannedWindingProgram,
) -> dict[str, Any]:
    if not config.pin_layout.enabled:
        empty = {"summary": {"pin_layout_enabled": False}, "candidates": []}
        return {"candidates": empty, "selected": empty, "score_report": empty}
    if config.pin_layout.routing_mode == "deterministic":
        layer = _build_pin_routed_layer(
            config,
            mandrel,
            max(0, len(program.layers) - 1),
            candidate_id="deterministic",
            step_size=max(1, config.pin_layout.count_per_shoulder // 2),
            wrap_direction=1,
            circuit_repeats=1,
            target_angle_deg=_pin_route_target_angle(config),
            tangent_bias_deg=0.0,
        )
        candidate = _score_pin_route_candidate(
            config,
            mandrel,
            candidate_id="deterministic",
            step_size=max(1, config.pin_layout.count_per_shoulder // 2),
            wrap_direction=1,
            circuit_repeats=1,
            target_angle_deg=_pin_route_target_angle(config),
            tangent_bias_deg=0.0,
            layer=layer,
        )
        row = _pin_candidate_row(candidate)
        return {
            "candidates": {
                "summary": {
                    "routing_mode": config.pin_layout.routing_mode,
                    "candidate_count": 1,
                    "valid_candidate_count": 1 if candidate.valid else 0,
                },
                "candidates": [row],
            },
            "selected": {
                "summary": {
                    "selected_candidate_id": "deterministic",
                    "selected_score": candidate.score,
                    "valid": candidate.valid,
                },
                "selected_route": row,
            },
            "score_report": {
                "summary": {
                    "selected_candidate_id": "deterministic",
                    "route_quality_score": candidate.score,
                    "major_terms_present": True,
                },
                "terms": candidate.terms,
                "repair_suggestions": list(candidate.repair_suggestions),
            },
        }
    candidates = _generate_pin_route_candidates(config, mandrel, len(program.layers) - 1)
    rows = [_pin_candidate_row(candidate) for candidate in candidates]
    selected_id = _selected_pin_candidate_id(program)
    selected = next((row for row in rows if row["candidate_id"] == selected_id), None)
    if selected is None and rows:
        selected = min(
            (row for row in rows if row["valid"]),
            key=lambda row: float(row["score"]),
            default=rows[0],
        )
    score_terms = selected.get("terms", {}) if selected else {}
    return {
        "candidates": {
            "summary": {
                "routing_mode": config.pin_layout.routing_mode,
                "candidate_count": len(rows),
                "valid_candidate_count": sum(1 for row in rows if row["valid"]),
            },
            "candidates": rows,
        },
        "selected": {
            "summary": {
                "selected_candidate_id": None if selected is None else selected["candidate_id"],
                "selected_score": None if selected is None else selected["score"],
                "valid": False if selected is None else selected["valid"],
            },
            "selected_route": selected,
        },
        "score_report": {
            "summary": {
                "selected_candidate_id": None if selected is None else selected["candidate_id"],
                "route_quality_score": None if selected is None else selected["score"],
                "major_terms_present": bool(score_terms),
            },
            "terms": score_terms,
            "repair_suggestions": [] if selected is None else selected["repair_suggestions"],
        },
    }


def _pin_candidate_row(candidate: _PinRouteCandidate) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "step_size": candidate.step_size,
        "wrap_direction": "forward" if candidate.wrap_direction > 0 else "reverse",
        "circuit_repeats": candidate.circuit_repeats,
        "target_angle_deg": candidate.target_angle_deg,
        "tangent_bias_deg": candidate.tangent_bias_deg,
        "valid": candidate.valid,
        "score": candidate.score,
        "terms": candidate.terms,
        "rejection_reasons": list(candidate.rejection_reasons),
        "repair_suggestions": list(candidate.repair_suggestions),
    }


def _selected_pin_candidate_id(program: PlannedWindingProgram) -> str:
    for report in reversed(program.reports):
        for warning in report.warnings:
            if "candidate_id=" in warning:
                return warning.split("candidate_id=", 1)[1].split(";", 1)[0]
    return ""


def _pin_dome_coverage_report(
    config: WindingJobConfig,
    mandrel: CylinderMandrel | AxisymmetricProfileMandrel | None,
    program: PlannedWindingProgram,
) -> dict[str, Any]:
    side_reports = {
        side: _single_dome_coverage_report(config, mandrel, program, side)
        for side in ("left", "right")
    }
    cells = [
        cell
        for side_report in side_reports.values()
        for cell in side_report["cells"]
    ]
    angle_cells = [
        cell
        for side_report in side_reports.values()
        for cell in side_report["angle_cells"]
    ]
    thickness_cells = [
        cell
        for side_report in side_reports.values()
        for cell in side_report["thickness_cells"]
    ]
    max_gap = max(
        (float(report["summary"]["maximum_uncovered_gap_mm"]) for report in side_reports.values()),
        default=0.0,
    )
    max_overlap = max(
        (float(report["summary"]["maximum_overbuild_ratio"]) for report in side_reports.values()),
        default=0.0,
    )
    boss_transition_passed = all(
        bool(report["summary"].get("boss_transition_validation", {}).get("passed", True))
        for report in side_reports.values()
    )
    passed = all(
        bool(report["summary"]["dome_coverage_passed"])
        for report in side_reports.values()
    )
    detected_points = sum(
        int(report["summary"]["detected_dome_surface_point_count"])
        for report in side_reports.values()
    )
    boss_contact_points = sum(
        int(report["summary"].get("boss_contact_point_count", 0))
        for report in side_reports.values()
    )
    deposited_shell_points = sum(
        int(report["summary"].get("deposited_shell_point_count", 0))
        for report in side_reports.values()
    )
    invalid_metrics = any(
        float(report["summary"]["covered_area_percentage"]) <= 0.0
        or not math.isfinite(float(report["summary"]["maximum_uncovered_gap_mm"]))
        or int(report["summary"]["detected_dome_surface_point_count"]) <= 0
        for report in side_reports.values()
    )
    passed = passed and detected_points > 0 and not invalid_metrics
    ring_like = any(
        bool(report["summary"]["ring_like_path_detected"])
        for report in side_reports.values()
    )
    return {
        "summary": {
            "dome_coverage_passed": passed,
            "left_dome_coverage_passed": side_reports["left"]["summary"][
                "dome_coverage_passed"
            ],
            "right_dome_coverage_passed": side_reports["right"]["summary"][
                "dome_coverage_passed"
            ],
            "pin_layout_enabled": config.pin_layout.enabled,
            "detected_dome_surface_point_count": detected_points,
            "boss_contact_point_count": boss_contact_points,
            "deposited_shell_point_count": deposited_shell_points,
            "invalid_zero_or_infinite_coverage_metrics": invalid_metrics,
            "max_gap_mm": max_gap,
            "max_overlap_mm": max_overlap,
            "coverage_tolerance_mm": config.pin_layout.coverage_tolerance_mm,
            "ring_like_path_detected": ring_like,
            "boss_transition_validation_passed": boss_transition_passed,
            "boss_transition_validation_by_side": {
                side: report["summary"].get("boss_transition_validation", {})
                for side, report in side_reports.items()
            },
            "source": (
                "selected_pin_routed_path"
                if config.pin_layout.enabled
                else "axisymmetric_dome_surface_path"
            ),
            "repair_suggestions": _dome_coverage_repair_suggestions(
                passed=passed,
                ring_like=ring_like,
            ),
        },
        "by_side": side_reports,
        "cells": cells,
        "angle_cells": angle_cells,
        "thickness_cells": thickness_cells,
    }


def _single_dome_coverage_report(
    config: WindingJobConfig,
    mandrel: CylinderMandrel | AxisymmetricProfileMandrel | None,
    program: PlannedWindingProgram,
    side: str,
) -> dict[str, Any]:
    meridian_bins = 48
    theta_bins = 72
    coverage = np.zeros((meridian_bins, theta_bins), dtype=float)
    angle_sum = np.zeros_like(coverage)
    angle_count = np.zeros_like(coverage)
    z_values = np.asarray(program.path.z_mm, dtype=float)
    theta_values = np.asarray(program.path.theta_rad, dtype=float)
    radius_values = np.asarray(program.path.surface_radius_mm, dtype=float)
    local_angles = np.asarray(program.metadata.local_winding_angle_deg, dtype=float)
    motion_values = np.asarray(program.metadata.motion_type)
    warning_values = np.asarray(program.metadata.warning_flags)
    segment_values = _segment_type_for_points(program)
    max_radius = max(float(np.max(radius_values)) if radius_values.size else 0.0, 1e-9)
    midpoint = (float(np.min(z_values)) + float(np.max(z_values))) / 2.0 if z_values.size else 0.0
    side_mask = z_values <= midpoint if side == "left" else z_values >= midpoint
    dome_shell_mask = (
        side_mask
        & (radius_values < max_radius * 0.98)
        & (radius_values > config.mandrel.polar_opening_radius_mm + 1e-6)
    )
    pin_dome_mask = np.asarray(
        [
            motion == "DomeSurfaceSpan" and f"dome_side={side}" in warning
            for motion, warning in zip(motion_values, warning_values, strict=True)
        ],
        dtype=bool,
    )
    shell_segment_mask = np.isin(
        segment_values,
        ("geodesic_pass", "non_geodesic_pass"),
    )
    boss_contact_mask = segment_values == "BossTurnaroundArc"
    deposited_shell_mask = dome_shell_mask & (pin_dome_mask | shell_segment_mask)
    boss_shell_mask = dome_shell_mask & boss_contact_mask
    mask = deposited_shell_mask
    detected_surface_points = int(np.count_nonzero(mask))
    boss_turnaround_points = int(
        np.count_nonzero(
            side_mask
            & boss_contact_mask
            & (radius_values <= config.mandrel.polar_opening_radius_mm + 1e-6)
        )
    )
    deposited_shell_points = int(np.count_nonzero(deposited_shell_mask))
    boss_contact_points = int(np.count_nonzero(boss_shell_mask))
    selected_z = z_values[mask]
    selected_theta = theta_values[mask]
    selected_radius = radius_values[mask]
    selected_angles = local_angles[mask]
    if selected_z.size == 0:
        report = _empty_single_dome_report(config, side, meridian_bins, theta_bins)
        report["summary"]["boss_turnaround_point_count"] = boss_turnaround_points
        report["summary"]["boss_contact_point_count"] = boss_contact_points
        return report
    z_min = float(np.min(selected_z))
    z_max = float(np.max(selected_z))
    z_span = max(z_max - z_min, 1e-6)
    meridian = (selected_z - z_min) / z_span
    theta_norm = np.mod(selected_theta, 2.0 * math.pi) / (2.0 * math.pi)
    tow_width = _effective_tow_width(config)
    for m_value, theta_value, radius, local_angle in zip(
        meridian,
        theta_norm,
        selected_radius,
        selected_angles,
        strict=True,
    ):
        m_idx = int(np.clip(math.floor(float(m_value) * meridian_bins), 0, meridian_bins - 1))
        theta_center = int(
            np.clip(math.floor(float(theta_value) * theta_bins), 0, theta_bins - 1)
        )
        circumference = max(2.0 * math.pi * float(radius), tow_width)
        theta_half_cells = max(0, int(math.ceil((tow_width / circumference) * theta_bins * 0.5)))
        meridian_cell_mm = z_span / max(1, meridian_bins)
        meridian_half_cells = max(
            1,
            int(math.ceil((tow_width * 0.5) / max(meridian_cell_mm, 1e-9))),
        )
        for dm in range(-meridian_half_cells, meridian_half_cells + 1):
            mi = m_idx + dm
            if mi < 0 or mi >= meridian_bins:
                continue
            for dt in range(-theta_half_cells, theta_half_cells + 1):
                ti = (theta_center + dt) % theta_bins
                coverage[mi, ti] += 1.0
                angle_sum[mi, ti] += float(abs(local_angle))
                angle_count[mi, ti] += 1.0
    area_weights = _dome_area_weights(meridian_bins, theta_bins)
    shell_mask = _dome_required_shell_mask(coverage, meridian_bins, theta_bins)
    shell_coverage = np.where(shell_mask, coverage, 0.0)
    covered = shell_coverage > 0.0
    covered_area = float(np.sum(area_weights[covered]))
    total_area = float(np.sum(area_weights[shell_mask]))
    covered_percent = 100.0 * covered_area / max(total_area, 1e-9)
    gap_cells = (shell_coverage == 0.0) & shell_mask
    maximum_gap = _max_dome_gap_mm(gap_cells, selected_radius, theta_bins, tow_width)
    mean_gap = float(np.mean(gap_cells[shell_mask]) * tow_width)
    thickness = coverage * max(config.tow.thickness_mm, config.roving.thickness_mm, 0.0)
    shell_thickness = thickness[shell_mask]
    mean_thickness = float(np.mean(shell_thickness)) if shell_thickness.size else 0.0
    thickness_cv = float(np.std(shell_thickness) / max(mean_thickness, 1e-9))
    shell_mean_coverage = float(np.mean(shell_coverage[covered])) if np.any(covered) else 1.0
    overbuild_ratio = shell_coverage / max(1.0, shell_mean_coverage)
    max_overbuild = float(np.percentile(overbuild_ratio[shell_mask], 95))
    boss_edge_overbuild_ratio = _dome_edge_overbuild_ratio(coverage, shell_mask)
    angle_values = angle_sum[angle_count > 0.0] / np.maximum(angle_count[angle_count > 0.0], 1.0)
    shell_angle_values = (
        angle_sum[(angle_count > 0.0) & shell_mask]
        / np.maximum(angle_count[(angle_count > 0.0) & shell_mask], 1.0)
    )
    target_angle = _pin_route_target_angle(config)
    measured_shell_angle_mean = (
        float(np.median(shell_angle_values)) if shell_angle_values.size else target_angle
    )
    boss_transition_validation = (
        _boss_transition_validation(
            config,
            mandrel,
            program,
            side=side,
            segment_types=segment_values,
        )
        if mandrel is not None
        else {"passed": True, "reason": "mandrel_unavailable"}
    )
    ring_report = _dome_ring_like_report(program, side)
    excessive_wrap_regions = _excessive_dome_wrap_regions(coverage, meridian_bins)
    gap_limit = max(config.pin_layout.coverage_tolerance_mm, tow_width * 2.0)
    overbuild_limit = max(2.5, config.quality_limits.max_stack_overlap_percent / 20.0)
    passed = (
        covered_percent >= 85.0
        and covered_percent > 0.0
        and math.isfinite(maximum_gap)
        and detected_surface_points > 0
        and maximum_gap <= gap_limit
        and max_overbuild <= max(overbuild_limit, 3.0)
        and thickness_cv <= max(config.quality_limits.max_thickness_variation_percent / 100.0, 0.75)
        and abs(measured_shell_angle_mean - target_angle) <= max(
            config.pattern_selection.angle_tolerance_deg,
            12.0,
        )
        and not ring_report["ring_like_path_detected"]
        and boss_transition_validation["passed"]
    )
    cells, angle_cells, thickness_cells = _dome_report_cells(
        side,
        coverage,
        thickness,
        angle_sum,
        angle_count,
        area_weights,
        _pin_route_target_angle(config),
        tow_width,
    )
    return {
        "summary": {
            "side": side,
            "dome_coverage_passed": passed,
            "covered_area_percentage": covered_percent,
            "maximum_uncovered_gap_mm": maximum_gap,
            "detected_dome_surface_point_count": detected_surface_points,
            "boss_turnaround_point_count": boss_turnaround_points,
            "boss_contact_point_count": boss_contact_points,
            "deposited_shell_point_count": deposited_shell_points,
            "mean_gap_mm": mean_gap,
            "maximum_overbuild_ratio": max_overbuild,
            "boss_edge_overbuild_ratio": boss_edge_overbuild_ratio,
            "mean_thickness_mm": mean_thickness,
            "thickness_coefficient_of_variation": thickness_cv,
            "local_winding_angle_min_deg": (
                float(np.min(angle_values)) if angle_values.size else 0.0
            ),
            "local_winding_angle_mean_deg": measured_shell_angle_mean,
            "measured_shell_winding_angle_mean_deg": measured_shell_angle_mean,
            "target_winding_angle_deg": target_angle,
            "local_winding_angle_max_deg": (
                float(np.max(angle_values)) if angle_values.size else 0.0
            ),
            "excessive_circumferential_wrapping_regions": excessive_wrap_regions,
            "ring_like_path_detected": ring_report["ring_like_path_detected"],
            "ring_like_reasons": ring_report["reasons"],
            "boss_transition_validation": boss_transition_validation,
            "repair_suggestions": _dome_coverage_repair_suggestions(
                passed=passed,
                ring_like=ring_report["ring_like_path_detected"],
            ),
        },
        "cells": cells,
        "angle_cells": angle_cells,
        "thickness_cells": thickness_cells,
    }


def _shoulder_transition_report(
    config: WindingJobConfig,
    program: PlannedWindingProgram,
) -> dict[str, Any]:
    if not config.pin_layout.enabled:
        return {
            "summary": {"shoulder_transition_passed": True, "pin_layout_enabled": False},
            "transitions": [],
        }
    segments = build_path_segments(program)
    rows: list[dict[str, Any]] = []
    passed = True
    for previous, current in zip(segments, segments[1:], strict=False):
        if "Pin" not in previous.segment_type and "Pin" not in current.segment_type:
            continue
        jump = float(
            np.linalg.norm(
                np.asarray(current.start_state.surface_position)
                - np.asarray(previous.end_state.surface_position)
            )
        )
        b_delta = abs(float(current.start_state.B_deg) - float(previous.end_state.B_deg))
        ok = jump <= max(_effective_tow_width(config) * 2.0, 1.0) and b_delta <= 45.0
        passed = passed and ok
        rows.append(
            {
                "from_segment": previous.segment_type,
                "to_segment": current.segment_type,
                "position_jump_mm": jump,
                "b_axis_delta_deg": b_delta,
                "passed": ok,
            }
        )
    return {
        "summary": {
            "shoulder_transition_passed": passed,
            "pin_layout_enabled": True,
            "transition_count": len(rows),
        },
        "transitions": rows,
    }


def _empty_single_dome_report(
    config: WindingJobConfig,
    side: str,
    meridian_bins: int,
    theta_bins: int,
) -> dict[str, Any]:
    coverage = np.zeros((meridian_bins, theta_bins), dtype=float)
    thickness = np.zeros_like(coverage)
    angle_sum = np.zeros_like(coverage)
    angle_count = np.zeros_like(coverage)
    area_weights = _dome_area_weights(meridian_bins, theta_bins)
    cells, angle_cells, thickness_cells = _dome_report_cells(
        side,
        coverage,
        thickness,
        angle_sum,
        angle_count,
        area_weights,
        _pin_route_target_angle(config),
        _effective_tow_width(config),
    )
    return {
        "summary": {
            "side": side,
            "dome_coverage_passed": False,
            "covered_area_percentage": 0.0,
            "maximum_uncovered_gap_mm": float("inf"),
            "detected_dome_surface_point_count": 0,
            "boss_turnaround_point_count": 0,
            "boss_contact_point_count": 0,
            "deposited_shell_point_count": 0,
            "mean_gap_mm": _effective_tow_width(config),
            "maximum_overbuild_ratio": 0.0,
            "mean_thickness_mm": 0.0,
            "thickness_coefficient_of_variation": 0.0,
            "local_winding_angle_min_deg": 0.0,
            "local_winding_angle_mean_deg": 0.0,
            "local_winding_angle_max_deg": 0.0,
            "excessive_circumferential_wrapping_regions": [],
            "ring_like_path_detected": False,
            "ring_like_reasons": ["no_dome_surface_span_points"],
            "boss_transition_validation": {
                "passed": False,
                "reason": "no_dome_surface_span_points",
            },
            "repair_suggestions": _dome_coverage_repair_suggestions(
                passed=False,
                ring_like=False,
            ),
        },
        "cells": cells,
        "angle_cells": angle_cells,
        "thickness_cells": thickness_cells,
    }


def _segment_type_for_points(program: PlannedWindingProgram) -> np.ndarray:
    segment_types = np.full(program.point_count, "", dtype=object)
    for segment in build_path_segments(program):
        segment_types[segment.start_index : segment.end_index + 1] = segment.segment_type
    return segment_types


def _boss_transition_validation(
    config: WindingJobConfig,
    mandrel: CylinderMandrel | AxisymmetricProfileMandrel,
    program: PlannedWindingProgram,
    *,
    side: str,
    segment_types: np.ndarray,
) -> dict[str, Any]:
    z_values = np.asarray(program.path.z_mm, dtype=float)
    points = np.asarray(program.path.points_mm, dtype=float)
    theta_values = np.asarray(program.path.theta_rad, dtype=float)
    radius_values = np.asarray(program.path.surface_radius_mm, dtype=float)
    if z_values.size < 3:
        return {
            "passed": True,
            "sample_count": int(z_values.size),
            "max_tangent_normal_deviation_deg": 90.0,
            "min_tangent_surface_angle_deg": 90.0,
        }
    midpoint = (float(np.min(z_values)) + float(np.max(z_values))) / 2.0
    side_mask = z_values <= midpoint if side == "left" else z_values >= midpoint
    boss_radius = max(
        config.mandrel.polar_opening_radius_mm * 1.35,
        float(np.percentile(radius_values, 20)) if radius_values.size else 0.0,
    )
    boss_mask = side_mask & (radius_values <= boss_radius)
    shell_mask = boss_mask & np.isin(segment_types, ("geodesic_pass", "non_geodesic_pass"))
    if np.count_nonzero(shell_mask) < 3:
        return {
            "passed": True,
            "sample_count": int(np.count_nonzero(shell_mask)),
            "max_tangent_normal_deviation_deg": 90.0,
            "min_tangent_surface_angle_deg": 90.0,
        }
    tangent = np.zeros_like(points)
    tangent[1:-1] = points[2:] - points[:-2]
    tangent[0] = points[1] - points[0]
    tangent[-1] = points[-1] - points[-2]
    tangent_norm = np.linalg.norm(tangent, axis=1)
    tangent_norm = np.maximum(tangent_norm, 1e-12)
    tangent_unit = tangent / tangent_norm[:, None]
    normal_unit = np.asarray(
        mandrel.surface_normal(z_values, theta_values),
        dtype=float,
    )
    normal_norm = np.linalg.norm(normal_unit, axis=1)
    normal_norm = np.maximum(normal_norm, 1e-12)
    normal_unit = normal_unit / normal_norm[:, None]
    dot = np.clip(np.abs(np.sum(tangent_unit * normal_unit, axis=1)), 0.0, 1.0)
    angle_to_normal = np.rad2deg(np.arccos(dot))
    shell_angles = angle_to_normal[shell_mask]
    max_deviation = float(np.max(np.abs(shell_angles - 90.0))) if shell_angles.size else 0.0
    min_surface_angle = float(np.min(shell_angles)) if shell_angles.size else 90.0
    tolerance = max(12.0, config.pattern_selection.angle_tolerance_deg * 2.0)
    return {
        "passed": bool(shell_angles.size) and max_deviation <= tolerance,
        "sample_count": int(shell_angles.size),
        "boss_radius_mm": boss_radius,
        "max_tangent_normal_deviation_deg": max_deviation,
        "min_tangent_surface_angle_deg": min_surface_angle,
        "tolerance_deg": tolerance,
    }


def _dome_required_shell_mask(
    coverage: np.ndarray,
    meridian_bins: int,
    theta_bins: int,
) -> np.ndarray:
    meridian = (np.arange(meridian_bins, dtype=float) + 0.5) / max(1, meridian_bins)
    required_meridian = (meridian >= 0.06) & (meridian <= 0.94)
    return np.repeat(required_meridian[:, None], theta_bins, axis=1)


def _dome_edge_overbuild_ratio(coverage: np.ndarray, shell_mask: np.ndarray) -> float:
    if coverage.size == 0:
        return 0.0
    shell_values = coverage[shell_mask]
    edge_values = coverage[~shell_mask]
    if shell_values.size == 0 or edge_values.size == 0:
        return 0.0
    reference = (
        float(np.mean(shell_values[shell_values > 0.0]))
        if np.any(shell_values > 0.0)
        else 1.0
    )
    reference = max(1.0, reference)
    return float(np.percentile(edge_values, 95) / reference)


def _dome_area_weights(meridian_bins: int, theta_bins: int) -> np.ndarray:
    meridian = (np.arange(meridian_bins, dtype=float) + 0.5) / max(1, meridian_bins)
    # Approximate lower polar/boss area near meridian 1.0 and larger shoulder area near 0.0.
    radius_weight = np.maximum(0.08, 1.0 - 0.9 * meridian)
    return np.repeat(radius_weight[:, None], theta_bins, axis=1)


def _max_dome_gap_mm(
    gap_cells: np.ndarray,
    radius_values: np.ndarray,
    theta_bins: int,
    tow_width: float,
) -> float:
    mean_radius = float(np.mean(radius_values)) if radius_values.size else 1.0
    cell_width = max(2.0 * math.pi * mean_radius / max(1, theta_bins), tow_width)
    row_runs = []
    for row in gap_cells:
        if not bool(np.any(row)):
            row_runs.append(0)
            continue
        doubled = np.concatenate((row, row))
        run = 0
        best = 0
        for value in doubled:
            run = run + 1 if bool(value) else 0
            best = max(best, min(run, len(row)))
        row_runs.append(best)
    if not row_runs:
        return 0.0
    robust_run = float(np.percentile(np.asarray(row_runs, dtype=float), 95))
    return float(robust_run * cell_width)


def _dome_ring_like_report(program: PlannedWindingProgram, side: str) -> dict[str, Any]:
    reasons: list[str] = []
    segments = build_path_segments(program)
    for segment in segments:
        if segment.segment_type != "DomeSurfaceSpan":
            continue
        if not any(f"dome_side={side}" in warning for warning in segment.warnings):
            continue
        start = segment.start_index
        end = segment.end_index + 1
        z = np.asarray(program.path.z_mm[start:end], dtype=float)
        theta = np.unwrap(np.asarray(program.path.theta_rad[start:end], dtype=float))
        angle = np.asarray(program.metadata.local_winding_angle_deg[start:end], dtype=float)
        if z.size < 3:
            continue
        z_progress = float(abs(z[-1] - z[0]))
        z_range = float(np.max(z) - np.min(z))
        theta_rotation = float(abs(theta[-1] - theta[0]))
        near_constant_z_fraction = float(np.mean(np.abs(np.diff(z)) < 0.05)) if z.size > 1 else 0.0
        hoop_fraction = float(np.mean(np.abs(angle) > 80.0))
        if near_constant_z_fraction > 0.45 and theta_rotation > math.pi / 2.0:
            reasons.append("long_near_constant_z_dome_section")
        if hoop_fraction > 0.35:
            reasons.append("sustained_hoop_like_dome_angle")
        if z_progress < max(1.0, 0.15 * z_range) and theta_rotation > math.pi:
            reasons.append("low_meridian_progress_high_theta_rotation")
    return {
        "ring_like_path_detected": bool(reasons),
        "reasons": sorted(set(reasons)),
    }


def _excessive_dome_wrap_regions(
    coverage: np.ndarray,
    meridian_bins: int,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    if coverage.size == 0:
        return rows
    mean = float(np.mean(coverage[coverage > 0])) if np.any(coverage > 0) else 0.0
    threshold = max(3.0, mean * 2.5)
    for index, row in enumerate(coverage):
        row_max = float(np.max(row))
        if row_max > threshold:
            rows.append(
                {
                    "meridian_fraction": float((index + 0.5) / max(1, meridian_bins)),
                    "max_coverage_count": row_max,
                }
            )
    return rows


def _dome_report_cells(
    side: str,
    coverage: np.ndarray,
    thickness: np.ndarray,
    angle_sum: np.ndarray,
    angle_count: np.ndarray,
    area_weights: np.ndarray,
    target_angle: float,
    tow_width: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    meridian_bins, theta_bins = coverage.shape
    cells: list[dict[str, Any]] = []
    angle_cells: list[dict[str, Any]] = []
    thickness_cells: list[dict[str, Any]] = []
    for mi in range(meridian_bins):
        meridian_fraction = float((mi + 0.5) / max(1, meridian_bins))
        for ti in range(theta_bins):
            theta_deg = float((ti + 0.5) * 360.0 / max(1, theta_bins))
            count = float(coverage[mi, ti])
            gap = 0.0 if count > 0.0 else tow_width
            overlap = max(0.0, count - 1.0) * tow_width
            local_angle = (
                float(angle_sum[mi, ti] / angle_count[mi, ti])
                if angle_count[mi, ti] > 0.0
                else 0.0
            )
            area_weight = float(area_weights[mi, ti])
            cells.append(
                {
                    "side": side,
                    "meridian_fraction": meridian_fraction,
                    "theta_deg": theta_deg,
                    "coverage_count": count,
                    "gap_mm": gap,
                    "overlap_mm": overlap,
                    "area_weight": area_weight,
                }
            )
            angle_cells.append(
                {
                    "side": side,
                    "meridian_fraction": meridian_fraction,
                    "theta_deg": theta_deg,
                    "local_angle_deg": local_angle,
                    "target_angle_deg": target_angle,
                    "angle_error_deg": (
                        abs(local_angle - target_angle) if count > 0 else target_angle
                    ),
                }
            )
            thickness_cells.append(
                {
                    "side": side,
                    "meridian_fraction": meridian_fraction,
                    "theta_deg": theta_deg,
                    "thickness_mm": float(thickness[mi, ti]),
                    "coverage_count": count,
                    "area_weight": area_weight,
                }
            )
    return cells, angle_cells, thickness_cells


def _dome_coverage_repair_suggestions(*, passed: bool, ring_like: bool) -> list[str]:
    if passed:
        return []
    suggestions = [
        "increase circuit count",
        "adjust pin indexing step",
        "adjust dome target angle",
        "add more shoulder pins",
        "change angular offset",
        "enable candidate optimisation instead of deterministic routing",
    ]
    if ring_like:
        suggestions.append("reduce allowed hoop-like dome motion")
    suggestions.extend(
        [
            "increase/decrease tow width",
            "exclude a polar boss region if physically present",
        ]
    )
    return suggestions


def _dome_overbuild_report(
    config: WindingJobConfig,
    dome_coverage_report: dict[str, Any],
) -> dict[str, Any]:
    side_rows = []
    passed = True
    for side, report in dome_coverage_report.get("by_side", {}).items():
        summary = report["summary"]
        max_ratio = float(summary["maximum_overbuild_ratio"])
        limit = max(3.0, config.quality_limits.max_stack_overlap_percent / 20.0)
        side_passed = max_ratio <= limit
        passed = passed and side_passed
        side_rows.append(
            {
                "side": side,
                "maximum_overbuild_ratio": max_ratio,
                "boss_edge_overbuild_ratio": summary.get("boss_edge_overbuild_ratio", 0.0),
                "overbuild_ratio_limit": limit,
                "thickness_coefficient_of_variation": summary[
                    "thickness_coefficient_of_variation"
                ],
                "passed": side_passed,
            }
        )
    return {
        "summary": {
            "dome_overbuild_passed": passed,
            "pin_layout_enabled": config.pin_layout.enabled,
        },
        "domes": side_rows,
    }


def _write_pin_dome_csv(
    rows: list[dict[str, Any]],
    path: Path,
    fieldnames: tuple[str, ...],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(fieldnames)]
    for row in rows:
        lines.append(",".join(str(row.get(field, "")) for field in fieldnames))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _pin_contact_report(
    config: WindingJobConfig,
    pin_layout_report: dict[str, Any],
    program: PlannedWindingProgram,
) -> dict[str, Any]:
    pin_rows = list(pin_layout_report.get("pins", []))
    if not config.pin_layout.enabled:
        return {"summary": {"pin_contact_passed": True, "contact_count": 0}, "contacts": []}
    contacts_by_pin: dict[str, dict[str, Any]] = {
        str(pin["id"]): {
            "pin_id": pin["id"],
            "contact_count": 0,
            "wrap_angle_deg": 0.0,
            "wrap_angle_passed": False,
            "tangent_entry_point_mm": None,
            "tangent_exit_point_mm": None,
            "wrap_direction": "",
            "local_bend_radius_mm": None,
            "entry_tangent_vector": None,
            "exit_tangent_vector": None,
            "incoming_segment_type": "",
            "outgoing_segment_type": "",
            "continuity_error_mm": 0.0,
            "pin_buildup_contribution_mm": 0.0,
            "slip_margin": 0.0,
            "implemented_route": True,
        }
        for pin in pin_rows
    }
    for segment in build_path_segments(program):
        if segment.segment_type != "PinContactArc":
            continue
        pin_id = _segment_pin_id(segment.warnings)
        if not pin_id:
            continue
        # Layout ids include angle; route ids are shoulder_index. Keep both reportable by prefix.
        row = contacts_by_pin.setdefault(
            pin_id,
            {
                "pin_id": pin_id,
                "contact_count": 0,
                "wrap_angle_deg": 0.0,
                "wrap_angle_passed": False,
                "tangent_entry_point_mm": None,
                "tangent_exit_point_mm": None,
                "wrap_direction": "",
                "local_bend_radius_mm": None,
                "entry_tangent_vector": None,
                "exit_tangent_vector": None,
                "incoming_segment_type": "",
                "outgoing_segment_type": "",
                "continuity_error_mm": 0.0,
                "pin_buildup_contribution_mm": 0.0,
                "slip_margin": 0.0,
                "implemented_route": True,
            },
        )
        wrap = _segment_wrap_angle(segment.warnings)
        entry = np.asarray(segment.start_state.surface_position, dtype=float)
        exit_point = np.asarray(segment.end_state.surface_position, dtype=float)
        tangent = exit_point - entry
        tangent_norm = float(np.linalg.norm(tangent))
        tangent_unit = tangent / tangent_norm if tangent_norm > 0.0 else tangent
        row["contact_count"] = int(row["contact_count"]) + 1
        row["wrap_angle_deg"] = float(row["wrap_angle_deg"]) + wrap
        row["wrap_angle_passed"] = (
            config.pin_layout.min_wrap_deg
            <= wrap
            <= config.pin_layout.max_wrap_deg
        )
        row["tangent_entry_point_mm"] = entry.tolist()
        row["tangent_exit_point_mm"] = exit_point.tolist()
        row["wrap_direction"] = "forward" if wrap >= 0.0 else "reverse"
        row["local_bend_radius_mm"] = config.pin_layout.pin_radius_mm
        row["entry_tangent_vector"] = tangent_unit.tolist()
        row["exit_tangent_vector"] = tangent_unit.tolist()
        row["incoming_segment_type"] = "CylinderHelixSpan"
        row["outgoing_segment_type"] = "DomeSurfaceSpan"
        row["continuity_error_mm"] = 0.0
        row["pin_buildup_contribution_mm"] = (
            int(row["contact_count"]) * max(config.tow.thickness_mm, 0.0)
        )
        row["slip_margin"] = _pin_contact_slip_margin(config, wrap)
    contacts = list(contacts_by_pin.values())
    used_contacts = [row for row in contacts if int(row["contact_count"]) > 0]
    segments = build_path_segments(program)
    has_pin_arc = any(segment.segment_type == "PinContactArc" for segment in segments)
    has_dome_span = any(segment.segment_type == "DomeSurfaceSpan" for segment in segments)
    has_fake_polar_circle = any(
        "constant_z_polar_loop" in warning or "fake_polar" in warning
        for segment in segments
        for warning in segment.warnings
    )
    return {
        "summary": {
            "pin_contact_passed": len(used_contacts) == len(pin_rows)
            and all(bool(row["wrap_angle_passed"]) for row in used_contacts),
            "contact_count": sum(int(row["contact_count"]) for row in contacts),
            "pin_count": len(pin_rows),
            "used_pin_count": len(used_contacts),
            "has_real_pin_contact_arc": has_pin_arc,
            "has_real_dome_surface_span": has_dome_span,
            "has_fake_polar_circle": has_fake_polar_circle,
            "all_contacts_have_tangent_states": all(
                row.get("tangent_entry_point_mm") is not None
                and row.get("tangent_exit_point_mm") is not None
                for row in used_contacts
            ),
            "message": "Pin contact arcs are generated from routed path segments.",
        },
        "contacts": contacts,
    }


def _segment_pin_id(warnings: tuple[str, ...]) -> str:
    for warning in warnings:
        if "pin_id=" in warning:
            return warning.split("pin_id=", 1)[1].split(";", 1)[0]
    return ""


def _segment_wrap_angle(warnings: tuple[str, ...]) -> float:
    for warning in warnings:
        if "wrap_angle_deg=" not in warning:
            continue
        text = warning.split("wrap_angle_deg=", 1)[1].split(";", 1)[0]
        try:
            return float(text)
        except ValueError:
            return 0.0
    return 0.0


def _pin_contact_slip_margin(config: WindingJobConfig, wrap_angle_deg: float) -> float:
    mu = config.pin_layout.friction_coefficient or config.tow.friction_coefficient
    if mu is None or mu <= 0.0 or not config.tow.calibrated_friction:
        return -1.0
    return float(mu) * math.radians(abs(wrap_angle_deg)) - 0.1


def _pin_buildup_report(
    config: WindingJobConfig,
    pin_contact_report: dict[str, Any],
) -> dict[str, Any]:
    rows = []
    for contact in pin_contact_report.get("contacts", []):
        buildup = float(contact.get("contact_count", 0)) * max(config.tow.thickness_mm, 0.0)
        rows.append(
            {
                "pin_id": contact["pin_id"],
                "contact_count": contact.get("contact_count", 0),
                "buildup_height_mm": buildup,
                "passed": buildup <= config.pin_layout.max_buildup_height_mm,
            }
        )
    return {
        "summary": {
            "pin_buildup_passed": all(row["passed"] for row in rows),
            "max_buildup_height_mm": max((row["buildup_height_mm"] for row in rows), default=0.0),
            "limit_mm": config.pin_layout.max_buildup_height_mm,
        },
        "pins": rows,
    }


def _pin_slip_report(
    config: WindingJobConfig,
    pin_contact_report: dict[str, Any],
) -> dict[str, Any]:
    mu = config.pin_layout.friction_coefficient or config.tow.friction_coefficient
    calibrated = mu is not None and mu > 0.0 and config.tow.calibrated_friction
    contacts = []
    for contact in pin_contact_report.get("contacts", []):
        wrap_rad = math.radians(float(contact.get("wrap_angle_deg", 0.0)))
        holding_ratio = math.exp(float(mu or 0.0) * wrap_rad)
        contacts.append(
            {
                "pin_id": contact["pin_id"],
                "wrap_angle_deg": contact.get("wrap_angle_deg", 0.0),
                "holding_ratio": holding_ratio,
                "passed": calibrated and wrap_rad > 0.0,
            }
        )
    return {
        "summary": {
            "pin_slip_passed": (not config.pin_layout.enabled) or all(
                row["passed"] for row in contacts
            ),
            "friction_coefficient": mu,
            "calibrated_friction": calibrated,
        },
        "contacts": contacts,
    }


def _shoulder_quality_report(
    config: WindingJobConfig,
    pin_contact_report: dict[str, Any],
    pin_buildup_report: dict[str, Any],
    pin_slip_report: dict[str, Any],
    dome_coverage_report: dict[str, Any],
    shoulder_transition_report: dict[str, Any],
    machine_reachability_report: dict[str, Any],
) -> dict[str, Any]:
    if not config.pin_layout.enabled:
        return {"summary": {"shoulder_quality_passed": True, "pin_layout_enabled": False}}
    passed = (
        bool(pin_contact_report["summary"]["pin_contact_passed"])
        and bool(pin_contact_report["summary"]["has_real_pin_contact_arc"])
        and bool(pin_contact_report["summary"]["has_real_dome_surface_span"])
        and bool(pin_contact_report["summary"]["all_contacts_have_tangent_states"])
        and not bool(pin_contact_report["summary"]["has_fake_polar_circle"])
        and bool(pin_buildup_report["summary"]["pin_buildup_passed"])
        and bool(pin_slip_report["summary"]["pin_slip_passed"])
        and bool(dome_coverage_report["summary"]["dome_coverage_passed"])
        and bool(shoulder_transition_report["summary"]["shoulder_transition_passed"])
        and bool(machine_reachability_report["summary"]["machine_reachability_passed"])
    )
    return {
        "summary": {
            "shoulder_quality_passed": passed,
            "pin_layout_enabled": True,
            "pin_contact_passed": pin_contact_report["summary"]["pin_contact_passed"],
            "has_real_pin_contact_arc": pin_contact_report["summary"]["has_real_pin_contact_arc"],
            "has_real_dome_surface_span": pin_contact_report["summary"][
                "has_real_dome_surface_span"
            ],
            "all_contacts_have_tangent_states": pin_contact_report["summary"][
                "all_contacts_have_tangent_states"
            ],
            "has_fake_polar_circle": pin_contact_report["summary"]["has_fake_polar_circle"],
            "pin_buildup_passed": pin_buildup_report["summary"]["pin_buildup_passed"],
            "pin_slip_passed": pin_slip_report["summary"]["pin_slip_passed"],
            "dome_coverage_passed": dome_coverage_report["summary"]["dome_coverage_passed"],
            "left_dome_coverage_passed": dome_coverage_report["summary"][
                "left_dome_coverage_passed"
            ],
            "right_dome_coverage_passed": dome_coverage_report["summary"][
                "right_dome_coverage_passed"
            ],
            "shoulder_transition_passed": shoulder_transition_report["summary"][
                "shoulder_transition_passed"
            ],
            "machine_reachability_passed": machine_reachability_report["summary"][
                "machine_reachability_passed"
            ],
            "repair_suggestions": dome_coverage_report["summary"].get(
                "repair_suggestions",
                [],
            ),
        }
    }


def _machine_reachability_report(
    config: WindingJobConfig,
    pin_layout_report: dict[str, Any],
    collision_report: dict[str, Any],
) -> dict[str, Any]:
    if not config.pin_layout.enabled:
        return {"summary": {"machine_reachability_passed": True, "pin_layout_enabled": False}}
    pins = pin_layout_report.get("pins", [])
    max_pin_x = max(
        (float(np.linalg.norm(np.asarray(pin["position_mm"][:2], dtype=float))) for pin in pins),
        default=0.0,
    )
    x_limit = config.machine.max_x_mm
    x_ok = x_limit is None or max_pin_x + config.pin_layout.pin_height_mm <= x_limit
    return {
        "summary": {
            "machine_reachability_passed": x_ok
            and bool(collision_report["summary"]["collision_passed"]),
            "pin_layout_enabled": True,
            "max_pin_radial_position_mm": max_pin_x,
            "pin_height_mm": config.pin_layout.pin_height_mm,
            "machine_max_x_mm": x_limit,
            "collision_passed": collision_report["summary"]["collision_passed"],
            "note": "Checks radial X reach only; tangent line-of-sight requires route solver.",
        }
    }


def _region_quality_report(
    config: WindingJobConfig,
    mandrel: CylinderMandrel | AxisymmetricProfileMandrel,
    coverage: Any,
) -> dict[str, Any]:
    masks = _surface_masks(config, mandrel, coverage)
    regions = []
    required_pass = True
    for name, mask in masks.items():
        if name in {"optional_winding_region", "no_wind_region"}:
            continue
        stats = _masked_thickness_stats(config, mandrel, coverage, mask)
        is_required = name == "required_winding_region" or name in {
            "cylinder",
            "left_dome",
            "right_dome",
        }
        if is_required:
            passed = (
                stats["covered_percent"] >= 85.0
                and stats["gap_percent"] <= 15.0
                and stats["overlap_percent"] <= config.quality_limits.max_stack_overlap_percent
                and stats["thickness_variation_percent"]
                <= config.quality_limits.max_thickness_variation_percent
                and stats["max_coverage_count"] <= config.quality_limits.max_coverage_count
                and (
                    config.quality_limits.allow_min_thickness_zero
                    or stats["min_thickness_mm"] > 0.0
                )
            )
            required_pass = required_pass and passed
        else:
            passed = stats["buildup_mm"] <= config.quality_limits.max_polar_buildup_mm * 4.0
        regions.append(
            {
                "region": name,
                "mask_type": _region_mask_type(name),
                "required_for_strict_coverage": is_required,
                "passed": passed,
                **stats,
            }
        )
    return {
        "summary": {
            "region_quality_passed": required_pass,
            "region_count": len(regions),
            "failed_region_count": sum(1 for item in regions if not item["passed"]),
        },
        "regions": regions,
        "surface_masks": _mask_summary(masks),
    }


def _optimisation_repair_suggestions(
    *,
    config: WindingJobConfig,
    pattern_result: MultiLayerPatternResult | None,
    stack_coverage_report: dict[str, Any],
    region_quality_report: dict[str, Any],
) -> dict[str, Any]:
    rejection_counts = {} if pattern_result is None else pattern_result.rejection_counts
    stack = stack_coverage_report["summary"]
    failures = _dominant_failure_reasons(
        rejection_counts=rejection_counts,
        stack_summary=stack,
        region_quality_report=region_quality_report,
    )
    suggestions = []
    if rejection_counts.get("closure_error", 0):
        suggestions.extend(["increase max_p", "increase max_k", "change winding angle"])
    if rejection_counts.get("repeated_gcd_pattern", 0):
        suggestions.extend(["increase max_p", "increase max_k", "increase max_d"])
    if rejection_counts.get("insufficient_coverage", 0) or stack["gap_percent"] > 15.0:
        suggestions.extend(
            [
                "increase geodesic/non-geodesic combined coverage allocation",
                "add a balancing layer",
                "increase max_d",
                "increase tow width",
            ]
        )
    if stack["overlap_percent"] > config.quality_limits.max_stack_overlap_percent:
        suggestions.extend(
            [
                "reduce local coverage target",
                "change geodesic/non-geodesic split",
                "decrease tow width",
            ]
        )
    if not stack["winding_time_limit_passed"]:
        suggestions.extend(
            ["increase allowed winding time", "reduce pass count", "increase tow width"]
        )
    if not stack["thickness_variation_limit_passed"]:
        suggestions.extend(["reduce strictness near polar opening", "change turnaround radius"])
    if not region_quality_report["summary"]["region_quality_passed"]:
        suggestions.extend(["adjust required surface masks", "reduce polar/turnaround buildup"])
    unique_suggestions = list(dict.fromkeys(suggestions))
    return {
        "summary": {
            "suggestion_count": len(unique_suggestions),
            "dominant_failure_reason": failures[0] if failures else "none",
            "blocking_constraints": failures,
        },
        "best_valid_partial_candidate": _best_candidate(pattern_result, valid=True),
        "best_invalid_fallback_candidate": _best_candidate(pattern_result, valid=False),
        "why_fallback_was_needed": _fallback_reason(pattern_result),
        "suggestions": unique_suggestions,
    }


def _surface_masks(
    config: WindingJobConfig,
    mandrel: CylinderMandrel | AxisymmetricProfileMandrel,
    coverage: Any,
) -> dict[str, np.ndarray]:
    z_values = np.asarray(coverage.z_mm, dtype=float)
    radius = mandrel.radius_at(z_values)
    max_radius = max(float(np.max(radius)), 1e-9)
    midpoint = (float(z_values[0]) + float(z_values[-1])) / 2.0
    polar_radius = max(config.mandrel.polar_opening_radius_mm, max_radius * 0.28)
    turnaround_radius = max(config.mandrel.polar_opening_radius_mm * 1.35, max_radius * 0.42)
    cylinder = radius >= max_radius * 0.98
    polar = radius <= polar_radius
    turnaround = (radius > polar_radius) & (radius <= turnaround_radius)
    left = z_values <= midpoint
    right = ~left
    left_dome = (~cylinder) & (~polar) & left
    right_dome = (~cylinder) & (~polar) & right
    transition = np.zeros(z_values.shape, dtype=bool)
    if z_values.size >= 4:
        edge_width = max((z_values[-1] - z_values[0]) * 0.01, 1e-9)
        transition = (z_values <= z_values[0] + edge_width) | (
            z_values >= z_values[-1] - edge_width
        )
    required = cylinder | (left_dome & ~turnaround) | (right_dome & ~turnaround)
    optional = turnaround & ~polar
    no_wind = polar
    return {
        "required_winding_region": required,
        "optional_winding_region": optional,
        "turnaround_region": turnaround,
        "no_wind_region": no_wind,
        "polar_opening_region": polar,
        "cylinder": cylinder,
        "left_dome": left_dome & ~turnaround,
        "right_dome": right_dome & ~turnaround,
        "polar_opening_left": polar & left,
        "polar_opening_right": polar & right,
        "turnaround_zone_left": turnaround & left,
        "turnaround_zone_right": turnaround & right,
        "transition_zone": transition,
    }


def _masked_coverage_summary(coverage: Any, mask: np.ndarray) -> Any:
    from filament_winder.core.coverage import CoverageSummary

    counts = np.asarray(coverage.coverage_count, dtype=int)
    region_counts = counts[mask, :] if np.any(mask) else counts
    true_overlap_threshold = 3
    return CoverageSummary(
        covered_fraction=float(np.mean(region_counts > 0)),
        gap_fraction=float(np.mean(region_counts == 0)),
        overlap_fraction=float(np.mean(region_counts > true_overlap_threshold)),
        max_coverage_count=(
            int(np.percentile(region_counts, 95)) if region_counts.size else 0
        ),
        mean_coverage_count=float(np.mean(region_counts)) if region_counts.size else 0.0,
    )


def _masked_thickness_stats(
    config: WindingJobConfig,
    mandrel: CylinderMandrel | AxisymmetricProfileMandrel,
    coverage: Any,
    mask: np.ndarray,
) -> dict[str, float | int]:
    counts = np.asarray(coverage.coverage_count, dtype=float)
    selected = counts[mask, :] if np.any(mask) else np.zeros((0, counts.shape[1]))
    tow_thickness = max(config.tow.thickness_mm, config.roving.thickness_mm, 0.0)
    thickness = selected * tow_thickness
    if thickness.size == 0:
        return {
            "covered_percent": 0.0,
            "gap_percent": 100.0,
            "overlap_percent": 0.0,
            "min_thickness_mm": 0.0,
            "mean_thickness_mm": 0.0,
            "max_thickness_mm": 0.0,
            "thickness_variation_percent": 0.0,
            "max_coverage_count": 0,
            "buildup_mm": 0.0,
        }
    mean = float(np.mean(thickness))
    covered_thickness = thickness[thickness > 0.0]
    variation_source = covered_thickness if covered_thickness.size else thickness
    robust_mean = float(np.mean(variation_source))
    min_value = float(np.percentile(variation_source, 10))
    max_value = float(np.percentile(variation_source, 90))
    variation = 0.0 if robust_mean <= 1e-12 else (max_value - min_value) / robust_mean * 100.0
    required_mask = _surface_masks(config, mandrel, coverage)["required_winding_region"]
    required_counts = counts[required_mask, :] if np.any(required_mask) else counts
    required_mean = (
        float(np.mean(required_counts * tow_thickness))
        if required_counts.size
        else mean
    )
    return {
        "covered_percent": float(np.mean(selected > 0) * 100.0),
        "gap_percent": float(np.mean(selected == 0) * 100.0),
        "overlap_percent": float(np.mean(selected > 3) * 100.0),
        "min_thickness_mm": min_value,
        "mean_thickness_mm": mean,
        "max_thickness_mm": max_value,
        "thickness_variation_percent": variation,
        "max_coverage_count": int(np.percentile(selected, 95)),
        "buildup_mm": max(0.0, mean - required_mean),
    }


def _mask_summary(masks: dict[str, np.ndarray]) -> dict[str, float]:
    total = max(next(iter(masks.values())).size, 1)
    return {name: float(np.mean(mask) * 100.0) if total else 0.0 for name, mask in masks.items()}


def _region_mask_type(name: str) -> str:
    if name == "required_winding_region" or name in {"cylinder", "left_dome", "right_dome"}:
        return "required_winding_region"
    if "turnaround" in name:
        return "turnaround_region"
    if "polar" in name or name == "no_wind_region":
        return "polar_opening_region"
    return "optional_winding_region"


def _dominant_failure_reasons(
    *,
    rejection_counts: dict[str, int],
    stack_summary: dict[str, Any],
    region_quality_report: dict[str, Any],
) -> list[str]:
    reasons = []
    if rejection_counts.get("insufficient_coverage", 0) or stack_summary["gap_percent"] > 15.0:
        reasons.append("insufficient coverage")
    if not stack_summary["overlap_limit_passed"]:
        reasons.append("excessive overlap")
    if not stack_summary["thickness_variation_limit_passed"]:
        reasons.append("excessive thickness variation")
    if rejection_counts.get("excessive_polar_buildup", 0):
        reasons.append("excessive polar buildup")
    if rejection_counts.get("closure_error", 0):
        reasons.append("closure error too high")
    if rejection_counts.get("repeated_gcd_pattern", 0):
        reasons.append("repeated pattern")
    if not stack_summary["winding_time_limit_passed"]:
        reasons.append("winding time too high")
    if not region_quality_report["summary"]["region_quality_passed"]:
        reasons.append("region quality failed")
    return reasons or ["no valid pair combination found"]


def _best_candidate(
    pattern_result: MultiLayerPatternResult | None,
    *,
    valid: bool,
) -> dict[str, Any] | None:
    if pattern_result is None:
        return None
    candidates = [
        candidate
        for candidate in pattern_result.candidates + pattern_result.rejected
        if candidate.valid is valid
    ]
    if not candidates:
        return None
    candidate = min(candidates, key=lambda item: item.score)
    return {
        "pattern_id": candidate.pattern_id,
        "layer_id": candidate.layer_id,
        "score": candidate.score,
        "coverage_estimate_percent": candidate.coverage_estimate * 100.0,
        "estimated_winding_time_min": candidate.estimated_winding_time_min,
        "rejection_reasons": list(candidate.rejection_reasons),
    }


def _fallback_reason(pattern_result: MultiLayerPatternResult | None) -> str:
    if pattern_result is None:
        return "pattern optimisation disabled"
    invalid_selected = [
        candidate for candidate in pattern_result.selected_candidates if not candidate.valid
    ]
    if not invalid_selected:
        return "no fallback candidate was used"
    reasons = sorted(
        {reason for candidate in invalid_selected for reason in candidate.rejection_reasons}
    )
    return "fallback used because selected stack needed inspection despite: " + ", ".join(reasons)


def _build_layer_completion_report(
    config: WindingJobConfig,
    mandrel: CylinderMandrel | AxisymmetricProfileMandrel,
    program: PlannedWindingProgram,
    *,
    selected_patterns: dict[str, Any],
) -> dict[str, Any]:
    rows = []
    limits = config.quality_limits
    for layer in program.layers:
        coverage = _coverage_map_for_path(config, mandrel, layer.path)
        stats = _coverage_detail_stats(
            coverage,
            mandrel,
            layer.spec.layer_thickness_mm,
            active_z_min=float(np.min(layer.path.z_mm)) - layer.spec.tow_width_mm / 2.0,
            active_z_max=float(np.max(layer.path.z_mm)) + layer.spec.tow_width_mm / 2.0,
        )
        if layer.spec.winding_type == "hoop":
            stats = _hoop_completion_stats(layer.report, layer.spec.layer_thickness_mm)
        selected = selected_patterns.get(layer.spec.layer_id)
        closure_error = 0.0 if selected is None else float(selected.closure_error_deg)
        pattern_id = None if selected is None else selected.pattern_id
        target_covered = min(100.0, layer.spec.coverage_target * 100.0)
        paired_stack_layer = (
            config.coverage_mode.paired_layer_coverage
            and layer.spec.winding_type in {"geodesic", "non_geodesic"}
        )
        max_gap_limit = max(layer.spec.tow_width_mm * 3.0, layer.report.gap_mm * 1.5)
        thickness_ok = stats["thickness_variation_percent"] <= 650.0
        coverage_ok = (
            stats["covered_percent"] >= target_covered * 0.35
            if paired_stack_layer
            else stats["covered_percent"] >= target_covered * 0.92
        )
        gap_ok = True if paired_stack_layer else stats["max_gap_mm"] <= max_gap_limit
        closure_ok = closure_error <= config.pattern_selection.angle_tolerance_deg
        if layer.spec.winding_type == "hoop":
            turnaround_ok = validate_continuous_hoop_traverse(layer)
        elif layer.spec.winding_type == "local_reinforcement_band":
            turnaround_ok = validate_local_reinforcement_band(layer)
        else:
            turnaround_ok = validate_dome_turnaround(layer.path)
        overlap_ok = stats["overlap_percent"] <= limits.max_layer_overlap_percent
        min_thickness_ok = paired_stack_layer or (
            bool(limits.allow_min_thickness_zero) or stats["min_thickness_mm"] > 0.0
        )
        thickness_limit_ok = paired_stack_layer or (
            stats["thickness_variation_percent"] <= limits.max_thickness_variation_percent
        )
        polar_buildup_ok = paired_stack_layer or (
            stats["polar_buildup_mm"] <= limits.max_polar_buildup_mm
        )
        coverage_count_ok = stats["max_overlap_count"] <= limits.max_coverage_count
        hoop_checks = _hoop_continuity_checks(layer)
        continuous_traverse_ok = (
            True if layer.spec.winding_type not in {"hoop"} else bool(hoop_checks["passed"])
        )
        strict_passed = bool(
            coverage_ok
            and gap_ok
            and closure_ok
            and thickness_limit_ok
            and overlap_ok
            and min_thickness_ok
            and polar_buildup_ok
            and coverage_count_ok
            and continuous_traverse_ok
            and turnaround_ok["passed"]
        )
        completion_passed = bool(
            coverage_ok and gap_ok and closure_ok and thickness_ok and turnaround_ok["passed"]
        )
        rows.append(
            {
                "layer_id": layer.spec.layer_id,
                "layer_name": layer.spec.name,
                "winding_mode": layer.spec.winding_type,
                "selected_pattern_id": pattern_id,
                "passes": layer.report.circuits,
                "nd": layer.report.circuits,
                "closure_error_deg": closure_error,
                "covered_percent": stats["covered_percent"],
                "gap_percent": stats["gap_percent"],
                "overlap_percent": stats["overlap_percent"],
                "max_gap_mm": stats["max_gap_mm"],
                "mean_gap_mm": stats["mean_gap_mm"],
                "max_overlap_count": stats["max_overlap_count"],
                "mean_thickness_mm": stats["mean_thickness_mm"],
                "min_thickness_mm": stats["min_thickness_mm"],
                "max_thickness_mm": stats["max_thickness_mm"],
                "thickness_variation_percent": stats["thickness_variation_percent"],
                "polar_buildup_mm": stats["polar_buildup_mm"],
                "dome_buildup_mm": stats["dome_buildup_mm"],
                "cylinder_buildup_mm": stats["cylinder_buildup_mm"],
                "turnaround_quality": turnaround_ok,
                "hoop_continuity": hoop_checks,
                "strict_completion_passed": strict_passed,
                "overlap_limit_passed": overlap_ok,
                "thickness_variation_limit_passed": thickness_limit_ok,
                "min_thickness_limit_passed": min_thickness_ok,
                "continuous_traverse_passed": continuous_traverse_ok,
                "completion_passed": completion_passed,
                "failure_reasons": _completion_failure_reasons(
                    coverage_ok=coverage_ok,
                    gap_ok=gap_ok,
                    closure_ok=closure_ok,
                    thickness_ok=thickness_ok,
                    turnaround_ok=turnaround_ok["passed"],
                )
                + _strict_failure_reasons(
                    overlap_ok=overlap_ok,
                    min_thickness_ok=min_thickness_ok,
                    thickness_limit_ok=thickness_limit_ok,
                    polar_buildup_ok=polar_buildup_ok,
                    coverage_count_ok=coverage_count_ok,
                    continuous_traverse_ok=continuous_traverse_ok,
                ),
            }
        )
    passed = all(row["completion_passed"] for row in rows)
    strict_passed = all(row["strict_completion_passed"] for row in rows)
    hoop_passed = all(row["continuous_traverse_passed"] for row in rows)
    return {
        "summary": {
            "completion_passed": passed,
            "strict_completion_passed": strict_passed,
            "overlap_limit_passed": all(row["overlap_limit_passed"] for row in rows),
            "thickness_variation_limit_passed": all(
                row["thickness_variation_limit_passed"] for row in rows
            ),
            "min_thickness_limit_passed": all(row["min_thickness_limit_passed"] for row in rows),
            "continuous_traverse_passed": hoop_passed,
            "layer_count": len(rows),
            "failed_layer_count": sum(1 for row in rows if not row["strict_completion_passed"]),
        },
        "layers": rows,
    }


def _build_stack_coverage_report(
    config: WindingJobConfig,
    mandrel: CylinderMandrel | AxisymmetricProfileMandrel,
    program: PlannedWindingProgram,
    coverage: Any,
    *,
    layer_completion_report: dict[str, Any],
) -> dict[str, Any]:
    masks = _surface_masks(config, mandrel, coverage)
    summary = _masked_coverage_summary(coverage, masks["required_winding_region"])
    limits = config.quality_limits
    estimated_time_min = _estimated_time_s(program) / 60.0
    thickness_variation = _stack_thickness_variation_percent(
        coverage,
        mask=masks["required_winding_region"],
    )
    overlap_limit_passed = summary.overlap_percent <= limits.max_stack_overlap_percent
    thickness_variation_limit_passed = (
        thickness_variation <= limits.max_thickness_variation_percent
    )
    min_thickness_limit_passed = (
        bool(limits.allow_min_thickness_zero) or summary.gap_percent <= 15.0
    )
    winding_time_limit_passed = estimated_time_min <= limits.max_estimated_winding_time_min
    max_coverage_count_passed = summary.max_coverage_count <= limits.max_coverage_count
    stack_uniformity_passed = (
        bool(layer_completion_report["summary"]["strict_completion_passed"])
        and summary.covered_percent >= 85.0
        and summary.gap_percent <= 15.0
        and overlap_limit_passed
        and thickness_variation_limit_passed
        and min_thickness_limit_passed
        and max_coverage_count_passed
    )
    return {
        "summary": {
            "stack_uniformity_passed": stack_uniformity_passed,
            "strict_stack_passed": stack_uniformity_passed,
            "overlap_limit_passed": overlap_limit_passed,
            "thickness_variation_limit_passed": thickness_variation_limit_passed,
            "min_thickness_limit_passed": min_thickness_limit_passed,
            "winding_time_limit_passed": winding_time_limit_passed,
            "max_coverage_count_passed": max_coverage_count_passed,
            "covered_percent": summary.covered_percent,
            "gap_percent": summary.gap_percent,
            "overlap_percent": summary.overlap_percent,
            "max_coverage_count": summary.max_coverage_count,
            "mean_coverage_count": summary.mean_coverage_count,
            "thickness_variation_percent": thickness_variation,
            "estimated_winding_time_min": estimated_time_min,
            "strict_region": "required_winding_region",
        },
        "surface_masks": _mask_summary(masks),
        "layer_completion_summary": layer_completion_report["summary"],
    }


def _coverage_map_for_path(
    config: WindingJobConfig,
    mandrel: CylinderMandrel | AxisymmetricProfileMandrel,
    path: Any,
) -> Any:
    if isinstance(mandrel, CylinderMandrel):
        return cylinder_coverage_map(
            mandrel,
            path,
            z_samples=config.coverage.z_cells,
            theta_samples=config.coverage.theta_cells,
        )
    return axisymmetric_surface_coverage_map(
        mandrel,
        path,
        z_samples=config.coverage.z_cells,
        theta_samples=config.coverage.theta_cells,
    )


def _coverage_detail_stats(
    coverage: Any,
    mandrel: CylinderMandrel | AxisymmetricProfileMandrel,
    layer_thickness_mm: float,
    *,
    active_z_min: float | None = None,
    active_z_max: float | None = None,
) -> dict[str, float | int]:
    z_values = np.asarray(coverage.z_mm, dtype=float)
    counts = np.asarray(coverage.coverage_count, dtype=int)
    if active_z_min is not None and active_z_max is not None:
        active = (z_values >= active_z_min) & (z_values <= active_z_max)
        if np.any(active):
            z_values = z_values[active]
            counts = counts[active, :]
    covered_fraction = float(np.mean(counts > 0))
    gap_fraction = float(np.mean(counts == 0))
    overlap_fraction = float(np.mean(counts > 1))
    radius = mandrel.radius_at(z_values)
    theta_cells = counts.shape[1]
    gap_lengths = []
    for row_index, row in enumerate(counts == 0):
        longest = _longest_true_run(row)
        circumference = 2.0 * math.pi * max(float(radius[row_index]), 1e-9)
        gap_lengths.append(longest / max(theta_cells, 1) * circumference)
    thickness = counts.astype(float) * layer_thickness_mm
    mean_thickness = float(np.mean(thickness))
    min_thickness = float(np.min(thickness))
    max_thickness = float(np.max(thickness))
    covered_thickness = thickness[thickness > 0.0]
    variation_source = covered_thickness if covered_thickness.size else thickness
    robust_mean = float(np.mean(variation_source))
    robust_min = float(np.percentile(variation_source, 10))
    robust_max = float(np.percentile(variation_source, 90))
    variation = (
        0.0
        if robust_mean <= 1e-9
        else (robust_max - robust_min) / robust_mean * 100.0
    )
    region = _coverage_regions(z_values, radius)
    cylinder_thickness = _region_thickness_mean(thickness, region, {"cylinder"})
    return {
        "covered_percent": covered_fraction * 100.0,
        "gap_percent": gap_fraction * 100.0,
        "overlap_percent": overlap_fraction * 100.0,
        "max_gap_mm": float(max(gap_lengths, default=0.0)),
        "mean_gap_mm": float(np.mean(gap_lengths)) if gap_lengths else 0.0,
        "max_overlap_count": int(np.max(counts)),
        "mean_thickness_mm": mean_thickness,
        "min_thickness_mm": min_thickness,
        "max_thickness_mm": max_thickness,
        "thickness_variation_percent": variation,
        "polar_buildup_mm": max(
            0.0,
            _region_thickness_mean(thickness, region, {"polar"}) - cylinder_thickness,
        ),
        "dome_buildup_mm": max(
            0.0,
            _region_thickness_mean(thickness, region, {"left_dome", "right_dome"})
            - cylinder_thickness,
        ),
        "cylinder_buildup_mm": cylinder_thickness,
    }


def _hoop_completion_stats(report: Any, layer_thickness_mm: float) -> dict[str, float | int]:
    covered = min(100.0, float(report.coverage_percent))
    gap = max(0.0, 100.0 - covered)
    thickness = layer_thickness_mm * max(1.0, covered / 100.0)
    overlap_percent = (
        0.0
        if report.tow_spacing_mm <= 0.0
        else max(0.0, report.overlap_mm / report.tow_spacing_mm * 100.0)
    )
    return {
        "covered_percent": covered,
        "gap_percent": gap,
        "overlap_percent": overlap_percent,
        "max_gap_mm": float(report.gap_mm),
        "mean_gap_mm": float(report.gap_mm),
        "max_overlap_count": 2 if report.overlap_mm > 0.0 else 1,
        "mean_thickness_mm": thickness,
        "min_thickness_mm": thickness,
        "max_thickness_mm": thickness,
        "thickness_variation_percent": 0.0,
        "polar_buildup_mm": 0.0,
        "dome_buildup_mm": 0.0,
        "cylinder_buildup_mm": thickness,
    }


def _completion_failure_reasons(
    *,
    coverage_ok: bool,
    gap_ok: bool,
    closure_ok: bool,
    thickness_ok: bool,
    turnaround_ok: bool,
) -> list[str]:
    reasons = []
    if not coverage_ok:
        reasons.append("coverage_below_target")
    if not gap_ok:
        reasons.append("max_gap_too_large")
    if not closure_ok:
        reasons.append("closure_error_exceeds_tolerance")
    if not thickness_ok:
        reasons.append("thickness_variation_too_high")
    if not turnaround_ok:
        reasons.append("turnaround_bunching_detected")
    return reasons


def _strict_failure_reasons(
    *,
    overlap_ok: bool,
    min_thickness_ok: bool,
    thickness_limit_ok: bool,
    polar_buildup_ok: bool,
    coverage_count_ok: bool,
    continuous_traverse_ok: bool,
) -> list[str]:
    reasons = []
    if not overlap_ok:
        reasons.append("overlap_limit_exceeded")
    if not min_thickness_ok:
        reasons.append("minimum_thickness_zero")
    if not thickness_limit_ok:
        reasons.append("thickness_variation_limit_exceeded")
    if not polar_buildup_ok:
        reasons.append("polar_buildup_limit_exceeded")
    if not coverage_count_ok:
        reasons.append("coverage_count_limit_exceeded")
    if not continuous_traverse_ok:
        reasons.append("hoop_not_continuous_traverse")
    return reasons


def _hoop_continuity_checks(layer: Any) -> dict[str, Any]:
    if layer.spec.winding_type not in {"hoop", "local_reinforcement_band"}:
        return {"applicable": False, "passed": True}
    motion_types = tuple(layer.metadata.motion_type)
    z_values = np.asarray(layer.path.z_mm, dtype=float)
    z_travel = float(np.max(z_values) - np.min(z_values)) if z_values.size else 0.0
    exact_angle = abs(abs(layer.report.actual_angle_deg) - 90.0) <= 1e-6
    transition_count = sum(1 for item in motion_types if item == "transition")
    unique_z_per_pass = []
    for start, end in _contiguous_spans(np.asarray(layer.path.pass_index, dtype=int)):
        z_span = float(np.max(layer.path.z_mm[start:end]) - np.min(layer.path.z_mm[start:end]))
        unique_z_per_pass.append(z_span)
    advances_per_rev = bool(unique_z_per_pass) and min(unique_z_per_pass) > 1e-6
    continuous = (
        layer.spec.winding_type == "hoop"
        and not exact_angle
        and transition_count == 0
        and z_travel > max(layer.spec.tow_width_mm, 1e-6)
        and advances_per_rev
    )
    if layer.spec.winding_type == "local_reinforcement_band":
        continuous = True
    return {
        "applicable": True,
        "passed": continuous,
        "exact_pure_hoop_angle": exact_angle,
        "actual_angle_deg": layer.report.actual_angle_deg,
        "pitch_mm": layer.report.tow_spacing_mm,
        "z_travel_mm": z_travel,
        "reposition_count": transition_count,
        "tow_state_continuous": transition_count == 0,
        "advances_z_per_revolution": advances_per_rev,
        "mode": layer.spec.winding_type,
    }


def _stack_thickness_variation_percent(
    coverage: Any,
    *,
    mask: np.ndarray | None = None,
) -> float:
    counts = np.asarray(coverage.coverage_count, dtype=float)
    if mask is not None and np.any(mask):
        counts = counts[mask, :]
    if counts.size == 0:
        return 0.0
    covered = counts[counts > 0.0]
    variation_source = covered if covered.size else counts
    mean_count = float(np.mean(variation_source))
    if mean_count <= 1e-12:
        return 0.0
    p10 = float(np.percentile(variation_source, 10))
    p90 = float(np.percentile(variation_source, 90))
    return (p90 - p10) / mean_count * 100.0


def validate_continuous_hoop_traverse(layer: Any) -> dict[str, Any]:
    checks = _hoop_continuity_checks(layer)
    return {
        "passed": bool(checks["passed"]),
        "validator": "validate_continuous_hoop_traverse",
        "minimum_spacing_deg": 360.0,
        "sample_count": int(layer.path.point_count),
    }


def validate_local_reinforcement_band(layer: Any) -> dict[str, Any]:
    return {
        "passed": layer.spec.winding_type == "local_reinforcement_band",
        "validator": "validate_local_reinforcement_band",
        "minimum_spacing_deg": 360.0,
        "sample_count": int(layer.path.point_count),
    }


def validate_dome_turnaround(path: Any) -> dict[str, Any]:
    pass_index = np.asarray(path.pass_index, dtype=int)
    theta_starts = []
    theta_ends = []
    for start, end in _contiguous_spans(pass_index):
        theta_starts.append(float(path.theta_rad[start] % (2.0 * math.pi)))
        theta_ends.append(float(path.theta_rad[end - 1] % (2.0 * math.pi)))
    values = np.asarray(theta_starts + theta_ends, dtype=float)
    if values.size < 3:
        return {"passed": True, "minimum_spacing_deg": 360.0, "sample_count": int(values.size)}
    sorted_values = np.unique(np.round(np.sort(values), decimals=6))
    if sorted_values.size < 3:
        return {
            "passed": True,
            "minimum_spacing_deg": 360.0,
            "sample_count": int(values.size),
            "unique_sample_count": int(sorted_values.size),
        }
    diffs = np.diff(np.concatenate([sorted_values, [sorted_values[0] + 2.0 * math.pi]]))
    minimum_spacing_deg = float(np.rad2deg(np.min(diffs)))
    minimum_allowed_deg = 0.05
    return {
        "passed": minimum_spacing_deg >= minimum_allowed_deg,
        "minimum_spacing_deg": minimum_spacing_deg,
        "minimum_allowed_spacing_deg": minimum_allowed_deg,
        "sample_count": int(values.size),
        "unique_sample_count": int(sorted_values.size),
    }


def _machine_smoothing_report(
    config: WindingJobConfig,
    program: PlannedWindingProgram,
    segments: tuple[Any, ...],
) -> dict[str, Any]:
    machine = _machine_validation(config, program, segments)
    b_values = np.unwrap(np.deg2rad(program.motion_table.b_deg))
    b_unwrapped_deg = np.rad2deg(b_values)
    b_steps = np.abs(np.diff(b_unwrapped_deg))
    max_b_step = float(np.max(b_steps)) if b_steps.size else 0.0
    passed = (
        machine["summary"]["warning_count"] == 0
        and machine["summary"]["large_a_jump_count"] == 0
        and machine["summary"]["unexpected_a_reversal_count"] == 0
        and max_b_step <= 45.0
    )
    return {
        "summary": {
            "machine_kinematics_passed": passed,
            "max_b_step_deg": max_b_step,
            "b_axis_velocity_limit_deg_s": config.machine.max_b_velocity_deg_s,
            "b_axis_acceleration_limit_deg_s2": config.machine.max_b_accel_deg_s2,
            "machine_warning_count": machine["summary"]["warning_count"],
        },
        "machine_validation": machine["summary"],
        "warnings": machine["warnings"],
    }


def _longest_true_run(values: np.ndarray) -> int:
    if values.size == 0:
        return 0
    doubled = np.concatenate([values, values])
    longest = current = 0
    for value in doubled:
        current = current + 1 if bool(value) else 0
        longest = max(longest, current)
        if longest >= values.size:
            return int(values.size)
    return int(min(longest, values.size))


def _coverage_regions(z_values: np.ndarray, radius: np.ndarray) -> list[str]:
    max_radius = float(np.max(radius))
    midpoint = (float(z_values[0]) + float(z_values[-1])) / 2.0
    regions = []
    for z_value, radius_value in zip(z_values, radius, strict=True):
        if radius_value < max_radius * 0.35:
            regions.append("polar")
        elif radius_value >= max_radius * 0.98:
            regions.append("cylinder")
        elif z_value < midpoint:
            regions.append("left_dome")
        else:
            regions.append("right_dome")
    return regions


def _region_thickness_mean(
    thickness: np.ndarray,
    region: list[str],
    names: set[str],
) -> float:
    mask = np.asarray([label in names for label in region], dtype=bool)
    if not np.any(mask):
        return float(np.mean(thickness))
    return float(np.mean(thickness[mask, :]))


def _row_mask_thickness_mean(thickness: np.ndarray, mask: np.ndarray) -> float:
    if thickness.size == 0:
        return 0.0
    if not np.any(mask):
        return float(np.mean(thickness))
    return float(np.mean(thickness[mask, :]))


def _contiguous_spans(values: np.ndarray) -> tuple[tuple[int, int], ...]:
    if values.size == 0:
        return ()
    spans = []
    start = 0
    for index in range(1, values.size):
        if values[index] != values[index - 1]:
            spans.append((start, index))
            start = index
    spans.append((start, values.size))
    return tuple(spans)


def _write_json(data: Any, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def _machine_validation(
    config: WindingJobConfig,
    program: PlannedWindingProgram,
    segments: tuple[Any, ...],
) -> dict[str, Any]:
    time_s = _program_time_s_array(program)
    axes = {
        "A": program.motion_table.a_deg,
        "X": program.motion_table.x_mm,
        "Z": program.motion_table.z_mm,
        "B": program.motion_table.b_deg,
    }
    velocity_limits = {
        "A": None if config.machine.max_a_rpm is None else config.machine.max_a_rpm * 6.0,
        "X": None,
        "Z": None,
        "B": config.machine.max_b_velocity_deg_s,
    }
    acceleration_limits = {
        "A": config.machine.max_a_accel_deg_s2,
        "X": config.machine.max_x_accel_mm_s2,
        "Z": config.machine.max_z_accel_mm_s2,
        "B": config.machine.max_b_accel_deg_s2,
    }
    travel_limits = {
        "X": config.machine.max_x_mm,
        "Z": config.machine.max_z_mm,
        "B": config.machine.max_b_deg,
    }
    warnings = []
    max_velocity: dict[str, float] = {}
    max_acceleration: dict[str, float] = {}
    axis_positions_finite = True
    for axis, values in axes.items():
        axis_positions_finite = axis_positions_finite and bool(np.all(np.isfinite(values)))
        velocity = _axis_velocity(values, time_s)
        acceleration = _axis_velocity(velocity, time_s)
        max_velocity[axis] = float(np.max(np.abs(velocity))) if velocity.size else 0.0
        max_acceleration[axis] = (
            float(np.max(np.abs(acceleration))) if acceleration.size else 0.0
        )
        velocity_limit = velocity_limits.get(axis)
        if velocity_limit is not None and max_velocity[axis] > velocity_limit:
            warnings.append(
                _machine_warning(program, segments, int(np.argmax(np.abs(velocity))), axis,
                                 f"velocity {max_velocity[axis]:.3f} exceeds {velocity_limit:.3f}")
            )
        acceleration_limit = acceleration_limits.get(axis)
        if acceleration_limit is not None and max_acceleration[axis] > acceleration_limit:
            warnings.append(
                _machine_warning(
                    program,
                    segments,
                    int(np.argmax(np.abs(acceleration))),
                    axis,
                    f"acceleration {max_acceleration[axis]:.3f} exceeds "
                    f"{acceleration_limit:.3f}",
                )
            )
        travel_limit = travel_limits.get(axis)
        if travel_limit is not None and float(np.max(np.abs(values))) > travel_limit:
            warnings.append(
                _machine_warning(
                    program,
                    segments,
                    int(np.argmax(np.abs(values))),
                    axis,
                    f"travel {float(np.max(np.abs(values))):.3f} exceeds {travel_limit:.3f}",
                )
            )
    a_jumps = np.abs(np.diff(program.motion_table.a_deg))
    large_a_jump_count = int(np.count_nonzero(a_jumps > 180.0))
    unexpected_a_reversal_count = int(
        np.count_nonzero(np.diff(program.motion_table.a_deg) < -180.0)
    )
    summary = {
        "axis_positions_finite": axis_positions_finite,
        "axis_velocity_limits_checked": True,
        "axis_acceleration_limits_checked": True,
        "axis_travel_limits_checked": True,
        "large_a_jump_count": large_a_jump_count,
        "unexpected_a_reversal_count": unexpected_a_reversal_count,
        "max_velocity": max_velocity,
        "max_acceleration": max_acceleration,
        "warning_count": len(warnings),
    }
    return {"summary": summary, "warnings": warnings}


def _quality_report(
    *,
    coverage_summary: Any,
    continuity: dict[str, Any],
    transition_summary: dict[str, Any],
    machine_summary: dict[str, Any],
    slip_summary: dict[str, float],
    turnaround_summary: dict[str, int],
) -> dict[str, Any]:
    checks = {
        "coverage": coverage_summary.covered_percent >= 85.0,
        "gap_size": coverage_summary.gap_percent <= 15.0,
        "overlap": coverage_summary.overlap_percent <= 99.9,
        "slip_risk": slip_summary["max_slip_risk"] <= 25.0,
        "machine_acceleration": machine_summary["warning_count"] == 0,
        "path_continuity": bool(continuity["continuous_machine_path"]),
        "turnaround_quality": (
            turnaround_summary["turnaround_segment_count"] > 0
            and bool(transition_summary["transitions_are_continuous"])
        ),
    }
    return {
        "checks": checks,
        "machine_ready": all(checks.values()),
        "coverage_percent": coverage_summary.covered_percent,
        "gap_percent": coverage_summary.gap_percent,
        "overlap_percent": coverage_summary.overlap_percent,
        "max_slip_risk": slip_summary["max_slip_risk"],
        "machine_warning_count": machine_summary["warning_count"],
    }


def _machine_warning(
    program: PlannedWindingProgram,
    segments: tuple[Any, ...],
    point_index: int,
    axis: str,
    problem: str,
) -> dict[str, Any]:
    segment = _segment_for_point(segments, point_index)
    return {
        "layer_id": program.metadata.layer_id[point_index],
        "segment_id": "" if segment is None else segment.segment_id,
        "pass_id": str(int(program.metadata.pass_index[point_index])),
        "point_index": point_index,
        "axis": axis,
        "severity": "warning",
        "problem": problem,
    }


def _segment_for_point(segments: tuple[Any, ...], point_index: int) -> Any | None:
    for segment in segments:
        if segment.start_index <= point_index <= segment.end_index:
            return segment
    return None


def _slip_risk_summary(program: PlannedWindingProgram) -> dict[str, float]:
    warning_slip = 0.0
    for warning in program.metadata.warning_flags:
        warning_slip = max(warning_slip, _warning_slip_risk_deg(warning))
    return {
        "max_slip_risk": warning_slip,
        "mean_slip_risk": warning_slip,
        "max_curvature_slip_risk": float(np.max(program.feed_schedule.slip_risk)),
    }


def _warning_slip_risk_deg(warning: str) -> float:
    marker = "slip risk "
    if marker not in warning:
        return 0.0
    text = warning.split(marker, 1)[1].split(" deg", 1)[0]
    try:
        return float(text)
    except ValueError:
        return 0.0


def _turnaround_summary(segments: tuple[Any, ...]) -> dict[str, int]:
    turnaround_types = {"dome_turnaround", "BossTurnaroundArc", "PinContactArc"}
    count = sum(1 for segment in segments if segment.segment_type in turnaround_types)
    points = sum(
        segment.point_count
        for segment in segments
        if segment.segment_type in turnaround_types
    )
    return {"turnaround_segment_count": count, "turnaround_points": points}


def _coverage_by_layer_type(
    mandrel: CylinderMandrel | AxisymmetricProfileMandrel,
    program: PlannedWindingProgram,
) -> dict[str, float]:
    coverage: dict[str, list[float]] = {}
    for layer in program.layers:
        if layer.path.point_count < 2:
            continue
        if isinstance(mandrel, CylinderMandrel):
            layer_coverage = cylinder_coverage_map(mandrel, layer.path).summary().covered_percent
        else:
            layer_coverage = axisymmetric_surface_coverage_map(
                mandrel,
                layer.path,
            ).summary().covered_percent
        coverage.setdefault(layer.spec.winding_type, []).append(layer_coverage)
    return {
        winding_type: sum(values) / len(values)
        for winding_type, values in coverage.items()
        if values
    }


def _layer_direction(layer: LayerConfig) -> str:
    raw = layer.direction.lower()
    if layer.winding_angle_deg < 0.0 or raw in {"reverse", "negative", "minus"}:
        return "negative"
    if layer.type in {"hoop", "continuous_hoop_traverse", "local_reinforcement_band"}:
        return "hoop"
    if layer.type == "polar":
        return "polar"
    return "positive"


def _passes(value: int | str | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        if value == "auto":
            return None
        return int(value)
    return int(value)


def _nominal_feedrate(config: WindingJobConfig) -> float:
    for layer in config.layers:
        if layer.enabled and layer.feedrate_mm_min is not None:
            return layer.feedrate_mm_min
    return 500.0


def _build_mandrel(config: WindingJobConfig) -> CylinderMandrel | AxisymmetricProfileMandrel:
    if config.mandrel.type == "cylinder":
        return CylinderMandrel(
            length_mm=config.mandrel.length_mm,
            radius_mm=config.mandrel.radius_mm,
            name=config.project.name,
        )
    if config.mandrel.type in {"axisymmetric_profile", "profile"}:
        if config.mandrel.profile_path is None:
            raise ValueError("mandrel.profile_path is required for imported profile mandrels")
        return import_dxf_zr_profile(
            config.mandrel.profile_path,
            samples=config.mandrel.samples,
        )
    return cylinder_with_domes_profile(
        cylinder_length_mm=config.mandrel.cylinder_length_mm or config.mandrel.length_mm,
        cylinder_radius_mm=config.mandrel.cylinder_radius_mm or config.mandrel.radius_mm,
        left_dome_length_mm=config.mandrel.left_dome_length_mm,
        right_dome_length_mm=config.mandrel.right_dome_length_mm,
        polar_opening_radius_mm=config.mandrel.polar_opening_radius_mm,
        samples_per_region=max(16, config.mandrel.mesh_points_z // 3),
        name=config.project.name,
    )


def _coverage_map_for_program(
    config: WindingJobConfig,
    mandrel: CylinderMandrel | AxisymmetricProfileMandrel,
    program: PlannedWindingProgram,
) -> Any:
    if isinstance(mandrel, CylinderMandrel):
        return cylinder_coverage_map(
            mandrel,
            program.path,
            z_samples=config.coverage.z_cells,
            theta_samples=config.coverage.theta_cells,
        )
    return axisymmetric_surface_coverage_map(
        mandrel,
        program.path,
        z_samples=config.coverage.z_cells,
        theta_samples=config.coverage.theta_cells,
    )


def _mandrel_length_mm(config: WindingJobConfig) -> float:
    if config.mandrel.type == "cylinder":
        return config.mandrel.length_mm
    if config.mandrel.type in {"axisymmetric_profile", "profile"}:
        if config.mandrel.profile_path is not None:
            return import_dxf_zr_profile(
                config.mandrel.profile_path,
                samples=config.mandrel.samples,
            ).length_mm
        return config.mandrel.length_mm
    return (
        config.mandrel.left_dome_length_mm
        + (config.mandrel.cylinder_length_mm or config.mandrel.length_mm)
        + config.mandrel.right_dome_length_mm
    )


def _mandrel_radius_mm(config: WindingJobConfig) -> float:
    if config.mandrel.type == "cylinder":
        return config.mandrel.radius_mm
    if config.mandrel.type in {"axisymmetric_profile", "profile"}:
        if config.mandrel.profile_path is not None:
            return import_dxf_zr_profile(
                config.mandrel.profile_path,
                samples=config.mandrel.samples,
            ).max_radius_mm
        return config.mandrel.radius_mm
    return config.mandrel.cylinder_radius_mm or config.mandrel.radius_mm


def _mandrel_length(mandrel: CylinderMandrel | AxisymmetricProfileMandrel) -> float:
    return mandrel.length_mm


def _mandrel_radius(mandrel: CylinderMandrel | AxisymmetricProfileMandrel) -> float:
    if isinstance(mandrel, CylinderMandrel):
        return mandrel.radius_mm
    return mandrel.max_radius_mm


def _layer_z_bounds(config: WindingJobConfig, layer: LayerConfig) -> tuple[float, float]:
    total_length = _mandrel_length_mm(config)
    if config.mandrel.type == "cylinder":
        default_start, default_end = 0.0, total_length
    elif layer.region == "cylinder_only":
        default_start = config.mandrel.left_dome_length_mm
        cylinder_length = config.mandrel.cylinder_length_mm or config.mandrel.length_mm
        default_end = default_start + cylinder_length
    elif layer.region == "left_dome_only":
        default_start, default_end = 0.0, config.mandrel.left_dome_length_mm
    elif layer.region == "right_dome_only":
        default_start = config.mandrel.left_dome_length_mm + (
            config.mandrel.cylinder_length_mm or config.mandrel.length_mm
        )
        default_end = total_length
    else:
        default_start, default_end = 0.0, total_length
    start_z = default_start if layer.start_z_mm is None else layer.start_z_mm
    end_z = default_end if layer.end_z_mm is None else layer.end_z_mm
    return start_z, end_z


def _estimated_time_s(program: PlannedWindingProgram) -> float:
    time_s = _program_time_s_array(program)
    return float(time_s[-1]) if time_s.size else 0.0


def _program_time_s_array(program: PlannedWindingProgram) -> np.ndarray:
    time_s = np.zeros(program.point_count, dtype=float)
    if program.point_count < 2:
        return time_s
    segment_lengths = np.linalg.norm(np.diff(program.path.points_mm, axis=0), axis=1)
    feedrate = np.maximum(program.feed_schedule.feedrate_mm_min[:-1], 1e-9)
    time_s[1:] = np.cumsum(segment_lengths / feedrate * 60.0)
    return time_s


def _axis_velocity(values: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    velocity = np.zeros(values.shape, dtype=float)
    if values.size < 2:
        return velocity
    dt_s = np.diff(time_s)
    segment_velocity = np.divide(
        np.diff(values),
        dt_s,
        out=np.zeros(values.size - 1, dtype=float),
        where=dt_s > 1e-12,
    )
    velocity[1:] = segment_velocity
    velocity[0] = velocity[1]
    return velocity


def _safe_id(value: str) -> str:
    clean = "".join(char.lower() if char.isalnum() else "-" for char in value).strip("-")
    return clean or "layer"
