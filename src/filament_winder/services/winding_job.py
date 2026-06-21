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
from filament_winder.core.geometry import (
    AxisymmetricProfileMandrel,
    CylinderMandrel,
    cylinder_with_domes_profile,
)
from filament_winder.core.path_planning import (
    MultiLayerPatternResult,
    PatternSearchRequest,
    PlannedWindingProgram,
    WindingLayerSpec,
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
from filament_winder.plot import plot_layer_diagnostics, plot_winding_program
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
    optimisation_repair_suggestions_path: Path | None
    plot_manifest_path: Path | None
    plot_paths: tuple[Path, ...]
    summary: dict[str, Any]


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
    pattern_optimisation_report = _pattern_optimisation_report(
        config,
        pattern_result,
        layer_completion_report=layer_completion_report,
        stack_coverage_report=stack_coverage_report,
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
            else selected_pattern.k * 360.0 / max(selected_pattern.nd, 1)
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
        "machine_ready": (
            quality_report["machine_ready"]
            and bool(layer_completion_report["summary"]["completion_passed"])
            and bool(layer_completion_report["summary"]["strict_completion_passed"])
            and bool(layer_completion_report["summary"]["continuous_traverse_passed"])
            and bool(stack_coverage_report["summary"]["stack_uniformity_passed"])
            and bool(stack_coverage_report["summary"]["strict_stack_passed"])
            and bool(region_quality_report["summary"]["region_quality_passed"])
            and bool(pattern_optimisation_report["summary"]["pattern_optimisation_passed"])
            and bool(machine_smoothing_report["summary"]["machine_kinematics_passed"])
            and bool(calibration_report["summary"]["calibration_passed"])
            and bool(friction_margin_report["summary"]["friction_margin_passed"])
            and bool(polar_overbuild_report["summary"]["polar_overbuild_passed"])
            and bool(collision_report["summary"]["collision_passed"])
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
        "machine_ready": (
            quality_report["machine_ready"]
            and bool(layer_completion_report["summary"]["completion_passed"])
            and bool(layer_completion_report["summary"]["strict_completion_passed"])
            and bool(layer_completion_report["summary"]["continuous_traverse_passed"])
            and bool(stack_coverage_report["summary"]["stack_uniformity_passed"])
            and bool(stack_coverage_report["summary"]["strict_stack_passed"])
            and bool(region_quality_report["summary"]["region_quality_passed"])
            and bool(pattern_optimisation_report["summary"]["pattern_optimisation_passed"])
            and bool(machine_smoothing_report["summary"]["machine_kinematics_passed"])
            and bool(calibration_report["summary"]["calibration_passed"])
            and bool(friction_margin_report["summary"]["friction_margin_passed"])
            and bool(polar_overbuild_report["summary"]["polar_overbuild_passed"])
            and bool(collision_report["summary"]["collision_passed"])
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
        "machine_ready": (
            bool(layer_completion_report["summary"]["strict_completion_passed"])
            and bool(stack_coverage_report["summary"]["strict_stack_passed"])
            and bool(machine_smoothing_report["summary"]["machine_kinematics_passed"])
            and bool(calibration_report["summary"]["calibration_passed"])
            and bool(friction_margin_report["summary"]["friction_margin_passed"])
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
    passed = (
        bool(layer_summary["strict_completion_passed"])
        and bool(stack_summary["strict_stack_passed"])
        and total_time <= config.quality_limits.max_estimated_winding_time_min
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
            "polar_buildup_mm": max(0.0, polar_mean - cylinder_mean),
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
    missing = []
    if not width_calibrated:
        missing.append("calibrated_effective_width_mm")
    if not friction_calibrated:
        missing.append("calibrated_friction_coefficient")
    return {
        "summary": {
            "calibration_passed": width_calibrated and friction_calibrated,
            "effective_width_calibrated": width_calibrated,
            "friction_calibrated": friction_calibrated,
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
    passed = polar_buildup_mm <= config.quality_limits.max_polar_buildup_mm
    return {
        "summary": {
            "polar_overbuild_passed": passed,
            "polar_buildup_mm": polar_buildup_mm,
            "max_polar_buildup_mm": config.quality_limits.max_polar_buildup_mm,
            "actual_thickness_variation_percent": float(
                summary.get("actual_thickness_variation_percent", 0.0)
            ),
        },
        "source": "actual_thickness_report",
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
        and winding_time_limit_passed
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
    minimum_allowed_deg = max(0.05, 360.0 / max(values.size * 20.0, 1.0))
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
    count = sum(1 for segment in segments if segment.segment_type == "dome_turnaround")
    points = sum(
        segment.point_count
        for segment in segments
        if segment.segment_type == "dome_turnaround"
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
