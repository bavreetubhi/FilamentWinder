"""Command-line entry point for the Version 0.1 prototype."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from filament_winder.app import CylinderPreviewConfig, ProfileDomePreviewConfig
from filament_winder.app.gui import GuiDependencyError, launch_cylinder_preview
from filament_winder.config import load_winding_config
from filament_winder.core.coverage import cylinder_coverage_map
from filament_winder.core.geometry import AxisymmetricProfileMandrel, CylinderMandrel
from filament_winder.core.kinematics import machine_path_from_surface_path
from filament_winder.core.path_planning import (
    CylinderPatternOptimizationRequest,
    HelicalPathConfig,
    HelicalPathGenerator,
    ProfileDomePathConfig,
    ProfileDomePathGenerator,
    ProfileTurnaroundPathConfig,
    ProfileTurnaroundPathGenerator,
    estimate_cylinder_pattern_closure,
    find_profile_safe_zone,
    optimize_cylinder_pattern,
)
from filament_winder.core.tow import generate_cylinder_tow_band
from filament_winder.core.validation import (
    AxisLimitConfig,
    NoGoZone,
    validate_motion_table,
)
from filament_winder.io import (
    GCodeOptions,
    export_coverage_csv,
    export_coverage_summary_csv,
    export_cylinder_preview_obj,
    export_gcode,
    export_winding_csv,
    import_dxf_zr_profile,
)
from filament_winder.project import (
    CylinderMandrelConfig,
    MachineConfig,
    OutputConfig,
    WindingConfig,
    WindingProject,
    save_project,
)
from filament_winder.services import (
    analyze_winding_patterns,
    format_path_validation_report,
    generate_winding_job,
    summarize_winding_job,
    validate_path_csv,
    validate_winding_job_config,
    with_pattern_method,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="filament-winder")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_project = subparsers.add_parser(
        "init-project",
        help="Write a starter YAML config for the CLI winding workflow",
    )
    init_project.add_argument(
        "--output",
        type=Path,
        default=Path("examples/cylinder_stack.yaml"),
        help="Starter config output path",
    )
    init_project.set_defaults(func=_run_init_project)

    validate = subparsers.add_parser("validate", help="Validate a winding job config")
    validate.add_argument("--config", type=Path, required=True, help="YAML/JSON/TOML config path")
    validate.set_defaults(func=_run_validate_config)

    generate = subparsers.add_parser(
        "generate",
        help="Generate paths, CSV, summary JSON, and plots from a config",
    )
    generate.add_argument("--config", type=Path, required=True, help="YAML/JSON/TOML config path")
    generate.add_argument(
        "--pattern-method",
        choices=("legacy_coverage", "textbook", "textbook_integer_closure"),
        help="Override pattern generation method for this run",
    )
    generate.set_defaults(func=_run_generate_config)

    patterns = subparsers.add_parser(
        "patterns",
        help="Search and rank textbook winding pattern candidates",
    )
    patterns.add_argument("--config", type=Path, required=True, help="YAML/JSON/TOML config path")
    patterns.set_defaults(func=_run_patterns_config)

    inspect_pattern = subparsers.add_parser(
        "inspect-pattern",
        help="Print details for one textbook pattern candidate",
    )
    inspect_pattern.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML/JSON/TOML config path",
    )
    inspect_pattern.add_argument("--candidate", required=True, help="Pattern candidate id")
    inspect_pattern.set_defaults(func=_run_inspect_pattern_config)

    inspect_layer = subparsers.add_parser(
        "inspect-layer",
        help="Print per-layer completion details from a generated config",
    )
    inspect_layer.add_argument("--config", type=Path, required=True, help="YAML/JSON/TOML config")
    inspect_layer.add_argument("--layer", required=True, help="Layer id or layer name")
    inspect_layer.set_defaults(func=_run_inspect_layer_config)

    plot_layers = subparsers.add_parser(
        "plot-layers",
        help="Regenerate per-layer, combined, coverage, gap, overlap, and thickness plots",
    )
    plot_layers.add_argument("--config", type=Path, required=True, help="YAML/JSON/TOML config")
    plot_layers.set_defaults(func=_run_plot_layers_config)

    export_csv = subparsers.add_parser("export-csv", help="Generate only the CSV output")
    export_csv.add_argument("--config", type=Path, required=True, help="YAML/JSON/TOML config path")
    export_csv.set_defaults(func=_run_export_csv_config)

    export_gcode_config = subparsers.add_parser(
        "export-gcode",
        help="Generate G-code from a config-driven continuous winding path",
    )
    export_gcode_config.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML/JSON/TOML config path",
    )
    export_gcode_config.add_argument("--output", type=Path, help="Optional G-code output path")
    export_gcode_config.set_defaults(func=_run_export_gcode_config)

    coverage = subparsers.add_parser("coverage", help="Generate and report coverage outputs")
    coverage.add_argument("--config", type=Path, required=True, help="YAML/JSON/TOML config path")
    coverage.set_defaults(func=_run_coverage_config)

    inspect = subparsers.add_parser("inspect", help="Inspect a winding job config")
    inspect.add_argument("--config", type=Path, required=True, help="YAML/JSON/TOML config path")
    inspect.set_defaults(func=_run_inspect_config)

    summary = subparsers.add_parser("summary", help="Generate and print a config summary")
    summary.add_argument("--config", type=Path, required=True, help="YAML/JSON/TOML config path")
    summary.set_defaults(func=_run_summary_config)

    plot = subparsers.add_parser("plot", help="Generate only plot outputs from a config")
    plot.add_argument("--config", type=Path, required=True, help="YAML/JSON/TOML config path")
    plot.set_defaults(func=_run_plot_config)

    validate_path = subparsers.add_parser(
        "validate-path",
        help="Validate an existing generated path CSV and optional summary JSON",
    )
    validate_path.add_argument("--path", type=Path, required=True, help="Generated path CSV")
    validate_path.add_argument("--summary", type=Path, help="Generated summary JSON")
    validate_path.set_defaults(func=_run_validate_path)

    validate_output = subparsers.add_parser(
        "validate-output",
        help="Generate configured outputs, then validate the resulting path CSV",
    )
    validate_output.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML/JSON/TOML config path",
    )
    validate_output.set_defaults(func=_run_validate_output)

    backend_check = subparsers.add_parser(
        "backend-check",
        help="Run full backend generation, validation, export, and report checks",
    )
    backend_check.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML/JSON/TOML config path",
    )
    backend_check.set_defaults(func=_run_backend_check)

    cylinder = subparsers.add_parser("cylinder", help="Generate a cylinder helical winding path")
    cylinder.add_argument("--length", type=float, required=True, help="Mandrel length in mm")
    cylinder.add_argument("--radius", type=float, required=True, help="Mandrel radius in mm")
    cylinder.add_argument("--tow-width", type=float, required=True, help="Tow width in mm")
    cylinder.add_argument("--angle", type=float, required=True, help="Winding angle in degrees")
    cylinder.add_argument("--points", type=int, default=500, help="Generated points per pass")
    cylinder.add_argument("--passes", type=int, default=1, help="Number of helical passes")
    cylinder.add_argument(
        "--phase-offset",
        type=float,
        help="Pass-to-pass phase offset in degrees; defaults to 360 / passes",
    )
    cylinder.add_argument(
        "--no-alternate-direction",
        action="store_true",
        help="Generate every pass from start Z to end Z instead of alternating direction",
    )
    cylinder.add_argument(
        "--clearance",
        type=float,
        default=25.0,
        help="Radial payout-head clearance from the mandrel surface in mm",
    )
    cylinder.add_argument(
        "--csv",
        type=Path,
        default=Path("exports/cylinder_path.csv"),
        help="CSV output path",
    )
    cylinder.add_argument("--gcode", type=Path, help="Optional GRBL-style G-code output path")
    cylinder.add_argument(
        "--feedrate",
        type=float,
        default=500.0,
        help="G-code feedrate in mm/min",
    )
    cylinder.add_argument(
        "--coverage-csv",
        type=Path,
        help="Optional approximate z-theta coverage CSV output path",
    )
    cylinder.add_argument(
        "--coverage-summary-csv",
        type=Path,
        help="Optional one-row coverage summary CSV output path",
    )
    cylinder.add_argument(
        "--coverage-z-samples",
        type=int,
        default=120,
        help="Coverage grid samples along Z",
    )
    cylinder.add_argument(
        "--coverage-theta-samples",
        type=int,
        default=180,
        help="Coverage grid samples around theta",
    )
    cylinder.add_argument(
        "--preview-obj",
        type=Path,
        help="Optional Wavefront OBJ preview output path",
    )
    cylinder.add_argument(
        "--project",
        type=Path,
        help="Optional versioned project JSON output path",
    )
    cylinder.add_argument(
        "--validate",
        action="store_true",
        help="Validate against any limits provided",
    )
    cylinder.add_argument("--a-min", type=float, help="Minimum A axis angle in degrees")
    cylinder.add_argument("--a-max", type=float, help="Maximum A axis angle in degrees")
    cylinder.add_argument("--x-min", type=float, help="Minimum X axis position in mm")
    cylinder.add_argument("--x-max", type=float, help="Maximum X axis position in mm")
    cylinder.add_argument("--z-min", type=float, help="Minimum Z axis position in mm")
    cylinder.add_argument("--z-max", type=float, help="Maximum Z axis position in mm")
    cylinder.add_argument("--b-min", type=float, help="Minimum B axis angle in degrees")
    cylinder.add_argument("--b-max", type=float, help="Maximum B axis angle in degrees")
    cylinder.add_argument(
        "--no-go-zone",
        action="append",
        default=[],
        metavar="NAME,X_MIN,X_MAX,Z_MIN,Z_MAX",
        help="Rectangular X-Z no-go zone; can be repeated",
    )
    cylinder.set_defaults(func=_run_cylinder)

    optimize = subparsers.add_parser(
        "optimize-cylinder",
        help="Find closed cylinder winding patterns for target coverage",
    )
    optimize.add_argument("--length", type=float, required=True, help="Mandrel length in mm")
    optimize.add_argument("--radius", type=float, required=True, help="Mandrel radius in mm")
    optimize.add_argument("--tow-width", type=float, required=True, help="Tow width in mm")
    optimize.add_argument(
        "--target-coverage",
        type=float,
        default=100.0,
        help="Target coverage percentage, e.g. 100 for full coverage",
    )
    optimize.add_argument("--min-angle", type=float, default=10.0, help="Minimum winding angle")
    optimize.add_argument("--max-angle", type=float, default=85.0, help="Maximum winding angle")
    optimize.add_argument(
        "--preferred-angle",
        type=float,
        default=45.0,
        help="Preferred winding angle used to rank otherwise similar candidates",
    )
    optimize.add_argument("--min-passes", type=int, default=1, help="Minimum pass count")
    optimize.add_argument("--max-passes", type=int, default=200, help="Maximum pass count")
    optimize.add_argument("--points", type=int, default=500, help="Generated points per pass")
    optimize.add_argument("--results", type=int, default=10, help="Number of candidates to print")
    optimize.set_defaults(func=_run_optimize_cylinder)

    dxf = subparsers.add_parser("dxf-info", help="Inspect an ASCII DXF Z-R profile")
    dxf.add_argument("path", type=Path, help="DXF profile path")
    dxf.add_argument("--samples", type=int, help="Optional resampled point count")
    dxf.set_defaults(func=_run_dxf_info)

    profile_turnaround = subparsers.add_parser(
        "profile-turnaround",
        help="Generate a pole/opening-safe turnaround path from an ASCII DXF Z-R profile",
    )
    profile_turnaround.add_argument("path", type=Path, help="DXF Z-R profile path")
    profile_turnaround.add_argument("--samples", type=int, help="Optional resampled profile count")
    profile_turnaround.add_argument("--angle", type=float, required=True, help="Winding angle")
    profile_turnaround.add_argument(
        "--tow-width",
        type=float,
        required=True,
        help="Tow width in mm",
    )
    profile_turnaround.add_argument(
        "--min-radius",
        type=float,
        default=5.0,
        help="Minimum radius to avoid poles/openings",
    )
    profile_turnaround.add_argument(
        "--points",
        type=int,
        default=500,
        help="Generated points per outbound/return span",
    )
    profile_turnaround.add_argument(
        "--turnaround-points",
        type=int,
        default=25,
        help="Generated points per turnaround arc",
    )
    profile_turnaround.add_argument(
        "--turnaround-angle",
        type=float,
        default=180.0,
        help="Turnaround arc angle in degrees",
    )
    profile_turnaround.add_argument("--circuits", type=int, default=1, help="Out-and-back cycles")
    profile_turnaround.add_argument(
        "--clearance",
        type=float,
        default=25.0,
        help="Radial payout-head clearance from the profile surface in mm",
    )
    profile_turnaround.add_argument(
        "--csv",
        type=Path,
        default=Path("exports/profile_turnaround_path.csv"),
        help="CSV output path",
    )
    profile_turnaround.add_argument(
        "--gcode",
        type=Path,
        help="Optional GRBL-style G-code output path",
    )
    profile_turnaround.add_argument(
        "--feedrate",
        type=float,
        default=500.0,
        help="G-code feedrate in mm/min",
    )
    profile_turnaround.set_defaults(func=_run_profile_turnaround)

    profile_dome = subparsers.add_parser(
        "profile-dome",
        help="Generate a dome-aware geodesic/helix path from an ASCII DXF Z-R profile",
    )
    profile_dome.add_argument("path", type=Path, help="DXF Z-R profile path")
    profile_dome.add_argument("--samples", type=int, help="Optional resampled profile count")
    profile_dome.add_argument(
        "--angle",
        type=float,
        required=True,
        help="Cylinder/max-radius helix angle in degrees",
    )
    profile_dome.add_argument(
        "--tow-width",
        type=float,
        required=True,
        help="Tow width in mm",
    )
    profile_dome.add_argument(
        "--turnaround-radius",
        type=float,
        help=(
            "Optional boss/opening turnaround radius in mm; must be no smaller "
            "than max_radius * sin(angle)"
        ),
    )
    profile_dome.add_argument(
        "--points",
        type=int,
        default=500,
        help="Generated points per outbound/return dome+helix span",
    )
    profile_dome.add_argument(
        "--turnaround-points",
        type=int,
        default=25,
        help="Generated points per constant-radius turnaround arc",
    )
    profile_dome.add_argument(
        "--turnaround-angle",
        type=float,
        default=180.0,
        help="Turnaround arc angle in degrees",
    )
    profile_dome.add_argument("--circuits", type=int, default=1, help="Out-and-back cycles")
    profile_dome.add_argument(
        "--clearance",
        type=float,
        default=25.0,
        help="Radial payout-head clearance from the profile surface in mm",
    )
    profile_dome.add_argument(
        "--csv",
        type=Path,
        default=Path("exports/profile_dome_path.csv"),
        help="CSV output path",
    )
    profile_dome.add_argument(
        "--gcode",
        type=Path,
        help="Optional GRBL-style G-code output path",
    )
    profile_dome.add_argument(
        "--feedrate",
        type=float,
        default=500.0,
        help="G-code feedrate in mm/min",
    )
    profile_dome.set_defaults(func=_run_profile_dome)

    preview = subparsers.add_parser("preview", help="Launch the live PySide6/VisPy preview")
    preview.add_argument(
        "--profile-dome",
        action="store_true",
        help="Open the GUI directly in profile-dome DXF preview mode",
    )
    preview.add_argument(
        "--profile",
        type=Path,
        default=Path("mandrels/profile.dxf"),
        help="DXF Z-R profile path for --profile-dome",
    )
    preview.add_argument(
        "--profile-samples",
        type=int,
        help="Optional resampled profile count for --profile-dome",
    )
    preview.add_argument(
        "--profile-path-mode",
        choices=("dome", "nosecone", "axisymmetric"),
        default="dome",
        help="Axisymmetric profile path mode for --profile-dome",
    )
    preview.add_argument("--length", type=float, default=1000.0, help="Mandrel length in mm")
    preview.add_argument("--radius", type=float, default=100.0, help="Mandrel radius in mm")
    preview.add_argument("--tow-width", type=float, default=6.0, help="Tow width in mm")
    preview.add_argument("--angle", type=float, default=45.0, help="Winding angle in degrees")
    preview.add_argument("--points", type=int, default=500, help="Generated points per pass")
    preview.add_argument("--passes", type=int, default=1, help="Number of helical passes")
    preview.add_argument(
        "--phase-offset",
        type=float,
        help="Pass-to-pass phase offset in degrees; defaults to 360 / passes",
    )
    preview.add_argument(
        "--no-alternate-direction",
        action="store_true",
        help="Generate every pass from start Z to end Z instead of alternating direction",
    )
    preview.add_argument(
        "--clearance",
        type=float,
        default=25.0,
        help="Radial payout-head clearance from the mandrel surface in mm",
    )
    preview.add_argument(
        "--turnaround-radius",
        type=float,
        help="Optional profile-dome turnaround radius in mm",
    )
    preview.add_argument(
        "--min-radius",
        type=float,
        default=5.0,
        help="Minimum profile radius for nosecone/axisymmetric turnaround modes",
    )
    preview.add_argument(
        "--turnaround-points",
        type=int,
        default=25,
        help="Profile-dome points per turnaround arc",
    )
    preview.add_argument(
        "--turnaround-angle",
        type=float,
        default=180.0,
        help="Profile-dome turnaround arc angle in degrees",
    )
    preview.add_argument(
        "--circuits",
        type=int,
        default=1,
        help="Profile-dome out-and-back cycles",
    )
    preview.add_argument(
        "--coverage-z-samples",
        type=int,
        default=120,
        help="Coverage grid samples along Z",
    )
    preview.add_argument(
        "--coverage-theta-samples",
        type=int,
        default=180,
        help="Coverage grid samples around theta",
    )
    preview.add_argument(
        "--debug-gui",
        action="store_true",
        help="Enable extra GUI logging and Qt exception safety hooks",
    )
    preview.set_defaults(func=_run_preview)
    return parser


def _run_cylinder(args: argparse.Namespace) -> int:
    mandrel = CylinderMandrel(length_mm=args.length, radius_mm=args.radius)
    config = HelicalPathConfig(
        winding_angle_deg=args.angle,
        tow_width_mm=args.tow_width,
        point_count=args.points,
        passes=args.passes,
        phase_offset_deg=args.phase_offset,
        alternate_direction=not args.no_alternate_direction,
    )
    surface_path = HelicalPathGenerator(mandrel, config).generate()
    motion_table = machine_path_from_surface_path(
        surface_path,
        radial_clearance_mm=args.clearance,
    )
    csv_path = export_winding_csv(surface_path, motion_table, args.csv)
    gcode_path = None
    coverage_path = None
    coverage_summary_path = None
    preview_obj_path = None
    closure = estimate_cylinder_pattern_closure(mandrel, config)

    if args.gcode is not None:
        gcode_path = export_gcode(
            motion_table,
            args.gcode,
            options=GCodeOptions(feedrate_mm_min=args.feedrate),
        )
    if args.coverage_csv is not None or args.coverage_summary_csv is not None:
        coverage_map = cylinder_coverage_map(
            mandrel,
            surface_path,
            z_samples=args.coverage_z_samples,
            theta_samples=args.coverage_theta_samples,
        )
        if args.coverage_csv is not None:
            coverage_path = export_coverage_csv(coverage_map, args.coverage_csv)
        summary = coverage_map.summary()
        print(
            "Coverage summary: "
            f"covered={summary.covered_percent:.2f}% "
            f"gap={summary.gap_percent:.2f}% "
            f"overlap={summary.overlap_percent:.2f}% "
            f"max_count={summary.max_coverage_count}"
        )
        if args.coverage_summary_csv is not None:
            coverage_summary_path = export_coverage_summary_csv(
                coverage_map,
                args.coverage_summary_csv,
            )
    if args.preview_obj is not None:
        tow_band = generate_cylinder_tow_band(mandrel, surface_path)
        preview_obj_path = export_cylinder_preview_obj(
            mandrel,
            surface_path,
            args.preview_obj,
            tow_band=tow_band,
        )
    if args.project is not None:
        project = WindingProject(
            name="Cylinder winding",
            mandrel=CylinderMandrelConfig(length_mm=args.length, radius_mm=args.radius),
            winding=WindingConfig(
                tow_width_mm=args.tow_width,
                winding_angle_deg=args.angle,
                point_count=args.points,
                passes=args.passes,
                phase_offset_deg=args.phase_offset,
                alternate_direction=not args.no_alternate_direction,
            ),
            machine=MachineConfig(
                radial_clearance_mm=args.clearance,
                feedrate_mm_min=args.feedrate,
            ),
            outputs=OutputConfig(
                csv_path=str(csv_path),
                gcode_path=None if gcode_path is None else str(gcode_path),
                coverage_csv_path=None if coverage_path is None else str(coverage_path),
                coverage_summary_csv_path=(
                    None if coverage_summary_path is None else str(coverage_summary_path)
                ),
                preview_obj_path=None if preview_obj_path is None else str(preview_obj_path),
            ),
        )
        save_project(project, args.project)

    if args.validate or _has_limit_args(args) or args.no_go_zone:
        report = validate_motion_table(
            motion_table,
            limits=AxisLimitConfig(
                a_min_deg=args.a_min,
                a_max_deg=args.a_max,
                x_min_mm=args.x_min,
                x_max_mm=args.x_max,
                z_min_mm=args.z_min,
                z_max_mm=args.z_max,
                b_min_deg=args.b_min,
                b_max_deg=args.b_max,
            ),
            no_go_zones=tuple(_parse_no_go_zone(value) for value in args.no_go_zone),
        )
        if report.issues:
            for issue in report.issues:
                print(f"Validation {issue.severity}: [{issue.code}] {issue.message}")
        else:
            print("Validation: passed")

    print(f"Wrote {surface_path.point_count} points to {csv_path}")
    if gcode_path is not None:
        print(f"Wrote G-code to {gcode_path}")
    if coverage_path is not None:
        print(f"Wrote coverage map to {coverage_path}")
    if coverage_summary_path is not None:
        print(f"Wrote coverage summary to {coverage_summary_path}")
    if preview_obj_path is not None:
        print(f"Wrote preview OBJ to {preview_obj_path}")
    if args.project is not None:
        print(f"Wrote project file to {args.project}")
    print(f"Final mandrel rotation: {surface_path.final_rotation_deg:.6f} deg")
    print(f"Final turns: {surface_path.final_turns:.6f}")
    print(
        "Pattern closure: "
        f"turns/pass={closure.rotations_per_pass:.6f}, "
        f"nearest={closure.nearest_integer_turns}, "
        f"error={closure.closure_error_deg:.6f} deg, "
        f"phase={closure.phase_offset_deg:.6f} deg, "
        f"band_spacing={closure.band_spacing_mm:.6f} mm"
    )
    return 0


def _run_init_project(args: argparse.Namespace) -> int:
    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        print(f"Config already exists: {output_path}", file=sys.stderr)
        return 1
    output_path.write_text(_starter_config_text(), encoding="utf-8")
    print(f"Wrote starter config: {output_path}")
    return 0


def _run_validate_config(args: argparse.Namespace) -> int:
    try:
        config = load_winding_config(args.config)
        warnings = validate_winding_job_config(config)
    except ValueError as exc:
        print(f"Config validation failed: {exc}", file=sys.stderr)
        return 2
    print(f"Config valid: {args.config}")
    for warning in warnings:
        print(f"Warning: {warning}")
    return 0


def _run_generate_config(args: argparse.Namespace) -> int:
    try:
        config = with_pattern_method(
            load_winding_config(args.config),
            getattr(args, "pattern_method", None),
        )
        result = generate_winding_job(config)
    except ValueError as exc:
        print(f"Generate failed: {exc}", file=sys.stderr)
        return 2
    print(summarize_winding_job(result))
    print(_format_manufacturing_status(result.summary))
    return 0


def _run_patterns_config(args: argparse.Namespace) -> int:
    try:
        config = with_pattern_method(load_winding_config(args.config), "textbook")
        result = analyze_winding_patterns(config)
        if result is None:
            raise ValueError("textbook pattern analysis did not run")
    except ValueError as exc:
        print(f"Pattern search failed: {exc}", file=sys.stderr)
        return 2
    print("Pattern Candidates")
    print("------------------")
    if not result.selected_candidates:
        print("No valid candidates found.")
    for candidate in result.selected_candidates:
        print(
            f"Best for {candidate.layer_name}: {candidate.pattern_id} "
            f"{candidate.pattern_type} p={candidate.p} k={candidate.k} d={candidate.d} "
            f"nd={candidate.nd} closure={candidate.closure_error_deg:.4f} deg "
            f"eff_width={candidate.effective_roving_width_mm:.3f} mm "
            f"coverage={candidate.coverage_estimate * 100.0:.2f}% "
            f"score={candidate.score:.3f}"
        )
    print("\nTop alternatives:")
    for candidate in result.candidates[:10]:
        print(
            f"- {candidate.pattern_id}: {candidate.pattern_type} "
            f"p={candidate.p} k={candidate.k} d={candidate.d} nd={candidate.nd} "
            f"closure={candidate.closure_error_deg:.4f} deg "
            f"time={candidate.estimated_winding_time_min:.2f} min "
            f"score={candidate.score:.3f}"
        )
    if result.rejection_counts:
        print("\nRejected candidate counts:")
        for reason, count in sorted(result.rejection_counts.items()):
            print(f"- {reason}: {count}")
    return 0


def _run_inspect_pattern_config(args: argparse.Namespace) -> int:
    try:
        config = with_pattern_method(load_winding_config(args.config), "textbook")
        result = analyze_winding_patterns(config)
        if result is None:
            raise ValueError("textbook pattern analysis did not run")
    except ValueError as exc:
        print(f"Inspect pattern failed: {exc}", file=sys.stderr)
        return 2
    candidate = next(
        (
            item
            for item in result.candidates + result.rejected
            if item.pattern_id == args.candidate
        ),
        None,
    )
    if candidate is None:
        print(f"Pattern candidate not found: {args.candidate}", file=sys.stderr)
        return 1
    print(f"Pattern: {candidate.pattern_id}")
    print(f"Layer: {candidate.layer_name}")
    print(f"Type: {candidate.pattern_type}")
    print(f"p={candidate.p} k={candidate.k} d={candidate.d} nd={candidate.nd}")
    print(f"Closure error: {candidate.closure_error_deg:.6f} deg")
    print(f"Effective roving width: {candidate.effective_roving_width_mm:.6f} mm")
    print(f"Coverage estimate: {candidate.coverage_estimate * 100.0:.3f}%")
    print(f"Estimated winding time: {candidate.estimated_winding_time_min:.3f} min")
    print(f"Score: {candidate.score:.6f}")
    print(f"Valid: {candidate.valid}")
    if candidate.rejection_reasons:
        print("Rejection reasons:")
        for reason in candidate.rejection_reasons:
            print(f"- {reason}")
    if candidate.warnings:
        print("Warnings:")
        for warning in candidate.warnings:
            print(f"- {warning}")
    print("Thickness summary:")
    for key, value in candidate.thickness_distribution.summary.to_dict().items():
        print(f"- {key}: {value:.6f}")
    return 0


def _run_inspect_layer_config(args: argparse.Namespace) -> int:
    try:
        config = load_winding_config(args.config)
        result = generate_winding_job(config, export_csv=False, make_plots=False)
    except ValueError as exc:
        print(f"Inspect layer failed: {exc}", file=sys.stderr)
        return 2
    report_path = result.layer_completion_report_path
    if report_path is None:
        print("Layer completion report was not generated", file=sys.stderr)
        return 2
    report = json.loads(report_path.read_text(encoding="utf-8"))
    layer = next(
        (
            item
            for item in report["layers"]
            if item["layer_id"] == args.layer or item["layer_name"] == args.layer
        ),
        None,
    )
    if layer is None:
        print(f"Layer not found: {args.layer}", file=sys.stderr)
        return 1
    print(f"Layer: {layer['layer_id']} ({layer['layer_name']})")
    print(f"Mode: {layer['winding_mode']}")
    print(f"Completion: {'PASS' if layer['completion_passed'] else 'FAIL'}")
    print(f"Coverage: {layer['covered_percent']:.3f}%")
    print(f"Gap: {layer['gap_percent']:.3f}% max_gap={layer['max_gap_mm']:.3f} mm")
    print(f"Overlap: {layer['overlap_percent']:.3f}% max_count={layer['max_overlap_count']}")
    print(f"Thickness variation: {layer['thickness_variation_percent']:.3f}%")
    if layer["failure_reasons"]:
        print("Failure reasons:")
        for reason in layer["failure_reasons"]:
            print(f"- {reason}")
    return 0


def _run_plot_layers_config(args: argparse.Namespace) -> int:
    try:
        config = load_winding_config(args.config)
        result = generate_winding_job(
            config,
            export_csv=False,
            export_summary=True,
            make_plots=True,
        )
    except ValueError as exc:
        print(f"Plot layers failed: {exc}", file=sys.stderr)
        return 2
    manifest_path = result.plot_manifest_path
    if manifest_path is None:
        print("Plot manifest was not generated")
        return 0
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(f"Wrote plot manifest: {manifest_path}")
    for item in manifest["plots"]:
        print(f"- {item['type']}: {item['path']}")
    return 0


def _run_export_csv_config(args: argparse.Namespace) -> int:
    try:
        config = load_winding_config(args.config)
        result = generate_winding_job(
            config,
            export_csv=True,
            export_summary=False,
            make_plots=False,
        )
    except ValueError as exc:
        print(f"CSV export failed: {exc}", file=sys.stderr)
        return 2
    if result.csv_path is None:
        print("CSV output disabled by config")
        return 0
    print(f"Wrote CSV: {result.csv_path}")
    return 0


def _run_export_gcode_config(args: argparse.Namespace) -> int:
    try:
        config = load_winding_config(args.config)
        result = generate_winding_job(
            config,
            export_csv=False,
            export_summary=False,
            make_plots=False,
        )
        output_path = args.output or (config.output.directory / "path.gcode")
        gcode_path = export_gcode(
            result.program.motion_table,
            output_path,
            options=GCodeOptions(
                feedrate_mm_min=500.0,
                feed_schedule=result.program.feed_schedule,
            ),
        )
    except ValueError as exc:
        print(f"G-code export failed: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote G-code: {gcode_path}")
    return 0


def _run_coverage_config(args: argparse.Namespace) -> int:
    try:
        config = load_winding_config(args.config)
        result = generate_winding_job(
            config,
            export_csv=False,
            export_summary=True,
            make_plots=False,
        )
    except ValueError as exc:
        print(f"Coverage failed: {exc}", file=sys.stderr)
        return 2
    coverage = result.summary["coverage_summary"]
    print(
        "Coverage Summary\n"
        "----------------\n"
        f"covered={coverage['overall_covered_percent']:.3f}%\n"
        f"gap={coverage['gap_percent']:.3f}%\n"
        f"overlap={coverage['overlap_percent']:.3f}%"
    )
    if result.coverage_grid_path is not None:
        print(f"Wrote coverage grid: {result.coverage_grid_path}")
    return 0


def _run_inspect_config(args: argparse.Namespace) -> int:
    try:
        config = load_winding_config(args.config)
        warnings = validate_winding_job_config(config)
    except ValueError as exc:
        print(f"Inspect failed: {exc}", file=sys.stderr)
        return 2
    enabled = [layer for layer in config.layers if layer.enabled]
    print(
        f"Project: {config.project.name}\n"
        f"Mandrel: {config.mandrel.type}, L={config.mandrel.length_mm:g} mm, "
        f"R={config.mandrel.radius_mm:g} mm\n"
        f"Tow: {config.tow.width_mm:g} mm x {config.tow.thickness_mm:g} mm\n"
        f"Layers: {len(enabled)} enabled / {len(config.layers)} total\n"
        f"Coverage grid: {config.coverage.z_cells} x {config.coverage.theta_cells}\n"
        f"Output: {config.output.directory}"
    )
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"- {warning}")
    return 0


def _run_summary_config(args: argparse.Namespace) -> int:
    try:
        config = load_winding_config(args.config)
        result = generate_winding_job(
            config,
            export_csv=False,
            export_summary=True,
            make_plots=False,
        )
    except ValueError as exc:
        print(f"Summary failed: {exc}", file=sys.stderr)
        return 2
    print(summarize_winding_job(result))
    return 0


def _run_plot_config(args: argparse.Namespace) -> int:
    try:
        config = load_winding_config(args.config)
        result = generate_winding_job(
            config,
            export_csv=False,
            export_summary=False,
            make_plots=True,
        )
    except ValueError as exc:
        print(f"Plot failed: {exc}", file=sys.stderr)
        return 2
    if not result.plot_paths:
        print("Plot output disabled by config")
        return 0
    for path in result.plot_paths:
        print(f"Wrote plot: {path}")
    return 0


def _run_validate_path(args: argparse.Namespace) -> int:
    try:
        result = validate_path_csv(args.path, summary_path=args.summary)
    except (OSError, ValueError) as exc:
        print(f"Path validation failed: {exc}", file=sys.stderr)
        return 2
    print(format_path_validation_report(result))
    return 0 if result.ok else 1


def _run_validate_output(args: argparse.Namespace) -> int:
    try:
        config = load_winding_config(args.config)
        result = generate_winding_job(
            config,
            export_csv=True,
            export_summary=True,
            make_plots=True,
        )
        if result.csv_path is None:
            print("Path validation failed: CSV output is disabled", file=sys.stderr)
            return 2
        validation = validate_path_csv(result.csv_path, summary_path=result.summary_path)
    except (OSError, ValueError) as exc:
        print(f"Output validation failed: {exc}", file=sys.stderr)
        return 2
    print(format_path_validation_report(validation))
    print(_format_manufacturing_status(result.summary))
    return 0 if validation.ok else 1


def _run_backend_check(args: argparse.Namespace) -> int:
    config_ok = False
    path_ok = False
    exports_ok = False
    try:
        config = load_winding_config(args.config)
        validate_winding_job_config(config)
        config_ok = True
        result = generate_winding_job(
            config,
            export_csv=True,
            export_summary=True,
            make_plots=True,
        )
        if result.csv_path is None:
            raise ValueError("CSV output is disabled")
        validation = validate_path_csv(result.csv_path, summary_path=result.summary_path)
        path_ok = validation.ok
        exports_ok = _backend_required_exports_exist(result)
    except (OSError, ValueError) as exc:
        print(f"Backend check failed: {exc}", file=sys.stderr)
        result = None
    summary = {} if result is None else result.summary
    layer_status = summary.get("layer_completion_status", {})
    stack_status = summary.get("stack_uniformity_status", {})
    region_status = summary.get("region_quality_status", {})
    machine_status = summary.get("machine_smoothing_status", {})
    optimisation_status = summary.get("pattern_optimisation_status", {})
    polar_status = summary.get("polar_overbuild_status", {})
    collision_status = summary.get("collision_status", {})
    checks = {
        "Config": config_ok,
        "Pattern optimisation": bool(
            isinstance(optimisation_status, dict)
            and optimisation_status.get("pattern_optimisation_passed")
        ),
        "Path generation": path_ok,
        "Hoop continuity": bool(
            isinstance(layer_status, dict)
            and layer_status.get("continuous_traverse_passed", True)
        ),
        "Layer quality": bool(
            isinstance(layer_status, dict)
            and layer_status.get("strict_completion_passed")
        ),
        "Stack uniformity": bool(
            isinstance(stack_status, dict)
            and stack_status.get("strict_stack_passed")
        ),
        "Region quality": bool(
            isinstance(region_status, dict)
            and region_status.get("region_quality_passed")
        ),
        "Machine kinematics": bool(
            isinstance(machine_status, dict)
            and machine_status.get("machine_kinematics_passed")
        ),
        "Polar overbuild report": bool(
            isinstance(polar_status, dict) and "polar_buildup_mm" in polar_status
        ),
        "Collision": bool(
            isinstance(collision_status, dict) and collision_status.get("collision_passed", True)
        ),
        "Exports": exports_ok,
    }
    overall = all(checks.values())
    print("Backend Check")
    print("-------------")
    for label, passed in checks.items():
        print(f"{label}: {'PASS' if passed else 'FAIL'}")
    print(f"Machine-ready: {str(bool(summary.get('machine_ready'))).lower()}")
    print(f"Overall backend-ready: {str(overall).lower()}")
    return 0 if overall else 1


def _backend_required_exports_exist(result: object) -> bool:
    paths = [
        getattr(result, "summary_path", None),
        getattr(result, "validation_report_path", None),
        getattr(result, "layer_completion_report_path", None),
        getattr(result, "stack_coverage_report_path", None),
        getattr(result, "region_quality_report_path", None),
        getattr(result, "calibration_report_path", None),
        getattr(result, "friction_margin_report_path", None),
        getattr(result, "polar_overbuild_report_path", None),
        getattr(result, "collision_report_path", None),
        getattr(result, "machine_smoothing_report_path", None),
        getattr(result, "pattern_optimisation_report_path", None),
        getattr(result, "candidate_pair_report_path", None),
        getattr(result, "actual_thickness_report_path", None),
        getattr(result, "optimisation_repair_suggestions_path", None),
        getattr(result, "selected_pattern_path", None),
        getattr(result, "pattern_candidates_path", None),
        getattr(result, "pattern_rejection_report_path", None),
        getattr(result, "plot_manifest_path", None),
        getattr(result, "csv_path", None),
        getattr(result, "gcode_path", None),
        getattr(result, "segments_path", None),
        getattr(result, "coverage_grid_path", None),
    ]
    return all(path is not None and Path(path).exists() for path in paths)


def _format_manufacturing_status(summary: dict[str, object]) -> str:
    manufacturing = summary.get("manufacturing_report")
    if not isinstance(manufacturing, dict):
        return ""
    layer_status = summary.get("layer_completion_status")
    stack_status = summary.get("stack_uniformity_status")
    region_status = summary.get("region_quality_status")
    smoothing_status = summary.get("machine_smoothing_status")
    optimisation_status = summary.get("pattern_optimisation_status")
    layer_pass = isinstance(layer_status, dict) and bool(layer_status.get("completion_passed"))
    stack_pass = isinstance(stack_status, dict) and bool(
        stack_status.get("stack_uniformity_passed")
    )
    machine_pass = isinstance(smoothing_status, dict) and bool(
        smoothing_status.get("machine_kinematics_passed")
    )
    optimisation_pass = isinstance(optimisation_status, dict) and bool(
        optimisation_status.get("pattern_optimisation_passed")
    )
    region_pass = isinstance(region_status, dict) and bool(
        region_status.get("region_quality_passed")
    )
    hoop_pass = isinstance(layer_status, dict) and bool(
        layer_status.get("continuous_traverse_passed", True)
    )
    machine_ready = bool(summary.get("machine_ready"))
    return (
        "\nManufacturing Readiness\n"
        "-----------------------\n"
        f"Layer completion: {'PASS' if layer_pass else 'FAIL'}\n"
        f"Hoop continuity: {'PASS' if hoop_pass else 'FAIL'}\n"
        f"Layer quality: {'PASS' if layer_pass else 'FAIL'}\n"
        f"Stack uniformity: {'PASS' if stack_pass else 'FAIL'}\n"
        f"Region quality: {'PASS' if region_pass else 'FAIL'}\n"
        f"Pattern optimisation: {'PASS' if optimisation_pass else 'FAIL'}\n"
        f"Machine kinematics: {'PASS' if machine_pass else 'FAIL'}\n"
        f"Overall machine-ready: {str(machine_ready).lower()}"
    )


def _starter_config_text() -> str:
    return """project:
  name: demo_cylinder_stack
  units: mm

machine:
  axis_order: [A, X, Z, B]
  mandrel_orientation: horizontal
  controller: grbl_compatible
  clearance_mm: 20

mandrel:
  type: cylinder
  length_mm: 1000
  radius_mm: 101.6
  mesh_points_z: 400
  mesh_points_theta: 360

tow:
  tow_id: carbon_tow
  name: carbon_tow
  width_mm: 6.0
  thickness_mm: 0.25

layers:
  - name: hoop_1
    enabled: true
    type: hoop
    winding_angle_deg: 90
    direction: forward
    passes: 1
    coverage_target: 1.0
    feedrate_mm_min: 500
    start_z_mm: 0
    end_z_mm: 1000
    colour: "#999999"
    points: 120

  - name: helical_plus_45
    enabled: true
    type: helical
    winding_angle_deg: 45
    direction: forward
    passes: auto
    coverage_target: 1.0
    feedrate_mm_min: 500
    start_z_mm: 0
    end_z_mm: 1000
    colour: "#cc4444"
    points: 240

  - name: helical_minus_45
    enabled: true
    type: helical
    winding_angle_deg: -45
    direction: reverse
    passes: auto
    coverage_target: 1.0
    feedrate_mm_min: 500
    start_z_mm: 0
    end_z_mm: 1000
    colour: "#4466cc"
    points: 240

output:
  directory: exports/demo_cylinder_stack
  csv: true
  summary_json: true
  gcode: false

plot:
  enabled: true
  show: false
  save: true
  formats: [png]
  modes: [unwrapped, three_d, debug_passes, debug_transitions]
  include_2d_unwrapped: true
  include_3d_path: true
"""


def _run_dxf_info(args: argparse.Namespace) -> int:
    profile = _load_dxf_profile(args.path, samples=args.samples)
    if profile is None:
        return 1
    print(f"Imported {profile.z_mm.size} Z-R profile points from {args.path}")
    print(f"Z range: {profile.start_z_mm:.6f} mm to {profile.end_z_mm:.6f} mm")
    print(f"Length: {profile.length_mm:.6f} mm")
    print(f"Max radius: {profile.max_radius_mm:.6f} mm")
    return 0


def _run_profile_turnaround(args: argparse.Namespace) -> int:
    profile = _load_dxf_profile(args.path, samples=args.samples)
    if profile is None:
        return 1
    try:
        safe_zone = find_profile_safe_zone(profile, min_radius_mm=args.min_radius)
        config = ProfileTurnaroundPathConfig(
            winding_angle_deg=args.angle,
            tow_width_mm=args.tow_width,
            points_per_span=args.points,
            turnaround_points=args.turnaround_points,
            min_radius_mm=args.min_radius,
            turnaround_angle_deg=args.turnaround_angle,
            circuits=args.circuits,
        )
        surface_path = ProfileTurnaroundPathGenerator(profile, config).generate()
    except ValueError as exc:
        print(f"Could not generate profile turnaround path: {exc}", file=sys.stderr)
        return 1
    motion_table = machine_path_from_surface_path(
        surface_path,
        radial_clearance_mm=args.clearance,
    )
    csv_path = export_winding_csv(surface_path, motion_table, args.csv)
    gcode_path = None
    if args.gcode is not None:
        gcode_path = export_gcode(
            motion_table,
            args.gcode,
            options=GCodeOptions(feedrate_mm_min=args.feedrate),
        )
    print(
        "Safe zone: "
        f"z={safe_zone.start_z_mm:.6f}..{safe_zone.end_z_mm:.6f} mm, "
        f"r>={safe_zone.min_radius_mm:.6f} mm"
    )
    print(f"Wrote {surface_path.point_count} points to {csv_path}")
    if gcode_path is not None:
        print(f"Wrote G-code to {gcode_path}")
    print(f"Final mandrel rotation: {surface_path.final_rotation_deg:.6f} deg")
    return 0


def _run_profile_dome(args: argparse.Namespace) -> int:
    profile = _load_dxf_profile(args.path, samples=args.samples)
    if profile is None:
        return 1
    try:
        config = ProfileDomePathConfig(
            winding_angle_deg=args.angle,
            tow_width_mm=args.tow_width,
            points_per_span=args.points,
            turnaround_points=args.turnaround_points,
            turnaround_angle_deg=args.turnaround_angle,
            circuits=args.circuits,
            turnaround_radius_mm=args.turnaround_radius,
        )
        generator = ProfileDomePathGenerator(profile, config)
        surface_path = generator.generate()
    except ValueError as exc:
        print(f"Could not generate profile dome path: {exc}", file=sys.stderr)
        return 1
    motion_table = machine_path_from_surface_path(
        surface_path,
        radial_clearance_mm=args.clearance,
    )
    csv_path = export_winding_csv(surface_path, motion_table, args.csv)
    gcode_path = None
    if args.gcode is not None:
        gcode_path = export_gcode(
            motion_table,
            args.gcode,
            options=GCodeOptions(feedrate_mm_min=args.feedrate),
        )
    print(
        "Dome winding zone: "
        f"z={generator.safe_zone.start_z_mm:.6f}..{generator.safe_zone.end_z_mm:.6f} mm, "
        f"turnaround_radius={generator.turnaround_radius_mm:.6f} mm, "
        f"geodesic_radius={generator.clairaut_radius_mm:.6f} mm"
    )
    print(f"Wrote {surface_path.point_count} points to {csv_path}")
    if gcode_path is not None:
        print(f"Wrote G-code to {gcode_path}")
    print(
        "Tow-eye angle range: "
        f"{motion_table.b_deg.min():.6f}..{motion_table.b_deg.max():.6f} deg"
    )
    print(f"Final mandrel rotation: {surface_path.final_rotation_deg:.6f} deg")
    return 0


def _load_dxf_profile(
    path: Path,
    *,
    samples: int | None,
) -> AxisymmetricProfileMandrel | None:
    try:
        return import_dxf_zr_profile(path, samples=samples)
    except FileNotFoundError:
        print(f"Profile file not found: {path}", file=sys.stderr)
    except (OSError, ValueError) as exc:
        print(f"Could not read DXF profile {path}: {exc}", file=sys.stderr)
    return None


def _run_optimize_cylinder(args: argparse.Namespace) -> int:
    result = optimize_cylinder_pattern(
        CylinderPatternOptimizationRequest(
            length_mm=args.length,
            radius_mm=args.radius,
            tow_width_mm=args.tow_width,
            point_count=args.points,
            target_coverage_fraction=args.target_coverage / 100.0,
            min_angle_deg=args.min_angle,
            max_angle_deg=args.max_angle,
            min_passes=args.min_passes,
            max_passes=args.max_passes,
            preferred_angle_deg=args.preferred_angle,
            max_results=args.results,
        )
    )
    if not result.candidates:
        print("No closed pattern candidates found for the requested constraints.")
        return 1

    print(
        "rank angle_deg passes turns/pass phase_deg coverage_% "
        "band_spacing_mm gap_overlap_mm score"
    )
    for rank, candidate in enumerate(result.candidates, start=1):
        print(
            f"{rank} "
            f"{candidate.winding_angle_deg:.6f} "
            f"{candidate.passes} "
            f"{candidate.turns_per_pass} "
            f"{candidate.phase_offset_deg:.6f} "
            f"{candidate.estimated_coverage_percent:.3f} "
            f"{candidate.band_spacing_mm:.6f} "
            f"{candidate.estimated_gap_overlap_mm:.6f} "
            f"{candidate.score:.6f}"
        )
    return 0


def _run_preview(args: argparse.Namespace) -> int:
    config = CylinderPreviewConfig(
        length_mm=args.length,
        radius_mm=args.radius,
        tow_width_mm=args.tow_width,
        winding_angle_deg=args.angle,
        points_per_pass=args.points,
        passes=args.passes,
        radial_clearance_mm=args.clearance,
        phase_offset_deg=args.phase_offset,
        alternate_direction=not args.no_alternate_direction,
        coverage_z_samples=args.coverage_z_samples,
        coverage_theta_samples=args.coverage_theta_samples,
    )
    try:
        if args.profile_dome:
            profile_config = ProfileDomePreviewConfig(
                profile_path=args.profile,
                samples=args.profile_samples,
                path_mode=args.profile_path_mode,
                tow_width_mm=args.tow_width,
                winding_angle_deg=args.angle,
                points_per_span=args.points,
                min_radius_mm=args.min_radius,
                turnaround_points=args.turnaround_points,
                turnaround_angle_deg=args.turnaround_angle,
                circuits=args.circuits,
                turnaround_radius_mm=args.turnaround_radius,
                radial_clearance_mm=args.clearance,
            )
            return launch_cylinder_preview(
                config,
                profile_config=profile_config,
                initial_mode="profile-dome",
                debug_gui=args.debug_gui,
            )
        return launch_cylinder_preview(config, debug_gui=args.debug_gui)
    except GuiDependencyError as exc:
        print(f"Preview unavailable: {exc}", file=sys.stderr)
        return 2


def _has_limit_args(args: argparse.Namespace) -> bool:
    return any(
        value is not None
        for value in (
            args.a_min,
            args.a_max,
            args.x_min,
            args.x_max,
            args.z_min,
            args.z_max,
            args.b_min,
            args.b_max,
        )
    )


def _parse_no_go_zone(raw_value: str) -> NoGoZone:
    parts = [part.strip() for part in raw_value.split(",")]
    if len(parts) != 5:
        raise argparse.ArgumentTypeError(
            "--no-go-zone must use NAME,X_MIN,X_MAX,Z_MIN,Z_MAX format"
        )
    name, x_min, x_max, z_min, z_max = parts
    try:
        return NoGoZone(
            name=name,
            x_min_mm=float(x_min),
            x_max_mm=float(x_max),
            z_min_mm=float(z_min),
            z_max_mm=float(z_max),
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
