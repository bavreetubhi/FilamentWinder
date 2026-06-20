"""Conversions between GUI preview settings and project files."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

from filament_winder.app.preview import (
    CylinderPreviewConfig,
    PatternPlannerConfig,
    ProfileDomePreviewConfig,
    ProfilePathMode,
)
from filament_winder.project import (
    CylinderMandrelConfig,
    MachineConfig,
    OutputConfig,
    PatternPlannerProjectConfig,
    ProfilePreviewProjectConfig,
    UiProjectConfig,
    WindingConfig,
    WindingProject,
)


@dataclass(frozen=True, slots=True)
class PreviewExportPaths:
    csv_path: str = "exports/cylinder_path.csv"
    gcode_path: str = "exports/cylinder_path.gcode"
    coverage_csv_path: str = "exports/cylinder_coverage.csv"
    coverage_summary_csv_path: str = "exports/cylinder_coverage_summary.csv"
    preview_obj_path: str = "exports/cylinder_preview.obj"


def export_paths_from_directory(
    directory: str | Path,
    *,
    prefix: str = "winding",
) -> PreviewExportPaths:
    output_dir = Path(directory)
    safe_prefix = _safe_export_prefix(prefix)
    return PreviewExportPaths(
        csv_path=str(output_dir / f"{safe_prefix}_path.csv"),
        gcode_path=str(output_dir / f"{safe_prefix}_path.gcode"),
        coverage_csv_path=str(output_dir / f"{safe_prefix}_coverage.csv"),
        coverage_summary_csv_path=str(output_dir / f"{safe_prefix}_coverage_summary.csv"),
        preview_obj_path=str(output_dir / f"{safe_prefix}_preview.obj"),
    )


def project_from_preview_config(
    config: CylinderPreviewConfig,
    *,
    name: str = "Cylinder winding",
    profile_config: ProfileDomePreviewConfig | None = None,
    pattern_config: PatternPlannerConfig | None = None,
    pattern_enabled: bool = False,
    preview_mode: str = "cylinder",
    export_paths: PreviewExportPaths | None = None,
    feedrate_mm_min: float = 500.0,
    graph: dict[str, object] | None = None,
) -> WindingProject:
    paths = PreviewExportPaths() if export_paths is None else export_paths
    profile = ProfileDomePreviewConfig() if profile_config is None else profile_config
    pattern = PatternPlannerConfig() if pattern_config is None else pattern_config
    return WindingProject(
        name=name.strip() or "Cylinder winding",
        mandrel=CylinderMandrelConfig(
            length_mm=config.length_mm,
            radius_mm=config.radius_mm,
        ),
        winding=WindingConfig(
            tow_width_mm=config.tow_width_mm,
            winding_angle_deg=config.winding_angle_deg,
            point_count=config.points_per_pass,
            passes=config.passes,
            phase_offset_deg=config.phase_offset_deg,
            alternate_direction=config.alternate_direction,
        ),
        machine=MachineConfig(
            radial_clearance_mm=config.radial_clearance_mm,
            feedrate_mm_min=feedrate_mm_min,
        ),
        outputs=OutputConfig(
            csv_path=paths.csv_path,
            gcode_path=paths.gcode_path,
            coverage_csv_path=paths.coverage_csv_path,
            coverage_summary_csv_path=paths.coverage_summary_csv_path,
            preview_obj_path=paths.preview_obj_path,
        ),
        profile=ProfilePreviewProjectConfig(
            profile_path=str(profile.profile_path),
            samples=profile.samples,
            path_mode=profile.path_mode,
            min_radius_mm=profile.min_radius_mm,
            turnaround_radius_mm=profile.turnaround_radius_mm,
            turnaround_points=profile.turnaround_points,
            turnaround_angle_deg=profile.turnaround_angle_deg,
            circuits=profile.circuits,
        ),
        pattern=PatternPlannerProjectConfig(
            enabled=pattern_enabled,
            coverage_target=pattern.coverage_target,
            include_hoop_layer=pattern.include_hoop_layer,
            balanced_pm_layers=pattern.balanced_pm_layers,
            max_angle_error_deg=pattern.max_angle_error_deg,
        ),
        ui=UiProjectConfig(preview_mode=preview_mode),
        graph={} if graph is None else dict(graph),
    )


def preview_config_from_project(
    project: WindingProject,
    *,
    base_config: CylinderPreviewConfig | None = None,
) -> CylinderPreviewConfig:
    base = CylinderPreviewConfig() if base_config is None else base_config
    return replace(
        base,
        length_mm=project.mandrel.length_mm,
        radius_mm=project.mandrel.radius_mm,
        tow_width_mm=project.winding.tow_width_mm,
        winding_angle_deg=project.winding.winding_angle_deg,
        points_per_pass=project.winding.point_count,
        passes=project.winding.passes,
        radial_clearance_mm=project.machine.radial_clearance_mm,
        phase_offset_deg=project.winding.phase_offset_deg,
        alternate_direction=project.winding.alternate_direction,
    )


def profile_config_from_project(
    project: WindingProject,
    *,
    base_config: ProfileDomePreviewConfig | None = None,
) -> ProfileDomePreviewConfig:
    base = ProfileDomePreviewConfig() if base_config is None else base_config
    return replace(
        base,
        profile_path=Path(project.profile.profile_path),
        samples=project.profile.samples,
        path_mode=_project_profile_path_mode(project.profile.path_mode),
        min_radius_mm=project.profile.min_radius_mm,
        turnaround_radius_mm=project.profile.turnaround_radius_mm,
        turnaround_points=project.profile.turnaround_points,
        turnaround_angle_deg=project.profile.turnaround_angle_deg,
        circuits=project.profile.circuits,
        tow_width_mm=project.winding.tow_width_mm,
        winding_angle_deg=project.winding.winding_angle_deg,
        points_per_span=project.winding.point_count,
        radial_clearance_mm=project.machine.radial_clearance_mm,
    )


def pattern_config_from_project(
    project: WindingProject,
    *,
    base_config: PatternPlannerConfig | None = None,
) -> PatternPlannerConfig:
    base = PatternPlannerConfig() if base_config is None else base_config
    return replace(
        base,
        coverage_target=project.pattern.coverage_target,
        include_hoop_layer=project.pattern.include_hoop_layer,
        balanced_pm_layers=project.pattern.balanced_pm_layers,
        max_angle_error_deg=project.pattern.max_angle_error_deg,
    )


def pattern_enabled_from_project(project: WindingProject) -> bool:
    return project.pattern.enabled


def preview_mode_from_project(project: WindingProject) -> str:
    if project.ui.preview_mode in {"cylinder", "profile-dome"}:
        return project.ui.preview_mode
    return "cylinder"


def export_paths_from_project(project: WindingProject) -> PreviewExportPaths:
    return PreviewExportPaths(
        csv_path=project.outputs.csv_path,
        gcode_path=project.outputs.gcode_path or PreviewExportPaths().gcode_path,
        coverage_csv_path=(
            project.outputs.coverage_csv_path or PreviewExportPaths().coverage_csv_path
        ),
        coverage_summary_csv_path=(
            project.outputs.coverage_summary_csv_path
            or PreviewExportPaths().coverage_summary_csv_path
        ),
        preview_obj_path=project.outputs.preview_obj_path or PreviewExportPaths().preview_obj_path,
    )


def _project_profile_path_mode(raw_value: str) -> ProfilePathMode:
    if raw_value in {"dome", "nosecone", "axisymmetric"}:
        return cast(ProfilePathMode, raw_value)
    return "dome"


def _safe_export_prefix(prefix: str) -> str:
    clean = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in prefix)
    return clean.strip("_") or "winding"
