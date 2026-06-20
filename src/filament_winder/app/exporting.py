"""Export current GUI preview state to files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from filament_winder.app.preview import (
    CylinderPreviewConfig,
    PatternPlannerConfig,
    ProfileDomePreviewConfig,
    build_cylinder_pattern_preview_scene,
    build_cylinder_preview_scene,
    build_profile_dome_pattern_preview_scene,
    build_profile_dome_preview_scene,
)
from filament_winder.app.project_binding import PreviewExportPaths
from filament_winder.core.coverage import cylinder_coverage_map
from filament_winder.core.geometry import CylinderMandrel
from filament_winder.io import (
    GCodeOptions,
    export_coverage_csv,
    export_coverage_summary_csv,
    export_cylinder_preview_obj,
    export_gcode,
    export_winding_csv,
    export_winding_program_csv,
)


@dataclass(frozen=True, slots=True)
class PreviewExportResult:
    csv_path: Path | None = None
    gcode_path: Path | None = None
    coverage_csv_path: Path | None = None
    coverage_summary_csv_path: Path | None = None
    preview_obj_path: Path | None = None

    @property
    def written_paths(self) -> tuple[Path, ...]:
        return tuple(
            path
            for path in (
                self.csv_path,
                self.gcode_path,
                self.coverage_csv_path,
                self.coverage_summary_csv_path,
                self.preview_obj_path,
            )
            if path is not None
        )


def export_preview_files(
    config: CylinderPreviewConfig,
    paths: PreviewExportPaths,
    *,
    feedrate_mm_min: float = 500.0,
    csv: bool = False,
    gcode: bool = False,
    coverage_csv: bool = False,
    coverage_summary_csv: bool = False,
    preview_obj: bool = False,
) -> PreviewExportResult:
    scene = build_cylinder_preview_scene(config)
    coverage_map = None
    csv_path = None
    gcode_path = None
    coverage_csv_path = None
    coverage_summary_csv_path = None
    preview_obj_path = None

    if csv:
        csv_path = export_winding_csv(scene.path, scene.motion_table, paths.csv_path)
    if gcode:
        gcode_path = export_gcode(
            scene.motion_table,
            paths.gcode_path,
            options=GCodeOptions(feedrate_mm_min=feedrate_mm_min),
        )
    if coverage_csv or coverage_summary_csv:
        coverage_map = cylinder_coverage_map(
            scene.mandrel,
            scene.path,
            z_samples=config.coverage_z_samples,
            theta_samples=config.coverage_theta_samples,
        )
    if coverage_csv:
        if coverage_map is None:
            raise RuntimeError("coverage map was not generated")
        coverage_csv_path = export_coverage_csv(coverage_map, paths.coverage_csv_path)
    if coverage_summary_csv:
        if coverage_map is None:
            raise RuntimeError("coverage map was not generated")
        coverage_summary_csv_path = export_coverage_summary_csv(
            coverage_map,
            paths.coverage_summary_csv_path,
        )
    if preview_obj:
        preview_obj_path = export_cylinder_preview_obj(
            scene.mandrel,
            scene.path,
            paths.preview_obj_path,
            tow_band=scene.tow_band,
        )

    return PreviewExportResult(
        csv_path=csv_path,
        gcode_path=gcode_path,
        coverage_csv_path=coverage_csv_path,
        coverage_summary_csv_path=coverage_summary_csv_path,
        preview_obj_path=preview_obj_path,
    )


def export_profile_dome_preview_files(
    config: ProfileDomePreviewConfig,
    paths: PreviewExportPaths,
    *,
    feedrate_mm_min: float = 500.0,
    csv: bool = False,
    gcode: bool = False,
) -> PreviewExportResult:
    scene = build_profile_dome_preview_scene(config)
    csv_path = None
    gcode_path = None

    if csv:
        csv_path = export_winding_csv(scene.path, scene.motion_table, paths.csv_path)
    if gcode:
        gcode_path = export_gcode(
            scene.motion_table,
            paths.gcode_path,
            options=GCodeOptions(feedrate_mm_min=feedrate_mm_min),
        )

    return PreviewExportResult(csv_path=csv_path, gcode_path=gcode_path)


def export_cylinder_pattern_preview_files(
    config: CylinderPreviewConfig,
    pattern_config: PatternPlannerConfig,
    paths: PreviewExportPaths,
    *,
    feedrate_mm_min: float = 500.0,
    csv: bool = False,
    gcode: bool = False,
    coverage_csv: bool = False,
    coverage_summary_csv: bool = False,
) -> PreviewExportResult:
    scene = build_cylinder_pattern_preview_scene(
        config,
        pattern_config,
        feedrate_mm_min=feedrate_mm_min,
    )
    coverage_map = None
    csv_path = None
    gcode_path = None
    coverage_csv_path = None
    coverage_summary_csv_path = None

    if csv:
        csv_path = export_winding_program_csv(scene.program, paths.csv_path)
    if gcode:
        gcode_path = export_gcode(
            scene.program.motion_table,
            paths.gcode_path,
            options=GCodeOptions(
                feedrate_mm_min=feedrate_mm_min,
                feed_schedule=scene.program.feed_schedule,
            ),
        )
    if coverage_csv or coverage_summary_csv:
        if not isinstance(scene.mandrel, CylinderMandrel):
            raise RuntimeError("cylinder pattern coverage requires a cylinder mandrel")
        coverage_map = cylinder_coverage_map(
            scene.mandrel,
            scene.program.path,
            z_samples=config.coverage_z_samples,
            theta_samples=config.coverage_theta_samples,
        )
    if coverage_csv:
        if coverage_map is None:
            raise RuntimeError("coverage map was not generated")
        coverage_csv_path = export_coverage_csv(coverage_map, paths.coverage_csv_path)
    if coverage_summary_csv:
        if coverage_map is None:
            raise RuntimeError("coverage map was not generated")
        coverage_summary_csv_path = export_coverage_summary_csv(
            coverage_map,
            paths.coverage_summary_csv_path,
        )

    return PreviewExportResult(
        csv_path=csv_path,
        gcode_path=gcode_path,
        coverage_csv_path=coverage_csv_path,
        coverage_summary_csv_path=coverage_summary_csv_path,
    )


def export_profile_dome_pattern_preview_files(
    config: ProfileDomePreviewConfig,
    pattern_config: PatternPlannerConfig,
    paths: PreviewExportPaths,
    *,
    feedrate_mm_min: float = 500.0,
    csv: bool = False,
    gcode: bool = False,
) -> PreviewExportResult:
    scene = build_profile_dome_pattern_preview_scene(
        config,
        pattern_config,
        feedrate_mm_min=feedrate_mm_min,
    )
    csv_path = None
    gcode_path = None

    if csv:
        csv_path = export_winding_program_csv(scene.program, paths.csv_path)
    if gcode:
        gcode_path = export_gcode(
            scene.program.motion_table,
            paths.gcode_path,
            options=GCodeOptions(
                feedrate_mm_min=feedrate_mm_min,
                feed_schedule=scene.program.feed_schedule,
            ),
        )

    return PreviewExportResult(csv_path=csv_path, gcode_path=gcode_path)
