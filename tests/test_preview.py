from __future__ import annotations

import numpy as np

from filament_winder.app.exporting import (
    export_cylinder_pattern_preview_files,
    export_preview_files,
    export_profile_dome_pattern_preview_files,
    export_profile_dome_preview_files,
)
from filament_winder.app.gui import gui_dependencies_available, missing_gui_dependency_message
from filament_winder.app.preview import (
    CylinderPreviewConfig,
    PatternPlannerConfig,
    ProfileDomePreviewConfig,
    build_cylinder_pattern_preview_scene,
    build_cylinder_preview_scene,
    build_profile_dome_pattern_preview_scene,
    build_profile_dome_preview_scene,
    offset_display_surface,
    orient_points_for_horizontal_view,
)
from filament_winder.app.project_binding import (
    PreviewExportPaths,
    export_paths_from_directory,
    export_paths_from_project,
    pattern_config_from_project,
    pattern_enabled_from_project,
    preview_config_from_project,
    preview_mode_from_project,
    profile_config_from_project,
    project_from_preview_config,
)
from filament_winder.cli import main


def _write_profile_dxf(path) -> None:
    path.write_text(
        "\n".join(
            [
                "0",
                "SECTION",
                "2",
                "ENTITIES",
                "0",
                "LWPOLYLINE",
                "90",
                "4",
                "10",
                "0",
                "20",
                "0",
                "10",
                "20",
                "20",
                "20",
                "10",
                "80",
                "20",
                "20",
                "10",
                "100",
                "20",
                "0",
                "0",
                "ENDSEC",
                "0",
                "EOF",
            ]
        ),
        encoding="utf-8",
    )


def test_cylinder_preview_scene_builds_mesh_path_and_tow_geometry() -> None:
    preview = build_cylinder_preview_scene(
        CylinderPreviewConfig(
            length_mm=100.0,
            radius_mm=20.0,
            tow_width_mm=3.0,
            winding_angle_deg=45.0,
            points_per_pass=20,
            passes=2,
            coverage_z_samples=12,
            coverage_theta_samples=24,
            mesh_theta_segments=12,
            mesh_z_segments=4,
        )
    )

    assert preview.path.point_count == 40
    assert preview.cylinder_vertices_mm.shape == (60, 3)
    assert preview.cylinder_faces.shape == (96, 3)
    assert preview.tow_vertices_mm.shape == (80, 3)
    assert preview.tow_faces.shape == (78, 3)
    assert preview.display_path_points_mm[0, 0] == -50.0
    assert preview.display_path_points_mm[-1, 0] == -50.0
    assert preview.coverage_summary.max_coverage_count >= 1


def test_profile_dome_preview_scene_builds_profile_mesh_and_path(tmp_path) -> None:
    dxf_path = tmp_path / "profile.dxf"
    _write_profile_dxf(dxf_path)

    preview = build_profile_dome_preview_scene(
        ProfileDomePreviewConfig(
            profile_path=dxf_path,
            tow_width_mm=3.0,
            winding_angle_deg=35.0,
            points_per_span=20,
            turnaround_points=5,
            mesh_theta_segments=12,
            mesh_z_segments=4,
        )
    )

    assert preview.path.point_count == 47
    assert preview.profile_vertices_mm.shape == (60, 3)
    assert preview.profile_faces.shape == (96, 3)
    assert preview.display_path_points_mm.shape == (47, 3)
    assert preview.turnaround_radius_mm > 0.0
    assert preview.motion_table.b_deg.max() == 90.0


def test_profile_nosecone_preview_uses_min_radius_turnaround(tmp_path) -> None:
    dxf_path = tmp_path / "profile.dxf"
    _write_profile_dxf(dxf_path)

    preview = build_profile_dome_preview_scene(
        ProfileDomePreviewConfig(
            profile_path=dxf_path,
            path_mode="nosecone",
            tow_width_mm=3.0,
            winding_angle_deg=35.0,
            points_per_span=20,
            min_radius_mm=5.0,
            turnaround_points=5,
            mesh_theta_segments=12,
            mesh_z_segments=4,
        )
    )

    assert preview.config.path_mode == "nosecone"
    assert preview.turnaround_radius_mm == 5.0
    assert np.min(preview.profile.radius_at(preview.path.z_mm)) >= 5.0 - 1e-9
    assert np.allclose(preview.motion_table.b_deg, 35.0)


def test_profile_axisymmetric_preview_handles_complex_profile(tmp_path) -> None:
    dxf_path = tmp_path / "complex_profile.dxf"
    dxf_path.write_text(
        "\n".join(
            [
                "0",
                "SECTION",
                "2",
                "ENTITIES",
                "0",
                "LWPOLYLINE",
                "90",
                "5",
                "10",
                "0",
                "20",
                "12",
                "10",
                "30",
                "20",
                "24",
                "10",
                "70",
                "20",
                "18",
                "10",
                "110",
                "20",
                "28",
                "10",
                "150",
                "20",
                "22",
                "0",
                "ENDSEC",
                "0",
                "EOF",
            ]
        ),
        encoding="utf-8",
    )

    preview = build_profile_dome_preview_scene(
        ProfileDomePreviewConfig(
            profile_path=dxf_path,
            path_mode="axisymmetric",
            tow_width_mm=3.0,
            winding_angle_deg=30.0,
            points_per_span=30,
            min_radius_mm=5.0,
            mesh_theta_segments=12,
            mesh_z_segments=4,
        )
    )

    assert preview.config.path_mode == "axisymmetric"
    assert preview.path.point_count == 30
    assert preview.path.final_rotation_deg > 0.0
    assert preview.display_profile_vertices_mm.shape == (60, 3)


def test_cylinder_pattern_preview_builds_layer_program() -> None:
    preview = build_cylinder_pattern_preview_scene(
        CylinderPreviewConfig(
            length_mm=80.0,
            radius_mm=10.0,
            tow_width_mm=20.0,
            winding_angle_deg=45.0,
            points_per_pass=10,
            mesh_theta_segments=12,
            mesh_z_segments=4,
        ),
        PatternPlannerConfig(
            coverage_target=0.5,
            include_hoop_layer=True,
            balanced_pm_layers=True,
            max_angle_error_deg=20.0,
        ),
        feedrate_mm_min=700.0,
    )

    assert len(preview.program.layers) == 3
    assert preview.program.point_count > sum(
        layer.path.point_count for layer in preview.program.layers
    )
    assert len(preview.display_layer_path_points_mm) == 3
    assert preview.display_transition_path_points_mm
    assert preview.program.reports[0].winding_type == "hoop"
    assert preview.program.feed_schedule.max_feedrate_mm_min <= 700.0


def test_profile_dome_pattern_preview_builds_geodesic_layer_program(tmp_path) -> None:
    dxf_path = tmp_path / "profile.dxf"
    _write_profile_dxf(dxf_path)

    preview = build_profile_dome_pattern_preview_scene(
        ProfileDomePreviewConfig(
            profile_path=dxf_path,
            tow_width_mm=20.0,
            winding_angle_deg=35.0,
            points_per_span=10,
            turnaround_points=4,
            mesh_theta_segments=12,
            mesh_z_segments=4,
        ),
        PatternPlannerConfig(coverage_target=0.5, balanced_pm_layers=False),
    )

    assert len(preview.program.layers) == 1
    assert preview.program.reports[0].winding_type == "dome"
    assert preview.program.reports[0].circuits == 4
    assert preview.display_layer_path_points_mm[0].shape[1] == 3


def test_profile_nosecone_pattern_preview_reports_nosecone_layer(tmp_path) -> None:
    dxf_path = tmp_path / "profile.dxf"
    _write_profile_dxf(dxf_path)

    preview = build_profile_dome_pattern_preview_scene(
        ProfileDomePreviewConfig(
            profile_path=dxf_path,
            path_mode="nosecone",
            tow_width_mm=20.0,
            winding_angle_deg=35.0,
            points_per_span=10,
            min_radius_mm=5.0,
            turnaround_points=4,
            mesh_theta_segments=12,
            mesh_z_segments=4,
        ),
        PatternPlannerConfig(coverage_target=0.5, balanced_pm_layers=False),
    )

    assert len(preview.program.layers) == 1
    assert preview.program.reports[0].winding_type == "nosecone"
    assert preview.program.metadata.winding_type[0] == "nosecone"


def test_horizontal_view_orientation_maps_engineering_z_to_display_x() -> None:
    display_points = orient_points_for_horizontal_view(
        points_mm=np.asarray(
            [
                [10.0, 0.0, 0.0],
                [0.0, 20.0, 100.0],
            ]
        ),
        length_mm=100.0,
    )

    assert display_points.tolist() == [
        [-50.0, 10.0, 0.0],
        [50.0, 0.0, 20.0],
    ]


def test_display_surface_offset_lifts_points_from_horizontal_axis() -> None:
    offset_points = offset_display_surface(
        np.asarray(
            [
                [0.0, 10.0, 0.0],
                [0.0, 0.0, 20.0],
            ]
        ),
        offset_mm=1.0,
    )

    assert offset_points.tolist() == [
        [0.0, 11.0, 0.0],
        [0.0, 0.0, 21.0],
    ]


def test_preview_cli_delegates_to_launcher(monkeypatch) -> None:
    captured = {}

    def fake_launch(config):
        captured["config"] = config
        return 0

    monkeypatch.setattr("filament_winder.cli.launch_cylinder_preview", fake_launch)

    result = main(
        [
            "preview",
            "--length",
            "500",
            "--radius",
            "50",
            "--tow-width",
            "6",
            "--angle",
            "35",
            "--points",
            "25",
            "--passes",
            "3",
        ]
    )

    assert result == 0
    assert captured["config"].length_mm == 500.0
    assert captured["config"].passes == 3


def test_preview_cli_can_launch_profile_dome_mode(monkeypatch, tmp_path) -> None:
    dxf_path = tmp_path / "profile.dxf"
    _write_profile_dxf(dxf_path)
    captured = {}

    def fake_launch(config, *, profile_config=None, initial_mode="cylinder"):
        captured["config"] = config
        captured["profile_config"] = profile_config
        captured["initial_mode"] = initial_mode
        return 0

    monkeypatch.setattr("filament_winder.cli.launch_cylinder_preview", fake_launch)

    result = main(
        [
            "preview",
            "--profile-dome",
            "--profile",
            str(dxf_path),
            "--tow-width",
            "3",
            "--angle",
            "35",
            "--points",
            "25",
            "--profile-path-mode",
            "nosecone",
            "--min-radius",
            "7",
        ]
    )

    assert result == 0
    assert captured["initial_mode"] == "profile-dome"
    assert captured["profile_config"].profile_path == dxf_path
    assert captured["profile_config"].path_mode == "nosecone"
    assert captured["profile_config"].min_radius_mm == 7.0
    assert captured["profile_config"].points_per_span == 25


def test_gui_dependency_message_is_actionable() -> None:
    assert isinstance(gui_dependencies_available(), bool)
    assert "pip install -e .[gui]" in missing_gui_dependency_message()


def test_preview_project_binding_round_trips_winding_inputs() -> None:
    config = CylinderPreviewConfig(
        length_mm=750.0,
        radius_mm=80.0,
        tow_width_mm=9.0,
        winding_angle_deg=37.0,
        points_per_pass=123,
        passes=5,
        radial_clearance_mm=32.0,
        phase_offset_deg=18.0,
        alternate_direction=False,
        mesh_theta_segments=32,
        mesh_z_segments=8,
    )

    project = project_from_preview_config(config, name="Test GUI project")
    restored = preview_config_from_project(project, base_config=CylinderPreviewConfig())

    assert project.name == "Test GUI project"
    assert restored.length_mm == 750.0
    assert restored.radius_mm == 80.0
    assert restored.tow_width_mm == 9.0
    assert restored.winding_angle_deg == 37.0
    assert restored.points_per_pass == 123
    assert restored.passes == 5
    assert restored.radial_clearance_mm == 32.0
    assert restored.phase_offset_deg == 18.0
    assert not restored.alternate_direction


def test_preview_project_binding_preserves_export_paths() -> None:
    paths = PreviewExportPaths(
        csv_path="out/path.csv",
        gcode_path="out/path.gcode",
        coverage_csv_path="out/coverage.csv",
        coverage_summary_csv_path="out/coverage_summary.csv",
        preview_obj_path="out/preview.obj",
    )

    project = project_from_preview_config(
        CylinderPreviewConfig(),
        export_paths=paths,
        feedrate_mm_min=750.0,
    )

    assert export_paths_from_project(project) == paths
    assert project.machine.feedrate_mm_min == 750.0


def test_export_paths_from_directory_uses_safe_prefix(tmp_path) -> None:
    paths = export_paths_from_directory(tmp_path, prefix="Tank A/Rev 1")

    assert paths.csv_path.endswith("Tank_A_Rev_1_path.csv")
    assert paths.gcode_path.endswith("Tank_A_Rev_1_path.gcode")
    assert paths.coverage_csv_path.endswith("Tank_A_Rev_1_coverage.csv")
    assert paths.coverage_summary_csv_path.endswith("Tank_A_Rev_1_coverage_summary.csv")
    assert paths.preview_obj_path.endswith("Tank_A_Rev_1_preview.obj")


def test_preview_project_binding_round_trips_profile_and_pattern_state() -> None:
    project = project_from_preview_config(
        CylinderPreviewConfig(
            length_mm=750.0,
            radius_mm=80.0,
            tow_width_mm=9.0,
            winding_angle_deg=37.0,
            points_per_pass=123,
        ),
        profile_config=ProfileDomePreviewConfig(
            profile_path="mandrels/nosecone.dxf",
            path_mode="nosecone",
            samples=200,
            min_radius_mm=7.5,
            turnaround_radius_mm=8.0,
            turnaround_points=9,
            turnaround_angle_deg=270.0,
            circuits=3,
        ),
        pattern_config=PatternPlannerConfig(
            coverage_target=0.75,
            include_hoop_layer=True,
            balanced_pm_layers=False,
            max_angle_error_deg=8.5,
        ),
        pattern_enabled=True,
        preview_mode="profile-dome",
    )

    profile = profile_config_from_project(project)
    pattern = pattern_config_from_project(project)

    assert profile.profile_path.name == "nosecone.dxf"
    assert profile.path_mode == "nosecone"
    assert profile.samples == 200
    assert profile.min_radius_mm == 7.5
    assert profile.turnaround_radius_mm == 8.0
    assert profile.turnaround_points == 9
    assert profile.turnaround_angle_deg == 270.0
    assert profile.circuits == 3
    assert pattern.coverage_target == 0.75
    assert pattern.include_hoop_layer
    assert not pattern.balanced_pm_layers
    assert pattern.max_angle_error_deg == 8.5
    assert pattern_enabled_from_project(project)
    assert preview_mode_from_project(project) == "profile-dome"


def test_preview_export_files_writes_all_outputs(tmp_path) -> None:
    paths = PreviewExportPaths(
        csv_path=str(tmp_path / "path.csv"),
        gcode_path=str(tmp_path / "path.gcode"),
        coverage_csv_path=str(tmp_path / "coverage.csv"),
        coverage_summary_csv_path=str(tmp_path / "coverage_summary.csv"),
        preview_obj_path=str(tmp_path / "preview.obj"),
    )

    result = export_preview_files(
        CylinderPreviewConfig(
            length_mm=100.0,
            radius_mm=20.0,
            tow_width_mm=3.0,
            winding_angle_deg=45.0,
            points_per_pass=10,
            passes=2,
            coverage_z_samples=10,
            coverage_theta_samples=12,
        ),
        paths,
        feedrate_mm_min=650.0,
        csv=True,
        gcode=True,
        coverage_csv=True,
        coverage_summary_csv=True,
        preview_obj=True,
    )

    assert len(result.written_paths) == 5
    for path in result.written_paths:
        assert path.exists()
        assert path.stat().st_size > 0


def test_profile_dome_preview_export_writes_csv_and_gcode(tmp_path) -> None:
    dxf_path = tmp_path / "profile.dxf"
    _write_profile_dxf(dxf_path)
    paths = PreviewExportPaths(
        csv_path=str(tmp_path / "profile_dome.csv"),
        gcode_path=str(tmp_path / "profile_dome.gcode"),
    )

    result = export_profile_dome_preview_files(
        ProfileDomePreviewConfig(
            profile_path=dxf_path,
            tow_width_mm=3.0,
            winding_angle_deg=35.0,
            points_per_span=20,
            turnaround_points=5,
        ),
        paths,
        feedrate_mm_min=650.0,
        csv=True,
        gcode=True,
    )

    assert len(result.written_paths) == 2
    for path in result.written_paths:
        assert path.exists()
        assert path.stat().st_size > 0


def test_profile_nosecone_preview_export_writes_csv_and_gcode(tmp_path) -> None:
    dxf_path = tmp_path / "profile.dxf"
    _write_profile_dxf(dxf_path)
    paths = PreviewExportPaths(
        csv_path=str(tmp_path / "profile_nosecone.csv"),
        gcode_path=str(tmp_path / "profile_nosecone.gcode"),
    )

    result = export_profile_dome_preview_files(
        ProfileDomePreviewConfig(
            profile_path=dxf_path,
            path_mode="nosecone",
            tow_width_mm=3.0,
            winding_angle_deg=35.0,
            points_per_span=20,
            min_radius_mm=5.0,
            turnaround_points=5,
        ),
        paths,
        feedrate_mm_min=650.0,
        csv=True,
        gcode=True,
    )

    assert len(result.written_paths) == 2
    for path in result.written_paths:
        assert path.exists()
        assert path.stat().st_size > 0


def test_cylinder_pattern_preview_export_writes_program_outputs(tmp_path) -> None:
    paths = PreviewExportPaths(
        csv_path=str(tmp_path / "pattern.csv"),
        gcode_path=str(tmp_path / "pattern.gcode"),
        coverage_csv_path=str(tmp_path / "pattern_coverage.csv"),
        coverage_summary_csv_path=str(tmp_path / "pattern_summary.csv"),
    )

    result = export_cylinder_pattern_preview_files(
        CylinderPreviewConfig(
            length_mm=80.0,
            radius_mm=10.0,
            tow_width_mm=20.0,
            winding_angle_deg=45.0,
            points_per_pass=10,
            coverage_z_samples=8,
            coverage_theta_samples=12,
        ),
        PatternPlannerConfig(
            coverage_target=0.5,
            include_hoop_layer=True,
            balanced_pm_layers=False,
            max_angle_error_deg=20.0,
        ),
        paths,
        feedrate_mm_min=650.0,
        csv=True,
        gcode=True,
        coverage_csv=True,
        coverage_summary_csv=True,
    )

    assert len(result.written_paths) == 4
    for path in result.written_paths:
        assert path.exists()
        assert path.stat().st_size > 0
    assert result.csv_path is not None
    assert "layer_name" in result.csv_path.read_text(encoding="utf-8")


def test_profile_dome_pattern_preview_export_writes_program_outputs(tmp_path) -> None:
    dxf_path = tmp_path / "profile.dxf"
    _write_profile_dxf(dxf_path)
    paths = PreviewExportPaths(
        csv_path=str(tmp_path / "dome_pattern.csv"),
        gcode_path=str(tmp_path / "dome_pattern.gcode"),
    )

    result = export_profile_dome_pattern_preview_files(
        ProfileDomePreviewConfig(
            profile_path=dxf_path,
            tow_width_mm=20.0,
            winding_angle_deg=35.0,
            points_per_span=10,
            turnaround_points=4,
        ),
        PatternPlannerConfig(coverage_target=0.5, balanced_pm_layers=False),
        paths,
        feedrate_mm_min=650.0,
        csv=True,
        gcode=True,
    )

    assert len(result.written_paths) == 2
    for path in result.written_paths:
        assert path.exists()
        assert path.stat().st_size > 0


def test_profile_axisymmetric_pattern_preview_export_writes_program_outputs(tmp_path) -> None:
    dxf_path = tmp_path / "profile.dxf"
    _write_profile_dxf(dxf_path)
    paths = PreviewExportPaths(
        csv_path=str(tmp_path / "axisymmetric_pattern.csv"),
        gcode_path=str(tmp_path / "axisymmetric_pattern.gcode"),
    )

    result = export_profile_dome_pattern_preview_files(
        ProfileDomePreviewConfig(
            profile_path=dxf_path,
            path_mode="axisymmetric",
            tow_width_mm=20.0,
            winding_angle_deg=35.0,
            points_per_span=10,
            min_radius_mm=5.0,
            turnaround_points=4,
        ),
        PatternPlannerConfig(coverage_target=0.5, balanced_pm_layers=False),
        paths,
        feedrate_mm_min=650.0,
        csv=True,
        gcode=True,
    )

    assert len(result.written_paths) == 2
    for path in result.written_paths:
        assert path.exists()
        assert path.stat().st_size > 0
    assert result.csv_path is not None
    assert "axisymmetric" in result.csv_path.read_text(encoding="utf-8")
