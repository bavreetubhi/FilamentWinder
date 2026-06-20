from __future__ import annotations

import csv

import numpy as np

from filament_winder.core.geometry import AxisymmetricProfileMandrel, CylinderMandrel
from filament_winder.core.path_planning import (
    WindingLayerSpec,
    WindingSchedule,
    axisymmetric_surface_coverage_map,
    plan_winding_schedule,
    validate_winding_program,
)
from filament_winder.io import export_winding_program_csv


def test_cylinder_schedule_plans_alternating_layers_with_transition() -> None:
    mandrel = CylinderMandrel(length_mm=100.0, radius_mm=20.0)
    schedule = WindingSchedule(
        layers=(
            WindingLayerSpec(
                name="+helical",
                winding_type="helical",
                target_angle_deg=52.0,
                tow_width_mm=10.0,
                point_count=20,
                direction="positive",
                max_angle_error_deg=8.0,
            ),
            WindingLayerSpec(
                name="-helical",
                winding_type="helical",
                target_angle_deg=52.0,
                tow_width_mm=10.0,
                point_count=20,
                direction="negative",
                max_angle_error_deg=8.0,
            ),
        ),
        nominal_feedrate_mm_min=500.0,
    )

    program = plan_winding_schedule(mandrel, schedule)
    report = validate_winding_program(program, max_angle_error_deg=8.0, max_gap_mm=2.0)

    assert len(program.layers) == 2
    assert program.point_count > sum(layer.path.point_count for layer in program.layers)
    assert program.reports[0].closes
    assert program.reports[1].closes
    assert np.max(program.motion_table.b_deg) > 0.0
    assert np.min(program.motion_table.b_deg) < 0.0
    assert "transition" in program.metadata.winding_type
    assert not report.has_errors


def test_hoop_schedule_uses_continuous_z_traverse() -> None:
    mandrel = CylinderMandrel(length_mm=30.0, radius_mm=10.0)
    schedule = WindingSchedule(
        layers=(
            WindingLayerSpec(
                name="hoop",
                winding_type="hoop",
                target_angle_deg=90.0,
                tow_width_mm=5.0,
                point_count=12,
                direction="hoop",
            ),
        )
    )

    program = plan_winding_schedule(mandrel, schedule)

    assert 0.0 < program.reports[0].actual_angle_deg < 90.0
    assert program.reports[0].circuits == 6
    assert program.path.pass_count == 6
    assert np.ptp(program.path.z_mm) == 30.0
    assert "transition" not in set(program.metadata.motion_type)
    assert np.all(program.motion_table.b_deg < 90.0)
    for pass_number in range(program.path.pass_count):
        mask = program.path.pass_index == pass_number
        assert np.ptp(program.path.z_mm[mask]) > 0.0


def test_local_reinforcement_band_can_use_fixed_region() -> None:
    mandrel = CylinderMandrel(length_mm=30.0, radius_mm=10.0)
    schedule = WindingSchedule(
        layers=(
            WindingLayerSpec(
                name="band",
                winding_type="local_reinforcement_band",
                target_angle_deg=90.0,
                tow_width_mm=5.0,
                point_count=12,
                direction="hoop",
                start_z_mm=10.0,
                end_z_mm=15.0,
            ),
        )
    )

    program = plan_winding_schedule(mandrel, schedule)

    assert program.reports[0].actual_angle_deg == 90.0
    assert np.allclose(program.motion_table.b_deg, 90.0)


def test_profile_dome_schedule_and_surface_coverage_map() -> None:
    profile = AxisymmetricProfileMandrel(
        z_mm=np.asarray([0.0, 20.0, 80.0, 100.0]),
        r_mm=np.asarray([0.0, 20.0, 20.0, 0.0]),
    )
    schedule = WindingSchedule(
        layers=(
            WindingLayerSpec(
                name="dome",
                winding_type="dome",
                target_angle_deg=35.0,
                tow_width_mm=3.0,
                point_count=20,
                turnaround_points=5,
            ),
        )
    )

    program = plan_winding_schedule(profile, schedule)
    coverage = axisymmetric_surface_coverage_map(profile, program.layers[0].path, z_samples=20)

    assert program.reports[0].circuits > 0
    assert program.reports[0].actual_angle_deg == 35.0
    assert program.motion_table.b_deg.max() == 90.0
    assert coverage.coverage_count.shape == (20, 180)
    assert coverage.summary().max_coverage_count >= 1


def test_profile_nosecone_schedule_uses_turnaround_min_radius() -> None:
    profile = AxisymmetricProfileMandrel(
        z_mm=np.asarray([0.0, 20.0, 80.0, 100.0]),
        r_mm=np.asarray([0.0, 20.0, 20.0, 0.0]),
    )
    schedule = WindingSchedule(
        layers=(
            WindingLayerSpec(
                name="nosecone",
                winding_type="nosecone",
                target_angle_deg=35.0,
                tow_width_mm=3.0,
                point_count=20,
                turnaround_points=5,
                turnaround_radius_mm=5.0,
            ),
        )
    )

    program = plan_winding_schedule(profile, schedule)

    assert program.reports[0].winding_type == "nosecone"
    assert np.min(profile.radius_at(program.layers[0].path.z_mm)) >= 5.0 - 1e-9
    assert np.allclose(program.layers[0].motion_table.b_deg, 35.0)


def test_profile_axisymmetric_schedule_reports_axisymmetric_layer() -> None:
    profile = AxisymmetricProfileMandrel(
        z_mm=np.asarray([0.0, 20.0, 60.0, 100.0]),
        r_mm=np.asarray([10.0, 25.0, 18.0, 22.0]),
    )
    schedule = WindingSchedule(
        layers=(
            WindingLayerSpec(
                name="complex",
                winding_type="axisymmetric",
                target_angle_deg=30.0,
                tow_width_mm=4.0,
                point_count=16,
                turnaround_radius_mm=5.0,
            ),
        )
    )

    program = plan_winding_schedule(profile, schedule)

    assert program.reports[0].winding_type == "axisymmetric"
    assert program.layers[0].path.point_count > 16
    assert program.motion_table.b_deg[0] == 30.0


def test_winding_program_csv_exports_machine_and_layer_metadata(tmp_path) -> None:
    mandrel = CylinderMandrel(length_mm=30.0, radius_mm=10.0)
    schedule = WindingSchedule(
        layers=(
            WindingLayerSpec(
                layer_id="hoop-layer",
                name="hoop",
                winding_type="hoop",
                target_angle_deg=90.0,
                tow_width_mm=5.0,
                point_count=12,
                direction="hoop",
            ),
        )
    )
    program = plan_winding_schedule(mandrel, schedule)

    output_path = export_winding_program_csv(program, tmp_path / "program.csv")

    with output_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == program.point_count
    assert rows[0]["layer_id"] == "hoop-layer"
    assert rows[0]["layer_name"] == "hoop"
    assert rows[0]["winding_type"] == "hoop"
    assert "time_s" in rows[0]
    assert "warning_flags" in rows[0]
    assert "feedrate_mm_min" in rows[0]


def test_disabled_layer_is_preserved_but_not_generated() -> None:
    mandrel = CylinderMandrel(length_mm=40.0, radius_mm=10.0)
    schedule = WindingSchedule(
        layers=(
            WindingLayerSpec(
                layer_id="disabled-layer",
                name="disabled",
                winding_type="helical",
                target_angle_deg=120.0,
                tow_width_mm=5.0,
                enabled=False,
            ),
            WindingLayerSpec(
                layer_id="active-layer",
                name="active",
                winding_type="helical",
                target_angle_deg=45.0,
                tow_width_mm=5.0,
                point_count=12,
                number_of_passes=2,
            ),
        )
    )

    program = plan_winding_schedule(mandrel, schedule)

    assert len(program.layers) == 1
    assert program.layers[0].spec.layer_id == "active-layer"
    assert set(program.metadata.layer_id) == {"active-layer"}


def test_layer_buildup_updates_effective_radius() -> None:
    mandrel = CylinderMandrel(length_mm=40.0, radius_mm=10.0)
    schedule = WindingSchedule(
        layers=(
            WindingLayerSpec(
                layer_id="base",
                name="base",
                winding_type="helical",
                target_angle_deg=40.0,
                tow_width_mm=5.0,
                layer_thickness_mm=2.0,
                point_count=12,
                number_of_passes=2,
                transition_mode="cut_restart",
            ),
            WindingLayerSpec(
                layer_id="second",
                name="second",
                winding_type="helical",
                target_angle_deg=55.0,
                tow_width_mm=4.0,
                layer_thickness_mm=1.0,
                point_count=12,
                number_of_passes=2,
                transition_mode="cut_restart",
            ),
        )
    )

    program = plan_winding_schedule(mandrel, schedule)

    assert program.layers[0].effective_radius_mm == 10.0
    assert program.layers[1].effective_radius_mm == 12.0
    assert program.layers[1].accumulated_thickness_before_mm == 2.0
    second_layer_mask = np.asarray(program.metadata.layer_id) == "second"
    assert np.min(program.metadata.local_radius_mm[second_layer_mask]) == 12.0


def test_layers_can_use_different_passes_feedrate_and_clearance() -> None:
    mandrel = CylinderMandrel(length_mm=40.0, radius_mm=10.0)
    schedule = WindingSchedule(
        layers=(
            WindingLayerSpec(
                layer_id="fast",
                name="fast",
                winding_type="helical",
                target_angle_deg=35.0,
                tow_width_mm=5.0,
                point_count=12,
                number_of_passes=2,
                feedrate_mm_min=600.0,
                mandrel_clearance_mm=20.0,
                transition_mode="cut_restart",
            ),
            WindingLayerSpec(
                layer_id="slow",
                name="slow",
                winding_type="helical",
                target_angle_deg=65.0,
                tow_width_mm=5.0,
                point_count=12,
                number_of_passes=3,
                feedrate_mm_min=300.0,
                mandrel_clearance_mm=40.0,
                transition_mode="cut_restart",
            ),
        ),
        nominal_feedrate_mm_min=500.0,
    )

    program = plan_winding_schedule(mandrel, schedule)

    assert program.layers[0].path.pass_count == 2
    assert program.layers[1].path.pass_count == 3
    assert (
        program.layers[0].feed_schedule.max_feedrate_mm_min
        > program.layers[1].feed_schedule.max_feedrate_mm_min
    )
    assert np.allclose(program.layers[0].motion_table.x_mm, 30.0)
    assert np.allclose(program.layers[1].motion_table.x_mm, 50.0)
