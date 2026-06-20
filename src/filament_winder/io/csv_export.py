"""CSV exporters for prototype winding programs."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from filament_winder.core.feedrate import FeedSchedule
from filament_winder.core.kinematics import MachineMotionTable
from filament_winder.core.path_planning import (
    PlannedWindingProgram,
    SurfacePath,
    build_path_segments,
    segment_labels_for_points,
)


def export_motion_table_csv(motion_table: MachineMotionTable, output_path: str | Path) -> Path:
    """Export A/X/Z/B machine positions."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["index", "A_deg", "X_mm", "Z_mm", "B_deg"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(motion_table.rows())
    return path


def export_winding_csv(
    surface_path: SurfacePath,
    motion_table: MachineMotionTable,
    output_path: str | Path,
    *,
    feed_schedule: FeedSchedule | None = None,
) -> Path:
    """Export surface centreline points and matching machine positions."""

    if surface_path.point_count != motion_table.point_count:
        raise ValueError("surface path and motion table must have the same point count")
    if feed_schedule is not None and feed_schedule.point_count != surface_path.point_count:
        raise ValueError("feed schedule must have the same point count as the surface path")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "index",
        "pass_index",
        "z_mm",
        "theta_rad",
        "surface_x_mm",
        "surface_y_mm",
        "surface_z_mm",
        "A_deg",
        "X_mm",
        "Z_mm",
        "B_deg",
    ]
    if feed_schedule is not None:
        fieldnames.extend(
            [
                "feedrate_mm_min",
                "curvature_1_per_mm",
                "curvature_radius_mm",
                "slip_risk",
            ]
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        pass_index = (
            np.zeros(surface_path.point_count, dtype=int)
            if surface_path.pass_index is None
            else surface_path.pass_index
        )
        for index in range(surface_path.point_count):
            row = {
                "index": index,
                "pass_index": int(pass_index[index]),
                "z_mm": float(surface_path.z_mm[index]),
                "theta_rad": float(surface_path.theta_rad[index]),
                "surface_x_mm": float(surface_path.x_mm[index]),
                "surface_y_mm": float(surface_path.y_mm[index]),
                "surface_z_mm": float(surface_path.z_mm[index]),
                "A_deg": float(motion_table.a_deg[index]),
                "X_mm": float(motion_table.x_mm[index]),
                "Z_mm": float(motion_table.z_mm[index]),
                "B_deg": float(motion_table.b_deg[index]),
            }
            if feed_schedule is not None:
                row.update(
                    {
                        "feedrate_mm_min": float(feed_schedule.feedrate_mm_min[index]),
                        "curvature_1_per_mm": float(
                            feed_schedule.curvature_1_per_mm[index]
                        ),
                        "curvature_radius_mm": float(
                            _csv_finite(feed_schedule.curvature_radius_mm[index])
                        ),
                        "slip_risk": float(feed_schedule.slip_risk[index]),
                    }
                )
            writer.writerow(row)
    return path


def export_winding_program_csv(
    program: PlannedWindingProgram,
    output_path: str | Path,
) -> Path:
    """Export a planned multi-layer program with planner metadata."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "index",
        "layer_id",
        "layer_index",
        "layer_name",
        "layer_type",
        "motion_type",
        "segment_id",
        "segment_type",
        "tow_state",
        "process_state",
        "circuit_index",
        "pass_id",
        "pass_index",
        "point_role",
        "winding_type",
        "time_s",
        "z_mm",
        "r_mm",
        "theta_mod_rad",
        "theta_unwrapped_rad",
        "theta_rad",
        "theta_deg_plot",
        "theta_deg",
        "x_surface_mm",
        "y_surface_mm",
        "z_surface_mm",
        "surface_x_mm",
        "surface_y_mm",
        "surface_z_mm",
        "local_radius_mm",
        "local_angle_deg",
        "local_winding_angle_deg",
        "a_deg",
        "A_deg",
        "x_mm",
        "X_mm",
        "b_deg",
        "Z_mm",
        "B_deg",
        "feedrate_mm_min",
        "surface_speed_mm_min",
        "A_velocity_deg_s",
        "X_velocity_mm_s",
        "Z_velocity_mm_s",
        "B_velocity_deg_s",
        "slip_risk_deg",
        "coverage_count",
        "warning_flags",
        "curvature_1_per_mm",
        "curvature_radius_mm",
        "slip_risk",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        time_s = _program_time_s(program)
        surface_speed = _surface_speed_mm_min(program, time_s)
        axis_velocity = _axis_velocity(program, time_s)
        point_segments = segment_labels_for_points(
            program.point_count,
            build_path_segments(program),
        )
        theta_unwrapped_rad = np.unwrap(program.path.theta_rad)
        theta_mod_rad = np.mod(program.path.theta_rad, 2.0 * np.pi)
        for index in range(program.point_count):
            segment_id, segment_type, point_role, tow_state, process_state = point_segments[index]
            writer.writerow(
                {
                    "index": index,
                    "layer_id": program.metadata.layer_id[index],
                    "layer_index": int(program.metadata.layer_index[index]),
                    "layer_name": program.metadata.layer_name[index],
                    "layer_type": program.metadata.winding_type[index],
                    "motion_type": program.metadata.motion_type[index],
                    "segment_id": segment_id,
                    "segment_type": segment_type,
                    "tow_state": tow_state,
                    "process_state": process_state,
                    "circuit_index": int(program.metadata.circuit_index[index]),
                    "pass_id": int(program.metadata.pass_index[index]),
                    "pass_index": int(program.metadata.pass_index[index]),
                    "point_role": point_role,
                    "winding_type": program.metadata.winding_type[index],
                    "time_s": float(time_s[index]),
                    "z_mm": float(program.path.z_mm[index]),
                    "r_mm": float(program.metadata.local_radius_mm[index]),
                    "theta_mod_rad": float(theta_mod_rad[index]),
                    "theta_unwrapped_rad": float(theta_unwrapped_rad[index]),
                    "theta_rad": float(program.path.theta_rad[index]),
                    "theta_deg_plot": float(np.rad2deg(theta_mod_rad[index])),
                    "theta_deg": float(program.path.theta_deg[index]),
                    "x_surface_mm": float(program.path.x_mm[index]),
                    "y_surface_mm": float(program.path.y_mm[index]),
                    "z_surface_mm": float(program.path.z_mm[index]),
                    "surface_x_mm": float(program.path.x_mm[index]),
                    "surface_y_mm": float(program.path.y_mm[index]),
                    "surface_z_mm": float(program.path.z_mm[index]),
                    "local_radius_mm": float(program.metadata.local_radius_mm[index]),
                    "local_angle_deg": float(program.metadata.local_winding_angle_deg[index]),
                    "local_winding_angle_deg": float(
                        program.metadata.local_winding_angle_deg[index]
                    ),
                    "a_deg": float(program.motion_table.a_deg[index]),
                    "A_deg": float(program.motion_table.a_deg[index]),
                    "x_mm": float(program.motion_table.x_mm[index]),
                    "X_mm": float(program.motion_table.x_mm[index]),
                    "b_deg": float(program.motion_table.b_deg[index]),
                    "Z_mm": float(program.motion_table.z_mm[index]),
                    "B_deg": float(program.motion_table.b_deg[index]),
                    "feedrate_mm_min": float(program.feed_schedule.feedrate_mm_min[index]),
                    "surface_speed_mm_min": float(surface_speed[index]),
                    "A_velocity_deg_s": float(axis_velocity[0][index]),
                    "X_velocity_mm_s": float(axis_velocity[1][index]),
                    "Z_velocity_mm_s": float(axis_velocity[2][index]),
                    "B_velocity_deg_s": float(axis_velocity[3][index]),
                    "slip_risk_deg": _warning_slip_risk_deg(
                        program.metadata.warning_flags[index]
                    ),
                    "coverage_count": 0,
                    "warning_flags": program.metadata.warning_flags[index],
                    "curvature_1_per_mm": float(
                        program.feed_schedule.curvature_1_per_mm[index]
                    ),
                    "curvature_radius_mm": float(
                        _csv_finite(program.feed_schedule.curvature_radius_mm[index])
                    ),
                    "slip_risk": float(program.feed_schedule.slip_risk[index]),
                }
            )
    return path


def _program_time_s(program: PlannedWindingProgram) -> np.ndarray:
    time_s = np.zeros(program.point_count, dtype=float)
    if program.point_count < 2:
        return time_s
    segment_lengths = np.linalg.norm(np.diff(program.path.points_mm, axis=0), axis=1)
    feedrate = np.maximum(program.feed_schedule.feedrate_mm_min[:-1], 1e-9)
    time_s[1:] = np.cumsum(segment_lengths / feedrate * 60.0)
    return time_s


def _surface_speed_mm_min(program: PlannedWindingProgram, time_s: np.ndarray) -> np.ndarray:
    if program.point_count < 2:
        return np.zeros(program.point_count, dtype=float)
    speed = np.zeros(program.point_count, dtype=float)
    segment_lengths = np.linalg.norm(np.diff(program.path.points_mm, axis=0), axis=1)
    dt_s = np.diff(time_s)
    segment_speed = np.divide(
        segment_lengths,
        dt_s,
        out=np.zeros_like(segment_lengths),
        where=dt_s > 1e-12,
    ) * 60.0
    speed[1:] = segment_speed
    speed[0] = speed[1]
    return speed


def _axis_velocity(
    program: PlannedWindingProgram,
    time_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        _velocity(program.motion_table.a_deg, time_s),
        _velocity(program.motion_table.x_mm, time_s),
        _velocity(program.motion_table.z_mm, time_s),
        _velocity(program.motion_table.b_deg, time_s),
    )


def _velocity(values: np.ndarray, time_s: np.ndarray) -> np.ndarray:
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


def _csv_finite(value: float) -> float:
    return float(value) if np.isfinite(value) else 1.0e12


def _warning_slip_risk_deg(warning: str) -> float:
    marker = "slip risk "
    if marker not in warning:
        return 0.0
    text = warning.split(marker, 1)[1].split(" deg", 1)[0]
    try:
        return float(text)
    except ValueError:
        return 0.0
