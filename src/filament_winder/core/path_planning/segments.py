"""Segment/state view over planned winding programs."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from filament_winder.core.path_planning.schedule import PlannedWindingProgram


@dataclass(frozen=True, slots=True)
class PathState:
    index: int
    z_mm: float
    theta_rad: float
    r_mm: float
    local_angle_deg: float
    surface_position: tuple[float, float, float]
    A_deg: float
    X_mm: float
    Z_mm: float
    B_deg: float
    feedrate_mm_min: float
    time_s: float
    warning_flags: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PathSegment:
    segment_id: str
    segment_type: str
    layer_id: str
    layer_name: str
    layer_type: str
    pass_id: str
    start_index: int
    end_index: int
    point_count: int
    start_state: PathState
    end_state: PathState
    tow_state: str
    process_state: str
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["start_state"] = self.start_state.to_dict()
        data["end_state"] = self.end_state.to_dict()
        return data


def build_path_segments(program: PlannedWindingProgram) -> tuple[PathSegment, ...]:
    """Build contiguous segment descriptors from planner metadata."""

    if program.point_count < 1:
        return ()
    time_s = _program_time_s(program)
    segments = []
    start = 0
    segment_number = 1
    for index in range(1, program.point_count):
        if _segment_break(program, index):
            segments.append(_make_segment(program, time_s, start, index - 1, segment_number))
            segment_number += 1
            start = index
    segments.append(_make_segment(program, time_s, start, program.point_count - 1, segment_number))
    return tuple(segments)


def segment_labels_for_points(
    point_count: int,
    segments: tuple[PathSegment, ...],
) -> tuple[tuple[str, str, str, str, str], ...]:
    labels = [("", "", "mid", "on", "winding") for _ in range(point_count)]
    for segment in segments:
        for index in range(segment.start_index, segment.end_index + 1):
            role = "mid"
            if index == segment.start_index:
                role = "start"
            elif index == segment.end_index:
                role = "end"
            labels[index] = (
                segment.segment_id,
                segment.segment_type,
                role,
                segment.tow_state,
                segment.process_state,
            )
    return tuple(labels)


def _segment_break(program: PlannedWindingProgram, index: int) -> bool:
    return (
        program.metadata.layer_id[index] != program.metadata.layer_id[index - 1]
        or program.metadata.pass_index[index] != program.metadata.pass_index[index - 1]
        or program.metadata.motion_type[index] != program.metadata.motion_type[index - 1]
    )


def _make_segment(
    program: PlannedWindingProgram,
    time_s: np.ndarray,
    start_index: int,
    end_index: int,
    segment_number: int,
) -> PathSegment:
    motion_type = program.metadata.motion_type[start_index]
    layer_type = program.metadata.winding_type[start_index]
    segment_type = _segment_type(
        motion_type,
        layer_type,
        program.path.z_mm[start_index:end_index + 1],
        program.path.theta_rad[start_index:end_index + 1],
    )
    layer_id = program.metadata.layer_id[start_index]
    pass_id = str(int(program.metadata.pass_index[start_index]))
    return PathSegment(
        segment_id=f"seg-{segment_number:05d}",
        segment_type=segment_type,
        layer_id=layer_id,
        layer_name=program.metadata.layer_name[start_index],
        layer_type=layer_type,
        pass_id=pass_id,
        start_index=start_index,
        end_index=end_index,
        point_count=end_index - start_index + 1,
        start_state=_state_at(program, time_s, start_index),
        end_state=_state_at(program, time_s, end_index),
        tow_state="on",
        process_state="transition" if motion_type == "transition" else "winding",
        warnings=_segment_warnings(program, start_index, end_index),
    )


def _state_at(program: PlannedWindingProgram, time_s: np.ndarray, index: int) -> PathState:
    return PathState(
        index=index,
        z_mm=float(program.path.z_mm[index]),
        theta_rad=float(program.path.theta_rad[index]),
        r_mm=float(program.metadata.local_radius_mm[index]),
        local_angle_deg=float(program.metadata.local_winding_angle_deg[index]),
        surface_position=(
            float(program.path.x_mm[index]),
            float(program.path.y_mm[index]),
            float(program.path.z_mm[index]),
        ),
        A_deg=float(program.motion_table.a_deg[index]),
        X_mm=float(program.motion_table.x_mm[index]),
        Z_mm=float(program.motion_table.z_mm[index]),
        B_deg=float(program.motion_table.b_deg[index]),
        feedrate_mm_min=float(program.feed_schedule.feedrate_mm_min[index]),
        time_s=float(time_s[index]),
        warning_flags=program.metadata.warning_flags[index],
    )


def _segment_type(
    motion_type: str,
    layer_type: str,
    z_mm: np.ndarray,
    theta_rad: np.ndarray,
) -> str:
    if motion_type == "transition":
        if layer_type in {"geodesic", "non_geodesic"}:
            return "dome_turnaround"
        z_span = float(np.max(z_mm) - np.min(z_mm)) if z_mm.size else 0.0
        theta_span = float(np.max(theta_rad) - np.min(theta_rad)) if theta_rad.size else 0.0
        if z_span > 1e-6 and abs(theta_span) <= 1e-6:
            return "axial_reposition"
        if z_span <= 1e-6 and abs(theta_span) > 1e-6:
            return "phase_reposition"
        return "layer_transition"
    if layer_type == "hoop":
        return "hoop_pass"
    if layer_type == "geodesic":
        return "geodesic_pass"
    if layer_type == "non_geodesic":
        return "non_geodesic_pass"
    if layer_type in {"dome", "polar", "nosecone", "axisymmetric"}:
        return "dome_turnaround"
    return "helical_pass"


def _segment_warnings(
    program: PlannedWindingProgram,
    start_index: int,
    end_index: int,
) -> tuple[str, ...]:
    warnings = {
        warning
        for warning in program.metadata.warning_flags[start_index:end_index + 1]
        if warning
    }
    return tuple(sorted(warnings))


def _program_time_s(program: PlannedWindingProgram) -> np.ndarray:
    time_s = np.zeros(program.point_count, dtype=float)
    if program.point_count < 2:
        return time_s
    segment_lengths = np.linalg.norm(np.diff(program.path.points_mm, axis=0), axis=1)
    feedrate = np.maximum(program.feed_schedule.feedrate_mm_min[:-1], 1e-9)
    time_s[1:] = np.cumsum(segment_lengths / feedrate * 60.0)
    return time_s
