"""ASCII DXF Z-R profile importer."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from filament_winder.core.geometry import AxisymmetricProfileMandrel

FloatArray = NDArray[np.float64]
Tag = tuple[str, str]
POINT_TOLERANCE = 1e-9


@dataclass(frozen=True, slots=True)
class _DxfPointEntity:
    entity_type: str
    layer: str
    points: FloatArray


def import_dxf_zr_profile(
    input_path: str | Path,
    *,
    samples: int | None = None,
    name: str | None = None,
) -> AxisymmetricProfileMandrel:
    """Import a Z-R profile from common ASCII DXF line and polyline entities.

    DXF X coordinates are interpreted as longitudinal Z values in mm. DXF Y
    coordinates are interpreted as radial R values in mm.
    """

    path = Path(input_path)
    text = path.read_text(encoding="utf-8", errors="ignore")
    points = _select_profile_points(_parse_ascii_dxf_entities(text))
    z_mm, r_mm = _clean_profile_points(points, samples=samples)
    return AxisymmetricProfileMandrel(
        z_mm=z_mm,
        r_mm=r_mm,
        name=path.stem if name is None else name,
    )


def _parse_ascii_dxf_entities(text: str) -> list[_DxfPointEntity]:
    tags = _read_tags(text)
    entities: list[_DxfPointEntity] = []
    for entity_type, body in _iter_entities(tags):
        layer = _entity_layer(body)
        points: list[tuple[float, float]] = []
        if entity_type == "LINE":
            points.extend(_line_points(body))
        elif entity_type == "LWPOLYLINE":
            points.extend(_xy_sequence_points(body))
        elif entity_type == "VERTEX":
            points.extend(_vertex_points(body))
        elif entity_type == "ARC":
            points.extend(_arc_points(body))
        elif entity_type == "CIRCLE":
            points.extend(_circle_points(body))
        elif entity_type == "ELLIPSE":
            points.extend(_ellipse_points(body))
        elif entity_type == "SPLINE":
            points.extend(_xy_sequence_points(body))
        if points:
            entities.append(
                _DxfPointEntity(
                    entity_type=entity_type,
                    layer=layer,
                    points=np.asarray(points, dtype=float),
                )
            )
    if not entities:
        raise ValueError("DXF profile did not contain LINE, LWPOLYLINE, or POLYLINE vertex points")
    return entities


def _select_profile_points(entities: list[_DxfPointEntity]) -> FloatArray:
    candidates: list[tuple[tuple[int, float, float, int], FloatArray]] = []
    for entity in entities:
        points = entity.points
        if points.ndim != 2 or points.shape[1] != 2:
            continue
        finite_points = points[np.all(np.isfinite(points), axis=1)]
        if finite_points.size == 0:
            continue
        upper_points = finite_points[finite_points[:, 1] >= -POINT_TOLERANCE].copy()
        if upper_points.size == 0:
            continue
        upper_points[:, 1] = np.maximum(upper_points[:, 1], 0.0)
        if np.max(upper_points[:, 1]) <= POINT_TOLERANCE:
            continue
        score = _profile_candidate_score(entity, upper_points)
        candidates.append((score, upper_points))
    if not candidates:
        raise ValueError("DXF profile did not contain a usable non-negative Z-R profile")
    _score, points = max(candidates, key=lambda candidate: candidate[0])
    return points


def _profile_candidate_score(
    entity: _DxfPointEntity,
    points: FloatArray,
) -> tuple[int, float, float, int]:
    layer = entity.layer.upper()
    if "CENTER" in layer or "CONSTRUCTION" in layer:
        layer_rank = -10
    elif "UPPER" in layer or "OUTER" in layer:
        layer_rank = 30
    elif "PROFILE" in layer and "LOWER" not in layer:
        layer_rank = 20
    elif "LOWER" in layer:
        layer_rank = -5
    else:
        layer_rank = 0
    z_span = float(np.max(points[:, 0]) - np.min(points[:, 0]))
    max_radius = float(np.max(points[:, 1]))
    return layer_rank, z_span, max_radius, int(points.shape[0])


def _read_tags(text: str) -> list[Tag]:
    lines = [line.strip() for line in text.splitlines()]
    tags: list[Tag] = []
    index = 0
    while index + 1 < len(lines):
        tags.append((lines[index], lines[index + 1]))
        index += 2
    return tags


def _iter_entities(tags: Iterable[Tag]) -> Iterable[tuple[str, list[Tag]]]:
    tag_list = list(tags)
    index = 0
    while index < len(tag_list):
        code, value = tag_list[index]
        if code != "0":
            index += 1
            continue
        entity_type = value.upper()
        body: list[Tag] = []
        index += 1
        while index < len(tag_list) and tag_list[index][0] != "0":
            body.append(tag_list[index])
            index += 1
        yield entity_type, body


def _line_points(tags: list[Tag]) -> list[tuple[float, float]]:
    values = _tag_values(tags)
    points: list[tuple[float, float]] = []
    if "10" in values and "20" in values:
        points.append((float(values["10"][0]), float(values["20"][0])))
    if "11" in values and "21" in values:
        points.append((float(values["11"][0]), float(values["21"][0])))
    return points


def _arc_points(tags: list[Tag], *, samples: int = 64) -> list[tuple[float, float]]:
    values = _tag_values(tags)
    required_codes = {"10", "20", "40", "50", "51"}
    if not required_codes.issubset(values):
        return []
    center_x = float(values["10"][0])
    center_y = float(values["20"][0])
    radius = float(values["40"][0])
    start_deg = float(values["50"][0])
    end_deg = float(values["51"][0])
    if radius < 0.0:
        return []
    if end_deg < start_deg:
        end_deg += 360.0
    theta = np.deg2rad(np.linspace(start_deg, end_deg, samples))
    return [
        (float(center_x + radius * np.cos(angle)), float(center_y + radius * np.sin(angle)))
        for angle in theta
    ]


def _circle_points(tags: list[Tag], *, samples: int = 128) -> list[tuple[float, float]]:
    values = _tag_values(tags)
    if not {"10", "20", "40"}.issubset(values):
        return []
    center_x = float(values["10"][0])
    center_y = float(values["20"][0])
    radius = float(values["40"][0])
    if radius < 0.0:
        return []
    theta = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
    return [
        (float(center_x + radius * np.cos(angle)), float(center_y + radius * np.sin(angle)))
        for angle in theta
    ]


def _ellipse_points(tags: list[Tag], *, samples: int = 128) -> list[tuple[float, float]]:
    values = _tag_values(tags)
    if not {"10", "20", "11", "21", "40"}.issubset(values):
        return []
    center = np.asarray([float(values["10"][0]), float(values["20"][0])], dtype=float)
    major_axis = np.asarray([float(values["11"][0]), float(values["21"][0])], dtype=float)
    ratio = float(values["40"][0])
    start_param = float(values.get("41", ["0.0"])[0])
    end_param = float(values.get("42", [str(2.0 * np.pi)])[0])
    if ratio < 0.0:
        return []
    if end_param < start_param:
        end_param += 2.0 * np.pi
    minor_axis = np.asarray([-major_axis[1], major_axis[0]], dtype=float) * ratio
    params = np.linspace(start_param, end_param, samples)
    points = (
        center
        + np.cos(params)[:, np.newaxis] * major_axis
        + np.sin(params)[:, np.newaxis] * minor_axis
    )
    return [(float(point[0]), float(point[1])) for point in points]


def _xy_sequence_points(tags: list[Tag]) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    pending_x: float | None = None
    for code, value in tags:
        if code == "10":
            pending_x = float(value)
        elif code == "20" and pending_x is not None:
            points.append((pending_x, float(value)))
            pending_x = None
    return points


def _vertex_points(tags: list[Tag]) -> list[tuple[float, float]]:
    values = _tag_values(tags)
    if "10" in values and "20" in values:
        return [(float(values["10"][0]), float(values["20"][0]))]
    return []


def _entity_layer(tags: list[Tag]) -> str:
    values = _tag_values(tags)
    return values.get("8", [""])[0]


def _tag_values(tags: list[Tag]) -> dict[str, list[str]]:
    values: dict[str, list[str]] = {}
    for code, value in tags:
        values.setdefault(code, []).append(value)
    return values


def _clean_profile_points(
    points: FloatArray,
    *,
    samples: int | None,
) -> tuple[FloatArray, FloatArray]:
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("profile points must have shape (n, 2)")
    if not np.all(np.isfinite(points)):
        raise ValueError("profile points must be finite")
    if np.any(points[:, 1] < 0.0):
        raise ValueError("profile radius values must be non-negative")
    order = np.argsort(points[:, 0])
    sorted_points = points[order]
    unique_z = np.unique(sorted_points[:, 0])
    if unique_z.size < 2:
        raise ValueError("profile needs at least two unique Z values")

    merged_r = np.empty(unique_z.size, dtype=float)
    for index, z_value in enumerate(unique_z):
        merged_r[index] = float(np.mean(sorted_points[sorted_points[:, 0] == z_value, 1]))

    if samples is not None:
        if samples < 2:
            raise ValueError("samples must be at least 2")
        resampled_z = np.linspace(float(unique_z[0]), float(unique_z[-1]), samples)
        resampled_r = np.interp(resampled_z, unique_z, merged_r)
        return resampled_z.astype(float, copy=False), resampled_r.astype(float, copy=False)
    return unique_z.astype(float, copy=False), merged_r.astype(float, copy=False)
