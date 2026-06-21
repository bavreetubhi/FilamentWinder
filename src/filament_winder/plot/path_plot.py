"""Basic headless winding path plots."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path
from typing import Any, cast

import numpy as np

from filament_winder.config import WindingJobConfig
from filament_winder.core.geometry import AxisymmetricProfileMandrel, CylinderMandrel
from filament_winder.core.path_planning import PlannedWindingProgram

Rgb = tuple[int, int, int]


def plot_winding_program(
    config: WindingJobConfig,
    mandrel: CylinderMandrel | AxisymmetricProfileMandrel,
    program: PlannedWindingProgram,
    output_dir: Path,
) -> tuple[Path, ...]:
    if not config.plot.enabled:
        return ()
    if "png" not in {item.lower() for item in config.plot.formats}:
        return ()
    output_dir.mkdir(parents=True, exist_ok=True)
    color_map = {
        f"{index + 1:02d}-{_safe_id(layer.name)}": _parse_color(layer.colour)
        for index, layer in enumerate(config.layers)
    }
    paths = []
    modes = {mode.lower() for mode in config.plot.modes}
    if config.plot.save and "unwrapped" in modes:
        path = output_dir / "path_unwrapped.png"
        _write_unwrapped_png(path, mandrel, program, color_map)
        paths.append(path)
    if config.plot.save and "three_d" in modes:
        path = output_dir / "path_3d.png"
        _write_projected_3d_png(path, mandrel, program, color_map)
        paths.append(path)
    if config.plot.save and "debug_passes" in modes:
        path = output_dir / "path_debug_passes.png"
        _write_unwrapped_png(path, mandrel, program, color_map, debug_passes=True)
        paths.append(path)
    if config.plot.save and "debug_transitions" in modes:
        path = output_dir / "path_debug_transitions.png"
        _write_unwrapped_png(path, mandrel, program, color_map, debug_transitions=True)
        paths.append(path)
    return tuple(paths)


def plot_layer_diagnostics(
    config: WindingJobConfig,
    mandrel: CylinderMandrel | AxisymmetricProfileMandrel,
    program: PlannedWindingProgram,
    coverage: object,
    output_dir: Path,
) -> tuple[tuple[Path, ...], dict[str, object]]:
    if not config.plot.save:
        return (), {"plots": []}
    output_dir.mkdir(parents=True, exist_ok=True)
    color_map = {
        f"{index + 1:02d}-{_safe_id(layer.name)}": _parse_color(layer.colour)
        for index, layer in enumerate(config.layers)
    }
    paths: list[Path] = []
    manifest: list[dict[str, str]] = []

    combined_unwrapped = output_dir / "combined_unwrapped.png"
    _write_unwrapped_png(combined_unwrapped, mandrel, program, color_map, debug_transitions=True)
    paths.append(combined_unwrapped)
    manifest.append({"type": "combined_unwrapped", "path": str(combined_unwrapped)})

    combined_3d = output_dir / "combined_3d.png"
    _write_projected_3d_png(combined_3d, mandrel, program, color_map)
    paths.append(combined_3d)
    manifest.append({"type": "combined_3d", "path": str(combined_3d)})

    for index, layer in enumerate(program.layers, start=1):
        label = f"layer_{index:02d}_{_safe_id(layer.spec.winding_type)}"
        layer_unwrapped = output_dir / f"{label}_unwrapped.png"
        _write_unwrapped_png(
            layer_unwrapped,
            mandrel,
            program,
            color_map,
            debug_transitions=True,
            layer_filter=layer.spec.layer_id,
        )
        paths.append(layer_unwrapped)
        manifest.append(
            {
                "type": "layer_unwrapped",
                "layer_id": layer.spec.layer_id,
                "path": str(layer_unwrapped),
            }
        )
        layer_3d = output_dir / f"{label}_3d.png"
        _write_projected_3d_png(
            layer_3d,
            mandrel,
            program,
            color_map,
            layer_filter=layer.spec.layer_id,
        )
        paths.append(layer_3d)
        manifest.append(
            {
                "type": "layer_3d",
                "layer_id": layer.spec.layer_id,
                "path": str(layer_3d),
            }
        )

    coverage_map = cast(Any, coverage)
    coverage_count = np.asarray(coverage_map.coverage_count, dtype=np.int_)
    heatmap_specs = (
        ("coverage_heatmap", coverage_count),
        ("gap_map", (coverage_count == 0).astype(int)),
        ("overlap_map", np.maximum(coverage_count - 1, 0)),
        ("thickness_distribution", coverage_count),
        ("stack_thickness_map", coverage_count),
        ("stack_overlap_map", np.maximum(coverage_count - 1, 0)),
        ("region_quality_map", _region_quality_values(mandrel, coverage_map)),
        ("strict_quality_summary", _strict_quality_values(coverage_count)),
    )
    for plot_type, values in heatmap_specs:
        path = output_dir / f"{plot_type}.png"
        _write_heatmap_png(path, values)
        paths.append(path)
        manifest.append({"type": plot_type, "path": str(path)})

    hoop_layers = [layer for layer in program.layers if layer.spec.winding_type == "hoop"]
    if hoop_layers:
        hoop_layer = hoop_layers[0]
        a_deg = np.rad2deg(hoop_layer.path.theta_rad)
        z_mm = hoop_layer.path.z_mm
        path = output_dir / "hoop_z_vs_a.png"
        _write_xy_line_png(path, a_deg, z_mm, (42, 116, 214))
        paths.append(path)
        manifest.append(
            {"type": "hoop_z_vs_a", "layer_id": hoop_layer.spec.layer_id, "path": str(path)}
        )

        pitch_values = _hoop_pitch_values(hoop_layer)
        path = output_dir / "hoop_pitch_debug.png"
        _write_xy_line_png(
            path,
            np.arange(pitch_values.size, dtype=float),
            pitch_values,
            (210, 115, 25),
        )
        paths.append(path)
        manifest.append(
            {
                "type": "hoop_pitch_debug",
                "layer_id": hoop_layer.spec.layer_id,
                "path": str(path),
            }
        )

    return tuple(paths), {"plots": manifest}


def plot_dome_coverage_maps(
    dome_coverage_report: dict[str, Any],
    output_dir: Path,
) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for side in ("left", "right"):
        cells = [
            cell
            for cell in dome_coverage_report.get("cells", [])
            if cell.get("side") == side
        ]
        if not cells:
            continue
        gap_values = _dome_cell_grid(cells, "gap_mm")
        overlap_values = _dome_cell_grid(cells, "overlap_mm")
        gap_path = output_dir / f"{side}_dome_gap_map.png"
        overlap_path = output_dir / f"{side}_dome_overlap_map.png"
        _write_heatmap_png(gap_path, gap_values)
        _write_heatmap_png(overlap_path, overlap_values)
        paths.extend([gap_path, overlap_path])
    return tuple(paths)


def plot_dome_motion_diagnostics(
    config: WindingJobConfig,
    mandrel: CylinderMandrel | AxisymmetricProfileMandrel,
    program: PlannedWindingProgram,
    output_dir: Path,
) -> tuple[Path, ...]:
    if not config.plot.save:
        return ()
    output_dir.mkdir(parents=True, exist_ok=True)
    color_map = {
        f"{index + 1:02d}-{_safe_id(layer.name)}": _parse_color(layer.colour)
        for index, layer in enumerate(config.layers)
    }
    plots: list[Path] = []

    shell_types = {"geodesic_pass", "non_geodesic_pass", "wind"}
    boss_types = {"BossTurnaroundArc"}
    transition_types = {"transition", "PinTransition", "FreeSpan", "PinContactArc"}

    shell_path = output_dir / "dome_shell_only_unwrapped.png"
    _write_unwrapped_png(
        shell_path,
        mandrel,
        program,
        color_map,
        motion_types_filter=shell_types,
    )
    plots.append(shell_path)

    boss_path = output_dir / "dome_boss_contact_unwrapped.png"
    _write_unwrapped_png(
        boss_path,
        mandrel,
        program,
        color_map,
        motion_types_filter=boss_types,
    )
    plots.append(boss_path)

    transition_path = output_dir / "dome_transition_moves_unwrapped.png"
    _write_unwrapped_png(
        transition_path,
        mandrel,
        program,
        color_map,
        motion_types_filter=transition_types,
    )
    plots.append(transition_path)

    boss_closeup_left = output_dir / "dome_boss_closeup_left.png"
    boss_closeup_right = output_dir / "dome_boss_closeup_right.png"
    _write_boss_closeup_png(
        boss_closeup_left,
        mandrel,
        program,
        color_map,
        side="left",
        motion_types_filter=shell_types | boss_types,
    )
    _write_boss_closeup_png(
        boss_closeup_right,
        mandrel,
        program,
        color_map,
        side="right",
        motion_types_filter=shell_types | boss_types,
    )
    plots.extend([boss_closeup_left, boss_closeup_right])
    return tuple(plots)


def _dome_cell_grid(cells: list[dict[str, Any]], field: str) -> np.ndarray:
    meridian_values = sorted({float(cell["meridian_fraction"]) for cell in cells})
    theta_values = sorted({float(cell["theta_deg"]) for cell in cells})
    meridian_index = {value: index for index, value in enumerate(meridian_values)}
    theta_index = {value: index for index, value in enumerate(theta_values)}
    grid = np.zeros((len(meridian_values), len(theta_values)), dtype=float)
    for cell in cells:
        grid[
            meridian_index[float(cell["meridian_fraction"])],
            theta_index[float(cell["theta_deg"])],
        ] = float(cell.get(field, 0.0))
    return grid


def _write_boss_closeup_png(
    path: Path,
    mandrel: CylinderMandrel | AxisymmetricProfileMandrel,
    program: PlannedWindingProgram,
    color_map: dict[str, Rgb],
    *,
    side: str,
    motion_types_filter: set[str],
) -> None:
    width, height = 1200, 720
    margin_left, margin_right = 70, 28
    margin_top, margin_bottom = 34, 58
    image = _new_image(width, height, (250, 252, 255))
    z = program.path.z_mm
    radius = program.path.surface_radius_mm
    motion_types = np.asarray(program.metadata.motion_type)
    z_min = _start_z(mandrel)
    z_max = _end_z(mandrel)
    z_mid = (z_min + z_max) / 2.0
    if side == "left":
        window = z <= z_mid
        x_min = z_min
        x_max = z_min + max((z_max - z_min) * 0.35, 1e-9)
    else:
        window = z >= z_mid
        x_min = z_max - max((z_max - z_min) * 0.35, 1e-9)
        x_max = z_max
    if not np.any(window):
        _write_png(path, image)
        return
    z_sel = z[window]
    r_sel = radius[window]
    motion_sel = motion_types[window]
    layer_ids = tuple(program.metadata.layer_id[i] for i, keep in enumerate(window) if keep)
    x_pixels = margin_left + ((z_sel - x_min) / max(x_max - x_min, 1e-9)) * (
        width - margin_left - margin_right
    )
    y_pixels = (
        height
        - margin_bottom
        - (r_sel / max(float(np.max(radius)), 1e-9)) * (height - margin_top - margin_bottom)
    )
    _draw_axes(image, margin_left, margin_top, width - margin_right, height - margin_bottom)
    _draw_grid(image, margin_left, margin_top, width - margin_right, height - margin_bottom)
    _draw_layered_polyline(
        image,
        x_pixels,
        y_pixels,
        layer_ids,
        color_map,
        motion_types=tuple(motion_sel),
        motion_types_filter=motion_types_filter,
        layer_filter=None,
    )
    _write_png(path, image)


def _write_unwrapped_png(
    path: Path,
    mandrel: CylinderMandrel | AxisymmetricProfileMandrel,
    program: PlannedWindingProgram,
    color_map: dict[str, Rgb],
    *,
    debug_passes: bool = False,
    debug_transitions: bool = False,
    layer_filter: str | None = None,
    motion_types_filter: set[str] | None = None,
) -> None:
    width, height = 1200, 720
    margin_left, margin_right = 70, 28
    margin_top, margin_bottom = 34, 58
    image = _new_image(width, height, (250, 252, 255))
    _draw_axes(image, margin_left, margin_top, width - margin_right, height - margin_bottom)

    z = program.path.z_mm
    theta_wrapped = np.mod(program.path.theta_deg, 360.0)
    z_min = _start_z(mandrel)
    z_span = max(_length(mandrel), 1e-9)
    x_pixels = margin_left + ((z - z_min) / z_span) * (width - margin_left - margin_right)
    y_pixels = (
        height
        - margin_bottom
        - (theta_wrapped / 360.0) * (height - margin_top - margin_bottom)
    )
    _draw_grid(image, margin_left, margin_top, width - margin_right, height - margin_bottom)
    _draw_layered_polyline(
        image,
        x_pixels,
        y_pixels,
        tuple(program.metadata.layer_id),
        color_map,
        motion_types=program.metadata.motion_type,
        debug_transitions=debug_transitions,
        layer_filter=layer_filter,
        motion_types_filter=motion_types_filter,
    )
    if debug_passes:
        _draw_pass_markers(image, x_pixels, y_pixels, tuple(program.metadata.pass_index))
    if debug_transitions:
        _draw_transition_markers(image, x_pixels, y_pixels, program.metadata.motion_type)
    _write_png(path, image)


def _write_projected_3d_png(
    path: Path,
    mandrel: CylinderMandrel | AxisymmetricProfileMandrel,
    program: PlannedWindingProgram,
    color_map: dict[str, Rgb],
    layer_filter: str | None = None,
) -> None:
    width, height = 1200, 720
    image = _new_image(width, height, (250, 252, 255))
    points = program.path.points_mm
    x3 = points[:, 2]
    y3 = points[:, 0] * 0.82 + points[:, 1] * 0.38
    x_min, x_max = _start_z(mandrel), _end_z(mandrel)
    radius = _radius(mandrel)
    y_extent = max(radius * 1.35, float(np.max(np.abs(y3))) * 1.05)
    margin_left, margin_right = 70, 28
    margin_top, margin_bottom = 34, 58
    x_pixels = margin_left + (x3 - x_min) / (x_max - x_min) * (
        width - margin_left - margin_right
    )
    y_pixels = (
        height
        - margin_bottom
        - ((y3 + y_extent) / (2.0 * y_extent)) * (height - margin_top - margin_bottom)
    )
    _draw_axes(image, margin_left, margin_top, width - margin_right, height - margin_bottom)
    _draw_cylinder_outline(
        image,
        margin_left,
        margin_top,
        width - margin_right,
        height - margin_bottom,
        y_extent,
        radius,
    )
    _draw_layered_polyline(
        image,
        x_pixels,
        y_pixels,
        tuple(program.metadata.layer_id),
        color_map,
        layer_filter=layer_filter,
    )
    _write_png(path, image)


def _write_heatmap_png(path: Path, values: np.ndarray) -> None:
    width, height = 1200, 720
    image = _new_image(width, height, (250, 252, 255))
    rows, cols = values.shape
    max_value = max(float(np.max(values)), 1.0)
    left, top, right, bottom = 70, 34, width - 28, height - 58
    _draw_axes(image, left, top, right, bottom)
    for row in range(rows):
        y0 = int(top + row / rows * (bottom - top))
        y1 = int(top + (row + 1) / rows * (bottom - top))
        for col in range(cols):
            x0 = int(left + col / cols * (right - left))
            x1 = int(left + (col + 1) / cols * (right - left))
            intensity = float(values[row, col]) / max_value
            color = _heatmap_color(intensity)
            for yy in range(y0, max(y0 + 1, y1)):
                for xx in range(x0, max(x0 + 1, x1)):
                    _set_pixel(image, width, height, xx, yy, color)
    _write_png(path, image)


def _write_xy_line_png(path: Path, x_values: np.ndarray, y_values: np.ndarray, color: Rgb) -> None:
    width, height = 1200, 720
    image = _new_image(width, height, (250, 252, 255))
    left, top, right, bottom = 70, 34, width - 28, height - 58
    _draw_axes(image, left, top, right, bottom)
    _draw_grid(image, left, top, right, bottom)
    if x_values.size < 2 or y_values.size < 2:
        _write_png(path, image)
        return
    x_min = float(np.min(x_values))
    x_span = max(float(np.max(x_values)) - x_min, 1e-9)
    y_min = float(np.min(y_values))
    y_span = max(float(np.max(y_values)) - y_min, 1e-9)
    x_pixels = left + (x_values - x_min) / x_span * (right - left)
    y_pixels = bottom - (y_values - y_min) / y_span * (bottom - top)
    for index in range(1, x_values.size):
        _draw_line(
            image,
            int(x_pixels[index - 1]),
            int(y_pixels[index - 1]),
            int(x_pixels[index]),
            int(y_pixels[index]),
            color,
        )
    _write_png(path, image)


def _hoop_pitch_values(layer: Any) -> np.ndarray:
    theta_turn = np.floor((layer.path.theta_rad - layer.path.theta_rad[0]) / (2.0 * np.pi))
    values = []
    for turn in np.unique(theta_turn.astype(int)):
        mask = theta_turn.astype(int) == turn
        if np.count_nonzero(mask) > 1:
            values.append(float(np.max(layer.path.z_mm[mask]) - np.min(layer.path.z_mm[mask])))
    return np.asarray(values or [0.0], dtype=float)


def _strict_quality_values(coverage_count: np.ndarray) -> np.ndarray:
    values = np.zeros((4, 4), dtype=float)
    values[0, 0] = float(np.mean(coverage_count > 0))
    values[1, 0] = float(np.mean(coverage_count == 0))
    values[2, 0] = float(np.mean(coverage_count > 1))
    values[3, 0] = float(np.max(coverage_count)) if coverage_count.size else 0.0
    return values


def _region_quality_values(
    mandrel: CylinderMandrel | AxisymmetricProfileMandrel,
    coverage_map: Any,
) -> np.ndarray:
    z_values = np.asarray(coverage_map.z_mm, dtype=float)
    radius = mandrel.radius_at(z_values)
    max_radius = max(float(np.max(radius)), 1e-9)
    values = np.zeros(np.asarray(coverage_map.coverage_count).shape, dtype=float)
    cylinder = radius >= max_radius * 0.98
    polar = radius <= max_radius * 0.28
    dome = (~cylinder) & (~polar)
    values[cylinder, :] = 0.75
    values[dome, :] = 0.5
    values[polar, :] = 0.15
    return values


def _heatmap_color(value: float) -> Rgb:
    clipped = max(0.0, min(1.0, value))
    red = int(42 + clipped * 200)
    green = int(96 + (1.0 - abs(clipped - 0.5) * 2.0) * 120)
    blue = int(160 - clipped * 130)
    return red, green, max(20, blue)


def _new_image(width: int, height: int, color: Rgb) -> bytearray:
    data = bytearray(width * height * 3)
    for index in range(0, len(data), 3):
        data[index:index + 3] = bytes(color)
    return data


def _draw_axes(image: bytearray, left: int, top: int, right: int, bottom: int) -> None:
    _draw_line(image, left, bottom, right, bottom, (38, 48, 58))
    _draw_line(image, left, top, left, bottom, (38, 48, 58))


def _draw_grid(image: bytearray, left: int, top: int, right: int, bottom: int) -> None:
    for step in range(1, 10):
        x = int(left + step * (right - left) / 10)
        _draw_line(image, x, top, x, bottom, (225, 231, 238))
    for step in range(1, 6):
        y = int(top + step * (bottom - top) / 6)
        _draw_line(image, left, y, right, y, (225, 231, 238))


def _draw_cylinder_outline(
    image: bytearray,
    left: int,
    top: int,
    right: int,
    bottom: int,
    y_extent: float,
    radius: float,
) -> None:
    mid = (top + bottom) // 2
    half = int(radius / y_extent * (bottom - top) / 2.0)
    _draw_line(image, left, mid - half, right, mid - half, (190, 199, 208))
    _draw_line(image, left, mid + half, right, mid + half, (190, 199, 208))


def _draw_layered_polyline(
    image: bytearray,
    x_values: np.ndarray,
    y_values: np.ndarray,
    layer_ids: tuple[str, ...],
    color_map: dict[str, Rgb],
    *,
    motion_types: tuple[str, ...] | None = None,
    debug_transitions: bool = False,
    layer_filter: str | None = None,
    motion_types_filter: set[str] | None = None,
) -> None:
    for index in range(1, len(x_values)):
        if layer_ids[index] != layer_ids[index - 1]:
            continue
        if layer_filter is not None and layer_ids[index] != layer_filter:
            continue
        if motion_types_filter is not None:
            if motion_types is None:
                continue
            if (
                motion_types[index] not in motion_types_filter
                and motion_types[index - 1] not in motion_types_filter
            ):
                continue
        is_transition = (
            motion_types is not None
            and (motion_types[index] == "transition" or motion_types[index - 1] == "transition")
        )
        color = (
            (20, 20, 20)
            if debug_transitions and is_transition
            else color_map.get(layer_ids[index], _fallback_color(layer_ids[index]))
        )
        _draw_line(
            image,
            int(x_values[index - 1]),
            int(y_values[index - 1]),
            int(x_values[index]),
            int(y_values[index]),
            color,
        )


def _draw_pass_markers(
    image: bytearray,
    x_values: np.ndarray,
    y_values: np.ndarray,
    pass_ids: tuple[int, ...],
) -> None:
    for index in range(len(pass_ids)):
        previous_changed = index == 0 or pass_ids[index] != pass_ids[index - 1]
        next_changed = index == len(pass_ids) - 1 or pass_ids[index] != pass_ids[index + 1]
        if previous_changed:
            _draw_square(image, int(x_values[index]), int(y_values[index]), 4, (18, 137, 62))
        if next_changed:
            _draw_square(image, int(x_values[index]), int(y_values[index]), 4, (190, 38, 38))


def _draw_transition_markers(
    image: bytearray,
    x_values: np.ndarray,
    y_values: np.ndarray,
    motion_types: tuple[str, ...],
) -> None:
    for index, motion_type in enumerate(motion_types):
        if motion_type == "transition" and index % 4 == 0:
            _draw_square(image, int(x_values[index]), int(y_values[index]), 3, (8, 8, 8))


def _draw_square(image: bytearray, x: int, y: int, radius: int, color: Rgb) -> None:
    width = 1200
    height = len(image) // (width * 3)
    for yy in range(y - radius, y + radius + 1):
        for xx in range(x - radius, x + radius + 1):
            _set_pixel(image, width, height, xx, yy, color)


def _draw_line(image: bytearray, x0: int, y0: int, x1: int, y1: int, color: Rgb) -> None:
    width = 1200
    height = len(image) // (width * 3)
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        _set_pixel(image, width, height, x, y, color)
        _set_pixel(image, width, height, x + 1, y, color)
        _set_pixel(image, width, height, x, y + 1, color)
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy


def _set_pixel(
    image: bytearray,
    width: int,
    height: int,
    x: int,
    y: int,
    color: Rgb,
) -> None:
    if x < 0 or y < 0 or x >= width or y >= height:
        return
    index = (y * width + x) * 3
    image[index:index + 3] = bytes(color)


def _write_png(path: Path, image: bytearray) -> None:
    width = 1200
    height = len(image) // (width * 3)
    rows = bytearray()
    stride = width * 3
    for row in range(height):
        rows.append(0)
        rows.extend(image[row * stride:(row + 1) * stride])
    payload = b"".join(
        (
            _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
            _png_chunk(b"IDAT", zlib.compress(bytes(rows), level=6)),
            _png_chunk(b"IEND", b""),
        )
    )
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + payload)


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    checksum = zlib.crc32(kind)
    checksum = zlib.crc32(data, checksum)
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum & 0xFFFFFFFF)


def _parse_color(value: str) -> Rgb:
    text = value.strip()
    if text.startswith("#") and len(text) == 7:
        try:
            return int(text[1:3], 16), int(text[3:5], 16), int(text[5:7], 16)
        except ValueError:
            pass
    return _fallback_color(text)


def _fallback_color(value: str) -> Rgb:
    hue = abs(hash(value)) % 360
    c = 180
    x = int(c * (1 - abs((hue / 60) % 2 - 1)))
    if hue < 60:
        return c, x, 60
    if hue < 120:
        return x, c, 60
    if hue < 180:
        return 60, c, x
    if hue < 240:
        return 60, x, c
    if hue < 300:
        return x, 60, c
    return c, 60, x


def _safe_id(value: str) -> str:
    clean = "".join(char.lower() if char.isalnum() else "-" for char in value).strip("-")
    return clean or "layer"


def _length(mandrel: CylinderMandrel | AxisymmetricProfileMandrel) -> float:
    return mandrel.length_mm


def _radius(mandrel: CylinderMandrel | AxisymmetricProfileMandrel) -> float:
    return mandrel.radius_mm if isinstance(mandrel, CylinderMandrel) else mandrel.max_radius_mm


def _start_z(mandrel: CylinderMandrel | AxisymmetricProfileMandrel) -> float:
    return 0.0 if isinstance(mandrel, CylinderMandrel) else mandrel.start_z_mm


def _end_z(mandrel: CylinderMandrel | AxisymmetricProfileMandrel) -> float:
    return mandrel.length_mm if isinstance(mandrel, CylinderMandrel) else mandrel.end_z_mm
