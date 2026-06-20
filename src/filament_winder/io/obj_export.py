"""Wavefront OBJ preview export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from filament_winder.core.geometry import CylinderMandrel
from filament_winder.core.path_planning import SurfacePath
from filament_winder.core.tow import TowBand

FloatArray = NDArray[np.float64]


def export_cylinder_preview_obj(
    mandrel: CylinderMandrel,
    surface_path: SurfacePath,
    output_path: str | Path,
    *,
    tow_band: TowBand | None = None,
    theta_segments: int = 48,
    z_segments: int = 16,
) -> Path:
    """Export a simple 3D preview OBJ with cylinder, path, and optional tow band."""

    if theta_segments < 8:
        raise ValueError("theta_segments must be at least 8")
    if z_segments < 2:
        raise ValueError("z_segments must be at least 2")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cylinder_vertices = _cylinder_vertices(mandrel, theta_segments, z_segments)

    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# FilamentWinder cylinder preview\n")
        handle.write("o mandrel\n")
        for vertex in cylinder_vertices:
            handle.write(_format_vertex(vertex))
        for z_index in range(z_segments):
            ring_start = z_index * theta_segments
            next_ring_start = (z_index + 1) * theta_segments
            for theta_index in range(theta_segments):
                a = ring_start + theta_index + 1
                b = ring_start + ((theta_index + 1) % theta_segments) + 1
                c = next_ring_start + ((theta_index + 1) % theta_segments) + 1
                d = next_ring_start + theta_index + 1
                handle.write(f"f {a} {b} {c} {d}\n")

        path_start = cylinder_vertices.shape[0] + 1
        handle.write("o helical_path\n")
        for vertex in surface_path.points_mm:
            handle.write(_format_vertex(vertex))
        path_indices = " ".join(
            str(path_start + index) for index in range(surface_path.point_count)
        )
        handle.write(f"l {path_indices}\n")

        if tow_band is not None:
            tow_start = path_start + surface_path.point_count
            handle.write("o tow_band\n")
            for vertex in tow_band.vertices_mm:
                handle.write(_format_vertex(vertex))
            for quad in tow_band.quad_indices:
                indices = " ".join(str(tow_start + int(index)) for index in quad)
                handle.write(f"f {indices}\n")
    return path


def _cylinder_vertices(
    mandrel: CylinderMandrel,
    theta_segments: int,
    z_segments: int,
) -> FloatArray:
    z_mm = np.linspace(0.0, mandrel.length_mm, z_segments + 1)
    theta_rad = np.linspace(0.0, 2.0 * np.pi, theta_segments, endpoint=False)
    vertices = []
    for z_value in z_mm:
        for theta_value in theta_rad:
            vertices.append(mandrel.surface_points([z_value], [theta_value])[0])
    return np.asarray(vertices, dtype=float)


def _format_vertex(vertex: FloatArray) -> str:
    return f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n"
