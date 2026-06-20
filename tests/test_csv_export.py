from __future__ import annotations

import csv

import pytest

from filament_winder.core.geometry import CylinderMandrel
from filament_winder.core.kinematics import machine_path_from_surface_path
from filament_winder.core.path_planning import HelicalPathConfig, HelicalPathGenerator
from filament_winder.io import export_winding_csv


def test_export_winding_csv_writes_surface_and_machine_columns(tmp_path) -> None:
    mandrel = CylinderMandrel(length_mm=100.0, radius_mm=20.0)
    config = HelicalPathConfig(winding_angle_deg=45.0, tow_width_mm=3.0, point_count=3)
    path = HelicalPathGenerator(mandrel, config).generate()
    motion = machine_path_from_surface_path(path, radial_clearance_mm=5.0)

    output_path = export_winding_csv(path, motion, tmp_path / "path.csv")

    with output_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 3
    assert rows[0].keys() == {
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
    }
    assert float(rows[-1]["Z_mm"]) == pytest.approx(100.0)
    assert float(rows[-1]["X_mm"]) == pytest.approx(25.0)
