from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from filament_winder.cli import main
from filament_winder.core.geometry import AxisymmetricProfileMandrel
from filament_winder.core.kinematics import machine_path_from_surface_path
from filament_winder.core.path_planning import (
    ProfileDomePathConfig,
    ProfileDomePathGenerator,
    ProfileTurnaroundPathConfig,
    ProfileTurnaroundPathGenerator,
    find_profile_safe_zone,
)


def _profile_with_poles() -> AxisymmetricProfileMandrel:
    return AxisymmetricProfileMandrel(
        z_mm=np.asarray([0.0, 20.0, 80.0, 100.0]),
        r_mm=np.asarray([0.0, 20.0, 20.0, 0.0]),
    )


def test_find_profile_safe_zone_interpolates_pole_avoidance_boundaries() -> None:
    safe_zone = find_profile_safe_zone(_profile_with_poles(), min_radius_mm=5.0)

    assert safe_zone.start_z_mm == pytest.approx(5.0)
    assert safe_zone.end_z_mm == pytest.approx(95.0)
    assert safe_zone.start_radius_mm == pytest.approx(5.0)
    assert safe_zone.end_radius_mm == pytest.approx(5.0)


def test_profile_turnaround_path_stays_inside_safe_radius() -> None:
    profile = _profile_with_poles()
    config = ProfileTurnaroundPathConfig(
        winding_angle_deg=35.0,
        tow_width_mm=3.0,
        points_per_span=20,
        turnaround_points=5,
        min_radius_mm=5.0,
        turnaround_angle_deg=180.0,
    )

    path = ProfileTurnaroundPathGenerator(profile, config).generate()
    radius = profile.radius_at(path.z_mm)

    assert path.point_count == 47
    assert path.pass_count == 2
    assert float(np.min(radius)) >= 5.0 - 1e-9
    assert np.min(path.z_mm) == pytest.approx(5.0)
    assert np.max(path.z_mm) == pytest.approx(95.0)
    assert path.theta_rad[-1] > path.theta_rad[0]
    assert np.count_nonzero(np.isclose(path.z_mm, 95.0)) >= 5
    assert np.count_nonzero(np.isclose(path.z_mm, 5.0)) >= 5


def test_profile_dome_path_uses_geodesic_turnaround_and_variable_tow_angle() -> None:
    profile = _profile_with_poles()
    config = ProfileDomePathConfig(
        winding_angle_deg=30.0,
        tow_width_mm=3.0,
        points_per_span=20,
        turnaround_points=5,
    )

    generator = ProfileDomePathGenerator(profile, config)
    path = generator.generate()
    radius = profile.radius_at(path.z_mm)
    motion = machine_path_from_surface_path(path, radial_clearance_mm=5.0)

    assert generator.clairaut_radius_mm == pytest.approx(10.0)
    assert generator.turnaround_radius_mm == pytest.approx(10.0)
    assert path.point_count == 47
    assert path.pass_count == 2
    assert float(np.min(radius)) >= 10.0 - 1e-9
    assert np.min(path.z_mm) == pytest.approx(10.0)
    assert np.max(path.z_mm) == pytest.approx(90.0)
    assert path.tow_eye_angle_deg is not None
    assert np.min(path.tow_eye_angle_deg) == pytest.approx(30.0, abs=0.1)
    assert np.max(path.tow_eye_angle_deg) == pytest.approx(90.0)
    assert np.min(motion.b_deg) == pytest.approx(30.0, abs=0.1)
    assert np.max(motion.b_deg) == pytest.approx(90.0)
    assert path.final_rotation_deg < 1000.0


def test_profile_turnaround_cli_exports_csv_and_gcode(tmp_path: Path) -> None:
    dxf_path = tmp_path / "profile.dxf"
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
    csv_path = tmp_path / "profile.csv"
    gcode_path = tmp_path / "profile.gcode"

    result = main(
        [
            "profile-turnaround",
            str(dxf_path),
            "--angle",
            "35",
            "--tow-width",
            "3",
            "--min-radius",
            "5",
            "--points",
            "20",
            "--turnaround-points",
            "5",
            "--csv",
            str(csv_path),
            "--gcode",
            str(gcode_path),
        ]
    )

    assert result == 0
    assert csv_path.exists()
    assert gcode_path.exists()
    assert csv_path.stat().st_size > 0
    assert gcode_path.stat().st_size > 0


def test_profile_dome_cli_exports_csv_and_gcode(tmp_path: Path) -> None:
    dxf_path = tmp_path / "profile.dxf"
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
    csv_path = tmp_path / "profile_dome.csv"
    gcode_path = tmp_path / "profile_dome.gcode"

    result = main(
        [
            "profile-dome",
            str(dxf_path),
            "--angle",
            "30",
            "--tow-width",
            "3",
            "--points",
            "20",
            "--turnaround-points",
            "5",
            "--csv",
            str(csv_path),
            "--gcode",
            str(gcode_path),
        ]
    )

    assert result == 0
    assert csv_path.exists()
    assert gcode_path.exists()
    assert csv_path.stat().st_size > 0
    assert gcode_path.stat().st_size > 0


def test_dxf_info_cli_missing_file_is_clean(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = main(["dxf-info", str(tmp_path / "missing.dxf")])

    captured = capsys.readouterr()
    assert result == 1
    assert "Profile file not found" in captured.err
    assert "Traceback" not in captured.err


def test_profile_turnaround_cli_missing_file_is_clean(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = main(
        [
            "profile-turnaround",
            str(tmp_path / "missing.dxf"),
            "--angle",
            "35",
            "--tow-width",
            "3",
        ]
    )

    captured = capsys.readouterr()
    assert result == 1
    assert "Profile file not found" in captured.err
    assert "Traceback" not in captured.err
