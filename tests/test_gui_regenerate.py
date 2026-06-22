from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402, I001
from PySide6 import QtCore, QtTest  # noqa: E402, I001

from filament_winder.app.gui import (  # noqa: E402, I001
    _PreviewWindow,
    _create_qapplication,
    _gui_logger,
    _load_gui_modules,
)
from filament_winder.app.preview import (  # noqa: E402
    CylinderPreviewConfig,
    ProfileDomePreviewConfig,
)
from filament_winder.core.path_planning import plan_winding_schedule  # noqa: E402


def _show_window(window: _PreviewWindow) -> None:
    window.widget.show()
    app = window._qt_widgets.QApplication.instance()
    if app is not None:
        app.processEvents()


def _click_button(window: _PreviewWindow, object_name: str) -> None:
    button = window.widget.findChild(window._qt_widgets.QPushButton, object_name)
    assert button is not None
    assert button.isVisible()
    assert button.isEnabled()
    QtTest.QTest.mouseClick(button, QtCore.Qt.MouseButton.LeftButton)
    app = window._qt_widgets.QApplication.instance()
    if app is not None:
        app.processEvents()


def _mandrel_vertices(window: _PreviewWindow) -> np.ndarray:
    assert window._visuals
    return window._visuals[0]._meshdata.get_vertices()


def test_regenerate_uses_main_setup_controls() -> None:
    modules = _load_gui_modules()
    app = _create_qapplication(modules[0], ["test"], _gui_logger())
    window = _PreviewWindow(
        *modules,
        CylinderPreviewConfig(),
        ProfileDomePreviewConfig(),
        "cylinder",
    )

    window.length.setValue(500.0)
    window.radius.setValue(60.0)
    window.angle.setValue(35.0)
    window.passes.setValue(3)
    window._regenerate_current()

    status = window.status.text()
    assert "Mode: cylinder" in status
    assert "L=500.0 mm" in status
    assert "R=60.0 mm" in status
    assert "Angle=35.0 deg" in status
    assert "Passes: auto ->" in status
    assert "Coverage target: 115.0%" in status
    assert window._simple_backend_config is not None
    assert window._simple_backend_config.layers[0].passes == "auto"
    assert window._simple_backend_config.layers[0].coverage_target == 1.15
    assert window._simple_backend_config.layers[0].direction == "forward"
    assert "Regenerated:" in status
    assert app is not None
    window.widget.close()


def test_regenerate_updates_profile_dome_mode() -> None:
    modules = _load_gui_modules()
    _create_qapplication(modules[0], ["test"], _gui_logger())
    window = _PreviewWindow(
        *modules,
        CylinderPreviewConfig(),
        ProfileDomePreviewConfig(),
        "cylinder",
    )

    window.mode.setCurrentText("Profile Dome")
    window.angle.setValue(40.0)
    window.tow_width.setValue(4.0)
    window.circuits.setValue(1)
    window._regenerate_current()

    status = window.status.text()
    assert "Mode: profile dome" in status
    assert "Angle=40.0 deg" in status
    assert "Tow=4.0 mm" in status
    assert "Passes: auto ->" in status
    assert window._simple_backend_config is not None
    assert window._simple_backend_config.mandrel.type == "cylinder_with_elliptical_domes"
    assert window._simple_backend_config.mandrel.left_dome_length_mm > 0.0
    assert window._simple_backend_config.mandrel.right_dome_length_mm > 0.0
    assert window._simple_backend_config.mandrel.polar_opening_radius_mm == 5.0
    assert window._simple_backend_config.layers[0].passes == "auto"
    assert window._simple_backend_config.layers[0].coverage_target == 1.15
    assert "Regenerated:" in status
    window.widget.close()


def test_workflow_regenerate_button_updates_cylinder_preview() -> None:
    modules = _load_gui_modules()
    _create_qapplication(modules[0], ["test"], _gui_logger())
    window = _PreviewWindow(
        *modules,
        CylinderPreviewConfig(),
        ProfileDomePreviewConfig(),
        "cylinder",
    )
    _show_window(window)

    window.length.setValue(450.0)
    window.radius.setValue(55.0)
    window.angle.setValue(30.0)
    _click_button(window, "workflowRegenerateButton")

    status = window.status.text()
    assert "L=450.0 mm" in status
    assert "R=55.0 mm" in status
    assert "Angle=30.0 deg" in status
    assert "Passes: auto ->" in status
    assert "Regenerated:" in status
    window.widget.close()


def test_regenerate_button_updates_rendered_mandrel_mesh() -> None:
    modules = _load_gui_modules()
    _create_qapplication(modules[0], ["test"], _gui_logger())
    window = _PreviewWindow(
        *modules,
        CylinderPreviewConfig(),
        ProfileDomePreviewConfig(),
        "cylinder",
    )
    _show_window(window)

    before_vertices = _mandrel_vertices(window)
    before_length_extent = float(before_vertices[:, 0].max() - before_vertices[:, 0].min())
    assert before_length_extent == 1000.0

    window.length.lineEdit().setText("320")
    window.radius.lineEdit().setText("40")
    _click_button(window, "workflowRegenerateButton")

    after_vertices = _mandrel_vertices(window)
    after_length_extent = float(after_vertices[:, 0].max() - after_vertices[:, 0].min())
    after_radial_extent = float(after_vertices[:, 1].max() - after_vertices[:, 1].min())
    assert after_length_extent == 320.0
    assert after_radial_extent == 80.0
    window.widget.close()


def test_viewport_regenerate_button_updates_profile_preview() -> None:
    modules = _load_gui_modules()
    _create_qapplication(modules[0], ["test"], _gui_logger())
    window = _PreviewWindow(
        *modules,
        CylinderPreviewConfig(),
        ProfileDomePreviewConfig(),
        "cylinder",
    )
    _show_window(window)

    window.mode.setCurrentText("Profile Dome")
    window.angle.setValue(42.0)
    window.tow_width.setValue(5.0)
    _click_button(window, "viewportRegenerateButton")

    status = window.status.text()
    assert "Mode: profile dome" in status
    assert "Angle=42.0 deg" in status
    assert "Tow=5.0 mm" in status
    assert "Regenerated:" in status
    window.widget.close()


def test_profile_regenerate_syncs_backend_graph_and_generator() -> None:
    modules = _load_gui_modules()
    _create_qapplication(modules[0], ["test"], _gui_logger())
    window = _PreviewWindow(
        *modules,
        CylinderPreviewConfig(),
        ProfileDomePreviewConfig(),
        "cylinder",
    )
    _show_window(window)

    window.mode.setCurrentText("Profile Dome")
    window.angle.setValue(38.0)
    window.tow_width.setValue(4.5)
    _click_button(window, "workflowRegenerateButton")

    config = window._backend_service.build_project_from_graph(window._node_graph)
    assert config.mandrel.type == "cylinder_with_elliptical_domes"
    assert config.mandrel.profile_path is None
    assert config.mandrel.polar_opening_radius_mm == 5.0
    assert config.layers[0].type == "geodesic"
    assert config.layers[0].winding_angle_deg == 38.0
    assert config.layers[0].passes == "auto"
    assert config.layers[0].coverage_target == 1.15
    assert config.tow.width_mm == 4.5

    result = window._backend_service.generate(
        config,
        export_csv=False,
        export_summary=False,
        make_plots=False,
    )
    assert result.program.point_count > 0
    assert result.program.reports[0].coverage_percent >= 100.0
    assert result.mandrel.length_mm > 0
    window.widget.close()


def test_auto_passes_ignore_manual_spinner_and_reach_full_coverage() -> None:
    modules = _load_gui_modules()
    _create_qapplication(modules[0], ["test"], _gui_logger())
    window = _PreviewWindow(
        *modules,
        CylinderPreviewConfig(),
        ProfileDomePreviewConfig(),
        "cylinder",
    )
    _show_window(window)

    window.passes.setValue(1)
    window.target_coverage.setValue(10.0)
    _click_button(window, "workflowRegenerateButton")

    config = window._backend_service.build_project_from_graph(window._node_graph)
    assert config.layers[0].passes == "auto"
    assert config.layers[0].coverage_target == 1.15

    mandrel = window._preview_mandrel_from_config(config)
    schedule = window._preview_schedule_from_config(config)
    planned = plan_winding_schedule(mandrel, schedule)
    assert planned.reports[0].circuits > 1
    assert planned.reports[0].coverage_percent >= 100.0
    window.widget.close()


def test_workflow_export_button_writes_outputs(tmp_path: Path) -> None:
    modules = _load_gui_modules()
    _create_qapplication(modules[0], ["test"], _gui_logger())
    window = _PreviewWindow(
        *modules,
        CylinderPreviewConfig(),
        ProfileDomePreviewConfig(),
        "cylinder",
    )
    _show_window(window)

    csv_path = tmp_path / "preview.csv"
    gcode_path = tmp_path / "preview.gcode"
    coverage_path = tmp_path / "coverage.csv"
    summary_path = tmp_path / "coverage_summary.csv"
    obj_path = tmp_path / "preview.obj"
    window.csv_output.setText(str(csv_path))
    window.gcode_output.setText(str(gcode_path))
    window.coverage_output.setText(str(coverage_path))
    window.coverage_summary_output.setText(str(summary_path))
    window.obj_output.setText(str(obj_path))

    _click_button(window, "workflowExportButton")

    assert csv_path.exists()
    assert gcode_path.exists()
    assert coverage_path.exists()
    assert summary_path.exists()
    assert obj_path.exists()
    assert "Exported:" in window.status.text()
    window.widget.close()
