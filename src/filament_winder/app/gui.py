"""Optional PySide6/VisPy live preview."""

from __future__ import annotations

import importlib
import importlib.util
import json
import logging
import sys
import traceback
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np

from filament_winder.app.exporting import (
    export_cylinder_pattern_preview_files,
    export_preview_files,
    export_profile_dome_pattern_preview_files,
    export_profile_dome_preview_files,
)
from filament_winder.app.node_graph import (
    GraphExecutionResult,
    NodeGraphController,
    NodeGraphExecutor,
    NodeGraphState,
    NodeInstance,
    NodeTypeDefinition,
    addable_node_type_ids,
    default_filament_winder_graph,
    default_node_registry,
)
from filament_winder.app.preview import (
    CylinderPreviewConfig,
    PatternPlannerConfig,
    ProfileDomePreviewConfig,
    ProfilePathMode,
    build_cylinder_pattern_preview_scene,
    build_cylinder_preview_scene,
    build_profile_dome_pattern_preview_scene,
    build_profile_dome_preview_scene,
    cylinder_mesh_arrays,
    offset_display_surface,
    orient_points_for_horizontal_view,
    profile_mesh_arrays,
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
from filament_winder.core.geometry import CylinderMandrel
from filament_winder.core.path_planning import (
    CylinderPatternOptimizationRequest,
    WindingLayerSpec,
    WindingSchedule,
    optimize_cylinder_pattern,
    plan_winding_schedule,
)
from filament_winder.io import (
    GCodeOptions,
    export_gcode,
    export_winding_program_csv,
    import_dxf_zr_profile,
)
from filament_winder.project import load_project, save_project

GUI_LOG_PATH = Path("logs/filament_winder_gui.log")


def _gui_logger() -> logging.Logger:
    logger = logging.getLogger("filament_winder.gui")
    if logger.handlers:
        return logger
    GUI_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(GUI_LOG_PATH, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


class GuiDependencyError(RuntimeError):
    """Raised when optional GUI dependencies are not installed."""


def gui_dependencies_available() -> bool:
    return (
        importlib.util.find_spec("PySide6") is not None
        and importlib.util.find_spec("vispy") is not None
    )


def missing_gui_dependency_message() -> str:
    return "Install GUI dependencies with: pip install -e .[gui]"


def launch_cylinder_preview(
    config: CylinderPreviewConfig | None = None,
    *,
    profile_config: ProfileDomePreviewConfig | None = None,
    initial_mode: str = "cylinder",
) -> int:
    """Launch a live cylinder preview window."""

    (
        qt_widgets,
        qt_core,
        qt_gui,
        vispy_scene,
        vispy_keys,
        vispy_arcball,
        vispy_quaternion,
    ) = _load_gui_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(sys.argv[:1])
    window = _PreviewWindow(
        qt_widgets,
        qt_core,
        qt_gui,
        vispy_scene,
        vispy_keys,
        vispy_arcball,
        vispy_quaternion,
        config or CylinderPreviewConfig(),
        profile_config or ProfileDomePreviewConfig(),
        initial_mode,
    )
    window.resize(1500, 900)
    window.show()
    return int(app.exec())


def _load_gui_modules() -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    if not gui_dependencies_available():
        raise GuiDependencyError(missing_gui_dependency_message())
    qt_widgets = importlib.import_module("PySide6.QtWidgets")
    qt_core = importlib.import_module("PySide6.QtCore")
    qt_gui = importlib.import_module("PySide6.QtGui")
    vispy_app = importlib.import_module("vispy.app")
    vispy_app.use_app("pyside6")
    vispy_scene = importlib.import_module("vispy.scene")
    vispy_keys = importlib.import_module("vispy.util.keys")
    vispy_arcball = importlib.import_module("vispy.scene.cameras.arcball")
    vispy_quaternion = importlib.import_module("vispy.util.quaternion")
    return qt_widgets, qt_core, qt_gui, vispy_scene, vispy_keys, vispy_arcball, vispy_quaternion


class _PreviewWindow:
    def __init__(
        self,
        qt_widgets: Any,
        qt_core: Any,
        qt_gui: Any,
        vispy_scene: Any,
        vispy_keys: Any,
        vispy_arcball: Any,
        vispy_quaternion: Any,
        config: CylinderPreviewConfig,
        profile_config: ProfileDomePreviewConfig,
        initial_mode: str,
    ) -> None:
        self._qt_widgets = qt_widgets
        self._qt_core = qt_core
        self._qt_gui = qt_gui
        self._vispy_scene = vispy_scene
        self._vispy_keys = vispy_keys
        self._arcball = vispy_arcball._arcball
        self._quaternion = vispy_quaternion.Quaternion
        self._logger = _gui_logger()
        self._logger.info("Application startup")
        self._config = config
        self._profile_config = profile_config
        self._visuals: list[Any] = []
        self._drag_last_pos: np.ndarray | None = None
        self._current_project_path: Path | None = None
        self._default_export_paths = PreviewExportPaths()
        self._node_registry = default_node_registry()
        self._node_graph = default_filament_winder_graph(
            length_mm=config.length_mm,
            radius_mm=config.radius_mm,
            tow_width_mm=config.tow_width_mm,
            angle_deg=config.winding_angle_deg,
            point_count=config.points_per_pass,
            radial_clearance_mm=config.radial_clearance_mm,
            profile_path=str(profile_config.profile_path),
            profile_mode="profile" if initial_mode == "profile-dome" else "cylinder",
        )
        self._node_items: dict[str, Any] = {}
        self._socket_items: dict[tuple[str, str, str], Any] = {}
        self._node_link_items: list[Any] = []
        self._node_group_items: list[Any] = []
        self._refreshing_node_links = False
        self._node_socket_drag: dict[str, str] | None = None
        self._node_temp_link_item: Any | None = None
        self._node_highlight_socket: Any | None = None
        self._last_node_result: GraphExecutionResult | None = None
        self._viewport_node_context: str | None = None
        self._node_thread_pool = qt_core.QThreadPool.globalInstance()
        self._node_workers: list[Any] = []

        self.widget = qt_widgets.QMainWindow()
        self.widget.setWindowTitle("FilamentWinder Preview")
        self.widget.setStyleSheet(_modern_stylesheet())
        central = qt_widgets.QWidget()
        root = qt_widgets.QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        main_splitter = qt_widgets.QSplitter(self._qt_core.Qt.Orientation.Vertical)
        main_splitter.setObjectName("mainVerticalSplitter")
        main_splitter.setChildrenCollapsible(False)

        self.canvas = vispy_scene.SceneCanvas(
            keys="interactive",
            bgcolor="#101418",
            show=False,
        )
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = vispy_scene.cameras.ArcballCamera(
            fov=45.0,
            distance=max(config.length_mm, config.radius_mm * 6.0) * 1.4,
            center=(0.0, 0.0, 0.0),
            scale_factor=max(config.length_mm, config.radius_mm * 6.0),
            translate_speed=1.0,
        )
        self.view.camera.interactive = False
        vispy_scene.visuals.XYZAxis(parent=self.view.scene)
        self.canvas.events.mouse_press.connect(self._on_mouse_press)
        self.canvas.events.mouse_move.connect(self._on_mouse_move)
        self.canvas.events.mouse_release.connect(self._on_mouse_release)
        self.canvas.events.mouse_wheel.connect(self._on_mouse_wheel)

        viewport_panel = self._build_viewport_panel(self.canvas.native)
        controls = self._build_controls(config)
        main_splitter.addWidget(viewport_panel)
        main_splitter.addWidget(controls)
        main_splitter.setStretchFactor(0, 2)
        main_splitter.setStretchFactor(1, 1)
        main_splitter.setSizes([700, 350])
        root.addWidget(main_splitter, 1)

        self.widget.setCentralWidget(central)
        if initial_mode == "profile-dome":
            self.mode.setCurrentText("Profile Dome")
        self._render_scene()

    def _build_viewport_panel(self, canvas_widget: Any) -> Any:
        qt_widgets = self._qt_widgets
        qt_core = self._qt_core
        panel = qt_widgets.QWidget()
        panel.setObjectName("viewportPanel")
        panel.setMinimumHeight(320)
        layout = qt_widgets.QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 8)
        layout.setSpacing(8)

        toolbar = qt_widgets.QHBoxLayout()
        toolbar.setSpacing(8)
        refresh = qt_widgets.QPushButton("Update Preview")
        self._connect_safe_button(refresh, "Update Preview", self._render_scene)
        reset_view = qt_widgets.QPushButton("Reset View")
        self._connect_safe_button(reset_view, "Reset View", self._reset_camera)
        fit_mandrel = qt_widgets.QPushButton("Fit Mandrel")
        self._connect_safe_button(fit_mandrel, "Fit Mandrel", self._reset_camera)
        self.show_tow_path = qt_widgets.QCheckBox("Tow Path")
        self.show_tow_path.setChecked(True)
        self.show_tow_path.toggled.connect(
            lambda _checked: self.run_safe_action("Toggle Tow Path", self._render_scene)
        )
        self.show_tow_band = qt_widgets.QCheckBox("Tow Band")
        self.show_tow_band.setChecked(True)
        self.show_tow_band.toggled.connect(
            lambda _checked: self.run_safe_action("Toggle Tow Band", self._render_scene)
        )
        self.show_coverage = qt_widgets.QCheckBox("Coverage")
        self.show_coverage.setEnabled(False)
        self.show_coverage.setToolTip("Coverage-map display is shown from analysis nodes.")
        self.show_machine_view = qt_widgets.QCheckBox("Machine View")
        self.show_machine_view.setEnabled(False)
        self.show_machine_view.setToolTip(
            "Machine clearance display is planned for controller nodes."
        )
        camera_help = qt_widgets.QLabel(
            "Left-drag orbit | Shift+left or middle-drag pan | wheel/right-drag zoom"
        )
        camera_help.setObjectName("viewportHelp")
        toolbar.addWidget(refresh)
        toolbar.addWidget(reset_view)
        toolbar.addWidget(fit_mandrel)
        toolbar.addSpacing(12)
        toolbar.addWidget(self.show_tow_path)
        toolbar.addWidget(self.show_tow_band)
        toolbar.addWidget(self.show_coverage)
        toolbar.addWidget(self.show_machine_view)
        toolbar.addStretch(1)
        toolbar.addWidget(camera_help)
        layout.addLayout(toolbar)

        viewport_frame = qt_widgets.QFrame()
        viewport_frame.setObjectName("viewportFrame")
        viewport_layout = qt_widgets.QStackedLayout(viewport_frame)
        viewport_layout.setContentsMargins(0, 0, 0, 0)
        viewport_layout.setStackingMode(qt_widgets.QStackedLayout.StackingMode.StackAll)
        viewport_layout.addWidget(canvas_widget)

        overlay = qt_widgets.QWidget()
        overlay.setAttribute(qt_core.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        overlay_layout = qt_widgets.QVBoxLayout(overlay)
        overlay_layout.setContentsMargins(12, 12, 12, 12)
        self.status = qt_widgets.QLabel("Viewport ready")
        self.status.setObjectName("viewportStatusOverlay")
        self.status.setWordWrap(True)
        self.status.setMinimumWidth(260)
        self.status.setMaximumWidth(520)
        overlay_layout.addWidget(
            self.status,
            0,
            qt_core.Qt.AlignmentFlag.AlignTop | qt_core.Qt.AlignmentFlag.AlignLeft,
        )
        overlay_layout.addStretch(1)
        viewport_layout.addWidget(overlay)
        layout.addWidget(viewport_frame, 1)
        return panel

    def __getattr__(self, name: str) -> Any:
        return getattr(self.widget, name)

    def run_safe_action(self, action_name: str, callback: Callable[[], Any]) -> Any | None:
        """Run a GUI action without allowing exceptions to escape into Qt."""

        self._logger.info("Button clicked: %s", action_name)
        self._logger.info("Action started: %s", action_name)
        previous_status = self.status.text() if hasattr(self, "status") else ""
        self._set_gui_status(f"Running: {action_name}")
        try:
            result = callback()
        except Exception as exc:  # noqa: BLE001 - protects Qt event loop
            self.handle_gui_error(action_name, exc)
            return None
        self._logger.info("Action completed: %s", action_name)
        if hasattr(self, "status") and self.status.text() in {
            f"Running: {action_name}",
            previous_status,
        }:
            self._set_gui_status(f"Complete: {action_name}")
        if hasattr(self, "node_status_log"):
            self.node_status_log.appendPlainText(f"Complete: {action_name}")
        return result

    def handle_gui_error(self, action_name: str, exc: BaseException) -> None:
        message = f"{action_name} failed: {exc}"
        self._logger.error("Action failed: %s\n%s", action_name, traceback.format_exc())
        self._set_gui_status(message)
        if hasattr(self, "node_status"):
            self.node_status.setText(message)
        if hasattr(self, "node_status_log"):
            self.node_status_log.appendPlainText(message)
        if hasattr(self, "node_debug_log"):
            self.node_debug_log.appendPlainText(message)
            self.node_debug_log.appendPlainText(traceback.format_exc())

    def _set_gui_status(self, message: str) -> None:
        if hasattr(self, "status"):
            self.status.setText(message)
        if hasattr(self, "node_status"):
            self.node_status.setText(message)
        if hasattr(self, "node_status_log"):
            self.node_status_log.appendPlainText(message)

    def _connect_safe_button(
        self,
        button: Any,
        action_name: str,
        callback: Callable[[], Any],
    ) -> None:
        button.clicked.connect(
            lambda _checked=False, name=action_name, cb=callback: self.run_safe_action(
                name,
                cb,
            )
        )

    def _build_controls(self, config: CylinderPreviewConfig) -> Any:
        qt_widgets = self._qt_widgets
        panel = qt_widgets.QWidget()
        panel.setObjectName("controlPanel")
        panel.setMinimumHeight(280)
        panel.setSizePolicy(
            qt_widgets.QSizePolicy.Policy.Expanding,
            qt_widgets.QSizePolicy.Policy.Expanding,
        )
        layout = qt_widgets.QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        tabs = qt_widgets.QTabWidget()
        tabs.setUsesScrollButtons(True)
        form = qt_widgets.QFormLayout()

        self.mode = qt_widgets.QComboBox()
        self.mode.addItems(["Cylinder", "Profile Dome"])
        self.mode.currentTextChanged.connect(self._on_mode_changed)

        self.length = self._double_spin(config.length_mm, 1.0, 10000.0, 10.0)
        self.radius = self._double_spin(config.radius_mm, 1.0, 2000.0, 1.0)
        self.tow_width = self._double_spin(config.tow_width_mm, 0.0, 200.0, 0.5)
        self.angle = self._double_spin(config.winding_angle_deg, 1.0, 89.0, 1.0)
        self.points = self._int_spin(config.points_per_pass, 2, 10000, 10)
        self.passes = self._int_spin(config.passes, 1, 200, 1)
        self.target_coverage = self._double_spin(100.0, 1.0, 300.0, 5.0)
        self.max_opt_passes = self._int_spin(max(200, config.passes), 1, 1000, 10)
        self.auto_phase = qt_widgets.QCheckBox()
        self.auto_phase.setChecked(config.phase_offset_deg is None)
        self.phase_offset = self._double_spin(
            _display_phase_offset(config),
            0.0,
            360.0,
            1.0,
        )
        self.phase_offset.setEnabled(config.phase_offset_deg is not None)
        self.clearance = self._double_spin(config.radial_clearance_mm, 0.0, 2000.0, 1.0)
        self.alternate = qt_widgets.QCheckBox()
        self.alternate.setChecked(config.alternate_direction)
        self.passes.valueChanged.connect(self._update_auto_phase_value)
        self.auto_phase.toggled.connect(self._on_auto_phase_toggled)

        mode_group = qt_widgets.QGroupBox("Preview Mode")
        mode_layout = qt_widgets.QFormLayout(mode_group)
        mode_layout.addRow("Mode", self.mode)

        project_group = qt_widgets.QGroupBox("Project")
        project_layout = qt_widgets.QVBoxLayout(project_group)
        self.project_name = qt_widgets.QLineEdit("Cylinder winding")
        self.project_path_label = qt_widgets.QLabel("Unsaved project")
        self.project_path_label.setWordWrap(True)
        project_buttons = qt_widgets.QGridLayout()
        load_project_button = qt_widgets.QPushButton("Import Project")
        self._connect_safe_button(load_project_button, "Open Project", self._load_project_dialog)
        save_project_button = qt_widgets.QPushButton("Export Project")
        self._connect_safe_button(save_project_button, "Save Project", self._save_project_dialog)
        save_as_project_button = qt_widgets.QPushButton("Export As")
        self._connect_safe_button(
            save_as_project_button,
            "Save Project As",
            lambda: self._save_project_dialog(force_dialog=True),
        )
        project_buttons.addWidget(load_project_button, 0, 0)
        project_buttons.addWidget(save_project_button, 0, 1)
        project_buttons.addWidget(save_as_project_button, 1, 0, 1, 2)
        project_layout.addWidget(self.project_name)
        project_layout.addLayout(project_buttons)
        project_layout.addWidget(self.project_path_label)

        winding_group = qt_widgets.QGroupBox("Cylinder Winding")
        winding_layout = qt_widgets.QVBoxLayout(winding_group)
        form.addRow("Length mm", self.length)
        form.addRow("Radius mm", self.radius)
        form.addRow("Tow width mm", self.tow_width)
        form.addRow("Angle deg", self.angle)
        form.addRow("Points/pass", self.points)
        form.addRow("Passes", self.passes)
        form.addRow("Target coverage %", self.target_coverage)
        form.addRow("Max opt passes", self.max_opt_passes)
        form.addRow("Auto phase", self.auto_phase)
        form.addRow("Phase offset deg", self.phase_offset)
        form.addRow("Clearance mm", self.clearance)
        form.addRow("Alternate direction", self.alternate)
        winding_layout.addLayout(form)

        layer_group = qt_widgets.QGroupBox("Layer Stack")
        layer_layout = qt_widgets.QVBoxLayout(layer_group)
        layer_header = qt_widgets.QHBoxLayout()
        self.use_layer_stack = qt_widgets.QCheckBox("Use layer stack")
        self.use_layer_stack.setChecked(True)
        self.use_layer_stack.toggled.connect(
            lambda _checked: self.run_safe_action("Toggle Layer Stack", self._render_scene)
        )
        add_layer = qt_widgets.QPushButton("+ Helical")
        self._connect_safe_button(
            add_layer,
            "Add Helical Layer",
            lambda: self._add_layer_preset("helical"),
        )
        add_hoop = qt_widgets.QPushButton("+ Hoop")
        self._connect_safe_button(
            add_hoop,
            "Add Hoop Layer",
            lambda: self._add_layer_preset("hoop"),
        )
        add_polar = qt_widgets.QPushButton("+ Polar")
        self._connect_safe_button(
            add_polar,
            "Add Polar Layer",
            lambda: self._add_layer_preset("polar"),
        )
        remove_layer = qt_widgets.QPushButton("Remove")
        self._connect_safe_button(remove_layer, "Delete Layer", self._remove_selected_layer_rows)
        move_up = qt_widgets.QPushButton("Up")
        self._connect_safe_button(move_up, "Move Layer Up", lambda: self._move_selected_layer(-1))
        move_down = qt_widgets.QPushButton("Down")
        self._connect_safe_button(
            move_down,
            "Move Layer Down",
            lambda: self._move_selected_layer(1),
        )
        layer_header.addWidget(self.use_layer_stack, 1)
        layer_header.addWidget(add_layer)
        layer_header.addWidget(add_hoop)
        layer_header.addWidget(add_polar)
        layer_header.addWidget(remove_layer)
        layer_header.addWidget(move_up)
        layer_header.addWidget(move_down)
        layer_layout.addLayout(layer_header)
        self.layer_table = qt_widgets.QTableWidget(0, 14)
        self.layer_table.setHorizontalHeaderLabels(
            [
                "Enabled",
                "Layer Name",
                "Type",
                "Angle",
                "Direction",
                "Passes",
                "Tow Width",
                "Thickness",
                "Feedrate",
                "Clearance",
                "Colour",
                "Coverage %",
                "Points",
                "Transition pts",
            ]
        )
        self.layer_table.horizontalHeader().setStretchLastSection(True)
        self.layer_table.verticalHeader().setVisible(False)
        self.layer_table.setMinimumHeight(260)
        self.layer_table.setSelectionBehavior(self.layer_table.SelectionBehavior.SelectRows)
        self.layer_table.setSelectionMode(self.layer_table.SelectionMode.ExtendedSelection)
        self.layer_table.itemChanged.connect(self._on_layer_table_changed)
        layer_layout.addWidget(self.layer_table)

        profile_group = qt_widgets.QGroupBox("Axisymmetric Profile")
        profile_layout = qt_widgets.QVBoxLayout(profile_group)
        profile_form = qt_widgets.QFormLayout()
        self.profile_path = qt_widgets.QLineEdit(str(self._profile_config.profile_path))
        self.profile_path_mode = qt_widgets.QComboBox()
        self.profile_path_mode.addItems(
            ["Dome (geodesic)", "Nosecone", "Axisymmetric"]
        )
        self.profile_path_mode.setCurrentText(
            _profile_path_mode_label(self._profile_config.path_mode)
        )
        self.profile_samples = self._int_spin(self._profile_config.samples or 0, 0, 10000, 10)
        self.profile_min_radius = self._double_spin(
            self._profile_config.min_radius_mm,
            0.001,
            2000.0,
            0.5,
        )
        self.turnaround_radius = self._double_spin(
            0.0
            if self._profile_config.turnaround_radius_mm is None
            else self._profile_config.turnaround_radius_mm,
            0.0,
            2000.0,
            0.5,
        )
        self.turnaround_points = self._int_spin(
            self._profile_config.turnaround_points,
            2,
            10000,
            1,
        )
        self.turnaround_angle = self._double_spin(
            self._profile_config.turnaround_angle_deg,
            1.0,
            720.0,
            5.0,
        )
        self.circuits = self._int_spin(self._profile_config.circuits, 1, 200, 1)
        profile_form.addRow(
            "DXF profile",
            self._open_path_selector(
                self.profile_path,
                "Choose DXF Z-R profile",
                "DXF Files (*.dxf);;All Files (*)",
            ),
        )
        profile_form.addRow("Path type", self.profile_path_mode)
        profile_form.addRow("Samples (0 auto)", self.profile_samples)
        profile_form.addRow("Min radius mm", self.profile_min_radius)
        profile_form.addRow("Turn radius (0 auto)", self.turnaround_radius)
        profile_form.addRow("Turn points", self.turnaround_points)
        profile_form.addRow("Turn angle deg", self.turnaround_angle)
        profile_form.addRow("Circuits", self.circuits)
        profile_layout.addLayout(profile_form)
        inspect_profile = qt_widgets.QPushButton("Inspect DXF Import")
        self._connect_safe_button(inspect_profile, "Import DXF", self._inspect_profile_import)
        profile_layout.addWidget(inspect_profile)

        pattern_group = qt_widgets.QGroupBox("Pattern Planner")
        pattern_layout = qt_widgets.QFormLayout(pattern_group)
        self.use_pattern_planner = qt_widgets.QCheckBox()
        self.use_pattern_planner.setChecked(False)
        self.pattern_coverage = self._double_spin(100.0, 1.0, 300.0, 5.0)
        self.include_hoop_layer = qt_widgets.QCheckBox()
        self.include_hoop_layer.setChecked(False)
        self.include_hoop_layer.setToolTip(
            "Applies to cylinder schedules; ignored for DXF dome schedules."
        )
        self.balanced_pm_layers = qt_widgets.QCheckBox()
        self.balanced_pm_layers.setChecked(True)
        self.pattern_max_angle_error = self._double_spin(5.0, 0.0, 20.0, 0.5)
        pattern_layout.addRow("Use full pattern", self.use_pattern_planner)
        pattern_layout.addRow("Layer coverage %", self.pattern_coverage)
        pattern_layout.addRow("Balanced +/-", self.balanced_pm_layers)
        pattern_layout.addRow("Include hoop", self.include_hoop_layer)
        pattern_layout.addRow("Max angle err deg", self.pattern_max_angle_error)
        self._populate_default_layers()

        export_group = qt_widgets.QGroupBox("Export")
        export_layout = qt_widgets.QVBoxLayout(export_group)
        export_form = qt_widgets.QFormLayout()
        self.feedrate = self._double_spin(500.0, 1.0, 100000.0, 50.0)
        self.csv_output = qt_widgets.QLineEdit(self._default_export_paths.csv_path)
        self.gcode_output = qt_widgets.QLineEdit(self._default_export_paths.gcode_path)
        self.coverage_output = qt_widgets.QLineEdit(
            self._default_export_paths.coverage_csv_path
        )
        self.coverage_summary_output = qt_widgets.QLineEdit(
            self._default_export_paths.coverage_summary_csv_path
        )
        self.obj_output = qt_widgets.QLineEdit(self._default_export_paths.preview_obj_path)
        export_form.addRow("Feed mm/min", self.feedrate)
        export_form.addRow(
            "CSV",
            self._path_selector(
                self.csv_output,
                "Choose CSV output",
                "CSV Files (*.csv);;All Files (*)",
            ),
        )
        export_form.addRow(
            "G-code",
            self._path_selector(
                self.gcode_output,
                "Choose G-code output",
                "G-code Files (*.gcode *.nc *.tap);;All Files (*)",
            ),
        )
        export_form.addRow(
            "Coverage",
            self._path_selector(
                self.coverage_output,
                "Choose coverage CSV output",
                "CSV Files (*.csv);;All Files (*)",
            ),
        )
        export_form.addRow(
            "Summary",
            self._path_selector(
                self.coverage_summary_output,
                "Choose coverage summary CSV output",
                "CSV Files (*.csv);;All Files (*)",
            ),
        )
        export_form.addRow(
            "OBJ",
            self._path_selector(
                self.obj_output,
                "Choose OBJ preview output",
                "OBJ Files (*.obj);;All Files (*)",
            ),
        )
        export_buttons = qt_widgets.QGridLayout()
        export_folder = qt_widgets.QPushButton("Set Export Folder")
        self._connect_safe_button(export_folder, "Choose Export Folder", self._choose_export_folder)
        export_csv = qt_widgets.QPushButton("Export CSV")
        self._connect_safe_button(export_csv, "Export CSV", lambda: self._export_current(csv=True))
        export_gcode = qt_widgets.QPushButton("Export G-code")
        self._connect_safe_button(
            export_gcode,
            "Export G-code",
            lambda: self._export_current(gcode=True),
        )
        export_coverage = qt_widgets.QPushButton("Export Coverage")
        self._connect_safe_button(
            export_coverage,
            "Export Coverage",
            lambda: self._export_current(coverage_csv=True, coverage_summary_csv=True),
        )
        export_obj = qt_widgets.QPushButton("Export OBJ")
        self._connect_safe_button(
            export_obj,
            "Export OBJ",
            lambda: self._export_current(preview_obj=True),
        )
        export_all = qt_widgets.QPushButton("Export All")
        self._connect_safe_button(
            export_all,
            "Export All",
            lambda: self._export_current(
                csv=True,
                gcode=True,
                coverage_csv=True,
                coverage_summary_csv=True,
                preview_obj=True,
            ),
        )
        export_buttons.addWidget(export_folder, 0, 0, 1, 2)
        export_buttons.addWidget(export_csv, 1, 0)
        export_buttons.addWidget(export_gcode, 1, 1)
        export_buttons.addWidget(export_coverage, 2, 0)
        export_buttons.addWidget(export_obj, 2, 1)
        export_buttons.addWidget(export_all, 3, 0, 1, 2)
        export_layout.addLayout(export_form)
        export_layout.addLayout(export_buttons)

        nodes_tab = self._build_node_workspace()
        tabs.addTab(self._scroll_tab(mode_group, project_group), "Setup")
        tabs.addTab(nodes_tab, "Nodes")
        tabs.addTab(self._scroll_tab(winding_group, layer_group, profile_group), "Path")
        tabs.addTab(self._scroll_tab(pattern_group), "Pattern")
        tabs.addTab(self._scroll_tab(export_group), "Export")
        tabs.setCurrentWidget(nodes_tab)

        layout.addWidget(tabs, 1)
        return panel

    def _build_node_workspace(self) -> Any:
        qt_widgets = self._qt_widgets
        container = qt_widgets.QWidget()
        container.setObjectName("nodeWorkspace")
        container_layout = qt_widgets.QVBoxLayout(container)
        container_layout.setContentsMargins(8, 8, 8, 8)
        container_layout.setSpacing(8)

        toolbar = qt_widgets.QHBoxLayout()
        toolbar.setSpacing(8)
        link_nodes = qt_widgets.QPushButton("Link")
        self._connect_safe_button(link_nodes, "Link", self._link_selected_nodes)
        unlink_nodes = qt_widgets.QPushButton("Unlink")
        self._connect_safe_button(unlink_nodes, "Unlink", self._unlink_selected_nodes)
        duplicate_nodes = qt_widgets.QPushButton("Duplicate")
        self._connect_safe_button(duplicate_nodes, "Duplicate Node", self._duplicate_selected_nodes)
        delete_nodes = qt_widgets.QPushButton("Delete")
        self._connect_safe_button(delete_nodes, "Delete Node", self._delete_selected_graph_items)
        collapse_nodes = qt_widgets.QPushButton("Collapse")
        self._connect_safe_button(
            collapse_nodes,
            "Collapse Node",
            lambda: self._set_selected_nodes_collapsed(True),
        )
        expand_nodes = qt_widgets.QPushButton("Expand")
        self._connect_safe_button(
            expand_nodes,
            "Expand Node",
            lambda: self._set_selected_nodes_collapsed(False),
        )
        group_nodes = qt_widgets.QPushButton("Group")
        self._connect_safe_button(group_nodes, "Group Nodes", self._group_selected_nodes)
        fit_nodes = qt_widgets.QPushButton("Fit")
        self._connect_safe_button(fit_nodes, "Fit Node Graph", self._fit_node_graph)
        run_selected = qt_widgets.QPushButton("Run Selected")
        self._connect_safe_button(
            run_selected,
            "Run Selected",
            lambda: self._execute_node_graph(execute_exports=False),
        )
        run_branch = qt_widgets.QPushButton("Run Branch")
        self._connect_safe_button(
            run_branch,
            "Run Branch",
            lambda: self._execute_node_graph(execute_exports=False),
        )
        run_graph = qt_widgets.QPushButton("Run Graph")
        self._connect_safe_button(
            run_graph,
            "Run Graph",
            lambda: self._execute_node_graph(execute_exports=False),
        )
        export_graph = qt_widgets.QPushButton("Run + Export")
        self._connect_safe_button(
            export_graph,
            "Run + Export",
            lambda: self._execute_node_graph(execute_exports=True),
        )
        optimize_pattern = qt_widgets.QPushButton("Optimize Pattern")
        self._connect_safe_button(optimize_pattern, "Optimise Pattern", self._optimize_pattern)
        toolbar.addWidget(link_nodes)
        toolbar.addWidget(unlink_nodes)
        toolbar.addWidget(duplicate_nodes)
        toolbar.addWidget(delete_nodes)
        toolbar.addWidget(collapse_nodes)
        toolbar.addWidget(expand_nodes)
        toolbar.addWidget(group_nodes)
        toolbar.addWidget(fit_nodes)
        toolbar.addStretch(1)
        toolbar.addWidget(run_selected)
        toolbar.addWidget(run_branch)
        toolbar.addWidget(run_graph)
        toolbar.addWidget(export_graph)
        toolbar.addWidget(optimize_pattern)
        container_layout.addLayout(toolbar)

        workspace_splitter = qt_widgets.QSplitter(self._qt_core.Qt.Orientation.Horizontal)
        workspace_splitter.setChildrenCollapsible(False)

        graph_panel = qt_widgets.QWidget()
        graph_layout = qt_widgets.QVBoxLayout(graph_panel)
        graph_layout.setContentsMargins(0, 0, 0, 0)
        graph_layout.setSpacing(0)

        self.node_scene = qt_widgets.QGraphicsScene()
        self.node_scene.setSceneRect(0.0, 0.0, 2200.0, 900.0)
        self.node_scene.selectionChanged.connect(self._on_node_selection_changed)
        self.node_scene.changed.connect(lambda _regions: self._refresh_node_links())
        self.node_view = qt_widgets.QGraphicsView(self.node_scene)
        self.node_view.setObjectName("nodeView")
        self.node_view.setMinimumHeight(240)
        self.node_view.setMinimumWidth(680)
        self.node_view.setRenderHint(self._qt_gui.QPainter.RenderHint.Antialiasing, True)
        self.node_view.setDragMode(qt_widgets.QGraphicsView.DragMode.RubberBandDrag)
        self.node_view.setTransformationAnchor(
            qt_widgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse
        )
        self.node_view.setContextMenuPolicy(
            self._qt_core.Qt.ContextMenuPolicy.CustomContextMenu
        )
        self.node_view.customContextMenuRequested.connect(self._show_node_context_menu)
        self._install_node_canvas_event_filter()
        self._install_node_shortcuts()
        graph_layout.addWidget(self.node_view, 1)

        bottom_tabs = qt_widgets.QTabWidget()
        bottom_tabs.setObjectName("nodeBottomTabs")
        bottom_tabs.setMinimumWidth(330)

        library_panel = qt_widgets.QWidget()
        library_panel.setObjectName("nodeBottomPanel")
        library_layout = qt_widgets.QVBoxLayout(library_panel)
        library_layout.setContentsMargins(8, 8, 8, 8)
        library_layout.setSpacing(8)
        self.node_search = qt_widgets.QLineEdit()
        self.node_search.setPlaceholderText("Search nodes")
        self.node_search.textChanged.connect(self._refresh_node_library)
        self.node_category_filter = qt_widgets.QComboBox()
        self.node_category_filter.currentTextChanged.connect(self._refresh_node_library)
        self.node_library = qt_widgets.QListWidget()
        self.node_library.itemDoubleClicked.connect(
            lambda _item: self.run_safe_action("Add Node", self._add_selected_node_type)
        )
        add_node = qt_widgets.QPushButton("Add Node")
        self._connect_safe_button(add_node, "Add Node", self._add_selected_node_type)
        library_tools = qt_widgets.QHBoxLayout()
        library_tools.addWidget(self.node_search, 2)
        library_tools.addWidget(self.node_category_filter, 1)
        library_tools.addWidget(add_node)
        library_layout.addLayout(library_tools)
        library_layout.addWidget(self.node_library, 1)

        inspector_panel = qt_widgets.QWidget()
        inspector_panel.setObjectName("nodeBottomPanel")
        inspector_layout = qt_widgets.QVBoxLayout(inspector_panel)
        inspector_layout.setContentsMargins(8, 8, 8, 8)
        inspector_layout.setSpacing(8)
        self.node_inspector_name = qt_widgets.QLineEdit()
        self.node_inspector_name.editingFinished.connect(self._apply_node_inspector)
        self.node_inspector_type = qt_widgets.QLabel("No node selected")
        self.node_inspector_status = qt_widgets.QLabel("")
        self.node_inspector_form_widget = qt_widgets.QWidget()
        self.node_inspector_form = qt_widgets.QFormLayout(self.node_inspector_form_widget)
        self.node_inspector_form.setContentsMargins(0, 0, 0, 0)
        self.node_inspector_form.setSpacing(8)
        inspector_scroll = qt_widgets.QScrollArea()
        inspector_scroll.setWidgetResizable(True)
        inspector_scroll.setObjectName("nodeInspectorScroll")
        inspector_scroll.setWidget(self.node_inspector_form_widget)
        self.node_inspector_settings = qt_widgets.QPlainTextEdit()
        self.node_inspector_settings.setMinimumHeight(90)
        self.node_inspector_settings.setMaximumHeight(150)
        self.node_inspector_settings.setPlaceholderText("Advanced settings JSON")
        apply_settings = qt_widgets.QPushButton("Apply Settings")
        self._connect_safe_button(apply_settings, "Apply Node Settings", self._apply_node_inspector)
        inspector_layout.addWidget(self.node_inspector_type)
        inspector_layout.addWidget(self.node_inspector_name)
        inspector_layout.addWidget(self.node_inspector_status)
        inspector_layout.addWidget(inspector_scroll, 1)
        inspector_layout.addWidget(self.node_inspector_settings)
        inspector_layout.addWidget(apply_settings)

        status_panel = qt_widgets.QWidget()
        status_panel.setObjectName("nodeBottomPanel")
        status_layout = qt_widgets.QVBoxLayout(status_panel)
        status_layout.setContentsMargins(8, 8, 8, 8)
        status_layout.setSpacing(8)
        self.node_status = qt_widgets.QLabel("Graph ready")
        self.node_status.setWordWrap(True)
        self.node_status_log = qt_widgets.QPlainTextEdit()
        self.node_status_log.setReadOnly(True)
        status_layout.addWidget(self.node_status)
        status_layout.addWidget(self.node_status_log, 1)

        debug_panel = qt_widgets.QWidget()
        debug_panel.setObjectName("nodeBottomPanel")
        debug_layout = qt_widgets.QVBoxLayout(debug_panel)
        debug_layout.setContentsMargins(8, 8, 8, 8)
        debug_layout.setSpacing(8)
        self.node_debug_log = qt_widgets.QPlainTextEdit()
        self.node_debug_log.setReadOnly(True)
        self.node_debug_log.setPlaceholderText(f"Debug log: {GUI_LOG_PATH}")
        debug_layout.addWidget(self.node_debug_log, 1)

        bottom_tabs.addTab(library_panel, "Node Library")
        bottom_tabs.addTab(inspector_panel, "Inspector")
        bottom_tabs.addTab(status_panel, "Execution / Status")
        bottom_tabs.addTab(debug_panel, "Debug Log")

        workspace_splitter.addWidget(graph_panel)
        workspace_splitter.addWidget(bottom_tabs)
        workspace_splitter.setStretchFactor(0, 3)
        workspace_splitter.setStretchFactor(1, 1)
        workspace_splitter.setSizes([1050, 360])
        container_layout.addWidget(workspace_splitter, 1)
        self._log_graph_event("Workflow tab opened")
        self._refresh_node_library()
        self._redraw_node_graph()
        self._schedule_fit_node_graph()
        return container

    def _fit_node_graph(self) -> None:
        bounds = self.node_scene.itemsBoundingRect().adjusted(-40.0, -40.0, 40.0, 40.0)
        if bounds.isNull() or bounds.isEmpty():
            return
        self.node_view.fitInView(bounds, self._qt_core.Qt.AspectRatioMode.KeepAspectRatio)

    def _schedule_fit_node_graph(self) -> None:
        if hasattr(self, "node_view"):
            self._qt_core.QTimer.singleShot(0, self._fit_node_graph)

    def _scale_node_graph(self, factor: float) -> None:
        self.node_view.scale(factor, factor)

    def _graph_controller(self) -> NodeGraphController:
        return NodeGraphController(self._node_graph, self._node_registry)

    def _set_node_status(self, message: str) -> None:
        if hasattr(self, "node_status"):
            self.node_status.setText(message)
        if hasattr(self, "node_status_log"):
            self.node_status_log.appendPlainText(message)

    def _log_graph_event(self, message: str) -> None:
        self._logger.info(message)
        if hasattr(self, "node_debug_log"):
            self.node_debug_log.appendPlainText(message)

    def _show_graph_error(self, message: str, exc: BaseException | None = None) -> None:
        detail = str(exc) if exc is not None else ""
        full_message = f"{message}: {detail}" if detail else message
        if exc is not None:
            self._logger.error("%s\n%s", full_message, traceback.format_exc())
        else:
            self._logger.error(full_message)
        if hasattr(self, "node_status"):
            self.node_status.setText(full_message)
        if hasattr(self, "node_status_log"):
            self.node_status_log.appendPlainText(full_message)
        if hasattr(self, "node_debug_log"):
            self.node_debug_log.appendPlainText(full_message)
            if exc is not None:
                self.node_debug_log.appendPlainText(traceback.format_exc())

    def _safe_add_node(self, type_id: str, position: Any | None = None) -> str | None:
        before_node_ids = set(self._node_graph.nodes)
        try:
            if type_id not in self._node_registry:
                raise ValueError(f"unknown node type: {type_id}")
            if position is None:
                position = self._next_node_add_position()
            else:
                position = self._bounded_node_position(position)
            x_pos = float(position.x())
            y_pos = float(position.y())
            self._log_graph_event(f"Node creation started: {type_id}")
            result = self._graph_controller().add_node(type_id, x=x_pos, y=y_pos)
            if not result.success or result.node_id is None:
                raise ValueError(result.error or "node creation failed")
            self._redraw_node_graph()
            self._select_node_item(result.node_id)
            self._set_node_status(f"Node created: {self._node_graph.nodes[result.node_id].name}")
            self._log_graph_event(f"Node creation succeeded: {type_id}")
            return result.node_id
        except Exception as exc:  # noqa: BLE001 - protects Qt event loop
            for node_id in set(self._node_graph.nodes) - before_node_ids:
                self._node_graph.delete_nodes((node_id,))
            self._show_graph_error(f"Failed to add node {type_id}", exc)
            return None

    def _next_node_add_position(self) -> Any:
        selected = self._selected_node_ids() if hasattr(self, "node_scene") else ()
        if selected:
            node = self._node_graph.nodes[selected[-1]]
            return self._bounded_node_position(
                self._qt_core.QPointF(node.x + 360.0, node.y)
            )
        if hasattr(self, "node_view"):
            return self._bounded_node_position(
                self.node_view.mapToScene(self.node_view.viewport().rect().center())
            )
        return self._qt_core.QPointF(120.0, 120.0)

    def _bounded_node_position(self, position: Any) -> Any:
        scene_rect = self.node_scene.sceneRect() if hasattr(self, "node_scene") else None
        if scene_rect is None:
            return position
        max_x = max(scene_rect.left(), scene_rect.right() - 320.0)
        max_y = max(scene_rect.top(), scene_rect.bottom() - 180.0)
        x_pos = min(max(float(position.x()), scene_rect.left() + 20.0), max_x)
        y_pos = min(max(float(position.y()), scene_rect.top() + 20.0), max_y)
        return self._qt_core.QPointF(x_pos, y_pos)

    def _install_node_canvas_event_filter(self) -> None:
        qt_core = self._qt_core

        class _NodeCanvasEventFilter(qt_core.QObject):  # type: ignore[name-defined, misc, valid-type]
            def __init__(self, owner: _PreviewWindow) -> None:
                super().__init__()
                self._owner = owner

            def eventFilter(self, obj: Any, event: Any) -> bool:  # noqa: N802
                return bool(self._owner._node_canvas_event_filter(obj, event))

        self._node_canvas_event_filter_object = _NodeCanvasEventFilter(self)
        self.node_view.viewport().installEventFilter(self._node_canvas_event_filter_object)

    def _install_node_shortcuts(self) -> None:
        shortcuts = (
            ("Delete", self._delete_selected_graph_items),
            ("Ctrl+D", self._duplicate_selected_nodes),
            ("Ctrl+G", self._group_selected_nodes),
            ("Ctrl+Shift+G", self._ungroup_selected_nodes),
            ("F", self._frame_selected_nodes),
            ("Ctrl+F", lambda: self.node_search.setFocus()),
        )
        self._node_shortcuts = []
        for key_sequence, callback in shortcuts:
            shortcut = self._qt_gui.QShortcut(
                self._qt_gui.QKeySequence(key_sequence),
                self.node_view,
            )
            shortcut.activated.connect(callback)
            self._node_shortcuts.append(shortcut)

    def _node_canvas_event_filter(self, _obj: Any, event: Any) -> bool:
        event_type = event.type()
        if (
            event_type == self._qt_core.QEvent.Type.MouseButtonPress
            and event.button() == self._qt_core.Qt.MouseButton.LeftButton
        ):
            return self._begin_socket_drag(event)
        if (
            event_type == self._qt_core.QEvent.Type.MouseMove
            and self._node_socket_drag is not None
        ):
            self._update_socket_drag(event)
            return True
        if (
            event_type == self._qt_core.QEvent.Type.MouseButtonRelease
            and self._node_socket_drag is not None
        ):
            self._finish_socket_drag(event)
            return True
        return False

    def _event_view_pos(self, event: Any) -> Any:
        if hasattr(event, "position"):
            return event.position().toPoint()
        return event.pos()

    def _begin_socket_drag(self, event: Any) -> bool:
        view_pos = self._event_view_pos(event)
        item = self._socket_item_at_view_pos(view_pos)
        if item is None or item.data(4) != "output":
            return False
        node_id = str(item.data(2))
        socket_name = str(item.data(3))
        self._node_socket_drag = {
            "source_node_id": node_id,
            "source_socket": socket_name,
        }
        start = item.sceneBoundingRect().center()
        path = self._node_link_path(start, start)
        pen = self._qt_gui.QPen(self._qt_gui.QColor("#8fb8dc"), 2.0)
        pen.setStyle(self._qt_core.Qt.PenStyle.DashLine)
        self._node_temp_link_item = self.node_scene.addPath(path, pen)
        self._node_temp_link_item.setZValue(100.0)
        self._log_graph_event(f"Link creation started: {node_id}.{socket_name}")
        event.accept()
        return True

    def _update_socket_drag(self, event: Any) -> None:
        if self._node_socket_drag is None or self._node_temp_link_item is None:
            return
        view_pos = self._event_view_pos(event)
        scene_pos = self.node_view.mapToScene(view_pos)
        source_item = self._socket_items.get(
            (
                self._node_socket_drag["source_node_id"],
                self._node_socket_drag["source_socket"],
                "output",
            )
        )
        if source_item is None:
            return
        self._node_temp_link_item.setPath(
            self._node_link_path(source_item.sceneBoundingRect().center(), scene_pos)
        )
        self._highlight_socket_target(self._socket_item_at_view_pos(view_pos))

    def _finish_socket_drag(self, event: Any) -> None:
        if self._node_socket_drag is None:
            return
        view_pos = self._event_view_pos(event)
        target_item = self._socket_item_at_view_pos(view_pos)
        if self._node_temp_link_item is not None:
            self.node_scene.removeItem(self._node_temp_link_item)
        self._node_temp_link_item = None
        self._reset_highlighted_socket()
        try:
            if target_item is None or target_item.data(4) != "input":
                raise ValueError("drop link on a compatible input socket")
            self._node_graph.add_link(
                self._node_socket_drag["source_node_id"],
                self._node_socket_drag["source_socket"],
                str(target_item.data(2)),
                str(target_item.data(3)),
                self._node_registry,
            )
        except ValueError as exc:
            self._show_graph_error("Link creation failed", exc)
        else:
            self._refresh_node_links()
            self._set_node_status("Link created")
            self._log_graph_event("Link creation succeeded")
        self._node_socket_drag = None
        event.accept()

    def _socket_item_at_view_pos(self, view_pos: Any) -> Any | None:
        item = self.node_view.itemAt(view_pos)
        while item is not None:
            if item.data(1) == "socket":
                return item
            item = item.parentItem()
        return None

    def _highlight_socket_target(self, item: Any | None) -> None:
        if item is self._node_highlight_socket:
            return
        self._reset_highlighted_socket()
        if item is None or item.data(4) != "input" or self._node_socket_drag is None:
            return
        self._node_highlight_socket = item
        try:
            self._node_graph.validate_link(
                self._node_socket_drag["source_node_id"],
                self._node_socket_drag["source_socket"],
                str(item.data(2)),
                str(item.data(3)),
                self._node_registry,
            )
        except ValueError:
            color = "#d65757"
        else:
            color = "#51a36f"
        item.setBrush(self._qt_gui.QBrush(self._qt_gui.QColor(color)))

    def _reset_highlighted_socket(self) -> None:
        if self._node_highlight_socket is not None:
            item = self._node_highlight_socket
            item.setBrush(
                self._qt_gui.QBrush(
                    self._qt_gui.QColor(_socket_kind_color(str(item.data(5))))
                )
            )
        self._node_highlight_socket = None

    def _node_link_path(self, start: Any, end: Any) -> Any:
        path = self._qt_gui.QPainterPath(start)
        direction = 1.0 if end.x() >= start.x() else -1.0
        lead = max(54.0, min(130.0, abs(end.x() - start.x()) * 0.25))
        start_lead = self._qt_core.QPointF(start.x() + direction * lead, start.y())
        end_lead = self._qt_core.QPointF(end.x() - direction * lead, end.y())
        path.lineTo(start_lead)
        dx = max(90.0, abs(end_lead.x() - start_lead.x()) * 0.5)
        path.cubicTo(
            start_lead.x() + direction * dx,
            start_lead.y(),
            end_lead.x() - direction * dx,
            end_lead.y(),
            end_lead.x(),
            end_lead.y(),
        )
        path.lineTo(end)
        return path

    def _refresh_node_library(self) -> None:
        query = self.node_search.text().strip().lower()
        selected_category = (
            self.node_category_filter.currentText()
            if hasattr(self, "node_category_filter")
            else "All"
        )
        categories = ["All"]
        for type_id in addable_node_type_ids(self._node_registry):
            category = self._node_registry[type_id].category
            if category not in categories:
                categories.append(category)
        if hasattr(self, "node_category_filter"):
            current_category = self.node_category_filter.currentText() or "All"
            self.node_category_filter.blockSignals(True)
            self.node_category_filter.clear()
            self.node_category_filter.addItems(categories)
            self.node_category_filter.setCurrentText(
                current_category if current_category in categories else "All"
            )
            self.node_category_filter.blockSignals(False)
            selected_category = self.node_category_filter.currentText()
        self.node_library.clear()
        for type_id in sorted(
            addable_node_type_ids(self._node_registry),
            key=lambda node_type_id: (
                self._node_registry[node_type_id].category,
                self._node_registry[node_type_id].label,
            ),
        ):
            definition = self._node_registry[type_id]
            if selected_category != "All" and definition.category != selected_category:
                continue
            searchable = f"{definition.category} {definition.label} {type_id}".lower()
            if query and query not in searchable:
                continue
            item = self._qt_widgets.QListWidgetItem(
                f"{definition.category} / {definition.label}"
            )
            item.setData(self._qt_core.Qt.ItemDataRole.UserRole, type_id)
            self.node_library.addItem(item)

    def _add_selected_node_type(self) -> None:
        item = self.node_library.currentItem()
        if item is None:
            return
        type_id = str(item.data(self._qt_core.Qt.ItemDataRole.UserRole))
        self._log_graph_event(f"Node type selected: {type_id}")
        self._safe_add_node(type_id)

    def _redraw_node_graph(self) -> None:
        self.node_scene.blockSignals(True)
        self.node_scene.clear()
        self._node_items.clear()
        self._socket_items.clear()
        self._node_link_items.clear()
        self._node_group_items.clear()
        self._draw_graph_groups()
        for node in self._node_graph.nodes.values():
            self._draw_graph_node(node)
        self.node_scene.blockSignals(False)
        self._refresh_node_links()
        self._on_node_selection_changed()

    def _draw_graph_groups(self) -> None:
        for group in self._node_graph.groups.values():
            rects = []
            for node_id in group.node_ids:
                node = self._node_graph.nodes.get(node_id)
                if node is None:
                    continue
                rects.append(
                    self._qt_core.QRectF(
                        node.x - 20.0,
                        node.y - 34.0,
                        node.width + 40.0,
                        node.height + 56.0,
                    )
                )
            if not rects:
                continue
            bounds = rects[0]
            for rect in rects[1:]:
                bounds = bounds.united(rect)
            item = self.node_scene.addRect(
                bounds,
                self._qt_gui.QPen(self._qt_gui.QColor(group.color), 1.5),
                self._qt_gui.QBrush(self._qt_gui.QColor(35, 48, 61, 80)),
            )
            item.setZValue(-10.0)
            item.setData(1, "group")
            item.setData(2, group.id)
            text = self.node_scene.addText(group.name)
            text.setDefaultTextColor(self._qt_gui.QColor("#9fb3c6"))
            text.setPos(bounds.x() + 8.0, bounds.y() + 4.0)
            text.setZValue(-9.0)
            text.setData(1, "group")
            text.setData(2, group.id)
            self._node_group_items.extend([item, text])

    def _draw_graph_node(self, node: NodeInstance) -> None:
        definition = self._node_registry[node.type_id]
        width = max(node.width, 280.0)
        height = 54.0 if node.collapsed else max(node.height, 150.0)
        node.width = width
        node.height = height
        rect = self.node_scene.addRect(
            0.0,
            0.0,
            width,
            height,
            self._qt_gui.QPen(self._qt_gui.QColor("#65788a"), 1.5),
            self._qt_gui.QBrush(self._qt_gui.QColor("#151c24")),
        )
        rect.setPos(node.x, node.y)
        rect.setZValue(10.0)
        rect.setFlag(self._qt_widgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        rect.setFlag(self._qt_widgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        rect.setData(0, node.id)
        rect.setData(1, "node")

        header = self.node_scene.addRect(
            0.0,
            0.0,
            width,
            32.0,
            self._qt_gui.QPen(self._qt_gui.QColor(definition.color), 1.0),
            self._qt_gui.QBrush(self._qt_gui.QColor(definition.color)),
        )
        header.setParentItem(rect)
        title = self.node_scene.addText(node.name)
        title.setDefaultTextColor(self._qt_gui.QColor("#f0f5fa"))
        title.setTextWidth(width - 48.0)
        title.setPos(10.0, 5.0)
        title.setParentItem(rect)

        status = self.node_scene.addEllipse(
            width - 24.0,
            9.0,
            12.0,
            12.0,
            self._qt_gui.QPen(self._qt_gui.QColor("#0b1015"), 1.0),
            self._qt_gui.QBrush(self._qt_gui.QColor(_node_status_color(node.status))),
        )
        status.setParentItem(rect)
        if not node.collapsed:
            details = self.node_scene.addText(
                self._node_summary_text(node, definition),
            )
            details.setDefaultTextColor(self._qt_gui.QColor("#b7c3cf"))
            details.setTextWidth(width - 28.0)
            details.setPos(14.0, 48.0)
            details.setParentItem(rect)
        self._draw_node_sockets(rect, node, definition, height)
        self._node_items[node.id] = rect

    def _draw_node_sockets(
        self,
        parent_item: Any,
        node: NodeInstance,
        definition: NodeTypeDefinition,
        height: float,
    ) -> None:
        self._draw_socket_side(
            parent_item,
            node,
            definition.inputs,
            side="input",
            x_pos=-7.0,
            height=height,
        )
        self._draw_socket_side(
            parent_item,
            node,
            definition.outputs,
            side="output",
            x_pos=node.width - 7.0,
            height=height,
        )

    def _draw_socket_side(
        self,
        parent_item: Any,
        node: NodeInstance,
        sockets: tuple[Any, ...],
        *,
        side: str,
        x_pos: float,
        height: float,
    ) -> None:
        if not sockets:
            return
        spacing = 26.0 if len(sockets) > 1 else 0.0
        socket_band_start = 58.0
        available_height = max(40.0, height - socket_band_start - 16.0)
        needed_height = spacing * (len(sockets) - 1)
        start_y = socket_band_start + max(0.0, (available_height - needed_height) / 2.0)
        for index, socket in enumerate(sockets):
            y_pos = start_y + index * spacing
            required_missing = (
                side == "input"
                and socket.required
                and not any(
                    link.target_node_id == node.id and link.target_socket == socket.name
                    for link in self._node_graph.links.values()
                )
            )
            color = "#c89438" if required_missing else _socket_kind_color(socket.kind)
            item = self.node_scene.addEllipse(
                x_pos,
                y_pos - 7.0,
                14.0,
                14.0,
                self._qt_gui.QPen(self._qt_gui.QColor("#0b1015"), 1.0),
                self._qt_gui.QBrush(self._qt_gui.QColor(color)),
            )
            item.setParentItem(parent_item)
            item.setZValue(40.0)
            item.setData(1, "socket")
            item.setData(2, node.id)
            item.setData(3, socket.name)
            item.setData(4, side)
            item.setData(5, socket.kind)
            item.setToolTip(f"{side}: {socket.name} ({socket.kind})")
            self._socket_items[(node.id, socket.name, side)] = item
            if not node.collapsed:
                label = self.node_scene.addText(socket.name)
                label.setDefaultTextColor(self._qt_gui.QColor("#8fa0ad"))
                label.setToolTip(socket.name)
                label.setParentItem(parent_item)
                if side == "input":
                    label.setTextWidth(max(82.0, node.width * 0.42))
                    label.setPos(16.0, y_pos - 11.0)
                else:
                    label.setTextWidth(max(82.0, node.width * 0.42))
                    label.setPos(node.width - max(126.0, node.width * 0.44), y_pos - 11.0)

    def _node_summary_text(
        self,
        node: NodeInstance,
        definition: NodeTypeDefinition,
    ) -> str:
        inputs = ", ".join(socket.name for socket in definition.inputs) or "no inputs"
        outputs = ", ".join(socket.name for socket in definition.outputs) or "no outputs"
        message = node.message or "Not computed"
        settings = ", ".join(
            f"{key}: {value}" for key, value in list(node.settings.items())[:2]
        )
        settings_line = f"\n{settings}" if settings else ""
        return (
            f"{definition.category}\nInputs: {inputs}\nOutputs: {outputs}\n"
            f"{message}{settings_line}"
        )

    def _refresh_node_links(self) -> None:
        if self._refreshing_node_links or not hasattr(self, "node_scene"):
            return
        self._refreshing_node_links = True
        for item in self._node_link_items:
            self.node_scene.removeItem(item)
        self._node_link_items.clear()
        for link in self._node_graph.links.values():
            source_item = self._socket_items.get(
                (link.source_node_id, link.source_socket, "output")
            )
            target_item = self._socket_items.get(
                (link.target_node_id, link.target_socket, "input")
            )
            if source_item is None or target_item is None:
                continue
            path = self._node_link_path(
                source_item.sceneBoundingRect().center(),
                target_item.sceneBoundingRect().center(),
            )
            pen = self._qt_gui.QPen(self._qt_gui.QColor("#6c8daa"), 2.0)
            item = self.node_scene.addPath(path, pen)
            item.setZValue(0.0)
            item.setFlag(self._qt_widgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
            item.setData(1, "link")
            item.setData(2, link.id)
            item.setToolTip(f"{link.source_socket} -> {link.target_socket}")
            self._node_link_items.append(item)
        self._sync_node_graph_from_scene()
        self._refreshing_node_links = False

    def _selected_node_ids(self) -> tuple[str, ...]:
        node_ids = []
        try:
            selected_items = self.node_scene.selectedItems()
        except RuntimeError:
            return ()
        for item in selected_items:
            try:
                node_id = item.data(0)
            except RuntimeError:
                continue
            if node_id in self._node_graph.nodes:
                node_ids.append(str(node_id))
        return tuple(dict.fromkeys(node_ids))

    def _selected_link_ids(self) -> tuple[str, ...]:
        link_ids = []
        try:
            selected_items = self.node_scene.selectedItems()
        except RuntimeError:
            return ()
        for item in selected_items:
            try:
                item_kind = item.data(1)
                link_id = item.data(2)
            except RuntimeError:
                continue
            if item_kind == "link" and link_id in self._node_graph.links:
                link_ids.append(str(link_id))
        return tuple(dict.fromkeys(link_ids))

    def _find_graph_item_at_view_pos(self, view_pos: Any, kind: str) -> Any | None:
        item = self.node_view.itemAt(view_pos)
        while item is not None:
            if item.data(1) == kind:
                return item
            item = item.parentItem()
        return None

    def _select_node_item(self, node_id: str) -> None:
        self.node_scene.clearSelection()
        item = self._node_items.get(node_id)
        if item is not None:
            item.setSelected(True)

    def _show_node_context_menu(self, view_pos: Any) -> None:
        scene_pos = self.node_view.mapToScene(view_pos)
        node_item = self._find_graph_item_at_view_pos(view_pos, "node")
        socket_item = self._find_graph_item_at_view_pos(view_pos, "socket")
        link_item = self._find_graph_item_at_view_pos(view_pos, "link")
        group_item = self._find_graph_item_at_view_pos(view_pos, "group")
        menu = self._qt_widgets.QMenu(self.node_view)
        if link_item is not None:
            link_id = str(link_item.data(2))
            delete_link = menu.addAction("Delete Link")
            inspect_type = menu.addAction("Inspect Data Type")
            action = menu.exec(self.node_view.viewport().mapToGlobal(view_pos))
            if action == delete_link:
                self._node_graph.links.pop(link_id, None)
                self._refresh_node_links()
            elif action == inspect_type and link_id in self._node_graph.links:
                link = self._node_graph.links[link_id]
                self.node_status.setText(
                    f"Link: {link.source_socket} -> {link.target_socket}"
                )
            return
        if socket_item is not None:
            node_id = str(socket_item.data(2))
            self._select_node_item(node_id)
            self.node_status.setText(
                f"Socket: {socket_item.data(4)} {socket_item.data(3)} "
                f"({socket_item.data(5)})"
            )
            return
        if group_item is not None:
            group_id = str(group_item.data(2))
            rename_group = menu.addAction("Rename Group")
            ungroup = menu.addAction("Ungroup")
            delete_group = menu.addAction("Delete Group Only")
            delete_group_nodes = menu.addAction("Delete Group and Nodes")
            action = menu.exec(self.node_view.viewport().mapToGlobal(view_pos))
            if action == rename_group:
                self._rename_group_dialog(group_id)
            elif action in (ungroup, delete_group):
                self._delete_group_only(group_id)
            elif action == delete_group_nodes:
                self._delete_group_and_nodes(group_id)
            return
        if node_item is not None:
            node_id = str(node_item.data(0))
            if node_id not in self._selected_node_ids():
                self._select_node_item(node_id)
            node = self._node_graph.nodes[node_id]
            rename_action = menu.addAction("Rename")
            duplicate_action = menu.addAction("Duplicate")
            delete_action = menu.addAction("Delete")
            collapse_action = menu.addAction("Expand" if node.collapsed else "Collapse")
            menu.addSeparator()
            run_action = menu.addAction("Run Node")
            downstream_action = menu.addAction("Run Downstream")
            group_action = menu.addAction("Create Group From Selection")
            action = menu.exec(self.node_view.viewport().mapToGlobal(view_pos))
            if action == rename_action:
                self._rename_selected_node_dialog()
            elif action == duplicate_action:
                self._duplicate_selected_nodes()
            elif action == delete_action:
                self._delete_selected_graph_items()
            elif action == collapse_action:
                self._set_selected_nodes_collapsed(not node.collapsed)
            elif action in {run_action, downstream_action}:
                self._execute_node_graph(execute_exports=False)
            elif action == group_action:
                self._group_selected_nodes()
            return
        add_menu = menu.addMenu("Add Node")
        actions: dict[Any, str] = {}
        for type_id, definition in sorted(
            self._node_registry.items(),
            key=lambda item: (item[1].category, item[1].label),
        ):
            action = add_menu.addAction(f"{definition.category} / {definition.label}")
            actions[action] = type_id
        menu.addSeparator()
        frame_all = menu.addAction("Frame All")
        auto_layout = menu.addAction("Auto Layout")
        group_action = menu.addAction("Create Group")
        action = menu.exec(self.node_view.viewport().mapToGlobal(view_pos))
        if action in actions:
            self._safe_add_node(actions[action], scene_pos)
        elif action == frame_all:
            self._fit_node_graph()
        elif action == auto_layout:
            self._auto_layout_node_graph()
        elif action == group_action:
            self._group_selected_nodes()

    def _rename_selected_node_dialog(self) -> None:
        selected = self._selected_node_ids()
        if len(selected) != 1:
            return
        node = self._node_graph.nodes[selected[0]]
        name, accepted = self._qt_widgets.QInputDialog.getText(
            self.widget,
            "Rename node",
            "Node name",
            text=node.name,
        )
        if accepted:
            self._node_graph.rename_node(node.id, name)
            self._redraw_node_graph()
            self._select_node_item(node.id)

    def _rename_group_dialog(self, group_id: str) -> None:
        group = self._node_graph.groups.get(group_id)
        if group is None:
            return
        name, accepted = self._qt_widgets.QInputDialog.getText(
            self.widget,
            "Rename group",
            "Group name",
            text=group.name,
        )
        if accepted:
            group.name = name.strip() or group.name
            self._redraw_node_graph()

    def _delete_group_only(self, group_id: str) -> None:
        group = self._node_graph.groups.pop(group_id, None)
        if group is None:
            return
        for node_id in group.node_ids:
            if node_id in self._node_graph.nodes:
                self._node_graph.nodes[node_id].group_id = None
        self._redraw_node_graph()

    def _delete_group_and_nodes(self, group_id: str) -> None:
        group = self._node_graph.groups.get(group_id)
        if group is None:
            return
        self._node_graph.delete_nodes(group.node_ids)
        self._node_graph.groups.pop(group_id, None)
        self._redraw_node_graph()

    def _on_node_selection_changed(self) -> None:
        try:
            selected = self._selected_node_ids()
            self._node_graph.selected_node_ids = selected
            if len(selected) != 1:
                self.node_inspector_type.setText(
                    "No node selected" if not selected else f"{len(selected)} nodes selected"
                )
                self.node_inspector_name.setText("")
                self.node_inspector_status.setText("")
                self.node_inspector_settings.setPlainText("")
                self.node_inspector_name.setEnabled(False)
                self.node_inspector_settings.setEnabled(False)
                self._rebuild_node_inspector_form(selected)
                self._update_viewport_context_for_selection(selected)
                return
            node = self._node_graph.nodes[selected[0]]
            definition = self._node_registry[node.type_id]
            self.node_inspector_name.setEnabled(True)
            self.node_inspector_settings.setEnabled(True)
            self.node_inspector_type.setText(f"{definition.category} / {definition.label}")
            self.node_inspector_name.setText(node.name)
            self.node_inspector_status.setText(f"Status: {node.status} - {node.message}")
            self.node_inspector_settings.setPlainText(
                json.dumps(node.settings, indent=2, sort_keys=True)
            )
            self._rebuild_node_inspector_form(selected)
            self._update_viewport_context_for_selection(selected)
        except RuntimeError:
            return

    def _rebuild_node_inspector_form(self, selected: tuple[str, ...]) -> None:
        self._clear_form_layout(self.node_inspector_form)
        if not selected:
            message = self._qt_widgets.QLabel("No node selected. Select a node to edit it.")
            message.setWordWrap(True)
            self.node_inspector_form.addRow(message)
            return
        if len(selected) > 1:
            message = self._qt_widgets.QLabel(f"{len(selected)} nodes selected")
            message.setWordWrap(True)
            actions = self._qt_widgets.QWidget()
            action_layout = self._qt_widgets.QGridLayout(actions)
            action_layout.setContentsMargins(0, 0, 0, 0)
            duplicate = self._qt_widgets.QPushButton("Duplicate")
            self._connect_safe_button(duplicate, "Duplicate Node", self._duplicate_selected_nodes)
            delete = self._qt_widgets.QPushButton("Delete")
            self._connect_safe_button(delete, "Delete Node", self._delete_selected_graph_items)
            group = self._qt_widgets.QPushButton("Group")
            self._connect_safe_button(group, "Group Nodes", self._group_selected_nodes)
            collapse = self._qt_widgets.QPushButton("Collapse")
            self._connect_safe_button(
                collapse,
                "Collapse Node",
                lambda: self._set_selected_nodes_collapsed(True),
            )
            expand = self._qt_widgets.QPushButton("Expand")
            self._connect_safe_button(
                expand,
                "Expand Node",
                lambda: self._set_selected_nodes_collapsed(False),
            )
            action_layout.addWidget(duplicate, 0, 0)
            action_layout.addWidget(delete, 0, 1)
            action_layout.addWidget(group, 1, 0)
            action_layout.addWidget(collapse, 1, 1)
            action_layout.addWidget(expand, 2, 0, 1, 2)
            self.node_inspector_form.addRow(message)
            self.node_inspector_form.addRow(actions)
            return
        node = self._node_graph.nodes[selected[0]]
        definition = self._node_registry[node.type_id]
        self.node_inspector_form.addRow("Node type", self._qt_widgets.QLabel(definition.label))
        self.node_inspector_form.addRow("Status", self._qt_widgets.QLabel(node.status))
        inputs = ", ".join(socket.name for socket in definition.inputs) or "None"
        outputs = ", ".join(socket.name for socket in definition.outputs) or "None"
        self.node_inspector_form.addRow("Inputs", self._qt_widgets.QLabel(inputs))
        self.node_inspector_form.addRow("Outputs", self._qt_widgets.QLabel(outputs))
        for key, value in node.settings.items():
            editor = self._setting_editor(node.id, key, value)
            self.node_inspector_form.addRow(_setting_label(key), editor)
        run_button = self._qt_widgets.QPushButton("Run Graph From This Node")
        self._connect_safe_button(
            run_button,
            "Run Graph From Node",
            lambda: self._execute_node_graph(execute_exports=False),
        )
        self.node_inspector_form.addRow(run_button)

    def _clear_form_layout(self, layout: Any) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _setting_editor(self, node_id: str, key: str, value: Any) -> Any:
        if isinstance(value, bool):
            editor = self._qt_widgets.QCheckBox()
            editor.setChecked(value)
            editor.toggled.connect(
                lambda checked, setting_key=key: self._update_node_setting(
                    node_id,
                    setting_key,
                    bool(checked),
                )
            )
            return editor
        if isinstance(value, int) and not isinstance(value, bool):
            editor = self._qt_widgets.QSpinBox()
            editor.setRange(-1_000_000, 1_000_000)
            editor.setValue(value)
            editor.valueChanged.connect(
                lambda new_value, setting_key=key: self._update_node_setting(
                    node_id,
                    setting_key,
                    int(new_value),
                )
            )
            return editor
        if isinstance(value, float):
            editor = self._qt_widgets.QDoubleSpinBox()
            editor.setRange(-1_000_000.0, 1_000_000.0)
            editor.setDecimals(4)
            editor.setValue(value)
            editor.valueChanged.connect(
                lambda new_value, setting_key=key: self._update_node_setting(
                    node_id,
                    setting_key,
                    float(new_value),
                )
            )
            return editor
        if isinstance(value, str):
            editor = self._qt_widgets.QLineEdit(value)
            editor.editingFinished.connect(
                lambda setting_key=key, field=editor: self._update_node_setting(
                    node_id,
                    setting_key,
                    field.text(),
                )
            )
            return editor
        summary = self._qt_widgets.QLabel("Edit nested value in advanced JSON")
        summary.setWordWrap(True)
        return summary

    def _update_node_setting(self, node_id: str, key: str, value: Any) -> None:
        node = self._node_graph.nodes.get(node_id)
        if node is None:
            return
        node.settings[key] = value
        self._node_graph.mark_downstream_dirty(node_id)
        self.node_inspector_status.setText(f"Status: {node.status} - {node.message}")
        self.node_inspector_settings.setPlainText(
            json.dumps(node.settings, indent=2, sort_keys=True)
        )
        self._set_node_status(f"Updated {node.name}: {key}")

    def _update_viewport_context_for_selection(self, selected: tuple[str, ...]) -> None:
        if len(selected) != 1:
            previous_context = self._viewport_node_context
            self._viewport_node_context = None
            if previous_context is not None and hasattr(self, "view"):
                self._render_scene()
            return
        node = self._node_graph.nodes.get(selected[0])
        if node is None:
            previous_context = self._viewport_node_context
            self._viewport_node_context = None
            if previous_context is not None and hasattr(self, "view"):
                self._render_scene()
            return
        previous_context = self._viewport_node_context
        self._viewport_node_context = node.type_id
        if previous_context != node.type_id and hasattr(self, "view"):
            self._render_scene()
        definition = self._node_registry[node.type_id]
        context = self._viewport_context_description(node.type_id)
        current_status = self.status.text() if hasattr(self, "status") else ""
        status_lines = current_status.splitlines()
        status_lines = [
            line for line in status_lines if not line.startswith("Selected node:")
        ]
        status_lines.append(f"Selected node: {definition.label} - {context}")
        self.status.setText("\n".join(status_lines))

    def _viewport_context_description(self, type_id: str) -> str:
        descriptions = {
            "mandrel_profile": "mandrel geometry only",
            "material_tow": "mandrel with tow settings context",
            "layer_stack": "active layer stack preview",
            "winding_pattern": "generated tow path preview",
            "path_optimisation": "optimised program preview",
            "coverage_analysis": "coverage-ready path preview",
            "simulation": "simulation path preview",
            "csv_export": "final export path preview",
            "gcode_export": "final machine path preview",
            "controller_run": "controller run preview",
        }
        return descriptions.get(type_id, "workflow preview")

    def _viewport_show_tow_path(self) -> bool:
        if self._viewport_node_context == "mandrel_profile":
            return False
        return not hasattr(self, "show_tow_path") or self.show_tow_path.isChecked()

    def _viewport_show_tow_band(self) -> bool:
        if self._viewport_node_context == "mandrel_profile":
            return False
        return not hasattr(self, "show_tow_band") or self.show_tow_band.isChecked()

    def _apply_node_inspector(self) -> None:
        selected = self._selected_node_ids()
        if len(selected) != 1:
            return
        node_id = selected[0]
        try:
            settings = json.loads(self.node_inspector_settings.toPlainText() or "{}")
            if not isinstance(settings, dict):
                raise ValueError("settings JSON must be an object")
        except (json.JSONDecodeError, ValueError) as exc:
            self.node_status.setText(f"Invalid node settings: {exc}")
            return
        self._node_graph.rename_node(node_id, self.node_inspector_name.text())
        self._node_graph.nodes[node_id].settings = settings
        self._node_graph.mark_downstream_dirty(node_id)
        self._redraw_node_graph()
        self._select_node_item(node_id)
        self._rebuild_node_inspector_form((node_id,))

    def _link_selected_nodes(self) -> None:
        selected = self._selected_node_ids()
        if len(selected) != 2:
            self._set_node_status("Select exactly two nodes to link")
            return
        source_id, target_id = selected
        self._log_graph_event(f"Link creation started: {source_id} -> {target_id}")
        link_args = self._first_compatible_link(source_id, target_id)
        if link_args is None:
            link_args = self._first_compatible_link(target_id, source_id)
        if link_args is None:
            self._set_node_status("No compatible socket pair between selected nodes")
            return
        try:
            self._node_graph.add_link(*link_args, registry=self._node_registry)
        except ValueError as exc:
            self._show_graph_error("Link creation failed", exc)
            return
        self._refresh_node_links()
        self._set_node_status("Link created")
        self._log_graph_event("Link creation succeeded")

    def _first_compatible_link(
        self,
        source_id: str,
        target_id: str,
    ) -> tuple[str, str, str, str] | None:
        source_def = self._node_registry[self._node_graph.nodes[source_id].type_id]
        target_def = self._node_registry[self._node_graph.nodes[target_id].type_id]
        for output_socket in source_def.outputs:
            for input_socket in target_def.inputs:
                try:
                    self._node_graph.validate_link(
                        source_id,
                        output_socket.name,
                        target_id,
                        input_socket.name,
                        self._node_registry,
                    )
                except ValueError:
                    continue
                return source_id, output_socket.name, target_id, input_socket.name
        return None

    def _unlink_selected_nodes(self) -> None:
        selected = self._selected_node_ids()
        self._node_graph.remove_links_for_nodes(selected)
        self._refresh_node_links()
        self._set_node_status("Selected node links removed")

    def _duplicate_selected_nodes(self) -> None:
        selected = self._selected_node_ids()
        new_nodes = [
            self._node_graph.duplicate_node(node_id, self._node_registry)
            for node_id in selected
        ]
        self._redraw_node_graph()
        if new_nodes:
            self._select_node_item(new_nodes[-1].id)

    def _delete_selected_nodes(self) -> None:
        selected = self._selected_node_ids()
        if not selected:
            return
        self._node_graph.delete_nodes(selected)
        self._redraw_node_graph()

    def _delete_selected_graph_items(self) -> None:
        link_ids = self._selected_link_ids()
        for link_id in link_ids:
            self._node_graph.links.pop(link_id, None)
        selected = self._selected_node_ids()
        if selected:
            self._node_graph.delete_nodes(selected)
        if link_ids or selected:
            self._redraw_node_graph()

    def _set_selected_nodes_collapsed(self, collapsed: bool) -> None:
        for node_id in self._selected_node_ids():
            self._node_graph.set_node_collapsed(node_id, collapsed)
        self._redraw_node_graph()

    def _group_selected_nodes(self) -> None:
        selected = self._selected_node_ids()
        if len(selected) < 2:
            self._set_node_status("Select at least two nodes to group")
            return
        self._node_graph.group_nodes(selected, name=f"Group {len(self._node_graph.groups) + 1}")
        self._redraw_node_graph()

    def _ungroup_selected_nodes(self) -> None:
        selected = set(self._selected_node_ids())
        if not selected:
            return
        for group_id, group in list(self._node_graph.groups.items()):
            if selected.intersection(group.node_ids):
                for node_id in group.node_ids:
                    if node_id in self._node_graph.nodes:
                        self._node_graph.nodes[node_id].group_id = None
                self._node_graph.groups.pop(group_id, None)
        self._redraw_node_graph()

    def _frame_selected_nodes(self) -> None:
        selected = self._selected_node_ids()
        if not selected:
            self._fit_node_graph()
            return
        rects = [
            self._node_items[node_id].sceneBoundingRect()
            for node_id in selected
            if node_id in self._node_items
        ]
        if not rects:
            return
        bounds = rects[0]
        for rect in rects[1:]:
            bounds = bounds.united(rect)
        self.node_view.fitInView(
            bounds.adjusted(-80.0, -80.0, 80.0, 80.0),
            self._qt_core.Qt.AspectRatioMode.KeepAspectRatio,
        )

    def _auto_layout_node_graph(self) -> None:
        try:
            ordered = self._node_graph.topological_node_ids()
        except ValueError:
            ordered = tuple(self._node_graph.nodes)
        columns: dict[int, list[str]] = {}
        depths: dict[str, int] = {}
        for node_id in ordered:
            incoming = self._node_graph.incoming_links(node_id)
            depth = 0
            if incoming:
                depth = 1 + max(depths.get(link.source_node_id, 0) for link in incoming)
            depths[node_id] = depth
            columns.setdefault(depth, []).append(node_id)
        for depth, node_ids in columns.items():
            for row, node_id in enumerate(node_ids):
                self._node_graph.set_node_position(
                    node_id,
                    60.0 + depth * 380.0,
                    70.0 + row * 200.0,
                )
        self._redraw_node_graph()
        self._schedule_fit_node_graph()

    def _sync_node_graph_from_scene(self) -> None:
        for node_id, item in self._node_items.items():
            position = item.pos()
            self._node_graph.set_node_position(node_id, float(position.x()), float(position.y()))

    def _execute_node_graph(self, *, execute_exports: bool) -> None:
        self._sync_node_graph_from_scene()
        self._log_graph_event("Graph execution started")
        try:
            ordered_node_ids = self._node_graph.topological_node_ids()
        except ValueError as exc:
            self._show_graph_error("Graph execution failed", exc)
            return
        for node_id in ordered_node_ids:
            node = self._node_graph.nodes[node_id]
            node.status = "processing"
            node.message = "Queued"
        self._redraw_node_graph()
        self._set_node_status("Graph running in background...")
        self._start_node_graph_worker(execute_exports=execute_exports)

    def _start_node_graph_worker(self, *, execute_exports: bool) -> None:
        qt_core = self._qt_core
        registry = self._node_registry
        graph_data = self._node_graph.to_dict()

        class _WorkerSignals(qt_core.QObject):  # type: ignore[name-defined, misc, valid-type]
            finished = qt_core.Signal(object, object)
            failed = qt_core.Signal(str)

        class _GraphWorker(qt_core.QRunnable):  # type: ignore[name-defined, misc, valid-type]
            def __init__(self) -> None:
                super().__init__()
                self.signals = _WorkerSignals()

            def run(self) -> None:
                try:
                    graph = NodeGraphState.from_dict(graph_data, registry)
                    result = NodeGraphExecutor(
                        registry,
                        execute_exports=execute_exports,
                    ).execute(graph)
                except Exception as exc:  # noqa: BLE001 - reported to UI thread
                    self.signals.failed.emit(str(exc))
                    return
                self.signals.finished.emit(result, graph.to_dict())

        worker = _GraphWorker()
        worker.signals.finished.connect(self._on_node_graph_worker_finished)
        worker.signals.failed.connect(self._on_node_graph_worker_failed)
        self._node_workers.append(worker)
        self._node_thread_pool.start(worker)

    def _on_node_graph_worker_finished(
        self,
        result: GraphExecutionResult,
        graph_data: dict[str, object],
    ) -> None:
        self._node_graph = NodeGraphState.from_dict(graph_data, self._node_registry)
        self._last_node_result = result
        self._redraw_node_graph()
        self._draw_node_graph_result(result)
        if result.warnings:
            warning_text = "\n".join(result.warnings)
            self._set_node_status(f"Graph completed with warnings:\n{warning_text}")
            return
        self._set_node_status(f"Graph complete: {len(result.executed_node_ids)} nodes executed")
        self._log_graph_event("Graph execution completed")

    def _on_node_graph_worker_failed(self, message: str) -> None:
        for node in self._node_graph.nodes.values():
            if node.status == "processing":
                node.status = "error"
                node.message = message
        self._redraw_node_graph()
        self._show_graph_error(f"Graph execution failed: {message}")

    def _draw_node_graph_result(self, result: GraphExecutionResult) -> None:
        program = None
        mandrel = None
        for outputs in result.node_outputs.values():
            if program is None and "program" in outputs:
                program = outputs["program"]
            if mandrel is None and "mandrel" in outputs:
                mandrel = outputs["mandrel"]
        if program is None or mandrel is None:
            return
        scene = self._vispy_scene
        for visual in self._visuals:
            visual.parent = None
        self._visuals.clear()
        if isinstance(mandrel, CylinderMandrel):
            mesh_vertices, mesh_faces = cylinder_mesh_arrays(
                mandrel,
                theta_segments=64,
                z_segments=24,
            )
            center_z = mandrel.length_mm / 2.0
            display_mandrel = orient_points_for_horizontal_view(
                mesh_vertices,
                length_mm=mandrel.length_mm,
                center_z_mm=center_z,
            )
            scale = max(mandrel.length_mm, mandrel.radius_mm * 6.0)
        else:
            mesh_vertices, mesh_faces = profile_mesh_arrays(
                mandrel,
                theta_segments=64,
                z_segments=48,
            )
            center_z = 0.5 * (mandrel.start_z_mm + mandrel.end_z_mm)
            display_mandrel = orient_points_for_horizontal_view(
                mesh_vertices,
                length_mm=mandrel.length_mm,
                center_z_mm=center_z,
            )
            scale = max(mandrel.length_mm, mandrel.max_radius_mm * 6.0)
        self._visuals.append(
            scene.visuals.Mesh(
                vertices=display_mandrel,
                faces=mesh_faces,
                color=(0.42, 0.49, 0.55, 1.0),
                shading="smooth",
                parent=self.view.scene,
            )
        )
        self._visuals[-1].set_gl_state("opaque", depth_test=True, cull_face=False)
        colors = (
            (0.10, 0.58, 1.0, 1.0),
            (1.0, 0.52, 0.12, 1.0),
            (0.30, 0.82, 0.44, 1.0),
            (0.86, 0.36, 0.95, 1.0),
        )
        if self._viewport_show_tow_path():
            for index, layer in enumerate(program.layers):
                points = orient_points_for_horizontal_view(
                    layer.path.points_mm,
                    length_mm=mandrel.length_mm,
                    center_z_mm=center_z,
                )
                points = offset_display_surface(points, offset_mm=0.75)
                self._visuals.append(
                    scene.visuals.Line(
                        pos=points,
                        color=colors[index % len(colors)],
                        width=2.0,
                        parent=self.view.scene,
                    )
                )
                self._visuals[-1].set_gl_state("opaque", depth_test=True)
        self.view.camera.distance = scale * 1.4
        self.view.camera.center = (0.0, 0.0, 0.0)
        self.view.camera.scale_factor = scale
        self.status.setText(
            f"Mode: node graph\nLayers: {len(program.layers)}\nPoints: {program.point_count}"
        )

    def _node_double_spin(self, source: Any) -> Any:
        spin = self._double_spin(
            float(source.value()),
            float(source.minimum()),
            float(source.maximum()),
            float(source.singleStep()),
        )
        if hasattr(source, "decimals"):
            spin.setDecimals(source.decimals())
        spin.valueChanged.connect(lambda value: self._set_spin_value(source, value))
        spin.valueChanged.connect(lambda _value: self._render_from_node())
        source.valueChanged.connect(lambda value: self._set_spin_value(spin, value, emit=False))
        return spin

    def _node_int_spin(self, source: Any) -> Any:
        spin = self._int_spin(
            int(source.value()),
            int(source.minimum()),
            int(source.maximum()),
            int(source.singleStep()),
        )
        spin.valueChanged.connect(lambda value: self._set_spin_value(source, value))
        spin.valueChanged.connect(lambda _value: self._render_from_node())
        source.valueChanged.connect(lambda value: self._set_spin_value(spin, value, emit=False))
        return spin

    def _node_checkbox(self, source: Any) -> Any:
        checkbox = self._qt_widgets.QCheckBox()
        checkbox.setChecked(source.isChecked())
        checkbox.toggled.connect(lambda checked: self._set_checkbox_value(source, checked))
        checkbox.toggled.connect(lambda _checked: self._render_from_node())
        source.toggled.connect(
            lambda checked: self._set_checkbox_value(checkbox, checked, emit=False)
        )
        return checkbox

    def _node_combo(self, source: Any) -> Any:
        combo = self._qt_widgets.QComboBox()
        combo.addItems([source.itemText(index) for index in range(source.count())])
        combo.setCurrentText(source.currentText())
        combo.currentTextChanged.connect(lambda text: self._set_combo_value(source, text))
        combo.currentTextChanged.connect(lambda _text: self._render_from_node())
        source.currentTextChanged.connect(
            lambda text: self._set_combo_value(combo, text, emit=False)
        )
        return combo

    def _node_line_edit(self, source: Any) -> Any:
        edit = self._qt_widgets.QLineEdit(source.text())
        edit.editingFinished.connect(lambda: self._set_line_edit_value(source, edit.text()))
        edit.editingFinished.connect(self._render_from_node)
        source.textChanged.connect(lambda text: self._set_line_edit_value(edit, text, emit=False))
        return edit

    def _set_spin_value(self, target: Any, value: float | int, *, emit: bool = True) -> None:
        if abs(float(target.value()) - float(value)) < 1e-9:
            return
        target.blockSignals(not emit)
        target.setValue(value)
        target.blockSignals(False)

    def _set_checkbox_value(self, target: Any, checked: bool, *, emit: bool = True) -> None:
        if target.isChecked() == checked:
            return
        target.blockSignals(not emit)
        target.setChecked(checked)
        target.blockSignals(False)

    def _set_combo_value(self, target: Any, text: str, *, emit: bool = True) -> None:
        if target.currentText() == text:
            return
        target.blockSignals(not emit)
        target.setCurrentText(text)
        target.blockSignals(False)

    def _set_line_edit_value(self, target: Any, text: str, *, emit: bool = True) -> None:
        if target.text() == text:
            return
        target.blockSignals(not emit)
        target.setText(text)
        target.blockSignals(False)

    def _render_from_node(self) -> None:
        if hasattr(self, "view"):
            self._render_scene()

    def _double_spin(self, value: float, minimum: float, maximum: float, step: float) -> Any:
        spin = self._qt_widgets.QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(3)
        spin.setSingleStep(step)
        spin.setValue(value)
        return spin

    def _int_spin(self, value: int, minimum: int, maximum: int, step: int) -> Any:
        spin = self._qt_widgets.QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setValue(value)
        return spin

    def _scroll_tab(self, *widgets: Any) -> Any:
        container = self._qt_widgets.QWidget()
        layout = self._qt_widgets.QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)
        for widget in widgets:
            layout.addWidget(widget)
        layout.addStretch(1)

        area = self._qt_widgets.QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(self._qt_widgets.QFrame.Shape.NoFrame)
        area.setWidget(container)
        return area

    def _path_selector(self, line_edit: Any, title: str, file_filter: str) -> Any:
        container = self._qt_widgets.QWidget()
        layout = self._qt_widgets.QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        browse = self._qt_widgets.QPushButton("...")
        browse.setFixedWidth(32)
        self._connect_safe_button(
            browse,
            f"Browse Output {title}",
            lambda: self._browse_output_path(line_edit, title, file_filter),
        )
        layout.addWidget(line_edit, 1)
        layout.addWidget(browse)
        return container

    def _open_path_selector(self, line_edit: Any, title: str, file_filter: str) -> Any:
        container = self._qt_widgets.QWidget()
        layout = self._qt_widgets.QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        browse = self._qt_widgets.QPushButton("...")
        browse.setFixedWidth(32)
        self._connect_safe_button(
            browse,
            f"Browse Input {title}",
            lambda: self._browse_input_path(line_edit, title, file_filter),
        )
        layout.addWidget(line_edit, 1)
        layout.addWidget(browse)
        return container

    def _current_config(self) -> CylinderPreviewConfig:
        return replace(
            self._config,
            length_mm=float(self.length.value()),
            radius_mm=float(self.radius.value()),
            tow_width_mm=float(self.tow_width.value()),
            winding_angle_deg=float(self.angle.value()),
            points_per_pass=int(self.points.value()),
            passes=int(self.passes.value()),
            radial_clearance_mm=float(self.clearance.value()),
            phase_offset_deg=(
                None if self.auto_phase.isChecked() else float(self.phase_offset.value())
            ),
            alternate_direction=bool(self.alternate.isChecked()),
        )

    def _current_profile_config(self) -> ProfileDomePreviewConfig:
        samples = int(self.profile_samples.value())
        turnaround_radius = float(self.turnaround_radius.value())
        return replace(
            self._profile_config,
            profile_path=Path(self.profile_path.text().strip() or "mandrels/profile.dxf"),
            samples=None if samples <= 0 else samples,
            path_mode=_profile_path_mode_from_label(self.profile_path_mode.currentText()),
            tow_width_mm=float(self.tow_width.value()),
            winding_angle_deg=float(self.angle.value()),
            points_per_span=int(self.points.value()),
            min_radius_mm=float(self.profile_min_radius.value()),
            turnaround_points=int(self.turnaround_points.value()),
            turnaround_angle_deg=float(self.turnaround_angle.value()),
            circuits=int(self.circuits.value()),
            turnaround_radius_mm=None if turnaround_radius <= 0.0 else turnaround_radius,
            radial_clearance_mm=float(self.clearance.value()),
        )

    def _current_pattern_config(self) -> PatternPlannerConfig:
        return PatternPlannerConfig(
            coverage_target=float(self.pattern_coverage.value()) / 100.0,
            include_hoop_layer=bool(self.include_hoop_layer.isChecked()),
            balanced_pm_layers=bool(self.balanced_pm_layers.isChecked()),
            max_angle_error_deg=float(self.pattern_max_angle_error.value()),
        )

    def _current_layer_schedule(self) -> WindingSchedule | None:
        if not self.use_layer_stack.isChecked():
            return None
        layers = []
        for row in range(self.layer_table.rowCount()):
            try:
                layers.append(self._layer_spec_from_row(row))
            except ValueError as exc:
                self.status.setText(f"Invalid layer row {row + 1}: {exc}")
                return None
        if not layers:
            return None
        return WindingSchedule(
            layers=tuple(layers),
            radial_clearance_mm=float(self.clearance.value()),
            nominal_feedrate_mm_min=float(self.feedrate.value()),
        )

    def _layer_spec_from_row(self, row: int) -> WindingLayerSpec:
        def get_text(col: int) -> str:
            widget = self.layer_table.cellWidget(row, col)
            if widget is not None and hasattr(widget, "currentText"):
                return str(widget.currentText()).strip()
            current = self.layer_table.item(row, col)
            return current.text().strip() if current is not None else ""

        name = get_text(0) or f"Layer {row + 1}"
        enabled = get_text(0).lower() != "disabled"
        name = get_text(1) or f"Layer {row + 1}"
        winding_type = get_text(2)
        direction = get_text(4) or "positive"
        passes_text = get_text(5)
        feedrate_text = get_text(8)
        clearance_text = get_text(9)
        return WindingLayerSpec(
            name=name,
            winding_type=cast(Any, winding_type or "helical"),
            target_angle_deg=float(get_text(3) or "45"),
            tow_width_mm=float(get_text(6) or "6"),
            layer_thickness_mm=max(0.0, float(get_text(7) or "0")),
            coverage_target=max(0.01, float(get_text(11) or "100") / 100.0),
            direction=cast(Any, direction),
            point_count=max(2, int(float(get_text(12) or "500"))),
            enabled=enabled,
            number_of_passes=(
                None if not passes_text or int(float(passes_text)) <= 0 else int(float(passes_text))
            ),
            feedrate_mm_min=None if not feedrate_text else float(feedrate_text),
            mandrel_clearance_mm=None if not clearance_text else float(clearance_text),
            colour=get_text(10) or "#1e90ff",
            transition_points=max(2, int(float(get_text(13) or "20"))),
        )

    def _set_layer_table_value(self, row: int, col: int, value: str) -> None:
        item = self.layer_table.item(row, col)
        if item is None:
            item = self._qt_widgets.QTableWidgetItem()
            self.layer_table.setItem(row, col, item)
        item.setText(value)

    def _set_layer_table_combo(
        self,
        row: int,
        col: int,
        values: tuple[str, ...],
        current_value: str,
    ) -> None:
        combo = self._qt_widgets.QComboBox()
        combo.addItems(list(values))
        combo.setCurrentText(current_value if current_value in values else values[0])
        combo.currentTextChanged.connect(self._on_layer_table_changed)
        self.layer_table.setCellWidget(row, col, combo)

    def _layer_row_values(self, row: int) -> list[str]:
        values = []
        for col in range(self.layer_table.columnCount()):
            widget = self.layer_table.cellWidget(row, col)
            if widget is not None and hasattr(widget, "currentText"):
                values.append(str(widget.currentText()))
                continue
            item = self.layer_table.item(row, col)
            values.append(item.text() if item is not None else "")
        return values

    def _add_layer_row(
        self,
        spec: WindingLayerSpec | None = None,
        *,
        render: bool = True,
    ) -> None:
        spec = spec or WindingLayerSpec(
            name=f"layer-{self.layer_table.rowCount() + 1}",
            winding_type="helical",
            target_angle_deg=45.0,
            tow_width_mm=float(self.tow_width.value()),
            coverage_target=float(self.pattern_coverage.value()) / 100.0,
            direction="positive",
        )
        row = self.layer_table.rowCount()
        self.layer_table.blockSignals(True)
        self.layer_table.insertRow(row)
        values = [
            "enabled" if spec.enabled else "disabled",
            spec.name,
            spec.winding_type,
            f"{spec.target_angle_deg:.3f}",
            spec.direction,
            "" if spec.number_of_passes is None else str(spec.number_of_passes),
            f"{spec.tow_width_mm:.3f}",
            f"{spec.layer_thickness_mm:.3f}",
            "" if spec.feedrate_mm_min is None else f"{spec.feedrate_mm_min:.3f}",
            "" if spec.mandrel_clearance_mm is None else f"{spec.mandrel_clearance_mm:.3f}",
            spec.colour,
            f"{spec.coverage_target * 100.0:.3f}",
            str(spec.point_count),
            str(spec.transition_points),
        ]
        for col, value in enumerate(values):
            if col == 0:
                self._set_layer_table_combo(row, col, ("enabled", "disabled"), value)
            elif col == 2:
                self._set_layer_table_combo(row, col, ("hoop", "helical", "polar"), value)
            elif col == 4:
                self._set_layer_table_combo(
                    row,
                    col,
                    ("positive", "negative", "alternating", "hoop", "polar"),
                    value,
                )
            else:
                self._set_layer_table_value(row, col, value)
        self.layer_table.blockSignals(False)
        if render:
            self._render_scene()

    def _add_layer_preset(self, winding_type: str) -> None:
        if winding_type == "hoop":
            spec = WindingLayerSpec(
                name=f"hoop-{self.layer_table.rowCount() + 1}",
                winding_type="hoop",
                target_angle_deg=90.0,
                tow_width_mm=float(self.tow_width.value()),
                coverage_target=float(self.pattern_coverage.value()) / 100.0,
                direction="hoop",
                point_count=max(12, int(self.points.value())),
                transition_points=20,
            )
        elif winding_type == "polar":
            spec = WindingLayerSpec(
                name=f"polar-{self.layer_table.rowCount() + 1}",
                winding_type="polar",
                target_angle_deg=min(85.0, max(60.0, float(self.angle.value()) + 15.0)),
                tow_width_mm=float(self.tow_width.value()),
                coverage_target=float(self.pattern_coverage.value()) / 100.0,
                direction="negative",
                point_count=int(self.points.value()),
                transition_points=24,
            )
        else:
            spec = WindingLayerSpec(
                name=f"helical-{self.layer_table.rowCount() + 1}",
                winding_type="helical",
                target_angle_deg=float(self.angle.value()),
                tow_width_mm=float(self.tow_width.value()),
                coverage_target=float(self.pattern_coverage.value()) / 100.0,
                direction="positive",
                point_count=int(self.points.value()),
                transition_points=20,
            )
        self._add_layer_row(spec)

    def _populate_default_layers(self) -> None:
        if self.layer_table.rowCount() > 0:
            return
        self._add_layer_row(
            WindingLayerSpec(
                name="hoop",
                winding_type="hoop",
                target_angle_deg=90.0,
                tow_width_mm=float(self.tow_width.value()),
                coverage_target=float(self.pattern_coverage.value()) / 100.0,
                direction="hoop",
                point_count=max(12, int(self.points.value())),
                transition_points=20,
            ),
            render=False,
        )
        self._add_layer_row(
            WindingLayerSpec(
                name="helical",
                winding_type="helical",
                target_angle_deg=float(self.angle.value()),
                tow_width_mm=float(self.tow_width.value()),
                coverage_target=float(self.pattern_coverage.value()) / 100.0,
                direction="positive",
                point_count=int(self.points.value()),
                transition_points=20,
            ),
            render=False,
        )
        self._add_layer_row(
            WindingLayerSpec(
                name="polar",
                winding_type="polar",
                target_angle_deg=min(85.0, max(60.0, float(self.angle.value()) + 15.0)),
                tow_width_mm=float(self.tow_width.value()),
                coverage_target=float(self.pattern_coverage.value()) / 100.0,
                direction="negative",
                point_count=int(self.points.value()),
                transition_points=24,
            ),
            render=False,
        )

    def _remove_selected_layer_rows(self) -> None:
        rows = sorted({index.row() for index in self.layer_table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.layer_table.removeRow(row)
        self._render_scene()

    def _move_selected_layer(self, direction: int) -> None:
        rows = sorted({index.row() for index in self.layer_table.selectedIndexes()})
        if not rows:
            return
        row = rows[0]
        target = row + direction
        if target < 0 or target >= self.layer_table.rowCount():
            return
        values = self._layer_row_values(row)
        self.layer_table.blockSignals(True)
        self.layer_table.removeRow(row)
        self.layer_table.insertRow(target)
        for col, value in enumerate(values):
            if col == 0:
                self._set_layer_table_combo(
                    row=target,
                    col=col,
                    values=("enabled", "disabled"),
                    current_value=value,
                )
            elif col == 2:
                self._set_layer_table_combo(
                    row=target,
                    col=col,
                    values=("hoop", "helical", "polar"),
                    current_value=value,
                )
            elif col == 4:
                self._set_layer_table_combo(
                    row=target,
                    col=col,
                    values=("positive", "negative", "alternating", "hoop", "polar"),
                    current_value=value,
                )
            else:
                self._set_layer_table_value(target, col, value)
        self.layer_table.blockSignals(False)
        self._render_scene()

    def _on_layer_table_changed(self, *_args: Any) -> None:
        self._render_scene()

    def _is_profile_dome_mode(self) -> bool:
        return self.mode.currentText() == "Profile Dome"

    def _is_pattern_planner_enabled(self) -> bool:
        return bool(self.use_pattern_planner.isChecked())

    def _current_export_paths(self) -> PreviewExportPaths:
        return PreviewExportPaths(
            csv_path=self.csv_output.text().strip() or self._default_export_paths.csv_path,
            gcode_path=self.gcode_output.text().strip() or self._default_export_paths.gcode_path,
            coverage_csv_path=(
                self.coverage_output.text().strip()
                or self._default_export_paths.coverage_csv_path
            ),
            coverage_summary_csv_path=(
                self.coverage_summary_output.text().strip()
                or self._default_export_paths.coverage_summary_csv_path
            ),
            preview_obj_path=self.obj_output.text().strip()
            or self._default_export_paths.preview_obj_path,
        )

    def _render_scene(self) -> None:
        if self._is_profile_dome_mode():
            if self._is_pattern_planner_enabled():
                self._render_profile_dome_pattern_scene()
                return
            self._render_profile_dome_scene()
            return
        if self._current_layer_schedule() is not None:
            self._render_custom_layer_scene()
            return
        if self._is_pattern_planner_enabled():
            self._render_cylinder_pattern_scene()
            return

        scene = self._vispy_scene
        try:
            preview = build_cylinder_preview_scene(self._current_config())
        except ValueError as exc:
            self.status.setText(f"Invalid preview inputs: {exc}")
            return

        for visual in self._visuals:
            visual.parent = None
        self._visuals.clear()

        self._visuals.append(
            scene.visuals.Mesh(
                vertices=preview.display_cylinder_vertices_mm,
                faces=preview.cylinder_faces,
                color=(0.42, 0.49, 0.55, 1.0),
                shading="smooth",
                parent=self.view.scene,
            )
        )
        self._visuals[-1].set_gl_state("opaque", depth_test=True, cull_face=False)
        if self._viewport_show_tow_path():
            self._visuals.append(
                scene.visuals.Line(
                    pos=preview.display_path_points_mm,
                    color=(0.10, 0.58, 1.0, 1.0),
                    width=2.5,
                    parent=self.view.scene,
                )
            )
            self._visuals[-1].set_gl_state("opaque", depth_test=True)
        if self._viewport_show_tow_band():
            self._visuals.append(
                scene.visuals.Mesh(
                    vertices=preview.display_tow_vertices_mm,
                    faces=preview.tow_faces,
                    color=(1.0, 0.52, 0.12, 0.82),
                    shading="flat",
                    parent=self.view.scene,
                )
            )
            self._visuals[-1].set_gl_state("translucent", depth_test=True, cull_face=False)
        self.view.camera.distance = (
            max(preview.config.length_mm, preview.config.radius_mm * 6.0) * 1.4
        )
        self.view.camera.center = (0.0, 0.0, 0.0)
        self.view.camera.scale_factor = max(
            preview.config.length_mm,
            preview.config.radius_mm * 6.0,
        )
        self.status.setText(
            f"Points: {preview.path.point_count}\n"
            f"Final turns: {preview.path.final_turns:.4f}\n"
            f"Closure error: {preview.closure.closure_error_deg:.3f} deg\n"
            f"Covered: {preview.coverage_summary.covered_percent:.2f}%\n"
            f"Gap: {preview.coverage_summary.gap_percent:.2f}%\n"
            f"Overlap: {preview.coverage_summary.overlap_percent:.2f}%"
        )

    def _render_custom_layer_scene(self) -> None:
        schedule = self._current_layer_schedule()
        if schedule is None:
            return
        scene = self._vispy_scene
        try:
            mandrel = self._current_custom_mandrel()
            program = plan_winding_schedule(mandrel, schedule)
        except (OSError, ValueError) as exc:
            self.status.setText(f"Invalid layer stack: {exc}")
            return
        for visual in self._visuals:
            visual.parent = None
        self._visuals.clear()
        mesh_vertices, mesh_faces = cylinder_mesh_arrays(
            mandrel,
            theta_segments=64,
            z_segments=24,
        )
        display_mandrel = orient_points_for_horizontal_view(
            mesh_vertices,
            length_mm=mandrel.length_mm,
        )
        self._visuals.append(
            scene.visuals.Mesh(
                vertices=display_mandrel,
                faces=mesh_faces,
                color=(0.42, 0.49, 0.55, 1.0),
                shading="smooth",
                parent=self.view.scene,
            )
        )
        self._visuals[-1].set_gl_state("opaque", depth_test=True, cull_face=False)
        colors = (
            (0.10, 0.58, 1.0, 1.0),
            (1.0, 0.52, 0.12, 1.0),
            (0.30, 0.82, 0.44, 1.0),
            (0.86, 0.36, 0.95, 1.0),
        )
        if self._viewport_show_tow_path():
            for index, layer in enumerate(program.layers):
                points = orient_points_for_horizontal_view(
                    layer.path.points_mm,
                    length_mm=mandrel.length_mm,
                    center_z_mm=0.5 * mandrel.length_mm,
                )
                points = offset_display_surface(points, offset_mm=0.75)
                self._visuals.append(
                    scene.visuals.Line(
                        pos=points,
                        color=colors[index % len(colors)],
                        width=2.0,
                        parent=self.view.scene,
                    )
                )
                self._visuals[-1].set_gl_state("opaque", depth_test=True)
        scale = max(mandrel.length_mm, mandrel.radius_mm * 6.0)
        self.view.camera.distance = scale * 1.4
        self.view.camera.center = (0.0, 0.0, 0.0)
        self.view.camera.scale_factor = scale
        self.status.setText(
            f"Mode: custom layer stack\n"
            f"Layers: {len(program.layers)}\n"
            f"Points: {program.point_count}\n"
            + "\n".join(
                f"{report.layer_name}: {report.winding_type}, {report.actual_angle_deg:.2f} deg"
                for report in program.reports
            )
        )

    def _current_custom_mandrel(self) -> Any:
        return CylinderMandrel(
            length_mm=float(self.length.value()),
            radius_mm=float(self.radius.value()),
        )

    def _render_cylinder_pattern_scene(self) -> None:
        try:
            preview = build_cylinder_pattern_preview_scene(
                self._current_config(),
                self._current_pattern_config(),
                feedrate_mm_min=float(self.feedrate.value()),
            )
        except ValueError as exc:
            self.status.setText(f"Invalid cylinder pattern inputs: {exc}")
            return
        self._draw_pattern_preview(preview, mode_label="cylinder pattern")

    def _render_profile_dome_pattern_scene(self) -> None:
        try:
            preview = build_profile_dome_pattern_preview_scene(
                self._current_profile_config(),
                self._current_pattern_config(),
                feedrate_mm_min=float(self.feedrate.value()),
            )
        except (OSError, ValueError) as exc:
            self.status.setText(f"Invalid profile dome pattern inputs: {exc}")
            return
        suffix = (
            "\nHoop layer is ignored for axisymmetric profile schedules."
            if self.include_hoop_layer.isChecked()
            else ""
        )
        self._draw_pattern_preview(
            preview,
            mode_label=f"profile {self._current_profile_config().path_mode} pattern",
            suffix=suffix,
        )

    def _draw_pattern_preview(
        self,
        preview: Any,
        *,
        mode_label: str,
        suffix: str = "",
    ) -> None:
        scene = self._vispy_scene
        for visual in self._visuals:
            visual.parent = None
        self._visuals.clear()

        self._visuals.append(
            scene.visuals.Mesh(
                vertices=preview.display_mandrel_vertices_mm,
                faces=preview.mandrel_faces,
                color=(0.42, 0.49, 0.55, 1.0),
                shading="smooth",
                parent=self.view.scene,
            )
        )
        self._visuals[-1].set_gl_state("opaque", depth_test=True, cull_face=False)

        colors = (
            (0.10, 0.58, 1.0, 1.0),
            (1.0, 0.52, 0.12, 1.0),
            (0.30, 0.82, 0.44, 1.0),
            (0.86, 0.36, 0.95, 1.0),
        )
        if self._viewport_show_tow_path():
            for index, points in enumerate(preview.display_layer_path_points_mm):
                if points.shape[0] < 2:
                    continue
                self._visuals.append(
                    scene.visuals.Line(
                        pos=points,
                        color=colors[index % len(colors)],
                        width=2.0,
                        parent=self.view.scene,
                    )
                )
                self._visuals[-1].set_gl_state("opaque", depth_test=True)
            for points in preview.display_transition_path_points_mm:
                self._visuals.append(
                    scene.visuals.Line(
                        pos=points,
                        color=(0.90, 0.90, 0.90, 0.75),
                        width=1.5,
                        parent=self.view.scene,
                    )
                )
                self._visuals[-1].set_gl_state("translucent", depth_test=True)

        radius_mm = _preview_radius_mm(preview.mandrel)
        scale = max(preview.mandrel.length_mm, radius_mm * 6.0)
        self.view.camera.distance = scale * 1.4
        self.view.camera.center = (0.0, 0.0, 0.0)
        self.view.camera.scale_factor = scale

        report_lines = [
            (
                f"{report.layer_name}: {report.winding_type}, "
                f"{report.actual_angle_deg:.2f} deg, "
                f"{report.circuits} circuits, "
                f"{report.coverage_percent:.1f}% coverage, "
                f"gap {report.gap_mm:.3f} mm, "
                f"overlap {report.overlap_mm:.3f} mm"
            )
            for report in preview.program.reports
        ]
        self.status.setText(
            f"Mode: {mode_label}\n"
            f"Layers: {len(preview.program.layers)}\n"
            f"Points: {preview.program.point_count}\n"
            + "\n".join(report_lines)
            + suffix
        )

    def _render_profile_dome_scene(self) -> None:
        scene = self._vispy_scene
        try:
            preview = build_profile_dome_preview_scene(self._current_profile_config())
        except (OSError, ValueError) as exc:
            self.status.setText(f"Invalid profile dome inputs: {exc}")
            return

        for visual in self._visuals:
            visual.parent = None
        self._visuals.clear()

        self._visuals.append(
            scene.visuals.Mesh(
                vertices=preview.display_profile_vertices_mm,
                faces=preview.profile_faces,
                color=(0.42, 0.49, 0.55, 1.0),
                shading="smooth",
                parent=self.view.scene,
            )
        )
        self._visuals[-1].set_gl_state("opaque", depth_test=True, cull_face=False)
        if self._viewport_show_tow_path():
            self._visuals.append(
                scene.visuals.Line(
                    pos=preview.display_path_points_mm,
                    color=(0.10, 0.58, 1.0, 1.0),
                    width=2.5,
                    parent=self.view.scene,
                )
            )
            self._visuals[-1].set_gl_state("opaque", depth_test=True)

        scale = max(preview.profile.length_mm, preview.profile.max_radius_mm * 6.0)
        self.view.camera.distance = scale * 1.4
        self.view.camera.center = (0.0, 0.0, 0.0)
        self.view.camera.scale_factor = scale
        self.status.setText(
            f"Mode: profile {preview.config.path_mode}\n"
            f"Profile: {preview.config.profile_path}\n"
            f"Points: {preview.path.point_count}\n"
            f"Final turns: {preview.path.final_turns:.4f}\n"
            f"Turn radius: {preview.turnaround_radius_mm:.3f} mm\n"
            f"Geodesic radius: {preview.geodesic_radius_mm:.3f} mm\n"
            f"Safe Z: {preview.safe_start_z_mm:.3f}..{preview.safe_end_z_mm:.3f} mm\n"
            f"B angle: {preview.motion_table.b_deg.min():.2f}.."
            f"{preview.motion_table.b_deg.max():.2f} deg"
        )

    def _apply_config_to_controls(self, config: CylinderPreviewConfig) -> None:
        self._config = config
        self.length.setValue(config.length_mm)
        self.radius.setValue(config.radius_mm)
        self.tow_width.setValue(config.tow_width_mm)
        self.angle.setValue(config.winding_angle_deg)
        self.points.setValue(config.points_per_pass)
        self.passes.setValue(config.passes)
        self.auto_phase.setChecked(config.phase_offset_deg is None)
        self.phase_offset.setValue(_display_phase_offset(config))
        self.phase_offset.setEnabled(config.phase_offset_deg is not None)
        self.clearance.setValue(config.radial_clearance_mm)
        self.alternate.setChecked(config.alternate_direction)

    def _apply_profile_config_to_controls(self, config: ProfileDomePreviewConfig) -> None:
        self._profile_config = config
        self.profile_path.setText(str(config.profile_path))
        self.profile_path_mode.setCurrentText(_profile_path_mode_label(config.path_mode))
        self.profile_samples.setValue(0 if config.samples is None else config.samples)
        self.profile_min_radius.setValue(config.min_radius_mm)
        self.turnaround_radius.setValue(
            0.0 if config.turnaround_radius_mm is None else config.turnaround_radius_mm
        )
        self.turnaround_points.setValue(config.turnaround_points)
        self.turnaround_angle.setValue(config.turnaround_angle_deg)
        self.circuits.setValue(config.circuits)

    def _apply_pattern_config_to_controls(
        self,
        config: PatternPlannerConfig,
        *,
        enabled: bool,
    ) -> None:
        self.use_pattern_planner.setChecked(enabled)
        self.pattern_coverage.setValue(config.coverage_target * 100.0)
        self.include_hoop_layer.setChecked(config.include_hoop_layer)
        self.balanced_pm_layers.setChecked(config.balanced_pm_layers)
        self.pattern_max_angle_error.setValue(config.max_angle_error_deg)

    def _apply_export_paths_to_controls(self, paths: PreviewExportPaths) -> None:
        self.csv_output.setText(paths.csv_path)
        self.gcode_output.setText(paths.gcode_path)
        self.coverage_output.setText(paths.coverage_csv_path)
        self.coverage_summary_output.setText(paths.coverage_summary_csv_path)
        self.obj_output.setText(paths.preview_obj_path)

    def _load_project_dialog(self) -> None:
        filename, _ = self._qt_widgets.QFileDialog.getOpenFileName(
            self.widget,
            "Load FilamentWinder project",
            "",
            "FilamentWinder Project (*.fwp.json);;JSON Files (*.json);;All Files (*)",
        )
        if not filename:
            return
        try:
            project = load_project(filename)
            config = preview_config_from_project(
                project,
                base_config=self._current_config(),
            )
            profile_config = profile_config_from_project(
                project,
                base_config=self._current_profile_config(),
            )
            pattern_config = pattern_config_from_project(
                project,
                base_config=self._current_pattern_config(),
            )
        except (OSError, KeyError, TypeError, ValueError) as exc:
            self.status.setText(f"Could not load project: {exc}")
            return
        self.project_name.setText(project.name)
        self.feedrate.setValue(project.machine.feedrate_mm_min)
        self._apply_export_paths_to_controls(export_paths_from_project(project))
        self._set_project_path(Path(filename))
        self._apply_config_to_controls(config)
        self._apply_profile_config_to_controls(profile_config)
        self._apply_pattern_config_to_controls(
            pattern_config,
            enabled=pattern_enabled_from_project(project),
        )
        self.mode.setCurrentText(
            "Profile Dome" if preview_mode_from_project(project) == "profile-dome" else "Cylinder"
        )
        self._apply_node_graph_from_project(project.graph)
        self._render_scene()

    def _save_project_dialog(self, *, force_dialog: bool = False) -> None:
        start_path = (
            str(self._current_project_path)
            if self._current_project_path is not None
            else "cylinder_project.fwp.json"
        )
        if self._current_project_path is not None and not force_dialog:
            filename = str(self._current_project_path)
        else:
            filename, _ = self._qt_widgets.QFileDialog.getSaveFileName(
                self.widget,
                "Export FilamentWinder project",
                start_path,
                "FilamentWinder Project (*.fwp.json);;JSON Files (*.json);;All Files (*)",
            )
            if not filename:
                return
        project = project_from_preview_config(
            self._current_config(),
            name=self.project_name.text(),
            profile_config=self._current_profile_config(),
            pattern_config=self._current_pattern_config(),
            pattern_enabled=self._is_pattern_planner_enabled(),
            preview_mode="profile-dome" if self._is_profile_dome_mode() else "cylinder",
            export_paths=self._current_export_paths(),
            feedrate_mm_min=float(self.feedrate.value()),
            graph=self._current_node_graph_data(),
        )
        try:
            saved_path = save_project(project, filename)
        except OSError as exc:
            self.status.setText(f"Could not save project: {exc}")
            return
        self._set_project_path(saved_path)
        self.status.setText(f"Saved project: {saved_path}")

    def _current_node_graph_data(self) -> dict[str, object]:
        if hasattr(self, "node_scene"):
            self._sync_node_graph_from_scene()
        self._log_graph_event("Project save graph state")
        return self._node_graph.to_dict()

    def _apply_node_graph_from_project(self, graph_data: dict[str, object]) -> None:
        if not graph_data:
            return
        try:
            self._node_graph = NodeGraphState.from_dict(graph_data, self._node_registry)
        except (KeyError, TypeError, ValueError) as exc:
            self._show_graph_error("Project load graph state failed", exc)
            return
        self._log_graph_event("Project load graph state")
        if hasattr(self, "node_scene"):
            self._redraw_node_graph()
            self._schedule_fit_node_graph()

    def _set_project_path(self, path: Path) -> None:
        self._current_project_path = path
        self.project_path_label.setText(str(path))

    def _choose_export_folder(self) -> None:
        folder = self._qt_widgets.QFileDialog.getExistingDirectory(
            self.widget,
            "Choose export folder",
            "exports",
        )
        if not folder:
            return
        prefix = _safe_output_prefix(self.project_name.text())
        self._apply_export_paths_to_controls(export_paths_from_directory(folder, prefix=prefix))
        self.status.setText(f"Export folder set: {folder}")

    def _inspect_profile_import(self) -> None:
        try:
            profile = import_dxf_zr_profile(
                self._current_profile_config().profile_path,
                samples=self._current_profile_config().samples,
            )
        except (OSError, ValueError) as exc:
            self.status.setText(f"DXF import failed: {exc}")
            return
        self.status.setText(
            "DXF import OK\n"
            f"Points: {profile.z_mm.size}\n"
            f"Z: {profile.start_z_mm:.3f}..{profile.end_z_mm:.3f} mm\n"
            f"Length: {profile.length_mm:.3f} mm\n"
            f"Max radius: {profile.max_radius_mm:.3f} mm"
        )

    def _browse_output_path(self, line_edit: Any, title: str, file_filter: str) -> None:
        filename, _ = self._qt_widgets.QFileDialog.getSaveFileName(
            self.widget,
            title,
            line_edit.text().strip(),
            file_filter,
        )
        if filename:
            line_edit.setText(filename)

    def _browse_input_path(self, line_edit: Any, title: str, file_filter: str) -> None:
        filename, _ = self._qt_widgets.QFileDialog.getOpenFileName(
            self.widget,
            title,
            line_edit.text().strip(),
            file_filter,
        )
        if filename:
            line_edit.setText(filename)

    def _on_mode_changed(self, *_args: Any) -> None:
        self._render_scene()

    def _export_current(
        self,
        *,
        csv: bool = False,
        gcode: bool = False,
        coverage_csv: bool = False,
        coverage_summary_csv: bool = False,
        preview_obj: bool = False,
    ) -> None:
        try:
            schedule = self._current_layer_schedule()
            if not self._is_profile_dome_mode() and schedule is not None:
                mandrel = self._current_custom_mandrel()
                program = plan_winding_schedule(mandrel, schedule)
                written: list[str] = []
                if csv:
                    csv_path = self._current_export_paths().csv_path
                    written.append(
                        str(export_winding_program_csv(program, csv_path))
                    )
                if gcode:
                    gcode_path = self._current_export_paths().gcode_path
                    written.append(
                        str(
                            export_gcode(
                                program.motion_table,
                                gcode_path,
                                options=GCodeOptions(
                                    feedrate_mm_min=float(self.feedrate.value()),
                                    feed_schedule=program.feed_schedule,
                                ),
                            )
                        )
                    )
                written_text = ", ".join(written)
                self.status.setText(
                    f"Exported custom layer stack: {written_text}"
                    if written_text
                    else "Nothing exported."
                )
                return
            if self._is_profile_dome_mode():
                if self._is_pattern_planner_enabled():
                    result = export_profile_dome_pattern_preview_files(
                        self._current_profile_config(),
                        self._current_pattern_config(),
                        self._current_export_paths(),
                        feedrate_mm_min=float(self.feedrate.value()),
                        csv=csv,
                        gcode=gcode,
                    )
                else:
                    result = export_profile_dome_preview_files(
                        self._current_profile_config(),
                        self._current_export_paths(),
                        feedrate_mm_min=float(self.feedrate.value()),
                        csv=csv,
                        gcode=gcode,
                    )
            elif self._is_pattern_planner_enabled():
                result = export_cylinder_pattern_preview_files(
                    self._current_config(),
                    self._current_pattern_config(),
                    self._current_export_paths(),
                    feedrate_mm_min=float(self.feedrate.value()),
                    csv=csv,
                    gcode=gcode,
                    coverage_csv=coverage_csv,
                    coverage_summary_csv=coverage_summary_csv,
                )
            else:
                result = export_preview_files(
                    self._current_config(),
                    self._current_export_paths(),
                    feedrate_mm_min=float(self.feedrate.value()),
                    csv=csv,
                    gcode=gcode,
                    coverage_csv=coverage_csv,
                    coverage_summary_csv=coverage_summary_csv,
                    preview_obj=preview_obj,
                )
        except (OSError, RuntimeError, ValueError) as exc:
            self.status.setText(f"Export failed: {exc}")
            return
        written_paths = ", ".join(str(path) for path in result.written_paths)
        skipped = (
            " Coverage and OBJ exports are cylinder-only."
            if self._is_profile_dome_mode()
            and (coverage_csv or coverage_summary_csv or preview_obj)
            else (
                " OBJ export is single-path cylinder-only."
                if self._is_pattern_planner_enabled() and preview_obj
                else ""
            )
        )
        self.status.setText(
            f"Exported: {written_paths}{skipped}"
            if written_paths
            else f"Nothing exported.{skipped}"
        )

    def _optimize_pattern(self) -> None:
        if self._is_profile_dome_mode():
            self.status.setText("Pattern optimization is currently cylinder-only")
            return
        try:
            result = optimize_cylinder_pattern(
                CylinderPatternOptimizationRequest(
                    length_mm=float(self.length.value()),
                    radius_mm=float(self.radius.value()),
                    tow_width_mm=float(self.tow_width.value()),
                    point_count=int(self.points.value()),
                    target_coverage_fraction=float(self.target_coverage.value()) / 100.0,
                    min_angle_deg=5.0,
                    max_angle_deg=85.0,
                    min_passes=1,
                    max_passes=int(self.max_opt_passes.value()),
                    preferred_angle_deg=float(self.angle.value()),
                    max_results=1,
                )
            )
        except ValueError as exc:
            self.status.setText(f"Optimization failed: {exc}")
            return
        if not result.candidates:
            self.status.setText("Optimization found no closed pattern candidates")
            return
        best = result.best
        self.angle.setValue(best.winding_angle_deg)
        self.passes.setValue(best.passes)
        self.auto_phase.setChecked(True)
        self.phase_offset.setValue(best.phase_offset_deg)
        self._render_scene()
        gap_label = "gap" if best.estimated_gap_overlap_mm >= 0.0 else "overlap"
        self.status.setText(
            f"Optimized: {best.winding_angle_deg:.3f} deg, "
            f"{best.passes} passes, {best.turns_per_pass} turns/pass, "
            f"{best.estimated_coverage_percent:.2f}% coverage, "
            f"{abs(best.estimated_gap_overlap_mm):.3f} mm {gap_label}"
        )

    def _on_auto_phase_toggled(self, checked: bool) -> None:
        self.phase_offset.setEnabled(not checked)
        if checked:
            self._update_auto_phase_value()

    def _update_auto_phase_value(self, *_args: Any) -> None:
        if self.auto_phase.isChecked():
            self.phase_offset.setValue(360.0 / max(1, int(self.passes.value())))

    def _reset_camera(self) -> None:
        if self._is_pattern_planner_enabled():
            try:
                if self._is_profile_dome_mode():
                    pattern_preview = build_profile_dome_pattern_preview_scene(
                        self._current_profile_config(),
                        self._current_pattern_config(),
                        feedrate_mm_min=float(self.feedrate.value()),
                    )
                else:
                    pattern_preview = build_cylinder_pattern_preview_scene(
                        self._current_config(),
                        self._current_pattern_config(),
                        feedrate_mm_min=float(self.feedrate.value()),
                    )
                radius_mm = _preview_radius_mm(pattern_preview.mandrel)
                scale = max(pattern_preview.mandrel.length_mm, radius_mm * 6.0)
            except (OSError, ValueError):
                scale = 1000.0
        elif self._is_profile_dome_mode():
            try:
                profile_preview = build_profile_dome_preview_scene(
                    self._current_profile_config()
                )
                scale = max(
                    profile_preview.profile.length_mm,
                    profile_preview.profile.max_radius_mm * 6.0,
                )
            except (OSError, ValueError):
                scale = 1000.0
        else:
            config = self._current_config()
            scale = max(config.length_mm, config.radius_mm * 6.0)
        self.view.camera.center = (0.0, 0.0, 0.0)
        self.view.camera.scale_factor = scale
        self.view.camera.distance = scale * 1.4

    def _on_mouse_press(self, event: Any) -> None:
        if event.button in (1, 2, 3):
            self._drag_last_pos = np.asarray(event.pos[:2], dtype=float)
            event.handled = True

    def _on_mouse_release(self, event: Any) -> None:
        self._drag_last_pos = None
        event.handled = True

    def _on_mouse_move(self, event: Any) -> None:
        if self._drag_last_pos is None or not event.buttons:
            return
        current_pos = np.asarray(event.pos[:2], dtype=float)
        previous_pos = self._drag_last_pos
        self._drag_last_pos = current_pos
        buttons = {int(button) for button in event.buttons}
        shift_pressed = self._vispy_keys.SHIFT in event.modifiers

        if 1 in buttons and shift_pressed or 3 in buttons:
            self._pan_camera(previous_pos, current_pos)
        elif 2 in buttons:
            self._zoom_camera((current_pos - previous_pos)[1])
        elif 1 in buttons:
            self._orbit_camera(previous_pos, current_pos)
        event.handled = True

    def _on_mouse_wheel(self, event: Any) -> None:
        self._scale_camera(1.1 ** -float(event.delta[1]))
        event.handled = True

    def _orbit_camera(self, previous_pos: np.ndarray, current_pos: np.ndarray) -> None:
        camera = self.view.camera
        viewbox_size = camera._viewbox.size
        camera._quaternion = (
            self._quaternion(*self._arcball(current_pos, viewbox_size))
            * self._quaternion(*self._arcball(previous_pos, viewbox_size))
            * camera._quaternion
        )
        camera.view_changed()

    def _pan_camera(self, previous_pos: np.ndarray, current_pos: np.ndarray) -> None:
        camera = self.view.camera
        norm = np.mean(camera._viewbox.size)
        if norm <= 0.0:
            return
        dist = (previous_pos - current_pos) / norm * camera.scale_factor
        dist[1] *= -1
        dx, dy, dz = camera._dist_to_trans(dist)
        flip = camera._flip_factors
        up, forward, right = camera._get_dim_vectors()
        dx, dy, dz = right * dx + forward * dy + up * dz
        dx, dy, dz = flip[0] * dx, flip[1] * dy, dz * flip[2]
        center = camera.center
        camera.center = (
            center[0] + dx,
            center[1] + dy,
            center[2] + dz,
        )
        camera.view_changed()

    def _zoom_camera(self, pixel_delta_y: float) -> None:
        self._scale_camera((1.0 + self.view.camera.zoom_factor) ** pixel_delta_y)

    def _scale_camera(self, factor: float) -> None:
        if not np.isfinite(factor) or factor <= 0.0:
            return
        camera = self.view.camera
        camera.scale_factor = max(camera.scale_factor * factor, 1e-6)
        if camera.distance is not None:
            camera.distance = max(camera.distance * factor, 1e-6)
        camera.view_changed()


def _display_phase_offset(config: CylinderPreviewConfig) -> float:
    if config.phase_offset_deg is not None:
        return config.phase_offset_deg
    return 360.0 / max(1, config.passes)


def _preview_radius_mm(mandrel: Any) -> float:
    if hasattr(mandrel, "max_radius_mm"):
        return float(mandrel.max_radius_mm)
    return float(mandrel.radius_mm)


def _node_status_color(status: str) -> str:
    colors = {
        "not_configured": "#7d8790",
        "ready": "#4f9f72",
        "warning": "#c89438",
        "error": "#d65757",
        "processing": "#3b82c4",
        "complete": "#51a36f",
        "dirty": "#c89438",
    }
    return colors.get(status, "#7d8790")


def _socket_kind_color(kind: str) -> str:
    colors = {
        "mandrel": "#2f9fcf",
        "tow": "#9b6fd3",
        "machine": "#c89438",
        "layer_stack": "#5fb56c",
        "program": "#55b3ad",
        "coverage": "#8cbf4f",
        "simulation": "#8f8de0",
        "export": "#d08a45",
        "any": "#9aa7b2",
    }
    return colors.get(kind, "#9aa7b2")


def _profile_path_mode_label(path_mode: ProfilePathMode) -> str:
    labels = {
        "dome": "Dome (geodesic)",
        "nosecone": "Nosecone",
        "axisymmetric": "Axisymmetric",
    }
    return labels[path_mode]


def _profile_path_mode_from_label(label: str) -> ProfilePathMode:
    if label.startswith("Nosecone"):
        return "nosecone"
    if label.startswith("Axisymmetric"):
        return "axisymmetric"
    return "dome"


def _safe_output_prefix(name: str) -> str:
    clean = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in name)
    return clean.strip("_") or "winding"


def _setting_label(name: str) -> str:
    return name.replace("_", " ").strip().capitalize()


def _modern_stylesheet() -> str:
    return """
    QWidget {
        background: #101418;
        color: #e8edf2;
        font-size: 12px;
    }
    QWidget#controlPanel {
        background: #151b22;
        border-right: 1px solid #29323d;
    }
    QWidget#nodeSidePanel {
        background: #111820;
        border: 1px solid #29323d;
        border-radius: 6px;
    }
    QWidget#nodeBottomPanel {
        background: #111820;
        border: 1px solid #29323d;
        border-radius: 6px;
    }
    QTabWidget#nodeBottomTabs::pane {
        border: 1px solid #29323d;
        background: #101820;
    }
    QGraphicsView#nodeView {
        background: #0b1015;
        border: 1px solid #29323d;
        border-radius: 6px;
    }
    QGroupBox {
        border: 1px solid #2d3742;
        border-radius: 6px;
        margin-top: 10px;
        padding: 8px;
        background: #171e26;
        font-weight: 600;
    }
    QGroupBox#nodeCard {
        border: 1px solid #3a5063;
        border-radius: 7px;
        margin-top: 14px;
        background: #151c24;
    }
    QGroupBox#nodeCard::title {
        background: #213142;
        border: 1px solid #3a5063;
        border-radius: 5px;
        subcontrol-origin: margin;
        left: 8px;
        padding: 3px 8px;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 8px;
        padding: 0 4px;
    }
    QTabWidget::pane {
        border: 1px solid #2d3742;
        border-radius: 6px;
        background: #111820;
    }
    QTabBar::tab {
        background: #202a34;
        border: 1px solid #2d3742;
        padding: 7px 10px;
        margin-right: 2px;
        border-top-left-radius: 5px;
        border-top-right-radius: 5px;
    }
    QTabBar::tab:selected {
        background: #2b6cb0;
        border-color: #3b82c4;
    }
    QLineEdit, QPlainTextEdit, QListWidget, QDoubleSpinBox, QSpinBox, QComboBox {
        background: #0f141a;
        border: 1px solid #33404d;
        border-radius: 4px;
        padding: 4px;
        selection-background-color: #2b6cb0;
    }
    QPushButton {
        background: #263341;
        border: 1px solid #3a4a5a;
        border-radius: 5px;
        padding: 6px 10px;
    }
    QPushButton:hover {
        background: #314255;
    }
    QPushButton:pressed {
        background: #1f6fb2;
    }
    QScrollArea {
        background: transparent;
    }
    """
