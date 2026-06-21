"""Optional PySide6/VisPy live preview."""

from __future__ import annotations

import importlib
import importlib.util
import json
import logging
import math
import sys
import time
import traceback
from collections.abc import Callable
from contextlib import suppress
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np

from filament_winder.app.backend_service import (
    BackendCheckResult,
    BackendService,
    LoadedPlotSet,
    LoadedReportSet,
)
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
    default_backend_winding_graph,
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
from filament_winder.config import WindingJobConfig
from filament_winder.core.geometry import (
    AxisymmetricProfileMandrel,
    CylinderMandrel,
    cylinder_with_domes_profile,
)
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
_ORIGINAL_EXCEPTHOOK = sys.excepthook
NODE_CANVAS_SCENE_RECT = (-10000.0, -6000.0, 20000.0, 12000.0)


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


def install_gui_exception_hook(logger: logging.Logger | None = None) -> None:
    """Install a last-resort hook so uncaught GUI errors are logged."""

    resolved_logger = _gui_logger() if logger is None else logger

    def _hook(exc_type: type[BaseException], exc: BaseException, tb: Any) -> None:
        resolved_logger.critical(
            "Unhandled GUI exception\n%s",
            "".join(traceback.format_exception(exc_type, exc, tb)),
        )
        _ORIGINAL_EXCEPTHOOK(exc_type, exc, tb)

    sys.excepthook = _hook


def _create_qapplication(qt_widgets: Any, argv: list[str], logger: logging.Logger) -> Any:
    existing = qt_widgets.QApplication.instance()
    if existing is not None:
        return existing

    class _SafeApplication(qt_widgets.QApplication):  # type: ignore[name-defined, misc, valid-type]
        def notify(self, receiver: Any, event: Any) -> bool:  # noqa: N802
            try:
                return bool(super().notify(receiver, event))
            except Exception:  # noqa: BLE001 - final GUI safety net
                logger.critical("Unhandled Qt event exception\n%s", traceback.format_exc())
                return False

    return _SafeApplication(argv)


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
    debug_gui: bool = False,
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
    logger = _gui_logger()
    if debug_gui:
        logger.setLevel(logging.DEBUG)
    install_gui_exception_hook(logger)
    app = _create_qapplication(qt_widgets, sys.argv[:1], logger)
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
        self._backend_service = BackendService()
        self._backend_busy = False
        self._closing = False
        self._safe_buttons: list[Any] = []
        self._last_backend_check: BackendCheckResult | None = None
        self._last_loaded_reports: LoadedReportSet | None = None
        self._last_loaded_plots: LoadedPlotSet | None = None
        self._current_plot_path: Path | None = None
        self._config = config
        self._profile_config = profile_config
        self._visuals: list[Any] = []
        self._drag_last_pos: np.ndarray | None = None
        self._current_project_path: Path | None = None
        self._default_export_paths = PreviewExportPaths()
        self._node_registry = default_node_registry()
        self._node_graph = default_backend_winding_graph()
        self._node_items: dict[str, Any] = {}
        self._socket_items: dict[tuple[str, str, str], Any] = {}
        self._node_link_items: list[Any] = []
        self._node_group_items: list[Any] = []
        self._retired_node_scenes: list[Any] = []
        self._node_inline_widgets: list[Any] = []
        self._refreshing_node_links = False
        self._node_link_refresh_queued = False
        self._node_position_sync_queued = False
        self._redrawing_node_graph = False
        self._applying_node_selection = False
        self._node_setting_render_pending = False
        self._node_graph_redraw_queued = False
        self._queued_node_selection: tuple[str, ...] = ()
        self._queued_link_selection: tuple[str, ...] = ()
        self._selected_node_id_cache: tuple[str, ...] = ()
        self._selected_link_id_cache: tuple[str, ...] = ()
        self._node_socket_drag: dict[str, str] | None = None
        self._node_temp_link_item: Any | None = None
        self._node_highlight_socket: Any | None = None
        self._last_node_result: GraphExecutionResult | None = None
        self._viewport_node_context: str | None = None
        self._node_thread_pool = qt_core.QThreadPool.globalInstance()
        self._node_workers: list[Any] = []
        self._task_started_at: float | None = None
        self._task_total_steps = 0
        self._task_completed_steps = 0
        self._task_name = ""
        self._task_progress_ticks = 0
        self._task_progress_timer = qt_core.QTimer()
        self._task_progress_timer.setInterval(500)
        self._task_progress_timer.timeout.connect(self._tick_task_progress)
        self._node_zoom = 1.0
        self._node_panning = False
        self._node_pan_last_pos: Any | None = None
        self._node_right_press_pos: Any | None = None
        self._node_suppress_context_menu = False
        self._node_space_down = False

        class _PreviewMainWindow(qt_widgets.QMainWindow):  # type: ignore[name-defined, misc, valid-type]
            def __init__(self, owner: _PreviewWindow) -> None:
                super().__init__()
                self._owner = owner

            def closeEvent(self, event: Any) -> None:  # noqa: N802
                self._owner._cleanup_node_scenes_for_close()
                super().closeEvent(event)

        self.widget = _PreviewMainWindow(self)
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
        import_config = qt_widgets.QPushButton("Import Config")
        self._connect_safe_button(
            import_config,
            "Import Backend Config",
            self._import_backend_config_dialog,
        )
        export_config = qt_widgets.QPushButton("Export Config")
        self._connect_safe_button(
            export_config,
            "Export Backend Config",
            self._export_backend_config_dialog,
        )
        backend_check = qt_widgets.QPushButton("Backend Check")
        self._connect_safe_button(backend_check, "Backend Check", self._run_backend_check)
        backend_csv = qt_widgets.QPushButton("Backend CSV")
        self._connect_safe_button(backend_csv, "Backend CSV", self._run_backend_csv_export)
        backend_gcode = qt_widgets.QPushButton("Backend G-code")
        self._connect_safe_button(
            backend_gcode,
            "Backend G-code",
            self._run_backend_gcode_export,
        )
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
        toolbar.addWidget(import_config)
        toolbar.addWidget(export_config)
        toolbar.addWidget(backend_check)
        toolbar.addWidget(backend_csv)
        toolbar.addWidget(backend_gcode)
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

        selected_count = len(self._selected_node_ids()) if hasattr(self, "node_scene") else 0
        self._logger.info("Button clicked: %s", action_name)
        self._logger.info("Action started: %s selected_nodes=%s", action_name, selected_count)
        self._append_gui_log(action_name, "started", f"selected_nodes={selected_count}")
        previous_status = self.status.text() if hasattr(self, "status") else ""
        self._set_gui_status(f"Running: {action_name}")
        try:
            result = callback()
        except Exception as exc:  # noqa: BLE001 - protects Qt event loop
            self.handle_gui_error(action_name, exc)
            return None
        self._logger.info("Action completed: %s", action_name)
        self._append_gui_log(action_name, "complete", "")
        if hasattr(self, "status") and self.status.text() in {
            f"Running: {action_name}",
            previous_status,
        }:
            self._set_gui_status(f"Complete: {action_name}")
        if hasattr(self, "node_status_log"):
            self.node_status_log.appendPlainText(f"Complete: {action_name}")
        return result

    def _run_button_action(
        self,
        button: Any,
        action_name: str,
        callback: Callable[[], Any],
    ) -> Any | None:
        button.setEnabled(False)
        try:
            return self.run_safe_action(action_name, callback)
        finally:
            if not self._backend_busy:
                button.setEnabled(True)

    def handle_gui_error(self, action_name: str, exc: BaseException) -> None:
        message = f"{action_name} failed: {exc}"
        self._logger.error("Action failed: %s\n%s", action_name, traceback.format_exc())
        self._append_gui_log(action_name, "failed", str(exc), traceback.format_exc())
        self._set_gui_status(message)
        for node_id in getattr(self._node_graph, "selected_node_ids", ()):
            if node_id in self._node_graph.nodes:
                self._node_graph.nodes[node_id].status = "failed"
                self._node_graph.nodes[node_id].message = str(exc)
        if hasattr(self, "node_status"):
            self.node_status.setText(message)
        if hasattr(self, "node_status_log"):
            self.node_status_log.appendPlainText(message)
        if hasattr(self, "node_debug_log"):
            self.node_debug_log.appendPlainText(message)
            self.node_debug_log.appendPlainText(traceback.format_exc())
        if hasattr(self, "node_scene"):
            self._redraw_node_graph()

    def _append_gui_log(
        self,
        action_name: str,
        status: str,
        message: str,
        traceback_text: str = "",
    ) -> None:
        if not hasattr(self, "node_status_log"):
            return
        timestamp = self._qt_core.QDateTime.currentDateTime().toString("yyyy-MM-dd HH:mm:ss")
        self.node_status_log.appendPlainText(
            f"{timestamp} | {action_name} | {status} | {message}".rstrip()
        )
        if traceback_text and hasattr(self, "node_debug_log"):
            self.node_debug_log.appendPlainText(
                f"{timestamp} | {action_name} | {status}\n{traceback_text}"
            )

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
        self._safe_buttons.append(button)
        button.clicked.connect(
            lambda _checked=False,
            clicked=button,
            name=action_name,
            cb=callback: self._run_button_action(
                clicked,
                name,
                cb,
            )
        )

    def _set_backend_busy(self, busy: bool, message: str) -> None:
        self._backend_busy = busy
        live_buttons = []
        for button in getattr(self, "_safe_buttons", []):
            try:
                button.setEnabled(not busy)
            except RuntimeError:
                continue
            live_buttons.append(button)
        self._safe_buttons = live_buttons
        self._set_gui_status(message)

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
        tabs.addTab(self._scroll_tab(winding_group, layer_group, profile_group), "Path")
        tabs.addTab(self._scroll_tab(pattern_group), "Pattern")
        tabs.addTab(self._scroll_tab(export_group), "Export")

        tabs.setVisible(False)
        self._legacy_control_tabs = tabs
        nodes_tab.setSizePolicy(
            self._qt_widgets.QSizePolicy.Policy.Expanding,
            self._qt_widgets.QSizePolicy.Policy.Expanding,
        )
        layout.addWidget(nodes_tab, 1)
        return panel

    def _build_node_workspace(self) -> Any:
        qt_widgets = self._qt_widgets
        container = qt_widgets.QWidget()
        container.setObjectName("nodeWorkspace")
        container_layout = qt_widgets.QVBoxLayout(container)
        container_layout.setContentsMargins(8, 8, 8, 8)
        container_layout.setSpacing(8)

        toolbar_widget = qt_widgets.QWidget()
        toolbar_widget.setObjectName("nodeToolbar")
        toolbar = qt_widgets.QHBoxLayout(toolbar_widget)
        toolbar.setContentsMargins(6, 6, 6, 6)
        toolbar.setSpacing(8)
        add_nodes = qt_widgets.QPushButton("Add Node")
        self._connect_safe_button(add_nodes, "Add Node", self._add_selected_node_type)
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
        ungroup_nodes = qt_widgets.QPushButton("Ungroup")
        self._connect_safe_button(ungroup_nodes, "Ungroup Nodes", self._ungroup_selected_nodes)
        move_left = qt_widgets.QPushButton("Left")
        self._connect_safe_button(
            move_left,
            "Move Node Left",
            lambda: self._move_selected_nodes(-40.0, 0.0),
        )
        move_right = qt_widgets.QPushButton("Right")
        self._connect_safe_button(
            move_right,
            "Move Node Right",
            lambda: self._move_selected_nodes(40.0, 0.0),
        )
        move_up = qt_widgets.QPushButton("Up")
        self._connect_safe_button(
            move_up,
            "Move Node Up",
            lambda: self._move_selected_nodes(0.0, -40.0),
        )
        move_down = qt_widgets.QPushButton("Down")
        self._connect_safe_button(
            move_down,
            "Move Node Down",
            lambda: self._move_selected_nodes(0.0, 40.0),
        )
        align_h = qt_widgets.QPushButton("Align H")
        self._connect_safe_button(
            align_h,
            "Align Horizontal",
            self._align_selected_nodes_horizontally,
        )
        align_v = qt_widgets.QPushButton("Align V")
        self._connect_safe_button(align_v, "Align Vertical", self._align_selected_nodes_vertically)
        distribute_h = qt_widgets.QPushButton("Dist H")
        self._connect_safe_button(
            distribute_h,
            "Distribute Horizontal",
            self._distribute_selected_nodes_horizontally,
        )
        distribute_v = qt_widgets.QPushButton("Dist V")
        self._connect_safe_button(
            distribute_v,
            "Distribute Vertical",
            self._distribute_selected_nodes_vertically,
        )
        fit_nodes = qt_widgets.QPushButton("Fit")
        self._connect_safe_button(fit_nodes, "Fit Node Graph", self._fit_node_graph)
        zoom_out = qt_widgets.QPushButton("Zoom -")
        self._connect_safe_button(zoom_out, "Zoom Out", lambda: self._zoom_node_graph(0.85))
        self.node_zoom_label = qt_widgets.QLabel("Zoom: 100%")
        self.node_zoom_label.setMinimumWidth(86)
        zoom_in = qt_widgets.QPushButton("Zoom +")
        self._connect_safe_button(zoom_in, "Zoom In", lambda: self._zoom_node_graph(1.18))
        reset_zoom = qt_widgets.QPushButton("Reset View")
        self._connect_safe_button(reset_zoom, "Reset Node View", self._reset_node_graph_view)
        center_selected = qt_widgets.QPushButton("Center Sel")
        self._connect_safe_button(center_selected, "Center Selected", self._frame_selected_nodes)
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
        update_everything = qt_widgets.QPushButton("Update Everything")
        update_everything.setToolTip("Rebuild config, paths, reports, plots, CSV, and G-code.")
        self._connect_safe_button(
            update_everything,
            "Update Everything",
            self._update_everything,
        )
        optimize_pattern = qt_widgets.QPushButton("Optimize Pattern")
        self._connect_safe_button(optimize_pattern, "Optimise Pattern", self._optimize_pattern)
        toolbar.addWidget(add_nodes)
        toolbar.addWidget(link_nodes)
        toolbar.addWidget(unlink_nodes)
        toolbar.addWidget(duplicate_nodes)
        toolbar.addWidget(delete_nodes)
        toolbar.addWidget(collapse_nodes)
        toolbar.addWidget(expand_nodes)
        toolbar.addWidget(group_nodes)
        toolbar.addWidget(ungroup_nodes)
        toolbar.addWidget(move_left)
        toolbar.addWidget(move_right)
        toolbar.addWidget(move_up)
        toolbar.addWidget(move_down)
        toolbar.addWidget(align_h)
        toolbar.addWidget(align_v)
        toolbar.addWidget(distribute_h)
        toolbar.addWidget(distribute_v)
        toolbar.addWidget(fit_nodes)
        toolbar.addWidget(zoom_out)
        toolbar.addWidget(self.node_zoom_label)
        toolbar.addWidget(zoom_in)
        toolbar.addWidget(reset_zoom)
        toolbar.addWidget(center_selected)
        toolbar.addStretch(1)
        toolbar.addWidget(run_selected)
        toolbar.addWidget(run_branch)
        toolbar.addWidget(run_graph)
        toolbar.addWidget(export_graph)
        toolbar.addWidget(update_everything)
        toolbar.addWidget(optimize_pattern)
        toolbar_scroll = qt_widgets.QScrollArea()
        toolbar_scroll.setObjectName("nodeToolbarScroll")
        toolbar_scroll.setWidgetResizable(True)
        toolbar_scroll.setHorizontalScrollBarPolicy(
            self._qt_core.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        toolbar_scroll.setVerticalScrollBarPolicy(self._qt_core.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        toolbar_scroll.setFrameShape(qt_widgets.QFrame.Shape.NoFrame)
        toolbar_scroll.setMaximumHeight(52)
        toolbar_scroll.setWidget(toolbar_widget)
        container_layout.addWidget(toolbar_scroll)

        workspace_splitter = qt_widgets.QSplitter(self._qt_core.Qt.Orientation.Horizontal)
        workspace_splitter.setChildrenCollapsible(False)

        graph_panel = qt_widgets.QWidget()
        graph_layout = qt_widgets.QVBoxLayout(graph_panel)
        graph_layout.setContentsMargins(0, 0, 0, 0)
        graph_layout.setSpacing(0)

        self.node_scene = qt_widgets.QGraphicsScene()
        self.node_scene.setSceneRect(*NODE_CANVAS_SCENE_RECT)
        self._connect_node_scene_signals(self.node_scene)

        class _NodeGraphicsView(qt_widgets.QGraphicsView):  # type: ignore[name-defined, misc, valid-type]
            def __init__(self, scene: Any, owner: _PreviewWindow) -> None:
                super().__init__(scene)
                self._owner = owner

            def drawBackground(self, painter: Any, rect: Any) -> None:  # noqa: N802
                super().drawBackground(painter, rect)
                minor = 24.0
                major = minor * 5.0
                left = int(rect.left()) - (int(rect.left()) % int(minor))
                top = int(rect.top()) - (int(rect.top()) % int(minor))
                minor_pen = self._owner._qt_gui.QPen(
                    self._owner._qt_gui.QColor(32, 43, 54),
                    1.0,
                )
                major_pen = self._owner._qt_gui.QPen(
                    self._owner._qt_gui.QColor(48, 62, 76),
                    1.0,
                )
                x = float(left)
                while x < rect.right():
                    painter.setPen(major_pen if abs(x % major) < 1e-6 else minor_pen)
                    painter.drawLine(
                        self._owner._qt_core.QPointF(x, rect.top()),
                        self._owner._qt_core.QPointF(x, rect.bottom()),
                    )
                    x += minor
                y = float(top)
                while y < rect.bottom():
                    painter.setPen(major_pen if abs(y % major) < 1e-6 else minor_pen)
                    painter.drawLine(
                        self._owner._qt_core.QPointF(rect.left(), y),
                        self._owner._qt_core.QPointF(rect.right(), y),
                    )
                    y += minor

        self.node_view = _NodeGraphicsView(self.node_scene, self)
        self.node_view.setObjectName("nodeView")
        self.node_view.setMinimumHeight(240)
        self.node_view.setMinimumWidth(680)
        self.node_view.setRenderHint(self._qt_gui.QPainter.RenderHint.Antialiasing, True)
        self.node_view.setViewportUpdateMode(
            qt_widgets.QGraphicsView.ViewportUpdateMode.FullViewportUpdate
        )
        self.node_view.setCacheMode(qt_widgets.QGraphicsView.CacheModeFlag.CacheNone)
        self.node_view.setDragMode(qt_widgets.QGraphicsView.DragMode.RubberBandDrag)
        self.node_view.setTransformationAnchor(
            qt_widgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse
        )
        self.node_view.setContextMenuPolicy(
            self._qt_core.Qt.ContextMenuPolicy.CustomContextMenu
        )
        self.node_view.customContextMenuRequested.connect(
            lambda pos: self.run_safe_action(
                "Node Context Menu",
                lambda: self._show_node_context_menu(pos),
            )
        )
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
        self.node_inspector_name.editingFinished.connect(
            lambda: self.run_safe_action("Apply Node Name", self._apply_node_inspector)
        )
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
        self.task_progress = qt_widgets.QProgressBar()
        self.task_progress.setObjectName("taskProgress")
        self.task_progress.setRange(0, 100)
        self.task_progress.setValue(0)
        self.task_progress.setTextVisible(False)
        self.task_progress.setMaximumWidth(220)
        self.task_time_label = qt_widgets.QLabel("Idle")
        self.task_time_label.setObjectName("taskTimeLabel")
        progress_layout = qt_widgets.QHBoxLayout()
        progress_layout.setContentsMargins(0, 0, 0, 0)
        progress_layout.addWidget(self.task_progress)
        progress_layout.addWidget(self.task_time_label, 1)
        self.node_status_log = qt_widgets.QPlainTextEdit()
        self.node_status_log.setReadOnly(True)
        status_layout.addWidget(self.node_status)
        status_layout.addLayout(progress_layout)
        status_layout.addWidget(self.node_status_log, 1)

        debug_panel = qt_widgets.QWidget()
        debug_panel.setObjectName("nodeBottomPanel")
        debug_layout = qt_widgets.QVBoxLayout(debug_panel)
        debug_layout.setContentsMargins(8, 8, 8, 8)
        debug_layout.setSpacing(8)
        self.node_debug_log = qt_widgets.QPlainTextEdit()
        self.node_debug_log.setReadOnly(True)
        self.node_debug_log.setPlaceholderText(f"Debug log: {GUI_LOG_PATH}")
        debug_actions = qt_widgets.QHBoxLayout()
        copy_log = qt_widgets.QPushButton("Copy Log")
        self._connect_safe_button(copy_log, "Copy Log", self._copy_gui_log)
        clear_log = qt_widgets.QPushButton("Clear Log")
        self._connect_safe_button(clear_log, "Clear Log", self._clear_gui_log)
        open_log = qt_widgets.QPushButton("Open Log File")
        self._connect_safe_button(open_log, "Open Log File", self._open_gui_log_file)
        debug_actions.addWidget(copy_log)
        debug_actions.addWidget(clear_log)
        debug_actions.addWidget(open_log)
        debug_actions.addStretch(1)
        debug_layout.addLayout(debug_actions)
        debug_layout.addWidget(self.node_debug_log, 1)

        reports_panel = qt_widgets.QWidget()
        reports_panel.setObjectName("nodeBottomPanel")
        reports_layout = qt_widgets.QVBoxLayout(reports_panel)
        reports_layout.setContentsMargins(8, 8, 8, 8)
        reports_layout.setSpacing(8)
        report_actions = qt_widgets.QHBoxLayout()
        refresh_reports = qt_widgets.QPushButton("Refresh Reports")
        self._connect_safe_button(
            refresh_reports,
            "Refresh Reports",
            self._refresh_backend_artifacts,
        )
        report_actions.addWidget(refresh_reports)
        report_actions.addStretch(1)
        self.backend_report_summary = qt_widgets.QLabel("Backend check has not run.")
        self.backend_report_summary.setWordWrap(True)
        self.report_list = qt_widgets.QListWidget()
        self.report_list.currentItemChanged.connect(
            lambda current, _previous: self.run_safe_action(
                "Open Report",
                lambda: self._show_report_item(current),
            )
        )
        self.report_detail = qt_widgets.QPlainTextEdit()
        self.report_detail.setReadOnly(True)
        reports_layout.addLayout(report_actions)
        reports_layout.addWidget(self.backend_report_summary)
        reports_layout.addWidget(self.report_list, 1)
        reports_layout.addWidget(self.report_detail, 2)

        plots_panel = qt_widgets.QWidget()
        plots_panel.setObjectName("nodeBottomPanel")
        plots_layout = qt_widgets.QVBoxLayout(plots_panel)
        plots_layout.setContentsMargins(8, 8, 8, 8)
        plots_layout.setSpacing(8)
        plot_actions = qt_widgets.QHBoxLayout()
        refresh_plots = qt_widgets.QPushButton("Refresh Plots")
        self._connect_safe_button(refresh_plots, "Refresh Plots", self._refresh_backend_artifacts)
        fit_plot = qt_widgets.QPushButton("Fit Plot")
        self._connect_safe_button(fit_plot, "Fit Plot", self._fit_current_plot)
        open_plot = qt_widgets.QPushButton("Open Plot")
        self._connect_safe_button(open_plot, "Open Plot", self._open_current_plot_external)
        plot_actions.addWidget(refresh_plots)
        plot_actions.addWidget(fit_plot)
        plot_actions.addWidget(open_plot)
        plot_actions.addStretch(1)
        self.plot_list = qt_widgets.QListWidget()
        self.plot_list.currentItemChanged.connect(
            lambda current, _previous: self.run_safe_action(
                "Open Plot",
                lambda: self._show_plot_item(current),
            )
        )
        self.plot_preview = qt_widgets.QLabel("No plot selected")
        self.plot_preview.setAlignment(self._qt_core.Qt.AlignmentFlag.AlignCenter)
        self.plot_preview.setMinimumHeight(220)
        self.plot_preview.setScaledContents(False)
        plot_scroll = qt_widgets.QScrollArea()
        plot_scroll.setWidgetResizable(True)
        plot_scroll.setWidget(self.plot_preview)
        plots_layout.addLayout(plot_actions)
        plots_layout.addWidget(self.plot_list, 1)
        plots_layout.addWidget(plot_scroll, 3)

        bottom_tabs.addTab(library_panel, "Node Library")
        bottom_tabs.addTab(inspector_panel, "Inspector")
        bottom_tabs.addTab(status_panel, "Execution / Status")
        bottom_tabs.addTab(reports_panel, "Reports")
        bottom_tabs.addTab(plots_panel, "Plots")
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
            self._set_node_status("No nodes to fit")
            return
        self.node_view.fitInView(bounds, self._qt_core.Qt.AspectRatioMode.KeepAspectRatio)
        fitted_zoom = float(self.node_view.transform().m11())
        self._node_zoom = self._bounded_node_zoom(fitted_zoom)
        if abs(fitted_zoom - self._node_zoom) > 1e-9:
            correction = self._node_zoom / max(fitted_zoom, 1e-9)
            self.node_view.scale(correction, correction)
        self._update_node_zoom_label()
        self._sync_node_view_state()

    def _schedule_fit_node_graph(self) -> None:
        if hasattr(self, "node_view"):
            self._qt_core.QTimer.singleShot(0, self._restore_node_view_state)

    def _scale_node_graph(self, factor: float) -> None:
        self._zoom_node_graph(factor)

    def _zoom_node_graph(self, factor: float) -> None:
        target_zoom = self._bounded_node_zoom(self._node_zoom * factor)
        if abs(target_zoom - self._node_zoom) < 1e-9:
            return
        scale_factor = target_zoom / max(self._node_zoom, 1e-9)
        self.node_view.scale(scale_factor, scale_factor)
        self._node_zoom = target_zoom
        self._update_node_zoom_label()
        self._sync_node_view_state()

    def _bounded_node_zoom(self, value: float) -> float:
        return max(0.25, min(3.0, value))

    def _reset_node_graph_view(self) -> None:
        center = self.node_view.mapToScene(self.node_view.viewport().rect().center())
        self.node_view.resetTransform()
        self._node_zoom = 1.0
        self.node_view.centerOn(center)
        self._update_node_zoom_label()
        self._sync_node_view_state()

    def _update_node_zoom_label(self) -> None:
        if hasattr(self, "node_zoom_label"):
            self.node_zoom_label.setText(f"Zoom: {self._node_zoom * 100.0:.0f}%")

    def _sync_node_view_state(self) -> None:
        if not hasattr(self, "node_view"):
            return
        center = self.node_view.mapToScene(self.node_view.viewport().rect().center())
        self._node_graph.view_zoom = self._node_zoom
        self._node_graph.view_center_x = float(center.x())
        self._node_graph.view_center_y = float(center.y())

    def _restore_node_view_state(self) -> None:
        if not hasattr(self, "node_view"):
            return
        self.node_view.resetTransform()
        self._node_zoom = self._bounded_node_zoom(float(self._node_graph.view_zoom))
        self.node_view.scale(self._node_zoom, self._node_zoom)
        self.node_view.centerOn(
            self._qt_core.QPointF(
                float(self._node_graph.view_center_x),
                float(self._node_graph.view_center_y),
            )
        )
        self._update_node_zoom_label()

    def _graph_controller(self) -> NodeGraphController:
        return NodeGraphController(self._node_graph, self._node_registry)

    def _set_node_status(self, message: str) -> None:
        if hasattr(self, "node_status"):
            self.node_status.setText(message)
        if hasattr(self, "node_status_log"):
            self.node_status_log.appendPlainText(message)

    def _start_task_progress(self, name: str, *, total_steps: int = 0) -> None:
        self._task_started_at = time.monotonic()
        self._task_total_steps = max(0, int(total_steps))
        self._task_completed_steps = 0
        self._task_progress_ticks = 0
        self._task_name = name
        if not hasattr(self, "task_progress"):
            return
        if self._task_total_steps:
            self.task_progress.setRange(0, self._task_total_steps)
            self.task_progress.setValue(0)
        else:
            self.task_progress.setRange(0, 100)
            self.task_progress.setValue(3)
        if hasattr(self, "_task_progress_timer"):
            self._task_progress_timer.start()
        self._update_task_progress_label()

    def _tick_task_progress(self) -> None:
        if not self._backend_busy and not self._task_name:
            return
        self._task_progress_ticks += 1
        if hasattr(self, "task_progress") and not self._task_total_steps:
            current = self.task_progress.value()
            next_value = min(95, max(current + 1, int(8 + self._task_progress_ticks * 1.5)))
            self.task_progress.setValue(next_value)
        self._update_task_progress_label()

    def _update_task_progress(self, completed_steps: int | None = None) -> None:
        if completed_steps is not None:
            self._task_completed_steps = max(0, int(completed_steps))
        elif self._task_total_steps:
            self._task_completed_steps = min(
                self._task_total_steps,
                self._task_completed_steps + 1,
            )
        if hasattr(self, "task_progress") and self._task_total_steps:
            self.task_progress.setValue(
                min(self._task_completed_steps, self._task_total_steps)
            )
        self._update_task_progress_label()

    def _finish_task_progress(self, message: str = "Complete") -> None:
        if hasattr(self, "_task_progress_timer"):
            self._task_progress_timer.stop()
        if hasattr(self, "task_progress"):
            self.task_progress.setRange(0, 100)
            self.task_progress.setValue(100)
        self._update_task_progress_label(done_message=message)
        self._task_name = ""

    def _fail_task_progress(self, message: str = "Failed") -> None:
        if hasattr(self, "_task_progress_timer"):
            self._task_progress_timer.stop()
        if hasattr(self, "task_progress"):
            self.task_progress.setRange(0, 100)
            self.task_progress.setValue(0)
        self._update_task_progress_label(done_message=message)
        self._task_name = ""

    def _update_task_progress_label(self, *, done_message: str | None = None) -> None:
        if not hasattr(self, "task_time_label"):
            return
        elapsed = (
            0.0
            if self._task_started_at is None
            else max(0.0, time.monotonic() - self._task_started_at)
        )
        task_name = self._task_name or "Task"
        if done_message is not None:
            self.task_time_label.setText(f"{task_name}: {done_message} in {elapsed:.1f}s")
            return
        if self._task_total_steps and self._task_completed_steps > 0:
            per_step = elapsed / max(1, self._task_completed_steps)
            remaining = max(0, self._task_total_steps - self._task_completed_steps) * per_step
            self.task_time_label.setText(
                f"{task_name}: {self._task_completed_steps}/{self._task_total_steps}, "
                f"{elapsed:.1f}s elapsed, ~{remaining:.1f}s left"
            )
        elif self._task_total_steps:
            self.task_time_label.setText(f"{task_name}: 0/{self._task_total_steps}")
        else:
            percent = self.task_progress.value() if hasattr(self, "task_progress") else 0
            self.task_time_label.setText(
                f"{task_name}: running {elapsed:.1f}s elapsed, progress {percent}%"
            )

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

    def _copy_gui_log(self) -> None:
        parts = []
        if hasattr(self, "node_status_log"):
            parts.append(self.node_status_log.toPlainText())
        if hasattr(self, "node_debug_log"):
            parts.append(self.node_debug_log.toPlainText())
        self._qt_widgets.QApplication.clipboard().setText("\n\n".join(parts).strip())
        self._set_node_status("GUI log copied to clipboard")

    def _clear_gui_log(self) -> None:
        if hasattr(self, "node_status_log"):
            self.node_status_log.clear()
        if hasattr(self, "node_debug_log"):
            self.node_debug_log.clear()
        self._set_node_status("GUI log panel cleared")

    def _open_gui_log_file(self) -> None:
        GUI_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        GUI_LOG_PATH.touch(exist_ok=True)
        self._qt_gui.QDesktopServices.openUrl(
            self._qt_core.QUrl.fromLocalFile(str(GUI_LOG_PATH.resolve()))
        )

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
                try:
                    return bool(self._owner._node_canvas_event_filter(obj, event))
                except Exception as exc:  # noqa: BLE001 - protects Qt event loop
                    self._owner.handle_gui_error("Node Canvas Event", exc)
                    return True

        self._node_canvas_event_filter_object = _NodeCanvasEventFilter(self)
        self.node_view.viewport().installEventFilter(self._node_canvas_event_filter_object)

    def _install_node_shortcuts(self) -> None:
        shortcuts = (
            ("Delete", "Delete Node", self._delete_selected_graph_items),
            ("Ctrl+D", "Duplicate Node", self._duplicate_selected_nodes),
            ("Ctrl+G", "Group Nodes", self._group_selected_nodes),
            ("Ctrl+Shift+G", "Ungroup Nodes", self._ungroup_selected_nodes),
            ("F", "Fit Node Graph", self._frame_selected_nodes),
            ("0", "Reset Node View", self._reset_node_graph_view),
            ("Ctrl+F", "Focus Node Search", lambda: self.node_search.setFocus()),
        )
        self._node_shortcuts = []
        for key_sequence, action_name, callback in shortcuts:
            shortcut = self._qt_gui.QShortcut(
                self._qt_gui.QKeySequence(key_sequence),
                self.node_view,
            )
            shortcut.activated.connect(
                lambda name=action_name, cb=callback: self.run_safe_action(name, cb)
            )
            self._node_shortcuts.append(shortcut)

    def _node_canvas_event_filter(self, _obj: Any, event: Any) -> bool:
        event_type = event.type()
        if event_type == self._qt_core.QEvent.Type.Wheel:
            return self._handle_node_wheel(event)
        if (
            event_type == self._qt_core.QEvent.Type.KeyPress
            and event.key() == self._qt_core.Qt.Key.Key_Space
        ):
            self._node_space_down = True
            event.accept()
            return True
        if (
            event_type == self._qt_core.QEvent.Type.KeyRelease
            and event.key() == self._qt_core.Qt.Key.Key_Space
        ):
            self._node_space_down = False
            event.accept()
            return True
        if (
            event_type == self._qt_core.QEvent.Type.MouseButtonPress
            and event.button() == self._qt_core.Qt.MouseButton.RightButton
        ):
            self._node_right_press_pos = self._event_view_pos(event)
            self._node_suppress_context_menu = False
            return False
        if (
            event_type == self._qt_core.QEvent.Type.MouseButtonPress
            and event.button() == self._qt_core.Qt.MouseButton.MiddleButton
        ):
            self._begin_node_pan(event)
            return True
        if (
            event_type == self._qt_core.QEvent.Type.MouseButtonPress
            and event.button() == self._qt_core.Qt.MouseButton.LeftButton
        ):
            if self._node_space_down:
                self._begin_node_pan(event)
                return True
            return self._begin_socket_drag(event)
        if event_type == self._qt_core.QEvent.Type.MouseMove and self._node_panning:
            self._update_node_pan(event)
            return True
        if (
            event_type == self._qt_core.QEvent.Type.MouseMove
            and self._node_right_press_pos is not None
            and event.buttons() & self._qt_core.Qt.MouseButton.RightButton
        ):
            current = self._event_view_pos(event)
            if (current - self._node_right_press_pos).manhattanLength() > 6:
                self._node_suppress_context_menu = True
                self._begin_node_pan(event)
                return True
        if event_type == self._qt_core.QEvent.Type.MouseButtonRelease and self._node_panning:
            self._finish_node_pan(event)
            self._node_right_press_pos = None
            return True
        if (
            event_type == self._qt_core.QEvent.Type.MouseButtonRelease
            and event.button() == self._qt_core.Qt.MouseButton.RightButton
        ):
            suppressed = self._node_suppress_context_menu
            self._node_right_press_pos = None
            return suppressed
        if (
            event_type == self._qt_core.QEvent.Type.MouseMove
            and self._node_socket_drag is not None
        ):
            self._update_socket_drag(event)
            return True
        if (
            event_type == self._qt_core.QEvent.Type.MouseMove
            and event.buttons() & self._qt_core.Qt.MouseButton.LeftButton
            and self._selected_node_ids()
        ):
            self._schedule_node_link_refresh()
            return False
        if (
            event_type == self._qt_core.QEvent.Type.MouseButtonRelease
            and self._node_socket_drag is not None
        ):
            self._finish_socket_drag(event)
            return True
        if (
            event_type == self._qt_core.QEvent.Type.MouseButtonRelease
            and event.button() == self._qt_core.Qt.MouseButton.LeftButton
            and self._selected_node_ids()
        ):
            self._schedule_node_link_refresh()
            self._schedule_node_position_sync()
            return False
        return False

    def _handle_node_wheel(self, event: Any) -> bool:
        delta = event.angleDelta().y()
        if delta == 0:
            return False
        factor = 1.15 if delta > 0 else 1.0 / 1.15
        before = self.node_view.mapToScene(self._event_view_pos(event))
        self._zoom_node_graph(factor)
        after = self.node_view.mapToScene(self._event_view_pos(event))
        movement = after - before
        self.node_view.translate(movement.x(), movement.y())
        self._sync_node_view_state()
        event.accept()
        return True

    def _begin_node_pan(self, event: Any) -> None:
        self._node_panning = True
        self._node_pan_last_pos = self._event_view_pos(event)
        self.node_view.setDragMode(self._qt_widgets.QGraphicsView.DragMode.NoDrag)
        self.node_view.setCursor(self._qt_core.Qt.CursorShape.ClosedHandCursor)
        event.accept()

    def _update_node_pan(self, event: Any) -> None:
        if self._node_pan_last_pos is None:
            return
        current = self._event_view_pos(event)
        delta = current - self._node_pan_last_pos
        self._node_pan_last_pos = current
        self.node_view.horizontalScrollBar().setValue(
            self.node_view.horizontalScrollBar().value() - delta.x()
        )
        self.node_view.verticalScrollBar().setValue(
            self.node_view.verticalScrollBar().value() - delta.y()
        )
        self._sync_node_view_state()
        event.accept()

    def _finish_node_pan(self, event: Any) -> None:
        self._node_panning = False
        self._node_pan_last_pos = None
        self.node_view.setDragMode(self._qt_widgets.QGraphicsView.DragMode.RubberBandDrag)
        self.node_view.unsetCursor()
        self._sync_node_view_state()
        event.accept()

    def _event_view_pos(self, event: Any) -> Any:
        if hasattr(event, "position"):
            return event.position().toPoint()
        return event.pos()

    def _begin_socket_drag(self, event: Any) -> bool:
        view_pos = self._event_view_pos(event)
        item = self._socket_item_at_view_pos(view_pos)
        if item is None:
            return False
        node_id = str(item.data(2))
        socket_name = str(item.data(3))
        side = str(item.data(4))
        self._node_socket_drag = {
            "node_id": node_id,
            "socket": socket_name,
            "side": side,
        }
        start = item.sceneBoundingRect().center()
        path = self._node_link_path(start, start)
        pen = self._qt_gui.QPen(self._qt_gui.QColor("#8fb8dc"), 2.0)
        pen.setStyle(self._qt_core.Qt.PenStyle.DashLine)
        self._node_temp_link_item = self.node_scene.addPath(path, pen)
        self._node_temp_link_item.setZValue(100.0)
        self._log_graph_event(f"Link creation started: {node_id}.{socket_name} ({side})")
        event.accept()
        return True

    def _update_socket_drag(self, event: Any) -> None:
        if self._node_socket_drag is None or self._node_temp_link_item is None:
            return
        view_pos = self._event_view_pos(event)
        scene_pos = self.node_view.mapToScene(view_pos)
        source_item = self._socket_items.get(
            (
                self._node_socket_drag["node_id"],
                self._node_socket_drag["socket"],
                self._node_socket_drag["side"],
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
            with suppress(RuntimeError):
                self.node_scene.removeItem(self._node_temp_link_item)
        self._node_temp_link_item = None
        self._reset_highlighted_socket()
        try:
            if target_item is None:
                raise ValueError("drop link on a compatible socket")
            link_args = self._link_args_from_socket_drag(target_item)
            result = self._graph_controller().link_nodes(*link_args)
            if not result.success:
                raise ValueError(result.error or "link creation failed")
        except ValueError as exc:
            self._show_graph_error("Link creation failed", exc)
        else:
            self._refresh_node_links()
            self._set_node_status("Link created")
            self._log_graph_event("Link creation succeeded")
        self._node_socket_drag = None
        event.accept()

    def _link_args_from_socket_drag(self, target_item: Any) -> tuple[str, str, str, str]:
        if self._node_socket_drag is None:
            raise ValueError("no active link drag")
        start_side = self._node_socket_drag["side"]
        end_side = str(target_item.data(4))
        start_node_id = self._node_socket_drag["node_id"]
        start_socket = self._node_socket_drag["socket"]
        end_node_id = str(target_item.data(2))
        end_socket = str(target_item.data(3))
        if start_side == "output" and end_side == "input":
            return start_node_id, start_socket, end_node_id, end_socket
        if start_side == "input" and end_side == "output":
            return end_node_id, end_socket, start_node_id, start_socket
        raise ValueError("drop link between an output socket and an input socket")

    def _socket_item_at_view_pos(self, view_pos: Any) -> Any | None:
        candidates = [self.node_view.itemAt(view_pos)]
        search_rect = self._qt_core.QRect(
            int(view_pos.x()) - 8,
            int(view_pos.y()) - 8,
            16,
            16,
        )
        candidates.extend(self.node_view.items(search_rect))
        for candidate in candidates:
            item = candidate
            while item is not None:
                if item.data(1) == "socket":
                    return item
                item = item.parentItem()
        return None

    def _highlight_socket_target(self, item: Any | None) -> None:
        if item is self._node_highlight_socket:
            return
        self._reset_highlighted_socket()
        if item is None or self._node_socket_drag is None:
            return
        self._node_highlight_socket = item
        try:
            self._node_graph.validate_link(
                *self._link_args_from_socket_drag(item),
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
            self._set_node_status("No node type selected. Pick a node from the library first.")
            return
        type_id = str(item.data(self._qt_core.Qt.ItemDataRole.UserRole))
        self._log_graph_event(f"Node type selected: {type_id}")
        self._safe_add_node(type_id)

    def _redraw_node_graph(self) -> None:
        keep_selected = tuple(
            node_id for node_id in self._selected_node_id_cache if node_id in self._node_graph.nodes
        )
        keep_links = tuple(
            link_id for link_id in self._selected_link_id_cache if link_id in self._node_graph.links
        )
        view_state = self._capture_node_view_state()
        self._redrawing_node_graph = True
        self._replace_node_scene_for_rebuild()
        self.node_scene.blockSignals(True)
        try:
            self._node_items.clear()
            self._socket_items.clear()
            self._node_link_items.clear()
            self._node_group_items.clear()
            self._node_inline_widgets.clear()
            self._draw_graph_groups()
            for node in self._node_graph.nodes.values():
                self._draw_graph_node(node)
            self._refresh_node_links()
            self._restore_scene_selection(keep_selected, keep_links)
        finally:
            self.node_scene.blockSignals(False)
            self._redrawing_node_graph = False
        self._restore_captured_node_view_state(view_state)
        self._selected_node_id_cache = keep_selected
        self._selected_link_id_cache = keep_links
        self._node_graph.selected_node_ids = keep_selected
        self._schedule_node_selection_ui_update()

    def _connect_node_scene_signals(self, scene: Any) -> None:
        scene.selectionChanged.connect(self._on_node_selection_changed)

    def _replace_node_scene_for_rebuild(self) -> None:
        old_scene = self.node_scene
        old_scene.blockSignals(True)
        self._retired_node_scenes.append(old_scene)
        self.node_scene = self._qt_widgets.QGraphicsScene()
        self.node_scene.setSceneRect(old_scene.sceneRect())
        self._connect_node_scene_signals(self.node_scene)
        self.node_view.setScene(self.node_scene)
        self._qt_core.QTimer.singleShot(
            2000,
            lambda scene=old_scene: self._dispose_retired_node_scene(scene),
        )

    def _capture_node_view_state(self) -> tuple[Any, Any] | None:
        if not hasattr(self, "node_view"):
            return None
        return (
            self.node_view.transform(),
            self.node_view.mapToScene(self.node_view.viewport().rect().center()),
        )

    def _restore_captured_node_view_state(self, view_state: tuple[Any, Any] | None) -> None:
        if view_state is None or not hasattr(self, "node_view"):
            return
        transform, center = view_state
        self.node_view.setTransform(transform)
        self.node_view.centerOn(center)
        self._node_zoom = self._bounded_node_zoom(float(transform.m11()))
        self._update_node_zoom_label()
        self._sync_node_view_state()

    def _dispose_retired_node_scene(self, scene: Any) -> None:
        if scene is self.node_scene:
            return
        try:
            scene.blockSignals(True)
            scene.clearSelection()
        except RuntimeError:
            pass
        if scene in self._retired_node_scenes:
            self._retired_node_scenes.remove(scene)
        scene.deleteLater()

    def _cleanup_node_scenes_for_close(self) -> None:
        self._closing = True
        if not hasattr(self, "node_view") or not hasattr(self, "node_scene"):
            return
        scenes = [self.node_scene, *self._retired_node_scenes]
        self._node_inline_widgets.clear()
        self._node_items.clear()
        self._socket_items.clear()
        self._node_link_items.clear()
        self._node_group_items.clear()
        self.node_view.setScene(self._qt_widgets.QGraphicsScene())
        for scene in scenes:
            try:
                scene.blockSignals(True)
                scene.clearSelection()
            except RuntimeError:
                continue

    def _queue_node_graph_redraw(
        self,
        selected_node_ids: tuple[str, ...] = (),
        selected_link_ids: tuple[str, ...] = (),
    ) -> None:
        self._queued_node_selection = tuple(
            node_id for node_id in selected_node_ids if node_id in self._node_graph.nodes
        )
        self._queued_link_selection = tuple(
            link_id for link_id in selected_link_ids if link_id in self._node_graph.links
        )
        self._selected_node_id_cache = self._queued_node_selection
        self._selected_link_id_cache = self._queued_link_selection
        if self._node_graph_redraw_queued:
            return
        self._node_graph_redraw_queued = True
        self._qt_core.QTimer.singleShot(0, self._flush_queued_node_graph_redraw)

    def _flush_queued_node_graph_redraw(self) -> None:
        self._node_graph_redraw_queued = False
        selected = self._queued_node_selection
        links = self._queued_link_selection
        self._queued_node_selection = ()
        self._queued_link_selection = ()
        self._selected_node_id_cache = selected
        self._selected_link_id_cache = links
        self._redraw_node_graph()
        self._set_scene_selection_by_ids(selected, links)

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
        inline_keys = self._inline_node_setting_keys(node)
        title_text = self._node_display_title(node)
        longest_label = max(
            [len(title_text), *(len(_setting_label(key)) for key in inline_keys)],
            default=len(title_text),
        )
        has_path_field = any(self._is_path_setting(key) for key in inline_keys)
        content_width = 214.0 + min(220.0, longest_label * 6.5)
        width = max(node.width, 380.0, content_width, 460.0 if has_path_field else 0.0)
        inline_height = self._inline_node_settings_height(inline_keys)
        height = (
            44.0
            if node.collapsed
            else max(node.height, 132.0, 80.0 + inline_height)
        )
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
        title = self.node_scene.addText(title_text)
        title.setDefaultTextColor(self._qt_gui.QColor("#f0f5fa"))
        title.setTextWidth(width - 72.0)
        title.setPos(34.0, 5.0)
        title.setParentItem(rect)
        collapse_hint = self.node_scene.addText("[+]" if node.collapsed else "[-]")
        collapse_hint.setDefaultTextColor(self._qt_gui.QColor("#d7e4ef"))
        collapse_hint.setPos(9.0, 5.0)
        collapse_hint.setParentItem(rect)

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
            details.setPos(14.0, 40.0)
            details.setParentItem(rect)
            self._draw_inline_node_settings(rect, node, inline_keys)
        self._draw_node_sockets(
            rect,
            node,
            definition,
            height,
            show_labels=not inline_keys,
        )
        self._node_items[node.id] = rect

    def _inline_node_setting_keys(self, node: NodeInstance) -> tuple[str, ...]:
        editable = [
            key
            for key, value in node.settings.items()
            if isinstance(value, bool | int | float | str)
        ]
        layer_keys = (
            "enabled",
            "name",
            "material",
            "winding_angle_deg",
            "angle_tolerance_deg",
            "start_z_mm",
            "end_z_mm",
            "points",
            "coverage_target",
        )
        simple_keys_by_type = {
            "project": ("name", "output_directory", "units"),
            "machine_backend": (
                "controller",
                "clearance_mm",
                "max_a_rpm",
                "max_x_mm",
                "max_z_mm",
            ),
            "mandrel_backend": (
                "mode",
                "type",
                "profile_path",
                "cylinder_length_mm",
                "cylinder_radius_mm",
                "left_dome_length_mm",
                "right_dome_length_mm",
                "polar_opening_radius_mm",
            ),
            "tow_backend": (
                "name",
                "width_mm",
                "thickness_mm",
                "effective_width_mm",
                "calibrated_effective_width",
                "friction_coefficient",
                "calibrated_friction",
                "fibre_type",
                "resin_system",
            ),
            "layer_backend": layer_keys,
            "hoop_layer": layer_keys,
            "geodesic_layer": layer_keys,
            "non_geodesic_layer": layer_keys,
            "layer_stack_backend": ("name", "ordering", "repeat_stack", "mirror_stack"),
            "pattern_optimisation_backend": (
                "method",
                "angle_tolerance_deg",
                "require_gcd_clean_pattern",
            ),
            "coverage_mode": ("stack_level_full_coverage", "paired_layer_coverage"),
            "pin_layout_backend": (
                "enabled",
                "routing_mode",
                "candidate_count",
                "route_step_size",
                "wrap_direction",
                "count_per_shoulder",
                "angular_offset_deg",
                "pin_radius_mm",
                "pin_height_mm",
                "pin_standoff_mm",
                "shoulder_zone_width_mm",
                "target_dome_angle_min_deg",
                "target_dome_angle_max_deg",
                "coverage_tolerance_mm",
            ),
            "plot_backend": ("enabled", "output_directory"),
            "csv_backend_export": ("enabled",),
            "gcode_backend_export": ("enabled",),
            "report_export": ("enabled",),
            "controller_run_backend": ("enabled", "port"),
        }
        if node.type_id in simple_keys_by_type:
            return tuple(key for key in simple_keys_by_type[node.type_id] if key in editable)
        priority = (
            "enabled",
            "name",
            "ply_order",
            "material",
            "mode",
            "type",
            "profile_path",
            "samples",
            "region",
            "direction",
            "winding_angle_deg",
            "target_angle_deg",
            "initial_angle_deg",
            "angle_tolerance_deg",
            "start_z_mm",
            "end_z_mm",
            "passes",
            "coverage_target",
            "width_mm",
            "thickness_mm",
            "cylinder_length_mm",
            "cylinder_radius_mm",
            "polar_opening_radius_mm",
            "feedrate_mm_min",
            "method",
            "output_directory",
        )
        ordered = [key for key in priority if key in editable]
        ordered.extend(key for key in editable if key not in ordered)
        return tuple(ordered)

    def _inline_node_settings_height(self, keys: tuple[str, ...]) -> float:
        if not keys:
            return 0.0
        return 28.0 * len(keys) + 12.0

    def _draw_inline_node_settings(
        self,
        parent_item: Any,
        node: NodeInstance,
        keys: tuple[str, ...],
    ) -> None:
        if node.type_id == "layer_stack_backend":
            self._draw_layer_stack_table(parent_item, node)
            return
        if not keys:
            return
        panel = self._qt_widgets.QWidget()
        panel.setObjectName("inlineNodeSettings")
        layout = self._qt_widgets.QFormLayout(panel)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(4)
        layout.setFieldGrowthPolicy(
            self._qt_widgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
        )
        for key in keys:
            editor = self._inline_node_setting_editor(node.id, key, node.settings[key])
            label = self._qt_widgets.QLabel(_setting_label(key))
            label.setObjectName("inlineNodeSettingLabel")
            label.setToolTip(key)
            label.setMinimumWidth(118)
            label.setMaximumWidth(168)
            layout.addRow(label, editor)
        proxy = self.node_scene.addWidget(panel)
        proxy.setParentItem(parent_item)
        proxy.setPos(14.0, 74.0)
        proxy.setZValue(60.0)
        proxy.resize(
            parent_item.rect().width() - 28.0,
            max(36.0, self._inline_node_settings_height(keys)),
        )
        self._node_inline_widgets.append(proxy)

    def _draw_layer_stack_table(self, parent_item: Any, node: NodeInstance) -> None:
        layers = self._connected_layer_nodes_for_stack(node.id)
        panel = self._qt_widgets.QWidget()
        panel.setObjectName("inlineNodeSettings")
        layout = self._qt_widgets.QVBoxLayout(panel)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(6)
        table = self._qt_widgets.QTableWidget(max(1, len(layers)), 6)
        table.setObjectName("layerStackNodeTable")
        table.setHorizontalHeaderLabels(
            ["Layer", "Angle", "Tol", "Start Z", "End Z", "Steps"]
        )
        table.verticalHeader().setVisible(False)
        table.setSelectionBehavior(self._qt_widgets.QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(self._qt_widgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        table.setDragDropMode(self._qt_widgets.QAbstractItemView.DragDropMode.InternalMove)
        table.setDefaultDropAction(self._qt_core.Qt.DropAction.MoveAction)
        table.setDragDropOverwriteMode(False)
        table.setMinimumHeight(138)
        table.setMaximumHeight(220)
        table.horizontalHeader().setStretchLastSection(True)
        table.blockSignals(True)
        for row, layer_node in enumerate(layers):
            self._populate_layer_stack_row(table, row, layer_node)
        if not layers:
            item = self._qt_widgets.QTableWidgetItem("Connect layer nodes")
            item.setFlags(item.flags() & ~self._qt_core.Qt.ItemFlag.ItemIsEditable)
            table.setItem(0, 1, item)
        table.blockSignals(False)
        def _on_cell_changed(row: int, col: int) -> None:
            self.run_safe_action(
                "Edit Layer Stack Row",
                lambda: self._update_layer_stack_row(node.id, table, row, col),
            )

        def _on_rows_moved(*_args: Any) -> None:
            self.run_safe_action(
                "Reorder Layer Stack",
                lambda: self._apply_layer_stack_table_order(node.id, table),
            )

        table.cellChanged.connect(_on_cell_changed)
        table.model().rowsMoved.connect(_on_rows_moved)
        copy_button = self._qt_widgets.QPushButton("Copy Selected Layer Row")

        def _copy_selected_layer_rows() -> None:
            self._copy_layer_stack_rows(node.id, table)

        self._connect_safe_button(
            copy_button,
            "Copy Layer Stack Row",
            _copy_selected_layer_rows,
        )
        hint = self._qt_widgets.QLabel(
            "Drag rows to change ply order. Angle selects Polar, Helical, or Hoop automatically."
        )
        hint.setObjectName("inlineNodeSettingLabel")
        layout.addWidget(table)
        layout.addWidget(copy_button)
        layout.addWidget(hint)
        proxy = self.node_scene.addWidget(panel)
        proxy.setParentItem(parent_item)
        proxy.setPos(14.0, 74.0)
        proxy.setZValue(60.0)
        proxy.resize(parent_item.rect().width() - 28.0, 285.0)
        self._node_inline_widgets.append(proxy)

    def _connected_layer_nodes_for_stack(self, stack_node_id: str) -> list[NodeInstance]:
        layer_nodes = [
            self._node_graph.nodes[link.source_node_id]
            for link in self._node_graph.links.values()
            if link.target_node_id == stack_node_id
            and link.target_socket == "layer"
            and link.source_node_id in self._node_graph.nodes
            and self._node_graph.nodes[link.source_node_id].type_id
            in {"layer_backend", "hoop_layer", "geodesic_layer", "non_geodesic_layer"}
        ]
        layer_nodes.sort(
            key=lambda layer: (
                _float_setting(layer.settings.get("ply_order"), 1_000_000.0),
                layer.y,
                layer.name,
            )
        )
        return layer_nodes

    def _populate_layer_stack_row(self, table: Any, row: int, layer_node: NodeInstance) -> None:
        values = (
            self._node_display_title(layer_node),
            layer_node.settings.get(
                "winding_angle_deg",
                layer_node.settings.get("target_angle_deg", ""),
            ),
            layer_node.settings.get("angle_tolerance_deg", ""),
            layer_node.settings.get("start_z_mm", ""),
            layer_node.settings.get("end_z_mm", ""),
            layer_node.settings.get("points", ""),
        )
        for col, value in enumerate(values):
            item = self._qt_widgets.QTableWidgetItem(str(value))
            item.setData(self._qt_core.Qt.ItemDataRole.UserRole, layer_node.id)
            if col == 0:
                item.setFlags(item.flags() & ~self._qt_core.Qt.ItemFlag.ItemIsEditable)
            table.setItem(row, col, item)

    def _update_layer_stack_row(self, _stack_node_id: str, table: Any, row: int, col: int) -> None:
        item = table.item(row, col)
        if item is None:
            return
        layer_node_id = item.data(self._qt_core.Qt.ItemDataRole.UserRole)
        if layer_node_id not in self._node_graph.nodes:
            return
        key_by_col = {
            1: "winding_angle_deg",
            2: "angle_tolerance_deg",
            3: "start_z_mm",
            4: "end_z_mm",
            5: "points",
        }
        key = key_by_col.get(col)
        if key is None:
            return
        value: Any = item.text().strip()
        if key in {"points"}:
            value = max(16, int(float(value or "220")))
        elif key in {"winding_angle_deg", "angle_tolerance_deg", "start_z_mm", "end_z_mm"}:
            value = "" if value == "" else float(value)
        node = self._node_graph.nodes[str(layer_node_id)]
        node.settings[key] = value
        if key == "winding_angle_deg":
            angle = _float_setting(value, 45.0)
            node.settings["name"] = f"{self._layer_kind_label(node).lower()}_{angle:g}deg"
        self._node_graph.mark_downstream_dirty(str(layer_node_id))
        self._queue_node_graph_redraw(self._selected_node_ids())
        self._schedule_node_setting_render()

    def _apply_layer_stack_table_order(self, _stack_node_id: str, table: Any) -> None:
        for row in range(table.rowCount()):
            item = table.item(row, 0) or table.item(row, 1)
            if item is None:
                continue
            layer_node_id = item.data(self._qt_core.Qt.ItemDataRole.UserRole)
            if layer_node_id in self._node_graph.nodes:
                self._node_graph.nodes[str(layer_node_id)].settings["ply_order"] = row + 1
        self._queue_node_graph_redraw(self._selected_node_ids())
        self._schedule_node_setting_render()

    def _copy_layer_stack_rows(self, stack_node_id: str, table: Any) -> None:
        selected_rows = sorted({index.row() for index in table.selectedIndexes()})
        if not selected_rows:
            self._set_node_status("No layer row selected to copy.")
            return
        new_ids = []
        for row in selected_rows:
            item = table.item(row, 0) or table.item(row, 1)
            if item is None:
                continue
            layer_node_id = item.data(self._qt_core.Qt.ItemDataRole.UserRole)
            if layer_node_id not in self._node_graph.nodes:
                continue
            source = self._node_graph.nodes[str(layer_node_id)]
            duplicate = self._node_graph.duplicate_node(
                source.id,
                self._node_registry,
                offset=(40.0, 80.0),
            )
            duplicate.settings["name"] = f"{source.settings.get('name', source.name)} copy"
            duplicate.settings["ply_order"] = (
                len(self._connected_layer_nodes_for_stack(stack_node_id)) + 1
            )
            self._node_graph.add_link(
                duplicate.id,
                "layer",
                stack_node_id,
                "layer",
                self._node_registry,
            )
            new_ids.append(duplicate.id)
        self._queue_node_graph_redraw(tuple(new_ids))
        self._schedule_node_setting_render()

    def _inline_node_setting_editor(self, node_id: str, key: str, value: Any) -> Any:
        if self._is_path_setting(key):
            editor = self._node_path_editor(node_id, key, value, compact=True)
            editor.setObjectName("inlineNodeSettingEditor")
            return editor
        editor = self._setting_editor(node_id, key, value, compact=True)
        editor.setObjectName("inlineNodeSettingEditor")
        editor.setToolTip(key)
        if hasattr(editor, "setMaximumHeight"):
            editor.setMaximumHeight(24)
        return editor

    def _draw_node_sockets(
        self,
        parent_item: Any,
        node: NodeInstance,
        definition: NodeTypeDefinition,
        height: float,
        *,
        show_labels: bool = True,
    ) -> None:
        self._draw_socket_side(
            parent_item,
            node,
            definition.inputs,
            side="input",
            x_pos=-7.0,
            height=height,
            show_labels=show_labels,
        )
        self._draw_socket_side(
            parent_item,
            node,
            definition.outputs,
            side="output",
            x_pos=node.width - 7.0,
            height=height,
            show_labels=show_labels,
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
        show_labels: bool,
    ) -> None:
        if not sockets:
            return
        spacing = 26.0 if len(sockets) > 1 else 0.0
        socket_band_start = 26.0 if node.collapsed else 54.0
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
            item.setZValue(120.0)
            item.setData(1, "socket")
            item.setData(2, node.id)
            item.setData(3, socket.name)
            item.setData(4, side)
            item.setData(5, socket.kind)
            item.setToolTip(f"{side}: {socket.name} ({socket.kind})")
            self._socket_items[(node.id, socket.name, side)] = item
            if show_labels and not node.collapsed:
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
        message = node.message or "Not computed"
        if self._is_layer_node(node):
            return f"Layers / {self._layer_kind_label(node)}\n{message}"
        return f"{definition.category} / {definition.label}\n{message}"

    def _node_display_title(self, node: NodeInstance) -> str:
        if not self._is_layer_node(node):
            return node.name
        angle = self._layer_angle_deg(node)
        return f"{self._layer_kind_label(node)} {angle:g} deg"

    def _is_layer_node(self, node: NodeInstance) -> bool:
        return node.type_id in {
            "layer_backend",
            "hoop_layer",
            "geodesic_layer",
            "non_geodesic_layer",
        }

    def _layer_angle_deg(self, node: NodeInstance) -> float:
        return _float_setting(
            node.settings.get(
                "winding_angle_deg",
                node.settings.get("target_angle_deg", node.settings.get("initial_angle_deg", 45.0)),
            ),
            45.0,
        )

    def _layer_kind_label(self, node: NodeInstance) -> str:
        angle = abs(self._layer_angle_deg(node))
        if angle < 15.0:
            return "Polar"
        if angle >= 85.0:
            return "Hoop"
        return "Helical"

    def _refresh_node_links(self) -> None:
        if self._refreshing_node_links or not hasattr(self, "node_scene"):
            return
        self._refreshing_node_links = True
        self._node_link_refresh_queued = False
        try:
            for item in self._node_link_items:
                try:
                    self.node_scene.removeItem(item)
                except RuntimeError:
                    continue
            self._node_link_items.clear()
            for link in tuple(self._node_graph.links.values()):
                source_item = self._socket_items.get(
                    (link.source_node_id, link.source_socket, "output")
                )
                target_item = self._socket_items.get(
                    (link.target_node_id, link.target_socket, "input")
                )
                if source_item is None or target_item is None:
                    continue
                try:
                    path = self._node_link_path(
                        source_item.sceneBoundingRect().center(),
                        target_item.sceneBoundingRect().center(),
                    )
                except RuntimeError:
                    continue
                pen = self._qt_gui.QPen(self._qt_gui.QColor("#6c8daa"), 2.0)
                item = self.node_scene.addPath(path, pen)
                item.setZValue(0.0)
                item.setFlag(self._qt_widgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
                item.setData(1, "link")
                item.setData(2, link.id)
                item.setToolTip(f"{link.source_socket} -> {link.target_socket}")
                self._node_link_items.append(item)
            if not self._redrawing_node_graph:
                self._sync_node_graph_from_scene()
        finally:
            self._refreshing_node_links = False

    def _schedule_node_link_refresh(self) -> None:
        if self._node_link_refresh_queued or self._refreshing_node_links:
            return
        self._node_link_refresh_queued = True
        self._qt_core.QTimer.singleShot(0, self._refresh_node_links)

    def _schedule_node_position_sync(self) -> None:
        if self._node_position_sync_queued:
            return
        self._node_position_sync_queued = True

        def _sync() -> None:
            self._node_position_sync_queued = False
            self._sync_node_graph_from_scene()

        self._qt_core.QTimer.singleShot(0, _sync)

    def _on_node_scene_changed(self) -> None:
        if self._redrawing_node_graph:
            return
        try:
            self._refresh_node_links()
        except Exception as exc:  # noqa: BLE001 - protects Qt event loop
            self.handle_gui_error("Node Scene Changed", exc)

    def _read_scene_selection_ids(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        node_ids = []
        link_ids = []
        try:
            selected_items = self.node_scene.selectedItems()
        except RuntimeError:
            return (), ()
        for item in selected_items:
            try:
                item_kind = item.data(1)
                node_id = item.data(0)
                link_id = item.data(2)
            except RuntimeError:
                continue
            if item_kind == "node" and node_id in self._node_graph.nodes:
                node_ids.append(str(node_id))
            elif item_kind == "link" and link_id in self._node_graph.links:
                link_ids.append(str(link_id))
        return tuple(dict.fromkeys(node_ids)), tuple(dict.fromkeys(link_ids))

    def _selected_node_ids(self) -> tuple[str, ...]:
        return tuple(
            node_id for node_id in self._selected_node_id_cache if node_id in self._node_graph.nodes
        )

    def _selected_link_ids(self) -> tuple[str, ...]:
        return tuple(
            link_id for link_id in self._selected_link_id_cache if link_id in self._node_graph.links
        )

    def _clear_node_scene_selection(self) -> None:
        if not hasattr(self, "node_scene"):
            return
        self.node_scene.blockSignals(True)
        try:
            self.node_scene.clearSelection()
        finally:
            self.node_scene.blockSignals(False)
        self._selected_node_id_cache = ()
        self._selected_link_id_cache = ()
        self._node_graph.selected_node_ids = ()

    def _restore_scene_selection(
        self,
        node_ids: tuple[str, ...],
        link_ids: tuple[str, ...] = (),
    ) -> None:
        for node_id in node_ids:
            item = self._node_items.get(node_id)
            if item is not None:
                try:
                    item.setSelected(True)
                except RuntimeError:
                    continue
        for link_id in link_ids:
            for item in self._node_link_items:
                try:
                    if item.data(2) == link_id:
                        item.setSelected(True)
                        break
                except RuntimeError:
                    continue

    def _set_scene_selection_by_ids(
        self,
        node_ids: tuple[str, ...],
        link_ids: tuple[str, ...] = (),
    ) -> None:
        node_ids = tuple(node_id for node_id in node_ids if node_id in self._node_graph.nodes)
        link_ids = tuple(link_id for link_id in link_ids if link_id in self._node_graph.links)
        self.node_scene.blockSignals(True)
        try:
            self.node_scene.clearSelection()
            self._restore_scene_selection(node_ids, link_ids)
        finally:
            self.node_scene.blockSignals(False)
        self._selected_node_id_cache = node_ids
        self._selected_link_id_cache = link_ids
        self._node_graph.selected_node_ids = node_ids
        self._schedule_node_selection_ui_update()

    def _delete_link_by_id(self, link_id: str) -> None:
        if link_id not in self._node_graph.links:
            self._set_node_status("No link selected.")
            return
        self._node_graph.links.pop(link_id, None)
        self._refresh_node_links()
        self._set_node_status("Link deleted")

    def _find_graph_item_at_view_pos(self, view_pos: Any, kind: str) -> Any | None:
        item = self.node_view.itemAt(view_pos)
        while item is not None:
            if item.data(1) == kind:
                return item
            item = item.parentItem()
        return None

    def _select_node_item(self, node_id: str) -> None:
        self._set_scene_selection_by_ids((node_id,))

    def _show_node_context_menu(self, view_pos: Any) -> None:
        if self._node_suppress_context_menu:
            self._node_suppress_context_menu = False
            return
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
                self.run_safe_action(
                    "Delete Link",
                    lambda: self._delete_link_by_id(link_id),
                )
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
                self.run_safe_action("Rename Group", lambda: self._rename_group_dialog(group_id))
            elif action in (ungroup, delete_group):
                self.run_safe_action("Ungroup Nodes", lambda: self._delete_group_only(group_id))
            elif action == delete_group_nodes:
                self.run_safe_action(
                    "Delete Group Nodes",
                    lambda: self._delete_group_and_nodes(group_id),
                )
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
                self.run_safe_action("Rename Node", self._rename_selected_node_dialog)
            elif action == duplicate_action:
                self.run_safe_action("Duplicate Node", self._duplicate_selected_nodes)
            elif action == delete_action:
                self.run_safe_action("Delete Node", self._delete_selected_graph_items)
            elif action == collapse_action:
                self.run_safe_action(
                    "Toggle Collapse Node",
                    lambda: self._set_selected_nodes_collapsed(not node.collapsed),
                )
            elif action in {run_action, downstream_action}:
                self.run_safe_action(
                    "Run Node",
                    lambda: self._execute_node_graph(execute_exports=False),
                )
            elif action == group_action:
                self.run_safe_action("Group Nodes", self._group_selected_nodes)
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
            self.run_safe_action(
                "Add Node",
                lambda: self._safe_add_node(actions[action], scene_pos),
            )
        elif action == frame_all:
            self.run_safe_action("Fit Node Graph", self._fit_node_graph)
        elif action == auto_layout:
            self.run_safe_action("Auto Layout", self._auto_layout_node_graph)
        elif action == group_action:
            self.run_safe_action("Group Nodes", self._group_selected_nodes)

    def _rename_selected_node_dialog(self) -> None:
        selected = self._selected_node_ids()
        if len(selected) != 1:
            self._set_node_status("Select exactly one node before using Rename.")
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
            self._queue_node_graph_redraw((node.id,))

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
            self._queue_node_graph_redraw(self._selected_node_ids())

    def _delete_group_only(self, group_id: str) -> None:
        group = self._node_graph.groups.pop(group_id, None)
        if group is None:
            return
        selected = self._selected_node_ids()
        self._clear_node_scene_selection()
        for node_id in group.node_ids:
            if node_id in self._node_graph.nodes:
                self._node_graph.nodes[node_id].group_id = None
        self._selected_node_id_cache = tuple(
            node_id for node_id in selected if node_id in self._node_graph.nodes
        )
        self._queue_node_graph_redraw(self._selected_node_id_cache)

    def _delete_group_and_nodes(self, group_id: str) -> None:
        group = self._node_graph.groups.get(group_id)
        if group is None:
            return
        node_ids = tuple(node_id for node_id in group.node_ids if node_id in self._node_graph.nodes)
        self._clear_node_scene_selection()
        result = self._graph_controller().delete_nodes(node_ids)
        if not result.success:
            self._show_graph_error(
                "Delete group nodes failed",
                RuntimeError(result.error or "unknown error"),
            )
            return
        self._node_graph.groups.pop(group_id, None)
        self._queue_node_graph_redraw()

    def _on_node_selection_changed(self) -> None:
        if self._redrawing_node_graph or self._applying_node_selection:
            return
        (
            self._selected_node_id_cache,
            self._selected_link_id_cache,
        ) = self._read_scene_selection_ids()
        self._node_graph.selected_node_ids = self._selected_node_id_cache
        self._logger.info(
            "Node selection changed: nodes=%s links=%s",
            len(self._selected_node_id_cache),
            len(self._selected_link_id_cache),
        )
        self._schedule_node_selection_ui_update()

    def _schedule_node_selection_ui_update(self) -> None:
        if not hasattr(self, "node_scene") or self._applying_node_selection:
            return
        self._applying_node_selection = True
        self._qt_core.QTimer.singleShot(0, self._apply_node_selection_changed)

    def _apply_node_selection_changed(self) -> None:
        try:
            selected = self._selected_node_ids()
            self._selected_node_id_cache = selected
            self._selected_link_id_cache = self._selected_link_ids()
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
        finally:
            self._applying_node_selection = False

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
            editor = (
                self._node_path_editor(node.id, key, value)
                if self._is_path_setting(key)
                else self._setting_editor(node.id, key, value)
            )
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

    def _setting_editor(
        self,
        node_id: str,
        key: str,
        value: Any,
        *,
        compact: bool = False,
    ) -> Any:
        if isinstance(value, bool):
            editor = self._qt_widgets.QCheckBox()
            editor.setChecked(value)
            editor.toggled.connect(
                lambda checked, setting_key=key: self.run_safe_action(
                    "Edit Node Setting",
                    lambda: self._update_node_setting(node_id, setting_key, bool(checked)),
                )
            )
            return editor
        if isinstance(value, int) and not isinstance(value, bool):
            editor = self._qt_widgets.QSpinBox()
            editor.setRange(-1_000_000, 1_000_000)
            if compact:
                editor.setButtonSymbols(self._qt_widgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
            editor.setValue(value)
            editor.valueChanged.connect(
                lambda new_value, setting_key=key: self.run_safe_action(
                    "Edit Node Setting",
                    lambda: self._update_node_setting(node_id, setting_key, int(new_value)),
                )
            )
            return editor
        if isinstance(value, float):
            editor = self._qt_widgets.QDoubleSpinBox()
            editor.setRange(-1_000_000.0, 1_000_000.0)
            editor.setDecimals(4)
            if compact:
                editor.setButtonSymbols(self._qt_widgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
            editor.setValue(value)
            editor.valueChanged.connect(
                lambda new_value, setting_key=key: self.run_safe_action(
                    "Edit Node Setting",
                    lambda: self._update_node_setting(node_id, setting_key, float(new_value)),
                )
            )
            return editor
        if isinstance(value, str):
            options = self._setting_options(key, value)
            if options:
                editor = self._qt_widgets.QComboBox()
                editor.setEditable(value not in options)
                editor.addItems(options)
                if value not in options:
                    editor.addItem(value)
                editor.setCurrentText(value)
                editor.currentTextChanged.connect(
                    lambda text, setting_key=key: self.run_safe_action(
                        "Edit Node Setting",
                        lambda: self._update_node_setting(node_id, setting_key, text),
                    )
                )
                return editor
            editor = self._qt_widgets.QLineEdit(value)
            editor.textEdited.connect(
                lambda text, setting_key=key: self.run_safe_action(
                    "Edit Node Setting",
                    lambda: self._update_node_setting(node_id, setting_key, text),
                )
            )
            return editor
        summary = self._qt_widgets.QLabel("Edit nested value in advanced JSON")
        summary.setWordWrap(True)
        return summary

    def _is_path_setting(self, key: str) -> bool:
        lowered = key.lower()
        return lowered.endswith("_path") or lowered.endswith("_file") or "directory" in lowered

    def _node_path_editor(
        self,
        node_id: str,
        key: str,
        value: Any,
        *,
        compact: bool = False,
    ) -> Any:
        container = self._qt_widgets.QWidget()
        layout = self._qt_widgets.QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        field = self._qt_widgets.QLineEdit(str(value))
        field.setToolTip(key)
        field.textEdited.connect(
            lambda text, setting_key=key: self.run_safe_action(
                "Edit Node Setting",
                lambda: self._update_node_setting(node_id, setting_key, text),
            )
        )
        browse = self._qt_widgets.QPushButton("...")
        browse.setFixedWidth(26 if compact else 32)

        def browse_path_setting() -> None:
            self._browse_node_path_setting(node_id, key, field)

        self._connect_safe_button(
            browse,
            f"Browse {key}",
            browse_path_setting,
        )
        layout.addWidget(field, 1)
        layout.addWidget(browse)
        container.setToolTip(key)
        return container

    def _browse_node_path_setting(self, node_id: str, key: str, field: Any) -> None:
        if "directory" in key.lower():
            selected = self._qt_widgets.QFileDialog.getExistingDirectory(
                self.widget,
                f"Choose {key}",
                field.text() or ".",
            )
        else:
            selected, _filter = self._qt_widgets.QFileDialog.getOpenFileName(
                self.widget,
                f"Choose {key}",
                field.text() or ".",
                "Mandrel/Profile files (*.dxf *.json *.csv);;All files (*.*)",
            )
        if not selected:
            return
        field.setText(selected)
        self._update_node_setting(node_id, key, selected)

    def _setting_options(self, key: str, current_value: str) -> list[str]:
        options = {
            "type": [
                "hoop",
                "helical",
                "polar",
                "geodesic",
                "non_geodesic",
                "cylinder",
                "cylinder_with_elliptical_domes",
                "axisymmetric_profile",
            ],
            "mode": ["dome", "cylinder", "profile"],
            "region": ["cylinder_only", "dome_to_dome", "left_dome", "right_dome"],
            "direction": ["forward", "reverse"],
            "passes": ["auto"],
            "method": ["textbook_integer_closure"],
            "units": ["mm", "in"],
            "controller": ["grbl_compatible"],
            "tow_band_model": ["rectangular_surface_band"],
            "fibre_type": ["carbon", "glass", "aramid", "basalt"],
        }.get(key, [])
        if current_value and current_value not in options and key in {
            "type",
            "region",
            "direction",
            "method",
            "units",
            "controller",
            "tow_band_model",
            "fibre_type",
            "mode",
        }:
            options = [*options, current_value]
        return options

    def _update_node_setting(self, node_id: str, key: str, value: Any) -> None:
        node = self._node_graph.nodes.get(node_id)
        if node is None:
            return
        if node.settings.get(key) == value:
            return
        node.settings[key] = value
        self._node_graph.mark_downstream_dirty(node_id)
        if hasattr(self, "node_inspector_status"):
            self.node_inspector_status.setText(f"Status: {node.status} - {node.message}")
        if hasattr(self, "node_inspector_settings") and node_id in self._selected_node_ids():
            self.node_inspector_settings.setPlainText(
                json.dumps(node.settings, indent=2, sort_keys=True)
            )
        self._set_node_status(f"Updated {node.name}: {key}")
        self._schedule_node_setting_render()

    def _schedule_node_setting_render(self) -> None:
        if self._node_setting_render_pending:
            return
        self._node_setting_render_pending = True
        self._qt_core.QTimer.singleShot(150, self._flush_node_setting_render)

    def _flush_node_setting_render(self) -> None:
        self._node_setting_render_pending = False
        self._update_viewport_context_for_selection(self._selected_node_ids())
        self._render_from_node()

    def _update_viewport_context_for_selection(self, selected: tuple[str, ...]) -> None:
        if len(selected) != 1:
            self._viewport_node_context = None
            return
        node = self._node_graph.nodes.get(selected[0])
        if node is None:
            self._viewport_node_context = None
            return
        self._viewport_node_context = node.type_id
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
            "project": "backend project metadata",
            "machine_backend": "machine limits and kinematics",
            "mandrel_backend": "domed mandrel geometry",
            "tow_backend": "tow and material settings",
            "layer_stack_backend": "backend layer stack",
            "layer_backend": "angle-driven layer definition",
            "pin_layout_backend": "symmetric shoulder cross-pin layout",
            "hoop_layer": "single hoop layer definition",
            "geodesic_layer": "single geodesic dome layer definition",
            "non_geodesic_layer": "single controlled non-geodesic layer definition",
            "coverage_mode": "tow-footprint coverage settings",
            "pattern_optimisation_backend": "textbook pattern search and selection",
            "validation_backend": "path and manufacturing validation",
            "backend_check": "full backend-ready gate",
            "plot_backend": "backend plot outputs",
            "csv_backend_export": "backend CSV output",
            "gcode_backend_export": "backend G-code output",
            "report_export": "manufacturing reports",
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

    def _display_layer_points(
        self,
        layer: Any,
        *,
        mandrel_length_mm: float,
        center_z_mm: float | None = None,
        display_offset_mm: float = 0.75,
    ) -> Any:
        points = orient_points_for_horizontal_view(
            layer.path.points_mm,
            length_mm=mandrel_length_mm,
            center_z_mm=center_z_mm,
        )
        return offset_display_surface(points, offset_mm=display_offset_mm)

    def _display_tow_band_mesh(
        self,
        layer: Any,
        *,
        mandrel_length_mm: float,
        center_z_mm: float | None = None,
        display_offset_mm: float = 0.9,
    ) -> tuple[Any, Any]:
        path_points = np.asarray(layer.path.points_mm, dtype=float)
        if path_points.shape[0] < 2:
            display_points = orient_points_for_horizontal_view(
                path_points,
                length_mm=mandrel_length_mm,
                center_z_mm=center_z_mm,
            )
            return display_points, np.zeros((0, 3), dtype=int)
        tangent = np.gradient(path_points, axis=0)
        tangent_norm = np.linalg.norm(tangent, axis=1)
        tangent_norm[tangent_norm <= 1e-9] = 1.0
        tangent = tangent / tangent_norm[:, None]
        surface_normal = self._path_surface_normals(layer)
        side = np.cross(surface_normal, tangent)
        side_norm = np.linalg.norm(side, axis=1)
        fallback = side_norm <= 1e-9
        side_norm[fallback] = 1.0
        side = side / side_norm[:, None]
        side[fallback] = np.asarray([0.0, 1.0, 0.0])
        half_width = max(float(layer.spec.tow_width_mm) * 0.5, 0.25)
        centerline = path_points + surface_normal * display_offset_mm
        vertices_mandrel = np.empty((centerline.shape[0] * 2, 3), dtype=float)
        vertices_mandrel[0::2] = centerline - side * half_width
        vertices_mandrel[1::2] = centerline + side * half_width
        vertices = orient_points_for_horizontal_view(
            vertices_mandrel,
            length_mm=mandrel_length_mm,
            center_z_mm=center_z_mm,
        )
        faces = []
        for index in range(centerline.shape[0] - 1):
            left0 = index * 2
            right0 = left0 + 1
            left1 = left0 + 2
            right1 = left0 + 3
            faces.append((left0, left1, right0))
            faces.append((right0, left1, right1))
        return vertices, np.asarray(faces, dtype=int)

    def _path_surface_normals(self, layer: Any) -> Any:
        points = np.asarray(layer.path.points_mm, dtype=float)
        radius = np.asarray(layer.path.surface_radius_mm, dtype=float)
        z_mm = np.asarray(layer.path.z_mm, dtype=float)
        if points.shape[0] == 0:
            return np.zeros((0, 3), dtype=float)
        radial = points.copy()
        radial[:, 2] = 0.0
        radial_norm = np.linalg.norm(radial[:, :2], axis=1)
        radial_norm[radial_norm <= 1e-9] = 1.0
        radial[:, 0] = radial[:, 0] / radial_norm
        radial[:, 1] = radial[:, 1] / radial_norm
        radial[:, 2] = 0.0
        if points.shape[0] < 3 or np.allclose(radius, radius[0]):
            return radial
        dz = np.gradient(z_mm)
        dr = np.gradient(radius)
        dr_dz = np.divide(
            dr,
            dz,
            out=np.zeros_like(dr, dtype=float),
            where=np.abs(dz) > 1e-9,
        )
        dr_dz = np.nan_to_num(dr_dz, nan=0.0, posinf=0.0, neginf=0.0)
        normals = radial.copy()
        normals[:, 2] = -dr_dz
        normal_norm = np.linalg.norm(normals, axis=1)
        normal_norm[normal_norm <= 1e-9] = 1.0
        return normals / normal_norm[:, None]

    def _apply_node_inspector(self) -> None:
        selected = self._selected_node_ids()
        if len(selected) != 1:
            self._set_node_status("Select exactly one node before applying inspector settings.")
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
        self._queue_node_graph_redraw((node_id,))
        self._schedule_node_setting_render()
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
        result = self._graph_controller().link_nodes(*link_args)
        if not result.success:
            self._show_graph_error(
                "Link creation failed",
                RuntimeError(result.error or "unknown error"),
            )
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
        if not selected:
            self._set_node_status("No node selected. Select nodes before using Unlink.")
            return
        result = self._graph_controller().unlink_nodes(selected)
        if not result.success:
            self._show_graph_error("Unlink failed", RuntimeError(result.error or "unknown error"))
            return
        self._refresh_node_links()
        self._set_node_status("Selected node links removed")

    def _duplicate_selected_nodes(self) -> None:
        selected = self._selected_node_ids()
        if not selected:
            self._set_node_status("No node selected. Select a node before using Duplicate.")
            return
        self._logger.info("Duplicate nodes: count=%s ids=%s", len(selected), selected)
        self._clear_node_scene_selection()
        new_nodes = []
        for node_id in selected:
            result = self._graph_controller().duplicate_node(node_id)
            if not result.success or result.node_id is None:
                self._show_graph_error(
                    "Duplicate node failed",
                    RuntimeError(result.error or "unknown error"),
                )
                return
            new_nodes.append(self._node_graph.nodes[result.node_id])
        new_ids = tuple(node.id for node in new_nodes)
        self._selected_node_id_cache = new_ids
        self._queue_node_graph_redraw(new_ids)

    def _delete_selected_nodes(self) -> None:
        selected = self._selected_node_ids()
        if not selected:
            self._set_node_status("No node selected. Select a node before using Delete.")
            return
        self._logger.info("Delete nodes: count=%s ids=%s", len(selected), selected)
        self._clear_node_scene_selection()
        result = self._graph_controller().delete_nodes(selected)
        if not result.success:
            self._show_graph_error(
                "Delete node failed",
                RuntimeError(result.error or "unknown error"),
            )
            return
        self._queue_node_graph_redraw()

    def _delete_selected_graph_items(self) -> None:
        link_ids = self._selected_link_ids()
        selected = self._selected_node_ids()
        if not link_ids and not selected:
            self._set_node_status("No node or link selected. Select an item before using Delete.")
            return
        self._logger.info(
            "Delete graph items: nodes=%s links=%s",
            len(selected),
            len(link_ids),
        )
        self._clear_node_scene_selection()
        for link_id in link_ids:
            self._node_graph.links.pop(link_id, None)
        if selected:
            result = self._graph_controller().delete_nodes(selected)
            if not result.success:
                self._show_graph_error(
                    "Delete node failed",
                    RuntimeError(result.error or "unknown error"),
                )
                return
        self._queue_node_graph_redraw()

    def _set_selected_nodes_collapsed(self, collapsed: bool) -> None:
        selected = self._selected_node_ids()
        if not selected:
            self._set_node_status("No node selected. Select a node before Collapse/Expand.")
            return
        self._clear_node_scene_selection()
        for node_id in selected:
            self._node_graph.set_node_collapsed(node_id, collapsed)
        self._selected_node_id_cache = selected
        self._queue_node_graph_redraw(selected)

    def _group_selected_nodes(self) -> None:
        selected = self._selected_node_ids()
        if len(selected) < 2:
            self._set_node_status("Select at least two nodes to group")
            return
        self._clear_node_scene_selection()
        self._node_graph.group_nodes(selected, name=f"Group {len(self._node_graph.groups) + 1}")
        self._selected_node_id_cache = selected
        self._queue_node_graph_redraw(selected)

    def _ungroup_selected_nodes(self) -> None:
        selected = set(self._selected_node_ids())
        if not selected:
            self._set_node_status("No node selected. Select grouped nodes before using Ungroup.")
            return
        self._clear_node_scene_selection()
        for group_id, group in list(self._node_graph.groups.items()):
            if selected.intersection(group.node_ids):
                for node_id in group.node_ids:
                    if node_id in self._node_graph.nodes:
                        self._node_graph.nodes[node_id].group_id = None
                self._node_graph.groups.pop(group_id, None)
        self._selected_node_id_cache = tuple(
            node_id for node_id in selected if node_id in self._node_graph.nodes
        )
        self._queue_node_graph_redraw(self._selected_node_id_cache)

    def _move_selected_nodes(self, dx: float, dy: float) -> None:
        selected = self._selected_node_ids()
        if not selected:
            self._set_node_status("No node selected. Select node(s) before moving.")
            return
        self._logger.info("Move nodes: count=%s dx=%s dy=%s", len(selected), dx, dy)
        self._clear_node_scene_selection()
        for node_id in selected:
            node = self._node_graph.nodes[node_id]
            self._node_graph.set_node_position(node_id, node.x + dx, node.y + dy)
            self._node_graph.mark_downstream_dirty(node_id)
        self._selected_node_id_cache = selected
        self._queue_node_graph_redraw(selected)
        self._set_node_status(f"Moved {len(selected)} node(s)")

    def _align_selected_nodes_horizontally(self) -> None:
        selected = self._selected_node_ids()
        if len(selected) < 2:
            self._set_node_status("Select at least two nodes before using Align H.")
            return
        self._clear_node_scene_selection()
        y_pos = min(self._node_graph.nodes[node_id].y for node_id in selected)
        for node_id in selected:
            self._node_graph.set_node_position(node_id, self._node_graph.nodes[node_id].x, y_pos)
            self._node_graph.mark_downstream_dirty(node_id)
        self._selected_node_id_cache = selected
        self._queue_node_graph_redraw(selected)

    def _align_selected_nodes_vertically(self) -> None:
        selected = self._selected_node_ids()
        if len(selected) < 2:
            self._set_node_status("Select at least two nodes before using Align V.")
            return
        self._clear_node_scene_selection()
        x_pos = min(self._node_graph.nodes[node_id].x for node_id in selected)
        for node_id in selected:
            self._node_graph.set_node_position(node_id, x_pos, self._node_graph.nodes[node_id].y)
            self._node_graph.mark_downstream_dirty(node_id)
        self._selected_node_id_cache = selected
        self._queue_node_graph_redraw(selected)

    def _distribute_selected_nodes_horizontally(self) -> None:
        selected = sorted(
            self._selected_node_ids(),
            key=lambda node_id: self._node_graph.nodes[node_id].x,
        )
        if len(selected) < 3:
            self._set_node_status("Select at least three nodes before using Distribute H.")
            return
        self._clear_node_scene_selection()
        start = self._node_graph.nodes[selected[0]].x
        end = self._node_graph.nodes[selected[-1]].x
        step = (end - start) / max(1, len(selected) - 1)
        for index, node_id in enumerate(selected):
            self._node_graph.set_node_position(
                node_id,
                start + step * index,
                self._node_graph.nodes[node_id].y,
            )
            self._node_graph.mark_downstream_dirty(node_id)
        self._selected_node_id_cache = tuple(selected)
        self._queue_node_graph_redraw(tuple(selected))

    def _distribute_selected_nodes_vertically(self) -> None:
        selected = sorted(
            self._selected_node_ids(),
            key=lambda node_id: self._node_graph.nodes[node_id].y,
        )
        if len(selected) < 3:
            self._set_node_status("Select at least three nodes before using Distribute V.")
            return
        self._clear_node_scene_selection()
        start = self._node_graph.nodes[selected[0]].y
        end = self._node_graph.nodes[selected[-1]].y
        step = (end - start) / max(1, len(selected) - 1)
        for index, node_id in enumerate(selected):
            self._node_graph.set_node_position(
                node_id,
                self._node_graph.nodes[node_id].x,
                start + step * index,
            )
            self._node_graph.mark_downstream_dirty(node_id)
        self._selected_node_id_cache = tuple(selected)
        self._queue_node_graph_redraw(tuple(selected))

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
        selected = self._selected_node_ids()
        self._selected_node_id_cache = selected
        self._queue_node_graph_redraw(selected)
        self._schedule_fit_node_graph()

    def _sync_node_graph_from_scene(self) -> None:
        for node_id, item in self._node_items.items():
            if node_id not in self._node_graph.nodes:
                continue
            try:
                position = item.pos()
            except RuntimeError:
                continue
            self._node_graph.set_node_position(node_id, float(position.x()), float(position.y()))
        self._sync_node_view_state()

    def _uses_backend_service_graph(self) -> bool:
        backend_types = {
            "project",
            "machine_backend",
            "mandrel_backend",
            "tow_backend",
            "layer_backend",
            "hoop_layer",
            "geodesic_layer",
            "non_geodesic_layer",
            "layer_stack_backend",
            "pattern_optimisation_backend",
            "coverage_mode",
            "pin_layout_backend",
            "validation_backend",
            "backend_check",
            "plot_backend",
            "csv_backend_export",
            "gcode_backend_export",
            "report_export",
        }
        return any(node.type_id in backend_types for node in self._node_graph.nodes.values())

    def _execute_node_graph(self, *, execute_exports: bool) -> None:
        if self._uses_backend_service_graph():
            if execute_exports:
                self._run_backend_generate()
            else:
                self._run_backend_check()
            return
        self._sync_node_graph_from_scene()
        self._log_graph_event("Graph execution started")
        try:
            ordered_node_ids = self._node_graph.topological_node_ids()
        except ValueError as exc:
            self._show_graph_error("Graph execution failed", exc)
            return
        self._start_task_progress("Graph", total_steps=len(ordered_node_ids))
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
                    with suppress(RuntimeError):
                        self.signals.failed.emit(str(exc))
                    return
                with suppress(RuntimeError):
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
        if self._closing:
            return
        self._node_graph = NodeGraphState.from_dict(graph_data, self._node_registry)
        self._last_node_result = result
        self._redraw_node_graph()
        self._draw_node_graph_result(result)
        self._update_task_progress(len(result.executed_node_ids))
        if result.warnings:
            warning_text = "\n".join(result.warnings)
            self._finish_task_progress("Complete with warnings")
            self._set_node_status(f"Graph completed with warnings:\n{warning_text}")
            return
        self._finish_task_progress("Complete")
        self._set_node_status(f"Graph complete: {len(result.executed_node_ids)} nodes executed")
        self._log_graph_event("Graph execution completed")

    def _on_node_graph_worker_failed(self, message: str) -> None:
        if self._closing:
            return
        for node in self._node_graph.nodes.values():
            if node.status == "processing":
                node.status = "error"
                node.message = message
        self._redraw_node_graph()
        self._fail_task_progress("Failed")
        self._show_graph_error(f"Graph execution failed: {message}")

    def _run_backend_check(self) -> None:
        self._start_backend_service_worker("backend_check")

    def _run_backend_generate(self) -> None:
        self._start_backend_service_worker("generate")

    def _update_everything(self) -> None:
        self._set_node_status("Updating everything from current node settings...")
        self._last_backend_check = None
        self._last_loaded_reports = None
        self._last_loaded_plots = None
        self._current_plot_path = None
        if hasattr(self, "backend_report_panel"):
            self.backend_report_panel.clear()
        if hasattr(self, "node_debug_log"):
            self.node_debug_log.appendPlainText("Update Everything: stale artifacts cleared")
        # Keep the current mandrel viewport visible while the backend job runs.
        # The finished callback redraws it from the updated node graph/config.
        self._start_backend_service_worker("generate")

    def _run_backend_csv_export(self) -> None:
        self._start_backend_service_worker("export_csv")

    def _run_backend_gcode_export(self) -> None:
        self._start_backend_service_worker("export_gcode")

    def _start_backend_service_worker(self, operation: str) -> None:
        self._sync_node_graph_from_scene()
        graph_data = self._node_graph.to_dict()
        registry = self._node_registry
        qt_core = self._qt_core
        self._logger.info("Backend %s start", operation)
        self._append_gui_log("Backend", "started", operation)
        self._start_task_progress(operation.replace("_", " ").title())
        self._mark_backend_nodes_running(operation)
        self._set_backend_busy(True, f"Backend {operation.replace('_', ' ')} running...")

        class _WorkerSignals(qt_core.QObject):  # type: ignore[name-defined, misc, valid-type]
            finished = qt_core.Signal(str, object, object)
            failed = qt_core.Signal(str, str, str)

        class _BackendWorker(qt_core.QRunnable):  # type: ignore[name-defined, misc, valid-type]
            def __init__(self) -> None:
                super().__init__()
                self.signals = _WorkerSignals()

            def run(self) -> None:
                try:
                    graph = NodeGraphState.from_dict(graph_data, registry)
                    service = BackendService()
                    if operation in {"backend_check", "generate"}:
                        payload: object = service.backend_check_safely(graph)
                    elif operation == "export_csv":
                        payload = service.export_csv(graph)
                    elif operation == "export_gcode":
                        payload = service.export_gcode(graph)
                    else:
                        raise ValueError(f"unsupported backend operation: {operation}")
                except Exception as exc:  # noqa: BLE001 - reported to UI thread
                    with suppress(RuntimeError):
                        self.signals.failed.emit(operation, str(exc), traceback.format_exc())
                    return
                with suppress(RuntimeError):
                    self.signals.finished.emit(operation, payload, graph.to_dict())

        worker = _BackendWorker()
        worker.signals.finished.connect(self._on_backend_service_worker_finished)
        worker.signals.failed.connect(self._on_backend_service_worker_failed)
        self._node_workers.append(worker)
        self._node_thread_pool.start(worker)

    def _mark_backend_nodes_running(self, operation: str) -> None:
        backend_types = {
            "project",
            "machine_backend",
            "mandrel_backend",
            "tow_backend",
            "layer_backend",
            "hoop_layer",
            "geodesic_layer",
            "non_geodesic_layer",
            "layer_stack_backend",
            "pattern_optimisation_backend",
            "coverage_mode",
            "validation_backend",
            "backend_check",
            "plot_backend",
            "csv_backend_export",
            "gcode_backend_export",
            "report_export",
        }
        for node in self._node_graph.nodes.values():
            if node.type_id in backend_types:
                node.status = "running"
                node.message = operation.replace("_", " ")
        self._redraw_node_graph()

    def _on_backend_service_worker_finished(
        self,
        operation: str,
        payload: object,
        graph_data: dict[str, object],
    ) -> None:
        if self._closing:
            return
        self._node_graph = NodeGraphState.from_dict(graph_data, self._node_registry)
        if isinstance(payload, BackendCheckResult):
            self._last_backend_check = payload
            self._apply_backend_check_result(payload)
            self._update_backend_report_panel(payload)
            self._load_backend_artifacts(payload.output_directory)
            status = "backend-ready" if payload.overall_ready else "not backend-ready"
            self._set_node_status(f"Backend check complete: {status}")
            if payload.traceback_text:
                self.node_debug_log.appendPlainText(payload.traceback_text)
            if operation == "generate":
                self._render_from_node()
        elif isinstance(payload, Path):
            self._mark_backend_nodes_passed(f"Wrote {payload}")
            self._set_node_status(f"{operation.replace('_', ' ').title()} complete: {payload}")
            self._refresh_backend_artifacts()
        else:
            self._mark_backend_nodes_passed(f"{operation} complete")
            self._set_node_status(f"{operation.replace('_', ' ').title()} complete")
        self._redraw_node_graph()
        self._logger.info("Backend %s complete", operation)
        self._append_gui_log("Backend", "complete", operation)
        self._finish_task_progress("Complete")
        self._set_backend_busy(False, f"Complete: {operation.replace('_', ' ')}")

    def _on_backend_service_worker_failed(
        self,
        operation: str,
        message: str,
        traceback_text: str,
    ) -> None:
        if self._closing:
            return
        for node in self._node_graph.nodes.values():
            if node.status == "running":
                node.status = "failed"
                node.message = message
        self._logger.error("Backend %s failed: %s\n%s", operation, message, traceback_text)
        self._append_gui_log("Backend", "failed", f"{operation}: {message}", traceback_text)
        self._redraw_node_graph()
        self._fail_task_progress("Failed")
        self._set_backend_busy(False, f"{operation.replace('_', ' ')} failed: {message}")
        self._show_graph_error(
            f"Backend {operation.replace('_', ' ')} failed",
            RuntimeError(message),
        )
        if hasattr(self, "node_debug_log"):
            self.node_debug_log.appendPlainText(traceback_text)

    def _apply_backend_check_result(self, result: BackendCheckResult) -> None:
        status = (
            "passed"
            if result.overall_ready
            else "failed"
            if result.traceback_text
            else "warning"
        )
        message = "Backend-ready" if result.overall_ready else "Review backend report"
        for node in self._node_graph.nodes.values():
            if node.type_id in {
                "project",
                "machine_backend",
                "mandrel_backend",
                "tow_backend",
                "layer_backend",
                "hoop_layer",
                "geodesic_layer",
                "non_geodesic_layer",
                "layer_stack_backend",
                "pattern_optimisation_backend",
                "coverage_mode",
                "validation_backend",
                "backend_check",
                "plot_backend",
                "csv_backend_export",
                "gcode_backend_export",
                "report_export",
            }:
                node.status = cast(Any, status)
                node.message = message
        for label, passed in result.checks.items():
            if label == "Pattern optimisation":
                self._set_nodes_by_type_status(
                    ("pattern_optimisation_backend",),
                    "passed" if passed else "failed",
                    label,
                )
            elif label in {"Machine kinematics", "Config"}:
                self._set_nodes_by_type_status(
                    ("machine_backend", "validation_backend"),
                    "passed" if passed else "failed",
                    label,
                )
            elif label == "Exports":
                self._set_nodes_by_type_status(
                    ("csv_backend_export", "gcode_backend_export", "report_export"),
                    "passed" if passed else "failed",
                    label,
                )

    def _set_nodes_by_type_status(
        self,
        type_ids: tuple[str, ...],
        status: str,
        message: str,
    ) -> None:
        for node in self._node_graph.nodes.values():
            if node.type_id in type_ids:
                node.status = cast(Any, status)
                node.message = message

    def _mark_backend_nodes_passed(self, message: str) -> None:
        for node in self._node_graph.nodes.values():
            if node.status == "running":
                node.status = "passed"
                node.message = message

    def _refresh_backend_artifacts(self) -> None:
        output_directory = (
            self._last_backend_check.output_directory
            if self._last_backend_check is not None
            else self._backend_service.build_project_from_graph(self._node_graph).output.directory
        )
        self._load_backend_artifacts(output_directory)
        self._set_node_status(f"Loaded backend artifacts from {output_directory}")

    def _load_backend_artifacts(self, output_directory: Path | None) -> None:
        if output_directory is None:
            return
        reports = self._backend_service.load_reports(output_directory)
        plots = self._backend_service.load_plots(output_directory)
        self._last_loaded_reports = reports
        self._last_loaded_plots = plots
        self._populate_report_list(reports)
        self._populate_plot_list(plots)

    def _update_backend_report_panel(self, result: BackendCheckResult) -> None:
        if not hasattr(self, "backend_report_summary"):
            return
        lines = [
            f"Backend-ready: {str(result.overall_ready).lower()}",
            f"Machine-ready: {str(result.machine_ready).lower()}",
            f"Summary hash: {result.summary_hash or 'not available'}",
        ]
        lines.extend(
            f"{label}: {'PASS' if passed else 'FAIL'}"
            for label, passed in result.checks.items()
        )
        if result.output_directory is not None:
            lines.append(f"Output: {result.output_directory}")
        self.backend_report_summary.setText("\n".join(lines))
        if hasattr(self, "report_detail"):
            detail = result.log
            if result.traceback_text:
                detail = f"{detail}\n\n{result.traceback_text}"
            self.report_detail.setPlainText(detail)

    def _populate_report_list(self, reports: LoadedReportSet) -> None:
        if not hasattr(self, "report_list"):
            return
        self.report_list.clear()
        for name, path in sorted(reports.paths.items()):
            item = self._qt_widgets.QListWidgetItem(name.replace("_", " ").title())
            item.setData(self._qt_core.Qt.ItemDataRole.UserRole, str(path))
            self.report_list.addItem(item)

    def _populate_plot_list(self, plots: LoadedPlotSet) -> None:
        if not hasattr(self, "plot_list"):
            return
        self.plot_list.clear()
        for path in plots.plots:
            item = self._qt_widgets.QListWidgetItem(path.name)
            item.setData(self._qt_core.Qt.ItemDataRole.UserRole, str(path))
            self.plot_list.addItem(item)
        if self.plot_list.count() and self.plot_list.currentItem() is None:
            self.plot_list.setCurrentRow(0)

    def _show_report_item(self, current: Any | None) -> None:
        if current is None or not hasattr(self, "report_detail"):
            return
        path = Path(str(current.data(self._qt_core.Qt.ItemDataRole.UserRole)))
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self.report_detail.setPlainText(f"Could not load report {path}: {exc}")
            return
        self.report_detail.setPlainText(json.dumps(parsed, indent=2, sort_keys=True))

    def _show_plot_item(self, current: Any | None) -> None:
        if current is None:
            return
        self._current_plot_path = Path(
            str(current.data(self._qt_core.Qt.ItemDataRole.UserRole))
        )
        self._fit_current_plot()

    def _fit_current_plot(self) -> None:
        if self._current_plot_path is None or not hasattr(self, "plot_preview"):
            return
        pixmap = self._qt_gui.QPixmap(str(self._current_plot_path))
        if pixmap.isNull():
            self.plot_preview.setText(f"Could not load plot: {self._current_plot_path}")
            return
        available_size = self.plot_preview.size()
        if available_size.width() > 40 and available_size.height() > 40:
            pixmap = pixmap.scaled(
                available_size,
                self._qt_core.Qt.AspectRatioMode.KeepAspectRatio,
                self._qt_core.Qt.TransformationMode.SmoothTransformation,
            )
        self.plot_preview.setPixmap(pixmap)
        self.plot_preview.setToolTip(str(self._current_plot_path))

    def _open_current_plot_external(self) -> None:
        if self._current_plot_path is None:
            self._set_node_status("No plot selected")
            return
        self._qt_gui.QDesktopServices.openUrl(
            self._qt_core.QUrl.fromLocalFile(str(self._current_plot_path))
        )

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
                points = self._display_layer_points(
                    layer,
                    mandrel_length_mm=mandrel.length_mm,
                    center_z_mm=center_z,
                )
                self._visuals.append(
                    scene.visuals.Line(
                        pos=points,
                        color=colors[index % len(colors)],
                        width=2.0,
                        parent=self.view.scene,
                    )
                )
                self._visuals[-1].set_gl_state("opaque", depth_test=True)
        if self._viewport_show_tow_band():
            for index, layer in enumerate(program.layers):
                vertices, faces = self._display_tow_band_mesh(
                    layer,
                    mandrel_length_mm=mandrel.length_mm,
                    center_z_mm=center_z,
                )
                if faces.size == 0:
                    continue
                color = colors[index % len(colors)]
                self._visuals.append(
                    scene.visuals.Mesh(
                        vertices=vertices,
                        faces=faces,
                        color=(color[0], color[1], color[2], 0.38),
                        shading="flat",
                        parent=self.view.scene,
                    )
                )
                self._visuals[-1].set_gl_state("translucent", depth_test=True, cull_face=False)
        self.status.setText(
            f"Mode: node graph\nLayers: {len(program.layers)}\nPoints: {program.point_count}"
        )

    def _render_pin_layout_visuals(self, config: WindingJobConfig, mandrel: Any) -> None:
        pins = config.pin_layout
        if not pins.enabled:
            return
        pin_rows = self._pin_layout_points(config, mandrel)
        if not pin_rows:
            return
        scene = self._vispy_scene
        for start, end in pin_rows:
            display = orient_points_for_horizontal_view(
                np.asarray([start, end], dtype=float),
                length_mm=mandrel.length_mm,
            )
            self._visuals.append(
                scene.visuals.Line(
                    pos=display,
                    color=(1.0, 0.82, 0.18, 1.0),
                    width=max(2.0, float(pins.pin_radius_mm)),
                    parent=self.view.scene,
                )
            )
            self._visuals[-1].set_gl_state("opaque", depth_test=True)

    def _pin_layout_points(self, config: WindingJobConfig, mandrel: Any) -> list[tuple[Any, Any]]:
        pins = config.pin_layout
        length = float(mandrel.length_mm)
        left = pins.left_shoulder_z_mm
        right = pins.right_shoulder_z_mm
        if left is None:
            left = config.mandrel.left_dome_length_mm or length * 0.25
        if right is None:
            right = length - (config.mandrel.right_dome_length_mm or length * 0.25)
        shoulders = ("left", "right") if pins.shoulders == "both" else (pins.shoulders,)
        z_by_shoulder = {"left": float(left), "right": float(right)}
        rows = []
        step = 360.0 / max(1, int(pins.count_per_shoulder))
        for shoulder in shoulders:
            z_mm = max(0.0, min(length, z_by_shoulder[shoulder]))
            radius = float(mandrel.radius_at(np.asarray([z_mm], dtype=float))[0])
            for index in range(max(1, int(pins.count_per_shoulder))):
                phi = math.radians((float(pins.angular_offset_deg) + index * step) % 360.0)
                radial = np.asarray([math.cos(phi), math.sin(phi), 0.0], dtype=float)
                surface = np.asarray([radius * radial[0], radius * radial[1], z_mm], dtype=float)
                start = surface + radial * float(pins.pin_standoff_mm)
                end = start + radial * float(pins.pin_height_mm)
                rows.append((start, end))
        return rows

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
        camera_state = self._capture_mandrel_camera_state()
        try:
            self._render_scene_contents()
        finally:
            self._restore_mandrel_camera_state(camera_state)

    def _capture_mandrel_camera_state(self) -> dict[str, Any] | None:
        if not hasattr(self, "view") or not hasattr(self.view, "camera"):
            return None
        camera = self.view.camera
        return {
            "center": tuple(camera.center),
            "distance": camera.distance,
            "scale_factor": camera.scale_factor,
            "quaternion": camera._quaternion,
        }

    def _restore_mandrel_camera_state(self, camera_state: dict[str, Any] | None) -> None:
        if camera_state is None or not hasattr(self, "view") or not hasattr(self.view, "camera"):
            return
        camera = self.view.camera
        camera.center = camera_state["center"]
        camera.distance = camera_state["distance"]
        camera.scale_factor = camera_state["scale_factor"]
        camera._quaternion = camera_state["quaternion"]
        camera.view_changed()

    def _render_scene_contents(self) -> None:
        if self._uses_backend_service_graph():
            self._render_node_graph_scene()
            return
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

    def _render_node_graph_scene(self) -> None:
        try:
            self._sync_node_graph_from_scene()
            config = self._backend_service.build_project_from_graph(self._node_graph)
            mandrel = self._preview_mandrel_from_config(config)
            schedule = self._preview_schedule_from_config(config)
            program = plan_winding_schedule(mandrel, schedule)
        except (OSError, ValueError) as exc:
            self.status.setText(f"Invalid node graph preview: {exc}")
            return
        scene = self._vispy_scene
        for visual in self._visuals:
            visual.parent = None
        self._visuals.clear()
        if isinstance(mandrel, CylinderMandrel):
            mesh_vertices, mesh_faces = cylinder_mesh_arrays(
                mandrel,
                theta_segments=96,
                z_segments=32,
            )
            radius_mm = mandrel.radius_mm
        else:
            mesh_vertices, mesh_faces = profile_mesh_arrays(
                mandrel,
                theta_segments=96,
                z_segments=max(24, min(180, config.mandrel.mesh_points_z)),
            )
            radius_mm = mandrel.max_radius_mm
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
        self._render_pin_layout_visuals(config, mandrel)
        colors = (
            (0.10, 0.58, 1.0, 1.0),
            (1.0, 0.52, 0.12, 1.0),
            (0.30, 0.82, 0.44, 1.0),
            (0.86, 0.36, 0.95, 1.0),
        )
        if self._viewport_show_tow_path():
            for index, layer in enumerate(program.layers):
                points = self._display_layer_points(layer, mandrel_length_mm=mandrel.length_mm)
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
        if self._viewport_show_tow_band():
            for index, layer in enumerate(program.layers):
                vertices, faces = self._display_tow_band_mesh(
                    layer,
                    mandrel_length_mm=mandrel.length_mm,
                )
                if faces.size == 0:
                    continue
                color = colors[index % len(colors)]
                self._visuals.append(
                    scene.visuals.Mesh(
                        vertices=vertices,
                        faces=faces,
                        color=(color[0], color[1], color[2], 0.38),
                        shading="flat",
                        parent=self.view.scene,
                    )
                )
                self._visuals[-1].set_gl_state("translucent", depth_test=True, cull_face=False)
        scale = max(mandrel.length_mm, radius_mm * 6.0)
        self.view.camera.distance = scale * 1.4
        self.view.camera.center = (0.0, 0.0, 0.0)
        self.view.camera.scale_factor = scale
        layer_lines = [
            (
                f"{report.layer_name}: {report.winding_type}, "
                f"{report.actual_angle_deg:.2f} deg, "
                f"{report.coverage_percent:.1f}%"
            )
            for report in program.reports[:5]
        ]
        self.status.setText(
            f"Mode: node graph preview\n"
            f"Mandrel: {config.mandrel.type}, L={mandrel.length_mm:.1f} mm, "
            f"R={radius_mm:.1f} mm\n"
            f"Layers: {len(program.layers)} | Points: {program.point_count}\n"
            + "\n".join(layer_lines)
        )

    def _preview_mandrel_from_config(
        self,
        config: WindingJobConfig,
    ) -> CylinderMandrel | AxisymmetricProfileMandrel:
        if config.mandrel.type == "cylinder":
            return CylinderMandrel(
                length_mm=config.mandrel.length_mm,
                radius_mm=config.mandrel.radius_mm,
                name=config.project.name,
            )
        if config.mandrel.type in {"axisymmetric_profile", "profile"}:
            if config.mandrel.profile_path is None:
                raise ValueError("mandrel profile_path is required")
            return import_dxf_zr_profile(
                config.mandrel.profile_path,
                samples=config.mandrel.samples,
            )
        return cylinder_with_domes_profile(
            cylinder_length_mm=config.mandrel.cylinder_length_mm
            or config.mandrel.length_mm,
            cylinder_radius_mm=config.mandrel.cylinder_radius_mm
            or config.mandrel.radius_mm,
            left_dome_length_mm=config.mandrel.left_dome_length_mm,
            right_dome_length_mm=config.mandrel.right_dome_length_mm,
            polar_opening_radius_mm=config.mandrel.polar_opening_radius_mm,
            samples_per_region=max(16, config.mandrel.mesh_points_z // 3),
            name=config.project.name,
        )

    def _preview_schedule_from_config(self, config: WindingJobConfig) -> WindingSchedule:
        return WindingSchedule(
            layers=tuple(
                WindingLayerSpec(
                    name=layer.name,
                    winding_type=cast(Any, layer.type),
                    target_angle_deg=(
                        layer.target_angle_deg
                        if layer.target_angle_deg is not None
                        else layer.winding_angle_deg
                    ),
                    tow_width_mm=layer.tow_width_mm or config.tow.width_mm,
                    layer_thickness_mm=layer.tow_thickness_mm or config.tow.thickness_mm,
                    coverage_target=layer.coverage_target,
                    direction=cast(Any, layer.direction),
                    point_count=max(2, layer.points),
                    layer_id=layer.name,
                    enabled=layer.enabled,
                    number_of_passes=self._preview_layer_pass_count(layer.passes),
                    start_z_mm=layer.start_z_mm,
                    end_z_mm=layer.end_z_mm,
                    feedrate_mm_min=layer.feedrate_mm_min,
                    transition_points=20,
                    turnaround_radius_mm=layer.turnaround_radius_mm,
                    phase_offset_deg=layer.phase_offset_deg,
                    colour=layer.colour,
                )
                for layer in config.layers
            ),
            radial_clearance_mm=config.machine.clearance_mm,
            nominal_feedrate_mm_min=float(
                next(
                    (
                        layer.feedrate_mm_min
                        for layer in config.layers
                        if layer.enabled and layer.feedrate_mm_min is not None
                    ),
                    500.0,
                )
            ),
        )

    def _preview_layer_pass_count(self, passes: int | str | None) -> int | None:
        if passes is None:
            return None
        if isinstance(passes, str) and passes.strip().lower() in {"", "auto"}:
            return None
        return int(passes)

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
                points = self._display_layer_points(
                    layer,
                    mandrel_length_mm=mandrel.length_mm,
                    center_z_mm=0.5 * mandrel.length_mm,
                )
                self._visuals.append(
                    scene.visuals.Line(
                        pos=points,
                        color=colors[index % len(colors)],
                        width=2.0,
                        parent=self.view.scene,
                    )
                )
                self._visuals[-1].set_gl_state("opaque", depth_test=True)
        if self._viewport_show_tow_band():
            for index, layer in enumerate(program.layers):
                vertices, faces = self._display_tow_band_mesh(
                    layer,
                    mandrel_length_mm=mandrel.length_mm,
                    center_z_mm=0.5 * mandrel.length_mm,
                )
                if faces.size == 0:
                    continue
                color = colors[index % len(colors)]
                self._visuals.append(
                    scene.visuals.Mesh(
                        vertices=vertices,
                        faces=faces,
                        color=(color[0], color[1], color[2], 0.38),
                        shading="flat",
                        parent=self.view.scene,
                    )
                )
                self._visuals[-1].set_gl_state("translucent", depth_test=True, cull_face=False)
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
            self._sync_node_view_state()
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
            self._restore_node_view_state()

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

    def _import_backend_config_dialog(self) -> None:
        filename, _ = self._qt_widgets.QFileDialog.getOpenFileName(
            self.widget,
            "Import backend config",
            "examples",
            "Config Files (*.yaml *.yml *.json);;All Files (*)",
        )
        if not filename:
            return
        graph = self._backend_service.import_config_to_graph(filename)
        self._node_graph = graph
        if hasattr(self, "node_scene"):
            self._redraw_node_graph()
            self._schedule_fit_node_graph()
        config = self._backend_service.build_project_from_graph(graph)
        if hasattr(self, "project_name"):
            self.project_name.setText(config.project.name)
        self._set_gui_status(f"Imported backend config: {filename}")
        self._load_backend_artifacts(config.output.directory)

    def _export_backend_config_dialog(self) -> None:
        filename, _ = self._qt_widgets.QFileDialog.getSaveFileName(
            self.widget,
            "Export backend config",
            "exports/gui_winding_job.yaml",
            "YAML Files (*.yaml *.yml);;JSON Files (*.json);;All Files (*)",
        )
        if not filename:
            return
        self._sync_node_graph_from_scene()
        path = self._backend_service.export_graph_to_config(self._node_graph, filename)
        self._set_gui_status(f"Exported backend config: {path}")

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
        "not_run": "#7d8790",
        "not_configured": "#7d8790",
        "ready": "#4f9f72",
        "running": "#3b82c4",
        "warning": "#c89438",
        "failed": "#d65757",
        "passed": "#51a36f",
        "stale": "#8f6bd3",
        "error": "#d65757",
        "processing": "#3b82c4",
        "complete": "#51a36f",
        "dirty": "#c89438",
    }
    return colors.get(status, "#7d8790")


def _socket_kind_color(kind: str) -> str:
    colors = {
        "project_config": "#6f8fd3",
        "machine_config": "#c89438",
        "mandrel": "#2f9fcf",
        "tow": "#9b6fd3",
        "machine": "#c89438",
        "layer": "#7bbf68",
        "layer_stack": "#5fb56c",
        "pattern_candidates": "#72a5d8",
        "selected_pattern": "#55b3ad",
        "winding_program": "#55b3ad",
        "program": "#55b3ad",
        "coverage": "#8cbf4f",
        "coverage_mode": "#8cbf4f",
        "simulation": "#8f8de0",
        "validation_report": "#a58de0",
        "plots": "#d0a845",
        "exports": "#d08a45",
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


def _float_setting(value: Any, default: float) -> float:
    try:
        if value in {None, ""}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


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
    QWidget#nodeWorkspace {
        background: #0f141a;
    }
    QWidget#nodeToolbar {
        background: #121a23;
        border: 1px solid #263442;
        border-radius: 6px;
    }
    QScrollArea#nodeToolbarScroll {
        background: transparent;
        border: 0;
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
    QWidget#inlineNodeSettings {
        background: #101820;
        border: 1px solid #253342;
        border-radius: 4px;
    }
    QLabel#inlineNodeSettingLabel {
        background: transparent;
        color: #b5c1cc;
        font-size: 10px;
        padding: 2px 0;
    }
    QLineEdit#inlineNodeSettingEditor,
    QDoubleSpinBox#inlineNodeSettingEditor,
    QSpinBox#inlineNodeSettingEditor,
    QComboBox#inlineNodeSettingEditor {
        background: #202830;
        border: 1px solid #3a4652;
        border-radius: 3px;
        min-height: 20px;
        padding: 1px 5px;
        font-size: 10px;
    }
    QLineEdit#inlineNodeSettingEditor:focus,
    QDoubleSpinBox#inlineNodeSettingEditor:focus,
    QSpinBox#inlineNodeSettingEditor:focus,
    QComboBox#inlineNodeSettingEditor:focus {
        border-color: #4f8dcc;
        background: #24313c;
    }
    QCheckBox#inlineNodeSettingEditor {
        background: transparent;
        spacing: 4px;
    }
    QPushButton {
        background: #263341;
        border: 1px solid #3a4a5a;
        border-radius: 5px;
        padding: 6px 10px;
    }
    QWidget#nodeToolbar QPushButton {
        background: #1b2530;
        border-color: #344656;
        padding: 5px 9px;
        min-height: 24px;
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
