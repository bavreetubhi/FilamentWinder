"""Rapid mandrel and winding settings debug GUI.

Run from the repository root:

    .venv\\Scripts\\python examples\\mandrel_settings_debug_gui.py

The left panel exposes the complete WindingJobConfig as editable scalar leaves.
Press "Rebuild" to regenerate the backend winding job and render the mandrel
and centerline path on the right.
"""

from __future__ import annotations

import json
import sys
import tempfile
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from filament_winder.app.backend_service import winding_config_to_mapping  # noqa: E402
from filament_winder.app.preview import (  # noqa: E402
    cylinder_mesh_arrays,
    offset_display_surface,
    orient_points_for_horizontal_view,
    profile_mesh_arrays,
)
from filament_winder.config import WindingJobConfig, load_winding_config  # noqa: E402
from filament_winder.core.geometry import AxisymmetricProfileMandrel, CylinderMandrel  # noqa: E402
from filament_winder.services import generate_winding_job  # noqa: E402

DEFAULT_CONFIG_PATH = REPO_ROOT / "examples" / "demo_domed_pressure_vessel.yaml"
SCALAR_TYPES = (str, int, float, bool, type(None))


def main() -> int:
    try:
        from PySide6 import QtCore, QtGui, QtWidgets
        from vispy import scene
    except ImportError as exc:
        print("Install GUI dependencies with: pip install -e .[gui]")
        print(exc)
        return 2

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    window = MandrelSettingsDebugWindow(QtCore, QtGui, QtWidgets, scene)
    window.resize(1500, 950)
    window.show()
    return int(app.exec())


class MandrelSettingsDebugWindow:
    def __init__(self, qt_core: Any, qt_gui: Any, qt_widgets: Any, vispy_scene: Any) -> None:
        self.QtCore = qt_core
        self.QtGui = qt_gui
        self.QtWidgets = qt_widgets
        self.scene = vispy_scene
        self.mapping = _default_mapping()
        self._item_paths: dict[int, tuple[Any, ...]] = {}
        self._updating_tree = False
        self._temp_output_dir = Path(tempfile.gettempdir()) / "filament_winder_debug_gui"

        self.window = qt_widgets.QMainWindow()
        self.window.setWindowTitle("Filament Winder Mandrel Settings Debug GUI")
        self.window.setCentralWidget(self._build_ui())
        self._populate_tree()
        self._rebuild()

    def resize(self, width: int, height: int) -> None:
        self.window.resize(width, height)

    def show(self) -> None:
        self.window.show()

    def _build_ui(self) -> Any:
        qt_widgets = self.QtWidgets
        splitter = qt_widgets.QSplitter()

        left = qt_widgets.QWidget()
        left_layout = qt_widgets.QVBoxLayout(left)

        toolbar = qt_widgets.QHBoxLayout()
        self.load_button = qt_widgets.QPushButton("Load YAML")
        self.save_button = qt_widgets.QPushButton("Save YAML")
        self.reset_button = qt_widgets.QPushButton("Reset demo")
        toolbar.addWidget(self.load_button)
        toolbar.addWidget(self.save_button)
        toolbar.addWidget(self.reset_button)
        left_layout.addLayout(toolbar)

        self.fast_preview = qt_widgets.QCheckBox("Fast preview: disable exports/plots")
        self.fast_preview.setChecked(True)
        left_layout.addWidget(self.fast_preview)

        self.tree = qt_widgets.QTreeWidget()
        self.tree.setColumnCount(3)
        self.tree.setHeaderLabels(("Setting", "Value", "Type"))
        self.tree.setAlternatingRowColors(True)
        self.tree.setUniformRowHeights(True)
        self.tree.itemChanged.connect(self._on_item_changed)
        left_layout.addWidget(self.tree, stretch=1)

        action_row = qt_widgets.QHBoxLayout()
        self.rebuild_button = qt_widgets.QPushButton("Rebuild")
        self.expand_button = qt_widgets.QPushButton("Expand all")
        self.collapse_button = qt_widgets.QPushButton("Collapse all")
        action_row.addWidget(self.rebuild_button)
        action_row.addWidget(self.expand_button)
        action_row.addWidget(self.collapse_button)
        left_layout.addLayout(action_row)

        self.status = qt_widgets.QPlainTextEdit()
        self.status.setReadOnly(True)
        self.status.setMaximumBlockCount(300)
        self.status.setMinimumHeight(130)
        left_layout.addWidget(self.status)

        right = qt_widgets.QWidget()
        right_layout = qt_widgets.QVBoxLayout(right)
        self.canvas = self.scene.SceneCanvas(keys="interactive", bgcolor="#101820")
        self.canvas.create_native()
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = "turntable"
        right_layout.addWidget(self.canvas.native, stretch=1)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes((520, 980))

        self.load_button.clicked.connect(self._load_yaml)
        self.save_button.clicked.connect(self._save_yaml)
        self.reset_button.clicked.connect(self._reset_demo)
        self.rebuild_button.clicked.connect(self._rebuild)
        self.expand_button.clicked.connect(self.tree.expandAll)
        self.collapse_button.clicked.connect(self.tree.collapseAll)
        return splitter

    def _populate_tree(self) -> None:
        self._updating_tree = True
        self.tree.clear()
        self._item_paths.clear()
        self._add_mapping_items(self.tree.invisibleRootItem(), self.mapping, ())
        self.tree.resizeColumnToContents(0)
        self.tree.resizeColumnToContents(2)
        self._updating_tree = False

    def _add_mapping_items(self, parent: Any, value: Any, path: tuple[Any, ...]) -> None:
        if isinstance(value, dict):
            for key in sorted(value):
                child = self._new_item(str(key), "", "section", editable=False)
                parent.addChild(child)
                self._add_mapping_items(child, value[key], (*path, key))
            return
        if isinstance(value, list):
            for index, item in enumerate(value):
                label = f"[{index}]"
                child = self._new_item(label, "", "list item", editable=False)
                parent.addChild(child)
                self._add_mapping_items(child, item, (*path, index))
            return
        label = str(path[-1]) if path else "root"
        item = self._new_item(label, _format_value(value), type(value).__name__, editable=True)
        parent.addChild(item)
        self._item_paths[id(item)] = path

    def _new_item(self, setting: str, value: str, type_name: str, *, editable: bool) -> Any:
        item = self.QtWidgets.QTreeWidgetItem((setting, value, type_name))
        flags = item.flags()
        if editable:
            item.setFlags(flags | self.QtCore.Qt.ItemIsEditable)
        else:
            item.setFlags(flags & ~self.QtCore.Qt.ItemIsEditable)
            item.setExpanded(True)
        return item

    def _on_item_changed(self, item: Any, column: int) -> None:
        if self._updating_tree or column != 1:
            return
        path = self._item_paths.get(id(item))
        if path is None:
            return
        try:
            old_value = _get_path(self.mapping, path)
            new_value = _parse_value(item.text(1), old_value)
            _set_path(self.mapping, path, new_value)
            item.setText(2, type(new_value).__name__)
            self._log(f"Changed {_path_text(path)} = {_format_value(new_value)}")
        except Exception as exc:  # noqa: BLE001 - keep GUI alive for debugging.
            self._log(f"Invalid value for {_path_text(path)}: {exc}")
            self._updating_tree = True
            item.setText(1, _format_value(_get_path(self.mapping, path)))
            self._updating_tree = False

    def _load_yaml(self) -> None:
        path, _selected = self.QtWidgets.QFileDialog.getOpenFileName(
            self.window,
            "Load winding YAML",
            str(REPO_ROOT),
            "YAML files (*.yaml *.yml);;All files (*.*)",
        )
        if not path:
            return
        try:
            self.mapping = winding_config_to_mapping(load_winding_config(path))
            self._populate_tree()
            self._log(f"Loaded {path}")
            self._rebuild()
        except Exception:  # noqa: BLE001 - show full traceback in GUI.
            self._log(traceback.format_exc())

    def _save_yaml(self) -> None:
        path, _selected = self.QtWidgets.QFileDialog.getSaveFileName(
            self.window,
            "Save winding YAML",
            str(REPO_ROOT / "exports" / "debug_settings.yaml"),
            "YAML files (*.yaml *.yml);;JSON files (*.json);;All files (*.*)",
        )
        if not path:
            return
        output = Path(path)
        try:
            if output.suffix.lower() == ".json":
                output.write_text(json.dumps(self.mapping, indent=2), encoding="utf-8")
            else:
                output.write_text(_dump_simple_yaml(self.mapping), encoding="utf-8")
            self._log(f"Saved {output}")
        except Exception:  # noqa: BLE001
            self._log(traceback.format_exc())

    def _reset_demo(self) -> None:
        self.mapping = _default_mapping()
        self._populate_tree()
        self._log("Reset to demo domed pressure vessel config")
        self._rebuild()

    def _rebuild(self) -> None:
        self.rebuild_button.setEnabled(False)
        try:
            preview_mapping = deepcopy(self.mapping)
            if self.fast_preview.isChecked():
                _force_fast_preview_outputs(preview_mapping, self._temp_output_dir)
            config = WindingJobConfig.from_mapping(preview_mapping)
            result = generate_winding_job(
                config,
                export_csv=False,
                export_summary=False,
                make_plots=False,
            )
            self._draw_result(result)
            self._log(_result_summary(result))
        except Exception:  # noqa: BLE001
            self._log(traceback.format_exc())
        finally:
            self.rebuild_button.setEnabled(True)

    def _draw_result(self, result: Any) -> None:
        self.view.scene.children.clear()
        mandrel = result.mandrel
        vertices, faces = _mandrel_mesh(mandrel)
        center_z = _center_z(mandrel)
        display_vertices = orient_points_for_horizontal_view(
            vertices,
            length_mm=mandrel.length_mm,
            center_z_mm=center_z,
        )
        mesh = self.scene.visuals.Mesh(
            vertices=display_vertices,
            faces=faces,
            color=(0.18, 0.25, 0.30, 0.55),
            shading="smooth",
            parent=self.view.scene,
        )
        mesh.set_gl_state("translucent", depth_test=True, cull_face=False)

        colors = {
            "wind": (1.0, 0.58, 0.10, 0.92),
            "DomeTurnaround": (1.0, 0.16, 0.62, 0.98),
            "transition": (0.82, 0.86, 0.90, 0.75),
            "other": (0.35, 0.80, 1.0, 0.85),
        }
        for points, label in _path_chunks(result.program, mandrel.length_mm, center_z):
            visual = self.scene.visuals.Line(
                pos=points,
                color=colors.get(label, colors["other"]),
                width=2.0 if label != "DomeTurnaround" else 2.8,
                parent=self.view.scene,
            )
            visual.set_gl_state("opaque", depth_test=True)

        axis = self.scene.visuals.XYZAxis(parent=self.view.scene)
        axis_scale = max(mandrel.length_mm, 1.0) * 0.08
        axis.transform = self.scene.transforms.STTransform(scale=(axis_scale,) * 3)
        scale = max(mandrel.length_mm, _mandrel_radius(mandrel) * 6.0, 1.0)
        self.view.camera.set_range(
            x=(-scale * 0.55, scale * 0.55),
            y=(-scale * 0.35, scale * 0.35),
            z=(-scale * 0.35, scale * 0.35),
        )

    def _log(self, message: str) -> None:
        self.status.appendPlainText(message.rstrip())


def _default_mapping() -> dict[str, Any]:
    return winding_config_to_mapping(load_winding_config(DEFAULT_CONFIG_PATH))


def _force_fast_preview_outputs(mapping: dict[str, Any], output_dir: Path) -> None:
    output = mapping.setdefault("output", {})
    output["directory"] = str(output_dir)
    output["csv"] = False
    output["gcode"] = False
    output["summary_json"] = False
    output["segments_json"] = False
    output["validation_report_json"] = False
    output["coverage_grid"] = False
    plot = mapping.setdefault("plot", {})
    plot["enabled"] = False
    plot["show"] = False
    plot["save"] = False


def _format_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float, str)):
        return str(value)
    return json.dumps(value)


def _parse_value(text: str, old_value: Any) -> Any:
    stripped = text.strip()
    if isinstance(old_value, bool):
        if stripped.lower() in {"true", "1", "yes", "on"}:
            return True
        if stripped.lower() in {"false", "0", "no", "off"}:
            return False
        raise ValueError("expected boolean")
    if old_value is None:
        if stripped.lower() in {"", "none", "null"}:
            return None
        return _json_or_string(stripped)
    if isinstance(old_value, int) and not isinstance(old_value, bool):
        return int(float(stripped))
    if isinstance(old_value, float):
        return float(stripped)
    if isinstance(old_value, str):
        return stripped
    return _json_or_string(stripped)


def _json_or_string(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _dump_simple_yaml(value: Any, *, indent: int = 0) -> str:
    if isinstance(value, dict):
        lines: list[str] = []
        for key, item in value.items():
            prefix = " " * indent + f"{key}:"
            if _is_scalar(item):
                lines.append(f"{prefix} {_yaml_scalar(item)}")
            else:
                lines.append(prefix)
                lines.append(_dump_simple_yaml(item, indent=indent + 2).rstrip())
        return "\n".join(lines) + "\n"
    if isinstance(value, list):
        lines = []
        for item in value:
            if _is_scalar(item):
                lines.append(" " * indent + f"- {_yaml_scalar(item)}")
            elif isinstance(item, dict):
                lines.extend(_dump_yaml_list_mapping_item(item, indent=indent))
            else:
                lines.append(" " * indent + "-")
                lines.append(_dump_simple_yaml(item, indent=indent + 2).rstrip())
        return "\n".join(lines) + "\n"
    return " " * indent + _yaml_scalar(value) + "\n"


def _dump_yaml_list_mapping_item(mapping: dict[str, Any], *, indent: int) -> list[str]:
    if not mapping:
        return [" " * indent + "- {}"]
    items = list(mapping.items())
    first_key, first_value = items[0]
    prefix = " " * indent + f"- {first_key}:"
    lines: list[str] = []
    if _is_scalar(first_value):
        lines.append(f"{prefix} {_yaml_scalar(first_value)}")
    else:
        lines.append(prefix)
        lines.append(_dump_simple_yaml(first_value, indent=indent + 4).rstrip())
    for key, value in items[1:]:
        nested_prefix = " " * (indent + 2) + f"{key}:"
        if _is_scalar(value):
            lines.append(f"{nested_prefix} {_yaml_scalar(value)}")
        else:
            lines.append(nested_prefix)
            lines.append(_dump_simple_yaml(value, indent=indent + 4).rstrip())
    return lines


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool)) or value is None


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        if value == "":
            return '""'
        if any(char in value for char in ":#[]{}-,&*!?|<>=@`\"'\n\t") or value.strip() != value:
            return json.dumps(value)
        lowered = value.lower()
        if lowered in {"true", "false", "null", "none", "yes", "no", "on", "off"}:
            return json.dumps(value)
        return value
    return str(value)


def _get_path(root: Any, path: tuple[Any, ...]) -> Any:
    current = root
    for part in path:
        current = current[part]
    return current


def _set_path(root: Any, path: tuple[Any, ...], value: Any) -> None:
    current = root
    for part in path[:-1]:
        current = current[part]
    current[path[-1]] = value


def _path_text(path: tuple[Any, ...]) -> str:
    parts = []
    for part in path:
        if isinstance(part, int):
            parts.append(f"[{part}]")
        else:
            parts.append(str(part))
    return ".".join(parts).replace(".[", "[")


def _mandrel_mesh(mandrel: Any) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(mandrel, CylinderMandrel):
        return cylinder_mesh_arrays(mandrel, theta_segments=96, z_segments=32)
    if isinstance(mandrel, AxisymmetricProfileMandrel):
        return profile_mesh_arrays(mandrel, theta_segments=128, z_segments=96)
    raise TypeError(f"unsupported mandrel type: {type(mandrel)!r}")


def _path_chunks(
    program: Any,
    length_mm: float,
    center_z: float | None,
) -> list[tuple[np.ndarray, str]]:
    labels = tuple(getattr(program.metadata, "motion_type", ()))
    if not labels:
        labels = tuple("wind" for _ in range(program.point_count))
    labels_array = np.asarray(labels, dtype=object)
    chunks: list[tuple[np.ndarray, str]] = []
    start = 0
    for index in range(1, program.point_count):
        if labels_array[index] != labels_array[index - 1]:
            _append_path_chunk(
                chunks,
                program.path.points_mm[start:index],
                str(labels_array[start]),
                length_mm,
                center_z,
            )
            start = index
    _append_path_chunk(
        chunks,
        program.path.points_mm[start:],
        str(labels_array[start]),
        length_mm,
        center_z,
    )
    return chunks


def _append_path_chunk(
    chunks: list[tuple[np.ndarray, str]],
    points: np.ndarray,
    label: str,
    length_mm: float,
    center_z: float | None,
) -> None:
    if points.shape[0] < 2:
        return
    display = offset_display_surface(
        orient_points_for_horizontal_view(points, length_mm=length_mm, center_z_mm=center_z),
        offset_mm=0.85,
    )
    chunks.append((display, label))


def _center_z(mandrel: Any) -> float | None:
    if isinstance(mandrel, AxisymmetricProfileMandrel):
        return 0.5 * (mandrel.start_z_mm + mandrel.end_z_mm)
    return None


def _mandrel_radius(mandrel: Any) -> float:
    if isinstance(mandrel, CylinderMandrel):
        return mandrel.radius_mm
    return mandrel.max_radius_mm


def _result_summary(result: Any) -> str:
    summary = result.summary
    lines = [
        "Rebuilt",
        f"points: {result.program.point_count}",
        f"layers: {len(result.program.layers)}",
        f"backend_ready: {summary.get('backend_ready')}",
        f"machine_ready: {summary.get('machine_ready')}",
        f"estimated_time_min: {summary.get('estimated_winding_time_min')}",
    ]
    coverage = summary.get("coverage_summary", {})
    if isinstance(coverage, dict):
        lines.append(
            "coverage: "
            f"covered={coverage.get('covered_percent')}%, "
            f"gap={coverage.get('gap_percent')}%, "
            f"overlap={coverage.get('overlap_percent')}%"
        )
    reports = [
        f"{report.layer_name}: {report.winding_type} passes={report.circuits} "
        f"gap={report.gap_mm:.3f} overlap={report.overlap_mm:.3f}"
        for report in result.program.reports
    ]
    lines.extend(reports)
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
