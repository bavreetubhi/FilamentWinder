from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from filament_winder.app.backend_service import BackendService, graph_to_config_mapping
from filament_winder.app.gui import _PreviewWindow
from filament_winder.app.node_graph import (
    NodeGraphController,
    NodeGraphExecutor,
    NodeGraphState,
    NodeInstance,
    NodeTypeDefinition,
    addable_node_type_ids,
    default_backend_winding_graph,
    default_filament_winder_graph,
    default_node_registry,
    validate_node_type_definition,
)
from filament_winder.project import CylinderMandrelConfig, WindingConfig, WindingProject


def test_default_node_graph_executes_backend_program() -> None:
    graph = default_filament_winder_graph(
        length_mm=120.0,
        radius_mm=20.0,
        tow_width_mm=8.0,
        angle_deg=45.0,
        point_count=16,
    )
    result = NodeGraphExecutor(default_node_registry()).execute(graph)

    assert not result.warnings
    assert len(result.executed_node_ids) == len(graph.nodes)
    assert any("program" in outputs for outputs in result.node_outputs.values())
    assert any("coverage" in outputs for outputs in result.node_outputs.values())


def test_node_graph_rejects_invalid_socket_links() -> None:
    registry = default_node_registry()
    graph = NodeGraphState()
    mandrel = graph.add_node("mandrel_profile", registry)
    tow = graph.add_node("material_tow", registry)

    with pytest.raises(ValueError):
        graph.add_link(mandrel.id, "mandrel", tow.id, "tow", registry)


def test_add_each_registered_node_type_does_not_crash() -> None:
    registry = default_node_registry()
    graph = NodeGraphState()
    controller = NodeGraphController(graph, registry)

    for index, type_id in enumerate(addable_node_type_ids(registry)):
        result = controller.add_node(type_id, x=float(index * 10), y=25.0)
        assert result.success, result.error
        assert result.node_id in graph.nodes


def test_add_node_with_default_settings_and_valid_sockets() -> None:
    registry = default_node_registry()
    for type_id in addable_node_type_ids(registry):
        definition = registry[type_id]
        validate_node_type_definition(definition)
        result = NodeGraphController(NodeGraphState(), registry).add_node(type_id)
        assert result.success


def test_add_invalid_node_type_returns_error() -> None:
    graph = NodeGraphState()
    result = NodeGraphController(graph, default_node_registry()).add_node("missing_node")

    assert not result.success
    assert result.error is not None
    assert graph.nodes == {}


def test_add_node_rollback_on_failure() -> None:
    class ExplodingGraph(NodeGraphState):
        def add_node(
            self,
            type_id: str,
            registry: dict[str, NodeTypeDefinition],
            *,
            name: str | None = None,
            x: float = 0.0,
            y: float = 0.0,
        ) -> NodeInstance:
            node = super().add_node(type_id, registry, name=name, x=x, y=y)
            raise RuntimeError(f"boom after {node.id}")

    graph = ExplodingGraph()
    result = NodeGraphController(graph, default_node_registry()).add_node("material_tow")

    assert not result.success
    assert graph.nodes == {}


def test_add_node_serializes_and_save_load_project() -> None:
    registry = default_node_registry()
    graph = NodeGraphState()
    result = NodeGraphController(graph, registry).add_node("material_tow", x=10.0, y=20.0)
    assert result.success
    project = WindingProject(
        name="single node",
        mandrel=CylinderMandrelConfig(length_mm=100.0, radius_mm=20.0),
        winding=WindingConfig(tow_width_mm=6.0, winding_angle_deg=45.0, point_count=12),
        graph=graph.to_dict(),
    )

    restored = NodeGraphState.from_dict(WindingProject.from_dict(project.to_dict()).graph, registry)

    assert tuple(restored.nodes) == tuple(graph.nodes)
    assert next(iter(restored.nodes.values())).x == 10.0


def test_main_window_uses_top_bottom_layout() -> None:
    source = inspect.getsource(_PreviewWindow.__init__)

    assert "QSplitter(self._qt_core.Qt.Orientation.Vertical)" in source
    assert "_build_viewport_panel(self.canvas.native)" in source
    assert "main_splitter.setSizes([700, 350])" in source


def test_node_workspace_uses_canvas_left_tools_right_layout() -> None:
    source = inspect.getsource(_PreviewWindow._build_node_workspace)

    assert "QSplitter(self._qt_core.Qt.Orientation.Horizontal)" in source
    assert "bottom_tabs.addTab(library_panel, \"Node Library\")" in source
    assert "container_layout.addLayout(toolbar)" in source


def test_gui_button_connections_use_safe_action_wrapper() -> None:
    source = inspect.getsource(_PreviewWindow)

    assert "def run_safe_action" in source
    assert "def handle_gui_error" in source
    assert "def _connect_safe_button" in source
    assert source.count(".clicked.connect(") == 1
    assert "button.clicked.connect(" in source


def test_node_graph_accepts_valid_socket_link_and_marks_downstream_stale() -> None:
    registry = default_node_registry()
    graph = NodeGraphState()
    mandrel = graph.add_node("mandrel_profile", registry)
    layer_stack = graph.add_node("layer_stack", registry)
    pattern = graph.add_node("winding_pattern", registry)
    graph.add_link(mandrel.id, "mandrel", pattern.id, "mandrel", registry)
    graph.add_link(layer_stack.id, "layer_stack", pattern.id, "layer_stack", registry)

    graph.nodes[pattern.id].status = "complete"
    graph.update_node_settings(mandrel.id, {"radius_mm": 22.0})

    assert graph.nodes[mandrel.id].status == "stale"
    assert graph.nodes[pattern.id].status == "stale"


def test_node_graph_round_trips_through_project_dict() -> None:
    registry = default_node_registry()
    graph = default_filament_winder_graph(point_count=12)
    first_node = next(iter(graph.nodes.values()))
    first_node.collapsed = True
    first_node.settings["custom_test_value"] = 123
    graph.group_nodes(tuple(list(graph.nodes)[:2]), name="Variant A")
    project = WindingProject(
        name="node graph project",
        mandrel=CylinderMandrelConfig(length_mm=100.0, radius_mm=20.0),
        winding=WindingConfig(tow_width_mm=6.0, winding_angle_deg=45.0, point_count=12),
        graph=graph.to_dict(),
    )
    loaded = WindingProject.from_dict(project.to_dict())
    restored = NodeGraphState.from_dict(loaded.graph, registry)

    assert len(restored.nodes) == len(graph.nodes)
    assert len(restored.links) == len(graph.links)
    assert len(restored.groups) == len(graph.groups)
    assert restored.to_dict()["nodes"][0]["x"] == graph.to_dict()["nodes"][0]["x"]
    assert restored.nodes[first_node.id].collapsed
    assert restored.nodes[first_node.id].settings["custom_test_value"] == 123


def test_node_graph_default_backend_workflow_exists() -> None:
    graph = default_backend_winding_graph()
    type_ids = {node.type_id for node in graph.nodes.values()}

    assert {
        "project",
        "machine_backend",
        "mandrel_backend",
        "tow_backend",
        "layer_stack_backend",
        "pattern_optimisation_backend",
        "validation_backend",
        "backend_check",
        "plot_backend",
        "csv_backend_export",
        "gcode_backend_export",
        "report_export",
    }.issubset(type_ids)
    assert len(graph.topological_node_ids()) == len(graph.nodes)


def test_node_graph_controller_duplicate_delete_and_link_do_not_crash() -> None:
    registry = default_node_registry()
    graph = NodeGraphState()
    controller = NodeGraphController(graph, registry)
    source = controller.add_node("tow_backend")
    target = controller.add_node("layer_stack_backend", x=300.0)

    assert source.success and source.node_id is not None
    assert target.success and target.node_id is not None
    link = controller.link_nodes(source.node_id, "tow", target.node_id, "tow")
    duplicate = controller.duplicate_node(target.node_id)
    delete = controller.delete_nodes((target.node_id,))

    assert link.success
    assert duplicate.success
    assert delete.success
    assert target.node_id not in graph.nodes


def test_graph_to_config_mapping_builds_backend_config() -> None:
    graph = default_backend_winding_graph()
    config = BackendService().build_project_from_graph(graph)
    mapping = graph_to_config_mapping(graph)

    assert config.mandrel.type == "cylinder_with_elliptical_domes"
    assert len(config.layers) == 3
    assert mapping["pattern_selection"]["method"] == "textbook_integer_closure"
    assert mapping["output"]["gcode"] is True


def test_import_config_creates_graph_and_export_graph_to_config(tmp_path: Path) -> None:
    service = BackendService()
    graph = service.import_config_to_graph("examples/demo_domed_pressure_vessel.yaml")
    exported = tmp_path / "gui_export.yaml"

    path = service.export_graph_to_config(graph, exported)
    restored = service.load_config(path)

    assert path.exists()
    assert restored.project.name == "demo_domed_pressure_vessel"
    assert restored.mandrel.type == "cylinder_with_elliptical_domes"
    assert len(restored.layers) == 3


def test_gui_project_save_load_preserves_backend_graph(tmp_path: Path) -> None:
    service = BackendService()
    graph = default_backend_winding_graph()
    first_node = next(iter(graph.nodes.values()))
    first_node.collapsed = True
    project_path = tmp_path / "backend_gui_project.fwgui.json"

    service.save_gui_project(graph, project_path)
    restored = service.load_gui_project(project_path)

    assert len(restored.nodes) == len(graph.nodes)
    assert restored.nodes[first_node.id].collapsed


def test_backend_check_failure_does_not_crash_app() -> None:
    graph = default_backend_winding_graph()
    tow = next(node for node in graph.nodes.values() if node.type_id == "tow_backend")
    tow.settings["width_mm"] = -1.0

    result = BackendService().backend_check_safely(graph)

    assert not result.overall_ready
    assert result.traceback_text
    assert result.checks == {"Backend check": False}


def test_backend_check_button_uses_worker() -> None:
    source = inspect.getsource(_PreviewWindow._start_backend_service_worker)

    assert "QRunnable" in source
    assert "backend_check_safely" in source
    assert "_set_backend_busy(True" in source


def test_plot_browser_loads_manifest(tmp_path: Path) -> None:
    plot_path = tmp_path / "combined_unwrapped.png"
    plot_path.write_bytes(b"not a real png but present")
    (tmp_path / "plot_manifest.json").write_text(
        json.dumps({"plots": [{"type": "combined_unwrapped", "path": str(plot_path)}]}),
        encoding="utf-8",
    )

    plots = BackendService().load_plots(tmp_path)

    assert plots.plots == (plot_path,)
    assert plots.manifest["plots"][0]["type"] == "combined_unwrapped"


def test_report_panel_loads_summary(tmp_path: Path) -> None:
    (tmp_path / "summary.json").write_text(
        json.dumps({"machine_ready": True, "project": {"name": "demo"}}),
        encoding="utf-8",
    )

    reports = BackendService().load_reports(tmp_path)

    assert reports.reports["summary"]["machine_ready"] is True
    assert reports.paths["summary"] == tmp_path / "summary.json"


def test_gui_launch_path_builds_backend_default_graph() -> None:
    source = inspect.getsource(_PreviewWindow.__init__)

    assert "default_backend_winding_graph()" in source
    assert "BackendService()" in source


def test_node_adjustment_buttons_use_safe_action_wrapper() -> None:
    source = inspect.getsource(_PreviewWindow._build_node_workspace)
    shortcut_source = inspect.getsource(_PreviewWindow._install_node_shortcuts)
    menu_source = inspect.getsource(_PreviewWindow._show_node_context_menu)

    assert "_connect_safe_button" in source
    assert "run_safe_action" in shortcut_source
    assert "run_safe_action" in menu_source


def test_node_adjustment_buttons_do_not_crash_without_selection() -> None:
    source = "\n".join(
        [
            inspect.getsource(_PreviewWindow._delete_selected_graph_items),
            inspect.getsource(_PreviewWindow._duplicate_selected_nodes),
            inspect.getsource(_PreviewWindow._move_selected_nodes),
            inspect.getsource(_PreviewWindow._group_selected_nodes),
        ]
    )

    assert "No node" in source
    assert "Select at least two nodes" in source
    assert "_set_node_status" in source


def test_node_canvas_zoom_pan_and_fit_view_are_implemented() -> None:
    source = "\n".join(
        [
            inspect.getsource(_PreviewWindow._node_canvas_event_filter),
            inspect.getsource(_PreviewWindow._handle_node_wheel),
            inspect.getsource(_PreviewWindow._begin_node_pan),
            inspect.getsource(_PreviewWindow._fit_node_graph),
            inspect.getsource(_PreviewWindow._reset_node_graph_view),
        ]
    )

    assert "Type.Wheel" in source
    assert "_bounded_node_zoom" in source
    assert "ClosedHandCursor" in source
    assert "fitInView" in source
    assert "resetTransform" in source


def test_node_position_and_view_state_preserved_after_zoom_pan_round_trip() -> None:
    graph = default_backend_winding_graph()
    first_node = next(iter(graph.nodes.values()))
    first_node.x = 123.0
    first_node.y = 456.0
    graph.view_zoom = 1.75
    graph.view_center_x = 777.0
    graph.view_center_y = 333.0

    restored = NodeGraphState.from_dict(graph.to_dict(), default_node_registry())

    assert restored.nodes[first_node.id].x == 123.0
    assert restored.nodes[first_node.id].y == 456.0
    assert restored.view_zoom == 1.75
    assert restored.view_center_x == 777.0
    assert restored.view_center_y == 333.0


def test_inspector_edit_marks_downstream_nodes_stale() -> None:
    graph = default_backend_winding_graph()
    tow = next(node for node in graph.nodes.values() if node.type_id == "tow_backend")
    pattern = next(
        node for node in graph.nodes.values() if node.type_id == "pattern_optimisation_backend"
    )

    graph.update_node_settings(tow.id, {"width_mm": 7.0})

    assert graph.nodes[tow.id].status == "stale"
    assert graph.nodes[pattern.id].status == "stale"


def test_project_save_load_preserves_node_view_transform(tmp_path: Path) -> None:
    service = BackendService()
    graph = default_backend_winding_graph()
    graph.view_zoom = 2.0
    graph.view_center_x = 1234.0
    graph.view_center_y = 567.0
    project_path = tmp_path / "view_state.fwgui.json"

    service.save_gui_project(graph, project_path)
    restored = service.load_gui_project(project_path)

    assert restored.view_zoom == 2.0
    assert restored.view_center_x == 1234.0
    assert restored.view_center_y == 567.0
