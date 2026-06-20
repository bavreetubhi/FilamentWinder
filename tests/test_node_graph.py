from __future__ import annotations

import inspect

import pytest

from filament_winder.app.gui import _PreviewWindow
from filament_winder.app.node_graph import (
    NodeGraphController,
    NodeGraphExecutor,
    NodeGraphState,
    addable_node_type_ids,
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
        def add_node(self, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
            node = super().add_node(*args, **kwargs)
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


def test_node_graph_accepts_valid_socket_link_and_marks_downstream_dirty() -> None:
    registry = default_node_registry()
    graph = NodeGraphState()
    mandrel = graph.add_node("mandrel_profile", registry)
    layer_stack = graph.add_node("layer_stack", registry)
    pattern = graph.add_node("winding_pattern", registry)
    graph.add_link(mandrel.id, "mandrel", pattern.id, "mandrel", registry)
    graph.add_link(layer_stack.id, "layer_stack", pattern.id, "layer_stack", registry)

    graph.nodes[pattern.id].status = "complete"
    graph.update_node_settings(mandrel.id, {"radius_mm": 22.0})

    assert graph.nodes[mandrel.id].status == "dirty"
    assert graph.nodes[pattern.id].status == "dirty"


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
