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
    controls_source = inspect.getsource(_PreviewWindow._build_controls)

    assert "QSplitter(self._qt_core.Qt.Orientation.Horizontal)" in source
    assert "FullViewportUpdate" in source
    assert "NODE_CANVAS_SCENE_RECT" in source
    assert "bottom_tabs.addTab(library_panel, \"Node Library\")" in source
    assert "toolbar_scroll.setWidget(toolbar_widget)" in source
    assert "container_layout.addWidget(toolbar_scroll)" in source
    assert "tabs.setVisible(False)" in controls_source
    assert 'tabs.addTab(nodes_tab, "Nodes")' not in controls_source
    assert "layout.addWidget(nodes_tab, 1)" in controls_source


def test_gui_button_connections_use_safe_action_wrapper() -> None:
    source = inspect.getsource(_PreviewWindow)

    assert "def run_safe_action" in source
    assert "def handle_gui_error" in source
    assert "def _append_gui_log" in source
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
    layer = controller.add_node("geodesic_layer", x=300.0)
    target = controller.add_node("layer_stack_backend", x=600.0)

    assert source.success and source.node_id is not None
    assert layer.success and layer.node_id is not None
    assert target.success and target.node_id is not None
    material_link = controller.link_nodes(source.node_id, "tow", layer.node_id, "material")
    link = controller.link_nodes(layer.node_id, "layer", target.node_id, "layer")
    duplicate = controller.duplicate_node(target.node_id)
    delete = controller.delete_nodes((target.node_id,))

    assert material_link.success
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
    assert [layer.name for layer in config.layers] == [
        "hoop_90deg",
        "helical_45deg",
        "polar_10deg",
    ]
    assert [layer.type for layer in config.layers] == ["hoop", "geodesic", "polar"]
    assert mapping["pattern_selection"]["method"] == "textbook_integer_closure"
    assert mapping["output"]["gcode"] is True


def test_connected_layer_nodes_define_stack_ply_order_and_layer_settings() -> None:
    graph = default_backend_winding_graph()
    layers = [node for node in graph.nodes.values() if node.type_id == "layer_backend"]
    geodesic = next(node for node in layers if node.settings["winding_angle_deg"] == 45.0)
    hoop = next(node for node in layers if node.settings["winding_angle_deg"] == 90.0)
    polar = next(node for node in layers if node.settings["winding_angle_deg"] == 10.0)

    geodesic.settings.update(
        {
            "ply_order": 1,
            "name": "first_helical",
            "angle_tolerance_deg": 0.25,
            "start_z_mm": 10.0,
            "end_z_mm": 900.0,
        }
    )
    hoop.settings["ply_order"] = 2
    polar.settings["ply_order"] = 3

    config = BackendService().build_project_from_graph(graph)

    assert [layer.name for layer in config.layers] == [
        "first_helical",
        "hoop_90deg",
        "polar_10deg",
    ]
    assert [layer.type for layer in config.layers] == ["geodesic", "hoop", "polar"]
    assert config.layers[0].angle_tolerance_deg == 0.25
    assert config.layers[0].start_z_mm == 10.0
    assert config.layers[0].end_z_mm == 900.0


def test_backend_graph_can_use_imported_axisymmetric_mandrel() -> None:
    graph = default_backend_winding_graph()
    mandrel = next(node for node in graph.nodes.values() if node.type_id == "mandrel_backend")
    profile_path = Path("mandrels/2000mm_8in_od_elliptical_dome_profile.dxf")
    mandrel.settings.update(
        {
            "mode": "profile",
            "type": "axisymmetric_profile",
            "profile_path": str(profile_path),
            "samples": 41,
        }
    )

    config = BackendService().build_project_from_graph(graph)
    mapping = graph_to_config_mapping(graph)

    assert config.mandrel.type == "axisymmetric_profile"
    assert config.mandrel.profile_path == profile_path
    assert config.mandrel.samples == 41
    assert mapping["mandrel"]["profile_path"] == str(profile_path)


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
    assert "suppress(RuntimeError)" in source


def test_backend_busy_prunes_deleted_buttons() -> None:
    source = inspect.getsource(_PreviewWindow._set_backend_busy)

    assert "live_buttons" in source
    assert "except RuntimeError" in source
    assert "self._safe_buttons = live_buttons" in source


def test_apply_node_settings_queues_safe_redraw() -> None:
    source = inspect.getsource(_PreviewWindow._apply_node_inspector)

    assert "_queue_node_graph_redraw((node_id,))" in source
    assert "_schedule_node_setting_render()" in source
    assert "_redraw_node_graph()" not in source


def test_node_path_settings_have_browse_controls() -> None:
    source = "\n".join(
        [
            inspect.getsource(_PreviewWindow._node_path_editor),
            inspect.getsource(_PreviewWindow._browse_node_path_setting),
            inspect.getsource(_PreviewWindow._draw_inline_node_settings),
        ]
    )

    assert "QFileDialog.getOpenFileName" in source
    assert "Mandrel/Profile files" in source
    assert "profile_path" in inspect.getsource(_PreviewWindow._inline_node_setting_keys)


def test_viewport_rendering_is_node_graph_controlled() -> None:
    render_source = inspect.getsource(_PreviewWindow._render_scene)
    render_contents_source = inspect.getsource(_PreviewWindow._render_scene_contents)
    node_render_source = inspect.getsource(_PreviewWindow._render_node_graph_scene)
    schedule_source = inspect.getsource(_PreviewWindow._preview_schedule_from_config)
    mandrel_source = inspect.getsource(_PreviewWindow._preview_mandrel_from_config)

    assert "_capture_mandrel_camera_state" in render_source
    assert "_restore_mandrel_camera_state" in render_source
    assert "self._uses_backend_service_graph()" in render_contents_source
    assert "self._render_node_graph_scene()" in render_contents_source
    assert "build_project_from_graph" in node_render_source
    assert "start_z_mm=layer.start_z_mm" in schedule_source
    assert "end_z_mm=layer.end_z_mm" in schedule_source
    assert "plan_winding_schedule(mandrel, schedule)" in node_render_source
    assert "cylinder_with_domes_profile" in mandrel_source
    assert "import_dxf_zr_profile" in mandrel_source


def test_global_gui_exception_safety_net_is_installed() -> None:
    import filament_winder.app.gui as gui

    launch_source = inspect.getsource(gui.launch_cylinder_preview)
    app_source = inspect.getsource(gui._create_qapplication)

    assert "install_gui_exception_hook" in launch_source
    assert "def notify" in app_source
    assert "traceback.format_exc()" in app_source


def test_log_panel_has_copy_clear_open_controls() -> None:
    source = inspect.getsource(_PreviewWindow._build_node_workspace)

    assert "Copy Log" in source
    assert "Clear Log" in source
    assert "Open Log File" in source
    assert "_copy_gui_log" in source
    assert "_clear_gui_log" in source
    assert "_open_gui_log_file" in source


def test_node_related_signal_paths_are_safe_wrapped() -> None:
    source = "\n".join(
        [
            inspect.getsource(_PreviewWindow._build_node_workspace),
            inspect.getsource(_PreviewWindow._setting_editor),
            inspect.getsource(_PreviewWindow._install_node_canvas_event_filter),
        ]
    )

    assert "Apply Node Name" in source
    assert "Open Report" in source
    assert "Open Plot" in source
    assert "Edit Node Setting" in source
    assert "run_safe_action" in source
    assert "handle_gui_error(\"Node Canvas Event\"" in source


def test_context_menu_signal_is_safe_wrapped() -> None:
    source = inspect.getsource(_PreviewWindow._build_node_workspace)

    assert "customContextMenuRequested.connect" in source
    assert "Node Context Menu" in source
    assert "run_safe_action" in source


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


def test_node_selection_is_cached_as_ids_not_qgraphics_items() -> None:
    source = "\n".join(
        [
            inspect.getsource(_PreviewWindow.__init__),
            inspect.getsource(_PreviewWindow._read_scene_selection_ids),
            inspect.getsource(_PreviewWindow._selected_node_ids),
            inspect.getsource(_PreviewWindow._on_node_selection_changed),
            inspect.getsource(_PreviewWindow._apply_node_selection_changed),
        ]
    )

    assert "_selected_node_id_cache: tuple[str, ...]" in source
    assert "_selected_link_id_cache: tuple[str, ...]" in source
    assert "selectedItems()" in source
    assert "_selected_node_id_cache" in inspect.getsource(_PreviewWindow._selected_node_ids)
    assert "_schedule_node_selection_ui_update" in source
    assert "_render_scene()" not in inspect.getsource(
        _PreviewWindow._update_viewport_context_for_selection
    )


def test_node_delete_clears_selection_blocks_signals_and_rebuilds_scene() -> None:
    redraw_source = inspect.getsource(_PreviewWindow._redraw_node_graph)
    replace_source = inspect.getsource(_PreviewWindow._replace_node_scene_for_rebuild)
    delete_source = "\n".join(
        [
            inspect.getsource(_PreviewWindow._delete_selected_nodes),
            inspect.getsource(_PreviewWindow._delete_selected_graph_items),
            inspect.getsource(_PreviewWindow._move_selected_nodes),
        ]
    )

    assert "blockSignals(True)" in redraw_source
    assert "self.node_scene.clear()" not in redraw_source
    assert "_replace_node_scene_for_rebuild()" in redraw_source
    assert "_retired_node_scenes.append(old_scene)" in replace_source
    assert "self.node_view.setScene(self.node_scene)" in replace_source
    assert "_restore_scene_selection" in redraw_source
    assert "_clear_node_scene_selection()" in delete_source
    assert "_graph_controller().delete_nodes" in delete_source


def test_node_scene_and_drag_link_callbacks_are_exception_guarded() -> None:
    scene_source = inspect.getsource(_PreviewWindow._on_node_scene_changed)
    link_source = inspect.getsource(_PreviewWindow._finish_socket_drag)
    refresh_source = inspect.getsource(_PreviewWindow._refresh_node_links)

    assert "handle_gui_error(\"Node Scene Changed\"" in scene_source
    assert "self._graph_controller().link_nodes" in link_source
    assert "suppress(RuntimeError)" in link_source
    assert "finally:" in refresh_source
    assert "self._refreshing_node_links = False" in refresh_source


def test_node_cards_have_inline_blender_style_setting_editors() -> None:
    draw_source = inspect.getsource(_PreviewWindow._draw_graph_node)
    inline_source = "\n".join(
        [
            inspect.getsource(_PreviewWindow._draw_inline_node_settings),
            inspect.getsource(_PreviewWindow._inline_node_setting_editor),
            inspect.getsource(_PreviewWindow._setting_editor),
            inspect.getsource(_PreviewWindow._update_node_setting),
        ]
    )

    assert "_draw_inline_node_settings" in draw_source
    assert "QGraphicsProxyWidget" not in inline_source
    assert "self.node_scene.addWidget(panel)" in inline_source
    assert "QDoubleSpinBox" in inline_source
    assert "QComboBox" in inline_source
    assert "QLineEdit" in inline_source
    assert "textEdited.connect" in inline_source
    assert "_schedule_node_setting_render()" in inline_source


def test_node_scene_rebuild_swaps_scenes_without_clearing_active_scene() -> None:
    redraw_source = inspect.getsource(_PreviewWindow._redraw_node_graph)
    replace_source = inspect.getsource(_PreviewWindow._replace_node_scene_for_rebuild)
    cleanup_source = inspect.getsource(_PreviewWindow._cleanup_node_scenes_for_close)

    assert "self.node_scene.clear()" not in redraw_source
    assert "self.node_view.setScene(self.node_scene)" in replace_source
    assert "_dispose_retired_node_scene" in replace_source
    assert "scene.clear()" not in cleanup_source


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


def test_node_linking_layer_stack_table_and_tow_band_rendering_are_supported() -> None:
    link_source = "\n".join(
        [
            inspect.getsource(_PreviewWindow._begin_socket_drag),
            inspect.getsource(_PreviewWindow._finish_socket_drag),
            inspect.getsource(_PreviewWindow._socket_item_at_view_pos),
        ]
    )
    stack_source = "\n".join(
        [
            inspect.getsource(_PreviewWindow._draw_layer_stack_table),
            inspect.getsource(_PreviewWindow._copy_layer_stack_rows),
        ]
    )
    render_source = "\n".join(
        [
            inspect.getsource(_PreviewWindow._display_tow_band_mesh),
            inspect.getsource(_PreviewWindow._render_node_graph_scene),
        ]
    )

    assert "_link_args_from_socket_drag" in link_source
    assert "self.node_view.items(search_rect)" in link_source
    assert "layerStackNodeTable" in stack_source
    assert "InternalMove" in stack_source
    assert "_copy_layer_stack_rows" in stack_source
    assert "_display_tow_band_mesh" in render_source
    assert "scene.visuals.Mesh" in render_source


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
