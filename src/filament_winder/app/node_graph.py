"""Node graph data model and execution for the engineering GUI."""

from __future__ import annotations

import copy
import math
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from filament_winder.core.coverage import cylinder_coverage_map
from filament_winder.core.geometry import AxisymmetricProfileMandrel, CylinderMandrel
from filament_winder.core.path_planning import (
    PlannedWindingProgram,
    WindingLayerSpec,
    WindingSchedule,
    axisymmetric_surface_coverage_map,
    plan_winding_schedule,
)
from filament_winder.io import GCodeOptions, export_gcode, export_winding_program_csv
from filament_winder.io.dxf_import import import_dxf_zr_profile

SocketKind = Literal[
    "project_config",
    "machine_config",
    "mandrel",
    "tow",
    "machine",
    "layer",
    "layer_stack",
    "pattern_candidates",
    "selected_pattern",
    "winding_program",
    "program",
    "coverage",
    "coverage_mode",
    "simulation",
    "validation_report",
    "plots",
    "exports",
    "export",
    "any",
]
NodeStatus = Literal[
    "not_run",
    "not_configured",
    "ready",
    "running",
    "warning",
    "failed",
    "passed",
    "stale",
    "error",
    "processing",
    "complete",
    "dirty",
]


@dataclass(frozen=True, slots=True)
class NodeSocketDefinition:
    name: str
    kind: SocketKind
    required: bool = True


@dataclass(frozen=True, slots=True)
class NodeTypeDefinition:
    type_id: str
    label: str
    category: str
    color: str
    inputs: tuple[NodeSocketDefinition, ...] = ()
    outputs: tuple[NodeSocketDefinition, ...] = ()
    default_settings: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class NodeInstance:
    id: str
    type_id: str
    name: str
    x: float = 0.0
    y: float = 0.0
    width: float = 280.0
    height: float = 150.0
    collapsed: bool = False
    group_id: str | None = None
    settings: dict[str, Any] = field(default_factory=dict)
    status: NodeStatus = "not_configured"
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type_id": self.type_id,
            "name": self.name,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "collapsed": self.collapsed,
            "group_id": self.group_id,
            "settings": copy.deepcopy(self.settings),
            "status": self.status,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NodeInstance:
        return cls(
            id=str(data["id"]),
            type_id=str(data["type_id"]),
            name=str(data.get("name", data["type_id"])),
            x=float(data.get("x", 0.0)),
            y=float(data.get("y", 0.0)),
            width=float(data.get("width", 280.0)),
            height=float(data.get("height", 150.0)),
            collapsed=bool(data.get("collapsed", False)),
            group_id=(None if data.get("group_id") in {None, ""} else str(data["group_id"])),
            settings=copy.deepcopy(data.get("settings", {})),
            status=_node_status(data.get("status", "not_configured")),
            message=str(data.get("message", "")),
        )


@dataclass(frozen=True, slots=True)
class NodeLink:
    id: str
    source_node_id: str
    source_socket: str
    target_node_id: str
    target_socket: str

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "source_node_id": self.source_node_id,
            "source_socket": self.source_socket,
            "target_node_id": self.target_node_id,
            "target_socket": self.target_socket,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NodeLink:
        return cls(
            id=str(data["id"]),
            source_node_id=str(data["source_node_id"]),
            source_socket=str(data["source_socket"]),
            target_node_id=str(data["target_node_id"]),
            target_socket=str(data["target_socket"]),
        )


@dataclass(slots=True)
class NodeGroup:
    id: str
    name: str
    node_ids: tuple[str, ...]
    color: str = "#33404d"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "node_ids": list(self.node_ids),
            "color": self.color,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NodeGroup:
        return cls(
            id=str(data["id"]),
            name=str(data.get("name", "Group")),
            node_ids=tuple(str(node_id) for node_id in data.get("node_ids", ())),
            color=str(data.get("color", "#33404d")),
        )


@dataclass(slots=True)
class NodeGraphState:
    nodes: dict[str, NodeInstance] = field(default_factory=dict)
    links: dict[str, NodeLink] = field(default_factory=dict)
    groups: dict[str, NodeGroup] = field(default_factory=dict)
    selected_node_ids: tuple[str, ...] = ()
    schema_version: int = 1
    view_zoom: float = 1.0
    view_center_x: float = 1100.0
    view_center_y: float = 450.0

    def add_node(
        self,
        type_id: str,
        registry: dict[str, NodeTypeDefinition],
        *,
        name: str | None = None,
        x: float = 0.0,
        y: float = 0.0,
    ) -> NodeInstance:
        definition = _definition(registry, type_id)
        node = NodeInstance(
            id=_new_id(type_id),
            type_id=type_id,
            name=name or definition.label,
            x=x,
            y=y,
            width=280.0,
            height=150.0,
            settings=copy.deepcopy(definition.default_settings),
        )
        self.nodes[node.id] = node
        return node

    def duplicate_node(
        self,
        node_id: str,
        registry: dict[str, NodeTypeDefinition],
        *,
        offset: tuple[float, float] = (32.0, 32.0),
    ) -> NodeInstance:
        source = self.nodes[node_id]
        _definition(registry, source.type_id)
        node = NodeInstance(
            id=_new_id(source.type_id),
            type_id=source.type_id,
            name=f"{source.name} copy",
            x=source.x + offset[0],
            y=source.y + offset[1],
            width=source.width,
            height=source.height,
            collapsed=source.collapsed,
            group_id=source.group_id,
            settings=copy.deepcopy(source.settings),
            status="not_configured",
        )
        self.nodes[node.id] = node
        return node

    def delete_nodes(self, node_ids: tuple[str, ...]) -> None:
        delete_set = set(node_ids)
        for node_id in delete_set:
            self.nodes.pop(node_id, None)
        for link_id, link in list(self.links.items()):
            if link.source_node_id in delete_set or link.target_node_id in delete_set:
                self.links.pop(link_id, None)
        for group_id, group in list(self.groups.items()):
            remaining = tuple(node_id for node_id in group.node_ids if node_id not in delete_set)
            if remaining:
                group.node_ids = remaining
            else:
                self.groups.pop(group_id, None)

    def rename_node(self, node_id: str, name: str) -> None:
        self.nodes[node_id].name = name.strip() or self.nodes[node_id].name

    def set_node_position(self, node_id: str, x: float, y: float) -> None:
        node = self.nodes[node_id]
        node.x = float(x)
        node.y = float(y)

    def set_node_collapsed(self, node_id: str, collapsed: bool) -> None:
        node = self.nodes[node_id]
        node.collapsed = collapsed
        node.height = 54.0 if collapsed else 150.0

    def update_node_settings(self, node_id: str, settings: dict[str, Any]) -> None:
        self.nodes[node_id].settings.update(copy.deepcopy(settings))
        self.mark_downstream_dirty(node_id)

    def add_link(
        self,
        source_node_id: str,
        source_socket: str,
        target_node_id: str,
        target_socket: str,
        registry: dict[str, NodeTypeDefinition],
    ) -> NodeLink:
        self.validate_link(
            source_node_id,
            source_socket,
            target_node_id,
            target_socket,
            registry,
        )
        for link_id, link in list(self.links.items()):
            if (
                target_socket != "layer"
                and link.target_node_id == target_node_id
                and link.target_socket == target_socket
            ):
                self.links.pop(link_id)
        link = NodeLink(
            id=_new_id("link"),
            source_node_id=source_node_id,
            source_socket=source_socket,
            target_node_id=target_node_id,
            target_socket=target_socket,
        )
        self.links[link.id] = link
        self.mark_downstream_dirty(source_node_id)
        return link

    def remove_links_for_nodes(self, node_ids: tuple[str, ...]) -> None:
        node_set = set(node_ids)
        for link_id, link in list(self.links.items()):
            if link.source_node_id in node_set or link.target_node_id in node_set:
                self.links.pop(link_id, None)

    def group_nodes(self, node_ids: tuple[str, ...], name: str = "Group") -> NodeGroup:
        group = NodeGroup(id=_new_id("group"), name=name, node_ids=tuple(node_ids))
        self.groups[group.id] = group
        for node_id in node_ids:
            if node_id in self.nodes:
                self.nodes[node_id].group_id = group.id
        return group

    def validate_link(
        self,
        source_node_id: str,
        source_socket: str,
        target_node_id: str,
        target_socket: str,
        registry: dict[str, NodeTypeDefinition],
    ) -> None:
        if source_node_id == target_node_id:
            raise ValueError("cannot link a node to itself")
        source = self.nodes[source_node_id]
        target = self.nodes[target_node_id]
        source_def = _definition(registry, source.type_id)
        target_def = _definition(registry, target.type_id)
        output_socket = _socket(source_def.outputs, source_socket)
        input_socket = _socket(target_def.inputs, target_socket)
        if output_socket is None:
            raise ValueError(f"node '{source.name}' has no output socket '{source_socket}'")
        if input_socket is None:
            raise ValueError(f"node '{target.name}' has no input socket '{target_socket}'")
        if not _compatible_socket_kinds(output_socket.kind, input_socket.kind):
            raise ValueError(
                f"cannot connect {output_socket.kind} output to {input_socket.kind} input"
            )
        if self._would_create_cycle(source_node_id, target_node_id):
            raise ValueError("link would create a cycle")

    def incoming_links(self, node_id: str) -> tuple[NodeLink, ...]:
        return tuple(link for link in self.links.values() if link.target_node_id == node_id)

    def outgoing_links(self, node_id: str) -> tuple[NodeLink, ...]:
        return tuple(link for link in self.links.values() if link.source_node_id == node_id)

    def topological_node_ids(self) -> tuple[str, ...]:
        dependencies = {node_id: set[str]() for node_id in self.nodes}
        for link in self.links.values():
            dependencies[link.target_node_id].add(link.source_node_id)
        ready = [node_id for node_id, deps in dependencies.items() if not deps]
        ordered: list[str] = []
        while ready:
            node_id = ready.pop(0)
            ordered.append(node_id)
            for link in self.outgoing_links(node_id):
                deps = dependencies[link.target_node_id]
                deps.discard(node_id)
                if (
                    not deps
                    and link.target_node_id not in ordered
                    and link.target_node_id not in ready
                ):
                    ready.append(link.target_node_id)
        if len(ordered) != len(self.nodes):
            raise ValueError("graph contains a cycle")
        return tuple(ordered)

    def downstream_node_ids(self, node_id: str) -> tuple[str, ...]:
        downstream: list[str] = []
        pending = [node_id]
        seen = {node_id}
        while pending:
            current = pending.pop(0)
            for link in self.outgoing_links(current):
                if link.target_node_id in seen:
                    continue
                seen.add(link.target_node_id)
                downstream.append(link.target_node_id)
                pending.append(link.target_node_id)
        return tuple(downstream)

    def mark_downstream_dirty(self, node_id: str) -> None:
        stale_ids = (node_id, *self.downstream_node_ids(node_id))
        for stale_id in stale_ids:
            if stale_id in self.nodes:
                self.nodes[stale_id].status = "stale"
                self.nodes[stale_id].message = "Needs recompute"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "links": [link.to_dict() for link in self.links.values()],
            "groups": [group.to_dict() for group in self.groups.values()],
            "selected_node_ids": list(self.selected_node_ids),
            "view": {
                "zoom": self.view_zoom,
                "center_x": self.view_center_x,
                "center_y": self.view_center_y,
            },
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any] | None,
        registry: dict[str, NodeTypeDefinition],
    ) -> NodeGraphState:
        if not data:
            return NodeGraphState()
        view_data = data.get("view", {})
        view = view_data if isinstance(view_data, dict) else {}
        graph = cls(
            schema_version=int(data.get("schema_version", 1)),
            view_zoom=float(view.get("zoom", data.get("view_zoom", 1.0))),
            view_center_x=float(view.get("center_x", data.get("view_center_x", 1100.0))),
            view_center_y=float(view.get("center_y", data.get("view_center_y", 450.0))),
        )
        for node_data in data.get("nodes", ()):
            node = NodeInstance.from_dict(node_data)
            if node.type_id not in registry:
                continue
            default_settings = copy.deepcopy(registry[node.type_id].default_settings)
            default_settings.update(node.settings)
            node.settings = default_settings
            graph.nodes[node.id] = node
        for link_data in data.get("links", ()):
            link = NodeLink.from_dict(link_data)
            if link.source_node_id in graph.nodes and link.target_node_id in graph.nodes:
                try:
                    graph.validate_link(
                        link.source_node_id,
                        link.source_socket,
                        link.target_node_id,
                        link.target_socket,
                        registry,
                    )
                except ValueError:
                    continue
                graph.links[link.id] = link
        for group_data in data.get("groups", ()):
            group = NodeGroup.from_dict(group_data)
            if all(node_id in graph.nodes for node_id in group.node_ids):
                graph.groups[group.id] = group
        graph.selected_node_ids = tuple(
            node_id for node_id in data.get("selected_node_ids", ()) if node_id in graph.nodes
        )
        return graph

    def _would_create_cycle(self, source_node_id: str, target_node_id: str) -> bool:
        pending = [target_node_id]
        seen: set[str] = set()
        while pending:
            current = pending.pop(0)
            if current == source_node_id:
                return True
            if current in seen:
                continue
            seen.add(current)
            pending.extend(link.target_node_id for link in self.outgoing_links(current))
        return False


@dataclass(frozen=True, slots=True)
class GraphExecutionResult:
    node_outputs: dict[str, dict[str, Any]]
    executed_node_ids: tuple[str, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GraphMutationResult:
    success: bool
    node_id: str | None = None
    node_ids: tuple[str, ...] = ()
    link_id: str | None = None
    error: str | None = None
    message: str = ""


class NodeGraphController:
    """Safe graph mutation facade used by the GUI."""

    def __init__(
        self,
        graph: NodeGraphState,
        registry: dict[str, NodeTypeDefinition],
    ) -> None:
        self._graph = graph
        self._registry = registry

    def add_node(
        self,
        type_id: str,
        *,
        x: float = 0.0,
        y: float = 0.0,
    ) -> GraphMutationResult:
        before_node_ids = set(self._graph.nodes)
        try:
            definition = _definition(self._registry, type_id)
            validate_node_type_definition(definition)
            node = self._graph.add_node(type_id, self._registry, x=x, y=y)
            validate_node_instance(node, definition)
        except Exception as exc:  # noqa: BLE001 - converted to controlled UI error
            for node_id in set(self._graph.nodes) - before_node_ids:
                self._graph.nodes.pop(node_id, None)
            return GraphMutationResult(success=False, error=str(exc))
        return GraphMutationResult(success=True, node_id=node.id)

    def duplicate_node(self, node_id: str) -> GraphMutationResult:
        before_node_ids = set(self._graph.nodes)
        try:
            node = self._graph.duplicate_node(node_id, self._registry)
            validate_node_instance(node, _definition(self._registry, node.type_id))
        except Exception as exc:  # noqa: BLE001 - converted to controlled UI error
            for added_node_id in set(self._graph.nodes) - before_node_ids:
                self._graph.nodes.pop(added_node_id, None)
            return GraphMutationResult(success=False, error=str(exc))
        return GraphMutationResult(success=True, node_id=node.id, node_ids=(node.id,))

    def duplicate_branch(self, root_node_id: str) -> GraphMutationResult:
        before_node_ids = set(self._graph.nodes)
        try:
            branch_node_ids = (root_node_id, *self._graph.downstream_node_ids(root_node_id))
            duplicates: dict[str, NodeInstance] = {}
            for index, source_id in enumerate(branch_node_ids):
                duplicates[source_id] = self._graph.duplicate_node(
                    source_id,
                    self._registry,
                    offset=(48.0, 190.0 + index * 8.0),
                )
            for link in tuple(self._graph.links.values()):
                if link.source_node_id in duplicates and link.target_node_id in duplicates:
                    self._graph.add_link(
                        duplicates[link.source_node_id].id,
                        link.source_socket,
                        duplicates[link.target_node_id].id,
                        link.target_socket,
                        self._registry,
                    )
        except Exception as exc:  # noqa: BLE001 - converted to controlled UI error
            for added_node_id in set(self._graph.nodes) - before_node_ids:
                self._graph.delete_nodes((added_node_id,))
            return GraphMutationResult(success=False, error=str(exc))
        duplicate_ids = tuple(node.id for node in duplicates.values())
        return GraphMutationResult(success=True, node_id=duplicate_ids[0], node_ids=duplicate_ids)

    def delete_nodes(self, node_ids: tuple[str, ...]) -> GraphMutationResult:
        try:
            self._graph.delete_nodes(node_ids)
        except Exception as exc:  # noqa: BLE001 - converted to controlled UI error
            return GraphMutationResult(success=False, error=str(exc))
        return GraphMutationResult(success=True, node_ids=tuple(node_ids))

    def link_nodes(
        self,
        source_node_id: str,
        source_socket: str,
        target_node_id: str,
        target_socket: str,
    ) -> GraphMutationResult:
        try:
            link = self._graph.add_link(
                source_node_id,
                source_socket,
                target_node_id,
                target_socket,
                self._registry,
            )
        except Exception as exc:  # noqa: BLE001 - converted to controlled UI error
            return GraphMutationResult(success=False, error=str(exc))
        return GraphMutationResult(success=True, link_id=link.id)

    def unlink_nodes(self, node_ids: tuple[str, ...]) -> GraphMutationResult:
        try:
            self._graph.remove_links_for_nodes(node_ids)
        except Exception as exc:  # noqa: BLE001 - converted to controlled UI error
            return GraphMutationResult(success=False, error=str(exc))
        return GraphMutationResult(success=True, node_ids=tuple(node_ids))


class NodeGraphExecutor:
    """Executes graph nodes through the existing filament winding backend."""

    def __init__(
        self,
        registry: dict[str, NodeTypeDefinition],
        *,
        execute_exports: bool = False,
    ) -> None:
        self._registry = registry
        self._execute_exports = execute_exports

    def execute(self, graph: NodeGraphState) -> GraphExecutionResult:
        outputs: dict[str, dict[str, Any]] = {}
        warnings: list[str] = []
        executed: list[str] = []
        for node_id in graph.topological_node_ids():
            node = graph.nodes[node_id]
            node.status = "processing"
            node.message = "Processing"
            try:
                inputs = self._inputs_for_node(graph, node, outputs)
                outputs[node_id] = self._execute_node(node, inputs)
            except Exception as exc:  # noqa: BLE001 - surfaced as node status
                node.status = "error"
                node.message = str(exc)
                warnings.append(f"{node.name}: {exc}")
                outputs[node_id] = {}
                continue
            node.status = "complete"
            node.message = "Complete"
            executed.append(node_id)
        return GraphExecutionResult(
            node_outputs=outputs,
            executed_node_ids=tuple(executed),
            warnings=tuple(warnings),
        )

    def _inputs_for_node(
        self,
        graph: NodeGraphState,
        node: NodeInstance,
        outputs: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        definition = _definition(self._registry, node.type_id)
        values: dict[str, Any] = {}
        for socket_def in definition.inputs:
            links = [
                link
                for link in graph.incoming_links(node.id)
                if link.target_socket == socket_def.name
            ]
            if not links:
                if socket_def.required:
                    raise ValueError(f"missing required input '{socket_def.name}'")
                continue
            link = links[-1]
            source_outputs = outputs.get(link.source_node_id, {})
            if link.source_socket not in source_outputs:
                raise ValueError(f"upstream output '{link.source_socket}' is unavailable")
            values[socket_def.name] = source_outputs[link.source_socket]
        return values

    def _execute_node(self, node: NodeInstance, inputs: dict[str, Any]) -> dict[str, Any]:
        if node.type_id == "mandrel_profile":
            return _execute_mandrel_profile(node)
        if node.type_id == "material_tow":
            return {"tow": copy.deepcopy(node.settings)}
        if node.type_id == "machine_config":
            return {"machine": copy.deepcopy(node.settings)}
        if node.type_id == "layer_stack":
            return _execute_layer_stack(node, inputs)
        if node.type_id == "winding_pattern":
            return _execute_winding_pattern(node, inputs)
        if node.type_id == "path_optimisation":
            return {"program": inputs["program"]}
        if node.type_id == "coverage_analysis":
            return _execute_coverage_analysis(inputs)
        if node.type_id == "simulation":
            return _execute_simulation(inputs)
        if node.type_id == "csv_export":
            return self._execute_csv_export(node, inputs)
        if node.type_id == "gcode_export":
            return self._execute_gcode_export(node, inputs)
        if node.type_id == "controller_run":
            return {"export": {"ready": False, "message": "Controller streaming not implemented"}}
        raise ValueError(f"unsupported node type: {node.type_id}")

    def _execute_csv_export(self, node: NodeInstance, inputs: dict[str, Any]) -> dict[str, Any]:
        path = str(node.settings.get("csv_path", "exports/node_graph_program.csv"))
        if not self._execute_exports:
            return {"export": {"ready": True, "path": path, "written": False}}
        output_path = export_winding_program_csv(inputs["program"], path)
        return {"export": {"ready": True, "path": str(output_path), "written": True}}

    def _execute_gcode_export(self, node: NodeInstance, inputs: dict[str, Any]) -> dict[str, Any]:
        path = str(node.settings.get("gcode_path", "exports/node_graph_program.gcode"))
        feedrate = float(node.settings.get("feedrate_mm_min", 500.0))
        program: PlannedWindingProgram = inputs["program"]
        if not self._execute_exports:
            return {"export": {"ready": True, "path": path, "written": False}}
        output_path = export_gcode(
            program.motion_table,
            path,
            options=GCodeOptions(
                feedrate_mm_min=feedrate,
                feed_schedule=program.feed_schedule,
            ),
        )
        return {"export": {"ready": True, "path": str(output_path), "written": True}}


def default_node_registry() -> dict[str, NodeTypeDefinition]:
    registry = {
        "mandrel_profile": NodeTypeDefinition(
            type_id="mandrel_profile",
            label="Mandrel Profile",
            category="Geometry",
            color="#2f6f9f",
            outputs=(NodeSocketDefinition("mandrel", "mandrel"),),
            default_settings={
                "mode": "cylinder",
                "length_mm": 1000.0,
                "radius_mm": 100.0,
                "profile_path": "mandrels/profile.dxf",
                "samples": 0,
            },
        ),
        "material_tow": NodeTypeDefinition(
            type_id="material_tow",
            label="Material / Tow",
            category="Material",
            color="#7a5aa6",
            outputs=(NodeSocketDefinition("tow", "tow"),),
            default_settings={
                "tow_width_mm": 6.0,
                "layer_thickness_mm": 0.0,
                "resin_fraction": 0.35,
            },
        ),
        "machine_config": NodeTypeDefinition(
            type_id="machine_config",
            label="Machine Config",
            category="Machine",
            color="#98733f",
            outputs=(NodeSocketDefinition("machine", "machine"),),
            default_settings={
                "radial_clearance_mm": 25.0,
                "feedrate_mm_min": 500.0,
            },
        ),
        "layer_stack": NodeTypeDefinition(
            type_id="layer_stack",
            label="Layer Stack",
            category="Winding",
            color="#487d53",
            inputs=(
                NodeSocketDefinition("tow", "tow", required=False),
            ),
            outputs=(NodeSocketDefinition("layer_stack", "layer_stack"),),
            default_settings={
                "layers": [
                    {
                        "layer_id": "helical-1",
                        "enabled": True,
                        "name": "helical",
                        "winding_type": "helical",
                        "target_angle_deg": 45.0,
                        "tow_width_mm": 6.0,
                        "layer_thickness_mm": 0.0,
                        "coverage_target": 1.0,
                        "direction": "positive",
                        "number_of_passes": None,
                        "feedrate_mm_min": None,
                        "mandrel_clearance_mm": None,
                        "colour": "#1e90ff",
                        "notes": "",
                        "point_count": 300,
                        "transition_points": 20,
                    }
                ]
            },
        ),
        "winding_pattern": NodeTypeDefinition(
            type_id="winding_pattern",
            label="Winding Pattern",
            category="Winding",
            color="#3f7d7d",
            inputs=(
                NodeSocketDefinition("mandrel", "mandrel"),
                NodeSocketDefinition("layer_stack", "layer_stack"),
                NodeSocketDefinition("machine", "machine", required=False),
            ),
            outputs=(NodeSocketDefinition("program", "program"),),
            default_settings={},
        ),
        "path_optimisation": NodeTypeDefinition(
            type_id="path_optimisation",
            label="Path Optimisation",
            category="Analysis",
            color="#517aa3",
            inputs=(NodeSocketDefinition("program", "program"),),
            outputs=(NodeSocketDefinition("program", "program"),),
            default_settings={"enabled": False},
        ),
        "coverage_analysis": NodeTypeDefinition(
            type_id="coverage_analysis",
            label="Coverage Analysis",
            category="Analysis",
            color="#657f3f",
            inputs=(
                NodeSocketDefinition("mandrel", "mandrel"),
                NodeSocketDefinition("program", "program"),
            ),
            outputs=(NodeSocketDefinition("coverage", "coverage"),),
            default_settings={"z_samples": 120, "theta_samples": 180},
        ),
        "simulation": NodeTypeDefinition(
            type_id="simulation",
            label="Simulation",
            category="Analysis",
            color="#6b6aa8",
            inputs=(NodeSocketDefinition("program", "program"),),
            outputs=(NodeSocketDefinition("simulation", "simulation"),),
            default_settings={},
        ),
        "csv_export": NodeTypeDefinition(
            type_id="csv_export",
            label="CSV Export",
            category="Export",
            color="#8a6741",
            inputs=(NodeSocketDefinition("program", "program"),),
            outputs=(NodeSocketDefinition("export", "export"),),
            default_settings={"csv_path": "exports/node_graph_program.csv"},
        ),
        "gcode_export": NodeTypeDefinition(
            type_id="gcode_export",
            label="G-code Export",
            category="Export",
            color="#8a523f",
            inputs=(NodeSocketDefinition("program", "program"),),
            outputs=(NodeSocketDefinition("export", "export"),),
            default_settings={
                "gcode_path": "exports/node_graph_program.gcode",
                "feedrate_mm_min": 500.0,
            },
        ),
        "controller_run": NodeTypeDefinition(
            type_id="controller_run",
            label="Controller / Machine Run",
            category="Machine",
            color="#8a3f3f",
            inputs=(NodeSocketDefinition("program", "program"),),
            outputs=(NodeSocketDefinition("export", "export"),),
            default_settings={"enabled": False, "port": ""},
        ),
    }
    registry.update(_backend_node_registry())
    return registry


def _backend_node_registry() -> dict[str, NodeTypeDefinition]:
    return {
        "project": NodeTypeDefinition(
            type_id="project",
            label="Project",
            category="Project",
            color="#4b6f9f",
            outputs=(NodeSocketDefinition("project_config", "project_config"),),
            default_settings={
                "name": "demo_domed_pressure_vessel",
                "units": "mm",
                "output_directory": "exports/demo_domed_pressure_vessel",
            },
        ),
        "machine_backend": NodeTypeDefinition(
            type_id="machine_backend",
            label="Machine Config",
            category="Machine",
            color="#98733f",
            inputs=(NodeSocketDefinition("project_config", "project_config"),),
            outputs=(NodeSocketDefinition("machine_config", "machine_config"),),
            default_settings={
                "controller": "grbl_compatible",
                "axis_order": ["A", "X", "Z", "B"],
                "clearance_mm": 20.0,
                "max_a_rpm": 120.0,
                "max_x_mm": 300.0,
                "max_z_mm": 1500.0,
                "max_b_deg": 7200.0,
                "max_b_velocity_deg_s": 240.0,
                "max_segment_length_mm": 20.0,
                "max_a_accel_deg_s2": 2500.0,
                "max_x_accel_mm_s2": 2500.0,
                "max_z_accel_mm_s2": 1000.0,
                "max_b_accel_deg_s2": 10000.0,
            },
        ),
        "mandrel_backend": NodeTypeDefinition(
            type_id="mandrel_backend",
            label="Mandrel",
            category="Mandrel",
            color="#2f6f9f",
            inputs=(NodeSocketDefinition("machine_config", "machine_config"),),
            outputs=(NodeSocketDefinition("mandrel", "mandrel"),),
            default_settings={
                "mode": "dome",
                "type": "cylinder_with_elliptical_domes",
                "cylinder_length_mm": 1000.0,
                "cylinder_radius_mm": 101.6,
                "left_dome_length_mm": 120.0,
                "right_dome_length_mm": 120.0,
                "polar_opening_radius_mm": 25.0,
                "profile_path": "mandrels/profile.dxf",
                "samples": 0,
                "mesh_points_z": 360,
                "mesh_points_theta": 360,
            },
        ),
        "tow_backend": NodeTypeDefinition(
            type_id="tow_backend",
            label="Material",
            category="Material",
            color="#7a5aa6",
            inputs=(NodeSocketDefinition("mandrel", "mandrel"),),
            outputs=(NodeSocketDefinition("tow", "tow"),),
            default_settings={
                "name": "carbon_tow",
                "width_mm": 6.0,
                "thickness_mm": 0.25,
                "effective_width_mm": 6.0,
                "calibrated_effective_width": False,
                "friction_coefficient": "",
                "calibrated_friction": False,
                "tension_N": "",
                "min_bend_radius_mm": "",
                "fibre_type": "carbon",
                "resin_system": "",
                "notes": "",
            },
        ),
        "layer_stack_backend": NodeTypeDefinition(
            type_id="layer_stack_backend",
            label="Layer Stack",
            category="Layers",
            color="#487d53",
            inputs=(
                NodeSocketDefinition("layer", "layer", required=False),
            ),
            outputs=(NodeSocketDefinition("layer_stack", "layer_stack"),),
            default_settings={
                "name": "primary_stack",
                "ordering": "ply_order",
                "repeat_stack": 1,
                "mirror_stack": False,
                "notes": "",
            },
        ),
        "layer_backend": NodeTypeDefinition(
            type_id="layer_backend",
            label="Layer",
            category="Layers",
            color="#4d8265",
            inputs=(NodeSocketDefinition("material", "tow", required=False),),
            outputs=(NodeSocketDefinition("layer", "layer"),),
            default_settings={
                "name": "layer_45deg",
                "enabled": True,
                "material": "carbon_tow",
                "winding_angle_deg": 45.0,
                "angle_tolerance_deg": 0.5,
                "start_z_mm": "",
                "end_z_mm": "",
                "direction": "forward",
                "coverage_target": 1.0,
                "turnaround_radius_mm": 28.0,
                "polar_opening_radius_mm": 25.0,
                "feedrate_mm_min": 450.0,
                "transition_before": True,
                "transition_after": True,
                "points": 220,
                "colour": "#1e90ff",
                "notes": "",
            },
        ),
        "hoop_layer": NodeTypeDefinition(
            type_id="hoop_layer",
            label="Hoop Layer",
            category="Layers",
            color="#6f8f3f",
            inputs=(NodeSocketDefinition("material", "tow", required=False),),
            outputs=(NodeSocketDefinition("layer", "layer"),),
            default_settings={
                "name": "hoop_cylinder",
                "enabled": True,
                "ply_order": 10,
                "material": "carbon_tow",
                "region": "cylinder_only",
                "type": "hoop",
                "winding_angle_deg": 90.0,
                "angle_tolerance_deg": 0.5,
                "start_z_mm": "",
                "end_z_mm": "",
                "passes": 1,
                "coverage_target": 1.0,
                "feedrate_mm_min": 500.0,
                "transition_before": False,
                "transition_after": True,
                "points": 80,
                "colour": "#1e90ff",
                "notes": "",
            },
        ),
        "geodesic_layer": NodeTypeDefinition(
            type_id="geodesic_layer",
            label="Geodesic Layer",
            category="Layers",
            color="#3f7d7d",
            inputs=(NodeSocketDefinition("material", "tow", required=False),),
            outputs=(NodeSocketDefinition("layer", "layer"),),
            default_settings={
                "name": "geodesic_dome_to_dome",
                "enabled": True,
                "ply_order": 20,
                "material": "carbon_tow",
                "region": "dome_to_dome",
                "type": "geodesic",
                "initial_angle_deg": 45.0,
                "winding_angle_deg": 45.0,
                "angle_tolerance_deg": 0.5,
                "start_z_mm": "",
                "end_z_mm": "",
                "direction": "forward",
                "passes": "auto",
                "turnaround_radius_mm": 28.0,
                "polar_opening_radius_mm": 25.0,
                "coverage_target": 1.0,
                "feedrate_mm_min": 450.0,
                "transition_before": True,
                "transition_after": True,
                "points": 140,
                "colour": "#1e90ff",
                "notes": "",
            },
        ),
        "non_geodesic_layer": NodeTypeDefinition(
            type_id="non_geodesic_layer",
            label="Non-Geodesic Layer",
            category="Layers",
            color="#5d718f",
            inputs=(NodeSocketDefinition("material", "tow", required=False),),
            outputs=(NodeSocketDefinition("layer", "layer"),),
            default_settings={
                "name": "non_geodesic_controlled",
                "enabled": True,
                "ply_order": 30,
                "material": "carbon_tow",
                "region": "dome_to_dome",
                "type": "non_geodesic",
                "target_angle_deg": 35.0,
                "winding_angle_deg": 35.0,
                "angle_tolerance_deg": 1.0,
                "start_z_mm": "",
                "end_z_mm": "",
                "direction": "reverse",
                "passes": "auto",
                "turnaround_radius_mm": 28.0,
                "polar_opening_radius_mm": 25.0,
                "coverage_target": 1.0,
                "feedrate_mm_min": 400.0,
                "transition_before": True,
                "transition_after": True,
                "points": 140,
                "colour": "#ff851b",
                "notes": "",
            },
        ),
        "coverage_mode": NodeTypeDefinition(
            type_id="coverage_mode",
            label="Coverage Mode",
            category="Pattern",
            color="#657f3f",
            inputs=(NodeSocketDefinition("project_config", "project_config", required=False),),
            outputs=(NodeSocketDefinition("coverage_mode", "coverage_mode"),),
            default_settings={
                "individual_layer_full_coverage": False,
                "stack_level_full_coverage": True,
                "paired_layer_coverage": True,
                "z_cells": 160,
                "theta_cells": 240,
                "tow_band_model": "rectangular_surface_band",
            },
        ),
        "pin_layout_backend": NodeTypeDefinition(
            type_id="pin_layout_backend",
            label="Pin Layout",
            category="Mandrel",
            color="#8f6241",
            inputs=(NodeSocketDefinition("mandrel", "mandrel", required=False),),
            outputs=(NodeSocketDefinition("any", "any"),),
            default_settings={
                "enabled": False,
                "layout_type": "shoulder_cross",
                "shoulders": "both",
                "count_per_shoulder": 4,
                "angular_offset_deg": 0.0,
                "left_shoulder_z_mm": "",
                "right_shoulder_z_mm": "",
                "shoulder_zone_width_mm": 60.0,
                "pin_radius_mm": 4.0,
                "pin_height_mm": 25.0,
                "pin_standoff_mm": 2.0,
                "pin_clearance_mm": 0.5,
                "min_wrap_deg": 120.0,
                "max_wrap_deg": 270.0,
                "max_buildup_height_mm": 8.0,
                "max_contact_balance_ratio": 1.25,
                "friction_coefficient": "",
                "min_bend_radius_mm": "",
                "route_family": "shoulder_cross_reinforcement",
                "routing_mode": "deterministic",
                "candidate_count": 192,
                "route_step_size": 0,
                "wrap_direction": "both",
                "target_dome_angle_min_deg": 25.0,
                "target_dome_angle_max_deg": 55.0,
                "coverage_tolerance_mm": 6.0,
            },
        ),
        "pattern_optimisation_backend": NodeTypeDefinition(
            type_id="pattern_optimisation_backend",
            label="Pattern Optimisation",
            category="Pattern",
            color="#517aa3",
            inputs=(
                NodeSocketDefinition("machine_config", "machine_config", required=False),
                NodeSocketDefinition("mandrel", "mandrel", required=False),
                NodeSocketDefinition("tow", "tow", required=False),
                NodeSocketDefinition("layer_stack", "layer_stack"),
                NodeSocketDefinition("coverage_mode", "coverage_mode", required=False),
            ),
            outputs=(
                NodeSocketDefinition("pattern_candidates", "pattern_candidates"),
                NodeSocketDefinition("selected_pattern", "selected_pattern"),
                NodeSocketDefinition("winding_program", "winding_program"),
            ),
            default_settings={
                "method": "textbook_integer_closure",
                "max_p": 500,
                "max_k": 500,
                "max_d": 20,
                "angle_tolerance_deg": 0.5,
                "require_gcd_clean_pattern": True,
                "candidate_count": 10,
                "target_layer_thickness_mm": 0.25,
                "max_layer_overlap_percent": 35.0,
                "max_stack_overlap_percent": 45.0,
                "max_thickness_variation_percent": 75.0,
                "max_polar_buildup_mm": 0.75,
                "max_coverage_count": 20,
                "max_estimated_winding_time_min": 1300.0,
            },
        ),
        "validation_backend": NodeTypeDefinition(
            type_id="validation_backend",
            label="Validation",
            category="Validation",
            color="#6b6aa8",
            inputs=(
                NodeSocketDefinition("winding_program", "winding_program"),
                NodeSocketDefinition("selected_pattern", "selected_pattern", required=False),
                NodeSocketDefinition("machine_config", "machine_config", required=False),
            ),
            outputs=(NodeSocketDefinition("validation_report", "validation_report"),),
            default_settings={},
        ),
        "backend_check": NodeTypeDefinition(
            type_id="backend_check",
            label="Backend Check",
            category="Validation",
            color="#6b6aa8",
            inputs=(NodeSocketDefinition("validation_report", "validation_report"),),
            outputs=(NodeSocketDefinition("validation_report", "validation_report"),),
            default_settings={},
        ),
        "plot_backend": NodeTypeDefinition(
            type_id="plot_backend",
            label="Plot",
            category="Export",
            color="#8a6741",
            inputs=(
                NodeSocketDefinition("winding_program", "winding_program"),
                NodeSocketDefinition("validation_report", "validation_report", required=False),
            ),
            outputs=(NodeSocketDefinition("plots", "plots"),),
            default_settings={
                "enabled": True,
                "save": True,
                "show": False,
                "formats": ["png"],
                "modes": ["unwrapped", "three_d", "debug_passes", "debug_transitions"],
            },
        ),
        "csv_backend_export": NodeTypeDefinition(
            type_id="csv_backend_export",
            label="CSV Export",
            category="Export",
            color="#8a6741",
            inputs=(NodeSocketDefinition("winding_program", "winding_program"),),
            outputs=(NodeSocketDefinition("exports", "exports"),),
            default_settings={"enabled": True},
        ),
        "gcode_backend_export": NodeTypeDefinition(
            type_id="gcode_backend_export",
            label="G-code Export",
            category="Export",
            color="#8a523f",
            inputs=(NodeSocketDefinition("winding_program", "winding_program"),),
            outputs=(NodeSocketDefinition("exports", "exports"),),
            default_settings={"enabled": True},
        ),
        "report_export": NodeTypeDefinition(
            type_id="report_export",
            label="Report Export",
            category="Export",
            color="#8a6741",
            inputs=(NodeSocketDefinition("validation_report", "validation_report"),),
            outputs=(NodeSocketDefinition("exports", "exports"),),
            default_settings={"enabled": True},
        ),
        "controller_run_backend": NodeTypeDefinition(
            type_id="controller_run_backend",
            label="Controller / Machine Run",
            category="Controller",
            color="#8a3f3f",
            inputs=(NodeSocketDefinition("exports", "exports", required=False),),
            outputs=(NodeSocketDefinition("exports", "exports"),),
            default_settings={"enabled": False, "port": ""},
        ),
    }


def validate_node_type_definition(definition: NodeTypeDefinition) -> None:
    if not definition.type_id.strip():
        raise ValueError("node type id cannot be empty")
    if not definition.label.strip():
        raise ValueError(f"node type '{definition.type_id}' has no display label")
    if not definition.category.strip():
        raise ValueError(f"node type '{definition.type_id}' has no category")
    if not definition.color.strip():
        raise ValueError(f"node type '{definition.type_id}' has no color")
    if not isinstance(definition.default_settings, dict):
        raise ValueError(f"node type '{definition.type_id}' default settings must be a dict")
    _validate_socket_definitions(definition.inputs, definition.type_id, "input")
    _validate_socket_definitions(definition.outputs, definition.type_id, "output")


def validate_node_instance(
    node: NodeInstance,
    definition: NodeTypeDefinition,
) -> None:
    if node.type_id != definition.type_id:
        raise ValueError("node instance type does not match definition")
    if not node.id.strip():
        raise ValueError("node id cannot be empty")
    if not node.name.strip():
        raise ValueError("node name cannot be empty")
    if not isinstance(node.settings, dict):
        raise ValueError("node settings must be a dict")


def addable_node_type_ids(
    registry: dict[str, NodeTypeDefinition],
) -> tuple[str, ...]:
    addable = []
    for type_id, definition in registry.items():
        try:
            validate_node_type_definition(definition)
        except ValueError:
            continue
        if type_id == definition.type_id:
            addable.append(type_id)
    return tuple(addable)


def default_filament_winder_graph(
    *,
    length_mm: float = 1000.0,
    radius_mm: float = 100.0,
    tow_width_mm: float = 6.0,
    angle_deg: float = 45.0,
    point_count: int = 300,
    feedrate_mm_min: float = 500.0,
    radial_clearance_mm: float = 25.0,
    csv_path: str = "exports/node_graph_program.csv",
    gcode_path: str = "exports/node_graph_program.gcode",
    profile_path: str = "mandrels/profile.dxf",
    profile_mode: str = "cylinder",
) -> NodeGraphState:
    registry = default_node_registry()
    graph = NodeGraphState()
    mandrel = graph.add_node("mandrel_profile", registry, x=40.0, y=220.0)
    mandrel.settings.update(
        {
            "mode": profile_mode,
            "length_mm": length_mm,
            "radius_mm": radius_mm,
            "profile_path": profile_path,
        }
    )
    tow = graph.add_node("material_tow", registry, x=40.0, y=40.0)
    tow.settings.update({"tow_width_mm": tow_width_mm})
    machine = graph.add_node("machine_config", registry, x=40.0, y=400.0)
    machine.settings.update(
        {
            "radial_clearance_mm": radial_clearance_mm,
            "feedrate_mm_min": feedrate_mm_min,
        }
    )
    layers = graph.add_node("layer_stack", registry, x=420.0, y=130.0)
    layers.settings["layers"] = [
        {
            "layer_id": "helical-1",
            "enabled": True,
            "name": "helical",
            "winding_type": "helical",
            "target_angle_deg": angle_deg,
            "tow_width_mm": tow_width_mm,
            "layer_thickness_mm": 0.0,
            "coverage_target": 1.0,
            "direction": "positive",
            "number_of_passes": None,
            "feedrate_mm_min": None,
            "mandrel_clearance_mm": None,
            "colour": "#1e90ff",
            "notes": "",
            "point_count": point_count,
            "transition_points": 20,
        }
    ]
    pattern = graph.add_node("winding_pattern", registry, x=800.0, y=220.0)
    coverage = graph.add_node("coverage_analysis", registry, x=1180.0, y=40.0)
    simulation = graph.add_node("simulation", registry, x=1180.0, y=220.0)
    csv_export = graph.add_node("csv_export", registry, x=1560.0, y=130.0)
    csv_export.settings["csv_path"] = csv_path
    gcode_export = graph.add_node("gcode_export", registry, x=1560.0, y=310.0)
    gcode_export.settings.update(
        {
            "gcode_path": gcode_path,
            "feedrate_mm_min": feedrate_mm_min,
        }
    )

    graph.add_link(mandrel.id, "mandrel", pattern.id, "mandrel", registry)
    graph.add_link(tow.id, "tow", layers.id, "tow", registry)
    graph.add_link(layers.id, "layer_stack", pattern.id, "layer_stack", registry)
    graph.add_link(machine.id, "machine", pattern.id, "machine", registry)
    graph.add_link(mandrel.id, "mandrel", coverage.id, "mandrel", registry)
    graph.add_link(pattern.id, "program", coverage.id, "program", registry)
    graph.add_link(pattern.id, "program", simulation.id, "program", registry)
    graph.add_link(pattern.id, "program", csv_export.id, "program", registry)
    graph.add_link(pattern.id, "program", gcode_export.id, "program", registry)
    return graph


def default_backend_winding_graph() -> NodeGraphState:
    """Default node workflow for the config-driven domed winding backend."""

    registry = default_node_registry()
    graph = NodeGraphState()
    project = graph.add_node("project", registry, x=40.0, y=60.0)
    machine = graph.add_node("machine_backend", registry, x=380.0, y=60.0)
    mandrel = graph.add_node("mandrel_backend", registry, x=720.0, y=60.0)
    tow = graph.add_node("tow_backend", registry, x=1060.0, y=60.0)
    pins = graph.add_node("pin_layout_backend", registry, x=720.0, y=440.0)
    hoop_layer = graph.add_node("layer_backend", registry, x=1400.0, y=-160.0)
    hoop_layer.settings.update(
        {
            "name": "hoop_90deg",
            "winding_angle_deg": 90.0,
            "angle_tolerance_deg": 0.5,
            "start_z_mm": "",
            "end_z_mm": "",
            "feedrate_mm_min": 500.0,
            "points": 160,
            "colour": "#1e90ff",
        }
    )
    helical_layer = graph.add_node("layer_backend", registry, x=1400.0, y=180.0)
    helical_layer.settings.update(
        {
            "name": "helical_45deg",
            "winding_angle_deg": 45.0,
            "angle_tolerance_deg": 0.5,
            "feedrate_mm_min": 450.0,
            "points": 260,
            "colour": "#1e90ff",
        }
    )
    polar_layer = graph.add_node("layer_backend", registry, x=1400.0, y=560.0)
    polar_layer.settings.update(
        {
            "name": "polar_10deg",
            "enabled": False,
            "winding_angle_deg": 10.0,
            "angle_tolerance_deg": 1.0,
            "feedrate_mm_min": 400.0,
            "points": 260,
            "colour": "#ff851b",
        }
    )
    layers = graph.add_node("layer_stack_backend", registry, x=1840.0, y=120.0)
    coverage = graph.add_node("coverage_mode", registry, x=1840.0, y=360.0)
    pattern = graph.add_node("pattern_optimisation_backend", registry, x=2220.0, y=120.0)
    validation = graph.add_node("validation_backend", registry, x=2600.0, y=100.0)
    backend_check = graph.add_node("backend_check", registry, x=2940.0, y=100.0)
    plots = graph.add_node("plot_backend", registry, x=3280.0, y=20.0)
    reports = graph.add_node("report_export", registry, x=3280.0, y=200.0)
    csv_export = graph.add_node("csv_backend_export", registry, x=3640.0, y=20.0)
    gcode_export = graph.add_node("gcode_backend_export", registry, x=3640.0, y=200.0)

    graph.add_link(project.id, "project_config", machine.id, "project_config", registry)
    graph.add_link(machine.id, "machine_config", mandrel.id, "machine_config", registry)
    graph.add_link(mandrel.id, "mandrel", tow.id, "mandrel", registry)
    graph.add_link(mandrel.id, "mandrel", pins.id, "mandrel", registry)
    graph.add_link(tow.id, "tow", hoop_layer.id, "material", registry)
    graph.add_link(tow.id, "tow", helical_layer.id, "material", registry)
    graph.add_link(tow.id, "tow", polar_layer.id, "material", registry)
    graph.add_link(hoop_layer.id, "layer", layers.id, "layer", registry)
    graph.add_link(helical_layer.id, "layer", layers.id, "layer", registry)
    graph.add_link(polar_layer.id, "layer", layers.id, "layer", registry)
    graph.add_link(project.id, "project_config", coverage.id, "project_config", registry)
    graph.add_link(machine.id, "machine_config", pattern.id, "machine_config", registry)
    graph.add_link(mandrel.id, "mandrel", pattern.id, "mandrel", registry)
    graph.add_link(tow.id, "tow", pattern.id, "tow", registry)
    graph.add_link(layers.id, "layer_stack", pattern.id, "layer_stack", registry)
    graph.add_link(coverage.id, "coverage_mode", pattern.id, "coverage_mode", registry)
    graph.add_link(pattern.id, "winding_program", validation.id, "winding_program", registry)
    graph.add_link(pattern.id, "selected_pattern", validation.id, "selected_pattern", registry)
    graph.add_link(machine.id, "machine_config", validation.id, "machine_config", registry)
    graph.add_link(
        validation.id,
        "validation_report",
        backend_check.id,
        "validation_report",
        registry,
    )
    graph.add_link(pattern.id, "winding_program", plots.id, "winding_program", registry)
    graph.add_link(validation.id, "validation_report", plots.id, "validation_report", registry)
    graph.add_link(
        validation.id,
        "validation_report",
        reports.id,
        "validation_report",
        registry,
    )
    graph.add_link(pattern.id, "winding_program", csv_export.id, "winding_program", registry)
    graph.add_link(pattern.id, "winding_program", gcode_export.id, "winding_program", registry)
    return graph


def _default_backend_layers() -> list[dict[str, Any]]:
    return [
        {
            "name": "hoop_cylinder_1",
            "enabled": True,
            "region": "cylinder_only",
            "type": "hoop",
            "winding_mode": "hoop",
            "winding_angle_deg": 90.0,
            "passes": 1,
            "coverage_target": 1.0,
            "feedrate_mm_min": 500.0,
            "transition_before": False,
            "transition_after": True,
            "points": 80,
            "colour": "#1e90ff",
        },
        {
            "name": "geodesic_dome_to_dome_1",
            "enabled": True,
            "region": "dome_to_dome",
            "type": "geodesic",
            "winding_mode": "geodesic",
            "initial_angle_deg": 45.0,
            "winding_angle_deg": 45.0,
            "direction": "forward",
            "passes": "auto",
            "turnaround_radius_mm": 28.0,
            "polar_opening_radius_mm": 25.0,
            "coverage_target": 1.0,
            "feedrate_mm_min": 450.0,
            "transition_before": True,
            "transition_after": True,
            "points": 140,
            "colour": "#1e90ff",
        },
        {
            "name": "non_geodesic_controlled_1",
            "enabled": True,
            "region": "dome_to_dome",
            "type": "non_geodesic",
            "winding_mode": "non_geodesic",
            "target_angle_deg": 35.0,
            "winding_angle_deg": 35.0,
            "direction": "reverse",
            "passes": "auto",
            "turnaround_radius_mm": 28.0,
            "polar_opening_radius_mm": 25.0,
            "coverage_target": 1.0,
            "feedrate_mm_min": 400.0,
            "transition_before": True,
            "transition_after": True,
            "points": 140,
            "colour": "#ff851b",
        },
    ]


def _execute_mandrel_profile(node: NodeInstance) -> dict[str, Any]:
    mode = str(node.settings.get("mode", "cylinder"))
    mandrel: CylinderMandrel | AxisymmetricProfileMandrel
    if mode == "profile":
        profile_path = Path(str(node.settings.get("profile_path", "mandrels/profile.dxf")))
        samples = int(node.settings.get("samples", 0) or 0)
        mandrel = import_dxf_zr_profile(profile_path, samples=None if samples <= 0 else samples)
    else:
        mandrel = CylinderMandrel(
            length_mm=float(node.settings.get("length_mm", 1000.0)),
            radius_mm=float(node.settings.get("radius_mm", 100.0)),
        )
    return {"mandrel": mandrel}


def _execute_layer_stack(node: NodeInstance, inputs: dict[str, Any]) -> dict[str, Any]:
    tow = inputs.get("tow", {})
    default_tow_width = float(tow.get("tow_width_mm", 6.0)) if isinstance(tow, dict) else 6.0
    layers = []
    raw_layers = node.settings.get("layers", [])
    if not isinstance(raw_layers, list) or not raw_layers:
        raise ValueError("layer stack must contain at least one layer")
    for index, raw_layer in enumerate(raw_layers):
        if not isinstance(raw_layer, dict):
            raise ValueError(f"layer {index + 1} must be an object")
        layers.append(
            WindingLayerSpec(
                name=str(raw_layer.get("name", f"layer-{index + 1}")),
                winding_type=str(raw_layer.get("winding_type", "helical")),  # type: ignore[arg-type]
                target_angle_deg=float(raw_layer.get("target_angle_deg", 45.0)),
                tow_width_mm=float(raw_layer.get("tow_width_mm", default_tow_width)),
                layer_thickness_mm=float(raw_layer.get("layer_thickness_mm", 0.0)),
                coverage_target=float(raw_layer.get("coverage_target", 1.0)),
                direction=str(raw_layer.get("direction", "positive")),  # type: ignore[arg-type]
                point_count=max(2, int(raw_layer.get("point_count", 300))),
                layer_id=str(raw_layer.get("layer_id", "")),
                enabled=bool(raw_layer.get("enabled", True)),
                number_of_passes=(
                    None
                    if raw_layer.get("number_of_passes") in {None, "", 0}
                    else int(raw_layer["number_of_passes"])
                ),
                feedrate_mm_min=(
                    None
                    if raw_layer.get("feedrate_mm_min") in {None, ""}
                    else float(raw_layer["feedrate_mm_min"])
                ),
                mandrel_clearance_mm=(
                    None
                    if raw_layer.get("mandrel_clearance_mm") in {None, ""}
                    else float(raw_layer["mandrel_clearance_mm"])
                ),
                colour=str(raw_layer.get("colour", "#1e90ff")),
                notes=str(raw_layer.get("notes", "")),
                max_angle_error_deg=float(raw_layer.get("max_angle_error_deg", 5.0)),
                transition_points=max(2, int(raw_layer.get("transition_points", 20))),
            )
        )
    return {"layer_stack": WindingSchedule(layers=tuple(layers))}


def _execute_winding_pattern(node: NodeInstance, inputs: dict[str, Any]) -> dict[str, Any]:
    del node
    mandrel: CylinderMandrel | AxisymmetricProfileMandrel = inputs["mandrel"]
    schedule: WindingSchedule = inputs["layer_stack"]
    machine = inputs.get("machine", {})
    if isinstance(machine, dict):
        schedule = WindingSchedule(
            layers=schedule.layers,
            radial_clearance_mm=float(
                machine.get("radial_clearance_mm", schedule.radial_clearance_mm)
            ),
            nominal_feedrate_mm_min=float(
                machine.get("feedrate_mm_min", schedule.nominal_feedrate_mm_min)
            ),
            minimum_feedrate_mm_min=schedule.minimum_feedrate_mm_min,
        )
    program = plan_winding_schedule(mandrel, schedule)
    return {"program": program}


def _execute_coverage_analysis(inputs: dict[str, Any]) -> dict[str, Any]:
    mandrel = inputs["mandrel"]
    program: PlannedWindingProgram = inputs["program"]
    if isinstance(mandrel, CylinderMandrel):
        coverage = cylinder_coverage_map(mandrel, program.path)
    else:
        coverage = axisymmetric_surface_coverage_map(mandrel, program.path)
    return {"coverage": coverage.summary()}


def _execute_simulation(inputs: dict[str, Any]) -> dict[str, Any]:
    program: PlannedWindingProgram = inputs["program"]
    motion = program.motion_table
    return {
        "simulation": {
            "point_count": program.point_count,
            "layers": len(program.layers),
            "x_min_mm": float(motion.x_mm.min()),
            "x_max_mm": float(motion.x_mm.max()),
            "a_min_deg": float(motion.a_deg.min()),
            "a_max_deg": float(motion.a_deg.max()),
            "b_min_deg": float(motion.b_deg.min()),
            "b_max_deg": float(motion.b_deg.max()),
        }
    }


def _definition(
    registry: dict[str, NodeTypeDefinition],
    type_id: str,
) -> NodeTypeDefinition:
    if type_id not in registry:
        raise ValueError(f"unknown node type: {type_id}")
    return registry[type_id]


def _socket(
    sockets: tuple[NodeSocketDefinition, ...],
    name: str,
) -> NodeSocketDefinition | None:
    return next((socket for socket in sockets if socket.name == name), None)


def _validate_socket_definitions(
    sockets: tuple[NodeSocketDefinition, ...],
    type_id: str,
    side: str,
) -> None:
    names: set[str] = set()
    for socket in sockets:
        if not socket.name.strip():
            raise ValueError(f"node type '{type_id}' has an empty {side} socket name")
        if socket.name in names:
            raise ValueError(
                f"node type '{type_id}' has duplicate {side} socket '{socket.name}'"
            )
        names.add(socket.name)
        if not socket.kind:
            raise ValueError(
                f"node type '{type_id}' socket '{socket.name}' has no data kind"
            )


def _compatible_socket_kinds(output_kind: SocketKind, input_kind: SocketKind) -> bool:
    aliases = {
        "program": "winding_program",
        "export": "exports",
        "machine": "machine_config",
    }
    normalised_output = aliases.get(output_kind, output_kind)
    normalised_input = aliases.get(input_kind, input_kind)
    return (
        normalised_output == normalised_input
        or normalised_input == "any"
        or normalised_output == "any"
    )


def _new_id(prefix: str) -> str:
    safe_prefix = "".join(char if char.isalnum() else "_" for char in prefix).strip("_")
    return f"{safe_prefix}_{uuid.uuid4().hex[:10]}"


def _node_status(raw_status: Any) -> NodeStatus:
    if raw_status in {
        "not_run",
        "not_configured",
        "ready",
        "running",
        "warning",
        "failed",
        "passed",
        "stale",
        "error",
        "processing",
        "complete",
        "dirty",
    }:
        return raw_status
    return "not_configured"


def finite_or_default(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default
