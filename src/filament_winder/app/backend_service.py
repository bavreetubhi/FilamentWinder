"""GUI-facing service facade for the config-driven winding backend."""

from __future__ import annotations

import hashlib
import json
import traceback
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from filament_winder.app.node_graph import (
    NodeGraphState,
    default_backend_winding_graph,
    default_node_registry,
)
from filament_winder.config import WindingJobConfig, load_winding_config
from filament_winder.core.path_planning import MultiLayerPatternResult
from filament_winder.services.path_validation import (
    PathValidationResult,
    validate_path_csv,
)
from filament_winder.services.winding_job import (
    WindingJobResult,
    analyze_winding_patterns,
    generate_winding_job,
    validate_winding_job_config,
)

GUI_PROJECT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class BackendCheckResult:
    checks: dict[str, bool]
    overall_ready: bool
    machine_ready: bool
    summary_hash: str
    output_directory: Path | None
    result: WindingJobResult | None
    validation: PathValidationResult | None
    reports: dict[str, Path]
    plots: tuple[Path, ...]
    log: str
    traceback_text: str = ""


@dataclass(frozen=True, slots=True)
class LoadedReportSet:
    output_directory: Path
    reports: dict[str, dict[str, Any]]
    paths: dict[str, Path]


@dataclass(frozen=True, slots=True)
class LoadedPlotSet:
    output_directory: Path
    plots: tuple[Path, ...]
    manifest: dict[str, Any]


class BackendService:
    """Small boundary between the PySide GUI and the backend winding engine."""

    def load_config(self, path: str | Path) -> WindingJobConfig:
        return load_winding_config(path)

    def save_config(self, config: WindingJobConfig, path: str | Path) -> Path:
        output_path = Path(path)
        mapping = winding_config_to_mapping(config)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix.lower() == ".json":
            output_path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
        elif output_path.suffix.lower() in {"", ".yaml", ".yml"}:
            output_path.write_text(_dump_simple_yaml(mapping), encoding="utf-8")
        else:
            raise ValueError("config export supports .json, .yaml, and .yml files")
        return output_path

    def build_project_from_graph(
        self,
        graph: NodeGraphState,
        *,
        branch_node_id: str | None = None,
    ) -> WindingJobConfig:
        return WindingJobConfig.from_mapping(
            graph_to_config_mapping(graph, branch_node_id=branch_node_id)
        )

    def import_config_to_graph(self, path: str | Path) -> NodeGraphState:
        return graph_from_winding_config(self.load_config(path))

    def export_graph_to_config(
        self,
        graph: NodeGraphState,
        path: str | Path,
        *,
        branch_node_id: str | None = None,
    ) -> Path:
        config = self.build_project_from_graph(graph, branch_node_id=branch_node_id)
        return self.save_config(config, path)

    def run_pattern_search(
        self,
        source: WindingJobConfig | NodeGraphState,
        *,
        branch_node_id: str | None = None,
    ) -> MultiLayerPatternResult | None:
        config = self._coerce_config(source, branch_node_id=branch_node_id)
        validate_winding_job_config(config)
        return analyze_winding_patterns(config)

    def generate(
        self,
        source: WindingJobConfig | NodeGraphState,
        *,
        branch_node_id: str | None = None,
        export_csv: bool | None = None,
        export_summary: bool | None = None,
        make_plots: bool | None = None,
    ) -> WindingJobResult:
        config = self._coerce_config(source, branch_node_id=branch_node_id)
        validate_winding_job_config(config)
        return generate_winding_job(
            config,
            export_csv=export_csv,
            export_summary=export_summary,
            make_plots=make_plots,
        )

    def validate_output(
        self,
        source: WindingJobConfig | NodeGraphState | WindingJobResult,
        *,
        branch_node_id: str | None = None,
    ) -> PathValidationResult:
        result = (
            source
            if isinstance(source, WindingJobResult)
            else self.generate(
                source,
                branch_node_id=branch_node_id,
                export_csv=True,
                export_summary=True,
                make_plots=False,
            )
        )
        if result.csv_path is None:
            raise ValueError("CSV output is disabled")
        return validate_path_csv(result.csv_path, summary_path=result.summary_path)

    def backend_check(
        self,
        source: WindingJobConfig | NodeGraphState,
        *,
        branch_node_id: str | None = None,
    ) -> BackendCheckResult:
        config = self._coerce_config(source, branch_node_id=branch_node_id)
        validate_winding_job_config(config)
        result = generate_winding_job(
            config,
            export_csv=True,
            export_summary=True,
            make_plots=True,
        )
        if result.csv_path is None:
            raise ValueError("CSV output is disabled")
        validation = validate_path_csv(result.csv_path, summary_path=result.summary_path)
        checks = _backend_checks(result, path_ok=validation.ok)
        machine_ready = bool(result.summary.get("machine_ready"))
        overall = machine_ready and all(checks.values())
        reports = _report_paths(result)
        plots = _plot_paths(result)
        log = _backend_check_log(checks, overall, machine_ready, result)
        return BackendCheckResult(
            checks=checks,
            overall_ready=overall,
            machine_ready=machine_ready,
            summary_hash=summary_hash(result.summary),
            output_directory=result.config.output.directory,
            result=result,
            validation=validation,
            reports=reports,
            plots=plots,
            log=log,
        )

    def backend_check_safely(
        self,
        source: WindingJobConfig | NodeGraphState,
        *,
        branch_node_id: str | None = None,
    ) -> BackendCheckResult:
        try:
            return self.backend_check(source, branch_node_id=branch_node_id)
        except Exception as exc:  # noqa: BLE001 - keeps GUI event loop alive
            return BackendCheckResult(
                checks={"Backend check": False},
                overall_ready=False,
                machine_ready=False,
                summary_hash="",
                output_directory=None,
                result=None,
                validation=None,
                reports={},
                plots=(),
                log=f"Backend check failed: {exc}",
                traceback_text=traceback.format_exc(),
            )

    def load_reports(self, output_directory: str | Path) -> LoadedReportSet:
        directory = Path(output_directory)
        reports: dict[str, dict[str, Any]] = {}
        paths: dict[str, Path] = {}
        for report_path in _candidate_report_paths(directory):
            if not report_path.exists() or report_path.suffix.lower() != ".json":
                continue
            try:
                data = json.loads(report_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                reports[report_path.stem] = data
                paths[report_path.stem] = report_path
        return LoadedReportSet(output_directory=directory, reports=reports, paths=paths)

    def load_plots(self, output_directory: str | Path) -> LoadedPlotSet:
        directory = Path(output_directory)
        manifest_path = directory / "plot_manifest.json"
        manifest: dict[str, Any] = {}
        plots: list[Path] = []
        if manifest_path.exists():
            raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(raw_manifest, dict):
                manifest = raw_manifest
                for raw_plot in raw_manifest.get("plots", ()):
                    if isinstance(raw_plot, dict) and raw_plot.get("path") is not None:
                        plots.append(_resolve_output_path(directory, str(raw_plot["path"])))
        if not plots:
            plots.extend(sorted(directory.glob("*.png")))
        existing = tuple(path for path in plots if path.exists() and path.suffix.lower() == ".png")
        return LoadedPlotSet(output_directory=directory, plots=existing, manifest=manifest)

    def export_csv(
        self,
        source: WindingJobConfig | NodeGraphState,
        *,
        branch_node_id: str | None = None,
    ) -> Path:
        result = self.generate(
            source,
            branch_node_id=branch_node_id,
            export_csv=True,
            export_summary=True,
            make_plots=False,
        )
        if result.csv_path is None:
            raise ValueError("CSV output is disabled")
        return result.csv_path

    def export_gcode(
        self,
        source: WindingJobConfig | NodeGraphState,
        *,
        branch_node_id: str | None = None,
    ) -> Path:
        config = self._coerce_config(source, branch_node_id=branch_node_id)
        mapping = winding_config_to_mapping(config)
        output = dict(_mapping(mapping.get("output"), "output"))
        output["gcode"] = True
        mapping["output"] = output
        result = generate_winding_job(
            WindingJobConfig.from_mapping(mapping),
            export_csv=True,
            export_summary=True,
            make_plots=False,
        )
        if result.gcode_path is None:
            raise ValueError("G-code output was not written")
        return result.gcode_path

    def save_gui_project(
        self,
        graph: NodeGraphState,
        path: str | Path,
        *,
        last_result: BackendCheckResult | None = None,
    ) -> Path:
        output_path = Path(path)
        payload = {
            "schema_version": GUI_PROJECT_SCHEMA_VERSION,
            "project_type": "filament_winder_gui",
            "node_graph": graph.to_dict(),
            "selected_output_folder": _selected_output_directory(graph),
            "last_backend_check": None
            if last_result is None
            else {
                "checks": last_result.checks,
                "overall_ready": last_result.overall_ready,
                "machine_ready": last_result.machine_ready,
                "summary_hash": last_result.summary_hash,
                "output_directory": None
                if last_result.output_directory is None
                else str(last_result.output_directory),
                "reports": {key: str(value) for key, value in last_result.reports.items()},
                "plots": [str(path_item) for path_item in last_result.plots],
            },
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return output_path

    def load_gui_project(self, path: str | Path) -> NodeGraphState:
        input_path = Path(path)
        data = json.loads(input_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("GUI project must contain a JSON object")
        schema_version = int(data.get("schema_version", 0))
        if schema_version != GUI_PROJECT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported GUI project schema {schema_version}; "
                f"expected {GUI_PROJECT_SCHEMA_VERSION}"
            )
        graph_data = data.get("node_graph")
        if not isinstance(graph_data, dict):
            raise ValueError("GUI project is missing node_graph")
        return NodeGraphState.from_dict(graph_data, default_node_registry())

    def _coerce_config(
        self,
        source: WindingJobConfig | NodeGraphState,
        *,
        branch_node_id: str | None,
    ) -> WindingJobConfig:
        if isinstance(source, WindingJobConfig):
            return source
        return self.build_project_from_graph(source, branch_node_id=branch_node_id)


def graph_from_winding_config(config: WindingJobConfig) -> NodeGraphState:
    graph = default_backend_winding_graph()
    for node in graph.nodes.values():
        if node.type_id == "project":
            node.settings.update(
                {
                    "name": config.project.name,
                    "units": config.project.units,
                    "output_directory": str(config.output.directory),
                }
            )
        elif node.type_id == "machine_backend":
            node.settings.update(_normalise_config_value(asdict(config.machine)))
        elif node.type_id == "mandrel_backend":
            node.settings.update(_normalise_config_value(asdict(config.mandrel)))
        elif node.type_id == "tow_backend":
            node.settings.update(_normalise_config_value(asdict(config.tow)))
        elif node.type_id == "layer_stack_backend":
            node.settings["layers"] = [
                _normalise_config_value(asdict(layer)) for layer in config.layers
            ]
        elif node.type_id == "coverage_mode":
            node.settings.update(_normalise_config_value(asdict(config.coverage_mode)))
            node.settings.update(_normalise_config_value(asdict(config.coverage)))
        elif node.type_id == "pattern_optimisation_backend":
            node.settings.update(_normalise_config_value(asdict(config.pattern_selection)))
            node.settings.update(_normalise_config_value(asdict(config.laminate_targets)))
            node.settings.update(_normalise_config_value(asdict(config.pattern_objectives)))
            node.settings.update(_normalise_config_value(asdict(config.hoop_winding)))
            node.settings.update(_normalise_config_value(asdict(config.quality_limits)))
        elif node.type_id == "plot_backend":
            node.settings.update(_normalise_config_value(asdict(config.plot)))
        elif node.type_id == "csv_backend_export":
            node.settings["enabled"] = config.output.csv
        elif node.type_id == "gcode_backend_export":
            node.settings["enabled"] = config.output.gcode
    return graph


def graph_to_config_mapping(
    graph: NodeGraphState,
    *,
    branch_node_id: str | None = None,
) -> dict[str, Any]:
    scoped_ids = _branch_scope(graph, branch_node_id)
    project = _node_settings(graph, scoped_ids, "project")
    machine = _node_settings(graph, scoped_ids, "machine_backend", "machine_config")
    mandrel = _node_settings(graph, scoped_ids, "mandrel_backend", "mandrel_profile")
    tow = _node_settings(graph, scoped_ids, "tow_backend", "material_tow")
    layer_stack = _node_settings(graph, scoped_ids, "layer_stack_backend", "layer_stack")
    pattern = _node_settings(graph, scoped_ids, "pattern_optimisation_backend")
    coverage = _node_settings(graph, scoped_ids, "coverage_mode")
    plot = _node_settings(graph, scoped_ids, "plot_backend")
    csv_export = _node_settings(graph, scoped_ids, "csv_backend_export", "csv_export")
    gcode_export = _node_settings(graph, scoped_ids, "gcode_backend_export", "gcode_export")

    output_directory = str(
        project.get("output_directory")
        or _export_parent(csv_export.get("csv_path"))
        or "exports/gui_winding_job"
    )
    tow_width = float(tow.get("width_mm", tow.get("tow_width_mm", 6.0)))
    tow_thickness = float(tow.get("thickness_mm", tow.get("layer_thickness_mm", 0.25)))
    layers = _layers_from_graph(layer_stack, graph, scoped_ids, tow_width, tow_thickness)
    mapping = {
        "project": {
            "name": str(project.get("name", "gui_winding_job")),
            "units": str(project.get("units", "mm")),
        },
        "machine": _machine_mapping(machine),
        "mandrel": _mandrel_mapping(mandrel),
        "tow": {
            "name": str(tow.get("name", "tow")),
            "width_mm": tow_width,
            "thickness_mm": tow_thickness,
            "fibre_type": str(tow.get("fibre_type", tow.get("fiber_type", ""))),
            "resin_system": str(tow.get("resin_system", "")),
            "notes": str(tow.get("notes", "")),
        },
        "roving": {
            "width_mm": tow_width,
            "thickness_mm": tow_thickness,
            "fiber_volume_fraction": float(tow.get("fiber_volume_fraction", 0.5)),
            "resin_factor": tow.get("resin_factor", "auto"),
        },
        "laminate_targets": {
            "mode": str(pattern.get("mode", "simplified")),
            "target_layer_thickness_mm": pattern.get("target_layer_thickness_mm", tow_thickness),
            "target_total_thickness_mm": pattern.get("target_total_thickness_mm"),
            "required_thickness_profile": pattern.get("required_thickness_profile"),
        },
        "pattern_selection": {
            "method": str(pattern.get("method", "textbook_integer_closure")),
            "max_p": int(pattern.get("max_p", 500)),
            "max_k": int(pattern.get("max_k", 500)),
            "max_d": int(pattern.get("max_d", 20)),
            "angle_tolerance_deg": float(pattern.get("angle_tolerance_deg", 0.5)),
            "require_gcd_clean_pattern": bool(
                pattern.get("require_gcd_clean_pattern", True)
            ),
            "candidate_count": int(pattern.get("candidate_count", 10)),
        },
        "pattern_objectives": {
            "minimise_winding_time": bool(pattern.get("minimise_winding_time", True)),
            "minimise_overlap": bool(pattern.get("minimise_overlap", True)),
            "minimise_polar_buildup": bool(pattern.get("minimise_polar_buildup", True)),
            "require_closed_pattern": bool(pattern.get("require_closed_pattern", True)),
        },
        "hoop_winding": {
            "mode": str(pattern.get("hoop_mode", pattern.get("mode", "continuous_traverse"))),
            "nominal_angle_deg": float(pattern.get("nominal_angle_deg", 89.0)),
            "min_angle_offset_from_pure_hoop_deg": float(
                pattern.get("min_angle_offset_from_pure_hoop_deg", 0.25)
            ),
            "axial_pitch_mode": str(pattern.get("axial_pitch_mode", "tow_width")),
            "allow_exact_pure_hoop": bool(pattern.get("allow_exact_pure_hoop", False)),
            "tow_state_during_traverse": str(pattern.get("tow_state_during_traverse", "on")),
        },
        "quality_limits": {
            "max_layer_overlap_percent": float(
                pattern.get("max_layer_overlap_percent", 35.0)
            ),
            "max_stack_overlap_percent": float(
                pattern.get("max_stack_overlap_percent", 45.0)
            ),
            "max_thickness_variation_percent": float(
                pattern.get("max_thickness_variation_percent", 75.0)
            ),
            "max_polar_buildup_mm": float(pattern.get("max_polar_buildup_mm", 0.75)),
            "max_coverage_count": int(pattern.get("max_coverage_count", 20)),
            "allow_min_thickness_zero": bool(pattern.get("allow_min_thickness_zero", False)),
            "max_estimated_winding_time_min": float(
                pattern.get("max_estimated_winding_time_min", 1300.0)
            ),
        },
        "coverage_mode": {
            "individual_layer_full_coverage": bool(
                coverage.get("individual_layer_full_coverage", False)
            ),
            "stack_level_full_coverage": bool(
                coverage.get("stack_level_full_coverage", True)
            ),
            "paired_layer_coverage": bool(coverage.get("paired_layer_coverage", True)),
        },
        "layers": layers,
        "coverage": {
            "z_cells": int(coverage.get("z_cells", 160)),
            "theta_cells": int(coverage.get("theta_cells", 240)),
            "tow_band_model": str(
                coverage.get("tow_band_model", "rectangular_surface_band")
            ),
        },
        "output": {
            "directory": output_directory,
            "csv": bool(csv_export.get("enabled", True)),
            "summary_json": True,
            "segments_json": True,
            "validation_report_json": True,
            "coverage_grid": True,
            "gcode": bool(gcode_export.get("enabled", True)),
        },
        "plot": {
            "enabled": bool(plot.get("enabled", True)),
            "show": bool(plot.get("show", False)),
            "save": bool(plot.get("save", True)),
            "formats": list(plot.get("formats", ["png"])),
            "modes": list(
                plot.get("modes", ["unwrapped", "three_d", "debug_passes", "debug_transitions"])
            ),
        },
    }
    return mapping


def winding_config_to_mapping(config: WindingJobConfig) -> dict[str, Any]:
    return _normalise_config_value(asdict(config))


def summary_hash(summary: Mapping[str, Any]) -> str:
    encoded = json.dumps(_normalise_config_value(dict(summary)), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _backend_checks(result: WindingJobResult, *, path_ok: bool) -> dict[str, bool]:
    summary = result.summary
    layer_status = summary.get("layer_completion_status", {})
    stack_status = summary.get("stack_uniformity_status", {})
    region_status = summary.get("region_quality_status", {})
    machine_status = summary.get("machine_smoothing_status", {})
    optimisation_status = summary.get("pattern_optimisation_status", {})
    return {
        "Config": True,
        "Pattern optimisation": bool(
            isinstance(optimisation_status, dict)
            and optimisation_status.get("pattern_optimisation_passed")
        ),
        "Path generation": path_ok,
        "Hoop continuity": bool(
            isinstance(layer_status, dict)
            and layer_status.get("continuous_traverse_passed", True)
        ),
        "Layer quality": bool(
            isinstance(layer_status, dict) and layer_status.get("strict_completion_passed")
        ),
        "Stack uniformity": bool(
            isinstance(stack_status, dict) and stack_status.get("strict_stack_passed")
        ),
        "Region quality": bool(
            isinstance(region_status, dict) and region_status.get("region_quality_passed")
        ),
        "Machine kinematics": bool(
            isinstance(machine_status, dict) and machine_status.get("machine_kinematics_passed")
        ),
        "Exports": _required_exports_exist(result),
    }


def _backend_check_log(
    checks: Mapping[str, bool],
    overall_ready: bool,
    machine_ready: bool,
    result: WindingJobResult,
) -> str:
    lines = ["Backend Check", "-------------"]
    lines.extend(f"{label}: {'PASS' if passed else 'FAIL'}" for label, passed in checks.items())
    lines.append(f"Machine-ready: {str(machine_ready).lower()}")
    lines.append(f"Overall backend-ready: {str(overall_ready).lower()}")
    lines.append(f"Output: {result.config.output.directory}")
    return "\n".join(lines)


def _required_exports_exist(result: WindingJobResult) -> bool:
    paths = [
        result.summary_path,
        result.validation_report_path,
        result.layer_completion_report_path,
        result.stack_coverage_report_path,
        result.region_quality_report_path,
        result.machine_smoothing_report_path,
        result.pattern_optimisation_report_path,
        result.candidate_pair_report_path,
        result.actual_thickness_report_path,
        result.optimisation_repair_suggestions_path,
        result.selected_pattern_path,
        result.pattern_candidates_path,
        result.pattern_rejection_report_path,
        result.plot_manifest_path,
        result.csv_path,
        result.gcode_path,
        result.segments_path,
        result.coverage_grid_path,
    ]
    return all(path is not None and path.exists() for path in paths)


def _report_paths(result: WindingJobResult) -> dict[str, Path]:
    paths = {
        "summary": result.summary_path,
        "validation_report": result.validation_report_path,
        "layer_completion_report": result.layer_completion_report_path,
        "stack_coverage_report": result.stack_coverage_report_path,
        "region_quality_report": result.region_quality_report_path,
        "machine_smoothing_report": result.machine_smoothing_report_path,
        "pattern_optimisation_report": result.pattern_optimisation_report_path,
        "candidate_pair_report": result.candidate_pair_report_path,
        "actual_thickness_report": result.actual_thickness_report_path,
        "optimisation_repair_suggestions": result.optimisation_repair_suggestions_path,
    }
    return {key: path for key, path in paths.items() if path is not None and path.exists()}


def _plot_paths(result: WindingJobResult) -> tuple[Path, ...]:
    return tuple(path for path in result.plot_paths if path.exists())


def _candidate_report_paths(directory: Path) -> tuple[Path, ...]:
    names = (
        "summary.json",
        "validation_report.json",
        "layer_completion_report.json",
        "stack_coverage_report.json",
        "region_quality_report.json",
        "machine_smoothing_report.json",
        "pattern_optimisation_report.json",
        "candidate_pair_report.json",
        "actual_thickness_report.json",
        "optimisation_repair_suggestions.json",
        "selected_pattern.json",
    )
    return tuple(directory / name for name in names)


def _resolve_output_path(output_directory: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute() or path.exists():
        return path
    candidate = output_directory / path.name
    return candidate if candidate.exists() else path


def _branch_scope(graph: NodeGraphState, branch_node_id: str | None) -> set[str] | None:
    if branch_node_id is None or branch_node_id not in graph.nodes:
        return None
    scoped = {branch_node_id, *graph.downstream_node_ids(branch_node_id)}
    pending = [branch_node_id]
    while pending:
        current = pending.pop(0)
        for link in graph.incoming_links(current):
            if link.source_node_id in scoped:
                continue
            scoped.add(link.source_node_id)
            pending.append(link.source_node_id)
    return scoped


def _node_settings(
    graph: NodeGraphState,
    scoped_ids: set[str] | None,
    *type_ids: str,
) -> dict[str, Any]:
    for node in graph.nodes.values():
        if scoped_ids is not None and node.id not in scoped_ids:
            continue
        if node.type_id in type_ids:
            return dict(node.settings)
    return {}


def _layers_from_graph(
    layer_stack: Mapping[str, Any],
    graph: NodeGraphState,
    scoped_ids: set[str] | None,
    tow_width_mm: float,
    tow_thickness_mm: float,
) -> list[dict[str, Any]]:
    raw_layers = layer_stack.get("layers")
    layers: list[dict[str, Any]] = []
    if isinstance(raw_layers, list) and raw_layers:
        layers = [
            _normalise_layer_mapping(raw_layer, index, tow_width_mm, tow_thickness_mm)
            for index, raw_layer in enumerate(raw_layers)
            if isinstance(raw_layer, Mapping)
        ]
    if layers:
        return layers
    for node in graph.nodes.values():
        if scoped_ids is not None and node.id not in scoped_ids:
            continue
        if node.type_id in {"hoop_layer", "geodesic_layer", "non_geodesic_layer"}:
            layers.append(
                _normalise_layer_mapping(
                    node.settings,
                    len(layers),
                    tow_width_mm,
                    tow_thickness_mm,
                )
            )
    if layers:
        return layers
    return [
        {
            "name": "geodesic_default",
            "type": "geodesic",
            "winding_angle_deg": 45.0,
            "region": "dome_to_dome",
            "passes": "auto",
            "coverage_target": 1.0,
            "tow_width_mm": tow_width_mm,
            "tow_thickness_mm": tow_thickness_mm,
            "points": 140,
        }
    ]


def _normalise_layer_mapping(
    raw_layer: Mapping[str, Any],
    index: int,
    tow_width_mm: float,
    tow_thickness_mm: float,
) -> dict[str, Any]:
    layer_type = str(
        raw_layer.get(
            "type",
            raw_layer.get("winding_mode", raw_layer.get("winding_type", "geodesic")),
        )
    )
    passes = raw_layer.get("passes", raw_layer.get("number_of_passes", "auto"))
    if passes in {None, "", 0}:
        passes = "auto"
    direction = str(raw_layer.get("direction", "forward"))
    if direction == "positive":
        direction = "forward"
    elif direction == "negative":
        direction = "reverse"
    return {
        "name": str(raw_layer.get("name", f"layer_{index + 1}")),
        "enabled": bool(raw_layer.get("enabled", True)),
        "region": str(raw_layer.get("region", "full_mandrel")),
        "type": layer_type,
        "winding_mode": raw_layer.get("winding_mode", layer_type),
        "initial_angle_deg": raw_layer.get("initial_angle_deg"),
        "target_angle_deg": raw_layer.get("target_angle_deg"),
        "winding_angle_deg": float(
            raw_layer.get(
                "winding_angle_deg",
                raw_layer.get("angle_deg", raw_layer.get("target_angle_deg", 45.0)),
            )
        ),
        "direction": direction,
        "passes": passes,
        "coverage_target": float(raw_layer.get("coverage_target", 1.0)),
        "turnaround_radius_mm": raw_layer.get("turnaround_radius_mm"),
        "polar_opening_radius_mm": raw_layer.get("polar_opening_radius_mm"),
        "tow_width_mm": _float_or_default(raw_layer.get("tow_width_mm"), tow_width_mm),
        "tow_thickness_mm": _float_or_default(
            raw_layer.get("tow_thickness_mm"),
            tow_thickness_mm,
        ),
        "feedrate_mm_min": raw_layer.get("feedrate_mm_min"),
        "start_z_mm": raw_layer.get("start_z_mm"),
        "end_z_mm": raw_layer.get("end_z_mm"),
        "transition_before": bool(raw_layer.get("transition_before", True)),
        "transition_after": bool(raw_layer.get("transition_after", True)),
        "phase_offset_deg": raw_layer.get("phase_offset_deg"),
        "colour": str(raw_layer.get("colour", raw_layer.get("color", "#1e90ff"))),
        "notes": str(raw_layer.get("notes", "")),
        "points": int(raw_layer.get("points", raw_layer.get("point_count", 500))),
    }


def _machine_mapping(machine: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "clearance_mm": float(
            machine.get("clearance_mm", machine.get("radial_clearance_mm", 25.0))
        ),
        "axis_order": list(machine.get("axis_order", ["A", "X", "Z", "B"])),
        "mandrel_orientation": str(machine.get("mandrel_orientation", "horizontal")),
        "controller": str(machine.get("controller", "generic")),
        "max_a_rpm": machine.get("max_a_rpm"),
        "max_x_mm": machine.get("max_x_mm"),
        "max_z_mm": machine.get("max_z_mm"),
        "max_b_deg": machine.get("max_b_deg"),
        "max_b_velocity_deg_s": machine.get("max_b_velocity_deg_s"),
        "max_segment_length_mm": machine.get("max_segment_length_mm"),
        "max_a_accel_deg_s2": machine.get("max_a_accel_deg_s2"),
        "max_x_accel_mm_s2": machine.get("max_x_accel_mm_s2"),
        "max_z_accel_mm_s2": machine.get("max_z_accel_mm_s2"),
        "max_b_accel_deg_s2": machine.get("max_b_accel_deg_s2"),
    }


def _float_or_default(value: Any, default: float) -> float:
    if value in {None, ""}:
        return default
    return float(value)


def _mandrel_mapping(mandrel: Mapping[str, Any]) -> dict[str, Any]:
    if str(mandrel.get("mode", "")) == "cylinder":
        return {
            "type": "cylinder",
            "length_mm": float(mandrel.get("length_mm", 1000.0)),
            "radius_mm": float(mandrel.get("radius_mm", 100.0)),
            "mesh_points_z": int(mandrel.get("mesh_points_z", 240)),
            "mesh_points_theta": int(mandrel.get("mesh_points_theta", 180)),
        }
    return {
        "type": str(mandrel.get("type", "cylinder_with_elliptical_domes")),
        "length_mm": float(mandrel.get("length_mm", mandrel.get("cylinder_length_mm", 1000.0))),
        "radius_mm": float(mandrel.get("radius_mm", mandrel.get("cylinder_radius_mm", 100.0))),
        "cylinder_length_mm": mandrel.get("cylinder_length_mm"),
        "cylinder_radius_mm": mandrel.get("cylinder_radius_mm"),
        "left_dome_length_mm": float(mandrel.get("left_dome_length_mm", 0.0)),
        "right_dome_length_mm": float(mandrel.get("right_dome_length_mm", 0.0)),
        "polar_opening_radius_mm": float(mandrel.get("polar_opening_radius_mm", 0.0)),
        "mesh_points_z": int(mandrel.get("mesh_points_z", 240)),
        "mesh_points_theta": int(mandrel.get("mesh_points_theta", 180)),
    }


def _selected_output_directory(graph: NodeGraphState) -> str:
    project = _node_settings(graph, None, "project")
    return str(project.get("output_directory", "exports/gui_winding_job"))


def _export_parent(path_value: Any) -> str | None:
    if path_value in {None, ""}:
        return None
    return str(Path(str(path_value)).parent)


def _normalise_config_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_normalise_config_value(item) for item in value]
    if isinstance(value, list):
        return [_normalise_config_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalise_config_value(item) for key, item in value.items()}
    return value


def _mapping(raw: object, name: str) -> Mapping[str, Any]:
    if isinstance(raw, Mapping):
        return raw
    raise ValueError(f"{name} must be a mapping")


def _dump_simple_yaml(data: Mapping[str, Any]) -> str:
    return "\n".join(_yaml_lines(data, 0)) + "\n"


def _yaml_lines(value: Any, indent: int) -> list[str]:
    prefix = " " * indent
    if isinstance(value, Mapping):
        lines: list[str] = []
        for key, item in value.items():
            if isinstance(item, (Mapping, list)):
                lines.append(f"{prefix}{key}:")
                lines.extend(_yaml_lines(item, indent + 2))
            else:
                lines.append(f"{prefix}{key}: {_yaml_scalar(item)}")
        return lines
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, Mapping):
                item_lines = _yaml_lines(item, indent + 2)
                if item_lines:
                    first = item_lines[0].lstrip()
                    lines.append(f"{prefix}- {first}")
                    lines.extend(item_lines[1:])
                else:
                    lines.append(f"{prefix}- {{}}")
            elif isinstance(item, list):
                lines.append(f"{prefix}-")
                lines.extend(_yaml_lines(item, indent + 2))
            else:
                lines.append(f"{prefix}- {_yaml_scalar(item)}")
        return lines
    return [f"{prefix}{_yaml_scalar(value)}"]


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    text = str(value)
    if not text or any(char in text for char in ":#[]{}&,") or text.strip() != text:
        return json.dumps(text)
    return text


def _dedupe_paths(paths: Iterable[Path]) -> tuple[Path, ...]:
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return tuple(unique)
