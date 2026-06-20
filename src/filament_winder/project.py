"""Versioned project-file storage."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

PROJECT_SCHEMA_VERSION = 3
SUPPORTED_PROJECT_SCHEMA_VERSIONS = {1, 2, 3}


@dataclass(frozen=True, slots=True)
class CylinderMandrelConfig:
    length_mm: float
    radius_mm: float


@dataclass(frozen=True, slots=True)
class WindingConfig:
    tow_width_mm: float
    winding_angle_deg: float
    point_count: int
    passes: int = 1
    phase_offset_deg: float | None = None
    alternate_direction: bool = True


@dataclass(frozen=True, slots=True)
class MachineConfig:
    radial_clearance_mm: float = 25.0
    feedrate_mm_min: float = 500.0


@dataclass(frozen=True, slots=True)
class OutputConfig:
    csv_path: str = "exports/cylinder_path.csv"
    gcode_path: str | None = None
    coverage_csv_path: str | None = None
    coverage_summary_csv_path: str | None = None
    preview_obj_path: str | None = None


@dataclass(frozen=True, slots=True)
class ProfilePreviewProjectConfig:
    profile_path: str = "mandrels/profile.dxf"
    samples: int | None = None
    path_mode: str = "dome"
    min_radius_mm: float = 5.0
    turnaround_radius_mm: float | None = None
    turnaround_points: int = 25
    turnaround_angle_deg: float = 180.0
    circuits: int = 1


@dataclass(frozen=True, slots=True)
class PatternPlannerProjectConfig:
    enabled: bool = False
    coverage_target: float = 1.0
    include_hoop_layer: bool = False
    balanced_pm_layers: bool = True
    max_angle_error_deg: float = 5.0


@dataclass(frozen=True, slots=True)
class UiProjectConfig:
    preview_mode: str = "cylinder"


@dataclass(frozen=True, slots=True)
class WindingProject:
    name: str
    mandrel: CylinderMandrelConfig
    winding: WindingConfig
    machine: MachineConfig = field(default_factory=MachineConfig)
    outputs: OutputConfig = field(default_factory=OutputConfig)
    profile: ProfilePreviewProjectConfig = field(default_factory=ProfilePreviewProjectConfig)
    pattern: PatternPlannerProjectConfig = field(default_factory=PatternPlannerProjectConfig)
    ui: UiProjectConfig = field(default_factory=UiProjectConfig)
    graph: dict[str, Any] = field(default_factory=dict)
    schema_version: int = PROJECT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["schema_version"] = self.schema_version
        data["project_type"] = "filament_winder"
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WindingProject:
        schema_version = int(data.get("schema_version", 0))
        if schema_version not in SUPPORTED_PROJECT_SCHEMA_VERSIONS:
            raise ValueError(
                f"unsupported project schema version {schema_version}; "
                f"expected one of {sorted(SUPPORTED_PROJECT_SCHEMA_VERSIONS)}"
            )
        return cls(
            name=str(data["name"]),
            mandrel=CylinderMandrelConfig(**data["mandrel"]),
            winding=WindingConfig(**data["winding"]),
            machine=MachineConfig(**data.get("machine", {})),
            outputs=OutputConfig(**data.get("outputs", {})),
            profile=ProfilePreviewProjectConfig(**data.get("profile", {})),
            pattern=PatternPlannerProjectConfig(**data.get("pattern", {})),
            ui=UiProjectConfig(**data.get("ui", {})),
            graph=dict(data.get("graph", {})),
            schema_version=PROJECT_SCHEMA_VERSION,
        )


def save_project(project: WindingProject, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(project.to_dict(), handle, indent=2)
        handle.write("\n")
    return path


def load_project(input_path: str | Path) -> WindingProject:
    path = Path(input_path)
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("project file must contain a JSON object")
    return WindingProject.from_dict(data)
