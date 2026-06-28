# FilamentWinder

Python-based 4-axis filament winding software for wet winding carbon fibre over
axisymmetric mandrels.

The first implementation is intentionally small and follows this engineering
pipeline:

```text
Mandrel/profile -> surface path or layer schedule -> coverage/validation -> A/X/Z/B machine path -> CSV / G-code
```

## Axis convention

| Axis | Meaning |
|---|---|
| `A` | Mandrel rotation |
| `X` | Radial distance from mandrel centreline |
| `Z` | Mandrel longitudinal axis |
| `B` | Tow-eye rotation |

## Current status

This repo contains the Version 0.1 foundation from the project plan:

- Python package structure under `src/filament_winder`.
- `CylinderMandrel` for constant-radius mandrels.
- `HelicalPathGenerator` for single-pass and multi-pass cylinder helical paths.
- Surface path output as `z`, `theta`, `x`, and `y`.
- Full-width tow band strip generation for cylinder paths.
- Pattern closure estimates and pass-to-pass phase offsets.
- Closed cylinder pattern optimization for target coverage.
- Approximate Z-theta cylinder coverage maps with gap/overlap summaries.
- First-order A/X/Z/B machine mapping.
- Machine limit validation and rectangular X-Z no-go-zone checks.
- Optional live PySide6/VisPy preview with a tabbed setup workflow for cylinder
  winding, DXF profile winding, pattern planning, project load/save, and export
  controls.
- CSV export for surface points and machine positions.
- GRBL-style G-code export through a modular post-processor.
- Wavefront OBJ preview export for the cylinder, helix, and tow band.
- Versioned JSON project files.
- ASCII DXF Z-R profile import for common `LINE`, `LWPOLYLINE`, and `POLYLINE`
  vertex data.
- First-order profile-aware helical paths over imported Z-R profiles, with
  singular-radius guardrails.
- Profile turnaround paths that avoid pole/opening radii below a configured
  minimum.
- Dome-aware profile winding paths that use a geodesic turnaround radius and
  transition into the requested helix angle on the max-radius section.
- Textbook dome profile winding with Clairaut-based geodesic turnarounds.
- Variable tow-eye angle output for dome paths.
- Layer-level winding schedules with pattern reports, transition moves,
  per-point layer/circuit metadata, feed targets, and winding-program CSV export.
- Axisymmetric surface coverage checks for changing-radius profile paths.
- Unit tests for the math core, exports, project files, DXF import, validation,
  and CLI outputs.

Direct controller communication, production collision simulation,
acceleration-limited motion, and calibrated non-geodesic slip modelling are not
included yet. Those belong to later milestones.

## Install

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .[dev]
```

Optional live preview dependencies:

```bash
pip install -e .[gui]
```

## Run

Use the root launcher for day-to-day startup:

```bash
python run.py
```

Launcher modes:

```bash
python run.py gui            # open the normal GUI
python run.py debug          # open the GUI with extra debug logging
python run.py cli --help     # show all CLI commands
```

CLI commands can be passed through after `cli`:

```bash
python run.py cli generate --config examples/cylinder_stack.yaml
```

## Generate a cylinder winding path

```bash
filament-winder cylinder ^
  --length 1000 ^
  --radius 100 ^
  --tow-width 6 ^
  --angle 45 ^
  --points 500 ^
  --passes 4 ^
  --clearance 25 ^
  --csv exports/cylinder_path.csv ^
  --gcode exports/cylinder_path.gcode ^
  --coverage-csv exports/cylinder_coverage.csv ^
  --coverage-summary-csv exports/cylinder_coverage_summary.csv ^
  --preview-obj exports/cylinder_preview.obj ^
  --project exports/cylinder_project.fwp.json ^
  --validate ^
  --x-min 0 ^
  --x-max 150 ^
  --z-min 0 ^
  --z-max 1000
```

The CSV contains:

```text
index,pass_index,z_mm,theta_rad,surface_x_mm,surface_y_mm,surface_z_mm,A_deg,X_mm,Z_mm,B_deg
```

Inspect an imported DXF Z-R profile:

```bash
filament-winder dxf-info mandrels/profile.dxf --samples 200
```

Generate a profile turnaround path that avoids winding into pole/opening
singularities:

```bash
filament-winder profile-turnaround mandrels/profile.dxf ^
  --angle 35 ^
  --tow-width 3 ^
  --min-radius 5 ^
  --csv exports/profile_turnaround_path.csv ^
  --gcode exports/profile_turnaround_path.gcode
```

The repo includes `mandrels/profile.dxf` as a small test profile and
`mandrels/2000mm_8in_od_elliptical_dome_profile.dxf` as a smoother 2000 mm by
8 inch OD elliptical-dome sample. Replace either with your own ASCII DXF Z-R
profile when you are ready to wind a real mandrel.

Generate a dome-aware path that winds the domes and the cylinder helix as one
continuous path:

```bash
filament-winder profile-dome mandrels/2000mm_8in_od_elliptical_dome_profile.dxf ^
  --angle 35 ^
  --tow-width 3 ^
  --csv exports/profile_dome_path.csv ^
  --gcode exports/profile_dome_path.gcode
```

`profile-dome` uses the max profile radius as the cylinder radius. It computes
the geodesic turnaround radius as `max_radius * sin(angle)`, follows the
Clairaut angle change over the domes, and outputs variable `B_deg` tow-eye
angles in the CSV/G-code.

Launch the live cylinder preview:

```bash
filament-winder preview --length 1000 --radius 100 --tow-width 6 --angle 45 --passes 4
```

Launch the GUI directly in DXF profile mode with the smoother 2000 mm by
8 inch OD sample profile:

```bash
filament-winder preview --profile-dome --profile mandrels/2000mm_8in_od_elliptical_dome_profile.dxf --tow-width 3 --angle 35 --points 500
```

The preview draws the mandrel horizontally. Camera controls:

- Left-drag: full orbit around the part
- Shift + left-drag or middle-drag: pan
- Mouse wheel or right-drag: zoom
- Reset View: return to the centered horizontal view

The left setup panel is split into tabs:

- Setup: preview mode and project load/save.
- Path: cylinder dimensions and DXF profile path settings.
- Pattern: full-layer planning settings.
- Export: CSV, G-code, coverage, summary, and OBJ paths.

The settings side of the window is vertically scrollable, and the divider
between settings and the 3D viewport can be dragged to resize the workspace.

The Preview Mode selector switches between the cylinder view and a DXF-backed
Profile Dome view. In Profile Dome mode, choose an ASCII DXF Z-R profile and
use the textbook geodesic dome winding path.

Set the tow width, winding angle, points/span, turnaround settings, and
circuits, then click Inspect DXF Import to verify the profile or Update
Preview to render the path. Profile exports write CSV and G-code; coverage and
OBJ exports remain cylinder-only unless cylinder pattern coverage is selected.

The Pattern Planner panel turns the preview from a single path into a full layer
program. Enable **Use full pattern** to generate:

- cylinder hoop, `+helical`, and `-helical` schedules with closure reports
- DXF profile schedules for textbook geodesic dome winding
- per-layer colored preview paths and visible transition moves
- winding-program CSV with layer/circuit metadata, local radius, local angle,
  A/X/Z/B, feedrate, curvature, and slip-risk columns
- G-code using the adaptive per-point feed schedule

For profile modes, the hoop checkbox is ignored because hoop planning is
currently cylinder-only.

The preview Project panel can import and export `.fwp.json` files. Saved GUI
projects preserve cylinder dimensions, profile path and mode, pattern planner
settings, tow width, winding angle, points per pass/span, pass count, phase
mode, alternating direction, radial clearance, feedrate, and export paths.

The Export panel can write the current preview to CSV, GRBL-style G-code,
coverage CSV, coverage summary CSV, and OBJ preview files. Pattern mode writes
program CSV/G-code; cylinder pattern mode can also export coverage CSVs. OBJ
export remains for the single-path cylinder preview. Use Set Export Folder to
assign all export paths at once from the current project name.

The Optimize Pattern button searches integer-turn closed cylinder patterns and
updates the GUI angle/pass count to the best candidate for the target coverage.

Find optimized cylinder candidates from the CLI:

```bash
filament-winder optimize-cylinder ^
  --length 1000 ^
  --radius 100 ^
  --tow-width 6 ^
  --target-coverage 100 ^
  --max-passes 120
```

## Python API example

```python
from filament_winder.core.geometry import CylinderMandrel
from filament_winder.core.kinematics import machine_path_from_surface_path
from filament_winder.core.path_planning import (
    HelicalPathConfig,
    HelicalPathGenerator,
    estimate_cylinder_pattern_closure,
)
from filament_winder.core.tow import generate_cylinder_tow_band
from filament_winder.io import export_cylinder_preview_obj, export_gcode, export_winding_csv

mandrel = CylinderMandrel(length_mm=1000.0, radius_mm=100.0)
config = HelicalPathConfig(
    winding_angle_deg=45.0,
    tow_width_mm=6.0,
    point_count=500,
    passes=4,
)
surface_path = HelicalPathGenerator(mandrel, config).generate()
motion_table = machine_path_from_surface_path(surface_path, radial_clearance_mm=25.0)
tow_band = generate_cylinder_tow_band(mandrel, surface_path)
closure = estimate_cylinder_pattern_closure(mandrel, config)

export_winding_csv(surface_path, motion_table, "exports/path.csv")
export_gcode(motion_table, "exports/path.gcode")
export_cylinder_preview_obj(mandrel, surface_path, "exports/path.obj", tow_band=tow_band)
print(closure)
```

## Coordinate foundation

All future geometry should be represented as an axisymmetric radius function:

```text
r = r(z)
```

Surface point:

```text
P(z, theta) = [r(z) cos(theta), r(z) sin(theta), z]
```

Cylinder winding relationship:

```text
tan(alpha) = r dtheta / dz
dtheta = tan(alpha) / r dz
```

Version 0.1 machine mapping:

```text
A = theta in degrees
Z = mandrel z coordinate
X = local radius + clearance
B = requested winding angle
```

## Project structure

```text
app/main.py
docs/version_0_1.md
examples/cylinder_v0_1.py
src/filament_winder/
  app/
  cli.py
  project.py
  core/
    geometry/
    path_planning/
    kinematics/
    coverage.py
    tow.py
    validation.py
  io/
tests/
```

## Test

```bash
pytest
```

## Filament winding path guide

See `docs/FILAMENT_WINDING_PATHS_AND_PATTERNS.md` for a beginner-oriented
explanation of winding angles, closure, circuits, starts, coverage, dome
turnarounds, layer schedules, transitions, machine output, and validation.

## Next development targets

1. Add a CLI command for saved multi-layer schedules.
2. Add editable arbitrary layer stacks and layer buildup thickness in the GUI.
3. Add DXF profile preview and cleaning controls.
4. Add controller communication with dry-run, pause/resume, and stop.
5. Add non-geodesic dome path tuning modes with calibrated slip limits.
