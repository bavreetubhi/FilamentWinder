# FilamentWinder

Python-based 4-axis filament winding software for wet winding carbon fibre over axisymmetric mandrels.

The first implementation is built around a clean engineering pipeline:

```text
Mandrel geometry -> surface tow path -> tow band / coverage -> A/X/Z/B machine path -> CSV / G-code
```

## Axis convention

| Axis | Meaning |
|---|---|
| `A` | Mandrel rotation |
| `X` | Radial distance from mandrel centreline |
| `Z` | Mandrel longitudinal axis |
| `B` | Tow-eye rotation |

## Current status

This repo contains a working Version 0.1 foundation:

- Axisymmetric mandrel model based on `r = r(z)`.
- Procedural cylinder mandrel generation.
- Vectorised helical cylinder winding path generation.
- Multi-pass alternating winding support.
- Full-width tow band mesh generation.
- Approximate z-theta coverage map.
- A/X/Z/B kinematic conversion.
- Machine limit validation.
- CSV export.
- GRBL-style / FluidNC-style G-code export.
- Versioned JSON project model.
- Optional DXF Z-R profile importer.
- CLI for generating a basic cylinder winding program.
- Tests covering geometry, path generation, export, project files, and coverage.

## Install

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e .[dev]
```

Optional DXF support:

```bash
pip install -e .[dxf]
```

Optional GUI dependencies:

```bash
pip install -e .[gui]
```

## Generate a cylinder winding path

```bash
filament-winder cylinder \
  --length 1000 \
  --radius 101.6 \
  --tow-width 6 \
  --angle 45 \
  --passes 2 \
  --points 1200 \
  --feed 500 \
  --csv exports/cylinder_path.csv \
  --gcode exports/cylinder_path.gcode
```

Outputs:

```text
exports/cylinder_path.csv
exports/cylinder_path.gcode
```

## Python API example

```python
from filament_winder.geometry import AxisymmetricMandrel
from filament_winder.path import HelicalLayerSpec, generate_cylinder_helix
from filament_winder.kinematics import machine_path_from_surface_path
from filament_winder.export import export_csv, export_gcode

mandrel = AxisymmetricMandrel.cylinder(length_mm=1000.0, radius_mm=101.6)
spec = HelicalLayerSpec(
    winding_angle_deg=45.0,
    tow_width_mm=6.0,
    passes=2,
    points_per_pass=1200,
)

surface_path = generate_cylinder_helix(mandrel, spec)
machine_path = machine_path_from_surface_path(mandrel, surface_path)

export_csv(machine_path, "exports/path.csv")
export_gcode(machine_path, "exports/path.gcode")
```

## Coordinate foundation

All geometry is represented as an axisymmetric radius function:

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

Machine mapping, first-order implementation:

```text
A = theta in degrees
Z = mandrel z coordinate
X = local radius + clearance
B = local tow tangent angle
```

## Project structure

```text
src/filament_winder/
  config.py       Machine, axis, and tow configuration
  geometry.py     Axisymmetric mandrel and mesh generation
  path.py         Helical path generation and pattern estimates
  tow.py          Full-width tow band mesh generation
  kinematics.py   A/X/Z/B machine path conversion
  validation.py   Mandrel and machine-path validation
  coverage.py     Approximate surface coverage maps
  export.py       CSV and GRBL-style G-code export
  project.py      Versioned project file models
  dxf.py          Optional DXF Z-R profile import
  cli.py          Command-line interface

app/main.py       Optional GUI entry point placeholder
tests/            Unit tests
examples/         Usage examples
docs/             Architecture and milestone notes
```

## Controller target

The exporter produces GRBL-style G-code with four coordinated axes:

```gcode
G1 Z0.0000 A0.0000 X126.6000 B45.0000 F500.000
```

Use this with a controller/firmware stack that supports configured multi-axis motion, such as FluidNC or grblHAL. Vanilla GRBL is useful as a protocol reference, but should not be treated as the final 4-axis machine platform.

## Test

```bash
pytest
```

## Next development targets

1. Real VisPy viewport for mandrel, path, tow band, axis triad, and machine preview.
2. DXF profile preview and cleaning UI.
3. Dome-aware geodesic and non-geodesic path generation.
4. Slip-risk and no-go-zone validation.
5. Direct serial sender with jog, home, unlock, dry-run, pause, resume, and stop.
6. Layer stack editor with gap/overlap maps and layer buildup.
