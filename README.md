# FilamentWinder

> Python-based 4-axis filament winding software for carbon fibre tubes, pressure vessels, domes, and nose cones.

![Status](https://img.shields.io/badge/status-planning%20%2F%20prototype-blue)
![Language](https://img.shields.io/badge/python-3.11%2B-informational)
![Machine](https://img.shields.io/badge/machine-4--axis-informational)
![Controller](https://img.shields.io/badge/controller-GRBL--compatible-informational)
![Process](https://img.shields.io/badge/process-wet%20winding-informational)

---

## Overview

**FilamentWinder** is a planned professional-grade software tool for designing, previewing, simulating, and exporting winding programs for a custom 4-axis filament winding machine.

The project targets the manufacture of:

- Carbon fibre tubes
- Pressure vessels with cylindrical bodies and domed ends
- Non-geodesic and geodesic dome profiles
- Nose cones and other axisymmetric mandrels

The software is being developed in **Python**, with a high-performance 3D visualisation layer for displaying the mandrel, tow path, winding layers, payout eye, no-go zones, and machine motion.

The long-term goal is not just to generate simple winding paths, but to create a complete engineering application capable of geometry import, path planning, simulation, coverage analysis, machine validation, and GRBL-compatible machine output.

---

## Machine Axis Convention

| Axis | Description |
|---|---|
| **A** | Mandrel rotation |
| **X** | Radial distance from mandrel centreline |
| **Z** | Mandrel longitudinal axis |
| **B** | Tow eye rotation |

Initial assumptions:

- Horizontal mandrel
- Stepper-driven axes
- Wet winding process
- Single continuous tow for the first prototype
- GRBL-compatible controller target
- Closed-loop feedback and software-controlled tensioning planned for later development

---

## Planned Software Stack

| Layer | Planned Technology |
|---|---|
| Main language | Python |
| GUI | PySide6 / Qt for Python |
| 3D viewport | VisPy or equivalent high-performance OpenGL viewer |
| Geometry and numerics | NumPy, SciPy, trimesh |
| DXF import | ezdxf |
| Optional engineering visualisation | PyVista / VTK |
| Project files | Versioned JSON/TOML |
| Machine output | CSV and GRBL-style G-code |
| Controller communication | Serial sender module |

The viewer should eventually support real-time or near-real-time display of:

- Mandrel geometry
- Carbon tow centreline
- Full-width tow band
- Completed layers
- Payout eye
- No-go zones
- Winding angle maps
- Coverage maps
- Gaps and overlaps
- Axis motion replay

---

## Core Design Principle

The project should not be built as a one-off script.

The correct pipeline is:

```text
Mandrel geometry
    ↓
Surface tow path
    ↓
Tow band / coverage model
    ↓
Machine axis path
    ↓
Timed motion table
    ↓
Simulation
    ↓
CSV / G-code / direct control
```

This separation keeps the software expandable as support is added for domes, non-geodesic winding, no-go zones, multi-layer laminate buildup, and live controller communication.

---

## Geometry Model

All supported mandrels are treated as axisymmetric surfaces defined by:

```text
r = r(z)
```

Where:

- `z` is the longitudinal mandrel position
- `r` is the local radius
- `theta` is the angular coordinate around the mandrel

A surface point is represented as:

```text
P(z, theta) = [r(z) cos(theta), r(z) sin(theta), z]
```

This allows the same geometric foundation to support:

- Straight cylinders
- Cylinders with domed ends
- Pressure vessels
- Custom DXF mandrels
- Nose cones
- Non-geodesic dome profiles

---

## Initial Mandrel Support

The first geometry system will support:

| Geometry | Source |
|---|---|
| Cylinder | Procedural parameters |
| Cylinder with domes | DXF Z-R profile |
| Pressure vessel | DXF Z-R profile |
| Nose cone | DXF Z-R profile or procedural profile |

DXF files are expected to contain a 2D radius profile:

| DXF Direction | Meaning |
|---|---|
| Horizontal axis | Mandrel Z position |
| Vertical axis | Radius |

The importer should clean, resample, smooth, and validate the profile before generating a 3D mesh.

---

## Winding Modes

Planned winding modes include:

- Hoop winding
- Helical winding
- Geodesic dome winding
- Controlled non-geodesic dome winding
- Nose cone winding
- Local reinforcement layers
- Multi-layer laminate schedules

The first prototype will focus on a simple helical layer over a cylinder, then expand to cylinder-plus-dome winding.

---

## Tow and Layer Features

The tow system should eventually support:

- 6K and 12K carbon fibre tow
- User-defined tow width
- Tow thickness
- Wet winding settings
- Full-width tow band display
- Centreline display toggle
- Layer colour control
- Gap and overlap analysis
- Local laminate thickness buildup
- Updated winding surface for later layers

---

## Machine Kinematics

For each generated tow path point, the software will produce machine axis positions:

| Axis | Value |
|---|---|
| **A** | Mandrel rotation angle |
| **Z** | Carriage longitudinal position |
| **X** | Payout radial distance |
| **B** | Tow eye orientation |

Initial approximation:

```text
A = theta in degrees
Z = mandrel z coordinate
X = mandrel radius + clearance distance
B = local tow tangent angle
```

Later versions will refine the B-axis and X-axis calculations based on payout-eye geometry, clearance requirements, tow tangent direction, tow twist, and collision limits.

---

## Export Targets

The software will support both engineering inspection and machine output.

Planned exports:

- CSV motion table
- GRBL-style G-code
- Simulation replay files
- Future post-processors for other controllers

Example CSV fields:

```text
index
time_s
z_mm
a_deg
x_mm
b_deg
feedrate_mm_min
layer_id
pass_id
warning_flags
```

Example G-code style:

```gcode
G21 ; millimetres
G90 ; absolute positioning
G94 ; feed per minute

; Layer 1 - Pass 1
G1 Z0.000 A0.000 X120.000 B15.000 F500
G1 Z10.000 A35.000 X120.000 B15.000 F500
```

---

## Roadmap

### Version 0.1 — Mathematical Core

- Axisymmetric mandrel class
- Procedural cylinder generator
- Simple helical path generator
- Tow centreline generation
- Basic CSV export
- Unit tests

### Version 0.2 — First Visual Prototype

- PySide6 desktop window
- 3D viewport
- Cylinder mesh display
- Tow path display
- Full-width tow band display
- Orbit, pan, and zoom controls

### Version 0.3 — DXF Mandrel Import

- Import DXF Z-R profiles
- Clean and resample profiles
- Generate mandrel surface mesh
- Preview imported geometry
- Support cylinder-plus-dome mandrels

### Version 0.4 — 4-Axis Kinematics

- Convert surface paths to A/X/Z/B motion
- Add radial clearance control
- Add feedrate calculation
- Add axis limit checks
- Generate timed motion tables

### Version 0.5 — Export System

- CSV exporter
- GRBL-style G-code exporter
- Post-processor abstraction
- Export preview

### Version 0.6 — Controller Communication

- Serial connection
- Jogging
- Homing
- Command streaming
- Pause, resume, and stop
- Dry-run mode

### Version 0.7 — Dome Winding

- Dome-aware path generation
- Geodesic mode
- Controlled non-geodesic mode
- Dome turnaround logic
- Feedrate reduction in high-risk regions

### Version 0.8 — Coverage Analysis

- Surface coverage grid
- Gap map
- Overlap map
- Fibre angle map
- Local layer thickness map

### Version 0.9 — Layer Stack System

- Multiple layers
- Alternating winding directions
- Hoop and helical layer scheduling
- Layer visibility toggles
- Updated winding radius after each layer

### Version 1.0 — First Usable Winder Software

Version 1.0 should include:

- DXF mandrel import
- Cylinder and dome winding
- Geodesic and limited non-geodesic path generation
- Single-tow wet winding workflow
- Centreline/full-width tow display toggle
- Coverage, gap, overlap, and fibre-angle maps
- Full A/X/Z/B machine simulation
- CSV export
- GRBL-compatible G-code export
- Direct controller connection
- Jogging and dry-run mode
- Project save/load
- Machine configuration save/load

---

## First Prototype Target

The first coding target is a clean helical winding path around a cylinder.

Inputs:

```text
mandrel_length_mm
mandrel_radius_mm
tow_width_mm
winding_angle_deg
number_of_points
```

Outputs:

```text
surface path points
A axis angle
Z axis position
simple CSV
basic 3D preview
```

Cylinder path relationship:

```text
tan(alpha) = r * dtheta / dz
```

Therefore:

```text
dtheta = tan(alpha) / r * dz
```

For the first working version:

```text
theta(z) = tan(alpha) / r * z
A = theta converted to degrees
Z = z
X = r + clearance
B = alpha
```

This creates the mathematical backbone for the complete winding application.

---

## Planned Repository Structure

```text
filament_winder/
│
├── app/
│   ├── main.py
│   ├── gui/
│   ├── viewport/
│   └── controllers/
│
├── core/
│   ├── geometry/
│   ├── materials/
│   ├── layers/
│   ├── path_planning/
│   ├── kinematics/
│   ├── simulation/
│   ├── validation/
│   └── export/
│
├── machine/
│   ├── machine_config.py
│   ├── axis_config.py
│   └── calibration.py
│
├── io/
│   ├── dxf_import.py
│   ├── project_files.py
│   ├── gcode_export.py
│   └── csv_export.py
│
├── tests/
├── examples/
└── docs/
```

---

## Long-Term Features

Planned future features:

- Closed-loop axis feedback
- Software-controlled tow tensioning
- Tow break detection
- Resin bath monitoring
- Tow cut/restart events
- No-go zone editor
- Collision checking
- Tow bridging risk estimation
- Slip-risk estimation
- Pattern closure optimisation
- Multi-tow support
- Advanced post-processors
- Calibration wizard
- Live winding dashboard

---

## Development Status

This repository is currently in the planning/prototype stage.

The immediate priority is to implement the mathematical core before adding hardware control. Direct machine control should only be introduced after the geometry, path planning, kinematics, and simulation layers are producing reliable outputs.

---

## Safety Notice

This project is intended for experimental composite manufacturing software development.

Before running any generated motion on real hardware:

- Verify all axis limits
- Run simulation first
- Use dry-run mode
- Keep emergency stop hardware active
- Keep the payout eye clear of the mandrel
- Confirm G-code compatibility with the selected controller
- Do not rely on visual preview alone for machine safety

---

## License

License not yet selected.

