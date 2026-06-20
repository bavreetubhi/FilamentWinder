# Version 0.1 Prototype

This milestone implements the deliberately small mathematical core from the
project plan:

- `CylinderMandrel` for a constant-radius mandrel.
- `HelicalPathGenerator` for single-pass and multi-pass cylinder helical paths.
- Surface point output as `z`, `theta`, `x`, and `y`.
- First-order A/X/Z/B machine mapping.
- CSV export containing both surface coordinates and machine positions.
- GRBL-style G-code export through a modular post-processor.
- Full-width cylinder tow band generation.
- Pattern closure estimates and pass-to-pass phase offsets.
- Closed cylinder pattern optimization for target coverage.
- Approximate Z-theta cylinder coverage maps with gap/overlap summaries.
- Machine limit validation and rectangular X-Z no-go-zone checks.
- Optional live PySide6/VisPy cylinder preview with editable winding inputs,
  project load/save controls, and export controls.
- Wavefront OBJ preview export for external 3D inspection.
- Versioned JSON project file save/load helpers.
- ASCII DXF Z-R profile import for common line and polyline data.
- First-order profile-aware helical paths over imported Z-R profiles.
- Profile turnaround paths that avoid pole/opening radii below a configured
  minimum.
- Dome-aware profile paths with a geodesic turnaround radius, cylinder helix
  transition, and variable tow-eye angle output.
- Unit tests for the math core, exports, project files, CLI outputs, and validation.

No full engineering GUI, controller streaming, or direct machine control is
included in this milestone.
