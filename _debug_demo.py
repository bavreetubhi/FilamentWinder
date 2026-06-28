"""Run the demo from Python with full traceback."""
from __future__ import annotations
import sys, traceback
from pathlib import Path
sys.path.insert(0, str(Path("src").resolve()))

from filament_winder.config.schema import WindingJobConfig
from filament_winder.services.winding_job import (
    generate_winding_job,
)

# Load config using the same method as the CLI
from filament_winder.services.winding_job import _build_mandrel

import yaml
with open("examples/demo_domed_pressure_vessel.yaml") as f:
    raw = yaml.safe_load(f)

config = WindingJobConfig.from_mapping(raw)

# Check mandrel shape
mandrel = _build_mandrel(config)
print(f"Mandrel: z=[{mandrel.start_z_mm:.3f}, {mandrel.end_z_mm:.3f}]")
print(f"Mandrel r range: {float(min(mandrel.r_mm)):.3f} - {mandrel.max_radius_mm:.3f}")
print(f"Mandrel dome_shape from config: {config.mandrel.dome_shape}")

# Build the service and run
try:
    result = generate_winding_job(config)
    print("Demo OK")
    print(f"Total points: {result.total_segments}")
except Exception:
    traceback.print_exc()
