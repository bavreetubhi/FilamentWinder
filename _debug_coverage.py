import sys; sys.path.insert(0, 'src')
from pathlib import Path
from filament_winder.config import load_winding_config
from filament_winder.services.winding_job import generate_winding_job
c = load_winding_config('examples/demo_domed_pressure_vessel.yaml')
r = generate_winding_job(c, make_plots=False)
import json
for side in ['left', 'right']:
    p = Path(getattr(r, f'{side}_dome_coverage_report_path'))
    j = json.loads(p.read_text('utf-8'))
    s = j['summary']
    print(f"{side}: mean_angle={s['measured_shell_winding_angle_mean_deg']:.2f}, "
          f"covered={s['covered_area_percentage']}%, gap={s['maximum_uncovered_gap_mm']}, "
          f"pass={s['dome_coverage_passed']}")
