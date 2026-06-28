import sys; sys.path.insert(0, 'src')
from filament_winder.config import load_winding_config
from filament_winder.services.winding_job import generate_winding_job
c = load_winding_config('examples/demo_domed_pressure_vessel.yaml')
r = generate_winding_job(c, make_plots=False)
print("mandrel type:", r.summary["mandrel"]["type"])
print("mandrel keys:", list(r.summary["mandrel"].keys()))
for k, v in r.summary["mandrel"].items():
    print(f"  {k}: {v}")
