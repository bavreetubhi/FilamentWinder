"""Generate a simple cylinder winding CSV from Python."""

from filament_winder.core.geometry import CylinderMandrel
from filament_winder.core.kinematics import machine_path_from_surface_path
from filament_winder.core.path_planning import HelicalPathConfig, HelicalPathGenerator
from filament_winder.core.tow import generate_cylinder_tow_band
from filament_winder.io import export_cylinder_preview_obj, export_gcode, export_winding_csv

mandrel = CylinderMandrel(length_mm=1000.0, radius_mm=100.0)
config = HelicalPathConfig(winding_angle_deg=45.0, tow_width_mm=6.0, point_count=250)
surface_path = HelicalPathGenerator(mandrel, config).generate()
motion_table = machine_path_from_surface_path(surface_path, radial_clearance_mm=25.0)
tow_band = generate_cylinder_tow_band(mandrel, surface_path)

export_winding_csv(surface_path, motion_table, "exports/example_cylinder_path.csv")
export_gcode(motion_table, "exports/example_cylinder_path.gcode")
export_cylinder_preview_obj(
    mandrel,
    surface_path,
    "exports/example_cylinder_preview.obj",
    tow_band=tow_band,
)
print(f"Final rotation: {surface_path.final_rotation_deg:.3f} deg")
