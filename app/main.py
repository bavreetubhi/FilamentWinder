"""Application entry point.

Use `filament-winder preview` for the optional live PySide6/VisPy preview.
Other commands remain available through the same CLI entry point.
"""

from filament_winder.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
