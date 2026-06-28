"""Friendly project launcher for FilamentWinder.

Common use:
    python run.py              # open the GUI (falls back to cli if no PySide6)
    python run.py gui          # open the GUI
    python run.py debug        # open the GUI with debug logging
    python run.py cli --help   # show command-line workflow help
    python run.py demo         # generate + plot the demo pressure vessel
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
os.chdir(PROJECT_ROOT)

from filament_winder.cli import main as cli_main  # noqa: E402, I001


GUI_COMMAND = "preview"
DEBUG_GUI_FLAGS = ("preview", "--debug-gui")
DEMO_CONFIG = PROJECT_ROOT / "examples" / "demo_domed_pressure_vessel.yaml"


def _gui_available() -> bool:
    try:
        import PySide6  # noqa: F401
        return True
    except ImportError:
        return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python run.py",
        description="Start FilamentWinder in GUI, CLI, or debug GUI mode.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run.py\n"
            "  python run.py gui\n"
            "  python run.py debug\n"
            "  python run.py cli generate --config examples/cylinder_stack.yaml\n"
            "  python run.py cli --help\n"
            "  python run.py demo\n"
        ),
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("gui", "cli", "debug", "debug-gui", "demo"),
        default="gui",
        help="Startup mode. Defaults to gui.",
    )
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Extra arguments passed to the selected mode.",
    )
    return parser


def _normalize_remainder(args: Sequence[str]) -> list[str]:
    remainder = list(args)
    if remainder[:1] == ["--"]:
        return remainder[1:]
    return remainder


def _run_demo() -> int:
    if not DEMO_CONFIG.exists():
        print(f"Demo config not found: {DEMO_CONFIG}", file=sys.stderr)
        return 1
    print(f"Generating demo: {DEMO_CONFIG.name}")
    ret = cli_main(["generate", str(DEMO_CONFIG)])
    if ret != 0:
        return ret
    print("Plotting results...")
    return cli_main(["plot", str(DEMO_CONFIG)])


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    namespace = parser.parse_args(argv)
    extra_args = _normalize_remainder(namespace.args)

    if namespace.mode == "demo":
        return _run_demo()

    if namespace.mode == "cli":
        if not extra_args:
            print("Tip: use `python run.py cli --help` to see all CLI commands.")
            return cli_main(["--help"])
        return cli_main(extra_args)

    if namespace.mode in {"debug", "debug-gui"}:
        print("Starting FilamentWinder debug GUI...")
        return cli_main([*DEBUG_GUI_FLAGS, *extra_args])

    if not _gui_available():
        print("PySide6 not installed — running demo (generate + plot) instead.")
        print("  Install GUI: pip install PySide6>=6.5 vispy>=0.14")
        return _run_demo()

    print("Starting FilamentWinder GUI...")
    return cli_main([GUI_COMMAND, *extra_args])


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
