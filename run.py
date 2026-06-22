"""Friendly project launcher for FilamentWinder.

Common use:
    python run.py              # open the GUI
    python run.py gui          # open the GUI
    python run.py debug        # open the GUI with debug logging
    python run.py cli --help   # show command-line workflow help
"""

from __future__ import annotations

import argparse
import os
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
        ),
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("gui", "cli", "debug", "debug-gui"),
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
    """Allow `python run.py cli -- generate ...` as well as without `--`."""

    remainder = list(args)
    if remainder[:1] == ["--"]:
        return remainder[1:]
    return remainder


def _print_cli_hint() -> None:
    print("Tip: use `python run.py cli --help` to see all CLI commands.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    namespace = parser.parse_args(argv)
    extra_args = _normalize_remainder(namespace.args)

    if namespace.mode == "cli":
        if not extra_args:
            _print_cli_hint()
            return cli_main(["--help"])
        return cli_main(extra_args)

    if namespace.mode in {"debug", "debug-gui"}:
        print("Starting FilamentWinder debug GUI...")
        return cli_main([*DEBUG_GUI_FLAGS, *extra_args])

    print("Starting FilamentWinder GUI...")
    return cli_main([GUI_COMMAND, *extra_args])


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
