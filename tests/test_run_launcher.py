from __future__ import annotations

import run


def test_run_defaults_to_gui(monkeypatch) -> None:
    calls: list[list[str]] = []

    monkeypatch.setattr(run, "cli_main", lambda argv: calls.append(argv) or 0)

    assert run.main([]) == 0
    assert calls == [["preview"]]


def test_run_starts_debug_gui(monkeypatch) -> None:
    calls: list[list[str]] = []

    monkeypatch.setattr(run, "cli_main", lambda argv: calls.append(argv) or 0)

    assert run.main(["debug", "--length", "500"]) == 0
    assert calls == [["preview", "--debug-gui", "--length", "500"]]


def test_run_passes_cli_arguments(monkeypatch) -> None:
    calls: list[list[str]] = []

    monkeypatch.setattr(run, "cli_main", lambda argv: calls.append(argv) or 0)

    assert run.main(["cli", "generate", "--config", "examples/cylinder_stack.yaml"]) == 0
    assert calls == [["generate", "--config", "examples/cylinder_stack.yaml"]]
