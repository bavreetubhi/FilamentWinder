"""Load human-editable CLI winding job configs."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from filament_winder.config.schema import WindingJobConfig


def load_winding_config(path: str | Path) -> WindingJobConfig:
    config_path = Path(path)
    try:
        data = _load_mapping(config_path)
    except OSError as exc:
        raise ValueError(f"could not read config {config_path}: {exc}") from exc
    return WindingJobConfig.from_mapping(data)


def _load_mapping(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        raw = json.loads(text)
    elif suffix == ".toml":
        raw = _load_toml(text)
    else:
        raw = _parse_simple_yaml(text)
    if not isinstance(raw, dict):
        raise ValueError("config root must be a mapping")
    return raw


def _load_toml(text: str) -> dict[str, Any]:
    try:
        import tomllib
    except ModuleNotFoundError as exc:  # pragma: no cover - Python 3.10 fallback
        raise ValueError("TOML configs require Python 3.11+ or use YAML/JSON") from exc
    raw = tomllib.loads(text)
    if not isinstance(raw, dict):
        raise ValueError("TOML config root must be a mapping")
    return raw


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    lines: list[tuple[int, str, int]] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = _strip_comment(raw_line).rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent % 2:
            raise ValueError(f"YAML line {line_number}: indentation must use multiples of 2")
        lines.append((indent, line.strip(), line_number))
    if not lines:
        return {}
    value, next_index = _parse_yaml_block(lines, 0, lines[0][0])
    if next_index != len(lines):
        _, _, line_number = lines[next_index]
        raise ValueError(f"YAML line {line_number}: could not parse remaining content")
    if not isinstance(value, dict):
        raise ValueError("YAML config root must be a mapping")
    return value


def _parse_yaml_block(
    lines: list[tuple[int, str, int]],
    index: int,
    indent: int,
) -> tuple[Any, int]:
    if lines[index][1].startswith("- "):
        return _parse_yaml_list(lines, index, indent)
    return _parse_yaml_mapping(lines, index, indent)


def _parse_yaml_mapping(
    lines: list[tuple[int, str, int]],
    index: int,
    indent: int,
) -> tuple[dict[str, Any], int]:
    mapping: dict[str, Any] = {}
    while index < len(lines):
        line_indent, content, line_number = lines[index]
        if line_indent < indent:
            break
        if line_indent > indent:
            raise ValueError(f"YAML line {line_number}: unexpected indentation")
        if content.startswith("- "):
            break
        key, value_text = _split_key_value(content, line_number)
        index += 1
        if value_text:
            mapping[key] = _parse_scalar(value_text)
            continue
        if index < len(lines) and lines[index][0] > line_indent:
            mapping[key], index = _parse_yaml_block(lines, index, lines[index][0])
        else:
            mapping[key] = {}
    return mapping, index


def _parse_yaml_list(
    lines: list[tuple[int, str, int]],
    index: int,
    indent: int,
) -> tuple[list[Any], int]:
    items: list[Any] = []
    while index < len(lines):
        line_indent, content, line_number = lines[index]
        if line_indent < indent:
            break
        if line_indent > indent:
            raise ValueError(f"YAML line {line_number}: unexpected indentation")
        if not content.startswith("- "):
            break
        item_text = content[2:].strip()
        index += 1
        if not item_text:
            if index < len(lines) and lines[index][0] > line_indent:
                item, index = _parse_yaml_block(lines, index, lines[index][0])
            else:
                item = {}
            items.append(item)
            continue
        if ":" in item_text and not item_text.startswith(("'", '"')):
            key, value_text = _split_key_value(item_text, line_number)
            item = {key: _parse_scalar(value_text)} if value_text else {key: {}}
            if index < len(lines) and lines[index][0] > line_indent:
                continuation, index = _parse_yaml_mapping(lines, index, lines[index][0])
                if not isinstance(continuation, dict):
                    raise ValueError(f"YAML line {line_number}: list item continuation invalid")
                item.update(continuation)
            items.append(item)
            continue
        items.append(_parse_scalar(item_text))
    return items, index


def _split_key_value(text: str, line_number: int) -> tuple[str, str]:
    if ":" not in text:
        raise ValueError(f"YAML line {line_number}: expected key: value")
    key, value = text.split(":", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"YAML line {line_number}: empty key")
    return key, value.strip()


def _strip_comment(line: str) -> str:
    in_quote: str | None = None
    escaped = False
    for index, char in enumerate(line):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char in {"'", '"'}:
            in_quote = None if in_quote == char else char if in_quote is None else in_quote
            continue
        if char == "#" and in_quote is None:
            return line[:index]
    return line


def _parse_scalar(text: str) -> Any:
    if text == "":
        return ""
    lower = text.lower()
    if lower in {"true", "yes", "on"}:
        return True
    if lower in {"false", "no", "off"}:
        return False
    if lower in {"null", "none"}:
        return None
    if text.startswith("[") and text.endswith("]"):
        return [_parse_scalar(item.strip()) for item in _split_inline_list(text[1:-1])]
    if (text.startswith('"') and text.endswith('"')) or (
        text.startswith("'") and text.endswith("'")
    ):
        try:
            return ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return text[1:-1]
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _split_inline_list(text: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    in_quote: str | None = None
    for char in text:
        if char in {"'", '"'}:
            in_quote = None if in_quote == char else char if in_quote is None else in_quote
        if char == "," and in_quote is None:
            parts.append("".join(current))
            current.clear()
            continue
        current.append(char)
    if current:
        parts.append("".join(current))
    return parts
