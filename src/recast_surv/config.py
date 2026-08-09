from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    # Configurations live under <workspace>/configs by convention.
    workspace = config_path.parent.parent
    return config, workspace


def resolve_path(workspace: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (workspace / path).resolve()


def artifacts_dir(config: dict[str, Any], workspace: Path) -> Path:
    path = resolve_path(workspace, config["project"]["artifacts_dir"])
    path.mkdir(parents=True, exist_ok=True)
    return path

