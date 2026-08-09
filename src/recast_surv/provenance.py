from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest().upper()


def validate_source_manifest(manifest_path: Path, workspace: Path) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path, sep="\t")
    required = {"file", "bytes", "sha256", "source_url"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Source manifest is missing columns: {sorted(missing)}")
    records = []
    for row in manifest.itertuples(index=False):
        source = (manifest_path.parent / str(row.file)).resolve()
        exists = source.is_file()
        size = source.stat().st_size if exists else None
        digest = sha256_file(source) if exists else None
        records.append(
            {
                "file": str(source.relative_to(workspace)),
                "exists": exists,
                "expected_bytes": int(row.bytes),
                "observed_bytes": size,
                "size_ok": exists and size == int(row.bytes),
                "expected_sha256": str(row.sha256).upper(),
                "observed_sha256": digest,
                "sha256_ok": exists and digest == str(row.sha256).upper(),
            }
        )
    result = pd.DataFrame(records)
    if not result[["exists", "size_ok", "sha256_ok"]].all(axis=None):
        bad = result.loc[~result[["exists", "size_ok", "sha256_ok"]].all(axis=1)]
        raise ValueError(f"Source-manifest validation failed:\n{bad.to_string(index=False)}")
    return result


def write_run_manifest(path: Path, command: str, inputs: Iterable[Path]) -> None:
    payload = {
        "command": command,
        "python": sys.version,
        "platform": platform.platform(),
        "inputs": [
            {
                "path": str(item),
                "bytes": item.stat().st_size,
                "sha256": sha256_file(item),
            }
            for item in inputs
            if item.is_file()
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

