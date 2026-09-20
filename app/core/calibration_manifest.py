"""Reproducibility manifest for scanner and dose calibrations."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from app.paths import PROJECT_ROOT


MANIFEST_NAME = "calibration_manifest.json"
ARTIFACT_NAMES = ("calibration_data.csv", "fit_parameters.csv", "field_flattening.npz")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def update_calibration_manifest(data_dir, section: str, details: dict, source_paths=()):
    """Atomically update one manifest section and recompute artifact hashes."""
    directory = Path(data_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / MANIFEST_NAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        manifest = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat()}

    sources = []
    for source in source_paths:
        path = Path(source)
        if path.is_file():
            sources.append({
                "filename": path.name,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            })

    artifacts = {}
    for name in ARTIFACT_NAMES:
        path = directory / name
        if path.is_file():
            artifacts[name] = {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }

    manifest[section] = {**details, "sources": sources}
    manifest["artifacts"] = artifacts
    manifest["software"] = {
        "git_commit": _git_commit(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
    identity_payload = json.dumps(artifacts, sort_keys=True, separators=(",", ":")).encode()
    manifest["calibration_id"] = hashlib.sha256(identity_payload).hexdigest()

    descriptor, temporary_name = tempfile.mkstemp(
        prefix="calibration_manifest_", suffix=".json", dir=directory
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, manifest_path)
    except Exception:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
        raise
    return manifest


def verify_calibration_manifest(data_dir):
    """Return artifact integrity results, or ``None`` for legacy calibrations."""
    directory = Path(data_dir).resolve()
    path = directory / MANIFEST_NAME
    if not path.is_file():
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"manifest": False, "error": str(exc)}
    results = {"manifest": True, "calibration_id": manifest.get("calibration_id")}
    for name, record in manifest.get("artifacts", {}).items():
        artifact = directory / name
        results[name] = bool(
            artifact.is_file() and sha256_file(artifact) == record.get("sha256")
        )
    return results
