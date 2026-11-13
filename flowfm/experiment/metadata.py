"""Portable experiment provenance and deterministic metadata snapshots."""

import importlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from flowfm.config import compute_config_hash


def validate_output_path(
    path,
    enforce_autodl_data_disk=False,
    data_root=Path("/root/autodl-tmp"),
):
    """Resolve an output path and optionally require it below the data root."""

    resolved = Path(path).expanduser().resolve()
    if enforce_autodl_data_disk:
        resolved_root = Path(data_root).expanduser().resolve()
        if resolved == resolved_root or resolved_root not in resolved.parents:
            raise ValueError(f"output path must be nested below {resolved_root}")
    return resolved


def _git_value(repo_dir: Path, *arguments):
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=str(repo_dir),
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (
        FileNotFoundError,
        OSError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ):
        return None
    return completed.stdout.strip()


def _git_provenance(repo_dir: Path):
    commit = _git_value(repo_dir, "rev-parse", "--verify", "HEAD")
    if commit is None:
        return None, None
    status = _git_value(repo_dir, "status", "--porcelain")
    return commit, None if status is None else bool(status)


def _torch_provenance():
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return None, None, None

    torch_version = str(getattr(torch, "__version__", "unknown"))
    version_info = getattr(torch, "version", None)
    cuda_runtime = getattr(version_info, "cuda", None)
    try:
        cuda_available = bool(torch.cuda.is_available())
    except (AttributeError, RuntimeError):
        cuda_available = None
    return torch_version, cuda_runtime, cuda_available


def capture_provenance(
    repo_dir,
    config,
    schema_version,
    dataset_hash,
    split_hash,
    seed=None,
):
    """Capture JSON-serializable software and experiment identity metadata."""

    if dataset_hash is not None and not isinstance(dataset_hash, str):
        raise ValueError("dataset_hash must be a string or None")
    if split_hash is not None and not isinstance(split_hash, str):
        raise ValueError("split_hash must be a string or None")

    commit, dirty = _git_provenance(Path(repo_dir).expanduser().resolve())
    torch_version, cuda_runtime, cuda_available = _torch_provenance()
    try:
        config_hash = compute_config_hash(config)
    except (TypeError, ValueError) as exc:
        raise ValueError("config is not canonically JSON serializable") from exc
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "git_commit": commit,
        "git_dirty": dirty,
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch_version,
        "cuda_runtime": cuda_runtime,
        "cuda_available": cuda_available,
        "config_hash": config_hash,
        "schema_version": schema_version,
        "dataset_hash": dataset_hash,
        "split_hash": split_hash,
        "seed": seed,
    }
    try:
        json.dumps(payload, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("provenance payload is not strict JSON serializable") from exc
    return payload


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(str(directory), flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def write_json_snapshot(path, payload):
    """Atomically write deterministic, human-readable UTF-8 JSON."""

    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    destination = validate_output_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(destination.parent),
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(serialized)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
        _fsync_directory(destination.parent)
    except Exception:
        if temporary_path is not None and temporary_path.is_file():
            temporary_path.unlink()
        raise

    return destination
