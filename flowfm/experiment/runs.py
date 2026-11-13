"""Safe, isolated experiment run directory creation."""

import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from flowfm.config import FlowConfig
from flowfm.experiment.metadata import (
    capture_provenance,
    validate_output_path,
    write_json_snapshot,
)


_RUN_SUBDIRECTORIES = ("checkpoints", "logs", "metrics", "predictions", "artifacts")


def _sanitize_experiment_name(name: str) -> str:
    if "\x00" in name or "/" in name or "\\" in name or name in (".", ".."):
        raise ValueError("unsafe experiment name: path traversal is not allowed")
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip()).strip("._-")
    if not sanitized:
        raise ValueError("unsafe experiment name: no usable filename characters")
    return sanitized


@dataclass(frozen=True)
class RunContext:
    """Paths for a fully published run; incomplete runs are rejected."""

    run_dir: Path
    checkpoints: Path
    logs: Path
    metrics: Path
    predictions: Path
    artifacts: Path
    config_snapshot: Path
    environment_snapshot: Path

    def __post_init__(self):
        if (self.run_dir / ".incomplete").exists():
            raise ValueError(f"run directory is incomplete: {self.run_dir}")
        required_directories = (
            self.checkpoints,
            self.logs,
            self.metrics,
            self.predictions,
            self.artifacts,
        )
        required_snapshots = (self.config_snapshot, self.environment_snapshot)
        if not self.run_dir.is_dir() or not all(
            path.is_dir() for path in required_directories
        ):
            raise ValueError(f"run directory is incomplete: {self.run_dir}")
        if not all(path.is_file() for path in required_snapshots):
            raise ValueError(f"run directory is incomplete: {self.run_dir}")


def create_run_context(
    config: FlowConfig,
    repo_dir,
    dataset_hash=None,
    split_hash=None,
    now=None,
    data_root=Path("/root/autodl-tmp"),
) -> RunContext:
    """Publish one unique run using atomic snapshots and an incomplete marker.

    Whole-directory creation is not atomic. Failures deliberately leave the
    directory and its ``.incomplete`` marker for diagnosis; no directory is
    recursively removed by this phase-one implementation.
    """

    if not isinstance(config, FlowConfig):
        raise TypeError("config must be a validated FlowConfig")
    experiment = config.experiment
    run_name = _sanitize_experiment_name(experiment["name"])
    output_root = validate_output_path(
        experiment["output_root"],
        enforce_autodl_data_disk=experiment["enforce_autodl_data_disk"],
        data_root=data_root,
    )
    if dataset_hash is not None and not isinstance(dataset_hash, str):
        raise ValueError("dataset_hash must be a string or None")
    if split_hash is not None and not isinstance(split_hash, str):
        raise ValueError("split_hash must be a string or None")

    timestamp = now if now is not None else datetime.now(timezone.utc)
    if not isinstance(timestamp, datetime):
        raise TypeError("now must be a datetime")
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    timestamp = timestamp.astimezone(timezone.utc)
    directory_name = "{}-{}-{}-{}".format(
        run_name,
        timestamp.strftime("%Y%m%dT%H%M%S%fZ"),
        uuid.uuid4().hex[:8],
        config.config_hash[:12],
    )
    run_dir = output_root / directory_name

    config_payload = config.to_dict()
    environment_payload = capture_provenance(
        repo_dir,
        config_payload,
        config.schema["version"],
        dataset_hash,
        split_hash,
        seed=experiment["seed"],
    )
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        run_dir.mkdir(exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(f"run directory already exists: {run_dir}") from exc

    incomplete_marker = run_dir / ".incomplete"
    paths = {name: run_dir / name for name in _RUN_SUBDIRECTORIES}
    config_snapshot = run_dir / "config.snapshot.json"
    environment_snapshot = run_dir / "environment.snapshot.json"
    try:
        incomplete_marker.write_text(
            "run publication incomplete\n",
            encoding="utf-8",
        )
        for path in paths.values():
            path.mkdir()
        write_json_snapshot(config_snapshot, config_payload)
        write_json_snapshot(environment_snapshot, environment_payload)
        incomplete_marker.unlink()
    except Exception as exc:
        raise RuntimeError(
            f"run publication failed; {run_dir} is marked incomplete"
        ) from exc

    return RunContext(
        run_dir=run_dir,
        config_snapshot=config_snapshot,
        environment_snapshot=environment_snapshot,
        **paths,
    )


__all__ = ["RunContext", "create_run_context"]
