"""Atomic, resumable training checkpoints with lazy PyTorch imports."""

import copy
import hashlib
import importlib
import math
import os
import random
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from flowfm.data.schemaflow_registry import (
    CanonicalFeatureRegistry,
    TokenizerMetadata,
)

CHECKPOINT_FORMAT_VERSION = 4
MODEL_DATA_METADATA_FORMAT_VERSION = 2
_REQUIRED_FIELDS = frozenset(
    {
        "format_version",
        "model_data_metadata",
        "model_state",
        "adapter_state",
        "optimizer_state",
        "scheduler_state",
        "epoch",
        "global_step",
        "git_commit",
        "schema_version",
        "seed",
        "config",
        "extra_state",
        "python_random_state",
        "numpy_random_state",
        "torch_cpu_rng_state",
        "torch_cuda_rng_states",
    }
)


def _require_torch():
    try:
        return importlib.import_module("torch")
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for checkpoint operations") from exc


def _optional_numpy():
    try:
        return importlib.import_module("numpy")
    except ImportError:
        return None


def _state_dict(component: Any) -> Any:
    return None if component is None else component.state_dict()


def _normalize_model_data_metadata(value, torch, *, schema_version=None):
    required = {
        "format_version",
        "feature_registry",
        "tokenizer_metadata",
        "categorical_table_sizes",
        "label_spaces",
    }
    allowed = (required, required | {"training_contract"})
    if not isinstance(value, Mapping) or set(value) not in allowed:
        raise ValueError(
            "model_data_metadata must contain the core fields and optional "
            "training_contract"
        )
    if value["format_version"] != MODEL_DATA_METADATA_FORMAT_VERSION:
        raise ValueError(
            "unsupported model_data_metadata format version: "
            f"{value['format_version']!r}"
        )
    try:
        registry = CanonicalFeatureRegistry.from_dict(value["feature_registry"])
        tokenizer = TokenizerMetadata.from_dict(value["tokenizer_metadata"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid model_data_metadata: {exc}") from exc

    if tokenizer.registry_version != registry.version:
        raise ValueError(
            "model_data_metadata registry version does not match tokenizer metadata"
        )
    if tokenizer.registry_hash != registry.registry_hash:
        raise ValueError(
            "model_data_metadata registry hash does not match tokenizer metadata"
        )
    if schema_version is not None and tokenizer.schema_version != schema_version:
        raise ValueError(
            "model_data_metadata Schema version does not match checkpoint"
        )

    expected_table_sizes = {}
    for name, vocabulary in tokenizer.categorical_vocabularies.items():
        spec = registry.by_name.get(name)
        if (
            spec is None
            or not spec.model_visible
            or spec.feature_type != "categorical"
            or spec.feature_id != vocabulary.feature_id
            or spec.value_policy != vocabulary.value_policy
        ):
            raise ValueError(
                f"model_data_metadata vocabulary does not match registry: {name}"
            )
        expected_table_sizes[name] = vocabulary.table_size

    table_sizes = value["categorical_table_sizes"]
    if not isinstance(table_sizes, Mapping):
        raise ValueError(
            "model_data_metadata categorical_table_sizes must be a mapping"
        )
    if dict(table_sizes) != expected_table_sizes:
        raise ValueError(
            "model_data_metadata categorical table sizes do not match vocabularies"
        )

    raw_label_spaces = value["label_spaces"]
    if not isinstance(raw_label_spaces, Mapping):
        raise ValueError("model_data_metadata label_spaces must be a mapping")
    from flowfm.data.labels import DatasetLabelSpace

    label_spaces = {}
    for dataset_id, payload in sorted(raw_label_spaces.items()):
        if not isinstance(dataset_id, str) or not dataset_id:
            raise ValueError("model_data_metadata label space keys must be strings")
        try:
            label_space = DatasetLabelSpace.from_dict(payload)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid model_data_metadata label space: {exc}"
            ) from exc
        if dataset_id != label_space.dataset_id:
            raise ValueError(
                "model_data_metadata label space key does not match dataset_id"
            )
        if (
            tokenizer.data_manifest.get("dataset_id") == dataset_id
            and label_space.schema_version != tokenizer.schema_version
        ):
            raise ValueError(
                "model_data_metadata label space Schema does not match tokenizer"
            )
        label_spaces[dataset_id] = label_space.to_dict()

    normalized = {
        "format_version": MODEL_DATA_METADATA_FORMAT_VERSION,
        "feature_registry": registry.to_dict(),
        "tokenizer_metadata": tokenizer.to_dict(),
        "categorical_table_sizes": expected_table_sizes,
        "label_spaces": label_spaces,
    }
    if "training_contract" in value:
        contract = value["training_contract"]
        contract_fields = {
            "format_version",
            "architecture",
            "task",
            "task_labels",
            "model_config",
            "class_balance",
            "class_counts_train_only",
            "class_weights_train_only",
            "legacy_encoder",
            "categorical_training_audit",
            "schema_version",
            "schema_hash",
            "data_manifest",
            "split_manifest_sha256",
            "preprocessor_hash",
            "training_sample_ids_hash",
            "tokenizer_metadata_hash",
            "registry_hash",
        }
        if not isinstance(contract, Mapping) or set(contract) != contract_fields:
            raise ValueError(
                "model_data_metadata training_contract has invalid fields"
            )
        if contract["format_version"] != 1:
            raise ValueError("unsupported training_contract format version")
        if contract["architecture"] not in {
            "rescae_bigru",
            "ft_transformer",
            "schema_bigru",
        }:
            raise ValueError("training_contract architecture is invalid")
        if contract["task"] not in {"binary", "coarse", "native"}:
            raise ValueError("training_contract task is invalid")
        labels = contract["task_labels"]
        if (
            not isinstance(labels, (list, tuple))
            or len(labels) <= 1
            or not all(isinstance(label, str) and label for label in labels)
        ):
            raise ValueError("training_contract task_labels are invalid")
        counts = contract["class_counts_train_only"]
        weights = contract["class_weights_train_only"]
        if (
            not isinstance(counts, (list, tuple))
            or len(counts) != len(labels)
            or not all(type(count) is int and count > 0 for count in counts)
        ):
            raise ValueError("training_contract class counts are invalid")
        if (
            not isinstance(weights, (list, tuple))
            or len(weights) != len(labels)
            or not all(
                isinstance(weight, (int, float))
                and not isinstance(weight, bool)
                and math.isfinite(weight)
                and weight > 0
                for weight in weights
            )
        ):
            raise ValueError("training_contract class weights are invalid")
        encoder = contract["legacy_encoder"]
        if (
            not isinstance(encoder, Mapping)
            or encoder.get("fit_scope") != "train_only"
            or type(encoder.get("output_dimension")) is not int
            or encoder["output_dimension"] <= 0
        ):
            raise ValueError("training_contract legacy encoder is invalid")
        if contract["schema_version"] != tokenizer.schema_version:
            raise ValueError("training_contract Schema version does not match tokenizer")
        if contract["schema_hash"] != tokenizer.schema_hash:
            raise ValueError("training_contract Schema hash does not match tokenizer")
        if dict(contract["data_manifest"]) != dict(tokenizer.data_manifest):
            raise ValueError("training_contract Data Manifest does not match tokenizer")
        for field in (
            "split_manifest_sha256",
            "preprocessor_hash",
            "training_sample_ids_hash",
        ):
            if contract[field] != getattr(tokenizer, field):
                raise ValueError(
                    f"training_contract {field} does not match tokenizer"
                )
        if contract["tokenizer_metadata_hash"] != tokenizer.metadata_hash:
            raise ValueError("training_contract tokenizer hash does not match tokenizer")
        if contract["registry_hash"] != registry.registry_hash:
            raise ValueError("training_contract registry hash does not match registry")
        normalized["training_contract"] = dict(contract)
    try:
        return _normalize_safe(normalized, torch, "model_data_metadata")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid model_data_metadata: {exc}") from exc


def _normalize_safe(value: Any, torch, field: str) -> Any:
    """Copy values into structures accepted by restricted torch loading."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field} contains a non-finite float")
        return value
    if torch.is_tensor(value):
        return value
    if isinstance(value, Mapping):
        normalized = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{field} mappings require string keys")
            normalized[key] = _normalize_safe(item, torch, field)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_normalize_safe(item, torch, field) for item in value]
    raise TypeError(f"{field} contains unsupported type {type(value).__name__}")


def _validate_restricted_state(value: Any, torch, field: str) -> None:
    """Validate values supported by PyTorch's restricted checkpoint loader."""

    if value is None or isinstance(value, (str, bytes, bool, int)):
        return
    if isinstance(value, float):
        return
    if torch.is_tensor(value):
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if type(key) not in (str, int):
                raise ValueError(
                    f"{field} contains unsupported mapping key "
                    f"{type(key).__name__}"
                )
            _validate_restricted_state(item, torch, field)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _validate_restricted_state(item, torch, field)
        return
    raise ValueError(f"{field} contains unsupported type {type(value).__name__}")


def _encode_python_rng_state(state):
    return {
        "version": state[0],
        "state": list(state[1]),
        "gauss": state[2],
    }


def _decode_python_rng_state(state):
    if isinstance(state, Mapping):
        if set(state) != {"version", "state", "gauss"}:
            raise ValueError("python_random_state has invalid fields")
        if not isinstance(state["state"], list):
            raise ValueError("python_random_state.state must be a list")
        return state["version"], tuple(state["state"]), state["gauss"]
    return state


def _encode_numpy_rng_state(state):
    if state is None:
        return None
    return {
        "bit_generator": state[0],
        "state": state[1].tolist(),
        "position": state[2],
        "has_gauss": state[3],
        "cached_gaussian": state[4],
    }


def _decode_numpy_rng_state(state, numpy):
    if not isinstance(state, Mapping):
        return state
    required = {
        "bit_generator",
        "state",
        "position",
        "has_gauss",
        "cached_gaussian",
    }
    if set(state) != required:
        raise ValueError("numpy_random_state has invalid fields")
    return (
        state["bit_generator"],
        numpy.asarray(state["state"], dtype=numpy.uint32),
        state["position"],
        state["has_gauss"],
        state["cached_gaussian"],
    )


def _validate_numpy_rng_structure(state):
    if isinstance(state, Mapping):
        required = {
            "bit_generator",
            "state",
            "position",
            "has_gauss",
            "cached_gaussian",
        }
        if set(state) != required:
            raise ValueError("numpy_random_state has invalid fields")
        if not isinstance(state["bit_generator"], str):
            raise ValueError("numpy_random_state.bit_generator must be a string")
        if not isinstance(state["state"], list) or not all(
            isinstance(item, int) for item in state["state"]
        ):
            raise ValueError("numpy_random_state.state must be an integer list")
        if not isinstance(state["position"], int) or not isinstance(
            state["has_gauss"], int
        ):
            raise ValueError("numpy_random_state counters must be integers")
        if not isinstance(state["cached_gaussian"], (int, float)):
            raise ValueError("numpy_random_state.cached_gaussian must be numeric")
        return
    if not isinstance(state, tuple) or len(state) != 5:
        raise ValueError("numpy_random_state must be a normalized mapping")


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


def save_checkpoint(
    path,
    model,
    *,
    epoch,
    global_step,
    git_commit,
    schema_version,
    seed,
    config,
    model_data_metadata,
    optimizer=None,
    scheduler=None,
    adapter=None,
    extra_state=None,
):
    """Atomically save all state needed to resume a training run."""

    torch = _require_torch()
    numpy = _optional_numpy()
    destination = Path(path).expanduser().resolve()
    cuda_states = None
    if torch.cuda.is_available():
        cuda_states = torch.cuda.get_rng_state_all()
    normalized_model_data_metadata = _normalize_model_data_metadata(
        model_data_metadata,
        torch,
        schema_version=schema_version,
    )

    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_data_metadata": normalized_model_data_metadata,
        "model_state": model.state_dict(),
        "adapter_state": _state_dict(adapter),
        "optimizer_state": _state_dict(optimizer),
        "scheduler_state": _state_dict(scheduler),
        "epoch": epoch,
        "global_step": global_step,
        "git_commit": git_commit,
        "schema_version": schema_version,
        "seed": seed,
        "config": _normalize_safe(config, torch, "config"),
        "extra_state": _normalize_safe(extra_state, torch, "extra_state"),
        "python_random_state": _encode_python_rng_state(random.getstate()),
        "numpy_random_state": (
            None if numpy is None else _encode_numpy_rng_state(numpy.random.get_state())
        ),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_states": cuda_states,
    }
    _validate_payload(
        payload,
        torch,
        numpy,
        {
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "adapter": adapter,
        },
        restore_rng=True,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=str(destination.parent),
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            torch.save(payload, temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
        _fsync_directory(destination.parent)
    except Exception:
        if temporary_path is not None and temporary_path.is_file():
            temporary_path.unlink()
        raise

    return payload


def _torch_load(torch, path: Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError as exc:
        message = str(exc)
        if "weights_only" not in message or "unexpected keyword" not in message:
            raise
        # Older PyTorch releases lack restricted loading. This compatibility
        # path is safe only for trusted checkpoints produced by this project.
        return torch.load(path, map_location=map_location)


def _normalize_loaded_rng_to_cpu(payload, torch):
    if not isinstance(payload, Mapping):
        return payload
    normalized = dict(payload)
    cpu_state = normalized.get("torch_cpu_rng_state")
    if torch.is_tensor(cpu_state):
        normalized["torch_cpu_rng_state"] = cpu_state.cpu()
    cuda_states = normalized.get("torch_cuda_rng_states")
    if isinstance(cuda_states, (list, tuple)):
        normalized["torch_cuda_rng_states"] = [
            state.cpu() if torch.is_tensor(state) else state for state in cuda_states
        ]
    return normalized


def _validate_tensor_rng(torch, value, field):
    if not torch.is_tensor(value):
        raise ValueError(f"malformed checkpoint: {field} must be a torch tensor")
    if value.dtype != torch.uint8 or value.device.type != "cpu" or value.dim() != 1:
        raise ValueError(
            f"malformed checkpoint: {field} must be a one-dimensional CPU uint8 tensor"
        )


def _validate_payload(payload: Any, torch, numpy, components, restore_rng):
    if not isinstance(payload, Mapping):
        raise ValueError("malformed checkpoint: payload must be a mapping")
    missing = sorted(_REQUIRED_FIELDS.difference(payload))
    if missing:
        raise ValueError(
            "malformed checkpoint: missing required fields: " + ", ".join(missing)
        )
    if payload["format_version"] != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            "unsupported checkpoint format version: "
            f"{payload['format_version']!r}; expected {CHECKPOINT_FORMAT_VERSION}"
        )
    if not isinstance(payload["model_state"], Mapping):
        raise ValueError("malformed checkpoint: model_state must be a mapping")
    _validate_restricted_state(payload["model_state"], torch, "model_state")

    for name in ("optimizer", "scheduler", "adapter"):
        state = payload[f"{name}_state"]
        supplied = components[name] is not None
        present = state is not None
        if present and not supplied:
            raise ValueError(
                f"checkpoint contains {name} state but caller omitted the {name}"
            )
        if supplied and not present:
            raise ValueError(
                f"caller supplied {name} but checkpoint state is missing"
            )
        if present and not isinstance(state, Mapping):
            raise ValueError(f"malformed checkpoint: {name}_state must be a mapping")
        if present:
            _validate_restricted_state(state, torch, f"{name}_state")

    if not isinstance(payload["schema_version"], str) or not payload["schema_version"]:
        raise ValueError("malformed checkpoint: schema_version must be a non-empty string")
    try:
        normalized_model_data_metadata = _normalize_model_data_metadata(
            payload["model_data_metadata"],
            torch,
            schema_version=payload["schema_version"],
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"malformed checkpoint: {exc}") from exc
    if normalized_model_data_metadata != payload["model_data_metadata"]:
        raise ValueError(
            "malformed checkpoint: model_data_metadata is not normalized"
        )
    if type(payload["epoch"]) is not int or payload["epoch"] < 0:
        raise ValueError("malformed checkpoint: epoch must be a non-negative integer")
    if type(payload["global_step"]) is not int or payload["global_step"] < 0:
        raise ValueError(
            "malformed checkpoint: global_step must be a non-negative integer"
        )
    if not isinstance(payload["git_commit"], str) or not re.fullmatch(
        r"[0-9a-f]{40}", payload["git_commit"]
    ):
        raise ValueError(
            "malformed checkpoint: git_commit must be a 40-character lowercase SHA"
        )
    if not isinstance(payload["seed"], int):
        raise ValueError("malformed checkpoint: seed must be an integer")
    if not isinstance(payload["config"], Mapping):
        raise ValueError("malformed checkpoint: config must be a mapping")
    try:
        _normalize_safe(payload["config"], torch, "config")
        _normalize_safe(payload["extra_state"], torch, "extra_state")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"malformed checkpoint: {exc}") from exc

    python_state = _decode_python_rng_state(payload["python_random_state"])
    try:
        random.Random().setstate(python_state)
    except (TypeError, ValueError) as exc:
        raise ValueError("malformed checkpoint: python_random_state is invalid") from exc

    numpy_state = None
    saved_numpy = payload["numpy_random_state"] is not None
    current_numpy = numpy is not None
    if restore_rng and saved_numpy != current_numpy:
        if saved_numpy:
            raise ValueError(
                "checkpoint contains NumPy RNG state but NumPy is unavailable"
            )
        raise ValueError(
            "NumPy is available but checkpoint NumPy RNG state is missing"
        )
    if saved_numpy:
        try:
            _validate_numpy_rng_structure(payload["numpy_random_state"])
        except (TypeError, ValueError) as exc:
            raise ValueError("malformed checkpoint: numpy_random_state is invalid") from exc
        if numpy is not None:
            try:
                numpy_state = _decode_numpy_rng_state(
                    payload["numpy_random_state"], numpy
                )
                numpy.random.RandomState().set_state(numpy_state)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "malformed checkpoint: numpy_random_state is invalid"
                ) from exc

    _validate_tensor_rng(torch, payload["torch_cpu_rng_state"], "torch_cpu_rng_state")
    cuda_states = payload["torch_cuda_rng_states"]
    saved_cuda = cuda_states is not None
    current_cuda = bool(torch.cuda.is_available())
    if restore_rng and saved_cuda != current_cuda:
        if saved_cuda:
            raise ValueError(
                "checkpoint contains CUDA RNG state but CUDA is unavailable"
            )
        raise ValueError(
            "CUDA is available but checkpoint CUDA RNG state is missing"
        )
    if cuda_states is not None:
        if not isinstance(cuda_states, (list, tuple)):
            raise ValueError(
                "malformed checkpoint: torch_cuda_rng_states must be a sequence"
            )
        for index, state in enumerate(cuda_states):
            _validate_tensor_rng(torch, state, f"torch_cuda_rng_states[{index}]")
        if restore_rng:
            device_count = torch.cuda.device_count()
            if len(cuda_states) != device_count:
                raise ValueError(
                    "checkpoint CUDA RNG device count does not match available devices"
                )
    if restore_rng:
        return python_state, numpy_state, cuda_states
    return None


def _snapshot_restore_state(torch, numpy, components, restore_rng):
    component_states = {
        name: copy.deepcopy(component.state_dict())
        for name, component in components.items()
        if component is not None
    }
    rng_state = None
    if restore_rng:
        rng_state = {
            "python": random.getstate(),
            "numpy": None if numpy is None else copy.deepcopy(numpy.random.get_state()),
            "torch": torch.get_rng_state().clone(),
            "cuda": (
                [state.clone() for state in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_available()
                else None
            ),
        }
    return component_states, rng_state


def _rollback(torch, numpy, components, component_states, rng_state):
    failures = []
    for name, component in components.items():
        if component is not None:
            try:
                component.load_state_dict(component_states[name])
            except Exception as exc:  # pragma: no cover - exceptional recovery path
                failures.append(f"{name}: {exc}")
    if rng_state is not None:
        for label, operation in (
            ("Python RNG", lambda: random.setstate(rng_state["python"])),
            (
                "NumPy RNG",
                lambda: numpy.random.set_state(rng_state["numpy"])
                if numpy is not None and rng_state["numpy"] is not None
                else None,
            ),
            ("torch CPU RNG", lambda: torch.set_rng_state(rng_state["torch"])),
            (
                "torch CUDA RNG",
                lambda: torch.cuda.set_rng_state_all(rng_state["cuda"])
                if rng_state["cuda"] is not None
                else None,
            ),
        ):
            try:
                operation()
            except Exception as exc:  # pragma: no cover - exceptional recovery path
                failures.append(f"{label}: {exc}")
    return failures


def load_checkpoint(
    path,
    model,
    *,
    optimizer=None,
    scheduler=None,
    adapter=None,
    map_location="cpu",
    restore_rng=True,
    expected_model_data_metadata,
):
    """Transactionally restore an exact training checkpoint.

    Optional components must match the checkpoint exactly. Checkpoint files
    must be trusted when using an older PyTorch release without restricted
    ``weights_only`` loading support.
    """

    torch = _require_torch()
    numpy = _optional_numpy()
    payload = _normalize_loaded_rng_to_cpu(
        _torch_load(torch, Path(path), map_location), torch
    )
    components = {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "adapter": adapter,
    }
    decoded_rng = _validate_payload(
        payload, torch, numpy, components, restore_rng
    )
    expected_metadata = _normalize_model_data_metadata(
        expected_model_data_metadata,
        torch,
    )
    if payload["model_data_metadata"] != expected_metadata:
        raise ValueError(
            "model_data_metadata mismatch; refusing incompatible checkpoint restore"
        )
    try:
        component_states, current_rng = _snapshot_restore_state(
            torch, numpy, components, restore_rng
        )
    except Exception as exc:
        raise RuntimeError("unable to snapshot current state before restore") from exc

    try:
        model.load_state_dict(payload["model_state"])
        for name in ("optimizer", "scheduler", "adapter"):
            if components[name] is not None:
                components[name].load_state_dict(payload[f"{name}_state"])

        if restore_rng:
            python_state, numpy_state, cuda_states = decoded_rng
            random.setstate(python_state)
            if numpy_state is not None:
                numpy.random.set_state(numpy_state)
            torch.set_rng_state(payload["torch_cpu_rng_state"])
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)
    except Exception as exc:
        rollback_failures = _rollback(
            torch, numpy, components, component_states, current_rng
        )
        message = "checkpoint restore failed; transaction rolled back"
        if rollback_failures:
            message += "; rollback errors: " + "; ".join(rollback_failures)
        raise RuntimeError(message) from exc

    return payload



def _validate_cross_dataset_metadata(source, expected):
    """Validate that only dataset-local vocabularies differ."""

    if source["feature_registry"] != expected["feature_registry"]:
        raise ValueError(
            "cross-dataset initialization requires an identical feature registry"
        )
    source_tokenizer = TokenizerMetadata.from_dict(
        source["tokenizer_metadata"]
    )
    target_tokenizer = TokenizerMetadata.from_dict(
        expected["tokenizer_metadata"]
    )
    if not source_tokenizer.schema_version or not target_tokenizer.schema_version:
        raise ValueError(
            "cross-dataset initialization requires non-empty Schema versions"
        )
    if (
        source_tokenizer.registry_version != target_tokenizer.registry_version
        or source_tokenizer.registry_hash != target_tokenizer.registry_hash
    ):
        raise ValueError(
            "cross-dataset initialization requires the same registry metadata"
        )
    source_vocabularies = source_tokenizer.categorical_vocabularies
    target_vocabularies = target_tokenizer.categorical_vocabularies
    if set(source_vocabularies) != set(target_vocabularies):
        raise ValueError(
            "cross-dataset initialization requires the same categorical features"
        )
    for name in sorted(source_vocabularies):
        source_vocab = source_vocabularies[name]
        target_vocab = target_vocabularies[name]
        if (
            source_vocab.feature_id != target_vocab.feature_id
            or source_vocab.value_policy != target_vocab.value_policy
        ):
            raise ValueError(
                "cross-dataset initialization found incompatible vocabulary "
                f"definition for {name}"
            )


def load_pretrained_encoder(
    path,
    encoder,
    *,
    map_location="cpu",
    expected_model_data_metadata,
    expected_model_config,
    allow_cross_dataset_init=False,
):
    """Load only a verified SchemaFlow encoder from a pretraining checkpoint."""

    torch = _require_torch()
    numpy = _optional_numpy()
    checkpoint_path = Path(path).expanduser().resolve()
    payload = _normalize_loaded_rng_to_cpu(
        _torch_load(torch, checkpoint_path, map_location),
        torch,
    )
    components = {
        "model": object(),
        "optimizer": (
            object() if payload.get("optimizer_state") is not None else None
        ),
        "scheduler": (
            object() if payload.get("scheduler_state") is not None else None
        ),
        "adapter": object() if payload.get("adapter_state") is not None else None,
    }
    _validate_payload(payload, torch, numpy, components, restore_rng=False)

    expected_metadata = _normalize_model_data_metadata(
        expected_model_data_metadata,
        torch,
    )
    transfer_mode = "exact"
    if payload["model_data_metadata"] != expected_metadata:
        if not allow_cross_dataset_init:
            raise ValueError(
                "model_data_metadata mismatch; refusing incompatible pretrained encoder"
            )
        _validate_cross_dataset_metadata(
            payload["model_data_metadata"], expected_metadata
        )
        transfer_mode = "cross_dataset_partial"

    config = payload["config"]
    if config.get("workflow") != "pretrain":
        raise ValueError("encoder initialization requires a pretraining checkpoint")
    source_model = config.get("model")
    if not isinstance(source_model, Mapping):
        raise ValueError("pretraining checkpoint model config is missing")
    if source_model.get("name") != "schemaflow_fm_pretrainer":
        raise ValueError("checkpoint model is not schemaflow_fm_pretrainer")
    fields = (
        "d_model",
        "nhead",
        "num_layers",
        "dim_feedforward",
        "dropout",
    )
    if not isinstance(expected_model_config, Mapping):
        raise TypeError("expected_model_config must be a mapping")
    try:
        source_architecture = {field: source_model[field] for field in fields}
        expected_architecture = {
            field: expected_model_config[field] for field in fields
        }
    except KeyError as exc:
        raise ValueError(
            f"pretrained encoder model config is missing {exc.args[0]}"
        ) from exc
    if source_architecture != expected_architecture:
        raise ValueError(
            "pretrained encoder architecture does not match supervised model"
        )

    prefix = "encoder."
    source_state = {
        key[len(prefix) :]: value
        for key, value in payload["model_state"].items()
        if isinstance(key, str) and key.startswith(prefix)
    }
    expected_keys = set(encoder.state_dict())
    if set(source_state) != expected_keys:
        missing = sorted(expected_keys - set(source_state))
        unexpected = sorted(set(source_state) - expected_keys)
        raise ValueError(
            "pretrained encoder state keys do not match target; "
            f"missing={missing}, unexpected={unexpected}"
        )

    current_state = copy.deepcopy(encoder.state_dict())
    skipped_keys = []
    transferable_state = {}
    for key, target_value in current_state.items():
        source_value = source_state.get(key)
        if source_value is None:
            raise ValueError(
                f"pretrained encoder state is missing target key: {key}"
            )
        if transfer_mode == "cross_dataset_partial" and key.startswith(
            "category_embeddings."
        ):
            skipped_keys.append(key)
            continue
        if tuple(source_value.shape) != tuple(target_value.shape):
            raise ValueError(
                "pretrained encoder state shape mismatch for "
                f"{key}: source={tuple(source_value.shape)} "
                f"target={tuple(target_value.shape)}"
            )
        transferable_state[key] = source_value
    try:
        encoder.load_state_dict(transferable_state, strict=False)
    except Exception as exc:
        try:
            encoder.load_state_dict(current_state, strict=True)
        except Exception as rollback_exc:
            raise RuntimeError(
                "pretrained encoder load failed and rollback failed"
            ) from rollback_exc
        raise RuntimeError(
            "pretrained encoder load failed; target encoder rolled back"
        ) from exc

    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": digest.hexdigest(),
        "source_epoch": payload["epoch"],
        "source_global_step": payload["global_step"],
        "source_git_commit": payload["git_commit"],
        "source_schema_version": payload["schema_version"],
        "source_model_config": source_architecture,
        "source_tokenizer_metadata_hash": payload["model_data_metadata"][
            "tokenizer_metadata"
        ]["metadata_hash"],
        "transfer_mode": transfer_mode,
        "loaded_encoder_key_count": len(transferable_state),
        "skipped_encoder_keys": skipped_keys,
    }
