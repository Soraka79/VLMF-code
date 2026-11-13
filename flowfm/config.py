"""Validated, deterministic YAML configuration for SchemaFlow-FM."""

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Optional

from flowfm.data.identity import canonical_json, stable_hash


WORKFLOWS = ("pretrain", "finetune", "evaluate", "ablate")
CONFIG_HASH_NAMESPACE = "flowfm-config-v1"
_REQUIRED_TOP_LEVEL_KEYS = frozenset(
    ("workflow", "experiment", "schema", "runtime")
)
_OPTIONAL_TOP_LEVEL_KEYS = frozenset(("data", "model", "training"))


def compute_config_hash(payload: Any) -> str:
    """Return the sole canonical hash used for FlowFM configurations."""

    return stable_hash(payload, namespace=CONFIG_HASH_NAMESPACE)


def _validate_strict_json(
    value: Any,
    location: str = "configuration",
    _ancestors=None,
) -> None:
    ancestors = set() if _ancestors is None else _ancestors
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if math.isfinite(value):
            return
        raise ValueError(f"{location} must contain only strict JSON values")
    if isinstance(value, list):
        identity = id(value)
        if identity in ancestors:
            raise ValueError(f"{location} contains a cyclic strict JSON value")
        ancestors.add(identity)
        try:
            for index, item in enumerate(value):
                _validate_strict_json(item, f"{location}[{index}]", ancestors)
        finally:
            ancestors.remove(identity)
        return
    if isinstance(value, dict):
        identity = id(value)
        if identity in ancestors:
            raise ValueError(f"{location} contains a cyclic strict JSON value")
        ancestors.add(identity)
        try:
            for key, item in value.items():
                if not isinstance(key, str):
                    raise ValueError(f"{location} must contain only strict JSON values")
                _validate_strict_json(item, f"{location}.{key}", ancestors)
        finally:
            ancestors.remove(identity)
        return
    raise ValueError(f"{location} must contain only strict JSON values")


def _require_mapping(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    section = payload[key]
    if not isinstance(section, dict):
        raise ValueError(f"{key} section must be a mapping")
    return dict(section)


def _validate_keys(section: str, payload: Mapping[str, Any], required, optional=()):
    required_keys = frozenset(required)
    allowed_keys = required_keys | frozenset(optional)
    missing = sorted(required_keys - payload.keys())
    unknown = sorted(payload.keys() - allowed_keys)
    if missing:
        raise ValueError(f"{section} section missing required keys: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{section} section has unknown keys: {', '.join(unknown)}")


def _require_clean_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a nonempty string")
    if value != value.strip():
        raise ValueError(f"{field} must not contain surrounding whitespace")
    return value


def _require_positive_int(value: Any, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _require_finite_number(
    value: Any,
    field: str,
    *,
    minimum=None,
    maximum=None,
    minimum_inclusive=True,
    maximum_inclusive=True,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field} must be a finite number")
    if minimum is not None:
        invalid = normalized < minimum if minimum_inclusive else normalized <= minimum
        if invalid:
            raise ValueError(f"{field} is below its allowed range")
    if maximum is not None:
        invalid = normalized > maximum if maximum_inclusive else normalized >= maximum
        if invalid:
            raise ValueError(f"{field} is above its allowed range")
    return normalized


def _validate_train_sections(payload, workflow):
    required_sections = ("data", "model", "training")
    missing = [name for name in required_sections if name not in payload]
    if missing:
        raise ValueError(
            "train execution requires data, model, and training sections"
        )
    if workflow not in ("finetune", "pretrain"):
        raise ValueError(
            "train execution is currently implemented only for finetune or pretrain"
        )

    data = _require_mapping(payload, "data")
    model = _require_mapping(payload, "model")
    training = _require_mapping(payload, "training")
    has_sources = workflow == "pretrain" and "sources" in data
    required_data_keys = (
        ("dataset_id", "data_manifest")
        if has_sources
        else ("dataset_id", "split_dir", "preprocessor_dir", "data_manifest")
    )
    _validate_keys(
        "data",
        data,
        required=required_data_keys,
        optional=("sources",) if has_sources else (),
    )
    for key in required_data_keys:
        _require_clean_string(data[key], f"data {key}")
    if "sources" in data:
        sources = data["sources"]
        if not isinstance(sources, list) or len(sources) < 2:
            raise ValueError("pretrain data sources must contain at least two entries")
        source_ids = set()
        required_source_keys = {
            "dataset_id",
            "split_dir",
            "preprocessor_dir",
            "data_manifest",
        }
        for index, source in enumerate(sources):
            if not isinstance(source, dict) or set(source) != required_source_keys:
                raise ValueError(
                    f"pretrain data sources[{index}] has invalid keys"
                )
            for key in sorted(required_source_keys):
                _require_clean_string(
                    source[key], f"pretrain data sources[{index}] {key}"
                )
            dataset_id = source["dataset_id"]
            if dataset_id in source_ids:
                raise ValueError("pretrain data source dataset_ids must be unique")
            source_ids.add(dataset_id)
    elif workflow == "pretrain" and data["dataset_id"].startswith(
        "SchemaFlow-MultiSource"
    ):
        raise ValueError("multi-source pretraining requires data sources")
    _validate_keys(
        "model",
        model,
        required=(
            "name",
            "d_model",
            "nhead",
            "num_layers",
            "dim_feedforward",
            "dropout",
            "adapter_dim",
        ),
    )
    expected_model_name = (
        "schemaflow_fm_supervised"
        if workflow == "finetune"
        else "schemaflow_fm_pretrainer"
    )
    if _require_clean_string(model["name"], "model name") != expected_model_name:
        raise ValueError(f"train model name must be {expected_model_name}")
    for key in (
        "d_model",
        "nhead",
        "num_layers",
        "dim_feedforward",
        "adapter_dim",
    ):
        _require_positive_int(model[key], f"model {key}")
    if model["d_model"] % model["nhead"]:
        raise ValueError("model d_model must be divisible by nhead")
    _require_finite_number(
        model["dropout"],
        "model dropout",
        minimum=0,
        maximum=1,
        maximum_inclusive=False,
    )

    _validate_keys(
        "training",
        training,
        required=(
            "epochs",
            "batch_size",
            "learning_rate",
            "weight_decay",
            "gradient_clip",
            "scheduler_gamma",
            "shuffle",
            "class_balance",
            "loss_weights",
        ),
        optional=(
            "mask_fraction",
            "schema_corruption_fraction",
            "contrastive_temperature",
            "reconstruction_weight",
            "contrastive_weight",
        )
        if workflow == "pretrain"
        else (
            "adaptation_mode",
            "evaluation_scope",
            "classification_loss",
            "consistency_weight",
            "consistency_missing_seed",
            "early_stopping_patience",
            "early_stopping_interval",
            "early_stopping_min_delta",
        ),
    )
    _require_positive_int(training["epochs"], "training epochs")
    _require_positive_int(training["batch_size"], "training batch_size")
    _require_finite_number(
        training["learning_rate"],
        "training learning_rate",
        minimum=0,
        minimum_inclusive=False,
    )
    _require_finite_number(
        training["weight_decay"],
        "training weight_decay",
        minimum=0,
    )
    _require_finite_number(
        training["gradient_clip"],
        "training gradient_clip",
        minimum=0,
        minimum_inclusive=False,
    )
    _require_finite_number(
        training["scheduler_gamma"],
        "training scheduler_gamma",
        minimum=0,
        maximum=1,
        minimum_inclusive=False,
    )
    if type(training["shuffle"]) is not bool:
        raise ValueError("training shuffle must be a boolean")
    if training["class_balance"] not in {"none", "inverse_frequency"}:
        raise ValueError(
            "training class_balance must be none or inverse_frequency"
        )
    adaptation_mode = training.get("adaptation_mode", "full")
    if workflow == "finetune" and adaptation_mode not in {
        "full",
        "adapter_category_heads",
    }:
        raise ValueError("invalid finetune adaptation_mode")
    evaluation_scope = training.get("evaluation_scope", "validation_and_test")
    if workflow == "finetune" and evaluation_scope not in {
        "validation_only",
        "validation_and_test",
    }:
        raise ValueError("invalid finetune evaluation_scope")
    classification_loss = training.get(
        "classification_loss", "weighted_cross_entropy"
    )
    if workflow == "finetune" and classification_loss not in {
        "weighted_cross_entropy",
        "balanced_softmax",
    }:
        raise ValueError("invalid finetune classification_loss")
    if (
        workflow == "finetune"
        and classification_loss == "balanced_softmax"
        and training["class_balance"] != "none"
    ):
        raise ValueError(
            "balanced_softmax requires training class_balance none"
        )
    consistency_weight = training.get("consistency_weight", 0.0)
    _require_finite_number(
        consistency_weight,
        "training consistency_weight",
        minimum=0,
        maximum=1,
    )
    consistency_missing_seed = training.get("consistency_missing_seed", 0)
    if type(consistency_missing_seed) is not int or consistency_missing_seed < 0:
        raise ValueError(
            "training consistency_missing_seed must be a non-negative integer"
        )
    early_stopping_patience = training.get("early_stopping_patience", 0)
    if type(early_stopping_patience) is not int or early_stopping_patience < 0:
        raise ValueError(
            "training early_stopping_patience must be a non-negative integer"
        )
    _require_positive_int(
        training.get("early_stopping_interval", 1),
        "training early_stopping_interval",
    )
    _require_finite_number(
        training.get("early_stopping_min_delta", 0.0),
        "training early_stopping_min_delta",
        minimum=0,
    )
    if workflow == "finetune" and early_stopping_patience > 0:
        if evaluation_scope != "validation_only":
            raise ValueError("early stopping requires validation_only scope")
        if training.get("early_stopping_interval", 1) > training["epochs"]:
            raise ValueError("early stopping interval exceeds training epochs")
    weights = training["loss_weights"]
    if not isinstance(weights, dict) or set(weights) != {
        "binary",
        "coarse",
        "native",
    }:
        raise ValueError(
            "training loss_weights must contain binary, coarse, and native"
        )
    normalized_weights = {
        key: _require_finite_number(
            weights[key], f"training loss_weights {key}", minimum=0
        )
        for key in ("binary", "coarse", "native")
    }
    if not any(value > 0 for value in normalized_weights.values()):
        raise ValueError("at least one training loss weight must be positive")
    for key, default in (
        ("mask_fraction", 0.3),
        ("schema_corruption_fraction", 0.1),
        ("contrastive_temperature", 0.1),
    ):
        value = training.get(key, default)
        _require_finite_number(
            value,
            f"training {key}",
            minimum=0,
            maximum=1 if key != "contrastive_temperature" else None,
            minimum_inclusive=False if key == "contrastive_temperature" else True,
        )
    objective_weights = {
        key: _require_finite_number(
            training.get(key, 1.0),
            f"training {key}",
            minimum=0,
        )
        for key in ("reconstruction_weight", "contrastive_weight")
    }
    if workflow == "pretrain" and not any(
        value > 0 for value in objective_weights.values()
    ):
        raise ValueError(
            "at least one pretraining objective weight must be positive"
        )
    return data, model, training


def _validate_payload(payload, expected_workflow=None):
    if not isinstance(payload, dict):
        raise ValueError("configuration must be a mapping")
    _validate_strict_json(payload)

    allowed_top_level = _REQUIRED_TOP_LEVEL_KEYS | _OPTIONAL_TOP_LEVEL_KEYS
    missing = sorted(_REQUIRED_TOP_LEVEL_KEYS - payload.keys())
    unknown = sorted(payload.keys() - allowed_top_level)
    if missing:
        raise ValueError(f"configuration missing required sections: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"configuration has unknown sections: {', '.join(unknown)}")

    workflow = _require_clean_string(payload["workflow"], "workflow")
    if workflow not in WORKFLOWS:
        raise ValueError(f"workflow must be one of: {', '.join(WORKFLOWS)}")
    if expected_workflow is not None and workflow != expected_workflow:
        raise ValueError(
            f"configuration workflow {workflow!r} does not match command {expected_workflow!r}"
        )

    experiment = _require_mapping(payload, "experiment")
    schema = _require_mapping(payload, "schema")
    runtime = _require_mapping(payload, "runtime")

    _validate_keys(
        "experiment",
        experiment,
        required=("name", "output_root", "seed"),
        optional=("enforce_autodl_data_disk",),
    )
    _require_clean_string(experiment["name"], "experiment name")
    _require_clean_string(experiment["output_root"], "experiment output_root")
    if isinstance(experiment["seed"], bool) or not isinstance(experiment["seed"], int):
        raise ValueError("experiment seed must be an integer and not a boolean")
    enforcement = experiment.setdefault("enforce_autodl_data_disk", False)
    if not isinstance(enforcement, bool):
        raise ValueError("experiment enforce_autodl_data_disk must be a boolean")

    _validate_keys("schema", schema, required=("version",))
    _require_clean_string(schema["version"], "schema version")

    _validate_keys("runtime", runtime, required=("device",), optional=("execution",))
    device = _require_clean_string(runtime["device"], "runtime device")
    if device not in ("auto", "cpu", "cuda"):
        raise ValueError("runtime device must be one of: auto, cpu, cuda")

    execution = runtime.get("execution", "validate_only")
    if execution not in ("validate_only", "train"):
        raise ValueError("runtime execution must be validate_only or train")

    optional_sections = {}
    if execution == "train":
        data, model, training = _validate_train_sections(payload, workflow)
        optional_sections = {
            "data": data,
            "model": model,
            "training": training,
        }
    elif any(name in payload for name in _OPTIONAL_TOP_LEVEL_KEYS):
        raise ValueError(
            "data, model, and training sections require runtime execution train"
        )

    normalized = {
        "workflow": workflow,
        "experiment": experiment,
        "schema": schema,
        "runtime": runtime,
    }
    normalized.update(optional_sections)
    return normalized


@dataclass(frozen=True)
class FlowConfig:
    """Immutable validated configuration with defensive mapping accessors."""

    _serialized: str
    config_hash: str

    def __post_init__(self):
        try:
            payload = json.loads(self._serialized)
        except (TypeError, ValueError) as exc:
            raise ValueError("serialized configuration must be canonical JSON") from exc
        normalized = _validate_payload(payload)
        if canonical_json(normalized) != self._serialized:
            raise ValueError("serialized configuration must be canonical JSON")
        expected_hash = compute_config_hash(normalized)
        if self.config_hash != expected_hash:
            raise ValueError("config_hash does not match the serialized configuration")

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self._serialized)

    @property
    def workflow(self) -> str:
        return self.to_dict()["workflow"]

    @property
    def experiment(self):
        return MappingProxyType(self.to_dict()["experiment"])

    @property
    def schema(self):
        return MappingProxyType(self.to_dict()["schema"])

    @property
    def runtime(self):
        return MappingProxyType(self.to_dict()["runtime"])

    @property
    def data(self):
        payload = self.to_dict().get("data")
        return None if payload is None else MappingProxyType(payload)

    @property
    def model(self):
        payload = self.to_dict().get("model")
        return None if payload is None else MappingProxyType(payload)

    @property
    def training(self):
        payload = self.to_dict().get("training")
        return None if payload is None else MappingProxyType(payload)


def load_config(path, expected_workflow: Optional[str] = None) -> FlowConfig:
    """Load phase-one YAML without unsafe constructors, aliases, or duplicate keys."""

    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to load FlowFM YAML configuration") from exc

    class RestrictedSafeLoader(yaml.SafeLoader):
        def compose_node(self, parent, index):
            if self.check_event(yaml.AliasEvent):
                event = self.get_event()
                raise ValueError(f"YAML aliases are not allowed: *{event.anchor}")
            return super().compose_node(parent, index)

        def construct_mapping(self, node, deep=False):
            seen = set()
            for key_node, _ in node.value:
                key = self.construct_object(key_node, deep=deep)
                if not isinstance(key, str):
                    raise ValueError("YAML mapping keys must be strings")
                if key in seen:
                    raise ValueError(f"duplicate YAML key: {key}")
                seen.add(key)
            return super().construct_mapping(node, deep=deep)

    config_path = Path(path).expanduser()
    try:
        with config_path.open("r", encoding="utf-8") as stream:
            payload = yaml.load(stream, Loader=RestrictedSafeLoader)
    except ValueError:
        raise
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML configuration: {exc}") from exc

    normalized = _validate_payload(payload, expected_workflow=expected_workflow)
    serialized = canonical_json(normalized)
    return FlowConfig(
        _serialized=serialized,
        config_hash=compute_config_hash(normalized),
    )


__all__ = [
    "CONFIG_HASH_NAMESPACE",
    "FlowConfig",
    "WORKFLOWS",
    "compute_config_hash",
    "load_config",
]
