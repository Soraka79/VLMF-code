"""Versioned, non-train-fitted label spaces for supervised FlowFM tasks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import torch

from flowfm.data.contracts import FlowSample
from flowfm.data.identity import stable_hash
from flowfm.data.schemas.nf_bot_iot_v2 import NF_BOT_IOT_V2_SCHEMA
from flowfm.data.schemas.nf_cse_cic_ids2018_v2 import NF_CSE_CIC_IDS2018_V2_SCHEMA
from flowfm.data.schemas.nf_ton_iot_v2 import NF_TON_IOT_V2_SCHEMA
from flowfm.data.schemas.nf_unsw_nb15_v2 import NF_UNSW_NB15_V2_SCHEMA


LABEL_SPACE_FORMAT_VERSION = 1
LABEL_SPACE_HASH_NAMESPACE = "flowfm-dataset-label-space-v1"
SHARED_BINARY_LABELS = ("benign", "attack")
SHARED_COARSE_LABELS = (
    "benign",
    "dos",
    "scan",
    "brute_force",
    "exploit",
    "botnet",
    "backdoor",
    "malware",
)


def _clean_labels(values: Sequence[str], field_name: str) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{field_name} must be a non-empty sequence")
    normalized = tuple(values)
    if not all(
        isinstance(value, str) and value and value.strip() == value
        for value in normalized
    ):
        raise ValueError(f"{field_name} must contain clean non-empty strings")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} must not contain duplicates")
    return normalized


@dataclass(frozen=True)
class DatasetLabelSpace:
    dataset_id: str
    schema_version: str
    mapping_version: str
    binary_labels: tuple[str, ...]
    coarse_labels: tuple[str, ...]
    native_labels: tuple[str, ...]
    native_to_coarse: Mapping[str, str]
    label_space_hash: str = ""
    _binary_to_id: Mapping[str, int] = field(init=False, repr=False, compare=False)
    _coarse_to_id: Mapping[str, int] = field(init=False, repr=False, compare=False)
    _native_to_id: Mapping[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        for field_name in ("dataset_id", "schema_version", "mapping_version"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value or value.strip() != value:
                raise ValueError(f"{field_name} must be a clean non-empty string")
        binary = _clean_labels(self.binary_labels, "binary_labels")
        coarse = _clean_labels(self.coarse_labels, "coarse_labels")
        native = _clean_labels(self.native_labels, "native_labels")
        if not isinstance(self.native_to_coarse, Mapping):
            raise TypeError("native_to_coarse must be a mapping")
        mapping = dict(self.native_to_coarse)
        if set(mapping) != set(native):
            raise ValueError("native_to_coarse keys must exactly match native_labels")
        if not all(value in coarse for value in mapping.values()):
            raise ValueError("native_to_coarse values must occur in coarse_labels")

        object.__setattr__(self, "binary_labels", binary)
        object.__setattr__(self, "coarse_labels", coarse)
        object.__setattr__(self, "native_labels", native)
        object.__setattr__(self, "native_to_coarse", MappingProxyType(mapping))
        object.__setattr__(
            self,
            "_binary_to_id",
            MappingProxyType({value: index for index, value in enumerate(binary)}),
        )
        object.__setattr__(
            self,
            "_coarse_to_id",
            MappingProxyType({value: index for index, value in enumerate(coarse)}),
        )
        object.__setattr__(
            self,
            "_native_to_id",
            MappingProxyType({value: index for index, value in enumerate(native)}),
        )
        expected_hash = stable_hash(self._hash_payload(), LABEL_SPACE_HASH_NAMESPACE)
        if self.label_space_hash and self.label_space_hash != expected_hash:
            raise ValueError("label_space_hash does not match label space")
        object.__setattr__(self, "label_space_hash", expected_hash)

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "format_version": LABEL_SPACE_FORMAT_VERSION,
            "dataset_id": self.dataset_id,
            "schema_version": self.schema_version,
            "mapping_version": self.mapping_version,
            "binary_labels": list(self.binary_labels),
            "coarse_labels": list(self.coarse_labels),
            "native_labels": list(self.native_labels),
            "native_to_coarse": dict(sorted(self.native_to_coarse.items())),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._hash_payload()
        payload["label_space_hash"] = self.label_space_hash
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DatasetLabelSpace":
        required = {
            "format_version",
            "dataset_id",
            "schema_version",
            "mapping_version",
            "binary_labels",
            "coarse_labels",
            "native_labels",
            "native_to_coarse",
            "label_space_hash",
        }
        if not isinstance(payload, Mapping) or set(payload) != required:
            raise ValueError("label space payload has invalid fields")
        if payload["format_version"] != LABEL_SPACE_FORMAT_VERSION:
            raise ValueError("unsupported label space format version")
        return cls(
            dataset_id=payload["dataset_id"],
            schema_version=payload["schema_version"],
            mapping_version=payload["mapping_version"],
            binary_labels=tuple(payload["binary_labels"]),
            coarse_labels=tuple(payload["coarse_labels"]),
            native_labels=tuple(payload["native_labels"]),
            native_to_coarse=payload["native_to_coarse"],
            label_space_hash=payload["label_space_hash"],
        )


@dataclass(frozen=True)
class SupervisedTargets:
    binary: torch.Tensor
    coarse: torch.Tensor
    native: torch.Tensor
    dataset_id: str

    def __post_init__(self) -> None:
        tensors = (self.binary, self.coarse, self.native)
        if not all(isinstance(value, torch.Tensor) for value in tensors):
            raise TypeError("supervised targets must be torch tensors")
        if not all(value.dtype == torch.int64 and value.ndim == 1 for value in tensors):
            raise ValueError("supervised targets must be rank-1 int64 tensors")
        if len({tuple(value.shape) for value in tensors}) != 1:
            raise ValueError("supervised targets must have identical shapes")
        if len({value.device for value in tensors}) != 1:
            raise ValueError("supervised targets must share one device")
        if not isinstance(self.dataset_id, str) or not self.dataset_id:
            raise ValueError("dataset_id must be a non-empty string")

    def to(self, device) -> "SupervisedTargets":
        return SupervisedTargets(
            binary=self.binary.to(device),
            coarse=self.coarse.to(device),
            native=self.native.to(device),
            dataset_id=self.dataset_id,
        )


def encode_supervised_targets(
    samples: Sequence[FlowSample],
    label_space: DatasetLabelSpace,
    *,
    device="cpu",
) -> SupervisedTargets:
    if not isinstance(samples, (list, tuple)) or not samples:
        raise ValueError("samples must be a non-empty sequence")
    if not isinstance(label_space, DatasetLabelSpace):
        raise TypeError("label_space must be DatasetLabelSpace")
    binary_ids = []
    coarse_ids = []
    native_ids = []
    for sample in samples:
        if not isinstance(sample, FlowSample):
            raise TypeError("samples must contain FlowSample values")
        if sample.dataset_id != label_space.dataset_id:
            raise ValueError("sample dataset does not match label space")
        try:
            native_id = label_space._native_to_id[sample.native_label]
        except KeyError as exc:
            raise ValueError(f"unknown native label: {sample.native_label!r}") from exc
        try:
            coarse_id = label_space._coarse_to_id[sample.coarse_label]
        except KeyError as exc:
            raise ValueError(f"unknown coarse label: {sample.coarse_label!r}") from exc
        try:
            binary_id = label_space._binary_to_id[sample.binary_label]
        except KeyError as exc:
            raise ValueError(f"unknown binary label: {sample.binary_label!r}") from exc
        expected_coarse = label_space.native_to_coarse[sample.native_label]
        expected_binary = "benign" if expected_coarse == "benign" else "attack"
        if sample.coarse_label != expected_coarse or (
            "benign" in label_space._binary_to_id
            and "attack" in label_space._binary_to_id
            and sample.binary_label != expected_binary
        ):
            raise ValueError("sample label hierarchy is inconsistent")
        binary_ids.append(binary_id)
        coarse_ids.append(coarse_id)
        native_ids.append(native_id)
    target_device = torch.device(device)
    return SupervisedTargets(
        binary=torch.tensor(binary_ids, dtype=torch.int64, device=target_device),
        coarse=torch.tensor(coarse_ids, dtype=torch.int64, device=target_device),
        native=torch.tensor(native_ids, dtype=torch.int64, device=target_device),
        dataset_id=label_space.dataset_id,
    )


NF_UNSW_LABEL_SPACE = DatasetLabelSpace(
    dataset_id=NF_UNSW_NB15_V2_SCHEMA.dataset_id,
    schema_version=NF_UNSW_NB15_V2_SCHEMA.version,
    mapping_version="nf-unsw-nb15-v2-label-space-v2",
    binary_labels=SHARED_BINARY_LABELS,
    coarse_labels=SHARED_COARSE_LABELS,
    native_labels=(
        "Benign",
        "Generic",
        "Exploits",
        "Fuzzers",
        "DoS",
        "Reconnaissance",
        "Analysis",
        "Backdoor",
        "Shellcode",
        "Worms",
    ),
    native_to_coarse={
        "Benign": "benign",
        "Generic": "exploit",
        "Exploits": "exploit",
        "Fuzzers": "exploit",
        "DoS": "dos",
        "Reconnaissance": "scan",
        "Analysis": "exploit",
        "Backdoor": "backdoor",
        "Shellcode": "exploit",
        "Worms": "malware",
    },
)


def filtered_shared_coarse_label_space(
    label_space: DatasetLabelSpace,
    supported_labels: Sequence[str],
    *,
    mapping_version: str,
) -> DatasetLabelSpace:
    """Return a versioned task view that preserves shared-coarse label order."""

    if not isinstance(label_space, DatasetLabelSpace):
        raise TypeError("label_space must be a DatasetLabelSpace")
    supported = _clean_labels(supported_labels, "supported_labels")
    expected = tuple(
        label for label in label_space.coarse_labels if label in set(supported)
    )
    if supported != expected:
        raise ValueError(
            "supported_labels must follow the source shared-coarse label order"
        )
    if set(label_space.native_to_coarse.values()) - set(supported):
        raise ValueError("filtered task omits a coarse label used by native labels")
    return DatasetLabelSpace(
        dataset_id=label_space.dataset_id,
        schema_version=label_space.schema_version,
        mapping_version=mapping_version,
        binary_labels=label_space.binary_labels,
        coarse_labels=supported,
        native_labels=label_space.native_labels,
        native_to_coarse=label_space.native_to_coarse,
    )


def coarse_task_label_space(
    label_space: DatasetLabelSpace,
    supported_labels: Sequence[str],
    *,
    mapping_version: str,
) -> DatasetLabelSpace:
    """Build a coarse task after a predeclared row-domain filter.

    Native labels whose coarse parent is outside the declared task are removed.
    This is only appropriate when the task and row filter are frozen before any
    model result is observed, such as after a historical-exposure audit.
    """

    if not isinstance(label_space, DatasetLabelSpace):
        raise TypeError("label_space must be a DatasetLabelSpace")
    supported = _clean_labels(supported_labels, "supported_labels")
    expected = tuple(
        label for label in label_space.coarse_labels if label in set(supported)
    )
    if supported != expected:
        raise ValueError(
            "supported_labels must follow the source shared-coarse label order"
        )
    native_labels = tuple(
        label
        for label in label_space.native_labels
        if label_space.native_to_coarse[label] in set(supported)
    )
    if not native_labels:
        raise ValueError("coarse task must retain at least one native label")
    return DatasetLabelSpace(
        dataset_id=label_space.dataset_id,
        schema_version=label_space.schema_version,
        mapping_version=mapping_version,
        binary_labels=label_space.binary_labels,
        coarse_labels=supported,
        native_labels=native_labels,
        native_to_coarse={
            label: label_space.native_to_coarse[label]
            for label in native_labels
        },
    )


NF_TON_LABEL_SPACE = DatasetLabelSpace(
    dataset_id=NF_TON_IOT_V2_SCHEMA.dataset_id,
    schema_version=NF_TON_IOT_V2_SCHEMA.version,
    mapping_version="nf-ton-iot-v2-label-space-v1",
    binary_labels=SHARED_BINARY_LABELS,
    coarse_labels=SHARED_COARSE_LABELS,
    native_labels=(
        "Benign",
        "backdoor",
        "ddos",
        "dos",
        "injection",
        "mitm",
        "password",
        "ransomware",
        "scanning",
        "xss",
    ),
    native_to_coarse={
        "Benign": "benign",
        "backdoor": "backdoor",
        "ddos": "dos",
        "dos": "dos",
        "injection": "exploit",
        "mitm": "exploit",
        "password": "brute_force",
        "ransomware": "malware",
        "scanning": "scan",
        "xss": "exploit",
    },
)


NF_CSE_LABEL_SPACE = DatasetLabelSpace(
    dataset_id=NF_CSE_CIC_IDS2018_V2_SCHEMA.dataset_id,
    schema_version=NF_CSE_CIC_IDS2018_V2_SCHEMA.version,
    mapping_version="nf-cse-cic-ids2018-v2-label-space-v1",
    binary_labels=SHARED_BINARY_LABELS,
    coarse_labels=SHARED_COARSE_LABELS,
    native_labels=(
        "Benign",
        "Bot",
        "Brute Force -Web",
        "Brute Force -XSS",
        "DDOS attack-HOIC",
        "DDOS attack-LOIC-UDP",
        "DDoS attacks-LOIC-HTTP",
        "DoS attacks-GoldenEye",
        "DoS attacks-Hulk",
        "DoS attacks-SlowHTTPTest",
        "DoS attacks-Slowloris",
        "FTP-BruteForce",
        "Infilteration",
        "SQL Injection",
        "SSH-Bruteforce",
    ),
    native_to_coarse={
        "Benign": "benign",
        "Bot": "botnet",
        "Brute Force -Web": "brute_force",
        "Brute Force -XSS": "brute_force",
        "DDOS attack-HOIC": "dos",
        "DDOS attack-LOIC-UDP": "dos",
        "DDoS attacks-LOIC-HTTP": "dos",
        "DoS attacks-GoldenEye": "dos",
        "DoS attacks-Hulk": "dos",
        "DoS attacks-SlowHTTPTest": "dos",
        "DoS attacks-Slowloris": "dos",
        "FTP-BruteForce": "brute_force",
        "Infilteration": "exploit",
        "SQL Injection": "exploit",
        "SSH-Bruteforce": "brute_force",
    },
)

NF_CSE_SHARED_COARSE_LABEL_SPACE = filtered_shared_coarse_label_space(
    NF_CSE_LABEL_SPACE,
    tuple(
        label
        for label in SHARED_COARSE_LABELS
        if label in set(NF_CSE_LABEL_SPACE.native_to_coarse.values())
    ),
    mapping_version="nf-cse-cic-ids2018-v2-shared-coarse-five-class-v1",
)


NF_BOT_LABEL_SPACE = DatasetLabelSpace(
    dataset_id=NF_BOT_IOT_V2_SCHEMA.dataset_id,
    schema_version=NF_BOT_IOT_V2_SCHEMA.version,
    mapping_version="nf-bot-iot-v2-label-space-v1",
    binary_labels=SHARED_BINARY_LABELS,
    coarse_labels=SHARED_COARSE_LABELS,
    native_labels=("Benign", "DDoS", "DoS", "Reconnaissance", "Theft"),
    native_to_coarse={
        "Benign": "benign",
        "DDoS": "dos",
        "DoS": "dos",
        "Reconnaissance": "scan",
        "Theft": "malware",
    },
)


__all__ = [
    "DatasetLabelSpace",
    "LABEL_SPACE_FORMAT_VERSION",
    "NF_BOT_LABEL_SPACE",
    "NF_CSE_LABEL_SPACE",
    "NF_CSE_SHARED_COARSE_LABEL_SPACE",
    "NF_TON_LABEL_SPACE",
    "NF_UNSW_LABEL_SPACE",
    "SHARED_BINARY_LABELS",
    "SHARED_COARSE_LABELS",
    "SupervisedTargets",
    "coarse_task_label_space",
    "encode_supervised_targets",
    "filtered_shared_coarse_label_space",
]
