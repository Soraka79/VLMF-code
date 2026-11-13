"""Real NF-UNSW single-batch CPU/CUDA acceptance for SchemaFlow-FM."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from flowfm.data.nf_unsw_dataset import (
    load_nf_unsw_development_samples,
)
from flowfm.data.nf_unsw_preprocessing import (
    load_nf_unsw_preprocessor,
)
from flowfm.data.nf_unsw_tokenization import (
    build_nf_unsw_tokenizer_metadata,
)
from flowfm.data.schemaflow_batch import (
    CATEGORICAL_FEATURE_TYPE_ID,
    collate_schemaflow_samples,
)
from flowfm.data.schemaflow_registry import (
    SCHEMAFLOW_MAIN_REGISTRY,
)
from flowfm.experiment.metadata import validate_output_path
from flowfm.models.schemaflow import SchemaFlowEncoder


AUTODL_DATA_ROOT = Path("/root/autodl-tmp")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one real NF-UNSW SchemaFlow batch on CPU or CUDA; "
            "this is not a training run."
        )
    )
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--preprocessor-dir", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--train-samples", type=_positive_int, default=4)
    parser.add_argument("--validation-samples", type=_positive_int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    return parser


def validate_data_manifest_path(
    path,
    *,
    data_root=AUTODL_DATA_ROOT,
) -> Path:
    resolved = validate_output_path(
        path,
        enforce_autodl_data_disk=True,
        data_root=data_root,
    )
    if not resolved.is_file():
        raise ValueError(
            f"data_manifest must reference an existing file: {resolved}"
        )
    return resolved


def _load_data_manifest(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid data Manifest: {path}") from exc
    if not isinstance(payload, dict) or not payload:
        raise ValueError("data Manifest must contain a non-empty JSON object")
    return payload


def _select_validation_unknowns(samples, metadata, count):
    selected = []
    scanned = 0
    chunk_size = 256
    for start in range(0, len(samples), chunk_size):
        chunk = samples[start : start + chunk_size]
        batch = collate_schemaflow_samples(chunk, metadata)
        row_has_unknown = (batch.category_ids == 1).any(dim=1)
        for index in row_has_unknown.nonzero(as_tuple=False).view(-1).tolist():
            selected.append(chunk[index])
            if len(selected) == count:
                return tuple(selected), start + len(chunk)
        scanned = start + len(chunk)
    raise RuntimeError(
        "validation split does not contain enough train-unseen categories; "
        f"requested={count}, found={len(selected)}, scanned={scanned}"
    )


def _unknown_counts_by_feature(batch) -> dict[str, int]:
    counts = {}
    first_row_ids = batch.feature_ids[0].tolist()
    for column, feature_id in enumerate(first_row_ids):
        spec = SCHEMAFLOW_MAIN_REGISTRY.by_id[feature_id]
        if spec.feature_type == "categorical":
            counts[spec.name] = int((batch.category_ids[:, column] == 1).sum().item())
    return {name: count for name, count in counts.items() if count}


def _exercise_device(batch, metadata, device_name, seed):
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    torch.manual_seed(seed)
    if device_name == "cuda":
        torch.cuda.manual_seed_all(seed)
    device = torch.device(device_name)
    device_batch = batch.to(device)
    model = SchemaFlowEncoder(metadata).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    optimizer.zero_grad(set_to_none=True)
    output = model(device_batch)
    loss = (
        output.flow_embedding.square().mean()
        + output.feature_embeddings.square().mean()
    )
    if not torch.isfinite(loss).item():
        raise RuntimeError("SchemaFlow smoke loss is not finite")
    loss.backward()
    unknown_gradient_l1 = sum(
        float(embedding.weight.grad[1].abs().sum().item())
        for embedding in model.category_embeddings.values()
        if embedding.weight.grad is not None
    )
    if unknown_gradient_l1 <= 0:
        raise RuntimeError("unknown category embeddings did not receive gradients")
    optimizer.step()
    padding_rows_zero = all(
        torch.equal(
            embedding.weight[0],
            torch.zeros_like(embedding.weight[0]),
        )
        for embedding in model.category_embeddings.values()
    )
    if not padding_rows_zero:
        raise RuntimeError("a category padding row changed after AdamW step")
    if device_name == "cuda":
        torch.cuda.synchronize()

    return {
        "device": device_name,
        "device_name": (
            torch.cuda.get_device_name(device)
            if device_name == "cuda"
            else "CPU"
        ),
        "loss": float(loss.detach().cpu().item()),
        "finite_loss": True,
        "unknown_gradient_l1": unknown_gradient_l1,
        "padding_rows_zero": padding_rows_zero,
        "model_parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
    }


def run_smoke(arguments) -> dict:
    started = time.monotonic()
    manifest_path = validate_data_manifest_path(arguments.data_manifest)
    data_manifest = _load_data_manifest(manifest_path)
    dataset = load_nf_unsw_development_samples(
        arguments.split_dir,
        arguments.preprocessor_dir,
        data_root=AUTODL_DATA_ROOT,
    )
    preprocessor = load_nf_unsw_preprocessor(
        Path(arguments.preprocessor_dir).expanduser().resolve()
        / "preprocessor.json"
    )
    metadata = build_nf_unsw_tokenizer_metadata(
        dataset,
        preprocessor,
        data_manifest=data_manifest,
    )

    train_samples = tuple(dataset.splits["train"][: arguments.train_samples])
    if len(train_samples) != arguments.train_samples:
        raise RuntimeError("train split is smaller than the requested smoke batch")
    train_batch = collate_schemaflow_samples(train_samples, metadata)
    if (train_batch.category_ids == 1).any().item():
        raise RuntimeError("train-only tokenizer produced unknown IDs for train samples")
    validation_samples, validation_scanned = _select_validation_unknowns(
        dataset.splits["val"],
        metadata,
        arguments.validation_samples,
    )
    batch = collate_schemaflow_samples(
        (*train_samples, *validation_samples),
        metadata,
    )
    categorical_positions = (
        batch.feature_type_ids == CATEGORICAL_FEATURE_TYPE_ID
    )
    unknown_count = int(
        ((batch.category_ids == 1) & categorical_positions).sum().item()
    )
    if unknown_count <= 0:
        raise RuntimeError("real smoke batch does not contain an unknown category")

    device_result = _exercise_device(
        batch,
        metadata,
        arguments.device,
        arguments.seed,
    )
    return {
        "status": "passed",
        "scope": "single_batch_no_checkpoint_no_full_training",
        "dataset_id": data_manifest.get("dataset_id"),
        "schema_version": metadata.schema_version,
        "schema_hash": metadata.schema_hash,
        "registry_version": metadata.registry_version,
        "registry_hash": metadata.registry_hash,
        "tokenizer_metadata_hash": metadata.metadata_hash,
        "data_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        "split_manifest_sha256": metadata.split_manifest_sha256,
        "preprocessor_hash": metadata.preprocessor_hash,
        "training_sample_ids_hash": metadata.training_sample_ids_hash,
        "batch": {
            "train_samples": len(train_samples),
            "validation_samples": len(validation_samples),
            "validation_rows_scanned": validation_scanned,
            "feature_count": batch.feature_count,
            "unknown_category_count": unknown_count,
            "unknown_counts_by_feature": _unknown_counts_by_feature(batch),
            "missing_value_count": int(batch.missing_mask.sum().item()),
            "schema_absent_count": int((~batch.schema_present_mask).sum().item()),
        },
        "categorical_table_sizes": {
            name: vocabulary.table_size
            for name, vocabulary in metadata.categorical_vocabularies.items()
        },
        "device_result": device_result,
        "elapsed_seconds": time.monotonic() - started,
    }


def main(argv=None) -> int:
    arguments = build_parser().parse_args(argv)
    result = run_smoke(arguments)
    print(
        "SCHEMAFLOW_SMOKE_RESULT "
        + json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
