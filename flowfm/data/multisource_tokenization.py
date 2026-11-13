"""Train-only tokenizer metadata for multi-dataset pretraining."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

from flowfm.data.contracts import FlowSample
from flowfm.data.identity import stable_hash
from flowfm.data.schemaflow_registry import (
    DEFAULT_MAX_KNOWN_CATEGORIES,
    PORT_BUCKET_POLICY_VERSION,
    CanonicalFeatureRegistry,
    TokenizerMetadata,
    build_port_vocabulary,
    fit_categorical_vocabulary,
)


_TRAINING_IDS_NAMESPACE = "schemaflow-multisource-training-ids-v1"
_SCHEMA_NAMESPACE = "schemaflow-multisource-schema-v1"
_PREPROCESSOR_NAMESPACE = "schemaflow-multisource-preprocessors-v1"


def build_multisource_tokenizer_metadata(
    source_samples: Mapping[str, Sequence[FlowSample]],
    source_tokenizers: Mapping[str, TokenizerMetadata],
    *,
    registry: CanonicalFeatureRegistry,
    schema_version: str,
    data_manifest: Mapping,
    split_manifest_sha256: str,
    max_known_categories: int = DEFAULT_MAX_KNOWN_CATEGORIES,
) -> TokenizerMetadata:
    """Merge category metadata using source training samples and nothing else."""
    if not isinstance(source_samples, Mapping) or len(source_samples) < 2:
        raise ValueError("multi-source tokenization requires at least two sources")
    if set(source_samples) != set(source_tokenizers):
        raise ValueError("multi-source sample and tokenizer sources must match")
    if not isinstance(registry, CanonicalFeatureRegistry):
        raise TypeError("registry must be a CanonicalFeatureRegistry")
    if not isinstance(schema_version, str) or not schema_version:
        raise ValueError("multi-source schema_version must be non-empty")
    if not isinstance(data_manifest, Mapping) or not data_manifest:
        raise ValueError("multi-source data_manifest must be non-empty")
    if not isinstance(split_manifest_sha256, str) or not split_manifest_sha256:
        raise ValueError("multi-source split hash must be non-empty")
    if type(max_known_categories) is not int or max_known_categories <= 0:
        raise ValueError("max_known_categories must be positive")

    categorical_specs = registry.categorical_model_features
    counts = {spec.name: Counter() for spec in categorical_specs}
    combined_ids = []
    source_schema_records = []
    source_preprocessor_records = []
    total_rows = 0

    for dataset_id in sorted(source_samples):
        tokenizer = source_tokenizers[dataset_id]
        if not isinstance(tokenizer, TokenizerMetadata):
            raise TypeError("source tokenizers must contain TokenizerMetadata")
        if (
            tokenizer.registry_version != registry.version
            or tokenizer.registry_hash != registry.registry_hash
        ):
            raise ValueError("source tokenizer does not match canonical registry")
        if tokenizer.data_manifest.get("dataset_id") != dataset_id:
            raise ValueError("source tokenizer Manifest does not match dataset_id")
        samples = tuple(source_samples[dataset_id])
        if not samples:
            raise ValueError("multi-source training sources must be non-empty")
        source_schema_records.append(
            {
                "dataset_id": dataset_id,
                "schema_version": tokenizer.schema_version,
                "schema_hash": tokenizer.schema_hash,
            }
        )
        source_preprocessor_records.append(
            {
                "dataset_id": dataset_id,
                "preprocessor_hash": tokenizer.preprocessor_hash,
                "training_sample_ids_hash": tokenizer.training_sample_ids_hash,
            }
        )
        total_rows += len(samples)
        for sample in samples:
            if not isinstance(sample, FlowSample):
                raise TypeError("multi-source training values must be FlowSample")
            if sample.dataset_id != dataset_id:
                raise ValueError("training sample dataset_id does not match source")
            combined_ids.append(
                {"dataset_id": dataset_id, "sample_id": sample.sample_id}
            )
            positions = {
                name: index for index, name in enumerate(sample.feature_ids)
            }
            if len(positions) != len(sample.feature_ids):
                raise ValueError("training sample feature IDs must be unique")
            for spec in categorical_specs:
                index = positions.get(spec.name)
                if index is None or sample.missing_mask[index]:
                    continue
                if not sample.categorical_mask[index]:
                    raise ValueError(f"feature type mismatch for {spec.name}")
                token = sample.category_tokens[index]
                if not isinstance(token, str) or not token:
                    raise ValueError(f"observed category token is invalid for {spec.name}")
                counts[spec.name][token] += 1

    identity_pairs = {
        (item["dataset_id"], item["sample_id"]) for item in combined_ids
    }
    if len(identity_pairs) != len(combined_ids):
        raise ValueError("dataset/sample identity is duplicated across sources")
    training_ids_hash = stable_hash(combined_ids, _TRAINING_IDS_NAMESPACE)
    vocabularies = {}
    for spec in categorical_specs:
        arguments = {
            "training_row_count": total_rows,
            "training_sample_ids_hash": training_ids_hash,
        }
        if spec.value_policy == PORT_BUCKET_POLICY_VERSION:
            vocabulary = build_port_vocabulary(
                spec, counts[spec.name], **arguments
            )
        else:
            vocabulary = fit_categorical_vocabulary(
                spec,
                counts[spec.name],
                max_known_categories=max_known_categories,
                **arguments,
            )
        vocabularies[spec.name] = vocabulary

    return TokenizerMetadata(
        registry_version=registry.version,
        registry_hash=registry.registry_hash,
        schema_version=schema_version,
        schema_hash=stable_hash(source_schema_records, _SCHEMA_NAMESPACE),
        categorical_vocabularies=vocabularies,
        data_manifest=dict(data_manifest),
        split_manifest_sha256=split_manifest_sha256,
        preprocessor_hash=stable_hash(
            source_preprocessor_records, _PREPROCESSOR_NAMESPACE
        ),
        training_sample_ids_hash=training_ids_hash,
    )


__all__ = ["build_multisource_tokenizer_metadata"]
