"""NF-UNSW train-only tokenizer metadata construction."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

from flowfm.data.identity import stable_hash
from flowfm.data.nf_unsw_dataset import NFUNSWDevelopmentSamples
from flowfm.data.nf_unsw_preprocessing import (
    TRAINING_SAMPLE_IDS_NAMESPACE,
    NFUNSWTrainOnlyPreprocessor,
)
from flowfm.data.schemaflow_registry import (
    DEFAULT_MAX_KNOWN_CATEGORIES,
    PORT_BUCKET_POLICY_VERSION,
    SCHEMAFLOW_MAIN_REGISTRY,
    TokenizerMetadata,
    build_port_vocabulary,
    fit_categorical_vocabulary,
)
from flowfm.data.schemas.nf_unsw_nb15_v2 import (
    NF_UNSW_NB15_V2_SCHEMA,
)


def _category_cap(
    configured: int | Mapping[str, int],
    feature_name: str,
) -> int:
    if type(configured) is int:
        value = configured
    elif isinstance(configured, Mapping):
        value = configured.get(
            feature_name,
            DEFAULT_MAX_KNOWN_CATEGORIES,
        )
    else:
        raise TypeError(
            "max_known_categories must be an integer or mapping"
        )
    if type(value) is not int or value <= 0:
        raise ValueError(
            f"category cap for {feature_name} must be a positive integer"
        )
    return value


def build_nf_unsw_tokenizer_metadata(
    dataset: NFUNSWDevelopmentSamples,
    preprocessor: NFUNSWTrainOnlyPreprocessor,
    *,
    data_manifest: Mapping[str, Any],
    max_known_categories: int | Mapping[str, int] = (
        DEFAULT_MAX_KNOWN_CATEGORIES
    ),
) -> TokenizerMetadata:
    """Fit canonical category metadata from NF-UNSW training samples only."""

    if not isinstance(dataset, NFUNSWDevelopmentSamples):
        raise TypeError(
            "dataset must be NFUNSWDevelopmentSamples"
        )
    if not isinstance(
        preprocessor, NFUNSWTrainOnlyPreprocessor
    ):
        raise TypeError(
            "preprocessor must be NFUNSWTrainOnlyPreprocessor"
        )
    if dataset.preprocessor_hash != preprocessor.preprocessor_hash:
        raise ValueError(
            "dataset preprocessor_hash does not match preprocessor"
        )
    if (
        dataset.schema_version
        != NF_UNSW_NB15_V2_SCHEMA.version
    ):
        raise ValueError(
            "dataset schema_version does not match NF-UNSW schema"
        )
    if (
        not isinstance(data_manifest, Mapping)
        or not data_manifest
    ):
        raise ValueError(
            "data_manifest must be a non-empty mapping"
        )
    if (
        data_manifest.get("dataset_id")
        != NF_UNSW_NB15_V2_SCHEMA.dataset_id
    ):
        raise ValueError(
            "data_manifest dataset_id does not match NF-UNSW"
        )

    training_samples = dataset.splits["train"]
    if len(training_samples) != preprocessor.training_row_count:
        raise ValueError(
            "training sample count does not match preprocessor fit"
        )
    training_sample_ids_hash = stable_hash(
        [sample.sample_id for sample in training_samples],
        TRAINING_SAMPLE_IDS_NAMESPACE,
    )
    if (
        training_sample_ids_hash
        != dataset.training_sample_ids_hash
    ):
        raise ValueError(
            "training_sample_ids_hash does not match train split"
        )

    categorical_specs = (
        SCHEMAFLOW_MAIN_REGISTRY.categorical_model_features
    )
    counts = {
        spec.name: Counter() for spec in categorical_specs
    }
    source_ids = {
        feature_name: {
            token: category_id
            for category_id, token in enumerate(
                vocabulary,
                start=2,
            )
        }
        for feature_name, vocabulary
        in preprocessor.categorical_vocabularies.items()
    }

    for sample in training_samples:
        if (
            sample.dataset_id
            != NF_UNSW_NB15_V2_SCHEMA.dataset_id
        ):
            raise ValueError(
                "training sample dataset_id does not match NF-UNSW"
            )
        if len(set(sample.feature_ids)) != len(
            sample.feature_ids
        ):
            raise ValueError(
                "training sample feature_ids must be unique"
            )
        positions = {
            name: index
            for index, name in enumerate(sample.feature_ids)
        }
        for spec in categorical_specs:
            try:
                index = positions[spec.name]
            except KeyError as exc:
                raise ValueError(
                    f"training sample is missing {spec.name}"
                ) from exc
            if not sample.categorical_mask[index]:
                raise ValueError(
                    f"training sample type mismatch for {spec.name}"
                )
            token = sample.category_tokens[index]
            if sample.missing_mask[index]:
                continue
            expected_local_id = source_ids[spec.name].get(
                token,
                1,
            )
            if expected_local_id == 1:
                raise ValueError(
                    f"training category for {spec.name} "
                    "is absent from the train-only preprocessor"
                )
            if sample.features[index] != expected_local_id:
                raise ValueError(
                    f"training category ID mismatch for {spec.name}"
                )
            counts[spec.name][token] += 1

    vocabularies = {}
    for spec in categorical_specs:
        arguments = {
            "training_row_count": len(training_samples),
            "training_sample_ids_hash": (
                training_sample_ids_hash
            ),
        }
        if spec.value_policy == PORT_BUCKET_POLICY_VERSION:
            vocabulary = build_port_vocabulary(
                spec,
                counts[spec.name],
                **arguments,
            )
        else:
            vocabulary = fit_categorical_vocabulary(
                spec,
                counts[spec.name],
                max_known_categories=_category_cap(
                    max_known_categories,
                    spec.name,
                ),
                **arguments,
            )
        vocabularies[spec.name] = vocabulary

    return TokenizerMetadata(
        registry_version=SCHEMAFLOW_MAIN_REGISTRY.version,
        registry_hash=SCHEMAFLOW_MAIN_REGISTRY.registry_hash,
        schema_version=NF_UNSW_NB15_V2_SCHEMA.version,
        schema_hash=NF_UNSW_NB15_V2_SCHEMA.schema_hash,
        categorical_vocabularies=vocabularies,
        data_manifest=dict(data_manifest),
        split_manifest_sha256=(
            dataset.split_manifest_sha256
        ),
        preprocessor_hash=preprocessor.preprocessor_hash,
        training_sample_ids_hash=training_sample_ids_hash,
    )


__all__ = ["build_nf_unsw_tokenizer_metadata"]
