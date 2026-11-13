"""Raw and model-visible schema for the official NF-ToN-IoT-v2 dataset."""

from flowfm.data.schemas.nf_unsw_nb15_v2 import (
    MODEL_FEATURES,
    RAW_FEATURES,
    TRACE_FEATURE_NAMES,
    DatasetFeatureSchema,
)


NF_TON_IOT_V2_SCHEMA = DatasetFeatureSchema(
    dataset_id="NF-ToN-IoT-v2",
    version="nf-ton-iot-v2-model-41-v1",
    raw_features=RAW_FEATURES,
    model_features=MODEL_FEATURES,
    trace_feature_names=TRACE_FEATURE_NAMES,
)


__all__ = ["NF_TON_IOT_V2_SCHEMA"]
