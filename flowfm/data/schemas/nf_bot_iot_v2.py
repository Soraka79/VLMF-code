"""Raw and model-visible schema for the official NF-BoT-IoT-v2 dataset."""

from flowfm.data.schemas.nf_unsw_nb15_v2 import (
    MODEL_FEATURES,
    RAW_FEATURES,
    TRACE_FEATURE_NAMES,
    DatasetFeatureSchema,
)


NF_BOT_IOT_V2_SCHEMA = DatasetFeatureSchema(
    dataset_id="NF-BoT-IoT-v2",
    version="nf-bot-iot-v2-model-41-v1",
    raw_features=RAW_FEATURES,
    model_features=MODEL_FEATURES,
    trace_feature_names=TRACE_FEATURE_NAMES,
)


__all__ = ["NF_BOT_IOT_V2_SCHEMA"]
