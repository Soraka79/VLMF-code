"""Exact lookup for versioned dataset feature schemas."""

from flowfm.data.schemas.nf_bot_iot_v2 import NF_BOT_IOT_V2_SCHEMA
from flowfm.data.schemas.nf_cse_cic_ids2018_v2 import NF_CSE_CIC_IDS2018_V2_SCHEMA
from flowfm.data.schemas.nf_ton_iot_v2 import NF_TON_IOT_V2_SCHEMA
from flowfm.data.schemas.nf_unsw_nb15_v2 import NF_UNSW_NB15_V2_SCHEMA


_SCHEMAS = {
    (schema.dataset_id, schema.version): schema
    for schema in (
        NF_UNSW_NB15_V2_SCHEMA,
        NF_TON_IOT_V2_SCHEMA,
        NF_CSE_CIC_IDS2018_V2_SCHEMA,
        NF_BOT_IOT_V2_SCHEMA,
    )
}


def get_dataset_schema(dataset_id: str, schema_version: str):
    try:
        return _SCHEMAS[(dataset_id, schema_version)]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "unregistered dataset Schema: "
            f"dataset_id={dataset_id!r}, schema_version={schema_version!r}"
        ) from exc


__all__ = ["get_dataset_schema"]
