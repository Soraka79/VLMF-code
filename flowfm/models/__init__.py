"""Model registration for SchemaFlow-FM."""

from .ft_transformer import (
    FTTransformerClassifier,
    FTTransformerClassifierOutput,
)
from .schema_bigru import (
    SchemaBiGRUClassifier,
    SchemaBiGRUClassifierOutput,
    SchemaBiGRUEncoder,
    SchemaBiGRUEncoderOutput,
)
from .schemaflow import (
    SchemaFlowEncoder,
    SchemaFlowEncoderOutput,
)
from .supervised import (
    ResidualAdapter,
    SchemaFlowSupervisedLoss,
    SchemaFlowSupervisedModel,
    SchemaFlowSupervisedOutput,
    compute_supervised_loss,
)

__all__ = [
    "ResidualAdapter",
    "FTTransformerClassifier",
    "FTTransformerClassifierOutput",
    "SchemaBiGRUClassifier",
    "SchemaBiGRUClassifierOutput",
    "SchemaBiGRUEncoder",
    "SchemaBiGRUEncoderOutput",
    "SchemaFlowEncoder",
    "SchemaFlowEncoderOutput",
    "SchemaFlowSupervisedLoss",
    "SchemaFlowSupervisedModel",
    "SchemaFlowSupervisedOutput",
    "compute_supervised_loss",
]
