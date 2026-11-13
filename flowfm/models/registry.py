"""Lazy model registry, including the preserved legacy baseline."""

from typing import Any, Callable, Dict, Tuple


ModelFactory = Callable[..., Any]


class ModelRegistry:
    def __init__(self) -> None:
        self._factories: Dict[str, ModelFactory] = {}

    def register(self, name: str, factory: ModelFactory) -> None:
        if name in self._factories:
            raise ValueError("model already registered: {}".format(name))
        self._factories[name] = factory

    def names(self) -> Tuple[str, ...]:
        return tuple(sorted(self._factories))

    def create(self, name: str, *args: Any, **kwargs: Any) -> Any:
        try:
            factory = self._factories[name]
        except KeyError as exc:
            raise KeyError("unknown model: {}".format(name)) from exc
        return factory(*args, **kwargs)


def _create_legacy_model(*args: Any, **kwargs: Any) -> Any:
    from shared_rescae_bigru import ResCAE_BiGRU

    return ResCAE_BiGRU(*args, **kwargs)


def _create_schema_bigru_model(*args: Any, **kwargs: Any) -> Any:
    from flowfm.models.schema_bigru import SchemaBiGRUClassifier

    return SchemaBiGRUClassifier(*args, **kwargs)


def _create_ft_transformer_model(*args: Any, **kwargs: Any) -> Any:
    from flowfm.models.ft_transformer import FTTransformerClassifier

    return FTTransformerClassifier(*args, **kwargs)


def _create_schemaflow_model(*args: Any, **kwargs: Any) -> Any:
    from flowfm.models.schemaflow import SchemaFlowEncoder

    return SchemaFlowEncoder(*args, **kwargs)


def _create_schemaflow_supervised_model(*args: Any, **kwargs: Any) -> Any:
    from flowfm.models.supervised import SchemaFlowSupervisedModel

    return SchemaFlowSupervisedModel(*args, **kwargs)


MODEL_REGISTRY = ModelRegistry()
MODEL_REGISTRY.register("rescae_bigru_legacy", _create_legacy_model)
MODEL_REGISTRY.register("schema_bigru", _create_schema_bigru_model)
MODEL_REGISTRY.register("ft_transformer_baseline", _create_ft_transformer_model)
MODEL_REGISTRY.register("schemaflow_fm", _create_schemaflow_model)
MODEL_REGISTRY.register(
    "schemaflow_fm_supervised",
    _create_schemaflow_supervised_model,
)
