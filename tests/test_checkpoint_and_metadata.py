import json
import os
import pickle
import random
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from flowfm.data.schemaflow_registry import (
    CanonicalFeatureRegistry,
    CanonicalFeatureSpec,
    TokenizerMetadata,
    fit_categorical_vocabulary,
)
from flowfm.data.labels import DatasetLabelSpace
from flowfm.config import compute_config_hash
from flowfm.experiment.metadata import (
    capture_provenance,
    validate_output_path,
    write_json_snapshot,
)
from flowfm.training.checkpoints import (
    CHECKPOINT_FORMAT_VERSION,
    load_checkpoint,
    save_checkpoint,
)


_TEST_GIT_COMMIT = "a" * 40


def _model_data_metadata(
    schema_version="checkpoint-test-schema-v1",
    manifest_hash="data-manifest-hash",
):
    protocol = CanonicalFeatureSpec(
        feature_id=3,
        name="PROTOCOL",
        feature_type="categorical",
        value_policy="vocabulary",
    )
    registry = CanonicalFeatureRegistry(
        "checkpoint-test-registry-v1",
        (protocol,),
    )
    vocabulary = fit_categorical_vocabulary(
        protocol,
        {"i:6": 8, "i:17": 4},
        training_row_count=12,
        training_sample_ids_hash="train-sample-hash",
        max_known_categories=1,
    )
    tokenizer = TokenizerMetadata(
        registry_version=registry.version,
        registry_hash=registry.registry_hash,
        schema_version=schema_version,
        schema_hash="schema-hash",
        categorical_vocabularies={"PROTOCOL": vocabulary},
        data_manifest={
            "dataset": "unit-test",
            "manifest_sha256": manifest_hash,
        },
        split_manifest_sha256="split-manifest-hash",
        preprocessor_hash="preprocessor-hash",
        training_sample_ids_hash="train-sample-hash",
    )
    label_space = DatasetLabelSpace(
        dataset_id="unit-test",
        schema_version=schema_version,
        mapping_version="checkpoint-labels-v1",
        binary_labels=("benign", "attack"),
        coarse_labels=("benign", "dos"),
        native_labels=("Benign", "DoS"),
        native_to_coarse={"Benign": "benign", "DoS": "dos"},
    )
    return {
        "format_version": 2,
        "feature_registry": registry.to_dict(),
        "tokenizer_metadata": tokenizer.to_dict(),
        "categorical_table_sizes": {
            "PROTOCOL": vocabulary.table_size,
        },
        "label_spaces": {"unit-test": label_space.to_dict()},
    }


def _model_data_metadata_v2(schema_version="checkpoint-test-schema-v1"):
    return _model_data_metadata(schema_version)


class _Stateful:
    def __init__(self, state):
        self.state = state
        self.loaded = []

    def state_dict(self):
        return self.state

    def load_state_dict(self, state):
        self.loaded.append(state)
        self.state = state


class _FailOnceStateful(_Stateful):
    def __init__(self, state):
        super().__init__(state)
        self.failures_remaining = 1

    def load_state_dict(self, state):
        if self.failures_remaining:
            self.failures_remaining -= 1
            raise RuntimeError("injected load failure")
        super().load_state_dict(state)


class _FakeDevice:
    def __init__(self, device_type="cpu"):
        self.type = device_type


class _FakeTensor:
    dtype = "uint8"

    def __init__(self, value, device_type="cpu"):
        self.value = value
        self.device = _FakeDevice(device_type)

    def dim(self):
        return 1

    def clone(self):
        return _FakeTensor(self.value, self.device.type)

    def cpu(self):
        return _FakeTensor(self.value, "cpu")

    def __eq__(self, other):
        return (
            isinstance(other, _FakeTensor)
            and self.value == other.value
            and self.device.type == other.device.type
        )


class _FakeCuda:
    restored_rng = None

    @staticmethod
    def is_available():
        return False

    @staticmethod
    def device_count():
        return 0

    @staticmethod
    def set_rng_state_all(state):
        _FakeCuda.restored_rng = state

    @staticmethod
    def get_rng_state_all():
        return [_FakeTensor("current-cuda-rng")]


class _FakeTorch:
    cuda = _FakeCuda()
    uint8 = "uint8"
    current_rng = _FakeTensor("fake-torch-rng")
    restored_rng = None

    @staticmethod
    def save(payload, handle):
        pickle.dump(payload, handle)

    @staticmethod
    def load(path, map_location="cpu"):
        del map_location
        with open(path, "rb") as handle:
            return pickle.load(handle)

    @staticmethod
    def get_rng_state():
        return _FakeTorch.current_rng.clone()

    @staticmethod
    def set_rng_state(state):
        _FakeTorch.restored_rng = state
        _FakeTorch.current_rng = state.clone()

    @staticmethod
    def is_tensor(value):
        return isinstance(value, _FakeTensor)


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self._numpy_patcher = mock.patch(
            "flowfm.training.checkpoints._optional_numpy",
            return_value=None,
        )
        self._numpy_patcher.start()
        self.addCleanup(self._numpy_patcher.stop)
        _FakeTorch.current_rng = _FakeTensor("fake-torch-rng")
        _FakeTorch.restored_rng = None

    def _save_payload(self, path, **overrides):
        payload = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "model_data_metadata": _model_data_metadata("ids-v2"),
            "model_state": {"weight": 3},
            "adapter_state": {"adapter": 1},
            "optimizer_state": {"momentum": 7},
            "scheduler_state": {"step": 4},
            "epoch": 4,
            "global_step": 24,
            "git_commit": _TEST_GIT_COMMIT,
            "schema_version": "ids-v2",
            "seed": 91,
            "config": {"batch_size": 32},
            "extra_state": {"best_f1": 0.8},
            "python_random_state": random.getstate(),
            "numpy_random_state": None,
            "torch_cpu_rng_state": _FakeTensor("saved-rng"),
            "torch_cuda_rng_states": None,
        }
        payload.update(overrides)
        with Path(path).open("wb") as handle:
            pickle.dump(payload, handle)
        return payload

    def test_checkpoint_roundtrip_includes_optional_state_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoints" / "epoch-4.pt"
            model = _Stateful({"weight": 3})
            optimizer = _Stateful({"momentum": 7})
            scheduler = _Stateful({"step": 4})
            adapter = _Stateful({"normalizer": [1, 2]})
            config = {"model": {"width": 16}, "batch_size": 32}

            with mock.patch(
                "flowfm.training.checkpoints._require_torch",
                return_value=_FakeTorch,
            ):
                save_checkpoint(
                    path,
                    model,
                    epoch=4,
                    global_step=24,
                    git_commit=_TEST_GIT_COMMIT,
                    schema_version="ids-v2",
                    seed=91,
                    config=config,
                    model_data_metadata=_model_data_metadata("ids-v2"),
                    optimizer=optimizer,
                    scheduler=scheduler,
                    adapter=adapter,
                    extra_state={"best_f1": 0.8},
                )
                model.state = {"weight": -1}
                optimizer.state = {}
                scheduler.state = {}
                adapter.state = {}
                result = load_checkpoint(
                    path,
                    model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    adapter=adapter,
                    expected_model_data_metadata=_model_data_metadata("ids-v2"),
                )

            self.assertEqual(model.state, {"weight": 3})
            self.assertEqual(optimizer.state, {"momentum": 7})
            self.assertEqual(scheduler.state, {"step": 4})
            self.assertEqual(adapter.state, {"normalizer": [1, 2]})
            self.assertEqual(result["epoch"], 4)
            self.assertEqual(result["schema_version"], "ids-v2")
            self.assertEqual(result["config"], config)
            self.assertEqual(result["extra_state"], {"best_f1": 0.8})
            self.assertEqual(_FakeTorch.restored_rng, _FakeTensor("fake-torch-rng"))

    def test_checkpoint_roundtrip_requires_complete_model_data_metadata(self):
        metadata = _model_data_metadata()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            model = _Stateful({"weight": _FakeTensor("model")})
            with mock.patch(
                "flowfm.training.checkpoints._require_torch",
                return_value=_FakeTorch,
            ):
                save_checkpoint(
                    path,
                    model,
                    epoch=1,
                    global_step=12,
                    git_commit=_TEST_GIT_COMMIT,
                    schema_version="checkpoint-test-schema-v1",
                    seed=7,
                    config={},
                    model_data_metadata=metadata,
                )
                payload = load_checkpoint(
                    path,
                    model,
                    expected_model_data_metadata=metadata,
                )

        self.assertEqual(payload["model_data_metadata"], metadata)
        tokenizer = payload["model_data_metadata"]["tokenizer_metadata"]
        vocabulary = tokenizer["categorical_vocabularies"]["PROTOCOL"]
        self.assertEqual(
            payload["model_data_metadata"]["categorical_table_sizes"],
            {"PROTOCOL": len(vocabulary["id_to_token"])},
        )
        self.assertEqual(tokenizer["reserved_category_ids"], {
            "missing_padding": 0,
            "unknown": 1,
        })
        self.assertEqual(tokenizer["data_manifest"]["manifest_sha256"], "data-manifest-hash")
        self.assertEqual(tokenizer["split_manifest_sha256"], "split-manifest-hash")
        self.assertEqual(tokenizer["preprocessor_hash"], "preprocessor-hash")
        self.assertEqual(tokenizer["training_sample_ids_hash"], "train-sample-hash")
        self.assertEqual(vocabulary["overflow_unique_count"], 1)
        self.assertEqual(vocabulary["overflow_frequency"], 4)

    def test_checkpoint_v4_requires_global_step_git_commit_and_label_spaces(self):
        metadata = _model_data_metadata_v2()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            model = _Stateful({"weight": _FakeTensor("model")})
            with mock.patch(
                "flowfm.training.checkpoints._require_torch",
                return_value=_FakeTorch,
            ):
                save_checkpoint(
                    path,
                    model,
                    epoch=1,
                    global_step=12,
                    git_commit="a" * 40,
                    schema_version="checkpoint-test-schema-v1",
                    seed=7,
                    config={},
                    model_data_metadata=metadata,
                )
                payload = load_checkpoint(
                    path,
                    model,
                    expected_model_data_metadata=metadata,
                )

        self.assertEqual(payload["global_step"], 12)
        self.assertEqual(payload["git_commit"], "a" * 40)
        self.assertEqual(
            payload["model_data_metadata"]["label_spaces"],
            metadata["label_spaces"],
        )

    def test_metadata_mismatch_rejects_before_model_or_optimizer_mutation(self):
        metadata = _model_data_metadata()
        mismatched = _model_data_metadata(manifest_hash="different-manifest")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            self._save_payload(
                path,
                schema_version="checkpoint-test-schema-v1",
                model_data_metadata=metadata,
                adapter_state=None,
                scheduler_state=None,
            )
            model = _Stateful({"model-original": True})
            optimizer = _Stateful({"optimizer-original": True})
            with mock.patch(
                "flowfm.training.checkpoints._require_torch",
                return_value=_FakeTorch,
            ):
                with self.assertRaisesRegex(ValueError, "model_data_metadata mismatch"):
                    load_checkpoint(
                        path,
                        model,
                        optimizer=optimizer,
                        expected_model_data_metadata=mismatched,
                    )

        self.assertEqual(model.state, {"model-original": True})
        self.assertEqual(optimizer.state, {"optimizer-original": True})
        self.assertEqual(model.loaded, [])
        self.assertEqual(optimizer.loaded, [])

    def test_load_rejects_malformed_payload_before_mutating_model(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.pt"
            with path.open("wb") as handle:
                pickle.dump({"format_version": 1, "model_state": {"bad": True}}, handle)
            model = _Stateful({"original": True})

            with mock.patch(
                "flowfm.training.checkpoints._require_torch",
                return_value=_FakeTorch,
            ):
                with self.assertRaisesRegex(ValueError, "missing required fields"):
                    load_checkpoint(
                        path,
                        model,
                        expected_model_data_metadata=_model_data_metadata("ids-v2"),
                    )

            self.assertEqual(model.state, {"original": True})
            self.assertEqual(model.loaded, [])

    def test_optional_component_presence_must_match_before_any_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            self._save_payload(path)
            model = _Stateful({"original": True})

            with mock.patch(
                "flowfm.training.checkpoints._require_torch",
                return_value=_FakeTorch,
            ):
                with self.assertRaisesRegex(ValueError, "optimizer.*caller"):
                    load_checkpoint(
                        path,
                        model,
                        expected_model_data_metadata=_model_data_metadata("ids-v2"),
                    )

            self.assertEqual(model.state, {"original": True})
            self.assertEqual(model.loaded, [])

            self._save_payload(
                path,
                adapter_state=None,
                optimizer_state=None,
                scheduler_state=None,
            )
            supplied_optimizer = _Stateful({"untouched": True})
            with mock.patch(
                "flowfm.training.checkpoints._require_torch",
                return_value=_FakeTorch,
            ):
                with self.assertRaisesRegex(ValueError, "optimizer.*missing"):
                    load_checkpoint(
                        path,
                        model,
                        optimizer=supplied_optimizer,
                        expected_model_data_metadata=_model_data_metadata("ids-v2"),
                    )
            self.assertEqual(model.loaded, [])
            self.assertEqual(supplied_optimizer.loaded, [])

    def test_malformed_optional_or_rng_state_is_rejected_before_mutation(self):
        malformed_cases = {
            "optimizer_state": [],
            "scheduler_state": "bad",
            "adapter_state": 4,
            "python_random_state": (999, (), None),
            "torch_cpu_rng_state": b"not-a-tensor",
            "global_step": -1,
            "git_commit": "not-a-commit",
        }
        for field, value in malformed_cases.items():
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "checkpoint.pt"
                self._save_payload(path, **{field: value})
                model = _Stateful({"model-original": True})
                optimizer = _Stateful({"optimizer-original": True})
                scheduler = _Stateful({"scheduler-original": True})
                adapter = _Stateful({"adapter-original": True})
                with mock.patch(
                    "flowfm.training.checkpoints._require_torch",
                    return_value=_FakeTorch,
                ):
                    with self.assertRaisesRegex(ValueError, field):
                        load_checkpoint(
                            path,
                            model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            adapter=adapter,
                            expected_model_data_metadata=_model_data_metadata("ids-v2"),
                        )
                for component in (model, optimizer, scheduler, adapter):
                    self.assertEqual(component.loaded, [])

    def test_rng_portability_failures_happen_before_mutation(self):
        cases = (
            (
                "numpy_random_state",
                {
                    "bit_generator": "MT19937",
                    "state": [1, 2, 3],
                    "position": 0,
                    "has_gauss": 0,
                    "cached_gaussian": 0.0,
                },
                "NumPy",
            ),
            ("torch_cuda_rng_states", [_FakeTensor("cuda")], "CUDA"),
        )
        for field, value, message in cases:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "checkpoint.pt"
                self._save_payload(
                    path,
                    adapter_state=None,
                    optimizer_state=None,
                    scheduler_state=None,
                    **{field: value},
                )
                model = _Stateful({"original": True})
                patches = [
                    mock.patch(
                        "flowfm.training.checkpoints._require_torch",
                        return_value=_FakeTorch,
                    )
                ]
                if field == "numpy_random_state":
                    patches.append(
                        mock.patch(
                            "flowfm.training.checkpoints._optional_numpy",
                            return_value=None,
                        )
                    )
                with patches[0]:
                    context = patches[1] if len(patches) == 2 else mock.patch.object(
                        _FakeCuda, "is_available", return_value=False
                    )
                    with context:
                        with self.assertRaisesRegex(ValueError, message):
                            load_checkpoint(
                                path,
                                model,
                                expected_model_data_metadata=_model_data_metadata("ids-v2"),
                            )
                self.assertEqual(model.loaded, [])

    def test_current_rng_backend_without_saved_state_fails_before_mutation(self):
        class _FakeNumpy:
            pass

        cases = ("numpy", "cuda")
        for backend in cases:
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "checkpoint.pt"
                self._save_payload(
                    path,
                    adapter_state=None,
                    optimizer_state=None,
                    scheduler_state=None,
                    numpy_random_state=None,
                    torch_cuda_rng_states=None,
                )
                model = _Stateful({"original": True})
                numpy_value = _FakeNumpy() if backend == "numpy" else None
                with mock.patch(
                    "flowfm.training.checkpoints._require_torch",
                    return_value=_FakeTorch,
                ), mock.patch(
                    "flowfm.training.checkpoints._optional_numpy",
                    return_value=numpy_value,
                ), mock.patch.object(
                    _FakeCuda, "is_available", return_value=backend == "cuda"
                ), mock.patch.object(
                    _FakeCuda, "device_count", return_value=1 if backend == "cuda" else 0
                ):
                    expected_message = "CUDA" if backend == "cuda" else "NumPy"
                    with self.assertRaisesRegex(ValueError, expected_message):
                        load_checkpoint(
                            path,
                            model,
                            expected_model_data_metadata=_model_data_metadata("ids-v2"),
                        )
                self.assertEqual(model.loaded, [])

    def test_restore_rng_false_is_explicit_backend_mismatch_escape(self):
        class _FakeNumpy:
            pass

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            self._save_payload(
                path,
                adapter_state=None,
                optimizer_state=None,
                scheduler_state=None,
                numpy_random_state=None,
                torch_cuda_rng_states=None,
            )
            model = _Stateful({"original": True})
            with mock.patch(
                "flowfm.training.checkpoints._require_torch",
                return_value=_FakeTorch,
            ), mock.patch(
                "flowfm.training.checkpoints._optional_numpy",
                return_value=_FakeNumpy(),
            ), mock.patch.object(
                _FakeCuda, "is_available", return_value=True
            ):
                load_checkpoint(
                    path,
                    model,
                    restore_rng=False,
                    expected_model_data_metadata=_model_data_metadata("ids-v2"),
                )
            self.assertEqual(model.state, {"weight": 3})

    def test_rng_tensors_are_moved_to_cpu_after_non_cpu_map_location(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            model_tensor = _FakeTensor("model-weight", "cuda")
            self._save_payload(
                path,
                model_state={"weight": model_tensor},
                adapter_state=None,
                optimizer_state=None,
                scheduler_state=None,
                torch_cpu_rng_state=_FakeTensor("cpu-rng", "cuda"),
                torch_cuda_rng_states=[_FakeTensor("cuda-rng", "cuda")],
            )
            model = _Stateful({"original": True})
            with mock.patch(
                "flowfm.training.checkpoints._require_torch",
                return_value=_FakeTorch,
            ), mock.patch.object(
                _FakeCuda, "is_available", return_value=True
            ), mock.patch.object(
                _FakeCuda, "device_count", return_value=1
            ):
                payload = load_checkpoint(
                    path,
                    model,
                    map_location="cuda",
                    expected_model_data_metadata=_model_data_metadata("ids-v2"),
                )

            self.assertEqual(model.state["weight"].device.type, "cuda")
            self.assertEqual(payload["torch_cpu_rng_state"].device.type, "cpu")
            self.assertEqual(payload["torch_cuda_rng_states"][0].device.type, "cpu")
            self.assertEqual(_FakeTorch.restored_rng.device.type, "cpu")

    def test_save_rejects_unsafe_component_state_before_creating_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            adapter = _Stateful({"unsafe": object()})
            with mock.patch(
                "flowfm.training.checkpoints._require_torch",
                return_value=_FakeTorch,
            ):
                with self.assertRaisesRegex(ValueError, "adapter_state"):
                    save_checkpoint(
                        path,
                        _Stateful({"weight": _FakeTensor("model")}),
                        epoch=1,
                        global_step=0,
                        git_commit=_TEST_GIT_COMMIT,
                        schema_version="v2",
                        seed=4,
                        config={},
                        model_data_metadata=_model_data_metadata("v2"),
                        adapter=adapter,
                    )
            self.assertFalse(path.exists())
            self.assertEqual(list(Path(directory).glob(".checkpoint.pt.*.tmp")), [])

    def test_save_accepts_nested_optimizer_state_with_integer_keys(self):
        optimizer = _Stateful(
            {
                "state": {0: {"momentum_buffer": _FakeTensor("momentum")}},
                "param_groups": [{"lr": 0.1, "params": [0]}],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            with mock.patch(
                "flowfm.training.checkpoints._require_torch",
                return_value=_FakeTorch,
            ):
                save_checkpoint(
                    path,
                    _Stateful({"weight": _FakeTensor("model")}),
                    epoch=1,
                    global_step=0,
                    git_commit=_TEST_GIT_COMMIT,
                    schema_version="v2",
                    seed=4,
                    config={},
                    model_data_metadata=_model_data_metadata("v2"),
                    optimizer=optimizer,
                )
            self.assertTrue(path.is_file())

    def test_save_accepts_reduce_lr_on_plateau_infinite_scheduler_sentinels(self):
        scheduler = _Stateful(
            {
                "factor": 0.1,
                "min_lrs": [0.0],
                "patience": 10,
                "verbose": False,
                "cooldown": 0,
                "cooldown_counter": 0,
                "mode": "min",
                "threshold": 0.0001,
                "threshold_mode": "rel",
                "eps": 1e-8,
                "last_epoch": 0,
                "_last_lr": [0.1],
                "mode_worse": float("inf"),
                "best": float("inf"),
                "num_bad_epochs": 0,
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            with mock.patch(
                "flowfm.training.checkpoints._require_torch",
                return_value=_FakeTorch,
            ):
                save_checkpoint(
                    path,
                    _Stateful({"weight": _FakeTensor("model")}),
                    epoch=0,
                    global_step=0,
                    git_commit=_TEST_GIT_COMMIT,
                    schema_version="v2",
                    seed=4,
                    config={},
                    model_data_metadata=_model_data_metadata("v2"),
                    scheduler=scheduler,
                )
            self.assertTrue(path.is_file())

    def test_component_failure_rolls_back_all_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            self._save_payload(path, adapter_state=None, scheduler_state=None)
            model = _Stateful({"model-original": True})
            optimizer = _FailOnceStateful({"optimizer-original": True})
            with mock.patch(
                "flowfm.training.checkpoints._require_torch",
                return_value=_FakeTorch,
            ):
                with self.assertRaisesRegex(RuntimeError, "transaction"):
                    load_checkpoint(
                        path,
                        model,
                        optimizer=optimizer,
                        expected_model_data_metadata=_model_data_metadata("ids-v2"),
                    )

            self.assertEqual(model.state, {"model-original": True})
            self.assertEqual(optimizer.state, {"optimizer-original": True})

    def test_rng_setter_failure_rolls_back_components_and_rng(self):
        class _FailOnceRngTorch(_FakeTorch):
            failures_remaining = 1

            @staticmethod
            def set_rng_state(state):
                if _FailOnceRngTorch.failures_remaining:
                    _FailOnceRngTorch.failures_remaining -= 1
                    raise RuntimeError("injected RNG failure")
                _FakeTorch.set_rng_state(state)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            saved_python_state = random.getstate()
            self._save_payload(
                path,
                adapter_state=None,
                optimizer_state=None,
                scheduler_state=None,
            )
            random.random()
            current_python_state = random.getstate()
            model = _Stateful({"model-original": True})
            with mock.patch(
                "flowfm.training.checkpoints._require_torch",
                return_value=_FailOnceRngTorch,
            ):
                with self.assertRaisesRegex(RuntimeError, "transaction"):
                    load_checkpoint(
                        path,
                        model,
                        expected_model_data_metadata=_model_data_metadata("ids-v2"),
                    )

            self.assertNotEqual(current_python_state, saved_python_state)
            self.assertEqual(random.getstate(), current_python_state)
            self.assertEqual(model.state, {"model-original": True})
            self.assertEqual(_FakeTorch.current_rng, _FakeTensor("fake-torch-rng"))

    def test_rng_types_are_validated_even_when_restore_is_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            self._save_payload(
                path,
                adapter_state=None,
                optimizer_state=None,
                scheduler_state=None,
                torch_cpu_rng_state=b"bad",
            )
            model = _Stateful({"original": True})
            with mock.patch(
                "flowfm.training.checkpoints._require_torch",
                return_value=_FakeTorch,
            ):
                with self.assertRaisesRegex(ValueError, "torch_cpu_rng_state"):
                    load_checkpoint(
                        path,
                        model,
                        restore_rng=False,
                        expected_model_data_metadata=_model_data_metadata("ids-v2"),
                    )
            self.assertEqual(model.loaded, [])

    def test_torch_load_does_not_hide_unrelated_type_errors(self):
        class _BrokenLoadTorch(_FakeTorch):
            @staticmethod
            def load(path, map_location="cpu", weights_only=True):
                del path, map_location, weights_only
                raise TypeError("corrupt tensor type")

        with mock.patch(
            "flowfm.training.checkpoints._require_torch",
            return_value=_BrokenLoadTorch,
        ):
            with self.assertRaisesRegex(TypeError, "corrupt tensor type"):
                load_checkpoint(
                    "unused.pt",
                    _Stateful({}),
                    expected_model_data_metadata=_model_data_metadata("ids-v2"),
                )

    def test_checkpoint_api_requires_torch_only_when_called(self):
        model = _Stateful({})
        with mock.patch(
            "flowfm.training.checkpoints.importlib.import_module",
            side_effect=ImportError("no torch"),
        ):
            with self.assertRaisesRegex(RuntimeError, "PyTorch is required"):
                save_checkpoint(
                    "unused.pt",
                    model,
                    epoch=0,
                    global_step=0,
                    git_commit=_TEST_GIT_COMMIT,
                    schema_version="v1",
                    seed=1,
                    config={},
                    model_data_metadata=_model_data_metadata("v1"),
                )


class MetadataTests(unittest.TestCase):
    def test_validate_output_path_enforces_real_data_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "autodl-tmp"
            root.mkdir()
            accepted = validate_output_path(
                root / "runs" / "one",
                enforce_autodl_data_disk=True,
                data_root=root,
            )
            self.assertEqual(accepted, (root / "runs" / "one").resolve())

            for rejected in (root, root.parent / "autodl-tmp-evil" / "run"):
                with self.subTest(rejected=rejected):
                    with self.assertRaises(ValueError):
                        validate_output_path(
                            rejected,
                            enforce_autodl_data_disk=True,
                            data_root=root,
                        )

    def test_capture_provenance_is_serializable_and_stable_for_config(self):
        config_a = {"batch_size": 16, "model": {"layers": [2, 4]}}
        config_b = {"model": {"layers": [2, 4]}, "batch_size": 16}
        with tempfile.TemporaryDirectory() as directory:
            first = capture_provenance(
                directory, config_a, "schema-1", "dataset-hash", "split-hash", seed=7
            )
            second = capture_provenance(
                directory, config_b, "schema-1", "dataset-hash", "split-hash", seed=7
            )

        json.dumps(first)
        self.assertEqual(first["config_hash"], second["config_hash"])
        self.assertEqual(
            first["config_hash"], compute_config_hash(config_a)
        )
        self.assertEqual(first["git_commit"], None)
        self.assertEqual(first["git_dirty"], None)
        self.assertEqual(first["schema_version"], "schema-1")
        self.assertEqual(first["dataset_hash"], "dataset-hash")
        self.assertEqual(first["split_hash"], "split-hash")
        self.assertEqual(first["seed"], 7)
        for field in (
            "timestamp_utc",
            "python_version",
            "platform",
            "torch_version",
            "cuda_runtime",
            "cuda_available",
        ):
            self.assertIn(field, first)

    def test_capture_provenance_reports_clean_and_dirty_git_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            subprocess.run(
                ["git", "-C", str(repo), "config", "user.email", "test@example.com"],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(repo), "config", "user.name", "Test User"],
                check=True,
            )
            tracked = repo / "tracked.txt"
            tracked.write_text("initial\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(repo), "add", "tracked.txt"], check=True)
            subprocess.run(
                ["git", "-C", str(repo), "commit", "-q", "-m", "initial"],
                check=True,
            )

            clean = capture_provenance(repo, {}, "v1", "data", "split")
            (repo / "untracked.txt").write_text("dirty\n", encoding="utf-8")
            dirty = capture_provenance(repo, {}, "v1", "data", "split")

        self.assertRegex(clean["git_commit"], r"^[0-9a-f]{40}$")
        self.assertFalse(clean["git_dirty"])
        self.assertEqual(dirty["git_commit"], clean["git_commit"])
        self.assertTrue(dirty["git_dirty"])

    def test_capture_provenance_rejects_non_json_runtime_field(self):
        with mock.patch(
            "flowfm.experiment.metadata.platform.platform", return_value=object()
        ):
            with tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(ValueError, "provenance.*JSON"):
                    capture_provenance(directory, {}, "v1", "data", "split")

    def test_capture_provenance_requires_string_or_none_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            for field, dataset_hash, split_hash in (
                ("dataset_hash", 123, None),
                ("split_hash", None, ["invalid"]),
            ):
                with self.subTest(field=field):
                    with self.assertRaisesRegex(ValueError, field + ".*string or None"):
                        capture_provenance(
                            directory,
                            {},
                            "v1",
                            dataset_hash,
                            split_hash,
                        )

    def test_write_json_snapshot_is_deterministic_and_roundtrips(self):
        payload = {"z": [3, 2], "a": {"unicode": "流"}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "snapshot.json"
            write_json_snapshot(path, payload)
            first_bytes = path.read_bytes()
            write_json_snapshot(path, payload)
            second_bytes = path.read_bytes()

            self.assertEqual(first_bytes, second_bytes)
            self.assertEqual(json.loads(first_bytes.decode("utf-8")), payload)
            self.assertTrue(first_bytes.endswith(b"\n"))
            self.assertLess(first_bytes.index(b'"a"'), first_bytes.index(b'"z"'))

    def test_write_json_snapshot_validates_before_replacing_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "snapshot.json"
            path.write_text('{"original": true}\n', encoding="utf-8")
            with self.assertRaises(TypeError):
                write_json_snapshot(path, {"bad": object()})
            self.assertEqual(path.read_text(encoding="utf-8"), '{"original": true}\n')


if __name__ == "__main__":
    unittest.main()
