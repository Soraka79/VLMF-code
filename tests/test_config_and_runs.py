import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from flowfm.config import (
    CONFIG_HASH_NAMESPACE,
    FlowConfig,
    _validate_strict_json,
    compute_config_hash,
    load_config,
)
from flowfm.data.identity import canonical_json, stable_hash
from flowfm.experiment.logging import JsonlLogger
from flowfm.experiment.runs import RunContext, create_run_context


def config_text(
    *,
    workflow="evaluate",
    name="schemaflow-test",
    output_root="/tmp/schemaflow-results",
    seed="42",
    enforce="false",
    schema_version="phase1-v1",
    device="auto",
):
    return f"""
workflow: {workflow}
experiment:
  name: {name}
  output_root: {output_root}
  seed: {seed}
  enforce_autodl_data_disk: {enforce}
schema:
  version: {schema_version}
runtime:
  device: {device}
"""


class DependencyTests(unittest.TestCase):
    def test_pyyaml_is_a_required_test_dependency(self):
        if importlib.util.find_spec("yaml") is None:
            project_root = Path(__file__).resolve().parents[1]
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "flowfm",
                    "evaluate",
                    "--config",
                    str(project_root / "configs" / "smoke-evaluate.yaml"),
                ],
                cwd=project_root,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertEqual(result.stdout, "")
            self.assertIn("flowfm: PyYAML is required", result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            self.fail(
                "PyYAML is required; install requirements.txt in the test environment"
            )


class SmokeConfigTests(unittest.TestCase):
    def test_has_one_unambiguous_smoke_config_per_workflow(self):
        project_root = Path(__file__).resolve().parents[1]
        self.assertFalse((project_root / "configs" / "smoke.yaml").exists())
        for workflow in ("pretrain", "finetune", "evaluate", "ablate"):
            with self.subTest(workflow=workflow):
                path = project_root / "configs" / f"smoke-{workflow}.yaml"
                self.assertTrue(path.is_file(), path)
                text = path.read_text(encoding="utf-8")
                self.assertIn(f"workflow: {workflow}\n", text)
                for field in (
                    "name: schemaflow-smoke",
                    "output_root: /root/autodl-tmp/Res-AE-Attn/results",
                    "seed: 42",
                    "enforce_autodl_data_disk: true",
                    "version: phase1-v1",
                    "device: auto",
                ):
                    self.assertIn(field, text)


class ConfigTests(unittest.TestCase):
    def load_text(self, text, expected_workflow=None):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(text, encoding="utf-8")
            return load_config(path, expected_workflow=expected_workflow)

    def test_loads_valid_yaml_and_applies_optional_default(self):
        config = self.load_text(
            config_text().replace("  enforce_autodl_data_disk: false\n", "")
        )

        self.assertEqual(config.workflow, "evaluate")
        self.assertEqual(config.experiment["name"], "schemaflow-test")
        self.assertEqual(config.experiment["seed"], 42)
        self.assertFalse(config.experiment["enforce_autodl_data_disk"])
        self.assertEqual(config.schema["version"], "phase1-v1")
        self.assertEqual(config.runtime["device"], "auto")

    def test_rejects_unsafe_yaml_constructor(self):
        with self.assertRaisesRegex(ValueError, "invalid YAML"):
            self.load_text("!!python/object/apply:os.system ['echo unsafe']\n")

    def test_rejects_duplicate_mapping_keys(self):
        duplicate = config_text().replace(
            "workflow: evaluate", "workflow: evaluate\nworkflow: pretrain"
        )
        with self.assertRaisesRegex(ValueError, "duplicate YAML key.*workflow"):
            self.load_text(duplicate)

    def test_rejects_non_string_yaml_mapping_keys(self):
        with self.assertRaisesRegex(
            ValueError, "YAML mapping keys must be strings"
        ):
            self.load_text("? [a, b]\n: value\n")

    def test_rejects_self_and_nested_yaml_aliases(self):
        cases = (
            (
                "self",
                """
workflow: evaluate
experiment: &experiment
  name: schemaflow-test
  output_root: /tmp/results
  seed: 42
  recursive: *experiment
schema: {version: phase1-v1}
runtime: {device: auto}
""",
            ),
            (
                "nested",
                """
workflow: evaluate
experiment:
  name: &name schemaflow-test
  output_root: /tmp/results
  seed: 42
schema: {version: phase1-v1}
runtime:
  device: *name
""",
            ),
        )
        for label, text in cases:
            with self.subTest(label=label):
                with self.assertRaisesRegex(ValueError, "YAML aliases are not allowed"):
                    self.load_text(text)

    def test_requires_exact_top_level_sections(self):
        valid = config_text()
        with self.assertRaisesRegex(ValueError, "missing.*runtime"):
            self.load_text(valid.replace("runtime:\n  device: auto\n", ""))
        with self.assertRaisesRegex(ValueError, "unknown.*extra"):
            self.load_text(valid + "extra: true\n")
        with self.assertRaisesRegex(ValueError, "experiment.*mapping"):
            self.load_text(
                valid.replace(
                    "experiment:\n  name: schemaflow-test\n"
                    "  output_root: /tmp/schemaflow-results\n"
                    "  seed: 42\n  enforce_autodl_data_disk: false\n",
                    "experiment: invalid\n",
                )
            )

    def test_rejects_invalid_or_mismatched_workflow(self):
        with self.assertRaisesRegex(ValueError, "workflow"):
            self.load_text(config_text(workflow="train"))
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.load_text(config_text(), expected_workflow="pretrain")

    def test_rejects_surrounding_whitespace_in_string_fields(self):
        cases = (
            (config_text(workflow="' evaluate'"), "workflow"),
            (config_text(name="'schemaflow-test '"), "experiment name"),
            (
                config_text(output_root="'/tmp/schemaflow-results '"),
                "output_root",
            ),
            (config_text(schema_version="' phase1-v1'"), "schema version"),
            (config_text(device="'auto '"), "runtime device"),
        )
        for text, field in cases:
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "surrounding whitespace"):
                    self.load_text(text)

    def test_validates_section_fields_and_rejects_unknown_keys(self):
        cases = [
            (config_text(name="''"), "name"),
            (config_text(seed="true"), "seed"),
            (config_text(device="mps"), "device"),
            (config_text(schema_version="''"), "version"),
            (config_text().replace("  seed: 42\n", ""), "seed"),
            (
                config_text().replace(
                    "  device: auto\n", "  device: auto\n  extra: 1\n"
                ),
                "unknown",
            ),
        ]
        for text, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    self.load_text(text)

    def test_rejects_non_finite_and_non_strict_json_values(self):
        with self.assertRaisesRegex(ValueError, "strict JSON"):
            self.load_text(config_text().replace("seed: 42", "seed: 42\n  bad: .nan"))
        with self.assertRaisesRegex(ValueError, "strict JSON"):
            self.load_text(
                config_text().replace("name: schemaflow-test", "name: 2026-07-20")
            )

    def test_hash_is_stable_across_yaml_key_order(self):
        first = self.load_text(config_text())
        second = self.load_text(
            """
runtime: {device: auto}
schema: {version: phase1-v1}
experiment:
  seed: 42
  output_root: /tmp/schemaflow-results
  name: schemaflow-test
  enforce_autodl_data_disk: false
workflow: evaluate
"""
        )

        self.assertEqual(first.config_hash, second.config_hash)
        self.assertEqual(CONFIG_HASH_NAMESPACE, "flowfm-config-v1")
        self.assertEqual(
            first.config_hash,
            compute_config_hash(first.to_dict()),
        )
        self.assertEqual(
            first.config_hash,
            stable_hash(first.to_dict(), namespace=CONFIG_HASH_NAMESPACE),
        )

    def test_config_is_immutable_and_to_dict_is_defensive(self):
        config = self.load_text(config_text())
        copied = config.to_dict()
        copied["experiment"]["name"] = "changed"

        self.assertEqual(config.experiment["name"], "schemaflow-test")
        with self.assertRaises(TypeError):
            config.experiment["name"] = "changed"

    def test_rejects_forged_config_hash(self):
        payload = {
            "workflow": "evaluate",
            "experiment": {
                "name": "schemaflow-test",
                "output_root": "/tmp/results",
                "seed": 42,
                "enforce_autodl_data_disk": False,
            },
            "schema": {"version": "phase1-v1"},
            "runtime": {"device": "auto"},
        }
        with self.assertRaisesRegex(ValueError, "config_hash.*does not match"):
            FlowConfig(canonical_json(payload), "0" * 64)

    def test_strict_json_validation_rejects_cycles_without_recursing_forever(self):
        cyclic = {}
        cyclic["self"] = cyclic
        with self.assertRaisesRegex(ValueError, "cyclic"):
            _validate_strict_json(cyclic)


class RunContextTests(unittest.TestCase):
    def load_for_root(self, output_root, enforce=False):
        existing_parent = next(parent for parent in output_root.parents if parent.exists())
        path = existing_parent / "config.yaml"
        path.write_text(
            config_text(
                output_root=str(output_root),
                enforce="true" if enforce else "false",
            ),
            encoding="utf-8",
        )
        return load_config(path)

    def test_rejects_name_path_traversal_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            config_path = base / "config.yaml"
            config_path.write_text(
                config_text(name="../escape", output_root=str(base / "out")),
                encoding="utf-8",
            )
            config = load_config(config_path)

            with self.assertRaisesRegex(ValueError, "unsafe experiment name"):
                create_run_context(config, base)
            self.assertFalse((base / "out").exists())

    def test_enforces_injected_data_root(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            allowed = base / "data"
            config = self.load_for_root(base / "outside", enforce=True)

            with self.assertRaisesRegex(ValueError, "nested below"):
                create_run_context(config, base, data_root=allowed)
            self.assertFalse((base / "outside").exists())

    def test_creates_unique_layout_and_snapshots(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            output_root = base / "data" / "results"
            config = self.load_for_root(output_root, enforce=True)
            now = datetime(2026, 7, 20, 3, 4, 5, tzinfo=timezone.utc)

            context = create_run_context(
                config,
                base,
                dataset_hash="dataset-hash",
                split_hash="split-hash",
                now=now,
                data_root=base / "data",
            )

            self.assertEqual(context.run_dir.parent, output_root.resolve())
            self.assertRegex(
                context.run_dir.name,
                r"^schemaflow-test-20260720T030405000000Z-[0-9a-f]{8}-[0-9a-f]{12}$",
            )
            self.assertFalse((context.run_dir / ".incomplete").exists())
            for name in ("checkpoints", "logs", "metrics", "predictions", "artifacts"):
                self.assertEqual(getattr(context, name), context.run_dir / name)
                self.assertTrue((context.run_dir / name).is_dir())

            config_snapshot = json.loads(
                context.config_snapshot.read_text(encoding="utf-8")
            )
            environment = json.loads(
                context.environment_snapshot.read_text(encoding="utf-8")
            )
            self.assertEqual(config_snapshot, config.to_dict())
            self.assertEqual(environment["config_hash"], config.config_hash)
            self.assertEqual(environment["dataset_hash"], "dataset-hash")
            self.assertEqual(environment["split_hash"], "split-hash")
            config_snapshot["experiment"]["name"] = "mutated"
            self.assertEqual(config.experiment["name"], "schemaflow-test")

            second = create_run_context(
                config,
                base,
                now=now,
                data_root=base / "data",
            )
            self.assertNotEqual(second.run_dir, context.run_dir)

    def test_failure_leaves_marked_incomplete_run_and_retry_is_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            output_root = base / "data" / "results"
            config = self.load_for_root(output_root, enforce=True)
            now = datetime(2026, 7, 20, 3, 4, 5, tzinfo=timezone.utc)

            with mock.patch(
                "flowfm.experiment.runs.write_json_snapshot",
                side_effect=OSError("injected snapshot failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "marked incomplete"):
                    create_run_context(
                        config,
                        base,
                        now=now,
                        data_root=base / "data",
                    )

            failed_runs = list(output_root.iterdir())
            self.assertEqual(len(failed_runs), 1)
            failed_run = failed_runs[0]
            self.assertTrue((failed_run / ".incomplete").is_file())

            context = create_run_context(
                config,
                base,
                now=now,
                data_root=base / "data",
            )
            self.assertNotEqual(context.run_dir, failed_run)
            self.assertFalse((context.run_dir / ".incomplete").exists())

    def test_run_context_rejects_an_incomplete_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            run_dir.mkdir()
            (run_dir / ".incomplete").write_text("incomplete\n", encoding="utf-8")
            paths = {
                name: run_dir / name
                for name in ("checkpoints", "logs", "metrics", "predictions", "artifacts")
            }
            with self.assertRaisesRegex(ValueError, "incomplete"):
                RunContext(
                    run_dir=run_dir,
                    config_snapshot=run_dir / "config.snapshot.json",
                    environment_snapshot=run_dir / "environment.snapshot.json",
                    **paths,
                )

            (run_dir / ".incomplete").unlink()
            with self.assertRaisesRegex(ValueError, "incomplete"):
                RunContext(
                    run_dir=run_dir,
                    config_snapshot=run_dir / "config.snapshot.json",
                    environment_snapshot=run_dir / "environment.snapshot.json",
                    **paths,
                )

    def test_dataset_and_split_hashes_must_be_strings_or_none(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            config = self.load_for_root(base / "results")
            for field, kwargs in (
                ("dataset_hash", {"dataset_hash": 123}),
                ("split_hash", {"split_hash": ["invalid"]}),
            ):
                with self.subTest(field=field):
                    with self.assertRaisesRegex(ValueError, field + ".*string or None"):
                        create_run_context(config, base, **kwargs)
            self.assertFalse((base / "results").exists())


class JsonlLoggerTests(unittest.TestCase):
    def test_appends_without_truncating_and_adds_utc_timestamp(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            logger = JsonlLogger(path)
            logger.log({"event": "first", "value": 1})
            logger.log({"event": "second"})

            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            first, second = map(json.loads, lines)
            self.assertEqual(first["event"], "first")
            for event in (first, second):
                timestamp = event["timestamp_utc"]
                self.assertRegex(
                    timestamp,
                    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$",
                )
                parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                self.assertEqual(parsed.utcoffset(), timezone.utc.utcoffset(parsed))
            self.assertEqual(
                lines[0],
                json.dumps(
                    first,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )

    def test_rejects_invalid_event_before_opening_or_truncating(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            logger = JsonlLogger(path)
            with self.assertRaisesRegex(ValueError, "strict JSON"):
                logger.log({"value": math.inf})
            self.assertFalse(path.exists())

            path.write_text('{"existing":true}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "strict JSON"):
                logger.log({"value": {1, 2}})
            self.assertEqual(path.read_text(encoding="utf-8"), '{"existing":true}\n')

    def test_rejects_reserved_timestamp_before_opening_or_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            logger = JsonlLogger(path)
            with self.assertRaisesRegex(ValueError, "timestamp_utc.*reserved"):
                logger.log({"event": "forged", "timestamp_utc": "fixed"})
            self.assertFalse(path.exists())

            original = '{"existing":true}\n'
            path.write_text(original, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "timestamp_utc.*reserved"):
                logger.log({"timestamp_utc": None})
            self.assertEqual(path.read_text(encoding="utf-8"), original)


class CliConfigIntegrationTests(unittest.TestCase):
    def test_real_entrypoint_publishes_validated_only_run_without_training(self):
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for command in ("pretrain", "finetune", "evaluate", "ablate"):
                with self.subTest(command=command):
                    output_root = base / command / "results"
                    config_path = base / f"{command}.yaml"
                    config_path.write_text(
                        config_text(
                            workflow=command,
                            output_root=str(output_root),
                        ),
                        encoding="utf-8",
                    )
                    result = subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "flowfm",
                            command,
                            "--config",
                            str(config_path),
                        ],
                        cwd=project_root,
                        capture_output=True,
                        text=True,
                    )

                    self.assertEqual(result.returncode, 0, result.stderr)
                    summary = json.loads(result.stdout)
                    run_dir = Path(summary.pop("run_dir"))
                    self.assertEqual(
                        summary,
                        {
                            "command": command,
                            "config": str(config_path.resolve()),
                            "config_hash": load_config(config_path).config_hash,
                            "device": "auto",
                            "experiment": "schemaflow-test",
                            "output_root": str(output_root),
                            "schema_version": "phase1-v1",
                            "status": "validated_only",
                            "training_started": False,
                        },
                    )
                    self.assertEqual(run_dir.parent, output_root.resolve())
                    self.assertTrue((run_dir / "config.snapshot.json").is_file())
                    self.assertTrue((run_dir / "environment.snapshot.json").is_file())
                    event = json.loads(
                        (run_dir / "logs" / "events.jsonl").read_text(
                            encoding="utf-8"
                        )
                    )
                    self.assertEqual(event["event"], "workflow_validated")
                    self.assertEqual(event["command"], command)
                    self.assertEqual(event["status"], "validated_only")
                    self.assertFalse(event["training_started"])
                    status = json.loads(
                        (run_dir / "metrics" / "status.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    self.assertEqual(
                        status,
                        {
                            "command": command,
                            "config_hash": load_config(config_path).config_hash,
                            "schema_version": "phase1-v1",
                            "status": "validated_only",
                            "training_started": False,
                        },
                    )

    def test_expected_config_errors_have_stable_cli_boundary(self):
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            output_root = base / "results"
            cases = (
                ("missing", base / "missing.yaml", "No such file"),
                ("directory", base / "config-dir", "Is a directory"),
                ("malformed", base / "malformed.yaml", "invalid YAML"),
                ("mismatch", base / "mismatch.yaml", "does not match"),
            )
            (base / "config-dir").mkdir()
            (base / "malformed.yaml").write_text("workflow: [\n", encoding="utf-8")
            (base / "mismatch.yaml").write_text(
                config_text(workflow="pretrain", output_root=str(output_root)),
                encoding="utf-8",
            )

            for label, path, message in cases:
                with self.subTest(label=label):
                    result = subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "flowfm",
                            "evaluate",
                            "--config",
                            str(path),
                        ],
                        cwd=project_root,
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(result.returncode, 2)
                    self.assertEqual(result.stdout, "")
                    self.assertTrue(result.stderr.startswith("flowfm: "))
                    self.assertIn(message, result.stderr)
                    self.assertNotIn("Traceback", result.stderr)
                    self.assertFalse(output_root.exists())


if __name__ == "__main__":
    unittest.main()
