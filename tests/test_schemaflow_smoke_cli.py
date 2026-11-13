import io
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

from scripts.smoke_schemaflow_nf_unsw import (
    build_parser,
    validate_data_manifest_path,
)


class SchemaFlowSmokeCLITests(unittest.TestCase):
    def test_parser_requires_frozen_inputs_and_explicit_device(self):
        parser = build_parser()
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args([])

        arguments = parser.parse_args(
            [
                "--split-dir",
                "/root/autodl-tmp/split",
                "--preprocessor-dir",
                "/root/autodl-tmp/preprocessor",
                "--data-manifest",
                "/root/autodl-tmp/manifest.json",
                "--device",
                "cpu",
            ]
        )
        self.assertEqual(arguments.device, "cpu")
        self.assertEqual(arguments.train_samples, 4)
        self.assertEqual(arguments.validation_samples, 4)

    def test_data_manifest_must_be_a_file_below_data_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "autodl-tmp"
            root.mkdir()
            manifest = root / "data" / "manifest.json"
            manifest.parent.mkdir()
            manifest.write_text('{"dataset_id": "fixture"}\n', encoding="utf-8")

            self.assertEqual(
                validate_data_manifest_path(manifest, data_root=root),
                manifest.resolve(),
            )
            for invalid in (
                root / "data" / "missing.json",
                root.parent / "outside.json",
            ):
                with self.subTest(invalid=invalid):
                    with self.assertRaises(ValueError):
                        validate_data_manifest_path(invalid, data_root=root)


if __name__ == "__main__":
    unittest.main()
