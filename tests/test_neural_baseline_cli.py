import unittest

from scripts.run_nf_unsw_neural_baseline import (
    build_parser,
    model_config_from_args,
)


class NeuralBaselineCLITests(unittest.TestCase):
    def test_parser_requires_one_model_and_one_single_task(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--model",
                "ft_transformer",
                "--task",
                "binary",
                "--device",
                "cpu",
            ]
        )
        self.assertEqual(args.model, "ft_transformer")
        self.assertEqual(args.task, "binary")
        self.assertEqual(args.dataset, "nf_unsw")
        self.assertEqual(args.batch_size, 256)
        self.assertEqual(args.epochs, 1)
        ton_args = parser.parse_args(
            [
                "--dataset", "nf_ton", "--model", "ft_transformer",
                "--task", "native",
            ]
        )
        self.assertEqual(ton_args.dataset, "nf_ton")
        with self.assertRaises(SystemExit):
            parser.parse_args(
                ["--model", "ft_transformer", "--task", "all"]
            )
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "--model",
                    "rescae_bigru",
                    "--task",
                    "binary",
                    "--batch-size",
                    "0",
                ]
            )

    def test_model_configs_preserve_legacy_and_exclude_schemaflow_objectives(self):
        parser = build_parser()
        legacy = parser.parse_args(
            ["--model", "rescae_bigru", "--task", "native"]
        )
        self.assertEqual(model_config_from_args(legacy), {})
        ft = parser.parse_args(
            ["--model", "ft_transformer", "--task", "coarse"]
        )
        config = model_config_from_args(ft)
        self.assertEqual(
            config,
            {
                "d_token": 64,
                "nhead": 8,
                "num_layers": 3,
                "dim_feedforward": 256,
                "dropout": 0.1,
            },
        )
        for forbidden in (
            "adapter_dim",
            "schema_corruption",
            "infonce",
            "pretraining",
        ):
            self.assertNotIn(forbidden, config)


if __name__ == "__main__":
    unittest.main()
