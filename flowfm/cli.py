"""Command-line parsing for phase-one SchemaFlow-FM workflows."""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from flowfm.config import load_config
from flowfm.experiment.logging import JsonlLogger
from flowfm.experiment.metadata import write_json_snapshot
from flowfm.experiment.runs import create_run_context


TRAINING_COMMANDS = ("pretrain", "finetune", "evaluate", "ablate")
DATA_COMMANDS = (
    "audit-conflicts",
    "prepare-nf-unsw-split",
    "prepare-nf-ton-split",
    "prepare-nf-ton-confirmatory-cohort",
    "derive-nf-ton-few-shot",
    "prepare-nf-cse-split",
    "prepare-nf-cse-confirmatory-split",
    "derive-nf-cse-few-shot",
    "fit-nf-unsw-preprocessor",
    "fit-nf-ton-preprocessor",
    "fit-nf-cse-preprocessor",
)
COMMANDS = (*TRAINING_COMMANDS, *DATA_COMMANDS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="flowfm")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in TRAINING_COMMANDS:
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--config", required=True, type=Path)
        if command == "finetune":
            command_parser.add_argument(
                "--allow-cross-dataset-init",
                action="store_true",
                help="allow Schema/registry-compatible partial encoder transfer",
            )
            checkpoint_group = command_parser.add_mutually_exclusive_group()
            checkpoint_group.add_argument("--resume-from", type=Path)
            checkpoint_group.add_argument("--initialize-from", type=Path)
        elif command == "pretrain":
            command_parser.add_argument("--resume-from", type=Path)

    audit_parser = subparsers.add_parser("audit-conflicts")
    audit_parser.add_argument("--config", required=True, type=Path)
    audit_parser.add_argument("--output", required=True, type=Path)

    split_parser = subparsers.add_parser("prepare-nf-unsw-split")
    split_parser.add_argument("--config", required=True, type=Path)
    split_parser.add_argument("--output", required=True, type=Path)

    nf_ton_split_parser = subparsers.add_parser("prepare-nf-ton-split")
    nf_ton_split_parser.add_argument("--config", required=True, type=Path)
    nf_ton_split_parser.add_argument("--output", required=True, type=Path)

    nf_ton_confirmatory_parser = subparsers.add_parser(
        "prepare-nf-ton-confirmatory-cohort"
    )
    nf_ton_confirmatory_parser.add_argument("--config", required=True, type=Path)
    nf_ton_confirmatory_parser.add_argument("--output", required=True, type=Path)

    nf_ton_few_shot_parser = subparsers.add_parser(
        "derive-nf-ton-few-shot"
    )
    nf_ton_few_shot_parser.add_argument(
        "--source-split-dir", required=True, type=Path
    )
    nf_ton_few_shot_parser.add_argument("--train-percent", required=True, type=int)
    nf_ton_few_shot_parser.add_argument("--seed", default=42, type=int)
    nf_ton_few_shot_parser.add_argument("--output", required=True, type=Path)

    nf_cse_split_parser = subparsers.add_parser("prepare-nf-cse-split")
    nf_cse_split_parser.add_argument("--config", required=True, type=Path)
    nf_cse_split_parser.add_argument("--output", required=True, type=Path)

    nf_cse_confirmatory_parser = subparsers.add_parser(
        "prepare-nf-cse-confirmatory-split"
    )
    nf_cse_confirmatory_parser.add_argument("--config", required=True, type=Path)
    nf_cse_confirmatory_parser.add_argument("--output", required=True, type=Path)

    nf_cse_few_shot_parser = subparsers.add_parser("derive-nf-cse-few-shot")
    nf_cse_few_shot_parser.add_argument(
        "--source-split-dir", required=True, type=Path
    )
    nf_cse_few_shot_parser.add_argument(
        "--train-percent", required=True, type=int
    )
    nf_cse_few_shot_parser.add_argument("--seed", required=True, type=int)
    nf_cse_few_shot_parser.add_argument("--output", required=True, type=Path)

    preprocessor_parser = subparsers.add_parser("fit-nf-unsw-preprocessor")
    preprocessor_parser.add_argument("--split-dir", required=True, type=Path)
    preprocessor_parser.add_argument("--output", required=True, type=Path)

    nf_ton_preprocessor_parser = subparsers.add_parser(
        "fit-nf-ton-preprocessor"
    )
    nf_ton_preprocessor_parser.add_argument(
        "--split-dir", required=True, type=Path
    )
    nf_ton_preprocessor_parser.add_argument(
        "--output", required=True, type=Path
    )

    nf_cse_preprocessor_parser = subparsers.add_parser(
        "fit-nf-cse-preprocessor"
    )
    nf_cse_preprocessor_parser.add_argument(
        "--split-dir", required=True, type=Path
    )
    nf_cse_preprocessor_parser.add_argument(
        "--output", required=True, type=Path
    )

    return parser


def _publish_validated_only_run(config, command: str, repo_dir: Path):
    context = create_run_context(config, repo_dir)
    status = {
        "command": command,
        "config_hash": config.config_hash,
        "schema_version": config.schema["version"],
        "status": "validated_only",
        "training_started": False,
    }
    incomplete_marker = context.run_dir / ".incomplete"
    incomplete_marker.write_text(
        "validation-only publication incomplete\n",
        encoding="utf-8",
    )
    try:
        write_json_snapshot(context.metrics / "status.json", status)
        JsonlLogger(context.logs / "events.jsonl").log(
            {
                "event": "workflow_validated",
                **status,
            }
        )
        incomplete_marker.unlink()
    except Exception as exc:
        raise RuntimeError(
            f"validation-only publication failed; {context.run_dir} is marked incomplete"
        ) from exc
    return context, status


def dispatch(args: argparse.Namespace):
    """Publish validation-only runs without starting model training."""
    if args.command not in COMMANDS:
        raise ValueError("unsupported command: {}".format(args.command))
    repo_dir = Path(__file__).resolve().parents[1]
    if args.command == "audit-conflicts":
        from flowfm.data.audit import run_conflict_audit

        return run_conflict_audit(
            args.config.expanduser().resolve(),
            args.output.expanduser().resolve(),
            repo_dir=repo_dir,
        )
    if args.command == "prepare-nf-unsw-split":
        from flowfm.data.nf_unsw_split import (
            publish_nf_unsw_development_split,
        )

        return publish_nf_unsw_development_split(
            args.config.expanduser().resolve(),
            args.output.expanduser().resolve(),
            repo_dir=repo_dir,
        )
    if args.command == "prepare-nf-ton-split":
        from flowfm.data.nf_ton_split import (
            publish_nf_ton_development_split,
        )

        return publish_nf_ton_development_split(
            args.config.expanduser().resolve(),
            args.output.expanduser().resolve(),
            repo_dir=repo_dir,
        )
    if args.command == "prepare-nf-ton-confirmatory-cohort":
        from flowfm.data.nf_ton_confirmatory import (
            publish_nf_ton_confirmatory_cohort,
        )

        return publish_nf_ton_confirmatory_cohort(
            args.config.expanduser().resolve(),
            args.output.expanduser().resolve(),
            repo_dir=repo_dir,
        )
    if args.command == "derive-nf-ton-few-shot":
        from flowfm.data.nf_ton_few_shot import (
            publish_nf_ton_few_shot_split,
        )

        return publish_nf_ton_few_shot_split(
            args.source_split_dir.expanduser().resolve(),
            args.output.expanduser().resolve(),
            train_percent=args.train_percent,
            seed=args.seed,
            repo_dir=repo_dir,
        )
    if args.command == "prepare-nf-cse-split":
        from flowfm.data.nf_cse_split import (
            publish_nf_cse_development_split,
        )

        return publish_nf_cse_development_split(
            args.config.expanduser().resolve(),
            args.output.expanduser().resolve(),
            repo_dir=repo_dir,
        )

    if args.command == "prepare-nf-cse-confirmatory-split":
        from flowfm.data.nf_cse_confirmatory import (
            publish_nf_cse_confirmatory_split,
        )

        return publish_nf_cse_confirmatory_split(
            args.config,
            args.output,
            repo_dir=repo_dir,
        )

    if args.command == "derive-nf-cse-few-shot":
        from flowfm.data.nf_cse_few_shot import publish_nf_cse_few_shot_split

        return publish_nf_cse_few_shot_split(
            args.source_split_dir,
            args.output,
            train_percent=args.train_percent,
            seed=args.seed,
            repo_dir=repo_dir,
        )
    if args.command == "fit-nf-unsw-preprocessor":
        from flowfm.data.nf_unsw_preprocessing import (
            publish_nf_unsw_train_only_preprocessor,
        )

        return publish_nf_unsw_train_only_preprocessor(
            args.split_dir.expanduser().resolve(),
            args.output.expanduser().resolve(),
            repo_dir=repo_dir,
        )
    if args.command == "fit-nf-ton-preprocessor":
        from flowfm.data.nf_ton_preprocessing import (
            publish_nf_ton_train_only_preprocessor,
        )

        return publish_nf_ton_train_only_preprocessor(
            args.split_dir.expanduser().resolve(),
            args.output.expanduser().resolve(),
            repo_dir=repo_dir,
        )
    if args.command == "fit-nf-cse-preprocessor":
        from flowfm.data.nf_cse_preprocessing import (
            publish_nf_cse_train_only_preprocessor,
        )

        return publish_nf_cse_train_only_preprocessor(
            args.split_dir.expanduser().resolve(),
            args.output.expanduser().resolve(),
            repo_dir=repo_dir,
        )
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path, expected_workflow=args.command)
    if config.runtime.get("execution", "validate_only") == "train":
        resume_from = (
            None
            if getattr(args, "resume_from", None) is None
            else args.resume_from.expanduser().resolve()
        )
        initialize_from = (
            None
            if getattr(args, "initialize_from", None) is None
            else args.initialize_from.expanduser().resolve()
        )
        allow_cross_dataset_init = bool(
            getattr(args, "allow_cross_dataset_init", False)
        )
        if args.command == "finetune":
            from flowfm.training.supervised import (
                run_nf_cse_finetune,
                run_nf_ton_finetune,
                run_nf_unsw_finetune,
            )

            dataset_id = config.data["dataset_id"]
            if dataset_id == "NF-UNSW-NB15-v2":
                runner = run_nf_unsw_finetune
            elif dataset_id == "NF-ToN-IoT-v2":
                runner = run_nf_ton_finetune
            elif dataset_id == "NF-CSE-CIC-IDS2018-v2":
                runner = run_nf_cse_finetune
            else:
                raise ValueError(
                    f"unsupported finetune dataset_id: {dataset_id}"
                )
            summary = runner(
                config,
                repo_dir=repo_dir,
                config_path=config_path,
                resume_from=resume_from,
                initialize_from=initialize_from,
                allow_cross_dataset_init=allow_cross_dataset_init,
            )
        elif args.command == "pretrain":
            from flowfm.training.multisource_pretraining import (
                MULTISOURCE_DATASET_ID,
                run_multisource_pretrain,
            )
            from flowfm.training.pretraining import run_nf_unsw_pretrain

            runner = (
                run_multisource_pretrain
                if config.data["dataset_id"] == MULTISOURCE_DATASET_ID
                else run_nf_unsw_pretrain
            )
            summary = runner(
                config,
                repo_dir=repo_dir,
                config_path=config_path,
                resume_from=resume_from,
            )
        else:
            raise ValueError(
                "train execution is currently supported only by pretrain or finetune"
            )
        return {
            "command": args.command,
            "config": str(config_path),
            "config_hash": config.config_hash,
            "device": config.runtime["device"],
            "experiment": config.experiment["name"],
            "output_root": config.experiment["output_root"],
            **summary,
        }
    context, status = _publish_validated_only_run(
        config,
        args.command,
        repo_dir,
    )
    return {
        "command": args.command,
        "config": str(config_path),
        "config_hash": config.config_hash,
        "device": config.runtime["device"],
        "experiment": config.experiment["name"],
        "output_root": config.experiment["output_root"],
        "run_dir": str(context.run_dir),
        "schema_version": config.schema["version"],
        "status": status["status"],
        "training_started": status["training_started"],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = dispatch(args)
    except (OSError, ImportError, RuntimeError, ValueError) as exc:
        print(f"flowfm: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0
