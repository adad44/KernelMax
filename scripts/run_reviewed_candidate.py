"""Single-candidate review/verification driver, independent of session/CLI work."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluator.candidate_execution import ReviewedCandidateEvaluator, WorkerRuntime
from evaluator.candidate_protocol import file_digest, read_json
from evaluator.schemas import load_contract
from orchestrator.candidate_manager import freeze_candidate
from orchestrator.experiment_budget import ExperimentBudget
from orchestrator.journal import ExperimentJournal
from orchestrator.run import run_candidate


def native_runtime(root: Path, contract) -> WorkerRuntime:
    """Use pinned download metadata, never infer checkpoint approval from its name.

    Independently verify the checkout with `hf cache verify` before this command.
    Shards retain their remote LFS SHA256. Workers check bytes unless the caller
    explicitly supplies a controller-owned, identity-bound verification receipt.
    """
    root = root.resolve()
    index = read_json(root / "model.safetensors.index.json")
    shards = set(index["weight_map"].values())
    support = {
        "config.json",
        "model.safetensors.index.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
    }
    manifest = {}
    for name in sorted(shards | support):
        metadata = root / ".cache/huggingface/download" / (name + ".metadata")
        revision, digest, *_ = metadata.read_text().splitlines()
        if revision != contract.model.revision:
            raise ValueError(
                f"checkpoint download metadata has another revision: {name}"
            )
        manifest[name] = digest if name in shards else file_digest(root / name)
    config = read_json(root / "config.json")
    return WorkerRuntime(
        mode="mlx_native",
        python=Path(sys.executable),
        hidden_size=config["hidden_size"],
        num_experts=config["num_local_experts"],
        model_path=root,
        model_revision=contract.model.revision,
        model_files=manifest,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "kernelmaxxing.yaml",
    )
    parser.add_argument(
        "--prepare-review",
        action="store_true",
        help="Freeze and print the hash; never execute",
    )
    authorization = parser.add_mutually_exclusive_group()
    authorization.add_argument(
        "--approved-sha256",
        action="append",
        default=[],
        help="Human/controller approval of an exact frozen hash",
    )
    authorization.add_argument(
        "--session-authorization", type=Path,
        help="Controller-owned standing user authorization; freezes/authorizes once without another prompt",
    )
    parser.add_argument(
        "--checkpoint-cache", type=Path,
        help="Reuse controller byte verification only while checkpoint pins and file identities match",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--fixture",
        action="store_true",
        help="Explicit NumPy expert fixtures, not model evidence",
    )
    mode.add_argument("--native-model", type=Path)
    parser.add_argument(
        "--expert-factory",
        help="Optional trusted module:symbol factory(model, mx=...) -> TrustedExpertAdapter",
    )
    parser.add_argument("--timeout-seconds", type=float, default=600)
    args = parser.parse_args(argv)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    candidate = args.candidate.resolve()
    if args.prepare_review:
        frozen = freeze_candidate(
            candidate,
            frozen_root=args.work_dir / "review" / uuid.uuid4().hex,
            candidates_root=candidate.parent,
        )
        print(
            json.dumps(
                {
                    "snapshot": str(frozen.directory),
                    "sha256": frozen.sha256,
                    "executed": False,
                }
            )
        )
        return 0
    if not (args.approved_sha256 or args.session_authorization) or not (args.fixture or args.native_model):
        parser.error(
            "execution requires an explicit mode and approved hash or standing session authorization"
        )
    if args.checkpoint_cache and not args.native_model:
        parser.error("checkpoint verification reuse requires --native-model")
    if args.checkpoint_cache and (
        args.checkpoint_cache.resolve().is_relative_to(candidate)
        or args.checkpoint_cache.resolve().is_relative_to(Path(__file__).resolve().parents[1] / "candidates")
    ):
        parser.error("checkpoint cache must be controller-owned, outside candidates")
    contract = load_contract(args.contract)
    if args.session_authorization:
        from scripts.loop_control import load_authorization

        session = load_authorization(args.session_authorization, contract)
        if args.native_model:
            if candidate.parent != Path(__file__).resolve().parents[1] / "candidates":
                parser.error("native standing authorization is limited to this repository's candidates")
            if args.native_model.resolve() != Path(session["model_path"]).resolve():
                parser.error("native model path disagrees with standing authorization")
    runtime = (
        native_runtime(args.native_model, contract)
        if args.native_model
        else WorkerRuntime()
    )
    if args.expert_factory:
        runtime = replace(runtime, expert_factory=args.expert_factory)
    if args.checkpoint_cache:
        from evaluator.checkpoint_verification import verify_checkpoint

        receipt, reused = verify_checkpoint(
            runtime.model_path, runtime.model_revision, runtime.model_files, args.checkpoint_cache,
        )
        write_path = args.work_dir / ("checkpoint-verification-" + uuid.uuid4().hex + ".json")
        from evaluator.candidate_protocol import write_json

        write_json(write_path, {"receipt": receipt, "reused": reused})
        runtime = replace(runtime, checkpoint_verification=receipt)
    options = {"runtime": runtime, "timeout_seconds": args.timeout_seconds}
    if args.session_authorization:
        from scripts.loop_control import SessionEvaluator

        evaluator = SessionEvaluator(args.session_authorization, args.work_dir / "artifacts", **options)
    else:
        evaluator = ReviewedCandidateEvaluator(
            frozenset(args.approved_sha256), args.work_dir / "artifacts", **options,
        )
    budget = ExperimentBudget(max_attempts=1)
    result = run_candidate(
        candidate,
        evaluator=evaluator,
        contract=contract,
        budget=budget,
        journal=ExperimentJournal(args.work_dir / "journal" / uuid.uuid4().hex),
        frozen_root=args.work_dir / "runs" / uuid.uuid4().hex,
        candidates_root=candidate.parent,
    )
    report = result.report
    summary = {
        "phase": result.state.phase.value if result.state else None,
        "snapshot_sha256": result.frozen_candidate.sha256
        if result.frozen_candidate
        else None,
        "attempts_started": budget.attempts_started,
        "error": result.error,
        "report_status": report.status.value if report else None,
        "correctness": {row.case_name: row.passed for row in report.correctness}
        if report
        else {},
        "errors": list(report.errors) if report else [],
        "verdict": None,
        "performance_measured": False,
    }
    print(json.dumps(summary, sort_keys=True))
    return (
        0
        if report
        and not report.errors
        and len(report.correctness) == len(contract.correctness.required_cases)
        and all(row.passed for row in report.correctness)
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
