"""Command-line entry point for the offline orchestrator demonstration."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Sequence

from evaluator.pass_through import TrustedPassThroughEvaluator
from evaluator.schemas import ExperimentContract, load_contract
from orchestrator.candidate_manager import create_candidate_id
from orchestrator.experiment_budget import ExperimentBudget
from orchestrator.journal import ExperimentJournal
from orchestrator.session import run_session
from target.adapter import TrustedExpertAdapter


ROOT = Path(__file__).resolve().parents[1]


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _fake_expert(activations, selected_expert_ids):
    """Deterministic NumPy-only expert used by the no-download demo."""

    return activations[:, :, None, :] + selected_expert_ids[:, :, :, None].astype(
        activations.dtype
    )


class _DemoEvaluator:
    """Inject one synthetic failure, then delegate to trusted diagnostics."""

    def __init__(self, trusted: TrustedPassThroughEvaluator, failed_candidate_id: str):
        self.trusted = trusted
        self.failed_candidate_id = failed_candidate_id
        self.failure_sent = False

    def __call__(self, run_identity, frozen_candidate, contract, *, on_diagnostic_passed=None):
        if (
            frozen_candidate.candidate_id == self.failed_candidate_id
            and not self.failure_sent
        ):
            self.failure_sent = True
            raise RuntimeError("synthetic demo evaluator failure")
        return self.trusted(
            run_identity,
            frozen_candidate,
            contract,
            on_diagnostic_passed=on_diagnostic_passed,
        )


def _write_fake_submission(directory: Path, candidate_id: str, name: str) -> None:
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "candidate_id": candidate_id,
                "name": name,
                "implementation_type": "mlx",
                "description": "Offline provenance fixture; candidate code is never run",
            }
        ),
        encoding="utf-8",
    )
    # Deliberately not executable by the trusted pass-through evaluator.
    for filename in ("wrapper.py", "implementation.py"):
        (directory / filename).write_text(
            'raise AssertionError("candidate code must not be executed")\n',
            encoding="utf-8",
        )


def run_demo(*, max_attempts: int = 2) -> str:
    """Run a fully offline multi-submission example and return its summary."""

    contract: ExperimentContract = load_contract(ROOT / "kernelmaxxing.yaml")
    with tempfile.TemporaryDirectory(prefix="kernelmax-session-") as temporary:
        root = Path(temporary)
        candidates_root = root / "candidates"
        candidates_root.mkdir()
        first_id = create_candidate_id()
        second_id = create_candidate_id()
        third_id = create_candidate_id()
        first = candidates_root / first_id
        second = candidates_root / second_id
        third = candidates_root / third_id
        _write_fake_submission(first, first_id, "Synthetic failure example")
        _write_fake_submission(second, second_id, "Trusted diagnostic example")
        _write_fake_submission(third, third_id, "Budget exhaustion example")

        def reference(layer_index, activations, selected_expert_ids):
            return _fake_expert(activations, selected_expert_ids)

        adapter = TrustedExpertAdapter(
            {layer: _fake_expert for layer in (0, 12, 23)},
            hidden_size=8,
            num_experts=32,
        )
        trusted_evaluator = TrustedPassThroughEvaluator(reference, adapter)
        evaluator = _DemoEvaluator(trusted_evaluator, first_id)
        budget = ExperimentBudget(max_attempts=max_attempts)
        journal = ExperimentJournal(root / "controller" / "journal")
        result = run_session(
            [first, first, second, third],
            evaluator=evaluator,
            contract=contract,
            budget=budget,
            journal=journal,
            frozen_root=root / "controller" / "frozen",
            candidates_root=candidates_root,
            session_id="offline-demo",
        )
        return result.summary()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m orchestrator.main",
        description="Run KernelMax's offline multi-submission orchestrator demo.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo = subparsers.add_parser(
        "demo",
        help="use fake experts and the trusted pass-through evaluator",
    )
    demo.add_argument(
        "--max-attempts",
        type=_positive_int,
        default=2,
        help="shared evaluator-attempt budget (default: 2)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "demo":
        print(run_demo(max_attempts=args.max_attempts))
        return 0
    return 2  # argparse's required subcommand makes this branch unreachable.


if __name__ == "__main__":
    raise SystemExit(main())
