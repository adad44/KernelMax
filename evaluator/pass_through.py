"""Trusted adapter diagnostics for an injected orchestrator evaluator interface.

The frozen submission supplies provenance only. Its code is never imported or
executed. These reports cannot establish candidate correctness or acceptance.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, replace

from evaluator.correctness_suite import run_correctness_suite
from evaluator.schemas import (
    CandidateIdentity,
    EnvironmentRecord,
    EvaluationReport,
    ExperimentContract,
    RunStatus,
)
from orchestrator.candidate_manager import FrozenCandidate, verify_frozen_candidate
from orchestrator.state_machine import RunIdentity
from target.adapter import TrustedExpertAdapter


@dataclass(frozen=True, slots=True)
class TrustedPassThroughEvaluator:
    """Bind trusted reference/adapter callables outside the submission workspace.

    Call as evaluator(run_identity, frozen_candidate, contract). The optional
    on_diagnostic_passed(report) hook receives a scoped diagnostic report, NOT a
    candidate CORRECTNESS_PASSED event or permission to start benchmarking.
    Native bindings/model hashes/runtime provenance remain caller responsibilities.
    environment=None is explicit missing evidence, not an invented host record.
    """

    reference: Callable
    adapter: TrustedExpertAdapter
    environment: EnvironmentRecord | None = None

    def __post_init__(self) -> None:
        if not callable(self.reference):
            raise TypeError("reference must be an explicitly trusted callable")
        if not isinstance(self.adapter, TrustedExpertAdapter):
            raise TypeError("adapter must be a TrustedExpertAdapter")
        if self.environment is not None and not isinstance(
            self.environment, EnvironmentRecord
        ):
            raise TypeError("environment must be an EnvironmentRecord or None")

    def __call__(
        self,
        run_identity: RunIdentity,
        frozen_candidate: FrozenCandidate,
        contract: ExperimentContract,
        *,
        on_diagnostic_passed: Callable[[EvaluationReport], None] | None = None,
    ) -> EvaluationReport:
        """Verify provenance/integrity, retain all checks, and return no verdict.

        Bad supplied identities/configuration raise before trusted experts run.
        Snapshot failures return failed reports. Suite execution errors retain
        all per-layer diagnostics. A completed numerical failure is a completed
        diagnostic, not an automatically assigned REJECTED verdict. Interruptions
        during the suite propagate to the controller, as in the existing suite.
        """
        if not isinstance(run_identity, RunIdentity):
            raise TypeError("run_identity must be a RunIdentity")
        if not isinstance(frozen_candidate, FrozenCandidate):
            raise TypeError("frozen_candidate must be a FrozenCandidate")
        if not isinstance(contract, ExperimentContract):
            raise TypeError("contract must be a validated ExperimentContract")
        if on_diagnostic_passed is not None and not callable(on_diagnostic_passed):
            raise TypeError("on_diagnostic_passed must be callable or None")

        candidate = CandidateIdentity(
            frozen_candidate.candidate_id,
            frozen_candidate.implementation_type,
            frozen_candidate.sha256,
        )
        if run_identity.candidate != candidate:
            raise ValueError("run identity does not match frozen candidate metadata")
        if run_identity.contract_sha256 != contract.sha256:
            raise ValueError("run identity does not match supplied contract hash")
        if (
            candidate.implementation_type
            not in contract.agent.allowed_implementation_types
        ):
            raise ValueError("candidate implementation is forbidden by the contract")

        report = EvaluationReport(
            run_id=run_identity.run_id,
            candidate=candidate,
            contract_sha256=contract.sha256,
            environment=self.environment,
            correctness=(),
            performance=(),
            status=RunStatus.COMPLETED,
            errors=(),
            verdict=None,
        )
        try:
            verify_frozen_candidate(frozen_candidate)
        except Exception as exc:
            return self._failed(report, "snapshot before diagnostics", exc)

        try:
            suite = run_correctness_suite(
                self.reference,
                self.adapter,
                policy=contract.correctness,
                hidden_size=self.adapter.hidden_size,
                num_experts=self.adapter.num_experts,
                mx=self.adapter.mx,
            )
        except Exception as exc:
            return self._failed(report, "suite setup/execution", exc)

        results = []
        for result in suite.correctness:
            detail = json.loads(result.detail)
            detail.update(
                evaluation_scope="trusted_pass_through",
                input_mode="numpy_fixture" if self.adapter.mx is None else "mlx_bf16",
                candidate_code_executed=False,
                candidate_correctness_established=False,
                performance_measured=False,
            )
            results.append(replace(result, detail=json.dumps(detail, sort_keys=True)))
        report = replace(
            report,
            correctness=tuple(results),
            errors=suite.errors,
            status=RunStatus.FAILED if suite.errors else RunStatus.COMPLETED,
        )
        try:
            verify_frozen_candidate(frozen_candidate)
        except Exception as exc:
            return self._failed(report, "snapshot after diagnostics", exc)

        if suite.passed and on_diagnostic_passed is not None:
            try:
                on_diagnostic_passed(report)
            except KeyboardInterrupt as exc:
                return replace(
                    report,
                    status=RunStatus.INTERRUPTED,
                    errors=(*report.errors, f"diagnostic callback interrupted: {exc}"),
                )
            except Exception as exc:
                return self._failed(report, "diagnostic callback", exc)
            try:
                verify_frozen_candidate(frozen_candidate)
            except Exception as exc:
                return self._failed(report, "snapshot after diagnostic callback", exc)
        run_identity.validate_report(report)
        return report

    @staticmethod
    def _failed(
        report: EvaluationReport, stage: str, exc: Exception
    ) -> EvaluationReport:
        return replace(
            report,
            status=RunStatus.FAILED,
            errors=(
                *report.errors,
                f"trusted pass-through diagnostic {stage}: {type(exc).__name__}: {exc}",
            ),
        )
