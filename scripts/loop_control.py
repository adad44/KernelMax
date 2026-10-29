"""Standing authorization handoff; never read execution policy from a candidate."""

from pathlib import Path

from evaluator.candidate_execution import ReviewedCandidateEvaluator, _protected_files
from evaluator.candidate_protocol import file_digest, read_json, write_json


ROOT = Path(__file__).resolve().parents[1]


def controller_sources():
    return _protected_files([
        Path(__file__), ROOT / "scripts/run_reviewed_candidate.py",
        ROOT / "scripts/run_candidate_loop.py",
    ])


def load_authorization(path, contract):
    path = Path(path).resolve()
    if path.is_relative_to(ROOT / "candidates"):
        raise ValueError("session authorization must be controller-owned, outside candidates")
    data = read_json(path)
    if (
        data.get("schema") != 1
        or data.get("authorized") is not True
        or data.get("repository") != str(ROOT)
        or data.get("contract_sha256") != contract.sha256
        or data.get("model_revision") != contract.model.revision
        or not isinstance(data.get("allowed_operations"), list)
        or "candidate_correctness" not in data.get("allowed_operations", [])
        or data.get("acceptance_enabled") is not False
        or data.get("protected_sources") != controller_sources()
        or not isinstance(data.get("authorization_basis"), str)
        or not data["authorization_basis"].strip()
    ):
        raise ValueError("standing authorization is missing, stale or outside this experiment")
    return data


class SessionEvaluator:
    """Authorize the controller's single frozen snapshot using standing consent.

    The orchestrator freezes once; this adapter checks session scope before
    passing that exact hash to the unchanged evaluator approval interface.
    No candidate code is imported here. This is authorization, not a claim that
    the human reviewed each snapshot, and grants no acceptance/publication.
    """

    supports_candidate_correctness = True

    def __init__(self, authorization, artifact_root, **options):
        self.authorization = Path(authorization).resolve()
        self.authorization_digest = file_digest(self.authorization)
        self.artifact_root = Path(artifact_root)
        self.options = options

    def __call__(self, identity, frozen, contract, **callbacks):
        if file_digest(self.authorization) != self.authorization_digest:
            raise ValueError("standing authorization changed during preparation")
        load_authorization(self.authorization, contract)
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        record = self.artifact_root / (identity.run_id + "-authorization.json")
        if record.exists():
            raise ValueError("authorization run ID already used")
        write_json(record, {
            "authorization": str(self.authorization),
            "authorization_sha256": self.authorization_digest,
            "candidate_id": frozen.candidate_id,
            "snapshot_sha256": frozen.sha256,
            "contract_sha256": contract.sha256,
            "run_id": identity.run_id,
            "human_reviewed_exact_snapshot": False,
            "authorization_kind": "standing_user_instruction",
            "scope": "correctness only; timing requires a separate valid diagnostic",
        })
        evaluator = ReviewedCandidateEvaluator(
            frozenset({frozen.sha256}), self.artifact_root,
            protected_paths=(self.authorization, *[
                ROOT / "scripts" / name for name in (
                    "loop_control.py", "run_reviewed_candidate.py", "run_candidate_loop.py"
                )
            ]),
            **self.options,
        )
        return evaluator(identity, frozen, contract, **callbacks)
