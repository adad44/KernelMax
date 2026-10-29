"""One-command local correctness stage using a controller-owned loop session.

Timing deliberately stays behind the correctness report and its existing
environment/noise checks. This command does not grade or publish speedups.
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluator.schemas import load_contract
from scripts.loop_control import ROOT, load_authorization
from scripts.run_reviewed_candidate import main as run_correctness


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=1200)
    args = parser.parse_args(argv)
    contract = load_contract(ROOT / "kernelmaxxing.yaml")
    session = load_authorization(args.session, contract)
    # An attempt directory is append-only evidence, not a resumable success cache.
    args.work_dir.mkdir(parents=True, exist_ok=False)
    return run_correctness([
        "--candidate", str(args.candidate), "--work-dir", str(args.work_dir),
        "--native-model", session["model_path"],
        "--session-authorization", str(args.session),
        "--checkpoint-cache", session["checkpoint_cache"],
        "--timeout-seconds", str(args.timeout_seconds),
    ])


if __name__ == "__main__":
    raise SystemExit(main())
