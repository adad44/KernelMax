"""Prepare the optional third snapshot by composing byte-identical sources.

No model execution. Invoke only after the controller decides both placements
are worth combining. The existing resident worker enforces the three-ID cap.
"""

import ast
import hashlib
import json
from pathlib import Path
import shutil
import sys

WORK = Path(__file__).resolve().parent
ROOT = WORK.parent.parent / "kernelmax-first-loop"
sys.path.insert(0, str(ROOT))
from orchestrator.candidate_manager import (
    create_candidate_id,
    freeze_candidate,
    validate_candidate,
    verify_frozen_candidate,
    FrozenCandidate,
)


def main():
    target = WORK / "combined-frozen.json"
    if target.exists():
        raise ValueError("Third snapshot already prepared; do not allocate another ID")
    parents = {}
    for kind, filename in (
        ("vocabulary", "vocabulary-frozen.json"),
        ("moe", "moe-frozen.json"),
    ):
        info = json.loads((WORK / filename).read_text())
        c = validate_candidate(info["directory"], Path(info["directory"]).parent)
        frozen = FrozenCandidate(
            **vars(c), sha256=info["sha256"], file_sha256=info["file_sha256"]
        )
        verify_frozen_candidate(frozen)
        parents[kind] = frozen
    identifier = create_candidate_id()
    directory = ROOT / "candidates" / identifier
    directory.mkdir()
    provenance = {}
    for kind, frozen in parents.items():
        child = directory / kind
        child.mkdir()
        hashes = {}
        for source in frozen.directory.iterdir():
            if source.is_file() and source.suffix in (".py", ".metal"):
                destination = child / source.name
                destination.write_bytes(source.read_bytes())
                actual = hashlib.sha256(destination.read_bytes()).hexdigest()
                if actual != frozen.file_sha256[source.name]:
                    raise ValueError("Vendored source hash mismatch")
                hashes[source.name] = actual
        provenance[kind] = {
            "candidate_id": frozen.candidate_id,
            "bundle_sha256": frozen.sha256,
            "files": hashes,
        }
    implementation = (WORK / "combined_implementation.py.in").read_text()
    ast.parse(implementation)
    (directory / "implementation.py").write_text(implementation)
    wrapper = """import importlib.util
from pathlib import Path
import sys
_name=__name__+'_combined'
_spec=importlib.util.spec_from_file_location(_name,Path(__file__).with_name('implementation.py'))
_module=importlib.util.module_from_spec(_spec);sys.modules[_name]=_module
_spec.loader.exec_module(_module)
prepare=_module.prepare
"""
    ast.parse(wrapper)
    (directory / "wrapper.py").write_text(wrapper)
    manifest = {
        "candidate_id": identifier,
        "name": "Vocabulary layout plus fused MoE decode block",
        "implementation_type": "mlx",
        "description": "Prepare(model) composes the two frozen placements without new arithmetic: lossless BF16 vocabulary projection only, plus three-stage post-attention normalization/routing/MXFP4expert/reduction/residual pipeline across24layers. Unchanged native prefill fallbacks, weights, precision and fresh caches. Declare byte-identical parent source provenance. Full-model numerical/route/token/cache checks and fresh strict matched confirmation required; no speedup/acceptance claim.",
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (directory / "source_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    (directory / "README.md").write_text(
        "# Combined full-model placement\n\nThis third sprint snapshot composes byte-identical vocabulary and MoE parent\nsources. No new numerical operation is introduced by composition. The trusted\ncontroller checks their interaction on complete-model outputs before timing.\nSidecar allocation remains explicit and native prefill fallbacks remain active.\nThis file contains no correctness, performance or acceptance verdict.\n"
    )
    frozen = freeze_candidate(
        directory, frozen_root=WORK / "frozen", candidates_root=ROOT / "candidates"
    )
    verify_frozen_candidate(frozen)
    value = {
        "candidate_id": frozen.candidate_id,
        "sha256": frozen.sha256,
        "file_sha256": dict(frozen.file_sha256),
        "directory": str(frozen.directory),
        "parent_provenance": provenance,
        "sprint_slot": 3,
        "template_sha256": hashlib.sha256(
            (WORK / "combined_implementation.py.in").read_bytes()
        ).hexdigest(),
    }
    target.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "candidate_id": frozen.candidate_id,
                "sha256": frozen.sha256,
                "slot": 3,
                "native_execution_started": False,
            }
        )
    )


if __name__ == "__main__":
    main()
