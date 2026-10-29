# Run catalog imported 2026-10-28

The [manifest](manifest.json) catalogs all 7,003 original run files. Every entry
records its workspace-relative path, size, SHA256, permissions, and storage
location. [inventory.tsv](inventory.tsv) breaks down retained and external files
by input directory. [external-artifacts.tsv](external-artifacts.tsv) lists the
artifacts that are kept outside Git.

Git retains 4,722 files: candidate and controller source snapshots, configurations,
summaries, timing samples, environment details, hashes, and failure records.
All outcomes remain represented, including interrupted, failed, inconclusive,
and exploratory attempts. Files under `files/` preserve their original bytes.

The other 2,281 files remain in local storage: numerical arrays, generated weight
layouts, compiled binaries, compressed traces, and bulk worker tensor inputs and
outputs. They are **not hosted by GitHub**. An external manifest entry's
`artifact_path` is relative to the recorded `artifact_root`:

```text
/Users/alandiaz/Downloads/KernelMaxxxing
```

For example, an external `work/.../correctness-512-128.npz` entry refers to the
original file under that workspace root. No original run files were deleted.
A separate full backup is also retained locally at:

```text
/Users/alandiaz/Downloads/KernelMaxxxing/local-run-artifacts/run-archive-2026-10-28
```

This is a local availability record, not a shared download location. Sharing
external artifacts requires copying them into separate artifact storage and
providing that location to collaborators.

## Coverage

The catalog includes all `baseline-results/`, `baseline-tools/`, the original
checkout's `KernelMax/reports/baselines/`, and all eleven experiment directories
under `work/`. These cover the first three optimization runs, the optimization
loop, reviewed-native execution, decode bottleneck/search/exploration, both
sprint rounds, and the changed-residency confirmation.

Python bytecode, tool caches, and `.DS_Store` files are omitted and listed in
`excluded_cache_files`. Unrelated slide/resume work, installed runtimes, and the
external source checkpoint are outside the import scope.

## Verify

Run from the repository root with Python 3.11 or later. Only the standard
library is required:

```sh
# Check only the evidence stored in Git; report external files as unchecked.
python3 -B scripts/archive_runs.py verify experiments/run-archive-2026-10-28

# Also check every external artifact against the manifest hashes.
python3 -B scripts/archive_runs.py verify experiments/run-archive-2026-10-28 --artifact-root /Users/alandiaz/Downloads/KernelMaxxxing
```

Use another `--artifact-root` after copying the original relative paths to a
different location. Historical absolute paths inside run records are preserved.
Verification does not execute experiments.

## Interpretation

Each attempt retains its original contract, source identity, scope, and status.
Read its own results, states, and audits. Historical baselines and exploratory
screens are not fresh confirmation evidence. The latest round's
[resident state](files/work/decode-sprint-r2-20261028/resident-state.json) records
a failed final checkpoint guard and `five_percent_improvement_verified: false`.
The earlier changed-residency confirmation also stopped without candidate
confirmation pairs. This catalog does not issue a verdict, establish a speedup,
or change acceptance gates.
