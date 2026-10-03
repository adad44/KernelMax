# AI/ML Club - KernelMax

**Academic Year:** 2026-2027 Fall Semester

**Project Duration:** September 2026 - December 2026

## Project Description

KernelMax uses an agentic auto-research loop to find and verify faster MLX/Metal inference on a local model. The project is building a controlled workflow that proposes candidate optimizations, checks correctness, benchmarks performance, and records whether each experiment is accepted or rejected.

This repository is currently an initial project scaffold. The implementation and dependency files will be filled in as development progresses.

**Key Objectives:**

- Establish a reproducible local inference baseline.
- Profile MLX/Metal workloads and identify performance bottlenecks.
- Generate optimization candidates within a restricted candidate workspace.
- Verify every candidate with protected correctness and performance checks.

## Lead Contact Information

**Project Lead:** Alan Diaz

**GitHub:** [@adad44](https://github.com/adad44)

## Contributors

| Name | Role | Email | GitHub |
| --- | --- | --- | --- |
| Alan Diaz | Project Lead | — | [@adad44](https://github.com/adad44) |
| Elina | Member | [elina.saini@sjsu.edu](mailto:elina.saini@sjsu.edu) | [@colorfulelina](https://github.com/colorfulelina) |
| Xavier | Member | [xavier.e.rodriguez@sjsu.edu](mailto:xavier.e.rodriguez@sjsu.edu) | [@xaviererodriguez-design](https://github.com/xaviererodriguez-design) |

## Project Kanban Board

A public project board has not been linked yet.

## Quick Start Guide

### Prerequisites

- Git
- Python
- An Apple silicon Mac for the planned MLX/Metal workflow

### Installation Guide

1. Clone the repository:

   ```bash
   git clone https://github.com/adad44/KernelMax.git
   cd KernelMax
   ```

2. Follow the setup and run instructions once the project dependencies and implementation are added.

> [!NOTE]
> KernelMax is still mostly a scaffold, so there is not yet a runnable model benchmark command. The offline target helpers below can be used and tested now.

### Offline fixed workload

The first implemented pieces in `target/` prepare fake inputs for a future
language-model benchmark. A workload is the set of inputs used for a test;
benchmarking will later measure how quickly a model handles those inputs.
These helpers do not run inference or measure performance.

- `target/model_loader.py` defines `LoadedModel`, a record holding a model and
  its tokenizer (the object that converts text to token IDs). Its `load_model`
  function calls a required, caller-supplied loader once. The loader returns
  `(model, tokenizer)`; tests can return fake objects. There is no default ML
  backend, download, or model dependency. Any future backend's behavior is the
  caller's responsibility.
- `target/workload.py` defines `generate_workload(seed=42, vocab_size=32000)`.
  It creates synthetic integer token IDs using its own seeded random generator.
  The same seed and vocabulary size produce the same inputs. No tokenizer or
  model is needed, and global random state is left unchanged.
- Each immutable `WorkloadCase` holds `input_ids` and `max_new_tokens=128`.
  Its `batch_size` property counts input rows, `prompt_length` counts tokens in
  the row, and `input_shape` returns those two dimensions. Exactly three cases
  are generated, with shapes `(1, 512)`, `(1, 2048)`, and `(1, 4096)`. The `1`
  means one prompt per batch; `128` limits future output tokens.
- `tests/test_model_loader.py` checks the interface using fake objects.
  `tests/test_workload.py` checks the fixed cases, shapes, token bounds,
  reproducibility, and invalid seed/vocabulary settings.

Python 3.9 or newer is sufficient for these standard-library helpers:

```python
from target.model_loader import load_model
from target.workload import generate_workload

# These objects stand in for a model and tokenizer; no model files are read.
loaded = load_model(loader=lambda: (object(), object()))
for case in generate_workload():
    print(case.input_shape, case.max_new_tokens)
```

With `pytest` available in your Python environment, run the offline tests from
the repository root:

```bash
python3 -m pytest -q
```

The other scaffold modules and test files remain placeholders.

## Technology Stack

- **Programming Language:** Python
- **ML/AI:** MLX and Metal (planned)
- **Configuration:** YAML
- **Development:** `uv` project files (setup pending)
- **Version Control:** Git and GitHub

## License

No license file has been added yet.

## Acknowledgments

- San Jose State University AI/ML Club
- The MLX and Apple silicon developer communities

---

**Last Updated:** September 29, 2026

**Next Review:** To be determined

---

This README follows the AI/ML Club standard project format.

## Repository Structure

```text
KernelMax/
├── README.md
├── AGENTS.md
├── kernelmaxxing.yaml
├── pyproject.toml
├── uv.lock
│
├── orchestrator/
│   ├── run.py
│   ├── state_machine.py
│   ├── experiment_budget.py
│   ├── candidate_manager.py
│   └── prompts/
│       ├── profiler.md
│       ├── mlx_optimizer.md
│       ├── metal_optimizer.md
│       └── failure_analyst.md
│
├── target/
│   ├── model_loader.py
│   ├── workload.py
│   ├── reference_moe.py
│   └── adapter.py
│
├── evaluator/                   # protected from optimization agents
│   ├── correctness.py
│   ├── benchmark.py
│   ├── environment.py
│   ├── verdict.py
│   └── schemas.py
│
├── candidates/                  # only agent-writable area
│   ├── README.md
│   └── candidate-template/
│       ├── manifest.json
│       ├── implementation.py
│       └── wrapper.py
│
├── tests/
│   ├── test_correctness.py
│   ├── test_contract.py
│   ├── test_candidate_boundary.py
│   └── test_verdict.py
│
├── reports/                     # evaluator-written
│   └── .gitkeep
│
└── .kernelmaxxing/              # evaluator-owned runtime state
    ├── state.json
    ├── journal.jsonl
    └── model-files.sha256
```
