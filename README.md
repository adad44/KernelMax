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
> KernelMax is currently a structure-only scaffold, so there is not yet a runnable installation or benchmark command.

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
