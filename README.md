# KernelMax
Uses an agentic auto research loop to find and verify faster MLX/Metal inference on a local model
KernelMaxxing/
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
