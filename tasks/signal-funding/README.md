# signal-funding

Build contrarian signal from funding rate

**Difficulty:** medium
**Category:** quant-research
**Tags:** signal, funding, contrarian, mean-reversion, crypto

## Quick Start

Restore the data bundle from the repository root before running this task; see `DATA.md`.

```bash
# Run starter code (will fail until implemented)
python run_local.py --starter

# Run evaluation only
python run_local.py --eval-only
```

## Task Overview

See [instruction.md](instruction.md) for full task details.

## Directory Structure

```
signal-funding/
├── instruction.md          # Agent-facing task description
├── README.md               # This file
├── task.toml               # Task configuration
├── run_local.py            # Local testing script
├── environment/
│   ├── Dockerfile          # Container definition
│   ├── docker-compose.yaml # Docker compose config
│   ├── data/
│   │   └── train.parquet   # Training data
│   └── starter_code/
│       └── code.py       # Template to implement
├── tests/
│   ├── test.parquet        # Held-out test data
│   ├── test.sh             # Test script
│   ├── test_outputs.py     # Pytest assertions
│   └── evaluate.py         # Evaluation logic
└── output/
    └── .gitkeep            # Output directory
```

## Evaluation Metrics

- **Information Coefficient (IC)**: Spearman correlation between signal and returns
- **IC Positive %**: Consistency of signal direction
- **DK t-stat**: Statistical significance with panel correlation adjustment
