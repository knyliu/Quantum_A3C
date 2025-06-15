# Quantum A3C

This repository contains a simple implementation of an asynchronous advantage actor-critic (A3C) agent that uses a variational quantum circuit (VQC) implemented with PennyLane and PyTorch. The original example trains the agent on a MiniGrid environment.

## Financial Extension

The repository now also includes an experimental **Quantum-Inspired Reinforcement Learning** example for financial data. The new module `financial_env.py` implements a lightweight trading environment using randomly generated price data. The training script `financial_a3c_vqc.py` adapts the A3C algorithm to this environment while keeping the quantum circuit based policy and value networks.

These additions serve as a starting point for exploring quantum-inspired reinforcement learning strategies in algorithmic trading scenarios.

### Running the Financial Example

Install the core dependencies before running the financial script:

- `torch`
- `pennylane`
- `numpy`
- `matplotlib`

Launch the training loop with:

```bash
python financial_a3c_vqc.py
```

The environment uses randomly generated price data by default. You can adapt
`financial_env.py` to feed real market prices for more realistic experiments.
