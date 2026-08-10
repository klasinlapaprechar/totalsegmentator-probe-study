#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.probes import run_probes

def main():
    rng = np.random.default_rng(0)
    n, d = 200, 128
    y = rng.integers(0, 2, size=n)
    X = rng.normal(size=(n, d))
    X[y == 1, :8] += 1.2  # plant a linear signal
    metrics = run_probes(X, y)
    print("smoke ok — synthetic embeddings")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}")

if __name__ == "__main__":
    main()
