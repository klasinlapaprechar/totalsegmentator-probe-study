"""Linear / MLP probes on frozen embedding matrices (public smoke)."""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

def run_probes(X: np.ndarray, y: np.ndarray, seed: int = 0) -> dict:
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
    out = {}
    # intensity-like baseline: mean embedding scalar threshold via logistic on 1-D
    base = LogisticRegression(max_iter=200).fit(Xtr.mean(1, keepdims=True), ytr)
    out["intensity_proxy_bal_acc"] = float(balanced_accuracy_score(yte, base.predict(Xte.mean(1, keepdims=True))))
    lin = LogisticRegression(max_iter=500).fit(Xtr, ytr)
    out["linear_bal_acc"] = float(balanced_accuracy_score(yte, lin.predict(Xte)))
    mlp = MLPClassifier(hidden_layer_sizes=(64,), max_iter=400, random_state=seed)
    mlp.fit(Xtr, ytr)
    out["mlp_bal_acc"] = float(balanced_accuracy_score(yte, mlp.predict(Xte)))
    return out
