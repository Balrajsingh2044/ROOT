"""Fair-wage engine: a small neural network with Monte-Carlo-dropout uncertainty.

HONEST SCOPE: the network is trained on SYNTHETIC data produced by a made-up wage
formula (see _synthetic_data). It demonstrates the method: a wage RANGE instead of a
single number, where the range reflects model uncertainty. It is NOT a market estimate.

Written in plain NumPy on purpose: no extra dependencies and it trains in about a second.
"""
from pathlib import Path

import numpy as np

WEIGHTS_PATH = Path(__file__).resolve().parent / "wage_model.npz"
CONFIG_VERSION = 1

HIDDEN = 32
DROPOUT = 0.2
EPOCHS = 600
LR = 0.01
MC_PASSES = 200

# Made-up formula (rupees per hour) used ONLY to generate synthetic training data.
BASE, SKILL_W, COMPLEXITY_W, DEMAND_W, NOISE_SD = 80.0, 200.0, 15.0, 10.0, 15.0
Y_OFFSET, Y_SCALE = 100.0, 300.0

DEMAND_LEVELS = {"low": 1, "medium": 3, "high": 5}


def demand_from_label(label) -> int:
    """'low' / 'medium' / 'high' (or a number 1-5) -> an integer 1-5."""
    if isinstance(label, (int, float)):
        return int(min(5, max(1, round(label))))
    return DEMAND_LEVELS.get(str(label).strip().lower(), 3)


def _synthetic_data(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    skill = rng.uniform(0.0, 1.0, n)
    complexity = rng.integers(1, 6, n)
    demand = rng.integers(1, 6, n)
    wage = (BASE + SKILL_W * skill + COMPLEXITY_W * complexity + DEMAND_W * demand
            + rng.normal(0.0, NOISE_SD, n))
    X = np.stack([skill, complexity / 5.0, demand / 5.0], axis=1)
    y = ((wage - Y_OFFSET) / Y_SCALE).reshape(-1, 1)
    return X, y


def _init_params(rng):
    return {
        "W1": rng.normal(0, np.sqrt(2 / 3), (3, HIDDEN)), "b1": np.zeros(HIDDEN),
        "W2": rng.normal(0, np.sqrt(2 / HIDDEN), (HIDDEN, HIDDEN)), "b2": np.zeros(HIDDEN),
        "W3": rng.normal(0, np.sqrt(1 / HIDDEN), (HIDDEN, 1)), "b3": np.zeros(1),
    }


def _forward(p, X, rng=None):
    """rng given -> dropout is active (training and Monte-Carlo inference)."""
    z1 = X @ p["W1"] + p["b1"]
    a1 = np.maximum(z1, 0.0)
    m1 = None
    if rng is not None:
        m1 = (rng.random(a1.shape) >= DROPOUT) / (1.0 - DROPOUT)
        a1 = a1 * m1
    z2 = a1 @ p["W2"] + p["b2"]
    a2 = np.maximum(z2, 0.0)
    m2 = None
    if rng is not None:
        m2 = (rng.random(a2.shape) >= DROPOUT) / (1.0 - DROPOUT)
        a2 = a2 * m2
    out = a2 @ p["W3"] + p["b3"]
    return out, (X, z1, a1, m1, z2, a2, m2)


def _backward(p, cache, dout):
    X, z1, a1, m1, z2, a2, m2 = cache
    g = {"W3": a2.T @ dout, "b3": dout.sum(0)}
    da2 = dout @ p["W3"].T
    if m2 is not None:
        da2 = da2 * m2
    dz2 = da2 * (z2 > 0)
    g["W2"], g["b2"] = a1.T @ dz2, dz2.sum(0)
    da1 = dz2 @ p["W2"].T
    if m1 is not None:
        da1 = da1 * m1
    dz1 = da1 * (z1 > 0)
    g["W1"], g["b1"] = X.T @ dz1, dz1.sum(0)
    return g


def train(seed=0):
    """Train the network on synthetic data (Adam, full batch, MSE loss)."""
    rng = np.random.default_rng(seed)
    X, y = _synthetic_data(seed=seed)
    p = _init_params(rng)
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v = {k: np.zeros_like(val) for k, val in p.items()}
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    for t in range(1, EPOCHS + 1):
        out, cache = _forward(p, X, rng)
        dout = 2.0 * (out - y) / len(X)
        grads = _backward(p, cache, dout)
        for k in p:
            m[k] = beta1 * m[k] + (1 - beta1) * grads[k]
            v[k] = beta2 * v[k] + (1 - beta2) * grads[k] ** 2
            p[k] -= LR * (m[k] / (1 - beta1 ** t)) / (np.sqrt(v[k] / (1 - beta2 ** t)) + eps)
    return p


_PARAMS = None


def _get_params():
    global _PARAMS
    if _PARAMS is not None:
        return _PARAMS
    if WEIGHTS_PATH.exists():
        try:
            data = np.load(WEIGHTS_PATH)
            if int(data["version"]) == CONFIG_VERSION:
                _PARAMS = {k: data[k] for k in ("W1", "b1", "W2", "b2", "W3", "b3")}
                return _PARAMS
        except Exception:
            pass  # unreadable cache: retrain below
    _PARAMS = train()
    try:
        np.savez(WEIGHTS_PATH, version=CONFIG_VERSION, **_PARAMS)
    except OSError:
        pass
    return _PARAMS


def _round5(x):
    return int(round(float(x) / 5.0) * 5)


def predict_wage(skill_score, complexity=3, demand=3, seed=None):
    """Return a wage range (rupees/hour) from the technique score (0-1).

    The range is the 2.5th-97.5th percentile of Monte-Carlo-dropout predictions, so it
    shows the model's own uncertainty. A wide range means the model is unsure.
    """
    skill = float(np.clip(skill_score, 0.0, 1.0))
    complexity = int(np.clip(round(complexity), 1, 5))
    demand = demand_from_label(demand)
    x = np.array([[skill, complexity / 5.0, demand / 5.0]])
    X = np.repeat(x, MC_PASSES, axis=0)
    out, _ = _forward(_get_params(), X, np.random.default_rng(seed))
    wages = out[:, 0] * Y_SCALE + Y_OFFSET
    lo, hi = np.percentile(wages, [2.5, 97.5])
    mean = float(wages.mean())
    lo_r, hi_r, mean_r = max(50, _round5(lo)), _round5(hi), _round5(mean)
    if hi_r <= lo_r:
        hi_r = lo_r + 5
    rel_width = (hi - lo) / max(mean, 1.0)
    confidence = "High" if rel_width < 0.12 else "Medium" if rel_width < 0.30 else "Low"
    return {
        "min": lo_r,
        "max": hi_r,
        "mean": mean_r,
        "text": f"\u20b9{lo_r} - \u20b9{hi_r}/hr",
        "confidence": confidence,
        "method": "Neural network + Monte-Carlo dropout (95% interval), synthetic training data",
    }


if __name__ == "__main__":
    for s in (0.2, 0.5, 0.8, 0.95):
        r = predict_wage(s, 3, 3, seed=1)
        print(s, r["text"], r["confidence"])
