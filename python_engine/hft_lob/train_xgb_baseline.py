"""
train_xgb_baseline.py
=====================
Phase 3 — Supervised XGBoost baseline.

Trains a 3-class classifier (down / flat / up) on rolling LOB features
to predict the *direction* of the mid-price `horizon` seconds ahead.

This is the GATE CHECK for the LOB-HFT v2 stack:
  - test accuracy must exceed 0.36 (better than random 1/3)
  - else PPO training is not worth attempting on this data slice.

Public API
----------
train_baseline(symbols: list[str], test_size: float = 0.2) -> dict
    Returns:
        {
            "accuracy": float,
            "passes_gate": bool,
            "n_train": int, "n_test": int,
            "model_path": str,
            "feature_importance": list[(name, score)],
        }
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

try:
    from lob_features import load_feature_df, add_rolling_features, build_xy, FEATURE_COLS
except ImportError:
    from hft_lob.lob_features import load_feature_df, add_rolling_features, build_xy, FEATURE_COLS

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints_lob")
os.makedirs(MODELS_DIR, exist_ok=True)

GATE_THRESHOLD = 0.36


def train_baseline(symbols: list[str], test_size: float = 0.2) -> dict:
    """
    Train an XGBoost direction classifier on LOB features.

    Uses CUDA on the RTX 3060 if available (XGBoost 2.x device="cuda").
    """
    try:
        import xgboost as xgb
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
    except ImportError as exc:
        raise RuntimeError(
            "xgboost / scikit-learn not installed in the engine venv."
        ) from exc

    df = load_feature_df(symbols)
    if df.empty:
        raise RuntimeError(
            f"No feature parquet files found for symbols={symbols}. "
            "Run /data/download-lob first."
        )

    df = add_rolling_features(df)
    X, y = build_xy(df, horizon=5)
    if len(X) < 1000:
        raise RuntimeError(
            f"Too few labelled samples ({len(X)}). Download more data."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        objective="multi:softprob",
        num_class=3,
        tree_method="hist",
        device="cuda",
        eval_metric="mlogloss",
        verbosity=0,
    )

    t0 = time.time()
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    train_secs = time.time() - t0

    preds = model.predict(X_test)
    acc = float(accuracy_score(y_test, preds))

    # Save the booster
    ts = int(time.time())
    model_name = f"xgb_baseline_{ts}.json"
    model_path = os.path.join(MODELS_DIR, model_name)
    model.save_model(model_path)

    # Feature importance
    try:
        importances = model.feature_importances_.tolist()
    except Exception:
        importances = [0.0] * len(FEATURE_COLS)
    fi = sorted(
        zip(FEATURE_COLS, importances),
        key=lambda kv: -kv[1],
    )

    result = {
        "accuracy": acc,
        "passes_gate": acc > GATE_THRESHOLD,
        "gate_threshold": GATE_THRESHOLD,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "train_seconds": round(train_secs, 2),
        "model_path": model_path,
        "feature_importance": [(n, float(s)) for n, s in fi[:15]],
        "symbols": symbols,
        "timestamp": ts,
    }

    # Persist a sidecar JSON for the dashboard
    meta_path = os.path.join(MODELS_DIR, f"xgb_baseline_{ts}.meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    logger.info(
        "[xgb-baseline] symbols=%s acc=%.4f gate=%s n_train=%d n_test=%d",
        symbols, acc, result["passes_gate"], len(X_train), len(X_test),
    )
    return result


if __name__ == "__main__":
    import sys
    syms = sys.argv[1:] or ["NVDA", "AAPL", "TSM", "META"]
    print(json.dumps(train_baseline(syms), indent=2))
