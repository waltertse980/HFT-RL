"""
model_registry.py
=================
Lightweight registry that lists LOB checkpoints and their metadata.
Used by the dashboard to populate model dropdowns for backtest / paper-trade.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
CHECKPOINTS_LOB = BASE_DIR / "checkpoints_lob"


def list_lob_checkpoints() -> list[dict]:
    """Return all LOB PPO checkpoints with sidecar meta if present."""
    if not CHECKPOINTS_LOB.exists():
        return []
    out: list[dict] = []
    for d in sorted(CHECKPOINTS_LOB.iterdir()):
        if not d.is_dir():
            continue
        model = d / "ppo_final.zip"
        if not model.exists():
            continue
        entry = {
            "checkpoint_dir": d.name,
            "model_path": str(model),
            "vecnorm_path": str(d / "vecnorm.pkl") if (d / "vecnorm.pkl").exists() else None,
        }
        meta = d / "meta.json"
        if meta.exists():
            try:
                entry["meta"] = json.loads(meta.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("[registry] bad meta in %s: %s", d, exc)
        out.append(entry)
    return out


def list_xgb_baselines() -> list[dict]:
    """Return all saved XGBoost baselines with sidecar meta."""
    if not CHECKPOINTS_LOB.exists():
        return []
    out: list[dict] = []
    for fn in sorted(os.listdir(CHECKPOINTS_LOB)):
        if not fn.startswith("xgb_baseline_") or not fn.endswith(".meta.json"):
            continue
        path = CHECKPOINTS_LOB / fn
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            out.append(data)
        except Exception as exc:
            logger.warning("[registry] bad xgb meta %s: %s", fn, exc)
    return out
