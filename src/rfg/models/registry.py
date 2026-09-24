"""S2S model adapter selection from checkpoint metadata."""
from __future__ import annotations

import json
from pathlib import Path


def model_family(model_path: str) -> str:
    config_path = Path(model_path) / "config.json"
    with config_path.open() as fh:
        config = json.load(fh)
    model_type = config.get("model_type")
    architectures = config.get("architectures") or []
    if model_type == "minicpmo" or "MiniCPMO" in architectures:
        return "minicpmo"
    return "qwen_omni"


def load_s2s_model(model_path: str, **kwargs):
    family = model_family(model_path)
    if family == "minicpmo":
        from rfg.models.minicpmo import MiniCPMOModel

        return MiniCPMOModel(model_path, **kwargs)

    from rfg.models.omni import OmniModel

    return OmniModel(model_path, **kwargs)
