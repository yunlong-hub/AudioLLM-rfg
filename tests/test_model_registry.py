import json

from rfg.models.registry import model_family


def test_model_family_detects_minicpmo(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "minicpmo", "architectures": ["MiniCPMO"]})
    )
    assert model_family(str(tmp_path)) == "minicpmo"


def test_model_family_defaults_to_qwen_omni(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen3_omni_moe", "architectures": ["Qwen3OmniMoeModel"]})
    )
    assert model_family(str(tmp_path)) == "qwen_omni"
