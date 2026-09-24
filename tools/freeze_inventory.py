#!/usr/bin/env python3
"""D0-1 工具链冻结清单生成器（可复算，不依赖网络）。

用法:
    python tools/freeze_inventory.py --out configs/freeze.d0.json
    python tools/freeze_inventory.py --print          # 只打印摘要

行为:
  * 对每个候选项记录: 存在性、绝对路径、文件清单、总大小、mtime。
  * 小文件 (< SMALL_LIMIT) 全量 sha256; 大文件只做 head/tail 1 MiB 的 partial sha256,
    并显式标注 `hash_mode`, 避免把 partial 当成全量哈希。
  * 对 transformers 模型目录额外抽取 config.json 的关键字段 (architectures / 是否含
    talker_config / enable_audio_output), 用于确认"能不能出声"。
  * 所有未探测到的字段写 null, 不做任何估计。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

SMALL_LIMIT = 2 * 1024 * 1024 * 1024  # 2 GiB
CHUNK = 1 << 20  # 1 MiB

PRETRAIN = "/workspace/yunlong/LLM/pretrain_model"

# ---------------------------------------------------------------- candidates
S2S_MODELS = [
    f"{PRETRAIN}/Qwen/Qwen3-Omni-30B-A3B-Instruct",
    f"{PRETRAIN}/Qwen/Qwen3-Omni-30B-A3B-Thinking",  # 无 talker_config, 不能出声
    f"{PRETRAIN}/Qwen/Qwen2.5-Omni-7B",
    f"{PRETRAIN}/Qwen/Qwen2.5-Omni-3B",
    f"{PRETRAIN}/Audio/Qwen2-Audio-7B-Instruct",  # 音频进/文本出, 不能进 SPEAK
]

ASR_MODELS = [
    f"{PRETRAIN}/Audio/whisper-large-v3",
    f"{PRETRAIN}/Audio/seamless-m4t-v2-large",
]

TTS_MODELS = [
    f"{PRETRAIN}/TTS/CosyVoice2-0.5B",
    f"{PRETRAIN}/TTS/Fun-CosyVoice3-0.5B-2512",
    f"{PRETRAIN}/TTS/chatterbox",
    f"{PRETRAIN}/TTS/Qwen3-TTS-12Hz-1.7B-Base",
]

EXTRACTOR_MODELS = [
    f"{PRETRAIN}/Qwen/Qwen2.5-7B-Instruct",
    f"{PRETRAIN}/Qwen/Qwen3-8B",
    f"{PRETRAIN}/Qwen/Qwen2.5-3B",
]

ENVS = [
    "/workspace/yunlong/anaconda3/envs/audio-llm",  # 唯一执行环境（用户指定）
]


def _load_decode_config(path: str = "configs/decode.json") -> dict:
    """读取 D0-2 冒烟后冻结的解码配置；缺失时返回全 null（绝不预估）。"""
    if os.path.isfile(path):
        try:
            with open(path) as fh:
                cfg = json.load(fh)
            cfg["_source"] = path
            return cfg
        except Exception as exc:
            return {"_error": f"{type(exc).__name__}: {exc}", "_source": path}
    return {"_source": None, "note": "configs/decode.json 不存在：D0-2 冒烟后回填"}


def sha256_file(path: str, limit_bytes: int | None = None) -> str:
    h = hashlib.sha256()
    read = 0
    with open(path, "rb") as fh:
        while True:
            if limit_bytes is not None and read >= limit_bytes:
                break
            want = CHUNK if limit_bytes is None else min(CHUNK, limit_bytes - read)
            if want <= 0:
                break
            buf = fh.read(want)
            if not buf:
                break
            h.update(buf)
            read += len(buf)
    return h.hexdigest()


def sha256_head_tail(path: str, span: int = CHUNK) -> str:
    size = os.path.getsize(path)
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read(span))
        if size > 2 * span:
            fh.seek(size - span)
            h.update(fh.read(span))
    return h.hexdigest()


def hash_entry(path: str) -> dict:
    size = os.path.getsize(path)
    if size <= SMALL_LIMIT:
        return {"sha256": sha256_file(path), "hash_mode": "full", "size_bytes": size}
    return {
        "sha256_head_tail_1MiB": sha256_head_tail(path),
        "hash_mode": "partial_head_tail_1MiB",
        "size_bytes": size,
    }


def model_fingerprint(root: str) -> dict:
    """抽取 config 关键字段, 判定该 checkpoint 能否输出语音。"""
    info: dict = {
        "path": root,
        "exists": os.path.isdir(root),
        "config": None,
        "can_speak": None,
        "n_files": None,
        "total_bytes": None,
        "files": [],
    }
    if not info["exists"]:
        return info

    names = sorted(os.listdir(root))
    info["n_files"] = len(names)
    total = 0
    for n in names:
        p = os.path.join(root, n)
        if os.path.isfile(p):
            total += os.path.getsize(p)
    info["total_bytes"] = total

    # 权重的分片清单 (只记录文件名+大小+partial hash, 不逐个全量哈希)
    weight_like = [n for n in names if n.endswith((".safetensors", ".pt", ".bin", ".onnx"))]
    for n in weight_like:
        p = os.path.join(root, n)
        e = hash_entry(p)
        e["file"] = n
        info["files"].append(e)

    cfg_path = os.path.join(root, "config.json")
    if os.path.isfile(cfg_path):
        e = hash_entry(cfg_path)
        e["file"] = "config.json"
        info["files"].append(e)
        try:
            cfg = json.load(open(cfg_path))
        except Exception as exc:  # pragma: no cover
            cfg = {"_parse_error": str(exc)}
        talker = cfg.get("talker_config") or (cfg.get("thinker_config") or {}).get("talker_config")
        info["config"] = {
            "architectures": cfg.get("architectures"),
            "model_type": cfg.get("model_type"),
            "has_talker_config": bool(talker),
            "enable_audio_output": cfg.get("enable_audio_output"),
            "enable_talker": cfg.get("enable_talker"),
            "has_code2wav": bool(cfg.get("code2wav_config") or cfg.get("token2wav_config")),
            "transformers_version": cfg.get("transformers_version"),
            "dtype": cfg.get("dtype") or cfg.get("torch_dtype"),
            "sha256": e["sha256"],
        }
        # 可出声的判据: 有 talker 结构且未显式关闭
        info["can_speak"] = bool(
            talker is not None and cfg.get("enable_audio_output") is not False
        )
    return info


def env_fingerprint(root: str) -> dict:
    info = {"path": root, "exists": os.path.isdir(root), "pip_freeze_sha256": None,
            "pip_freeze_n_files": None, "python": None}
    if not info["exists"]:
        return info
    py = os.path.join(root, "bin", "python")
    info["python"] = py if os.path.isfile(py) else None
    # conda-meta 下每个包一个 json, 用其清单的聚合哈希代表环境状态 (不调用 pip, 避免慢)
    meta = os.path.join(root, "conda-meta")
    if os.path.isdir(meta):
        names = sorted(n for n in os.listdir(meta) if n.endswith(".json"))
        info["pip_freeze_n_files"] = len(names)
        h = hashlib.sha256()
        for n in names:
            h.update(n.encode())
        info["pip_freeze_sha256"] = h.hexdigest()
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="configs/freeze.d0.json")
    ap.add_argument("--print", dest="do_print", action="store_true")
    args = ap.parse_args()

    inv = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": os.uname().nodename,
        "git_commit": None,
        "hash_policy": {
            "small_file_limit_bytes": SMALL_LIMIT,
            "large_file_mode": "sha256 over head+tail 1MiB (labeled partial)",
        },
        "s2s_models": [model_fingerprint(p) for p in S2S_MODELS],
        "asr_models": [model_fingerprint(p) for p in ASR_MODELS],
        "tts_models": [model_fingerprint(p) for p in TTS_MODELS],
        "extractor_models": [model_fingerprint(p) for p in EXTRACTOR_MODELS],
        "conda_envs": [env_fingerprint(p) for p in ENVS],
        "decode_config": _load_decode_config(),
        "prompts": {"fact_extraction_llm": {"sha256": None, "model": None},
                    "ef_forced_readback": {"sha256": None}},
        "runtime": {"cuda_visible_devices": "2,3", "output_audio": "wav mono pcm_s16le"},
        "hosts": {"A22": "本机（GPU2/3 可用，GPU0/1 被他人占用）",
                  "A23": "远程 ssh A23-direct（GPU0/1 空闲）；/workspace 共享"},
    }

    # git commit (仓库尚未初始化时为 null)
    try:
        import subprocess
        inv["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        ).stdout.strip() or None
    except Exception:
        inv["git_commit"] = None

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(inv, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    if args.do_print:
        for group in ("s2s_models", "asr_models", "tts_models", "extractor_models"):
            print(f"[{group}]")
            for m in inv[group]:
                if not m["exists"]:
                    print(f"  MISSING {m['path']}")
                    continue
                cfg = m["config"] or {}
                print(f"  {os.path.basename(m['path']):38s} can_speak={m['can_speak']} "
                      f"talker={cfg.get('has_talker_config')} files={m['n_files']} "
                      f"{m['total_bytes']/2**30:.1f}GiB")
    print(f"written -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
