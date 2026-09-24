#!/usr/bin/env python3
"""用 vLLM-Omni 离线引擎跑五条件推理（Qwen2.5-Omni-7B 的修复路径）。

为什么单独一个入口
------------------
`transformers` 路径下 Qwen2.5-Omni-7B 的 talker 会产出越界 codec token，触发
`vectorized_gather_kernel index out of bounds`（device-side assert），之后 CUDA 上下文被污染，
同进程内剩余题目全部瞬时失败（实测 11/200 成功）。vLLM-Omni 的 talker→code2wav 实现与
transformers 完全不同，因此用**换栈**绕开该缺陷。

产物与 `scripts/infer/infer.py` **逐字段同构**，故 `readback.py` / `extract_facts.py` / `score.py`
无需改动即可消费：
  exp/<run_id>/predictions/<model_slug>/<item_id>/<COND>.{json,wav}
  exp/<run_id>/metrics/infer_vllm_<model_slug>.json

**口径警告（必须写进论文）**：本入口使用 vLLM-Omni + vLLM 栈，与其余模型
（Qwen3-Omni-30B / Qwen2.5-Omni-3B / Step-Audio-2-mini，均是 transformers 栈）**不是同一栈**。
7B 的数值只能作为独立栈的稳健性证据，不可与主表直接并列比较。

环境：`.venvs/vllm_omni`（继承 conda `audio-llm` 的 torch 2.11.0+cu130，另装 vllm/vllm-omni 0.24.0）。

用法：
  CUDA_VISIBLE_DEVICES=2,3 .venvs/vllm_omni/bin/python scripts/infer/infer_vllm_omni.py \
      --model /workspace/yunlong/LLM/pretrain_model/Qwen/Qwen2.5-Omni-7B \
      --run-id d3_7b_stack --limit 5
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2,3")

from rfg.run.conditions import (CONDITIONS, FORCED_SOURCE, WANTS_AUDIO, ef_prompt,  # noqa: E402
                                speakable_format)
from rfg.run.conditions import INSTRUCTION  # noqa: E402

AUDIO_CONDS = tuple(c for c in CONDITIONS if WANTS_AUDIO[c])
TEXT_CONDS = tuple(c for c in CONDITIONS if not WANTS_AUDIO[c])

SYSTEM = ("You are Qwen, a virtual human developed by the Qwen Team, Alibaba "
          "Group, capable of perceiving auditory and visual inputs, as well as "
          "generating text and speech.")
AUDIO_PLACEHOLDER = "<|audio_bos|><|AUDIO|><|audio_eos|>"
SAMPLE_RATE = 24000
MAX_OUTPUT_AUDIO_SEC = 30.0


def slugify(model_path: str) -> str:
    return os.path.basename(model_path.rstrip("/"))


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


_COMPAT_KEYS = ("model", "items", "decode", "instruction", "stack", "runtime")


def configs_compatible(prev: dict | None, cur: dict) -> bool:
    if not isinstance(prev, dict):
        return False
    return all(prev.get(k) == cur.get(k) for k in _COMPAT_KEYS)


def load_items(path: str) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build_prompt(question_text: str | None, instruction: str | None, with_audio: bool) -> str:
    body = " ".join(x for x in (question_text, instruction) if x)
    audio = AUDIO_PLACEHOLDER if with_audio else ""
    return (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{audio}{body}<|im_end|>\n"
            f"<|im_start|>assistant\n")


def condition_requests(cond: str, item: dict, upstream: dict) -> tuple[str, list, list[str]]:
    """返回 (prompt, 需要的多模态 key, 输出模态)。"""
    q_text = item.get("question_text")

    if cond == "READ":
        return build_prompt(q_text, INSTRUCTION, with_audio=False), [], ["text"]
    if cond == "LISTEN":
        return build_prompt(None, INSTRUCTION, with_audio=True), ["audio"], ["text"]
    if cond in ("SPEAK", "SPEAKD"):
        return build_prompt(None, INSTRUCTION, with_audio=True), ["audio"], ["text", "audio"]
    if cond == "ECHO":
        return build_prompt(q_text, INSTRUCTION, with_audio=False), [], ["text", "audio"]

    src = FORCED_SOURCE.get(cond)
    text = upstream.get(src) if src else None
    if not text:
        raise RuntimeError(f"上游 {src} 无文本，无法构造 {cond}")
    if cond == "EFW":
        text = speakable_format(text)
    with_audio = cond in ("EFA", "EFB")
    return build_prompt(None, ef_prompt(text), with_audio=with_audio), \
        (["audio"] if with_audio else []), ["text", "audio"]


def save_wav(path: str, tensor) -> float:
    import soundfile as sf

    arr = tensor.detach().cpu().numpy()
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    sf.write(path, arr, SAMPLE_RATE)
    return len(arr) / SAMPLE_RATE


def run_generate(omni, prompts: list[dict], sampling_params_list):
    """一次 generate；返回 {request_id: {modality: output}}。"""
    out: dict[str, dict] = {}
    for stage_outputs in omni.generate(prompts, sampling_params_list):
        o = stage_outputs.request_output
        out.setdefault(o.request_id, {})[stage_outputs.final_output_type] = o
    return out


def _load_audio(path: str):
    import numpy as np
    import soundfile as sf

    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return np.asarray(wav, dtype=np.float32), sr


def _item_for_rid(rid: str, meta: list[dict]) -> dict | None:
    """vLLM-Omni 的 request_id 形如 ``"<下标>_<uuid>"``（实测 0.24.0）。

    只取首个下划线前的整数作为输入列表下标；格式不符则返回 None 并记为错误，
    绝不猜测映射（错配会把 A 题的音频写到 B 题名下，属静默污染）。
    """
    head = str(rid).split("_", 1)[0]
    try:
        i = int(head)
    except (TypeError, ValueError):
        return None
    return meta[i] if 0 <= i < len(meta) else None


def _base_rec(iid: str, cond: str, cfg: dict, chash: str) -> dict:
    return {"item_id": iid, "condition": cond, "config_hash": chash, "config": cfg,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "stack": "vllm-omni"}


def _write_error(path: str, iid: str, cond: str, cfg: dict, chash: str, exc: Exception) -> None:
    rec = _base_rec(iid, cond, cfg, chash)
    rec.update({"error": f"{type(exc).__name__}: {exc}", "text": None})
    json.dump(rec, open(path, "w"), indent=2, ensure_ascii=False)


def _text_of(payload: dict) -> str | None:
    o = payload.get("text")
    if o is None:
        return None
    try:
        return o.outputs[0].text
    except Exception:
        return None


def _write_text_batch(outs: dict, todo: list[dict], cond: str, cfg: dict, chash: str,
                      rec_path, errors: list[str]) -> None:
    for rid, payload in outs.items():
        it = _item_for_rid(rid, todo)
        if it is None:
            errors.append(f"{cond}: 无法映射 request_id={rid}")
            continue
        rec = _base_rec(it["id"], cond, cfg, chash)
        rec.update({"text": _text_of(payload), "output_modality": "text",
                    "input_modality": "audio" if cond == "LISTEN" else "text",
                    "audio": None, "audio_duration_sec": None})
        if not rec["text"]:
            rec["error"] = "无文本输出"
        json.dump(rec, open(rec_path(it["id"], cond), "w"), indent=2, ensure_ascii=False)


def _write_audio_record(path: str, iid: str, cond: str, payload: dict,
                        cfg: dict, chash: str) -> None:
    rec = _base_rec(iid, cond, cfg, chash)
    rec.update({"text": _text_of(payload), "output_modality": "speech",
                "input_modality": "audio" if cond in ("SPEAK", "EFA", "EFB", "SPEAKD") else "text"})
    o = payload.get("audio")
    if o is None:
        raise RuntimeError("语音条件未产出音频（疑似静默回退为文本）")
    tensor = o.outputs[0].multimodal_output["audio"]
    if tensor is None or tensor.numel() == 0:
        raise RuntimeError("语音条件音频为空")
    wav_path = path[:-5] + ".wav"
    dur = save_wav(wav_path, tensor)
    rec["audio"] = wav_path
    rec["audio_duration_sec"] = round(dur, 3)
    if dur > MAX_OUTPUT_AUDIO_SEC:
        rec["long_audio"] = True
        rec["long_audio_note"] = f"{dur}s > {MAX_OUTPUT_AUDIO_SEC}s 上限（疑似退化）"
    json.dump(rec, open(path, "w"), indent=2, ensure_ascii=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--items", default="data/pilot/items.jsonl")
    ap.add_argument("--run-id", default="d3_7b_stack")
    ap.add_argument("--out-root", default="exp")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--conditions", default="READ,LISTEN,SPEAK,ECHO,EF")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--thinker-temperature", type=float, default=0.0)
    ap.add_argument("--talker-temperature", type=float, default=0.9)
    ap.add_argument("--talker-top-p", type=float, default=0.8)
    ap.add_argument("--talker-top-k", type=int, default=40)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--talker-max-new-tokens", type=int, default=2048)
    ap.add_argument(
        "--stage-configs-path",
        default="",
        help="可选的 vLLM-Omni stage YAML；用于记录并复现逐 stage 显存配置",
    )
    ap.add_argument("--batch-size", type=int, default=24,
                    help="每次 generate 的题数。整批返回才落盘，故必须分批：既能看到增量进度，"
                         "也让中断后可续跑（默认 24）")
    ap.add_argument("--metrics-tag", default="",
                    help="可选的指标文件后缀；多机分片时用于避免 worker 相互覆盖汇总")
    ap.add_argument("--enforce-eager", action="store_true",
                    help="传给 vLLM 的 enforce_eager（排障用，规避图捕获相关问题）")
    args = ap.parse_args()

    from vllm.sampling_params import SamplingParams
    from vllm_omni.entrypoints.omni import Omni

    conds = tuple(c.strip() for c in args.conditions.split(",") if c.strip())
    unknown = [c for c in conds if c not in CONDITIONS]
    if unknown:
        print(f"未知条件 {unknown}；合法值 {CONDITIONS}", file=sys.stderr)
        return 2
    for up in sorted({FORCED_SOURCE[c] for c in conds if c in FORCED_SOURCE}):
        if up not in conds:
            print(f"{sorted(c for c in conds if FORCED_SOURCE.get(c) == up)} 依赖 {up}，自动加入", flush=True)
            conds = (up,) + conds

    mslug = slugify(args.model)
    run_dir = os.path.join(args.out_root, args.run_id)
    pred_root = os.path.join(run_dir, "predictions", mslug)
    os.makedirs(pred_root, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "metrics"), exist_ok=True)

    runtime_cfg = None
    if args.stage_configs_path:
        stage_path = os.path.abspath(args.stage_configs_path)
        if not os.path.isfile(stage_path):
            print(f"stage config 不存在: {stage_path}", file=sys.stderr)
            return 2
        runtime_cfg = {"stage_configs_path": stage_path,
                       "stage_configs_sha256": file_sha256(stage_path)}

    cfg = {"model": args.model, "items": args.items, "stack": "vllm-omni",
           "decode": {"temperature": args.thinker_temperature,
                      "talker_temperature": args.talker_temperature,
                      "talker_top_p": args.talker_top_p, "talker_top_k": args.talker_top_k,
                      "do_sample": args.thinker_temperature > 0,
                      "max_new_tokens": args.max_new_tokens,
                      "talker_max_new_tokens": args.talker_max_new_tokens,
                      "seed": args.seed},
           "instruction": INSTRUCTION, "conditions": list(conds),
           "runtime": runtime_cfg}
    chash = config_hash(cfg)
    print(f"[infer_vllm] model={args.model} run={args.run_id} config_hash={chash}", flush=True)

    items = load_items(args.items)
    if args.offset:
        items = items[args.offset:]
    if args.limit:
        items = items[: args.limit]
    print(f"[infer_vllm] items={len(items)} offset={args.offset} conds={conds}", flush=True)

    t0 = time.time()
    omni_kwargs = {"model": args.model}
    if args.enforce_eager:
        omni_kwargs["enforce_eager"] = True
    if args.stage_configs_path:
        omni_kwargs["stage_configs_path"] = os.path.abspath(args.stage_configs_path)
    omni = Omni(**omni_kwargs)
    load_sec = time.time() - t0
    print(f"[infer_vllm] 引擎加载 {load_sec:.1f}s", flush=True)

    thinker_sp = SamplingParams(temperature=args.thinker_temperature, top_p=1.0, top_k=-1,
                                max_tokens=args.max_new_tokens, seed=args.seed,
                                detokenize=True, repetition_penalty=1.1)
    talker_sp = SamplingParams(temperature=args.talker_temperature, top_p=args.talker_top_p,
                               top_k=args.talker_top_k, max_tokens=args.talker_max_new_tokens,
                               seed=args.seed, detokenize=True, repetition_penalty=1.05,
                               stop_token_ids=[8294])
    code2wav_sp = SamplingParams(temperature=0.0, top_p=1.0, top_k=-1, max_tokens=2048,
                                 seed=args.seed, detokenize=True, repetition_penalty=1.1)
    sp_list = [thinker_sp, talker_sp, code2wav_sp]

    errors: list[str] = []
    latencies: list[float] = []
    n_written = 0

    def rec_path(iid: str, cond: str) -> str:
        d = os.path.join(pred_root, iid)
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"{cond}.json")

    def already_done(iid: str, cond: str) -> dict | None:
        p = rec_path(iid, cond)
        if not os.path.exists(p):
            return None
        try:
            prev = json.load(open(p))
        except Exception:
            return None
        if prev.get("error"):
            return None          # 失败记录必须重试
        if prev.get("config_hash") != chash and not configs_compatible(prev.get("config"), cfg):
            raise RuntimeError(f"{iid}/{cond}: 既有记录配置不一致，拒绝覆盖")
        return prev

    # ---------- 阶段 A：文本输出条件 ----------
    upstream: dict[str, dict[str, str]] = {}
    for cond in [c for c in TEXT_CONDS if c in conds]:
        todo = []
        for it in items:
            prev = already_done(it["id"], cond)
            if prev is not None:
                upstream.setdefault(it["id"], {})[cond] = prev.get("text")
            else:
                todo.append(it)
        print(f"[infer_vllm] {cond}: 待跑 {len(todo)}/{len(items)}", flush=True)
        if not todo:
            continue
        bs = max(1, args.batch_size)
        for c0 in range(0, len(todo), bs):
            chunk = todo[c0 : c0 + bs]
            prompts = []
            for it in chunk:
                pr, keys, mods = condition_requests(cond, it, {})
                d = {"prompt": pr, "modalities": mods}
                if keys:
                    d["multi_modal_data"] = {"audio": _load_audio(it["question_audio"]["path"])}
                prompts.append(d)
            t1 = time.time()
            try:
                outs = run_generate(omni, prompts, sp_list)
            except Exception as exc:
                errors.append(f"{cond}[{c0}:{c0+len(chunk)}] 批量失败: {type(exc).__name__}: {exc}")
                traceback.print_exc()
                continue
            latencies.append(time.time() - t1)
            _write_text_batch(outs, chunk, cond, cfg, chash, rec_path, errors)
            for it in chunk:
                p = rec_path(it["id"], cond)
                if os.path.exists(p):
                    t = json.load(open(p)).get("text")
                    if t:
                        upstream.setdefault(it["id"], {})[cond] = t
            print(f"[infer_vllm] {cond}: {min(c0+bs, len(todo))}/{len(todo)} "
                  f"用时 {time.time()-t1:.1f}s", flush=True)

    # ---------- 阶段 B：音频输出条件 ----------
    for cond in [c for c in AUDIO_CONDS if c in conds]:
        todo = [it for it in items if already_done(it["id"], cond) is None]
        print(f"[infer_vllm] {cond}: 待跑 {len(todo)}/{len(items)}", flush=True)
        if not todo:
            continue
        prompts, meta = [], []
        for it in todo:
            up = dict(upstream.get(it["id"], {}))
            for u in ("READ", "LISTEN"):
                if u not in up:
                    p = os.path.join(pred_root, it["id"], f"{u}.json")
                    if os.path.exists(p):
                        try:
                            t = json.load(open(p)).get("text")
                            if t:
                                up[u] = t
                        except Exception:
                            pass
            try:
                pr, keys, mods = condition_requests(cond, it, up)
            except Exception as exc:
                errors.append(f"{it['id']}/{cond}: {exc}")
                _write_error(rec_path(it["id"], cond), it["id"], cond, cfg, chash, exc)
                continue
            d = {"prompt": pr, "modalities": mods}
            if keys:
                d["multi_modal_data"] = {"audio": _load_audio(it["question_audio"]["path"])}
            prompts.append(d)
            meta.append(it)
        if not prompts:
            continue
        bs = max(1, args.batch_size)
        for c0 in range(0, len(prompts), bs):
            cp, cm = prompts[c0 : c0 + bs], meta[c0 : c0 + bs]
            t1 = time.time()
            try:
                outs = run_generate(omni, cp, sp_list)
            except Exception as exc:
                errors.append(f"{cond}[{c0}:{c0+len(cp)}] 批量失败: {type(exc).__name__}: {exc}")
                traceback.print_exc()
                continue
            latencies.append(time.time() - t1)
            for rid, payload in outs.items():
                it = _item_for_rid(rid, cm)
                if it is None:
                    errors.append(f"{cond}: 无法把 request_id={rid} 映射回题目")
                    continue
                try:
                    _write_audio_record(rec_path(it["id"], cond), it["id"], cond, payload, cfg, chash)
                    n_written += 1
                except Exception as exc:
                    errors.append(f"{it['id']}/{cond}: {type(exc).__name__}: {exc}")
                    _write_error(rec_path(it["id"], cond), it["id"], cond, cfg, chash, exc)
            print(f"[infer_vllm] {cond}: {min(c0+bs, len(prompts))}/{len(prompts)} "
                  f"用时 {time.time()-t1:.1f}s", flush=True)

    omni.close()
    wall = time.time() - t0
    if args.metrics_tag and not all(c.isalnum() or c in "-_" for c in args.metrics_tag):
        raise ValueError("metrics-tag 只能包含字母、数字、连字符和下划线")
    summary = {"model": args.model, "run_id": args.run_id, "stack": "vllm-omni",
               "config_hash": chash, "n_items": len(items), "n_records_written": n_written,
               "offset": args.offset, "limit": args.limit, "conditions": list(conds),
               "batch_size": args.batch_size, "metrics_tag": args.metrics_tag or None,
               "load_sec": round(load_sec, 1), "wall_sec": round(wall, 1),
               "mean_batch_latency_sec": (sum(latencies) / len(latencies)) if latencies else None,
               "errors": errors[:200], "n_errors_total": len(errors)}
    suffix = f"_{args.metrics_tag}" if args.metrics_tag else ""
    out = os.path.join(run_dir, "metrics", f"infer_vllm_{mslug}{suffix}.json")
    json.dump(summary, open(out, "w"), indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in summary.items() if k != "errors"}, indent=2, ensure_ascii=False))
    print(f"errors: {len(errors)} (前 5 条) {errors[:5]}")
    print(f"written -> {out}")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
