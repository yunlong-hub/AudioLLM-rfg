#!/usr/bin/env python3
"""D0-4：五条件推理管线（单模型一次调用，支持多机多卡并行）。

产物结构（按运行隔离）：
  exp/<run_id>/
    ├── configs/<model>_<hash>.yaml      # 本运行最终生效配置（多模型/分片安全）
    ├── logs/infer_<model_slug>.log
    ├── predictions/<model_slug>/<item_id>/<COND>.json   # 文本 + 元数据
    ├── predictions/<model_slug>/<item_id>/<COND>.wav    # 语音条件的音频
    └── metrics/infer_<model_slug>.json                  # 完成率/错误/时延汇总

可续跑：每条记录写入 config_hash；哈希一致则跳过，不一致即报错停止（拒绝静默混用配置）。
语音条件强制断言"音频非空且时长 > 0"，否则记为 error，绝不静默回退为文本。

用法：
  CUDA_VISIBLE_DEVICES=2,3 python scripts/infer.py --model <path> --run-id d0_pilot
  CUDA_VISIBLE_DEVICES=0,1 python scripts/infer.py --model <path> --run-id d0_pilot --limit 20
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
import traceback

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2,3")

from rfg.models.registry import load_s2s_model, model_family  # noqa: E402
from rfg.run.conditions import (CONDITION_GEN_KWARGS, CONDITIONS, WANTS_AUDIO, ef_prompt,  # noqa: E402
                                 item_instruction, speakable_format)
from rfg.run.conditions import INSTRUCTION  # noqa: E402


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


# 输出语音时长上限（研究方案 §9 消融：15s / 30s；主实验取 30s）。超限只标 flag，不静默截断。
MAX_OUTPUT_AUDIO_SEC = 30.0


# config_hash 覆盖整份 cfg（含 conditions）。扩展条件集合会改变 hash，但那不是解码配置变更，
# 因此 hash 不一致时再按"解码相关字段"逐项比对，既允许扩展条件、又拒绝静默混用解码配置。
_COMPAT_KEYS = ("model", "items", "decode", "instruction", "device_map")


def configs_compatible(prev_cfg: dict | None, cur_cfg: dict) -> bool:
    if not isinstance(prev_cfg, dict):
        return False
    return all(prev_cfg.get(k) == cur_cfg.get(k) for k in _COMPAT_KEYS)


def slugify(model_path: str) -> str:
    return os.path.basename(model_path.rstrip("/"))


def load_items(path: str) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def run_item(omni, item: dict, out_dir: str, cfg: dict, chash: str,
             conditions: tuple[str, ...] = CONDITIONS) -> dict:
    """跑完一条题目的指定条件，返回汇总。READ 的文本答案供 EF 使用。"""
    iid = item["id"]
    item_dir = os.path.join(out_dir, iid)
    os.makedirs(item_dir, exist_ok=True)
    q_audio = (item.get("question_audio") or {}).get("path")
    instruction = item_instruction(item)
    results: dict[str, dict] = {}
    read_text: str | None = None
    listen_text: str | None = None

    # 条件之间有依赖：READ 必须先跑（EF 需要它的文本答案）
    if "READ" in conditions:
        conditions = tuple(dict.fromkeys(("READ",) + tuple(conditions)))

    def do(cond: str, fn) -> None:
        nonlocal read_text, listen_text
        path = os.path.join(item_dir, f"{cond}.json")
        if os.path.exists(path):
            with open(path) as fh:
                prev = json.load(fh)
            if prev.get("config_hash") != chash and not configs_compatible(prev.get("config"), cfg):
                raise RuntimeError(f"{iid}/{cond}: 已存在记录的解码配置与本次不一致，拒绝覆盖 "
                                   f"({prev.get('config_hash')} != {chash})")
            # 失败记录必须重试：CUDA assert 之类会污染进程上下文，只有新进程才能恢复，
            # 因此这里把"带 error 的记录"视为未完成，交给下一次运行补齐。
            if not prev.get("error"):
                results[cond] = prev
                if cond == "READ":
                    read_text = prev.get("text")
                elif cond == "LISTEN":
                    listen_text = prev.get("text")
                return
        rec: dict = {"item_id": iid, "condition": cond, "config_hash": chash,
                     "config": cfg, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
        try:
            res = fn()
            rec.update({"text": res.text, "output_modality": res.output_modality,
                        "latency_sec": round(res.latency_sec or 0, 3),
                        "input_modality": ("audio" if cond in ("LISTEN", "SPEAK", "EFA", "EFB")
                                           else "text")})
            if WANTS_AUDIO[cond]:
                # 硬断言：语音条件必须有音频，禁止静默回退
                if res.audio is None or res.audio.size == 0:
                    raise RuntimeError("语音条件未产出音频（疑似静默回退为文本）")
                wav = os.path.join(item_dir, f"{cond}.wav")
                omni.save_wav(wav, res)
                rec["audio"] = wav
                rec["audio_duration_sec"] = round(res.duration_sec or 0, 3)
                # 长语音退化护栏：超上限不删除，但显式标 flag，供稳健性分析剔除/单列
                if rec["audio_duration_sec"] > MAX_OUTPUT_AUDIO_SEC:
                    rec["long_audio"] = True
                    rec["long_audio_note"] = (f"{rec['audio_duration_sec']}s > "
                                              f"{MAX_OUTPUT_AUDIO_SEC}s 上限（疑似退化）")
            else:
                rec["audio"] = None
                rec["audio_duration_sec"] = None
            rec["raw_text"] = res.meta.get("raw_text")
        except Exception as exc:
            rec["error"] = f"{type(exc).__name__}: {exc}"
            rec["text"] = None
            traceback.print_exc()
        with open(path, "w") as fh:
            json.dump(rec, fh, indent=2, ensure_ascii=False)
        results[cond] = rec
        if cond == "READ":
            read_text = rec.get("text")
        elif cond == "LISTEN":
            listen_text = rec.get("text")

    if not q_audio or not os.path.exists(q_audio):
        raise FileNotFoundError(f"{iid}: 缺少 question_audio（先跑 scripts/tts_questions.py）")

    # 上游文本从既有记录恢复：这样"只补跑某个新条件"时无需重跑 READ/LISTEN
    for _up, _setter in (("READ", "read"), ("LISTEN", "listen")):
        _p = os.path.join(item_dir, f"{_up}.json")
        if os.path.exists(_p):
            try:
                _t = json.load(open(_p)).get("text")
            except Exception:
                _t = None
            if _t:
                if _setter == "read":
                    read_text = _t
                else:
                    listen_text = _t

    if "READ" in conditions:
        do("READ", lambda: omni.chat([{"type": "text", "text": f"{item['question_text']} {instruction}"}],
                                     want_audio=False))
    if "LISTEN" in conditions:
        do("LISTEN", lambda: omni.chat([{"type": "audio", "audio": q_audio},
                                        {"type": "text", "text": instruction}], want_audio=False))
    if "SPEAK" in conditions:
        do("SPEAK", lambda: omni.chat([{"type": "audio", "audio": q_audio},
                                       {"type": "text", "text": instruction}], want_audio=True))
    if "ECHO" in conditions:
        do("ECHO", lambda: omni.chat([{"type": "text", "text": f"{item['question_text']} {instruction}"}],
                                     want_audio=True))
    if "EF" in conditions:
        if read_text:
            do("EF", lambda: omni.chat([{"type": "text", "text": ef_prompt(read_text)}], want_audio=True))
        else:
            results["EF"] = {"item_id": iid, "condition": "EF", "error": "upstream READ gave no text",
                             "config_hash": chash}
    # EFB：语音输入 + 给定 READ 文本朗读。与 EF 目标句完全相同，唯一差别是输入模态
    # （文本问 vs 语音问），因此 EF vs EFB 是"输入模态是否影响渲染保真"的因果对照。
    if "EFB" in conditions:
        if read_text:
            do("EFB", lambda: omni.chat(
                [{"type": "audio", "audio": q_audio},
                 {"type": "text", "text": ef_prompt(read_text)}], want_audio=True))
        else:
            results["EFB"] = {"item_id": iid, "condition": "EFB",
                              "error": "upstream READ gave no text", "config_hash": chash}
    # SPEAKD：与 SPEAK 同提示词，但用退化控制解码（talker 重复惩罚 + 长度上限）
    if "SPEAKD" in conditions:
        do("SPEAKD", lambda: omni.chat([{"type": "audio", "audio": q_audio},
                                        {"type": "text", "text": instruction}], want_audio=True,
                                       extra_gen_kwargs=CONDITION_GEN_KWARGS["SPEAKD"]))
    # EFW：把 READ 文本改写成"可朗读形式"（数字→词形）后朗读 → 零训练缓解的对照
    if "EFW" in conditions:
        if read_text:
            do("EFW", lambda: omni.chat(
                [{"type": "text", "text": f"{item['question_text']} {instruction}"},
                 {"type": "text", "text": ef_prompt(speakable_format(read_text))}],
                want_audio=True))
        else:
            results["EFW"] = {"item_id": iid, "condition": "EFW",
                              "error": "upstream READ gave no text", "config_hash": chash}
    # EFA：语音输入 + 给定 LISTEN 文本朗读（闭合"输入模态 × 内容来源"2×2）
    if "EFA" in conditions:
        if listen_text:
            do("EFA", lambda: omni.chat(
                [{"type": "audio", "audio": q_audio},
                 {"type": "text", "text": ef_prompt(listen_text)}], want_audio=True))
        else:
            results["EFA"] = {"item_id": iid, "condition": "EFA",
                              "error": "upstream LISTEN gave no text", "config_hash": chash}
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--items", default="data/pilot/items.jsonl")
    ap.add_argument("--run-id", default="d0_pilot")
    ap.add_argument("--out-root", default="exp")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0, help="跳过前 N 题（多卡分片用）")
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--talker-max-new-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--device-map", default="auto")
    args = ap.parse_args()

    conds = tuple(c.strip() for c in args.conditions.split(",") if c.strip())
    unknown = [c for c in conds if c not in CONDITIONS]
    if unknown:
        print(f"未知条件 {unknown}；合法值 {CONDITIONS}", file=sys.stderr)
        return 2
    if "EF" in conds and "READ" not in conds:
        print("EF 依赖 READ 的文本答案，自动加入 READ", flush=True)
        conds = ("READ",) + conds

    run_dir = os.path.join(args.out_root, args.run_id)
    pred_root = os.path.join(run_dir, "predictions", slugify(args.model))
    os.makedirs(pred_root, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "metrics"), exist_ok=True)

    family = model_family(args.model)
    cfg = {"model": args.model, "items": args.items,
           "decode": {"temperature": args.temperature, "do_sample": args.temperature > 0,
                      "max_new_tokens": args.max_new_tokens,
                      "talker_max_new_tokens": args.talker_max_new_tokens},
           "instruction": INSTRUCTION, "conditions": list(conds),
           "device_map": args.device_map, "adapter_family": family}
    if family == "minicpmo":
        cfg["adapter_options"] = {
            "dtype": "float16",
            "token2wav_dtype": "float32",
            "tts_alignment_guard": "shared_prefix",
            "attn_implementation": "sdpa",
            "init_vision": False,
            "reference_voice": "assets/HT_ref_audio.wav",
            "seed": 0,
        }
    chash = config_hash(cfg)
    print(f"[infer] model={args.model} run={args.run_id} config_hash={chash}", flush=True)

    items = load_items(args.items)
    # 分片：多个 worker 可并行处理同一 run 的不同区段（各自 GPU），产物按 item 落盘天然汇合
    if args.offset:
        items = items[args.offset:]
    if args.limit:
        items = items[: args.limit]
    print(f"[infer] items={len(items)} offset={args.offset}", flush=True)

    t0 = time.time()
    omni = load_s2s_model(args.model, device_map=args.device_map)
    load_sec = time.time() - t0

    t_start = time.time()
    n_ok = n_err = 0
    errors: list[str] = []
    latencies: list[float] = []
    print(f"[infer] 条件: {conds}", flush=True)
    for idx, item in enumerate(items, 1):
        try:
            results = run_item(omni, item, pred_root, cfg, chash, conditions=conds)
            bad = [c for c, r in results.items() if r.get("error")]
            n_ok += 1
            n_err += len(bad)
            for c in bad:
                errors.append(f"{item['id']}/{c}: {results[c]['error']}")
            latencies.extend(r.get("latency_sec") or 0 for r in results.values())
        except Exception as exc:
            n_err += 1
            errors.append(f"{item['id']}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
        if idx % 5 == 0 or idx == len(items):
            el = time.time() - t_start
            print(f"  [{idx}/{len(items)}] ok_items={n_ok} cond_errors={n_err} "
                  f"elapsed={el:.0f}s eta={el/idx*(len(items)-idx):.0f}s", flush=True)

    wall = time.time() - t_start
    summary = {"model": args.model, "run_id": args.run_id, "config_hash": chash,
               "n_items": len(items), "n_items_ok": n_ok, "n_condition_errors": n_err,
               "load_sec": round(load_sec, 1), "wall_sec": round(wall, 1),
               "mean_condition_latency_sec": (sum(latencies) / len(latencies)) if latencies else None,
               "errors": errors[:200], "n_errors_total": len(errors),
               "all_conditions_present": n_err == 0 and n_ok == len(items)}
    # 汇总文件名带分片号：多 worker 并行时避免互相覆盖对方的汇总
    shard_tag = f"_off{args.offset}" if args.offset else ""
    out = os.path.join(run_dir, "metrics", f"infer_{slugify(args.model)}{shard_tag}.json")
    summary["shard_offset"] = args.offset
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    config_dir = os.path.join(run_dir, "configs")
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, f"{slugify(args.model)}_{chash}.yaml")
    fd, config_tmp = tempfile.mkstemp(prefix=f".{slugify(args.model)}_{chash}.",
                                      suffix=".yaml.tmp", dir=config_dir, text=True)
    with os.fdopen(fd, "w") as fh:
        import yaml

        yaml.safe_dump(cfg, fh, allow_unicode=True, sort_keys=False)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(config_tmp, config_path)

    print(json.dumps({k: v for k, v in summary.items() if k != "errors"},
                     indent=2, ensure_ascii=False))
    print(f"errors: {len(errors)} (前 5 条) {errors[:5]}")
    print(f"written -> {out}")
    return 0 if summary["all_conditions_present"] else 1


if __name__ == "__main__":
    sys.exit(main())
