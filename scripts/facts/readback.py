#!/usr/bin/env python3
"""D0-4 回读阶段：对推理产出的语音做双 ASR 回读。

对每个语音条件（SPEAK/ECHO/EF）：
  * ASR-1（whisper-large-v3）与 ASR-2（seamless-m4t-v2-large）各转写一次，单通道失败不互相拖累；
  * `asr_agree`：两侧规范化 WER ≤ 阈值（冒烟口径；D0-5 会改为"事实集合一致"）；
  * `readback_wer`：回读文本 vs **同题上游文本答案**（SPEAK→LISTEN，ECHO/EF→READ，见 conditions.py）。

可续跑：已存在且 asr 版本一致的记录直接复用；写入前先落临时文件再替换。

用法：
  CUDA_VISIBLE_DEVICES=2 python scripts/readback.py --run-id d0_pilot [--model <path>] [--limit 20]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

from rfg.facts.readback_state import (  # noqa: E402
    channel_complete,
    channel_reusable_for_chunking,
    completed_text,
)
from rfg.facts.readback_update import merge_readback_observations  # noqa: E402
from rfg.models.asr import SeamlessReadback, WhisperReadback  # noqa: E402
from rfg.run.conditions import READBACK_REFERENCE  # noqa: E402

PRETRAIN = "/workspace/yunlong/LLM/pretrain_model"
WHISPER = f"{PRETRAIN}/Audio/whisper-large-v3"
SEAMLESS = f"{PRETRAIN}/Audio/seamless-m4t-v2-large"
SPEECH_CONDS = tuple(READBACK_REFERENCE)
ASR_CHANNELS = ("asr1", "asr2")


def parse_channels(value: str) -> tuple[str, ...]:
    channels = tuple(part.strip() for part in value.split(",") if part.strip())
    invalid = [channel for channel in channels if channel not in ASR_CHANNELS]
    if not channels or invalid:
        raise argparse.ArgumentTypeError(
            f"channels must be a comma-separated subset of {','.join(ASR_CHANNELS)}"
        )
    return tuple(dict.fromkeys(channels))


def safe(fn, *a, **kw):
    try:
        return fn(*a, **kw), None
    except Exception as exc:  # 单通道失败不拖累另一通道
        return None, f"{type(exc).__name__}: {exc}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="d0_pilot")
    ap.add_argument("--out-root", default="exp")
    ap.add_argument("--model", default=None, help="只处理该模型目录；默认全部")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0, help="跳过前 N 题（多卡分片用）")
    ap.add_argument("--agree-threshold", type=float, default=0.2)
    ap.add_argument("--intelligible-wer", type=float, default=0.5,
                    help="回读 vs 模型自身文本的 WER 超过它即判为生成失效（非事实渲染丢失）")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument(
        "--channels",
        type=parse_channels,
        default=ASR_CHANNELS,
        help="要补齐的 ASR 通道：asr1、asr2 或 asr1,asr2；单通道模式降低显存峰值",
    )
    ap.add_argument(
        "--low-memory",
        action="store_true",
        help="Seamless 禁用生成缓存，供大模型旁路回读降低峰值显存",
    )
    ap.add_argument(
        "--chunk-seconds",
        type=float,
        default=0.0,
        help="长音频按静音点切成不超过该秒数的块；0 保持单段推理",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="重新计算指定通道，即使已有成功结果",
    )
    args = ap.parse_args()

    run_dir = os.path.join(args.out_root, args.run_id)
    pred_root = os.path.join(run_dir, "predictions")
    models = [args.model] if args.model else (
        [d for d in sorted(os.listdir(pred_root)) if os.path.isdir(os.path.join(pred_root, d))]
        if os.path.isdir(pred_root) else [])
    if not models:
        print(f"没有可处理的模型目录: {pred_root}", file=sys.stderr)
        return 1

    print(f"[readback] models={models} channels={args.channels}", flush=True)
    recognizers = {}
    if "asr1" in args.channels:
        recognizers["asr1"] = WhisperReadback(
            WHISPER,
            device=args.device,
            chunk_seconds=args.chunk_seconds or None,
        )
    if "asr2" in args.channels:
        recognizers["asr2"] = SeamlessReadback(
            SEAMLESS,
            device=args.device,
            low_memory=args.low_memory,
            chunk_seconds=args.chunk_seconds or None,
        )

    for mslug in models:
        mdir = os.path.join(pred_root, mslug)
        item_ids = sorted(d for d in os.listdir(mdir) if os.path.isdir(os.path.join(mdir, d)))
        if args.offset:
            item_ids = item_ids[args.offset:]
        if args.limit:
            item_ids = item_ids[: args.limit]
        stats = {"n_speech": 0, "n_both_nonempty": 0, "n_agree": 0, "n_intelligible": 0,
                 "n_reused": 0, "errors": []}
        wers: list[float] = []
        t0 = time.time()
        for k, iid in enumerate(item_ids, 1):
            idir = os.path.join(mdir, iid)
            # 上游文本：SPEAK→LISTEN，ECHO/EF→READ
            upstream: dict[str, str | None] = {}
            for ref in ("READ", "LISTEN"):
                p = os.path.join(idir, f"{ref}.json")
                if os.path.exists(p):
                    with open(p) as fh:
                        upstream[ref] = json.load(fh).get("text")
            for cond in SPEECH_CONDS:
                cpath = os.path.join(idir, f"{cond}.json")
                wav = os.path.join(idir, f"{cond}.wav")
                if not os.path.exists(cpath) or not os.path.exists(wav):
                    continue
                with open(cpath) as fh:
                    rec = json.load(fh)
                prev = rec.get("readback") or {}
                duration = rec.get("audio_duration_sec")

                def reusable(channel: str) -> bool:
                    return channel_reusable_for_chunking(
                        prev,
                        channel,
                        duration_sec=duration,
                        chunk_seconds=args.chunk_seconds or None,
                        expected_version=recognizers[channel].name,
                    )

                if (
                    not args.force
                    and all(reusable(channel) for channel in args.channels)
                ):
                    stats["n_reused"] += 1
                    stats["n_speech"] += 1
                    stats["n_agree"] += int(bool(prev.get("asr_agree")))
                    stats["n_both_nonempty"] += int(bool(prev.get("asr1")) and bool(prev.get("asr2")))
                    if prev.get("readback_wer") is not None:
                        wers.append(prev["readback_wer"])
                    continue

                ref_cond = READBACK_REFERENCE[cond]
                ref_text = upstream.get(ref_cond)
                updates = {}
                for channel in args.channels:
                    if not args.force and reusable(channel):
                        continue
                    result, error = safe(recognizers[channel].transcribe, wav)
                    updates[channel] = {
                        "text": result.text if result else None,
                        "error": error,
                        "sec": round(result.latency_sec, 3) if result else None,
                        "model": recognizers[channel].name,
                    }
                    if error:
                        stats["errors"].append(f"{iid}/{cond} {channel}: {error}")

                rec["readback"] = merge_readback_observations(
                    prev,
                    updates,
                    reference_condition=ref_cond,
                    reference_text=ref_text,
                    self_text=rec.get("text"),
                    agree_threshold=args.agree_threshold,
                    intelligible_wer=args.intelligible_wer,
                )
                # Multiple disjoint readback shards can touch adjacent
                # conditions in the same item directory.  A PID-scoped
                # temporary name keeps their atomic replacements independent.
                tmp = f"{cpath}.tmp.{os.getpid()}"
                with open(tmp, "w") as fh:
                    json.dump(rec, fh, indent=2, ensure_ascii=False)
                os.replace(tmp, cpath)

                t1 = completed_text(rec["readback"], "asr1")
                t2 = completed_text(rec["readback"], "asr2")
                stats["n_speech"] += 1
                stats["n_both_nonempty"] += int(bool(t1) and bool(t2))
                stats["n_agree"] += int(bool(rec["readback"].get("asr_agree")))
                stats["n_intelligible"] += int(rec["readback"]["audio_intelligible"])
                if rec["readback"].get("readback_wer") is not None:
                    wers.append(rec["readback"]["readback_wer"])
            if k % 20 == 0 or k == len(item_ids):
                el = time.time() - t0
                print(f"  [{mslug} {k}/{len(item_ids)}] speech={stats['n_speech']} "
                      f"agree={stats['n_agree']} reused={stats['n_reused']} "
                      f"elapsed={el:.0f}s eta={el/k*(len(item_ids)-k):.0f}s", flush=True)

        summary = {"model_dir": mslug, "run_id": args.run_id,
                   "n_items": len(item_ids), **{k: v for k, v in stats.items() if k != "errors"},
                   "agreement_rate": (stats["n_agree"] / stats["n_speech"]) if stats["n_speech"] else None,
                   "intelligibility_rate": (stats["n_intelligible"] / stats["n_speech"])
                   if stats["n_speech"] else None,
                   "n_intelligible": stats["n_intelligible"],
                   "mean_readback_wer": (sum(wers) / len(wers)) if wers else None,
                   "agree_threshold": args.agree_threshold,
                   "n_errors_total": len(stats["errors"]), "errors": stats["errors"][:100]}
        _tag = f"_off{args.offset}" if args.offset else ""
        if args.channels != ASR_CHANNELS:
            _tag += "_" + "-".join(args.channels)
        out = os.path.join(run_dir, "metrics", f"readback_{mslug}{_tag}.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as fh:
            json.dump(summary, fh, indent=2, ensure_ascii=False)
        print(json.dumps({k: v for k, v in summary.items() if k != "errors"},
                         indent=2, ensure_ascii=False))
        print(f"written -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
