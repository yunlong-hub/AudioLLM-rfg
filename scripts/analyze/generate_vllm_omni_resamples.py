#!/usr/bin/env python3
"""Generate fixed-seed vLLM-Omni speech candidates for FRR evaluation."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

from rfg.facts.readback_state import completed_text
from scripts.infer.infer_vllm_omni import (
    MAX_OUTPUT_AUDIO_SEC,
    _item_for_rid,
    _load_audio,
    _text_of,
    condition_requests,
    config_hash,
    file_sha256,
    run_generate,
    save_wav,
)


def atomic_json(path: Path, record: dict) -> None:
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2))
    os.replace(temporary, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--items", default="data/pilot/items.jsonl")
    ap.add_argument("--pilot-pred", default="exp/d3_7b_stack/predictions")
    ap.add_argument("--out-root", default="exp/d0_resample")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seeds", default="101,202,303")
    ap.add_argument("--candidate-offset", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--thinker-temperature", type=float, default=0.0)
    ap.add_argument("--talker-temperature", type=float, default=0.9)
    ap.add_argument("--talker-top-p", type=float, default=0.8)
    ap.add_argument("--talker-top-k", type=int, default=40)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--talker-max-new-tokens", type=int, default=512)
    ap.add_argument("--stage-configs-path", required=True)
    ap.add_argument("--metrics-tag", required=True)
    args = ap.parse_args()

    from vllm.sampling_params import SamplingParams
    from vllm_omni.entrypoints.omni import Omni

    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        ap.error("--seeds must contain at least one integer")
    if args.offset < 0 or args.limit < 0 or args.candidate_offset < 0:
        ap.error("--offset, --limit, and --candidate-offset must be non-negative")
    candidates = list(enumerate(seeds, start=args.candidate_offset))

    model_slug = os.path.basename(args.model.rstrip("/"))
    rows = [json.loads(line) for line in Path(args.items).read_text().splitlines() if line.strip()]
    selected = rows[args.offset : args.offset + args.limit if args.limit else None]
    base_root = Path(args.pilot_pred) / model_slug
    items = []
    for item in selected:
        base_path = base_root / item["id"] / "SPEAK.json"
        if not base_path.is_file():
            continue
        base = json.loads(base_path.read_text())
        readback = base.get("readback") or {}
        if (base.get("text") and completed_text(readback, "asr1") is not None
                and completed_text(readback, "asr2") is not None
                and completed_text(readback, "asr3") is not None):
            items.append(item)
    if len(items) != len(selected):
        raise RuntimeError(
            f"base coverage incomplete in selected range: usable={len(items)} selected={len(selected)}"
        )

    stage_path = os.path.abspath(args.stage_configs_path)
    out_model = Path(args.out_root) / model_slug
    metrics_dir = Path(args.out_root) / "metrics"
    out_model.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    runtime = {
        "stage_configs_path": stage_path,
        "stage_configs_sha256": file_sha256(stage_path),
    }
    shared_config = {
        "model": args.model,
        "items": args.items,
        "pilot_pred": args.pilot_pred,
        "stack": "vllm-omni",
        "instruction_condition": "SPEAK",
        "runtime": runtime,
        "decode": {
            "thinker_temperature": args.thinker_temperature,
            "talker_temperature": args.talker_temperature,
            "talker_top_p": args.talker_top_p,
            "talker_top_k": args.talker_top_k,
            "max_new_tokens": args.max_new_tokens,
            "talker_max_new_tokens": args.talker_max_new_tokens,
        },
    }

    expected = len(items) * len(seeds)
    reusable = 0
    for item in items:
        for candidate, seed in candidates:
            path = out_model / item["id"] / f"R{candidate}.json"
            seed_config = {**shared_config, "seed": seed, "candidate": candidate}
            if path.is_file() and path.with_suffix(".wav").is_file():
                record = json.loads(path.read_text())
                if record.get("config_hash") != config_hash(seed_config):
                    raise RuntimeError(f"incompatible existing candidate: {path}")
                reusable += 1
    if reusable == expected:
        print(f"[vllm-resample] all {expected} candidates reusable", flush=True)
        return 0

    print(
        f"[vllm-resample] model={model_slug} range={args.offset}:{args.offset + len(selected)} "
        f"items={len(items)} seeds={seeds} reusable={reusable}/{expected}",
        flush=True,
    )
    started = time.time()
    omni = Omni(model=args.model, stage_configs_path=stage_path)
    load_sec = time.time() - started
    errors: list[str] = []
    written = 0
    try:
        for candidate, seed in candidates:
            seed_config = {**shared_config, "seed": seed, "candidate": candidate}
            chash = config_hash(seed_config)
            pending = []
            for item in items:
                record_path = out_model / item["id"] / f"R{candidate}.json"
                if record_path.is_file() and record_path.with_suffix(".wav").is_file():
                    continue
                pending.append(item)
            thinker = SamplingParams(
                temperature=args.thinker_temperature, top_p=1.0, top_k=-1,
                max_tokens=args.max_new_tokens, seed=seed, detokenize=True,
                repetition_penalty=1.1,
            )
            talker = SamplingParams(
                temperature=args.talker_temperature, top_p=args.talker_top_p,
                top_k=args.talker_top_k, max_tokens=args.talker_max_new_tokens,
                seed=seed, detokenize=True, repetition_penalty=1.05,
                stop_token_ids=[8294],
            )
            code2wav = SamplingParams(
                temperature=0.0, top_p=1.0, top_k=-1, max_tokens=2048,
                seed=seed, detokenize=True, repetition_penalty=1.1,
            )
            sampling = [thinker, talker, code2wav]
            for start in range(0, len(pending), max(1, args.batch_size)):
                chunk = pending[start : start + max(1, args.batch_size)]
                prompts = []
                for item in chunk:
                    prompt, keys, modalities = condition_requests("SPEAK", item, {})
                    request = {"prompt": prompt, "modalities": modalities}
                    if keys:
                        request["multi_modal_data"] = {
                            "audio": _load_audio(item["question_audio"]["path"])
                        }
                    prompts.append(request)
                batch_started = time.time()
                try:
                    outputs = run_generate(omni, prompts, sampling)
                    for request_id, payload in outputs.items():
                        item = _item_for_rid(request_id, chunk)
                        if item is None:
                            raise RuntimeError(f"cannot map request_id={request_id}")
                        audio = payload.get("audio")
                        if audio is None:
                            raise RuntimeError(f"{item['id']}/R{candidate}: missing audio")
                        tensor = audio.outputs[0].multimodal_output["audio"]
                        item_dir = out_model / item["id"]
                        item_dir.mkdir(parents=True, exist_ok=True)
                        wav_path = item_dir / f"R{candidate}.wav"
                        temporary_wav = item_dir / f"R{candidate}.tmp.{os.getpid()}.wav"
                        duration = save_wav(str(temporary_wav), tensor)
                        os.replace(temporary_wav, wav_path)
                        record = {
                            "item_id": item["id"], "condition": f"R{candidate}",
                            "seed": seed, "text": _text_of(payload),
                            "audio": str(wav_path), "audio_duration_sec": round(duration, 3),
                            "stack": "vllm-omni", "config_hash": chash,
                            "config": seed_config,
                        }
                        if duration > MAX_OUTPUT_AUDIO_SEC:
                            record["long_audio"] = True
                        atomic_json(item_dir / f"R{candidate}.json", record)
                        written += 1
                except Exception as exc:
                    errors.append(
                        f"R{candidate}[{start}:{start + len(chunk)}]: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    traceback.print_exc()
                print(
                    f"[vllm-resample] R{candidate} "
                    f"{min(start + len(chunk), len(pending))}/{len(pending)} "
                    f"batch_sec={time.time() - batch_started:.1f} errors={len(errors)}",
                    flush=True,
                )
    finally:
        omni.close()

    missing = []
    for item in items:
        for candidate, _ in candidates:
            record_path = out_model / item["id"] / f"R{candidate}.json"
            audio_path = record_path.with_suffix(".wav")
            if not record_path.is_file() or not audio_path.is_file():
                missing.append(f"{item['id']}/R{candidate}")
    if missing:
        errors.append(
            f"incomplete output: missing={len(missing)} examples={missing[:20]}"
        )

    summary = {
        "model": model_slug, "offset": args.offset, "limit": args.limit,
        "n_items": len(items), "seeds": seeds,
        "candidate_offset": args.candidate_offset, "n_expected": expected,
        "n_reused": reusable, "n_written": written, "n_missing": len(missing),
        "n_errors": len(errors),
        "errors": errors[:100], "load_sec": round(load_sec, 1),
        "wall_sec": round(time.time() - started, 1),
        "stage_config": runtime, "metrics_tag": args.metrics_tag,
    }
    metrics_path = metrics_dir / f"generate_{model_slug}_{args.metrics_tag}.json"
    atomic_json(metrics_path, summary)
    print(f"written -> {metrics_path}", flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
