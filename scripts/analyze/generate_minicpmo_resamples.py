#!/usr/bin/env python3
"""Generate fixed-text, fixed-seed MiniCPM-o candidates for incremental FRR pools.

The existing SPEAK artifact is candidate one. Candidate indices can start at a
non-zero offset, so an existing N=4 pool (R0--R2) can be extended to N=8
(R3--R6) without rewriting prior artifacts. Exact equality with the baseline
internal text is enforced so the experiment isolates speech rendering rather
than answer-generation variance.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import soundfile as sf

from rfg.models.registry import load_s2s_model
from rfg.run.conditions import item_instruction
from rfg.run.resample_generation import atomic_json, candidate_complete, indexed_seeds


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="pretrain_model/Audio/MiniCPM-o-4_5")
    parser.add_argument("--items", default="data/main600/items.jsonl")
    parser.add_argument("--base-pred", default="exp/d1_main/predictions")
    parser.add_argument("--out", default="exp/d0_resample")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--n", type=int, default=3,
                        help="new candidates; existing SPEAK is candidate one")
    parser.add_argument("--seeds", default="101,202,303")
    parser.add_argument("--candidate-offset", type=int, default=0,
                        help="first R index; use 3 when extending N=4 to N=8")
    args = parser.parse_args()

    candidates = indexed_seeds(args.seeds, args.n, args.candidate_offset)
    rows = [json.loads(line) for line in Path(args.items).read_text().splitlines() if line.strip()]
    selected = rows[args.offset : args.offset + args.limit if args.limit else None]
    model_slug = Path(args.model).name
    base_root = Path(args.base_pred) / model_slug
    output_root = Path(args.out) / model_slug
    output_root.mkdir(parents=True, exist_ok=True)

    eligible: list[tuple[dict, str]] = []
    for item in selected:
        base_json = base_root / item["id"] / "SPEAK.json"
        base_wav = base_root / item["id"] / "SPEAK.wav"
        if not base_json.exists() or not base_wav.exists():
            continue
        base = json.loads(base_json.read_text())
        if base.get("error") or not base.get("text"):
            continue
        eligible.append((item, base["text"].strip()))
    print(
        f"[minicpmo-resample] range={args.offset}:{args.offset + len(selected)} "
        f"eligible={len(eligible)} candidates={candidates}",
        flush=True,
    )
    if len(eligible) != len(selected):
        print(f"[minicpmo-resample] missing valid baseline artifacts for "
              f"{len(selected) - len(eligible)} items", flush=True)
        return 1

    pending = sum(
        not candidate_complete(
            output_root / item["id"] / f"R{candidate_index}.json",
            output_root / item["id"] / f"R{candidate_index}.wav",
        )
        for item, _ in eligible
        for candidate_index, _ in candidates
    )
    if not pending:
        print("[minicpmo-resample] all candidates already complete", flush=True)
        return 0

    model = load_s2s_model(args.model, device_map="auto")
    print(f"[minicpmo-resample] model loaded; pending={pending}", flush=True)
    completed = failed = 0
    for item_index, (item, baseline_text) in enumerate(eligible, 1):
        item_id = item["id"]
        question_audio = (item.get("question_audio") or {}).get("path")
        if not question_audio:
            print(f"[minicpmo-resample] {item_id}: missing question audio", flush=True)
            failed += len(candidates)
            continue
        output_dir = output_root / item_id
        output_dir.mkdir(parents=True, exist_ok=True)
        content = [
            {"type": "audio", "audio": question_audio},
            {"type": "text", "text": item_instruction(item)},
        ]
        for candidate_index, seed in candidates:
            json_path = output_dir / f"R{candidate_index}.json"
            wav_path = output_dir / f"R{candidate_index}.wav"
            if candidate_complete(json_path, wav_path):
                completed += 1
                continue
            started = time.time()
            record = {
                "item_id": item_id,
                "seed": seed,
                "protocol": "deterministic semantic decode; stochastic TTS seed",
                "temperature": 0.0,
            }
            try:
                result = model.chat(content, want_audio=True, temperature=0.0, seed=seed)
                generated_text = (result.text or "").strip()
                if generated_text != baseline_text:
                    raise RuntimeError(
                        "semantic text changed under deterministic decoding: "
                        f"baseline={baseline_text!r} generated={generated_text!r}"
                    )
                temporary_wav = wav_path.with_suffix(".tmp.wav")
                model.save_wav(str(temporary_wav), result)
                info = sf.info(temporary_wav)
                temporary_wav.replace(wav_path)
                record.update({
                    "text": generated_text,
                    "baseline_text": baseline_text,
                    "text_matches_baseline": True,
                    "audio": str(wav_path),
                    "audio_duration_sec": round(info.frames / info.samplerate, 3),
                    "latency_sec": round(time.time() - started, 3),
                    "sample_rate": info.samplerate,
                    "tts_sampling": {
                        "top_p": 0.85,
                        "min_p": 0.01,
                        "top_k": 25,
                        "repetition_penalty": 1.05,
                        "temperature": 0.8,
                    },
                })
                completed += 1
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                failed += 1
            atomic_json(json_path, record)
        if item_index % 5 == 0 or item_index == len(eligible):
            print(f"[{item_index}/{len(eligible)}] candidates_ok={completed} errors={failed}",
                  flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
