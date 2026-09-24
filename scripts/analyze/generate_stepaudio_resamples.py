#!/usr/bin/env python3
"""Generate fixed-seed Step-Audio candidates for the N=4 FRR experiment.

The original SPEAK artifact is candidate one.  This command adds R0/R1/R2
under ``exp/d0_resample/Step-Audio-2-mini/<item_id>/`` and is safe to shard or
resume: a candidate is reused only when both its JSON and non-empty WAV exist.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.infer import infer_stepaudio as step  # noqa: E402
from rfg.run.resample_generation import (  # noqa: E402
    atomic_json,
    candidate_complete,
    parse_seeds,
    seed_everything,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", default="data/pilot/items.jsonl")
    parser.add_argument("--pilot-pred", default="exp/d2_stepaudio/predictions")
    parser.add_argument("--out", default="exp/d0_resample")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--n", type=int, default=3, help="new candidates; original SPEAK is the fourth")
    parser.add_argument("--seeds", default="101,202,303")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds, args.n)
    rows = [json.loads(line) for line in Path(args.items).read_text().splitlines() if line.strip()]
    selected = rows[args.offset : args.offset + args.limit if args.limit else None]
    model_slug = "Step-Audio-2-mini"
    base_root = Path(args.pilot_pred) / model_slug
    output_root = Path(args.out) / model_slug
    output_root.mkdir(parents=True, exist_ok=True)

    eligible = []
    for item in selected:
        item_id = item["id"]
        base_json = base_root / item_id / "SPEAK.json"
        base_wav = base_root / item_id / "SPEAK.wav"
        if base_json.exists() and base_wav.exists():
            eligible.append(item)
    print(
        f"[step-resample] range={args.offset}:{args.offset + len(selected)} "
        f"eligible={len(eligible)} seeds={seeds}",
        flush=True,
    )
    if len(eligible) != len(selected):
        print(
            f"[step-resample] missing baseline artifacts for "
            f"{len(selected) - len(eligible)} items",
            flush=True,
        )
        return 1
    pending = sum(
        not candidate_complete(
            output_root / item["id"] / f"R{candidate_index}.json",
            output_root / item["id"] / f"R{candidate_index}.wav",
        )
        for item in eligible
        for candidate_index in range(len(seeds))
    )
    if not pending:
        print("[step-resample] all candidates already complete", flush=True)
        return 0

    model = step.StepAudio2(step.MODEL)
    token_to_wav = step.Token2wav(os.path.join(step.MODEL, "token2wav"))
    print(f"[step-resample] model loaded; pending={pending}", flush=True)

    completed = failed = 0
    for item_index, item in enumerate(eligible, 1):
        item_id = item["id"]
        question_audio = (item.get("question_audio") or {}).get("path")
        if not question_audio:
            print(f"[step-resample] {item_id}: missing question audio", flush=True)
            failed += len(seeds)
            continue
        output_dir = output_root / item_id
        output_dir.mkdir(parents=True, exist_ok=True)
        messages = [{
            "role": "human",
            "content": [
                {"type": "audio", "audio": question_audio},
                {"type": "text", "text": step.item_instruction(item)},
            ],
        }]
        for candidate_index, seed in enumerate(seeds):
            json_path = output_dir / f"R{candidate_index}.json"
            wav_path = output_dir / f"R{candidate_index}.wav"
            if candidate_complete(json_path, wav_path):
                completed += 1
                continue
            started = time.time()
            record = {"item_id": item_id, "seed": seed}
            try:
                seed_everything(seed)
                text_raw, speech_tokens = step.call(model, messages, want_audio=True)
                speech_tokens = [token for token in speech_tokens if token < 6561]
                if not speech_tokens:
                    raise RuntimeError("no valid speech tokens")
                audio_bytes = token_to_wav(speech_tokens, step.PROMPT_WAV)
                temporary_wav = wav_path.with_suffix(".tmp.wav")
                temporary_wav.write_bytes(audio_bytes)
                info = step._sf.info(temporary_wav)
                os.replace(temporary_wav, wav_path)
                record.update({
                    "text_raw": text_raw,
                    "text": step.clean_text(text_raw),
                    "audio": str(wav_path),
                    "audio_duration_sec": round(info.frames / info.samplerate, 3),
                    "n_speech_tokens": len(speech_tokens),
                    "latency_sec": round(time.time() - started, 3),
                })
                completed += 1
            except Exception as exc:  # retryable record for resumable long jobs
                record["error"] = f"{type(exc).__name__}: {exc}"
                failed += 1
            atomic_json(json_path, record)
        if item_index % 5 == 0 or item_index == len(eligible):
            print(
                f"[{item_index}/{len(eligible)}] candidates_ok={completed} errors={failed}",
                flush=True,
            )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
