#!/usr/bin/env python3
"""Build the E1-a human-anchoring listening packet (proportional stratified sample).

Why a separate packet from `data/adjudication/three_asr_sample100.*`
-------------------------------------------------------------------
The existing 100-item packet deliberately oversamples ASR-disagreement strata
(40/35/25) so that a small sample covers the difficult cases.  That makes it
unsuitable for estimating the human-versus-ASR gap over the population: the
stratum weights do not match the pool.

This tool draws a sample whose **stratum proportions match the candidate pool**
(795/385/206 -> 34/17/9 at n=60) and whose model split is also proportional
(30B 586, 3B 600, Step-Audio 200).  The per-item human transcript is then an
unbiased estimate of "facts a listener can recover", which is what the paper
needs in order to state an upper/lower bound around render loss.

Blinding
--------
The distributed page contains only `audit_id` and an anonymous audio file name
(`e1a_001.wav`, ...).  No model name, item id, internal text, ASR hypothesis, or
source path is embedded in the page or in the copied audio names.

This tool never overwrites: it refuses to run if its output manifest exists.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rfg.audit.adjudication import AuditSource, collect_candidates  # noqa: E402

DEFAULT_SOURCES = (
    "d1_main|Qwen3-Omni-30B-A3B-Instruct|exp/d1_main/facts/Qwen3-Omni-30B-A3B-Instruct.jsonl",
    "d1_main|Qwen2.5-Omni-3B|exp/d1_main/facts/Qwen2.5-Omni-3B.jsonl",
    "d2_stepaudio|Step-Audio-2-mini|exp/d2_stepaudio/facts/Step-Audio-2-mini.jsonl",
)
OUT = ROOT / "data/adjudication/e1a60"

PAGE = r'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><title>独立语音听辨 · __ROLE__</title>
<style>
body{font:16px/1.6 system-ui;background:#f4f6f9;color:#203044;margin:0}
main{max-width:820px;margin:32px auto;padding:26px;background:#fff;border-radius:12px}
h1{margin:0 0 6px;font-size:24px}p,small{color:#566477}audio,textarea{width:100%;box-sizing:border-box}
textarea{padding:12px;border:1px solid #bbc8d6;border-radius:6px;font:15px/1.5 monospace}
label{display:block;margin:16px 0 6px;font-weight:600}
button,select{padding:9px 15px;border:1px solid #b9c7d5;border-radius:6px;background:#fff;cursor:pointer}
button.primary{background:#0072b2;color:#fff;border-color:#0072b2}nav{display:flex;gap:10px;flex-wrap:wrap;margin:16px 0}
#status{color:#0072b2;font-weight:600}
</style><main>
<h1>独立语音听辨 · __ROLE__</h1>
<p>任务：听音频，把听到的话<b>逐字打出来</b>。不要纠正语法，不要改数字，不要参考任何模型输出。
听不清就按你听到的写；<b>完全没有可辨识语音时留空</b>，并在备注里说明。</p>
<p><small>进度保存在本机浏览器里；关闭页面前请先点“保存本项”并“导出标注 JSONL”。音频全部来自本地文件，不上传。</small></p>
<nav><button id="prev">上一项</button><select id="item"></select><button id="next">下一项</button>
<button id="download" class="primary">导出标注 JSONL</button></nav>
<div id="status"></div><h2 id="sample"></h2>
<audio id="audio" controls preload="none"></audio>
<label for="transcript">逐字转写</label><textarea id="transcript" rows="4"></textarea>
<label for="notes">备注／不确定之处（可留空）</label><textarea id="notes" rows="2"></textarea>
<nav><button id="save" class="primary">保存本项</button><button id="load">载入已导出的 JSONL</button>
<input id="file" type="file" accept=".jsonl" hidden></nav>
</main><script>
const seed = __ROWS__;
const role = "__ROLE__";
const key = "rfg-e1a60-" + role;
const $ = id => document.getElementById(id);
let rows = structuredClone(seed), index = 0, dirty = false;
function restore(incoming) {
  const map = new Map(incoming.map(r => [r.audit_id, r]));
  if (map.size !== seed.length || incoming.length !== seed.length || seed.some(r => !map.has(r.audit_id)))
    throw Error("样本编号与当前标注包不一致");
  rows = seed.map(r => ({...r, annotation: map.get(r.audit_id).annotation}));
}
try { const saved = localStorage.getItem(key); if (saved) restore(JSON.parse(saved)); }
catch(e) { alert("未载入本地缓存：" + e.message + "。可载入之前导出的 JSONL。"); }
function progress() {
  const done = rows.filter(r => typeof r.annotation.transcript === "string").length;
  $("status").textContent = "已保存 " + done + " / " + rows.length + " 项";
}
function show() {
  const row = rows[index], a = row.annotation;
  $("item").value = index; $("sample").textContent = row.audit_id;
  $("audio").src = row.audio;
  $("transcript").value = a.transcript ?? "";
  $("notes").value = a.notes ?? ""; dirty = false; progress();
}
function persist() {
  try { localStorage.setItem(key, JSON.stringify(rows)); }
  catch(e) { alert("本地缓存不可用，请立即导出文件保存进度。"); }
}
function save() {
  rows[index].annotation = {transcript: $("transcript").value,
                            facts: [], notes: $("notes").value || null};
  dirty = false; persist(); progress(); return true;
}
rows.forEach((r,i) => { const o = document.createElement("option"); o.value=i;
  o.textContent=(i+1)+" · "+r.audit_id; $("item").append(o); });
function navigate(target) { if (dirty) save(); index=target; show(); }
for (const id of ["transcript","notes"]) $(id).oninput = () => { dirty = true; };
$("item").onchange = () => navigate(Number($("item").value));
$("prev").onclick = () => navigate(Math.max(0,index-1));
$("next").onclick = () => navigate(Math.min(rows.length-1,index+1));
$("save").onclick = save;
$("download").onclick = () => {
  if (dirty) save();
  const text = rows.map(r => JSON.stringify({audit_id:r.audit_id,annotation:r.annotation})).join("\n")+"\n";
  const url = URL.createObjectURL(new Blob([text],{type:"application/x-ndjson"}));
  const a = document.createElement("a"); a.href=url; a.download=role+".labels.jsonl"; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
$("load").onclick = () => $("file").click();
$("file").onchange = async () => {
  try { restore((await $("file").files[0].text()).trim().split(/\r?\n/).map(JSON.parse)); persist(); show(); }
  catch(e) { alert("未载入："+e.message); }
};
show();
</script></html>'''


def _source(value: str) -> AuditSource:
    run_id, model_slug, facts_path = value.split("|", 2)
    return AuditSource(run_id, model_slug, Path(facts_path))


def _apportion(weights: dict[str, int], total: int) -> dict[str, int]:
    """Largest-remainder apportionment so the split matches the pool."""
    pool = sum(weights.values())
    exact = {k: total * v / pool for k, v in weights.items()}
    out = {k: int(v) for k, v in exact.items()}
    for key in sorted(exact, key=lambda k: (-(exact[k] - out[k]), k))[: total - sum(out.values())]:
        out[key] += 1
    return out


def proportional_sample(candidates: list[dict], n: int, seed: int) -> list[dict]:
    """Sample proportionally within each agreement stratum and each model."""
    rng = random.Random(seed)
    by_stratum_model: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in candidates:
        by_stratum_model[row["stratum"]][row["model"]].append(row)
    for cells in by_stratum_model.values():
        for rows in cells.values():
            rows.sort(key=lambda r: hashlib.sha256(
                f"{seed}:{r['run_id']}:{r['item_id']}".encode()).hexdigest())

    stratum_quota = _apportion(Counter(r["stratum"] for r in candidates), n)
    selected: list[dict] = []
    used: set[tuple[str, str]] = set()
    for stratum, amount in sorted(stratum_quota.items()):
        model_weights = {m: len(rows) for m, rows in by_stratum_model[stratum].items()}
        model_quota = _apportion(model_weights, amount)
        for model, take in sorted(model_quota.items()):
            for row in by_stratum_model[stratum][model]:
                if take == 0:
                    break
                if (row["run_id"], row["item_id"]) in used:
                    continue
                used.add((row["run_id"], row["item_id"]))
                selected.append(row)
                take -= 1
    rng.shuffle(selected)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", type=_source)
    parser.add_argument("--condition", default="SPEAK")
    parser.add_argument("--size", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()

    manifest = args.out_dir / "e1a60.manifest.json"
    if manifest.exists():
        raise SystemExit(f"refusing to overwrite existing packet: {manifest}")

    sources = args.source or [_source(v) for v in DEFAULT_SOURCES]
    candidates = collect_candidates(sources, args.condition)
    selected = proportional_sample(candidates, args.size, args.seed)

    rows: list[dict] = []
    audio_dir = args.out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    for number, row in enumerate(selected, start=1):
        audit_id = f"e1a_{number:03d}"
        shutil.copyfile(row["audio"], audio_dir / f"{audit_id}.wav")
        rows.append({
            "audit_id": audit_id,
            "condition": row["condition"],
            "stratum": row["stratum"],
            "category": row["category"],
            "model": row["model"],
            "run_id": row["run_id"],
            "item_id": row["item_id"],
            "source_audio": row["audio"],
            "internal_text": row["internal_text"],
            "asr": row["asr"],
            "pairwise_wer": row["pairwise_wer"],
        })

    master = args.out_dir / "e1a60.master.jsonl"
    with master.open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    for role in ("annotator_1", "annotator_2"):
        public_rows = [{"audit_id": r["audit_id"], "audio": f"audio/{r['audit_id']}.wav",
                        "annotation": {"transcript": None, "facts": [], "notes": None}}
                       for r in rows]
        (args.out_dir / f"{role}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in public_rows))
        html = PAGE.replace("__ROLE__", role).replace(
            "__ROWS__", json.dumps(public_rows, ensure_ascii=False).replace("<", "\\u003c"))
        (args.out_dir / f"{role}.html").write_text(html)

    summary = {
        "protocol": ("proportional stratified sample of the SPEAK candidate pool; "
                     "stratum and model splits follow the pool, unlike the 100-item "
                     "disagreement-enriched packet"),
        "seed": args.seed,
        "n_candidates": len(candidates),
        "n_selected": len(rows),
        "pool_strata": dict(Counter(r["stratum"] for r in candidates)),
        "selected_strata": dict(Counter(r["stratum"] for r in rows)),
        "pool_models": dict(Counter(r["model"] for r in candidates)),
        "selected_models": dict(Counter(r["model"] for r in rows)),
        "selected_categories": dict(Counter(r["category"] for r in rows)),
        "condition": args.condition,
        "master": str(master.relative_to(ROOT)),
        "role": "human transcription only; facts stay empty and are filled later by "
                "the shared extractor on the adjudicated transcript",
        "sources": [s.__dict__ | {"facts_path": str(s.facts_path)} for s in sources],
    }
    manifest.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in
                      ("n_selected", "selected_strata", "selected_models")},
                     ensure_ascii=False, indent=2))
    print(f"master : {master}")
    print(f"pages  : {args.out_dir}/annotator_1.html , annotator_2.html")
    print(f"audio  : {audio_dir}")

    # Stratum weights of the pool, for unbiased reweighting during scoring.
    weights = {k: v / len(candidates) for k, v in summary["pool_strata"].items()}
    (args.out_dir / "e1a60.stratum_weights.json").write_text(
        json.dumps(weights, ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
