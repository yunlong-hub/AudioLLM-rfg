#!/usr/bin/env python3
"""Generate the adjudication page for the E1-a packet.

The third listener hears each clip again and resolves the two transcripts: pick
candidate A or B, or type a corrected version.  The page embeds only the
anonymous audio and the two candidates; model identity, internal text, and the
ASR hypotheses are never included.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKET = ROOT / "data/adjudication/e1a60"

PAGE = r'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><title>分歧条目裁定</title>
<style>
body{font:16px/1.6 system-ui;background:#f4f6f9;color:#203044;margin:0}
main{max-width:880px;margin:32px auto;padding:26px;background:#fff;border-radius:12px}
h1{margin:0 0 6px;font-size:24px}p,small{color:#566477}audio,textarea{width:100%;box-sizing:border-box}
textarea{padding:12px;border:1px solid #bbc8d6;border-radius:6px;font:15px/1.5 monospace}
label{display:block;margin:16px 0 6px;font-weight:600}
.cand{border:1px solid #dbe3ec;border-radius:8px;padding:10px 12px;margin:8px 0;background:#fbfcfe}
.cand b{color:#0072b2}
button,select{padding:9px 15px;border:1px solid #b9c7d5;border-radius:6px;background:#fff;cursor:pointer}
button.primary{background:#0072b2;color:#fff;border-color:#0072b2}
nav{display:flex;gap:10px;flex-wrap:wrap;margin:16px 0}#status{color:#0072b2;font-weight:600}
</style><main>
<h1>分歧条目裁定</h1>
<p>请再听一遍音频，判断两位听辨者哪一份转写更接近你听到的内容：
选 A 或选 B 会自动填入下方文本框，你也可以直接修改成最终版本。
若两份都不准确，以你听到的为准填写。<b>没有任何可辨识语音时留空</b>。</p>
<nav><button id="prev">上一项</button><select id="item"></select><button id="next">下一项</button>
<button id="download" class="primary">导出裁定 JSONL</button></nav>
<div id="status"></div><h2 id="sample"></h2>
<audio id="audio" controls preload="none"></audio>
<label>听辨者 A</label><div class="cand"><b>A</b> <span id="ca"></span>
<button id="pickA">采用 A</button></div>
<label>听辨者 B</label><div class="cand"><b>B</b> <span id="cb"></span>
<button id="pickB">采用 B</button></div>
<label for="final">最终转写（可编辑）</label><textarea id="final" rows="3"></textarea>
<label for="notes">备注（可留空）</label><textarea id="notes" rows="2"></textarea>
<nav><button id="save" class="primary">保存本项</button><button id="load">载入已导出的 JSONL</button>
<input id="file" type="file" accept=".jsonl" hidden></nav>
<p><small>进度保存在本机浏览器；关闭页面前请先保存并导出。音频来自本地文件，不上传。</small></p>
</main><script>
const seed = __ROWS__;
const key = "rfg-e1a60-adjudication";
const $ = id => document.getElementById(id);
let rows = structuredClone(seed), index = 0, dirty = false;
function restore(incoming) {
  const map = new Map(incoming.map(r => [r.audit_id, r]));
  if (map.size !== seed.length || incoming.length !== seed.length || seed.some(r => !map.has(r.audit_id)))
    throw Error("样本编号与当前裁定包不一致");
  rows = seed.map(r => ({...r, adjudicated: map.get(r.audit_id).adjudicated}));
}
try { const saved = localStorage.getItem(key); if (saved) restore(JSON.parse(saved)); }
catch(e) { alert("未载入本地缓存：" + e.message); }
function progress() {
  const done = rows.filter(r => typeof r.adjudicated.transcript === "string").length;
  $("status").textContent = "已裁定 " + done + " / " + rows.length + " 项";
}
function show() {
  const row = rows[index], a = row.adjudicated;
  $("item").value = index; $("sample").textContent = row.audit_id;
  $("audio").src = row.audio;
  $("ca").textContent = row.candidate_a || "（空）";
  $("cb").textContent = row.candidate_b || "（空）";
  $("final").value = a.transcript ?? ""; $("notes").value = a.notes ?? "";
  dirty = false; progress();
}
function persist() {
  try { localStorage.setItem(key, JSON.stringify(rows)); }
  catch(e) { alert("本地缓存不可用，请立即导出。"); }
}
function save() {
  rows[index].adjudicated = {transcript: $("final").value, facts: [], notes: $("notes").value || null};
  dirty = false; persist(); progress(); return true;
}
function navigate(target) { if (dirty) save(); index = target; show(); }
rows.forEach((r,i) => { const o = document.createElement("option"); o.value=i;
  o.textContent=(i+1)+" · "+r.audit_id; $("item").append(o); });
$("item").onchange = () => navigate(Number($("item").value));
$("prev").onclick = () => navigate(Math.max(0,index-1));
$("next").onclick = () => navigate(Math.min(rows.length-1,index+1));
$("pickA").onclick = () => { $("final").value = rows[index].candidate_a || ""; dirty = true; };
$("pickB").onclick = () => { $("final").value = rows[index].candidate_b || ""; dirty = true; };
$("save").onclick = save;
for (const id of ["final","notes"]) $(id).oninput = () => { dirty = true; };
$("download").onclick = () => {
  if (dirty) save();
  const text = rows.map(r => JSON.stringify({audit_id:r.audit_id,adjudicated:r.adjudicated})).join("\n")+"\n";
  const url = URL.createObjectURL(new Blob([text],{type:"application/x-ndjson"}));
  const a = document.createElement("a"); a.href=url; a.download="adjudication.labels.jsonl"; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
$("load").onclick = () => $("file").click();
$("file").onchange = async () => {
  try { restore((await $("file").files[0].text()).trim().split(/\r?\n/).map(JSON.parse)); persist(); show(); }
  catch(e) { alert("未载入："+e.message); }
};
show();
</script></html>'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing page (never touches human label files)")
    args = parser.parse_args()

    sheet_path = args.packet / "adjudication.jsonl"
    if not sheet_path.exists():
        raise SystemExit(f"run tools/merge_e1a_labels.py first; missing {sheet_path}")
    rows = [json.loads(line) for line in sheet_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    output = args.packet / "adjudication.html"
    if output.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite {output} (use --force to regenerate)")
    html = PAGE.replace("__ROWS__", json.dumps(rows, ensure_ascii=False).replace("<", "\\u003c"))
    output.write_text(html, encoding="utf-8")
    print(f"adjudication page: {output}")
    print(f"items: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
