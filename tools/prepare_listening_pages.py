"""Create local listening pages from existing blind sheets, or import their labels.

No model/ASR text or source audio paths enter the distributed pages. Imported
labels are written to new files; the existing sheets and human data stay intact.
"""
from pathlib import Path
import argparse
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
SHEETS = ROOT / "data/adjudication/blinded"
OUT = ROOT / "data/adjudication/listening"
ROLES = ("annotator_1", "annotator_2")

PAGE = r'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><title>独立语音听辨</title>
<style>
body{font:16px/1.6 system-ui;background:#f4f6f9;color:#203044;margin:0}
main{max-width:850px;margin:35px auto;padding:28px;background:white;border-radius:12px}
h1{margin:0;font-size:25px}p{color:#566477}audio,textarea{width:100%;box-sizing:border-box}
textarea{padding:12px;border:1px solid #bbc8d6;border-radius:6px;font:15px/1.5 monospace}
label{display:block;margin:18px 0 6px;font-weight:600}
button,select{padding:9px 15px;border:1px solid #b9c7d5;border-radius:6px;background:white;cursor:pointer}
button.primary{background:#0072b2;color:white;border-color:#0072b2}nav{display:flex;gap:10px;flex-wrap:wrap;margin:20px 0}
#status{color:#0072b2}small{color:#566477}
</style><main>
<h1>独立语音听辨 · __ROLE__</h1>
<p>先听音频，再逐字转写并记录能听到的事实。不要查看模型、内部文本、ASR 输出或另一位标注者的答案。
听不清的内容不要补猜，可在备注中说明。每项完成后点击“保存本项”。</p>
<nav><button id="prev">上一项</button><select id="item"></select><button id="next">下一项</button>
<button id="download" class="primary">导出标注 JSONL</button></nav>
<div id="status"></div><h2 id="sample"></h2><audio id="audio" controls preload="none"></audio>
<label for="transcript">逐字转写</label><textarea id="transcript" rows="4"></textarea>
<small>仅在没有可辨识语音时留空。不要把原文整理成更正确的回答。</small>
<label for="facts">可听到的事实（JSON 数组）</label><textarea id="facts" rows="5"></textarea>
<small>格式示例（非本题答案）：[{"type":"number","value":"42","polarity":"+"}]。
类型可用 number、unit、proper_noun、negation、content；无可辨事实填 []。</small>
<label for="notes">备注／不确定之处</label><textarea id="notes" rows="2"></textarea>
<nav><button id="save" class="primary">保存本项</button><button id="load">载入已导出的 JSONL</button>
<input id="file" type="file" accept=".jsonl" hidden></nav>
<small>浏览器本地保存用于续做；请定期导出文件。关闭页面前先保存本项并导出。音频完全来自本地，不上传。</small>
</main><script>
const seed = __ROWS__;
const role = "__ROLE__";
const key = "rfg-listening-" + role + "-three-asr-sample100";
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
  const done = rows.filter(r => typeof r.annotation.transcript === "string" && Array.isArray(r.annotation.facts)).length;
  $("status").textContent = "已保存 " + done + " / " + rows.length + " 项";
}
function show() {
  const row = rows[index], a = row.annotation;
  $("item").value = index; $("sample").textContent = row.audit_id;
  $("audio").src = row.audio;
  $("transcript").value = a.transcript ?? "";
  $("facts").value = a.facts === null ? "" : JSON.stringify(a.facts, null, 2);
  $("notes").value = a.notes ?? ""; dirty=false; progress();
}
function persist() {
  try { localStorage.setItem(key, JSON.stringify(rows)); }
  catch(e) { alert("本地缓存不可用，请立即导出文件保存进度。"); }
}
function save() {
  try {
    const facts = JSON.parse($("facts").value);
    if (!Array.isArray(facts) || facts.some(f => !f || typeof f.type !== "string" || !f.type.trim()
      || typeof f.value !== "string" || !f.value.trim() || !["+","-"].includes(f.polarity)))
      throw Error("事实须为含 type、value、polarity 的数组，极性填写 + 或 -");
    rows[index].annotation = {transcript: $("transcript").value, facts, notes: $("notes").value || null};
    dirty=false; persist(); progress(); return true;
  } catch(e) { alert(e.message); return false; }
}
rows.forEach((r,i) => { const o = document.createElement("option"); o.value=i; o.textContent=(i+1)+" · "+r.audit_id; $("item").append(o); });
function navigate(target) { if (dirty && !save()) { $("item").value=index; return; } index=target; show(); }
for (const id of ["transcript","facts","notes"]) $(id).oninput=() => { dirty=true; };
$("item").onchange = () => navigate(Number($("item").value));
$("prev").onclick = () => navigate(Math.max(0,index-1));
$("next").onclick = () => navigate(Math.min(rows.length-1,index+1));
$("save").onclick = save;
$("download").onclick = () => {
  if (dirty && !save()) return;
  const text = rows.map(r => JSON.stringify({audit_id:r.audit_id,annotation:r.annotation})).join("\n")+"\n";
  const url = URL.createObjectURL(new Blob([text],{type:"application/x-ndjson"}));
  const a=document.createElement("a"); a.href=url; a.download=role+".labels.jsonl"; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
$("load").onclick = () => $("file").click();
$("file").onchange = async () => {
  try { restore((await $("file").files[0].text()).trim().split(/\r?\n/).map(JSON.parse)); persist(); show(); }
  catch(e) { alert("未载入："+e.message); }
};
show();
</script></html>'''


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sheet(role):
    return SHEETS / f"three_asr_sample100.{role}.jsonl"


def build():
    audio_dir = OUT / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    for role in ROLES:
        public_rows = []
        for row in read_rows(sheet(role)):
            name = row["audit_id"] + ".wav"
            shutil.copyfile(row["audio"], audio_dir / name)
            public_rows.append({"audit_id": row["audit_id"], "audio": "audio/" + name,
                                "annotation": row["annotation"]})
        html = PAGE.replace("__ROLE__", role).replace(
            "__ROWS__", json.dumps(public_rows, ensure_ascii=False).replace("<", "\\u003c"))
        output = OUT / f"{role}.html"
        output.write_text(html)
        print(output)


def import_labels(role, path):
    original = read_rows(sheet(role))
    labels = read_rows(path)
    indexed = {r["audit_id"]: r["annotation"] for r in labels}
    if len(indexed) != len(labels) or set(indexed) != {r["audit_id"] for r in original}:
        raise ValueError("label file must cover each source audit ID exactly once")
    for row in original:
        annotation = indexed[row["audit_id"]]
        if not isinstance(annotation.get("transcript"), str) or not isinstance(annotation.get("facts"), list):
            raise ValueError(f"unfinished annotation: {row['audit_id']}")
        row["annotation"] = annotation
    output = OUT / f"{role}.completed.jsonl"
    with output.open("x") as handle:
        for row in original:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--import-labels", type=Path)
    parser.add_argument("--annotator", choices=ROLES)
    args = parser.parse_args()
    if args.import_labels:
        if not args.annotator:
            parser.error("--import-labels requires --annotator")
        import_labels(args.annotator, args.import_labels)
    else:
        build()
