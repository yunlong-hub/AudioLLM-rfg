"""重采样语料的事实装配：把 `exp/d0_resample/` 的样本与主口径内部文本对齐成一批可抽取文本。

背景
----
`scripts/analyze/probe_randomness.py`、`probe_resample.py`、`probe_rerender.py` 都要看
"同一题目的多个独立样本里，内部文本的每条事实被说出几次"。样本文本（`R*.json` 的 `asr1`）
不在 `exp/<run>/facts/<model>.jsonl` 里，必须重新做**双通道**抽取；内部文本与单样本回读
（`SPEAK`）则已在主口径产物中，直接作为 priors 复用，保证两个口径不会分裂。

键（key）约定与 `extract_facts.py` 对齐
--------------------------------------
* 内部文本：`<item_id>|SPEAK#internal`
* 单样本回读：`<item_id>|SPEAK`
* 重采样样本：`<item_id>|R<k>`；第二、三路回读分别追加 ``#asr2`` / ``#asr3``
"""
from __future__ import annotations

import json
import os
import re

from rfg.facts.extract import extraction_key

INTERNAL_COND = "SPEAK#internal"
BASE_COND = "SPEAK"
_R_RE = re.compile(r"^R(\d+)\.json$")


def load_resample_corpus(pred_root: str, resample_root: str, mslug: str,
                         ) -> tuple[list[dict], dict[str, str]]:
    """装配一题多样本的语料。

    返回 `(items, texts)`：

    * `items`：每题一条，含
      `item_id` / `internal_text` / `internal_key` / `sample_keys`（第 0 个是主口径的
      `SPEAK` 回读，其余是重采样样本，按 R 序号升序）；
    * `texts`：键 -> 文本，直接喂给 `rfg.facts.extract.extract_dual`。

    只收"内部文本与 `asr1` 都非空"的题（与历史探针一致：缺回读的题整体排除，不当作全损）。
    """
    mdir = os.path.join(resample_root, mslug)
    pdir = os.path.join(pred_root, mslug)
    if not os.path.isdir(mdir):
        raise FileNotFoundError(f"缺少重采样目录 {mdir}")
    if not os.path.isdir(pdir):
        raise FileNotFoundError(f"缺少主口径预测目录 {pdir}")

    items: list[dict] = []
    texts: dict[str, str] = {}
    for iid in sorted(os.listdir(mdir)):
        idir = os.path.join(mdir, iid)
        if not os.path.isdir(idir):
            continue
        sp = os.path.join(pdir, iid, "SPEAK.json")
        if not os.path.exists(sp):
            continue
        with open(sp) as fh:
            base = json.load(fh)
        base_readback = base.get("readback") or {}
        asr1 = base_readback.get("asr1")
        asr2 = base_readback.get("asr2")
        asr3 = base_readback.get("asr3")
        if not base.get("text") or not asr1 or not asr2:
            continue

        internal_key = extraction_key(iid, INTERNAL_COND)
        texts[internal_key] = base["text"]
        sample_keys = [extraction_key(iid, BASE_COND)]
        sample_keys_asr2 = [extraction_key(iid, BASE_COND + "#asr2")]
        sample_keys_asr3 = [extraction_key(iid, BASE_COND + "#asr3")] if asr3 else []
        asr3_complete = bool(asr3)
        texts[sample_keys[0]] = asr1
        texts[sample_keys_asr2[0]] = asr2
        if asr3:
            texts[sample_keys_asr3[0]] = asr3

        rfiles = sorted((f for f in os.listdir(idir) if _R_RE.match(f)),
                        key=lambda f: int(_R_RE.match(f).group(1)))
        for f in rfiles:
            with open(os.path.join(idir, f)) as fh:
                rec = json.load(fh)
            if not rec.get("asr1") or not rec.get("asr2"):
                continue
            condition = f[: -len(".json")]
            key1 = extraction_key(iid, condition)
            key2 = extraction_key(iid, condition + "#asr2")
            sample_keys.append(key1)
            sample_keys_asr2.append(key2)
            texts[key1] = rec["asr1"]
            texts[key2] = rec["asr2"]
            if asr3_complete and rec.get("asr3"):
                key3 = extraction_key(iid, condition + "#asr3")
                sample_keys_asr3.append(key3)
                texts[key3] = rec["asr3"]
            else:
                asr3_complete = False
                sample_keys_asr3 = []

        items.append({"item_id": iid, "internal_text": base["text"],
                      "internal_key": internal_key, "sample_keys": sample_keys,
                      "sample_keys_asr2": sample_keys_asr2,
                      "sample_keys_asr3": sample_keys_asr3})
    return items, texts
