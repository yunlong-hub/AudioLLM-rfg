#!/usr/bin/env python3
"""论文数值一致性检查：扫描稿件里的数值，标出**不在权威产物里**的。

背景：本项目发生过多次"论文数值与产物对不上"的事故，根因是手抄数值
跨越了多次抽取器/口径变更。`reports/numbers_audit.md` 是唯一权威来源，
本脚本把"稿件里的每个数值都能在产物里找到出处"变成可自动检查的约束。

稿件 = `papers/AudioLLM-rfg/main.tex`（当前唯一的论文载体；此前的
Markdown 稿已按用户要求删除，如需找回见 git 历史）。

用法：
    python tools/check_paper_numbers.py            # 检查并列出可疑数值
    python tools/check_paper_numbers.py --strict   # 有可疑值即非零退出

白名单（`docs/paper_numbers_whitelist.txt`，每行一个）用于登记确实不属于
产物范畴的常量（协议设定、外部引用、模型规格等）；未登记的疑点会全部列出，
便于逐条确认是"漏同步"还是"新的合法常量"。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_ROOT, "pyproject.toml")):
    _ROOT = os.path.dirname(_ROOT)

PAPERS = [
    "papers/AudioLLM-rfg/main.tex",
]
WHITELIST = "docs/paper_numbers_whitelist.txt"

# 稿件里出现的数值形态：0.1234 / 0.123 / 12.3% / 0.103--0.158 里的端点
NUM_RE = re.compile(r"(?<![\w.])(\d+\.\d+)(?![\w])")


def audit_values() -> list[float]:
    """把**全部权威产物**里的数值摊平成一个浮点列表。

    值域 = 审计 + 各 run 的 metrics（分数/分解/解剖/随机性/重采样/重渲染判定）
    + 负结果子组。只有能由这些产物（在四舍五入容差内）得到的数值才允许写进论文；
    其余必须登记进白名单。

    用数值容差而非字符串匹配，才能正确处理稿件的四舍五入（0.3426 → 0.343）
    与符号（−0.3426 的绝对值 0.343）。
    """
    import glob

    vals: list[float] = []

    def walk(x):
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, (int, float)) and not isinstance(x, bool):
            f = float(x)
            vals.append(f)
            vals.append(abs(f))
            vals.append(abs(f) * 100)   # 百分数写法

    sources = ["reports/numbers_audit.json"]
    sources += glob.glob(os.path.join(_ROOT, "exp", "*", "metrics", "*.json"))
    sources += glob.glob(os.path.join(_ROOT, "exp", "*", "resample_*.json"))
    sources += glob.glob(os.path.join(_ROOT, "reports", "*.json"))

    for rel in sources:
        p = rel if os.path.isabs(rel) else os.path.join(_ROOT, rel)
        if not os.path.exists(p):
            continue
        try:
            walk(json.load(open(p)))
        except Exception:
            continue

    # Markdown 报告里的数值（表格中的呈现值）也纳入
    for p in glob.glob(os.path.join(_ROOT, "reports", "*.md")):
        try:
            for m in re.finditer(r"-?\d+\.\d+", open(p).read()):
                f = float(m.group(0))
                vals += [f, abs(f), abs(f) * 100]
        except Exception:
            continue
    return vals


def is_known(value: str, known: list[float]) -> bool:
    """稿件里的字符串数值是否可由权威产物四舍五入得到。"""
    v = float(value)
    dec = len(value.split(".")[1])
    tol = 0.5 * (10 ** -dec) + 1e-9
    return any(abs(k - v) < tol for k in known)


def load_whitelist() -> set[str]:
    p = os.path.join(_ROOT, WHITELIST)
    if not os.path.exists(p):
        return set()
    out = set()
    for line in open(p):
        line = line.split("#", 1)[0].strip()
        if line:
            out.add(line)
    return out


def scan(path: str) -> dict[str, list[int]]:
    found: dict[str, list[int]] = {}
    full = os.path.join(_ROOT, path)
    if not os.path.exists(full):
        return found
    for lineno, line in enumerate(open(full), 1):
        for m in NUM_RE.finditer(line):
            found.setdefault(m.group(1), []).append(lineno)
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true",
                    help="存在未登记的可疑数值时以非零码退出")
    args = ap.parse_args()

    known = audit_values()
    allow = load_whitelist()
    print(f"审计数值 {len(known)} 个；白名单 {len(allow)} 条\n")

    total_suspicious = 0
    for path in PAPERS:
        found = scan(path)
        if not found:
            print(f"## {path}: 文件不存在，跳过")
            continue
        susp = {v: ls for v, ls in found.items()
                if not is_known(v, known) and v not in allow}
        print(f"## {path}")
        print(f"   数值个数 {len(found)}；未在审计/白名单中 {len(susp)}")
        for v, ls in sorted(susp.items()):
            lines = ",".join(str(x) for x in ls[:6])
            more = "…" if len(ls) > 6 else ""
            print(f"     {v:>10}  行 {lines}{more}")
        total_suspicious += len(susp)
        print()

    if total_suspicious:
        print(f"共 {total_suspicious} 个待确认数值。")
        print(f"逐个核对：确属审计口径的应当出现在 reports/numbers_audit.md；")
        print(f"确属协议常量的登记到 {WHITELIST}（每行一个）。")
    else:
        print("稿件中所有数值均可在权威审计或白名单中找到出处 ✓")
    return 1 if (args.strict and total_suspicious) else 0


if __name__ == "__main__":
    sys.exit(main())
