"""论文↔产物 数值一致性回归测试。

本项目发生过多次"论文数值与产物对不上"的事故（口径分裂、抽取器中途变更、
手抄数值跨越版本）。`tools/check_paper_numbers.py` 把"稿件里每个数值都能
由 reports/ 与 exp/ 下的权威产物在四舍五入容差内得到"变成可自动检查的约束，
本测试确保该约束在每次改动后仍然成立。

若失败：先看它列出的数值。确属旧口径残留 → 同步稿件；确属合法常量 →
登记到 `docs/paper_numbers_whitelist.txt`；确属该落盘却没落盘 → 让对应探针
把该值写进产物，而不是加白名单。
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKER = os.path.join(ROOT, "tools", "check_paper_numbers.py")


@pytest.mark.skipif(not os.path.exists(CHECKER), reason="检查器不存在")
def test_paper_numbers_traceable_to_artifacts():
    r = subprocess.run([sys.executable, CHECKER, "--strict"],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, (
        "稿件中存在无法由权威产物解释的数值（旧口径残留或未登记的常量）：\n"
        + r.stdout + r.stderr
    )
