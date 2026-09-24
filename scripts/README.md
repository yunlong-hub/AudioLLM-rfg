# scripts/ — 入口脚本与流水线

本目录只放**可执行入口**；可复用的实现全部在 `src/rfg/`（已 `pip install -e .`，任意 cwd 下 `import rfg` 均可用）。

所有脚本都从**仓库根**读写相对路径（`data/`、`exp/`、`reports/`）——移动文件后脚本会自行向上查找到含 `pyproject.toml` 的目录并 `cd` 到那里，见各文件顶部的 `_ROOT` 解析。

## 目录职责

| 目录 | 职责 |
| --- | --- |
| `data/` | 构造评测题目与题面音频 |
| `infer/` | 跑条件推理，产出 `exp/<run_id>/predictions/` |
| `facts/` | 回读 → 事实抽取 → 打分（主指标） |
| `analyze/` | 各类探针与闸门判定，产出 `reports/*.md` |
| `pipelines/` | 把上面若干步串起来的 shell 入口 |
| `diagnostics/` | 单模型排障脚本（非主流水线，按需手动跑） |

## 主流水线

```
items.jsonl ──infer──> predictions/ ──readback──> readback/*.json
                                          │
                                          ├─extract_facts──> facts/*.json
                                          │                      │
                                          └──────────────── score ┘──> metrics.json
                                                                       │
                                                                  analyze/* ──> reports/*.md
```

### 端到端命令

```bash
PY=/workspace/yunlong/anaconda3/envs/audio-llm/bin/python

# 1) 题目与题面音频（通常只需一次）
$PY scripts/data/build_items.py --split pilot     # -> data/pilot/items.jsonl + 音频
$PY scripts/data/build_items.py --split main      # -> data/main600/items.jsonl + 音频
$PY scripts/data/tts_questions.py                 # 题面 TTS（若 build_items 未含）

# 2) 推理（分片可选；见「分片」一节）
scripts/pipelines/run_infer.sh audio-llm <model_path> <run_id> - 0 0 <gpu>

# 3) 后处理四步（回读 → 抽取 → 打分 → 闸门）
scripts/pipelines/run_d0_pipeline.sh <run_id> <model_slug> <gpu>

# 4) 机制三探针（分解 / 2×2 / 长度）
scripts/pipelines/run_mechanism_pipeline.sh <run_id> <model_slug> <gpu>
```

## 各脚本

### `data/`

| 脚本 | 作用 |
| --- | --- |
| `build_items.py` | 由模板生成 pilot(200) / main(600) 题目，含事实标注，并合成题面音频 |
| `tts_questions.py` | 题面音频的 TTS 引擎封装入口 |

### `infer/`

| 脚本 | 作用 |
| --- | --- |
| `infer.py` | 六条件推理主入口（READ/LISTEN/SPEAK/ECHO/EF/EFA/EFB/EFW/SPEAKD） |
| `infer_stepaudio.py` | Step-Audio-2-mini 专用推理（独立运行时与 audio token 处理） |
| `smoke.py` | 单模型冒烟：能否加载、能否出音频 |

### `facts/`

| 脚本 | 作用 |
| --- | --- |
| `readback.py` | 双 ASR 回读（Whisper-large-v3 + Seamless-M4T-v2）与一致率/WER |
| `extract_facts.py` | 双通道事实抽取（规则 + LLM），含 κ 与合并 |
| `score.py` | 主指标：PG/RFG/RG/RFG_EF/分项/RFG_corr + 按题聚类 bootstrap |

### `analyze/`

| 脚本 | 产出 | 作用 |
| --- | --- | --- |
| `d0_gate.py` | `reports/d0_gate.md` | D0 通行闸门（数据完整性与指标下限） |
| `probe_ef.py` | `reports/probe_ef*.md` | 三段分解 Δ_plan / Δ_render / RFG |
| `probe_grid.py` | `reports/mechanism*.md` | 2×2（输入模态 × 内容来源） |
| `probe_length.py` | `reports/length_control*.md` | 长度混杂控制（Spearman ρ） |
| `probe_taxonomy.py` | `reports/taxonomy*.md` | SPEAK 失败分类（sub/del/drift/pol） |
| `probe_randomness.py` | `reports/randomness.md` | 事实内 vs 事实间方差、位置置换检验、分型结构地板 |
| `probe_resample.py` | `exp/<run>/resample_*.json` | 事实一致性重采样（FRR，N=4）+ 评估器条件下的 candidate oracle |
| `probe_rerender.py` | — | 定向重渲染执行（产出新音频） |
| `judge_rerender.py` | `reports/rerender.md` | 定向重渲染的效果判定 |
| `probe_subgroups.py` | `reports/negative_contrasts.md` | SPEAKD 干预 与 数字面子组两个负结果对比 |
| `probe_stepaudio_gate.py` | — | Step-Audio 内部文本可取出性闸门 |

跨脚本复用：`probe_grid.py` / `probe_length.py` 从同目录的 `probe_ef.py` 导入
`INTERNAL` / `bootstrap_diff` / `load_facts` / `pooled`。**这几个文件必须留在同一目录。**

### `pipelines/`

| 脚本 | 用法 |
| --- | --- |
| `run_infer.sh` | `run_infer.sh <env> <model_path> <run_id> [conditions] [offset] [limit] [gpu]` |
| `run_d0_pipeline.sh` | `run_d0_pipeline.sh <run_id> [model_slug] [gpu]` |
| `run_mechanism_pipeline.sh` | `run_mechanism_pipeline.sh <run_id> <model_slug> [gpu] [items]` |

环境变量 `PY` / `LLM_MODEL` 可覆盖默认解释器与抽取用 LLM。

### `diagnostics/`

Qwen2.5-Omni-7B 语音输出损坏的排查脚本（**待办实验**用，非主流水线）：

| 脚本 | 作用 |
| --- | --- |
| `smoke_7b.py` | 复核 7B 语音输出是否真的损坏 |
| `smoke7b_knobs.py` | talker 采样 knob 扫描 |
| `probe_7b_talker.py` | 采样配置 × 可懂度，判断能否作为第三个数据点 |
| `probe_stepid_text.py` | Step-Audio 内部文本可取出性门槛测试 |

## 运行约定与坑

- **只用空闲卡**：跑前 `nvidia-smi` 确认；优先级 A22 → A23 → A31 → A42。
- **A42 需要单独 venv**：驱动 525.60.13 配 cu130 的 torch 会失败，用
  `.venvs/audio_llm_cu12`（py3.10 / torch 2.7.1+cu126），走 `run_infer.sh llm_a42 ...`。
- **分片反而更慢**：`/workspace` 是共享 NFS，多进程并行读同一批音频常比单进程串行慢
  （30B 回读：串行 586 条约 1.5 h；4 路并行 10 min 只推进 4 条）。除非确实需要，别分片。
- **不要并发写同一文件**：历史上出现过两个写者把两个 JSON 对象拼进同一个文件导致解析崩溃。
- **`reports/numbers_audit.md` 是论文数值的唯一权威来源**，由 `tools/audit_numbers.py` 生成。
  改抽取器后必须重跑该脚本，并逐项核对 `papers/AudioLLM-rfg/main.tex`。

## 相关工具

- `tools/audit_numbers.py` — **数值审计（论文数值唯一权威来源）**。同时输出"分解链（共同样本集）"
  与"主指标（来自 `score.py`）"两节，并与 `probe_ef.json` 交叉校验，分歧即非零退出。
- `tools/check_paper_numbers.py` — **论文数值一致性检查**。扫描两份稿件里的每个数值，
  标出无法由 `reports/`+`exp/` 权威产物解释的项；`--strict` 有疑点即非零退出
  （已接入 `pytest`）。合法常量登记在 `docs/paper_numbers_whitelist.txt`。
- `tools/freeze_inventory.py` — 工具链冻结清单

### 改数值时的正确顺序

```bash
# 1) 重跑受影响的探针（它们会重写 reports/*.md 与 exp/*/metrics/*.json）
# 2) 重跑审计，确认交叉校验 0 分歧
python tools/audit_numbers.py
# 3) 同步稿件，然后验证每个数值都有出处
python tools/check_paper_numbers.py --strict
pytest tests/ -q
```

**不要手工重算论文里的数**。若某个量在产物里没有，正确做法是让对应探针把它落盘
（例如分型结构地板由 `probe_randomness.py` 写入 `randomness.json` 的 `by_type`），
而不是把算出来的值直接抄进稿件——本项目已经因为手抄数值跨越抽取器变更而出过事故。

## 常见坑

- **事实抽取只有一个入口**：新文本一律走 `src/rfg/facts/extract.py::extract_dual()`
  （规则 ∪ LLM → `merge()` → 按 `text_hash`+`prompt_sha256` 缓存）。在任何探针里直接
  `extract_rules()` 当事实源都会造成第二套口径——历史上审计与三个探针都犯过这个错。
- **涉及集合/字典迭代的合并逻辑必须验证跨进程确定性**：`merge()` 曾只按 value 合并，
  而 `number:95:+` 与 `negation:95:-` 共享同一 value，覆盖顺序随 `PYTHONHASHSEED` 变化，
  导致同一文本在不同进程抽出不同事实集。用多 `PYTHONHASHSEED` 比对产物 md5。
