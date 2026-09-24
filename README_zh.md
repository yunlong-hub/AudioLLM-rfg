# AudioLLM-RFG

<p align="center"><a href="README.md">English</a> · <strong>简体中文</strong></p>

论文 **Reusing the Internal Spoken Plan: Localizing Loss and Recovering Facts in Speech Language Models** 的官方实现。

[论文（即将公开）](#引用) · [代码](https://github.com/yunlong-hub/AudioLLM-rfg)

## 项目简介

语音语言模型可能在内部文本中保留了正确事实，却在生成的语音波形中丢失这些事实。AudioLLM-RFG 沿生成路径比较带符号的原子事实，从而定位事实损失：

```text
文本或语音输入
      ↓
内部口语计划
      ↓
生成的语音波形
      ↓
独立 ASR 回读
```

该框架区分两类错误来源：

- **计划损失（Plan loss）：** 模型从纯文本输出切换到语音输出时，内部答案中的事实发生变化。
- **渲染损失（Render loss）：** 内部口语计划中存在的事实无法再从生成的语音中恢复。

![AudioLLM-RFG 路径审计与计划引导恢复框架](docs/assets/method_overview.svg)

上图为论文中的方法总览：内部口语计划既是定位事实损失的共享参照，也是候选语音恢复的选择依据。

该方法进一步复用内部计划，从多个语音候选中选择更忠实的结果。候选选择与最终评测使用互斥的 ASR 视角，因此不需要外部参考答案，也不会用参与选择的识别器评测自身结果。

## 主要特性

- 沿纯文本答案、内部口语计划、语音波形和 ASR 回读进行路径级定位。
- 使用带符号原子事实表示数字、单位、命名实体、短事实内容和极性。
- 通过固定文本和匹配输入控制，区分答案内容变化与语音实现错误。
- 使用 Whisper、SeamlessM4T 和 Fun-ASR 三个识别器进行评测。
- 采用留出 ASR 的候选选择和 leave-one-ASR-out 评测。
- 提供 `N = 1, 2, 4, 8` 的候选预算曲线。
- 按问题聚类计算 Bootstrap 置信区间。

## 评测模型

论文评测了三个模型家族的五个检查点：

| 家族 | 检查点 | 论文采用的推理栈 |
| --- | --- | --- |
| Qwen | Qwen3-Omni-30B-A3B-Instruct | Transformers |
| Qwen | Qwen2.5-Omni-3B | Transformers |
| Qwen | Qwen2.5-Omni-7B | vLLM-Omni |
| Step-Audio | Step-Audio-2-mini | Step-Audio 官方运行时 |
| MiniCPM | MiniCPM-o-4.5 | Transformers / 模型自带 chat 接口 |

三个识别器视角分别为 Whisper-large-v3、SeamlessM4T-v2-large 和 Fun-ASR-Nano-2512。

## 仓库结构

```text
AudioLLM-rfg/
├── configs/              # 可移植的解码与环境配置
├── docs/                 # 方法与复现说明
├── scripts/
│   ├── data/             # 受控数据构建与问题语音合成
│   ├── infer/            # 语音语言模型推理
│   ├── facts/            # ASR 回读、事实抽取与评分
│   ├── analyze/          # 控制实验、稳健性检查与恢复分析
│   └── pipelines/        # 端到端流程入口
├── src/rfg/
│   ├── data/             # 受控样本构建
│   ├── models/           # 模型与 ASR 适配器
│   ├── facts/            # 事实格式、抽取与回读处理
│   ├── score/            # 路径损失与候选选择指标
│   └── run/              # 共用执行逻辑
├── tests/                # 单元测试与契约测试
├── tools/                # 审计、分析与绘图工具
└── pyproject.toml
```

模型权重、数据集、生成语音、预测结果、日志和实验产物均不会提交到 Git。

## 安装

项目要求 Python 3.10 或更高版本。建议创建隔离环境并以可编辑模式安装：

```bash
git clone https://github.com/yunlong-hub/AudioLLM-rfg.git
cd AudioLLM-rfg

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

五个语音模型并不共用同一套推理栈。请按照各上游模型的官方文档和许可证分别安装。论文中，Qwen2.5-Omni-7B 使用 vLLM-Omni，Step-Audio-2-mini 使用独立的官方运行时。

本仓库不会自动下载任何模型权重。

## 快速开始

### 1. 验证受控数据生成器

```bash
python scripts/data/build_items.py --split pilot --verify-only
```

生成包含 200 个问题的 pilot 清单：

```bash
python scripts/data/build_items.py \
  --split pilot \
  --out data/pilot/items.jsonl
```

Pilot 包含五类问题，每类 40 个：数字、单位、专有名词、否定和短事实内容。

### 2. 生成模型输出

准备好问题语音和上游模型权重后，运行：

```bash
python scripts/infer/infer.py \
  --model /path/to/speech-language-model \
  --items data/pilot/items.jsonl \
  --run-id pilot \
  --conditions READ,LISTEN,SPEAK,ECHO,EF
```

五种受控条件如下：

| 条件 | 输入 | 输出 | 用途 |
| --- | --- | --- | --- |
| `READ` | 文本 | 文本 | 文本输入基线 |
| `LISTEN` | 语音 | 文本 | 语音输入的计划基线 |
| `SPEAK` | 语音 | 文本 + 语音 | 主要语音输出路径 |
| `ECHO` | 文本 | 文本 + 语音 | 输入模态控制 |
| `EF` | 固定文本 | 语音 | 固定内容渲染控制 |

### 3. 回读、抽取事实并评分

后续阶段读取 `exp/<run_id>/predictions/` 中的结果：

```bash
# ASR 回读
python scripts/facts/readback.py --run-id pilot

# 确定性的规则事实抽取
python scripts/facts/extract_facts.py --run-id pilot

# 可选：规则 + LLM 事实抽取
python scripts/facts/extract_facts.py \
  --run-id pilot \
  --with-llm \
  --llm-model /path/to/Qwen2.5-7B-Instruct

# 路径级评分
python scripts/facts/score.py \
  --run-id pilot \
  --items data/pilot/items.jsonl
```

重新计算识别器特定结果时，可使用 `--readback-mode asr1`、`asr2`、`asr3` 或可用的 union 模式。各阶段参数见 `python <script> --help` 和 [`scripts/README.md`](scripts/README.md)。

### 4. 评估候选恢复

候选生成和留出 ASR 评测由以下脚本实现：

- `scripts/analyze/generate_vllm_omni_resamples.py`
- `scripts/analyze/generate_stepaudio_resamples.py`
- `scripts/analyze/generate_minicpmo_resamples.py`
- `scripts/analyze/evaluate_frr_loo.py`
- `scripts/analyze/evaluate_frr_curve.py`

这些脚本将输出写入已被 Git 忽略的 `exp/` 目录。

## 主要结果

### 路径审计能够定位事实损失发生的位置

![五个语音语言模型检查点的路径级定位结果](docs/assets/path_results.svg)

在五个检查点上：

- Qwen3-Omni-30B、Qwen2.5-Omni-3B/7B 和 MiniCPM-o 的内部口语计划几乎保留全部抽取事实，计划损失仅为 `0.0%–0.1%`；但在单个识别器下，语音回读会损失 `11.7%–22.2%` 的事实。
- Step-Audio-2 呈现互补的计划主导型错误，说明该审计能够区分两种真实存在的失败模式，而非将所有错误都归因于语音渲染。
- VoiceBench 结果把计划到波形的事实损失与下游任务准确率联系起来：即使内部计划正确，回读准确率仍可能下降。

### 内部计划可以指导恢复

![计划引导选择与相同文本恢复结果](docs/assets/recovery_results.svg)

候选数为 8 时，计划引导的事实选择在五个检查点上将参考计划损失降低 `2.5–11.3` 个百分点，并在参考计划保真度上优于随机选择和不使用计划的 ASR 一致性选择。相同文本候选池中的恢复效果仍然存在，说明收益并非仅来自改变模型答案。

15 个模型/识别器比较中有 14 个相同文本置信区间不包含零，唯一例外是 MiniCPM-o 经 Seamless 评测的结果。

### 不同选择目标下，计划引导均保持稳定收益

![六种候选选择策略比较](docs/assets/selection_methods.svg)

在完整候选池上，Plan-Fact 对每个检查点都比原始样本、随机选择和不使用计划的 ASR 一致性选择具有更低的参考计划损失。Plan-Text 同样有效；Oracle 是不可部署的评估侧上界，而不是实际基线。

### 渲染损失会造成可测量的下游影响

![从内部计划到 ASR 回读的 VoiceBench 准确率下降](docs/assets/voicebench_drops.svg)

准确率下降图隔离了内部计划转换为语音时的下游代价。不同模型和识别器之间差异明显：Qwen2.5-7B 的两个识别器视角未观察到下降，而 Qwen2.5-3B 经 Seamless 回读后下降 23.1 个百分点。这支持使用多个识别器报告结果，而不是依赖单一 ASR 视角。

### 增加候选预算通常能够改善恢复

![一到八个候选语音下的参考计划损失降低](docs/assets/candidate_curve.svg)

从一个候选增加到八个候选后，五个检查点在最终预算下的恢复效果均得到增强。曲线采用留出 ASR 的 leave-one-recognizer-out 评测：用于评价候选的识别器绝不会参与该候选的选择。

所有图中数值均为论文报告的聚合结果。作为代码仓库，本项目不包含原始预测、生成语音和逐样本评测记录。

## 测试

运行完整测试：

```bash
pytest -q
```

测试覆盖事实归一化、带符号事实匹配、回读状态、模型注册、基准评分、候选选择、leave-one-ASR-out 评测和实验保护机制。

## 数据与模型策略

本仓库仅发布代码，不重新分发：

- 语音语言模型或 ASR 权重；
- 受上游许可证约束的基准数据集；
- 生成的语音或模型预测；
- 逐样本实验记录或内部评测缓存。

请从官方来源下载各模型和基准，并遵守相应使用条款。本地资源应保存在 `pretrain_model/`、`data/`、`exp/` 和 `output/` 等已忽略目录中。

## 引用

论文公开后将补充链接和存档标识。目前请使用：

```bibtex
@article{zhang2026reusing,
  title   = {Reusing the Internal Spoken Plan: Localizing Loss and Recovering Facts in Speech Language Models},
  author  = {Zhang, Yunlong and Wang, Yonghe and Bao, Feilong},
  year    = {2026},
  note    = {Preprint}
}
```

## 联系方式

如有问题，请提交 GitHub Issue，或联系 `cszyl@mail.imu.edu.cn`。
