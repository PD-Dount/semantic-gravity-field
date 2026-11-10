# The semantic gravity field is a token prior and a cosine

Six falsification tests for field-inspired attention. Code, raw results and manuscripts.

[English](#the-semantic-gravity-field-is-a-token-prior-and-a-cosine) · [中文](#语义引力场就是一个-token-先验加一条余弦)

Physically inspired attention variants give each token a *mass* and a *capture radius* and let tokens interact through a distance-decaying potential. This repository implements one complete proposal — a global attraction potential with a three-zone collapse, a mass-conserving merge rule and an irreversible escape zone — as a single parameterised attention layer, and tests it six independent ways.

## Headline results

| # | Test | Result |
|---|---|---|
| 1 | Algebraic reduction | At constant bandwidth the logit is exactly `α·cosθ + β·cos²θ + b_j`; fit to the full field is numerically exact (R² = 1.000000) |
| 2 | Equal-parameter ablation | Five arms are indistinguishable in accuracy (0.206–0.214, chance 0.0625); three cost 2.0–2.8× per optimiser step |
| 3 | Gradient-sign voting | The learned field strength *decreases* under AdamW from three different initialisations |
| 4 | Cost-ratio scaling (AMD DCU) | Ratio grows with context, 2.42× at 512 → 4.83× at 2048; effective attended keys within 0.4% of the causal baseline |
| 5 | Iterated dynamics | Bistable (all-collapse or all-isolated) up to N = 8192; no stable multi-cluster state |
| 6 | Memory eviction / real attention | Bit-identical to LRFU (2001); the gravity basis is indistinguishable from parameter-matched controls (ΔR² ≤ 0.003) |

All tests are small-scale and use dense kernels: this falsifies deployment paths, not the metaphor. The transferable output is a diagnostic protocol — parameter-matched control bases, gradient-sign voting under multiple initialisations, effective-participation accounting, cost-ratio scaling — plus six implementation traps that silently invalidate otherwise correct experiments.

## Layout

| path | contents |
|---|---|
| `paper/` | Manuscript (EN/ZH, Markdown + docx), arXiv abstract, submission notes, Figures 1–2 |
| `gravity_lab/` | Attention layer, config, cost accounting, merge rule, tasks, training loop |
| `field_sim/` | Dynamical field simulator, phase sweeps, DCU benchmark driver |
| `results/` | Raw per-run JSON/JSONL, DCU long-sequence benchmarks, phase diagrams, attention fits |
| `*.py` | Experiment drivers: ablation, analysis, plots, attention fitting, memory probe, contract tests |
| `SPEC.md` | Frozen mathematical definition every arm must match |
| `PREREG.md` | Decision rules written before the runs |
| `REPORT.md` | Integrated report (Chinese), including the four defects the implementation exposed |
| `docs/original_proposal_zh.md` | The original proposal under test (Chinese) |
| `skills/idea-to-algebra/` | The diagnostic protocol as a reusable agent skill |

## Reproduce

```bash
python -m pip install torch

# 12 contract tests: A0 is bit-identical to native SDPA, A1 is shift-invariant, mass is conserved
python smoke_contract.py

# equal-parameter, equal-step, equal-seed ablation (CPU, 4 cores)
python run_ablation.py --arms A0 A1 A2 A3 A4 --seeds 0 1 2
python analyze.py
python plot.py

# the remaining tests
python attn_fit.py        # §3.7, needs distilgpt2
python memory_probe.py    # §3.5, LRFU equivalence
python selforg.py         # §3.4, iterated dynamics
python field_sim/sim.py   # phase diagram
```

Every table and figure in the paper is regenerated from `results/`; no number in the manuscript is hand-entered.

## Paper

- [`paper/manuscript_en.md`](paper/manuscript_en.md) — arXiv version
- [`paper/manuscript_zh.md`](paper/manuscript_zh.md) — Chinese version
- [`paper/arxiv_abstract.txt`](paper/arxiv_abstract.txt) — plain-text abstract for the submission form
- [`paper/README_SUBMIT.md`](paper/README_SUBMIT.md) — venue advice, reviewer objections, open items

## Citation

```bibtex
@misc{zhang2026semanticgravity,
  title  = {The Semantic Gravity Field Is a Token Prior and a Cosine: Six Falsification Tests for Field-Inspired Attention},
  author = {Zhang, Junbin},
  year   = {2026},
  note   = {Preprint. \url{https://github.com/PD-Dount/semantic-gravity-field}}
}
```

## License

Text, figures and results: [CC BY 4.0](LICENSE). Code: [MIT](LICENSE-CODE).

## Contact

Zhang Junbin — GitHub [@PD-Dount](https://github.com/PD-Dount)

---

# 语义引力场就是一个 token 先验加一条余弦

面向"场式注意力"的六项证伪实验。代码、原始结果与论文稿。

[English](#the-semantic-gravity-field-is-a-token-prior-and-a-cosine) · [中文](#语义引力场就是一个-token-先验加一条余弦)

物理启发的注意力变体给每个 token 一个**质量**与**捕获半径**，让 token 之间按距离衰减势相互作用。本仓库把一份完整的提案（全局吸引势 + 三区坍缩 + 质量守恒合并 + 不可逆逃逸）实现为单一参数化注意力层，并对它做六项独立检验。

## 六项结论

| # | 检验 | 结果 |
|---|---|---|
| 1 | 代数归约 | 带宽恒定时 logit 精确等于 `α·cosθ + β·cos²θ + b_j`；完整场参数化与之一致到数值精度（R² = 1.000000） |
| 2 | 等参数量消融 | 五个臂精度统计打平（0.206–0.214，随机基线 0.0625）；其中三个臂每步贵 2.0–2.8 倍 |
| 3 | 梯度符号投票 | 三个不同初值下，AdamW 一律把学到的场强度**调小** |
| 4 | 成本比随长度变化（AMD DCU） | 512 时 2.42× → 2048 时 4.83×；有效参与 key 数与因果基线只差 0.4% |
| 5 | 迭代动力学 | 双稳态（全坍缩或全隔离），到 N = 8192 仍无稳定多簇解 |
| 6 | 记忆淘汰 / 真实注意力 | 与 2001 年的 LRFU 逐位相同；引力基与同参数量控制基无差别（ΔR² ≤ 0.003） |

全部检验都在小尺度、稠密实现内完成：证伪的是**落地路径**，不是隐喻本身。可迁移的产出是一套诊断协议——参数匹配控制基、多初值梯度投票、有效参与数记账、成本比标定——以及六个会静默毁掉实验的实现陷阱。

## 目录

| 路径 | 内容 |
|---|---|
| `paper/` | 中英文稿件（Markdown + docx）、arXiv 摘要、投稿说明、Figure 1–2 |
| `gravity_lab/` | 注意力层、配置、成本记账、合并规则、任务、训练循环 |
| `field_sim/` | 场动力学模拟、相图扫描、DCU 基准驱动 |
| `results/` | 全部逐次实验 JSON/JSONL、DCU 长序列基准、相图、注意力拟合 |
| `*.py` | 实验驱动：消融、分析、绘图、注意力拟合、记忆淘汰探针、契约测试 |
| `SPEC.md` | 冻结的数学定义，所有臂必须逐字对齐 |
| `PREREG.md` | 跑实验前写死的判据 |
| `REPORT.md` | 整合测试报告（中文），含实现暴露的四个致命问题 |
| `docs/original_proposal_zh.md` | 被检验的原始方案（中文原文） |
| `skills/idea-to-algebra/` | 诊断协议，打包成可复用 skill |

## 复现

```bash
python -m pip install torch

# 12 项契约测试：A0 与原生 SDPA 逐位相等、A1 平移不变、质量守恒
python smoke_contract.py

# 等参数量、等步数、等种子的消融（CPU，4 核）
python run_ablation.py --arms A0 A1 A2 A3 A4 --seeds 0 1 2
python analyze.py
python plot.py

# 其余检验
python attn_fit.py        # §3.7，需要 distilgpt2
python memory_probe.py    # §3.5，与 LRFU 等价
python selforg.py         # §3.4，迭代动力学
python field_sim/sim.py   # 相图
```

论文里每张表、每张图都由 `results/` 重新生成，正文数字没有一个是手工填的。

## 论文

- [`paper/manuscript_zh.md`](paper/manuscript_zh.md) — 中文版
- [`paper/语义引力场_中文.docx`](paper/语义引力场_中文.docx) — 中文 Word 版
- [`paper/manuscript_en.md`](paper/manuscript_en.md) — 英文版（arXiv 用）
- [`paper/arxiv_abstract.txt`](paper/arxiv_abstract.txt) — 投稿表单用纯文本摘要
- [`paper/README_SUBMIT.md`](paper/README_SUBMIT.md) — 选定投哪里、审稿人会打什么、还差什么

## 引用

```bibtex
@misc{zhang2026semanticgravity,
  title  = {The Semantic Gravity Field Is a Token Prior and a Cosine: Six Falsification Tests for Field-Inspired Attention},
  author = {Zhang, Junbin},
  year   = {2026},
  note   = {Preprint. \url{https://github.com/PD-Dount/semantic-gravity-field}}
}
```

## 许可

文字、图表与结果：[CC BY 4.0](LICENSE)。代码：[MIT](LICENSE-CODE)。

## 联系

张俊滨 — GitHub [@PD-Dount](https://github.com/PD-Dount)
