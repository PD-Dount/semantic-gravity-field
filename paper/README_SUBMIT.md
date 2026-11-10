# 投稿说明 / Submission notes

## 文件

| 文件 | 用途 |
|---|---|
| `manuscript_en.md` | 英文主稿，arXiv 版 |
| `manuscript_zh.md` | 中文版，面向国内团队与工程读者 |
| `figures/fig1_arms.png` | 六臂消融（精度/损失/等墙钟/成本-精度） |
| `figures/fig2_dynamics.png` | 吸引×排斥相图 |
| `data_bench_dcu.json` | 长序列基准原始数据（AMD DCU） |

## 建议投递

- **首选**：arXiv 预印本，primary `cs.LG`，cross-list `cs.CL`。负面/证伪结果在 arXiv 上传播最快，且不需要通过"新颖性"评审。
- **可选**：TMLR（Transactions on Machine Learning Research）——评审标准是"结论是否正确且有用"，明确接受严谨的负面结果与复现研究，不需要 SOTA。
- **可选**：NeurIPS / ICML 的负面结果或科学方法论 workshop；或 *ReScience C*（复现类期刊）。
- **不建议**：正向 ML 会议主 track。这类工作会被"没有大规模实验"一票否决，而我们已明确把这一条写进局限。

## arXiv 表单

**Title:** The Semantic Gravity Field Is a Token Prior and a Cosine: Six Falsification Tests for Field-Inspired Attention

**Authors:** [待填]

**Abstract:** 直接取 `manuscript_en.md` 的 Abstract 段（约 320 词，超 1920 字符限制时删去最后一句"transferable contribution"）。

**Comments:** 12 pages, 2 figures, code at [repo]

**License:** 建议 CC BY 4.0（便于他人复用实验资产）

## 作者必须自己补的四项

1. 作者名已填 Zhang Junbin / 张俊滨；**单位待填**。
2. 代码仓库地址：https://github.com/PD-Dount/semantic-gravity-field
3. 是否有基金/致谢。
4. 若目标期刊需要，补 Declaration of interests（本工作无商业利益关联）。

## 审稿人最可能打的三点（已在正文处理，投稿信里可直接引）

1. **"只在 9 万参数的模型上做，说明不了什么。"**
   → 已写进 §5「我们不主张什么 / 尺度」，并给出两个不依赖任务可解性的检验（§3.4 成本比与有效参与、§3.7 真实模型注意力拟合）。**建议在 cover letter 里主动写出来，不要等审稿人提。**

2. **"你们证伪的是实现，不是想法。"**
   → 摘要与 §5 第一段明确区分"证伪落地路径"与"证伪隐喻"，并列出三条未被检验的边界（大规模训练、融合稀疏核、学出来的动力学）。

3. **"$\cos^2$ 与 $\cos^3$ 打平，会不会只是这一层的巧合？"**
   → §3.7 是 6 层 × 12 头 × 5 折留出的均值，且逐层报告在 `results/attn_fit.json`。可补充"逐层 R²"表格进附录以加强。

## 已知的、投稿前应该再补的两件事（可选）

- **融合核基线**：本次长序列基准的 A0 也是非融合稠密实现。补一个 `F.scaled_dot_product_attention` 基线，A3/A0 的比值会从 4.83× 进一步变大（预计 6–8×），结论只会更强。
- **第二个预训练模型**：目前只有 distilgpt2。加一个 pythia-70m 或 gpt2-medium 跑同样的拟合，可以挡住"模型太小"的质疑。

## 图表清单

- Figure 1：四联图（a 精度-步数 / b 损失-步数 / c 精度-墙钟 / d 成本-精度）
- Figure 2：四联图（dense 与 top-16 两组的分类相图 + NMI 热图）
- 正文 6 张表，均可由 `results/` 下 JSON 重新生成
