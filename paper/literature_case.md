# 文献调研：这项研究为什么有意义

> 给导师的一页纸。回答一个问题：**"证伪一个东西"算不算贡献？**

---

## 一、最关键的一条：被检验的对象是**已发表**的，不是我们自己造的

**[Sharma, R. (2025). Attention as Spacetime Curvature: A General Relativistic Analogy for Language Models. Zenodo. DOI: 10.5281/zenodo.15718512][s1]**

- 2025-06-23 发布，CC BY 4.0，Zenodo 有 DOI，**299 次下载 / 385 次浏览**
- 核心论断，逐条摘录：
  - "token importance is analogous to **gravitational mass**"
  - "attention scores represent a **gravitational pull**"
  - "the mathematical representation of Scaled Dot-Product Attention is likened to a **potential field in gravity**"
  - "token representations evolve by following **geodesics** through an attention-curved representation space"
  - "naturally explains contextual word meaning and disambiguation"
  - 提出用 Ollivier-Ricci 曲率去量化，并声称对 "interpretability, optimization, and robustness" 有实用价值

**这条文献的意义**：它把"质量 = 重要性"、"注意力 = 引力势"这两个论断**写成了可检验的公开主张**。我们的六项实验，前五项在检验"引力势"这一半，第六项（§3.7，拟合真实模型注意力 logit）在检验"质量 = 重要性"这一半。

**而且结果是分裂的、不是全盘否定**：
- "token 重要性 ≈ 质量"这一半 **得到了支持** —— 我们实测 59.2% 的 log-attention 结构就是一个逐 token 常数。这是对 Sharma 论点的正面证据。
- "曲率/测地线/距离衰减"这一半 **没有得到支持** —— 曲率项只额外解释 0.36%，且与任意同参数量的单调基无法区分。

这比"全盘否定"有价值得多：**我们的论文实际上是在给这条文献划出"哪一半成立、哪一半不成立"的边界**，还顺带给了它一个可以直接用的量化结果（59% / 23% / 0.4% 的分解）。

---

## 二、这条"物理类比"线里，有的成了真结果，有的没有——我们的工作是分辨器

**[Ramsauer et al. (2021). Hopfield Networks is All You Need. ICLR. arXiv:2008.02217][s2]**
物理/能量类比在注意力上**成功**的范例：把注意力证明成现代 Hopfield 网络的能量最小化，得到可证明的容量结论。它成功的原因正是——**类比背后有一条真的等价关系**。

我们的论文对"引力"类比做的，是同一件事的方法论：**先找等价关系**。
找到的结果是：引力势注意力 ≡ softmax + 秩一 key 偏置 + 余弦单调变换。
这是一个精确恒等式（数值 2.1e-17），不是经验观察。**没有等价关系可找，就是类比不成立的证据。**

**这条对照是给导师看的最有力的一句**：同一个领域里，"精力类比"既产出过 Hopfield 那样的硬结果，也产出过纯叙事；区别就在于有没有人能写出那条等价关系。**我们证明了引力这条写不出来。**

---

## 三、社区已经明确呼吁发表这类结果

**[Karl, F., Kemeter, L. M., Dax, G., Sierak, P. (2024). Position: Embracing Negative Results in Machine Learning. arXiv:2406.03980][s3]**

- 明确论点："predictive performance alone is not a good indicator for the worth of a publication"
- "put out a call for the publication of 'negative' results"
- 指出用性能单一指标评价论文"fosters problems like inefficiencies of the machine learning research community as a whole"
- 给出"concrete measures for the community to move towards a paradigm where their publication is normalized"

**这条文献可以直接写进 Introduction 与 cover letter**：我们不是在做一件边缘的事，我们是在回应社区自己发出的号召。

---

## 四、有专门的场地接收这类工作

| 场地 | 性质 | 适配度 |
|---|---|---|
| **arXiv (cs.LG)** | 预印本，无评审门槛 | 立即可以投 |
| **TMLR** | 评审标准是"结论是否正确且有用"，明确不要求 SOTA 或新颖方法 | 主投目标 |
| **ReScience C** | 专门做**复现**的期刊（[rescience.github.io][s4]），有完整投稿规范与 FAQ | 备选，若强调"可复现"路线 |
| NeurIPS/ICML 主 track | 以性能与新颖性为导向 | **不投**，会被"规模太小"一句话打死 |

---

## 五、把"意义"说成三条，每条都有对应的可交付物

**意义 1：给一整类设计下了一个精确判决，而且是免费的。**
任何形如 $\sum_j m_i m_j f(d_{ij})$ 再归一化的注意力，都恒等于 softmax 加一个秩一偏置。这个引理不依赖我们的实验规模，**别人可以直接拿去做设计前的第一道筛子**。这是论文里唯一"永久有效"的部分。

**意义 2：给了真实注意力一个可复用的定量分解。**
59% 逐 token 先验 + 23% 余弦系数 + 0.4% 其余。做可解释性、做 KV 压缩、做 attention sink 的人都会想要这三个数。

**意义 3：六个实现陷阱，是别人重写这类机制时一定会踩的。**
位置编码缺失让联想召回任务不可学（且单 batch 记忆会伪造 100% 收敛）；零初始化让偶函数项梯度恒为零；相对阈值的不可逆剔除在大 N 上会自我毁灭（实测 1024/1024 个单元全删）。**这六条会在任何人第二次实现这套东西时救回几周。**

---

## 六、诚实的风险，也写给导师看

1. **规模小。** 9 万参数的模型、distilgpt2 一个模型。这一条已经在论文 §5 明确写进"我们不主张什么"。
2. **Sharma 那篇是独立研究者的 Zenodo 预印本，不是顶会论文。** 所以"我们在证伪一条主流路线"这个说法要克制——准确表述是"一条有公开、可检验主张的类比路线"。
3. **它不会改变任何人的工程实践。** 没有人正在生产环境里用引力场注意力，所以这篇的社会影响上限不高。它的价值在**方法论**与**引理**，不在"避免了重大损失"。

---

## 七、给导师的一句话版本

> 我们没有造出新方法，但我们对一条**已公开发表**的物理类比路线给出了精确的代数归约（引力势注意力 ≡ softmax + token 偏置 + 余弦单调变换）、四条独立的证伪路径、一个对真实模型注意力的可复用定量分解（59/23/0.4），以及六个会静默毁掉实验的实现陷阱。ICML 2024 有 position paper 明确呼吁发表这类结果，TMLR 的评审标准也接纳。**这不是自我否定，是把一个模糊的类比推到了精确的边界。**

---

[s1]: https://zenodo.org/records/15718512
[s2]: https://arxiv.org/abs/2008.02217
[s3]: https://arxiv.org/abs/2406.03980
[s4]: https://rescience.github.io/
