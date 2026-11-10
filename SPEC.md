# SPEC — 冻结的数学定义（实现必须逐字对齐；有异议先回报 Lead，不要自创变体）

记 $x\in\mathbb R^{B\times N\times D}$，$d_h=D/H$，$h$ 为 head 下标。
$q=xW_q,\;k=xW_k,\;v=xW_v$，$\hat q=q/\lVert q\rVert,\;\hat k=k/\lVert k\rVert$。

**统一的 logit（所有 arm 共用同一个公式，靠开关置零）**

$$\ell_{ij}=\alpha_h\,\frac{q_i\cdot k_j}{\sqrt{d_h}}
 \;+\;\mathbb 1_{\text{mass}}\,\log m_j
 \;-\;\mathbb 1_{\text{rad}}\,\kappa_h\left(\frac{D_{ij}^2+\epsilon}{2\sigma_{ij}^2+\epsilon}\right)^{p_h/2}
 \;+\;\mathbb 1_{\text{sink}}\,b_h^{\text{sink}}$$

$w_{ij}=\mathrm{softmax}_j(\ell_{ij})$（加 causal 掩码）。**这就是标准 softmax 注意力加一个改过的 logit**，
不再写"非 softmax 的引力加权"——那个写法在数学上恒等，只制造误解。

各分量：

| 分量 | 定义 | 参数 |
|---|---|---|
| 质量 | $\log m_j=\log\mathrm{softplus}(x_j w_m^{(h)})+b_h^{\rm mass}$（$b$ 在 log 域，才真的是常数平移） | $w_m^{(h)}\in\mathbb R^{D}$，$b_h^{\rm mass}$ 标量 |
| 半径/带宽 | $r_i=\mathrm{softplus}(x_i w_r^{(h)}+c)$，$\sigma_{ij}=\tfrac{s_h}{2}(r_i+r_j)$ | $w_r^{(h)}$，$s_h>0$ 初值 `init_radius` |
| 各向异性距离 | $D_{ij}^2=\lVert (I+U_hU_h^\top)(\hat q_i-\hat k_j)\rVert^2$ | $U_h\in\mathbb R^{d_h\times r}$，**$N(0,0.01)$** |
| 形状指数 | $p_h=1+\mathrm{softplus}(\text{raw}_h)$，raw 初值 $-\infty$ 侧 3 档 ⇒ $p\approx$ `p_init` | 单调可学，$p\to\infty$ 逼近硬截断 |
| 真空天体 | 额外 1 个 key：logit $=b_h^{\text{sink}}$，value $=v_h^{\text{sink}}\in\mathbb R^{d_h}$ | 吸收"全都不相关"的归一化质量 |

$U_h$ 不能零初始化：$D^2$ 对 $U$ 是偶函数，$U=0$ 处梯度精确为 0，一阶法永远学不动（实测 `U.grad.max()==0.0`）。

**A0**：$\alpha=1$，其余全关 ⇒ 必须与原生 attention 数值一致（容差 1e-6），这是对齐测试。
**A1**：开 mass。**A2**：开 radius（$\kappa=1$，$p=1$，$U=0$，无 sink）。
**A3**：mass+radius+aniso+shape_p+sink，$\kappa$ 初值 `kappa_init`。
**A4**：A3 + 不可逆逃逸。令 $\bar m_j$ 为 key $j$ 收到注意力质量的 EMA（detach，逐层更新）。
从 `commit_start_layer` 起，若 $\bar m_j<\tau\cdot\mathrm{mean}(\bar m)$ 则 key 永久出局（logit $-10^9$），
出局状态记录在 `ctx['dead']` 中，跨层做 OR，**不可撤销**。用 STE 让 $v_j$ 乘 $(1-p_j)$ 保持梯度。
**A5**（stretch）：A3 + 远场合并。块大小 $g$（`merge`）。查询 $i$ 的注意力键集 =
$$[\text{已合并远场块}] \;\cup\;[\text{本块内的原始键（causal 掩码）}]$$
合并块 $m$（覆盖 $[mg,(m+1)g)$），块内权重 $\pi_{jm}=\mathrm{softmax}_{j\in m}\big(\mathrm{sim}(k_j,c_m)\big)$，**逐块归一化 $\sum_j\pi_{jm}=1$**：
$$M_m=\textstyle\sum_{j\in m}m_j,\qquad
\bar k_m=\frac{\sum_j\pi_{jm}m_jk_j}{\sum_j\pi_{jm}m_j},\qquad
\bar v_m=\frac{\sum_j\pi_{jm}m_jv_j}{\sum_j\pi_{jm}m_j},\qquad
\bar r_m=\frac{\sum_j\pi_{jm}m_jr_j}{\sum_j\pi_{jm}m_j}$$
块质量必须是**块内质量和** $M_m=\sum_j m_j$，不是 $\sum_j\pi_{jm}m_j$ —— 后者在 $\sum_j\pi_{jm}=1$
下等于块内加权均值，与守恒律 $\sum_mM_m=\sum_jm_j$ 数学上不能同时成立（实测会差 $g$ 倍）。
$\bar r_m$ 的分母同样必须用 $\sum_j\pi_{jm}m_j$，用 $M_m$ 会偏小约 $g$ 倍。
**质量守恒**：$\sum_mM_m=\sum_jm_j$；断言相对误差 <1e-5（实测 1e-7 量级）。

## 物理约束（要断言，不是注释）
1. 查询侧质量在 A0–A3 的输出上严格无效（$\partial\text{out}/\partial m_i\equiv0$）——所以 $m$ 只挂在 key 侧，别再自欺。
2. 三场同轴时 $\prod_k e^{-d^2/2\sigma_k^2}=e^{-d^2/2\sigma_{\rm eff}^2}$ —— 所以要 anisotropy，不要三个并列高斯。
3. 软带宽与硬筛选解耦只靠 $p$；断言 $p$ 训练后单调上升或给出反例。

## 输出 stats（每个 forward 都要给，浮点）
`eff_part`=mean(1/Σ_j w²)，`sparsity`=mean(w<1e-4)，`escape_frac`，`kv_len`，`p_mean`，`kappa`，`sigma_mean`。
