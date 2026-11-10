# The semantic gravity field is a token prior and a cosine

Six falsification tests for field-inspired attention. Code, raw results and manuscripts.

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
