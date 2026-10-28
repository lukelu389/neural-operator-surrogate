# Mamba: selective state spaces

**Citation verified 2026-10-27:** Albert Gu and Tri Dao,
*Mamba: Linear-Time Sequence Modeling with Selective State Spaces*, COLM 2024.
The preprint dates to December 2023; this note reads
[arXiv v2, 31 May 2024](https://arxiv.org/abs/2312.00752v2).
The [authors' publication list](https://dao-lab.ai/publications/) confirms COLM.
No published scores were independently reproduced.

## Method

Mamba makes the state-space input/output projections and discretization step
depend on the input token. This selection changes the recurrence across tokens,
so a fixed convolution no longer suffices. The paper combines a selective scan,
local convolution, projections, and gating; a fused implementation reduces
memory traffic during training.
[Sections 3.2–3.4, Algorithms 1–2, Appendix D](https://arxiv.org/html/2312.00752v2).

## Claims and actual baselines

Reported gains cover language, DNA, and audio, with linear sequence-length
scaling. Language comparisons include a GPT-3-style Transformer and an enhanced
Transformer recipe; selective-copying ablations compare S4, H3, and Hyena layers.
Speed experiments distinguish scan kernels from end-to-end autoregressive
throughput, including FlashAttention-2. The authors also caution that selectivity
can hurt continuous-signal tasks where time-invariant SSMs work well.
[Sections 4.1–4.6 and 5](https://arxiv.org/html/2312.00752v2).

## What could be missing? — our assessment

Those workloads do not establish an advantage for smooth implied-volatility
surfaces. Here the entire parameter vector is known before any coordinate is
queried. Content selection may help distinguish wings and maturities, or simply
increase optimization difficulty. Both outcomes are legitimate and require the
same validation budget as the other models.

Two forms of efficiency must remain separate. A recurrence can have linear
arithmetic complexity yet run slowly when dispatched through a Python loop.
Conversely, autoregressive generation benefits from a compact recurrent state in
a way that simultaneous prediction of a short option surface may not. A timing
claim requires our measured batch sizes, hardware, precision, and warm-up policy.

## Consequences for this project — our decisions

Select a **bidirectional Mamba-style selective SSM** for M3: Heston-conditioned
coordinate embeddings, a selective recurrence in forward and reverse
maturity-major order, and a per-coordinate IV head. This is our adaptation to
surface regression, not a replication of the original language model. The two
passes give each output information from both directions; they do not guarantee
permutation invariance or arbitrage consistency.

At T3.3, start with an explicit PyTorch reference recurrence that can be checked
for shapes, gradients, masking, and reversal alignment. Its transparency fits
this project's small grid and existing dependency choice. Do not claim the
paper's fused-kernel performance for this implementation. S4 remains a credible
alternative, but its full structured-kernel machinery adds implementation work
without present evidence that this task needs it.

Widths, state size, optimizer settings, and trials remain undecided until the
relevant gates. Register query-set behavior and ordering checks before final
evaluation, include all directional computation in speed measurements, and
enforce the common parameter budget. No model has been built or trained yet.
