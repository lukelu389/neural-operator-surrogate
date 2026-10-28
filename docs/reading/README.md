# T0.2 reading notes

Eight compact, approximately one-page notes cover the core papers in PLAN.md.
Each identifies the version read, summarizes the method and actual comparisons,
and separates the authors' claims from our critical assessment and project
implications. Citation verification is not benchmark reproduction. Dates on the
notes record source review on 27–28 October 2026.

| Note | Relevance to the project |
| --- | --- |
| [01 — COS method](01-cos-method.md) | Label generation, truncation/convergence studies, fair numerical baselines |
| [02 — The Little Heston Trap](02-little-heston-trap.md) | Stable characteristic-function branches and independent validation |
| [03 — Deep Learning Volatility](03-deep-learning-volatility.md) | Forward-surrogate calibration and its limits |
| [04 — Neural pricing and IV](04-neural-pricing-and-iv.md) | Pricing/inversion error, quote exclusions, and timing definitions |
| [05 — Fourier Neural Operator](05-fourier-neural-operator.md) | Function-space learning for optional Track B only |
| [06 — DeepONet](06-deeponet.md) | Branch/trunk factorization and honest naming of M1 |
| [07 — S4](07-s4.md) | Structured recurrences, sequence order, and benchmark caveats |
| [08 — Mamba](08-mamba.md) | Selected M3 direction: a bidirectional selective SSM adaptation |

## Decisions carried forward

Keep the MLP as a serious baseline. Distinguish finite-parameter Heston surface
regression from learning an operator on input functions. Reading FNO does not
authorize starting Track B. Published timing ratios do not establish speedups
against our vectorized COS baseline, and no published score is a project result.

For T3.3, select a Mamba-style selective recurrence with forward and reverse
coordinate passes. Start from an explicit PyTorch reference implementation;
document its differences from the original Mamba architecture and kernels.
Model dimensions, optimizer settings, and trial budgets are not selected here.
This is an implementation direction required by the reading plan, not a claim
that the architecture will outperform M0 or satisfy the evaluation targets.

## Source access and remaining limits

Primary paper texts, author/institutional repositories, and publisher metadata
support the notes. DeepONet's published text was read through an explicitly
identified academic mirror. Some publisher pages were inaccessible; accessible
paper copies were used instead, with the version and access limits recorded in
the relevant note. The Little Heston Trap note uses the complete 2006 manuscript
and verified 2007 publication metadata; final magazine typesetting and a DOI
were not verified. No reference PDF is committed.

Luke's own Poseidon-related research notes were not supplied or found in this
repository. A focused search of the connected notes repository returned no
match. Their contents, code-reuse opportunities, and conclusions remain
unreviewed; no personal research claims have been invented. Reuse can be assessed
when the source is supplied. This does not prevent completing the eight core
paper notes specified by T0.2.

To review, open each note and follow its linked section/table references.
`make check` verifies the existing software scaffold; it does not validate
literature claims or reproduce the papers. T0.3 remains the next ticket, and
the evaluation protocol is still undrafted and unfrozen.
