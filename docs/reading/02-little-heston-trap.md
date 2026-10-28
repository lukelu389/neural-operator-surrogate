# The Little Heston Trap: preserve the complex branch

Hansjörg Albrecher, Philipp Mayer, Wim Schoutens and Jurgen Tistaert,
*The Little Heston Trap*, Wilmott (2007), no. 1, 83–92.
[Coauthor publication list](https://www.opt.math.tugraz.at/~mayer/research.html).
The accessible manuscript is dated 11 September 2006, with first version
6 December 2005. The publisher's 2012 web-post date is not the publication year.
[Manuscript, title page](https://www.ma.imperial.ac.uk/~ajacquie/IC_Num_Methods/IC_Num_Methods_Docs/Literature/HestonTrap.pdf),
[publisher page](https://www.wilmott.com/the-little-heston-trap-wilmott-magazine-article-hansjorg-albrecher-philipp-mayer-wim-schoutens-jurgen-tistaert/).

## Method, claims, and comparisons from the paper

The paper compares two Heston characteristic-function representations. The
preferred form uses a decaying exponential and the reciprocal ratio
`g2 = 1/g1`. The alternative can cross the complex logarithm's branch cut,
creating discontinuities despite formal algebraic equivalence. The authors
derive instability conditions; Theorem 3 proves branch stability for the second
form under the paper's model, damping-moment and branch conventions.
Comparisons include both representations in Carr–Madan pricing and Monte Carlo
prices, rather than competing learned models. Their examples show that satisfying
the Feller inequality does not by itself remove the problem.
[Manuscript, §3 equations (1)–(2), §4 Table 2, and §5 Theorem 3](https://www.ma.imperial.ac.uk/~ajacquie/IC_Num_Methods/IC_Num_Methods_Docs/Literature/HestonTrap.pdf).

## Critical assessment and project implications — our reasoning

The central implementation lesson is to preserve a consistent branch convention
throughout the formula. A symbolic simplification is not sufficient evidence
that a floating-point implementation has the same behavior. In T1.2, I recommend
implementing the second representation directly, documenting the square-root
choice and treating zero-frequency and limiting cases explicitly. Adding
branch-tracking machinery to the first representation would introduce avoidable
implementation complexity for this project.

Our validation must distinguish the paper's branch-stability argument from
complete numerical accuracy. It does not certify our dtype, broadcasting,
autodiff behavior, or treatment of cancellation and small denominators. Those
are implementation questions requiring tests. Characteristic-function
normalization, conjugate symmetry on real frequencies, continuity, and a
deterministic-variance limit are useful candidate diagnostics; none replaces the
planned independent price checks.

We should include deliberately stressful parameter and maturity cases in pricer
unit tests even when they lie outside the proposed training grid. Such
deterministic numerical fixtures must remain separate from the sealed research
test and OOD datasets. This keeps software validation possible without allowing
model-selection leakage. Whether the eventual sampling box imposes Feller is
a separate Gate 1 decision, not a substitute for choosing a stable formula.

The remaining gap for calibration is derivative reliability. Smooth-looking
prices do not establish accurate parameter sensitivities in our implementation.
Later Greek and calibration tickets should compare derivatives against
independent finite-difference checks across step sizes before interpreting an
optimizer failure as an architectural weakness. This note does not implement
those checks or claim that they pass.

## Verification status

Verified 2026-10-27 using the complete 21-page manuscript and coauthor metadata.
The final magazine typesetting and a DOI were not verified. No numerical result
was reproduced; all implementation recommendations above remain untested.
