# COS method: accurate labels require numerical validation

F. Fang and C. W. Oosterlee, *A Novel Pricing Method for European Options Based on
Fourier-Cosine Series Expansions*, SIAM Journal on Scientific Computing 31(2),
826–848 (2008), [DOI: 10.1137/080718061](https://doi.org/10.1137/080718061).
The publisher dates online publication to 14 November 2008; its issue spans
2008–2009. This note uses the published paper, not an earlier manuscript.
[Publisher metadata](https://epubs.siam.org/doi/abs/10.1137/080718061?journalCode=sjoce3).

## Method, claims, and comparisons from the paper

COS approximates the transition density on a finite interval with cosine terms,
obtains coefficients from the characteristic function, and integrates against
analytic payoff coefficients. The authors separate truncation and series errors;
exponential convergence depends on density regularity. For a fixed maturity and
strike set, cost is linear in the number of terms. Their numerical competitors
are Carr–Madan FFT and CONV, using Simpson quadrature. Heston timing tables
compare COS with Carr–Madan; the short-maturity variance-gamma example has
convergence comparable to CONV. Thus superiority is conditional, not universal.
The truncation interval is based on cumulants.
[Published paper, §§2–5, especially §5.1, Remark 5.3, Tables 4–5 and Figure 5](https://ir.cwi.nl/pub/13283/13283D.pdf).

## Critical assessment and project implications — our reasoning

The missing evidence for this project is a uniform accuracy guarantee over our
eventual Heston parameter box, maturity grid, and wings. A successful example
cannot certify every training label. We should vary both the number of COS
terms and interval width: convergence in just one setting could conceal error
from the other. These sweeps belong in T1.2, before any training data exist.

I would adopt COS as the label generator because its characteristic-function
interface fits the chosen Heston model and the planned batched implementation.
Monte Carlo remains an independent check with sampling uncertainty and
discretization error; it should not silently become the primary label source.
Black–Scholes limits and put–call parity provide additional checks with different
failure sensitivities. Agreement between two COS resolutions alone is insufficient.

Price accuracy and implied-volatility accuracy need separate acceptance checks.
Our concern is that a seemingly negligible price discrepancy could dominate an
IV target at a difficult quote. T1.3 must therefore report inversion failures
explicitly, and T1.4 must preserve the mask instead of dropping inconvenient
coordinates. No tolerance or term count is selected by this note.

For the later speed comparison, use matched accuracy, precision, batch size and
hardware, including vectorized GPU COS. We should measure full surface generation
and the required IV conversion, then identify exactly which stages are timed.
A historical runtime cannot establish a speedup for our implementation.

## Verification status

Verified 2026-10-27 against publisher metadata and the complete CWI-hosted final
PDF. No pricing code was run, and no reported timing was reproduced.
