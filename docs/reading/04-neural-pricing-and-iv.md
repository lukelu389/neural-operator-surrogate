# Neural pricing and implied-volatility inversion

## Verified reference and version

Shuaiqiang Liu, Cornelis W. Oosterlee, and Sander M. Bohte. *Pricing Options and
Computing Implied Volatilities using Neural Networks*. **Risks 7(1), article 16
(2019)**, published 9 February 2019.
[DOI: 10.3390/risks7010016](https://doi.org/10.3390/risks7010016).
[arXiv](https://arxiv.org/abs/1901.08943) first submitted 25 January 2019, revised
23 April 2019. References below use the
[22-page published PDF hosted by CWI](https://ir.cwi.nl/pub/28645/28645.pdf).

## Method, claims, and actual comparisons

MLPs approximate Black–Scholes pricing, Heston COS pricing, and price-to-IV
inversion. IV inputs include log time value; examples with time value below
`1e-7` are excluded. ([Sections 3, 4.2–4.4](https://ir.cwi.nl/pub/28645/28645.pdf))

The authors report accurate, faster inference. Table 8 compares IV-ANN with
Newton–Raphson, Brent, secant, and bisection on CPU/GPU for 20,000 selected
at-the-money contracts. The approximately 100-fold GPU claim there is versus
Newton–Raphson. Heston-ANN followed by IV-ANN is separately compared with COS
followed by Brent; smaller evaluation domains improve accuracy.
([Section 4.3.2, Table 8; Section 4.4.2, Table 11](https://ir.cwi.nl/pub/28645/28645.pdf))

## Critical assessment and project implications

Our assessment: learning prices and learning their inverse introduce distinct
error channels. In this project, a direct Heston-parameter-to-IV target avoids
composing two learned approximations, but its labels still depend on accurate
pricing and inversion. Gate 1 must establish both before training. This is a
reason to validate the full label pipeline, not evidence that direct IV learning
will outperform every alternative.

For our evaluation, numerical difficulty must remain visible. Preserve an explicit
failure mask and report how exclusion changes coverage by maturity and moneyness.
Do not quietly adopt a time-value cutoff from another experiment: tolerances need
units, a numerical justification, and a recorded effect on this project's grid.
Deep wings and short maturities deserve separate reporting because a small price
perturbation can imply a large IV perturbation when vega is small.

We still need evidence for arbitrary query behavior, parameter shifts, derivative
accuracy, and static arbitrage under our protocol. Good aggregate price fit cannot
establish those properties. Freeze the intended domains before observing final
results; a narrower domain must not become an after-the-fact remedy for errors.
Use the prescribed three seeds and tail metrics to test reliability.

Speed comparisons here must time the complete requested task: producing the same
IV surface with the surrogate and with vectorized COS plus inversion. Record
warm-up, precision, batch size, and hardware. An IV-root-finder timing alone does
not answer that surface-pricing question.

## Access and verification limits

Checked 27 October 2026. Read the complete CWI-hosted journal PDF and verified
preprint history at arXiv. Direct publisher access returned HTTP 429 and DOI
retrieval failed, so journal sections/tables were checked against the institutional
PDF. Reported performance is the authors' result; no implementation, benchmark,
or reproducibility claim was independently reproduced in this ticket.
