# Deep Learning Volatility

## Verified reference and version

Blanka Horvath, Aitor Muguruza, and Mehdi Tomas. *Deep learning volatility: a deep
neural network perspective on pricing and calibration in (rough) volatility
models*. **Quantitative Finance 21(1), 11–27 (2021)**.
[DOI: 10.1080/14697688.2020.1817974](https://doi.org/10.1080/14697688.2020.1817974).
Published online 26 October 2020; [arXiv](https://arxiv.org/abs/1901.09647) first
submitted 28 January 2019, revised 22 August 2019. Section references below use
the [arXiv v2 manuscript](https://arxiv.org/pdf/1901.09647v2).

## Method, claims, and actual comparisons

A feedforward network learns parameters-to-IV-grid values offline; online
calibration optimizes parameters through this approximation. The two stages
separate learning a forward map from solving its inverse problem.
([Sections 2.2, 3.2–3.3](https://arxiv.org/pdf/1901.09647v2))

The authors report fast surface evaluation and calibration. Pricing accuracy
and CPU timing are compared with Monte Carlo for rough Bergomi and one-factor
Bergomi, including piecewise constant forward variance. Calibration compares
optimizers, selecting Levenberg–Marquardt; a historical SPX experiment compares
neural and brute-force Monte Carlo calibration. These are not Heston-versus-GPU-COS
timings. ([Section 4.1, Table 2; Sections 4.2–4.2.2](https://arxiv.org/pdf/1901.09647v2))

## Critical assessment and project implications

Our assessment: the useful transferable idea is the forward-surrogate calibration
workflow. A large speed advantage against simulation cannot establish the
advantage against this project's faster semi-analytic Heston reference. We must
measure the latter directly, with identical batch sizes and hardware disclosure,
and distinguish training cost from repeated online use.

For our purposes, a successful calibration needs three separate checks: whether
the fitted surface matches observations, whether the original pricer agrees at
the fitted parameters, and whether those parameters are identifiable. A small
surface residual alone cannot settle the other two. Local surrogate error and
Jacobian error can redirect optimization even when average prediction error is
small. Gate 5 should therefore retain the planned COS re-evaluation, matched
multi-start budgets, noise/missing-quote cases, and parameter-versus-fit analysis.

The evidence needed for our architecture question remains a controlled comparison
under our own training budget. Retain the MLP as a serious baseline; assess
coordinate models on held-out queries and distribution shifts without importing
any published calibration accuracy as a target achieved here. Report tails and
seed variability, and keep synthetic results explicitly labeled. Historical-data
experiments in another study do not validate this repository on market data.

## Access and verification limits

Checked 27 October 2026. Read the arXiv full text. The
[King's accepted manuscript](https://kclpure.kcl.ac.uk/ws/files/139539661/DLV_Revision.pdf)
confirms the journal reference; the publisher's
[issue listing](https://www.tandfonline.com/toc/rquf20/21/1) supplied the online
date through search, but direct access returned HTTP 403. Publisher pagination
was not used for manuscript section citations. No code or timings were reproduced.
