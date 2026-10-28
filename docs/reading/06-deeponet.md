# DeepONet

**Citation verified 2026-10-28:** Lu Lu, Pengzhan Jin, Guofei Pang, Zhongqiang Zhang, and George Em Karniadakis, *Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators*, Nature Machine Intelligence 3, 218–229 (2021). [Publisher DOI: 10.1038/s42256-021-00302-5](https://doi.org/10.1038/s42256-021-00302-5). The related [2019 preprint](https://arxiv.org/abs/1910.03193) has a different title and three authors; do not substitute its bibliography for the journal article.

## Verified method, claims, and comparisons

DeepONet combines branch features of sampled input functions with trunk features of output coordinates through a dot product. The paper reports operator approximation and empirical generalization across explicit operators and differential equations. Its approximation theorem does not guarantee optimization or generalization. See [published paper, pp. 219–221, Fig. 1](https://aiichironakano.github.io/cs653/Lu-DeepONet-NMachineIntell21.pdf) (academic mirror).

Baselines include FNNs, ResNets, attention-equipped LSTM/GRU Seq2Seq models, and CNNs for selected tasks. Antiderivative comparisons use different output-sampling datasets for Seq2Seq and DeepONet. For the fractional Laplacian, the CNN is slightly more accurate. See [pp. 221–224, Figs. 2–3](https://aiichironakano.github.io/cs653/Lu-DeepONet-NMachineIntell21.pdf).

## Independent assessment and project implications

For this project, call M1 “DeepONet-style.” Its branch receives a finite Heston parameter vector, not samples of an arbitrary input function. Its trunk receives moneyness and maturity, and its prediction is a scalar implied volatility. That is a useful factorized coordinate model; labeling the experiment as general function-space learning would overstate what the data test. The later optional payoff task would support a more direct operator interpretation.

The factorization provides a concrete engineering advantage to investigate: reuse the branch embedding across many contract queries and reuse trunk evaluations across surfaces on the same grid. Actual benefits depend on batch size and implementation, so benchmark complete surface inference rather than an isolated dot product. The width of the shared representation is also an expressiveness choice that must stay inside the declared parameter and tuning budgets.

An arbitrary-coordinate interface is not evidence of accurate interpolation or extrapolation. The planned query-generalization comparison should retain the spline-interpolated MLP baseline. Coordinate derivatives require their own checks against the validated numerical pricer; low IV error alone cannot establish accurate Greeks. Likewise, neither the network name nor the factorization enforces financial constraints, so arbitrage diagnostics remain necessary.

The paper's examples cannot replace a matched Heston comparison. Use identical labels, splits, optimizer family, tuning effort, and seeds, and preserve the project's holdout restrictions. Do not import external experiment conventions into our evaluation protocol. No model is implemented or selected by this reading note. M1 belongs to T3.1; payoff-operator experiments remain optional and wait until Gate 5 is complete.
