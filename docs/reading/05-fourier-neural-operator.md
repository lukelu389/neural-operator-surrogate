# Fourier Neural Operator

**Citation verified 2026-10-28:** Zongyi Li et al., *Fourier Neural Operator for Parametric Partial Differential Equations*, ICLR 2021. Read [arXiv:2010.08895v3](https://arxiv.org/abs/2010.08895v3), dated 17 May 2021; its arXiv DOI is 10.48550/arXiv.2010.08895, not a conference DOI.

## Verified method, claims, and comparisons

FNO learns function-to-function maps using truncated Fourier-mode multiplication, a pointwise linear path, and nonlinearities. The authors report PDE prediction, resolution transfer, and faster inference than numerical solvers. These are Burgers, Darcy, and Navier–Stokes experiments. See [paper, sections 3–5, pp. 4–9](https://arxiv.org/pdf/2010.08895v3).

Burgers baselines are NN, GCN, FCN, PCA-NN, graph operators GNO/MGNO, and low-rank LNO. Darcy includes reduced-basis RBM instead of GCN. Navier–Stokes compares ResNet, U-Net, TF-Net, FNO-2D, and FNO-3D; parameter counts differ. LNO is described as similar to unstacked DeepONet, not as an original-DeepONet replication. See [section 5 and Tables 1, 3–4, pp. 7–8, 15](https://arxiv.org/pdf/2010.08895v3).

## Independent assessment and project implications

The relevant question here is whether spectral structure helps our particular pricing task under the project's equal-budget rules. Published PDE rankings cannot answer that question. They should motivate a hypothesis, not determine the winner or justify adding FNO to Track A. The current Heston input is a short parameter vector plus contract coordinates; representing that vector on an artificial grid would introduce an arbitrary spatial interpretation. The planned MLP remains the appropriate initial reference.

The optional payoff-to-price task has a clearer operator interpretation: a whole payoff function enters and a price function emerges. For that future experiment, the Black–Scholes convolution hypothesis in PLAN.md makes FNO a sensible candidate, but the known convolution also creates a demanding numerical baseline. A learned approximation must earn its accuracy and speed claims against that baseline at matched tolerances, hardware, and batch sizes. Offline data generation and training costs belong in the discussion.

Before interpreting any future resolution result, distinguish a denser output grid from genuinely new information in the input. Denser sampling cannot recover payoff features absent from the coarse input. Domain truncation, boundary treatment, and strike kinks could dominate observed error; tests should isolate those effects and examine price derivatives as well as average price error. A small aggregate error could conceal poor Gamma near a kink.

This note changes no protocol or architecture. Track B remains optional, unimplemented, and blocked until Gate 5 completes. Reading the paper verifies the reference and motivates later questions; it does not verify its benchmark reproducibility or establish any project performance result.
