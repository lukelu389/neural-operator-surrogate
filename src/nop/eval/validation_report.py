"""Validation-only seed summaries and static figures; never model selection."""

import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from nop.eval.metrics import iv_metrics

_ERROR_METRICS = (
    "mean_absolute_error",
    "root_mean_squared_error",
    "p99_absolute_error",
    "max_absolute_error",
)
_METRICS = (
    "attempted",
    "valid_count",
    "reference_failure_count",
    "reference_failure_rate",
    "prediction_failure_count",
    "prediction_failure_rate",
    *_ERROR_METRICS,
)
MetricValues = dict[str, int | float | None]
AggregateValues = dict[str, dict[str, float | None]]


def _aggregate(values: list[MetricValues | None]) -> AggregateValues:
    result: AggregateValues = {}
    for name in _METRICS:
        available = [value[name] if value is not None else None for value in values]
        if any(value is None for value in available):
            result[name] = {"mean": None, "std": None}
            continue
        numbers = [float(value) for value in available if value is not None]
        mean, std = statistics.mean(numbers), statistics.stdev(numbers)
        result[name] = {
            "mean": mean if math.isfinite(mean) else None,
            "std": std if math.isfinite(std) else None,
        }
    return result


@torch.no_grad()
def validation_summary(
    predictions: dict[int, torch.Tensor],
    reference: torch.Tensor,
    valid: torch.Tensor,
    required_seeds: tuple[int, ...],
) -> dict[str, Any]:
    """Summarize decimal-IV predictions ``[N,T,Z]`` on one common reference mask.

    ``per_seed`` uses string keys for JSON roundtrips. Its strata are ``overall``,
    ``per_maturity[T]``, ``per_moneyness[Z]``, and ``per_cell[T][Z]``. Every present
    seed retains the complete ``iv_metrics`` result, including failure counts.
    Missing seeds have status ``missing`` and None strata. ``aggregate`` has the
    same strata, with each metric replaced by ``{mean, std}``; std uses ddof=1.
    Undefined values propagate per metric and stratum, with no complete-case
    averaging or pooling of quote-level tails across seeds. At least three
    distinct required seeds are necessary for this research report.
    """
    if (
        len(required_seeds) < 3
        or len(set(required_seeds)) != len(required_seeds)
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in required_seeds)
    ):
        raise ValueError("required_seeds must contain at least three distinct integer seeds")
    if any(
        isinstance(seed, bool) or not isinstance(seed, int) or seed not in required_seeds
        for seed in predictions
    ):
        raise ValueError("predictions contain an undeclared seed")
    if reference.ndim != 3 or 0 in reference.shape:
        raise ValueError("reference must have nonempty shape [N,T,Z]")
    if reference.dtype not in (torch.float32, torch.float64):
        raise TypeError("reference must have float32 or float64 dtype")
    if valid.dtype != torch.bool:
        raise TypeError("valid must have Boolean dtype")
    if valid.shape != reference.shape or valid.device != reference.device:
        raise ValueError("reference and valid must share shape and device")
    selected = reference[valid]
    if not torch.isfinite(selected).all() or (selected <= 0).any():
        raise ValueError("reference IV must be finite and positive at valid quotes")

    _, maturity_count, z_count = reference.shape
    per_seed: dict[str, Any] = {}
    for seed in required_seeds:
        if seed not in predictions:
            per_seed[str(seed)] = {
                "status": "missing",
                "overall": None,
                "per_maturity": None,
                "per_moneyness": None,
                "per_cell": None,
            }
            continue
        prediction = predictions[seed]
        overall = iv_metrics(prediction, reference, valid)
        status = "complete"
        if overall["prediction_failure_count"]:
            status = "invalid_predictions"
        elif any(overall[name] is None for name in _ERROR_METRICS):
            status = "undefined_metrics"
        per_seed[str(seed)] = {
            "status": status,
            "overall": overall,
            "per_maturity": [
                iv_metrics(prediction[:, t], reference[:, t], valid[:, t])
                for t in range(maturity_count)
            ],
            "per_moneyness": [
                iv_metrics(prediction[:, :, z], reference[:, :, z], valid[:, :, z])
                for z in range(z_count)
            ],
            "per_cell": [
                [
                    iv_metrics(prediction[:, t, z], reference[:, t, z], valid[:, t, z])
                    for z in range(z_count)
                ]
                for t in range(maturity_count)
            ],
        }
    seeds = list(per_seed.values())
    aggregate = {
        "overall": _aggregate([seed["overall"] for seed in seeds]),
        "per_maturity": [
            _aggregate(
                [
                    seed["per_maturity"][t] if seed["per_maturity"] is not None else None
                    for seed in seeds
                ]
            )
            for t in range(maturity_count)
        ],
        "per_moneyness": [
            _aggregate(
                [
                    seed["per_moneyness"][z] if seed["per_moneyness"] is not None else None
                    for seed in seeds
                ]
            )
            for z in range(z_count)
        ],
        "per_cell": [
            [
                _aggregate(
                    [
                        seed["per_cell"][t][z] if seed["per_cell"] is not None else None
                        for seed in seeds
                    ]
                )
                for z in range(z_count)
            ]
            for t in range(maturity_count)
        ],
    }
    attempted, valid_count = reference.numel(), int(valid.sum().item())
    return {
        "version": 1,
        "split": "validation",
        "units": "vol_points",
        "grid_shape": [maturity_count, z_count],
        "required_seeds": list(required_seeds),
        "missing_seeds": [seed for seed in required_seeds if seed not in predictions],
        "complete": all(seed["status"] == "complete" for seed in seeds),
        "reference_coverage": {
            "attempted": attempted,
            "valid_count": valid_count,
            "reference_failure_count": attempted - valid_count,
            "reference_failure_rate": (attempted - valid_count) / attempted,
        },
        "per_seed": per_seed,
        "aggregate": aggregate,
        "aggregation": "mean_and_sample_std_ddof_1_across_required_seeds_without_pooling",
    }


def write_validation_figures(
    report: dict[str, Any],
    histories: dict[str, list[dict[str, Any]]],
    maturity: list[float],
    z: list[float],
    output: Path,
    dpi: int,
    *,
    model_label: str = "M0",
) -> list[Path]:
    """Write Agg PNG learning curves and per-cell mean/sample-std MAE heatmaps.

    History keys are ``trial_00`` or ``seed_11`` and rows follow the shared
    training engine. Undefined values remain gaps or gray cells. Grid labels
    follow the frozen native coordinates ``k = 0.2 z sqrt(T)``, T in years.
    Figures are descriptive validation artifacts, never selection inputs.
    """
    if report["split"] != "validation":
        raise ValueError("figures require a validation report")
    if [len(maturity), len(z)] != report["grid_shape"]:
        raise ValueError("maturity and z axes must match report grid_shape")
    if not all(math.isfinite(value) and value > 0 for value in maturity) or not all(
        math.isfinite(value) for value in z
    ):
        raise ValueError("coordinate axes must be finite, with positive maturities")
    if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi < 1:
        raise ValueError("dpi must be a positive integer")
    if any(not key.startswith(("trial_", "seed_")) for key in histories):
        raise ValueError("history keys must start with trial_ or seed_")
    output.mkdir(parents=True, exist_ok=True)

    curves = Figure(figsize=(12, 8), layout="constrained")
    FigureCanvasAgg(curves)
    axes = curves.subplots(2, 2)
    for row, (prefix, title) in enumerate(
        (("trial_", "Tuning trials"), ("seed_", "Selected configuration"))
    ):
        selected_histories = [
            (key, history) for key, history in sorted(histories.items()) if key.startswith(prefix)
        ]
        for key, history in selected_histories:
            epochs = [entry["epoch"] for entry in history]
            train_loss = [
                np.nan if entry["train_loss"] is None else entry["train_loss"] for entry in history
            ]
            mae = [entry["validation_metrics"]["mean_absolute_error"] for entry in history]
            axes[row, 0].plot(epochs, train_loss, label=key, linewidth=1.4)
            axes[row, 1].plot(
                epochs,
                [np.nan if value is None else value for value in mae],
                label=key,
                linewidth=1.4,
            )
        for column in range(2):
            axis = axes[row, column]
            axis.set_title(f"{title}: {'training' if column == 0 else 'validation'}")
            axis.set_xlabel("Epoch")
            axis.set_ylabel("Standardized masked MSE" if column == 0 else "MAE (vol points)")
            axis.grid(alpha=0.2)
            if not selected_histories:
                axis.text(0.5, 0.5, "No completed epochs", ha="center", transform=axis.transAxes)
        if selected_histories:
            axes[row, 1].legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    curves.suptitle(f"{model_label} learning curves — training and validation only")
    curves_path = output / "learning_curves.png"
    curves.savefig(curves_path, dpi=dpi)
    curves.clear()

    heatmaps = Figure(figsize=(12, 5), layout="constrained")
    FigureCanvasAgg(heatmaps)
    heat_axes = heatmaps.subplots(1, 2)
    for axis, statistic, title in zip(
        heat_axes,
        ("mean", "std"),
        ("Mean of per-seed MAE", "Sample standard deviation of per-seed MAE"),
        strict=True,
    ):
        values = np.array(
            [
                [cell["mean_absolute_error"][statistic] for cell in row]
                for row in report["aggregate"]["per_cell"]
            ],
            dtype=float,
        )
        finite = values[np.isfinite(values)]
        maximum = float(finite.max()) if finite.size else 0.0
        image = axis.imshow(
            np.ma.masked_invalid(values),
            origin="lower",
            aspect="auto",
            cmap="viridis",
            vmin=0,
            vmax=maximum if maximum > 0 else 1.0,
        )
        axis.set_facecolor("#dddddd")
        axis.set_xticks(
            range(len(z)), labels=[f"{round(value, 3) or 0:g}" for value in z], rotation=45
        )
        axis.set_yticks(range(len(maturity)), labels=[f"{value:.4g}" for value in maturity])
        axis.set_xlabel("Scaled moneyness z")
        axis.set_ylabel("Maturity T (years)")
        axis.set_title(title)
        if finite.size:
            colorbar = heatmaps.colorbar(image, ax=axis, label="IV error (vol points)")
            if maximum == 0:
                colorbar.set_ticks([0])
        else:
            axis.text(
                0.5,
                0.5,
                "Undefined across required seeds",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
    heatmaps.suptitle(f"{model_label} validation-only error heatmaps")
    heatmaps.supxlabel(
        "k = log(K/F(T)) = 0.2 z √T; 1 vol point = 0.01 IV. "
        "Gray: undefined across required seeds; sample std uses ddof=1.",
        fontsize=9,
    )
    heatmaps_path = output / "validation_error_heatmaps.png"
    heatmaps.savefig(heatmaps_path, dpi=dpi)
    heatmaps.clear()
    return [curves_path, heatmaps_path]
