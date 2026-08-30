"""
visualization.py
------------------
matplotlib-based plotting of Monte Carlo results: histograms, PDF/CDF
overlays, and a compact statistical-summary dashboard with yield annotation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .engine import MonteCarloResults, SweepResult
from .optimization import OptimizationResult
from .stability import AutoFitResult, KharitonovResult


def _valid_values(results: MonteCarloResults, metric: str) -> np.ndarray:
    values = results.metrics[metric]
    mask = results.success & ~np.isnan(values)
    return values[mask]


def plot_histogram(
    results: MonteCarloResults,
    metric: str,
    bins: int = 30,
    lower_limit: Optional[float] = None,
    upper_limit: Optional[float] = None,
    ax: Optional[Axes] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> Union[Figure, Axes]:
    """Histogram of one metric, annotated with its mean and (optionally) spec limits."""
    values = _valid_values(results, metric)
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(7, 4.5))
    if len(values) == 0:
        ax.text(0.5, 0.5, "no valid samples", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.hist(values, bins=bins, color="#4C72B0", edgecolor="white", alpha=0.9)
        ax.axvline(np.mean(values), color="black", linestyle="--", linewidth=1.2,
                   label=f"mean = {np.mean(values):.4g}")
        if lower_limit is not None:
            ax.axvline(lower_limit, color="#C44E52", linewidth=1.2, label=f"lower = {lower_limit:.4g}")
        if upper_limit is not None:
            ax.axvline(upper_limit, color="#C44E52", linewidth=1.2, label=f"upper = {upper_limit:.4g}")
        ax.legend(fontsize=8)
    ax.set_xlabel(metric)
    ax.set_ylabel("count")
    ax.set_title(f"{metric}  (n={len(values)})")
    if own_fig:
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
        return fig
    return ax


def plot_pdf_cdf(
    results: MonteCarloResults,
    metric: str,
    bins: int = 40,
    save_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """Side-by-side estimated PDF (histogram, density-normalized) and empirical CDF."""
    values = _valid_values(results, metric)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    if len(values) == 0:
        for ax in (ax1, ax2):
            ax.text(0.5, 0.5, "no valid samples", ha="center", va="center", transform=ax.transAxes)
    else:
        ax1.hist(values, bins=bins, density=True, color="#4C72B0", edgecolor="white", alpha=0.9)
        sorted_vals = np.sort(values)
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax2.plot(sorted_vals, cdf, color="#55A868", linewidth=1.8)
        ax2.grid(alpha=0.3)

    ax1.set_title(f"PDF (estimated): {metric}")
    ax1.set_xlabel(metric)
    ax1.set_ylabel("density")
    ax2.set_title(f"CDF (empirical): {metric}")
    ax2.set_xlabel(metric)
    ax2.set_ylabel("cumulative probability")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_summary_dashboard(
    results: MonteCarloResults,
    metrics: Optional[List[str]] = None,
    limits: Optional[Dict[str, Dict[str, float]]] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """
    Grid of histograms, one per metric. `limits`, if given, maps metric name
    to {"lower": ..., "upper": ...} spec limits, each optional; any metric
    with limits gets a yield-percentage annotation.
    """
    metrics = metrics or list(results.metrics.keys())
    limits = limits or {}
    n = len(metrics)
    ncols = min(3, n) or 1
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.8 * nrows), squeeze=False)

    for idx, metric in enumerate(metrics):
        ax = axes[idx // ncols][idx % ncols]
        lims = limits.get(metric, {})
        plot_histogram(results, metric, ax=ax, lower_limit=lims.get("lower"), upper_limit=lims.get("upper"))
        if "lower" in lims or "upper" in lims:
            y = results.compute_yield(metric, lims.get("lower"), lims.get("upper"))
            ax.text(0.02, 0.95, f"yield: {y:.1f}%", transform=ax.transAxes,
                    va="top", fontsize=9, color="#C44E52")

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle(
        f"Monte Carlo summary -- {results.n_success}/{results.n_samples} runs "
        f"succeeded ({results.yield_pct:.1f}%)", fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


# ---------------------------------------------------------------------------
# Trade-off analysis: 2D sweeps and optimization convergence
# ---------------------------------------------------------------------------

def plot_contour(
    sweep: SweepResult,
    metric: str,
    kind: str = "contourf",
    levels: int = 20,
    mark_best: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """
    2D contour/heatmap of `metric` over a sweep_2d() grid -- the "sweet
    spot" trade-off map. `kind`: "contourf" (smooth filled contours, implies
    interpolation between grid points) or "pcolormesh" (flat-shaded heatmap,
    one cell per actually-simulated point -- more honest about resolution
    for a coarse grid). `mark_best`: "max" or "min" to star the best point.
    """
    if kind not in ("contourf", "pcolormesh"):
        raise ValueError("kind must be 'contourf' or 'pcolormesh'")
    X, Y = np.meshgrid(sweep.values_x, sweep.values_y)
    Z = np.where(sweep.success, sweep.metrics[metric], np.nan)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    if kind == "contourf":
        cs = ax.contourf(X, Y, Z, levels=levels, cmap="viridis")
    else:
        cs = ax.pcolormesh(X, Y, Z, shading="nearest", cmap="viridis")
    fig.colorbar(cs, ax=ax, label=metric)

    if mark_best:
        best = sweep.best(metric, mode=mark_best)
        ax.plot(best[sweep.param_x], best[sweep.param_y], marker="*", markersize=18,
                color="red", markeredgecolor="black", linestyle="none",
                label=f"{mark_best} {metric} = {best[metric]:.4g}")
        ax.legend(loc="upper right")

    ax.set_xlabel(sweep.param_x)
    ax.set_ylabel(sweep.param_y)
    ax.set_title(f"{metric} vs {sweep.param_x} / {sweep.param_y}  "
                 f"({sweep.n_success}/{sweep.n_points} points ok)")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_surface_3d(
    sweep: SweepResult,
    metric: str,
    save_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """True 3D surface -- an alternative view of the same sweep_2d() data as plot_contour()."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registers the '3d' projection

    X, Y = np.meshgrid(sweep.values_x, sweep.values_y)
    Z = np.where(sweep.success, sweep.metrics[metric], np.nan)

    fig = plt.figure(figsize=(8, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none", antialiased=True)
    ax.set_xlabel(sweep.param_x)
    ax.set_ylabel(sweep.param_y)
    ax.set_zlabel(metric)
    ax.set_title(f"{metric} surface over {sweep.param_x} / {sweep.param_y}")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_optimization_convergence(
    result: OptimizationResult,
    save_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """Objective value vs. evaluation number -- shows whether/how fast CircuitOptimizer converged."""
    obj_values = [step.objective_value for step in result.history]
    fig, ax = plt.subplots(figsize=(7, 4))
    if obj_values:
        ax.plot(range(1, len(obj_values) + 1), obj_values, marker="o", markersize=3, linewidth=1)
        best_idx = int(np.argmin(obj_values))
        ax.axvline(best_idx + 1, color="#C44E52", linestyle="--", linewidth=1,
                   label=f"best @ evaluation {best_idx + 1}")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "no evaluations recorded", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("evaluation")
    ax.set_ylabel("objective value")
    ax.set_title("Optimization convergence")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_splane_stability(
    kharitonov: KharitonovResult,
    all_run_roots: Optional[List[np.ndarray]] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """
    Plots roots on the s-plane: the 4 Kharitonov polynomials' roots (one
    marker style each, per the module docstring's stability_analysis.py
    interpretation notes) plus, optionally, every individual corner run's
    own roots underneath for direct comparison (pass e.g.
    [np.roots(row) for row in corner_result.stacked_den_coeffs()]).
    The imaginary axis is drawn as the stability boundary; shading marks
    the unstable (right-half-plane) region.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    all_roots_for_scaling = list(kharitonov.roots.values())
    if all_run_roots:
        all_roots_for_scaling += list(all_run_roots)
    all_real = np.concatenate([r.real for r in all_roots_for_scaling])
    all_imag = np.concatenate([r.imag for r in all_roots_for_scaling])
    x_margin = 0.15 * (all_real.max() - all_real.min() + 1e-9)
    y_margin = 0.15 * (all_imag.max() - all_imag.min() + 1e-9)
    x_min, x_max = all_real.min() - x_margin, max(all_real.max() + x_margin, x_margin)
    y_min, y_max = all_imag.min() - y_margin, all_imag.max() + y_margin

    ax.axvspan(0, x_max, color="#C44E52", alpha=0.08, zorder=0, label="unstable (RHP)")
    ax.axvline(0, color="black", linewidth=1.2, zorder=1)

    if all_run_roots:
        for i, roots in enumerate(all_run_roots):
            ax.scatter(roots.real, roots.imag, marker=".", s=25, color="#888888",
                       alpha=0.5, zorder=2, label="individual corner runs" if i == 0 else None)

    markers = {"K1": "o", "K2": "s", "K3": "^", "K4": "D"}
    colors = {"K1": "#4C72B0", "K2": "#55A868", "K3": "#C44E52", "K4": "#8172B2"}
    for name in ("K1", "K2", "K3", "K4"):
        roots = kharitonov.roots[name]
        stable_txt = "stable" if kharitonov.stable[name] else "UNSTABLE"
        ax.scatter(roots.real, roots.imag, marker=markers[name], s=90, color=colors[name],
                   edgecolor="black", linewidth=0.8, zorder=3, label=f"{name} ({stable_txt})")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Re(s)")
    ax.set_ylabel("Im(s)")
    verdict = "ROBUSTLY STABLE" if kharitonov.all_stable else "NOT CONFIRMED STABLE"
    ax.set_title(f"Kharitonov roots on the s-plane -- {verdict}")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_order_selection(auto_fit: AutoFitResult, save_path: Optional[Union[str, Path]] = None) -> Figure:
    """
    Fit residual (RSS) and information criterion (AIC/BIC) for every
    (num_order, den_order) candidate tried by auto_fit_transfer_function(),
    with the selected order marked. Use this to sanity-check an automatic
    order choice before trusting it -- e.g. confirm it isn't just sitting at
    the largest order tried (raise max_order if so) or on a razor-thin tie.
    """
    candidates = sorted(auto_fit.candidates, key=lambda c: (c.den_order, c.num_order))
    labels = [f"n{c.num_order}d{c.den_order}" for c in candidates]
    x = np.arange(len(candidates))
    best_idx = next(i for i, c in enumerate(candidates) if c is auto_fit.best)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.semilogy(x, [c.rss for c in candidates], marker="o", markersize=4, color="#4C72B0")
    ax1.axvline(best_idx, color="#C44E52", linestyle="--", linewidth=1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=90, fontsize=7)
    ax1.set_ylabel("RSS (log scale)")
    ax1.set_title("Fit residual by candidate order")
    ax1.grid(alpha=0.3)

    ax2.plot(x, [c.aic for c in candidates], marker="o", markersize=4, label="AIC")
    ax2.plot(x, [c.bic for c in candidates], marker="s", markersize=4, label="BIC")
    ax2.axvline(best_idx, color="#C44E52", linestyle="--", linewidth=1,
                label=f"selected: num={auto_fit.best.num_order}, den={auto_fit.best.den_order}")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=90, fontsize=7)
    ax2.set_ylabel("information criterion")
    ax2.set_title(f"Order selection ({auto_fit.criterion.upper()})")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig



