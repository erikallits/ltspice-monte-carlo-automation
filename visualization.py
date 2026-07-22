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

from .engine import MonteCarloResults


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
