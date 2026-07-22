"""
ltspice_mc
-----------
A circuit-agnostic Monte Carlo simulation framework for LTspice, built on
top of spicelib. See README.md for the full guide.

Quick start
-----------
    from ltspice_mc import MonteCarloEngine, Gaussian, Uniform, compute_rms, compute_gain_db

    engine = MonteCarloEngine("my_circuit.asc", parallel_sims=8)
    engine.add_parameter("R", Gaussian(mean=1000, std=10))
    engine.add_parameter("C", Uniform(9e-9, 11e-9))
    engine.add_metric("vout_rms", lambda raw: compute_rms(raw, "V(out)"))
    results = engine.run(n_samples=500, seed=42)
    print(results.summary())

Only `MonteCarloEngine.run()` (and anything under `ltspice_mc.runner`)
actually requires spicelib + a local LTspice install. Everything else --
parameter discovery, distributions, metric math, plotting -- works with
just numpy/matplotlib, so it can be developed and unit-tested independently.
"""

from .distributions import Custom, Distribution, Gaussian, Tolerance, Uniform
from .engine import MetricSpec, MonteCarloEngine, MonteCarloResults, ParamSpec
from .metrics import (
    compute_bandwidth,
    compute_gain_db,
    compute_mean,
    compute_overshoot_pct,
    compute_peak_to_peak,
    compute_phase_deg,
    compute_phase_margin,
    compute_rms,
    compute_settling_time,
    get_axis,
    get_signal,
)
from .netlist_parameterizer import (
    DetectedComponent,
    MonteCarloChoice,
    ParameterizationResult,
    detect_components,
    interactive_parameterize,
)
from .param_utils import DiscoveredParameter, discover_parameters, is_asc_file, parse_spice_value
from .visualization import plot_histogram, plot_pdf_cdf, plot_summary_dashboard

__all__ = [
    # distributions
    "Distribution", "Gaussian", "Uniform", "Tolerance", "Custom",
    # parameter discovery
    "discover_parameters", "parse_spice_value", "DiscoveredParameter", "is_asc_file",
    # automatic component detection / parameterization
    "detect_components", "interactive_parameterize",
    "DetectedComponent", "MonteCarloChoice", "ParameterizationResult",
    # metrics
    "get_signal", "get_axis", "compute_rms", "compute_mean", "compute_peak_to_peak",
    "compute_overshoot_pct", "compute_settling_time", "compute_gain_db", "compute_phase_deg",
    "compute_bandwidth", "compute_phase_margin",
    # engine
    "MonteCarloEngine", "MonteCarloResults", "ParamSpec", "MetricSpec",
    # visualization
    "plot_histogram", "plot_pdf_cdf", "plot_summary_dashboard",
]

__version__ = "0.2.0"
