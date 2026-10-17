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
from .engine import MetricSpec, MonteCarloEngine, MonteCarloResults, ParamSpec, SweepResult
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
from .optimization import CircuitOptimizer, OptimizationResult, OptimizationStep, OptimizationVariable
from .param_utils import DiscoveredParameter, discover_parameters, is_asc_file, parse_spice_value
from .relationships import (
    E_SERIES,
    Relationship,
    hold_cutoff_frequency_constant,
    hold_rc_product_constant,
    resolve_with_relationships,
    snap_to_series,
)
from .stability import (
    AutoFitResult,
    CornerRun,
    CornerSweepResult,
    FrequencySweepQuality,
    KharitonovResult,
    OrderCandidate,
    StabilityAnalysis,
    analytical_den_second_order,
    analytical_den_single_pole,
    auto_fit_transfer_function,
    build_stability_analysis,
    check_all_runs_stable,
    check_frequency_sweep,
    check_kharitonov_stability,
    fit_transfer_function_lstsq,
    is_hurwitz_stable,
    kharitonov_polynomials,
    make_fitted_extractor,
    set_ac_analysis,
    stability_margin,
    transfer_function_to_latex,
)
from .visualization import (
    plot_contour,
    plot_histogram,
    plot_optimization_convergence,
    plot_order_selection,
    plot_pdf_cdf,
    plot_splane_stability,
    plot_stability_margin_histogram,
    plot_summary_dashboard,
    plot_surface_3d,
    plot_temperature_stability_profile,
    plot_transfer_function_latex,
)

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
    "MonteCarloEngine", "MonteCarloResults", "ParamSpec", "MetricSpec", "SweepResult",
    # component relationships / E-series
    "Relationship", "resolve_with_relationships", "snap_to_series", "E_SERIES",
    "hold_rc_product_constant", "hold_cutoff_frequency_constant",
    # optimization
    "CircuitOptimizer", "OptimizationVariable", "OptimizationResult", "OptimizationStep",
    # temperature / corner sweeps / transfer-function & Kharitonov stability
    "CornerRun", "CornerSweepResult", "KharitonovResult", "StabilityAnalysis",
    "AutoFitResult", "OrderCandidate", "FrequencySweepQuality",
    "analytical_den_single_pole", "analytical_den_second_order", "fit_transfer_function_lstsq",
    "auto_fit_transfer_function", "make_fitted_extractor", "check_frequency_sweep", "set_ac_analysis",
    "is_hurwitz_stable", "kharitonov_polynomials", "check_kharitonov_stability",
    "check_all_runs_stable", "build_stability_analysis", "stability_margin",
    "transfer_function_to_latex",
    # visualization
    "plot_histogram", "plot_pdf_cdf", "plot_summary_dashboard",
    "plot_contour", "plot_surface_3d", "plot_optimization_convergence",
    "plot_splane_stability", "plot_order_selection",
    "plot_stability_margin_histogram", "plot_temperature_stability_profile",
    "plot_transfer_function_latex",
]

__version__ = "0.7.0"
