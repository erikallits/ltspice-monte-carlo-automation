"""
General-purpose Monte Carlo + Corner + Kharitonov stability analysis for
ANY linear circuit LTspice can AC-simulate -- no assumption about topology,
component names, or system order. Only the CONFIG block below should need
editing for a different circuit; everything past it is generic.

Three genuinely different stability claims come out of this, printed and
plotted separately rather than collapsed into one verdict -- see
stability.py's module docstring for exactly what each one does and does
NOT tell you:
  1. Empirical Monte Carlo stability  -- statistical, from random sampling.
  2. Corner stability                 -- exhaustive over chosen extremes.
  3. Kharitonov robust stability      -- conservative independent-box bound.

The transfer function's order is never hand-specified: it's determined
ONCE automatically (auto_fit_transfer_function via determine_transfer_
function_order(), balancing fit quality against model complexity through
BIC/AIC) from a single nominal-parameter simulation, then reused as a FIXED
order for every Monte Carlo sample and every corner -- required for
Kharitonov's coefficient bounds to be meaningful at all. Inspect
order_selection.png before trusting the automatic choice, especially for
anything spec- or safety-relevant; automatic order selection from frequency
response data is a genuinely harder, less reliable problem than knowing
your circuit's structure. If you DO know it, stability.analytical_den_
single_pole / analytical_den_second_order (or your own formula) are exact
and preferable -- see the README.

Requires a local LTspice + spicelib install to actually run:
    pip install spicelib numpy matplotlib scipy
    python run_my_circuit.py
"""
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np

from ltspice_mc import (
    MonteCarloEngine, Uniform, interactive_parameterize,
    make_fitted_extractor,
)
from ltspice_mc.visualization import (
    plot_order_selection, plot_splane_stability, plot_summary_dashboard,
)

# =============================================================================
# CONFIG -- edit this block for your circuit; nothing below should need it.
# =============================================================================

CIRCUIT_PATH = Path("circuit.asc")   # your schematic or netlist

OUT_SIGNAL = "V(vout)"               # transfer-function output node
IN_SIGNAL = "V(vin)"                 # input node; set to None to treat OUT_SIGNAL itself as H(s)

TEMP_RANGE_C = (-40.0, 125.0)        # set to None to leave temperature out entirely

MAX_ORDER = 6                        # ceiling for automatic order selection -- raise if
                                      # order_selection.png / the printed summary flags the
                                      # choice as sitting at this ceiling
ORDER_CRITERION = "bic"              # "bic" (parsimonious, recommended) or "aic"

RUN_MONTE_CARLO = True
N_MC_SAMPLES = 500

RUN_CORNER_SWEEP = True
CORNER_POINTS_PER_PARAM = 3          # e.g. 3 -> (min, nominal, max) per component/temperature

# Optional additional scalar metrics, beyond stability -- e.g.:
#   EXTRA_METRICS = {"bandwidth_hz": lambda raw: compute_bandwidth(raw, OUT_SIGNAL, IN_SIGNAL)}
# Leave empty for a purely stability-focused run.
EXTRA_METRICS = {}

# =============================================================================


def build_corner_values(distributions, n_points=3):
    """n_points-corner set spanning each component's tolerance, taken
    directly from the Tolerance objects interactive_parameterize() already
    built -- no retyping component ranges by hand."""
    corners = {}
    for name, dist in distributions.items():
        delta = dist.nominal * dist.tolerance_pct / 100.0
        if n_points <= 1 or delta == 0:
            corners[name] = [dist.nominal]
        else:
            corners[name] = sorted(set(np.linspace(dist.nominal - delta, dist.nominal + delta, n_points).tolist()))
    return corners


def main():
    if not CIRCUIT_PATH.exists():
        print(f"[Error] '{CIRCUIT_PATH}' not found.")
        return

    print("--- Automatic component/parameter discovery ---")
    result = interactive_parameterize(CIRCUIT_PATH)
    if not result.distributions:
        print("[Error] No components selected for analysis. Exiting.")
        return
    print()

    # One engine for the whole script. enable_temperature_control() must be
    # called before ANY run()/run_corner_sweep() -- it queues the ".temp
    # {TEMP_C}" instruction that makes TEMP_C actually affect the simulation.
    engine = MonteCarloEngine(result.output_path)
    if TEMP_RANGE_C is not None:
        engine.enable_temperature_control()

    for name, dist in result.distributions.items():
        engine.add_parameter(name, dist)
    if TEMP_RANGE_C is not None:
        engine.add_parameter("TEMP_C", Uniform(*TEMP_RANGE_C), warn_if_undeclared=False)

    for metric_name, metric_fn in EXTRA_METRICS.items():
        engine.add_metric(metric_name, metric_fn)

    # -------------------------------------------------------------------
    # Determine the transfer function order ONCE, from a single simulation
    # at nominal parameter values. Every sample below reuses this fixed
    # order -- mixing orders across samples would make coefficient bounds
    # (and therefore Kharitonov) meaningless.
    # -------------------------------------------------------------------
    print(f"Determining transfer function order (max_order={MAX_ORDER}, criterion={ORDER_CRITERION})...")
    order_result = engine.determine_transfer_function_order(
        OUT_SIGNAL, IN_SIGNAL, max_order=MAX_ORDER, criterion=ORDER_CRITERION,
    )
    print(order_result.summary())
    print()

    plot_order_selection(order_result, save_path="order_selection.png")
    print("Saved order_selection.png -- inspect this before trusting the automatic order choice.\n")

    extract_den = make_fitted_extractor(
        OUT_SIGNAL, IN_SIGNAL, order_result.best.num_order, order_result.best.den_order,
    )

    # -------------------------------------------------------------------
    # PART 1: empirical Monte Carlo stability (+ any EXTRA_METRICS)
    # -------------------------------------------------------------------
    mc_results = None
    if RUN_MONTE_CARLO:
        print(f"=== Monte Carlo: {N_MC_SAMPLES} samples ===")
        start = time.time()
        mc_results = engine.run(n_samples=N_MC_SAMPLES, seed=42, extract_den_fn=extract_den)
        print(f"Completed in {time.time() - start:.1f}s")
        print(mc_results.summary())
        print()
        print(mc_results.stability_analysis().summary())
        print()

        if mc_results.metrics:
            plot_summary_dashboard(mc_results, save_path="monte_carlo_dashboard.png")
            print("Saved monte_carlo_dashboard.png")

    # -------------------------------------------------------------------
    # PART 2: exhaustive corner sweep + Kharitonov robust stability
    # -------------------------------------------------------------------
    if RUN_CORNER_SWEEP:
        corner_values = build_corner_values(result.distributions, CORNER_POINTS_PER_PARAM)
        if TEMP_RANGE_C is not None:
            corner_values["TEMP_C"] = sorted({TEMP_RANGE_C[0], sum(TEMP_RANGE_C) / 2, TEMP_RANGE_C[1]})

        n_combos = 1
        for v in corner_values.values():
            n_combos *= len(v)
        print(f"=== Corner sweep: {' x '.join(str(len(v)) for v in corner_values.values())} = {n_combos} combinations ===")

        corner_results = engine.run_corner_sweep(corner_values=corner_values, extract_den_fn=extract_den)
        print(corner_results.summary())
        print()

        if corner_results.n_success == 0:
            print("[Error] No corner simulation succeeded -- check that OUT_SIGNAL/IN_SIGNAL "
                  "actually exist as node names in your circuit.")
        else:
            corner_stability = corner_results.stability_analysis()
            print(corner_stability.summary())
            print()

            stack = corner_results.stacked_den_coeffs()
            plot_splane_stability(
                corner_stability.kharitonov,
                all_run_roots=[np.roots(row) for row in stack],
                save_path="kharitonov_splane.png",
            )
            print("Saved kharitonov_splane.png")

    plt.show()


if __name__ == "__main__":
    main()
