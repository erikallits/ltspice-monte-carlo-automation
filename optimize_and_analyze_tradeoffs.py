"""
End-to-end example: Component Interdependency & Circuit Optimization.

Circuit: the same RC low-pass filter from rc_lowpass.cir (.param R=1k C=100n).

Demonstrates all three pieces together:
  1. A Relationship that holds R*C constant so C can be swept without the
     cutoff frequency drifting -- i.e. "if C1 increases, how should R1
     adjust to keep fc constant".
  2. CircuitOptimizer (scipy.optimize under the hood) finding the R, C pair
     that hits a target -3 dB bandwidth, with R constrained to real E24
     resistor values.
  3. sweep_2d() + plot_contour() to visualize the R/C trade-off space and
     mark the sweet spot directly on the map.

Requires a local LTspice + spicelib install (pip install spicelib) --
see tests/test_relationships_optimization.py for the parts of this (E-series
snapping, relationship resolution, and the optimizer's control flow) that
are verified without LTspice, against a synthetic objective function.
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ltspice_mc import (
    CircuitOptimizer, OptimizationVariable, Relationship,
    hold_rc_product_constant, resolve_with_relationships, snap_to_series,
    MonteCarloEngine, compute_bandwidth,
)
from ltspice_mc.visualization import plot_contour, plot_optimization_convergence

logging.basicConfig(level=logging.INFO, format="%(message)s")

CIRCUIT = Path(__file__).parent / "rc_lowpass.cir"


def main():
    # --- 1. Explicit relationship: hold the RC time constant (hence fc) fixed ---
    target_tau = 1000.0 * 100e-9  # R=1k, C=100n nominal -> tau = 1e-4 s (~1591.5 Hz)
    hold_fc = Relationship(
        dependent="R",
        solve=hold_rc_product_constant("C", target_product=target_tau),
        description="R adjusts automatically so R*C (and therefore fc) stays constant as C varies",
    )
    print("R for a few C values, fc held constant via the relationship:")
    for c in (50e-9, 100e-9, 200e-9):
        r = resolve_with_relationships({"C": c}, [hold_fc])["R"]
        print(f"  C={c*1e9:.0f}nF -> R={r:.1f} ohm")

    # --- 2. Optimize: find the E24-constrained R (with C free) that hits
    #        a target -3 dB bandwidth of 2000 Hz ---
    optimizer = CircuitOptimizer(
        CIRCUIT,
        free_variables=[
            OptimizationVariable("R", bounds=(200, 20000), series="E24", x0=1000),
            OptimizationVariable("C", bounds=(10e-9, 1e-6), x0=100e-9),
        ],
    )
    result = optimizer.target(
        metric_fn=lambda raw: compute_bandwidth(raw, "V(out)", "V(in)"),
        target_value=2000.0,
        maxiter=200,
    )
    print("\n" + result.summary())
    print(f"(R landed on the E24 grid: {result.resolved_values['R']} == "
          f"{snap_to_series(result.resolved_values['R'], 'E24')})")

    fig1 = plot_optimization_convergence(result)
    fig1.savefig("optimization_convergence.png", dpi=150)

    # --- 3. Trade-off map: sweep R x C directly and visualize bandwidth ---
    engine = MonteCarloEngine(CIRCUIT, parallel_sims=4)
    engine.add_metric("bandwidth_hz", lambda raw: compute_bandwidth(raw, "V(out)", "V(in)"))
    sweep = engine.sweep_2d(
        param_x="R", values_x=[300, 500, 750, 1000, 1500, 2200, 3300, 4700, 6800, 10000],
        param_y="C", values_y=[20e-9, 47e-9, 100e-9, 220e-9, 470e-9],
    )
    fig2 = plot_contour(sweep, "bandwidth_hz", kind="contourf", mark_best=None)
    fig2.savefig("rc_tradeoff_map.png", dpi=150)
    print("\nSaved optimization_convergence.png and rc_tradeoff_map.png")


if __name__ == "__main__":
    main()
