"""
Example: combined Corner + Temperature stability analysis.

Sweeps R and C corner values together with a -40..125C temperature corner,
extracts each run's single-pole D(s) = 1 + s*R*C analytically (exact -- see
stability.py for why this is preferred over curve-fitting whenever you know
your topology), then checks robust stability two complementary ways:

  1. check_all_runs_stable() -- the ACTUAL simulated corners, no relaxation.
  2. check_kharitonov_stability() -- the conservative "what if the
     coefficients could vary independently" bound. Read stability.py's
     module docstring before trusting a Kharitonov "fail": it does not by
     itself mean your real circuit is unstable.

Requires a local LTspice + spicelib install to actually run:
    pip install spicelib numpy matplotlib scipy
    python temperature_corner_stability.py
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ltspice_mc import (
    MonteCarloEngine, Tolerance,
    analytical_den_single_pole, check_all_runs_stable, check_kharitonov_stability,
)
from ltspice_mc.visualization import plot_splane_stability
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")

CIRCUIT = Path(__file__).parent / "rc_lowpass.cir"


def main():
    engine = MonteCarloEngine(CIRCUIT, output_folder="./mc_temp", parallel_sims=4)

    # Temperature becomes a normal .param -- must be called before any run.
    engine.enable_temperature_control()  # adds .temp {TEMP_C} to the netlist

    # R and C don't need add_parameter() here since we're not doing
    # statistical MC -- run_corner_sweep takes their discrete corner values
    # directly. (add_parameter is only consulted for params NOT listed in
    # corner_values, to decide what to hold them at -- see fixed_params.)

    # D(s) = 1 + s*R*C for this single-pole RC low-pass -- exact, analytical,
    # no curve-fitting needed since we know the topology.
    extract_den = analytical_den_single_pole(lambda p: p["R"] * p["C"])

    result = engine.run_corner_sweep(
        corner_values={
            "R": [950, 1000, 1050],           # +/- 5% resistor corner
            "C": [90e-9, 100e-9, 110e-9],      # +/- 10% capacitor corner
            "TEMP_C": [-40, 27, 125],           # cold / room / hot
        },
        extract_den_fn=extract_den,
    )
    print(result.summary())

    # --- Check 1: the actual simulated corners, directly, no relaxation ---
    stack = result.stacked_den_coeffs()
    direct_stable = check_all_runs_stable(stack)
    print(f"\nDirect per-run check: {direct_stable.sum()}/{len(direct_stable)} corners stable")

    # --- Check 2: Kharitonov's conservative independent-coefficient bound ---
    kharitonov = check_kharitonov_stability(result.coefficient_bounds())
    print("\n" + kharitonov.summary())

    fig = plot_splane_stability(
        kharitonov,
        all_run_roots=[np.roots(row) for row in stack],
        save_path="kharitonov_splane.png",
    )
    print("\nSaved kharitonov_splane.png")


if __name__ == "__main__":
    main()
