"""
Example: Monte Carlo analysis of a simple RC low-pass filter's -3 dB
bandwidth and passband gain, varying R and C by realistic component
tolerances (5% resistor, 10% capacitor).

Requires a local LTspice install + spicelib:
    pip install spicelib numpy matplotlib
    python run_rc_filter_mc.py

(This script itself cannot be executed in a sandbox without LTspice --
see tests/test_framework.py for logic that IS verified without LTspice.)
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ltspice_mc import MonteCarloEngine, Tolerance, compute_bandwidth, compute_gain_db
from ltspice_mc.visualization import plot_summary_dashboard

logging.basicConfig(level=logging.INFO, format="%(message)s")

CIRCUIT = Path(__file__).parent / "rc_lowpass.cir"


def main():
    engine = MonteCarloEngine(CIRCUIT, output_folder="./mc_temp", parallel_sims=4)
    print("Discovered .param values:", engine.discovered_parameters)

    # 5% resistor, 10% capacitor -- typical off-the-shelf tolerances.
    # Note: only .param names are ever touched -- R1/C1 designators are
    # never referenced anywhere in this script.
    engine.add_parameter("R", Tolerance(nominal=1000, tolerance_pct=5, kind="gaussian"))
    engine.add_parameter("C", Tolerance(nominal=100e-9, tolerance_pct=10, kind="uniform"))

    engine.add_metric("bandwidth_hz", lambda raw: compute_bandwidth(raw, "V(out)", "V(in)"))
    engine.add_metric("passband_gain_db", lambda raw: compute_gain_db(raw, "V(out)", "V(in)", at_freq=10))

    results = engine.run(n_samples=300, seed=1)
    print(results.summary())

    fig = plot_summary_dashboard(results, limits={"bandwidth_hz": {"lower": 1200, "upper": 2000}})
    fig.savefig("mc_summary.png", dpi=150)
    print("Saved mc_summary.png")


if __name__ == "__main__":
    main()
