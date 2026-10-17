"""
End-to-end example: auto-detect R/C/L components in a netlist, interactively
ask only for what can't be inferred (include? tolerance? distribution?),
rewrite the netlist, then feed the result straight into MonteCarloEngine.

No manual {R1}-style editing and no hand-written .param lines anywhere.

Run:
    python auto_parameterize_and_run.py path/to/your_circuit.net
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ltspice_mc import MonteCarloEngine, interactive_parameterize, compute_rms
from ltspice_mc.visualization import plot_summary_dashboard

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {Path(__file__).name} <circuit.net | circuit.asc>")
        sys.exit(1)

    circuit_path = Path(sys.argv[1])

    # Step 1: detect components and ask only for tolerance/distribution/include
    # -- nominal values are read from the file automatically.
    result = interactive_parameterize(circuit_path)

    if not result.distributions:
        print("Nothing was selected for Monte Carlo -- nothing to run.")
        return

    # Step 2: hand the rewritten, now fully .param-driven file straight to
    # the existing engine. It never sees a component name, only .param keys.
    engine = MonteCarloEngine(result.output_path, parallel_sims=4)
    for ref, dist in result.distributions.items():
        engine.add_parameter(ref, dist)

    # Adjust this to whatever node/metric you actually care about.
    out_node = input("\nOutput node to measure (e.g. V(out)): ").strip() or "V(out)"
    engine.add_metric("output_rms", lambda raw: compute_rms(raw, out_node))

    n = input("Number of Monte Carlo samples [200]: ").strip()
    n_samples = int(n) if n else 200

    results = engine.run(n_samples=n_samples, seed=42)
    print(results.summary())

    fig = plot_summary_dashboard(results)
    fig.savefig("mc_summary.png", dpi=150)
    print("Saved mc_summary.png")


if __name__ == "__main__":
    main()
