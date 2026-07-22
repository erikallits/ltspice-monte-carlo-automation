# ltspice_mc

Circuit-agnostic Monte Carlo simulation framework for LTspice. Varies only
`.param` values (never component designators like R1/C1), runs LTspice in
parallel via spicelib, extracts KPIs from `.raw` output, and aggregates
everything in memory as NumPy arrays.

## Install

```bash
pip install spicelib numpy matplotlib
```

LTspice itself must be installed separately (Windows/macOS native, or Linux
via Wine) — spicelib auto-detects standard install paths.

## Quickstart

```python
from ltspice_mc import MonteCarloEngine, Gaussian, Uniform, compute_rms, compute_bandwidth

engine = MonteCarloEngine("my_circuit.asc", parallel_sims=8)
print(engine.discovered_parameters)          # every .param found, no component names needed

engine.add_parameter("R", Gaussian(mean=1000, std=10))
engine.add_parameter("C", Uniform(9e-9, 11e-9))
engine.add_metric("vout_rms", lambda raw: compute_rms(raw, "V(out)"))
engine.add_metric("bw", lambda raw: compute_bandwidth(raw, "V(out)", "V(in)"))

results = engine.run(n_samples=500, seed=42)
print(results.summary())
print("Yield:", results.compute_yield("bw", lower=1200, upper=2000), "%")

from ltspice_mc.visualization import plot_summary_dashboard
plot_summary_dashboard(results, limits={"bw": {"lower": 1200, "upper": 2000}}).savefig("summary.png")
```

Works identically for a `.asc` schematic or a plain SPICE netlist (`.net`/`.cir`) —
file type is auto-detected from the extension.

## Automatic component detection (no more hand-editing {R1} and .param lines)

`interactive_parameterize()` scans a netlist/schematic for standard 2-terminal R/C/L
components, shows you what it found, and asks *only* what it can't infer (include in MC?
tolerance? distribution?) — nominal values are read from the file automatically:

```python
from ltspice_mc import interactive_parameterize, MonteCarloEngine

result = interactive_parameterize("my_circuit.net")   # asks questions on stdin
# -> Detected resistor: R1 = 10k
#    Detected resistor: R2 = 4.7k
#    Detected capacitor: C1 = 100n
#    R1 (resistor), current value 10k:
#      Include in Monte Carlo? [Y/n]:
#      Nominal value [10k]:
#      Tolerance % [5]:
#      Distribution [uniform/gaussian] (default: uniform):
#    ...

engine = MonteCarloEngine(result.output_path, parallel_sims=8)
for ref, dist in result.distributions.items():
    engine.add_parameter(ref, dist)   # ready-made Tolerance() distributions, keyed by ref designator
```

Behind the scenes this rewrites `R1 in out 10k` → `R1 in out {R1}` and adds `.param R1=10000`
(or the equivalent `SYMATTR Value {R1}` + `TEXT ... !.param ...` for `.asc` files) — every other
line is left untouched. Already-parameterized components (`{...}`) are left alone; components
whose value isn't a plain number (model references, behavioral `R=<expr>` forms — a deliberately
out-of-scope 3-terminal case) are flagged and skipped rather than risk corrupting the circuit.
See `examples/auto_parameterize_and_run.py` for the full loop (detect → ask → rewrite → simulate).

## Module map

| File | Responsibility |
|---|---|
| `param_utils.py` | Regex-based `.param` discovery/parsing — no spicelib needed, works on plain text |
| `netlist_parameterizer.py` | Auto-detects R/C/L components, prompts interactively, rewrites the netlist/schematic to be `.param`-driven |
| `distributions.py` | `Gaussian`, `Uniform`, `Tolerance`, `Custom` samplers + rejection sampling for bounds |
| `metrics.py` | RMS, mean, peak-to-peak, overshoot, settling time, gain, phase, bandwidth, phase margin |
| `runner.py` | spicelib `SimRunner`/`AscEditor`/`SpiceEditor` wrapper — the only module requiring spicelib |
| `engine.py` | `MonteCarloEngine` — sampling → parallel runs → callback → NumPy aggregation |
| `visualization.py` | Histograms, PDF/CDF, summary dashboard, yield annotation |

## Requirements traceability

- **Circuit-agnostic, `.param`-only**: `param_utils.discover_parameters()` never parses component
  designators; `runner.py` only ever calls `editor.set_parameters(**kwargs)`.
- **Sampling**: `Gaussian`/`Uniform`/`Tolerance`/`Custom` (the last wraps any `f(n, rng) -> array`,
  including scipy.stats distributions).
- **LTspice automation**: `spicelib.SimRunner` (current equivalent of the legacy `SimCommander`),
  `AscEditor`/`SpiceEditor.set_parameters()`, parallel execution via `parallel_sims`.
- **Output handling**: `spicelib.raw.raw_read.RawRead`, per-run callback processes and deletes
  `.raw`/`.log` immediately rather than letting them pile up.
- **Aggregation**: everything lands in NumPy arrays on `MonteCarloResults`; no intermediate files.
- **Visualization**: `plot_histogram`, `plot_pdf_cdf`, `plot_summary_dashboard`.
- **Robustness**: missing/aborted runs recorded as NaN + logged error (not a crash — see
  `engine.py`'s callback); out-of-range samples use rejection sampling with a clip fallback.

## Note on spicelib's built-in `Montecarlo` toolkit

spicelib ships a `spicelib.sim.toolkit.montecarlo.Montecarlo` helper, but it varies component
*values* via tolerances set per reference/class (`mc.set_tolerance('R', 0.01)`), which conflicts
with this spec's "no component identifiers, `.param`-only" constraint — so this framework is built
directly on the lower-level `SimRunner`/`AscEditor` primitives instead.

## Tested vs. not tested here

This sandbox has no LTspice install and no network access to install spicelib, so
`tests/test_framework.py` (38 tests, all passing) validates every piece of logic that doesn't
require an actual LTspice process: `.param` parsing, sampling statistics, rejection sampling, and
all metric math (validated against synthetic signals with known analytic answers — e.g. bandwidth
of a synthetic single-pole response, phase margin of a synthetic loop gain). The full
`MonteCarloEngine.run()` orchestration loop (sampling → callback → aggregation → error handling)
is also tested end-to-end using a fake runner, so only the `runner.py`↔spicelib boundary itself is
unverified by me — that requires your local LTspice + `pip install spicelib`.

`examples/rc_lowpass.cir` is a plain-text SPICE netlist (not a hand-drawn `.asc`), which I can
write with full confidence since it has no schematic geometry to get wrong. Point
`MonteCarloEngine` at your own `.asc` file the same way — `AscEditor` handles it identically.
