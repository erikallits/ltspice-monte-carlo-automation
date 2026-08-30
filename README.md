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

## Component interdependency & circuit optimization

Three pieces, in `relationships.py` / `optimization.py` / `engine.sweep_2d()`:

**1. Explicit relationships** — derive one parameter from others instead of letting them drift independently:

```python
from ltspice_mc import Relationship, hold_rc_product_constant, resolve_with_relationships

# R adjusts automatically to keep R*C (and therefore fc) constant as C varies
hold_fc = Relationship(dependent="R", solve=hold_rc_product_constant("C", target_product=1e-4))
resolve_with_relationships({"C": 200e-9}, [hold_fc])   # -> {"C": 200e-9, "R": 500.0}
```

`hold_rc_product_constant` / `hold_cutoff_frequency_constant` cover the topology-independent RC case directly.
Q-factor and gain-bandwidth product genuinely depend on circuit topology (series RLC vs. Sallen-Key vs.
multiple-feedback, etc.) — there's no single correct formula for all of them, so define your own `Relationship`
the same way rather than trusting a one-size-fits-all built-in.

**2. Optimization** — `CircuitOptimizer` drives `scipy.optimize.minimize` through sequential LTspice runs:

```python
from ltspice_mc import CircuitOptimizer, OptimizationVariable, compute_bandwidth

optimizer = CircuitOptimizer("circuit.asc", free_variables=[
    OptimizationVariable("R", bounds=(200, 20000), series="E24"),   # snapped to a real E24 value every evaluation
    OptimizationVariable("C", bounds=(10e-9, 1e-6)),
])
result = optimizer.target(lambda raw: compute_bandwidth(raw, "V(out)", "V(in)"), target_value=2000.0)
print(result.summary())
```

`.minimize()` / `.maximize()` / `.target()` cover the common cases. Nelder-Mead is the default method because
E-series snapping makes the objective piecewise-constant, which gradient-based methods handle poorly; its
initial exploration step is deliberately widened (see `optimization.py`) so it can't get stuck reading a flat
E-series plateau as "already converged" right at the starting point.

**3. Trade-off maps** — `MonteCarloEngine.sweep_2d()` runs a full parameter grid in parallel and hands it to
`plot_contour()` / `plot_surface_3d()`:

```python
sweep = engine.sweep_2d("R", [300, 500, 750, 1000, 1500, 2200], "C", [20e-9, 47e-9, 100e-9, 220e-9])
plot_contour(sweep, "bandwidth_hz", mark_best="max").savefig("tradeoff.png")
```

See `examples/optimize_and_analyze_tradeoffs.py` for all three together.

## Temperature, Corner sweeps, and Kharitonov robust stability

`stability.py` extracts a transfer function's denominator D(s) for every combined Corner+Temperature
run and checks robust stability two complementary ways. **Read this before trusting a verdict:**

> Kharitonov's theorem applies exactly to coefficients that vary *independently* within a box. Your
> circuit's D(s) coefficients are NOT independent — they're all derived from the same handful of
> R/L/C/temperature values. A Kharitonov **pass** is a valid, genuinely useful robustness guarantee
> (it covers worse coefficient correlations than your circuit can actually produce). A Kharitonov
> **fail** does *not* by itself prove your real circuit is unstable — always cross-check with
> `check_all_runs_stable()`, which tests your actual simulated corners directly, no relaxation.

```python
from ltspice_mc import MonteCarloEngine, analytical_den_single_pole, check_all_runs_stable, check_kharitonov_stability
from ltspice_mc.visualization import plot_splane_stability

engine = MonteCarloEngine("circuit.asc")
engine.enable_temperature_control()  # adds `.temp {TEMP_C}` -- call before any run/sweep

extract_den = analytical_den_single_pole(lambda p: p["R"] * p["C"])  # D(s) = 1 + s*R*C, exact
result = engine.run_corner_sweep(
    corner_values={"R": [950, 1000, 1050], "C": [90e-9, 100e-9, 110e-9], "TEMP_C": [-40, 27, 125]},
    extract_den_fn=extract_den,
)

direct = check_all_runs_stable(result.stacked_den_coeffs())         # ground truth, your actual corners
kharitonov = check_kharitonov_stability(result.coefficient_bounds())  # conservative independent-box bound
plot_splane_stability(kharitonov).savefig("splane.png")
```

Two ways to get D(s) per run: **analytical** (`analytical_den_single_pole` / `analytical_den_second_order`
— exact, recommended whenever you know your topology) or **numerical least-squares fit**
(`fit_transfer_function_lstsq`, Levy's method via linear algebra, not a single scipy.signal call —
you must specify the model order, and fit quality depends on your frequency sweep covering the poles;
read its docstring). `scipy.signal.tf2zpk`/`np.roots` are the right tools once you have coefficients —
getting from raw AC-sweep samples *to* coefficients is the part that needs care.

See `examples/temperature_corner_stability.py` for the full loop.

### Automatic order selection, and empirical Monte Carlo stability

Don't want to hand-specify the transfer function's order? `determine_transfer_function_order()` runs
**one** simulation at nominal parameter values and picks (num_order, den_order) automatically via
BIC/AIC (balancing fit quality against model complexity — plain least-squares RSS can only improve as
order increases, so *some* complexity penalty is mandatory or this would just always pick `max_order`).
That order is then held **fixed** for every subsequent sample — mixing orders across a batch would make
coefficient-wise bounds, and therefore Kharitonov, meaningless:

```python
from ltspice_mc import MonteCarloEngine, make_fitted_extractor
from ltspice_mc.visualization import plot_order_selection

engine = MonteCarloEngine("circuit.asc")
engine.add_parameter("R1", ...); engine.add_parameter("C1", ...)  # etc.

order = engine.determine_transfer_function_order("V(out)", "V(in)", max_order=6, criterion="bic")
print(order.summary())
plot_order_selection(order).savefig("order_selection.png")  # inspect before trusting it

extract_den = make_fitted_extractor("V(out)", "V(in)", order.best.num_order, order.best.den_order)
results = engine.run(n_samples=500, extract_den_fn=extract_den)   # empirical MC stability
print(results.stability_analysis().summary())                     # MC + Kharitonov, clearly labeled

corner_results = engine.run_corner_sweep(corner_values={...}, extract_den_fn=extract_den)  # same fixed order
print(corner_results.stability_analysis().summary())              # corner + Kharitonov, clearly labeled
```

Blind order selection from frequency-response data is a genuinely **harder, less reliable** problem
than knowing your circuit's topology — this is the fallback for when you don't know it, not a reason to
stop checking `order_selection.png` when it matters. `MonteCarloResults.stability_analysis()` and
`CornerSweepResult.stability_analysis()` both return the same `StabilityAnalysis` type, deliberately
kept separate from a single collapsed verdict — see `stability.py`'s module docstring for exactly what
each of the three checks (empirical MC / exhaustive corner / conservative Kharitonov) does and does not
prove. `examples/run_my_circuit.py`'s `CONFIG` block is a topology-agnostic template covering all three.

## Module map

| File | Responsibility |
|---|---|
| `param_utils.py` | Regex-based `.param` discovery/parsing — no spicelib needed, works on plain text |
| `netlist_parameterizer.py` | Auto-detects R/C/L components, prompts interactively, rewrites the netlist/schematic to be `.param`-driven |
| `distributions.py` | `Gaussian`, `Uniform`, `Tolerance`, `Custom` samplers + rejection sampling for bounds |
| `metrics.py` | RMS, mean, peak-to-peak, overshoot, settling time, gain, phase, bandwidth, phase margin |
| `relationships.py` | E-series (E6/E12/E24/E96) snapping + explicit component-to-component relationship equations |
| `optimization.py` | `CircuitOptimizer` — scipy.optimize-driven search over LTspice runs |
| `stability.py` | Transfer-function extraction (analytical + least-squares fit) and Kharitonov robust stability |
| `runner.py` | spicelib `SimRunner`/`AscEditor`/`SpiceEditor` wrapper — the only module requiring spicelib |
| `engine.py` | `MonteCarloEngine` — sampling/sweep/corner grid → parallel runs → callback → NumPy aggregation |
| `visualization.py` | Histograms, PDF/CDF, dashboard, contour/surface plots, s-plane stability plot |

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
