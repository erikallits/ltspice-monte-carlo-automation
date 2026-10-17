"""
engine.py
----------
Top-level orchestration: wires together parameter discovery, Monte Carlo
sampling, the LTspice runner, and metric extraction, and aggregates
everything into in-memory NumPy arrays -- never per-run files -- plus a
lightweight results/summary object.
"""

from __future__ import annotations

import itertools
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np

from .distributions import Distribution, Gaussian, Tolerance, Uniform
from .metrics import get_axis, get_signal
from .param_utils import DiscoveredParameter, discover_parameters
from .relationships import Relationship, resolve_with_relationships
from .stability import CornerRun, CornerSweepResult

_logger = logging.getLogger("ltspice_mc.engine")

MetricFn = Callable[..., float]


def _distribution_center(dist: Distribution, name: str) -> float:
    """Best single 'hold this constant at' value for a parameter not being
    swept: mean for Gaussian, midpoint for Uniform, nominal for Tolerance."""
    if isinstance(dist, Tolerance):
        return dist.nominal
    if isinstance(dist, Gaussian):
        return dist.mean
    if isinstance(dist, Uniform):
        return 0.5 * (dist.low + dist.high)
    raise TypeError(
        f"Can't infer a center value for '{name}' ({type(dist).__name__}); "
        "pass its value explicitly via sweep_2d(..., fixed_params={...})."
    )


@dataclass
class ParamSpec:
    name: str
    distribution: Distribution


@dataclass
class MetricSpec:
    name: str
    function: MetricFn


@dataclass
class MonteCarloResults:
    """All Monte Carlo output, held entirely in memory as NumPy arrays."""

    n_samples: int
    parameters: Dict[str, np.ndarray]
    metrics: Dict[str, np.ndarray]
    success: np.ndarray
    errors: List[Optional[str]]
    den_coeffs: List[Optional[np.ndarray]] = field(default_factory=list)  # only populated if run(extract_den_fn=...)

    @property
    def n_success(self) -> int:
        return int(self.success.sum())

    @property
    def yield_pct(self) -> float:
        return 100.0 * self.n_success / self.n_samples if self.n_samples else float("nan")

    def stacked_den_coeffs(self) -> np.ndarray:
        """(n_successful_samples, n_coeffs) array, highest power first --
        requires run() to have been called with extract_den_fn=..."""
        from .stability import _stack_coeffs
        return _stack_coeffs(self.den_coeffs, self.success)

    def coefficient_bounds(self):
        """[(a_i_min, a_i_max), ...] across successful samples, LOWEST power first."""
        stacked = self.stacked_den_coeffs()
        lowest_first = stacked[:, ::-1]
        return list(zip(lowest_first.min(axis=0).tolist(), lowest_first.max(axis=0).tolist()))

    def stability_analysis(self):
        """The full empirical-MC + Kharitonov stability picture -- see
        stability.py's module docstring for what each part does and does
        NOT tell you. Requires run() to have been called with extract_den_fn=..."""
        from .stability import build_stability_analysis
        return build_stability_analysis("monte_carlo", self.den_coeffs, list(self.success))

    def summary(self) -> str:
        lines = [
            f"Monte Carlo results: {self.n_success}/{self.n_samples} runs succeeded "
            f"({self.yield_pct:.1f}%)."
        ]
        for name, values in self.metrics.items():
            valid = values[self.success & ~np.isnan(values)]
            if len(valid) == 0:
                lines.append(f"  {name}: no valid samples")
                continue
            lines.append(
                f"  {name}: mean={np.mean(valid):.6g}  std={np.std(valid):.6g}  "
                f"min={np.min(valid):.6g}  max={np.max(valid):.6g}  n={len(valid)}"
            )
        if any(e is not None for e in self.errors):
            n_err = sum(1 for e in self.errors if e is not None)
            lines.append(f"  ({n_err} run(s) failed -- see .errors for details)")
        return "\n".join(lines)

    def compute_yield(self, metric: str, lower: Optional[float] = None,
                       upper: Optional[float] = None) -> float:
        """Percentage of *successful* runs whose metric falls within
        [lower, upper] (either bound may be omitted for a one-sided spec)."""
        values = self.metrics[metric]
        mask = self.success & ~np.isnan(values)
        valid = values[mask]
        if len(valid) == 0:
            return float("nan")
        in_spec = np.ones_like(valid, dtype=bool)
        if lower is not None:
            in_spec &= valid >= lower
        if upper is not None:
            in_spec &= valid <= upper
        return 100.0 * in_spec.sum() / len(valid)


@dataclass
class SweepResult:
    """Output of MonteCarloEngine.sweep_2d(): a metric evaluated over every
    (param_x, param_y) grid point, held in memory as 2D NumPy arrays shaped
    (len(values_y), len(values_x)) -- matching numpy.meshgrid/contourf's
    convention, so results plug directly into visualization.plot_contour()."""

    param_x: str
    values_x: np.ndarray
    param_y: str
    values_y: np.ndarray
    metrics: Dict[str, np.ndarray]
    success: np.ndarray
    errors: List[List[Optional[str]]]

    @property
    def n_points(self) -> int:
        return int(self.success.size)

    @property
    def n_success(self) -> int:
        return int(self.success.sum())

    def best(self, metric: str, mode: str = "max") -> Dict[str, float]:
        """Grid point (param_x, param_y, and the metric's own value) that
        maximizes/minimizes `metric` among successful points."""
        if mode not in ("max", "min"):
            raise ValueError("mode must be 'max' or 'min'")
        values = np.where(self.success, self.metrics[metric], np.nan)
        if np.all(np.isnan(values)):
            raise ValueError(f"No successful grid points for metric '{metric}'")
        idx = np.nanargmax(values) if mode == "max" else np.nanargmin(values)
        iy, ix = np.unravel_index(idx, values.shape)
        return {self.param_x: float(self.values_x[ix]), self.param_y: float(self.values_y[iy]),
                metric: float(values[iy, ix])}


class MonteCarloEngine:
    """
    Circuit-agnostic Monte Carlo driver. Works with ANY LTspice `.asc`
    schematic (or plain SPICE netlist) that exposes its variability through
    `.param` statements -- nothing here ever references a component
    designator like R1 or C1.

    Typical usage:
        engine = MonteCarloEngine("my_circuit.asc", parallel_sims=8)
        engine.add_parameter("R", Gaussian(mean=1000, std=10))
        engine.add_metric("vout_rms", lambda raw: compute_rms(raw, "V(out)"))
        results = engine.run(n_samples=500, seed=42)
    """

    def __init__(
        self,
        schematic_path: Union[str, Path],
        output_folder: Union[str, Path] = "./ltspice_mc_temp",
        parallel_sims: int = 4,
        ltspice_exe: Optional[str] = None,
        sim_timeout: float = 600.0,
        extra_instructions: Optional[List[str]] = None,
    ):
        self.schematic_path = Path(schematic_path)
        if not self.schematic_path.exists():
            raise FileNotFoundError(f"Schematic/netlist not found: {self.schematic_path}")

        self.discovered_parameters: Dict[str, DiscoveredParameter] = discover_parameters(
            self.schematic_path
        )
        self._param_specs: Dict[str, ParamSpec] = {}
        self._metric_specs: Dict[str, MetricSpec] = {}
        self._output_folder = output_folder
        self._parallel_sims = parallel_sims
        self._ltspice_exe = ltspice_exe
        self._sim_timeout = sim_timeout
        self._extra_instructions = extra_instructions
        self._runner = None  # created lazily -- spicelib only required at run() time

    # -- configuration ------------------------------------------------------

    def add_parameter(self, name: str, distribution: Distribution,
                       warn_if_undeclared: bool = True) -> "MonteCarloEngine":
        """Register a `.param` name to be varied according to `distribution`."""
        if warn_if_undeclared and name not in self.discovered_parameters:
            _logger.warning(
                "Parameter '%s' was not found among the .param statements "
                "discovered in %s (found: %s). It will still be sent to "
                "LTspice via set_parameters(), which is fine if it's meant "
                "to be newly introduced -- just double check the spelling "
                "if that wasn't intentional.",
                name, self.schematic_path.name, sorted(self.discovered_parameters),
            )
        self._param_specs[name] = ParamSpec(name, distribution)
        return self

    def add_metric(self, name: str, function: MetricFn) -> "MonteCarloEngine":
        """
        Register a derived KPI. `function` receives one argument -- the
        RawRead object for a single completed run -- and returns a float:
            engine.add_metric("vout_rms", lambda raw: compute_rms(raw, "V(out)"))
        """
        self._metric_specs[name] = MetricSpec(name, function)
        return self

    def enable_temperature_control(self, param_name: str = "TEMP_C") -> "MonteCarloEngine":
        """
        Makes LTspice's simulation temperature a normal, Python-controlled
        `.param`, by queuing a `.temp {param_name}` instruction -- after
        this, register temperature exactly like any other parameter:
            engine.enable_temperature_control()
            engine.add_parameter("TEMP_C", Uniform(-40, 125), warn_if_undeclared=False)
        Must be called before the first run()/sweep_2d()/run_corner_sweep()
        (i.e. before the underlying LTspiceRunner is created), since the
        instruction has to be queued before spicelib builds the netlist.
        """
        if self._runner is not None:
            raise RuntimeError(
                "enable_temperature_control() must be called before the first run(), "
                "sweep_2d(), or run_corner_sweep() -- the simulator is already running "
                "with whatever instructions were configured at that point."
            )
        instruction = f".temp {{{param_name}}}"
        if self._extra_instructions is None:
            self._extra_instructions = [instruction]
        elif instruction not in self._extra_instructions:
            self._extra_instructions.append(instruction)
        return self

    def _get_runner(self):
        if self._runner is None:
            from .runner import LTspiceRunner
            self._runner = LTspiceRunner(
                schematic_path=self.schematic_path,
                output_folder=self._output_folder,
                parallel_sims=self._parallel_sims,
                ltspice_exe=self._ltspice_exe,
                sim_timeout=self._sim_timeout,
                extra_instructions=self._extra_instructions,
            )
        return self._runner

    def _read_raw(self, raw_path: str, traces_to_read: Optional[List[str]]):
        """
        Isolated in its own method so tests can monkeypatch just this one
        seam (returning a mock RawRead-like object) to exercise the full
        sampling/callback/aggregation pipeline without needing spicelib or
        LTspice installed at all. See tests/test_framework.py.
        """
        from spicelib import RawRead
        return RawRead(raw_path, traces_to_read=traces_to_read or "*")

    def determine_transfer_function_order(
        self,
        out_signal: str,
        in_signal: Optional[str] = None,
        nominal_params: Optional[Dict[str, float]] = None,
        max_order: int = 6,
        criterion: str = "bic",
    ):
        """
        Runs ONE simulation at `nominal_params` (defaults to the center of
        every registered parameter's distribution, per add_parameter()) and
        auto-selects the transfer function's (num_order, den_order) from it
        via stability.auto_fit_transfer_function().

        Call this ONCE, before a batch run()/run_corner_sweep(), and reuse
        the result's fixed order for every sample -- via
        stability.make_fitted_extractor(out_signal, in_signal,
        result.best.num_order, result.best.den_order) -- rather than
        re-running order selection per sample. See stability.py's module
        docstring for why a consistent order across the whole batch matters.
        """
        from .stability import auto_fit_transfer_function

        if nominal_params is None:
            nominal_params = {
                name: _distribution_center(spec.distribution, name)
                for name, spec in self._param_specs.items()
            }
        raw_path, _log_path = self._get_runner().run_case_blocking(nominal_params)
        if raw_path is None:
            raise RuntimeError(f"Simulation aborted at nominal_params={nominal_params} -- no .raw file produced")
        raw = self._read_raw(raw_path, None)
        freq = get_axis(raw)
        h = get_signal(raw, out_signal) / get_signal(raw, in_signal) if in_signal else get_signal(raw, out_signal)
        return auto_fit_transfer_function(freq, h, max_order=max_order, criterion=criterion)

    # -- execution ------------------------------------------------------

    def run(
        self,
        n_samples: int,
        seed: Optional[int] = None,
        cleanup_files: bool = True,
        progress: bool = True,
        raw_traces_to_read: Optional[List[str]] = None,
        extract_den_fn: Optional[Callable[[Any, Dict[str, float]], np.ndarray]] = None,
    ) -> MonteCarloResults:
        """
        Draw `n_samples` from every registered parameter distribution, run
        one LTspice simulation per sample (in parallel, per `parallel_sims`),
        compute every registered metric from each result as soon as it's
        ready, and return everything aggregated as NumPy arrays in memory.

        `extract_den_fn(raw, resolved_params) -> array`, if given, ALSO
        extracts each sample's D(s) coefficients (highest power first) for
        empirical stability analysis (see MonteCarloResults.stability_analysis()
        and stability.py's module docstring) -- same signature as
        run_corner_sweep()'s extract_den_fn, and same warning: every sample
        must use a FIXED order (e.g. via make_fitted_extractor() built from
        a prior determine_transfer_function_order() call), not a fresh
        auto_fit_transfer_function() per sample.
        """
        if not self._param_specs:
            raise ValueError("No parameters registered -- call add_parameter() at least once.")
        if not self._metric_specs and extract_den_fn is None:
            raise ValueError(
                "Nothing to measure -- call add_metric() at least once, and/or pass extract_den_fn "
                "for transfer-function/stability analysis."
            )
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")

        rng = np.random.default_rng(seed)
        param_samples = {
            name: spec.distribution.sample(n_samples, rng)
            for name, spec in self._param_specs.items()
        }

        metric_arrays = {name: np.full(n_samples, np.nan) for name in self._metric_specs}
        success = np.zeros(n_samples, dtype=bool)
        errors: List[Optional[str]] = [None] * n_samples
        den_coeffs: List[Optional[np.ndarray]] = [None] * n_samples
        lock = threading.Lock()
        completed = [0]
        start_time = time.monotonic()
        progress_step = max(1, n_samples // 20)

        def make_callback(index: int, resolved: Dict[str, float]):
            def callback(raw_path: Optional[str], log_path: Optional[str]) -> None:
                result_ok = False
                error_msg = None
                values: Dict[str, float] = {}
                den: Optional[np.ndarray] = None

                if raw_path is None:
                    error_msg = "simulation aborted -- no .raw file produced"
                elif not Path(raw_path).exists():
                    error_msg = f"missing .raw file: {raw_path}"
                else:
                    try:
                        raw = self._read_raw(raw_path, raw_traces_to_read)
                        for m_name, m_spec in self._metric_specs.items():
                            values[m_name] = float(m_spec.function(raw))
                        if extract_den_fn is not None:
                            den = np.asarray(extract_den_fn(raw, resolved), dtype=float)
                        result_ok = True
                    except Exception as exc:  # noqa: BLE001 -- deliberately broad:
                        # one bad sample (bad convergence, missing node, etc.)
                        # must not crash the rest of the batch.
                        error_msg = f"{type(exc).__name__}: {exc}"

                with lock:
                    if result_ok:
                        for m_name, v in values.items():
                            metric_arrays[m_name][index] = v
                        den_coeffs[index] = den
                        success[index] = True
                    else:
                        errors[index] = error_msg
                        _logger.warning("Run %d failed: %s", index, error_msg)
                    completed[0] += 1
                    if progress and (completed[0] % progress_step == 0 or completed[0] == n_samples):
                        elapsed = time.monotonic() - start_time
                        _logger.info(
                            "Progress: %d/%d runs (%.0f%%), %d ok, %.1fs elapsed",
                            completed[0], n_samples, 100 * completed[0] / n_samples,
                            int(success.sum()), elapsed,
                        )

                if cleanup_files:
                    for p in (raw_path, log_path):
                        if p:
                            try:
                                Path(p).unlink(missing_ok=True)
                            except OSError:
                                pass

            return callback

        runner = self._get_runner()
        for i in range(n_samples):
            values_i = {name: float(arr[i]) for name, arr in param_samples.items()}
            runner.run_case(values_i, callback=make_callback(i, values_i))

        all_ok = runner.wait_all()
        if not all_ok:
            _logger.warning(
                "Not all simulations completed successfully (%d/%d ok). "
                "See MonteCarloResults.errors for per-run detail.",
                runner.n_ok, runner.n_total,
            )
        if cleanup_files:
            runner.cleanup()

        return MonteCarloResults(
            n_samples=n_samples,
            parameters=param_samples,
            metrics=metric_arrays,
            success=success,
            errors=errors,
            den_coeffs=den_coeffs,
        )

    # -- trade-off analysis --------------------------------------------

    def sweep_2d(
        self,
        param_x: str,
        values_x: Sequence[float],
        param_y: str,
        values_y: Sequence[float],
        fixed_params: Optional[Dict[str, float]] = None,
        relationships: Optional[List[Relationship]] = None,
        cleanup_files: bool = True,
        progress: bool = True,
        raw_traces_to_read: Optional[List[str]] = None,
    ) -> SweepResult:
        """
        Runs one LTspice simulation at every (param_x, param_y) grid point
        (in parallel, per parallel_sims), evaluating every metric already
        registered via add_metric() at each point -- this is the "sweet
        spot" trade-off map: feed the result to
        ltspice_mc.visualization.plot_contour()/plot_surface_3d().

        Every OTHER registered parameter is held fixed while sweeping: at
        the value given in `fixed_params` if present, else at its own
        distribution's mean/midpoint/nominal. `relationships`, if given, are
        resolved at every grid point AFTER param_x/param_y/fixed_params are
        set -- e.g. to keep a third, dependent component consistent with
        whichever pair you're actually sweeting.
        """
        if not self._metric_specs:
            raise ValueError("No metrics registered -- call add_metric() at least once.")

        values_x = np.asarray(values_x, dtype=float)
        values_y = np.asarray(values_y, dtype=float)
        if values_x.ndim != 1 or values_y.ndim != 1 or len(values_x) == 0 or len(values_y) == 0:
            raise ValueError("values_x/values_y must be non-empty 1D sequences")
        nx, ny = len(values_x), len(values_y)
        grid_shape = (ny, nx)

        dependent_names = {r.dependent for r in (relationships or [])}
        fixed = dict(fixed_params or {})
        for name, spec in self._param_specs.items():
            if name in (param_x, param_y) or name in fixed or name in dependent_names:
                continue
            fixed[name] = _distribution_center(spec.distribution, name)

        metric_arrays = {name: np.full(grid_shape, np.nan) for name in self._metric_specs}
        success = np.zeros(grid_shape, dtype=bool)
        errors: List[List[Optional[str]]] = [[None] * nx for _ in range(ny)]
        lock = threading.Lock()
        completed = [0]
        total = nx * ny
        progress_step = max(1, total // 20)
        start_time = time.monotonic()

        def make_callback(iy: int, ix: int, resolved: Dict[str, float]):
            def callback(raw_path: Optional[str], log_path: Optional[str]) -> None:
                result_ok = False
                error_msg = None
                values: Dict[str, float] = {}

                if raw_path is None:
                    error_msg = "simulation aborted -- no .raw file produced"
                elif not Path(raw_path).exists():
                    error_msg = f"missing .raw file: {raw_path}"
                else:
                    try:
                        raw = self._read_raw(raw_path, raw_traces_to_read)
                        for m_name, m_spec in self._metric_specs.items():
                            values[m_name] = float(m_spec.function(raw))
                        result_ok = True
                    except Exception as exc:  # noqa: BLE001 -- one bad grid point must not kill the sweep
                        error_msg = f"{type(exc).__name__}: {exc}"

                with lock:
                    if result_ok:
                        for m_name, v in values.items():
                            metric_arrays[m_name][iy, ix] = v
                        success[iy, ix] = True
                    else:
                        errors[iy][ix] = error_msg
                        _logger.warning(
                            "Sweep point (%s=%.6g, %s=%.6g) failed: %s",
                            param_x, resolved[param_x], param_y, resolved[param_y], error_msg,
                        )
                    completed[0] += 1
                    if progress and (completed[0] % progress_step == 0 or completed[0] == total):
                        elapsed = time.monotonic() - start_time
                        _logger.info(
                            "Sweep progress: %d/%d points (%.0f%%), %d ok, %.1fs elapsed",
                            completed[0], total, 100 * completed[0] / total, int(success.sum()), elapsed,
                        )

                if cleanup_files:
                    for p in (raw_path, log_path):
                        if p:
                            try:
                                Path(p).unlink(missing_ok=True)
                            except OSError:
                                pass

            return callback

        runner = self._get_runner()
        for iy, y in enumerate(values_y):
            for ix, x in enumerate(values_x):
                point = dict(fixed)
                point[param_x] = float(x)
                point[param_y] = float(y)
                resolved = resolve_with_relationships(point, relationships)
                runner.run_case(resolved, callback=make_callback(iy, ix, resolved))

        runner.wait_all()
        if cleanup_files:
            runner.cleanup()

        return SweepResult(
            param_x=param_x, values_x=values_x, param_y=param_y, values_y=values_y,
            metrics=metric_arrays, success=success, errors=errors,
        )

    def run_corner_sweep(
        self,
        corner_values: Dict[str, Sequence[float]],
        extract_den_fn: Callable[[Any, Dict[str, float]], np.ndarray],
        fixed_params: Optional[Dict[str, float]] = None,
        relationships: Optional[List[Relationship]] = None,
        cleanup_files: bool = True,
        progress: bool = True,
        raw_traces_to_read: Optional[List[str]] = None,
    ) -> CornerSweepResult:
        """
        Runs one LTspice simulation for every combination in the Cartesian
        product of corner_values -- e.g.
            {"R1": [990, 1000, 1010], "C1": [90e-9, 100e-9], "TEMP_C": [-40, 27, 125]}
        runs all 3*2*3 = 18 combinations (call enable_temperature_control()
        first so "TEMP_C" -- or whatever name you used -- actually reaches
        LTspice). Every other registered/discovered parameter is held at
        fixed_params, or its distribution's center if not given there.

        `extract_den_fn(raw, resolved_params) -> array` computes each run's
        D(s) coefficients (highest power first); see stability.py for
        ready-made analytical and least-squares-fit implementations.
        `relationships` are resolved the same way as in sweep_2d(), so a
        dependent component can be derived rather than swept directly.

        Feed the result to stability.check_kharitonov_stability(
        result.coefficient_bounds()) and stability.check_all_runs_stable(
        result.stacked_den_coeffs()) for the two complementary stability
        checks described in stability.py's module docstring.
        """
        if not corner_values:
            raise ValueError("corner_values must have at least one parameter")

        names = list(corner_values.keys())
        value_lists = [list(v) for v in corner_values.values()]
        combos = list(itertools.product(*value_lists))
        n = len(combos)

        fixed = dict(fixed_params or {})
        for name, spec in self._param_specs.items():
            if name in corner_values or name in fixed:
                continue
            fixed[name] = _distribution_center(spec.distribution, name)

        runs: List[Optional[CornerRun]] = [None] * n
        lock = threading.Lock()
        completed = [0]
        progress_step = max(1, n // 20)
        start_time = time.monotonic()

        def make_callback(index: int, resolved: Dict[str, float]):
            def callback(raw_path: Optional[str], log_path: Optional[str]) -> None:
                success = False
                error: Optional[str] = None
                den: Optional[np.ndarray] = None
                if raw_path is None:
                    error = "simulation aborted -- no .raw file produced"
                elif not Path(raw_path).exists():
                    error = f"missing .raw file: {raw_path}"
                else:
                    try:
                        raw = self._read_raw(raw_path, raw_traces_to_read)
                        den = np.asarray(extract_den_fn(raw, resolved), dtype=float)
                        success = True
                    except Exception as exc:  # noqa: BLE001 -- one bad corner must not kill the sweep
                        error = f"{type(exc).__name__}: {exc}"

                with lock:
                    runs[index] = CornerRun(params=resolved, success=success, error=error, den_coeffs=den)
                    if not success:
                        _logger.warning("Corner run %s failed: %s", resolved, error)
                    completed[0] += 1
                    if progress and (completed[0] % progress_step == 0 or completed[0] == n):
                        elapsed = time.monotonic() - start_time
                        _logger.info("Corner sweep progress: %d/%d (%.0f%%), %.1fs elapsed",
                                     completed[0], n, 100 * completed[0] / n, elapsed)

                if cleanup_files:
                    for p in (raw_path, log_path):
                        if p:
                            try:
                                Path(p).unlink(missing_ok=True)
                            except OSError:
                                pass

            return callback

        runner = self._get_runner()
        for i, combo in enumerate(combos):
            point = dict(fixed)
            point.update(dict(zip(names, combo)))
            resolved = resolve_with_relationships(point, relationships)
            runner.run_case(resolved, callback=make_callback(i, resolved))

        runner.wait_all()
        if cleanup_files:
            runner.cleanup()

        return CornerSweepResult(param_names=names, runs=[r for r in runs if r is not None])


