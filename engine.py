"""
engine.py
----------
Top-level orchestration: wires together parameter discovery, Monte Carlo
sampling, the LTspice runner, and metric extraction, and aggregates
everything into in-memory NumPy arrays -- never per-run files -- plus a
lightweight results/summary object.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np

from .distributions import Distribution
from .param_utils import DiscoveredParameter, discover_parameters

_logger = logging.getLogger("ltspice_mc.engine")

MetricFn = Callable[..., float]


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

    @property
    def n_success(self) -> int:
        return int(self.success.sum())

    @property
    def yield_pct(self) -> float:
        return 100.0 * self.n_success / self.n_samples if self.n_samples else float("nan")

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

    # -- execution ------------------------------------------------------

    def run(
        self,
        n_samples: int,
        seed: Optional[int] = None,
        cleanup_files: bool = True,
        progress: bool = True,
        raw_traces_to_read: Optional[List[str]] = None,
    ) -> MonteCarloResults:
        """
        Draw `n_samples` from every registered parameter distribution, run
        one LTspice simulation per sample (in parallel, per `parallel_sims`),
        compute every registered metric from each result as soon as it's
        ready, and return everything aggregated as NumPy arrays in memory.
        """
        if not self._param_specs:
            raise ValueError("No parameters registered -- call add_parameter() at least once.")
        if not self._metric_specs:
            raise ValueError("No metrics registered -- call add_metric() at least once.")
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
        lock = threading.Lock()
        completed = [0]
        start_time = time.monotonic()
        progress_step = max(1, n_samples // 20)

        def make_callback(index: int):
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
                    except Exception as exc:  # noqa: BLE001 -- deliberately broad:
                        # one bad sample (bad convergence, missing node, etc.)
                        # must not crash the rest of the batch.
                        error_msg = f"{type(exc).__name__}: {exc}"

                with lock:
                    if result_ok:
                        for m_name, v in values.items():
                            metric_arrays[m_name][index] = v
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
            runner.run_case(values_i, callback=make_callback(i))

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
        )
