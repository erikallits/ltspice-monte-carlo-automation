"""
optimization.py
------------------
Runs LTspice iteratively, driven by scipy.optimize, to find component
values that hit a target performance metric -- respecting E-series
constraints and any Relationship equations that tie components together.

Needs scipy (lazily imported, only when .minimize()/.maximize()/.target()
is actually called) and, for real use, a local LTspice + spicelib install
(see runner.py). The control flow -- bounds, E-series snapping,
relationship resolution, and the scipy.optimize wiring itself -- is fully
unit-testable with a synthetic stand-in for "run LTspice and extract a
metric" (see tests/test_relationships_optimization.py), which exercises
real scipy.optimize.minimize() against a known analytic objective. Only the
runner<->spicelib boundary itself needs real LTspice to verify.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .relationships import Relationship, resolve_with_relationships, snap_to_series

_logger = logging.getLogger("ltspice_mc.optimization")

MetricFn = Callable[[Any], float]  # same shape as MonteCarloEngine metrics: fn(raw) -> float


def _default_initial_simplex(x0: np.ndarray, bounds: List[Tuple[float, float]],
                              relative_step: float = 0.15) -> np.ndarray:
    """
    scipy's own default Nelder-Mead initial-simplex step (~5% of x0) can be
    SMALLER than the spacing between adjacent E-series grid values (E24 is
    ~10% apart, but E96 is only ~2.4% apart, and 5% sits uncomfortably close
    to both). If every vertex of the starting simplex snaps to the same
    E-series value, Nelder-Mead sees zero variation in its very first
    evaluations and reports false convergence without ever actually moving
    -- which is exactly what happened before this existed. An explicit,
    larger initial step avoids that regardless of whether E-series is used.
    """
    n = len(x0)
    simplex = np.tile(np.asarray(x0, dtype=float), (n + 1, 1))
    for i in range(n):
        lo, hi = bounds[i]
        step = abs(x0[i]) * relative_step if x0[i] != 0 else max(relative_step, (hi - lo) * relative_step)
        candidate = x0[i] + step
        if candidate > hi:
            candidate = x0[i] - step  # bounced off the upper bound -- step the other way instead
        simplex[i + 1, i] = float(np.clip(candidate, lo, hi))
    return simplex


@dataclass
class OptimizationVariable:
    """One free .param the optimizer is allowed to move."""
    name: str
    bounds: Tuple[float, float]
    series: Optional[str] = None   # e.g. "E24", "E96" -- snapped to the nearest standard value every evaluation
    x0: Optional[float] = None     # initial guess; defaults to the bounds midpoint

    def __post_init__(self):
        if self.bounds[0] > self.bounds[1]:
            raise ValueError(f"bounds[0] ({self.bounds[0]}) > bounds[1] ({self.bounds[1]}) for '{self.name}'")

    def initial_value(self) -> float:
        return self.x0 if self.x0 is not None else 0.5 * (self.bounds[0] + self.bounds[1])


@dataclass
class OptimizationStep:
    """One evaluated point, kept for convergence inspection/plotting."""
    free_values: Dict[str, float]
    resolved_values: Dict[str, float]  # free values + everything derived via Relationships
    metric_value: float
    objective_value: float


@dataclass
class OptimizationResult:
    success: bool
    message: str
    free_values: Dict[str, float]       # best free-variable values found (post E-series snapping)
    resolved_values: Dict[str, float]   # best full parameter set, including derived ones
    metric_value: float
    n_evaluations: int
    history: List[OptimizationStep] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Optimization {'succeeded' if self.success else 'did not fully converge'}: {self.message}",
            f"  Evaluations: {self.n_evaluations}",
            f"  Best metric value: {self.metric_value:.6g}",
        ]
        lines += [f"  {k} = {v:.6g}" for k, v in self.resolved_values.items()]
        return "\n".join(lines)


class CircuitOptimizer:
    """
    Wraps LTspiceRunner in a sequential, scipy.optimize-driven loop: each
    objective-function call runs exactly ONE LTspice simulation at a
    specific point (sequential by nature, since each point depends on where
    the last one landed) and returns one scalar. Free variables are
    E-series snapped (if requested), then any Relationship equations are
    resolved to fill in dependent parameters, all *before* the simulation
    runs -- so LTspice only ever sees the fully resolved, buildable
    parameter set.
    """

    def __init__(
        self,
        schematic_path: Union[str, Path],
        free_variables: List[OptimizationVariable],
        relationships: Optional[List[Relationship]] = None,
        output_folder: Union[str, Path] = "./ltspice_mc_temp",
        ltspice_exe: Optional[str] = None,
        sim_timeout: float = 600.0,
        extra_instructions: Optional[List[str]] = None,
    ):
        self.schematic_path = Path(schematic_path)
        if not self.schematic_path.exists():
            raise FileNotFoundError(f"Schematic/netlist not found: {self.schematic_path}")
        if not free_variables:
            raise ValueError("Provide at least one OptimizationVariable")
        names = [v.name for v in free_variables]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate free-variable names: {names}")
        dependent_names = {r.dependent for r in (relationships or [])}
        overlap = dependent_names & set(names)
        if overlap:
            raise ValueError(f"{overlap} cannot be both a free variable AND a Relationship's dependent")

        self.free_variables = free_variables
        self.relationships = relationships or []
        self._output_folder = output_folder
        self._ltspice_exe = ltspice_exe
        self._sim_timeout = sim_timeout
        self._extra_instructions = extra_instructions
        self._runner = None  # created lazily -- spicelib only required when actually simulating
        self.history: List[OptimizationStep] = []

    def _get_runner(self):
        if self._runner is None:
            from .runner import LTspiceRunner
            self._runner = LTspiceRunner(
                schematic_path=self.schematic_path, output_folder=self._output_folder,
                parallel_sims=1, ltspice_exe=self._ltspice_exe, sim_timeout=self._sim_timeout,
                extra_instructions=self._extra_instructions,
            )
        return self._runner

    def _read_raw(self, raw_path: str):
        """Isolated seam (same pattern as MonteCarloEngine._read_raw) so
        tests can substitute a synthetic result without spicelib installed."""
        from spicelib import RawRead
        return RawRead(raw_path, traces_to_read="*")

    def resolve(self, x: np.ndarray) -> Dict[str, float]:
        """
        E-series-snaps every free variable, then resolves all Relationships,
        in order: free -> snap -> relationships. Exposed publicly and takes
        no simulation to run, so "what would we build" can be inspected or
        tested independent of LTspice.
        """
        free_values = {}
        for var, xi in zip(self.free_variables, x):
            v = float(np.clip(xi, var.bounds[0], var.bounds[1]))
            if var.series:
                v = snap_to_series(v, var.series)
            free_values[var.name] = v
        return resolve_with_relationships(free_values, self.relationships)

    def evaluate(self, x: np.ndarray, metric_fn: MetricFn) -> Tuple[Dict[str, float], float]:
        """Resolves x to a full parameter set, runs ONE simulation, and
        returns (resolved_values, metric_value)."""
        resolved = self.resolve(x)
        raw_path, _log_path = self._get_runner().run_case_blocking(resolved)
        if raw_path is None:
            raise RuntimeError(f"Simulation aborted for parameters {resolved} -- no .raw file produced")
        raw = self._read_raw(raw_path)
        return resolved, float(metric_fn(raw))

    def minimize(
        self,
        metric_fn: MetricFn,
        objective: Optional[Callable[[float], float]] = None,
        method: str = "Nelder-Mead",
        maxiter: int = 100,
        **scipy_kwargs,
    ) -> OptimizationResult:
        """
        Finds free-variable values minimizing `objective(metric_value)`.
        `objective` defaults to the identity (minimize the metric directly).

        Nelder-Mead is the default because E-series snapping makes the
        objective piecewise-constant with jumps -- exactly the kind of
        rugged, non-smooth landscape derivative-free simplex search handles
        far better than gradient-based methods, which can get stuck reading
        a zero gradient on a flat snapped plateau. Pass `series=None` on
        your variables and a gradient-based `method=` if you'd rather
        optimize on a continuous landscape.
        """
        from scipy.optimize import minimize as scipy_minimize

        objective = objective if objective is not None else (lambda m: m)
        self.history = []

        def scipy_objective(x: np.ndarray) -> float:
            resolved, metric_value = self.evaluate(x, metric_fn)
            obj_value = float(objective(metric_value))
            self.history.append(OptimizationStep(
                free_values={v.name: resolved[v.name] for v in self.free_variables},
                resolved_values=dict(resolved), metric_value=metric_value, objective_value=obj_value,
            ))
            return obj_value

        x0 = np.array([v.initial_value() for v in self.free_variables])
        bounds = [v.bounds for v in self.free_variables]

        user_options = dict(scipy_kwargs.pop("options", {}))
        options: Dict[str, Any] = {"maxiter": maxiter}
        if method == "Nelder-Mead" and "initial_simplex" not in user_options:
            options["initial_simplex"] = _default_initial_simplex(x0, bounds)
        options.update(user_options)

        result = scipy_minimize(scipy_objective, x0, method=method, bounds=bounds,
                                 options=options, **scipy_kwargs)

        # The best point EVER evaluated, not necessarily scipy's final x --
        # some methods' last internal step isn't the best one they found,
        # and re-running LTspice just to double check would waste a sim.
        best = min(self.history, key=lambda s: s.objective_value) if self.history else None
        return OptimizationResult(
            success=bool(result.success), message=str(result.message),
            free_values=best.free_values if best else {},
            resolved_values=best.resolved_values if best else {},
            metric_value=best.metric_value if best else float("nan"),
            n_evaluations=len(self.history), history=list(self.history),
        )

    def maximize(self, metric_fn: MetricFn, **kwargs) -> OptimizationResult:
        """Convenience wrapper: maximize the metric directly (e.g. bandwidth)."""
        return self.minimize(metric_fn, objective=lambda m: -m, **kwargs)

    def target(self, metric_fn: MetricFn, target_value: float, **kwargs) -> OptimizationResult:
        """Convenience wrapper: drive the metric as close as possible to
        target_value (e.g. a specific phase margin)."""
        return self.minimize(metric_fn, objective=lambda m: abs(m - target_value), **kwargs)
