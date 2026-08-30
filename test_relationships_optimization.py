"""
Tests for relationships.py, optimization.py, and MonteCarloEngine.sweep_2d().
All run without spicelib or LTspice installed:

  * E-series snapping and Relationship resolution are pure math -- tested directly.
  * CircuitOptimizer is tested against a SYNTHETIC objective function (a known
    analytic formula standing in for "the metric LTspice would report"),
    injected via the same monkeypatched-seam pattern used for MonteCarloEngine
    in test_framework.py. This exercises REAL scipy.optimize.minimize() end to
    end through our glue code (bounds, E-series snapping, relationship
    resolution, history tracking) -- only the runner<->spicelib boundary
    itself is left unverified here.
  * sweep_2d() is tested the same way, with a fake parallel runner.
"""

import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ltspice_mc.relationships import (
    E_SERIES, Relationship, hold_cutoff_frequency_constant, hold_rc_product_constant,
    resolve_with_relationships, snap_to_series,
)
from ltspice_mc.optimization import CircuitOptimizer, OptimizationVariable, OptimizationResult
from ltspice_mc.distributions import Gaussian, Uniform, Tolerance
from ltspice_mc.engine import MonteCarloEngine, SweepResult


def _write_dummy_schematic() -> Path:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".cir", delete=False)
    f.write(".param R=1k C=10n\nR1 in out {R}\nC1 out 0 {C}\n.end\n")
    f.close()
    return Path(f.name)


# ===========================================================================
# E-series
# ===========================================================================

class TestESeries(unittest.TestCase):
    def test_series_lengths(self):
        self.assertEqual(len(E_SERIES["E96"]), 96)
        self.assertEqual(len(E_SERIES["E24"]), 24)
        self.assertEqual(len(E_SERIES["E12"]), 12)
        self.assertEqual(len(E_SERIES["E6"]), 6)

    def test_exact_grid_value_unchanged(self):
        self.assertAlmostEqual(snap_to_series(4700, "E24"), 4700)
        self.assertAlmostEqual(snap_to_series(10000, "E96"), 10000)

    def test_rounds_to_nearest(self):
        # 5000 is between E24's 4700 and 5100; closer to 5100? |5000-4700|=300,
        # |5000-5100|=100 -> 5100 wins.
        self.assertAlmostEqual(snap_to_series(5000, "E24"), 5100)

    def test_decade_boundary_crossing(self):
        # 990 in E6: nearest same-decade candidate is 680 (dist 310), but the
        # NEXT decade's 1000 (dist 10) is actually closer -- must cross decades.
        self.assertAlmostEqual(snap_to_series(990, "E6"), 1000)

    def test_preserves_decade_for_large_and_small_values(self):
        self.assertAlmostEqual(snap_to_series(4_600_000, "E24"), 4_700_000)  # nearer 4.7 than 4.3
        self.assertAlmostEqual(snap_to_series(0.00047, "E24"), 0.00047)      # already exact

    def test_case_insensitive_series_name(self):
        self.assertAlmostEqual(snap_to_series(4700, "e24"), 4700)

    def test_invalid_series_raises(self):
        with self.assertRaises(ValueError):
            snap_to_series(1000, "E999")

    def test_nonpositive_value_raises(self):
        with self.assertRaises(ValueError):
            snap_to_series(-100, "E24")
        with self.assertRaises(ValueError):
            snap_to_series(0, "E24")


# ===========================================================================
# Relationships
# ===========================================================================

class TestRelationships(unittest.TestCase):
    def test_resolve_no_relationships_passthrough(self):
        free = {"R": 1000.0}
        self.assertEqual(resolve_with_relationships(free, None), free)
        self.assertEqual(resolve_with_relationships(free, []), free)

    def test_single_relationship(self):
        rel = Relationship(dependent="C", solve=lambda v: 1e-6 / v["R"])
        resolved = resolve_with_relationships({"R": 1000.0}, [rel])
        self.assertAlmostEqual(resolved["C"], 1e-9)
        self.assertAlmostEqual(resolved["R"], 1000.0)

    def test_chained_relationships(self):
        # C derived from R, then D derived from C -- must see the resolved C, not miss it.
        rel_c = Relationship(dependent="C", solve=lambda v: v["R"] * 2)
        rel_d = Relationship(dependent="D", solve=lambda v: v["C"] + 1)
        resolved = resolve_with_relationships({"R": 10.0}, [rel_c, rel_d])
        self.assertAlmostEqual(resolved["C"], 20.0)
        self.assertAlmostEqual(resolved["D"], 21.0)

    def test_hold_rc_product_constant(self):
        solve = hold_rc_product_constant("C", target_product=1e-6)  # tau = R*C = 1us
        self.assertAlmostEqual(solve({"C": 1e-9}), 1000.0)  # R = tau/C = 1e-6/1e-9 = 1000

    def test_hold_cutoff_frequency_constant(self):
        fc = 1591.549431  # ~ R=1k, C=100n
        solve = hold_cutoff_frequency_constant("C", target_fc_hz=fc)
        r = solve({"C": 100e-9})
        self.assertAlmostEqual(r, 1000.0, delta=0.5)

    def test_division_by_zero_raises_clear_error(self):
        solve = hold_rc_product_constant("C", target_product=1e-6)
        with self.assertRaises(ValueError):
            solve({"C": 0.0})


# ===========================================================================
# OptimizationVariable
# ===========================================================================

class TestOptimizationVariable(unittest.TestCase):
    def test_bounds_validation(self):
        with self.assertRaises(ValueError):
            OptimizationVariable(name="R", bounds=(1000, 100))

    def test_default_initial_value_is_midpoint(self):
        v = OptimizationVariable(name="R", bounds=(1000, 3000))
        self.assertAlmostEqual(v.initial_value(), 2000.0)

    def test_explicit_x0_respected(self):
        v = OptimizationVariable(name="R", bounds=(1000, 3000), x0=1200)
        self.assertAlmostEqual(v.initial_value(), 1200.0)


# ===========================================================================
# CircuitOptimizer -- construction and resolve() (no simulation needed)
# ===========================================================================

class TestCircuitOptimizerResolve(unittest.TestCase):
    def setUp(self):
        self.schematic = _write_dummy_schematic()

    def tearDown(self):
        self.schematic.unlink()

    def test_missing_schematic_raises(self):
        with self.assertRaises(FileNotFoundError):
            CircuitOptimizer("does_not_exist.asc", [OptimizationVariable("R", (100, 1000))])

    def test_no_free_variables_raises(self):
        with self.assertRaises(ValueError):
            CircuitOptimizer(self.schematic, [])

    def test_duplicate_names_raise(self):
        with self.assertRaises(ValueError):
            CircuitOptimizer(self.schematic, [
                OptimizationVariable("R", (100, 1000)), OptimizationVariable("R", (1, 2)),
            ])

    def test_free_variable_cannot_also_be_a_dependent(self):
        rel = Relationship(dependent="R", solve=lambda v: 1.0)
        with self.assertRaises(ValueError):
            CircuitOptimizer(self.schematic, [OptimizationVariable("R", (100, 1000))], relationships=[rel])

    def test_resolve_plain_no_relationships(self):
        opt = CircuitOptimizer(self.schematic, [
            OptimizationVariable("R", (100, 10000)), OptimizationVariable("C", (1e-9, 1e-6)),
        ])
        resolved = opt.resolve(np.array([2500.0, 5e-8]))
        self.assertAlmostEqual(resolved["R"], 2500.0)
        self.assertAlmostEqual(resolved["C"], 5e-8)

    def test_resolve_clips_to_bounds(self):
        opt = CircuitOptimizer(self.schematic, [OptimizationVariable("R", (100, 1000))])
        self.assertAlmostEqual(opt.resolve(np.array([5000.0]))["R"], 1000.0)
        self.assertAlmostEqual(opt.resolve(np.array([-5.0]))["R"], 100.0)

    def test_resolve_applies_e_series_before_relationships(self):
        # R is E24-snapped; the relationship must see the SNAPPED value, not the raw one.
        rel = Relationship(dependent="C", solve=lambda v: 1e-6 / v["R"])
        opt = CircuitOptimizer(
            self.schematic, [OptimizationVariable("R", (100, 10000), series="E24")], relationships=[rel],
        )
        resolved = opt.resolve(np.array([5000.0]))  # nearest E24 to 5000 is 5100
        self.assertAlmostEqual(resolved["R"], 5100.0)
        self.assertAlmostEqual(resolved["C"], 1e-6 / 5100.0)


# ===========================================================================
# CircuitOptimizer -- full minimize()/maximize()/target() against a
# synthetic objective, exercising real scipy.optimize
# ===========================================================================

class _FakeBlockingRunner:
    """Stands in for LTspiceRunner.run_case_blocking(): instead of touching
    LTspice, it just remembers the resolved params so the test's fake
    _read_raw() can hand them straight to a synthetic metric function."""

    def __init__(self, tmp_dir: Path):
        self.tmp_dir = tmp_dir
        self.n_calls = 0
        self.last_params = None

    def run_case_blocking(self, param_values):
        self.n_calls += 1
        self.last_params = dict(param_values)
        p = self.tmp_dir / f"opt_run_{self.n_calls}.raw"
        p.touch()
        return str(p), None

    def cleanup(self):
        pass


class TestCircuitOptimizerMinimize(unittest.TestCase):
    def setUp(self):
        self.schematic = _write_dummy_schematic()
        self.tmp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        self.schematic.unlink()

    def _wire_fake_runner(self, optimizer: CircuitOptimizer) -> _FakeBlockingRunner:
        fake = _FakeBlockingRunner(self.tmp_dir)
        optimizer._get_runner = lambda: fake
        optimizer._read_raw = lambda raw_path: dict(fake.last_params)  # "raw" IS the params dict
        return fake

    def test_minimize_single_variable_finds_known_minimum(self):
        opt = CircuitOptimizer(self.schematic, [OptimizationVariable("R", (0, 10000), x0=100)])
        self._wire_fake_runner(opt)

        def metric_fn(params):
            return (params["R"] - 4321.0) ** 2  # unconstrained minimum at R=4321

        result = opt.minimize(metric_fn, maxiter=500)
        self.assertIsInstance(result, OptimizationResult)
        self.assertAlmostEqual(result.resolved_values["R"], 4321.0, delta=5.0)
        self.assertEqual(result.n_evaluations, len(result.history))
        self.assertGreater(result.n_evaluations, 0)

    def test_maximize_finds_known_maximum(self):
        opt = CircuitOptimizer(self.schematic, [OptimizationVariable("R", (0, 10000), x0=100)])
        self._wire_fake_runner(opt)

        def metric_fn(params):
            return -((params["R"] - 7000.0) ** 2)  # unconstrained max at R=7000

        result = opt.maximize(metric_fn, maxiter=500)
        self.assertAlmostEqual(result.resolved_values["R"], 7000.0, delta=5.0)

    def test_target_two_variables_hits_target_frequency(self):
        opt = CircuitOptimizer(self.schematic, [
            OptimizationVariable("R", (100, 100000), x0=1000),
            OptimizationVariable("C", (1e-9, 1e-6), x0=1e-8),
        ])
        self._wire_fake_runner(opt)

        def bandwidth_metric(params):
            return 1.0 / (2 * math.pi * params["R"] * params["C"])

        result = opt.target(bandwidth_metric, target_value=1000.0, maxiter=800)
        achieved_fc = 1.0 / (2 * math.pi * result.resolved_values["R"] * result.resolved_values["C"])
        self.assertAlmostEqual(achieved_fc, 1000.0, delta=20.0)

    def test_e_series_constraint_actually_applied_at_optimum(self):
        opt = CircuitOptimizer(self.schematic, [OptimizationVariable("R", (100, 10000), series="E24", x0=100)])
        self._wire_fake_runner(opt)

        def metric_fn(params):
            return (params["R"] - 4321.0) ** 2

        result = opt.minimize(metric_fn, maxiter=500)
        # The winning R must be an exact E24 value, not an arbitrary float.
        self.assertAlmostEqual(result.resolved_values["R"], snap_to_series(result.resolved_values["R"], "E24"))
        # And it should have landed on (or very near) the E24 value closest to 4321.
        self.assertAlmostEqual(result.resolved_values["R"], snap_to_series(4321, "E24"))

    def test_relationship_constrained_optimization(self):
        # Free variable R; C is DERIVED to hold R*C = 1e-6 constant. Optimizer
        # should still be able to drive R toward its unconstrained target.
        rel = Relationship(dependent="C", solve=hold_rc_product_constant("R", target_product=1e-6))
        opt = CircuitOptimizer(
            self.schematic, [OptimizationVariable("R", (100, 20000), x0=500)], relationships=[rel],
        )
        self._wire_fake_runner(opt)

        def metric_fn(params):
            self.assertAlmostEqual(params["R"] * params["C"], 1e-6, delta=1e-9)  # relationship must hold every eval
            return (params["R"] - 8000.0) ** 2

        result = opt.minimize(metric_fn, maxiter=500)
        self.assertAlmostEqual(result.resolved_values["R"], 8000.0, delta=20.0)
        self.assertAlmostEqual(result.resolved_values["R"] * result.resolved_values["C"], 1e-6, delta=1e-9)

    def test_history_best_matches_reported_result(self):
        opt = CircuitOptimizer(self.schematic, [OptimizationVariable("R", (0, 10000), x0=9000)])
        self._wire_fake_runner(opt)
        result = opt.minimize(lambda p: (p["R"] - 123.0) ** 2, maxiter=300)
        best_in_history = min(result.history, key=lambda s: s.objective_value)
        self.assertAlmostEqual(best_in_history.metric_value, result.metric_value)
        self.assertEqual(best_in_history.resolved_values, result.resolved_values)


# ===========================================================================
# MonteCarloEngine.sweep_2d()
# ===========================================================================

class _FakeSweepRunner:
    """Same shape as test_framework.py's _FakeLTspiceRunner: run_case() with
    a callback, fed by a real (empty) file so the engine's existence check
    passes, exactly like a completed LTspice run would leave behind."""

    def __init__(self, tmp_dir: Path):
        self.tmp_dir = tmp_dir
        self._i = 0
        self.n_ok = 0
        self.n_total = 0
        self.calls = []

    def run_case(self, param_values, callback):
        self._i += 1
        self.n_total += 1
        self.n_ok += 1
        self.calls.append(dict(param_values))
        raw_path = self.tmp_dir / f"sweep_{self._i}.raw"
        raw_path.touch()
        callback(str(raw_path), None)

    def wait_all(self, timeout=None):
        return self.n_ok == self.n_total

    def cleanup(self):
        pass


class MockRawWithParams:
    """Minimal object the test's metric function reads params back out of --
    mirrors the trick used for CircuitOptimizer above."""
    def __init__(self, params):
        self.params = params


class TestSweep2D(unittest.TestCase):
    def setUp(self):
        self.schematic = _write_dummy_schematic()
        self.tmp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        self.schematic.unlink()

    def _engine_with_fake_runner(self) -> "tuple[MonteCarloEngine, _FakeSweepRunner]":
        engine = MonteCarloEngine(self.schematic)
        fake = _FakeSweepRunner(self.tmp_dir)
        engine._get_runner = lambda: fake
        engine._read_raw = lambda raw_path, traces: MockRawWithParams(fake.calls[-1])
        return engine, fake

    def test_grid_shape_matches_values(self):
        engine, fake = self._engine_with_fake_runner()
        engine.add_metric("m", lambda raw: raw.params["R"] + raw.params["C"])
        result = engine.sweep_2d("R", [1, 2, 3], "C", [10, 20], progress=False)
        self.assertIsInstance(result, SweepResult)
        self.assertEqual(result.metrics["m"].shape, (2, 3))
        self.assertEqual(result.success.shape, (2, 3))
        self.assertEqual(result.n_points, 6)
        self.assertEqual(fake.n_total, 6)

    def test_values_correctly_indexed(self):
        engine, fake = self._engine_with_fake_runner()
        engine.add_metric("sum", lambda raw: raw.params["R"] + raw.params["C"])
        result = engine.sweep_2d("R", [1, 2], "C", [100, 200], progress=False)
        # metrics[iy, ix] must correspond to values_y[iy], values_x[ix]
        for iy, cval in enumerate([100, 200]):
            for ix, rval in enumerate([1, 2]):
                self.assertAlmostEqual(result.metrics["sum"][iy, ix], rval + cval)

    def test_fixed_params_held_constant(self):
        engine = MonteCarloEngine(self.schematic)
        engine.add_parameter("R", Gaussian(mean=1000, std=10), warn_if_undeclared=False)
        fake = _FakeSweepRunner(self.tmp_dir)
        engine._get_runner = lambda: fake
        engine._read_raw = lambda raw_path, traces: MockRawWithParams(fake.calls[-1])
        engine.add_metric("m", lambda raw: raw.params["EXTRA"])

        result = engine.sweep_2d("X", [1, 2], "Y", [10], fixed_params={"EXTRA": 42.0}, progress=False)
        self.assertTrue(np.all(result.metrics["m"] == 42.0))
        self.assertTrue(all(c["EXTRA"] == 42.0 for c in fake.calls))

    def test_non_swept_registered_param_defaults_to_distribution_center(self):
        engine = MonteCarloEngine(self.schematic)
        engine.add_parameter("R", Gaussian(mean=1234.0, std=10), warn_if_undeclared=False)
        fake = _FakeSweepRunner(self.tmp_dir)
        engine._get_runner = lambda: fake
        engine._read_raw = lambda raw_path, traces: MockRawWithParams(fake.calls[-1])
        engine.add_metric("r_value", lambda raw: raw.params["R"])

        result = engine.sweep_2d("X", [1], "Y", [1], progress=False)
        self.assertAlmostEqual(result.metrics["r_value"][0, 0], 1234.0)

    def test_relationships_resolved_per_grid_point(self):
        engine, fake = self._engine_with_fake_runner()
        rel = Relationship(dependent="C", solve=hold_rc_product_constant("R", target_product=1e-6))
        engine.add_metric("product", lambda raw: raw.params["R"] * raw.params["C"])

        result = engine.sweep_2d("R", [100, 500, 1000], "UNUSED", [0], relationships=[rel], progress=False)
        self.assertTrue(np.allclose(result.metrics["product"], 1e-6))

    def test_best_returns_correct_grid_point(self):
        engine, fake = self._engine_with_fake_runner()
        engine.add_metric("m", lambda raw: -((raw.params["R"] - 5) ** 2 + (raw.params["C"] - 50) ** 2))
        result = engine.sweep_2d("R", [1, 5, 9], "C", [10, 50, 90], progress=False)
        best = result.best("m", mode="max")
        self.assertAlmostEqual(best["R"], 5)
        self.assertAlmostEqual(best["C"], 50)
        self.assertAlmostEqual(best["m"], 0.0)

    def test_no_metrics_registered_raises(self):
        engine = MonteCarloEngine(self.schematic)
        with self.assertRaises(ValueError):
            engine.sweep_2d("R", [1], "C", [1])

    def test_partial_failure_shows_as_nan_not_success(self):
        engine = MonteCarloEngine(self.schematic)

        class _FlakyRunner(_FakeSweepRunner):
            def run_case(self, param_values, callback):
                self._i += 1
                self.n_total += 1
                if param_values["R"] == 2:
                    callback(None, None)  # simulate an aborted run
                    return
                self.n_ok += 1
                self.calls.append(dict(param_values))
                p = self.tmp_dir / f"sweep_{self._i}.raw"
                p.touch()
                callback(str(p), None)

        fake = _FlakyRunner(self.tmp_dir)
        engine._get_runner = lambda: fake
        engine._read_raw = lambda raw_path, traces: MockRawWithParams(fake.calls[-1])
        engine.add_metric("m", lambda raw: raw.params["R"])

        result = engine.sweep_2d("R", [1, 2, 3], "C", [0], progress=False)
        self.assertFalse(result.success[0, 1])  # R=2 column failed
        self.assertTrue(np.isnan(result.metrics["m"][0, 1]))
        self.assertTrue(result.success[0, 0] and result.success[0, 2])
        self.assertIsNotNone(result.errors[0][1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
