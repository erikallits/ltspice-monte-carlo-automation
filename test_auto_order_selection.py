"""
Tests for automatic transfer-function order selection and the unified
StabilityAnalysis shared between MonteCarloResults and CornerSweepResult.

The single most important thing tested here: auto_fit_transfer_function
must recover the TRUE order from synthetic data of KNOWN order, and must
NOT simply default to max_order (which is what would happen if the
complexity penalty were missing or broken, since least-squares RSS can only
improve or plateau as more parameters are added).
"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ltspice_mc.distributions import Gaussian
from ltspice_mc.engine import MonteCarloEngine
from ltspice_mc.stability import (
    CornerRun, CornerSweepResult,
    auto_fit_transfer_function, build_stability_analysis,
    make_fitted_extractor,
)


def _synthetic_ac_data(true_num, true_den, freq_decades=(-1, 4), n=300, noise_rel=1e-4, seed=0):
    """Analytic H(jw) = N(s)/D(s) for a KNOWN polynomial pair, plus small
    relative complex noise -- realistic enough that order selection isn't
    trivially degenerate at exactly RSS=0, without being dominated by noise."""
    freq = np.logspace(*freq_decades, n)
    s = 1j * 2 * np.pi * freq
    h_true = np.polyval(true_num, s) / np.polyval(true_den, s)
    rng = np.random.default_rng(seed)
    noise = noise_rel * np.abs(h_true) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    return freq, h_true + noise


# ===========================================================================
# Automatic order selection -- the critical correctness property
# ===========================================================================

class TestAutoFitTransferFunction(unittest.TestCase):
    def test_recovers_known_first_order(self):
        # D(s) = s + 5000 (leading coeff 1, matching fit's normalization)
        freq, h = _synthetic_ac_data(true_num=[5000.0], true_den=[1.0, 5000.0])
        result = auto_fit_transfer_function(freq, h, max_order=6, criterion="bic")
        self.assertEqual(result.best.den_order, 1)
        self.assertEqual(result.best.num_order, 0)

    def test_recovers_known_second_order(self):
        f0, q = 2000.0, 3.0
        w0 = 2 * np.pi * f0
        freq, h = _synthetic_ac_data(true_num=[w0 ** 2], true_den=[1.0, w0 / q, w0 ** 2])
        result = auto_fit_transfer_function(freq, h, max_order=6, criterion="bic")
        self.assertEqual(result.best.den_order, 2)

    def test_does_not_default_to_max_order(self):
        # A clean 1st-order system with max_order=6 available. If the
        # complexity penalty were missing/broken, RSS-only selection would
        # trivially always pick the highest order tried, every time.
        freq, h = _synthetic_ac_data(true_num=[8000.0], true_den=[1.0, 8000.0])
        result = auto_fit_transfer_function(freq, h, max_order=6, criterion="bic")
        self.assertEqual(result.best.den_order, 1)
        self.assertLess(result.best.den_order, 6)
        # every higher-order candidate must actually have been tried (and rejected)
        tried_orders = {c.den_order for c in result.candidates}
        self.assertIn(6, tried_orders)

    def test_bic_selection_matches_true_order_across_seeds(self):
        # Robustness check across several noise realizations, not just one
        # lucky seed.
        f0, q = 1500.0, 1.5
        w0 = 2 * np.pi * f0
        for seed in range(5):
            freq, h = _synthetic_ac_data(
                true_num=[w0 ** 2], true_den=[1.0, w0 / q, w0 ** 2], seed=seed,
            )
            result = auto_fit_transfer_function(freq, h, max_order=6, criterion="bic")
            self.assertEqual(result.best.den_order, 2, msg=f"failed for seed={seed}")

    def test_summary_flags_max_order_ceiling(self):
        # A 4th-order true system with max_order capped AT the true order --
        # selection lands exactly at the ceiling, and summary() must say so.
        w0 = 2 * np.pi * 1000.0
        true_den = np.poly([-w0, -w0 * 1.5, -w0 * 0.7, -w0 * 2.0])  # 4 real poles, all stable
        freq, h = _synthetic_ac_data(true_num=[w0 ** 4], true_den=true_den, freq_decades=(1, 5))
        result = auto_fit_transfer_function(freq, h, max_order=4, criterion="bic")
        self.assertEqual(result.best.den_order, 4)
        self.assertIn("largest den_order tried", result.summary())

    def test_invalid_criterion_raises(self):
        freq, h = _synthetic_ac_data(true_num=[1.0], true_den=[1.0, 1.0])
        with self.assertRaises(ValueError):
            auto_fit_transfer_function(freq, h, criterion="nope")

    def test_invalid_max_order_raises(self):
        freq, h = _synthetic_ac_data(true_num=[1.0], true_den=[1.0, 1.0])
        with self.assertRaises(ValueError):
            auto_fit_transfer_function(freq, h, max_order=0)

    def test_too_few_points_raises(self):
        with self.assertRaises(ValueError):
            auto_fit_transfer_function(np.array([1.0]), np.array([1 + 0j]), max_order=6)

    def test_condition_number_is_finite_for_well_posed_fit(self):
        freq, h = _synthetic_ac_data(true_num=[5000.0], true_den=[1.0, 5000.0])
        result = auto_fit_transfer_function(freq, h, max_order=3, criterion="bic")
        self.assertTrue(np.isfinite(result.best.condition_number))
        self.assertGreater(result.best.condition_number, 0)


class TestMakeFittedExtractor(unittest.TestCase):
    def test_uses_fixed_order_consistently(self):
        f0, q = 1000.0, 2.0
        w0 = 2 * np.pi * f0
        freq, h = _synthetic_ac_data(true_num=[w0 ** 2], true_den=[1.0, w0 / q, w0 ** 2], noise_rel=0.0)

        class _Trace:
            def __init__(self, wave):
                self._wave = wave
            def get_wave(self, step=0):
                return self._wave

        class _Raw:
            def get_axis(self, step=0):
                return freq
            def get_trace(self, name):
                return {"V(out)": _Trace(h), "V(in)": _Trace(np.ones_like(h))}[name]

        extract = make_fitted_extractor("V(out)", "V(in)", num_order=0, den_order=2)
        den = extract(_Raw(), {"anything": 1.0})  # resolved_params accepted but unused
        self.assertEqual(len(den), 3)  # den_order=2 -> 3 coefficients
        np.testing.assert_allclose(den, [1.0, w0 / q, w0 ** 2], rtol=1e-3)


# ===========================================================================
# Unified StabilityAnalysis -- same shared logic behind both MC and corner
# ===========================================================================

class TestBuildStabilityAnalysis(unittest.TestCase):
    def test_monte_carlo_source_labeling(self):
        den_list = [np.array([1.0, 6.0, 11.0, 6.0])] * 8 + [None]
        success = [True] * 8 + [False]
        analysis = build_stability_analysis("monte_carlo", den_list, success)
        self.assertEqual(analysis.source, "monte_carlo")
        self.assertEqual(analysis.n_success, 8)
        self.assertTrue(np.all(analysis.stable_mask))
        self.assertAlmostEqual(analysis.stable_fraction, 1.0)
        self.assertIn("Empirical Monte Carlo", analysis.summary())
        self.assertIn("STATISTICAL estimate", analysis.summary())

    def test_corner_source_labeling(self):
        den_list = [np.array([1.0, 6.0, 11.0, 6.0]), np.array([1.0, -2.0, 1.0, 0.5])]
        success = [True, True]
        analysis = build_stability_analysis("corner_sweep", den_list, success)
        self.assertEqual(analysis.source, "corner_sweep")
        self.assertIn("Corner stability", analysis.summary())
        self.assertIn("Exhaustive over the corners", analysis.summary())

    def test_inconsistent_orders_error_mentions_fixed_order_guidance(self):
        den_list = [np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0])]
        with self.assertRaises(ValueError) as ctx:
            build_stability_analysis("monte_carlo", den_list, [True, True])
        self.assertIn("make_fitted_extractor", str(ctx.exception))

    def test_corner_sweep_result_stability_analysis_matches_shared_helper(self):
        runs = [
            CornerRun(params={}, success=True, den_coeffs=np.array([1.0, 6.0, 11.0, 6.0])),
            CornerRun(params={}, success=True, den_coeffs=np.array([1.0, 4.0, -5.0, -6.0])),
        ]
        result = CornerSweepResult(param_names=[], runs=runs)
        analysis = result.stability_analysis()
        np.testing.assert_array_equal(analysis.stable_mask, [True, False])
        self.assertEqual(analysis.source, "corner_sweep")


# ===========================================================================
# Engine integration: run(extract_den_fn=...), determine_transfer_function_order()
# ===========================================================================

class _FakeRunner:
    def __init__(self, tmp_dir: Path):
        self.tmp_dir = tmp_dir
        self.n = 0
        self.n_ok = 0
        self.n_total = 0

    def run_case(self, param_values, callback):
        self.n += 1
        self.n_total += 1
        self.n_ok += 1
        p = self.tmp_dir / f"run_{self.n}.raw"
        p.touch()
        callback(str(p), None)

    def run_case_blocking(self, param_values):
        self.n += 1
        p = self.tmp_dir / f"blocking_{self.n}.raw"
        p.touch()
        return str(p), None

    def wait_all(self, timeout=None):
        return self.n_ok == self.n_total

    def cleanup(self):
        pass


class TestEngineTransferFunctionIntegration(unittest.TestCase):
    def setUp(self):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".cir", delete=False)
        f.write(".param R=1k C=10n\nR1 in out {R}\nC1 out 0 {C}\n.end\n")
        f.close()
        self.schematic = Path(f.name)
        self.tmp_dir = Path(tempfile.gettempdir())

    def tearDown(self):
        self.schematic.unlink()

    def test_run_with_extract_den_fn_populates_den_coeffs(self):
        engine = MonteCarloEngine(self.schematic)
        engine.add_parameter("R", Gaussian(mean=1000, std=10))
        engine.add_metric("dummy", lambda raw: 1.0)

        fake = _FakeRunner(self.tmp_dir)
        engine._get_runner = lambda: fake
        engine._read_raw = lambda path, traces: object()  # extract_den_fn below ignores it

        def extract(raw, resolved):
            return np.array([1.0, resolved["R"]])  # trivially encode R into the "coefficients"

        results = engine.run(n_samples=5, seed=1, cleanup_files=False, progress=False, extract_den_fn=extract)
        self.assertEqual(len(results.den_coeffs), 5)
        self.assertTrue(all(c is not None for c in results.den_coeffs))
        stack = results.stacked_den_coeffs()
        self.assertEqual(stack.shape, (5, 2))

    def test_run_without_extract_den_fn_leaves_den_coeffs_empty(self):
        engine = MonteCarloEngine(self.schematic)
        engine.add_parameter("R", Gaussian(mean=1000, std=10))
        engine.add_metric("dummy", lambda raw: 1.0)
        fake = _FakeRunner(self.tmp_dir)
        engine._get_runner = lambda: fake
        engine._read_raw = lambda path, traces: object()

        results = engine.run(n_samples=3, seed=1, cleanup_files=False, progress=False)
        self.assertTrue(all(c is None for c in results.den_coeffs))
        with self.assertRaises(ValueError):
            results.stacked_den_coeffs()

    def test_run_with_only_extract_den_fn_no_metrics_is_valid(self):
        # A purely stability-focused run: no add_metric() call at all,
        # just extract_den_fn. Must NOT raise "no metrics registered".
        engine = MonteCarloEngine(self.schematic)
        engine.add_parameter("R", Gaussian(mean=1000, std=10))
        fake = _FakeRunner(self.tmp_dir)
        engine._get_runner = lambda: fake
        engine._read_raw = lambda path, traces: object()

        results = engine.run(
            n_samples=3, seed=1, cleanup_files=False, progress=False,
            extract_den_fn=lambda raw, resolved: np.array([1.0, 5.0]),
        )
        self.assertEqual(results.n_success, 3)
        self.assertEqual(results.metrics, {})

    def test_run_with_neither_metrics_nor_extract_den_fn_still_raises(self):
        engine = MonteCarloEngine(self.schematic)
        engine.add_parameter("R", Gaussian(mean=1000, std=10))
        with self.assertRaises(ValueError):
            engine.run(n_samples=3)

    def test_monte_carlo_results_stability_analysis_end_to_end(self):
        engine = MonteCarloEngine(self.schematic)
        engine.add_parameter("R", Gaussian(mean=1000, std=10))
        engine.add_metric("dummy", lambda raw: 1.0)
        fake = _FakeRunner(self.tmp_dir)
        engine._get_runner = lambda: fake
        engine._read_raw = lambda path, traces: object()

        # Fixed, always-stable 2nd order D(s) regardless of R -- just checking wiring.
        extract = lambda raw, resolved: np.array([1.0, 6.0, 11.0, 6.0])
        results = engine.run(n_samples=6, seed=2, cleanup_files=False, progress=False, extract_den_fn=extract)
        analysis = results.stability_analysis()
        self.assertEqual(analysis.source, "monte_carlo")
        self.assertTrue(np.all(analysis.stable_mask))
        self.assertTrue(analysis.kharitonov.all_stable)

    def test_determine_transfer_function_order_uses_nominal_and_runs_once(self):
        engine = MonteCarloEngine(self.schematic)
        engine.add_parameter("R", Gaussian(mean=1000, std=10))

        fake = _FakeRunner(self.tmp_dir)
        engine._get_runner = lambda: fake

        f0, q = 1500.0, 2.0
        w0 = 2 * np.pi * f0
        freq, h = _synthetic_ac_data(true_num=[w0 ** 2], true_den=[1.0, w0 / q, w0 ** 2])

        class _Trace:
            def __init__(self, wave):
                self._wave = wave
            def get_wave(self, step=0):
                return self._wave

        class _Raw:
            def get_axis(self, step=0):
                return freq
            def get_trace(self, name):
                return {"V(out)": _Trace(h), "V(in)": _Trace(np.ones_like(h))}[name]

        seen_params = {}

        def fake_read_raw(path, traces):
            return _Raw()

        engine._read_raw = fake_read_raw

        result = engine.determine_transfer_function_order("V(out)", "V(in)", max_order=6)
        self.assertEqual(result.best.den_order, 2)
        self.assertEqual(fake.n, 1)  # exactly one simulation, not a batch


if __name__ == "__main__":
    unittest.main(verbosity=2)
