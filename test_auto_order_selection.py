"""
Tests for automatic transfer-function order selection and the unified
StabilityAnalysis shared between MonteCarloResults and CornerSweepResult.

The single most important thing tested here: auto_fit_transfer_function
must recover the TRUE order from synthetic data of KNOWN order, and must
NOT simply default to max_order (which is what would happen if the
complexity penalty were missing or broken, since least-squares RSS can only
improve or plateau as more parameters are added).
"""

import math
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
        # Tests summary()'s ceiling-detection logic directly, via a
        # constructed AutoFitResult -- decoupled from whether a specific
        # synthetic frequency sweep happens to land the SEARCH at the
        # ceiling, which now also depends on conditioning (tested
        # separately below), not just AIC/BIC.
        from ltspice_mc.stability import AutoFitResult, OrderCandidate
        candidates = [
            OrderCandidate(0, 1, np.array([1.0]), np.array([1.0, 1.0]), rss=10.0, aic=5.0, bic=6.0, condition_number=10.0),
            OrderCandidate(0, 2, np.array([1.0]), np.array([1.0, 1.0, 1.0]), rss=1.0, aic=1.0, bic=2.0, condition_number=10.0),
        ]
        result = AutoFitResult(best=candidates[1], candidates=candidates, criterion="bic")
        self.assertIn("largest den_order tried", result.summary())

    def test_ill_conditioned_candidate_does_not_win_even_with_better_bic(self):
        # Direct regression test for the reported bug: an ill-conditioned
        # higher-order candidate with a SPURIOUSLY better (lower) BIC must
        # NOT be selected over a well-conditioned lower-order one.
        from ltspice_mc.stability import OrderCandidate, _select_best_candidate
        good = OrderCandidate(0, 2, np.array([1.0]), np.array([1.0, 1.0, 1.0]),
                               rss=3.5e-25, aic=-2516, bic=-2510, condition_number=1e4)
        bad = OrderCandidate(2, 4, np.array([1.0, 1.0, 1.0]), np.array([1.0] * 5),
                              rss=1.65e-25, aic=-2540, bic=-2527, condition_number=5.84e28)
        # Without any filter, `bad` would win on raw BIC (-2527 < -2510) --
        # this is exactly the reported failure. With the default threshold,
        # `good` must win instead.
        self.assertLess(bad.bic, good.bic)  # sanity: confirms bad WOULD win unfiltered
        selected = _select_best_candidate([good, bad], criterion="bic", max_condition_number=1e10)
        self.assertIs(selected, good)

    def test_all_candidates_ill_conditioned_falls_back_with_warning(self):
        from ltspice_mc.stability import OrderCandidate, _select_best_candidate
        c1 = OrderCandidate(0, 1, np.array([1.0]), np.array([1.0, 1.0]), rss=1.0, aic=1.0, bic=1.0, condition_number=1e15)
        c2 = OrderCandidate(0, 2, np.array([1.0]), np.array([1.0, 1.0, 1.0]), rss=1.0, aic=2.0, bic=2.0, condition_number=1e20)
        with self.assertLogs("ltspice_mc.stability", level="WARNING"):
            selected = _select_best_candidate([c1, c2], criterion="bic", max_condition_number=1e10)
        self.assertIs(selected, c1)  # least-bad conditioning wins the fallback, not AIC/BIC

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


class TestTransferFunctionLatex(unittest.TestCase):
    """The key property: every term, sign, power and number comes from the
    coefficient VALUES passed in -- nothing about the equation is hardcoded."""

    def test_first_order_exact_string(self):
        from ltspice_mc.stability import transfer_function_to_latex
        latex = transfer_function_to_latex([5000.0], [1.0, 5000.0])
        self.assertEqual(latex, "\\mathrm{H}(s) = \\frac{5000}{s + 5000}")

    def test_order_changes_output_dynamically(self):
        # Same call, different coefficient counts -> correspondingly
        # different equations. If anything were hardcoded, these would match.
        from ltspice_mc.stability import transfer_function_to_latex
        first = transfer_function_to_latex([1.0], [1.0, 2.0])
        third = transfer_function_to_latex([1.0], [1.0, 2.0, 3.0, 4.0])
        self.assertNotEqual(first, third)
        self.assertIn("s^{3}", third)
        self.assertNotIn("s^{3}", first)

    def test_zero_coefficients_omitted(self):
        from ltspice_mc.stability import transfer_function_to_latex
        # s^2 + 0*s + 4  ->  the zero term must not appear at all
        latex = transfer_function_to_latex([1.0], [1.0, 0.0, 4.0])
        self.assertIn("s^{2} + 4", latex)
        self.assertNotIn("0 s", latex)

    def test_negative_coefficients_render_as_minus(self):
        from ltspice_mc.stability import transfer_function_to_latex
        latex = transfer_function_to_latex([1.0], [1.0, -2.5, 4.0])
        self.assertIn("- 2.5 s", latex)
        self.assertNotIn("+ -", latex)

    def test_unit_coefficient_omits_redundant_one(self):
        from ltspice_mc.stability import transfer_function_to_latex
        latex = transfer_function_to_latex([1.0], [1.0, 1.0])
        self.assertIn("\\frac{1}{s + 1}", latex)
        self.assertNotIn("1 s", latex)

    def test_scientific_notation_for_large_values(self):
        from ltspice_mc.stability import transfer_function_to_latex
        latex = transfer_function_to_latex([1.0], [1.0, 3.95e7])
        self.assertIn("\\times 10^{7}", latex)
        self.assertNotIn("e+07", latex)  # must not leak Python's float repr

    def test_scientific_notation_for_tiny_values(self):
        from ltspice_mc.stability import transfer_function_to_latex
        latex = transfer_function_to_latex([1e-7], [1.0, 2.5e-5])
        self.assertIn("\\times 10^{-7}", latex)
        self.assertNotIn("e-07", latex)

    def test_custom_variable_name(self):
        from ltspice_mc.stability import transfer_function_to_latex
        latex = transfer_function_to_latex([1.0], [1.0, 2.0], variable="p")
        self.assertIn("\\mathrm{H}(p)", latex)
        self.assertIn("p + 2", latex)
        self.assertNotIn("s", latex)

    def test_always_contains_a_fraction(self):
        from ltspice_mc.stability import transfer_function_to_latex
        for num, den in ([[1.0], [1.0, 2.0]], [[3.0, 1.0], [1.0, 0.0, 5.0]]):
            self.assertIn("\\frac{", transfer_function_to_latex(num, den))

    def test_renders_without_mathtext_warnings(self):
        import warnings
        import matplotlib
        matplotlib.use("Agg")
        from ltspice_mc.visualization import plot_transfer_function_latex
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any mathtext parse issue -> failure
            fig = plot_transfer_function_latex(
                [1.0, 0.0, 3.95e7], [1.0, 1256.6, 3.95e7], subtitle="num=2, den=2",
            )
        self.assertIsNotNone(fig)

    def test_end_to_end_from_a_real_fit(self):
        # The realistic path: fit synthetic data, render whatever came out.
        from ltspice_mc.stability import fit_transfer_function_lstsq, transfer_function_to_latex
        freq = np.logspace(1, 5, 200)
        w0, q = 2 * math.pi * 1000, 5.0
        s = 1j * 2 * math.pi * freq
        h = (s ** 2 + w0 ** 2) / (s ** 2 + s * w0 / q + w0 ** 2)
        num, den = fit_transfer_function_lstsq(freq, h, num_order=2, den_order=2)
        latex = transfer_function_to_latex(num, den)
        self.assertIn("\\frac{", latex)
        self.assertIn("s^{2}", latex)


class TestStabilityMarginAndPlots(unittest.TestCase):
    def test_stability_margin_matches_direct_root_finding(self):
        from ltspice_mc.stability import stability_margin
        stack = np.array([[1.0, 6.0, 11.0, 6.0], [1.0, 4.0, -5.0, -6.0]])
        margins = stability_margin(stack)
        self.assertEqual(len(margins), 2)
        # first is stable (all roots negative real part) -> margin < 0
        self.assertLess(margins[0], 0)
        # second has a root at s=+1 (unstable) -> margin > 0
        self.assertGreater(margins[1], 0)

    def test_plot_stability_margin_histogram_smoke(self):
        import matplotlib
        matplotlib.use("Agg")
        from ltspice_mc.visualization import plot_stability_margin_histogram
        stack = np.array([[1.0, 6.0, 11.0, 6.0]] * 10)
        fig = plot_stability_margin_histogram(stack)
        self.assertIsNotNone(fig)
        self.assertIn("100.0% stable", fig.axes[0].get_title())

    def test_plot_temperature_stability_profile_smoke(self):
        import matplotlib
        matplotlib.use("Agg")
        from ltspice_mc.visualization import plot_temperature_stability_profile
        stack = np.array([[1.0, 6.0, 11.0, 6.0]] * 10)
        temps = np.linspace(-40, 125, 10)
        fig = plot_temperature_stability_profile(temps, stack)
        self.assertEqual(len(fig.axes), 2)


class TestFrequencyNormalization(unittest.TestCase):
    def test_normalized_and_unnormalized_fits_agree(self):
        # For a well-conditioned case, fitting with omega_ref=1 (old
        # behavior) and the default (geometric-mean normalization) must
        # give the SAME true-s coefficients -- normalization is purely an
        # internal conditioning aid and must not change the answer.
        from ltspice_mc.stability import fit_transfer_function_lstsq
        freq = np.logspace(1, 4, 50)
        w0, q = 2 * math.pi * 500, 1.2
        s = 1j * 2 * math.pi * freq
        h = w0 ** 2 / (s ** 2 + s * w0 / q + w0 ** 2)

        num1, den1 = fit_transfer_function_lstsq(freq, h, 0, 2, omega_ref=1.0)
        num2, den2 = fit_transfer_function_lstsq(freq, h, 0, 2, omega_ref=None)
        np.testing.assert_allclose(den1, den2, rtol=1e-6)
        np.testing.assert_allclose(num1, num2, rtol=1e-6)

    def test_wide_sweep_notch_recovers_exact_order_and_poles(self):
        # Direct regression test for the reported notch-filter case: a
        # sparse, LINEAR, 5-decade sweep (mirroring ".ac lin 20 0.1 10k")
        # must still recover the true 2nd-order poles, not collapse to a
        # spurious 1st-order fit due to poor conditioning.
        freq = np.linspace(0.1, 10000, 20)
        w0, q = 2 * math.pi * 1000.0, 5.0
        s = 1j * 2 * math.pi * freq
        h = (s ** 2 + w0 ** 2) / (s ** 2 + s * w0 / q + w0 ** 2)

        result = auto_fit_transfer_function(freq, h, max_order=6, criterion="bic")
        self.assertEqual(result.best.den_order, 2)
        self.assertLess(result.best.condition_number, 1e7)  # was 1.265e10 before normalization

        roots = np.roots(result.best.den_coeffs)
        expected_real = -w0 / (2 * q)
        expected_imag = w0 * math.sqrt(1 - 1 / (4 * q ** 2))
        found_real = sorted(roots.real)[0]
        found_imag = sorted(abs(roots.imag))[0]
        self.assertAlmostEqual(found_real, expected_real, delta=1.0)
        self.assertAlmostEqual(found_imag, expected_imag, delta=1.0)


class TestFrequencySweepQuality(unittest.TestCase):
    def test_flags_linear_wide_sweep(self):
        from ltspice_mc.stability import check_frequency_sweep
        freq = np.linspace(0.1, 10000, 20)  # exactly the reported .ac lin 20 0.1 10k
        quality = check_frequency_sweep(freq)
        self.assertFalse(quality.is_adequate)
        self.assertFalse(quality.looks_log_spaced)
        self.assertTrue(any("LINEARLY spaced" in w for w in quality.warnings))

    def test_dense_log_sweep_is_adequate(self):
        from ltspice_mc.stability import check_frequency_sweep
        freq = np.logspace(0, 6, 600)  # dense, log-spaced, 6 decades
        quality = check_frequency_sweep(freq)
        self.assertTrue(quality.is_adequate)
        self.assertTrue(quality.looks_log_spaced)

    def test_sparse_sweep_flagged_even_if_log_spaced(self):
        from ltspice_mc.stability import check_frequency_sweep
        freq = np.logspace(0, 6, 8)  # log-spaced but very few points
        quality = check_frequency_sweep(freq)
        self.assertFalse(quality.is_adequate)

    def test_auto_fit_result_surfaces_sweep_warning_first(self):
        freq = np.linspace(0.1, 10000, 20)
        w0 = 2 * math.pi * 1000.0
        s = 1j * 2 * math.pi * freq
        h = w0 ** 2 / (s ** 2 + s * w0 / 2 + w0 ** 2)
        result = auto_fit_transfer_function(freq, h, max_order=4)
        summary = result.summary()
        self.assertTrue(summary.startswith("FREQUENCY SWEEP WARNING"))


class TestSetAcAnalysis(unittest.TestCase):
    def _write(self, text):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".net", delete=False)
        f.write(text)
        f.close()
        return Path(f.name)

    def test_replaces_existing_ac_directive(self):
        from ltspice_mc.stability import set_ac_analysis
        text = "R1 in out 1k\nC1 out 0 1n\n.ac lin 20 0.1 10k\n.end\n"
        path = self._write(text)
        try:
            out = set_ac_analysis(path, sweep_type="dec", points=100, fstart=1, fstop=1e6)
            new_text = out.read_text()
            self.assertIn(".ac dec 100 1 1e+06", new_text)
            self.assertNotIn(".ac lin", new_text)
            self.assertIn("R1 in out 1k", new_text)  # untouched
            self.assertIn("C1 out 0 1n", new_text)   # untouched
        finally:
            path.unlink()
            out.unlink(missing_ok=True)

    def test_inserts_before_end_if_no_existing_directive(self):
        from ltspice_mc.stability import set_ac_analysis
        text = "R1 in out 1k\n.end\n"
        path = self._write(text)
        try:
            out = set_ac_analysis(path, points=50, fstart=10, fstop=1e5)
            lines = out.read_text().splitlines()
            end_idx = lines.index(".end")
            ac_idx = next(i for i, l in enumerate(lines) if l.startswith(".ac"))
            self.assertLess(ac_idx, end_idx)
        finally:
            path.unlink()
            out.unlink(missing_ok=True)

    def test_duplicate_ac_lines_collapsed_to_one(self):
        from ltspice_mc.stability import set_ac_analysis
        text = ".ac lin 10 1 100\n.ac dec 5 1 1000\nR1 in out 1k\n.end\n"
        path = self._write(text)
        try:
            out = set_ac_analysis(path, points=200, fstart=1, fstop=1e6)
            lines = out.read_text().splitlines()
            ac_lines = [l for l in lines if l.strip().lower().startswith(".ac")]
            self.assertEqual(len(ac_lines), 1)
        finally:
            path.unlink()
            out.unlink(missing_ok=True)

    def test_asc_file_raises_clear_error(self):
        from ltspice_mc.stability import set_ac_analysis
        path = self._write("Version 4\n")
        asc_path = path.with_suffix(".asc")
        path.rename(asc_path)
        try:
            with self.assertRaises(ValueError):
                set_ac_analysis(asc_path)
        finally:
            asc_path.unlink()



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
