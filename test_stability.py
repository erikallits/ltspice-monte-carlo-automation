"""
Tests for stability.py and MonteCarloEngine.run_corner_sweep() /
enable_temperature_control(). Everything here runs without spicelib or
LTspice: Kharitonov/Hurwitz math is pure numpy, the TF fit is checked
against analytically-generated synthetic frequency-response data with a
KNOWN true answer, and the corner-sweep control flow is exercised through
the same fake-runner pattern used in test_framework.py.
"""

import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ltspice_mc.engine import MonteCarloEngine
from ltspice_mc.distributions import Gaussian
from ltspice_mc.stability import (
    CornerRun, CornerSweepResult,
    analytical_den_second_order, analytical_den_single_pole,
    check_all_runs_stable, check_kharitonov_stability,
    fit_transfer_function_lstsq, is_hurwitz_stable, kharitonov_polynomials,
)


# ===========================================================================
# Hurwitz stability
# ===========================================================================

class TestHurwitzStability(unittest.TestCase):
    def test_known_stable(self):
        # (s+1)(s+2)(s+3) = s^3 + 6s^2 + 11s + 6 -- roots at -1,-2,-3
        self.assertTrue(is_hurwitz_stable([1, 6, 11, 6]))

    def test_known_unstable(self):
        # (s-1)^2 = s^2 - 2s + 1 -- roots at +1, +1
        self.assertFalse(is_hurwitz_stable([1, -2, 1]))

    def test_marginal_on_imaginary_axis_is_not_stable(self):
        # s^2 + 1 -- roots at +/- j, real part exactly 0
        self.assertFalse(is_hurwitz_stable([1, 0, 1]))

    def test_simple_first_order_stable(self):
        # s + 5 -- root at -5
        self.assertTrue(is_hurwitz_stable([1, 5]))


# ===========================================================================
# Kharitonov polynomial construction -- checked against a hand-derived
# example that itself was cross-checked against Wikipedia's formula.
# ===========================================================================

class TestKharitonovConstruction(unittest.TestCase):
    def test_hand_verified_example(self):
        # coeff_bounds lowest-power-first: a0 in [1,2], a1 in [3,4], a2 in [5,6]
        bounds = [(1, 2), (3, 4), (5, 6)]
        k1, k2, k3, k4 = kharitonov_polynomials(bounds)
        # Expected, highest-power-first (see derivation in the accompanying
        # message / module docstring):
        np.testing.assert_array_almost_equal(k1, [6, 3, 1])
        np.testing.assert_array_almost_equal(k2, [5, 4, 2])
        np.testing.assert_array_almost_equal(k3, [6, 4, 1])
        np.testing.assert_array_almost_equal(k4, [5, 3, 2])

    def test_degenerate_no_uncertainty_collapses_to_single_polynomial(self):
        # min == max for every coefficient -> all four K's must be IDENTICAL
        # to the one true polynomial, and Kharitonov's verdict must agree
        # exactly with directly checking that polynomial.
        true_poly = [1.0, 6.0, 11.0, 6.0]  # lowest-first: a0..a3 = 6,11,6,1
        bounds = [(c, c) for c in true_poly[::-1]]  # lowest-power-first
        result = check_kharitonov_stability(bounds)
        for name in ("K1", "K2", "K3", "K4"):
            np.testing.assert_array_almost_equal(result.polynomials[name], true_poly)
        self.assertEqual(result.all_stable, is_hurwitz_stable(true_poly))
        self.assertTrue(result.all_stable)  # this specific poly IS stable (roots -1,-2,-3)

    def test_invalid_bounds_raise(self):
        with self.assertRaises(ValueError):
            kharitonov_polynomials([(5, 1)])  # min > max

    def test_check_kharitonov_stability_end_to_end_stable_case(self):
        # A single-pole family: D(s) = 1 + s*tau, tau in [1e-4, 2e-4] --
        # always exactly one real negative root regardless of tau -> stable
        # for the whole box, so Kharitonov must confirm it.
        bounds = [(1.0, 1.0), (1e-4, 2e-4)]  # a0=1 fixed, a1=tau in range
        result = check_kharitonov_stability(bounds)
        self.assertTrue(result.all_stable)
        self.assertIn("STABLE", result.summary())

    def test_check_kharitonov_stability_end_to_end_unstable_case(self):
        # D(s) = a0 + a1*s with a0 allowed to be negative -> a root at
        # s = -a0/a1 can land in the RHP for some combination -> must fail.
        bounds = [(-1.0, 1.0), (1.0, 1.0)]
        result = check_kharitonov_stability(bounds)
        self.assertFalse(result.all_stable)
        self.assertIn("cross-check", result.summary())


class TestCheckAllRunsStable(unittest.TestCase):
    def test_mixed_stable_and_unstable_runs(self):
        stack = np.array([
            [1, 6, 11, 6],   # stable
            [1, -2, 1, 0],   # unstable-ish leading example -- just needs >=1 bad root; use a clean unstable cubic instead
        ])
        # Replace the second row with an unambiguous unstable cubic: (s-1)(s+2)(s+3) = s^3+4s^2-5s-6
        stack[1] = [1, 4, -5, -6]
        result = check_all_runs_stable(stack)
        np.testing.assert_array_equal(result, [True, False])


# ===========================================================================
# Transfer function extraction
# ===========================================================================

class TestAnalyticalExtraction(unittest.TestCase):
    def test_single_pole(self):
        extract = analytical_den_single_pole(lambda p: p["R1"] * p["C1"])
        den = extract(None, {"R1": 1000.0, "C1": 1e-7})  # tau = 1e-4
        np.testing.assert_array_almost_equal(den, [1e-4, 1.0])

    def test_second_order(self):
        extract = analytical_den_second_order(
            omega0_fn=lambda p: 2 * math.pi * p["FC"], q_fn=lambda p: p["Q"],
        )
        den = extract(None, {"FC": 1000.0, "Q": 0.707})
        w0 = 2 * math.pi * 1000.0
        expected = [1 / w0 ** 2, 1 / (0.707 * w0), 1.0]
        np.testing.assert_array_almost_equal(den, expected)


class TestLstsqFit(unittest.TestCase):
    def test_recovers_known_second_order_lowpass(self):
        # H(s) = w0^2 / (s^2 + s*w0/Q + w0^2) -- generate exact synthetic
        # frequency-response data, then verify the fit recovers the TRUE
        # normalized denominator [1, w0/Q, w0^2] (leading coeff 1, matching
        # fit_transfer_function_lstsq's own normalization).
        f0 = 1000.0
        w0 = 2 * math.pi * f0
        q = 2.0
        freq = np.logspace(math.log10(f0) - 2, math.log10(f0) + 2, 400)
        s = 1j * 2 * math.pi * freq
        true_den = s ** 2 + s * w0 / q + w0 ** 2
        h = (w0 ** 2) / true_den

        num, den = fit_transfer_function_lstsq(freq, h, num_order=0, den_order=2)

        expected_den = np.array([1.0, w0 / q, w0 ** 2])
        # Relative tolerance -- this is a numerical fit, not exact algebra.
        np.testing.assert_allclose(den, expected_den, rtol=1e-3)
        self.assertAlmostEqual(num[0] / den[-1], 1.0, delta=1e-3)  # DC gain recovered

    def test_recovers_known_single_pole(self):
        tau = 5e-5
        freq = np.logspace(1, 7, 300)
        s = 1j * 2 * math.pi * freq
        h = 1.0 / (1.0 + s * tau)

        num, den = fit_transfer_function_lstsq(freq, h, num_order=0, den_order=1)
        # fit_transfer_function_lstsq normalizes the HIGHEST-order (s^1)
        # coefficient to 1, so "1 + s*tau" (leading s^0 coeff = 1) comes back
        # scaled by 1/tau: D(s) = (1/tau) + s -> [1.0, 1/tau], not [tau, 1.0].
        np.testing.assert_allclose(den, [1.0, 1.0 / tau], rtol=1e-3)

    def test_too_few_points_raises(self):
        with self.assertRaises(ValueError):
            fit_transfer_function_lstsq(np.array([1.0, 2.0]), np.array([1 + 0j, 1 + 0j]),
                                         num_order=2, den_order=3)


# ===========================================================================
# CornerSweepResult helpers
# ===========================================================================

class TestCornerSweepResult(unittest.TestCase):
    def test_stacked_and_bounds(self):
        runs = [
            CornerRun(params={"R": 1}, success=True, den_coeffs=np.array([1.0, 4.0, 6.0])),
            CornerRun(params={"R": 2}, success=True, den_coeffs=np.array([1.0, 5.0, 7.0])),
            CornerRun(params={"R": 3}, success=False, error="boom"),
        ]
        result = CornerSweepResult(param_names=["R"], runs=runs)
        self.assertEqual(result.n_points, 3)
        self.assertEqual(result.n_success, 2)
        stacked = result.stacked_den_coeffs()
        self.assertEqual(stacked.shape, (2, 3))
        bounds = result.coefficient_bounds()  # lowest-power-first
        # highest-power-first coeffs were [1,4,6] and [1,5,7] -> lowest-first: [6,4,1],[7,5,1]
        self.assertEqual(bounds, [(6.0, 7.0), (4.0, 5.0), (1.0, 1.0)])

    def test_inconsistent_order_raises(self):
        runs = [
            CornerRun(params={}, success=True, den_coeffs=np.array([1.0, 2.0])),
            CornerRun(params={}, success=True, den_coeffs=np.array([1.0, 2.0, 3.0])),
        ]
        result = CornerSweepResult(param_names=[], runs=runs)
        with self.assertRaises(ValueError):
            result.stacked_den_coeffs()

    def test_no_successful_runs_raises(self):
        result = CornerSweepResult(param_names=[], runs=[CornerRun(params={}, success=False, error="x")])
        with self.assertRaises(ValueError):
            result.stacked_den_coeffs()


# ===========================================================================
# MonteCarloEngine.run_corner_sweep / enable_temperature_control
# ===========================================================================

class _FakeCornerRunner:
    """Records every resolved parameter set it's asked to run, and
    synchronously invokes the callback with a real (empty) touched file,
    mirroring a completed LTspice run -- same pattern as test_framework.py."""

    def __init__(self, tmp_dir: Path):
        self.tmp_dir = tmp_dir
        self.calls = []
        self.n_ok = 0
        self.n_total = 0

    def run_case(self, param_values, callback):
        self.calls.append(dict(param_values))
        self.n_total += 1
        self.n_ok += 1
        raw_path = self.tmp_dir / f"corner_{self.n_total}.raw"
        raw_path.touch()
        callback(str(raw_path), None)

    def wait_all(self, timeout=None):
        return self.n_ok == self.n_total

    def cleanup(self):
        pass


class TestRunCornerSweep(unittest.TestCase):
    def setUp(self):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".cir", delete=False)
        f.write(".param R=1k C=10n\nR1 in out {R}\nC1 out 0 {C}\n.end\n")
        f.close()
        self.schematic = Path(f.name)
        self.tmp_dir = Path(tempfile.gettempdir())

    def tearDown(self):
        self.schematic.unlink()

    def test_cartesian_product_and_temperature_included(self):
        engine = MonteCarloEngine(self.schematic)
        engine.enable_temperature_control()
        engine.add_parameter("R", Gaussian(mean=1000, std=10))  # not swept -> held at its mean

        fake = _FakeCornerRunner(self.tmp_dir)
        engine._get_runner = lambda: fake
        engine._read_raw = lambda path, traces: None  # extract_den_fn below ignores `raw` entirely

        # A trivial "extract" that just reads back R*C as if it were tau --
        # den_coeffs shape isn't the point of this test, resolution logic is.
        def extract(raw, resolved):
            return np.array([resolved["R"] * resolved["C"], 1.0])

        result = engine.run_corner_sweep(
            corner_values={"C": [9e-9, 10e-9, 11e-9], "TEMP_C": [-40, 27, 125]},
            extract_den_fn=extract, cleanup_files=False, progress=False,
        )

        self.assertEqual(result.n_points, 9)  # 3 x 3 cartesian product
        self.assertEqual(result.n_success, 9)  # every run must have actually succeeded, not just been called
        self.assertEqual(len(fake.calls), 9)
        temps_seen = sorted({round(c["TEMP_C"]) for c in fake.calls})
        self.assertEqual(temps_seen, [-40, 27, 125])
        # R was registered but not swept -> every call held it at the Gaussian's mean
        self.assertTrue(all(c["R"] == 1000 for c in fake.calls))
        # extract_den_fn's actual output must have made it into the result.
        sample = result.runs[0]
        self.assertAlmostEqual(sample.den_coeffs[0], sample.params["R"] * sample.params["C"])

    def test_enable_temperature_control_adds_temp_instruction(self):
        engine = MonteCarloEngine(self.schematic)
        engine.enable_temperature_control(param_name="MYTEMP")
        self.assertIn(".temp {MYTEMP}", engine._extra_instructions)

    def test_enable_temperature_control_after_runner_created_raises(self):
        engine = MonteCarloEngine(self.schematic)
        engine._runner = object()  # simulate "already created"
        with self.assertRaises(RuntimeError):
            engine.enable_temperature_control()

    def test_fixed_params_override_distribution_center(self):
        engine = MonteCarloEngine(self.schematic)
        engine.add_parameter("R", Gaussian(mean=1000, std=10))
        fake = _FakeCornerRunner(self.tmp_dir)
        engine._get_runner = lambda: fake
        engine._read_raw = lambda path, traces: None

        engine.run_corner_sweep(
            corner_values={"C": [10e-9]},
            extract_den_fn=lambda raw, resolved: np.array([1.0]),
            fixed_params={"R": 4700.0},  # overrides the Gaussian's mean of 1000
            cleanup_files=False, progress=False,
        )
        self.assertEqual(fake.calls[0]["R"], 4700.0)

    def test_failed_run_recorded_not_crashing(self):
        engine = MonteCarloEngine(self.schematic)

        class _FailingRunner(_FakeCornerRunner):
            def run_case(self, param_values, callback):
                self.n_total += 1
                callback(None, None)  # simulate an aborted run

        fake = _FailingRunner(self.tmp_dir)
        engine._get_runner = lambda: fake

        result = engine.run_corner_sweep(
            corner_values={"C": [10e-9, 11e-9]},
            extract_den_fn=lambda raw, resolved: np.array([1.0]),
            cleanup_files=False, progress=False,
        )
        self.assertEqual(result.n_success, 0)
        self.assertTrue(all(r.error is not None for r in result.runs))


if __name__ == "__main__":
    unittest.main(verbosity=2)
