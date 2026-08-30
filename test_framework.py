"""
Tests that validate ltspice_mc's logic WITHOUT requiring spicelib or a real
LTspice install -- everything here runs on plain numpy + stdlib. This covers:

  1. .param parsing (parse_spice_value, discover_parameters)
  2. Sampling distributions (Gaussian, Uniform, Tolerance, rejection sampling)
  3. Metric math (RMS, gain, bandwidth, phase margin) against synthetic
     signals with known analytic answers, via a mock RawRead-like object
  4. The full MonteCarloEngine orchestration loop (sampling -> callback ->
     aggregation -> success/error bookkeeping), using a fake runner and a
     monkeypatched `_read_raw` seam so no LTspice process is ever launched

Running actual LTspice simulations is NOT covered here -- that requires
spicelib + LTspice installed locally, which this environment doesn't have.
"""

import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ltspice_mc.param_utils import discover_parameters, parse_spice_value
from ltspice_mc.distributions import Gaussian, Uniform, Tolerance, Custom
from ltspice_mc.metrics import (
    compute_rms, compute_mean, compute_peak_to_peak, compute_gain_db,
    compute_phase_deg, compute_bandwidth, compute_phase_margin,
    compute_overshoot_pct, compute_settling_time,
)
from ltspice_mc.engine import MonteCarloEngine, MonteCarloResults


# ===========================================================================
# 1. .param parsing
# ===========================================================================

class TestParseSpiceValue(unittest.TestCase):
    def test_plain_numbers(self):
        self.assertAlmostEqual(parse_spice_value("5"), 5.0)
        self.assertAlmostEqual(parse_spice_value("3.3"), 3.3)
        self.assertAlmostEqual(parse_spice_value("-2.5"), -2.5)
        self.assertAlmostEqual(parse_spice_value("1e-3"), 1e-3)

    def test_suffixes(self):
        self.assertAlmostEqual(parse_spice_value("1k"), 1000.0)
        self.assertAlmostEqual(parse_spice_value("2.2Meg"), 2.2e6)
        self.assertAlmostEqual(parse_spice_value("10n"), 10e-9)
        self.assertAlmostEqual(parse_spice_value("3.3u"), 3.3e-6)
        self.assertAlmostEqual(parse_spice_value("1p"), 1e-12)
        self.assertAlmostEqual(parse_spice_value("1T"), 1e12)

    def test_milli_is_not_mega(self):
        # This is the classic SPICE gotcha this parser must get right.
        self.assertAlmostEqual(parse_spice_value("100m"), 0.1)
        self.assertAlmostEqual(parse_spice_value("100M"), 0.1)
        self.assertNotAlmostEqual(parse_spice_value("100m"), 100e6)

    def test_meg_checked_before_m(self):
        self.assertAlmostEqual(parse_spice_value("5Meg"), 5e6)
        self.assertAlmostEqual(parse_spice_value("5MEG"), 5e6)

    def test_embedded_decimal_notation(self):
        # "2k7" style notation (spicelib itself added support for this).
        self.assertAlmostEqual(parse_spice_value("2k7"), 2700.0)
        self.assertAlmostEqual(parse_spice_value("4n7"), 4.7e-9)

    def test_units_suffix_ignored(self):
        self.assertAlmostEqual(parse_spice_value("5V"), 5.0)

    def test_expressions_return_none(self):
        self.assertIsNone(parse_spice_value("{R*2}"))
        self.assertIsNone(parse_spice_value(""))


class TestDiscoverParameters(unittest.TestCase):
    def _write(self, text: str, suffix: str) -> Path:
        f = tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False)
        f.write(text)
        f.close()
        return Path(f.name)

    def test_plain_netlist(self):
        netlist = (
            "* test netlist\n"
            ".param R=1k C=10n\n"
            "R1 in out {R}\n"
            "C1 out 0 {C}\n"
            ".end\n"
        )
        path = self._write(netlist, ".cir")
        try:
            params = discover_parameters(path)
            self.assertIn("R", params)
            self.assertIn("C", params)
            self.assertAlmostEqual(params["R"].nominal, 1000.0)
            self.assertAlmostEqual(params["C"].nominal, 10e-9)
            # crucially: component designators must NOT show up as parameters
            self.assertNotIn("R1", params)
            self.assertNotIn("C1", params)
        finally:
            path.unlink()

    def test_asc_style_directive(self):
        asc_text = (
            "Version 4\n"
            "SHEET 1 880 680\n"
            "SYMBOL res 96 64 R90\n"
            "SYMATTR InstName R1\n"
            "SYMATTR Value {R}\n"
            "TEXT 80 280 Left 2 !.param R=1k C=10n\n"
            "TEXT 80 320 Left 2 !.ac dec 100 1 1Meg\n"
        )
        path = self._write(asc_text, ".asc")
        try:
            params = discover_parameters(path)
            self.assertIn("R", params)
            self.assertIn("C", params)
            self.assertAlmostEqual(params["R"].nominal, 1000.0)
            self.assertNotIn("R1", params)
        finally:
            path.unlink()

    def test_commented_param_is_ignored(self):
        netlist = "* .param R=999\n.param R=1k\n"
        path = self._write(netlist, ".cir")
        try:
            params = discover_parameters(path)
            self.assertAlmostEqual(params["R"].nominal, 1000.0)
        finally:
            path.unlink()


# ===========================================================================
# 2. Distributions
# ===========================================================================

class TestDistributions(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(12345)

    def test_gaussian_statistics(self):
        d = Gaussian(mean=100.0, std=5.0)
        samples = d.sample(200_000, self.rng)
        self.assertAlmostEqual(np.mean(samples), 100.0, delta=0.1)
        self.assertAlmostEqual(np.std(samples), 5.0, delta=0.1)

    def test_uniform_statistics(self):
        d = Uniform(low=10.0, high=20.0)
        samples = d.sample(200_000, self.rng)
        self.assertGreaterEqual(samples.min(), 10.0)
        self.assertLessEqual(samples.max(), 20.0)
        self.assertAlmostEqual(np.mean(samples), 15.0, delta=0.05)

    def test_rejection_sampling_respects_bounds(self):
        # Deliberately huge std relative to bounds, forcing many rejections.
        d = Gaussian(mean=0.0, std=100.0, min_value=-1.0, max_value=1.0)
        samples = d.sample(5000, self.rng)
        self.assertTrue(np.all(samples >= -1.0))
        self.assertTrue(np.all(samples <= 1.0))

    def test_tolerance_uniform(self):
        d = Tolerance(nominal=1000.0, tolerance_pct=1.0, kind="uniform")
        samples = d.sample(50_000, self.rng)
        self.assertGreaterEqual(samples.min(), 990.0 - 1e-6)
        self.assertLessEqual(samples.max(), 1010.0 + 1e-6)

    def test_tolerance_gaussian_three_sigma(self):
        d = Tolerance(nominal=1000.0, tolerance_pct=3.0, kind="gaussian", sigma_multiple=3.0)
        # 3% of 1000 = 30, over 3 sigma -> sigma = 10
        samples = d.sample(200_000, self.rng)
        self.assertAlmostEqual(np.std(samples), 10.0, delta=0.2)

    def test_custom_distribution(self):
        d = Custom(lambda n, rng: rng.exponential(scale=2.0, size=n))
        samples = d.sample(100_000, self.rng)
        self.assertGreater(samples.min(), -1e-9)
        self.assertAlmostEqual(np.mean(samples), 2.0, delta=0.05)

    def test_negative_bounds_raises(self):
        with self.assertRaises(ValueError):
            Uniform(low=10, high=5)


# ===========================================================================
# 3. Metrics, via a mock RawRead
# ===========================================================================

class _MockTrace:
    def __init__(self, wave):
        self._wave = np.asarray(wave)

    def get_wave(self, step: int = 0):
        return self._wave


class MockRawRead:
    """Duck-typed stand-in for spicelib.raw.raw_read.RawRead."""

    def __init__(self, axis, traces):
        self._axis = np.asarray(axis)
        self._traces = {name: _MockTrace(wave) for name, wave in traces.items()}

    def get_axis(self, step: int = 0):
        return self._axis

    def get_trace(self, name):
        return self._traces[name]


class TestTimeDomainMetrics(unittest.TestCase):
    def test_rms_of_sine_wave(self):
        t = np.linspace(0, 1.0, 200_000)
        amplitude = 3.0
        y = amplitude * np.sin(2 * np.pi * 50 * t)
        raw = MockRawRead(t, {"V(out)": y})
        rms = compute_rms(raw, "V(out)")
        self.assertAlmostEqual(rms, amplitude / math.sqrt(2), delta=0.01)

    def test_rms_windowing_excludes_transient(self):
        t = np.linspace(0, 2.0, 400_000)
        # A decaying transient added on top of a steady sine -- windowing
        # away the first second should recover the steady-state RMS.
        y = 2.0 * np.sin(2 * np.pi * 10 * t) + 5.0 * np.exp(-t * 20)
        raw = MockRawRead(t, {"V(out)": y})
        rms_windowed = compute_rms(raw, "V(out)", t_start=1.0)
        self.assertAlmostEqual(rms_windowed, 2.0 / math.sqrt(2), delta=0.02)

    def test_mean_of_dc_signal(self):
        t = np.linspace(0, 1.0, 1000)
        y = np.full_like(t, 4.2)
        raw = MockRawRead(t, {"V(out)": y})
        self.assertAlmostEqual(compute_mean(raw, "V(out)"), 4.2, places=6)

    def test_peak_to_peak(self):
        t = np.linspace(0, 1.0, 1000)
        y = np.sin(2 * np.pi * 5 * t) * 2.0 + 1.0
        raw = MockRawRead(t, {"V(out)": y})
        self.assertAlmostEqual(compute_peak_to_peak(raw, "V(out)"), 4.0, delta=0.01)

    def test_rms_rejects_complex_data(self):
        t = np.linspace(0, 1, 10)
        raw = MockRawRead(t, {"V(out)": (1 + 1j) * np.ones(10)})
        with self.assertRaises(ValueError):
            compute_rms(raw, "V(out)")

    def test_overshoot_and_settling_time(self):
        t = np.linspace(0, 1.0, 100_000)
        tau = 0.05
        # Second-order-ish step response with overshoot to 1.15, settling to 1.0
        y = 1.0 + 0.20 * np.exp(-t / tau) * np.cos(2 * np.pi * 8 * t)
        raw = MockRawRead(t, {"V(out)": y})
        overshoot = compute_overshoot_pct(raw, "V(out)", final_value=1.0)
        self.assertGreater(overshoot, 5.0)
        settling = compute_settling_time(raw, "V(out)", tolerance_pct=2.0, final_value=1.0)
        self.assertTrue(0.0 < settling < 1.0)


class TestFrequencyDomainMetrics(unittest.TestCase):
    """
    Validate gain/bandwidth/phase-margin against a synthetic single-pole
    RC low-pass transfer function H(f) = 1 / (1 + j f / fc), computed
    analytically -- i.e. this tests the *math*, independent of LTspice.
    """

    def setUp(self):
        self.fc = 1591.5  # ~ R=1k, C=100n
        self.freq = np.logspace(0, 6, 4000)  # 1 Hz to 1 MHz, dense grid
        h = 1.0 / (1.0 + 1j * self.freq / self.fc)
        self.raw = MockRawRead(self.freq, {"V(in)": np.ones_like(self.freq, dtype=complex), "V(out)": h})

    def test_gain_at_dc_is_0db(self):
        gain_dc = compute_gain_db(self.raw, "V(out)", "V(in)", at_freq=1.0)
        self.assertAlmostEqual(gain_dc, 0.0, delta=0.01)

    def test_gain_at_corner_is_minus_3db(self):
        gain_fc = compute_gain_db(self.raw, "V(out)", "V(in)", at_freq=self.fc)
        self.assertAlmostEqual(gain_fc, -3.0103, delta=0.05)

    def test_bandwidth_matches_analytic_corner(self):
        bw = compute_bandwidth(self.raw, "V(out)", "V(in)")
        self.assertAlmostEqual(bw, self.fc, delta=self.fc * 0.02)

    def test_phase_at_corner_is_minus_45deg(self):
        phase_fc = compute_phase_deg(self.raw, "V(out)", "V(in)", at_freq=self.fc)
        self.assertAlmostEqual(phase_fc, -45.0, delta=0.5)

    def test_gain_rejects_real_data(self):
        raw = MockRawRead(self.freq, {"V(out)": np.ones_like(self.freq)})
        with self.assertRaises(ValueError):
            compute_gain_db(raw, "V(out)")

    def test_phase_margin_matches_closed_form_single_pole(self):
        # Loop gain L(f) = A0 / (1 + j f/fp): a single dominant pole should
        # give a phase margin approaching 90 degrees for high DC gain A0.
        fp = 1000.0
        A0 = 100.0  # 40 dB
        freq = np.logspace(0, 8, 20000)
        loop_gain = A0 / (1.0 + 1j * freq / fp)
        raw = MockRawRead(freq, {"LOOPGAIN": loop_gain})

        pm = compute_phase_margin(raw, "LOOPGAIN")

        # Independent closed-form check: solve |L(f)| = 1 analytically,
        # then PM = 180 + phase(f_cross) = 180 - arctan(f_cross / fp).
        f_cross_analytic = fp * math.sqrt(A0 ** 2 - 1)
        phase_analytic = -math.degrees(math.atan(f_cross_analytic / fp))
        pm_analytic = 180.0 + phase_analytic

        self.assertAlmostEqual(pm, pm_analytic, delta=0.5)
        # Sanity: for a single dominant pole, PM should be close to 90 deg.
        self.assertAlmostEqual(pm, 90.0, delta=2.0)

    def test_phase_margin_no_crossing_returns_nan(self):
        # Loop gain that never reaches 0 dB.
        freq = np.logspace(0, 6, 1000)
        loop_gain = 0.1 * np.ones_like(freq, dtype=complex)
        raw = MockRawRead(freq, {"LOOPGAIN": loop_gain})
        self.assertTrue(math.isnan(compute_phase_margin(raw, "LOOPGAIN")))


# ===========================================================================
# 4. Full engine orchestration, with a fake runner (no spicelib/LTspice)
# ===========================================================================

class _FakeLTspiceRunner:
    """
    Stands in for ltspice_mc.runner.LTspiceRunner. `outcomes` is a list of
    'ok' or 'fail', one per expected run_case() call, consumed in order.

    A real completed LTspice run leaves an actual .raw file on disk, and
    engine.py's callback deliberately checks for that (see the "missing .raw
    file" branch) -- so 'ok' outcomes here must create a real (even if
    empty) file for that check to pass, just like production.
    """

    def __init__(self, outcomes, tmp_dir: Path):
        self.outcomes = list(outcomes)
        self.tmp_dir = tmp_dir
        self._i = 0
        self.n_ok = 0
        self.n_total = 0
        self.cleanup_called = False

    def run_case(self, param_values, callback):
        outcome = self.outcomes[self._i]
        self._i += 1
        self.n_total += 1
        if outcome == "ok":
            self.n_ok += 1
            raw_path = self.tmp_dir / f"fake_run_{self._i}.raw"
            log_path = self.tmp_dir / f"fake_run_{self._i}.log"
            raw_path.touch()
            log_path.touch()
            callback(str(raw_path), str(log_path))
        else:
            callback(None, None)

    def wait_all(self, timeout=None):
        return self.n_ok == self.n_total

    def cleanup(self):
        self.cleanup_called = True


class TestMonteCarloEngine(unittest.TestCase):
    def setUp(self):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".cir", delete=False)
        f.write(".param R=1k C=10n\nR1 in out {R}\nC1 out 0 {C}\n.end\n")
        f.close()
        self.schematic = Path(f.name)

    def tearDown(self):
        self.schematic.unlink()

    def test_discovers_parameters_on_init(self):
        engine = MonteCarloEngine(self.schematic)
        self.assertIn("R", engine.discovered_parameters)
        self.assertIn("C", engine.discovered_parameters)

    def test_run_without_parameters_raises(self):
        engine = MonteCarloEngine(self.schematic)
        engine.add_metric("m", lambda raw: 0.0)
        with self.assertRaises(ValueError):
            engine.run(n_samples=10)

    def test_run_without_metrics_raises(self):
        engine = MonteCarloEngine(self.schematic)
        engine.add_parameter("R", Gaussian(1000, 10))
        with self.assertRaises(ValueError):
            engine.run(n_samples=10)

    def test_undeclared_parameter_warns_not_raises(self):
        engine = MonteCarloEngine(self.schematic)
        with self.assertLogs("ltspice_mc.engine", level="WARNING"):
            engine.add_parameter("NOT_A_REAL_PARAM", Gaussian(1, 0.1))
        # Should not raise -- just warns.
        self.assertIn("NOT_A_REAL_PARAM", engine._param_specs)

    def test_full_pipeline_all_success(self):
        engine = MonteCarloEngine(self.schematic)
        engine.add_parameter("R", Gaussian(mean=1000, std=10))
        engine.add_parameter("C", Uniform(9e-9, 11e-9))
        engine.add_metric("dummy_metric", lambda raw: 42.0)

        n = 25
        engine._get_runner = lambda: _FakeLTspiceRunner(["ok"] * n, Path(tempfile.gettempdir()))
        engine._read_raw = lambda path, traces: MockRawRead([0], {"unused": [0]})

        results = engine.run(n_samples=n, seed=7, cleanup_files=False, progress=False)

        self.assertIsInstance(results, MonteCarloResults)
        self.assertEqual(results.n_samples, n)
        self.assertEqual(results.n_success, n)
        self.assertAlmostEqual(results.yield_pct, 100.0)
        self.assertTrue(np.all(results.metrics["dummy_metric"] == 42.0))
        self.assertEqual(len(results.parameters["R"]), n)
        self.assertEqual(len(results.parameters["C"]), n)

    def test_full_pipeline_partial_failure(self):
        engine = MonteCarloEngine(self.schematic)
        engine.add_parameter("R", Gaussian(mean=1000, std=10))
        engine.add_metric("dummy_metric", lambda raw: 1.0)

        n = 20
        outcomes = ["ok" if i % 4 != 0 else "fail" for i in range(n)]  # 25% fail
        engine._get_runner = lambda: _FakeLTspiceRunner(outcomes, Path(tempfile.gettempdir()))
        engine._read_raw = lambda path, traces: MockRawRead([0], {"unused": [0]})

        results = engine.run(n_samples=n, seed=1, cleanup_files=False, progress=False)

        expected_ok = outcomes.count("ok")
        self.assertEqual(results.n_success, expected_ok)
        self.assertEqual(sum(1 for e in results.errors if e is not None), n - expected_ok)
        # Failed runs must show up as NaN, not as 0 or garbage.
        self.assertEqual(np.sum(np.isnan(results.metrics["dummy_metric"])), n - expected_ok)

    def test_metric_exception_marks_run_as_failed_not_crash(self):
        engine = MonteCarloEngine(self.schematic)
        engine.add_parameter("R", Gaussian(mean=1000, std=10))

        def flaky_metric(raw):
            raise RuntimeError("simulated bad node lookup")

        engine.add_metric("flaky", flaky_metric)
        n = 5
        engine._get_runner = lambda: _FakeLTspiceRunner(["ok"] * n, Path(tempfile.gettempdir()))
        engine._read_raw = lambda path, traces: MockRawRead([0], {"unused": [0]})

        # Must not raise -- a single bad metric computation should be caught
        # and recorded as a per-run failure, not crash the whole batch.
        results = engine.run(n_samples=n, seed=3, cleanup_files=False, progress=False)
        self.assertEqual(results.n_success, 0)
        self.assertEqual(sum(1 for e in results.errors if e is not None), n)

    def test_compute_yield_and_summary(self):
        results = MonteCarloResults(
            n_samples=10,
            parameters={"R": np.linspace(990, 1010, 10)},
            metrics={"bw": np.array([1500, 1550, 1600, 1650, np.nan, 1700, 1400, 1750, 1300, 1620])},
            success=np.array([True] * 4 + [False] + [True] * 5),
            errors=[None] * 4 + ["boom"] + [None] * 5,
        )
        y = results.compute_yield("bw", lower=1450, upper=1700)
        # valid (successful, non-nan) values: 1500,1550,1600,1650,1700,1400,1750,1300,1620 (9 values)
        # within [1450,1700]: 1500,1550,1600,1650,1700,1620 -> 6 of 9
        self.assertAlmostEqual(y, 6 / 9 * 100, places=3)
        summary_text = results.summary()
        self.assertIn("bw", summary_text)
        self.assertIn("9/10", summary_text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
