"""
Tests for netlist_parameterizer.py -- component detection, interactive
prompting (via injected fake input/print, no real stdin needed), and file
rewriting for both plain netlists and .asc schematics. All pure text/logic,
no spicelib or LTspice required.
"""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ltspice_mc.param_utils import discover_parameters
from ltspice_mc.distributions import Uniform, Gaussian
from ltspice_mc.netlist_parameterizer import (
    detect_components_in_netlist, detect_components_in_asc,
    prompt_for_choices, build_distribution,
    apply_to_netlist, apply_to_asc, interactive_parameterize,
    DetectedComponent, MonteCarloChoice,
)


def _write(text: str, suffix: str) -> Path:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False)
    f.write(text)
    f.close()
    return Path(f.name)


class ScriptedIO:
    """Fake input()/print() pair: feeds a scripted list of answers, records
    every prompt actually shown so tests can assert on the conversation."""

    def __init__(self, answers):
        self._answers = list(answers)
        self.prompts = []
        self.printed = []

    def input_fn(self, prompt):
        self.prompts.append(prompt)
        if not self._answers:
            raise AssertionError(f"ScriptedIO ran out of answers at prompt: {prompt!r}")
        return self._answers.pop(0)

    def print_fn(self, msg):
        self.printed.append(msg)


# ===========================================================================
# Detection: plain netlist
# ===========================================================================

class TestDetectNetlist(unittest.TestCase):
    def test_basic_rcl_detected(self):
        text = (
            "* test\n"
            "V1 in 0 AC 1\n"
            "R1 in mid 10k\n"
            "C1 mid out 100n\n"
            "L1 out 0 1m\n"
            "D1 out 0 DMOD\n"
            ".model DMOD D\n"
            ".end\n"
        )
        comps = detect_components_in_netlist(text)
        by_ref = {c.ref: c for c in comps}
        self.assertEqual(set(by_ref), {"R1", "C1", "L1"})  # V1, D1 excluded
        self.assertEqual(by_ref["R1"].kind, "resistor")
        self.assertAlmostEqual(by_ref["R1"].nominal, 10000.0)
        self.assertEqual(by_ref["R1"].nodes, ["in", "mid"])
        self.assertEqual(by_ref["C1"].kind, "capacitor")
        self.assertAlmostEqual(by_ref["C1"].nominal, 100e-9)
        self.assertEqual(by_ref["L1"].kind, "inductor")
        self.assertAlmostEqual(by_ref["L1"].nominal, 1e-3)
        self.assertTrue(all(c.supported for c in comps))
        self.assertTrue(all(not c.already_parameterized for c in comps))

    def test_already_parameterized_is_flagged_not_reconverted(self):
        text = ".param R=1k\nR1 in out {R}\n"
        comps = detect_components_in_netlist(text)
        self.assertEqual(len(comps), 1)
        self.assertTrue(comps[0].already_parameterized)
        self.assertIsNone(comps[0].nominal)

    def test_model_referencing_resistor_marked_unsupported(self):
        # 3-terminal semiconductor resistor -- deliberately out of scope;
        # must be flagged, not silently mis-converted.
        text = "R1 n1 n2 n3 RMOD L=1u W=2u\n"
        comps = detect_components_in_netlist(text)
        self.assertEqual(len(comps), 1)
        self.assertFalse(comps[0].supported)

    def test_inductor_with_ic_trailing_preserved_in_detection(self):
        text = "L1 out 0 10u IC=0\n"
        comps = detect_components_in_netlist(text)
        self.assertEqual(len(comps), 1)
        self.assertAlmostEqual(comps[0].nominal, 10e-6)

    def test_comments_and_directives_not_matched(self):
        text = "* R1 fake comment 10k\n.param foo=1\n+ continuation\n"
        comps = detect_components_in_netlist(text)
        self.assertEqual(len(comps), 0)

    def test_kinds_filter(self):
        text = "R1 a b 1k\nC1 a b 1n\nL1 a b 1u\n"
        comps = detect_components_in_netlist(text, kinds=("R",))
        self.assertEqual([c.ref for c in comps], ["R1"])


# ===========================================================================
# Detection: .asc schematic
# ===========================================================================

class TestDetectAsc(unittest.TestCase):
    def test_basic_asc_detection(self):
        text = (
            "Version 4\nSHEET 1 880 680\n"
            "SYMBOL res 96 64 R90\n"
            "SYMATTR InstName R1\n"
            "SYMATTR Value 10k\n"
            "SYMBOL cap 224 128 R0\n"
            "SYMATTR InstName C1\n"
            "SYMATTR Value 100n\n"
            "SYMBOL diode 300 200 R0\n"
            "SYMATTR InstName D1\n"
            "SYMATTR Value 1N4148\n"
        )
        comps = detect_components_in_asc(text)
        by_ref = {c.ref: c for c in comps}
        self.assertEqual(set(by_ref), {"R1", "C1"})
        self.assertAlmostEqual(by_ref["R1"].nominal, 10000.0)
        self.assertAlmostEqual(by_ref["C1"].nominal, 100e-9)

    def test_attribute_order_independent(self):
        # SYMATTR Value appearing before SYMATTR InstName must still work.
        text = "SYMBOL res 0 0 R0\nSYMATTR Value 2.2k\nSYMATTR InstName R7\n"
        comps = detect_components_in_asc(text)
        self.assertEqual(len(comps), 1)
        self.assertEqual(comps[0].ref, "R7")
        self.assertAlmostEqual(comps[0].nominal, 2200.0)

    def test_missing_value_attribute_is_supported_and_asks(self):
        # This is exactly what happens when a component is placed in
        # LTspice but its value was never explicitly set -- LTspice simply
        # doesn't write a SYMATTR Value line at all. Nothing to preserve,
        # so this must be treated as fully supported (just no default to
        # pre-fill), not skipped as "unsupported".
        text = "SYMBOL res 0 0 R0\nSYMATTR InstName R1\n"
        comps = detect_components_in_asc(text)
        self.assertEqual(len(comps), 1)
        self.assertTrue(comps[0].supported)
        self.assertIsNone(comps[0].nominal)
        self.assertFalse(comps[0].already_parameterized)
        self.assertEqual(comps[0].raw_value, "")


# ===========================================================================
# Interactive prompting (scripted, no real stdin)
# ===========================================================================

class TestPromptForChoices(unittest.TestCase):
    def test_full_walkthrough_include_with_overrides(self):
        comps = [DetectedComponent(ref="R1", kind="resistor", nodes=["a", "b"], raw_value="10k",
                                    nominal=10000.0, already_parameterized=False, supported=True, line_index=0)]
        io = ScriptedIO(answers=["y", "", "2", "gaussian"])  # include, keep nominal, 2%, gaussian
        choices = prompt_for_choices(comps, input_fn=io.input_fn, print_fn=io.print_fn)
        self.assertEqual(len(choices), 1)
        c = choices[0]
        self.assertTrue(c.include)
        self.assertAlmostEqual(c.nominal, 10000.0)
        self.assertAlmostEqual(c.tolerance_pct, 2.0)
        self.assertEqual(c.distribution, "gaussian")

    def test_nominal_override(self):
        comps = [DetectedComponent(ref="R1", kind="resistor", nodes=[], raw_value="10k",
                                    nominal=10000.0, already_parameterized=False, supported=True, line_index=0)]
        io = ScriptedIO(answers=["y", "4.7k", "5", ""])  # override nominal to 4.7k, default distribution
        choices = prompt_for_choices(comps, input_fn=io.input_fn, print_fn=io.print_fn)
        self.assertAlmostEqual(choices[0].nominal, 4700.0)
        self.assertEqual(choices[0].distribution, "uniform")  # default on blank

    def test_exclude_skips_remaining_questions(self):
        comps = [DetectedComponent(ref="R1", kind="resistor", nodes=[], raw_value="10k",
                                    nominal=10000.0, already_parameterized=False, supported=True, line_index=0)]
        io = ScriptedIO(answers=["n"])  # only ONE answer needed -- include=No short-circuits
        choices = prompt_for_choices(comps, input_fn=io.input_fn, print_fn=io.print_fn)
        self.assertEqual(len(choices), 1)
        self.assertFalse(choices[0].include)

    def test_invalid_input_reprompts(self):
        comps = [DetectedComponent(ref="R1", kind="resistor", nodes=[], raw_value="10k",
                                    nominal=10000.0, already_parameterized=False, supported=True, line_index=0)]
        io = ScriptedIO(answers=["maybe", "y", "notanumber", "1k", "5", "uniform"])
        choices = prompt_for_choices(comps, input_fn=io.input_fn, print_fn=io.print_fn)
        self.assertTrue(choices[0].include)
        self.assertAlmostEqual(choices[0].nominal, 1000.0)

    def test_already_parameterized_and_unsupported_not_prompted(self):
        comps = [
            DetectedComponent(ref="R1", kind="resistor", nodes=[], raw_value="{R1}", nominal=None,
                               already_parameterized=True, supported=True, line_index=0),
            DetectedComponent(ref="R2", kind="resistor", nodes=[], raw_value="RMOD", nominal=None,
                               already_parameterized=False, supported=False, reason_unsupported="model", line_index=1),
        ]
        io = ScriptedIO(answers=[])  # must not be asked anything at all
        choices = prompt_for_choices(comps, input_fn=io.input_fn, print_fn=io.print_fn)
        self.assertEqual(choices, [])


class TestBuildDistribution(unittest.TestCase):
    def test_uniform_from_tolerance(self):
        choice = MonteCarloChoice(ref="R1", include=True, nominal=1000.0, tolerance_pct=5.0, distribution="uniform")
        dist = build_distribution(choice)
        from ltspice_mc.distributions import Tolerance
        self.assertIsInstance(dist, Tolerance)
        samples = dist.sample(10000, __import__("numpy").random.default_rng(0))
        self.assertGreaterEqual(samples.min(), 950.0 - 1e-6)
        self.assertLessEqual(samples.max(), 1050.0 + 1e-6)


# ===========================================================================
# Applying: netlist rewrite
# ===========================================================================

class TestApplyToNetlist(unittest.TestCase):
    def test_rewrite_replaces_value_and_adds_param(self):
        text = (
            "* my circuit\n"
            "V1 in 0 AC 1\n"
            "R1 in mid 10k\n"
            "C1 mid out 100n\n"
            "L1 out 0 1m IC=0\n"
            ".ac dec 100 1 1Meg\n"
            ".end\n"
        )
        path = _write(text, ".cir")
        try:
            comps = detect_components_in_netlist(text)
            choices = [
                MonteCarloChoice(ref="R1", include=True, nominal=10000.0, tolerance_pct=1.0, distribution="uniform"),
                MonteCarloChoice(ref="C1", include=True, nominal=100e-9, tolerance_pct=10.0, distribution="gaussian"),
                MonteCarloChoice(ref="L1", include=False, nominal=1e-3),
            ]
            result = apply_to_netlist(path, comps, choices)
            new_text = result.output_path.read_text()

            self.assertIn("R1 in mid {R1}", new_text)
            self.assertIn("C1 mid out {C1}", new_text)
            self.assertIn("L1 out 0 1m IC=0", new_text)  # excluded -> untouched, IC preserved
            self.assertIn("V1 in 0 AC 1", new_text)      # untouched line
            self.assertIn(".param R1=10000", new_text)
            self.assertIn(".param C1=1e-07", new_text)
            self.assertEqual(set(result.distributions), {"R1", "C1"})
            self.assertEqual(len(result.skipped), 1)
            self.assertEqual(result.skipped[0].ref, "L1")
        finally:
            path.unlink()
            result.output_path.unlink(missing_ok=True)

    def test_unrelated_lines_byte_identical(self):
        text = "* header\nV1 in 0 AC 1\nR1 in out 10k\n.ac dec 10 1 1k\n.end\n"
        path = _write(text, ".cir")
        try:
            comps = detect_components_in_netlist(text)
            choices = [MonteCarloChoice(ref="R1", include=True, nominal=10000.0, tolerance_pct=5.0)]
            result = apply_to_netlist(path, comps, choices)
            new_lines = result.output_path.read_text().splitlines()
            old_lines = text.splitlines()
            for line in old_lines:
                if not line.startswith("R1"):
                    self.assertIn(line, new_lines)
        finally:
            path.unlink()
            result.output_path.unlink(missing_ok=True)

    def test_output_is_discoverable_by_param_utils(self):
        """Round-trip: the file this module writes must be readable by the
        existing discover_parameters() so it plugs straight into MonteCarloEngine."""
        text = "R1 in out 10k\nC1 out 0 100n\n.end\n"
        path = _write(text, ".cir")
        try:
            comps = detect_components_in_netlist(text)
            choices = [
                MonteCarloChoice(ref="R1", include=True, nominal=10000.0, tolerance_pct=5.0),
                MonteCarloChoice(ref="C1", include=True, nominal=100e-9, tolerance_pct=10.0),
            ]
            result = apply_to_netlist(path, comps, choices)
            discovered = discover_parameters(result.output_path)
            self.assertIn("R1", discovered)
            self.assertIn("C1", discovered)
            self.assertAlmostEqual(discovered["R1"].nominal, 10000.0)
            self.assertAlmostEqual(discovered["C1"].nominal, 100e-9)
        finally:
            path.unlink()
            result.output_path.unlink(missing_ok=True)

    def test_inserts_after_existing_param_block(self):
        text = ".param VDD=5\nR1 in out 10k\n.end\n"
        path = _write(text, ".cir")
        try:
            comps = detect_components_in_netlist(text)
            choices = [MonteCarloChoice(ref="R1", include=True, nominal=10000.0, tolerance_pct=5.0)]
            result = apply_to_netlist(path, comps, choices)
            lines = result.output_path.read_text().splitlines()
            vdd_idx = next(i for i, l in enumerate(lines) if l.startswith(".param VDD"))
            r1_idx = next(i for i, l in enumerate(lines) if l.startswith(".param R1"))
            self.assertEqual(r1_idx, vdd_idx + 1)
        finally:
            path.unlink()
            result.output_path.unlink(missing_ok=True)


# ===========================================================================
# Applying: .asc rewrite
# ===========================================================================

class TestApplyToAsc(unittest.TestCase):
    def test_rewrite_symattr_and_adds_directive(self):
        text = (
            "Version 4\nSHEET 1 880 680\n"
            "SYMBOL res 96 64 R90\nSYMATTR InstName R1\nSYMATTR Value 10k\n"
            "SYMBOL cap 224 128 R0\nSYMATTR InstName C1\nSYMATTR Value 100n\n"
        )
        path = _write(text, ".asc")
        try:
            comps = detect_components_in_asc(text)
            choices = [
                MonteCarloChoice(ref="R1", include=True, nominal=10000.0, tolerance_pct=1.0),
                MonteCarloChoice(ref="C1", include=True, nominal=100e-9, tolerance_pct=10.0),
            ]
            result = apply_to_asc(path, comps, choices)
            new_text = result.output_path.read_text()
            self.assertIn("SYMATTR Value {R1}", new_text)
            self.assertIn("SYMATTR Value {C1}", new_text)
            self.assertIn("R1=10000", new_text)
            self.assertIn("C1=1e-07", new_text)
            self.assertIn("!.param", new_text)

            discovered = discover_parameters(result.output_path)
            self.assertIn("R1", discovered)
            self.assertIn("C1", discovered)
        finally:
            path.unlink()
            result.output_path.unlink(missing_ok=True)

    def test_extends_existing_param_directive_in_place(self):
        text = (
            "Version 4\nSHEET 1 880 680\n"
            "SYMBOL res 96 64 R90\nSYMATTR InstName R1\nSYMATTR Value 10k\n"
            "TEXT 80 280 Left 2 !.param VDD=5\n"
        )
        path = _write(text, ".asc")
        try:
            comps = detect_components_in_asc(text)
            choices = [MonteCarloChoice(ref="R1", include=True, nominal=10000.0, tolerance_pct=1.0)]
            result = apply_to_asc(path, comps, choices)
            new_lines = result.output_path.read_text().splitlines()
            param_lines = [l for l in new_lines if "!.param" in l]
            self.assertEqual(len(param_lines), 1)  # extended, not duplicated
            self.assertIn("VDD=5", param_lines[0])
            self.assertIn("R1=10000", param_lines[0])
        finally:
            path.unlink()
            result.output_path.unlink(missing_ok=True)

    def test_new_directive_placed_below_existing_geometry(self):
        text = "Version 4\nSHEET 1 880 680\nSYMBOL res 96 64 R90\nSYMATTR InstName R1\nSYMATTR Value 10k\nFLAG 200 300 out\n"
        path = _write(text, ".asc")
        try:
            comps = detect_components_in_asc(text)
            choices = [MonteCarloChoice(ref="R1", include=True, nominal=10000.0, tolerance_pct=1.0)]
            result = apply_to_asc(path, comps, choices)
            new_lines = result.output_path.read_text().splitlines()
            text_line = next(l for l in new_lines if l.startswith("TEXT") and "!.param" in l)
            y = int(text_line.split()[2])
            self.assertGreater(y, 300)  # below the FLAG at y=300 and SHEET height 680... actually max coordinate wins
        finally:
            path.unlink()
            result.output_path.unlink(missing_ok=True)

    def test_inserts_new_value_line_when_none_existed(self):
        text = (
            "Version 4\nSHEET 1 880 680\n"
            "SYMBOL res 304 48 R90\nSYMATTR InstName R1\n"
        )
        path = _write(text, ".asc")
        try:
            comps = detect_components_in_asc(text)
            self.assertEqual(comps[0].line_index, -1)  # no existing Value line
            choices = [MonteCarloChoice(ref="R1", include=True, nominal=10000.0, tolerance_pct=5.0)]
            result = apply_to_asc(path, comps, choices)
            new_text = result.output_path.read_text()
            self.assertIn("SYMATTR InstName R1", new_text)
            self.assertIn("SYMATTR Value {R1}", new_text)
            lines = new_text.splitlines()
            inst_idx = next(i for i, l in enumerate(lines) if "InstName R1" in l)
            self.assertIn("SYMATTR Value {R1}", lines[inst_idx + 1])
            discovered = discover_parameters(result.output_path)
            self.assertIn("R1", discovered)
            self.assertAlmostEqual(discovered["R1"].nominal, 10000.0)
        finally:
            path.unlink()
            result.output_path.unlink(missing_ok=True)

    def test_multiple_missing_values_processed_safely_reproduces_bug_report(self):
        # Mirrors the actual reported case: R1 and C1 both placed with no
        # value ever set. Both need a brand-new Value line inserted, which
        # shifts line numbers -- this specifically tests that processing
        # order doesn't corrupt the second (or later) insertion.
        text = (
            "Version 4\nSHEET 1 880 680\n"
            "SYMBOL voltage 48 112 R0\nSYMATTR Value2 AC 1\nSYMATTR Value \"\"\nSYMATTR InstName V1\n"
            "SYMBOL res 304 48 R90\nSYMATTR InstName R1\n"
            "SYMBOL cap 368 144 R0\nSYMATTR InstName C1\n"
        )
        path = _write(text, ".asc")
        try:
            comps = detect_components_in_asc(text)
            by_ref = {c.ref: c for c in comps}
            self.assertEqual(set(by_ref), {"R1", "C1"})  # V1 excluded (not R/C/L)
            self.assertTrue(by_ref["R1"].supported and by_ref["R1"].nominal is None)
            self.assertTrue(by_ref["C1"].supported and by_ref["C1"].nominal is None)

            choices = [
                MonteCarloChoice(ref="R1", include=True, nominal=10000.0, tolerance_pct=5.0),
                MonteCarloChoice(ref="C1", include=True, nominal=100e-9, tolerance_pct=10.0),
            ]
            result = apply_to_asc(path, comps, choices)
            new_text = result.output_path.read_text()
            lines = new_text.splitlines()

            r1_inst = next(i for i, l in enumerate(lines) if "InstName R1" in l)
            c1_inst = next(i for i, l in enumerate(lines) if "InstName C1" in l)
            self.assertIn("SYMATTR Value {R1}", lines[r1_inst + 1])
            self.assertIn("SYMATTR Value {C1}", lines[c1_inst + 1])
            self.assertIn('SYMATTR Value ""', new_text)
            self.assertIn("SYMATTR Value2 AC 1", new_text)

            discovered = discover_parameters(result.output_path)
            self.assertAlmostEqual(discovered["R1"].nominal, 10000.0)
            self.assertAlmostEqual(discovered["C1"].nominal, 100e-9)
        finally:
            path.unlink()
            result.output_path.unlink(missing_ok=True)


# ===========================================================================
# Full one-call orchestration
# ===========================================================================

class TestInteractiveParameterize(unittest.TestCase):
    def test_end_to_end_netlist(self):
        text = "R1 in out 10k\nC1 out 0 100n\n.ac dec 100 1 1Meg\n.end\n"
        path = _write(text, ".cir")
        io = ScriptedIO(answers=[
            "y", "", "1", "uniform",     # R1: include, keep nominal, 1%, uniform
            "y", "", "10", "gaussian",   # C1: include, keep nominal, 10%, gaussian
        ])
        try:
            result = interactive_parameterize(path, input_fn=io.input_fn, print_fn=io.print_fn)
            self.assertTrue(result.output_path.exists())
            self.assertEqual(set(result.distributions), {"R1", "C1"})
            discovered = discover_parameters(result.output_path)
            self.assertIn("R1", discovered)
            self.assertIn("C1", discovered)
        finally:
            path.unlink()
            result.output_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
