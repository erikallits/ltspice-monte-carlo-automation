"""
netlist_parameterizer.py
--------------------------
Automatic detection of passive (R/C/L) components in a plain SPICE netlist
or LTspice .asc schematic, and interactive conversion of their values into
.param-driven form -- so nobody has to hand-edit {R1}-style parameter
references or write .param lines themselves.

This is a *preprocessing* step, run once per circuit. It produces a new
netlist/schematic file plus a ready dict of Distribution objects keyed by
reference designator. MonteCarloEngine itself still knows nothing about
component designators -- it only ever consumes the resulting .param names
(which, by the convention used here, simply equal the original reference
designators, e.g. the .param created for R1 is named "R1").

Typical use:

    from ltspice_mc.netlist_parameterizer import interactive_parameterize
    from ltspice_mc import MonteCarloEngine

    result = interactive_parameterize("my_circuit.net")   # asks questions on stdin
    engine = MonteCarloEngine(result.output_path, parallel_sims=8)
    for ref, dist in result.distributions.items():
        engine.add_parameter(ref, dist)
    ...

Design notes / deliberate scope limits (see README for the full rationale):
  * Only the standard 2-terminal form is handled: `Rxxx n1 n2 value [...]`,
    `Cxxx n1 n2 value [IC=...]`, `Lxxx n1 n2 value [IC=...]`. Semiconductor
    resistors/model-referencing devices, and behavioral `R=<expr>` forms,
    are detected but marked unsupported and left untouched -- guessing on
    those risks silently corrupting a working circuit, which is worse than
    just skipping them.
  * A component whose value is already `{...}` is left alone (it's already
    parameterized) and reported separately, never overwritten.
  * A component with NO value set at all (LTspice only writes a SYMATTR
    Value line if someone actually typed one in) is fully supported: there's
    nothing to preserve, so the user is just asked for a starting value with
    no pre-filled default, and a brand-new Value line is inserted.
  * All other lines are copied through completely unchanged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Union

from .distributions import Distribution, Tolerance
from .param_utils import is_asc_file, parse_spice_value

# Reference-designator prefix -> human-readable kind. SPICE itself uses the
# first letter of a component's name to determine its type; this is the
# same convention, so detection needs no knowledge of any specific circuit.
DEFAULT_KIND_BY_PREFIX: Dict[str, str] = {"R": "resistor", "C": "capacitor", "L": "inductor"}
DEFAULT_TOLERANCE_PCT: Dict[str, float] = {"resistor": 5.0, "capacitor": 10.0, "inductor": 10.0}

_NETLIST_COMPONENT_RE = re.compile(r"^([A-Za-z])(\S*)\s+(\S+)\s+(\S+)\s+(\S+)(.*)$")
_SYMBOL_RE = re.compile(r"^SYMBOL\s+(\S+)\s+", re.IGNORECASE)
_SYMATTR_RE = re.compile(r"^SYMATTR\s+(\S+)\s+(.*)$", re.IGNORECASE)
_ASC_PARAM_TEXT_RE = re.compile(r"^(TEXT\s+-?\d+\s+-?\d+\s+\S+\s+\d+\s+!)(\.param\b.*)$", re.IGNORECASE)
_ASC_ANY_COORD_RE = re.compile(r"^(?:WIRE|SYMBOL|FLAG|TEXT)\s+-?\d+\s+(-?\d+)", re.IGNORECASE)
_SHEET_RE = re.compile(r"^SHEET\s+\d+\s+\d+\s+(\d+)", re.IGNORECASE)


@dataclass
class DetectedComponent:
    ref: str                                  # e.g. "R1"
    kind: str                                 # "resistor" | "capacitor" | "inductor"
    nodes: List[str]                          # [] for .asc-sourced components (no net-tracing done)
    raw_value: str                            # exact text currently in the value slot ("" if none exists)
    nominal: Optional[float]                  # parsed numeric nominal, or None if unresolvable/absent
    already_parameterized: bool               # True if raw_value is already "{...}"
    supported: bool                           # False only if there's a value we can't safely touch
    reason_unsupported: Optional[str] = None
    line_index: int = field(default=-1, repr=False)           # existing Value line, or -1 if none exists
    instname_line_index: int = field(default=-1, repr=False)  # .asc only: anchor for inserting a new Value line

    def describe(self) -> str:
        if self.already_parameterized:
            tag = " (already parameterized -- left as-is)"
        elif not self.supported:
            tag = f" (skipped: {self.reason_unsupported})"
        elif self.nominal is None:
            tag = " (no value set in the schematic -- you'll be asked for one)"
        else:
            tag = ""
        return f"Detected {self.kind}: {self.ref} = {self.raw_value}{tag}"


@dataclass
class MonteCarloChoice:
    ref: str
    include: bool
    nominal: float
    tolerance_pct: Optional[float] = None
    distribution: str = "uniform"  # "uniform" | "gaussian"


@dataclass
class ParameterizationResult:
    output_path: Path
    param_lines_added: List[str]
    distributions: Dict[str, Distribution]
    skipped: List[DetectedComponent]

    def summary(self) -> str:
        lines = [f"Wrote {self.output_path}", f"Added {len(self.param_lines_added)} .param assignment(s):"]
        lines += [f"  {p}" for p in self.param_lines_added]
        if self.skipped:
            lines.append(f"Skipped {len(self.skipped)} component(s):")
            lines += [f"  {c.describe()}" for c in self.skipped]
        return "\n".join(lines)


def _classify(ref: str, kinds: Sequence[str]) -> Optional[str]:
    if not ref:
        return None
    prefix = ref[0].upper()
    if prefix in kinds:
        return DEFAULT_KIND_BY_PREFIX.get(prefix, prefix.lower())
    return None


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def detect_components_in_netlist(text: str, kinds: Sequence[str] = ("R", "C", "L")) -> List[DetectedComponent]:
    """Scan a plain SPICE netlist (.net/.cir/.sp) for standard 2-terminal R/C/L lines."""
    components: List[DetectedComponent] = []
    lines = text.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith(("*", ".", "+")):
            continue
        m = _NETLIST_COMPONENT_RE.match(stripped)
        if not m:
            continue
        prefix, rest_of_name, n1, n2, value, _trailing = m.groups()
        ref = prefix + rest_of_name
        kind = _classify(ref, kinds)
        if kind is None:
            continue
        already_param = value.startswith("{")
        nominal = None if already_param else parse_spice_value(value)
        supported = already_param or nominal is not None
        reason = None if supported else f"value '{value}' is not a plain numeric literal (model reference or expression?)"
        components.append(DetectedComponent(
            ref=ref, kind=kind, nodes=[n1, n2], raw_value=value, nominal=nominal,
            already_parameterized=already_param, supported=supported,
            reason_unsupported=reason, line_index=i,
        ))
    return components


def detect_components_in_asc(text: str, kinds: Sequence[str] = ("R", "C", "L")) -> List[DetectedComponent]:
    """Scan an LTspice .asc schematic for SYMBOL blocks whose InstName is an R/C/L designator."""
    components: List[DetectedComponent] = []
    lines = text.splitlines()

    inst_name: Optional[str] = None
    inst_name_line_idx: Optional[int] = None
    value: Optional[str] = None
    value_line_idx: Optional[int] = None

    def flush():
        nonlocal inst_name, inst_name_line_idx, value, value_line_idx
        if inst_name is not None:
            kind = _classify(inst_name, kinds)
            if kind is not None:
                if value is None:
                    # LTspice only writes a SYMATTR Value line if someone
                    # actually set one -- a freshly-placed component can
                    # legitimately have none yet. Nothing to preserve here,
                    # so this is fully supported: the user just gets asked
                    # for a starting value with no pre-filled default.
                    already_param, nominal, supported, reason = False, None, True, None
                elif value.startswith("{"):
                    already_param, nominal, supported, reason = True, None, True, None
                else:
                    nominal = parse_spice_value(value)
                    already_param, supported = False, nominal is not None
                    reason = None if supported else f"value '{value}' is not a plain numeric literal"
                components.append(DetectedComponent(
                    ref=inst_name, kind=kind, nodes=[], raw_value=value or "",
                    nominal=nominal, already_parameterized=already_param,
                    supported=supported, reason_unsupported=reason,
                    line_index=value_line_idx if value_line_idx is not None else -1,
                    instname_line_index=inst_name_line_idx if inst_name_line_idx is not None else -1,
                ))
        inst_name, inst_name_line_idx, value, value_line_idx = None, None, None, None

    for i, line in enumerate(lines):
        stripped = line.strip()
        if _SYMBOL_RE.match(stripped):
            flush()
            continue
        m = _SYMATTR_RE.match(stripped)
        if m:
            attr, attr_value = m.group(1), m.group(2).strip()
            if attr.lower() == "instname":
                inst_name = attr_value
                inst_name_line_idx = i
            elif attr.lower() == "value":
                value = attr_value
                value_line_idx = i
    flush()
    return components


def detect_components(path: Union[str, Path], kinds: Sequence[str] = ("R", "C", "L")) -> List[DetectedComponent]:
    """Auto-dispatches to the .asc or plain-netlist detector based on file extension."""
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    return detect_components_in_asc(text, kinds) if is_asc_file(path) else detect_components_in_netlist(text, kinds)


# ---------------------------------------------------------------------------
# Interactive prompting (I/O is injectable so this is fully unit-testable)
# ---------------------------------------------------------------------------

def _ask_yes_no(prompt: str, default: bool, input_fn, print_fn) -> bool:
    while True:
        raw = input_fn(prompt).strip().lower()
        if raw == "":
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print_fn("  Please answer y or n.")


def _ask_float(prompt: str, default: Optional[float], input_fn, print_fn, min_value: Optional[float] = None) -> float:
    while True:
        raw = input_fn(prompt).strip()
        if raw == "":
            if default is None:
                print_fn("  A value is required.")
                continue
            return float(default)
        value = parse_spice_value(raw)
        if value is None:
            try:
                value = float(raw)
            except ValueError:
                print_fn(f"  Could not parse '{raw}' as a number (suffixes like 1k, 10n are fine).")
                continue
        if min_value is not None and value < min_value:
            print_fn(f"  Value must be >= {min_value}.")
            continue
        return value


def _ask_choice(prompt: str, options: Sequence[str], default: str, input_fn, print_fn) -> str:
    while True:
        raw = input_fn(prompt).strip().lower()
        if raw == "":
            return default
        if raw in options:
            return raw
        print_fn(f"  Please choose one of: {', '.join(options)}")


def prompt_for_choices(
    components: Sequence[DetectedComponent],
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
) -> List[MonteCarloChoice]:
    """
    Walks the user through every detected, convertible component. Already-
    parameterized and unsupported components are shown but never prompted
    for (there's nothing meaningful to ask). Nominal value is always
    pre-filled from the netlist/schematic -- Enter accepts it as-is.
    """
    choices: List[MonteCarloChoice] = []
    to_prompt = [c for c in components if c.supported and not c.already_parameterized]

    print_fn(f"Detected {len(components)} candidate passive component(s):")
    for c in components:
        print_fn(f"  - {c.describe()}")

    for c in to_prompt:
        value_desc = f"current value {c.raw_value}" if c.raw_value else "no value set yet"
        print_fn(f"\n{c.ref} ({c.kind}), {value_desc}:")
        include = _ask_yes_no("  Include in Monte Carlo? [Y/n]: ", default=True, input_fn=input_fn, print_fn=print_fn)
        if not include:
            choices.append(MonteCarloChoice(ref=c.ref, include=False, nominal=c.nominal or 0.0))
            continue

        if c.nominal is None:
            nominal_prompt = "  Nominal value (nothing set in the schematic -- entry required): "
        else:
            nominal_prompt = f"  Nominal value [{c.raw_value}]: "
        nominal = _ask_float(nominal_prompt, default=c.nominal, input_fn=input_fn, print_fn=print_fn)
        default_tol = DEFAULT_TOLERANCE_PCT.get(c.kind, 5.0)
        tolerance = _ask_float(
            f"  Tolerance % [{default_tol:g}]: ", default=default_tol,
            input_fn=input_fn, print_fn=print_fn, min_value=0.0,
        )
        distribution = _ask_choice(
            "  Distribution [uniform/gaussian] (default: uniform): ",
            options=("uniform", "gaussian"), default="uniform", input_fn=input_fn, print_fn=print_fn,
        )
        choices.append(MonteCarloChoice(
            ref=c.ref, include=True, nominal=nominal, tolerance_pct=tolerance, distribution=distribution,
        ))

    return choices


def build_distribution(choice: MonteCarloChoice) -> Distribution:
    """Reuses the existing Tolerance convenience distribution -- keeps the
    uniform/gaussian-from-tolerance math in exactly one place."""
    return Tolerance(nominal=choice.nominal, tolerance_pct=choice.tolerance_pct or 0.0, kind=choice.distribution)


# ---------------------------------------------------------------------------
# Applying choices: rewrite the netlist / schematic
# ---------------------------------------------------------------------------

def _format_param_value(value: float) -> str:
    return f"{value:.10g}"


def _find_netlist_param_insertion_point(lines: List[str]) -> int:
    last_param_idx = -1
    for i, line in enumerate(lines):
        if line.strip().lower().startswith(".param"):
            last_param_idx = i
    if last_param_idx >= 0:
        return last_param_idx + 1
    return 1 if lines and lines[0].strip().startswith("*") else 0


def apply_to_netlist(
    path: Union[str, Path],
    components: Sequence[DetectedComponent],
    choices: Sequence[MonteCarloChoice],
    output_path: Optional[Union[str, Path]] = None,
) -> ParameterizationResult:
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    comp_by_ref = {c.ref: c for c in components}
    included = [c for c in choices if c.include]

    new_param_lines: List[str] = []
    distributions: Dict[str, Distribution] = {}

    for choice in included:
        comp = comp_by_ref[choice.ref]
        line = lines[comp.line_index]
        m = _NETLIST_COMPONENT_RE.match(line.strip())
        if not m:
            raise ValueError(f"Line {comp.line_index} no longer matches the expected component pattern: {line!r}")
        leading_ws = line[: len(line) - len(line.lstrip())]
        prefix, rest_of_name, n1, n2, _old_value, trailing = m.groups()
        lines[comp.line_index] = f"{leading_ws}{prefix}{rest_of_name} {n1} {n2} {{{choice.ref}}}{trailing}"

        new_param_lines.append(f".param {choice.ref}={_format_param_value(choice.nominal)}")
        distributions[choice.ref] = build_distribution(choice)

    insert_at = _find_netlist_param_insertion_point(lines)
    for offset, param_line in enumerate(new_param_lines):
        lines.insert(insert_at + offset, param_line)

    out_path = Path(output_path) if output_path else path.with_name(path.stem + "_parameterized" + path.suffix)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    included_refs = {c.ref for c in included}
    skipped = [c for c in components if c.ref not in included_refs]
    return ParameterizationResult(output_path=out_path, param_lines_added=new_param_lines,
                                   distributions=distributions, skipped=skipped)


def _insert_or_extend_asc_param_directive(lines: List[str], new_assignments: List[str]) -> None:
    """
    Prefers extending an existing `.param` TEXT directive line -- this
    sidesteps any risk of a newly-added TEXT block visually overlapping
    other schematic elements, since no new visual element is added at all.
    Only if no `.param` directive exists yet does this add a new TEXT line,
    placed below every other coordinate-bearing element on the sheet so it
    cannot overlap anything.
    """
    for i, line in enumerate(lines):
        m = _ASC_PARAM_TEXT_RE.match(line.strip())
        if m:
            prefix, body = m.group(1), m.group(2)
            lines[i] = prefix + body.rstrip() + " " + " ".join(new_assignments)
            return

    max_y = 0
    for line in lines:
        stripped = line.strip()
        cm = _ASC_ANY_COORD_RE.match(stripped)
        if cm:
            max_y = max(max_y, int(cm.group(1)))
        sm = _SHEET_RE.match(stripped)
        if sm:
            max_y = max(max_y, int(sm.group(1)))
    directive = ".param " + " ".join(new_assignments)
    lines.append(f"TEXT 40 {max_y + 40} Left 2 !{directive}")


def apply_to_asc(
    path: Union[str, Path],
    components: Sequence[DetectedComponent],
    choices: Sequence[MonteCarloChoice],
    output_path: Optional[Union[str, Path]] = None,
) -> ParameterizationResult:
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    comp_by_ref = {c.ref: c for c in components}
    included = [c for c in choices if c.include]

    new_param_lines: List[str] = []
    distributions: Dict[str, Distribution] = {}

    # Descending order matters: inserting a new SYMATTR Value line (for a
    # component that never had one) shifts every later line index down by
    # one. Processing highest-index components first means any insertion
    # only affects indices we've already finished with, never ones still
    # queued -- so recorded line_index/instname_line_index values for
    # not-yet-processed components stay correct throughout the loop.
    included = sorted(included, key=lambda ch: comp_by_ref[ch.ref].instname_line_index, reverse=True)

    for choice in included:
        comp = comp_by_ref[choice.ref]
        if comp.line_index >= 0:
            # Existing SYMATTR Value line -- overwrite it in place.
            line = lines[comp.line_index]
            if not _SYMATTR_RE.match(line.strip()):
                raise ValueError(f"Line {comp.line_index} no longer looks like a SYMATTR Value line: {line!r}")
            leading_ws = line[: len(line) - len(line.lstrip())]
            lines[comp.line_index] = f"{leading_ws}SYMATTR Value {{{choice.ref}}}"
        else:
            # No Value attribute exists at all -- insert a new one right
            # after this component's InstName line.
            if comp.instname_line_index < 0:
                raise ValueError(f"Could not find an InstName line for {choice.ref} to attach a new Value to")
            inst_line = lines[comp.instname_line_index]
            leading_ws = inst_line[: len(inst_line) - len(inst_line.lstrip())]
            lines.insert(comp.instname_line_index + 1, f"{leading_ws}SYMATTR Value {{{choice.ref}}}")

        new_param_lines.append(f"{choice.ref}={_format_param_value(choice.nominal)}")
        distributions[choice.ref] = build_distribution(choice)

    if new_param_lines:
        _insert_or_extend_asc_param_directive(lines, new_param_lines)

    out_path = Path(output_path) if output_path else path.with_name(path.stem + "_parameterized" + path.suffix)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    included_refs = {c.ref for c in included}
    skipped = [c for c in components if c.ref not in included_refs]
    return ParameterizationResult(output_path=out_path, param_lines_added=new_param_lines,
                                   distributions=distributions, skipped=skipped)


# ---------------------------------------------------------------------------
# One-call orchestration
# ---------------------------------------------------------------------------

def interactive_parameterize(
    schematic_or_netlist_path: Union[str, Path],
    kinds: Sequence[str] = ("R", "C", "L"),
    output_path: Optional[Union[str, Path]] = None,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
) -> ParameterizationResult:
    """
    Detect -> prompt -> rewrite, in one call. This is the only function most
    scripts need: point it at a schematic or netlist, answer the prompts,
    get back a rewritten file plus ready-to-use Distribution objects.
    """
    path = Path(schematic_or_netlist_path)
    components = detect_components(path, kinds)
    choices = prompt_for_choices(components, input_fn=input_fn, print_fn=print_fn)
    apply_fn = apply_to_asc if is_asc_file(path) else apply_to_netlist
    result = apply_fn(path, components, choices, output_path=output_path)
    print_fn("\n" + result.summary())
    return result