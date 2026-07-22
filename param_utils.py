"""
param_utils.py
---------------
Circuit-agnostic discovery and parsing of SPICE `.param` definitions.

This module never looks at component reference designators (R1, C1, M1, ...).
It only understands `.param name=value` text, wherever it appears:

  * As a plain SPICE netlist line:            .param R=1k C=10n
  * Inside an LTspice .asc schematic, where
    directives are stored in a TEXT command:  TEXT 40 296 Left 2 !.param R=1k C=10n

Both forms are pure text, so parameter discovery works even on a machine
without spicelib installed. spicelib is only needed later, to actually
*write* new values back into the netlist/schematic and to launch LTspice
(see runner.py).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

# SPICE engineering suffixes. NOTE: SPICE is deliberately quirky here --
#   "M"/"m"   = milli  (1e-3)   <-- NOT mega!
#   "MEG"     = mega   (1e6)    <-- must be matched before the single "M"
# Order matters: longer/more-specific suffixes must be checked first.
_SUFFIX_MULTIPLIERS = (
    ("meg", 1e6),
    ("t", 1e12),
    ("g", 1e9),
    ("k", 1e3),
    ("m", 1e-3),
    ("u", 1e-6),
    ("\u00b5", 1e-6),  # micro sign
    ("n", 1e-9),
    ("p", 1e-12),
    ("f", 1e-15),
)

_NUMERIC_RE = re.compile(r"^\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*([a-zA-Z\u00b5]*)\s*$")
# European-style embedded-decimal notation, e.g. "2k7" == 2.7k, "4n7" == 4.7n
_EMBEDDED_DECIMAL_RE = re.compile(r"^\s*([+-]?\d+)([a-zA-Z\u00b5]{1,3})(\d+)\s*$")

_PARAM_LINE_RE = re.compile(r"\.param\b(.*)", re.IGNORECASE)
_ASC_DIRECTIVE_RE = re.compile(r"^TEXT\s+-?\d+\s+-?\d+\s+\S+\s+\d+\s+!(.*)$", re.IGNORECASE)
_ASSIGNMENT_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(\{[^}]*\}|\S+)")


def _lookup_suffix(suffix: str) -> Optional[float]:
    suffix = suffix.lower()
    if not suffix:
        return 1.0
    for tag, mult in _SUFFIX_MULTIPLIERS:
        if suffix.startswith(tag):
            return mult
    return None


def parse_spice_value(text: str) -> Optional[float]:
    """
    Convert a SPICE numeric literal (e.g. '1k', '2.2Meg', '10n', '2k7', '5')
    to a float. Returns None if `text` is not a resolvable numeric literal
    (e.g. it's an expression like '{R*2}' or references another parameter).
    """
    text = text.strip()
    if not text or text.startswith("{") or text.startswith("'") or text.startswith('"'):
        return None

    m = _NUMERIC_RE.match(text)
    if m:
        mantissa, suffix = m.groups()
        try:
            value = float(mantissa)
        except ValueError:
            return None
        mult = _lookup_suffix(suffix)
        # Unknown trailing unit text (e.g. "5V", "10Ohm") -> SPICE ignores it;
        # fall back to the bare mantissa rather than failing outright.
        return value * mult if mult is not None else value

    m2 = _EMBEDDED_DECIMAL_RE.match(text)
    if m2:
        int_part, suffix, frac_part = m2.groups()
        mult = _lookup_suffix(suffix)
        if mult is not None:
            return float(f"{int_part}.{frac_part}") * mult

    return None


@dataclass
class DiscoveredParameter:
    """A single `.param` definition found while scanning a schematic/netlist."""
    name: str
    raw_value: str
    nominal: Optional[float]
    source_line: str

    def __repr__(self) -> str:  # pragma: no cover - cosmetic only
        nominal_txt = "unresolved (expression)" if self.nominal is None else f"{self.nominal:g}"
        return f"DiscoveredParameter(name={self.name!r}, raw={self.raw_value!r}, nominal={nominal_txt})"


def _iter_param_statement_bodies(text: str):
    """
    Yield (body, source_line) for every `.param` statement found, from both
    plain SPICE lines and LTspice .asc `TEXT ... !` directive lines. Handles
    the `+` line-continuation convention used by SPICE netlists.
    """
    raw_lines = text.splitlines()
    logical_lines = []
    for line in raw_lines:
        if line.startswith("+") and logical_lines:
            logical_lines[-1] += " " + line[1:]
        else:
            logical_lines.append(line)

    for line in logical_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("*"):
            continue  # blank line or full-line SPICE comment

        asc_match = _ASC_DIRECTIVE_RE.match(stripped)
        if asc_match:
            directive_body = asc_match.group(1)
            # LTspice encodes literal newlines inside one TEXT block as the
            # two characters backslash-n; split those into separate statements.
            for sub in directive_body.split("\\n"):
                pm = _PARAM_LINE_RE.search(sub)
                if pm:
                    yield pm.group(1), line
            continue

        if stripped.lower().startswith(".param"):
            pm = _PARAM_LINE_RE.search(stripped)
            if pm:
                yield pm.group(1), line


def discover_parameters(schematic_path: Union[str, Path]) -> Dict[str, DiscoveredParameter]:
    """
    Scan an LTspice `.asc` schematic OR a plain SPICE netlist (`.net`/`.cir`/`.sp`)
    for every top-level `.param name=value` definition -- with zero knowledge
    of component reference designators. This is the mechanism that makes the
    framework circuit-agnostic: it tells you *only* what's tunable via
    `.param`, which is exactly the set of things `MonteCarloEngine` is allowed
    to vary.

    Returns a dict keyed by parameter name, in first-seen order.
    """
    path = Path(schematic_path)
    text = path.read_text(encoding="utf-8", errors="replace")

    discovered: Dict[str, DiscoveredParameter] = {}
    for body, source_line in _iter_param_statement_bodies(text):
        for assign in _ASSIGNMENT_RE.finditer(body):
            name, raw_value = assign.group(1), assign.group(2)
            discovered[name] = DiscoveredParameter(
                name=name,
                raw_value=raw_value,
                nominal=parse_spice_value(raw_value),
                source_line=source_line.strip(),
            )
    return discovered


def is_asc_file(path: Union[str, Path]) -> bool:
    return Path(path).suffix.lower() == ".asc"
