"""
relationships.py
-------------------
Two tools for reasoning about how components relate to each other, rather
than tuning each one blindly:

1. E-series standard value snapping (`snap_to_series`) -- real resistors and
   capacitors only come in IEC 60063 preferred values. Optimization that
   ignores this hands you a "perfect" 8734.2 ohm answer nobody can buy.

2. `Relationship` -- an explicit, named equation that derives one parameter
   from the others, so a dependent quantity (like an RC cutoff frequency)
   can be held constant while a free variable is swept or optimized, instead
   of just letting both drift independently.

Neither of these needs LTspice or spicelib -- they're pure math/text and are
tested standalone in tests/test_relationships_optimization.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

# IEC 60063 preferred-value mantissas (one decade, 1.0 <= value < 10.0).
# E96/E24/E12/E6 verified against the published standard tables.
E_SERIES: Dict[str, List[float]] = {
    "E6": [1.0, 1.5, 2.2, 3.3, 4.7, 6.8],
    "E12": [1.0, 1.2, 1.5, 1.8, 2.2, 2.7, 3.3, 3.9, 4.7, 5.6, 6.8, 8.2],
    "E24": [1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0,
            3.3, 3.6, 3.9, 4.3, 4.7, 5.1, 5.6, 6.2, 6.8, 7.5, 8.2, 9.1],
    "E96": [1.00, 1.02, 1.05, 1.07, 1.10, 1.13, 1.15, 1.18, 1.21, 1.24, 1.27, 1.30,
            1.33, 1.37, 1.40, 1.43, 1.47, 1.50, 1.54, 1.58, 1.62, 1.65, 1.69, 1.74,
            1.78, 1.82, 1.87, 1.91, 1.96, 2.00, 2.05, 2.10, 2.15, 2.21, 2.26, 2.32,
            2.37, 2.43, 2.49, 2.55, 2.61, 2.67, 2.74, 2.80, 2.87, 2.94, 3.01, 3.09,
            3.16, 3.24, 3.32, 3.40, 3.48, 3.57, 3.65, 3.74, 3.83, 3.92, 4.02, 4.12,
            4.22, 4.32, 4.42, 4.53, 4.64, 4.75, 4.87, 4.99, 5.11, 5.23, 5.36, 5.49,
            5.62, 5.76, 5.90, 6.04, 6.19, 6.34, 6.49, 6.65, 6.81, 6.98, 7.15, 7.32,
            7.50, 7.68, 7.87, 8.06, 8.25, 8.45, 8.66, 8.87, 9.09, 9.31, 9.53, 9.76],
}
assert len(E_SERIES["E96"]) == 96 and len(E_SERIES["E24"]) == 24  # catch transcription slips early


def snap_to_series(value: float, series: str) -> float:
    """Rounds `value` to the nearest IEC 60063 preferred value in the given
    series (case-insensitive: "E24", "E96", "E12", "E6"), preserving its
    decade (e.g. 8734 -> 8660 in E96, 4300000 -> 4700000 in E24)."""
    if value <= 0:
        raise ValueError("snap_to_series requires a positive value")
    series = series.upper()
    if series not in E_SERIES:
        raise ValueError(f"Unknown series {series!r}; choose from {sorted(E_SERIES)}")
    base = E_SERIES[series]
    decade = 10.0 ** math.floor(math.log10(value))
    # Check the decade at, above, and below in case the nearest neighbour
    # crosses a decade boundary (e.g. 9.9 is closer to 10.0 than to 9.76).
    candidates = [d * decade * mult for mult in (0.1, 1.0, 10.0) for d in base]
    return min(candidates, key=lambda c: abs(c - value))


# ---------------------------------------------------------------------------
# Explicit component relationships
# ---------------------------------------------------------------------------

@dataclass
class Relationship:
    """
    Derives one parameter's value from the others whenever they change,
    instead of leaving it as an independent free variable.

    `solve` receives a dict of every OTHER currently-resolved parameter
    value (by .param name) and returns the value for `dependent`.
    """
    dependent: str
    solve: Callable[[Dict[str, float]], float]
    description: str = ""

    def __call__(self, other_values: Dict[str, float]) -> float:
        return self.solve(other_values)


def resolve_with_relationships(
    free_values: Dict[str, float],
    relationships: Optional[List[Relationship]] = None,
) -> Dict[str, float]:
    """Applies each Relationship in order, feeding each one the union of the
    original free values plus every dependent value resolved so far -- so
    relationships can be chained (B derived from A, C derived from B)."""
    resolved = dict(free_values)
    for rel in relationships or []:
        resolved[rel.dependent] = rel.solve(resolved)
    return resolved


# --- Common, topology-independent relationship builders ---------------------
# RC time constant / cutoff frequency are unambiguous regardless of circuit
# topology, so these are provided directly. Q-factor and gain-bandwidth
# product genuinely depend on topology (series RLC vs Sallen-Key vs
# multiple-feedback, etc.) -- there's no single formula that's correct for
# all of them, so those are shown as a documented example instead of a
# built-in (see README): define your own via `Relationship` the same way.

def hold_rc_product_constant(driving_param: str, target_product: float) -> Callable[[Dict[str, float]], float]:
    """Returns a `solve` function for a Relationship that keeps R*C (a time
    constant, in seconds) fixed at `target_product` as `driving_param` varies."""
    def solve(values: Dict[str, float]) -> float:
        driving = values[driving_param]
        if driving == 0:
            raise ValueError(f"'{driving_param}' is 0 -- cannot hold R*C constant (division by zero).")
        return target_product / driving
    return solve


def hold_cutoff_frequency_constant(driving_param: str, target_fc_hz: float) -> Callable[[Dict[str, float]], float]:
    """Same as hold_rc_product_constant, expressed as a -3 dB corner
    frequency for a single-pole RC stage: fc = 1 / (2*pi*R*C)."""
    target_tau = 1.0 / (2.0 * math.pi * target_fc_hz)
    return hold_rc_product_constant(driving_param, target_tau)
