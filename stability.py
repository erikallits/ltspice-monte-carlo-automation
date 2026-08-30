"""
stability.py
--------------
Transfer-function extraction from simulation results, and robust stability
analysis of the resulting denominator polynomial D(s) -- circuit-agnostic:
nothing here assumes a topology, component names, or a fixed system order.

IMPORTANT interpretation note, read before trusting ANY verdict from this
module. Three genuinely different questions get answered here, and they
must not be collapsed into one number:

  1. EMPIRICAL Monte Carlo stability (MonteCarloResults.stability_analysis()):
     what fraction of RANDOMLY SAMPLED realizations, per your declared
     parameter distributions, were individually Hurwitz stable. A
     statistical estimate of LIKELY behavior. Absence of an unstable sample
     in N draws never proves stability holds everywhere -- it's evidence,
     not a certificate. More samples narrow the estimate, they don't
     remove this ceiling.

  2. CORNER stability (CornerSweepResult.stability_analysis()): whether
     EVERY explicitly simulated corner combination (Cartesian product of
     discrete parameter/temperature extremes) was stable, and which ones
     weren't if not. Exhaustive over the corners you chose to check, but
     says nothing about points between them.

  3. KHARITONOV robust stability (both result types expose it): the
     conservative "coefficients vary independently within a box" bound.
     Kharitonov's theorem applies EXACTLY to that independent-box family --
     your real D(s) coefficients are NOT independent (they all derive from
     the same handful of physical component/temperature values), so the
     real, correlated family is a strict subset of what Kharitonov checks.
     Consequence, asymmetric and important:
       * Kharitonov PASS => your circuit IS robustly stable across the
         corner set (the independent box contains your real family, so
         stability of the bigger box implies stability of the smaller
         one) -- a genuine, strong guarantee.
       * Kharitonov FAIL does NOT prove your real circuit is unstable --
         it may only mean the artificial independent relaxation is worse
         than anything your circuit can physically produce. Treat a fail
         as "investigate further" using check #1/#2 above, not as a
         confirmed defect.

Two ways to get D(s) for one run:
  * Analytical (recommended whenever you know your topology): supply a
    Python function of the resolved component values --
    analytical_den_single_pole / analytical_den_second_order for the
    unambiguous, topology-independent RC and standard 2nd-order forms, or
    write your own. Exact, no fitting error, no order-selection needed.
  * Numerical least-squares fit from the AC-sweep data
    (fit_transfer_function_lstsq, Levy's method), with the order either
    fixed by you or chosen automatically via auto_fit_transfer_function().
    Read both docstrings before trusting the result -- blind order
    selection from frequency-response data is a genuinely harder, less
    reliable problem than knowing your circuit's structure.

If using the fit method across a whole Monte Carlo/corner BATCH: determine
the order ONCE (auto_fit_transfer_function on one nominal-parameter run, or
MonteCarloEngine.determine_transfer_function_order()), then reuse that FIXED
order for every sample via make_fitted_extractor(). Never re-run automatic
order selection per sample -- besides being slow, different samples could
then select different orders, which makes coefficient-wise bounds (and
therefore Kharitonov) meaningless: a_5 doesn't mean anything if half your
samples are only 3rd order.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

_logger = logging.getLogger("ltspice_mc.stability")


# ---------------------------------------------------------------------------
# Transfer function extraction -- analytical
# ---------------------------------------------------------------------------

def analytical_den_single_pole(tau_fn: Callable[[Dict[str, float]], float]) -> Callable[[Any, Dict[str, float]], np.ndarray]:
    """
    For a single-pole stage D(s) = 1 + s*tau (pole at s = -1/tau). `tau_fn`
    computes tau from the resolved parameter dict, e.g. lambda p: p["R1"]*p["C1"].
    Returns an extract_den_fn(raw, resolved_params) -- `raw` is accepted but
    ignored, since this path never touches the simulation output at all.
    """
    def extract(raw, resolved_params: Dict[str, float]) -> np.ndarray:
        tau = tau_fn(resolved_params)
        return np.array([tau, 1.0])  # highest power first: [s^1, s^0]
    return extract


def analytical_den_second_order(
    omega0_fn: Callable[[Dict[str, float]], float],
    q_fn: Callable[[Dict[str, float]], float],
) -> Callable[[Any, Dict[str, float]], np.ndarray]:
    """
    Standard 2nd-order form D(s) = 1 + s/(Q*omega0) + (s/omega0)^2.
    `omega0_fn`/`q_fn` compute the corner frequency (rad/s) and Q factor
    from the resolved parameter dict. How Q/omega0 relate to your actual
    R/L/C values is topology-specific -- you supply that mapping.
    """
    def extract(raw, resolved_params: Dict[str, float]) -> np.ndarray:
        w0 = omega0_fn(resolved_params)
        q = q_fn(resolved_params)
        return np.array([1.0 / w0 ** 2, 1.0 / (q * w0), 1.0])  # highest power first: [s^2, s^1, s^0]
    return extract


# ---------------------------------------------------------------------------
# Transfer function extraction -- numerical least-squares fit (Levy's method)
# ---------------------------------------------------------------------------

def _build_lstsq_system(freq_hz: np.ndarray, h_complex: np.ndarray, num_order: int, den_order: int):
    """Shared matrix construction, used by both the fit itself and the
    automatic-order-selection condition-number diagnostic."""
    s = 1j * 2.0 * np.pi * np.asarray(freq_hz, dtype=float)
    h = np.asarray(h_complex, dtype=complex)
    cols = [s ** m for m in range(num_order + 1)] + [-h * (s ** k) for k in range(den_order)]
    A = np.stack(cols, axis=1)
    b = h * (s ** den_order)
    return np.vstack([A.real, A.imag]), np.concatenate([b.real, b.imag])


def fit_transfer_function_lstsq(
    freq_hz: np.ndarray,
    h_complex: np.ndarray,
    num_order: int,
    den_order: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Levy's method: a single linear least-squares solve fitting a rational
    transfer function N(s)/D(s) to measured frequency-response data
    H(j*2*pi*f), with the denominator's leading coefficient normalized to 1.

    Returns (num_coeffs, den_coeffs), both highest-power-first (the
    convention numpy.roots/numpy.polyval expect).

    Real limitations, please read before trusting the output:
      * You must know/choose num_order and den_order for THIS function --
        use auto_fit_transfer_function() below if you want that chosen for
        you, or determine it once and reuse a fixed order (see module
        docstring) for a whole batch.
      * This is a LINEAR approximation (not iterative refinement like
        Sanathanan-Koerner or vector fitting), fast and simple but
        implicitly weights high-frequency samples more heavily and can be
        ill-conditioned for high orders or a sweep that doesn't span well
        past the poles/zeros of interest. Use a dense sweep covering at
        least 1-2 decades on either side of the expected pole locations.
      * Prefer analytical_den_single_pole/second_order (or your own
        topology-derived formula) whenever you actually know the circuit's
        structure -- it's exact, this is not.
    """
    if len(freq_hz) < num_order + den_order + 1:
        raise ValueError(
            f"Need at least {num_order + den_order + 1} frequency points for a "
            f"(num_order={num_order}, den_order={den_order}) fit, got {len(freq_hz)}."
        )
    A_ri, b_ri = _build_lstsq_system(freq_hz, h_complex, num_order, den_order)
    x, *_ = np.linalg.lstsq(A_ri, b_ri, rcond=None)
    num_coeffs = x[: num_order + 1][::-1]
    den_coeffs = np.concatenate([x[num_order + 1:], [1.0]])[::-1]
    return num_coeffs, den_coeffs


def make_fitted_extractor(
    out_signal: str,
    in_signal: Optional[str],
    num_order: int,
    den_order: int,
) -> Callable[[Any, Optional[Dict[str, float]]], np.ndarray]:
    """
    Builds an extract_den_fn(raw, resolved_params) using a FIXED order --
    typically from a prior determine_transfer_function_order() /
    auto_fit_transfer_function() call -- for consistent per-sample
    extraction across an entire Monte Carlo / corner batch. `resolved_params`
    is accepted for signature compatibility with run_corner_sweep()/run()
    but ignored -- this path only reads the simulation output.
    """
    from .metrics import get_axis, get_signal  # local import: avoids a hard
    # dependency cycle at module import time (metrics.py doesn't need stability.py)

    def extract(raw, resolved_params: Optional[Dict[str, float]] = None) -> np.ndarray:
        freq = get_axis(raw)
        h = get_signal(raw, out_signal) / get_signal(raw, in_signal) if in_signal else get_signal(raw, out_signal)
        _num, den = fit_transfer_function_lstsq(freq, h, num_order=num_order, den_order=den_order)
        return den

    return extract


# ---------------------------------------------------------------------------
# Automatic order selection
# ---------------------------------------------------------------------------

@dataclass
class OrderCandidate:
    num_order: int
    den_order: int
    num_coeffs: np.ndarray
    den_coeffs: np.ndarray
    rss: float                  # residual sum of squares, real+imag parts as separate real observations
    aic: float
    bic: float
    condition_number: float     # of the least-squares system; large values flag a numerically marginal fit


@dataclass
class AutoFitResult:
    best: OrderCandidate
    candidates: List[OrderCandidate]
    criterion: str  # "aic" | "bic"

    def summary(self) -> str:
        lines = [
            f"Automatic transfer-function order selection ({self.criterion.upper()}): "
            f"num_order={self.best.num_order}, den_order={self.best.den_order}",
            f"  RSS={self.best.rss:.4g}  AIC={self.best.aic:.4g}  BIC={self.best.bic:.4g}  "
            f"condition number={self.best.condition_number:.3g}",
        ]
        if self.best.condition_number > 1e8:
            lines.append(
                "  WARNING: this fit's condition number is very large -- the selected order may not "
                "be numerically well-supported by the data. Inspect plot_order_selection() before trusting it."
            )
        key = (lambda c: c.aic) if self.criterion == "aic" else (lambda c: c.bic)
        ranked = sorted(self.candidates, key=key)
        if len(ranked) > 1 and ranked[0].den_order == max(c.den_order for c in self.candidates):
            lines.append(
                "  NOTE: the selected order sits at the largest den_order tried -- consider raising "
                "max_order to confirm this isn't an artificial ceiling."
            )
        lines.append("  Candidates tried (best first):")
        for c in ranked[:8]:
            lines.append(f"    num={c.num_order} den={c.den_order}: RSS={c.rss:.4g} AIC={c.aic:.4g} BIC={c.bic:.4g}")
        return "\n".join(lines)


def auto_fit_transfer_function(
    freq_hz: np.ndarray,
    h_complex: np.ndarray,
    max_order: int = 6,
    criterion: str = "bic",
    force_proper: bool = True,
) -> AutoFitResult:
    """
    Tries every (num_order, den_order) pair with den_order from 1 to
    max_order (num_order from 0 to den_order if force_proper -- the norm
    for passive/most active circuits -- else 0 to max_order), fits each via
    fit_transfer_function_lstsq, and picks the one minimizing the chosen
    information criterion.

    `criterion="bic"` (default) penalizes extra parameters more heavily
    than "aic" as more data is available, which is the more appropriate
    choice here: the goal is finding the circuit's TRUE, parsimonious
    order, not just the best possible predictive fit (least squares RSS can
    only improve or stay flat as order increases, so SOME penalty for
    complexity is mandatory or this would just always pick max_order).

    This is fundamentally a harder, less reliable problem than knowing your
    circuit's topology -- see the module docstring. Read the returned
    AutoFitResult.summary() (it flags a large condition number or a result
    sitting at max_order) and consider plot_order_selection() before
    trusting the answer, especially for anything safety- or spec-relevant.
    """
    if criterion not in ("aic", "bic"):
        raise ValueError("criterion must be 'aic' or 'bic'")
    if max_order < 1:
        raise ValueError("max_order must be >= 1")

    freq_hz = np.asarray(freq_hz, dtype=float)
    h_complex = np.asarray(h_complex, dtype=complex)
    s = 1j * 2.0 * np.pi * freq_hz
    n_obs = 2 * len(freq_hz)  # real + imaginary parts as separate real observations

    candidates: List[OrderCandidate] = []
    for den_order in range(1, max_order + 1):
        num_orders = range(0, den_order + 1) if force_proper else range(0, max_order + 1)
        for num_order in num_orders:
            if len(freq_hz) < num_order + den_order + 1:
                continue
            try:
                num, den = fit_transfer_function_lstsq(freq_hz, h_complex, num_order, den_order)
            except Exception:
                continue
            if not (np.all(np.isfinite(num)) and np.all(np.isfinite(den))):
                continue

            h_model = np.polyval(num, s) / np.polyval(den, s)
            rss = float(np.sum(np.abs(h_complex - h_model) ** 2))
            if not (np.isfinite(rss) and rss > 0):
                continue

            k = (num_order + 1) + den_order  # free parameters (leading den coeff is fixed to 1)
            aic = n_obs * math.log(rss / n_obs) + 2 * k
            bic = n_obs * math.log(rss / n_obs) + k * math.log(n_obs)

            A_ri, _b_ri = _build_lstsq_system(freq_hz, h_complex, num_order, den_order)
            singular_values = np.linalg.svd(A_ri, compute_uv=False)
            cond = float(singular_values.max() / singular_values.min()) if singular_values.min() > 0 else float("inf")

            candidates.append(OrderCandidate(num_order, den_order, num, den, rss, aic, bic, cond))

    if not candidates:
        raise ValueError(
            f"No (num_order, den_order) candidate up to max_order={max_order} could be fit -- "
            "try a larger max_order, or check that freq_hz/h_complex contain valid AC-analysis data."
        )

    key = (lambda c: c.aic) if criterion == "aic" else (lambda c: c.bic)
    best = min(candidates, key=key)
    return AutoFitResult(best=best, candidates=candidates, criterion=criterion)


# ---------------------------------------------------------------------------
# Hurwitz / Kharitonov stability
# ---------------------------------------------------------------------------

def is_hurwitz_stable(den_coeffs: Sequence[float]) -> bool:
    """True iff every root of the polynomial (highest-power-first, as from
    numpy.roots' convention) has strictly negative real part. Roots exactly
    on the imaginary axis (marginal stability) count as NOT stable, matching
    the standard strict definition of Hurwitz stability."""
    roots = np.roots(np.asarray(den_coeffs, dtype=float))
    return bool(np.all(roots.real < 0))


def kharitonov_polynomials(
    coeff_bounds: Sequence[Tuple[float, float]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    coeff_bounds[i] = (a_i_min, a_i_max), LOWEST power first (a_0 = constant
    term) -- the conventional way interval-polynomial uncertainty is stated.

    Returns (K1, K2, K3, K4), each a coefficient array HIGHEST-power-first
    (ready for np.roots()/is_hurwitz_stable()), built via Kharitonov's
    original construction (verified against Wikipedia's "Kharitonov's
    theorem" and the Encyclopedia of Mathematics entry):
        K1 = a0- + a1- s + a2+ s^2 + a3+ s^3 + a4- s^4 + a5- s^5 + ...
        K2 = a0+ + a1+ s + a2- s^2 + a3- s^3 + a4+ s^4 + a5+ s^5 + ...
        K3 = a0- + a1+ s + a2+ s^2 + a3- s^3 + a4- s^4 + a5+ s^5 + ...
        K4 = a0+ + a1- s + a2- s^2 + a3+ s^3 + a4+ s^4 + a5- s^5 + ...
    """
    n = len(coeff_bounds)
    a_min = np.array([b[0] for b in coeff_bounds], dtype=float)
    a_max = np.array([b[1] for b in coeff_bounds], dtype=float)
    if np.any(a_min > a_max):
        raise ValueError("Every (min, max) pair must satisfy min <= max")

    r = np.arange(n) % 4
    lo_at_01 = np.isin(r, [0, 1])
    lo_at_03 = np.isin(r, [0, 3])

    k1 = np.where(lo_at_01, a_min, a_max)
    k2 = np.where(lo_at_01, a_max, a_min)
    k3 = np.where(lo_at_03, a_min, a_max)
    k4 = np.where(lo_at_03, a_max, a_min)
    return k1[::-1], k2[::-1], k3[::-1], k4[::-1]


@dataclass
class KharitonovResult:
    coeff_bounds: List[Tuple[float, float]]
    polynomials: Dict[str, np.ndarray]
    stable: Dict[str, bool]
    roots: Dict[str, np.ndarray]

    @property
    def all_stable(self) -> bool:
        return all(self.stable.values())

    def summary(self) -> str:
        lines = [f"Kharitonov robust stability: {'STABLE' if self.all_stable else 'NOT CONFIRMED STABLE'}"]
        for name in ("K1", "K2", "K3", "K4"):
            verdict = "stable" if self.stable[name] else "UNSTABLE"
            max_real = float(np.max(self.roots[name].real))
            lines.append(f"  {name}: {verdict}  (max root real part = {max_real:.4g})")
        if not self.all_stable:
            lines.append(
                "  Note: a Kharitonov failure does not by itself prove the actual circuit is "
                "unstable -- cross-check with the direct per-run stability check."
            )
        return "\n".join(lines)


def check_kharitonov_stability(coeff_bounds: Sequence[Tuple[float, float]]) -> KharitonovResult:
    """Builds the 4 Kharitonov polynomials for the given coefficient bounds
    and checks each one for Hurwitz stability."""
    k1, k2, k3, k4 = kharitonov_polynomials(coeff_bounds)
    polynomials = {"K1": k1, "K2": k2, "K3": k3, "K4": k4}
    roots = {name: np.roots(coeffs) for name, coeffs in polynomials.items()}
    stable = {name: bool(np.all(r.real < 0)) for name, r in roots.items()}
    return KharitonovResult(coeff_bounds=list(coeff_bounds), polynomials=polynomials, stable=stable, roots=roots)


def check_all_runs_stable(den_coeffs_stack: np.ndarray) -> np.ndarray:
    """
    Direct, non-conservative stability check: tests each ACTUALLY SIMULATED
    run's own D(s) individually (den_coeffs_stack: shape (n_runs, n_coeffs),
    highest power first). No independence assumption, no relaxation.
    Returns a boolean array.
    """
    stack = np.asarray(den_coeffs_stack, dtype=float)
    return np.array([is_hurwitz_stable(row) for row in stack])


# ---------------------------------------------------------------------------
# Shared stacking/analysis logic, used by BOTH MonteCarloResults and
# CornerSweepResult so the two paths behave identically and stay in sync.
# ---------------------------------------------------------------------------

def _stack_coeffs(den_coeffs_list: Sequence[Optional[np.ndarray]], success_mask: Sequence[bool]) -> np.ndarray:
    ok = [np.asarray(c) for c, s in zip(den_coeffs_list, success_mask) if s and c is not None]
    if not ok:
        raise ValueError(
            "No successful runs produced a transfer function -- did you pass extract_den_fn to "
            "run()/run_corner_sweep()?"
        )
    lengths = {len(c) for c in ok}
    if len(lengths) > 1:
        raise ValueError(
            f"Extracted D(s) have inconsistent orders across runs: {sorted(lengths)} coefficients. "
            "Every run in a batch must share the same polynomial order for coefficient-wise bounds "
            "(and Kharitonov) to be meaningful. If you're using the least-squares fit, don't call "
            "auto_fit_transfer_function() per sample -- determine the order ONCE (e.g. via "
            "MonteCarloEngine.determine_transfer_function_order()) and reuse a FIXED order for "
            "every sample via make_fitted_extractor()."
        )
    return np.stack(ok)


@dataclass
class StabilityAnalysis:
    """
    The three distinct stability claims this framework can make (see the
    module docstring for what each one does and does NOT tell you), bundled
    together but kept clearly labeled rather than collapsed into one verdict.
    """
    source: str  # "monte_carlo" | "corner_sweep"
    n_points: int
    n_success: int
    stable_mask: np.ndarray       # direct, per-point Hurwitz check
    stable_fraction: float
    coeff_bounds: List[Tuple[float, float]]
    kharitonov: KharitonovResult

    def summary(self) -> str:
        if self.source == "monte_carlo":
            headline = (
                f"Empirical Monte Carlo stability: {int(self.stable_mask.sum())}/{len(self.stable_mask)} "
                f"random samples were Hurwitz stable ({100 * self.stable_fraction:.2f}%). This is a "
                "STATISTICAL estimate, not a certificate -- no unstable sample in N draws does not "
                "prove stability holds everywhere in the parameter space."
            )
        else:
            headline = (
                f"Corner stability: {int(self.stable_mask.sum())}/{len(self.stable_mask)} simulated "
                f"corners were Hurwitz stable ({100 * self.stable_fraction:.2f}%). Exhaustive over the "
                "corners you chose to check, not necessarily between them."
            )
        lines = [headline, "", self.kharitonov.summary()]
        return "\n".join(lines)


def build_stability_analysis(
    source: str,
    den_coeffs_list: Sequence[Optional[np.ndarray]],
    success_mask: Sequence[bool],
) -> StabilityAnalysis:
    stack = _stack_coeffs(den_coeffs_list, success_mask)
    direct_stable = check_all_runs_stable(stack)
    lowest_first = stack[:, ::-1]
    bounds = list(zip(lowest_first.min(axis=0).tolist(), lowest_first.max(axis=0).tolist()))
    kharitonov = check_kharitonov_stability(bounds)
    return StabilityAnalysis(
        source=source,
        n_points=len(den_coeffs_list),
        n_success=int(sum(1 for s in success_mask if s)),
        stable_mask=direct_stable,
        stable_fraction=float(direct_stable.mean()) if len(direct_stable) else float("nan"),
        coeff_bounds=bounds,
        kharitonov=kharitonov,
    )


# ---------------------------------------------------------------------------
# Corner-sweep result types (produced by MonteCarloEngine.run_corner_sweep)
# ---------------------------------------------------------------------------

@dataclass
class CornerRun:
    params: Dict[str, float]
    success: bool
    error: Optional[str] = None
    den_coeffs: Optional[np.ndarray] = None  # highest power first; set only if success


@dataclass
class CornerSweepResult:
    param_names: List[str]
    runs: List[CornerRun] = field(default_factory=list)

    @property
    def n_points(self) -> int:
        return len(self.runs)

    @property
    def n_success(self) -> int:
        return sum(1 for r in self.runs if r.success)

    def stacked_den_coeffs(self) -> np.ndarray:
        """(n_successful_runs, n_coeffs) array, highest power first."""
        return _stack_coeffs([r.den_coeffs for r in self.runs], [r.success for r in self.runs])

    def coefficient_bounds(self) -> List[Tuple[float, float]]:
        """[(a_i_min, a_i_max), ...] across all successful runs, LOWEST
        power first -- directly usable as kharitonov_polynomials() input."""
        stacked = self.stacked_den_coeffs()
        lowest_first = stacked[:, ::-1]
        return list(zip(lowest_first.min(axis=0).tolist(), lowest_first.max(axis=0).tolist()))

    def stability_analysis(self) -> StabilityAnalysis:
        """The full, clearly-labeled corner + Kharitonov stability picture -- see StabilityAnalysis."""
        return build_stability_analysis(
            "corner_sweep", [r.den_coeffs for r in self.runs], [r.success for r in self.runs],
        )

    def summary(self) -> str:
        lines = [f"Corner sweep: {self.n_success}/{self.n_points} runs produced a transfer function."]
        n_err = self.n_points - self.n_success
        if n_err:
            lines.append(f"  ({n_err} run(s) failed -- see .runs[i].error for detail)")
        return "\n".join(lines)
