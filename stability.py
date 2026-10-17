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
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

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

def _coerce_freq_real(freq_hz: np.ndarray) -> np.ndarray:
    """
    Defense-in-depth mirror of metrics.py's _coerce_real: freq_hz normally
    already arrives real (get_axis() fixes this at the source), but these
    functions are public and may be called directly with a raw axis that
    still has LTspice's complex-but-really-real AC-axis storage quirk.
    Explicit, validated coercion instead of relying on an implicit,
    warning-producing dtype cast that silently discards data either way.
    """
    freq_hz = np.asarray(freq_hz)
    if not np.iscomplexobj(freq_hz):
        return freq_hz.astype(float)
    max_imag = float(np.max(np.abs(freq_hz.imag))) if freq_hz.size else 0.0
    max_real = float(np.max(np.abs(freq_hz.real))) if freq_hz.size else 0.0
    if max_imag > 1e-6 * max(max_real, 1.0):
        raise ValueError(
            f"freq_hz has a non-negligible imaginary part (max |imag|={max_imag:.4g} vs "
            f"max |real|={max_real:.4g}) -- refusing to silently discard it."
        )
    return freq_hz.real


def _default_omega_ref(freq_hz: np.ndarray) -> float:
    """
    Geometric mean of the (positive) frequency range, as an angular
    frequency -- centers the normalized s_hat's dynamic range around 1
    regardless of how many decades the sweep spans.
    """
    freq_hz = _coerce_freq_real(freq_hz)
    positive = freq_hz[freq_hz > 0]
    if len(positive) == 0:
        return 1.0
    return float(2.0 * np.pi * np.sqrt(positive.min() * positive.max()))


def _build_lstsq_system(
    freq_hz: np.ndarray, h_complex: np.ndarray, num_order: int, den_order: int, omega_ref: float = 1.0,
):
    """
    Shared matrix construction, used by both the fit itself and the
    automatic-order-selection condition-number diagnostic. Builds the
    system in NORMALIZED s_hat = s/omega_ref space: raw s^k for k>=2 over a
    multi-decade sweep spans many more orders of magnitude than s_hat^k
    does, which is most of what actually causes ill-conditioning here --
    normalizing fixes it at the source rather than leaving it for a
    condition-number filter to catch after the fact (see
    fit_transfer_function_lstsq for how the result gets un-normalized back
    to true s afterward).
    """
    s_hat = (1j * 2.0 * np.pi * _coerce_freq_real(freq_hz)) / omega_ref
    h = np.asarray(h_complex, dtype=complex)
    cols = [s_hat ** m for m in range(num_order + 1)] + [-h * (s_hat ** k) for k in range(den_order)]
    A = np.stack(cols, axis=1)
    b = h * (s_hat ** den_order)
    return np.vstack([A.real, A.imag]), np.concatenate([b.real, b.imag])


def fit_transfer_function_lstsq(
    freq_hz: np.ndarray,
    h_complex: np.ndarray,
    num_order: int,
    den_order: int,
    omega_ref: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Levy's method: a single linear least-squares solve fitting a rational
    transfer function N(s)/D(s) to measured frequency-response data
    H(j*2*pi*f), with the denominator's leading coefficient normalized to 1.

    Returns (num_coeffs, den_coeffs), both highest-power-first (the
    convention numpy.roots/numpy.polyval expect), in terms of the TRUE
    (unnormalized) s -- `omega_ref` (default: geometric mean of the
    frequency range) is purely an internal conditioning aid; the returned
    coefficients don't depend on it (verified directly in
    tests/test_auto_order_selection.py by fitting the same data with
    omega_ref=1 and the default, and checking they agree).

    Real limitations, please read before trusting the output:
      * You must know/choose num_order and den_order for THIS function --
        use auto_fit_transfer_function() below if you want that chosen for
        you, or determine it once and reuse a fixed order (see module
        docstring) for a whole batch.
      * This is a LINEAR approximation (not iterative refinement like
        Sanathanan-Koerner or vector fitting), fast and simple but
        implicitly weights high-frequency samples more heavily. Frequency
        normalization fixes the conditioning damage from a wide-span sweep,
        but it can't fix a sweep that's simply too SPARSE, or too LINEARLY
        spaced, to resolve your circuit's poles/zeros at all -- see
        check_frequency_sweep() and use a dense, log-spaced (.ac dec ...)
        sweep, especially for anything frequency-selective like a notch.
      * Prefer analytical_den_single_pole/second_order (or your own
        topology-derived formula) whenever you actually know the circuit's
        structure -- it's exact, this is not.
    """
    freq_hz = _coerce_freq_real(freq_hz)
    if len(freq_hz) < num_order + den_order + 1:
        raise ValueError(
            f"Need at least {num_order + den_order + 1} frequency points for a "
            f"(num_order={num_order}, den_order={den_order}) fit, got {len(freq_hz)}."
        )
    if omega_ref is None:
        omega_ref = _default_omega_ref(freq_hz)

    A_ri, b_ri = _build_lstsq_system(freq_hz, h_complex, num_order, den_order, omega_ref)
    x, *_ = np.linalg.lstsq(A_ri, b_ri, rcond=None)
    num_hat = x[: num_order + 1][::-1]                            # highest-power-first, in s_hat
    den_hat = np.concatenate([x[num_order + 1:], [1.0]])[::-1]    # highest-power-first, in s_hat, leading=1

    # D(s) := omega_ref^den_order * D_hat(s/omega_ref) keeps the leading
    # coefficient exactly 1 automatically (den_hat's leading term already
    # is 1 by construction); N(s) must be scaled by the SAME factor to
    # preserve H(s) = N(s)/D(s) unchanged.
    den_powers = np.arange(den_order, -1, -1)
    den_coeffs = den_hat * (omega_ref ** (den_order - den_powers))
    num_powers = np.arange(num_order, -1, -1)
    num_coeffs = num_hat * (omega_ref ** (den_order - num_powers))
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
# LaTeX rendering of a fitted transfer function
# ---------------------------------------------------------------------------

def _format_coefficient_latex(value: float, sig_figs: int = 4) -> str:
    """Formats one coefficient for display: scientific notation (as a
    proper `m \\times 10^{e}`, not Python's `e+07`) outside a comfortable
    decimal range, plain decimal otherwise."""
    if value == 0:
        return "0"
    abs_val = abs(value)
    if abs_val >= 1e4 or abs_val < 1e-3:
        exponent = int(math.floor(math.log10(abs_val)))
        mantissa = value / (10.0 ** exponent)
        mantissa_str = f"{mantissa:.{max(sig_figs - 1, 1)}g}"
        return f"{mantissa_str} \\times 10^{{{exponent}}}"
    return f"{value:.{sig_figs}g}"


def _polynomial_to_latex(coeffs: Sequence[float], variable: str = "s", sig_figs: int = 4) -> str:
    """coeffs: highest-power-first (numpy.roots convention). Builds a LaTeX
    polynomial string entirely from the given coefficients -- every term,
    sign, and power is derived from the actual values, nothing hardcoded."""
    coeffs = list(coeffs)
    n = len(coeffs)
    terms: List[Tuple[str, str]] = []
    for i, c in enumerate(coeffs):
        if c == 0:
            continue
        power = n - 1 - i
        coeff_str = _format_coefficient_latex(abs(c), sig_figs)
        sign = "-" if c < 0 else "+"

        if power == 0:
            term = coeff_str
        elif power == 1:
            term = variable if coeff_str == "1" else f"{coeff_str} {variable}"
        else:
            term = f"{variable}^{{{power}}}" if coeff_str == "1" else f"{coeff_str} {variable}^{{{power}}}"
        terms.append((sign, term))

    if not terms:
        return "0"
    latex = terms[0][1] if terms[0][0] == "+" else f"-{terms[0][1]}"
    for sign, term in terms[1:]:
        latex += f" {sign} {term}"
    return latex


def transfer_function_to_latex(
    num_coeffs: Sequence[float], den_coeffs: Sequence[float], variable: str = "s", sig_figs: int = 4,
) -> str:
    """
    Renders H(s) = N(s)/D(s) as a LaTeX string, generated entirely from the
    actual coefficient VALUES passed in (both highest-power-first, matching
    every other coefficient array in this module) -- nothing about the
    equation's shape, terms, or numbers is hardcoded; a 1st-order and a
    6th-order fit produce correspondingly different output automatically.
    Compatible with matplotlib mathtext (wrap in `$...$`) as well as a real
    LaTeX renderer -- see visualization.plot_transfer_function_latex().
    """
    num_latex = _polynomial_to_latex(num_coeffs, variable, sig_figs)
    den_latex = _polynomial_to_latex(den_coeffs, variable, sig_figs)
    return f"\\mathrm{{H}}({variable}) = \\frac{{{num_latex}}}{{{den_latex}}}"


# ---------------------------------------------------------------------------
# Frequency sweep quality diagnostic
# ---------------------------------------------------------------------------

@dataclass
class FrequencySweepQuality:
    n_points: int
    looks_log_spaced: bool
    min_freq: float
    max_freq: float
    decades_spanned: float
    warnings: List[str] = field(default_factory=list)

    @property
    def is_adequate(self) -> bool:
        return not self.warnings


def check_frequency_sweep(freq_hz: np.ndarray, min_points_per_decade: float = 20.0) -> FrequencySweepQuality:
    """
    Sanity-checks an AC sweep's suitability for transfer-function fitting,
    independent of anything else in this module. Rational-function fitting
    (and therefore automatic order selection, and therefore Kharitonov)
    needs DENSE, roughly LOG-spaced frequency coverage: poles and zeros
    shape a response across decades, not linear Hz, so a sparse or
    linearly-spaced sweep starves the fit of resolution exactly where it
    matters -- most severely for anything frequency-selective, like a notch
    or a sharp resonance, where the interesting behavior can fall entirely
    between two widely-spaced sample points.

    This diagnoses a DATA problem, not a code bug -- if your `.ac` directive
    looks like `.ac lin 20 0.1 10k`, that's very likely why a fit looks
    wrong, not a defect in the fitting itself. See set_ac_analysis() to
    replace it with a dense, log-spaced sweep.
    """
    freq_hz = _coerce_freq_real(freq_hz)
    freq_hz = np.sort(freq_hz[freq_hz > 0])
    n = len(freq_hz)
    if n < 2:
        return FrequencySweepQuality(n, False, float("nan"), float("nan"), float("nan"),
                                      ["fewer than 2 positive frequency points -- can't assess sweep quality"])

    decades = float(math.log10(freq_hz[-1] / freq_hz[0])) if freq_hz[0] > 0 else float("nan")
    ratios = freq_hz[1:] / freq_hz[:-1]
    diffs = np.diff(freq_hz)
    log_cv = float(np.std(ratios) / np.mean(ratios)) if np.mean(ratios) > 0 else float("inf")
    lin_cv = float(np.std(diffs) / np.mean(diffs)) if np.mean(diffs) > 0 else float("inf")
    looks_log_spaced = log_cv < lin_cv

    warnings_list: List[str] = []
    if n < 20:
        warnings_list.append(
            f"only {n} frequency points -- fits (especially for anything frequency-selective, like a "
            "notch or sharp resonance) need dense coverage; consider '.ac dec 100 ...' or denser."
        )
    if not looks_log_spaced and decades > 1:
        warnings_list.append(
            f"the sweep spans {decades:.1f} decades but looks LINEARLY spaced, not log-spaced -- most "
            "points end up bunched at the high-frequency end with almost none at low frequency. Use "
            "'.ac dec <points_per_decade> <fstart> <fstop>' instead of '.ac lin ...' for anything "
            "spanning more than about a decade."
        )
    elif decades > 0 and (n / decades) < min_points_per_decade:
        warnings_list.append(
            f"only {n / decades:.1f} points per decade over {decades:.1f} decades -- consider raising "
            f"the point count in your .ac directive (aim for {min_points_per_decade:.0f}+ per decade) "
            "for a more reliable fit."
        )

    return FrequencySweepQuality(
        n_points=n, looks_log_spaced=looks_log_spaced, min_freq=float(freq_hz[0]), max_freq=float(freq_hz[-1]),
        decades_spanned=decades, warnings=warnings_list,
    )


def set_ac_analysis(
    path: Union[str, Path],
    sweep_type: str = "dec",
    points: int = 100,
    fstart: float = 1.0,
    fstop: float = 1e6,
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Replaces any existing `.ac` directive in a plain SPICE netlist with a
    new one -- surgically (finds and replaces the line(s) in place), not by
    appending a second, conflicting `.ac` directive -- so you can guarantee
    dense, log-spaced coverage for reliable transfer-function fitting
    without hand-editing the netlist yourself.

    `sweep_type`: "dec" (points per decade, recommended -- matches how
    poles/zeros shape the response) or "oct"/"lin" if you specifically want
    those instead. Pick `fstart`/`fstop` with real margin around whatever
    frequency you care about (a notch, a corner frequency, ...) -- at least
    a decade on either side if you can.

    Only supports plain netlists (.net/.cir) for now -- generate a netlist
    from LTspice (File > Generate Netlist) first if you're working from a
    .asc schematic directly.
    """
    path = Path(path)
    if path.suffix.lower() == ".asc":
        raise ValueError(
            "set_ac_analysis() only edits plain netlists right now, not .asc schematics directly -- "
            "generate a netlist (File > Generate Netlist in LTspice) and point this at that file "
            "instead, or edit the .ac directive in the schematic yourself."
        )
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    new_directive = f".ac {sweep_type} {points} {fstart:g} {fstop:g}"
    ac_re = re.compile(r"^\.ac\b", re.IGNORECASE)

    kept: List[str] = []
    replaced = False
    for line in lines:
        if ac_re.match(line.strip()):
            if not replaced:
                kept.append(new_directive)
                replaced = True
            # else: a second/redundant .ac line -- drop it rather than
            # leaving two directives that could conflict.
        else:
            kept.append(line)
    if not replaced:
        end_idx = next((i for i, l in enumerate(kept) if l.strip().lower() == ".end"), len(kept))
        kept.insert(end_idx, new_directive)

    out_path = Path(output_path) if output_path else path
    out_path.write_text("\n".join(kept) + "\n", encoding="utf-8")
    return out_path


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
    sweep_quality: Optional[FrequencySweepQuality] = None

    def summary(self) -> str:
        lines = []
        if self.sweep_quality is not None and not self.sweep_quality.is_adequate:
            lines.append("FREQUENCY SWEEP WARNING (check this first -- it can invalidate everything below):")
            for w in self.sweep_quality.warnings:
                lines.append(f"  - {w}")
            lines.append("")
        lines += [
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


def _select_best_candidate(
    candidates: List["OrderCandidate"], criterion: str, max_condition_number: float,
) -> "OrderCandidate":
    """
    Picks the AIC/BIC-best candidate AMONG the numerically well-conditioned
    ones -- pulled out as its own function so this selection rule (the
    actual fix for the "ill-conditioned high order spuriously wins" failure
    mode) is testable precisely, with constructed candidates, independent
    of whatever a specific synthetic frequency sweep happens to produce.
    """
    key = (lambda c: c.aic) if criterion == "aic" else (lambda c: c.bic)
    well_conditioned = [c for c in candidates if c.condition_number <= max_condition_number]
    if well_conditioned:
        return min(well_conditioned, key=key)
    best = min(candidates, key=lambda c: c.condition_number)
    _logger.warning(
        "No candidate had condition number <= %.2g (best available: %.3g). "
        "The returned order is the least-bad option, not a trustworthy fit -- treat with real "
        "suspicion, especially if it disagrees with what you'd expect from the circuit's topology.",
        max_condition_number, best.condition_number,
    )
    return best


def auto_fit_transfer_function(
    freq_hz: np.ndarray,
    h_complex: np.ndarray,
    max_order: int = 6,
    criterion: str = "bic",
    force_proper: bool = True,
    max_condition_number: float = 1e10,
) -> AutoFitResult:
    """
    Tries every (num_order, den_order) pair with den_order from 1 to
    max_order (num_order from 0 to den_order if force_proper -- the norm
    for passive/most active circuits -- else 0 to max_order), fits each via
    fit_transfer_function_lstsq, and picks the one minimizing the chosen
    information criterion AMONG numerically well-conditioned candidates.

    `criterion="bic"` (default) penalizes extra parameters more heavily
    than "aic" as more data is available, which is the more appropriate
    choice here: the goal is finding the circuit's TRUE, parsimonious
    order, not just the best possible predictive fit (least squares RSS can
    only improve or stay flat as order increases, so SOME penalty for
    complexity is mandatory or this would just always pick max_order).

    `max_condition_number` (default 1e10, well below double precision's
    ~1e16 floor) excludes candidates whose least-squares system is too
    ill-conditioned to trust, REGARDLESS of what AIC/BIC say about them.
    This matters more than it might look: once RSS bottoms out near the
    numerical noise floor (a near-perfect fit at the true order), a
    higher-order candidate can appear to fit EVEN better by a tiny, totally
    spurious margin -- not because it captures real structure, but because
    it has enough free parameters to also fit floating-point noise. AIC/BIC
    alone can't tell that apart from genuine improvement; the condition
    number can, because an ill-conditioned fit's COEFFICIENTS (and
    therefore its roots/poles) are essentially undetermined even when its
    RSS looks fine. Without this filter, such a candidate can win outright
    and hand you Hurwitz/Kharitonov roots that are numerical noise, not a
    real stability result -- don't lower this unless you understand that
    tradeoff.

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

    freq_hz = _coerce_freq_real(freq_hz)
    h_complex = np.asarray(h_complex, dtype=complex)
    s = 1j * 2.0 * np.pi * freq_hz
    n_obs = 2 * len(freq_hz)  # real + imaginary parts as separate real observations
    omega_ref = _default_omega_ref(freq_hz)

    candidates: List[OrderCandidate] = []
    for den_order in range(1, max_order + 1):
        num_orders = range(0, den_order + 1) if force_proper else range(0, max_order + 1)
        for num_order in num_orders:
            if len(freq_hz) < num_order + den_order + 1:
                continue
            try:
                num, den = fit_transfer_function_lstsq(freq_hz, h_complex, num_order, den_order, omega_ref=omega_ref)
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

            # Condition number of the SAME normalized system actually
            # solved above -- must use the same omega_ref, or this would be
            # diagnosing a different (unnormalized) system than the one
            # that produced `num`/`den`.
            A_ri, _b_ri = _build_lstsq_system(freq_hz, h_complex, num_order, den_order, omega_ref=omega_ref)
            singular_values = np.linalg.svd(A_ri, compute_uv=False)
            cond = float(singular_values.max() / singular_values.min()) if singular_values.min() > 0 else float("inf")

            candidates.append(OrderCandidate(num_order, den_order, num, den, rss, aic, bic, cond))

    if not candidates:
        raise ValueError(
            f"No (num_order, den_order) candidate up to max_order={max_order} could be fit -- "
            "try a larger max_order, or check that freq_hz/h_complex contain valid AC-analysis data."
        )

    sweep_quality = check_frequency_sweep(freq_hz)
    best = _select_best_candidate(candidates, criterion, max_condition_number)
    return AutoFitResult(best=best, candidates=candidates, criterion=criterion, sweep_quality=sweep_quality)




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


def stability_margin(den_coeffs_stack: np.ndarray) -> np.ndarray:
    """
    max(Re(roots)) per row of den_coeffs_stack (shape (n_runs, n_coeffs),
    highest power first) -- negative means Hurwitz stable, with magnitude
    as a rough margin (more negative = further from the instability
    boundary); positive means unstable. The natural, circuit-agnostic
    scalar to histogram/plot per Monte Carlo sample or corner: unlike a
    domain-specific metric (bandwidth, gain, ...), this is always available
    the moment you have D(s), regardless of topology.
    """
    stack = np.asarray(den_coeffs_stack, dtype=float)
    return np.array([float(np.max(np.roots(row).real)) for row in stack])


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
