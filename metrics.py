"""
metrics.py
-----------
Extraction of derived KPIs (RMS, gain, phase margin, bandwidth, ...) from
simulation output.

Every function here is written against a small duck-typed protocol --
anything with `.get_axis(step=0)` and `.get_trace(name).get_wave(step=0)`,
which is exactly what `spicelib.raw.raw_read.RawRead` provides. That means
these functions can be, and are (see tests/test_framework.py), unit-tested
with a lightweight mock instead of a real RawRead/LTspice output, decoupling
the math from the LTspice/spicelib dependency.
"""

from __future__ import annotations

from typing import Optional, Protocol, Union, runtime_checkable

import numpy as np

# numpy 2.0 renamed trapz -> trapezoid (and later numpy versions remove the
# old name outright); support both so this works across numpy 1.x and 2.x.
_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


@runtime_checkable
class RawLike(Protocol):
    def get_axis(self, step: int = 0) -> np.ndarray: ...
    def get_trace(self, name: str): ...


def get_signal(raw: RawLike, name: str, step: int = 0) -> np.ndarray:
    """Fetch any trace (voltage, current) by name, as a numpy array (real or complex)."""
    trace = raw.get_trace(name)
    return np.asarray(trace.get_wave(step))


def get_axis(raw: RawLike, step: int = 0) -> np.ndarray:
    """Fetch the X axis: time for .tran, frequency for .ac."""
    return np.asarray(raw.get_axis(step))


def _windowed(t: np.ndarray, y: np.ndarray, t_start: Optional[float], t_end: Optional[float]):
    mask = np.ones_like(t, dtype=bool)
    if t_start is not None:
        mask &= t >= t_start
    if t_end is not None:
        mask &= t <= t_end
    return t[mask], y[mask]


# --------------------------------------------------------------------------
# Time-domain metrics
# --------------------------------------------------------------------------

def compute_rms(raw: RawLike, signal: str, t_start: Optional[float] = None,
                 t_end: Optional[float] = None) -> float:
    """
    RMS value of a transient signal, optionally windowed to [t_start, t_end]
    (use the window to exclude a startup transient and measure steady-state RMS).
    """
    t = get_axis(raw)
    y = get_signal(raw, signal)
    if np.iscomplexobj(y):
        raise ValueError(
            f"'{signal}' looks like AC-analysis (complex) data, not a transient "
            "signal -- compute_rms expects real-valued time-domain data."
        )
    t, y = _windowed(t, y, t_start, t_end)
    if len(t) < 2 or (t[-1] - t[0]) <= 0:
        return float("nan")
    return float(np.sqrt(_trapz(y ** 2, t) / (t[-1] - t[0])))


def compute_mean(raw: RawLike, signal: str, t_start: Optional[float] = None,
                  t_end: Optional[float] = None) -> float:
    """Time-averaged (DC) value of a transient signal over [t_start, t_end]."""
    t = get_axis(raw)
    y = get_signal(raw, signal)
    t, y = _windowed(t, y, t_start, t_end)
    if len(t) < 2 or (t[-1] - t[0]) <= 0:
        return float("nan")
    return float(_trapz(y, t) / (t[-1] - t[0]))


def compute_peak_to_peak(raw: RawLike, signal: str, t_start: Optional[float] = None,
                          t_end: Optional[float] = None) -> float:
    t = get_axis(raw)
    y = get_signal(raw, signal)
    _, y = _windowed(t, y, t_start, t_end)
    if len(y) == 0:
        return float("nan")
    return float(np.max(y) - np.min(y))


def compute_overshoot_pct(raw: RawLike, signal: str, final_value: Optional[float] = None) -> float:
    """
    Overshoot as a percentage of the final (settled) value. If `final_value`
    isn't given, it's estimated from the mean of the last 5% of the trace.
    """
    y = get_signal(raw, signal)
    if len(y) == 0:
        return float("nan")
    if final_value is None:
        tail = max(1, len(y) // 20)
        final_value = float(np.mean(y[-tail:]))
    if final_value == 0:
        return float("nan")
    peak = float(np.max(y)) if final_value > 0 else float(np.min(y))
    return max((peak - final_value) / abs(final_value) * 100.0, 0.0)


def compute_settling_time(raw: RawLike, signal: str, tolerance_pct: float = 2.0,
                           final_value: Optional[float] = None) -> float:
    """
    First time after which the signal stays within +/- tolerance_pct of its
    final value for the rest of the simulation. Returns nan if it never settles.
    """
    t = get_axis(raw)
    y = get_signal(raw, signal)
    if len(y) == 0:
        return float("nan")
    if final_value is None:
        tail = max(1, len(y) // 20)
        final_value = float(np.mean(y[-tail:]))
    band = abs(final_value) * tolerance_pct / 100.0
    if band == 0:
        band = np.finfo(float).eps
    within = np.abs(y - final_value) <= band
    outside_idx = np.flatnonzero(~within)
    if len(outside_idx) == 0:
        return float(t[0])
    last_outside = outside_idx[-1]
    if last_outside + 1 >= len(t):
        return float("nan")
    return float(t[last_outside + 1])


# --------------------------------------------------------------------------
# Frequency-domain (AC) metrics
# --------------------------------------------------------------------------

def _mag_phase_db_deg(raw: RawLike, out_signal: str, in_signal: Optional[str] = None):
    freq = get_axis(raw)
    h_out = get_signal(raw, out_signal)
    h = h_out / get_signal(raw, in_signal) if in_signal is not None else h_out
    if not np.iscomplexobj(h):
        raise ValueError(
            f"'{out_signal}' does not look like AC-analysis (complex) data -- "
            "gain/phase-margin/bandwidth all require output from a .ac simulation."
        )
    order = np.argsort(freq)
    freq, h = freq[order], h[order]
    mag_db = 20.0 * np.log10(np.maximum(np.abs(h), 1e-300))
    phase_deg = np.degrees(np.unwrap(np.angle(h)))
    return freq, mag_db, phase_deg


def compute_gain_db(raw: RawLike, out_signal: str, in_signal: Optional[str] = None,
                     at_freq: Optional[float] = None) -> Union[float, np.ndarray]:
    """
    Gain in dB from an AC analysis. If `in_signal` is given, computes
    20*log10(|out/in|); otherwise `out_signal` is treated as the transfer
    function directly (e.g. a loop-gain probe). Returns the full array unless
    `at_freq` is given, in which case the value is interpolated (linearly in
    log-frequency) at that frequency.
    """
    freq, mag_db, _ = _mag_phase_db_deg(raw, out_signal, in_signal)
    if at_freq is None:
        return mag_db
    return float(np.interp(np.log10(at_freq), np.log10(freq), mag_db))


def compute_phase_deg(raw: RawLike, out_signal: str, in_signal: Optional[str] = None,
                       at_freq: Optional[float] = None) -> Union[float, np.ndarray]:
    """Phase in degrees (unwrapped) from an AC analysis. See compute_gain_db for arguments."""
    freq, _, phase_deg = _mag_phase_db_deg(raw, out_signal, in_signal)
    if at_freq is None:
        return phase_deg
    return float(np.interp(np.log10(at_freq), np.log10(freq), phase_deg))


def _find_crossing_log(freq: np.ndarray, series: np.ndarray, level: float) -> Optional[float]:
    """
    First frequency (scanning low -> high) at which `series` crosses down
    through `level`, interpolated linearly in log10(frequency) for accuracy
    on a Bode plot. Returns None if no such crossing exists in range.
    """
    above = series - level
    sign_change = np.flatnonzero((above[:-1] >= 0) & (above[1:] < 0))
    if len(sign_change) == 0:
        return None
    i = int(sign_change[0])
    x0, x1 = np.log10(freq[i]), np.log10(freq[i + 1])
    y0, y1 = series[i], series[i + 1]
    if y1 == y0:
        return float(freq[i])
    frac = (level - y0) / (y1 - y0)
    return float(10 ** (x0 + frac * (x1 - x0)))


def compute_bandwidth(raw: RawLike, out_signal: str, in_signal: Optional[str] = None,
                       drop_db: float = 3.0, reference_db: Optional[float] = None) -> float:
    """
    Frequency at which the gain has dropped `drop_db` below `reference_db`.
    `reference_db` defaults to the gain at the lowest simulated frequency
    (i.e. DC/passband gain for a low-pass response). Returns nan if the gain
    never drops that far within the simulated sweep.
    """
    freq, mag_db, _ = _mag_phase_db_deg(raw, out_signal, in_signal)
    ref = mag_db[0] if reference_db is None else reference_db
    crossing = _find_crossing_log(freq, mag_db, ref - drop_db)
    return float("nan") if crossing is None else crossing


def compute_phase_margin(raw: RawLike, loop_gain_signal: str, in_signal: Optional[str] = None,
                          phase_offset_deg: float = 0.0) -> float:
    """
    Classic phase margin: PM = 180 + phase(f) evaluated at the frequency
    where |loop_gain| first crosses 0 dB (scanning from low to high frequency).

    `loop_gain_signal` should be the AC trace of your loop-gain probe (e.g.
    from a loop broken with an AC source injected in series, per the
    standard injection/Middlebrook technique) -- this function does not
    break the loop for you, it only processes whatever complex ratio you
    hand it as `loop_gain_signal` (optionally divided by `in_signal`).

    Sign conventions for loop-gain phase vary by measurement setup; if your
    result looks offset by a fixed amount, pass `phase_offset_deg` to correct
    it (e.g. 180 if your probe already includes an inversion). Returns nan
    if the loop gain never crosses 0 dB within the simulated range.
    """
    freq, mag_db, phase_deg = _mag_phase_db_deg(raw, loop_gain_signal, in_signal)
    f_cross = _find_crossing_log(freq, mag_db, 0.0)
    if f_cross is None:
        return float("nan")
    phase_at_cross = float(np.interp(np.log10(f_cross), np.log10(freq), phase_deg))
    return 180.0 + phase_at_cross + phase_offset_deg
