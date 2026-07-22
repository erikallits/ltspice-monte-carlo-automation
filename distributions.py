"""
distributions.py
-----------------
Statistical sampling for Monte Carlo parameter variation.

Every Distribution knows how to draw `n` samples given a numpy Generator.
Optional min/max bounds are enforced through rejection sampling (out-of-range
draws are re-sampled) with a clip-based fallback after too many rounds --
this is what satisfies the "invalid parameter samples -> fallback or
rejection sampling" robustness requirement.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Callable, Optional

import numpy as np

_logger = logging.getLogger("ltspice_mc.distributions")


class Distribution(ABC):
    """Base class for all parameter-sampling distributions."""

    def __init__(
        self,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        max_rejection_rounds: int = 200,
    ):
        if min_value is not None and max_value is not None and min_value > max_value:
            raise ValueError(f"min_value ({min_value}) > max_value ({max_value})")
        self.min_value = min_value
        self.max_value = max_value
        self.max_rejection_rounds = max_rejection_rounds

    @abstractmethod
    def _draw(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw n raw samples, ignoring bounds. Implemented by subclasses."""
        raise NotImplementedError

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw n samples, honoring min_value/max_value via rejection sampling."""
        if n <= 0:
            return np.empty(0, dtype=float)
        values = self._draw(n, rng)
        if self.min_value is None and self.max_value is None:
            return values

        accepted = np.full(n, np.nan)
        pending = np.ones(n, dtype=bool)
        rounds = 0
        while pending.any() and rounds < self.max_rejection_rounds:
            n_pending = int(pending.sum())
            candidates = self._draw(n_pending, rng)
            valid = np.ones(n_pending, dtype=bool)
            if self.min_value is not None:
                valid &= candidates >= self.min_value
            if self.max_value is not None:
                valid &= candidates <= self.max_value
            pending_idx = np.flatnonzero(pending)
            accept_idx = pending_idx[valid]
            accepted[accept_idx] = candidates[valid]
            pending[accept_idx] = False
            rounds += 1

        if pending.any():
            n_left = int(pending.sum())
            _logger.warning(
                "%d sample(s) still outside [%s, %s] after %d rejection rounds; "
                "clipping instead of resampling further.",
                n_left, self.min_value, self.max_value, rounds,
            )
            pending_idx = np.flatnonzero(pending)
            fallback = self._draw(n_left, rng)
            accepted[pending_idx] = np.clip(fallback, self.min_value, self.max_value)
        return accepted


class Gaussian(Distribution):
    """Normal / Gaussian distribution: value ~ N(mean, std**2)."""

    def __init__(self, mean: float, std: float, **bounds):
        super().__init__(**bounds)
        if std < 0:
            raise ValueError("std must be >= 0")
        self.mean = mean
        self.std = std

    def _draw(self, n, rng):
        return rng.normal(loc=self.mean, scale=self.std, size=n)

    def __repr__(self):
        return f"Gaussian(mean={self.mean:g}, std={self.std:g})"


class Uniform(Distribution):
    """Uniform distribution over [low, high]."""

    def __init__(self, low: float, high: float, **bounds):
        super().__init__(**bounds)
        if low > high:
            raise ValueError("low must be <= high")
        self.low = low
        self.high = high

    def _draw(self, n, rng):
        return rng.uniform(low=self.low, high=self.high, size=n)

    def __repr__(self):
        return f"Uniform(low={self.low:g}, high={self.high:g})"


class Tolerance(Distribution):
    """
    Convenience wrapper for the common "component tolerance" spec, e.g. a
    "1% resistor". `kind='uniform'` treats the tolerance as a hard +/- limit
    (matches how most component datasheets state tolerance). `kind='gaussian'`
    treats it as a `sigma_multiple`-sigma window (defaults to 3-sigma, a
    common assumption when a Gaussian model is preferred for the same spec).
    """

    def __init__(
        self,
        nominal: float,
        tolerance_pct: float,
        kind: str = "uniform",
        sigma_multiple: float = 3.0,
        **bounds,
    ):
        super().__init__(**bounds)
        self.nominal = nominal
        self.tolerance_pct = tolerance_pct
        self.kind = kind.lower()
        self.sigma_multiple = sigma_multiple
        delta = abs(nominal) * tolerance_pct / 100.0
        if self.kind == "uniform":
            self._impl: Distribution = Uniform(nominal - delta, nominal + delta)
        elif self.kind in ("gaussian", "normal"):
            std = delta / sigma_multiple if sigma_multiple > 0 else 0.0
            self._impl = Gaussian(nominal, std)
        else:
            raise ValueError(f"Unknown Tolerance kind: {kind!r} (use 'uniform' or 'gaussian')")

    def _draw(self, n, rng):
        return self._impl._draw(n, rng)

    def __repr__(self):
        return (f"Tolerance(nominal={self.nominal:g}, tolerance_pct={self.tolerance_pct:g}, "
                f"kind={self.kind!r})")


class Custom(Distribution):
    """
    Wraps an arbitrary user-defined sampler: any callable(n, rng) -> array-like
    of length n. This is the escape hatch for "user-defined statistical
    models", including direct use of scipy.stats distributions, e.g.:

        from scipy import stats
        Custom(lambda n, rng: stats.lognorm(s=0.2, scale=1000).rvs(size=n, random_state=rng))
    """

    def __init__(self, sampler_fn: Callable[[int, np.random.Generator], np.ndarray], **bounds):
        super().__init__(**bounds)
        self._sampler_fn = sampler_fn

    def _draw(self, n, rng):
        result = np.asarray(self._sampler_fn(n, rng), dtype=float)
        if result.shape != (n,):
            raise ValueError(f"Custom sampler must return an array of shape ({n},); got {result.shape}")
        return result

    def __repr__(self):
        return f"Custom({getattr(self._sampler_fn, '__name__', repr(self._sampler_fn))})"
