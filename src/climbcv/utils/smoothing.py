from __future__ import annotations

import numpy as np


class OneEuroFilter:
    """Adaptive low-pass filter for time series data (One Euro)."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0) -> None:
        if min_cutoff <= 0 or d_cutoff <= 0:
            raise ValueError("min_cutoff and d_cutoff must be > 0")
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._last_time: float | None = None
        self._x_prev: np.ndarray | None = None
        self._dx_prev: np.ndarray | None = None

    def reset(self) -> None:
        self._last_time = None
        self._x_prev = None
        self._dx_prev = None

    def _alpha(self, cutoff: np.ndarray | float, dt: float) -> np.ndarray:
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def apply(self, x: np.ndarray, timestamp_s: float) -> np.ndarray:
        if self._last_time is None or self._x_prev is None or self._dx_prev is None:
            self._last_time = timestamp_s
            self._x_prev = x
            self._dx_prev = np.zeros_like(x)
            return x

        dt = max(timestamp_s - self._last_time, 1e-6)
        dx = (x - self._x_prev) / dt
        alpha_d = self._alpha(self.d_cutoff, dt)
        dx_hat = alpha_d * dx + (1.0 - alpha_d) * self._dx_prev

        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        cutoff = np.maximum(cutoff, 1e-6)
        alpha = self._alpha(cutoff, dt)
        x_hat = alpha * x + (1.0 - alpha) * self._x_prev

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._last_time = timestamp_s
        return x_hat
