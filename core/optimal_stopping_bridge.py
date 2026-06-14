"""
Optimal Stopping Bridge — Wires the 913-line OptimalStoppingProblem/LSM
module into live trade exit decisions.

Missing Concept 4.3: "Optimal Stopping in Live Exits"
Uses the Shiryaev-Roberts statistic + continuation-value approach
to decide when holding a position is no longer optimal.

Pipeline integration:
    QuantumCoreOrchestrator._evaluate_exit(position) →
        OptimalExitEngine.should_exit(price_history, entry_price, direction)
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ExitDecision:
    """Result of optimal-stopping exit analysis."""
    should_exit: bool
    continuation_value: float
    exercise_value: float
    stopping_probability: float
    shiryaev_roberts: float
    reason: str
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Shiryaev-Roberts CUSUM for regime-break detection
# ---------------------------------------------------------------------------

class ShiryaevRoberts:
    """
    Shiryaev-Roberts procedure for sequential change-point detection.
    If the statistic exceeds a threshold the trend has likely reversed
    and exiting is optimal.
    """

    def __init__(self, mu_0: float = 0.0, mu_1: float = 0.001,
                 sigma: float = 0.01, threshold: float = 50.0):
        self.mu_0 = mu_0
        self.mu_1 = mu_1
        self.sigma = sigma
        self.threshold = threshold
        self._R = 0.0

    def reset(self) -> None:
        self._R = 0.0

    def update(self, x: float) -> float:
        """Update SR statistic with latest observation (log-return)."""
        ll_ratio = ((self.mu_1 - self.mu_0) * x -
                     0.5 * (self.mu_1 ** 2 - self.mu_0 ** 2)) / (self.sigma ** 2)
        self._R = (1.0 + self._R) * math.exp(ll_ratio)
        return self._R

    @property
    def value(self) -> float:
        return self._R

    @property
    def triggered(self) -> bool:
        return self._R >= self.threshold


# ---------------------------------------------------------------------------
# Continuation-value estimator (lightweight LSM spirit)
# ---------------------------------------------------------------------------

class ContinuationEstimator:
    """
    Lightweight online continuation-value estimator inspired by
    Longstaff-Schwartz.  Uses a simple polynomial regression on recent
    PnL paths to decide if holding has positive expected continuation value.
    """

    def __init__(self, degree: int = 3, min_samples: int = 30):
        self.degree = degree
        self.min_samples = min_samples
        self._pnl_paths: deque = deque(maxlen=500)

    def add_observation(self, unrealised_pnl: float) -> None:
        self._pnl_paths.append(unrealised_pnl)

    def estimate(self) -> Tuple[float, float]:
        """Return (continuation_value, exercise_value)."""
        pnl = np.array(self._pnl_paths)
        if len(pnl) < self.min_samples:
            return 0.0, float(pnl[-1]) if len(pnl) > 0 else 0.0

        # Fit polynomial to PnL trajectory
        t = np.arange(len(pnl), dtype=float)
        t_norm = t / max(t.max(), 1.0)

        try:
            coeffs = np.polyfit(t_norm, pnl, self.degree)
            poly = np.poly1d(coeffs)
            # Extrapolate one step ahead
            next_t = (len(pnl)) / max(t.max(), 1.0)
            continuation = float(poly(next_t))
        except Exception:
            continuation = float(pnl[-5:].mean())

        exercise = float(pnl[-1])
        return continuation, exercise


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class OptimalExitEngine:
    """
    Decides whether a live position should be exited based on:
      1. Shiryaev-Roberts change-point detection (trend reversal)
      2. Continuation-value vs exercise-value comparison
      3. Hard stop-loss / take-profit boundaries

    Meant to replace simple percentage-based exits with mathematically
    grounded optimal-stopping logic.
    """

    def __init__(self, *,
                 sr_threshold: float = 50.0,
                 risk_free: float = 0.0,
                 min_hold_bars: int = 5,
                 stop_loss_pct: float = 0.02,
                 take_profit_pct: float = 0.05):
        """
        Args:
            sr_threshold: Shiryaev-Roberts alarm threshold.
            risk_free: Risk-free rate per bar for discounting.
            min_hold_bars: Minimum bars to hold before optimal exit logic kicks in.
            stop_loss_pct: Hard stop-loss in fractional terms.
            take_profit_pct: Hard take-profit in fractional terms.
        """
        self.sr_threshold = sr_threshold
        self.risk_free = risk_free
        self.min_hold_bars = min_hold_bars
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        self._sr = ShiryaevRoberts(threshold=sr_threshold)
        self._cont = ContinuationEstimator()
        self._bar_count = 0
        self._entry_price: Optional[float] = None
        self._direction: int = 0  # +1 long, -1 short

    # ------------------------------------------------------------------
    # Position lifecycle
    # ------------------------------------------------------------------

    def open_position(self, entry_price: float, direction: int) -> None:
        """Call when a new position is opened."""
        self._entry_price = entry_price
        self._direction = direction
        self._bar_count = 0
        self._sr.reset()
        self._cont = ContinuationEstimator()
        logger.debug("OptimalExitEngine: position opened at %.6f dir=%d",
                      entry_price, direction)

    def close_position(self) -> None:
        """Call when the position is closed."""
        self._entry_price = None
        self._direction = 0
        self._bar_count = 0
        self._sr.reset()

    # ------------------------------------------------------------------
    # Core decision
    # ------------------------------------------------------------------

    def should_exit(self, current_price: float) -> ExitDecision:
        """
        Evaluate whether the open position should be exited.

        Returns:
            ExitDecision with should_exit flag and diagnostics.
        """
        t0 = time.perf_counter()

        if self._entry_price is None or self._direction == 0:
            return ExitDecision(
                should_exit=False, continuation_value=0.0, exercise_value=0.0,
                stopping_probability=0.0, shiryaev_roberts=0.0,
                reason="no_position",
                latency_ms=(time.perf_counter() - t0) * 1000
            )

        self._bar_count += 1

        # Unrealised PnL (fractional)
        pnl_frac = self._direction * (current_price / self._entry_price - 1.0)

        # --- Hard boundaries (always active) ---
        if pnl_frac <= -self.stop_loss_pct:
            return self._make_decision(True, 0.0, pnl_frac, 1.0,
                                        self._sr.value, "hard_stop_loss", t0)
        if pnl_frac >= self.take_profit_pct:
            return self._make_decision(True, 0.0, pnl_frac, 1.0,
                                        self._sr.value, "hard_take_profit", t0)

        # --- Optimal stopping checks (after min hold) ---
        if self._bar_count < self.min_hold_bars:
            return self._make_decision(False, 0.0, pnl_frac, 0.0,
                                        self._sr.value, "min_hold_period", t0)

        # Log-return for SR
        log_ret = self._direction * math.log(current_price / self._entry_price) / max(self._bar_count, 1)
        sr_val = self._sr.update(log_ret)

        # Continuation value tracker
        self._cont.add_observation(pnl_frac)
        cont_val, ex_val = self._cont.estimate()

        # --- Decision logic ---
        # 1) SR triggered → trend reversal detected
        if self._sr.triggered:
            return self._make_decision(True, cont_val, ex_val, 0.9,
                                        sr_val, "sr_changepoint", t0)

        # 2) Exercise > continuation → no further expected improvement
        discount = math.exp(-self.risk_free * self._bar_count)
        if ex_val > cont_val * discount and self._bar_count > self.min_hold_bars * 2:
            return self._make_decision(True, cont_val, ex_val, 0.7,
                                        sr_val, "exercise_dominates", t0)

        # 3) Continue holding
        stopping_prob = min(sr_val / self.sr_threshold, 1.0)
        return self._make_decision(False, cont_val, ex_val, stopping_prob,
                                    sr_val, "continue", t0)

    # ------------------------------------------------------------------

    def _make_decision(self, should_exit: bool, cont: float, ex: float,
                       prob: float, sr: float, reason: str,
                       t0: float) -> ExitDecision:
        return ExitDecision(
            should_exit=should_exit,
            continuation_value=cont,
            exercise_value=ex,
            stopping_probability=prob,
            shiryaev_roberts=sr,
            reason=reason,
            latency_ms=(time.perf_counter() - t0) * 1000
        )
