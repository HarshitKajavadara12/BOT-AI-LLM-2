"""
MPC Hedging Bridge — Wires the ModelPredictiveControl class from
core/math_engine/optimal_control.py into real-time dynamic hedging.

Missing Concept 4.4: "MPC for Dynamic Hedging"
Treats the portfolio as a state-space model and solves a receding-horizon
optimal control problem each bar to compute hedge adjustments.

State vector x = [position_value, hedge_ratio, volatility_estimate]
Control u = [delta_hedge_size]

Pipeline integration:
    QuantumCoreOrchestrator._compute_hedge_adjustment(position, forecast) →
        MPCHedgingEngine.compute_hedge(state, target)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Try importing the real MPC solver
# ---------------------------------------------------------------------------
try:
    from core.math_engine.optimal_control import ModelPredictiveControl
    _HAS_MPC = True
except Exception:
    _HAS_MPC = False
    logger.info("MPC module not importable — falling back to simplified hedging")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class HedgeAction:
    """Output of an MPC hedge computation."""
    delta_hedge: float          # Fractional hedge adjustment [-1, 1]
    hedge_ratio: float          # Target hedge ratio after action
    predicted_cost: float       # MPC objective value
    horizon_used: int
    latency_ms: float = 0.0
    method: str = "mpc"


# ---------------------------------------------------------------------------
# Simplified MPC wrapper for hedging
# ---------------------------------------------------------------------------

class MPCHedgingEngine:
    """
    Uses Model Predictive Control to compute optimal hedge adjustments
    for an open position.

    State:  x = [normalised_position, hedge_ratio, vol_estimate]
    Control: u = [delta_hedge]

    At each bar the engine:
        1. Estimates current state from live data
        2. Defines a reference trajectory (target hedge ratio)
        3. Solves receding-horizon optimisation
        4. Returns the first control action (delta_hedge)
    """

    def __init__(self, *,
                 horizon: int = 10,
                 position_cost: float = 1.0,
                 hedge_cost: float = 0.5,
                 control_cost: float = 0.1,
                 max_hedge_step: float = 0.3,
                 vol_decay: float = 0.95):
        """
        Args:
            horizon: MPC look-ahead horizon (bars).
            position_cost: Q weight on position deviation.
            hedge_cost: Q weight on hedge ratio deviation.
            control_cost: R weight on hedge adjustments (transaction cost proxy).
            max_hedge_step: Maximum absolute hedge adjustment per bar.
            vol_decay: Mean-reversion speed of vol estimate.
        """
        self.horizon = horizon
        self.max_hedge_step = max_hedge_step
        self.vol_decay = vol_decay

        # State-space matrices ------------------------------------------------
        # State: [position_value, hedge_ratio, vol_estimate]
        # The position evolves stochastically (not controlled).
        # The hedge ratio is directly adjusted by the control.
        # Volatility decays towards a long-run mean.
        n_states = 3
        n_controls = 1

        self._A = np.array([
            [1.0,  0.0,  0.0],     # position value (random walk placeholder)
            [0.0,  1.0,  0.0],     # hedge ratio persists
            [0.0,  0.0,  vol_decay] # vol mean-reverts
        ])

        self._B = np.array([
            [0.0],       # control does not directly change position
            [1.0],       # control adjusts hedge ratio
            [0.0],       # control does not affect vol
        ])

        self._Q = np.diag([position_cost, hedge_cost, 0.1])
        self._R = np.array([[control_cost]])

        # Build real MPC controller if available
        self._mpc: Optional[ModelPredictiveControl] = None
        if _HAS_MPC:
            try:
                self._mpc = ModelPredictiveControl(
                    A=self._A, B=self._B, Q=self._Q, R=self._R,
                    N=horizon, u_bounds=(-max_hedge_step, max_hedge_step)
                )
                logger.info("MPCHedgingEngine: real MPC solver loaded (horizon=%d)", horizon)
            except Exception as e:
                logger.warning("MPCHedgingEngine: MPC init failed — %s", e)

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    def compute_hedge(self, *,
                      normalised_position: float,
                      current_hedge_ratio: float,
                      vol_estimate: float,
                      target_hedge_ratio: float = 1.0,
                      target_vol: float = 0.01) -> HedgeAction:
        """
        Compute the optimal hedge adjustment.

        Args:
            normalised_position: Current position value normalised to [-1, 1].
            current_hedge_ratio: Fraction hedged [0, 1].
            vol_estimate: Recent realised vol (sigma).
            target_hedge_ratio: Desired hedge ratio (1.0 = fully hedged).
            target_vol: Long-run vol target.

        Returns:
            HedgeAction with the recommended delta_hedge.
        """
        t0 = time.perf_counter()

        x0 = np.array([normalised_position, current_hedge_ratio, vol_estimate])
        x_ref = np.array([0.0, target_hedge_ratio, target_vol])

        if self._mpc is not None:
            return self._solve_with_mpc(x0, x_ref, t0)
        else:
            return self._solve_heuristic(x0, x_ref, t0)

    # ------------------------------------------------------------------

    def _solve_with_mpc(self, x0: np.ndarray, x_ref: np.ndarray,
                        t0: float) -> HedgeAction:
        try:
            U = self._mpc.solve_mpc(x0, x_ref)
            delta = float(np.clip(U[0, 0], -self.max_hedge_step, self.max_hedge_step))
            cost = float(self._mpc.mpc_cost(U.ravel(), x0, x_ref))

            new_hr = float(np.clip(x0[1] + delta, 0.0, 1.0))

            return HedgeAction(
                delta_hedge=delta,
                hedge_ratio=new_hr,
                predicted_cost=cost,
                horizon_used=self.horizon,
                latency_ms=(time.perf_counter() - t0) * 1000,
                method="mpc"
            )
        except Exception as e:
            logger.warning("MPC solve failed: %s — falling back to heuristic", e)
            return self._solve_heuristic(x0, x_ref, t0)

    def _solve_heuristic(self, x0: np.ndarray, x_ref: np.ndarray,
                         t0: float) -> HedgeAction:
        """Proportional-control fallback when MPC solver is unavailable."""
        gap = x_ref[1] - x0[1]  # hedge_ratio gap
        delta = float(np.clip(0.3 * gap, -self.max_hedge_step, self.max_hedge_step))
        new_hr = float(np.clip(x0[1] + delta, 0.0, 1.0))

        # Simple quadratic cost estimate
        cost = float(np.sum(self._Q @ (x0 - x_ref) ** 2) +
                      self._R[0, 0] * delta ** 2)

        return HedgeAction(
            delta_hedge=delta,
            hedge_ratio=new_hr,
            predicted_cost=cost,
            horizon_used=1,
            latency_ms=(time.perf_counter() - t0) * 1000,
            method="heuristic"
        )
