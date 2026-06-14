"""
QUANTUM-FORGE: Gaussian Process Live Prediction Bridge
=======================================================
Wraps the OnlineGaussianProcess from intelligence/probabilistic_ml
and provides live uncertainty-aware predictions for the main pipeline.

GP predictions include:
  - Point estimate (mean prediction)
  - Uncertainty (std dev)
  - Prediction interval (95% CI)
  - Calibration score (how well uncertainty matches reality)
"""

import numpy as np
import logging
import torch
from typing import Dict, Optional, Tuple
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger("GPBridge")


@dataclass
class GPPrediction:
    """Prediction from the Gaussian Process."""
    mean: float              # Point estimate [-1, 1]
    std: float               # Uncertainty
    lower_95: float          # 95% CI lower bound
    upper_95: float          # 95% CI upper bound
    calibration: float       # How well-calibrated (0=poor, 1=perfect)
    n_training_points: int   # How much data GP has seen


class GPPredictionBridge:
    """
    Online GP that updates incrementally with each new feature+return pair.
    
    Gracefully degrades to neutral (0.0) if gpytorch is not available.
    """

    def __init__(self, feature_dim: int = 32, max_points: int = 500):
        self.feature_dim = feature_dim
        self.max_points = max_points
        self._gp = None
        self._available = False
        
        # Buffers for tracking calibration
        self._predictions: deque = deque(maxlen=200)
        self._actuals: deque = deque(maxlen=200)

        # Feature/target buffers (for lazy init)
        self._X_buffer: deque = deque(maxlen=max_points)
        self._y_buffer: deque = deque(maxlen=max_points)
        self._initialized = False
        self._min_init_samples = 30

        try:
            import gpytorch  # noqa: F401
            self._available = True
            logger.info("GPBridge: gpytorch available — GP predictions enabled")
        except ImportError:
            logger.warning("GPBridge: gpytorch not installed — GP predictions disabled")

    def update(self, features: np.ndarray, actual_return: float):
        """
        Feed a new observation (features at time t, return at t+1).
        
        This is called after we observe the actual result.
        """
        self._X_buffer.append(features.copy())
        self._y_buffer.append(np.clip(actual_return * 10, -1, 1))  # Scale returns

        if not self._available:
            return

        if not self._initialized and len(self._X_buffer) >= self._min_init_samples:
            self._initialize_gp()
        elif self._initialized and len(self._X_buffer) % 10 == 0:
            self._update_gp()

    def predict(self, features: np.ndarray) -> GPPrediction:
        """
        Get GP prediction with uncertainty for the given features.
        
        Returns neutral prediction if GP is not available or not yet trained.
        """
        if not self._initialized or not self._available:
            return GPPrediction(
                mean=0.0, std=1.0, lower_95=-1.0, upper_95=1.0,
                calibration=0.5, n_training_points=len(self._X_buffer),
            )

        try:
            x_tensor = torch.FloatTensor(features).unsqueeze(0)
            mean, std = self._gp.predict(x_tensor, return_std=True)

            mean_val = float(mean.item())
            std_val = float(std.item()) if std is not None else 0.5

            # Clip prediction to reasonable range
            mean_val = np.clip(mean_val, -1.0, 1.0)
            std_val = max(std_val, 0.01)

            lower = mean_val - 1.96 * std_val
            upper = mean_val + 1.96 * std_val

            # Track for calibration
            self._predictions.append((mean_val, std_val))

            return GPPrediction(
                mean=mean_val,
                std=std_val,
                lower_95=float(np.clip(lower, -2, 2)),
                upper_95=float(np.clip(upper, -2, 2)),
                calibration=self._compute_calibration(),
                n_training_points=len(self._X_buffer),
            )

        except Exception as e:
            logger.debug(f"GP prediction error: {e}")
            return GPPrediction(
                mean=0.0, std=1.0, lower_95=-1.0, upper_95=1.0,
                calibration=0.5, n_training_points=len(self._X_buffer),
            )

    def _initialize_gp(self):
        """Initialize the OnlineGP with buffered data."""
        try:
            from intelligence.probabilistic_ml.gaussian_processes import OnlineGaussianProcess

            X = torch.FloatTensor(np.array(list(self._X_buffer)))
            y = torch.FloatTensor(np.array(list(self._y_buffer)))

            self._gp = OnlineGaussianProcess(
                initial_x=X,
                initial_y=y,
                kernel_type='rbf',
                max_inducing=min(self.max_points, len(X)),
                forgetting_factor=0.995,
            )
            self._initialized = True
            logger.info(f"GPBridge: Initialized with {len(X)} samples")

        except Exception as e:
            logger.warning(f"GPBridge: Failed to initialize GP: {e}")
            self._available = False

    def _update_gp(self):
        """Incrementally update GP with recent data."""
        if self._gp is None:
            return

        try:
            # Get last 10 new points
            n_new = min(10, len(self._X_buffer))
            recent_X = list(self._X_buffer)[-n_new:]
            recent_y = list(self._y_buffer)[-n_new:]

            X = torch.FloatTensor(np.array(recent_X))
            y = torch.FloatTensor(np.array(recent_y))

            self._gp.update(X, y)

        except Exception as e:
            logger.debug(f"GP update failed: {e}")

    def _compute_calibration(self) -> float:
        """
        Compute calibration score: what fraction of actuals fall within
        the predicted 95% CI?
        """
        if len(self._predictions) < 10 or len(self._actuals) < 10:
            return 0.5

        n_in_ci = 0
        n_total = min(len(self._predictions), len(self._actuals))

        preds = list(self._predictions)[-n_total:]
        acts = list(self._actuals)[-n_total:]

        for (mean, std), actual in zip(preds, acts):
            lower = mean - 1.96 * std
            upper = mean + 1.96 * std
            if lower <= actual <= upper:
                n_in_ci += 1

        # Ideal calibration for 95% CI is 0.95
        coverage = n_in_ci / n_total
        # Score: 1.0 if coverage=0.95, decreasing otherwise
        return float(1.0 - abs(coverage - 0.95) / 0.95)

    def get_status(self) -> Dict:
        """Return GP status for monitoring."""
        return {
            "available": self._available,
            "initialized": self._initialized,
            "n_training_points": len(self._X_buffer),
            "calibration": self._compute_calibration(),
        }
