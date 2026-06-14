"""
QUANTUM-FORGE: Conformal Prediction & Bayesian Optimization
=============================================================
P2 6.2 — Conformal Prediction: distribution-free prediction intervals.
P2 6.3 — Bayesian Optimization: hyperparameter tuning for ML models.

Conformal prediction provides statistically valid prediction intervals
without distributional assumptions.

Bayesian optimization finds optimal hyperparameters with minimal
evaluations using Gaussian process surrogate.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Callable
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger("AdvancedML")


# ─── Conformal Prediction ───────────────────────────────────────────────

@dataclass
class ConformalInterval:
    """Prediction with conformal interval."""
    prediction: float
    lower: float
    upper: float
    coverage: float     # target coverage (e.g. 0.95)
    nonconformity: float  # nonconformity score used


class ConformalPredictor:
    """
    Split conformal prediction for trading signals.
    
    Wraps any point predictor and produces valid prediction intervals.
    
    Algorithm:
    1. Compute nonconformity scores on calibration set
    2. For new point, add prediction ± quantile(scores) 
    3. Guaranteed coverage under exchangeability assumption
    """

    def __init__(self, coverage: float = 0.95, cal_size: int = 200):
        self.coverage = coverage
        self.cal_size = cal_size
        self._scores: deque = deque(maxlen=cal_size)
        self._quantile: Optional[float] = None

    def calibrate(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calibrate using held-out data.
        
        Args:
            y_true: Actual values (calibration set)
            y_pred: Predicted values (calibration set)
        """
        scores = np.abs(y_true - y_pred)
        for s in scores:
            self._scores.append(float(s))
        self._update_quantile()

    def update(self, y_true: float, y_pred: float):
        """Online update with a single observation."""
        score = abs(y_true - y_pred)
        self._scores.append(score)
        self._update_quantile()

    def predict(self, point_prediction: float) -> ConformalInterval:
        """
        Produce a conformal prediction interval.
        
        Args:
            point_prediction: Base model's point prediction
            
        Returns:
            ConformalInterval with valid coverage guarantee
        """
        if self._quantile is None or len(self._scores) < 10:
            # Not enough calibration data — return wide interval
            return ConformalInterval(
                prediction=point_prediction,
                lower=point_prediction - 1.0,
                upper=point_prediction + 1.0,
                coverage=self.coverage,
                nonconformity=1.0,
            )

        return ConformalInterval(
            prediction=point_prediction,
            lower=point_prediction - self._quantile,
            upper=point_prediction + self._quantile,
            coverage=self.coverage,
            nonconformity=self._quantile,
        )

    def _update_quantile(self):
        """Recompute the conformal quantile."""
        if len(self._scores) < 10:
            return
        
        scores = sorted(self._scores)
        n = len(scores)
        # Quantile index for (1-alpha) coverage
        idx = int(np.ceil((n + 1) * self.coverage)) - 1
        idx = min(idx, n - 1)
        self._quantile = scores[idx]

    def get_status(self) -> Dict:
        return {
            "coverage_target": self.coverage,
            "calibration_samples": len(self._scores),
            "current_quantile": self._quantile,
        }


# ─── Bayesian Optimization ──────────────────────────────────────────────

@dataclass
class BOTrialResult:
    """Result of a single Bayesian optimization trial."""
    params: Dict[str, float]
    score: float
    iteration: int


class BayesianOptimizer:
    """
    Simple Bayesian optimization for hyperparameter tuning.
    
    Uses a surrogate model (GP or random forest) with Expected Improvement
    acquisition to find optimal hyperparameters.
    
    For production use with gpytorch, wrap this around the full BO pipeline.
    For lightweight use, this implements a simple random forest surrogate.
    """

    def __init__(
        self,
        param_bounds: Dict[str, Tuple[float, float]],
        n_initial: int = 5,
        n_iterations: int = 30,
    ):
        """
        Args:
            param_bounds: {param_name: (low, high)} for each hyperparameter
            n_initial: Number of random initial evaluations
            n_iterations: Total optimization iterations
        """
        self.param_bounds = param_bounds
        self.n_initial = n_initial
        self.n_iterations = n_iterations
        self.param_names = list(param_bounds.keys())

        self._X: List[np.ndarray] = []  # evaluated points
        self._y: List[float] = []       # scores
        self._best_score = -float("inf")
        self._best_params: Optional[Dict[str, float]] = None
        self._history: List[BOTrialResult] = []

    def optimize(
        self,
        objective_fn: Callable[[Dict[str, float]], float],
    ) -> Tuple[Dict[str, float], float]:
        """
        Run Bayesian optimization.
        
        Args:
            objective_fn: Function mapping params dict → score (higher is better)
            
        Returns:
            (best_params, best_score)
        """
        # Phase 1: Random initial points
        for i in range(self.n_initial):
            params = self._random_sample()
            score = self._evaluate(objective_fn, params, i)
            logger.info(f"BO init {i+1}/{self.n_initial}: score={score:.4f}")

        # Phase 2: Guided search
        for i in range(self.n_initial, self.n_iterations):
            params = self._suggest_next()
            score = self._evaluate(objective_fn, params, i)
            if i % 5 == 0:
                logger.info(
                    f"BO iter {i+1}/{self.n_iterations}: "
                    f"score={score:.4f}, best={self._best_score:.4f}"
                )

        return self._best_params or {}, self._best_score

    def _random_sample(self) -> Dict[str, float]:
        """Sample random point from bounds."""
        return {
            name: np.random.uniform(low, high)
            for name, (low, high) in self.param_bounds.items()
        }

    def _suggest_next(self) -> Dict[str, float]:
        """Suggest next point using Expected Improvement."""
        if len(self._X) < 3:
            return self._random_sample()

        try:
            from sklearn.ensemble import RandomForestRegressor
            
            X = np.array(self._X)
            y = np.array(self._y)
            
            # Fit surrogate
            rf = RandomForestRegressor(n_estimators=50, random_state=42)
            rf.fit(X, y)
            
            # Generate candidates
            n_candidates = 1000
            candidates = np.array([
                [np.random.uniform(*self.param_bounds[name]) for name in self.param_names]
                for _ in range(n_candidates)
            ])
            
            # Predict mean and std from RF trees
            tree_preds = np.array([tree.predict(candidates) for tree in rf.estimators_])
            mu = tree_preds.mean(axis=0)
            sigma = tree_preds.std(axis=0) + 1e-8
            
            # Expected Improvement
            best_y = max(self._y)
            z = (mu - best_y) / sigma
            from scipy.stats import norm
            ei = sigma * (z * norm.cdf(z) + norm.pdf(z))
            
            # Select best candidate
            best_idx = np.argmax(ei)
            return {
                name: float(candidates[best_idx, i])
                for i, name in enumerate(self.param_names)
            }
            
        except ImportError:
            # Fallback: random with perturbation around best
            if self._best_params is not None:
                params = {}
                for name, (low, high) in self.param_bounds.items():
                    current = self._best_params[name]
                    noise = (high - low) * 0.1 * np.random.randn()
                    params[name] = np.clip(current + noise, low, high)
                return params
            return self._random_sample()

    def _evaluate(
        self,
        objective_fn: Callable,
        params: Dict[str, float],
        iteration: int,
    ) -> float:
        """Evaluate objective and update history."""
        try:
            score = float(objective_fn(params))
        except Exception as e:
            logger.warning(f"BO evaluation failed: {e}")
            score = -float("inf")

        x = np.array([params[name] for name in self.param_names])
        self._X.append(x)
        self._y.append(score)

        if score > self._best_score:
            self._best_score = score
            self._best_params = dict(params)

        self._history.append(BOTrialResult(
            params=dict(params), score=score, iteration=iteration
        ))

        return score

    def get_history(self) -> List[Dict]:
        """Get optimization history."""
        return [
            {"iteration": r.iteration, "score": r.score, "params": r.params}
            for r in self._history
        ]
