"""
Conformal Prediction for Financial Risk Assessment
Uncertainty quantification with distribution-free coverage guarantees
"""

import torch
import numpy as np
from typing import Optional, Tuple, Dict, Any, List, Union, Callable
from abc import ABC, abstractmethod
import warnings
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
import pandas as pd


class ConformityScore(ABC):
    """Abstract base class for conformity scores"""
    
    @abstractmethod
    def compute_score(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """Compute conformity scores"""
        pass


class AbsoluteError(ConformityScore):
    """Absolute error conformity score"""
    
    def compute_score(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> np.ndarray:
        return np.abs(y_true - y_pred)


class SignedError(ConformityScore):
    """Signed error conformity score for asymmetric intervals"""
    
    def compute_score(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> np.ndarray:
        return y_true - y_pred


class NormalizedError(ConformityScore):
    """Error normalized by prediction magnitude"""
    
    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon
    
    def compute_score(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> np.ndarray:
        return np.abs(y_true - y_pred) / (np.abs(y_pred) + self.epsilon)


class QuantileScore(ConformityScore):
    """Quantile-based conformity score"""
    
    def __init__(self, quantile: float = 0.5):
        self.quantile = quantile
    
    def compute_score(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> np.ndarray:
        errors = y_true - y_pred
        return np.where(errors >= 0, self.quantile * errors, (self.quantile - 1) * errors)


class ConditionalConformityScore(ConformityScore):
    """Conformity score that adapts to input features"""
    
    def __init__(
        self,
        base_score: ConformityScore,
        conditioning_model: Optional[BaseEstimator] = None
    ):
        self.base_score = base_score
        self.conditioning_model = conditioning_model
        self.fitted = False
    
    def fit(self, X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray):
        """Fit conditioning model"""
        base_scores = self.base_score.compute_score(y_true, y_pred)
        
        if self.conditioning_model is not None:
            self.conditioning_model.fit(X, base_scores)
            self.fitted = True
    
    def compute_score(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        X: Optional[np.ndarray] = None
    ) -> np.ndarray:
        base_scores = self.base_score.compute_score(y_true, y_pred)
        
        if self.fitted and X is not None and self.conditioning_model is not None:
            # Normalize by predicted conformity score
            predicted_scores = self.conditioning_model.predict(X)
            predicted_scores = np.maximum(predicted_scores, 1e-8)  # Avoid division by zero
            return base_scores / predicted_scores
        
        return base_scores


class ConformalPredictor:
    """
    Base conformal predictor implementation
    """
    
    def __init__(
        self,
        model: Any,
        conformity_score: ConformityScore,
        alpha: float = 0.1
    ):
        """
        Initialize conformal predictor
        
        Args:
            model: Underlying prediction model
            conformity_score: Conformity score function
            alpha: Miscoverage level (1-alpha is target coverage)
        """
        self.model = model
        self.conformity_score = conformity_score
        self.alpha = alpha
        self.calibration_scores = None
        self.fitted = False
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_cal: np.ndarray,
        y_cal: np.ndarray
    ):
        """
        Fit conformal predictor
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_cal: Calibration features
            y_cal: Calibration targets
        """
        # Train base model
        self.model.fit(X_train, y_train)
        
        # Get calibration predictions
        y_cal_pred = self.model.predict(X_cal)
        
        # Compute calibration scores
        if isinstance(self.conformity_score, ConditionalConformityScore):
            self.conformity_score.fit(X_cal, y_cal, y_cal_pred)
            self.calibration_scores = self.conformity_score.compute_score(y_cal, y_cal_pred, X_cal)
        else:
            self.calibration_scores = self.conformity_score.compute_score(y_cal, y_cal_pred)
        
        self.fitted = True
    
    def predict(
        self,
        X_test: np.ndarray,
        return_intervals: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Make conformal predictions
        
        Args:
            X_test: Test features
            return_intervals: Whether to return prediction intervals
        
        Returns:
            If return_intervals=False: point predictions
            If return_intervals=True: (predictions, lower_bounds, upper_bounds)
        """
        if not self.fitted:
            raise ValueError("Conformal predictor must be fitted before making predictions")
        
        # Get point predictions
        y_pred = self.model.predict(X_test)
        
        if not return_intervals:
            return y_pred
        
        # Compute quantile of calibration scores
        n_cal = len(self.calibration_scores)
        quantile_level = (1 - self.alpha) * (1 + 1/n_cal)
        quantile = np.quantile(self.calibration_scores, quantile_level)
        
        # Create prediction intervals
        if isinstance(self.conformity_score, SignedError):
            # Asymmetric intervals for signed error
            lower_bounds = y_pred - quantile
            upper_bounds = y_pred + quantile
        else:
            # Symmetric intervals
            lower_bounds = y_pred - quantile
            upper_bounds = y_pred + quantile
        
        return y_pred, lower_bounds, upper_bounds
    
    def predict_quantiles(
        self,
        X_test: np.ndarray,
        quantiles: List[float]
    ) -> Dict[float, np.ndarray]:
        """
        Predict multiple quantiles
        
        Args:
            X_test: Test features
            quantiles: List of quantile levels
        
        Returns:
            Dictionary mapping quantile levels to predictions
        """
        if not self.fitted:
            raise ValueError("Conformal predictor must be fitted before making predictions")
        
        y_pred = self.model.predict(X_test)
        n_cal = len(self.calibration_scores)
        
        results = {}
        
        for q in quantiles:
            if q <= 0 or q >= 1:
                raise ValueError("Quantiles must be between 0 and 1")
            
            # Adjust quantile for finite sample
            adjusted_q = q * (1 + 1/n_cal)
            score_quantile = np.quantile(self.calibration_scores, adjusted_q)
            
            if isinstance(self.conformity_score, SignedError):
                if q < 0.5:
                    results[q] = y_pred - score_quantile
                else:
                    results[q] = y_pred + score_quantile
            else:
                results[q] = y_pred + score_quantile * (2*q - 1)
        
        return results


class SplitConformalPredictor(ConformalPredictor):
    """
    Split conformal predictor with automatic train/calibration split
    """
    
    def __init__(
        self,
        model: Any,
        conformity_score: ConformityScore,
        alpha: float = 0.1,
        calibration_ratio: float = 0.2,
        random_state: Optional[int] = None
    ):
        super().__init__(model, conformity_score, alpha)
        self.calibration_ratio = calibration_ratio
        self.random_state = random_state
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit with automatic train/calibration split
        
        Args:
            X: Features
            y: Targets
        """
        # Split data
        X_train, X_cal, y_train, y_cal = train_test_split(
            X, y,
            test_size=self.calibration_ratio,
            random_state=self.random_state
        )
        
        # Call parent fit method
        super().fit(X_train, y_train, X_cal, y_cal)


class AdaptiveConformalPredictor(ConformalPredictor):
    """
    Adaptive conformal predictor that adjusts to changing distributions
    """
    
    def __init__(
        self,
        model: Any,
        conformity_score: ConformityScore,
        alpha: float = 0.1,
        adaptation_rate: float = 0.01,
        window_size: int = 100
    ):
        super().__init__(model, conformity_score, alpha)
        self.adaptation_rate = adaptation_rate
        self.window_size = window_size
        self.score_buffer = []
        self.current_quantile = None
    
    def update(self, X_new: np.ndarray, y_new: np.ndarray):
        """
        Update predictor with new observations
        
        Args:
            X_new: New features
            y_new: New targets (observed after prediction)
        """
        if not self.fitted:
            raise ValueError("Predictor must be fitted before updating")
        
        # Get predictions for new data
        y_pred = self.model.predict(X_new)
        
        # Compute new conformity scores
        if isinstance(self.conformity_score, ConditionalConformityScore):
            new_scores = self.conformity_score.compute_score(y_new, y_pred, X_new)
        else:
            new_scores = self.conformity_score.compute_score(y_new, y_pred)
        
        # Update score buffer
        self.score_buffer.extend(new_scores.tolist())
        
        # Maintain window size
        if len(self.score_buffer) > self.window_size:
            self.score_buffer = self.score_buffer[-self.window_size:]
        
        # Update quantile estimate
        if len(self.score_buffer) >= 10:  # Minimum samples for stable estimate
            new_quantile = np.quantile(self.score_buffer, 1 - self.alpha)
            
            if self.current_quantile is None:
                self.current_quantile = new_quantile
            else:
                # Exponential moving average
                self.current_quantile = (
                    (1 - self.adaptation_rate) * self.current_quantile +
                    self.adaptation_rate * new_quantile
                )
    
    def predict(
        self,
        X_test: np.ndarray,
        return_intervals: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Make adaptive predictions"""
        if not self.fitted:
            raise ValueError("Conformal predictor must be fitted before making predictions")
        
        y_pred = self.model.predict(X_test)
        
        if not return_intervals:
            return y_pred
        
        # Use adaptive quantile if available, otherwise use calibration quantile
        if self.current_quantile is not None:
            quantile = self.current_quantile
        else:
            n_cal = len(self.calibration_scores)
            quantile_level = (1 - self.alpha) * (1 + 1/n_cal)
            quantile = np.quantile(self.calibration_scores, quantile_level)
        
        # Create intervals
        lower_bounds = y_pred - quantile
        upper_bounds = y_pred + quantile
        
        return y_pred, lower_bounds, upper_bounds


class ConditionalConformalPredictor(ConformalPredictor):
    """
    Conformal predictor with conditional coverage
    """
    
    def __init__(
        self,
        model: Any,
        conformity_score: ConformityScore,
        alpha: float = 0.1,
        conditioning_features: Optional[List[int]] = None
    ):
        super().__init__(model, conformity_score, alpha)
        self.conditioning_features = conditioning_features
        self.conditional_quantiles = {}
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_cal: np.ndarray,
        y_cal: np.ndarray
    ):
        """Fit with conditional calibration"""
        # Train base model
        self.model.fit(X_train, y_train)
        
        # Get calibration predictions
        y_cal_pred = self.model.predict(X_cal)
        
        # Compute calibration scores
        if isinstance(self.conformity_score, ConditionalConformityScore):
            self.conformity_score.fit(X_cal, y_cal, y_cal_pred)
            scores = self.conformity_score.compute_score(y_cal, y_cal_pred, X_cal)
        else:
            scores = self.conformity_score.compute_score(y_cal, y_cal_pred)
        
        # Group by conditioning features
        if self.conditioning_features is not None:
            conditioning_data = X_cal[:, self.conditioning_features]
            
            # Simple binning strategy (could be improved)
            for i, feature_idx in enumerate(self.conditioning_features):
                feature_values = conditioning_data[:, i]
                # Create bins based on quantiles
                bins = np.quantile(feature_values, [0, 0.33, 0.67, 1.0])
                bin_indices = np.digitize(feature_values, bins) - 1
                bin_indices = np.clip(bin_indices, 0, 2)
                
                for bin_idx in range(3):
                    mask = bin_indices == bin_idx
                    if np.sum(mask) > 5:  # Minimum samples per bin
                        bin_scores = scores[mask]
                        n_bin = len(bin_scores)
                        quantile_level = (1 - self.alpha) * (1 + 1/n_bin)
                        quantile = np.quantile(bin_scores, quantile_level)
                        self.conditional_quantiles[(i, bin_idx)] = quantile
        
        self.calibration_scores = scores
        self.fitted = True


class FinancialConformalPredictor:
    """
    Specialized conformal predictor for financial applications
    """
    
    def __init__(
        self,
        model: Any,
        alpha: float = 0.1,
        volatility_adaptive: bool = True,
        asymmetric_intervals: bool = True
    ):
        self.model = model
        self.alpha = alpha
        self.volatility_adaptive = volatility_adaptive
        self.asymmetric_intervals = asymmetric_intervals
        
        # Choose appropriate conformity score
        if asymmetric_intervals:
            self.conformity_score = SignedError()
        elif volatility_adaptive:
            self.conformity_score = NormalizedError()
        else:
            self.conformity_score = AbsoluteError()
        
        self.predictor = SplitConformalPredictor(
            model=model,
            conformity_score=self.conformity_score,
            alpha=alpha
        )
    
    def fit(self, returns: np.ndarray, features: np.ndarray):
        """
        Fit financial conformal predictor
        
        Args:
            returns: Historical returns
            features: Feature matrix
        """
        self.predictor.fit(features, returns)
    
    def predict_returns(
        self,
        features: np.ndarray,
        confidence_level: float = 0.9
    ) -> Dict[str, np.ndarray]:
        """
        Predict returns with confidence intervals
        
        Args:
            features: Feature matrix
            confidence_level: Confidence level (e.g., 0.9 for 90% confidence)
        
        Returns:
            Dictionary with predictions and intervals
        """
        alpha = 1 - confidence_level
        self.predictor.alpha = alpha
        
        predictions, lower, upper = self.predictor.predict(features, return_intervals=True)
        
        return {
            'predictions': predictions,
            'lower_bound': lower,
            'upper_bound': upper,
            'interval_width': upper - lower
        }
    
    def predict_var(
        self,
        features: np.ndarray,
        confidence_level: float = 0.95
    ) -> Dict[str, np.ndarray]:
        """
        Predict Value at Risk with conformal intervals
        
        Args:
            features: Feature matrix
            confidence_level: VaR confidence level
        
        Returns:
            VaR predictions with uncertainty
        """
        # Predict return distribution
        return_pred = self.predict_returns(features, confidence_level=0.9)
        
        # Compute VaR (negative of lower quantile)
        var_estimate = -return_pred['lower_bound']
        
        # Uncertainty in VaR estimate
        var_uncertainty = return_pred['interval_width']
        
        return {
            'var_estimate': var_estimate,
            'var_lower': var_estimate - var_uncertainty/2,
            'var_upper': var_estimate + var_uncertainty/2,
            'var_uncertainty': var_uncertainty
        }


def evaluate_coverage(
    y_true: np.ndarray,
    predictions: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    confidence_level: float = 0.9
) -> Dict[str, float]:
    """
    Evaluate coverage properties of conformal predictions
    
    Args:
        y_true: True values
        predictions: Point predictions
        lower_bounds: Lower bounds of intervals
        upper_bounds: Upper bounds of intervals
        confidence_level: Target confidence level
    
    Returns:
        Coverage metrics
    """
    # Coverage rate
    in_interval = (y_true >= lower_bounds) & (y_true <= upper_bounds)
    coverage_rate = np.mean(in_interval)
    
    # Interval width
    interval_widths = upper_bounds - lower_bounds
    avg_width = np.mean(interval_widths)
    
    # Prediction error
    prediction_errors = np.abs(y_true - predictions)
    avg_error = np.mean(prediction_errors)
    
    # Coverage efficiency (narrower intervals are better)
    efficiency = coverage_rate / avg_width if avg_width > 0 else 0
    
    return {
        'coverage_rate': coverage_rate,
        'target_coverage': confidence_level,
        'coverage_gap': abs(coverage_rate - confidence_level),
        'average_width': avg_width,
        'average_error': avg_error,
        'efficiency': efficiency
    }


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    print("Testing Conformal Prediction...")
    
    # Generate synthetic financial data
    n_samples = 1000
    n_features = 5
    
    # Features (market indicators)
    X = np.random.randn(n_samples, n_features)
    
    # Returns with heteroskedastic noise
    volatility = 0.01 + 0.005 * np.abs(X[:, 0])  # Vol depends on first feature
    noise = np.random.normal(0, volatility)
    y = 0.5 * X[:, 0] + 0.3 * X[:, 1] - 0.2 * X[:, 2] + noise
    
    # Split data
    train_size = int(0.6 * n_samples)
    cal_size = int(0.2 * n_samples)
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_cal = X[train_size:train_size+cal_size]
    y_cal = y[train_size:train_size+cal_size]
    X_test = X[train_size+cal_size:]
    y_test = y[train_size+cal_size:]
    
    # Test different conformity scores
    from sklearn.ensemble import RandomForestRegressor
    
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    
    conformity_scores = {
        'absolute': AbsoluteError(),
        'normalized': NormalizedError(),
        'signed': SignedError()
    }
    
    results = {}
    
    for score_name, score_func in conformity_scores.items():
        print(f"\nTesting {score_name} conformity score...")
        
        # Create conformal predictor
        cp = ConformalPredictor(model, score_func, alpha=0.1)
        cp.fit(X_train, y_train, X_cal, y_cal)
        
        # Make predictions
        pred, lower, upper = cp.predict(X_test, return_intervals=True)
        
        # Evaluate coverage
        metrics = evaluate_coverage(y_test, pred, lower, upper, confidence_level=0.9)
        results[score_name] = metrics
        
        print(f"  Coverage rate: {metrics['coverage_rate']:.3f}")
        print(f"  Average width: {metrics['average_width']:.4f}")
        print(f"  Average error: {metrics['average_error']:.4f}")
    
    print("\nTesting Financial Conformal Predictor...")
    
    # Test financial-specific predictor
    financial_cp = FinancialConformalPredictor(
        model=RandomForestRegressor(n_estimators=50, random_state=42),
        alpha=0.1,
        volatility_adaptive=True,
        asymmetric_intervals=True
    )
    
    # Combine train and calibration data for simple interface
    X_combined = np.vstack([X_train, X_cal])
    y_combined = np.hstack([y_train, y_cal])
    
    financial_cp.fit(y_combined, X_combined)
    
    # Predict returns
    return_pred = financial_cp.predict_returns(X_test, confidence_level=0.9)
    
    # Evaluate
    financial_metrics = evaluate_coverage(
        y_test,
        return_pred['predictions'],
        return_pred['lower_bound'],
        return_pred['upper_bound'],
        confidence_level=0.9
    )
    
    print(f"Financial CP coverage rate: {financial_metrics['coverage_rate']:.3f}")
    print(f"Financial CP average width: {financial_metrics['average_width']:.4f}")
    
    # Test VaR prediction
    var_pred = financial_cp.predict_var(X_test, confidence_level=0.95)
    print(f"Average VaR estimate: {np.mean(var_pred['var_estimate']):.4f}")
    print(f"Average VaR uncertainty: {np.mean(var_pred['var_uncertainty']):.4f}")
    
    # Test adaptive predictor
    print("\nTesting Adaptive Conformal Predictor...")
    
    adaptive_cp = AdaptiveConformalPredictor(
        model=RandomForestRegressor(n_estimators=50, random_state=42),
        conformity_score=AbsoluteError(),
        alpha=0.1,
        adaptation_rate=0.05
    )
    
    adaptive_cp.fit(X_train, y_train, X_cal, y_cal)
    
    # Simulate online updates
    batch_size = 20
    for i in range(0, len(X_test), batch_size):
        end_idx = min(i + batch_size, len(X_test))
        X_batch = X_test[i:end_idx]
        y_batch = y_test[i:end_idx]
        
        # Make predictions (before observing true values)
        pred, lower, upper = adaptive_cp.predict(X_batch, return_intervals=True)
        
        # Update with observed values
        adaptive_cp.update(X_batch, y_batch)
    
    print("Adaptive conformal predictor updated successfully")
    
    print("\nDone!")