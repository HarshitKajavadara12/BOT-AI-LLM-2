"""
QUANTUM-FORGE: SVM Hyperplane Classifier for Regime Detection
==============================================================
Implements the SVM-based regime classifier that was stated as a
requirement but never implemented.

Fixes Missing Concept 2.5 (Hyperplane Classifiers).

Uses a Support Vector Machine to classify market regimes based on
feature vectors. This provides a hyperplane-based decision boundary
that separates BULL / BEAR / NEUTRAL / HIGH_VOL regimes.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
from datetime import datetime

logger = logging.getLogger("SVMClassifier")


@dataclass
class SVMRegimePrediction:
    """Prediction from the SVM regime classifier."""
    predicted_regime: str
    confidence: float
    decision_margins: Dict[str, float]
    support_vectors_count: int
    timestamp: str = ""


class SVMRegimeClassifier:
    """
    SVM-based regime classifier using hyperplane separation.
    
    Uses One-vs-Rest strategy with RBF kernel for multi-class:
        - BULL (strong uptrend)
        - BEAR (strong downtrend)
        - NEUTRAL (sideways)
        - HIGH_VOL (extreme volatility)
    
    Implemented from scratch (no sklearn dependency at runtime)
    using simplified RBF kernel SVM for low-latency inference.
    """
    
    REGIMES = ["BULL", "BEAR", "NEUTRAL", "HIGH_VOL"]
    
    def __init__(
        self,
        feature_dim: int = 10,
        gamma: float = 0.1,
        C: float = 1.0,
        lookback: int = 100,
    ):
        self.feature_dim = feature_dim
        self.gamma = gamma
        self.C = C
        self.lookback = lookback
        
        # Online training buffer
        self._feature_buffer: deque = deque(maxlen=1000)
        self._label_buffer: deque = deque(maxlen=1000)
        
        # Model state (weights for linear approximation)
        # We use Nystroem approximation: map to RBF feature space, then linear SVM
        self.n_components = 50
        self._reference_points: Optional[np.ndarray] = None
        self._weights: Dict[str, np.ndarray] = {}
        self._biases: Dict[str, float] = {}
        self._fitted = False
        
        # Prediction history
        self.prediction_history: deque = deque(maxlen=200)
    
    def _rbf_kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute RBF kernel matrix K(X, Y)."""
        # K(x, y) = exp(-gamma * ||x - y||^2)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if Y.ndim == 1:
            Y = Y.reshape(1, -1)
        
        # Compute pairwise squared distances efficiently
        XX = np.sum(X ** 2, axis=1)[:, np.newaxis]
        YY = np.sum(Y ** 2, axis=1)[np.newaxis, :]
        distances = XX + YY - 2 * X @ Y.T
        
        return np.exp(-self.gamma * distances)
    
    def _nystroem_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Nystroem approximation of RBF kernel — maps to finite-dimensional
        feature space where linear SVM approximates RBF SVM.
        """
        if self._reference_points is None:
            return X  # Fallback to raw features
        
        K = self._rbf_kernel(X, self._reference_points)
        return K  # Shape: (n_samples, n_components)
    
    def _auto_label(self, features: np.ndarray, returns: np.ndarray, volatility: float) -> str:
        """
        Auto-label a data point based on simple heuristics.
        Used for online/self-supervised learning.
        """
        mean_ret = np.mean(returns[-5:]) if len(returns) >= 5 else 0.0
        
        if volatility > 0.04:
            return "HIGH_VOL"
        elif mean_ret > 0.01:
            return "BULL"
        elif mean_ret < -0.01:
            return "BEAR"
        else:
            return "NEUTRAL"
    
    def fit(self, features: np.ndarray, labels: np.ndarray):
        """
        Train the SVM classifier on labeled data.
        
        Args:
            features: (n_samples, feature_dim)
            labels: (n_samples,) — string regime labels
        """
        if len(features) < 20:
            logger.debug("Not enough data to fit SVM")
            return
        
        # Select reference points for Nystroem approximation
        n_ref = min(self.n_components, len(features))
        indices = np.random.choice(len(features), n_ref, replace=False)
        self._reference_points = features[indices].copy()
        
        # Transform to RBF feature space
        X_transformed = self._nystroem_transform(features)
        
        # One-vs-Rest: train one linear classifier per regime
        for regime in self.REGIMES:
            y_binary = (labels == regime).astype(float) * 2 - 1  # {-1, +1}
            
            # Simple SGD-based linear SVM
            w, b = self._train_linear_svm(X_transformed, y_binary)
            self._weights[regime] = w
            self._biases[regime] = b
        
        self._fitted = True
        logger.info(f"SVM classifier fitted on {len(features)} samples, "
                     f"{n_ref} reference points")
    
    def _train_linear_svm(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lr: float = 0.01,
        epochs: int = 100,
    ) -> Tuple[np.ndarray, float]:
        """Train linear SVM with hinge loss via SGD."""
        n, d = X.shape
        w = np.zeros(d)
        b = 0.0
        
        for epoch in range(epochs):
            # Shuffle
            perm = np.random.permutation(n)
            for i in perm:
                margin = y[i] * (X[i] @ w + b)
                if margin < 1:
                    w += lr * (y[i] * X[i] - (self.C / n) * w)
                    b += lr * y[i]
                else:
                    w -= lr * (self.C / n) * w
        
        return w, b
    
    def predict(self, features: np.ndarray) -> SVMRegimePrediction:
        """
        Predict market regime from feature vector.
        
        Args:
            features: (feature_dim,) or (1, feature_dim)
        
        Returns:
            SVMRegimePrediction with predicted regime and margins
        """
        if not self._fitted:
            return SVMRegimePrediction(
                predicted_regime="NEUTRAL",
                confidence=0.0,
                decision_margins={},
                support_vectors_count=0,
            )
        
        x = features.reshape(1, -1) if features.ndim == 1 else features
        x_transformed = self._nystroem_transform(x)[0]
        
        # Compute decision function for each regime
        margins = {}
        for regime in self.REGIMES:
            w = self._weights[regime]
            b = self._biases[regime]
            margins[regime] = float(x_transformed @ w + b)
        
        # Predicted class = highest margin
        predicted = max(margins, key=margins.get)
        
        # Confidence = margin of winner vs second best
        sorted_margins = sorted(margins.values(), reverse=True)
        confidence = float(np.clip(
            (sorted_margins[0] - sorted_margins[1]) / (abs(sorted_margins[0]) + 1e-10),
            0.0, 1.0
        ))
        
        pred = SVMRegimePrediction(
            predicted_regime=predicted,
            confidence=confidence,
            decision_margins=margins,
            support_vectors_count=len(self._reference_points) if self._reference_points is not None else 0,
            timestamp=datetime.now().isoformat(),
        )
        
        self.prediction_history.append(pred)
        return pred
    
    def online_update(self, features: np.ndarray, returns: np.ndarray, volatility: float):
        """
        Add a new data point for online learning.
        Periodically re-fits when buffer is large enough.
        """
        label = self._auto_label(features, returns, volatility)
        self._feature_buffer.append(features)
        self._label_buffer.append(label)
        
        # Re-fit every 100 new points
        if len(self._feature_buffer) >= 100 and len(self._feature_buffer) % 50 == 0:
            X = np.array(self._feature_buffer)
            y = np.array(self._label_buffer)
            self.fit(X, y)
