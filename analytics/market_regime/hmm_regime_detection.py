"""
Hidden Markov Model Regime Detection Framework
Advanced regime detection using Hidden Markov Models for market state identification
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import warnings
import math
from collections import defaultdict, deque
from scipy import stats
from scipy.optimize import minimize
from scipy.special import logsumexp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans


class RegimeType(Enum):
    """Types of market regimes"""
    BULL_MARKET = "BULL_MARKET"             # Strong upward trend
    BEAR_MARKET = "BEAR_MARKET"             # Strong downward trend
    SIDEWAYS = "SIDEWAYS"                   # Low volatility, no clear trend
    HIGH_VOLATILITY = "HIGH_VOLATILITY"     # High volatility period
    LOW_VOLATILITY = "LOW_VOLATILITY"       # Low volatility period
    CRISIS = "CRISIS"                       # Extreme market stress
    RECOVERY = "RECOVERY"                   # Post-crisis recovery


class RegimeModel(Enum):
    """HMM model types"""
    GAUSSIAN = "GAUSSIAN"                   # Gaussian emissions
    MULTIVARIATE_GAUSSIAN = "MULTIVARIATE_GAUSSIAN"  # Multivariate Gaussian
    STUDENT_T = "STUDENT_T"                 # Student-t emissions
    MIXTURE = "MIXTURE"                     # Mixture of distributions


@dataclass
class RegimeState:
    """Individual regime state characteristics"""
    regime_id: int
    regime_name: str
    mean_return: float = 0.0
    volatility: float = 0.0
    
    # State-specific parameters
    persistence: float = 0.0               # Probability of staying in state
    expected_duration: float = 0.0         # Expected duration in periods
    
    # Statistical properties
    skewness: float = 0.0
    kurtosis: float = 0.0
    var_95: float = 0.0
    var_99: float = 0.0
    
    # Regime characteristics
    trend_strength: float = 0.0            # Strength of trend
    volatility_regime: str = "NORMAL"      # LOW, NORMAL, HIGH
    market_stress: float = 0.0             # Market stress indicator (0-1)
    
    # Period information
    total_periods: int = 0
    avg_period_length: float = 0.0
    longest_period: int = 0


@dataclass
class RegimeDetectionResults:
    """Results from regime detection analysis"""
    
    # Model information
    n_regimes: int = 0
    model_type: str = ""
    log_likelihood: float = 0.0
    aic: float = 0.0
    bic: float = 0.0
    
    # Regime classifications
    regime_sequence: List[int] = field(default_factory=list)
    regime_probabilities: np.ndarray = field(default_factory=lambda: np.array([]))
    smoothed_probabilities: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # State information
    regime_states: List[RegimeState] = field(default_factory=list)
    transition_matrix: np.ndarray = field(default_factory=lambda: np.array([]))
    steady_state_probs: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Time series
    dates: Optional[pd.DatetimeIndex] = None
    returns: Optional[pd.Series] = None
    
    # Regime periods
    regime_periods: List[Dict[str, Any]] = field(default_factory=list)
    
    # Model diagnostics
    convergence_achieved: bool = False
    n_iterations: int = 0
    final_log_likelihood: float = 0.0


class HMMRegimeDetector(ABC):
    """Abstract base class for HMM-based regime detection"""
    
    @abstractmethod
    def fit(self, returns: pd.Series, **kwargs) -> RegimeDetectionResults:
        """Fit HMM model to return series"""
        pass
    
    @abstractmethod
    def predict_regime(self, returns: pd.Series) -> np.ndarray:
        """Predict regime for new data"""
        pass


class GaussianHMMDetector(HMMRegimeDetector):
    """Gaussian Hidden Markov Model for regime detection"""
    
    def __init__(self, 
                 n_regimes: int = 3,
                 max_iter: int = 1000,
                 tolerance: float = 1e-6,
                 random_state: int = 42):
        
        self.n_regimes = n_regimes
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.random_state = random_state
        
        # Model parameters
        self.transition_matrix = None
        self.emission_means = None
        self.emission_vars = None
        self.initial_probs = None
        
        # Fitted model
        self.is_fitted = False
        self.log_likelihood_history = []
    
    def fit(self, returns: pd.Series, **kwargs) -> RegimeDetectionResults:
        """Fit Gaussian HMM to return series"""
        # Use deterministic initialization via KMeans' random_state; do not set global seed
        
        if len(returns) < 10:
            raise ValueError("Need at least 10 observations to fit HMM")
        
        # Prepare data
        observations = returns.values.reshape(-1, 1)
        T = len(observations)
        
        # Initialize parameters
        self._initialize_parameters(observations)
        
        # EM algorithm
        log_likelihood_prev = -np.inf
        
        for iteration in range(self.max_iter):
            # E-step: Forward-backward algorithm
            log_alpha, log_beta, log_gamma, log_xi = self._forward_backward(observations)
            
            # M-step: Update parameters
            self._update_parameters(observations, log_gamma, log_xi)
            
            # Calculate log-likelihood
            current_log_likelihood = self._calculate_log_likelihood(observations)
            self.log_likelihood_history.append(current_log_likelihood)
            
            # Check convergence
            if abs(current_log_likelihood - log_likelihood_prev) < self.tolerance:
                break
                
            log_likelihood_prev = current_log_likelihood
        
        self.is_fitted = True
        
        # Generate results
        results = self._generate_results(returns, log_gamma, iteration + 1)
        
        return results
    
    def predict_regime(self, returns: pd.Series) -> np.ndarray:
        """Predict most likely regime sequence"""
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        observations = returns.values.reshape(-1, 1)
        
        # Use Viterbi algorithm for most likely path
        path = self._viterbi_decode(observations)
        
        return path
    
    def _initialize_parameters(self, observations: np.ndarray):
        """Initialize HMM parameters"""
        
        T, D = observations.shape
        
        # Use K-means for initialization
        kmeans = KMeans(n_clusters=self.n_regimes, random_state=self.random_state)
        cluster_labels = kmeans.fit_predict(observations)
        
        # Initialize transition matrix (slightly favor staying in same state)
        self.transition_matrix = np.full((self.n_regimes, self.n_regimes), 
                                       0.1 / (self.n_regimes - 1))
        np.fill_diagonal(self.transition_matrix, 0.9)
        
        # Keep transition matrix as initialized (slight diagonal bias);
        # avoid adding synthetic random perturbations so results rely on data-driven initialization.
        self.transition_matrix = self.transition_matrix / self.transition_matrix.sum(axis=1, keepdims=True)
        
        # Initialize emission parameters from K-means clusters
        self.emission_means = np.zeros((self.n_regimes, D))
        self.emission_vars = np.ones((self.n_regimes, D))
        
        for k in range(self.n_regimes):
            cluster_data = observations[cluster_labels == k]
            if len(cluster_data) > 0:
                self.emission_means[k] = cluster_data.mean(axis=0)
                self.emission_vars[k] = cluster_data.var(axis=0)
                
                # Ensure minimum variance
                self.emission_vars[k] = np.maximum(self.emission_vars[k], 1e-6)
        
        # Initial state probabilities (uniform)
        self.initial_probs = np.ones(self.n_regimes) / self.n_regimes
    
    def _forward_backward(self, observations: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Forward-backward algorithm"""
        
        T, D = observations.shape
        
        # Forward pass
        log_alpha = np.zeros((T, self.n_regimes))
        
        # Initial step
        emission_probs = self._calculate_emission_probs(observations[0])
        log_alpha[0] = np.log(self.initial_probs) + np.log(emission_probs)
        
        # Forward recursion
        for t in range(1, T):
            emission_probs = self._calculate_emission_probs(observations[t])
            for j in range(self.n_regimes):
                log_alpha[t, j] = (
                    logsumexp(log_alpha[t-1] + np.log(self.transition_matrix[:, j])) +
                    np.log(emission_probs[j])
                )
        
        # Backward pass
        log_beta = np.zeros((T, self.n_regimes))
        
        # Final step (log(1) = 0)
        log_beta[T-1] = 0
        
        # Backward recursion
        for t in range(T-2, -1, -1):
            emission_probs = self._calculate_emission_probs(observations[t+1])
            for i in range(self.n_regimes):
                log_beta[t, i] = logsumexp(
                    np.log(self.transition_matrix[i, :]) +
                    np.log(emission_probs) +
                    log_beta[t+1]
                )
        
        # Calculate gamma (state probabilities)
        log_gamma = log_alpha + log_beta
        log_gamma = log_gamma - logsumexp(log_gamma, axis=1, keepdims=True)
        
        # Calculate xi (transition probabilities)
        log_xi = np.zeros((T-1, self.n_regimes, self.n_regimes))
        
        for t in range(T-1):
            emission_probs = self._calculate_emission_probs(observations[t+1])
            for i in range(self.n_regimes):
                for j in range(self.n_regimes):
                    log_xi[t, i, j] = (
                        log_alpha[t, i] +
                        np.log(self.transition_matrix[i, j]) +
                        np.log(emission_probs[j]) +
                        log_beta[t+1, j]
                    )
            
            # Normalize
            log_xi[t] = log_xi[t] - logsumexp(log_xi[t])
        
        return log_alpha, log_beta, log_gamma, log_xi
    
    def _calculate_emission_probs(self, observation: np.ndarray) -> np.ndarray:
        """Calculate emission probabilities for all states"""
        
        probs = np.zeros(self.n_regimes)
        
        for k in range(self.n_regimes):
            # Gaussian probability density
            diff = observation - self.emission_means[k]
            var = self.emission_vars[k]
            
            # Multivariate Gaussian (simplified for 1D)
            log_prob = -0.5 * np.sum(
                np.log(2 * np.pi * var) + (diff ** 2) / var
            )
            probs[k] = np.exp(log_prob)
        
        # Ensure numerical stability
        probs = np.maximum(probs, 1e-100)
        
        return probs
    
    def _update_parameters(self, observations: np.ndarray, 
                          log_gamma: np.ndarray, log_xi: np.ndarray):
        """M-step: Update model parameters"""
        
        T, D = observations.shape
        gamma = np.exp(log_gamma)
        xi = np.exp(log_xi)
        
        # Update initial probabilities
        self.initial_probs = gamma[0]
        
        # Update transition matrix
        for i in range(self.n_regimes):
            for j in range(self.n_regimes):
                numerator = np.sum(xi[:, i, j])
                denominator = np.sum(gamma[:-1, i])
                
                if denominator > 1e-10:
                    self.transition_matrix[i, j] = numerator / denominator
                else:
                    self.transition_matrix[i, j] = 1.0 / self.n_regimes
        
        # Normalize transition matrix
        self.transition_matrix = (
            self.transition_matrix / self.transition_matrix.sum(axis=1, keepdims=True)
        )
        
        # Update emission parameters
        for k in range(self.n_regimes):
            gamma_sum = np.sum(gamma[:, k])
            
            if gamma_sum > 1e-10:
                # Update mean
                self.emission_means[k] = np.sum(
                    gamma[:, k].reshape(-1, 1) * observations, axis=0
                ) / gamma_sum
                
                # Update variance
                diff = observations - self.emission_means[k]
                self.emission_vars[k] = np.sum(
                    gamma[:, k].reshape(-1, 1) * (diff ** 2), axis=0
                ) / gamma_sum
                
                # Ensure minimum variance
                self.emission_vars[k] = np.maximum(self.emission_vars[k], 1e-6)
    
    def _calculate_log_likelihood(self, observations: np.ndarray) -> float:
        """Calculate model log-likelihood"""
        
        T = len(observations)
        log_likelihood = 0.0
        
        # Forward pass to calculate likelihood
        log_alpha = np.zeros((T, self.n_regimes))
        
        # Initial step
        emission_probs = self._calculate_emission_probs(observations[0])
        log_alpha[0] = np.log(self.initial_probs) + np.log(emission_probs)
        
        # Forward recursion
        for t in range(1, T):
            emission_probs = self._calculate_emission_probs(observations[t])
            for j in range(self.n_regimes):
                log_alpha[t, j] = (
                    logsumexp(log_alpha[t-1] + np.log(self.transition_matrix[:, j])) +
                    np.log(emission_probs[j])
                )
        
        # Total log-likelihood
        log_likelihood = logsumexp(log_alpha[T-1])
        
        return log_likelihood
    
    def _viterbi_decode(self, observations: np.ndarray) -> np.ndarray:
        """Viterbi algorithm for most likely state sequence"""
        
        T = len(observations)
        
        # Viterbi trellis
        log_delta = np.zeros((T, self.n_regimes))
        psi = np.zeros((T, self.n_regimes), dtype=int)
        
        # Initialization
        emission_probs = self._calculate_emission_probs(observations[0])
        log_delta[0] = np.log(self.initial_probs) + np.log(emission_probs)
        
        # Forward pass
        for t in range(1, T):
            emission_probs = self._calculate_emission_probs(observations[t])
            for j in range(self.n_regimes):
                transition_scores = log_delta[t-1] + np.log(self.transition_matrix[:, j])
                psi[t, j] = np.argmax(transition_scores)
                log_delta[t, j] = np.max(transition_scores) + np.log(emission_probs[j])
        
        # Backward pass (traceback)
        path = np.zeros(T, dtype=int)
        path[T-1] = np.argmax(log_delta[T-1])
        
        for t in range(T-2, -1, -1):
            path[t] = psi[t+1, path[t+1]]
        
        return path
    
    def _generate_results(self, returns: pd.Series, log_gamma: np.ndarray, 
                         n_iterations: int) -> RegimeDetectionResults:
        """Generate comprehensive results"""
        
        results = RegimeDetectionResults()
        
        # Basic model information
        results.n_regimes = self.n_regimes
        results.model_type = "Gaussian HMM"
        results.log_likelihood = self.log_likelihood_history[-1] if self.log_likelihood_history else 0
        results.convergence_achieved = n_iterations < self.max_iter
        results.n_iterations = n_iterations
        results.final_log_likelihood = results.log_likelihood
        
        # Calculate AIC and BIC
        n_params = (
            self.n_regimes * (self.n_regimes - 1) +  # Transition matrix
            self.n_regimes +                         # Initial probabilities  
            2 * self.n_regimes                       # Emission parameters (mean, var)
        )
        
        results.aic = -2 * results.log_likelihood + 2 * n_params
        results.bic = -2 * results.log_likelihood + n_params * np.log(len(returns))
        
        # Regime sequence and probabilities
        gamma = np.exp(log_gamma)
        results.regime_sequence = np.argmax(gamma, axis=1).tolist()
        results.regime_probabilities = gamma
        results.smoothed_probabilities = gamma
        
        # Time series information
        results.dates = returns.index
        results.returns = returns
        
        # Transition matrix and steady state
        results.transition_matrix = self.transition_matrix.copy()
        results.steady_state_probs = self._calculate_steady_state()
        
        # Generate regime states
        results.regime_states = self._analyze_regime_states(returns, gamma)
        
        # Identify regime periods
        results.regime_periods = self._identify_regime_periods(returns, results.regime_sequence)
        
        return results
    
    def _calculate_steady_state(self) -> np.ndarray:
        """Calculate steady-state probabilities"""
        
        # Solve π = πP where π is the steady state
        eigenvals, eigenvecs = np.linalg.eig(self.transition_matrix.T)
        
        # Find eigenvector corresponding to eigenvalue 1
        stationary_idx = np.argmin(np.abs(eigenvals - 1.0))
        stationary = np.real(eigenvecs[:, stationary_idx])
        
        # Normalize to probabilities
        stationary = stationary / stationary.sum()
        
        return np.abs(stationary)  # Ensure positive values
    
    def _analyze_regime_states(self, returns: pd.Series, gamma: np.ndarray) -> List[RegimeState]:
        """Analyze characteristics of each regime state"""
        
        states = []
        
        for k in range(self.n_regimes):
            state = RegimeState(regime_id=k, regime_name=f"Regime_{k}")
            
            # Weight returns by state probability
            weighted_returns = returns * gamma[:, k]
            total_weight = gamma[:, k].sum()
            
            if total_weight > 0:
                # Basic statistics
                state.mean_return = weighted_returns.sum() / total_weight
                
                # Weighted variance
                weighted_var = ((returns - state.mean_return) ** 2 * gamma[:, k]).sum() / total_weight
                state.volatility = np.sqrt(weighted_var)
                
                # State persistence (diagonal of transition matrix)
                state.persistence = self.transition_matrix[k, k]
                state.expected_duration = 1 / (1 - state.persistence) if state.persistence < 1 else np.inf
                
                # Get periods where this regime is most likely
                most_likely_periods = gamma[:, k] > 0.5
                regime_returns = returns[most_likely_periods]
                
                if len(regime_returns) > 2:
                    state.skewness = stats.skew(regime_returns)
                    state.kurtosis = stats.kurtosis(regime_returns, fisher=True)
                    
                    # VaR calculations
                    if len(regime_returns) >= 20:
                        state.var_95 = abs(np.percentile(regime_returns, 5))
                        state.var_99 = abs(np.percentile(regime_returns, 1))
                
                # Regime characteristics
                state.trend_strength = abs(state.mean_return) / (state.volatility + 1e-8)
                
                # Volatility regime classification
                if state.volatility < 0.01:
                    state.volatility_regime = "LOW"
                elif state.volatility > 0.025:
                    state.volatility_regime = "HIGH"
                else:
                    state.volatility_regime = "NORMAL"
                
                # Market stress indicator (based on volatility and negative returns)
                stress_factor = min(state.volatility / 0.03, 1.0)  # Normalize by 3% daily vol
                negative_return_factor = max(-state.mean_return / 0.02, 0.0)  # Normalize by -2% daily return
                state.market_stress = min(stress_factor + negative_return_factor, 1.0)
                
                # Period analysis
                regime_sequence = np.array(gamma[:, k] > 0.5, dtype=int)
                periods = self._find_consecutive_periods(regime_sequence)
                
                state.total_periods = len(periods)
                if periods:
                    period_lengths = [end - start + 1 for start, end in periods]
                    state.avg_period_length = np.mean(period_lengths)
                    state.longest_period = max(period_lengths)
            
            states.append(state)
        
        return states
    
    def _find_consecutive_periods(self, binary_sequence: np.ndarray) -> List[Tuple[int, int]]:
        """Find consecutive periods of 1s in binary sequence"""
        
        periods = []
        start = None
        
        for i, value in enumerate(binary_sequence):
            if value == 1 and start is None:
                start = i
            elif value == 0 and start is not None:
                periods.append((start, i - 1))
                start = None
        
        # Handle case where sequence ends with 1s
        if start is not None:
            periods.append((start, len(binary_sequence) - 1))
        
        return periods
    
    def _identify_regime_periods(self, returns: pd.Series, 
                               regime_sequence: List[int]) -> List[Dict[str, Any]]:
        """Identify distinct regime periods"""
        
        periods = []
        current_regime = regime_sequence[0]
        period_start = 0
        
        for i, regime in enumerate(regime_sequence[1:], 1):
            if regime != current_regime:
                # End of current period
                period_data = {
                    'regime_id': current_regime,
                    'start_idx': period_start,
                    'end_idx': i - 1,
                    'start_date': returns.index[period_start],
                    'end_date': returns.index[i - 1],
                    'duration': i - period_start,
                    'returns': returns.iloc[period_start:i],
                    'mean_return': returns.iloc[period_start:i].mean(),
                    'volatility': returns.iloc[period_start:i].std(),
                    'total_return': (1 + returns.iloc[period_start:i]).prod() - 1
                }
                
                periods.append(period_data)
                
                # Start new period
                current_regime = regime
                period_start = i
        
        # Add final period
        if period_start < len(regime_sequence):
            period_data = {
                'regime_id': current_regime,
                'start_idx': period_start,
                'end_idx': len(regime_sequence) - 1,
                'start_date': returns.index[period_start],
                'end_date': returns.index[-1],
                'duration': len(regime_sequence) - period_start,
                'returns': returns.iloc[period_start:],
                'mean_return': returns.iloc[period_start:].mean(),
                'volatility': returns.iloc[period_start:].std(),
                'total_return': (1 + returns.iloc[period_start:]).prod() - 1
            }
            
            periods.append(period_data)
        
        return periods


class MultivariateHMMDetector(HMMRegimeDetector):
    """Multivariate Gaussian HMM for regime detection with multiple features"""
    
    def __init__(self, 
                 n_regimes: int = 3,
                 features: List[str] = ['returns', 'volatility'],
                 max_iter: int = 1000,
                 tolerance: float = 1e-6,
                 random_state: int = 42):
        
        self.n_regimes = n_regimes
        self.features = features
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.random_state = random_state
        
        # Use sklearn's GaussianMixture for multivariate case
        self.hmm_model = None
        self.scaler = StandardScaler()
        
        # Fitted model
        self.is_fitted = False
    
    def fit(self, returns: pd.Series, **kwargs) -> RegimeDetectionResults:
        """Fit multivariate HMM"""
        
        # Prepare features
        feature_data = self._prepare_features(returns, **kwargs)
        
        if feature_data.shape[1] == 0:
            raise ValueError("No valid features prepared")
        
        # Scale features
        scaled_features = self.scaler.fit_transform(feature_data)
        
        # For simplicity, use Gaussian Mixture Model as approximation
        # In a full implementation, would use proper HMM
        self.hmm_model = GaussianMixture(
            n_components=self.n_regimes,
            max_iter=self.max_iter,
            tol=self.tolerance,
            random_state=self.random_state
        )
        
        self.hmm_model.fit(scaled_features)
        self.is_fitted = True
        
        # Generate results
        results = self._generate_multivariate_results(returns, scaled_features)
        
        return results
    
    def predict_regime(self, returns: pd.Series, **kwargs) -> np.ndarray:
        """Predict regime for new data"""
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        feature_data = self._prepare_features(returns, **kwargs)
        scaled_features = self.scaler.transform(feature_data)
        
        return self.hmm_model.predict(scaled_features)
    
    def _prepare_features(self, returns: pd.Series, **kwargs) -> np.ndarray:
        """Prepare feature matrix for multivariate analysis"""
        
        features_dict = {}
        
        if 'returns' in self.features:
            features_dict['returns'] = returns.values
        
        if 'volatility' in self.features:
            # Rolling volatility
            window = kwargs.get('volatility_window', 20)
            volatility = returns.rolling(window).std().fillna(returns.std())
            features_dict['volatility'] = volatility.values
        
        if 'volume' in self.features and 'volume' in kwargs:
            features_dict['volume'] = kwargs['volume'].reindex(returns.index).fillna(0).values
        
        if 'momentum' in self.features:
            # Price momentum
            window = kwargs.get('momentum_window', 10)
            momentum = returns.rolling(window).mean().fillna(0)
            features_dict['momentum'] = momentum.values
        
        if 'mean_reversion' in self.features:
            # Mean reversion indicator
            window = kwargs.get('mean_reversion_window', 20)
            rolling_mean = returns.rolling(window).mean()
            mean_reversion = (returns - rolling_mean).fillna(0)
            features_dict['mean_reversion'] = mean_reversion.values
        
        # Combine features
        if features_dict:
            feature_matrix = np.column_stack(list(features_dict.values()))
            
            # Remove NaN rows
            valid_rows = ~np.isnan(feature_matrix).any(axis=1)
            feature_matrix = feature_matrix[valid_rows]
            
            return feature_matrix
        else:
            return np.array([]).reshape(0, 0)
    
    def _generate_multivariate_results(self, returns: pd.Series, 
                                     features: np.ndarray) -> RegimeDetectionResults:
        """Generate results for multivariate model"""
        
        results = RegimeDetectionResults()
        
        # Basic model information
        results.n_regimes = self.n_regimes
        results.model_type = "Multivariate Gaussian HMM"
        results.log_likelihood = self.hmm_model.score(features)
        
        # Regime predictions
        regime_sequence = self.hmm_model.predict(features)
        regime_probs = self.hmm_model.predict_proba(features)
        
        results.regime_sequence = regime_sequence.tolist()
        results.regime_probabilities = regime_probs
        results.smoothed_probabilities = regime_probs
        
        # Time information (align with valid features)
        valid_indices = ~np.isnan(features).any(axis=1)
        results.dates = returns.index[valid_indices]
        results.returns = returns[valid_indices]
        
        return results


class RegimeAnalyzer:
    """
    Comprehensive regime analysis framework
    """
    
    def __init__(self):
        self.detectors = {
            'gaussian_hmm': GaussianHMMDetector,
            'multivariate_hmm': MultivariateHMMDetector
        }
    
    def detect_regimes(self, 
                      returns: pd.Series,
                      method: str = 'gaussian_hmm',
                      n_regimes: int = 3,
                      **kwargs) -> RegimeDetectionResults:
        """
        Detect market regimes using specified method
        
        Args:
            returns: Return series
            method: Detection method ('gaussian_hmm', 'multivariate_hmm')
            n_regimes: Number of regimes to detect
            **kwargs: Additional parameters for specific methods
            
        Returns:
            RegimeDetectionResults object
        """
        
        if method not in self.detectors:
            raise ValueError(f"Unknown method: {method}")
        
        detector_class = self.detectors[method]
        detector = detector_class(n_regimes=n_regimes, **kwargs)
        
        results = detector.fit(returns, **kwargs)
        
        return results
    
    def compare_models(self, 
                      returns: pd.Series,
                      regime_counts: List[int] = [2, 3, 4, 5],
                      method: str = 'gaussian_hmm') -> Dict[str, Any]:
        """Compare models with different numbers of regimes"""
        
        comparison_results = {}
        
        for n_regimes in regime_counts:
            try:
                results = self.detect_regimes(returns, method, n_regimes)
                
                comparison_results[f'{n_regimes}_regimes'] = {
                    'n_regimes': n_regimes,
                    'log_likelihood': results.log_likelihood,
                    'aic': results.aic,
                    'bic': results.bic,
                    'convergence': results.convergence_achieved,
                    'regime_persistence': np.diag(results.transition_matrix).mean() if results.transition_matrix.size > 0 else 0
                }
                
            except Exception as e:
                print(f"Model with {n_regimes} regimes failed: {e}")
                continue
        
        # Select best model based on BIC
        if comparison_results:
            best_model = min(comparison_results.keys(), 
                           key=lambda k: comparison_results[k]['bic'])
            
            comparison_results['best_model'] = best_model
            comparison_results['best_n_regimes'] = comparison_results[best_model]['n_regimes']
        
        return comparison_results
    
    def analyze_regime_transitions(self, results: RegimeDetectionResults) -> Dict[str, Any]:
        """Analyze regime transition characteristics"""
        
        if len(results.regime_sequence) == 0:
            return {}
        
        analysis = {}
        
        # Transition frequency
        transitions = []
        for i in range(1, len(results.regime_sequence)):
            if results.regime_sequence[i] != results.regime_sequence[i-1]:
                transitions.append((results.regime_sequence[i-1], results.regime_sequence[i]))
        
        analysis['total_transitions'] = len(transitions)
        analysis['transition_frequency'] = len(transitions) / len(results.regime_sequence)
        
        # Most common transitions
        if transitions:
            from collections import Counter
            transition_counts = Counter(transitions)
            analysis['most_common_transitions'] = transition_counts.most_common(5)
        
        # Regime persistence analysis
        if results.transition_matrix.size > 0:
            persistence = np.diag(results.transition_matrix)
            analysis['regime_persistence'] = {
                f'regime_{i}': persistence[i] for i in range(len(persistence))
            }
            analysis['average_persistence'] = persistence.mean()
        
        # Regime duration analysis
        regime_durations = defaultdict(list)
        current_regime = results.regime_sequence[0]
        duration = 1
        
        for regime in results.regime_sequence[1:]:
            if regime == current_regime:
                duration += 1
            else:
                regime_durations[current_regime].append(duration)
                current_regime = regime
                duration = 1
        
        # Add final duration
        regime_durations[current_regime].append(duration)
        
        analysis['regime_durations'] = {}
        for regime, durations in regime_durations.items():
            analysis['regime_durations'][f'regime_{regime}'] = {
                'mean_duration': np.mean(durations),
                'median_duration': np.median(durations),
                'max_duration': max(durations),
                'total_occurrences': len(durations)
            }
        
        return analysis
    
    def plot_regime_analysis(self, 
                           results: RegimeDetectionResults,
                           save_path: Optional[str] = None):
        """Create comprehensive regime analysis plots"""
        
        if results.returns is None or len(results.returns) == 0:
            print("No return data available for plotting")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Returns with regime coloring
        ax1 = axes[0, 0]
        
        # Create cumulative returns
        cumulative_returns = (1 + results.returns).cumprod()
        
        # Color by regime
        colors = ['red', 'blue', 'green', 'orange', 'purple'][:results.n_regimes]
        
        for regime in range(results.n_regimes):
            mask = np.array(results.regime_sequence) == regime
            if results.dates is not None:
                regime_dates = results.dates[mask]
                regime_values = cumulative_returns.iloc[mask]
                
                ax1.scatter(regime_dates, regime_values, 
                          c=colors[regime], alpha=0.7, s=10, 
                          label=f'Regime {regime}')
        
        ax1.set_title('Cumulative Returns by Regime')
        ax1.set_ylabel('Cumulative Return')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Regime probabilities over time
        ax2 = axes[0, 1]
        
        if results.smoothed_probabilities.size > 0 and results.dates is not None:
            for regime in range(results.n_regimes):
                probs = results.smoothed_probabilities[:, regime]
                ax2.plot(results.dates, probs, 
                        color=colors[regime], label=f'Regime {regime}')
            
            ax2.set_title('Regime Probabilities Over Time')
            ax2.set_ylabel('Probability')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Transition matrix heatmap
        ax3 = axes[1, 0]
        
        if results.transition_matrix.size > 0:
            im = ax3.imshow(results.transition_matrix, cmap='Blues', aspect='auto')
            ax3.set_title('Transition Matrix')
            ax3.set_xlabel('To Regime')
            ax3.set_ylabel('From Regime')
            
            # Add text annotations
            for i in range(results.n_regimes):
                for j in range(results.n_regimes):
                    text = ax3.text(j, i, f'{results.transition_matrix[i, j]:.2f}',
                                  ha="center", va="center", color="black")
            
            plt.colorbar(im, ax=ax3)
        
        # 4. Regime characteristics
        ax4 = axes[1, 1]
        
        if results.regime_states:
            regime_ids = [state.regime_id for state in results.regime_states]
            mean_returns = [state.mean_return * 252 for state in results.regime_states]  # Annualized
            volatilities = [state.volatility * np.sqrt(252) for state in results.regime_states]  # Annualized
            
            scatter = ax4.scatter(volatilities, mean_returns, 
                                c=regime_ids, cmap='viridis', s=100)
            
            for i, state in enumerate(results.regime_states):
                ax4.annotate(f'R{state.regime_id}', 
                           (volatilities[i], mean_returns[i]),
                           xytext=(5, 5), textcoords='offset points')
            
            ax4.set_title('Regime Risk-Return Profile')
            ax4.set_xlabel('Annualized Volatility')
            ax4.set_ylabel('Annualized Return')
            ax4.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax4)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Regime analysis plot saved to {save_path}")
        
        plt.show()
    
    def generate_regime_report(self, results: RegimeDetectionResults) -> str:
        """Generate comprehensive regime analysis report"""
        
        report = []
        report.append("="*70)
        report.append("HIDDEN MARKOV MODEL REGIME DETECTION REPORT")
        report.append("="*70)
        
        # Model information
        report.append(f"\nMODEL INFORMATION:")
        report.append(f"  Model Type: {results.model_type}")
        report.append(f"  Number of Regimes: {results.n_regimes}")
        report.append(f"  Log-Likelihood: {results.log_likelihood:.2f}")
        report.append(f"  AIC: {results.aic:.2f}")
        report.append(f"  BIC: {results.bic:.2f}")
        report.append(f"  Convergence: {'Yes' if results.convergence_achieved else 'No'}")
        report.append(f"  Iterations: {results.n_iterations}")
        
        # Data summary
        if results.returns is not None:
            report.append(f"\nDATA SUMMARY:")
            report.append(f"  Observations: {len(results.returns)}")
            report.append(f"  Period: {results.dates[0]} to {results.dates[-1]}")
            report.append(f"  Total Return: {(1 + results.returns).prod() - 1:.2%}")
            report.append(f"  Annualized Return: {results.returns.mean() * 252:.2%}")
            report.append(f"  Annualized Volatility: {results.returns.std() * np.sqrt(252):.2%}")
        
        # Regime characteristics
        if results.regime_states:
            report.append(f"\nREGIME CHARACTERISTICS:")
            
            for state in results.regime_states:
                report.append(f"\n  Regime {state.regime_id}:")
                report.append(f"    Mean Return (Ann.): {state.mean_return * 252:.2%}")
                report.append(f"    Volatility (Ann.): {state.volatility * np.sqrt(252):.2%}")
                report.append(f"    Persistence: {state.persistence:.3f}")
                report.append(f"    Expected Duration: {state.expected_duration:.1f} periods")
                report.append(f"    Volatility Regime: {state.volatility_regime}")
                report.append(f"    Market Stress: {state.market_stress:.2f}")
                report.append(f"    Trend Strength: {state.trend_strength:.2f}")
                
                if state.total_periods > 0:
                    report.append(f"    Total Periods: {state.total_periods}")
                    report.append(f"    Avg Period Length: {state.avg_period_length:.1f}")
                    report.append(f"    Longest Period: {state.longest_period}")
        
        # Transition matrix
        if results.transition_matrix.size > 0:
            report.append(f"\nTRANSITION MATRIX:")
            report.append("  " + "".join([f"{'To R' + str(j):>8}" for j in range(results.n_regimes)]))
            
            for i in range(results.n_regimes):
                row_str = f"From R{i}: "
                for j in range(results.n_regimes):
                    row_str += f"{results.transition_matrix[i, j]:>7.3f} "
                report.append("  " + row_str)
        
        # Steady state probabilities
        if results.steady_state_probs.size > 0:
            report.append(f"\nSTEADY STATE PROBABILITIES:")
            for i, prob in enumerate(results.steady_state_probs):
                report.append(f"  Regime {i}: {prob:.3f}")
        
        # Recent regime periods
        if results.regime_periods:
            report.append(f"\nRECENT REGIME PERIODS (Last 10):")
            report.append(f"  {'Start':<12} {'End':<12} {'Regime':<8} {'Duration':<10} {'Return':<10}")
            report.append("  " + "-"*60)
            
            for period in results.regime_periods[-10:]:
                start_str = period['start_date'].strftime('%Y-%m-%d')
                end_str = period['end_date'].strftime('%Y-%m-%d')
                regime_str = f"R{period['regime_id']}"
                duration_str = f"{period['duration']}d"
                return_str = f"{period['total_return']:.1%}"
                
                report.append(f"  {start_str:<12} {end_str:<12} {regime_str:<8} "
                             f"{duration_str:<10} {return_str:<10}")
        
        return "\n".join(report)
