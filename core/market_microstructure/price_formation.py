"""
Price Formation Models for QUANTUM-FORGE
Implements advanced models for understanding price discovery and formation mechanisms.
"""

import numpy as np
import pandas as pd
from scipy import stats, optimize, signal
from scipy.integrate import odeint
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler, RobustScaler
from typing import Dict, List, Tuple, Optional, Union, Callable
import warnings
from numba import jit, prange
from dataclasses import dataclass
from enum import Enum
from collections import deque, defaultdict
import networkx as nx
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.statespace.kalman_filter import KalmanFilter
warnings.filterwarnings('ignore')

class PriceDiscoveryMechanism(Enum):
    """Price discovery mechanisms."""
    INFORMATION_DRIVEN = "information"
    LIQUIDITY_DRIVEN = "liquidity"
    NOISE_DRIVEN = "noise"
    MOMENTUM_DRIVEN = "momentum"
    MICROSTRUCTURE_DRIVEN = "microstructure"

@dataclass
class PriceComponent:
    """Components of price formation."""
    efficient_price: float
    microstructure_noise: float
    temporary_impact: float
    permanent_impact: float
    informed_component: float
    noise_component: float

@dataclass
class PriceDiscoveryMetrics:
    """Price discovery efficiency metrics."""
    information_share: float
    price_efficiency: float
    noise_to_signal_ratio: float
    discovery_speed: float
    variance_ratio: float
    autocorrelation_measure: float

class HasbrouckModel:
    """Hasbrouck (1991) information shares model for price discovery."""
    
    def __init__(self, max_lags: int = 5):
        """
        Initialize Hasbrouck model.
        
        Args:
            max_lags: Maximum lags for VAR model
        """
        self.max_lags = max_lags
        self.var_model = None
        self.information_shares = {}
        self.fitted = False
    
    def fit(self, price_data: Dict[str, np.ndarray]) -> bool:
        """
        Fit VAR model and calculate information shares.
        
        Args:
            price_data: Dictionary of venue -> price series
        
        Returns:
            True if fitting successful
        """
        if len(price_data) < 2:
            return False
        
        # Create DataFrame from price data
        df = pd.DataFrame(price_data)
        
        # Calculate returns
        returns_df = df.pct_change().dropna()
        
        if len(returns_df) < 50:
            return False
        
        try:
            # Fit VAR model
            self.var_model = VAR(returns_df)
            var_results = self.var_model.fit(maxlags=self.max_lags, ic='aic')
            
            # Get Cholesky decomposition of residual covariance matrix
            residual_cov = var_results.sigma_u
            chol_decomp = np.linalg.cholesky(residual_cov)
            
            # Calculate information shares
            venues = list(price_data.keys())
            n_venues = len(venues)
            
            # Information share is based on the permanent component variance
            total_variance = np.trace(residual_cov)
            
            for i, venue in enumerate(venues):
                # Information share = (cholesky factor)^2 / total variance
                venue_variance = chol_decomp[i, i]**2
                self.information_shares[venue] = venue_variance / total_variance
            
            self.fitted = True
            return True
            
        except Exception as e:
            return False
    
    def get_information_shares(self) -> Dict[str, float]:
        """Get calculated information shares."""
        return self.information_shares.copy() if self.fitted else {}
    
    def calculate_price_discovery_metrics(self) -> Dict[str, float]:
        """Calculate price discovery efficiency metrics."""
        if not self.fitted or not self.var_model:
            return {}
        
        # Get VAR results
        var_results = self.var_model.fit(maxlags=self.max_lags, ic='aic')
        
        # Calculate metrics
        metrics = {}
        
        # Information concentration (Herfindahl index of information shares)
        info_shares = list(self.information_shares.values())
        metrics['information_concentration'] = sum(share**2 for share in info_shares)
        
        # Model fit quality
        metrics['aic'] = var_results.aic
        metrics['bic'] = var_results.bic
        
        # Residual analysis
        residuals = var_results.resid
        metrics['residual_autocorr'] = np.mean([
            np.abs(np.corrcoef(col[:-1], col[1:])[0, 1]) 
            for col in residuals.T if len(col) > 1
        ])
        
        return metrics

class GloStenHarrisModel:
    """Glosten and Harris (1988) price decomposition model."""
    
    def __init__(self):
        """Initialize Glosten-Harris model parameters."""
        self.phi = 0.0          # Permanent price impact
        self.theta = 0.0        # Temporary price impact  
        self.sigma_u = 1.0      # Efficient price innovation variance
        self.sigma_s = 1.0      # Microstructure noise variance
        
        self.fitted = False
    
    def fit(self, prices: np.ndarray, trade_directions: np.ndarray) -> bool:
        """
        Fit Glosten-Harris model using MLE.
        
        Args:
            prices: Transaction prices
            trade_directions: Trade direction indicators (+1 buy, -1 sell)
        
        Returns:
            True if fitting successful
        """
        if len(prices) < 50 or len(trade_directions) != len(prices):
            return False
        
        # Calculate price changes
        price_changes = np.diff(prices)
        directions = trade_directions[1:]  # Align with price changes
        lagged_directions = trade_directions[:-1]
        
        if len(price_changes) < 20:
            return False
        
        try:
            # Set up regression: Δp_t = φ*q_t + θ*(q_t - q_{t-1}) + u_t
            # Where q_t is trade direction indicator
            
            X = np.column_stack([
                directions,                    # φ term
                directions - lagged_directions  # θ term
            ])
            
            y = price_changes
            
            # OLS estimation
            reg = LinearRegression().fit(X, y)
            self.phi = reg.coef_[0]    # Permanent impact
            self.theta = reg.coef_[1]  # Temporary impact
            
            # Estimate variances from residuals
            residuals = y - reg.predict(X)
            self.sigma_s = np.std(residuals)
            
            # Efficient price innovation variance (simplified)
            self.sigma_u = np.std(price_changes) * 0.7  # Rough approximation
            
            self.fitted = True
            return True
            
        except Exception as e:
            return False
    
    def decompose_price_change(self, price_change: float, 
                             trade_direction: float, 
                             prev_trade_direction: float) -> PriceComponent:
        """
        Decompose price change into components.
        
        Args:
            price_change: Observed price change
            trade_direction: Current trade direction
            prev_trade_direction: Previous trade direction
        
        Returns:
            PriceComponent with decomposition
        """
        if not self.fitted:
            return PriceComponent(0, 0, 0, 0, 0, 0)
        
        # Permanent impact
        permanent_impact = self.phi * trade_direction
        
        # Temporary impact 
        temporary_impact = self.theta * (trade_direction - prev_trade_direction)
        
        # Microstructure noise (residual)
        microstructure_noise = (price_change - permanent_impact - temporary_impact)
        
        # Efficient price change (permanent component)
        efficient_price = permanent_impact
        
        # Informed vs noise components (simplified allocation)
        informed_component = permanent_impact * 0.8  # Most permanent impact is informational
        noise_component = microstructure_noise + temporary_impact * 0.5
        
        return PriceComponent(
            efficient_price=efficient_price,
            microstructure_noise=microstructure_noise,
            temporary_impact=temporary_impact,
            permanent_impact=permanent_impact,
            informed_component=informed_component,
            noise_component=noise_component
        )

class StructuralVARModel:
    """Structural VAR model for price formation analysis."""
    
    def __init__(self, identification: str = 'cholesky'):
        """
        Initialize structural VAR model.
        
        Args:
            identification: Identification scheme ('cholesky', 'blanchard_quah')
        """
        self.identification = identification
        self.var_model = None
        self.structural_shocks = None
        self.impulse_responses = None
        self.fitted = False
    
    def fit(self, data: pd.DataFrame, max_lags: int = 5) -> bool:
        """
        Fit structural VAR model.
        
        Args:
            data: DataFrame with variables (returns, order flow, etc.)
            max_lags: Maximum lags for VAR
        
        Returns:
            True if fitting successful
        """
        if len(data) < 100 or data.shape[1] < 2:
            return False
        
        try:
            # Fit reduced form VAR
            self.var_model = VAR(data)
            var_results = self.var_model.fit(maxlags=max_lags, ic='aic')
            
            # Structural identification
            if self.identification == 'cholesky':
                # Cholesky identification
                sigma_u = var_results.sigma_u
                self.structural_matrix = np.linalg.cholesky(sigma_u)
            
            # Calculate impulse response functions
            self.impulse_responses = var_results.irf(periods=20)
            
            self.fitted = True
            return True
            
        except Exception as e:
            return False
    
    def get_variance_decomposition(self, periods: int = 10) -> Dict[str, pd.DataFrame]:
        """Get forecast error variance decomposition."""
        if not self.fitted:
            return {}
        
        var_results = self.var_model.fit()
        return var_results.fevd(periods=periods)
    
    def calculate_information_content(self, shock_var: str, response_var: str) -> float:
        """
        Calculate information content of one variable for another.
        
        Args:
            shock_var: Variable providing the shock
            response_var: Variable responding
        
        Returns:
            Information content measure
        """
        if not self.fitted or not self.impulse_responses:
            return 0.0
        
        try:
            # Get impulse response
            irf_data = self.impulse_responses.irfs
            
            # Find variable indices
            var_names = list(self.var_model.endog_names)
            shock_idx = var_names.index(shock_var)
            response_idx = var_names.index(response_var)
            
            # Calculate cumulative response
            cumulative_response = np.cumsum(irf_data[:, response_idx, shock_idx])
            
            # Information content as proportion of total variation explained
            total_variation = np.sum(irf_data[:, response_idx, :]**2)
            shock_variation = np.sum(irf_data[:, response_idx, shock_idx]**2)
            
            return shock_variation / total_variation if total_variation > 0 else 0.0
            
        except Exception as e:
            return 0.0

class NonlinearPriceModel:
    """Nonlinear price formation model with regime switching."""
    
    def __init__(self, n_regimes: int = 2):
        """
        Initialize nonlinear price model.
        
        Args:
            n_regimes: Number of market regimes
        """
        self.n_regimes = n_regimes
        self.regime_models = {}
        self.regime_probabilities = np.ones(n_regimes) / n_regimes
        self.transition_matrix = np.ones((n_regimes, n_regimes)) / n_regimes
        
        self.fitted = False
    
    def fit(self, prices: np.ndarray, features: np.ndarray) -> bool:
        """
        Fit regime-switching price model.
        
        Args:
            prices: Price series
            features: Feature matrix for price prediction
        
        Returns:
            True if fitting successful
        """
        if len(prices) < 100 or features.shape[0] != len(prices):
            return False
        
        try:
            # Calculate returns
            returns = np.diff(np.log(prices))
            features_aligned = features[1:]  # Align with returns
            
            # Simple regime classification based on volatility
            rolling_vol = pd.Series(returns).rolling(20).std().fillna(method='bfill')
            vol_threshold = np.percentile(rolling_vol, 70)
            
            regimes = (rolling_vol > vol_threshold).astype(int)
            
            # Fit separate models for each regime
            for regime_id in range(self.n_regimes):
                regime_mask = (regimes == regime_id)
                
                if np.sum(regime_mask) > 10:  # Minimum observations
                    regime_returns = returns[regime_mask]
                    regime_features = features_aligned[regime_mask]
                    
                    # Fit regime-specific model
                    model = GradientBoostingRegressor(
                        n_estimators=50,
                        max_depth=3,
                        random_state=42
                    )
                    
                    model.fit(regime_features, regime_returns)
                    self.regime_models[regime_id] = model
                    
                    # Update regime probability
                    self.regime_probabilities[regime_id] = np.mean(regime_mask)
            
            # Estimate transition probabilities (simplified)
            for i in range(self.n_regimes):
                for j in range(self.n_regimes):
                    # Count transitions from regime i to regime j
                    transitions = 0
                    total_i_states = 0
                    
                    for t in range(len(regimes) - 1):
                        if regimes[t] == i:
                            total_i_states += 1
                            if regimes[t+1] == j:
                                transitions += 1
                    
                    if total_i_states > 0:
                        self.transition_matrix[i, j] = transitions / total_i_states
            
            self.fitted = True
            return True
            
        except Exception as e:
            return False
    
    def predict_regime_probabilities(self, features: np.ndarray) -> np.ndarray:
        """
        Predict regime probabilities for new observations.
        
        Args:
            features: Feature matrix
        
        Returns:
            Regime probabilities for each observation
        """
        if not self.fitted:
            return np.tile(self.regime_probabilities, (len(features), 1))
        
        # Simple approach: use model confidence as regime indicator
        regime_probs = np.zeros((len(features), self.n_regimes))
        
        for regime_id, model in self.regime_models.items():
            try:
                predictions = model.predict(features)
                # Use prediction variance as confidence measure
                regime_probs[:, regime_id] = 1.0 / (1.0 + np.abs(predictions))
            except:
                regime_probs[:, regime_id] = self.regime_probabilities[regime_id]
        
        # Normalize probabilities
        row_sums = regime_probs.sum(axis=1, keepdims=True)
        regime_probs = regime_probs / np.maximum(row_sums, 1e-8)
        
        return regime_probs
    
    def predict_returns(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict returns with uncertainty.
        
        Args:
            features: Feature matrix
        
        Returns:
            Tuple of (predicted returns, prediction uncertainties)
        """
        if not self.fitted:
            return np.zeros(len(features)), np.ones(len(features))
        
        regime_probs = self.predict_regime_probabilities(features)
        predictions = np.zeros(len(features))
        uncertainties = np.zeros(len(features))
        
        for regime_id, model in self.regime_models.items():
            try:
                regime_preds = model.predict(features)
                predictions += regime_probs[:, regime_id] * regime_preds
                
                # Simple uncertainty estimate
                uncertainties += regime_probs[:, regime_id] * np.abs(regime_preds)
            except:
                continue
        
        return predictions, uncertainties

class PriceFormationAnalyzer:
    """Comprehensive price formation analysis framework."""
    
    def __init__(self):
        """Initialize price formation analyzer."""
        self.models = {
            'hasbrouck': HasbrouckModel(),
            'glosten_harris': GloStenHarrisModel(),
            'structural_var': StructuralVARModel(),
            'nonlinear': NonlinearPriceModel()
        }
        
        self.price_history = deque(maxlen=10000)
        self.volume_history = deque(maxlen=10000)
        self.trade_direction_history = deque(maxlen=10000)
        
    def add_market_data(self, price: float, volume: int, trade_direction: int):
        """Add market data observation."""
        self.price_history.append(price)
        self.volume_history.append(volume)
        self.trade_direction_history.append(trade_direction)
    
    def calculate_price_efficiency_metrics(self, prices: np.ndarray, 
                                         returns: np.ndarray = None) -> PriceDiscoveryMetrics:
        """
        Calculate comprehensive price efficiency metrics.
        
        Args:
            prices: Price series
            returns: Return series (calculated if not provided)
        
        Returns:
            PriceDiscoveryMetrics object
        """
        if returns is None:
            returns = np.diff(np.log(prices))
        
        if len(returns) < 50:
            return PriceDiscoveryMetrics(0, 0, 0, 0, 0, 0)
        
        # Variance ratio test for efficiency
        variance_ratios = []
        for k in [2, 4, 8, 16]:
            if len(returns) > k * 10:
                # k-period variance ratio
                k_period_var = np.var(returns.reshape(-1, k).sum(axis=1)) if len(returns) % k == 0 else np.var(returns[:-len(returns)%k].reshape(-1, k).sum(axis=1))
                single_period_var = np.var(returns) * k
                
                if single_period_var > 0:
                    vr = k_period_var / single_period_var
                    variance_ratios.append(abs(vr - 1.0))
        
        variance_ratio = np.mean(variance_ratios) if variance_ratios else 0.0
        
        # Autocorrelation measure
        autocorr_measure = 0.0
        for lag in range(1, min(11, len(returns)//4)):
            if len(returns) > lag:
                autocorr = np.corrcoef(returns[:-lag], returns[lag:])[0, 1]
                if not np.isnan(autocorr):
                    autocorr_measure += abs(autocorr) / lag
        
        # Noise-to-signal ratio (using microstructure noise proxy)
        price_changes = np.diff(prices)
        bid_ask_bounce = np.mean(np.abs(price_changes[1:] + price_changes[:-1])) if len(price_changes) > 1 else 0
        fundamental_volatility = np.std(returns)
        noise_to_signal = bid_ask_bounce / max(fundamental_volatility, 1e-8)
        
        # Price efficiency (inverse of autocorrelation and variance ratio deviations)
        efficiency_score = 1.0 / (1.0 + autocorr_measure + variance_ratio)
        
        # Discovery speed (how quickly prices incorporate information)
        # Approximated by the speed of mean reversion
        discovery_speed = 0.0
        if len(returns) > 20:
            # Half-life of mean reversion
            try:
                ar_coef = np.corrcoef(returns[:-1], returns[1:])[0, 1]
                if abs(ar_coef) < 1:
                    half_life = -np.log(2) / np.log(abs(ar_coef)) if ar_coef != 0 else np.inf
                    discovery_speed = 1.0 / max(half_life, 1.0) if half_life != np.inf else 1.0
            except:
                discovery_speed = 0.0
        
        # Information share (simplified - would need multiple venues)
        information_share = min(1.0, efficiency_score * 2)
        
        return PriceDiscoveryMetrics(
            information_share=information_share,
            price_efficiency=efficiency_score,
            noise_to_signal_ratio=noise_to_signal,
            discovery_speed=discovery_speed,
            variance_ratio=variance_ratio,
            autocorrelation_measure=autocorr_measure
        )
    
    def identify_price_formation_regime(self, recent_prices: np.ndarray, 
                                      recent_volumes: np.ndarray) -> PriceDiscoveryMechanism:
        """
        Identify dominant price formation mechanism.
        
        Args:
            recent_prices: Recent price observations
            recent_volumes: Recent volume observations
        
        Returns:
            Identified price discovery mechanism
        """
        if len(recent_prices) < 20:
            return PriceDiscoveryMechanism.NOISE_DRIVEN
        
        returns = np.diff(np.log(recent_prices))
        volume_changes = np.diff(recent_volumes) if len(recent_volumes) > 1 else np.array([0])
        
        # Price-volume correlation
        if len(returns) == len(volume_changes) and len(returns) > 5:
            try:
                pv_corr = np.corrcoef(np.abs(returns), volume_changes[:-1] if len(volume_changes) > len(returns) else volume_changes)[0, 1]
            except:
                pv_corr = 0
        else:
            pv_corr = 0
        
        # Volatility clustering
        volatility_clustering = 0
        if len(returns) > 10:
            vol_proxy = np.abs(returns)
            for lag in range(1, min(6, len(vol_proxy)//2)):
                autocorr = np.corrcoef(vol_proxy[:-lag], vol_proxy[lag:])[0, 1]
                if not np.isnan(autocorr):
                    volatility_clustering += autocorr
        
        # Momentum measure
        momentum_strength = 0
        if len(returns) > 5:
            for lag in range(1, min(6, len(returns)//2)):
                momentum = np.corrcoef(returns[:-lag], returns[lag:])[0, 1]
                if not np.isnan(momentum) and momentum > 0:
                    momentum_strength += momentum
        
        # Microstructure effects (bid-ask bounce)
        microstructure_effect = 0
        if len(returns) > 2:
            bounce_correlation = np.corrcoef(returns[:-1], returns[1:])[0, 1]
            if not np.isnan(bounce_correlation) and bounce_correlation < 0:
                microstructure_effect = abs(bounce_correlation)
        
        # Decision logic
        if abs(pv_corr) > 0.3 and not np.isnan(pv_corr):
            return PriceDiscoveryMechanism.INFORMATION_DRIVEN
        elif momentum_strength > 0.2:
            return PriceDiscoveryMechanism.MOMENTUM_DRIVEN
        elif microstructure_effect > 0.3:
            return PriceDiscoveryMechanism.MICROSTRUCTURE_DRIVEN
        elif volatility_clustering > 0.5:
            return PriceDiscoveryMechanism.LIQUIDITY_DRIVEN
        else:
            return PriceDiscoveryMechanism.NOISE_DRIVEN
    
    def estimate_information_arrival_rate(self, prices: np.ndarray, 
                                        volumes: np.ndarray, 
                                        time_intervals: np.ndarray) -> float:
        """
        Estimate the rate of information arrival to the market.
        
        Args:
            prices: Price series
            volumes: Volume series
            time_intervals: Time intervals between observations
        
        Returns:
            Estimated information arrival rate (events per unit time)
        """
        if len(prices) < 50:
            return 0.0
        
        returns = np.diff(np.log(prices))
        
        # Identify significant price moves (potential information events)
        return_threshold = np.std(returns) * 2  # 2 standard deviations
        significant_moves = np.abs(returns) > return_threshold
        
        # Count information events
        info_events = np.sum(significant_moves)
        total_time = np.sum(time_intervals) if len(time_intervals) > 0 else len(prices)
        
        # Information arrival rate
        arrival_rate = info_events / max(total_time, 1.0)
        
        # Adjust for volume (higher volume may indicate more information)
        if len(volumes) > 1:
            avg_volume = np.mean(volumes)
            volume_adjustment = np.mean(volumes[1:][significant_moves]) / max(avg_volume, 1.0)
            arrival_rate *= volume_adjustment
        
        return arrival_rate

