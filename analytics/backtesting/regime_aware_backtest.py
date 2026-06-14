"""
Regime-Aware Backtesting Framework
Advanced backtesting system that adapts to different market regimes
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
from collections import deque, defaultdict
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats
import concurrent.futures


class MarketRegime(Enum):
    """Market regime types"""
    BULL_MARKET = "BULL_MARKET"              # Rising markets, low volatility
    BEAR_MARKET = "BEAR_MARKET"              # Falling markets, high volatility
    SIDEWAYS = "SIDEWAYS"                    # Range-bound markets
    HIGH_VOLATILITY = "HIGH_VOLATILITY"      # High volatility periods
    LOW_VOLATILITY = "LOW_VOLATILITY"        # Low volatility periods
    CRISIS = "CRISIS"                        # Crisis periods
    RECOVERY = "RECOVERY"                    # Post-crisis recovery
    UNKNOWN = "UNKNOWN"                      # Unclassified regime


@dataclass
class RegimeCharacteristics:
    """Characteristics of a market regime"""
    regime: MarketRegime
    mean_return: float = 0.0
    volatility: float = 0.0
    correlation: float = 0.0               # Average cross-asset correlation
    volume_ratio: float = 1.0              # Volume relative to normal
    spread_widening: float = 1.0           # Spread widening factor
    momentum_persistence: float = 0.5      # Momentum persistence factor
    mean_reversion_speed: float = 0.1      # Mean reversion speed
    
    # Trading environment characteristics
    liquidity_factor: float = 1.0          # Liquidity availability
    transaction_cost_multiplier: float = 1.0  # Cost multiplier
    strategy_performance_multiplier: Dict[str, float] = field(default_factory=dict)


@dataclass
class RegimeDetectionConfig:
    """Configuration for regime detection"""
    lookback_window: int = 252            # Days for regime detection
    update_frequency: int = 5             # Days between updates
    min_regime_duration: int = 20         # Minimum regime duration
    confidence_threshold: float = 0.7     # Confidence threshold for regime change
    
    # Feature settings
    use_returns: bool = True
    use_volatility: bool = True
    use_correlation: bool = True
    use_volume: bool = True
    use_spreads: bool = False
    
    # Model settings
    n_regimes: int = 4                    # Number of regimes to detect
    random_state: int = 42


class RegimeDetector(ABC):
    """Abstract base class for regime detection"""
    
    @abstractmethod
    def fit(self, data: pd.DataFrame) -> None:
        """Fit the regime detection model"""
        pass
    
    @abstractmethod
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Predict regimes for given data"""
        pass
    
    @abstractmethod
    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """Predict regime probabilities"""
        pass


class GMMRegimeDetector(RegimeDetector):
    """Gaussian Mixture Model based regime detector"""
    
    def __init__(self, config: RegimeDetectionConfig):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        
    def fit(self, data: pd.DataFrame) -> None:
        """Fit GMM regime detection model"""
        
        # Extract features
        features = self._extract_features(data)
        
        if features.empty:
            raise ValueError("No features could be extracted from data")
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features.fillna(0))
        
        # Fit GMM
        self.model = GaussianMixture(
            n_components=self.config.n_regimes,
            covariance_type='full',
            random_state=self.config.random_state,
            max_iter=200
        )
        
        self.model.fit(features_scaled)
        self.feature_names = features.columns.tolist()
        
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Predict regimes using GMM"""
        
        if self.model is None:
            raise RuntimeError("Model must be fitted before prediction")
        
        features = self._extract_features(data)
        
        if features.empty:
            return np.array([0] * len(data))
        
        features_scaled = self.scaler.transform(features.fillna(0))
        return self.model.predict(features_scaled)
    
    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """Predict regime probabilities using GMM"""
        
        if self.model is None:
            raise RuntimeError("Model must be fitted before prediction")
        
        features = self._extract_features(data)
        
        if features.empty:
            return np.ones((len(data), self.config.n_regimes)) / self.config.n_regimes
        
        features_scaled = self.scaler.transform(features.fillna(0))
        return self.model.predict_proba(features_scaled)
    
    def _extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract features for regime detection"""
        
        features = pd.DataFrame(index=data.index)
        
        # Assume data has columns: open, high, low, close, volume for each symbol
        # or is already returns data
        
        if 'close' in data.columns or any('_close' in col for col in data.columns):
            # Price data format
            close_cols = [col for col in data.columns if col.endswith('_close') or col == 'close']
            
            if not close_cols:
                return pd.DataFrame()
            
            # Calculate returns
            returns_data = pd.DataFrame()
            for col in close_cols:
                symbol = col.replace('_close', '') if '_close' in col else 'asset'
                returns_data[f'{symbol}_return'] = data[col].pct_change()
            
        else:
            # Assume already returns data
            returns_data = data.copy()
        
        if returns_data.empty:
            return pd.DataFrame()
        
        # Feature extraction
        window = min(self.config.lookback_window, len(returns_data))
        
        if self.config.use_returns:
            # Mean returns
            for col in returns_data.columns:
                if 'return' in col or returns_data[col].dtype in [np.float64, np.float32]:
                    features[f'{col}_mean'] = returns_data[col].rolling(window=window).mean()
        
        if self.config.use_volatility:
            # Volatility features
            for col in returns_data.columns:
                if 'return' in col or returns_data[col].dtype in [np.float64, np.float32]:
                    features[f'{col}_vol'] = returns_data[col].rolling(window=window).std()
                    
                    # Volatility of volatility
                    vol_series = returns_data[col].rolling(window=20).std()
                    features[f'{col}_vol_of_vol'] = vol_series.rolling(window=window//2).std()
        
        if self.config.use_correlation and len(returns_data.columns) > 1:
            # Cross-correlation features
            returns_matrix = returns_data.select_dtypes(include=[np.number])
            
            if returns_matrix.shape[1] > 1:
                # Rolling correlation
                def calc_mean_corr(x):
                    corr_matrix = x.corr()
                    if corr_matrix.empty:
                        return np.nan
                    
                    # Mean correlation (excluding diagonal)
                    corr_values = corr_matrix.values
                    mask = ~np.eye(corr_values.shape[0], dtype=bool)
                    return np.nanmean(corr_values[mask])
                
                features['mean_correlation'] = returns_matrix.rolling(window=window).apply(
                    calc_mean_corr, raw=False)
        
        if self.config.use_volume and any('volume' in col for col in data.columns):
            # Volume features
            volume_cols = [col for col in data.columns if 'volume' in col]
            
            for col in volume_cols:
                # Volume change
                features[f'{col}_change'] = data[col].pct_change()
                
                # Volume trend
                features[f'{col}_trend'] = data[col].rolling(window=window).apply(
                    lambda x: stats.linregress(range(len(x)), x)[0] if len(x) > 1 else 0,
                    raw=True
                )
        
        # Market stress indicators
        if len(returns_data.columns) > 0:
            # Skewness and kurtosis
            for col in returns_data.columns:
                if returns_data[col].dtype in [np.float64, np.float32]:
                    features[f'{col}_skew'] = returns_data[col].rolling(window=window).skew()
                    features[f'{col}_kurt'] = returns_data[col].rolling(window=window).kurt()
            
            # VIX-like measure (if we have enough assets)
            if len(returns_data.columns) >= 3:
                # Simple volatility index
                vol_data = returns_data.rolling(window=20).std()
                features['vix_proxy'] = vol_data.mean(axis=1)
        
        return features.dropna()


class HMMRegimeDetector(RegimeDetector):
    """Hidden Markov Model based regime detector (simplified version)"""
    
    def __init__(self, config: RegimeDetectionConfig):
        self.config = config
        self.transition_matrix = None
        self.emission_params = None
        self.initial_probs = None
        self.scaler = StandardScaler()
        
    def fit(self, data: pd.DataFrame) -> None:
        """Fit simplified HMM model"""
        
        # Extract features
        features = self._extract_features(data)
        
        if features.empty:
            raise ValueError("No features could be extracted")
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features.fillna(0))
        
        # Use K-means for initial state assignment
        kmeans = KMeans(n_clusters=self.config.n_regimes, 
                       random_state=self.config.random_state)
        states = kmeans.fit_predict(features_scaled)
        
        # Estimate transition matrix
        self._estimate_transition_matrix(states)
        
        # Estimate emission parameters
        self._estimate_emission_params(features_scaled, states)
        
        # Initial probabilities
        self.initial_probs = np.bincount(states[:self.config.n_regimes]) / len(states)
        if len(self.initial_probs) < self.config.n_regimes:
            # Pad with small probabilities
            self.initial_probs = np.pad(self.initial_probs, 
                                      (0, self.config.n_regimes - len(self.initial_probs)),
                                      constant_values=0.01)
        self.initial_probs /= self.initial_probs.sum()
        
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Predict regimes using simplified HMM"""
        
        probabilities = self.predict_proba(data)
        return np.argmax(probabilities, axis=1)
    
    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """Predict regime probabilities using simplified HMM"""
        
        if self.transition_matrix is None:
            raise RuntimeError("Model must be fitted before prediction")
        
        features = self._extract_features(data)
        
        if features.empty:
            return np.ones((len(data), self.config.n_regimes)) / self.config.n_regimes
        
        features_scaled = self.scaler.transform(features.fillna(0))
        
        # Simplified forward algorithm
        return self._forward_algorithm(features_scaled)
    
    def _extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract features for HMM - simpler feature set"""
        
        features = pd.DataFrame(index=data.index)
        
        # Focus on return and volatility features
        if 'close' in data.columns or any('_close' in col for col in data.columns):
            close_cols = [col for col in data.columns if col.endswith('_close') or col == 'close']
            
            returns_data = pd.DataFrame()
            for col in close_cols:
                symbol = col.replace('_close', '') if '_close' in col else 'asset'
                returns_data[f'{symbol}_return'] = data[col].pct_change()
        else:
            returns_data = data.copy()
        
        if returns_data.empty:
            return pd.DataFrame()
        
        # Simple features
        window = 20  # Shorter window for HMM
        
        for col in returns_data.columns:
            if returns_data[col].dtype in [np.float64, np.float32]:
                features[f'{col}_mean'] = returns_data[col].rolling(window=window).mean()
                features[f'{col}_std'] = returns_data[col].rolling(window=window).std()
        
        return features.dropna()
    
    def _estimate_transition_matrix(self, states: np.ndarray):
        """Estimate transition matrix from state sequence"""
        
        n_states = self.config.n_regimes
        self.transition_matrix = np.zeros((n_states, n_states))
        
        for i in range(len(states) - 1):
            current_state = min(states[i], n_states - 1)
            next_state = min(states[i + 1], n_states - 1)
            self.transition_matrix[current_state, next_state] += 1
        
        # Normalize rows
        for i in range(n_states):
            row_sum = self.transition_matrix[i].sum()
            if row_sum > 0:
                self.transition_matrix[i] /= row_sum
            else:
                # Uniform distribution if no transitions observed
                self.transition_matrix[i] = 1.0 / n_states
    
    def _estimate_emission_params(self, features: np.ndarray, states: np.ndarray):
        """Estimate emission parameters (Gaussian for each state)"""
        
        n_states = self.config.n_regimes
        n_features = features.shape[1]
        
        self.emission_params = {
            'means': np.zeros((n_states, n_features)),
            'covs': np.zeros((n_states, n_features, n_features))
        }
        
        for state in range(n_states):
            state_mask = states == state
            
            if np.sum(state_mask) > 0:
                state_features = features[state_mask]
                self.emission_params['means'][state] = np.mean(state_features, axis=0)
                
                if len(state_features) > 1:
                    self.emission_params['covs'][state] = np.cov(state_features, rowvar=False)
                else:
                    self.emission_params['covs'][state] = np.eye(n_features) * 0.01
            else:
                # Default parameters
                self.emission_params['means'][state] = np.zeros(n_features)
                self.emission_params['covs'][state] = np.eye(n_features)
    
    def _forward_algorithm(self, features: np.ndarray) -> np.ndarray:
        """Simplified forward algorithm for regime probabilities"""
        
        n_obs = features.shape[0]
        n_states = self.config.n_regimes
        
        # Forward probabilities
        alpha = np.zeros((n_obs, n_states))
        
        # Initialize
        for state in range(n_states):
            alpha[0, state] = (self.initial_probs[state] * 
                             self._emission_probability(features[0], state))
        
        # Forward pass
        for t in range(1, n_obs):
            for state in range(n_states):
                alpha[t, state] = (np.sum(alpha[t-1] * self.transition_matrix[:, state]) *
                                 self._emission_probability(features[t], state))
            
            # Normalize to prevent underflow
            alpha[t] /= np.sum(alpha[t])
        
        return alpha
    
    def _emission_probability(self, observation: np.ndarray, state: int) -> float:
        """Calculate emission probability for observation given state"""
        
        try:
            mean = self.emission_params['means'][state]
            cov = self.emission_params['covs'][state]
            
            # Multivariate normal probability
            diff = observation - mean
            
            # Add small regularization to diagonal
            cov_reg = cov + np.eye(cov.shape[0]) * 1e-6
            
            # Calculate probability (log space for numerical stability)
            log_prob = -0.5 * (np.dot(diff, np.linalg.solve(cov_reg, diff)) +
                              np.log(np.linalg.det(2 * np.pi * cov_reg)))
            
            return np.exp(log_prob)
        
        except (np.linalg.LinAlgError, RuntimeWarning):
            # Fallback to simple Gaussian
            return np.exp(-0.5 * np.sum(observation ** 2))


class RegimeAwareStrategy(ABC):
    """Abstract base class for regime-aware strategies"""
    
    @abstractmethod
    def calculate_signals(self, market_data: pd.DataFrame, 
                         current_regime: MarketRegime,
                         regime_probs: np.ndarray) -> Dict[str, float]:
        """Calculate trading signals based on market regime"""
        pass
    
    @abstractmethod
    def adjust_position_sizes(self, signals: Dict[str, float], 
                            current_regime: MarketRegime,
                            regime_probs: np.ndarray) -> Dict[str, float]:
        """Adjust position sizes based on regime"""
        pass


class AdaptiveMomentumStrategy(RegimeAwareStrategy):
    """Momentum strategy that adapts to market regimes"""
    
    def __init__(self, lookback_periods: Dict[MarketRegime, int] = None,
                 position_scaling: Dict[MarketRegime, float] = None):
        """
        Initialize adaptive momentum strategy
        
        Args:
            lookback_periods: Lookback periods for each regime
            position_scaling: Position scaling factors for each regime
        """
        
        self.lookback_periods = lookback_periods or {
            MarketRegime.BULL_MARKET: 60,
            MarketRegime.BEAR_MARKET: 20,
            MarketRegime.SIDEWAYS: 120,
            MarketRegime.HIGH_VOLATILITY: 10,
            MarketRegime.LOW_VOLATILITY: 120,
            MarketRegime.CRISIS: 5,
            MarketRegime.RECOVERY: 30,
            MarketRegime.UNKNOWN: 60
        }
        
        self.position_scaling = position_scaling or {
            MarketRegime.BULL_MARKET: 1.2,
            MarketRegime.BEAR_MARKET: 0.3,
            MarketRegime.SIDEWAYS: 0.8,
            MarketRegime.HIGH_VOLATILITY: 0.5,
            MarketRegime.LOW_VOLATILITY: 1.0,
            MarketRegime.CRISIS: 0.2,
            MarketRegime.RECOVERY: 1.1,
            MarketRegime.UNKNOWN: 0.6
        }
    
    def calculate_signals(self, market_data: pd.DataFrame, 
                         current_regime: MarketRegime,
                         regime_probs: np.ndarray) -> Dict[str, float]:
        """Calculate momentum signals adapted to regime"""
        
        signals = {}
        lookback = self.lookback_periods.get(current_regime, 60)
        
        # Get price columns
        price_cols = [col for col in market_data.columns 
                     if col.endswith('_close') or col == 'close']
        
        for col in price_cols:
            symbol = col.replace('_close', '') if '_close' in col else 'asset'
            
            if len(market_data) < lookback + 1:
                signals[symbol] = 0.0
                continue
            
            # Calculate returns over different horizons
            prices = market_data[col].dropna()
            
            if len(prices) < lookback + 1:
                signals[symbol] = 0.0
                continue
            
            # Regime-specific momentum calculation
            if current_regime == MarketRegime.BULL_MARKET:
                # Longer-term momentum in bull markets
                momentum = (prices.iloc[-1] / prices.iloc[-lookback] - 1)
                
            elif current_regime == MarketRegime.BEAR_MARKET:
                # Short-term reversal signals in bear markets
                short_momentum = (prices.iloc[-1] / prices.iloc[-lookback//3] - 1)
                momentum = -short_momentum  # Contrarian
                
            elif current_regime == MarketRegime.HIGH_VOLATILITY:
                # Very short-term momentum in volatile periods
                momentum = (prices.iloc[-1] / prices.iloc[-min(lookback//4, 5)] - 1)
                
            elif current_regime == MarketRegime.SIDEWAYS:
                # Mean reversion in sideways markets
                mean_price = prices.iloc[-lookback:].mean()
                momentum = -(prices.iloc[-1] / mean_price - 1)  # Mean reversion
                
            else:
                # Standard momentum
                momentum = (prices.iloc[-1] / prices.iloc[-lookback] - 1)
            
            # Scale by regime uncertainty
            regime_confidence = np.max(regime_probs) if len(regime_probs) > 0 else 0.5
            momentum *= regime_confidence
            
            signals[symbol] = np.clip(momentum, -1.0, 1.0)
        
        return signals
    
    def adjust_position_sizes(self, signals: Dict[str, float], 
                            current_regime: MarketRegime,
                            regime_probs: np.ndarray) -> Dict[str, float]:
        """Adjust position sizes based on regime"""
        
        scaling_factor = self.position_scaling.get(current_regime, 1.0)
        regime_confidence = np.max(regime_probs) if len(regime_probs) > 0 else 0.5
        
        # Reduce positions when regime is uncertain
        uncertainty_penalty = 0.5 + 0.5 * regime_confidence
        final_scaling = scaling_factor * uncertainty_penalty
        
        return {symbol: signal * final_scaling 
                for symbol, signal in signals.items()}


class RegimeAwareBacktester:
    """
    Backtesting framework that incorporates market regime detection
    """
    
    def __init__(self, 
                 regime_detector: RegimeDetector,
                 strategy: RegimeAwareStrategy,
                 regime_mapping: Dict[int, MarketRegime] = None,
                 rebalance_frequency: str = 'daily'):
        """
        Initialize regime-aware backtester
        
        Args:
            regime_detector: Regime detection model
            strategy: Regime-aware trading strategy
            regime_mapping: Mapping from detector output to regime types
            rebalance_frequency: How often to rebalance
        """
        
        self.regime_detector = regime_detector
        self.strategy = strategy
        self.rebalance_frequency = rebalance_frequency
        
        # Default regime mapping
        self.regime_mapping = regime_mapping or {
            0: MarketRegime.LOW_VOLATILITY,
            1: MarketRegime.BULL_MARKET,
            2: MarketRegime.HIGH_VOLATILITY,
            3: MarketRegime.BEAR_MARKET
        }
        
        # Tracking variables
        self.regime_history: List[Tuple[datetime, MarketRegime, np.ndarray]] = []
        self.position_history: List[Tuple[datetime, Dict[str, float]]] = []
        self.performance_history: List[Tuple[datetime, float]] = []
        
    def run_backtest(self, market_data: pd.DataFrame, 
                    initial_capital: float = 100000,
                    transaction_costs: float = 0.001) -> Dict[str, Any]:
        """Run regime-aware backtest"""
        
        if market_data.empty:
            raise ValueError("Market data cannot be empty")
        
        print(f"Running regime-aware backtest on {len(market_data)} data points...")
        
        # Fit regime detector on initial data
        train_window = min(252, len(market_data) // 2)  # Use first half or 1 year for training
        train_data = market_data.iloc[:train_window]
        
        try:
            self.regime_detector.fit(train_data)
            print(f"Regime detector fitted on {len(train_data)} training samples")
        except Exception as e:
            print(f"Warning: Regime detector fitting failed: {e}")
            # Use simple volatility-based regimes as fallback
            return self._run_simple_backtest(market_data, initial_capital, transaction_costs)
        
        # Initialize portfolio
        portfolio_value = initial_capital
        positions = {}
        cash = initial_capital
        
        # Get price columns
        price_cols = [col for col in market_data.columns 
                     if col.endswith('_close') or col == 'close']
        
        if not price_cols:
            raise ValueError("No price columns found in market data")
        
        symbols = [col.replace('_close', '') if '_close' in col else 'asset' 
                  for col in price_cols]
        
        # Initialize positions
        for symbol in symbols:
            positions[symbol] = 0.0
        
        # Rolling window for regime detection
        regime_window = 252  # 1 year
        
        print(f"Starting backtest loop for {len(symbols)} symbols...")
        
        try:
            for i, (date, row) in enumerate(market_data.iterrows()):
                
                if i % 50 == 0:
                    print(f"Processing day {i+1}/{len(market_data)}: {date}")
                
                # Get current prices
                current_prices = {}
                for j, symbol in enumerate(symbols):
                    price_col = price_cols[j]
                    current_prices[symbol] = row[price_col]
                
                # Skip if any prices are missing
                if any(pd.isna(price) or price <= 0 for price in current_prices.values()):
                    continue
                
                # Detect current regime
                start_idx = max(0, i - regime_window)
                regime_data = market_data.iloc[start_idx:i+1]
                
                if len(regime_data) < 20:  # Need minimum data
                    current_regime = MarketRegime.UNKNOWN
                    regime_probs = np.array([0.25, 0.25, 0.25, 0.25])
                else:
                    try:
                        regime_predictions = self.regime_detector.predict(regime_data)
                        regime_probabilities = self.regime_detector.predict_proba(regime_data)
                        
                        # Current regime
                        current_regime_id = regime_predictions[-1]
                        current_regime = self.regime_mapping.get(current_regime_id, MarketRegime.UNKNOWN)
                        regime_probs = regime_probabilities[-1] if len(regime_probabilities) > 0 else np.array([0.25, 0.25, 0.25, 0.25])
                        
                    except Exception as e:
                        print(f"Warning: Regime detection failed at {date}: {e}")
                        current_regime = MarketRegime.UNKNOWN
                        regime_probs = np.array([0.25, 0.25, 0.25, 0.25])
                
                # Store regime
                self.regime_history.append((date, current_regime, regime_probs.copy()))
                
                # Calculate signals
                try:
                    signals = self.strategy.calculate_signals(regime_data, current_regime, regime_probs)
                    target_positions = self.strategy.adjust_position_sizes(signals, current_regime, regime_probs)
                except Exception as e:
                    print(f"Warning: Strategy calculation failed at {date}: {e}")
                    target_positions = {symbol: 0.0 for symbol in symbols}
                
                # Execute trades
                for symbol in symbols:
                    current_pos = positions.get(symbol, 0.0)
                    target_pos = target_positions.get(symbol, 0.0)
                    trade_size = target_pos - current_pos
                    
                    if abs(trade_size) > 0.001:  # Minimum trade size
                        trade_value = trade_size * current_prices[symbol]
                        trade_cost = abs(trade_value) * transaction_costs
                        
                        # Check if we have enough cash
                        if trade_value + trade_cost <= cash:
                            positions[symbol] = target_pos
                            cash -= trade_value + trade_cost
                
                # Calculate portfolio value
                positions_value = sum(pos * current_prices[symbol] 
                                    for symbol, pos in positions.items())
                portfolio_value = cash + positions_value
                
                # Store results
                self.position_history.append((date, positions.copy()))
                self.performance_history.append((date, portfolio_value))
        
        except Exception as e:
            print(f"Error during backtest: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"Backtest completed. Processed {len(self.performance_history)} periods.")
        
        # Generate results
        return self._generate_results(initial_capital)
    
    def _run_simple_backtest(self, market_data: pd.DataFrame, 
                           initial_capital: float, transaction_costs: float) -> Dict[str, Any]:
        """Fallback simple backtest without regime detection"""
        
        print("Running simple backtest without regime detection...")
        
        # Simple volatility-based regime classification
        returns_data = market_data.pct_change().dropna()
        
        if returns_data.empty:
            return {'error': 'No valid return data'}
        
        # Calculate rolling volatility
        vol_window = 20
        if 'close' in returns_data.columns:
            volatility = returns_data['close'].rolling(vol_window).std()
        else:
            # Use first numeric column
            numeric_cols = returns_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                return {'error': 'No numeric data found'}
            volatility = returns_data[numeric_cols[0]].rolling(vol_window).std()
        
        # Simple regime assignment
        vol_threshold = volatility.quantile(0.7)
        
        portfolio_value = initial_capital
        performance_history = []
        
        for i, (date, row) in enumerate(market_data.iterrows()):
            if i < vol_window:
                continue
                
            current_vol = volatility.iloc[i]
            if current_vol > vol_threshold:
                current_regime = MarketRegime.HIGH_VOLATILITY
            else:
                current_regime = MarketRegime.LOW_VOLATILITY
            
            regime_probs = np.array([0.5, 0.5, 0.0, 0.0])
            
            self.regime_history.append((date, current_regime, regime_probs))
            performance_history.append((date, portfolio_value))
            
            # Simple buy-and-hold performance simulation
            if i == vol_window:
                base_value = portfolio_value
            else:
                # Simulate portfolio growth
                portfolio_value = base_value * (1 + returns_data.iloc[:i].mean(axis=1).sum())
        
        self.performance_history = performance_history
        return self._generate_results(initial_capital)
    
    def _generate_results(self, initial_capital: float) -> Dict[str, Any]:
        """Generate backtest results"""
        
        if not self.performance_history:
            return {'error': 'No performance history available'}
        
        # Extract performance data
        dates = [date for date, _ in self.performance_history]
        values = [value for _, value in self.performance_history]
        
        if len(values) < 2:
            return {'error': 'Insufficient performance data'}
        
        # Calculate returns
        returns = np.diff(values) / values[:-1]
        
        # Performance metrics
        total_return = (values[-1] - initial_capital) / initial_capital
        
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
            
            # Maximum drawdown
            peak = np.maximum.accumulate(values)
            drawdowns = (np.array(values) - peak) / peak
            max_drawdown = np.min(drawdowns)
            
            volatility = np.std(returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
            max_drawdown = 0.0
            volatility = 0.0
        
        # Regime analysis
        regime_counts = defaultdict(int)
        for _, regime, _ in self.regime_history:
            regime_counts[regime] += 1
        
        regime_distribution = {regime.value: count / len(self.regime_history) * 100 
                             for regime, count in regime_counts.items()}
        
        # Regime-specific performance
        regime_performance = self._analyze_regime_performance()
        
        results = {
            'initial_capital': initial_capital,
            'final_value': values[-1],
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'n_periods': len(values),
            'regime_distribution': regime_distribution,
            'regime_performance': regime_performance,
            'equity_curve': list(zip(dates, values))
        }
        
        return results
    
    def _analyze_regime_performance(self) -> Dict[str, Dict[str, float]]:
        """Analyze performance by regime"""
        
        if len(self.performance_history) != len(self.regime_history):
            return {}
        
        regime_returns = defaultdict(list)
        
        for i in range(1, len(self.performance_history)):
            _, prev_value = self.performance_history[i-1]
            _, curr_value = self.performance_history[i]
            _, regime, _ = self.regime_history[i]
            
            if prev_value > 0:
                period_return = (curr_value - prev_value) / prev_value
                regime_returns[regime].append(period_return)
        
        # Calculate statistics by regime
        regime_stats = {}
        
        for regime, returns in regime_returns.items():
            if len(returns) > 0:
                regime_stats[regime.value] = {
                    'mean_return': np.mean(returns),
                    'volatility': np.std(returns),
                    'sharpe_ratio': np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0,
                    'n_periods': len(returns),
                    'win_rate': np.sum(np.array(returns) > 0) / len(returns)
                }
        
        return regime_stats
