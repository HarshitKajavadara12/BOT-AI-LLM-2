"""
Factor Combination Framework
Advanced factor combination and portfolio construction for quantitative research
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
import warnings
import math
from scipy import optimize, stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import concurrent.futures


class CombinationMethod(Enum):
    """Factor combination methods"""
    EQUAL_WEIGHT = "EQUAL_WEIGHT"           # Equal weighting
    IC_WEIGHT = "IC_WEIGHT"                 # IC-based weighting
    VOLATILITY_WEIGHT = "VOLATILITY_WEIGHT" # Inverse volatility weighting
    SHARPE_WEIGHT = "SHARPE_WEIGHT"         # Sharpe ratio weighting
    MEAN_VARIANCE = "MEAN_VARIANCE"         # Mean-variance optimization
    RISK_PARITY = "RISK_PARITY"             # Risk parity
    BLACK_LITTERMAN = "BLACK_LITTERMAN"     # Black-Litterman
    PCA_COMBINATION = "PCA_COMBINATION"     # Principal component analysis
    HIERARCHICAL = "HIERARCHICAL"           # Hierarchical risk parity
    ADAPTIVE_WEIGHT = "ADAPTIVE_WEIGHT"     # Adaptive weighting
    ML_ENSEMBLE = "ML_ENSEMBLE"             # Machine learning ensemble
    REGIME_DEPENDENT = "REGIME_DEPENDENT"   # Regime-dependent weighting


class RiskModel(Enum):
    """Risk models for combination"""
    SAMPLE_COVARIANCE = "SAMPLE_COVARIANCE"     # Sample covariance
    SHRINKAGE = "SHRINKAGE"                     # Shrinkage estimator
    EXPONENTIAL_WEIGHT = "EXPONENTIAL_WEIGHT"   # Exponentially weighted
    FACTOR_MODEL = "FACTOR_MODEL"               # Factor model
    ROBUST = "ROBUST"                           # Robust estimator


@dataclass
class CombinationConfig:
    """Factor combination configuration"""
    method: CombinationMethod
    risk_model: RiskModel = RiskModel.SAMPLE_COVARIANCE
    
    # Weighting parameters
    lookback_window: int = 252              # Lookback window for estimation
    min_weight: float = 0.0                 # Minimum factor weight
    max_weight: float = 1.0                 # Maximum factor weight
    weight_sum_constraint: float = 1.0      # Sum of weights constraint
    
    # Risk parameters
    risk_aversion: float = 1.0              # Risk aversion parameter
    target_volatility: Optional[float] = None  # Target volatility
    max_tracking_error: Optional[float] = None  # Maximum tracking error
    
    # Regularization
    l1_penalty: float = 0.0                 # L1 regularization
    l2_penalty: float = 0.0                 # L2 regularization
    turnover_penalty: float = 0.0           # Turnover penalty
    
    # Rebalancing
    rebalance_frequency: str = "monthly"    # Rebalancing frequency
    min_rebalance_threshold: float = 0.05   # Minimum rebalance threshold
    
    # Optimization
    max_iterations: int = 1000              # Maximum optimization iterations
    tolerance: float = 1e-6                 # Convergence tolerance
    
    # Other parameters
    allow_short: bool = True                # Allow short positions
    transaction_costs: float = 0.0          # Transaction costs (bps)


@dataclass
class CombinationResult:
    """Factor combination results"""
    weights: pd.Series                      # Factor weights
    combined_factor: pd.Series              # Combined factor values
    
    # Performance metrics
    sharpe_ratio: float = 0.0               # Sharpe ratio
    volatility: float = 0.0                 # Volatility
    max_drawdown: float = 0.0               # Maximum drawdown
    information_ratio: float = 0.0          # Information ratio
    
    # Risk metrics
    var_95: float = 0.0                     # 95% VaR
    expected_shortfall: float = 0.0         # Expected shortfall
    
    # Combination metrics
    weight_concentration: float = 0.0       # Weight concentration (HHI)
    effective_factors: int = 0              # Effective number of factors
    diversification_ratio: float = 0.0     # Diversification ratio
    
    # Metadata
    method: CombinationMethod = CombinationMethod.EQUAL_WEIGHT
    combination_date: datetime = field(default_factory=datetime.now)
    n_factors_combined: int = 0
    optimization_time: float = 0.0


class BaseFactorCombiner(ABC):
    """Base class for factor combination"""
    
    def __init__(self, config: CombinationConfig):
        self.config = config
        self.is_fitted = False
        self.factor_stats: Dict[str, Dict[str, float]] = {}
        
    @abstractmethod
    def fit(self, factors: pd.DataFrame, returns: Optional[pd.Series] = None) -> 'BaseFactorCombiner':
        """Fit the combiner"""
        pass
    
    @abstractmethod
    def get_weights(self) -> pd.Series:
        """Get factor weights"""
        pass
    
    def combine(self, factors: pd.DataFrame) -> pd.Series:
        """Combine factors using fitted weights"""
        
        if not self.is_fitted:
            raise ValueError("Combiner must be fitted before combining")
        
        weights = self.get_weights()
        
        # Align factors and weights
        common_factors = factors.columns.intersection(weights.index)
        
        if len(common_factors) == 0:
            raise ValueError("No common factors between data and weights")
        
        factor_subset = factors[common_factors]
        weight_subset = weights[common_factors]
        
        # Normalize weights
        weight_subset = weight_subset / weight_subset.sum()
        
        # Combine factors
        combined = factor_subset.dot(weight_subset)
        
        return combined
    
    def fit_combine(self, factors: pd.DataFrame, returns: Optional[pd.Series] = None) -> pd.Series:
        """Fit and combine factors"""
        self.fit(factors, returns)
        return self.combine(factors)
    
    def calculate_factor_stats(self, factors: pd.DataFrame, returns: Optional[pd.Series] = None):
        """Calculate factor statistics"""
        
        self.factor_stats = {}
        
        for factor_name in factors.columns:
            factor_values = factors[factor_name].dropna()
            
            if len(factor_values) < 30:  # Minimum periods
                continue
            
            stats_dict = {
                'mean': factor_values.mean(),
                'std': factor_values.std(),
                'skew': factor_values.skew(),
                'kurt': factor_values.kurtosis(),
                'sharpe': factor_values.mean() / factor_values.std() if factor_values.std() > 0 else 0,
                'autocorr': factor_values.autocorr(lag=1) if len(factor_values) > 1 else 0
            }
            
            # Add IC statistics if returns provided
            if returns is not None:
                aligned_returns = returns.reindex(factor_values.index).dropna()
                common_index = factor_values.index.intersection(aligned_returns.index)
                
                if len(common_index) > 30:
                    factor_aligned = factor_values[common_index]
                    returns_aligned = aligned_returns[common_index]
                    
                    ic, _ = stats.spearmanr(factor_aligned, returns_aligned)
                    stats_dict['ic'] = ic if not np.isnan(ic) else 0.0
            
            self.factor_stats[factor_name] = stats_dict


class EqualWeightCombiner(BaseFactorCombiner):
    """Equal weight factor combination"""
    
    def __init__(self, config: CombinationConfig):
        super().__init__(config)
        self.weights: Optional[pd.Series] = None
    
    def fit(self, factors: pd.DataFrame, returns: Optional[pd.Series] = None) -> 'EqualWeightCombiner':
        """Fit equal weight combiner"""
        
        n_factors = len(factors.columns)
        equal_weight = 1.0 / n_factors
        
        self.weights = pd.Series(equal_weight, index=factors.columns)
        self.calculate_factor_stats(factors, returns)
        self.is_fitted = True
        
        return self
    
    def get_weights(self) -> pd.Series:
        """Get equal weights"""
        if self.weights is None:
            raise ValueError("Combiner not fitted")
        return self.weights.copy()


class ICWeightCombiner(BaseFactorCombiner):
    """IC-based factor weighting"""
    
    def __init__(self, config: CombinationConfig):
        super().__init__(config)
        self.weights: Optional[pd.Series] = None
    
    def fit(self, factors: pd.DataFrame, returns: Optional[pd.Series] = None) -> 'ICWeightCombiner':
        """Fit IC-based combiner"""
        
        if returns is None:
            raise ValueError("Returns required for IC-based weighting")
        
        self.calculate_factor_stats(factors, returns)
        
        # Calculate IC-based weights
        ic_values = {}
        
        for factor_name, stats_dict in self.factor_stats.items():
            ic_values[factor_name] = abs(stats_dict.get('ic', 0.0))
        
        # Convert to weights
        ic_series = pd.Series(ic_values)
        
        if ic_series.sum() > 0:
            self.weights = ic_series / ic_series.sum()
        else:
            # Fallback to equal weights
            self.weights = pd.Series(1.0 / len(ic_series), index=ic_series.index)
        
        self.is_fitted = True
        return self
    
    def get_weights(self) -> pd.Series:
        """Get IC-based weights"""
        if self.weights is None:
            raise ValueError("Combiner not fitted")
        return self.weights.copy()


class VolatilityWeightCombiner(BaseFactorCombiner):
    """Inverse volatility weighting"""
    
    def __init__(self, config: CombinationConfig):
        super().__init__(config)
        self.weights: Optional[pd.Series] = None
    
    def fit(self, factors: pd.DataFrame, returns: Optional[pd.Series] = None) -> 'VolatilityWeightCombiner':
        """Fit volatility-based combiner"""
        
        self.calculate_factor_stats(factors, returns)
        
        # Calculate inverse volatility weights
        inv_vol_values = {}
        
        for factor_name, stats_dict in self.factor_stats.items():
            vol = stats_dict.get('std', 1.0)
            inv_vol_values[factor_name] = 1.0 / vol if vol > 0 else 0.0
        
        # Convert to weights
        inv_vol_series = pd.Series(inv_vol_values)
        
        if inv_vol_series.sum() > 0:
            self.weights = inv_vol_series / inv_vol_series.sum()
        else:
            # Fallback to equal weights
            self.weights = pd.Series(1.0 / len(inv_vol_series), index=inv_vol_series.index)
        
        self.is_fitted = True
        return self
    
    def get_weights(self) -> pd.Series:
        """Get volatility-based weights"""
        if self.weights is None:
            raise ValueError("Combiner not fitted")
        return self.weights.copy()


class MeanVarianceCombiner(BaseFactorCombiner):
    """Mean-variance optimization combiner"""
    
    def __init__(self, config: CombinationConfig):
        super().__init__(config)
        self.weights: Optional[pd.Series] = None
        self.expected_returns: Optional[pd.Series] = None
        self.covariance_matrix: Optional[pd.DataFrame] = None
    
    def fit(self, factors: pd.DataFrame, returns: Optional[pd.Series] = None) -> 'MeanVarianceCombiner':
        """Fit mean-variance combiner"""
        
        # Calculate expected returns (historical mean or IC-based)
        self.expected_returns = self._calculate_expected_returns(factors, returns)
        
        # Calculate covariance matrix
        self.covariance_matrix = self._calculate_covariance_matrix(factors)
        
        # Optimize weights
        self.weights = self._optimize_weights()
        
        self.is_fitted = True
        return self
    
    def _calculate_expected_returns(self, factors: pd.DataFrame, 
                                  returns: Optional[pd.Series] = None) -> pd.Series:
        """Calculate expected returns for factors"""
        
        if returns is not None:
            # Use IC-based expected returns
            expected_returns = {}
            
            for factor_name in factors.columns:
                factor_values = factors[factor_name].dropna()
                aligned_returns = returns.reindex(factor_values.index).dropna()
                common_index = factor_values.index.intersection(aligned_returns.index)
                
                if len(common_index) > 30:
                    factor_aligned = factor_values[common_index]
                    returns_aligned = aligned_returns[common_index]
                    
                    ic, _ = stats.spearmanr(factor_aligned, returns_aligned)
                    expected_returns[factor_name] = ic if not np.isnan(ic) else 0.0
                else:
                    expected_returns[factor_name] = 0.0
            
            return pd.Series(expected_returns)
        
        else:
            # Use historical means
            return factors.mean()
    
    def _calculate_covariance_matrix(self, factors: pd.DataFrame) -> pd.DataFrame:
        """Calculate factor covariance matrix"""
        
        # Handle missing values
        factors_clean = factors.fillna(method='ffill').dropna()
        
        if self.config.risk_model == RiskModel.SAMPLE_COVARIANCE:
            return factors_clean.cov()
        
        elif self.config.risk_model == RiskModel.SHRINKAGE:
            # Ledoit-Wolf shrinkage
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf()
            cov_shrunk = lw.fit(factors_clean).covariance_
            return pd.DataFrame(cov_shrunk, index=factors.columns, columns=factors.columns)
        
        elif self.config.risk_model == RiskModel.EXPONENTIAL_WEIGHT:
            # Exponentially weighted covariance
            return factors_clean.ewm(halflife=63).cov().iloc[-len(factors.columns):]
        
        else:
            return factors_clean.cov()
    
    def _optimize_weights(self) -> pd.Series:
        """Optimize factor weights using mean-variance"""
        
        n_factors = len(self.expected_returns)
        
        # Initial guess (equal weights)
        x0 = np.ones(n_factors) / n_factors
        
        # Constraints
        constraints = []
        
        # Weights sum to 1
        constraints.append({
            'type': 'eq',
            'fun': lambda w: np.sum(w) - self.config.weight_sum_constraint
        })
        
        # Bounds
        bounds = [(self.config.min_weight, self.config.max_weight) for _ in range(n_factors)]
        
        # Objective function
        def objective(weights):
            portfolio_return = np.dot(weights, self.expected_returns)
            portfolio_variance = np.dot(weights, np.dot(self.covariance_matrix, weights))
            
            # Mean-variance utility
            utility = portfolio_return - 0.5 * self.config.risk_aversion * portfolio_variance
            
            # Add regularization
            if self.config.l1_penalty > 0:
                utility -= self.config.l1_penalty * np.sum(np.abs(weights))
            
            if self.config.l2_penalty > 0:
                utility -= self.config.l2_penalty * np.sum(weights ** 2)
            
            return -utility  # Minimize negative utility
        
        # Optimize
        try:
            result = optimize.minimize(
                objective,
                x0=x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': self.config.max_iterations, 'ftol': self.config.tolerance}
            )
            
            if result.success:
                optimal_weights = result.x
            else:
                warnings.warn("Optimization failed, using equal weights")
                optimal_weights = x0
                
        except Exception as e:
            warnings.warn(f"Optimization error: {e}, using equal weights")
            optimal_weights = x0
        
        return pd.Series(optimal_weights, index=self.expected_returns.index)
    
    def get_weights(self) -> pd.Series:
        """Get optimized weights"""
        if self.weights is None:
            raise ValueError("Combiner not fitted")
        return self.weights.copy()


class RiskParityCombiner(BaseFactorCombiner):
    """Risk parity factor combination"""
    
    def __init__(self, config: CombinationConfig):
        super().__init__(config)
        self.weights: Optional[pd.Series] = None
        self.covariance_matrix: Optional[pd.DataFrame] = None
    
    def fit(self, factors: pd.DataFrame, returns: Optional[pd.Series] = None) -> 'RiskParityCombiner':
        """Fit risk parity combiner"""
        
        # Calculate covariance matrix
        self.covariance_matrix = self._calculate_covariance_matrix(factors)
        
        # Optimize for risk parity
        self.weights = self._optimize_risk_parity()
        
        self.is_fitted = True
        return self
    
    def _calculate_covariance_matrix(self, factors: pd.DataFrame) -> pd.DataFrame:
        """Calculate factor covariance matrix"""
        factors_clean = factors.fillna(method='ffill').dropna()
        return factors_clean.cov()
    
    def _optimize_risk_parity(self) -> pd.Series:
        """Optimize for risk parity weights"""
        
        n_factors = len(self.covariance_matrix)
        
        # Initial guess (inverse volatility)
        volatilities = np.sqrt(np.diag(self.covariance_matrix))
        x0 = (1 / volatilities) / np.sum(1 / volatilities)
        
        # Risk parity objective function
        def risk_parity_objective(weights):
            # Portfolio volatility
            portfolio_var = np.dot(weights, np.dot(self.covariance_matrix, weights))
            portfolio_vol = np.sqrt(portfolio_var)
            
            # Risk contributions
            marginal_contrib = np.dot(self.covariance_matrix, weights) / portfolio_vol
            risk_contrib = weights * marginal_contrib
            
            # Target equal risk contribution
            target_contrib = portfolio_vol / n_factors
            
            # Sum of squared deviations from equal risk contribution
            return np.sum((risk_contrib - target_contrib) ** 2)
        
        # Constraints
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        
        # Bounds
        bounds = [(0.01, 0.99) for _ in range(n_factors)]
        
        try:
            result = optimize.minimize(
                risk_parity_objective,
                x0=x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': self.config.max_iterations}
            )
            
            if result.success:
                optimal_weights = result.x
            else:
                warnings.warn("Risk parity optimization failed, using inverse volatility weights")
                optimal_weights = x0
                
        except Exception as e:
            warnings.warn(f"Risk parity optimization error: {e}")
            optimal_weights = x0
        
        return pd.Series(optimal_weights, index=self.covariance_matrix.index)
    
    def get_weights(self) -> pd.Series:
        """Get risk parity weights"""
        if self.weights is None:
            raise ValueError("Combiner not fitted")
        return self.weights.copy()


class PCACombiner(BaseFactorCombiner):
    """PCA-based factor combination"""
    
    def __init__(self, config: CombinationConfig, n_components: int = 3):
        super().__init__(config)
        self.n_components = n_components
        self.pca_model = None
        self.weights: Optional[pd.Series] = None
    
    def fit(self, factors: pd.DataFrame, returns: Optional[pd.Series] = None) -> 'PCACombiner':
        """Fit PCA combiner"""
        
        # Standardize factors
        factors_clean = factors.fillna(method='ffill').dropna()
        scaler = StandardScaler()
        factors_scaled = scaler.fit_transform(factors_clean)
        
        # Fit PCA
        self.pca_model = PCA(n_components=self.n_components)
        self.pca_model.fit(factors_scaled)
        
        # Use first principal component as weights
        first_pc = self.pca_model.components_[0]
        weights = np.abs(first_pc)  # Take absolute values
        weights = weights / np.sum(weights)  # Normalize
        
        self.weights = pd.Series(weights, index=factors.columns)
        
        self.is_fitted = True
        return self
    
    def get_weights(self) -> pd.Series:
        """Get PCA-based weights"""
        if self.weights is None:
            raise ValueError("Combiner not fitted")
        return self.weights.copy()


class AdaptiveWeightCombiner(BaseFactorCombiner):
    """Adaptive factor weighting based on recent performance"""
    
    def __init__(self, config: CombinationConfig, adaptation_window: int = 63):
        super().__init__(config)
        self.adaptation_window = adaptation_window
        self.weights_history: List[pd.Series] = []
        self.current_weights: Optional[pd.Series] = None
    
    def fit(self, factors: pd.DataFrame, returns: Optional[pd.Series] = None) -> 'AdaptiveWeightCombiner':
        """Fit adaptive combiner"""
        
        if returns is None:
            raise ValueError("Returns required for adaptive weighting")
        
        # Calculate time-varying weights
        self._calculate_adaptive_weights(factors, returns)
        
        self.is_fitted = True
        return self
    
    def _calculate_adaptive_weights(self, factors: pd.DataFrame, returns: pd.Series):
        """Calculate adaptive weights over time"""
        
        # Align data
        common_index = factors.index.intersection(returns.index)
        factors_aligned = factors.loc[common_index]
        returns_aligned = returns.loc[common_index]
        
        # Rolling window calculation
        self.weights_history = []
        
        for i in range(self.adaptation_window, len(common_index)):
            window_start = i - self.adaptation_window
            window_end = i
            
            # Window data
            factor_window = factors_aligned.iloc[window_start:window_end]
            return_window = returns_aligned.iloc[window_start:window_end]
            
            # Calculate IC-based weights for this window
            ic_weights = {}
            
            for factor_name in factor_window.columns:
                factor_values = factor_window[factor_name].dropna()
                return_values = return_window.reindex(factor_values.index).dropna()
                
                if len(factor_values) > 20:
                    ic, _ = stats.spearmanr(factor_values, return_values)
                    ic_weights[factor_name] = abs(ic) if not np.isnan(ic) else 0.0
                else:
                    ic_weights[factor_name] = 0.0
            
            # Normalize weights
            ic_series = pd.Series(ic_weights)
            if ic_series.sum() > 0:
                weights = ic_series / ic_series.sum()
            else:
                weights = pd.Series(1.0 / len(ic_series), index=ic_series.index)
            
            self.weights_history.append(weights)
        
        # Use most recent weights
        if self.weights_history:
            self.current_weights = self.weights_history[-1]
        else:
            # Fallback to equal weights
            self.current_weights = pd.Series(1.0 / len(factors.columns), index=factors.columns)
    
    def get_weights(self) -> pd.Series:
        """Get current adaptive weights"""
        if self.current_weights is None:
            raise ValueError("Combiner not fitted")
        return self.current_weights.copy()


class FactorCombiner:
    """
    Main factor combination engine
    """
    
    def __init__(self):
        self.combiners: Dict[CombinationMethod, BaseFactorCombiner] = {}
        self.combination_history: List[CombinationResult] = []
        
    def create_combiner(self, method: CombinationMethod, config: CombinationConfig) -> BaseFactorCombiner:
        """Create combiner instance"""
        
        combiner_map = {
            CombinationMethod.EQUAL_WEIGHT: EqualWeightCombiner,
            CombinationMethod.IC_WEIGHT: ICWeightCombiner,
            CombinationMethod.VOLATILITY_WEIGHT: VolatilityWeightCombiner,
            CombinationMethod.MEAN_VARIANCE: MeanVarianceCombiner,
            CombinationMethod.RISK_PARITY: RiskParityCombiner,
            CombinationMethod.PCA_COMBINATION: PCACombiner,
            CombinationMethod.ADAPTIVE_WEIGHT: AdaptiveWeightCombiner
        }
        
        if method not in combiner_map:
            raise ValueError(f"Unsupported combination method: {method}")
        
        return combiner_map[method](config)
    
    def combine_factors(self, factors: pd.DataFrame, returns: Optional[pd.Series] = None,
                       method: CombinationMethod = CombinationMethod.EQUAL_WEIGHT,
                       config: Optional[CombinationConfig] = None) -> CombinationResult:
        """Combine factors using specified method"""
        
        if config is None:
            config = CombinationConfig(method=method)
        
        start_time = time.time()
        
        # Create and fit combiner
        combiner = self.create_combiner(method, config)
        
        try:
            # Fit combiner and get weights
            combiner.fit(factors, returns)
            weights = combiner.get_weights()
            
            # Combine factors
            combined_factor = combiner.combine(factors)
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(combined_factor, returns)
            
            # Calculate combination metrics
            combination_metrics = self._calculate_combination_metrics(weights, factors)
            
            # Create result
            result = CombinationResult(
                weights=weights,
                combined_factor=combined_factor,
                method=method,
                n_factors_combined=len(weights),
                optimization_time=time.time() - start_time,
                **performance_metrics,
                **combination_metrics
            )
            
            # Store in history
            self.combination_history.append(result)
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Factor combination failed: {e}")
    
    def _calculate_performance_metrics(self, combined_factor: pd.Series, 
                                     returns: Optional[pd.Series] = None) -> Dict[str, float]:
        """Calculate performance metrics for combined factor"""
        
        factor_values = combined_factor.dropna()
        
        if len(factor_values) < 30:
            return {'sharpe_ratio': 0.0, 'volatility': 0.0, 'max_drawdown': 0.0, 
                   'information_ratio': 0.0, 'var_95': 0.0, 'expected_shortfall': 0.0}
        
        # Basic statistics
        sharpe_ratio = factor_values.mean() / factor_values.std() if factor_values.std() > 0 else 0.0
        volatility = factor_values.std()
        
        # Maximum drawdown
        cumulative = (1 + factor_values).cumprod()
        running_max = cumulative.expanding().max()
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = abs(drawdowns.min())
        
        # VaR and Expected Shortfall
        var_95 = abs(np.percentile(factor_values, 5))
        worse_returns = factor_values[factor_values <= -var_95]
        expected_shortfall = abs(worse_returns.mean()) if len(worse_returns) > 0 else var_95
        
        # Information ratio (if returns provided)
        information_ratio = 0.0
        if returns is not None:
            aligned_returns = returns.reindex(factor_values.index).dropna()
            common_index = factor_values.index.intersection(aligned_returns.index)
            
            if len(common_index) > 30:
                factor_aligned = factor_values[common_index]
                returns_aligned = aligned_returns[common_index]
                
                ic, _ = stats.spearmanr(factor_aligned, returns_aligned)
                if not np.isnan(ic):
                    # Rolling IC for IR calculation
                    window_size = min(63, len(common_index) // 4)
                    if window_size >= 20:
                        rolling_ics = []
                        for i in range(window_size, len(common_index)):
                            window_factor = factor_aligned.iloc[i-window_size:i]
                            window_returns = returns_aligned.iloc[i-window_size:i]
                            ic_window, _ = stats.spearmanr(window_factor, window_returns)
                            if not np.isnan(ic_window):
                                rolling_ics.append(ic_window)
                        
                        if rolling_ics:
                            ic_mean = np.mean(rolling_ics)
                            ic_std = np.std(rolling_ics)
                            information_ratio = ic_mean / ic_std if ic_std > 0 else 0.0
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'information_ratio': information_ratio,
            'var_95': var_95,
            'expected_shortfall': expected_shortfall
        }
    
    def _calculate_combination_metrics(self, weights: pd.Series, factors: pd.DataFrame) -> Dict[str, float]:
        """Calculate combination-specific metrics"""
        
        # Weight concentration (Herfindahl-Hirschman Index)
        weight_concentration = np.sum(weights ** 2)
        
        # Effective number of factors
        effective_factors = 1.0 / weight_concentration if weight_concentration > 0 else 0.0
        
        # Diversification ratio
        individual_vols = factors.std()
        weighted_vol = np.sum(weights * individual_vols)
        
        # Portfolio volatility (approximate)
        factor_corr = factors.corr()
        portfolio_var = np.dot(weights, np.dot(factor_corr * np.outer(individual_vols, individual_vols), weights))
        portfolio_vol = np.sqrt(portfolio_var) if portfolio_var > 0 else weighted_vol
        
        diversification_ratio = weighted_vol / portfolio_vol if portfolio_vol > 0 else 1.0
        
        return {
            'weight_concentration': weight_concentration,
            'effective_factors': effective_factors,
            'diversification_ratio': diversification_ratio
        }
    
    def compare_methods(self, factors: pd.DataFrame, returns: Optional[pd.Series] = None) -> pd.DataFrame:
        """Compare different combination methods"""
        
        methods = [
            CombinationMethod.EQUAL_WEIGHT,
            CombinationMethod.IC_WEIGHT,
            CombinationMethod.VOLATILITY_WEIGHT,
            CombinationMethod.MEAN_VARIANCE,
            CombinationMethod.RISK_PARITY
        ]
        
        comparison_data = []
        
        for method in methods:
            try:
                config = CombinationConfig(method=method)
                result = self.combine_factors(factors, returns, method, config)
                
                comparison_data.append({
                    'method': method.value,
                    'sharpe_ratio': result.sharpe_ratio,
                    'information_ratio': result.information_ratio,
                    'volatility': result.volatility,
                    'max_drawdown': result.max_drawdown,
                    'weight_concentration': result.weight_concentration,
                    'effective_factors': result.effective_factors,
                    'diversification_ratio': result.diversification_ratio,
                    'optimization_time': result.optimization_time
                })
                
            except Exception as e:
                warnings.warn(f"Method {method} failed: {e}")
                continue
        
        return pd.DataFrame(comparison_data)
    
    def get_combination_summary(self) -> pd.DataFrame:
        """Get summary of all combination results"""
        
        if not self.combination_history:
            return pd.DataFrame()
        
        summary_data = []
        
        for result in self.combination_history:
            summary_data.append({
                'method': result.method.value,
                'n_factors': result.n_factors_combined,
                'sharpe_ratio': result.sharpe_ratio,
                'information_ratio': result.information_ratio,
                'volatility': result.volatility,
                'max_drawdown': result.max_drawdown,
                'effective_factors': result.effective_factors,
                'combination_date': result.combination_date,
                'optimization_time': result.optimization_time
            })
        
        return pd.DataFrame(summary_data)
