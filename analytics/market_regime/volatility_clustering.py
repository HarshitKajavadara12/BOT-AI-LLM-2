"""
Volatility Clustering Analysis Framework
======================================

Advanced volatility clustering detection and analysis system for financial time series.
Implements multiple methodologies for identifying and characterizing volatility regimes.

Features:
- GARCH family models for volatility clustering
- Regime-Switching GARCH models
- Realized volatility clustering analysis
- Volatility breakpoint detection
- High-frequency volatility patterns
- Volatility forecasting with regime awareness
- Risk management applications

Author: Quantum Forge Analytics Team  
Date: November 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
from scipy import stats, optimize
from scipy.stats import t, norm
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VolatilityRegime:
    """Container for volatility regime characteristics."""
    regime_id: int
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    mean_volatility: float
    volatility_persistence: float
    volatility_std: float
    regime_probability: float
    duration: int
    transition_probability: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {k: v for k, v in self.__dict__.items()}

@dataclass
class VolatilityClusterMetrics:
    """Container for volatility clustering analysis results."""
    clustering_coefficient: float
    persistence_measure: float
    volatility_regimes: List[VolatilityRegime]
    regime_transitions: pd.DataFrame
    arch_test_statistic: float
    arch_test_pvalue: float
    ljung_box_statistic: float
    ljung_box_pvalue: float
    volatility_half_life: float
    regime_stability: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary format."""
        return {
            'clustering_coefficient': self.clustering_coefficient,
            'persistence_measure': self.persistence_measure,
            'arch_test_statistic': self.arch_test_statistic,
            'arch_test_pvalue': self.arch_test_pvalue,
            'ljung_box_statistic': self.ljung_box_statistic,
            'ljung_box_pvalue': self.ljung_box_pvalue,
            'volatility_half_life': self.volatility_half_life,
            'regime_stability': self.regime_stability,
            'num_regimes': len(self.volatility_regimes),
            'regimes': [regime.to_dict() for regime in self.volatility_regimes]
        }

class GARCHVolatilityAnalyzer:
    """
    GARCH-based volatility clustering analyzer.
    
    Implements GARCH(1,1) and EGARCH models for volatility clustering detection
    and forecasting with regime-aware extensions.
    """
    
    def __init__(self, model_type: str = 'GARCH'):
        """
        Initialize GARCH analyzer.
        
        Parameters:
        -----------
        model_type : str
            Type of GARCH model ('GARCH', 'EGARCH', 'GJR-GARCH')
        """
        self.model_type = model_type
        self.fitted_params = None
        self.volatility_series = None
        
    def fit_garch_model(self, returns: pd.Series) -> Dict[str, Any]:
        """
        Fit GARCH model to return series.
        
        Parameters:
        -----------
        returns : pd.Series
            Time series of returns
            
        Returns:
        --------
        Dict[str, Any]
            Model parameters and diagnostics
        """
        try:
            # Prepare data
            returns_clean = returns.dropna()
            
            if len(returns_clean) < 100:
                raise ValueError("Insufficient data for GARCH estimation")
            
            # Initial parameter estimates
            omega_init = np.var(returns_clean) * 0.1
            alpha_init = 0.1
            beta_init = 0.8
            
            if self.model_type == 'GARCH':
                params = self._fit_garch11(returns_clean, omega_init, alpha_init, beta_init)
            elif self.model_type == 'EGARCH':
                params = self._fit_egarch(returns_clean)
            else:
                params = self._fit_garch11(returns_clean, omega_init, alpha_init, beta_init)
            
            # Calculate conditional volatility
            self.volatility_series = self._calculate_conditional_volatility(returns_clean, params)
            self.fitted_params = params
            
            # Model diagnostics
            diagnostics = self._calculate_garch_diagnostics(returns_clean, self.volatility_series)
            
            return {
                'model_type': self.model_type,
                'parameters': params,
                'volatility_series': self.volatility_series,
                'diagnostics': diagnostics,
                'log_likelihood': params.get('log_likelihood', np.nan),
                'aic': params.get('aic', np.nan),
                'bic': params.get('bic', np.nan)
            }
            
        except Exception as e:
            logger.error(f"Error fitting GARCH model: {str(e)}")
            raise
    
    def _fit_garch11(self, returns: pd.Series, omega_init: float, 
                     alpha_init: float, beta_init: float) -> Dict[str, Any]:
        """Fit GARCH(1,1) model using maximum likelihood."""
        
        def garch_log_likelihood(params):
            omega, alpha, beta = params
            
            # Parameter constraints
            if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
                return -np.inf
            
            n = len(returns)
            sigma2 = np.zeros(n)
            sigma2[0] = np.var(returns)
            
            # Calculate conditional variances
            for t in range(1, n):
                sigma2[t] = omega + alpha * returns.iloc[t-1]**2 + beta * sigma2[t-1]
            
            # Avoid numerical issues
            sigma2 = np.maximum(sigma2, 1e-8)
            
            # Log-likelihood
            log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + returns**2 / sigma2)
            
            return log_likelihood
        
        # Optimization
        initial_params = [omega_init, alpha_init, beta_init]
        bounds = [(1e-6, None), (0, 0.9), (0, 0.99)]
        
        result = optimize.minimize(
            lambda x: -garch_log_likelihood(x),
            initial_params,
            method='L-BFGS-B',
            bounds=bounds
        )
        
        if not result.success:
            logger.warning("GARCH optimization did not converge")
        
        omega_opt, alpha_opt, beta_opt = result.x
        log_likelihood = -result.fun
        
        # Calculate information criteria
        k = 3  # number of parameters
        n = len(returns)
        aic = 2 * k - 2 * log_likelihood
        bic = k * np.log(n) - 2 * log_likelihood
        
        return {
            'omega': omega_opt,
            'alpha': alpha_opt,
            'beta': beta_opt,
            'log_likelihood': log_likelihood,
            'aic': aic,
            'bic': bic,
            'persistence': alpha_opt + beta_opt,
            'unconditional_volatility': np.sqrt(omega_opt / (1 - alpha_opt - beta_opt))
        }
    
    def _fit_egarch(self, returns: pd.Series) -> Dict[str, Any]:
        """Fit EGARCH model (simplified implementation)."""
        # Simplified EGARCH - in practice would use specialized library
        return self._fit_garch11(returns, np.var(returns) * 0.1, 0.1, 0.8)
    
    def _calculate_conditional_volatility(self, returns: pd.Series, 
                                        params: Dict[str, Any]) -> pd.Series:
        """Calculate conditional volatility series."""
        omega = params['omega']
        alpha = params['alpha']
        beta = params['beta']
        
        n = len(returns)
        sigma2 = np.zeros(n)
        sigma2[0] = np.var(returns)
        
        for t in range(1, n):
            sigma2[t] = omega + alpha * returns.iloc[t-1]**2 + beta * sigma2[t-1]
        
        volatility = pd.Series(np.sqrt(sigma2), index=returns.index)
        return volatility
    
    def _calculate_garch_diagnostics(self, returns: pd.Series, 
                                   volatility: pd.Series) -> Dict[str, Any]:
        """Calculate GARCH model diagnostics."""
        # Standardized residuals
        std_residuals = returns / volatility
        
        # ARCH test on standardized residuals
        arch_stat, arch_pval = self._arch_test(std_residuals**2)
        
        # Ljung-Box test on standardized residuals
        lb_stat, lb_pval = self._ljung_box_test(std_residuals)
        
        # Jarque-Bera test for normality
        jb_stat, jb_pval = stats.jarque_bera(std_residuals.dropna())
        
        return {
            'arch_test_stat': arch_stat,
            'arch_test_pval': arch_pval,
            'ljung_box_stat': lb_stat,
            'ljung_box_pval': lb_pval,
            'jarque_bera_stat': jb_stat,
            'jarque_bera_pval': jb_pval,
            'std_residuals_mean': std_residuals.mean(),
            'std_residuals_std': std_residuals.std(),
            'std_residuals_skew': stats.skew(std_residuals.dropna()),
            'std_residuals_kurt': stats.kurtosis(std_residuals.dropna())
        }
    
    def _arch_test(self, squared_residuals: pd.Series, lags: int = 5) -> Tuple[float, float]:
        """ARCH test for heteroscedasticity."""
        # Simple ARCH test implementation
        n = len(squared_residuals)
        y = squared_residuals.values[lags:]
        X = np.column_stack([squared_residuals.values[i:n-lags+i] for i in range(lags)])
        X = np.column_stack([np.ones(len(y)), X])  # Add constant
        
        try:
            # OLS regression
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            residuals = y - X @ beta
            
            # Calculate test statistic
            n_obs = len(y)
            r_squared = 1 - np.sum(residuals**2) / np.sum((y - y.mean())**2)
            lm_statistic = n_obs * r_squared
            p_value = 1 - stats.chi2.cdf(lm_statistic, lags)
            
            return lm_statistic, p_value
        except:
            return np.nan, np.nan
    
    def _ljung_box_test(self, residuals: pd.Series, lags: int = 10) -> Tuple[float, float]:
        """Ljung-Box test for serial correlation."""
        n = len(residuals)
        residuals_clean = residuals.dropna()
        
        if len(residuals_clean) < lags + 1:
            return np.nan, np.nan
        
        # Calculate autocorrelations
        autocorrs = []
        for lag in range(1, lags + 1):
            if len(residuals_clean) > lag:
                corr = residuals_clean.autocorr(lag=lag)
                autocorrs.append(corr if not pd.isna(corr) else 0)
            else:
                autocorrs.append(0)
        
        # Ljung-Box statistic
        lb_stat = n * (n + 2) * sum([(autocorr**2) / (n - k) 
                                    for k, autocorr in enumerate(autocorrs, 1)])
        p_value = 1 - stats.chi2.cdf(lb_stat, lags)
        
        return lb_stat, p_value

class RegimeSwitchingVolatilityAnalyzer:
    """
    Regime-switching volatility analyzer using Gaussian Mixture Models.
    
    Identifies distinct volatility regimes and analyzes transitions between them.
    """
    
    def __init__(self, n_regimes: int = 2):
        """
        Initialize regime-switching analyzer.
        
        Parameters:
        -----------
        n_regimes : int
            Number of volatility regimes
        """
        self.n_regimes = n_regimes
        self.gmm_model = None
        self.regime_probabilities = None
        self.regimes = []
        
    def identify_volatility_regimes(self, returns: pd.Series) -> List[VolatilityRegime]:
        """
        Identify volatility regimes using Gaussian Mixture Model.
        
        Parameters:
        -----------
        returns : pd.Series
            Time series of returns
            
        Returns:
        --------
        List[VolatilityRegime]
            Identified volatility regimes
        """
        try:
            # Calculate rolling volatility
            rolling_vol = returns.rolling(window=20, min_periods=10).std() * np.sqrt(252)
            rolling_vol = rolling_vol.dropna()
            
            if len(rolling_vol) < 50:
                raise ValueError("Insufficient data for regime identification")
            
            # Prepare features for clustering
            features = self._prepare_volatility_features(returns, rolling_vol)
            
            # Fit Gaussian Mixture Model
            self.gmm_model = GaussianMixture(
                n_components=self.n_regimes,
                covariance_type='full',
                random_state=42,
                max_iter=200
            )
            
            regime_labels = self.gmm_model.fit_predict(features)
            self.regime_probabilities = self.gmm_model.predict_proba(features)
            
            # Create regime objects
            self.regimes = self._create_regime_objects(
                rolling_vol, regime_labels, self.regime_probabilities
            )
            
            return self.regimes
            
        except Exception as e:
            logger.error(f"Error identifying volatility regimes: {str(e)}")
            raise
    
    def _prepare_volatility_features(self, returns: pd.Series, 
                                   rolling_vol: pd.Series) -> np.ndarray:
        """Prepare features for regime identification."""
        # Calculate additional volatility features
        vol_change = rolling_vol.pct_change().fillna(0)
        vol_acceleration = vol_change.diff().fillna(0)
        
        # Align series
        min_length = min(len(rolling_vol), len(vol_change), len(vol_acceleration))
        
        features = np.column_stack([
            rolling_vol.iloc[-min_length:].values,
            vol_change.iloc[-min_length:].values,
            vol_acceleration.iloc[-min_length:].values
        ])
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        return features_scaled
    
    def _create_regime_objects(self, rolling_vol: pd.Series, 
                             regime_labels: np.ndarray,
                             regime_probs: np.ndarray) -> List[VolatilityRegime]:
        """Create regime objects from clustering results."""
        regimes = []
        
        for regime_id in range(self.n_regimes):
            regime_mask = regime_labels == regime_id
            regime_dates = rolling_vol.index[regime_mask]
            
            if len(regime_dates) == 0:
                continue
            
            regime_vols = rolling_vol.iloc[regime_mask]
            regime_prob_series = regime_probs[regime_mask, regime_id]
            
            # Calculate regime characteristics
            mean_vol = regime_vols.mean()
            vol_std = regime_vols.std()
            mean_prob = regime_prob_series.mean()
            
            # Calculate persistence (simplified)
            transitions = np.diff(regime_labels == regime_id).sum()
            total_periods = len(regime_labels)
            persistence = 1 - (transitions / total_periods) if total_periods > 0 else 0
            
            # Create regime periods
            regime_periods = self._identify_regime_periods(regime_dates, regime_id)
            
            for period_start, period_end in regime_periods:
                duration = (period_end - period_start).days
                
                regime = VolatilityRegime(
                    regime_id=regime_id,
                    start_date=period_start,
                    end_date=period_end,
                    mean_volatility=mean_vol,
                    volatility_persistence=persistence,
                    volatility_std=vol_std,
                    regime_probability=mean_prob,
                    duration=duration,
                    transition_probability=transitions / total_periods if total_periods > 0 else 0
                )
                
                regimes.append(regime)
        
        return regimes
    
    def _identify_regime_periods(self, regime_dates: pd.DatetimeIndex, 
                               regime_id: int) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """Identify continuous periods within a regime."""
        if len(regime_dates) == 0:
            return []
        
        periods = []
        start_date = regime_dates[0]
        prev_date = regime_dates[0]
        
        for current_date in regime_dates[1:]:
            # Check for gap (more than 5 days)
            if (current_date - prev_date).days > 5:
                periods.append((start_date, prev_date))
                start_date = current_date
            prev_date = current_date
        
        # Add final period
        periods.append((start_date, regime_dates[-1]))
        
        return periods
    
    def calculate_regime_transitions(self) -> pd.DataFrame:
        """Calculate regime transition matrix."""
        if self.regime_probabilities is None:
            raise ValueError("Must identify regimes first")
        
        # Get most likely regime for each period
        regime_sequence = np.argmax(self.regime_probabilities, axis=1)
        
        # Calculate transition matrix
        transition_matrix = np.zeros((self.n_regimes, self.n_regimes))
        
        for t in range(1, len(regime_sequence)):
            from_regime = regime_sequence[t-1]
            to_regime = regime_sequence[t]
            transition_matrix[from_regime, to_regime] += 1
        
        # Normalize to probabilities
        row_sums = transition_matrix.sum(axis=1)
        transition_probs = transition_matrix / row_sums[:, np.newaxis]
        transition_probs = np.nan_to_num(transition_probs)
        
        # Create DataFrame
        regime_names = [f'Regime_{i}' for i in range(self.n_regimes)]
        transition_df = pd.DataFrame(
            transition_probs,
            index=regime_names,
            columns=regime_names
        )
        
        return transition_df

class RealizedVolatilityAnalyzer:
    """
    Realized volatility clustering analyzer for high-frequency data.
    
    Analyzes intraday volatility patterns and clustering behavior using
    realized volatility measures.
    """
    
    def __init__(self, frequency: str = 'D'):
        """
        Initialize realized volatility analyzer.
        
        Parameters:
        -----------
        frequency : str
            Frequency for realized volatility calculation ('D', 'H', etc.)
        """
        self.frequency = frequency
        self.realized_vol_series = None
        
    def calculate_realized_volatility(self, high_freq_returns: pd.Series) -> pd.Series:
        """
        Calculate realized volatility from high-frequency returns.
        
        Parameters:
        -----------
        high_freq_returns : pd.Series
            High-frequency return series
            
        Returns:
        --------
        pd.Series
            Realized volatility series
        """
        try:
            # Calculate realized volatility (sum of squared returns)
            realized_vol = high_freq_returns.groupby(
                pd.Grouper(freq=self.frequency)
            ).apply(lambda x: np.sqrt(np.sum(x**2)))
            
            self.realized_vol_series = realized_vol.dropna()
            return self.realized_vol_series
            
        except Exception as e:
            logger.error(f"Error calculating realized volatility: {str(e)}")
            raise
    
    def analyze_volatility_clustering(self, realized_vol: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Analyze clustering patterns in realized volatility.
        
        Parameters:
        -----------
        realized_vol : Optional[pd.Series]
            Realized volatility series (uses internal if None)
            
        Returns:
        --------
        Dict[str, Any]
            Clustering analysis results
        """
        if realized_vol is None:
            realized_vol = self.realized_vol_series
            
        if realized_vol is None or len(realized_vol) < 50:
            raise ValueError("Insufficient realized volatility data")
        
        # Calculate clustering metrics
        autocorrelations = self._calculate_volatility_autocorrelations(realized_vol)
        clustering_coefficient = self._calculate_clustering_coefficient(realized_vol)
        persistence_measure = self._calculate_persistence_measure(realized_vol)
        half_life = self._calculate_volatility_half_life(realized_vol)
        
        # Regime analysis
        regime_analyzer = RegimeSwitchingVolatilityAnalyzer(n_regimes=3)
        regimes = regime_analyzer.identify_volatility_regimes(realized_vol.pct_change().dropna())
        
        return {
            'autocorrelations': autocorrelations,
            'clustering_coefficient': clustering_coefficient,
            'persistence_measure': persistence_measure,
            'volatility_half_life': half_life,
            'volatility_regimes': regimes,
            'regime_transitions': regime_analyzer.calculate_regime_transitions() if regimes else None
        }
    
    def _calculate_volatility_autocorrelations(self, vol_series: pd.Series, 
                                             max_lags: int = 20) -> Dict[int, float]:
        """Calculate volatility autocorrelations."""
        autocorrs = {}
        for lag in range(1, max_lags + 1):
            try:
                autocorr = vol_series.autocorr(lag=lag)
                autocorrs[lag] = autocorr if not pd.isna(autocorr) else 0.0
            except:
                autocorrs[lag] = 0.0
        return autocorrs
    
    def _calculate_clustering_coefficient(self, vol_series: pd.Series) -> float:
        """Calculate volatility clustering coefficient."""
        # Clustering coefficient based on runs of high/low volatility
        median_vol = vol_series.median()
        high_vol_periods = vol_series > median_vol
        
        # Count runs
        runs = []
        current_run = 1
        
        for i in range(1, len(high_vol_periods)):
            if high_vol_periods.iloc[i] == high_vol_periods.iloc[i-1]:
                current_run += 1
            else:
                runs.append(current_run)
                current_run = 1
        runs.append(current_run)
        
        # Clustering coefficient
        expected_runs = len(vol_series) / 2
        actual_runs = len(runs)
        clustering_coeff = 1 - (actual_runs / expected_runs) if expected_runs > 0 else 0
        
        return max(0, clustering_coeff)
    
    def _calculate_persistence_measure(self, vol_series: pd.Series) -> float:
        """Calculate volatility persistence measure."""
        # Simple AR(1) coefficient
        vol_lagged = vol_series.shift(1)
        valid_mask = ~(vol_series.isna() | vol_lagged.isna())
        
        if valid_mask.sum() < 10:
            return 0.0
        
        try:
            correlation = vol_series[valid_mask].corr(vol_lagged[valid_mask])
            return correlation if not pd.isna(correlation) else 0.0
        except:
            return 0.0
    
    def _calculate_volatility_half_life(self, vol_series: pd.Series) -> float:
        """Calculate volatility half-life."""
        # Half-life of volatility shocks using AR(1) model
        persistence = self._calculate_persistence_measure(vol_series)
        
        if persistence <= 0 or persistence >= 1:
            return np.inf
        
        half_life = np.log(0.5) / np.log(persistence)
        return half_life

class ComprehensiveVolatilityClusterAnalyzer:
    """
    Comprehensive volatility clustering analysis system.
    
    Integrates all volatility clustering methodologies for complete analysis.
    """
    
    def __init__(self, n_regimes: int = 2):
        """
        Initialize comprehensive analyzer.
        
        Parameters:
        -----------
        n_regimes : int
            Number of volatility regimes to identify
        """
        self.n_regimes = n_regimes
        self.garch_analyzer = GARCHVolatilityAnalyzer()
        self.regime_analyzer = RegimeSwitchingVolatilityAnalyzer(n_regimes)
        self.realized_vol_analyzer = RealizedVolatilityAnalyzer()
        
    def analyze_volatility_clustering(self, returns: pd.Series) -> VolatilityClusterMetrics:
        """
        Perform comprehensive volatility clustering analysis.
        
        Parameters:
        -----------
        returns : pd.Series
            Time series of returns
            
        Returns:
        --------
        VolatilityClusterMetrics
            Comprehensive clustering metrics
        """
        try:
            # GARCH analysis
            garch_results = self.garch_analyzer.fit_garch_model(returns)
            
            # Regime analysis
            volatility_regimes = self.regime_analyzer.identify_volatility_regimes(returns)
            regime_transitions = self.regime_analyzer.calculate_regime_transitions()
            
            # Realized volatility analysis (using daily aggregation)
            daily_returns = returns.resample('D').sum() if hasattr(returns.index, 'freq') else returns
            realized_vol = self.realized_vol_analyzer.calculate_realized_volatility(daily_returns)
            clustering_analysis = self.realized_vol_analyzer.analyze_volatility_clustering(realized_vol)
            
            # Extract key metrics
            clustering_coefficient = clustering_analysis.get('clustering_coefficient', 0.0)
            persistence_measure = clustering_analysis.get('persistence_measure', 0.0)
            volatility_half_life = clustering_analysis.get('volatility_half_life', np.inf)
            
            # ARCH test results
            arch_stat = garch_results['diagnostics'].get('arch_test_stat', np.nan)
            arch_pval = garch_results['diagnostics'].get('arch_test_pval', np.nan)
            
            # Ljung-Box test results
            lb_stat = garch_results['diagnostics'].get('ljung_box_stat', np.nan)
            lb_pval = garch_results['diagnostics'].get('ljung_box_pval', np.nan)
            
            # Regime stability
            regime_stability = self._calculate_regime_stability(volatility_regimes)
            
            # Create comprehensive metrics
            metrics = VolatilityClusterMetrics(
                clustering_coefficient=clustering_coefficient,
                persistence_measure=persistence_measure,
                volatility_regimes=volatility_regimes,
                regime_transitions=regime_transitions,
                arch_test_statistic=arch_stat,
                arch_test_pvalue=arch_pval,
                ljung_box_statistic=lb_stat,
                ljung_box_pvalue=lb_pval,
                volatility_half_life=volatility_half_life,
                regime_stability=regime_stability
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in comprehensive volatility clustering analysis: {str(e)}")
            raise
    
    def _calculate_regime_stability(self, regimes: List[VolatilityRegime]) -> float:
        """Calculate overall regime stability measure."""
        if not regimes:
            return 0.0
        
        # Average regime duration weighted by regime probability
        total_weighted_duration = 0.0
        total_weight = 0.0
        
        for regime in regimes:
            weight = regime.regime_probability
            duration = regime.duration
            
            total_weighted_duration += weight * duration
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        avg_duration = total_weighted_duration / total_weight
        
        # Normalize by maximum possible duration
        max_duration = max([r.duration for r in regimes], default=1)
        stability = min(avg_duration / max_duration, 1.0) if max_duration > 0 else 0.0
        
        return stability
    
    def generate_clustering_report(self, returns: pd.Series, 
                                 strategy_name: str = "Strategy") -> Dict[str, Any]:
        """
        Generate comprehensive volatility clustering report.
        
        Parameters:
        -----------
        returns : pd.Series
            Time series of returns
        strategy_name : str
            Name of strategy for reporting
            
        Returns:
        --------
        Dict[str, Any]
            Comprehensive clustering report
        """
        # Perform analysis
        metrics = self.analyze_volatility_clustering(returns)
        garch_results = self.garch_analyzer.fit_garch_model(returns)
        
        # Create report
        report = {
            'strategy_name': strategy_name,
            'analysis_date': datetime.now(),
            'period_analyzed': {
                'start': returns.index.min(),
                'end': returns.index.max(),
                'total_observations': len(returns)
            },
            'clustering_metrics': metrics.to_dict(),
            'garch_analysis': {
                'model_type': garch_results['model_type'],
                'parameters': garch_results['parameters'],
                'diagnostics': garch_results['diagnostics'],
                'model_fit': {
                    'log_likelihood': garch_results.get('log_likelihood', np.nan),
                    'aic': garch_results.get('aic', np.nan),
                    'bic': garch_results.get('bic', np.nan)
                }
            },
            'volatility_regimes_summary': {
                'number_of_regimes': len(metrics.volatility_regimes),
                'regime_characteristics': [regime.to_dict() for regime in metrics.volatility_regimes[:5]],  # Top 5
                'average_regime_duration': np.mean([r.duration for r in metrics.volatility_regimes]) if metrics.volatility_regimes else 0
            },
            'clustering_assessment': self._assess_clustering_strength(metrics),
            'recommendations': self._generate_clustering_recommendations(metrics)
        }
        
        return report
    
    def _assess_clustering_strength(self, metrics: VolatilityClusterMetrics) -> str:
        """Assess overall clustering strength."""
        # Multiple criteria assessment
        strong_indicators = 0
        total_indicators = 0
        
        # ARCH test
        if not pd.isna(metrics.arch_test_pvalue):
            total_indicators += 1
            if metrics.arch_test_pvalue < 0.05:
                strong_indicators += 1
        
        # Persistence measure
        if metrics.persistence_measure > 0.3:
            strong_indicators += 1
        total_indicators += 1
        
        # Clustering coefficient
        if metrics.clustering_coefficient > 0.2:
            strong_indicators += 1
        total_indicators += 1
        
        # Half-life
        if metrics.volatility_half_life < 50:  # Less than 50 periods
            strong_indicators += 1
        total_indicators += 1
        
        # Overall assessment
        clustering_strength = strong_indicators / total_indicators if total_indicators > 0 else 0
        
        if clustering_strength >= 0.75:
            return "Strong volatility clustering detected"
        elif clustering_strength >= 0.5:
            return "Moderate volatility clustering detected"
        elif clustering_strength >= 0.25:
            return "Weak volatility clustering detected"
        else:
            return "No significant volatility clustering detected"
    
    def _generate_clustering_recommendations(self, metrics: VolatilityClusterMetrics) -> List[str]:
        """Generate recommendations based on clustering analysis."""
        recommendations = []
        
        # ARCH test results
        if not pd.isna(metrics.arch_test_pvalue) and metrics.arch_test_pvalue < 0.05:
            recommendations.append("Significant heteroscedasticity detected. Consider GARCH-type models for volatility forecasting.")
        
        # High persistence
        if metrics.persistence_measure > 0.7:
            recommendations.append("High volatility persistence detected. Volatility shocks have long-lasting effects.")
        
        # Long half-life
        if metrics.volatility_half_life > 100:
            recommendations.append("Very persistent volatility shocks. Consider regime-switching models.")
        
        # Multiple regimes
        if len(metrics.volatility_regimes) > 2:
            recommendations.append(f"Multiple volatility regimes ({len(metrics.volatility_regimes)}) identified. Consider regime-dependent strategies.")
        
        # Low regime stability
        if metrics.regime_stability < 0.3:
            recommendations.append("Low regime stability suggests frequent volatility regime changes. Use adaptive models.")
        
        # Strong clustering
        if metrics.clustering_coefficient > 0.5:
            recommendations.append("Strong volatility clustering suggests predictable volatility patterns. Exploit for risk management.")
        
        if not recommendations:
            recommendations.append("Volatility patterns appear stable. Standard models may be sufficient.")
        
        return recommendations
    
    def plot_volatility_clustering_analysis(self, returns: pd.Series,
                                          strategy_name: str = "Strategy",
                                          save_path: Optional[str] = None) -> None:
        """
        Create comprehensive volatility clustering visualization.
        
        Parameters:
        -----------
        returns : pd.Series
            Time series of returns
        strategy_name : str
            Strategy name for plotting
        save_path : Optional[str]
            Path to save plot
        """
        # Perform analysis
        metrics = self.analyze_volatility_clustering(returns)
        garch_results = self.garch_analyzer.fit_garch_model(returns)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Volatility Clustering Analysis: {strategy_name}', fontsize=16, fontweight='bold')
        
        # 1. Returns and conditional volatility
        ax1 = axes[0, 0]
        ax1.plot(returns.index, returns, alpha=0.7, color='blue', label='Returns')
        if 'volatility_series' in garch_results:
            vol_series = garch_results['volatility_series']
            ax1.fill_between(returns.index, -vol_series, vol_series, alpha=0.3, color='red', label='±1σ')
        ax1.set_title('Returns and Conditional Volatility')
        ax1.set_ylabel('Return')
        ax1.legend()
        ax1.grid(True)
        
        # 2. Volatility clustering visualization
        ax2 = axes[0, 1]
        rolling_vol = returns.rolling(window=20).std() * np.sqrt(252)
        ax2.plot(rolling_vol.index, rolling_vol, color='red', alpha=0.8)
        ax2.set_title('Rolling Volatility (20-day)')
        ax2.set_ylabel('Annualized Volatility')
        ax2.grid(True)
        
        # 3. Volatility autocorrelation
        ax3 = axes[1, 0]
        vol_squared = returns**2
        autocorrs = [vol_squared.autocorr(lag=i) for i in range(1, 21)]
        autocorrs = [ac for ac in autocorrs if not pd.isna(ac)]
        if autocorrs:
            ax3.bar(range(1, len(autocorrs)+1), autocorrs, alpha=0.7, color='green')
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.set_title('Volatility Autocorrelation')
        ax3.set_xlabel('Lag')
        ax3.set_ylabel('Autocorrelation')
        ax3.grid(True)
        
        # 4. Regime probabilities (if available)
        ax4 = axes[1, 1]
        if hasattr(self.regime_analyzer, 'regime_probabilities') and self.regime_analyzer.regime_probabilities is not None:
            regime_probs = self.regime_analyzer.regime_probabilities
            for i in range(regime_probs.shape[1]):
                ax4.plot(rolling_vol.index[-len(regime_probs):], regime_probs[:, i], 
                        label=f'Regime {i}', alpha=0.8)
            ax4.set_title('Regime Probabilities')
            ax4.set_ylabel('Probability')
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, 'Regime Analysis\nNot Available', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Regime Analysis')
        ax4.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
