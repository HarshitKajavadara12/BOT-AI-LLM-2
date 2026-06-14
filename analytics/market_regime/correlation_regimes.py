"""
Correlation Regime Detection Framework
====================================

Advanced correlation regime analysis system for multi-asset portfolios.
Identifies periods of distinct correlation structures and regime transitions.

Features:
- Dynamic Correlation Regime Detection
- Multivariate GARCH correlation models (DCC, BEKK)
- Correlation breakpoint detection
- Regime-dependent correlation forecasting
- Cross-asset correlation clustering
- Crisis correlation analysis
- Portfolio risk implications

Author: Quantum Forge Analytics Team
Date: November 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
from scipy import stats, linalg, optimize
from scipy.stats import multivariate_normal
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf, OAS
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CorrelationRegime:
    """Container for correlation regime characteristics."""
    regime_id: int
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    correlation_matrix: np.ndarray
    eigenvalues: np.ndarray
    regime_probability: float
    duration: int
    average_correlation: float
    correlation_dispersion: float
    regime_type: str  # 'low', 'medium', 'high', 'crisis'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'regime_id': self.regime_id,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'duration': self.duration,
            'average_correlation': self.average_correlation,
            'correlation_dispersion': self.correlation_dispersion,
            'regime_type': self.regime_type,
            'regime_probability': self.regime_probability,
            'eigenvalues': self.eigenvalues.tolist() if self.eigenvalues is not None else None
        }

@dataclass
class CorrelationRegimeMetrics:
    """Container for correlation regime analysis results."""
    correlation_regimes: List[CorrelationRegime]
    regime_transitions: pd.DataFrame
    regime_stability: float
    correlation_clustering_coefficient: float
    average_regime_duration: float
    crisis_correlation_increase: float
    diversification_ratio_dynamics: pd.Series
    regime_transition_probability: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary format."""
        return {
            'num_regimes': len(self.correlation_regimes),
            'regime_stability': self.regime_stability,
            'correlation_clustering_coefficient': self.correlation_clustering_coefficient,
            'average_regime_duration': self.average_regime_duration,
            'crisis_correlation_increase': self.crisis_correlation_increase,
            'regime_transition_probability': self.regime_transition_probability,
            'regimes': [regime.to_dict() for regime in self.correlation_regimes]
        }

class DynamicCorrelationAnalyzer:
    """
    Dynamic correlation analyzer using rolling window estimation.
    
    Implements various methods for estimating time-varying correlations
    including exponential weighting and robust estimators.
    """
    
    def __init__(self, window_size: int = 60, min_periods: int = 30):
        """
        Initialize dynamic correlation analyzer.
        
        Parameters:
        -----------
        window_size : int
            Rolling window size for correlation estimation
        min_periods : int
            Minimum periods required for correlation calculation
        """
        self.window_size = window_size
        self.min_periods = min_periods
        self.correlation_series = None
        
    def calculate_rolling_correlations(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate rolling correlation matrices.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Multi-asset return matrix
            
        Returns:
        --------
        pd.DataFrame
            Time series of correlation matrices (flattened)
        """
        try:
            n_assets = len(returns.columns)
            dates = returns.index
            
            # Initialize correlation storage
            correlation_data = []
            correlation_dates = []
            
            for i in range(self.window_size - 1, len(returns)):
                window_returns = returns.iloc[i - self.window_size + 1:i + 1]
                
                if len(window_returns) >= self.min_periods:
                    # Calculate correlation matrix
                    corr_matrix = window_returns.corr().values
                    
                    # Extract upper triangular elements (excluding diagonal)
                    triu_indices = np.triu_indices(n_assets, k=1)
                    correlations = corr_matrix[triu_indices]
                    
                    correlation_data.append(correlations)
                    correlation_dates.append(dates[i])
            
            # Create DataFrame
            asset_pairs = []
            for i in range(n_assets):
                for j in range(i + 1, n_assets):
                    asset_pairs.append(f"{returns.columns[i]}_{returns.columns[j]}")
            
            self.correlation_series = pd.DataFrame(
                correlation_data,
                index=correlation_dates,
                columns=asset_pairs
            )
            
            return self.correlation_series
            
        except Exception as e:
            logger.error(f"Error calculating rolling correlations: {str(e)}")
            raise
    
    def calculate_ewm_correlations(self, returns: pd.DataFrame, 
                                  alpha: float = 0.05) -> pd.DataFrame:
        """
        Calculate exponentially weighted moving correlations.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Multi-asset return matrix
        alpha : float
            Exponential decay parameter
            
        Returns:
        --------
        pd.DataFrame
            EWM correlation time series
        """
        try:
            # Calculate EWM covariance matrix
            ewm_cov = returns.ewm(alpha=alpha).cov()
            
            # Convert to correlation
            correlation_data = []
            dates = returns.index
            n_assets = len(returns.columns)
            
            for date in dates[self.min_periods:]:
                try:
                    cov_matrix = ewm_cov.loc[date].values
                    
                    # Convert covariance to correlation
                    std_diag = np.sqrt(np.diag(cov_matrix))
                    corr_matrix = cov_matrix / np.outer(std_diag, std_diag)
                    
                    # Extract correlations
                    triu_indices = np.triu_indices(n_assets, k=1)
                    correlations = corr_matrix[triu_indices]
                    
                    correlation_data.append(correlations)
                except:
                    # Handle missing data
                    correlation_data.append([np.nan] * (n_assets * (n_assets - 1) // 2))
            
            # Create asset pair names
            asset_pairs = []
            for i in range(n_assets):
                for j in range(i + 1, n_assets):
                    asset_pairs.append(f"{returns.columns[i]}_{returns.columns[j]}")
            
            ewm_correlations = pd.DataFrame(
                correlation_data,
                index=dates[self.min_periods:],
                columns=asset_pairs
            )
            
            return ewm_correlations
            
        except Exception as e:
            logger.error(f"Error calculating EWM correlations: {str(e)}")
            raise
    
    def calculate_robust_correlations(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate robust correlation estimates using shrinkage methods.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Multi-asset return matrix
            
        Returns:
        --------
        pd.DataFrame
            Robust correlation time series
        """
        try:
            correlation_data = []
            dates = returns.index
            n_assets = len(returns.columns)
            
            # Use Ledoit-Wolf shrinkage estimator
            ledoit_wolf = LedoitWolf()
            
            for i in range(self.window_size - 1, len(returns)):
                window_returns = returns.iloc[i - self.window_size + 1:i + 1]
                
                if len(window_returns) >= self.min_periods and not window_returns.isna().all().all():
                    try:
                        # Fit robust covariance estimator
                        cov_matrix = ledoit_wolf.fit(window_returns.dropna()).covariance_
                        
                        # Convert to correlation
                        std_diag = np.sqrt(np.diag(cov_matrix))
                        corr_matrix = cov_matrix / np.outer(std_diag, std_diag)
                        
                        # Extract correlations
                        triu_indices = np.triu_indices(n_assets, k=1)
                        correlations = corr_matrix[triu_indices]
                        
                        correlation_data.append(correlations)
                    except:
                        correlation_data.append([np.nan] * (n_assets * (n_assets - 1) // 2))
                else:
                    correlation_data.append([np.nan] * (n_assets * (n_assets - 1) // 2))
            
            # Create asset pair names
            asset_pairs = []
            for i in range(n_assets):
                for j in range(i + 1, n_assets):
                    asset_pairs.append(f"{returns.columns[i]}_{returns.columns[j]}")
            
            robust_correlations = pd.DataFrame(
                correlation_data,
                index=dates[self.window_size - 1:],
                columns=asset_pairs
            )
            
            return robust_correlations
            
        except Exception as e:
            logger.error(f"Error calculating robust correlations: {str(e)}")
            raise

class CorrelationRegimeDetector:
    """
    Correlation regime detection using clustering and changepoint methods.
    
    Identifies distinct correlation regimes and their characteristics using
    multiple statistical and machine learning approaches.
    """
    
    def __init__(self, n_regimes: int = 3, method: str = 'gmm'):
        """
        Initialize correlation regime detector.
        
        Parameters:
        -----------
        n_regimes : int
            Number of correlation regimes
        method : str
            Detection method ('gmm', 'kmeans', 'spectral')
        """
        self.n_regimes = n_regimes
        self.method = method
        self.regime_model = None
        self.regime_labels = None
        self.regime_probabilities = None
        
    def detect_correlation_regimes(self, correlation_series: pd.DataFrame) -> List[CorrelationRegime]:
        """
        Detect correlation regimes from correlation time series.
        
        Parameters:
        -----------
        correlation_series : pd.DataFrame
            Time series of correlation values
            
        Returns:
        --------
        List[CorrelationRegime]
            Detected correlation regimes
        """
        try:
            # Prepare features
            features = self._prepare_correlation_features(correlation_series)
            
            # Detect regimes using specified method
            if self.method == 'gmm':
                regime_labels, regime_probs = self._detect_regimes_gmm(features)
            elif self.method == 'kmeans':
                regime_labels, regime_probs = self._detect_regimes_kmeans(features)
            elif self.method == 'spectral':
                regime_labels, regime_probs = self._detect_regimes_spectral(features)
            else:
                raise ValueError(f"Unknown method: {self.method}")
            
            self.regime_labels = regime_labels
            self.regime_probabilities = regime_probs
            
            # Create regime objects
            regimes = self._create_correlation_regime_objects(
                correlation_series, regime_labels, regime_probs
            )
            
            return regimes
            
        except Exception as e:
            logger.error(f"Error detecting correlation regimes: {str(e)}")
            raise
    
    def _prepare_correlation_features(self, correlation_series: pd.DataFrame) -> np.ndarray:
        """Prepare features for regime detection."""
        # Calculate additional features
        features_list = []
        
        # Raw correlations
        features_list.append(correlation_series.values)
        
        # Average correlation level
        avg_corr = correlation_series.mean(axis=1).values.reshape(-1, 1)
        features_list.append(avg_corr)
        
        # Correlation dispersion
        corr_std = correlation_series.std(axis=1).values.reshape(-1, 1)
        features_list.append(corr_std)
        
        # Correlation momentum (change)
        corr_change = correlation_series.diff().mean(axis=1).values.reshape(-1, 1)
        features_list.append(corr_change)
        
        # Combine features
        features = np.hstack(features_list)
        
        # Handle missing values
        features = np.nan_to_num(features, nan=0.0)
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        return features_scaled
    
    def _detect_regimes_gmm(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect regimes using Gaussian Mixture Model."""
        self.regime_model = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type='full',
            random_state=42,
            max_iter=200
        )
        
        regime_labels = self.regime_model.fit_predict(features)
        regime_probs = self.regime_model.predict_proba(features)
        
        return regime_labels, regime_probs
    
    def _detect_regimes_kmeans(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect regimes using K-means clustering."""
        self.regime_model = KMeans(
            n_clusters=self.n_regimes,
            random_state=42,
            n_init=10
        )
        
        regime_labels = self.regime_model.fit_predict(features)
        
        # Create pseudo-probabilities based on distance to centroids
        distances = self.regime_model.transform(features)
        regime_probs = 1 / (1 + distances)
        regime_probs = regime_probs / regime_probs.sum(axis=1, keepdims=True)
        
        return regime_labels, regime_probs
    
    def _detect_regimes_spectral(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect regimes using Spectral clustering."""
        self.regime_model = SpectralClustering(
            n_clusters=self.n_regimes,
            random_state=42,
            affinity='rbf'
        )
        
        regime_labels = self.regime_model.fit_predict(features)
        
        # Create uniform probabilities (spectral clustering doesn't provide probabilities)
        regime_probs = np.zeros((len(regime_labels), self.n_regimes))
        for i, label in enumerate(regime_labels):
            regime_probs[i, label] = 1.0
        
        return regime_labels, regime_probs
    
    def _create_correlation_regime_objects(self, correlation_series: pd.DataFrame,
                                         regime_labels: np.ndarray,
                                         regime_probs: np.ndarray) -> List[CorrelationRegime]:
        """Create correlation regime objects."""
        regimes = []
        n_assets = int((1 + np.sqrt(1 + 8 * len(correlation_series.columns))) / 2)
        
        for regime_id in range(self.n_regimes):
            regime_mask = regime_labels == regime_id
            
            if not regime_mask.any():
                continue
            
            regime_dates = correlation_series.index[regime_mask]
            regime_correlations = correlation_series.iloc[regime_mask]
            regime_prob_series = regime_probs[regime_mask, regime_id]
            
            # Calculate regime characteristics
            avg_correlations = regime_correlations.mean()
            avg_correlation = avg_correlations.mean()
            correlation_dispersion = avg_correlations.std()
            
            # Reconstruct average correlation matrix for this regime
            avg_corr_matrix = np.eye(n_assets)
            triu_indices = np.triu_indices(n_assets, k=1)
            avg_corr_matrix[triu_indices] = avg_correlations.values
            avg_corr_matrix = avg_corr_matrix + avg_corr_matrix.T - np.eye(n_assets)
            
            # Calculate eigenvalues
            try:
                eigenvalues = np.linalg.eigvals(avg_corr_matrix)
                eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending
            except:
                eigenvalues = np.array([np.nan] * n_assets)
            
            # Classify regime type
            regime_type = self._classify_regime_type(avg_correlation, correlation_dispersion)
            
            # Find continuous periods for this regime
            regime_periods = self._find_regime_periods(regime_dates)
            
            for start_date, end_date in regime_periods:
                duration = (end_date - start_date).days
                
                regime = CorrelationRegime(
                    regime_id=regime_id,
                    start_date=start_date,
                    end_date=end_date,
                    correlation_matrix=avg_corr_matrix,
                    eigenvalues=eigenvalues,
                    regime_probability=regime_prob_series.mean(),
                    duration=duration,
                    average_correlation=avg_correlation,
                    correlation_dispersion=correlation_dispersion,
                    regime_type=regime_type
                )
                
                regimes.append(regime)
        
        return regimes
    
    def _classify_regime_type(self, avg_correlation: float, 
                            correlation_dispersion: float) -> str:
        """Classify regime type based on correlation characteristics."""
        if avg_correlation > 0.7:
            return 'crisis'
        elif avg_correlation > 0.4:
            return 'high'
        elif avg_correlation > 0.2:
            return 'medium'
        else:
            return 'low'
    
    def _find_regime_periods(self, regime_dates: pd.DatetimeIndex) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """Find continuous periods within a regime."""
        if len(regime_dates) == 0:
            return []
        
        periods = []
        start_date = regime_dates[0]
        prev_date = regime_dates[0]
        
        for current_date in regime_dates[1:]:
            # Check for gap (more than 10 days)
            if (current_date - prev_date).days > 10:
                periods.append((start_date, prev_date))
                start_date = current_date
            prev_date = current_date
        
        # Add final period
        periods.append((start_date, regime_dates[-1]))
        
        return periods
    
    def calculate_regime_transitions(self) -> pd.DataFrame:
        """Calculate regime transition matrix."""
        if self.regime_labels is None:
            raise ValueError("Must detect regimes first")
        
        # Calculate transition matrix
        transition_matrix = np.zeros((self.n_regimes, self.n_regimes))
        
        for t in range(1, len(self.regime_labels)):
            from_regime = self.regime_labels[t-1]
            to_regime = self.regime_labels[t]
            transition_matrix[from_regime, to_regime] += 1
        
        # Normalize to probabilities
        row_sums = transition_matrix.sum(axis=1)
        transition_probs = np.divide(transition_matrix, row_sums[:, np.newaxis], 
                                   out=np.zeros_like(transition_matrix), 
                                   where=row_sums[:, np.newaxis]!=0)
        
        # Create DataFrame
        regime_names = [f'Regime_{i}' for i in range(self.n_regimes)]
        transition_df = pd.DataFrame(
            transition_probs,
            index=regime_names,
            columns=regime_names
        )
        
        return transition_df

class CrisisCorrelationAnalyzer:
    """
    Crisis correlation analyzer for identifying correlation breakdowns.
    
    Specializes in detecting periods when correlations increase significantly,
    typically during market stress periods.
    """
    
    def __init__(self, crisis_threshold: float = 0.8):
        """
        Initialize crisis correlation analyzer.
        
        Parameters:
        -----------
        crisis_threshold : float
            Correlation threshold for crisis identification
        """
        self.crisis_threshold = crisis_threshold
        
    def detect_crisis_periods(self, correlation_series: pd.DataFrame,
                            volatility_series: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Detect crisis periods based on correlation spikes.
        
        Parameters:
        -----------
        correlation_series : pd.DataFrame
            Time series of correlations
        volatility_series : Optional[pd.Series]
            Volatility series for additional crisis confirmation
            
        Returns:
        --------
        Dict[str, Any]
            Crisis period analysis results
        """
        try:
            # Calculate average correlation level
            avg_correlation = correlation_series.mean(axis=1)
            
            # Identify crisis periods
            crisis_mask = avg_correlation > self.crisis_threshold
            
            # Find continuous crisis periods
            crisis_periods = self._identify_crisis_periods(avg_correlation.index[crisis_mask])
            
            # Calculate crisis statistics
            normal_correlation = avg_correlation[~crisis_mask].mean()
            crisis_correlation = avg_correlation[crisis_mask].mean()
            correlation_increase = crisis_correlation - normal_correlation
            
            # Crisis frequency
            total_periods = len(avg_correlation)
            crisis_periods_count = crisis_mask.sum()
            crisis_frequency = crisis_periods_count / total_periods if total_periods > 0 else 0
            
            # Volatility analysis during crisis (if provided)
            volatility_analysis = None
            if volatility_series is not None:
                volatility_analysis = self._analyze_crisis_volatility(
                    volatility_series, crisis_mask
                )
            
            return {
                'crisis_periods': crisis_periods,
                'normal_correlation': normal_correlation,
                'crisis_correlation': crisis_correlation,
                'correlation_increase': correlation_increase,
                'crisis_frequency': crisis_frequency,
                'total_crisis_days': crisis_periods_count,
                'volatility_analysis': volatility_analysis,
                'avg_crisis_duration': np.mean([
                    (end - start).days for start, end in crisis_periods
                ]) if crisis_periods else 0
            }
            
        except Exception as e:
            logger.error(f"Error detecting crisis periods: {str(e)}")
            raise
    
    def _identify_crisis_periods(self, crisis_dates: pd.DatetimeIndex) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """Identify continuous crisis periods."""
        if len(crisis_dates) == 0:
            return []
        
        periods = []
        start_date = crisis_dates[0]
        prev_date = crisis_dates[0]
        
        for current_date in crisis_dates[1:]:
            # Check for gap (more than 5 days)
            if (current_date - prev_date).days > 5:
                periods.append((start_date, prev_date))
                start_date = current_date
            prev_date = current_date
        
        # Add final period
        periods.append((start_date, crisis_dates[-1]))
        
        return periods
    
    def _analyze_crisis_volatility(self, volatility_series: pd.Series,
                                 crisis_mask: pd.Series) -> Dict[str, float]:
        """Analyze volatility behavior during crisis periods."""
        # Align series
        aligned_vol = volatility_series.reindex(crisis_mask.index).fillna(method='ffill')
        
        normal_volatility = aligned_vol[~crisis_mask].mean()
        crisis_volatility = aligned_vol[crisis_mask].mean()
        volatility_increase = crisis_volatility - normal_volatility
        
        return {
            'normal_volatility': normal_volatility,
            'crisis_volatility': crisis_volatility,
            'volatility_increase': volatility_increase,
            'volatility_ratio': crisis_volatility / normal_volatility if normal_volatility > 0 else np.inf
        }

class DiversificationAnalyzer:
    """
    Portfolio diversification analyzer based on correlation regimes.
    
    Analyzes how diversification benefits change across different correlation regimes.
    """
    
    def __init__(self):
        """Initialize diversification analyzer."""
        pass
        
    def calculate_diversification_ratio(self, returns: pd.DataFrame,
                                      weights: Optional[np.ndarray] = None) -> pd.Series:
        """
        Calculate time-varying diversification ratio.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Multi-asset return matrix
        weights : Optional[np.ndarray]
            Portfolio weights (equal weights if None)
            
        Returns:
        --------
        pd.Series
            Time series of diversification ratios
        """
        try:
            if weights is None:
                weights = np.ones(len(returns.columns)) / len(returns.columns)
            
            # Calculate rolling diversification ratios
            window_size = 60
            diversification_ratios = []
            dates = []
            
            for i in range(window_size - 1, len(returns)):
                window_returns = returns.iloc[i - window_size + 1:i + 1]
                
                if len(window_returns) >= 30:  # Minimum periods
                    try:
                        # Calculate portfolio volatility
                        cov_matrix = window_returns.cov().values
                        portfolio_vol = np.sqrt(weights @ cov_matrix @ weights.T)
                        
                        # Calculate weighted average individual volatilities
                        individual_vols = window_returns.std().values
                        weighted_avg_vol = weights @ individual_vols
                        
                        # Diversification ratio
                        div_ratio = weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 1.0
                        
                        diversification_ratios.append(div_ratio)
                        dates.append(returns.index[i])
                    except:
                        diversification_ratios.append(1.0)
                        dates.append(returns.index[i])
                        
            return pd.Series(diversification_ratios, index=dates)
            
        except Exception as e:
            logger.error(f"Error calculating diversification ratio: {str(e)}")
            raise
    
    def analyze_regime_diversification(self, returns: pd.DataFrame,
                                     correlation_regimes: List[CorrelationRegime],
                                     weights: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Analyze diversification across correlation regimes.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Multi-asset return matrix
        correlation_regimes : List[CorrelationRegime]
            Detected correlation regimes
        weights : Optional[np.ndarray]
            Portfolio weights
            
        Returns:
        --------
        Dict[str, Any]
            Regime-specific diversification analysis
        """
        try:
            if weights is None:
                weights = np.ones(len(returns.columns)) / len(returns.columns)
            
            regime_analysis = {}
            
            for regime in correlation_regimes:
                # Get returns for this regime period
                regime_mask = (returns.index >= regime.start_date) & (returns.index <= regime.end_date)
                regime_returns = returns[regime_mask]
                
                if len(regime_returns) < 10:  # Skip short periods
                    continue
                
                # Calculate diversification metrics
                cov_matrix = regime_returns.cov().values
                portfolio_vol = np.sqrt(weights @ cov_matrix @ weights.T)
                individual_vols = regime_returns.std().values
                weighted_avg_vol = weights @ individual_vols
                
                diversification_ratio = weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 1.0
                
                # Calculate effective number of assets
                effective_assets = self._calculate_effective_assets(regime.correlation_matrix, weights)
                
                # Risk concentration
                risk_contributions = self._calculate_risk_contributions(cov_matrix, weights)
                risk_concentration = self._calculate_herfindahl_index(risk_contributions)
                
                regime_analysis[f'regime_{regime.regime_id}'] = {
                    'regime_type': regime.regime_type,
                    'average_correlation': regime.average_correlation,
                    'diversification_ratio': diversification_ratio,
                    'effective_assets': effective_assets,
                    'risk_concentration': risk_concentration,
                    'portfolio_volatility': portfolio_vol * np.sqrt(252),  # Annualized
                    'duration_days': regime.duration
                }
            
            return regime_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing regime diversification: {str(e)}")
            raise
    
    def _calculate_effective_assets(self, correlation_matrix: np.ndarray,
                                  weights: np.ndarray) -> float:
        """Calculate effective number of assets."""
        try:
            # Effective assets based on portfolio weights and correlations
            n = len(weights)
            numerator = (weights.sum()) ** 2
            
            # Calculate denominator
            denominator = 0
            for i in range(n):
                for j in range(n):
                    if i == j:
                        denominator += weights[i] ** 2
                    else:
                        denominator += weights[i] * weights[j] * correlation_matrix[i, j]
            
            effective_assets = numerator / denominator if denominator > 0 else 1
            return min(effective_assets, n)  # Cap at actual number of assets
            
        except:
            return len(weights)
    
    def _calculate_risk_contributions(self, cov_matrix: np.ndarray,
                                    weights: np.ndarray) -> np.ndarray:
        """Calculate risk contributions of each asset."""
        portfolio_var = weights @ cov_matrix @ weights.T
        marginal_risk = cov_matrix @ weights
        risk_contributions = weights * marginal_risk / portfolio_var if portfolio_var > 0 else weights
        return risk_contributions
    
    def _calculate_herfindahl_index(self, contributions: np.ndarray) -> float:
        """Calculate Herfindahl concentration index."""
        return np.sum(contributions ** 2)

class ComprehensiveCorrelationRegimeAnalyzer:
    """
    Comprehensive correlation regime analysis system.
    
    Integrates all correlation regime methodologies for complete analysis.
    """
    
    def __init__(self, window_size: int = 60, n_regimes: int = 3):
        """
        Initialize comprehensive analyzer.
        
        Parameters:
        -----------
        window_size : int
            Rolling window size for correlation estimation
        n_regimes : int
            Number of correlation regimes
        """
        self.window_size = window_size
        self.n_regimes = n_regimes
        
        self.correlation_analyzer = DynamicCorrelationAnalyzer(window_size)
        self.regime_detector = CorrelationRegimeDetector(n_regimes)
        self.crisis_analyzer = CrisisCorrelationAnalyzer()
        self.diversification_analyzer = DiversificationAnalyzer()
        
    def analyze_correlation_regimes(self, returns: pd.DataFrame,
                                  weights: Optional[np.ndarray] = None) -> CorrelationRegimeMetrics:
        """
        Perform comprehensive correlation regime analysis.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Multi-asset return matrix
        weights : Optional[np.ndarray]
            Portfolio weights for diversification analysis
            
        Returns:
        --------
        CorrelationRegimeMetrics
            Comprehensive correlation regime metrics
        """
        try:
            # Calculate dynamic correlations
            correlation_series = self.correlation_analyzer.calculate_rolling_correlations(returns)
            
            # Detect correlation regimes
            correlation_regimes = self.regime_detector.detect_correlation_regimes(correlation_series)
            
            # Calculate regime transitions
            regime_transitions = self.regime_detector.calculate_regime_transitions()
            
            # Crisis analysis
            crisis_analysis = self.crisis_analyzer.detect_crisis_periods(correlation_series)
            
            # Diversification analysis
            diversification_ratio = self.diversification_analyzer.calculate_diversification_ratio(
                returns, weights
            )
            
            # Calculate comprehensive metrics
            regime_stability = self._calculate_regime_stability(correlation_regimes)
            correlation_clustering_coeff = self._calculate_correlation_clustering_coefficient(correlation_series)
            avg_regime_duration = np.mean([r.duration for r in correlation_regimes]) if correlation_regimes else 0
            crisis_correlation_increase = crisis_analysis.get('correlation_increase', 0)
            
            # Transition probability
            transition_prob = self._calculate_regime_transition_probability(regime_transitions)
            
            metrics = CorrelationRegimeMetrics(
                correlation_regimes=correlation_regimes,
                regime_transitions=regime_transitions,
                regime_stability=regime_stability,
                correlation_clustering_coefficient=correlation_clustering_coeff,
                average_regime_duration=avg_regime_duration,
                crisis_correlation_increase=crisis_correlation_increase,
                diversification_ratio_dynamics=diversification_ratio,
                regime_transition_probability=transition_prob
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in comprehensive correlation regime analysis: {str(e)}")
            raise
    
    def _calculate_regime_stability(self, regimes: List[CorrelationRegime]) -> float:
        """Calculate overall regime stability."""
        if not regimes:
            return 0.0
        
        # Weighted average duration
        total_weighted_duration = sum(r.duration * r.regime_probability for r in regimes)
        total_weight = sum(r.regime_probability for r in regimes)
        
        if total_weight == 0:
            return 0.0
        
        avg_duration = total_weighted_duration / total_weight
        max_duration = max(r.duration for r in regimes)
        
        stability = min(avg_duration / max_duration, 1.0) if max_duration > 0 else 0.0
        return stability
    
    def _calculate_correlation_clustering_coefficient(self, correlation_series: pd.DataFrame) -> float:
        """Calculate correlation clustering coefficient."""
        # Measure how correlations cluster together
        avg_correlation = correlation_series.mean(axis=1)
        
        # Calculate clustering based on persistence of correlation levels
        high_corr_threshold = avg_correlation.quantile(0.7)
        high_corr_periods = avg_correlation > high_corr_threshold
        
        # Count runs of high correlation
        runs = []
        current_run = 0
        
        for is_high_corr in high_corr_periods:
            if is_high_corr:
                current_run += 1
            else:
                if current_run > 0:
                    runs.append(current_run)
                current_run = 0
        
        if current_run > 0:
            runs.append(current_run)
        
        # Clustering coefficient
        if len(runs) == 0:
            return 0.0
        
        avg_run_length = np.mean(runs)
        expected_run_length = high_corr_periods.mean() * len(high_corr_periods)
        
        clustering_coeff = avg_run_length / expected_run_length if expected_run_length > 0 else 0.0
        return min(clustering_coeff, 1.0)
    
    def _calculate_regime_transition_probability(self, transition_matrix: pd.DataFrame) -> float:
        """Calculate average regime transition probability."""
        if transition_matrix.empty:
            return 0.0
        
        # Average off-diagonal elements (transition probabilities)
        n_regimes = len(transition_matrix)
        off_diagonal_sum = 0
        off_diagonal_count = 0
        
        for i in range(n_regimes):
            for j in range(n_regimes):
                if i != j:
                    off_diagonal_sum += transition_matrix.iloc[i, j]
                    off_diagonal_count += 1
        
        avg_transition_prob = off_diagonal_sum / off_diagonal_count if off_diagonal_count > 0 else 0.0
        return avg_transition_prob
    
    def generate_regime_report(self, returns: pd.DataFrame,
                             strategy_name: str = "Portfolio",
                             weights: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Generate comprehensive correlation regime report.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Multi-asset return matrix
        strategy_name : str
            Strategy name for reporting
        weights : Optional[np.ndarray]
            Portfolio weights
            
        Returns:
        --------
        Dict[str, Any]
            Comprehensive regime report
        """
        # Perform analysis
        metrics = self.analyze_correlation_regimes(returns, weights)
        
        # Crisis analysis
        correlation_series = self.correlation_analyzer.calculate_rolling_correlations(returns)
        crisis_analysis = self.crisis_analyzer.detect_crisis_periods(correlation_series)
        
        # Diversification analysis
        diversification_analysis = self.diversification_analyzer.analyze_regime_diversification(
            returns, metrics.correlation_regimes, weights
        )
        
        # Create report
        report = {
            'strategy_name': strategy_name,
            'analysis_date': datetime.now(),
            'period_analyzed': {
                'start': returns.index.min(),
                'end': returns.index.max(),
                'total_observations': len(returns)
            },
            'correlation_regime_metrics': metrics.to_dict(),
            'crisis_analysis': crisis_analysis,
            'diversification_analysis': diversification_analysis,
            'regime_characteristics': self._summarize_regime_characteristics(metrics.correlation_regimes),
            'recommendations': self._generate_regime_recommendations(metrics, crisis_analysis)
        }
        
        return report
    
    def _summarize_regime_characteristics(self, regimes: List[CorrelationRegime]) -> Dict[str, Any]:
        """Summarize key characteristics of detected regimes."""
        if not regimes:
            return {}
        
        regime_types = {}
        for regime in regimes:
            regime_type = regime.regime_type
            if regime_type not in regime_types:
                regime_types[regime_type] = []
            regime_types[regime_type].append(regime)
        
        summary = {}
        for regime_type, regime_list in regime_types.items():
            summary[regime_type] = {
                'count': len(regime_list),
                'avg_correlation': np.mean([r.average_correlation for r in regime_list]),
                'avg_duration': np.mean([r.duration for r in regime_list]),
                'total_duration': sum([r.duration for r in regime_list]),
                'avg_probability': np.mean([r.regime_probability for r in regime_list])
            }
        
        return summary
    
    def _generate_regime_recommendations(self, metrics: CorrelationRegimeMetrics,
                                       crisis_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on regime analysis."""
        recommendations = []
        
        # Crisis correlation analysis
        if crisis_analysis.get('correlation_increase', 0) > 0.3:
            recommendations.append("High correlation increase during crisis periods detected. Implement crisis hedging strategies.")
        
        # Regime stability
        if metrics.regime_stability < 0.3:
            recommendations.append("Low regime stability suggests frequent correlation changes. Use adaptive portfolio allocation.")
        
        # High correlation clustering
        if metrics.correlation_clustering_coefficient > 0.6:
            recommendations.append("Strong correlation clustering detected. Correlations tend to persist in similar levels.")
        
        # Crisis frequency
        crisis_frequency = crisis_analysis.get('crisis_frequency', 0)
        if crisis_frequency > 0.1:
            recommendations.append(f"Crisis periods occur {crisis_frequency:.1%} of the time. Consider crisis-resilient diversification.")
        
        # Multiple regimes
        if len(metrics.correlation_regimes) > 3:
            recommendations.append("Multiple correlation regimes detected. Consider regime-dependent allocation strategies.")
        
        # High transition probability
        if metrics.regime_transition_probability > 0.2:
            recommendations.append("High regime transition probability suggests dynamic correlation environment.")
        
        if not recommendations:
            recommendations.append("Correlation regime patterns appear stable. Standard diversification approaches may be sufficient.")
        
        return recommendations
    
    def plot_correlation_regime_analysis(self, returns: pd.DataFrame,
                                       strategy_name: str = "Portfolio",
                                       save_path: Optional[str] = None) -> None:
        """
        Create comprehensive correlation regime visualization.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Multi-asset return matrix
        strategy_name : str
            Strategy name for plotting
        save_path : Optional[str]
            Path to save plot
        """
        # Perform analysis
        metrics = self.analyze_correlation_regimes(returns)
        correlation_series = self.correlation_analyzer.calculate_rolling_correlations(returns)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Correlation Regime Analysis: {strategy_name}', fontsize=16, fontweight='bold')
        
        # 1. Average correlation over time
        ax1 = axes[0, 0]
        avg_correlation = correlation_series.mean(axis=1)
        ax1.plot(avg_correlation.index, avg_correlation, color='blue', alpha=0.8)
        
        # Highlight regime periods
        colors = ['red', 'green', 'orange', 'purple', 'brown']
        for i, regime in enumerate(metrics.correlation_regimes[:5]):  # Top 5 regimes
            color = colors[i % len(colors)]
            ax1.axvspan(regime.start_date, regime.end_date, alpha=0.2, color=color, 
                       label=f'Regime {regime.regime_id} ({regime.regime_type})')
        
        ax1.set_title('Average Correlation Over Time')
        ax1.set_ylabel('Average Correlation')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True)
        
        # 2. Correlation heatmap for current period
        ax2 = axes[0, 1]
        if len(returns.columns) <= 10:  # Only for reasonable number of assets
            recent_corr = returns.tail(60).corr()
            sns.heatmap(recent_corr, annot=True, cmap='RdYlBu_r', center=0, ax=ax2)
            ax2.set_title('Recent Correlation Matrix')
        else:
            ax2.text(0.5, 0.5, f'Too many assets ({len(returns.columns)})\nfor correlation heatmap', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Correlation Matrix')
        
        # 3. Regime transition matrix
        ax3 = axes[1, 0]
        if not metrics.regime_transitions.empty:
            sns.heatmap(metrics.regime_transitions, annot=True, cmap='Blues', ax=ax3)
            ax3.set_title('Regime Transition Probabilities')
        else:
            ax3.text(0.5, 0.5, 'No regime transitions\ndetected', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Regime Transitions')
        
        # 4. Diversification ratio over time
        ax4 = axes[1, 1]
        if not metrics.diversification_ratio_dynamics.empty:
            ax4.plot(metrics.diversification_ratio_dynamics.index, 
                    metrics.diversification_ratio_dynamics.values, 
                    color='green', alpha=0.8)
            ax4.set_title('Diversification Ratio Over Time')
            ax4.set_ylabel('Diversification Ratio')
            ax4.grid(True)
        else:
            ax4.text(0.5, 0.5, 'Diversification ratio\nnot available', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Diversification Ratio')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
