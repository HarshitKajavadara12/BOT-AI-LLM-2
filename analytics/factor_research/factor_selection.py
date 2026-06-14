"""
Factor Selection Framework
Advanced factor selection and screening for quantitative research
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
import time
from collections import defaultdict
from scipy import stats
from sklearn.feature_selection import (
    SelectKBest, SelectPercentile, RFE, RFECV,
    f_regression, mutual_info_regression,
    VarianceThreshold, SelectFromModel
)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import concurrent.futures


class SelectionMethod(Enum):
    """Factor selection methods"""
    CORRELATION = "CORRELATION"             # Correlation-based selection
    IC_ANALYSIS = "IC_ANALYSIS"             # Information Coefficient analysis
    MUTUAL_INFO = "MUTUAL_INFO"             # Mutual information
    VARIANCE_THRESHOLD = "VARIANCE_THRESHOLD"  # Variance threshold
    UNIVARIATE_F = "UNIVARIATE_F"          # Univariate F-test
    LASSO = "LASSO"                        # LASSO regularization
    RIDGE = "RIDGE"                        # Ridge regularization
    ELASTIC_NET = "ELASTIC_NET"            # Elastic Net regularization
    RANDOM_FOREST = "RANDOM_FOREST"        # Random Forest importance
    RECURSIVE_ELIMINATION = "RECURSIVE_ELIMINATION"  # Recursive feature elimination
    FORWARD_SELECTION = "FORWARD_SELECTION"  # Forward selection
    BACKWARD_ELIMINATION = "BACKWARD_ELIMINATION"  # Backward elimination
    BORUTA = "BORUTA"                      # Boruta algorithm
    STABILITY_SELECTION = "STABILITY_SELECTION"  # Stability selection


class SelectionCriteria(Enum):
    """Selection criteria"""
    TOP_K = "TOP_K"                        # Select top K factors
    TOP_PERCENTILE = "TOP_PERCENTILE"      # Select top percentile
    THRESHOLD = "THRESHOLD"                # Select above threshold
    CROSS_VALIDATION = "CROSS_VALIDATION"  # Cross-validation based
    INFORMATION_RATIO = "INFORMATION_RATIO"  # Information ratio based


@dataclass
class SelectionConfig:
    """Factor selection configuration"""
    method: SelectionMethod
    criteria: SelectionCriteria = SelectionCriteria.TOP_K
    
    # Selection parameters
    k: int = 10                            # Number of factors to select
    percentile: float = 0.2                # Percentile for selection (20%)
    threshold: float = 0.05                # Threshold value
    
    # Analysis parameters
    min_periods: int = 252                 # Minimum periods for analysis
    ic_method: str = "spearman"            # IC calculation method
    return_horizon: int = 1                # Return prediction horizon
    
    # Cross-validation parameters
    cv_folds: int = 5                      # Number of CV folds
    test_size: float = 0.2                 # Test set size
    
    # Regularization parameters
    alpha_range: Tuple[float, float] = (0.001, 10.0)  # Alpha range for regularization
    l1_ratio_range: Tuple[float, float] = (0.1, 0.9)   # L1 ratio for Elastic Net
    
    # Other parameters
    random_state: int = 42
    n_jobs: int = -1
    verbose: bool = False


@dataclass
class SelectionResult:
    """Factor selection results"""
    selected_factors: List[str]
    factor_scores: Dict[str, float]
    selection_metrics: Dict[str, Any]
    
    # Method-specific results
    method: SelectionMethod
    criteria: SelectionCriteria
    
    # Performance metrics
    in_sample_score: Optional[float] = None
    out_of_sample_score: Optional[float] = None
    stability_score: Optional[float] = None
    
    # Factor analysis
    factor_correlations: Optional[pd.DataFrame] = None
    ic_analysis: Optional[Dict[str, Any]] = None
    
    # Metadata
    selection_date: datetime = field(default_factory=datetime.now)
    total_factors_considered: int = 0
    selection_time_seconds: float = 0.0


class BaseFactorSelector(ABC):
    """Base class for factor selection"""
    
    def __init__(self, config: SelectionConfig):
        self.config = config
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, factors: pd.DataFrame, returns: pd.Series) -> 'BaseFactorSelector':
        """Fit the selector"""
        pass
    
    @abstractmethod
    def select(self, factors: pd.DataFrame) -> List[str]:
        """Select factors"""
        pass
    
    @abstractmethod
    def get_scores(self) -> Dict[str, float]:
        """Get factor scores"""
        pass
    
    def fit_select(self, factors: pd.DataFrame, returns: pd.Series) -> List[str]:
        """Fit and select factors"""
        self.fit(factors, returns)
        return self.select(factors)


class CorrelationSelector(BaseFactorSelector):
    """Correlation-based factor selection"""
    
    def __init__(self, config: SelectionConfig):
        super().__init__(config)
        self.correlations: Dict[str, float] = {}
    
    def fit(self, factors: pd.DataFrame, returns: pd.Series) -> 'CorrelationSelector':
        """Fit correlation selector"""
        
        # Align data
        common_index = factors.index.intersection(returns.index)
        factors_aligned = factors.loc[common_index]
        returns_aligned = returns.loc[common_index]
        
        # Calculate correlations
        self.correlations = {}
        
        for factor_name in factors_aligned.columns:
            factor_values = factors_aligned[factor_name].dropna()
            returns_subset = returns_aligned.loc[factor_values.index]
            
            if len(factor_values) >= self.config.min_periods:
                if self.config.ic_method == "spearman":
                    corr, _ = stats.spearmanr(factor_values, returns_subset)
                else:
                    corr, _ = stats.pearsonr(factor_values, returns_subset)
                
                self.correlations[factor_name] = abs(corr) if not np.isnan(corr) else 0.0
            else:
                self.correlations[factor_name] = 0.0
        
        self.is_fitted = True
        return self
    
    def select(self, factors: pd.DataFrame) -> List[str]:
        """Select factors based on correlation"""
        
        if not self.is_fitted:
            raise ValueError("Selector must be fitted before selecting")
        
        # Sort by correlation strength
        sorted_factors = sorted(
            self.correlations.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Apply selection criteria
        if self.config.criteria == SelectionCriteria.TOP_K:
            selected = [name for name, _ in sorted_factors[:self.config.k]]
        elif self.config.criteria == SelectionCriteria.TOP_PERCENTILE:
            n_select = int(len(sorted_factors) * self.config.percentile)
            selected = [name for name, _ in sorted_factors[:n_select]]
        elif self.config.criteria == SelectionCriteria.THRESHOLD:
            selected = [name for name, score in sorted_factors if score >= self.config.threshold]
        else:
            selected = [name for name, _ in sorted_factors[:self.config.k]]
        
        return selected
    
    def get_scores(self) -> Dict[str, float]:
        """Get correlation scores"""
        return self.correlations.copy()


class ICSelector(BaseFactorSelector):
    """Information Coefficient-based factor selection"""
    
    def __init__(self, config: SelectionConfig):
        super().__init__(config)
        self.ic_metrics: Dict[str, Dict[str, float]] = {}
    
    def fit(self, factors: pd.DataFrame, returns: pd.Series) -> 'ICSelector':
        """Fit IC selector"""
        
        # Calculate forward returns if needed
        if self.config.return_horizon > 1:
            returns = returns.shift(-self.config.return_horizon)
        
        # Align data
        common_index = factors.index.intersection(returns.index)
        factors_aligned = factors.loc[common_index]
        returns_aligned = returns.loc[common_index]
        
        # Calculate IC metrics for each factor
        self.ic_metrics = {}
        
        for factor_name in factors_aligned.columns:
            ic_analysis = self._calculate_ic_metrics(
                factors_aligned[factor_name], 
                returns_aligned
            )
            self.ic_metrics[factor_name] = ic_analysis
        
        self.is_fitted = True
        return self
    
    def _calculate_ic_metrics(self, factor_values: pd.Series, 
                            returns: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive IC metrics"""
        
        # Remove NaN values
        valid_data = pd.DataFrame({'factor': factor_values, 'returns': returns}).dropna()
        
        if len(valid_data) < self.config.min_periods:
            return {
                'ic_mean': 0.0,
                'ic_std': 0.0,
                'ic_ir': 0.0,
                'ic_hit_rate': 0.0,
                'ic_skew': 0.0
            }
        
        # Calculate rolling IC
        window_size = min(63, len(valid_data) // 4)  # Quarterly windows
        
        if window_size < 20:
            # Calculate single IC
            if self.config.ic_method == "spearman":
                ic, _ = stats.spearmanr(valid_data['factor'], valid_data['returns'])
            else:
                ic, _ = stats.pearsonr(valid_data['factor'], valid_data['returns'])
            
            ic = ic if not np.isnan(ic) else 0.0
            
            return {
                'ic_mean': ic,
                'ic_std': 0.0,
                'ic_ir': 0.0,
                'ic_hit_rate': 1.0 if ic > 0 else 0.0,
                'ic_skew': 0.0
            }
        
        # Rolling IC calculation
        rolling_ics = []
        
        for i in range(window_size, len(valid_data)):
            window_data = valid_data.iloc[i-window_size:i]
            
            if self.config.ic_method == "spearman":
                ic, _ = stats.spearmanr(window_data['factor'], window_data['returns'])
            else:
                ic, _ = stats.pearsonr(window_data['factor'], window_data['returns'])
            
            if not np.isnan(ic):
                rolling_ics.append(ic)
        
        if not rolling_ics:
            return {
                'ic_mean': 0.0,
                'ic_std': 0.0,
                'ic_ir': 0.0,
                'ic_hit_rate': 0.0,
                'ic_skew': 0.0
            }
        
        rolling_ics = np.array(rolling_ics)
        
        # Calculate IC metrics
        ic_mean = np.mean(rolling_ics)
        ic_std = np.std(rolling_ics)
        ic_ir = ic_mean / ic_std if ic_std > 0 else 0.0
        ic_hit_rate = np.mean(rolling_ics > 0)
        ic_skew = stats.skew(rolling_ics)
        
        return {
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            'ic_ir': ic_ir,
            'ic_hit_rate': ic_hit_rate,
            'ic_skew': ic_skew
        }
    
    def select(self, factors: pd.DataFrame) -> List[str]:
        """Select factors based on IC analysis"""
        
        if not self.is_fitted:
            raise ValueError("Selector must be fitted before selecting")
        
        # Sort by IC Information Ratio
        sorted_factors = sorted(
            self.ic_metrics.items(),
            key=lambda x: x[1]['ic_ir'],
            reverse=True
        )
        
        # Apply selection criteria
        if self.config.criteria == SelectionCriteria.TOP_K:
            selected = [name for name, _ in sorted_factors[:self.config.k]]
        elif self.config.criteria == SelectionCriteria.TOP_PERCENTILE:
            n_select = int(len(sorted_factors) * self.config.percentile)
            selected = [name for name, _ in sorted_factors[:n_select]]
        elif self.config.criteria == SelectionCriteria.THRESHOLD:
            selected = [name for name, metrics in sorted_factors 
                       if metrics['ic_ir'] >= self.config.threshold]
        else:
            selected = [name for name, _ in sorted_factors[:self.config.k]]
        
        return selected
    
    def get_scores(self) -> Dict[str, float]:
        """Get IC scores (using IC IR)"""
        return {name: metrics['ic_ir'] for name, metrics in self.ic_metrics.items()}
    
    def get_ic_analysis(self) -> Dict[str, Dict[str, float]]:
        """Get detailed IC analysis"""
        return self.ic_metrics.copy()


class LassoSelector(BaseFactorSelector):
    """LASSO-based factor selection"""
    
    def __init__(self, config: SelectionConfig):
        super().__init__(config)
        self.lasso_model = None
        self.feature_importance: Dict[str, float] = {}
    
    def fit(self, factors: pd.DataFrame, returns: pd.Series) -> 'LassoSelector':
        """Fit LASSO selector"""
        
        # Align and clean data
        common_index = factors.index.intersection(returns.index)
        X = factors.loc[common_index].fillna(0)
        y = returns.loc[common_index].fillna(0)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit LASSO with cross-validation
        alphas = np.logspace(
            np.log10(self.config.alpha_range[0]),
            np.log10(self.config.alpha_range[1]),
            50
        )
        
        self.lasso_model = LassoCV(
            alphas=alphas,
            cv=self.config.cv_folds,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs
        )
        
        self.lasso_model.fit(X_scaled, y)
        
        # Extract feature importance
        self.feature_importance = {}
        for i, factor_name in enumerate(factors.columns):
            self.feature_importance[factor_name] = abs(self.lasso_model.coef_[i])
        
        self.is_fitted = True
        return self
    
    def select(self, factors: pd.DataFrame) -> List[str]:
        """Select factors based on LASSO coefficients"""
        
        if not self.is_fitted:
            raise ValueError("Selector must be fitted before selecting")
        
        # Select non-zero coefficients
        non_zero_factors = [
            name for name, coef in self.feature_importance.items() 
            if coef > 1e-6
        ]
        
        # Sort by coefficient magnitude
        sorted_factors = sorted(
            [(name, coef) for name, coef in self.feature_importance.items() if coef > 1e-6],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Apply selection criteria
        if self.config.criteria == SelectionCriteria.TOP_K:
            selected = [name for name, _ in sorted_factors[:self.config.k]]
        elif self.config.criteria == SelectionCriteria.TOP_PERCENTILE:
            n_select = int(len(sorted_factors) * self.config.percentile)
            selected = [name for name, _ in sorted_factors[:n_select]]
        else:
            selected = non_zero_factors
        
        return selected
    
    def get_scores(self) -> Dict[str, float]:
        """Get LASSO coefficients"""
        return self.feature_importance.copy()


class RandomForestSelector(BaseFactorSelector):
    """Random Forest-based factor selection"""
    
    def __init__(self, config: SelectionConfig):
        super().__init__(config)
        self.rf_model = None
        self.feature_importance: Dict[str, float] = {}
    
    def fit(self, factors: pd.DataFrame, returns: pd.Series) -> 'RandomForestSelector':
        """Fit Random Forest selector"""
        
        # Align and clean data
        common_index = factors.index.intersection(returns.index)
        X = factors.loc[common_index].fillna(method='ffill').fillna(0)
        y = returns.loc[common_index].fillna(0)
        
        # Fit Random Forest
        self.rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs
        )
        
        self.rf_model.fit(X, y)
        
        # Extract feature importance
        self.feature_importance = {}
        for i, factor_name in enumerate(factors.columns):
            self.feature_importance[factor_name] = self.rf_model.feature_importances_[i]
        
        self.is_fitted = True
        return self
    
    def select(self, factors: pd.DataFrame) -> List[str]:
        """Select factors based on Random Forest importance"""
        
        if not self.is_fitted:
            raise ValueError("Selector must be fitted before selecting")
        
        # Sort by importance
        sorted_factors = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Apply selection criteria
        if self.config.criteria == SelectionCriteria.TOP_K:
            selected = [name for name, _ in sorted_factors[:self.config.k]]
        elif self.config.criteria == SelectionCriteria.TOP_PERCENTILE:
            n_select = int(len(sorted_factors) * self.config.percentile)
            selected = [name for name, _ in sorted_factors[:n_select]]
        elif self.config.criteria == SelectionCriteria.THRESHOLD:
            selected = [name for name, score in sorted_factors if score >= self.config.threshold]
        else:
            selected = [name for name, _ in sorted_factors[:self.config.k]]
        
        return selected
    
    def get_scores(self) -> Dict[str, float]:
        """Get Random Forest importance scores"""
        return self.feature_importance.copy()


class StabilitySelector(BaseFactorSelector):
    """Stability-based factor selection"""
    
    def __init__(self, config: SelectionConfig):
        super().__init__(config)
        self.stability_scores: Dict[str, float] = {}
        self.n_bootstrap = 100
    
    def fit(self, factors: pd.DataFrame, returns: pd.Series) -> 'StabilitySelector':
        """Fit stability selector using bootstrap sampling"""
        
        # Align data
        common_index = factors.index.intersection(returns.index)
        X = factors.loc[common_index].fillna(method='ffill').fillna(0)
        y = returns.loc[common_index].fillna(0)
        
        # Bootstrap stability selection
        selection_frequency = defaultdict(int)
        
        for bootstrap_iter in range(self.n_bootstrap):
            # Deterministic block bootstrap: choose contiguous blocks with evenly spaced starts
            n_samples = len(X)
            block_size = max(1, int(0.8 * n_samples))
            if n_samples <= block_size:
                bootstrap_indices = np.arange(n_samples)
            else:
                # Evenly spaced deterministic start positions
                max_start = n_samples - block_size
                start = int((bootstrap_iter * max_start) / max(1, (self.n_bootstrap - 1)))
                bootstrap_indices = np.arange(start, start + block_size) % n_samples

            X_bootstrap = X.iloc[bootstrap_indices]
            y_bootstrap = y.iloc[bootstrap_indices]
            
            # Fit LASSO on bootstrap sample
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_bootstrap)
            
            lasso = LassoCV(
                cv=3,
                random_state=self.config.random_state + bootstrap_iter,
                n_jobs=1
            )
            
            try:
                lasso.fit(X_scaled, y_bootstrap)
                
                # Count selected features
                for i, factor_name in enumerate(X.columns):
                    if abs(lasso.coef_[i]) > 1e-6:
                        selection_frequency[factor_name] += 1
                        
            except Exception:
                continue
        
        # Calculate stability scores
        self.stability_scores = {}
        for factor_name in factors.columns:
            self.stability_scores[factor_name] = selection_frequency[factor_name] / self.n_bootstrap
        
        self.is_fitted = True
        return self
    
    def select(self, factors: pd.DataFrame) -> List[str]:
        """Select factors based on stability scores"""
        
        if not self.is_fitted:
            raise ValueError("Selector must be fitted before selecting")
        
        # Sort by stability score
        sorted_factors = sorted(
            self.stability_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Apply selection criteria
        if self.config.criteria == SelectionCriteria.TOP_K:
            selected = [name for name, _ in sorted_factors[:self.config.k]]
        elif self.config.criteria == SelectionCriteria.TOP_PERCENTILE:
            n_select = int(len(sorted_factors) * self.config.percentile)
            selected = [name for name, _ in sorted_factors[:n_select]]
        elif self.config.criteria == SelectionCriteria.THRESHOLD:
            selected = [name for name, score in sorted_factors if score >= self.config.threshold]
        else:
            selected = [name for name, _ in sorted_factors[:self.config.k]]
        
        return selected
    
    def get_scores(self) -> Dict[str, float]:
        """Get stability scores"""
        return self.stability_scores.copy()


class FactorSelector:
    """
    Main factor selection engine
    """
    
    def __init__(self):
        self.selectors: Dict[SelectionMethod, BaseFactorSelector] = {}
        self.selection_history: List[SelectionResult] = []
        
    def create_selector(self, method: SelectionMethod, config: SelectionConfig) -> BaseFactorSelector:
        """Create selector instance"""
        
        selector_map = {
            SelectionMethod.CORRELATION: CorrelationSelector,
            SelectionMethod.IC_ANALYSIS: ICSelector,
            SelectionMethod.LASSO: LassoSelector,
            SelectionMethod.RANDOM_FOREST: RandomForestSelector,
            SelectionMethod.STABILITY_SELECTION: StabilitySelector
        }
        
        if method not in selector_map:
            raise ValueError(f"Unsupported selection method: {method}")
        
        return selector_map[method](config)
    
    def select_factors(self, factors: pd.DataFrame, returns: pd.Series, 
                      method: SelectionMethod, config: SelectionConfig) -> SelectionResult:
        """Select factors using specified method"""
        
        start_time = time.time()
        
        # Create and fit selector
        selector = self.create_selector(method, config)
        
        try:
            # Perform selection
            selected_factors = selector.fit_select(factors, returns)
            factor_scores = selector.get_scores()
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(
                factors, returns, selected_factors
            )
            
            # Create result
            result = SelectionResult(
                selected_factors=selected_factors,
                factor_scores=factor_scores,
                selection_metrics=performance_metrics,
                method=method,
                criteria=config.criteria,
                total_factors_considered=len(factors.columns),
                selection_time_seconds=time.time() - start_time
            )
            
            # Add method-specific analysis
            if method == SelectionMethod.IC_ANALYSIS and isinstance(selector, ICSelector):
                result.ic_analysis = selector.get_ic_analysis()
            
            # Store in history
            self.selection_history.append(result)
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Factor selection failed: {e}")
    
    def multi_method_selection(self, factors: pd.DataFrame, returns: pd.Series,
                             methods: List[SelectionMethod], 
                             configs: List[SelectionConfig]) -> Dict[SelectionMethod, SelectionResult]:
        """Perform factor selection using multiple methods"""
        
        if len(methods) != len(configs):
            raise ValueError("Number of methods must match number of configs")
        
        results = {}
        
        for method, config in zip(methods, configs):
            try:
                result = self.select_factors(factors, returns, method, config)
                results[method] = result
            except Exception as e:
                warnings.warn(f"Selection method {method} failed: {e}")
                continue
        
        return results
    
    def ensemble_selection(self, factors: pd.DataFrame, returns: pd.Series,
                         methods: List[SelectionMethod], configs: List[SelectionConfig],
                         voting_threshold: float = 0.5) -> List[str]:
        """Ensemble factor selection using voting"""
        
        # Get results from multiple methods
        results = self.multi_method_selection(factors, returns, methods, configs)
        
        if not results:
            return []
        
        # Count votes for each factor
        factor_votes = defaultdict(int)
        total_methods = len(results)
        
        for method, result in results.items():
            for factor_name in result.selected_factors:
                factor_votes[factor_name] += 1
        
        # Select factors that meet voting threshold
        min_votes = int(total_methods * voting_threshold)
        selected_factors = [
            factor_name for factor_name, votes in factor_votes.items()
            if votes >= min_votes
        ]
        
        return selected_factors
    
    def _calculate_performance_metrics(self, factors: pd.DataFrame, returns: pd.Series,
                                     selected_factors: List[str]) -> Dict[str, Any]:
        """Calculate performance metrics for selected factors"""
        
        if not selected_factors:
            return {"error": "No factors selected"}
        
        try:
            # Select factor subset
            selected_factor_data = factors[selected_factors]
            
            # Align data
            common_index = selected_factor_data.index.intersection(returns.index)
            X = selected_factor_data.loc[common_index].fillna(method='ffill').fillna(0)
            y = returns.loc[common_index].fillna(0)
            
            if len(X) < 20:
                return {"error": "Insufficient data for performance calculation"}
            
            # Split into train/test
            split_point = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
            y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
            
            # Fit simple linear model
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Calculate metrics
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test) if len(X_test) > 0 else None
            
            # Calculate factor correlations
            factor_corr = X.corr()
            avg_correlation = factor_corr.abs().mean().mean()
            max_correlation = factor_corr.abs().max().max()
            
            return {
                "n_selected": len(selected_factors),
                "train_r2": train_score,
                "test_r2": test_score,
                "avg_factor_correlation": avg_correlation,
                "max_factor_correlation": max_correlation,
                "data_points": len(X)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def get_selection_summary(self) -> pd.DataFrame:
        """Get summary of all selection results"""
        
        if not self.selection_history:
            return pd.DataFrame()
        
        summary_data = []
        
        for result in self.selection_history:
            summary_data.append({
                'method': result.method.value,
                'criteria': result.criteria.value,
                'n_selected': len(result.selected_factors),
                'total_considered': result.total_factors_considered,
                'selection_time': result.selection_time_seconds,
                'train_r2': result.selection_metrics.get('train_r2'),
                'test_r2': result.selection_metrics.get('test_r2'),
                'avg_correlation': result.selection_metrics.get('avg_factor_correlation'),
                'selection_date': result.selection_date
            })
        
        return pd.DataFrame(summary_data)
    
    def compare_methods(self, factors: pd.DataFrame, returns: pd.Series) -> pd.DataFrame:
        """Compare different selection methods"""
        
        methods = [
            SelectionMethod.CORRELATION,
            SelectionMethod.IC_ANALYSIS,
            SelectionMethod.LASSO,
            SelectionMethod.RANDOM_FOREST,
            SelectionMethod.STABILITY_SELECTION
        ]
        
        configs = [SelectionConfig(method=method, k=10) for method in methods]
        
        results = self.multi_method_selection(factors, returns, methods, configs)
        
        comparison_data = []
        
        for method, result in results.items():
            comparison_data.append({
                'method': method.value,
                'n_selected': len(result.selected_factors),
                'train_r2': result.selection_metrics.get('train_r2', 0),
                'test_r2': result.selection_metrics.get('test_r2', 0),
                'avg_correlation': result.selection_metrics.get('avg_factor_correlation', 0),
                'selection_time': result.selection_time_seconds,
                'top_factors': ', '.join(result.selected_factors[:5])
            })
        
        return pd.DataFrame(comparison_data)
