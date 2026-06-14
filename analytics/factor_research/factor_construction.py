"""
Factor Construction Framework
Advanced factor engineering and construction for quantitative research
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
import queue
import time
from collections import deque, defaultdict
import warnings
import math
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import concurrent.futures


class FactorType(Enum):
    """Types of factors"""
    FUNDAMENTAL = "FUNDAMENTAL"         # Financial statement based
    TECHNICAL = "TECHNICAL"             # Price/volume based  
    MOMENTUM = "MOMENTUM"               # Price momentum
    REVERSAL = "REVERSAL"               # Mean reversion
    VOLATILITY = "VOLATILITY"           # Volatility based
    QUALITY = "QUALITY"                 # Quality metrics
    VALUE = "VALUE"                     # Valuation metrics
    GROWTH = "GROWTH"                   # Growth metrics
    SENTIMENT = "SENTIMENT"             # Market sentiment
    MACRO = "MACRO"                     # Macroeconomic
    ALTERNATIVE = "ALTERNATIVE"         # Alternative data
    COMPOSITE = "COMPOSITE"             # Combined factors


class NormalizationMethod(Enum):
    """Normalization methods"""
    ZSCORE = "ZSCORE"                   # Z-score normalization
    RANK = "RANK"                       # Rank normalization
    QUANTILE = "QUANTILE"               # Quantile normalization
    ROBUST = "ROBUST"                   # Robust scaling
    MINMAX = "MINMAX"                   # Min-max scaling
    WINSORIZE = "WINSORIZE"             # Winsorization
    NONE = "NONE"                       # No normalization


class OutlierMethod(Enum):
    """Outlier handling methods"""
    WINSORIZE = "WINSORIZE"             # Winsorize at percentiles
    CLIP = "CLIP"                       # Clip values
    REMOVE = "REMOVE"                   # Remove outliers
    TRANSFORM = "TRANSFORM"             # Transform (log, sqrt)
    NONE = "NONE"                       # No outlier handling


@dataclass
class FactorSpec:
    """Factor specification and metadata"""
    factor_name: str
    factor_type: FactorType
    description: str = ""
    
    # Construction parameters
    lookback_periods: int = 20
    min_periods: int = 10
    frequency: str = "daily"            # daily, weekly, monthly
    
    # Data requirements
    required_fields: List[str] = field(default_factory=list)
    universe_filter: Optional[str] = None
    
    # Processing parameters
    normalization: NormalizationMethod = NormalizationMethod.ZSCORE
    outlier_handling: OutlierMethod = OutlierMethod.WINSORIZE
    winsorize_limits: Tuple[float, float] = (0.05, 0.95)
    
    # Quality metrics
    min_coverage: float = 0.80          # Minimum data coverage
    max_missing_consecutive: int = 5    # Max consecutive missing values
    
    # Metadata
    created_date: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    version: str = "1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'factor_name': self.factor_name,
            'factor_type': self.factor_type.value,
            'description': self.description,
            'lookback_periods': self.lookback_periods,
            'min_periods': self.min_periods,
            'frequency': self.frequency,
            'required_fields': self.required_fields,
            'normalization': self.normalization.value,
            'outlier_handling': self.outlier_handling.value,
            'winsorize_limits': self.winsorize_limits,
            'min_coverage': self.min_coverage,
            'created_date': self.created_date.isoformat(),
            'version': self.version
        }


@dataclass
class FactorQuality:
    """Factor quality metrics"""
    factor_name: str
    
    # Coverage metrics
    coverage: float = 0.0               # Data coverage percentage
    missing_pct: float = 0.0            # Missing data percentage
    zero_pct: float = 0.0               # Zero values percentage
    
    # Distribution metrics
    mean: float = 0.0
    std: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    
    # Stability metrics
    autocorrelation: float = 0.0        # First-order autocorrelation
    volatility: float = 0.0             # Factor volatility
    turnover: float = 0.0               # Factor turnover
    
    # Cross-sectional metrics
    cross_sectional_dispersion: float = 0.0
    outlier_percentage: float = 0.0
    
    # Quality score
    overall_quality_score: float = 0.0
    
    # Timestamp
    calculation_date: datetime = field(default_factory=datetime.now)


class BaseFactor(ABC):
    """Base class for factor construction"""
    
    def __init__(self, spec: FactorSpec):
        self.spec = spec
        self.data_cache: Dict[str, pd.DataFrame] = {}
        self.factor_values: Optional[pd.DataFrame] = None
        self.quality_metrics: Optional[FactorQuality] = None
        
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate factor values"""
        pass
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess input data"""
        
        # Check required fields
        missing_fields = [field for field in self.spec.required_fields if field not in data.columns]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
        
        # Handle missing values
        data = self._handle_missing_values(data)
        
        # Handle outliers
        data = self._handle_outliers(data)
        
        return data
    
    def postprocess_factor(self, factor_values: pd.DataFrame) -> pd.DataFrame:
        """Postprocess factor values"""
        
        # Apply normalization
        if self.spec.normalization != NormalizationMethod.NONE:
            factor_values = self._normalize_factor(factor_values)
        
        # Final quality check
        factor_values = self._final_quality_check(factor_values)
        
        return factor_values
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in input data"""
        
        # Forward fill with limit
        data = data.fillna(method='ffill', limit=self.spec.max_missing_consecutive)
        
        # Check coverage after handling
        coverage = 1 - data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
        
        if coverage < self.spec.min_coverage:
            warnings.warn(f"Data coverage {coverage:.2%} below minimum {self.spec.min_coverage:.2%}")
        
        return data
    
    def _handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers in input data"""
        
        if self.spec.outlier_handling == OutlierMethod.WINSORIZE:
            lower_pct, upper_pct = self.spec.winsorize_limits
            
            for col in data.select_dtypes(include=[np.number]).columns:
                lower_bound = data[col].quantile(lower_pct)
                upper_bound = data[col].quantile(upper_pct)
                data[col] = data[col].clip(lower_bound, upper_bound)
        
        elif self.spec.outlier_handling == OutlierMethod.CLIP:
            # Clip at 3 standard deviations
            for col in data.select_dtypes(include=[np.number]).columns:
                mean_val = data[col].mean()
                std_val = data[col].std()
                lower_bound = mean_val - 3 * std_val
                upper_bound = mean_val + 3 * std_val
                data[col] = data[col].clip(lower_bound, upper_bound)
        
        return data
    
    def _normalize_factor(self, factor_values: pd.DataFrame) -> pd.DataFrame:
        """Normalize factor values"""
        
        if self.spec.normalization == NormalizationMethod.ZSCORE:
            # Cross-sectional z-score
            factor_values = factor_values.apply(lambda x: (x - x.mean()) / x.std(), axis=1)
        
        elif self.spec.normalization == NormalizationMethod.RANK:
            # Cross-sectional rank
            factor_values = factor_values.rank(axis=1, pct=True)
        
        elif self.spec.normalization == NormalizationMethod.QUANTILE:
            # Quantile normalization
            from scipy.stats import norm
            factor_values = factor_values.rank(axis=1, pct=True).apply(lambda x: norm.ppf(x))
        
        elif self.spec.normalization == NormalizationMethod.ROBUST:
            # Robust scaling (median and MAD)
            factor_values = factor_values.apply(
                lambda x: (x - x.median()) / x.mad(), axis=1
            )
        
        elif self.spec.normalization == NormalizationMethod.MINMAX:
            # Min-max scaling
            factor_values = factor_values.apply(
                lambda x: (x - x.min()) / (x.max() - x.min()), axis=1
            )
        
        return factor_values
    
    def _final_quality_check(self, factor_values: pd.DataFrame) -> pd.DataFrame:
        """Final quality check and cleanup"""
        
        # Remove rows/columns with all NaN
        factor_values = factor_values.dropna(how='all', axis=0)
        factor_values = factor_values.dropna(how='all', axis=1)
        
        # Replace inf with NaN
        factor_values = factor_values.replace([np.inf, -np.inf], np.nan)
        
        return factor_values
    
    def calculate_quality_metrics(self, factor_values: pd.DataFrame) -> FactorQuality:
        """Calculate factor quality metrics"""
        
        if factor_values.empty:
            return FactorQuality(factor_name=self.spec.factor_name)
        
        # Flatten factor values for analysis
        flat_values = factor_values.values.flatten()
        flat_values = flat_values[~np.isnan(flat_values)]
        
        if len(flat_values) == 0:
            return FactorQuality(factor_name=self.spec.factor_name)
        
        # Coverage metrics
        total_cells = factor_values.shape[0] * factor_values.shape[1]
        missing_cells = factor_values.isnull().sum().sum()
        coverage = 1 - (missing_cells / total_cells)
        missing_pct = missing_cells / total_cells
        zero_pct = (flat_values == 0).sum() / len(flat_values)
        
        # Distribution metrics
        mean_val = np.mean(flat_values)
        std_val = np.std(flat_values)
        skew_val = stats.skew(flat_values)
        kurt_val = stats.kurtosis(flat_values)
        
        # Stability metrics
        # Time series autocorrelation (if enough data)
        autocorr = 0.0
        if factor_values.shape[0] > 1:
            try:
                # Average autocorrelation across assets
                autocorrs = []
                for col in factor_values.columns:
                    series = factor_values[col].dropna()
                    if len(series) > 1:
                        autocorrs.append(series.autocorr(lag=1))
                
                if autocorrs:
                    autocorr = np.nanmean(autocorrs)
            except:
                autocorr = 0.0
        
        # Factor volatility - standard deviation of factor means over time
        factor_means = factor_values.mean(axis=1, skipna=True)
        volatility = factor_means.std() if len(factor_means) > 1 else 0.0
        
        # Cross-sectional dispersion
        cross_dispersions = factor_values.std(axis=1, skipna=True)
        cross_sectional_dispersion = cross_dispersions.mean()
        
        # Outlier percentage (beyond 3 standard deviations)
        if std_val > 0:
            outliers = np.abs(flat_values - mean_val) > 3 * std_val
            outlier_percentage = outliers.sum() / len(flat_values)
        else:
            outlier_percentage = 0.0
        
        # Overall quality score (0-100)
        quality_score = self._calculate_quality_score(
            coverage, missing_pct, std_val, outlier_percentage, autocorr
        )
        
        return FactorQuality(
            factor_name=self.spec.factor_name,
            coverage=coverage,
            missing_pct=missing_pct,
            zero_pct=zero_pct,
            mean=mean_val,
            std=std_val,
            skewness=skew_val,
            kurtosis=kurt_val,
            autocorrelation=autocorr,
            volatility=volatility,
            cross_sectional_dispersion=cross_sectional_dispersion,
            outlier_percentage=outlier_percentage,
            overall_quality_score=quality_score
        )
    
    def _calculate_quality_score(self, coverage: float, missing_pct: float, 
                                std_val: float, outlier_pct: float, autocorr: float) -> float:
        """Calculate overall quality score"""
        
        score = 100.0
        
        # Penalize low coverage
        if coverage < 0.95:
            score -= (0.95 - coverage) * 100
        
        # Penalize high missing percentage
        score -= missing_pct * 50
        
        # Penalize zero standard deviation (no signal)
        if std_val == 0:
            score -= 50
        
        # Penalize high outlier percentage
        score -= outlier_pct * 30
        
        # Penalize very high autocorrelation (stale factor)
        if abs(autocorr) > 0.8:
            score -= (abs(autocorr) - 0.8) * 100
        
        return max(0.0, min(100.0, score))


class MomentumFactor(BaseFactor):
    """Price momentum factor"""
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum factor"""
        
        if 'close' not in data.columns:
            raise ValueError("Close price required for momentum factor")
        
        # Calculate returns over lookback period
        returns = data['close'].pct_change(self.spec.lookback_periods)
        
        # Create factor DataFrame
        factor_values = pd.DataFrame(
            returns,
            columns=[self.spec.factor_name]
        )
        
        return factor_values


class ReversalFactor(BaseFactor):
    """Short-term reversal factor"""
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate reversal factor"""
        
        if 'close' not in data.columns:
            raise ValueError("Close price required for reversal factor")
        
        # Calculate short-term returns (negative for reversal)
        short_returns = data['close'].pct_change(self.spec.lookback_periods)
        reversal_factor = -short_returns  # Negative for reversal
        
        factor_values = pd.DataFrame(
            reversal_factor,
            columns=[self.spec.factor_name]
        )
        
        return factor_values


class VolatilityFactor(BaseFactor):
    """Volatility factor"""
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility factor"""
        
        if 'close' not in data.columns:
            raise ValueError("Close price required for volatility factor")
        
        # Calculate rolling volatility
        returns = data['close'].pct_change()
        volatility = returns.rolling(
            window=self.spec.lookback_periods,
            min_periods=self.spec.min_periods
        ).std()
        
        factor_values = pd.DataFrame(
            volatility,
            columns=[self.spec.factor_name]
        )
        
        return factor_values


class ValueFactor(BaseFactor):
    """Value factor (requires fundamental data)"""
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate value factor"""
        
        required_cols = ['close', 'book_value', 'earnings']
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns for value factor: {missing_cols}")
        
        # Calculate P/B and P/E ratios
        pb_ratio = data['close'] / data['book_value']
        pe_ratio = data['close'] / data['earnings']
        
        # Combine into value score (lower is better for value)
        value_score = -(pb_ratio.rank(pct=True) + pe_ratio.rank(pct=True)) / 2
        
        factor_values = pd.DataFrame(
            value_score,
            columns=[self.spec.factor_name]
        )
        
        return factor_values


class QualityFactor(BaseFactor):
    """Quality factor (requires fundamental data)"""
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate quality factor"""
        
        required_cols = ['roe', 'debt_to_equity', 'earnings_growth']
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns for quality factor: {missing_cols}")
        
        # Normalize individual quality metrics
        roe_score = data['roe'].rank(pct=True)
        debt_score = (1 - data['debt_to_equity'].rank(pct=True))  # Lower debt is better
        growth_score = data['earnings_growth'].rank(pct=True)
        
        # Combine quality metrics
        quality_score = (roe_score + debt_score + growth_score) / 3
        
        factor_values = pd.DataFrame(
            quality_score,
            columns=[self.spec.factor_name]
        )
        
        return factor_values


class TechnicalFactor(BaseFactor):
    """Technical analysis based factor"""
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical factor (RSI-based)"""
        
        if 'close' not in data.columns:
            raise ValueError("Close price required for technical factor")
        
        # Calculate RSI
        rsi = self._calculate_rsi(data['close'], self.spec.lookback_periods)
        
        # Convert to factor score (mean reversion signal)
        technical_score = (rsi - 50) / 50  # Normalize around 0
        
        factor_values = pd.DataFrame(
            technical_score,
            columns=[self.spec.factor_name]
        )
        
        return factor_values
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi


class CompositeFactor(BaseFactor):
    """Composite factor combining multiple factors"""
    
    def __init__(self, spec: FactorSpec, component_factors: List[BaseFactor], 
                 weights: Optional[List[float]] = None):
        super().__init__(spec)
        self.component_factors = component_factors
        self.weights = weights if weights else [1.0] * len(component_factors)
        
        if len(self.weights) != len(self.component_factors):
            raise ValueError("Number of weights must match number of component factors")
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate composite factor"""
        
        factor_values_list = []
        
        # Calculate each component factor
        for factor in self.component_factors:
            try:
                component_values = factor.calculate(data)
                factor_values_list.append(component_values.iloc[:, 0])  # First column
            except Exception as e:
                warnings.warn(f"Failed to calculate component factor {factor.spec.factor_name}: {e}")
                continue
        
        if not factor_values_list:
            raise ValueError("No component factors could be calculated")
        
        # Combine factors with weights
        composite_values = pd.Series(0.0, index=factor_values_list[0].index)
        total_weight = 0.0
        
        for i, (factor_values, weight) in enumerate(zip(factor_values_list, self.weights)):
            if not factor_values.empty:
                composite_values += factor_values.fillna(0) * weight
                total_weight += weight
        
        if total_weight > 0:
            composite_values /= total_weight
        
        factor_df = pd.DataFrame(
            composite_values,
            columns=[self.spec.factor_name]
        )
        
        return factor_df


class FactorConstructor:
    """
    Main factor construction engine
    """
    
    def __init__(self):
        self.factor_registry: Dict[str, BaseFactor] = {}
        self.factor_cache: Dict[str, pd.DataFrame] = {}
        self.quality_cache: Dict[str, FactorQuality] = {}
        
        # Threading for parallel construction
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        
    def register_factor(self, factor: BaseFactor):
        """Register a factor for construction"""
        self.factor_registry[factor.spec.factor_name] = factor
    
    def construct_factor(self, factor_name: str, data: pd.DataFrame, 
                        force_recalculate: bool = False) -> pd.DataFrame:
        """Construct a single factor"""
        
        if factor_name not in self.factor_registry:
            raise ValueError(f"Factor {factor_name} not registered")
        
        # Check cache
        if not force_recalculate and factor_name in self.factor_cache:
            return self.factor_cache[factor_name].copy()
        
        factor = self.factor_registry[factor_name]
        
        try:
            # Preprocess data
            processed_data = factor.preprocess_data(data)
            
            # Calculate factor
            factor_values = factor.calculate(processed_data)
            
            # Postprocess factor
            factor_values = factor.postprocess_factor(factor_values)
            
            # Calculate quality metrics
            quality = factor.calculate_quality_metrics(factor_values)
            self.quality_cache[factor_name] = quality
            
            # Cache results
            self.factor_cache[factor_name] = factor_values
            
            return factor_values.copy()
            
        except Exception as e:
            raise RuntimeError(f"Failed to construct factor {factor_name}: {e}")
    
    def construct_multiple_factors(self, factor_names: List[str], data: pd.DataFrame,
                                 parallel: bool = True) -> Dict[str, pd.DataFrame]:
        """Construct multiple factors"""
        
        if parallel:
            return self._construct_parallel(factor_names, data)
        else:
            return self._construct_sequential(factor_names, data)
    
    def _construct_sequential(self, factor_names: List[str], 
                            data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Construct factors sequentially"""
        
        results = {}
        
        for factor_name in factor_names:
            try:
                factor_values = self.construct_factor(factor_name, data)
                results[factor_name] = factor_values
            except Exception as e:
                warnings.warn(f"Failed to construct factor {factor_name}: {e}")
                continue
        
        return results
    
    def _construct_parallel(self, factor_names: List[str], 
                          data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Construct factors in parallel"""
        
        future_to_factor = {
            self.executor.submit(self.construct_factor, name, data): name
            for name in factor_names
        }
        
        results = {}
        
        for future in concurrent.futures.as_completed(future_to_factor):
            factor_name = future_to_factor[future]
            try:
                factor_values = future.result()
                results[factor_name] = factor_values
            except Exception as e:
                warnings.warn(f"Failed to construct factor {factor_name}: {e}")
                continue
        
        return results
    
    def get_factor_quality(self, factor_name: str) -> Optional[FactorQuality]:
        """Get factor quality metrics"""
        return self.quality_cache.get(factor_name)
    
    def get_quality_report(self) -> pd.DataFrame:
        """Get quality report for all factors"""
        
        if not self.quality_cache:
            return pd.DataFrame()
        
        quality_data = []
        
        for factor_name, quality in self.quality_cache.items():
            quality_data.append({
                'factor_name': quality.factor_name,
                'coverage': quality.coverage,
                'missing_pct': quality.missing_pct,
                'mean': quality.mean,
                'std': quality.std,
                'skewness': quality.skewness,
                'kurtosis': quality.kurtosis,
                'autocorrelation': quality.autocorrelation,
                'volatility': quality.volatility,
                'outlier_percentage': quality.outlier_percentage,
                'quality_score': quality.overall_quality_score
            })
        
        return pd.DataFrame(quality_data)
    
    def create_factor_universe(self, data: pd.DataFrame, 
                             factor_names: Optional[List[str]] = None) -> pd.DataFrame:
        """Create factor universe matrix"""
        
        if factor_names is None:
            factor_names = list(self.factor_registry.keys())
        
        # Construct all factors
        factor_results = self.construct_multiple_factors(factor_names, data)
        
        if not factor_results:
            return pd.DataFrame()
        
        # Combine into single DataFrame
        factor_universe = pd.DataFrame()
        
        for factor_name, factor_values in factor_results.items():
            if not factor_values.empty:
                factor_universe[factor_name] = factor_values.iloc[:, 0]
        
        return factor_universe
    
    def save_factor_specs(self, filename: str):
        """Save factor specifications to file"""
        
        specs = {}
        for name, factor in self.factor_registry.items():
            specs[name] = factor.spec.to_dict()
        
        import json
        with open(filename, 'w') as f:
            json.dump(specs, f, indent=2, default=str)
    
    def clear_cache(self):
        """Clear factor cache"""
        self.factor_cache.clear()
        self.quality_cache.clear()
