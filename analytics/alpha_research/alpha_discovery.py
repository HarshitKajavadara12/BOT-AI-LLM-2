"""
Alpha Discovery Framework
========================

Advanced alpha discovery system for systematic identification of profitable
trading opportunities and investment signals across multiple asset classes.

Features:
- Multi-factor alpha discovery using various methodologies
- Cross-sectional and time-series alpha signals
- Alternative data integration for alpha generation
- Machine learning-based alpha mining
- Technical and fundamental alpha strategies
- Market microstructure alpha signals
- Regime-dependent alpha discovery
- Alpha signal validation and screening

Author: Quantum Forge Analytics Team
Date: November 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
from scipy import stats, optimize
from scipy.stats import spearmanr as spearman, pearsonr
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AlphaSignal:
    """Container for alpha signal characteristics and metrics."""
    signal_name: str
    signal_type: str  # 'cross_sectional', 'time_series', 'hybrid'
    alpha_values: pd.Series
    information_coefficient: float
    sharpe_ratio: float
    turnover: float
    capacity: float
    decay_half_life: float
    correlation_with_returns: float
    sector_neutrality: float
    risk_adjusted_ic: float
    significance_pvalue: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'signal_name': self.signal_name,
            'signal_type': self.signal_type,
            'information_coefficient': self.information_coefficient,
            'sharpe_ratio': self.sharpe_ratio,
            'turnover': self.turnover,
            'capacity': self.capacity,
            'decay_half_life': self.decay_half_life,
            'correlation_with_returns': self.correlation_with_returns,
            'sector_neutrality': self.sector_neutrality,
            'risk_adjusted_ic': self.risk_adjusted_ic,
            'significance_pvalue': self.significance_pvalue
        }

@dataclass
class AlphaDiscoveryResults:
    """Container for comprehensive alpha discovery results."""
    discovered_signals: List[AlphaSignal]
    signal_correlations: pd.DataFrame
    top_signals: List[AlphaSignal]
    signal_performance_summary: pd.DataFrame
    discovery_methodology: str
    discovery_date: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary format."""
        return {
            'num_signals_discovered': len(self.discovered_signals),
            'num_top_signals': len(self.top_signals),
            'discovery_methodology': self.discovery_methodology,
            'discovery_date': self.discovery_date,
            'top_signals': [signal.to_dict() for signal in self.top_signals],
            'performance_summary': self.signal_performance_summary.to_dict() if not self.signal_performance_summary.empty else {}
        }

class TechnicalAlphaGenerator:
    """
    Technical alpha signal generator using various technical indicators.
    
    Generates alpha signals based on price, volume, and market microstructure
    patterns using both traditional and advanced technical analysis methods.
    """
    
    def __init__(self):
        """Initialize technical alpha generator."""
        self.generated_signals = {}
        
    def generate_momentum_signals(self, price_data: pd.DataFrame,
                                volume_data: Optional[pd.DataFrame] = None,
                                lookback_periods: List[int] = [5, 10, 20, 60]) -> Dict[str, pd.DataFrame]:
        """
        Generate momentum-based alpha signals.
        
        Parameters:
        -----------
        price_data : pd.DataFrame
            Price data (assets as columns, dates as index)
        volume_data : Optional[pd.DataFrame]
            Volume data for volume-weighted signals
        lookback_periods : List[int]
            Lookback periods for momentum calculation
            
        Returns:
        --------
        Dict[str, pd.DataFrame]
            Dictionary of momentum alpha signals
        """
        signals = {}
        
        try:
            # Price momentum signals
            for period in lookback_periods:
                # Simple price momentum
                momentum = price_data.pct_change(period)
                signals[f'price_momentum_{period}d'] = momentum
                
                # Risk-adjusted momentum
                vol = price_data.pct_change().rolling(period).std()
                risk_adj_momentum = momentum / (vol * np.sqrt(period))
                signals[f'risk_adj_momentum_{period}d'] = risk_adj_momentum
                
                # Momentum acceleration
                if period >= 10:
                    accel = momentum - price_data.pct_change(period // 2)
                    signals[f'momentum_acceleration_{period}d'] = accel
            
            # Volume-weighted momentum (if volume data available)
            if volume_data is not None:
                returns = price_data.pct_change()
                for period in lookback_periods:
                    # Volume-weighted average price momentum
                    vwap_momentum = self._calculate_vwap_momentum(price_data, volume_data, period)
                    signals[f'vwap_momentum_{period}d'] = vwap_momentum
                    
                    # Volume-price trend
                    volume_trend = volume_data.pct_change(period)
                    price_trend = price_data.pct_change(period)
                    vp_signal = price_trend * np.sign(volume_trend)
                    signals[f'volume_price_trend_{period}d'] = vp_signal
            
            # Cross-sectional momentum (relative to universe)
            for period in lookback_periods:
                momentum = price_data.pct_change(period)
                cross_sectional_momentum = momentum.subtract(momentum.mean(axis=1), axis=0)
                signals[f'cross_sectional_momentum_{period}d'] = cross_sectional_momentum
            
            self.generated_signals.update(signals)
            return signals
            
        except Exception as e:
            logger.error(f"Error generating momentum signals: {str(e)}")
            raise
    
    def generate_mean_reversion_signals(self, price_data: pd.DataFrame,
                                      lookback_periods: List[int] = [5, 10, 20]) -> Dict[str, pd.DataFrame]:
        """
        Generate mean reversion alpha signals.
        
        Parameters:
        -----------
        price_data : pd.DataFrame
            Price data
        lookback_periods : List[int]
            Lookback periods for mean reversion
            
        Returns:
        --------
        Dict[str, pd.DataFrame]
            Dictionary of mean reversion alpha signals
        """
        signals = {}
        
        try:
            returns = price_data.pct_change()
            
            for period in lookback_periods:
                # Simple mean reversion
                rolling_mean = returns.rolling(period).mean()
                mean_reversion = -rolling_mean  # Negative of recent average return
                signals[f'mean_reversion_{period}d'] = mean_reversion
                
                # Bollinger Band mean reversion
                rolling_std = returns.rolling(period).std()
                z_score = (returns - rolling_mean) / rolling_std
                bollinger_signal = -z_score  # Fade extreme moves
                signals[f'bollinger_reversion_{period}d'] = bollinger_signal
                
                # RSI-based mean reversion
                rsi = self._calculate_rsi(price_data, period)
                rsi_signal = -(rsi - 50) / 50  # Normalize and invert
                signals[f'rsi_reversion_{period}d'] = rsi_signal
            
            # Overnight gap mean reversion
            if len(price_data) > 1:
                overnight_returns = price_data.iloc[1:] / price_data.iloc[:-1].values - 1
                overnight_returns.index = price_data.index[1:]
                gap_reversion = -overnight_returns  # Fade gaps
                signals['gap_reversion'] = gap_reversion
            
            self.generated_signals.update(signals)
            return signals
            
        except Exception as e:
            logger.error(f"Error generating mean reversion signals: {str(e)}")
            raise
    
    def generate_volatility_signals(self, price_data: pd.DataFrame,
                                  lookback_periods: List[int] = [10, 20, 60]) -> Dict[str, pd.DataFrame]:
        """
        Generate volatility-based alpha signals.
        
        Parameters:
        -----------
        price_data : pd.DataFrame
            Price data
        lookback_periods : List[int]
            Lookback periods for volatility calculation
            
        Returns:
        --------
        Dict[str, pd.DataFrame]
            Dictionary of volatility alpha signals
        """
        signals = {}
        
        try:
            returns = price_data.pct_change()
            
            for period in lookback_periods:
                # Volatility trend
                vol = returns.rolling(period).std()
                vol_trend = vol.pct_change(period // 2)
                signals[f'volatility_trend_{period}d'] = vol_trend
                
                # Relative volatility
                vol_rank = vol.rolling(period * 2).rank(pct=True)
                signals[f'relative_volatility_{period}d'] = vol_rank - 0.5
                
                # Volatility mean reversion
                vol_zscore = (vol - vol.rolling(period * 2).mean()) / vol.rolling(period * 2).std()
                vol_mean_reversion = -vol_zscore
                signals[f'volatility_mean_reversion_{period}d'] = vol_mean_reversion
            
            # Volatility skew signal
            skew_period = 60
            rolling_skew = returns.rolling(skew_period).skew()
            skew_signal = -rolling_skew  # Fade negative skew
            signals[f'volatility_skew_{skew_period}d'] = skew_signal
            
            self.generated_signals.update(signals)
            return signals
            
        except Exception as e:
            logger.error(f"Error generating volatility signals: {str(e)}")
            raise
    
    def _calculate_vwap_momentum(self, price_data: pd.DataFrame,
                               volume_data: pd.DataFrame, period: int) -> pd.DataFrame:
        """Calculate volume-weighted average price momentum."""
        try:
            # Calculate VWAP
            typical_price = price_data  # Simplified - would use (H+L+C)/3 with OHLC data
            vwap = (typical_price * volume_data).rolling(period).sum() / volume_data.rolling(period).sum()
            
            # VWAP momentum
            vwap_momentum = typical_price / vwap - 1
            return vwap_momentum
            
        except Exception as e:
            logger.warning(f"Error calculating VWAP momentum: {str(e)}")
            return pd.DataFrame(0, index=price_data.index, columns=price_data.columns)
    
    def _calculate_rsi(self, price_data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Relative Strength Index."""
        try:
            returns = price_data.pct_change()
            gains = returns.where(returns > 0, 0)
            losses = -returns.where(returns < 0, 0)
            
            avg_gains = gains.rolling(period).mean()
            avg_losses = losses.rolling(period).mean()
            
            rs = avg_gains / avg_losses
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception as e:
            logger.warning(f"Error calculating RSI: {str(e)}")
            return pd.DataFrame(50, index=price_data.index, columns=price_data.columns)

class FundamentalAlphaGenerator:
    """
    Fundamental alpha signal generator using financial statement data.
    
    Generates alpha signals based on fundamental ratios, earnings quality,
    and financial statement analysis.
    """
    
    def __init__(self):
        """Initialize fundamental alpha generator."""
        self.generated_signals = {}
        
    def generate_value_signals(self, fundamental_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Generate value-based alpha signals.
        
        Parameters:
        -----------
        fundamental_data : pd.DataFrame
            Fundamental data (P/E, P/B, EV/EBITDA, etc.)
            
        Returns:
        --------
        Dict[str, pd.DataFrame]
            Dictionary of value alpha signals
        """
        signals = {}
        
        try:
            # Traditional value signals (inverted ratios)
            if 'PE_RATIO' in fundamental_data.columns:
                pe_signal = -fundamental_data['PE_RATIO'].rank(pct=True) + 0.5
                signals['pe_value_signal'] = pe_signal
            
            if 'PB_RATIO' in fundamental_data.columns:
                pb_signal = -fundamental_data['PB_RATIO'].rank(pct=True) + 0.5
                signals['pb_value_signal'] = pb_signal
            
            if 'EV_EBITDA' in fundamental_data.columns:
                ev_ebitda_signal = -fundamental_data['EV_EBITDA'].rank(pct=True) + 0.5
                signals['ev_ebitda_value_signal'] = ev_ebitda_signal
            
            # Composite value score
            value_columns = [col for col in ['PE_RATIO', 'PB_RATIO', 'EV_EBITDA'] 
                           if col in fundamental_data.columns]
            
            if value_columns:
                # Rank and average multiple value metrics
                value_ranks = fundamental_data[value_columns].rank(pct=True)
                composite_value = -value_ranks.mean(axis=1) + 0.5
                signals['composite_value_signal'] = composite_value
            
            self.generated_signals.update(signals)
            return signals
            
        except Exception as e:
            logger.error(f"Error generating value signals: {str(e)}")
            raise
    
    def generate_quality_signals(self, fundamental_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Generate quality-based alpha signals.
        
        Parameters:
        -----------
        fundamental_data : pd.DataFrame
            Fundamental data including quality metrics
            
        Returns:
        --------
        Dict[str, pd.DataFrame]
            Dictionary of quality alpha signals
        """
        signals = {}
        
        try:
            # ROE signal
            if 'ROE' in fundamental_data.columns:
                roe_signal = fundamental_data['ROE'].rank(pct=True) - 0.5
                signals['roe_quality_signal'] = roe_signal
            
            # ROA signal
            if 'ROA' in fundamental_data.columns:
                roa_signal = fundamental_data['ROA'].rank(pct=True) - 0.5
                signals['roa_quality_signal'] = roa_signal
            
            # Debt-to-equity signal (lower is better)
            if 'DEBT_TO_EQUITY' in fundamental_data.columns:
                debt_signal = -fundamental_data['DEBT_TO_EQUITY'].rank(pct=True) + 0.5
                signals['debt_quality_signal'] = debt_signal
            
            # Current ratio signal
            if 'CURRENT_RATIO' in fundamental_data.columns:
                # Optimal current ratio around 1.5-2.0
                optimal_current_ratio = 1.75
                current_ratio_signal = -np.abs(fundamental_data['CURRENT_RATIO'] - optimal_current_ratio)
                current_ratio_signal = current_ratio_signal.rank(pct=True) - 0.5
                signals['current_ratio_quality_signal'] = current_ratio_signal
            
            # Composite quality score
            quality_columns = [col for col in ['ROE', 'ROA'] if col in fundamental_data.columns]
            negative_quality_columns = [col for col in ['DEBT_TO_EQUITY'] if col in fundamental_data.columns]
            
            if quality_columns or negative_quality_columns:
                quality_scores = []
                
                # Positive quality metrics
                if quality_columns:
                    positive_ranks = fundamental_data[quality_columns].rank(pct=True)
                    quality_scores.append(positive_ranks.mean(axis=1))
                
                # Negative quality metrics (inverted)
                if negative_quality_columns:
                    negative_ranks = fundamental_data[negative_quality_columns].rank(pct=True)
                    quality_scores.append(-negative_ranks.mean(axis=1) + 1)
                
                if quality_scores:
                    composite_quality = pd.concat(quality_scores, axis=1).mean(axis=1) - 0.5
                    signals['composite_quality_signal'] = composite_quality
            
            self.generated_signals.update(signals)
            return signals
            
        except Exception as e:
            logger.error(f"Error generating quality signals: {str(e)}")
            raise
    
    def generate_growth_signals(self, fundamental_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Generate growth-based alpha signals.
        
        Parameters:
        -----------
        fundamental_data : pd.DataFrame
            Fundamental data including growth metrics
            
        Returns:
        --------
        Dict[str, pd.DataFrame]
            Dictionary of growth alpha signals
        """
        signals = {}
        
        try:
            # Revenue growth signal
            if 'REVENUE_GROWTH' in fundamental_data.columns:
                revenue_growth_signal = fundamental_data['REVENUE_GROWTH'].rank(pct=True) - 0.5
                signals['revenue_growth_signal'] = revenue_growth_signal
            
            # Earnings growth signal
            if 'EARNINGS_GROWTH' in fundamental_data.columns:
                earnings_growth_signal = fundamental_data['EARNINGS_GROWTH'].rank(pct=True) - 0.5
                signals['earnings_growth_signal'] = earnings_growth_signal
            
            # Book value growth signal
            if 'BOOK_VALUE_GROWTH' in fundamental_data.columns:
                bv_growth_signal = fundamental_data['BOOK_VALUE_GROWTH'].rank(pct=True) - 0.5
                signals['book_value_growth_signal'] = bv_growth_signal
            
            # PEG ratio signal (lower is better for growth at reasonable price)
            if 'PE_RATIO' in fundamental_data.columns and 'EARNINGS_GROWTH' in fundamental_data.columns:
                peg_ratio = fundamental_data['PE_RATIO'] / fundamental_data['EARNINGS_GROWTH']
                peg_signal = -peg_ratio.rank(pct=True) + 0.5
                signals['peg_growth_signal'] = peg_signal
            
            # Composite growth score
            growth_columns = [col for col in ['REVENUE_GROWTH', 'EARNINGS_GROWTH', 'BOOK_VALUE_GROWTH'] 
                            if col in fundamental_data.columns]
            
            if growth_columns:
                growth_ranks = fundamental_data[growth_columns].rank(pct=True)
                composite_growth = growth_ranks.mean(axis=1) - 0.5
                signals['composite_growth_signal'] = composite_growth
            
            self.generated_signals.update(signals)
            return signals
            
        except Exception as e:
            logger.error(f"Error generating growth signals: {str(e)}")
            raise

class MachineLearningAlphaGenerator:
    """
    Machine learning-based alpha signal generator.
    
    Uses various ML algorithms to discover non-linear patterns and
    complex relationships in financial data for alpha generation.
    """
    
    def __init__(self, random_state: int = 42):
        """Initialize ML alpha generator."""
        self.random_state = random_state
        self.trained_models = {}
        self.feature_importance = {}
        
    def generate_ml_alpha_signals(self, features: pd.DataFrame,
                                returns: pd.DataFrame,
                                model_types: List[str] = ['random_forest', 'gradient_boosting'],
                                lookforward_periods: List[int] = [1, 5, 10]) -> Dict[str, pd.DataFrame]:
        """
        Generate ML-based alpha signals.
        
        Parameters:
        -----------
        features : pd.DataFrame
            Feature matrix (assets as columns, dates as index)
        returns : pd.DataFrame
            Forward returns to predict
        model_types : List[str]
            Types of ML models to use
        lookforward_periods : List[int]
            Forward-looking periods to predict
            
        Returns:
        --------
        Dict[str, pd.DataFrame]
            Dictionary of ML alpha signals
        """
        signals = {}
        
        try:
            for period in lookforward_periods:
                # Calculate forward returns
                forward_returns = returns.shift(-period)
                
                for model_type in model_types:
                    # Train models for each asset
                    asset_signals = {}
                    
                    for asset in returns.columns:
                        try:
                            # Prepare training data
                            X, y = self._prepare_ml_training_data(
                                features, forward_returns[asset], period
                            )
                            
                            if len(X) < 100:  # Minimum training samples
                                continue
                            
                            # Train model
                            model = self._train_ml_model(X, y, model_type)
                            
                            if model is not None:
                                # Generate predictions (signals)
                                asset_predictions = self._generate_ml_predictions(
                                    model, features, asset
                                )
                                asset_signals[asset] = asset_predictions
                                
                                # Store model and feature importance
                                model_key = f"{model_type}_{asset}_{period}d"
                                self.trained_models[model_key] = model
                                
                                if hasattr(model, 'feature_importances_'):
                                    self.feature_importance[model_key] = model.feature_importances_
                        
                        except Exception as e:
                            logger.warning(f"Error training ML model for {asset}: {str(e)}")
                            continue
                    
                    if asset_signals:
                        signal_df = pd.DataFrame(asset_signals)
                        signal_name = f'ml_{model_type}_alpha_{period}d'
                        signals[signal_name] = signal_df
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating ML alpha signals: {str(e)}")
            raise
    
    def _prepare_ml_training_data(self, features: pd.DataFrame,
                                target: pd.Series, period: int) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for ML models."""
        # Align features and target
        common_dates = features.index.intersection(target.index)
        
        # Remove last 'period' observations (no forward returns available)
        training_dates = common_dates[:-period] if len(common_dates) > period else common_dates
        
        if len(training_dates) == 0:
            return np.array([]), np.array([])
        
        X = features.loc[training_dates].values
        y = target.loc[training_dates].values
        
        # Remove NaN values
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[valid_mask]
        y = y[valid_mask]
        
        return X, y
    
    def _train_ml_model(self, X: np.ndarray, y: np.ndarray, model_type: str):
        """Train ML model."""
        try:
            if len(X) < 50:  # Minimum training samples
                return None
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Initialize model
            if model_type == 'random_forest':
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=20,
                    min_samples_leaf=10,
                    random_state=self.random_state
                )
            elif model_type == 'gradient_boosting':
                model = GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    min_samples_split=20,
                    min_samples_leaf=10,
                    random_state=self.random_state
                )
            elif model_type == 'linear':
                model = Ridge(alpha=1.0)
            else:
                logger.warning(f"Unknown model type: {model_type}")
                return None
            
            # Train model
            model.fit(X_scaled, y)
            
            # Store scaler with model
            model.scaler = scaler
            
            return model
            
        except Exception as e:
            logger.warning(f"Error training {model_type} model: {str(e)}")
            return None
    
    def _generate_ml_predictions(self, model, features: pd.DataFrame, asset: str) -> pd.Series:
        """Generate ML predictions as alpha signals."""
        try:
            # Scale features
            X = features.values
            X_scaled = model.scaler.transform(X)
            
            # Generate predictions
            predictions = model.predict(X_scaled)
            
            # Convert to Series
            prediction_series = pd.Series(predictions, index=features.index)
            
            # Normalize predictions to alpha signal
            prediction_series = (prediction_series.rank(pct=True) - 0.5) * 2
            
            return prediction_series
            
        except Exception as e:
            logger.warning(f"Error generating ML predictions: {str(e)}")
            return pd.Series(0, index=features.index)

class AlphaSignalEvaluator:
    """
    Alpha signal evaluation and screening system.
    
    Evaluates the quality and potential of discovered alpha signals using
    various metrics and statistical tests.
    """
    
    def __init__(self):
        """Initialize alpha signal evaluator."""
        self.evaluation_results = {}
        
    def evaluate_alpha_signal(self, signal: pd.Series, returns: pd.Series,
                            signal_name: str, benchmark_returns: Optional[pd.Series] = None) -> AlphaSignal:
        """
        Evaluate a single alpha signal.
        
        Parameters:
        -----------
        signal : pd.Series
            Alpha signal values
        returns : pd.Series  
            Forward returns to evaluate against
        signal_name : str
            Name of the signal
        benchmark_returns : Optional[pd.Series]
            Benchmark returns for relative evaluation
            
        Returns:
        --------
        AlphaSignal
            Evaluated alpha signal object
        """
        try:
            # Align signal and returns
            common_index = signal.index.intersection(returns.index)
            aligned_signal = signal.loc[common_index]
            aligned_returns = returns.loc[common_index]
            
            # Remove NaN values
            valid_mask = ~(aligned_signal.isna() | aligned_returns.isna())
            clean_signal = aligned_signal[valid_mask]
            clean_returns = aligned_returns[valid_mask]
            
            if len(clean_signal) < 30:  # Minimum observations
                logger.warning(f"Insufficient data for signal evaluation: {signal_name}")
                return self._create_empty_alpha_signal(signal_name)
            
            # Calculate metrics
            information_coefficient = self._calculate_information_coefficient(clean_signal, clean_returns)
            sharpe_ratio = self._calculate_signal_sharpe_ratio(clean_signal, clean_returns)
            turnover = self._calculate_turnover(clean_signal)
            capacity = self._estimate_capacity(clean_signal, clean_returns)
            decay_half_life = self._calculate_decay_half_life(clean_signal, clean_returns)
            correlation_with_returns = clean_signal.corr(clean_returns)
            sector_neutrality = self._calculate_sector_neutrality(clean_signal)
            risk_adjusted_ic = self._calculate_risk_adjusted_ic(clean_signal, clean_returns)
            significance_pvalue = self._calculate_significance_test(clean_signal, clean_returns)
            
            # Determine signal type
            signal_type = self._determine_signal_type(clean_signal, clean_returns)
            
            alpha_signal = AlphaSignal(
                signal_name=signal_name,
                signal_type=signal_type,
                alpha_values=clean_signal,
                information_coefficient=information_coefficient,
                sharpe_ratio=sharpe_ratio,
                turnover=turnover,
                capacity=capacity,
                decay_half_life=decay_half_life,
                correlation_with_returns=correlation_with_returns,
                sector_neutrality=sector_neutrality,
                risk_adjusted_ic=risk_adjusted_ic,
                significance_pvalue=significance_pvalue
            )
            
            return alpha_signal
            
        except Exception as e:
            logger.error(f"Error evaluating alpha signal {signal_name}: {str(e)}")
            return self._create_empty_alpha_signal(signal_name)
    
    def _calculate_information_coefficient(self, signal: pd.Series, returns: pd.Series) -> float:
        """Calculate Information Coefficient (Spearman correlation)."""
        try:
            ic, _ = spearman(signal.values, returns.values)
            return ic if not np.isnan(ic) else 0.0
        except:
            return 0.0
    
    def _calculate_signal_sharpe_ratio(self, signal: pd.Series, returns: pd.Series) -> float:
        """Calculate signal-based Sharpe ratio."""
        try:
            # Create long-short portfolio based on signal
            signal_ranks = signal.rank(pct=True)
            
            # Long top quintile, short bottom quintile
            long_mask = signal_ranks >= 0.8
            short_mask = signal_ranks <= 0.2
            
            if long_mask.sum() == 0 or short_mask.sum() == 0:
                return 0.0
            
            long_returns = returns[long_mask].mean()
            short_returns = returns[short_mask].mean()
            
            portfolio_returns = long_returns - short_returns
            
            # Calculate Sharpe ratio
            if isinstance(portfolio_returns, pd.Series):
                mean_return = portfolio_returns.mean()
                std_return = portfolio_returns.std()
            else:
                # Single value
                return 0.0
            
            sharpe = mean_return / std_return if std_return > 0 else 0.0
            return sharpe
            
        except:
            return 0.0
    
    def _calculate_turnover(self, signal: pd.Series) -> float:
        """Calculate signal turnover."""
        try:
            signal_change = signal.diff().abs()
            avg_signal_level = signal.abs().mean()
            turnover = signal_change.mean() / avg_signal_level if avg_signal_level > 0 else 0.0
            return turnover
        except:
            return 0.0
    
    def _estimate_capacity(self, signal: pd.Series, returns: pd.Series) -> float:
        """Estimate signal capacity (simplified)."""
        try:
            # Simple capacity estimate based on signal strength and volatility
            ic = abs(self._calculate_information_coefficient(signal, returns))
            volatility = returns.std()
            
            # Higher IC and lower volatility suggest higher capacity
            capacity_score = ic / volatility if volatility > 0 else 0.0
            
            # Scale to reasonable range (0-100)
            normalized_capacity = min(capacity_score * 1000, 100)
            return normalized_capacity
            
        except:
            return 0.0
    
    def _calculate_decay_half_life(self, signal: pd.Series, returns: pd.Series) -> float:
        """Calculate signal decay half-life."""
        try:
            # Calculate IC at different forward periods
            max_periods = min(20, len(signal) // 4)
            ics = []
            
            for period in range(1, max_periods + 1):
                if period < len(returns):
                    forward_returns = returns.shift(-period)
                    common_idx = signal.index.intersection(forward_returns.index)
                    
                    if len(common_idx) > 10:
                        period_ic = abs(spearman(
                            signal.loc[common_idx].values,
                            forward_returns.loc[common_idx].values
                        )[0])
                        ics.append(period_ic if not np.isnan(period_ic) else 0)
                    else:
                        ics.append(0)
            
            if not ics or max(ics) == 0:
                return np.inf
            
            # Find half-life (period where IC drops to half of maximum)
            max_ic = max(ics)
            half_ic = max_ic / 2
            
            half_life_period = next(
                (i + 1 for i, ic in enumerate(ics) if ic <= half_ic),
                len(ics)
            )
            
            return float(half_life_period)
            
        except:
            return np.inf
    
    def _calculate_sector_neutrality(self, signal: pd.Series) -> float:
        """Calculate sector neutrality (simplified - assumes no sector data)."""
        # Placeholder - would calculate cross-sectional mean of signal by sector
        try:
            cross_sectional_mean = signal.mean()
            neutrality = 1.0 - abs(cross_sectional_mean)
            return max(0.0, neutrality)
        except:
            return 0.5
    
    def _calculate_risk_adjusted_ic(self, signal: pd.Series, returns: pd.Series) -> float:
        """Calculate risk-adjusted Information Coefficient."""
        try:
            ic = self._calculate_information_coefficient(signal, returns)
            return_vol = returns.std()
            
            # Adjust IC by return volatility
            risk_adjusted_ic = ic / return_vol if return_vol > 0 else 0.0
            return risk_adjusted_ic
            
        except:
            return 0.0
    
    def _calculate_significance_test(self, signal: pd.Series, returns: pd.Series) -> float:
        """Calculate statistical significance of signal-return relationship."""
        try:
            _, p_value = spearman(signal.values, returns.values)
            return p_value if not np.isnan(p_value) else 1.0
        except:
            return 1.0
    
    def _determine_signal_type(self, signal: pd.Series, returns: pd.Series) -> str:
        """Determine signal type based on characteristics."""
        try:
            # Simple heuristic based on signal properties
            signal_autocorr = signal.autocorr(lag=1)
            
            if abs(signal_autocorr) > 0.7:
                return 'time_series'
            elif abs(signal_autocorr) < 0.3:
                return 'cross_sectional'
            else:
                return 'hybrid'
                
        except:
            return 'hybrid'
    
    def _create_empty_alpha_signal(self, signal_name: str) -> AlphaSignal:
        """Create empty alpha signal for failed evaluations."""
        return AlphaSignal(
            signal_name=signal_name,
            signal_type='unknown',
            alpha_values=pd.Series(),
            information_coefficient=0.0,
            sharpe_ratio=0.0,
            turnover=0.0,
            capacity=0.0,
            decay_half_life=np.inf,
            correlation_with_returns=0.0,
            sector_neutrality=0.0,
            risk_adjusted_ic=0.0,
            significance_pvalue=1.0
        )

class ComprehensiveAlphaDiscovery:
    """
    Comprehensive alpha discovery system integrating all methodologies.
    
    Coordinates technical, fundamental, and ML-based alpha discovery
    with comprehensive evaluation and screening.
    """
    
    def __init__(self, min_ic_threshold: float = 0.02, max_pvalue: float = 0.05):
        """
        Initialize comprehensive alpha discovery system.
        
        Parameters:
        -----------
        min_ic_threshold : float
            Minimum Information Coefficient threshold for signal selection
        max_pvalue : float
            Maximum p-value for statistical significance
        """
        self.min_ic_threshold = min_ic_threshold
        self.max_pvalue = max_pvalue
        
        self.technical_generator = TechnicalAlphaGenerator()
        self.fundamental_generator = FundamentalAlphaGenerator()
        self.ml_generator = MachineLearningAlphaGenerator()
        self.evaluator = AlphaSignalEvaluator()
        
    def discover_alpha_signals(self, price_data: pd.DataFrame,
                             returns_data: pd.DataFrame,
                             volume_data: Optional[pd.DataFrame] = None,
                             fundamental_data: Optional[pd.DataFrame] = None,
                             additional_features: Optional[pd.DataFrame] = None) -> AlphaDiscoveryResults:
        """
        Discover alpha signals using all available methodologies.
        
        Parameters:
        -----------
        price_data : pd.DataFrame
            Price data
        returns_data : pd.DataFrame
            Forward returns data
        volume_data : Optional[pd.DataFrame]
            Volume data
        fundamental_data : Optional[pd.DataFrame]
            Fundamental data
        additional_features : Optional[pd.DataFrame]
            Additional features for ML models
            
        Returns:
        --------
        AlphaDiscoveryResults
            Comprehensive discovery results
        """
        try:
            all_signals = {}
            
            # Technical signal generation
            logger.info("Generating technical alpha signals...")
            
            # Momentum signals
            momentum_signals = self.technical_generator.generate_momentum_signals(
                price_data, volume_data
            )
            all_signals.update(momentum_signals)
            
            # Mean reversion signals
            mean_reversion_signals = self.technical_generator.generate_mean_reversion_signals(price_data)
            all_signals.update(mean_reversion_signals)
            
            # Volatility signals
            volatility_signals = self.technical_generator.generate_volatility_signals(price_data)
            all_signals.update(volatility_signals)
            
            # Fundamental signal generation
            if fundamental_data is not None:
                logger.info("Generating fundamental alpha signals...")
                
                value_signals = self.fundamental_generator.generate_value_signals(fundamental_data)
                all_signals.update(value_signals)
                
                quality_signals = self.fundamental_generator.generate_quality_signals(fundamental_data)
                all_signals.update(quality_signals)
                
                growth_signals = self.fundamental_generator.generate_growth_signals(fundamental_data)
                all_signals.update(growth_signals)
            
            # ML signal generation
            if additional_features is not None:
                logger.info("Generating ML-based alpha signals...")
                
                ml_signals = self.ml_generator.generate_ml_alpha_signals(
                    additional_features, returns_data
                )
                all_signals.update(ml_signals)
            
            # Evaluate all signals
            logger.info(f"Evaluating {len(all_signals)} alpha signals...")
            evaluated_signals = []
            
            for signal_name, signal_data in all_signals.items():
                if isinstance(signal_data, pd.DataFrame):
                    # Multi-asset signal - evaluate each asset
                    for asset in signal_data.columns:
                        if asset in returns_data.columns:
                            asset_signal_name = f"{signal_name}_{asset}"
                            alpha_signal = self.evaluator.evaluate_alpha_signal(
                                signal_data[asset], returns_data[asset], asset_signal_name
                            )
                            evaluated_signals.append(alpha_signal)
                elif isinstance(signal_data, pd.Series):
                    # Single signal
                    if len(returns_data.columns) == 1:
                        alpha_signal = self.evaluator.evaluate_alpha_signal(
                            signal_data, returns_data.iloc[:, 0], signal_name
                        )
                        evaluated_signals.append(alpha_signal)
            
            # Filter and rank signals
            filtered_signals = self._filter_signals(evaluated_signals)
            top_signals = self._rank_signals(filtered_signals)
            
            # Calculate signal correlations
            signal_correlations = self._calculate_signal_correlations(top_signals)
            
            # Create performance summary
            performance_summary = self._create_performance_summary(top_signals)
            
            # Create results object
            results = AlphaDiscoveryResults(
                discovered_signals=evaluated_signals,
                signal_correlations=signal_correlations,
                top_signals=top_signals,
                signal_performance_summary=performance_summary,
                discovery_methodology="comprehensive_multi_method",
                discovery_date=datetime.now()
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error in comprehensive alpha discovery: {str(e)}")
            raise
    
    def _filter_signals(self, signals: List[AlphaSignal]) -> List[AlphaSignal]:
        """Filter signals based on quality criteria."""
        filtered = []
        
        for signal in signals:
            # Apply filtering criteria
            if (abs(signal.information_coefficient) >= self.min_ic_threshold and
                signal.significance_pvalue <= self.max_pvalue and
                not np.isinf(signal.decay_half_life) and
                signal.turnover < 10.0):  # Reasonable turnover
                
                filtered.append(signal)
        
        return filtered
    
    def _rank_signals(self, signals: List[AlphaSignal], top_n: int = 20) -> List[AlphaSignal]:
        """Rank signals by quality score."""
        try:
            # Calculate composite quality score
            for signal in signals:
                quality_score = (
                    abs(signal.information_coefficient) * 0.4 +
                    abs(signal.sharpe_ratio) * 0.3 +
                    (1 / (1 + signal.turnover)) * 0.1 +
                    (signal.capacity / 100) * 0.1 +
                    (1 - signal.significance_pvalue) * 0.1
                )
                signal.quality_score = quality_score
            
            # Sort by quality score
            ranked_signals = sorted(signals, key=lambda x: getattr(x, 'quality_score', 0), reverse=True)
            
            return ranked_signals[:top_n]
            
        except Exception as e:
            logger.warning(f"Error ranking signals: {str(e)}")
            return signals[:top_n]
    
    def _calculate_signal_correlations(self, signals: List[AlphaSignal]) -> pd.DataFrame:
        """Calculate correlations between top signals."""
        try:
            if len(signals) < 2:
                return pd.DataFrame()
            
            # Align all signal values
            signal_data = {}
            for signal in signals:
                if not signal.alpha_values.empty:
                    signal_data[signal.signal_name] = signal.alpha_values
            
            if len(signal_data) < 2:
                return pd.DataFrame()
            
            signal_df = pd.DataFrame(signal_data)
            correlation_matrix = signal_df.corr()
            
            return correlation_matrix
            
        except Exception as e:
            logger.warning(f"Error calculating signal correlations: {str(e)}")
            return pd.DataFrame()
    
    def _create_performance_summary(self, signals: List[AlphaSignal]) -> pd.DataFrame:
        """Create performance summary DataFrame."""
        try:
            if not signals:
                return pd.DataFrame()
            
            summary_data = []
            for signal in signals:
                summary_data.append({
                    'signal_name': signal.signal_name,
                    'signal_type': signal.signal_type,
                    'information_coefficient': signal.information_coefficient,
                    'sharpe_ratio': signal.sharpe_ratio,
                    'turnover': signal.turnover,
                    'capacity': signal.capacity,
                    'decay_half_life': signal.decay_half_life,
                    'significance_pvalue': signal.significance_pvalue,
                    'quality_score': getattr(signal, 'quality_score', 0)
                })
            
            summary_df = pd.DataFrame(summary_data)
            return summary_df
            
        except Exception as e:
            logger.warning(f"Error creating performance summary: {str(e)}")
            return pd.DataFrame()
    
    def generate_discovery_report(self, results: AlphaDiscoveryResults,
                                strategy_name: str = "Strategy") -> Dict[str, Any]:
        """
        Generate comprehensive alpha discovery report.
        
        Parameters:
        -----------
        results : AlphaDiscoveryResults
            Discovery results
        strategy_name : str
            Strategy name for reporting
            
        Returns:
        --------
        Dict[str, Any]
            Comprehensive discovery report
        """
        report = {
            'strategy_name': strategy_name,
            'discovery_summary': results.to_dict(),
            'signal_analysis': {
                'total_signals_generated': len(results.discovered_signals),
                'top_signals_selected': len(results.top_signals),
                'average_ic': np.mean([s.information_coefficient for s in results.top_signals]) if results.top_signals else 0,
                'average_sharpe': np.mean([s.sharpe_ratio for s in results.top_signals]) if results.top_signals else 0,
                'signal_type_distribution': self._analyze_signal_types(results.top_signals)
            },
            'recommendations': self._generate_alpha_recommendations(results),
            'next_steps': self._suggest_next_steps(results)
        }
        
        return report
    
    def _analyze_signal_types(self, signals: List[AlphaSignal]) -> Dict[str, int]:
        """Analyze distribution of signal types."""
        type_counts = {}
        for signal in signals:
            signal_type = signal.signal_type
            type_counts[signal_type] = type_counts.get(signal_type, 0) + 1
        
        return type_counts
    
    def _generate_alpha_recommendations(self, results: AlphaDiscoveryResults) -> List[str]:
        """Generate recommendations based on discovery results."""
        recommendations = []
        
        if len(results.top_signals) == 0:
            recommendations.append("No significant alpha signals discovered. Consider expanding feature set or adjusting thresholds.")
            return recommendations
        
        # IC analysis
        avg_ic = np.mean([abs(s.information_coefficient) for s in results.top_signals])
        if avg_ic > 0.05:
            recommendations.append("Strong alpha signals discovered with high Information Coefficients. Consider portfolio implementation.")
        elif avg_ic > 0.02:
            recommendations.append("Moderate alpha signals found. Consider combining multiple signals for improved performance.")
        
        # Turnover analysis
        avg_turnover = np.mean([s.turnover for s in results.top_signals])
        if avg_turnover > 5.0:
            recommendations.append("High turnover signals detected. Consider transaction cost impact in implementation.")
        
        # Signal diversity
        signal_types = set(s.signal_type for s in results.top_signals)
        if len(signal_types) > 1:
            recommendations.append("Diverse signal types discovered. Consider ensemble approach combining different alpha sources.")
        
        # Decay analysis
        avg_decay = np.mean([s.decay_half_life for s in results.top_signals if not np.isinf(s.decay_half_life)])
        if not np.isnan(avg_decay) and avg_decay < 5:
            recommendations.append("Fast-decaying signals detected. Consider high-frequency trading implementation.")
        
        return recommendations
    
    def _suggest_next_steps(self, results: AlphaDiscoveryResults) -> List[str]:
        """Suggest next steps for alpha research."""
        next_steps = []
        
        if results.top_signals:
            next_steps.append("1. Implement alpha combination framework to optimize signal weighting")
            next_steps.append("2. Conduct out-of-sample validation of top signals")
            next_steps.append("3. Analyze signal performance across different market regimes")
            next_steps.append("4. Implement portfolio construction with discovered alphas")
            next_steps.append("5. Monitor signal decay and implement refresh procedures")
        else:
            next_steps.append("1. Expand feature engineering to include alternative data sources")
            next_steps.append("2. Investigate more sophisticated ML models (deep learning, ensemble methods)")
            next_steps.append("3. Analyze signal performance in different market conditions")
            next_steps.append("4. Consider regime-dependent alpha discovery")
        
        return next_steps

