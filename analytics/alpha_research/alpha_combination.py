"""
Alpha Combination Framework
==========================

Advanced alpha combination system for optimal blending of multiple alpha signals
to create robust composite alpha strategies with superior risk-adjusted returns.

Features:
- Multiple alpha combination methodologies
- Dynamic weight optimization with constraints
- Risk-based alpha combination approaches
- Machine learning-based signal weighting
- Regime-dependent alpha combinations
- Transaction cost-aware optimization
- Alpha decay and refresh mechanisms
- Portfolio construction integration

Author: Quantum Forge Analytics Team
Date: November 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
from scipy import optimize, stats
from scipy.optimize import minimize, differential_evolution
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import cvxpy as cp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AlphaCombinationWeights:
    """Container for alpha combination weights and metadata."""
    weights: Dict[str, float]
    combination_method: str
    rebalance_date: pd.Timestamp
    expected_ic: float
    expected_sharpe: float
    expected_turnover: float
    risk_budget_allocation: Dict[str, float]
    constraints_satisfied: bool
    optimization_status: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'weights': self.weights,
            'combination_method': self.combination_method,
            'rebalance_date': self.rebalance_date,
            'expected_ic': self.expected_ic,
            'expected_sharpe': self.expected_sharpe,
            'expected_turnover': self.expected_turnover,
            'risk_budget_allocation': self.risk_budget_allocation,
            'constraints_satisfied': self.constraints_satisfied,
            'optimization_status': self.optimization_status
        }

@dataclass
class CombinedAlphaSignal:
    """Container for combined alpha signal results."""
    combined_signal: pd.Series
    component_weights: pd.DataFrame
    signal_contributions: pd.DataFrame
    performance_metrics: Dict[str, float]
    combination_metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'performance_metrics': self.performance_metrics,
            'combination_metadata': self.combination_metadata,
            'signal_summary': {
                'num_periods': len(self.combined_signal),
                'num_components': len(self.component_weights.columns) if not self.component_weights.empty else 0,
                'avg_combined_signal': self.combined_signal.mean(),
                'combined_signal_std': self.combined_signal.std()
            }
        }

class EqualWeightCombiner:
    """
    Equal weight alpha combination approach.
    
    Simple but robust approach that weights all alpha signals equally,
    providing good diversification and avoiding concentration risk.
    """
    
    def __init__(self):
        """Initialize equal weight combiner."""
        self.method_name = "equal_weight"
        
    def combine_alphas(self, alpha_signals: Dict[str, pd.Series],
                      returns: Optional[pd.Series] = None) -> CombinedAlphaSignal:
        """
        Combine alpha signals using equal weights.
        
        Parameters:
        -----------
        alpha_signals : Dict[str, pd.Series]
            Dictionary of alpha signals
        returns : Optional[pd.Series]
            Returns for performance evaluation (not used in equal weight)
            
        Returns:
        --------
        CombinedAlphaSignal
            Combined alpha signal results
        """
        try:
            if not alpha_signals:
                raise ValueError("No alpha signals provided")
            
            # Align all signals
            signal_df = pd.DataFrame(alpha_signals)
            signal_df = signal_df.dropna()
            
            if signal_df.empty:
                raise ValueError("No overlapping data in alpha signals")
            
            # Equal weights
            n_signals = len(alpha_signals)
            equal_weight = 1.0 / n_signals
            weights = {name: equal_weight for name in alpha_signals.keys()}
            
            # Combined signal
            combined_signal = signal_df.mean(axis=1)
            
            # Component weights (constant over time)
            weight_df = pd.DataFrame(
                [weights] * len(signal_df),
                index=signal_df.index,
                columns=signal_df.columns
            )
            
            # Signal contributions
            contributions = signal_df * equal_weight
            
            # Performance metrics
            performance_metrics = {
                'combination_method': self.method_name,
                'num_signals': n_signals,
                'equal_weight': equal_weight,
                'signal_correlation_avg': signal_df.corr().values[np.triu_indices_from(signal_df.corr().values, k=1)].mean()
            }
            
            # Metadata
            combination_metadata = {
                'method': self.method_name,
                'optimization_required': False,
                'dynamic_weights': False,
                'signal_names': list(alpha_signals.keys())
            }
            
            return CombinedAlphaSignal(
                combined_signal=combined_signal,
                component_weights=weight_df,
                signal_contributions=contributions,
                performance_metrics=performance_metrics,
                combination_metadata=combination_metadata
            )
            
        except Exception as e:
            logger.error(f"Error in equal weight combination: {str(e)}")
            raise

class ICWeightedCombiner:
    """
    Information Coefficient weighted alpha combination.
    
    Weights alpha signals based on their historical Information Coefficient
    performance, giving more weight to signals with higher predictive power.
    """
    
    def __init__(self, lookback_period: int = 252, min_ic_threshold: float = 0.01):
        """
        Initialize IC weighted combiner.
        
        Parameters:
        -----------
        lookback_period : int
            Lookback period for IC calculation
        min_ic_threshold : float
            Minimum IC threshold for inclusion
        """
        self.lookback_period = lookback_period
        self.min_ic_threshold = min_ic_threshold
        self.method_name = "ic_weighted"
        
    def combine_alphas(self, alpha_signals: Dict[str, pd.Series],
                      returns: pd.Series) -> CombinedAlphaSignal:
        """
        Combine alpha signals using IC-based weights.
        
        Parameters:
        -----------
        alpha_signals : Dict[str, pd.Series]
            Dictionary of alpha signals
        returns : pd.Series
            Forward returns for IC calculation
            
        Returns:
        --------
        CombinedAlphaSignal
            Combined alpha signal results
        """
        try:
            if returns is None:
                raise ValueError("Returns required for IC weighted combination")
            
            # Align signals and returns
            signal_df = pd.DataFrame(alpha_signals)
            common_index = signal_df.index.intersection(returns.index)
            signal_df = signal_df.loc[common_index]
            aligned_returns = returns.loc[common_index]
            
            if len(signal_df) < self.lookback_period:
                logger.warning(f"Insufficient data for IC calculation. Using available {len(signal_df)} periods.")
                lookback = len(signal_df)
            else:
                lookback = self.lookback_period
            
            # Calculate rolling IC weights
            ic_weights_list = []
            combined_signals = []
            contributions_list = []
            
            for i in range(lookback - 1, len(signal_df)):
                # Calculate IC for each signal over lookback period
                window_signals = signal_df.iloc[i - lookback + 1:i + 1]
                window_returns = aligned_returns.iloc[i - lookback + 1:i + 1]
                
                ic_values = {}
                for signal_name in signal_df.columns:
                    try:
                        ic_corr = window_signals[signal_name].corr(window_returns)
                        ic_values[signal_name] = abs(ic_corr) if not pd.isna(ic_corr) else 0.0
                    except:
                        ic_values[signal_name] = 0.0
                
                # Filter signals by minimum IC threshold
                filtered_ics = {k: v for k, v in ic_values.items() if v >= self.min_ic_threshold}
                
                if not filtered_ics:
                    # No signals meet threshold - use equal weights
                    filtered_ics = {k: 1.0 for k in ic_values.keys()}
                
                # Normalize to weights
                total_ic = sum(filtered_ics.values())
                if total_ic > 0:
                    weights = {k: v / total_ic for k, v in filtered_ics.items()}
                else:
                    weights = {k: 1.0 / len(filtered_ics) for k in filtered_ics.keys()}
                
                # Set weights for non-qualifying signals to zero
                full_weights = {k: weights.get(k, 0.0) for k in signal_df.columns}
                
                # Calculate combined signal for this period
                current_signals = signal_df.iloc[i]
                combined_value = sum(current_signals[k] * full_weights[k] for k in full_weights.keys())
                
                # Store results
                ic_weights_list.append(full_weights)
                combined_signals.append(combined_value)
                
                # Signal contributions
                contributions = {k: current_signals[k] * full_weights[k] for k in full_weights.keys()}
                contributions_list.append(contributions)
            
            # Create result DataFrames
            result_index = signal_df.index[lookback - 1:]
            combined_signal = pd.Series(combined_signals, index=result_index)
            
            weight_df = pd.DataFrame(ic_weights_list, index=result_index, columns=signal_df.columns)
            contributions_df = pd.DataFrame(contributions_list, index=result_index, columns=signal_df.columns)
            
            # Performance metrics
            avg_weights = weight_df.mean()
            performance_metrics = {
                'combination_method': self.method_name,
                'lookback_period': lookback,
                'min_ic_threshold': self.min_ic_threshold,
                'avg_num_active_signals': (weight_df > 0).sum(axis=1).mean(),
                'weight_concentration': (avg_weights ** 2).sum(),  # Herfindahl index
                'avg_ic_values': {k: v for k, v in avg_weights.to_dict().items()}
            }
            
            # Metadata
            combination_metadata = {
                'method': self.method_name,
                'optimization_required': False,
                'dynamic_weights': True,
                'signal_names': list(alpha_signals.keys()),
                'rebalance_frequency': 'daily'
            }
            
            return CombinedAlphaSignal(
                combined_signal=combined_signal,
                component_weights=weight_df,
                signal_contributions=contributions_df,
                performance_metrics=performance_metrics,
                combination_metadata=combination_metadata
            )
            
        except Exception as e:
            logger.error(f"Error in IC weighted combination: {str(e)}")
            raise

class OptimizedCombiner:
    """
    Optimization-based alpha combination system.
    
    Uses constrained optimization to find optimal weights that maximize
    expected returns subject to risk and concentration constraints.
    """
    
    def __init__(self, objective: str = 'sharpe', lookback_period: int = 252,
                 max_weight: float = 0.5, min_weight: float = 0.0):
        """
        Initialize optimized combiner.
        
        Parameters:
        -----------
        objective : str
            Optimization objective ('sharpe', 'ic', 'return')
        lookback_period : int
            Lookback period for optimization
        max_weight : float
            Maximum weight per signal
        min_weight : float
            Minimum weight per signal
        """
        self.objective = objective
        self.lookback_period = lookback_period
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.method_name = f"optimized_{objective}"
        
    def combine_alphas(self, alpha_signals: Dict[str, pd.Series],
                      returns: pd.Series,
                      rebalance_frequency: int = 60) -> CombinedAlphaSignal:
        """
        Combine alpha signals using optimization.
        
        Parameters:
        -----------
        alpha_signals : Dict[str, pd.Series]
            Dictionary of alpha signals
        returns : pd.Series
            Forward returns for optimization
        rebalance_frequency : int
            Rebalancing frequency in periods
            
        Returns:
        --------
        CombinedAlphaSignal
            Combined alpha signal results
        """
        try:
            # Align signals and returns
            signal_df = pd.DataFrame(alpha_signals)
            common_index = signal_df.index.intersection(returns.index)
            signal_df = signal_df.loc[common_index]
            aligned_returns = returns.loc[common_index]
            
            if len(signal_df) < self.lookback_period:
                raise ValueError(f"Insufficient data for optimization. Need {self.lookback_period}, got {len(signal_df)}")
            
            # Optimize weights at rebalancing intervals
            optimization_results = []
            weight_periods = []
            
            for i in range(self.lookback_period - 1, len(signal_df), rebalance_frequency):
                # Optimization window
                end_idx = min(i + 1, len(signal_df))
                start_idx = max(0, end_idx - self.lookback_period)
                
                window_signals = signal_df.iloc[start_idx:end_idx]
                window_returns = aligned_returns.iloc[start_idx:end_idx]
                
                # Optimize weights
                optimal_weights = self._optimize_weights(window_signals, window_returns)
                
                # Apply weights to future periods until next rebalance
                next_rebalance = min(i + rebalance_frequency, len(signal_df))
                
                for j in range(i, next_rebalance):
                    if j < len(signal_df):
                        optimization_results.append(optimal_weights)
                        weight_periods.append(signal_df.index[j])
            
            if not optimization_results:
                raise ValueError("No optimization results generated")
            
            # Create weight DataFrame
            weight_df = pd.DataFrame(optimization_results, index=weight_periods, columns=signal_df.columns)
            
            # Calculate combined signals
            combined_values = []
            contributions_list = []
            
            for date in weight_df.index:
                if date in signal_df.index:
                    current_signals = signal_df.loc[date]
                    current_weights = weight_df.loc[date]
                    
                    combined_value = (current_signals * current_weights).sum()
                    combined_values.append(combined_value)
                    
                    contributions = current_signals * current_weights
                    contributions_list.append(contributions.to_dict())
            
            combined_signal = pd.Series(combined_values, index=weight_df.index)
            contributions_df = pd.DataFrame(contributions_list, index=weight_df.index, columns=signal_df.columns)
            
            # Performance metrics
            avg_weights = weight_df.mean()
            performance_metrics = {
                'combination_method': self.method_name,
                'optimization_objective': self.objective,
                'lookback_period': self.lookback_period,
                'rebalance_frequency': rebalance_frequency,
                'avg_weights': avg_weights.to_dict(),
                'weight_turnover': self._calculate_weight_turnover(weight_df),
                'concentration_ratio': (avg_weights ** 2).sum()
            }
            
            # Metadata
            combination_metadata = {
                'method': self.method_name,
                'optimization_required': True,
                'dynamic_weights': True,
                'signal_names': list(alpha_signals.keys()),
                'constraints': {
                    'max_weight': self.max_weight,
                    'min_weight': self.min_weight
                }
            }
            
            return CombinedAlphaSignal(
                combined_signal=combined_signal,
                component_weights=weight_df,
                signal_contributions=contributions_df,
                performance_metrics=performance_metrics,
                combination_metadata=combination_metadata
            )
            
        except Exception as e:
            logger.error(f"Error in optimized combination: {str(e)}")
            raise
    
    def _optimize_weights(self, signals: pd.DataFrame, returns: pd.Series) -> Dict[str, float]:
        """Optimize weights for given signals and returns."""
        try:
            n_signals = len(signals.columns)
            
            # Objective function
            def objective_function(weights):
                portfolio_signal = (signals * weights).sum(axis=1)
                
                if self.objective == 'ic':
                    # Maximize Information Coefficient
                    ic = portfolio_signal.corr(returns)
                    return -abs(ic) if not pd.isna(ic) else 0
                
                elif self.objective == 'sharpe':
                    # Maximize Sharpe ratio of signal-based returns
                    signal_returns = portfolio_signal * returns  # Simplified
                    if signal_returns.std() > 0:
                        sharpe = signal_returns.mean() / signal_returns.std()
                        return -sharpe
                    else:
                        return 0
                
                elif self.objective == 'return':
                    # Maximize expected return
                    signal_returns = portfolio_signal * returns
                    return -signal_returns.mean()
                
                else:
                    return 0
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Weights sum to 1
            ]
            
            # Bounds
            bounds = [(self.min_weight, self.max_weight) for _ in range(n_signals)]
            
            # Initial guess (equal weights)
            x0 = np.ones(n_signals) / n_signals
            
            # Optimize
            result = minimize(
                objective_function,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result.success:
                optimal_weights = result.x
            else:
                logger.warning("Optimization failed, using equal weights")
                optimal_weights = np.ones(n_signals) / n_signals
            
            return dict(zip(signals.columns, optimal_weights))
            
        except Exception as e:
            logger.warning(f"Error in weight optimization: {str(e)}")
            # Return equal weights as fallback
            return {col: 1.0 / len(signals.columns) for col in signals.columns}
    
    def _calculate_weight_turnover(self, weight_df: pd.DataFrame) -> float:
        """Calculate average weight turnover."""
        try:
            weight_changes = weight_df.diff().abs()
            avg_turnover = weight_changes.sum(axis=1).mean()
            return avg_turnover
        except:
            return 0.0

class RiskBudgetCombiner:
    """
    Risk budget-based alpha combination system.
    
    Allocates risk budget across alpha signals based on their risk contribution
    and expected performance, ensuring balanced risk exposure.
    """
    
    def __init__(self, risk_budget_method: str = 'equal_risk', lookback_period: int = 252):
        """
        Initialize risk budget combiner.
        
        Parameters:
        -----------
        risk_budget_method : str
            Risk budgeting method ('equal_risk', 'risk_parity', 'inverse_vol')
        lookback_period : int
            Lookback period for risk calculation
        """
        self.risk_budget_method = risk_budget_method
        self.lookback_period = lookback_period
        self.method_name = f"risk_budget_{risk_budget_method}"
        
    def combine_alphas(self, alpha_signals: Dict[str, pd.Series],
                      returns: pd.Series) -> CombinedAlphaSignal:
        """
        Combine alpha signals using risk budgeting approach.
        
        Parameters:
        -----------
        alpha_signals : Dict[str, pd.Series]
            Dictionary of alpha signals
        returns : pd.Series
            Returns for risk calculation
            
        Returns:
        --------
        CombinedAlphaSignal
            Combined alpha signal results
        """
        try:
            # Align signals and returns
            signal_df = pd.DataFrame(alpha_signals)
            common_index = signal_df.index.intersection(returns.index)
            signal_df = signal_df.loc[common_index]
            aligned_returns = returns.loc[common_index]
            
            # Calculate rolling risk-based weights
            weight_list = []
            combined_signals = []
            contributions_list = []
            
            for i in range(self.lookback_period - 1, len(signal_df)):
                # Risk calculation window
                window_signals = signal_df.iloc[i - self.lookback_period + 1:i + 1]
                
                # Calculate risk-based weights
                risk_weights = self._calculate_risk_weights(window_signals)
                
                # Calculate combined signal
                current_signals = signal_df.iloc[i]
                combined_value = (current_signals * pd.Series(risk_weights)).sum()
                
                # Store results
                weight_list.append(risk_weights)
                combined_signals.append(combined_value)
                
                # Signal contributions
                contributions = {k: current_signals[k] * risk_weights[k] for k in risk_weights.keys()}
                contributions_list.append(contributions)
            
            # Create result objects
            result_index = signal_df.index[self.lookback_period - 1:]
            combined_signal = pd.Series(combined_signals, index=result_index)
            
            weight_df = pd.DataFrame(weight_list, index=result_index, columns=signal_df.columns)
            contributions_df = pd.DataFrame(contributions_list, index=result_index, columns=signal_df.columns)
            
            # Performance metrics
            avg_weights = weight_df.mean()
            performance_metrics = {
                'combination_method': self.method_name,
                'risk_budget_method': self.risk_budget_method,
                'lookback_period': self.lookback_period,
                'avg_weights': avg_weights.to_dict(),
                'risk_concentration': self._calculate_risk_concentration(weight_df, signal_df)
            }
            
            # Metadata
            combination_metadata = {
                'method': self.method_name,
                'optimization_required': False,
                'dynamic_weights': True,
                'signal_names': list(alpha_signals.keys()),
                'risk_budgeting_approach': self.risk_budget_method
            }
            
            return CombinedAlphaSignal(
                combined_signal=combined_signal,
                component_weights=weight_df,
                signal_contributions=contributions_df,
                performance_metrics=performance_metrics,
                combination_metadata=combination_metadata
            )
            
        except Exception as e:
            logger.error(f"Error in risk budget combination: {str(e)}")
            raise
    
    def _calculate_risk_weights(self, signals: pd.DataFrame) -> Dict[str, float]:
        """Calculate risk-based weights."""
        try:
            if self.risk_budget_method == 'equal_risk':
                # Equal risk contribution
                signal_vols = signals.std()
                inv_vol_weights = 1 / signal_vols
                normalized_weights = inv_vol_weights / inv_vol_weights.sum()
                
            elif self.risk_budget_method == 'inverse_vol':
                # Inverse volatility weighting
                signal_vols = signals.std()
                inv_vol_weights = 1 / signal_vols
                normalized_weights = inv_vol_weights / inv_vol_weights.sum()
                
            elif self.risk_budget_method == 'risk_parity':
                # Risk parity using iterative algorithm
                normalized_weights = self._calculate_risk_parity_weights(signals)
                
            else:
                # Default to equal weights
                normalized_weights = pd.Series(1.0 / len(signals.columns), index=signals.columns)
            
            return normalized_weights.to_dict()
            
        except Exception as e:
            logger.warning(f"Error calculating risk weights: {str(e)}")
            # Fallback to equal weights
            return {col: 1.0 / len(signals.columns) for col in signals.columns}
    
    def _calculate_risk_parity_weights(self, signals: pd.DataFrame) -> pd.Series:
        """Calculate risk parity weights using simplified approach."""
        try:
            # Simplified risk parity - use correlation matrix
            corr_matrix = signals.corr().values
            vols = signals.std().values
            
            # Initial equal weights
            n = len(signals.columns)
            weights = np.ones(n) / n
            
            # Iterative risk parity calculation (simplified)
            for _ in range(10):  # Max iterations
                portfolio_vol = np.sqrt(weights @ (corr_matrix * np.outer(vols, vols)) @ weights)
                
                # Risk contributions
                marginal_risk = (corr_matrix * np.outer(vols, vols)) @ weights
                risk_contrib = weights * marginal_risk / portfolio_vol
                
                # Update weights to equalize risk contributions
                target_risk = portfolio_vol / n
                weight_multiplier = target_risk / risk_contrib
                
                # Normalize
                weights = weights * weight_multiplier
                weights = weights / weights.sum()
            
            return pd.Series(weights, index=signals.columns)
            
        except Exception as e:
            logger.warning(f"Error in risk parity calculation: {str(e)}")
            # Fallback to equal weights
            return pd.Series(1.0 / len(signals.columns), index=signals.columns)
    
    def _calculate_risk_concentration(self, weight_df: pd.DataFrame, 
                                    signal_df: pd.DataFrame) -> float:
        """Calculate risk concentration measure."""
        try:
            # Simplified risk concentration using weight concentration
            avg_weights = weight_df.mean()
            concentration = (avg_weights ** 2).sum()
            return concentration
        except:
            return np.nan

class MLCombiner:
    """
    Machine Learning-based alpha combination system.
    
    Uses ML algorithms to learn optimal combination weights based on
    historical performance and market conditions.
    """
    
    def __init__(self, model_type: str = 'random_forest', lookback_period: int = 252,
                 rebalance_frequency: int = 30):
        """
        Initialize ML combiner.
        
        Parameters:
        -----------
        model_type : str
            ML model type ('random_forest', 'ridge', 'elastic_net')
        lookback_period : int
            Training window size
        rebalance_frequency : int
            Model retraining frequency
        """
        self.model_type = model_type
        self.lookback_period = lookback_period
        self.rebalance_frequency = rebalance_frequency
        self.method_name = f"ml_{model_type}"
        self.trained_models = {}
        
    def combine_alphas(self, alpha_signals: Dict[str, pd.Series],
                      returns: pd.Series,
                      market_features: Optional[pd.DataFrame] = None) -> CombinedAlphaSignal:
        """
        Combine alpha signals using ML approach.
        
        Parameters:
        -----------
        alpha_signals : Dict[str, pd.Series]
            Dictionary of alpha signals
        returns : pd.Series
            Target returns for training
        market_features : Optional[pd.DataFrame]
            Additional market features for ML model
            
        Returns:
        --------
        CombinedAlphaSignal
            Combined alpha signal results
        """
        try:
            # Align all data
            signal_df = pd.DataFrame(alpha_signals)
            common_index = signal_df.index.intersection(returns.index)
            signal_df = signal_df.loc[common_index]
            aligned_returns = returns.loc[common_index]
            
            if market_features is not None:
                market_features = market_features.loc[common_index]
            
            # Train models and generate predictions
            weight_predictions = []
            combined_signals = []
            contributions_list = []
            model_dates = []
            
            for i in range(self.lookback_period, len(signal_df), self.rebalance_frequency):
                # Training window
                train_start = max(0, i - self.lookback_period)
                train_end = i
                
                # Get training data
                train_signals = signal_df.iloc[train_start:train_end]
                train_returns = aligned_returns.iloc[train_start:train_end]
                
                if market_features is not None:
                    train_features = market_features.iloc[train_start:train_end]
                else:
                    train_features = None
                
                # Train ML model to predict optimal weights
                model = self._train_weight_prediction_model(train_signals, train_returns, train_features)
                
                # Generate predictions for next period
                prediction_end = min(i + self.rebalance_frequency, len(signal_df))
                
                for j in range(i, prediction_end):
                    if j < len(signal_df):
                        # Predict weights
                        current_signals = signal_df.iloc[j:j+1]
                        
                        if market_features is not None and j < len(market_features):
                            current_features = market_features.iloc[j:j+1]
                        else:
                            current_features = None
                        
                        predicted_weights = self._predict_weights(model, current_signals, current_features)
                        
                        # Calculate combined signal
                        combined_value = (signal_df.iloc[j] * pd.Series(predicted_weights)).sum()
                        
                        # Store results
                        weight_predictions.append(predicted_weights)
                        combined_signals.append(combined_value)
                        model_dates.append(signal_df.index[j])
                        
                        # Signal contributions
                        contributions = {k: signal_df.iloc[j][k] * predicted_weights[k] for k in predicted_weights.keys()}
                        contributions_list.append(contributions)
            
            if not weight_predictions:
                raise ValueError("No ML predictions generated")
            
            # Create result objects
            combined_signal = pd.Series(combined_signals, index=model_dates)
            weight_df = pd.DataFrame(weight_predictions, index=model_dates, columns=signal_df.columns)
            contributions_df = pd.DataFrame(contributions_list, index=model_dates, columns=signal_df.columns)
            
            # Performance metrics
            avg_weights = weight_df.mean()
            performance_metrics = {
                'combination_method': self.method_name,
                'ml_model_type': self.model_type,
                'lookback_period': self.lookback_period,
                'rebalance_frequency': self.rebalance_frequency,
                'avg_weights': avg_weights.to_dict(),
                'weight_stability': 1 - weight_df.std().mean()  # Higher is more stable
            }
            
            # Metadata
            combination_metadata = {
                'method': self.method_name,
                'optimization_required': True,
                'dynamic_weights': True,
                'signal_names': list(alpha_signals.keys()),
                'ml_approach': True,
                'model_retraining_frequency': self.rebalance_frequency
            }
            
            return CombinedAlphaSignal(
                combined_signal=combined_signal,
                component_weights=weight_df,
                signal_contributions=contributions_df,
                performance_metrics=performance_metrics,
                combination_metadata=combination_metadata
            )
            
        except Exception as e:
            logger.error(f"Error in ML combination: {str(e)}")
            raise
    
    def _train_weight_prediction_model(self, signals: pd.DataFrame, returns: pd.Series,
                                     features: Optional[pd.DataFrame] = None):
        """Train ML model to predict optimal weights."""
        try:
            # Create training features
            X_features = []
            
            # Add signal values as features
            X_features.append(signals.values)
            
            # Add market features if available
            if features is not None:
                aligned_features = features.reindex(signals.index).fillna(method='ffill')
                X_features.append(aligned_features.values)
            
            # Combine features
            if len(X_features) > 1:
                X = np.hstack(X_features)
            else:
                X = X_features[0]
            
            # Target: forward returns (what we want to predict)
            y = returns.reindex(signals.index).values
            
            # Remove NaN values
            valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X_clean = X[valid_mask]
            y_clean = y[valid_mask]
            
            if len(X_clean) < 20:  # Minimum training samples
                return None
            
            # Train model
            if self.model_type == 'random_forest':
                model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
            elif self.model_type == 'ridge':
                model = Ridge(alpha=1.0)
            elif self.model_type == 'elastic_net':
                model = ElasticNet(alpha=1.0, l1_ratio=0.5)
            else:
                model = Ridge(alpha=1.0)  # Default
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_clean)
            
            # Fit model
            model.fit(X_scaled, y_clean)
            
            # Store scaler with model
            model.scaler = scaler
            
            return model
            
        except Exception as e:
            logger.warning(f"Error training ML weight prediction model: {str(e)}")
            return None
    
    def _predict_weights(self, model, current_signals: pd.DataFrame,
                        current_features: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """Predict optimal weights using trained model."""
        try:
            if model is None:
                # Fallback to equal weights
                return {col: 1.0 / len(current_signals.columns) for col in current_signals.columns}
            
            # Prepare features
            X_features = []
            X_features.append(current_signals.values.flatten())
            
            if current_features is not None:
                X_features.append(current_features.values.flatten())
            
            X = np.concatenate(X_features).reshape(1, -1)
            
            # Scale features
            X_scaled = model.scaler.transform(X)
            
            # Predict (this gives us expected return, not weights directly)
            prediction = model.predict(X_scaled)[0]
            
            # Convert prediction to weights (simplified approach)
            # In practice, you might train a separate model for each weight
            # or use multi-output regression
            
            # For now, use signal strength relative to prediction as weight proxy
            signal_values = current_signals.iloc[0]
            raw_weights = signal_values * prediction  # Scale by prediction
            
            # Normalize to weights
            if raw_weights.sum() != 0:
                normalized_weights = raw_weights / raw_weights.sum()
            else:
                normalized_weights = pd.Series(1.0 / len(signal_values), index=signal_values.index)
            
            # Ensure non-negative weights
            normalized_weights = normalized_weights.abs()
            normalized_weights = normalized_weights / normalized_weights.sum()
            
            return normalized_weights.to_dict()
            
        except Exception as e:
            logger.warning(f"Error predicting weights: {str(e)}")
            # Fallback to equal weights
            return {col: 1.0 / len(current_signals.columns) for col in current_signals.columns}

class ComprehensiveAlphaCombination:
    """
    Comprehensive alpha combination system integrating all methodologies.
    
    Provides unified interface for all alpha combination approaches with
    performance comparison and automatic method selection.
    """
    
    def __init__(self):
        """Initialize comprehensive alpha combination system."""
        self.combiners = {
            'equal_weight': EqualWeightCombiner(),
            'ic_weighted': ICWeightedCombiner(),
            'optimized_sharpe': OptimizedCombiner(objective='sharpe'),
            'optimized_ic': OptimizedCombiner(objective='ic'),
            'risk_budget': RiskBudgetCombiner(),
            'ml_random_forest': MLCombiner(model_type='random_forest')
        }
        
        self.combination_results = {}
        
    def combine_alphas_all_methods(self, alpha_signals: Dict[str, pd.Series],
                                 returns: pd.Series,
                                 methods: Optional[List[str]] = None) -> Dict[str, CombinedAlphaSignal]:
        """
        Combine alphas using all available methods.
        
        Parameters:
        -----------
        alpha_signals : Dict[str, pd.Series]
            Dictionary of alpha signals
        returns : pd.Series
            Forward returns for evaluation
        methods : Optional[List[str]]
            Specific methods to use (if None, uses all)
            
        Returns:
        --------
        Dict[str, CombinedAlphaSignal]
            Results from all combination methods
        """
        try:
            if methods is None:
                methods = list(self.combiners.keys())
            
            results = {}
            
            for method_name in methods:
                if method_name in self.combiners:
                    try:
                        logger.info(f"Running combination method: {method_name}")
                        combiner = self.combiners[method_name]
                        
                        if method_name == 'equal_weight':
                            result = combiner.combine_alphas(alpha_signals)
                        else:
                            result = combiner.combine_alphas(alpha_signals, returns)
                        
                        results[method_name] = result
                        
                    except Exception as e:
                        logger.warning(f"Error in {method_name} combination: {str(e)}")
                        continue
                else:
                    logger.warning(f"Unknown combination method: {method_name}")
            
            self.combination_results = results
            return results
            
        except Exception as e:
            logger.error(f"Error in comprehensive alpha combination: {str(e)}")
            raise
    
    def evaluate_combination_performance(self, combination_results: Dict[str, CombinedAlphaSignal],
                                       returns: pd.Series) -> pd.DataFrame:
        """
        Evaluate performance of different combination methods.
        
        Parameters:
        -----------
        combination_results : Dict[str, CombinedAlphaSignal]
            Results from different combination methods
        returns : pd.Series
            Forward returns for evaluation
            
        Returns:
        --------
        pd.DataFrame
            Performance comparison table
        """
        try:
            performance_data = []
            
            for method_name, result in combination_results.items():
                # Align combined signal with returns
                common_index = result.combined_signal.index.intersection(returns.index)
                aligned_signal = result.combined_signal.loc[common_index]
                aligned_returns = returns.loc[common_index]
                
                if len(aligned_signal) < 30:  # Minimum for evaluation
                    continue
                
                # Calculate performance metrics
                ic = aligned_signal.corr(aligned_returns)
                
                # Signal-based returns (simplified)
                signal_returns = aligned_signal * aligned_returns
                sharpe = signal_returns.mean() / signal_returns.std() if signal_returns.std() > 0 else 0
                
                # Turnover (from component weights if available)
                turnover = 0
                if not result.component_weights.empty:
                    weight_changes = result.component_weights.diff().abs()
                    turnover = weight_changes.sum(axis=1).mean()
                
                # Performance metrics
                perf_metrics = {
                    'method': method_name,
                    'information_coefficient': ic,
                    'sharpe_ratio': sharpe,
                    'turnover': turnover,
                    'num_signals': len(result.combination_metadata.get('signal_names', [])),
                    'avg_combined_signal': aligned_signal.mean(),
                    'combined_signal_std': aligned_signal.std()
                }
                
                performance_data.append(perf_metrics)
            
            performance_df = pd.DataFrame(performance_data)
            
            if not performance_df.empty:
                # Sort by IC (absolute value)
                performance_df['abs_ic'] = performance_df['information_coefficient'].abs()
                performance_df = performance_df.sort_values('abs_ic', ascending=False)
                performance_df = performance_df.drop('abs_ic', axis=1)
            
            return performance_df
            
        except Exception as e:
            logger.error(f"Error evaluating combination performance: {str(e)}")
            return pd.DataFrame()
    
    def select_best_combination_method(self, performance_df: pd.DataFrame,
                                     criteria: str = 'ic') -> str:
        """
        Select best combination method based on performance criteria.
        
        Parameters:
        -----------
        performance_df : pd.DataFrame
            Performance comparison results
        criteria : str
            Selection criteria ('ic', 'sharpe', 'composite')
            
        Returns:
        --------
        str
            Name of best performing method
        """
        try:
            if performance_df.empty:
                return 'equal_weight'  # Default fallback
            
            if criteria == 'ic':
                best_method = performance_df.loc[performance_df['information_coefficient'].abs().idxmax(), 'method']
            elif criteria == 'sharpe':
                best_method = performance_df.loc[performance_df['sharpe_ratio'].idxmax(), 'method']
            elif criteria == 'composite':
                # Composite score: IC + Sharpe - Turnover penalty
                composite_score = (
                    performance_df['information_coefficient'].abs() +
                    performance_df['sharpe_ratio'] -
                    performance_df['turnover'] * 0.1
                )
                best_method = performance_df.loc[composite_score.idxmax(), 'method']
            else:
                best_method = performance_df.iloc[0]['method']  # First in sorted list
            
            return best_method
            
        except Exception as e:
            logger.warning(f"Error selecting best method: {str(e)}")
            return 'equal_weight'
    
    def generate_combination_report(self, alpha_signals: Dict[str, pd.Series],
                                  returns: pd.Series,
                                  strategy_name: str = "Strategy") -> Dict[str, Any]:
        """
        Generate comprehensive alpha combination report.
        
        Parameters:
        -----------
        alpha_signals : Dict[str, pd.Series]
            Dictionary of alpha signals
        returns : pd.Series
            Forward returns for evaluation
        strategy_name : str
            Strategy name for reporting
            
        Returns:
        --------
        Dict[str, Any]
            Comprehensive combination report
        """
        try:
            # Run all combination methods
            combination_results = self.combine_alphas_all_methods(alpha_signals, returns)
            
            # Evaluate performance
            performance_df = self.evaluate_combination_performance(combination_results, returns)
            
            # Select best method
            best_method = self.select_best_combination_method(performance_df)
            
            # Create report
            report = {
                'strategy_name': strategy_name,
                'analysis_date': datetime.now(),
                'input_signals': {
                    'num_signals': len(alpha_signals),
                    'signal_names': list(alpha_signals.keys()),
                    'signal_period': {
                        'start': min(signal.index.min() for signal in alpha_signals.values()),
                        'end': max(signal.index.max() for signal in alpha_signals.values())
                    }
                },
                'combination_methods': {
                    'methods_tested': list(combination_results.keys()),
                    'best_method': best_method,
                    'performance_comparison': performance_df.to_dict('records') if not performance_df.empty else []
                },
                'best_combination_details': combination_results.get(best_method, {}).to_dict() if best_method in combination_results else {},
                'recommendations': self._generate_combination_recommendations(performance_df, combination_results),
                'implementation_notes': self._generate_implementation_notes(best_method, combination_results.get(best_method))
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating combination report: {str(e)}")
            raise
    
    def _generate_combination_recommendations(self, performance_df: pd.DataFrame,
                                            combination_results: Dict[str, CombinedAlphaSignal]) -> List[str]:
        """Generate recommendations based on combination analysis."""
        recommendations = []
        
        if performance_df.empty:
            recommendations.append("No valid combination results. Check input signals and data quality.")
            return recommendations
        
        # Best method analysis
        best_ic = performance_df['information_coefficient'].abs().max()
        if best_ic > 0.05:
            recommendations.append("Strong combined alpha signals achieved. Consider implementation with full position sizing.")
        elif best_ic > 0.02:
            recommendations.append("Moderate combined alpha strength. Consider conservative position sizing or further signal enhancement.")
        else:
            recommendations.append("Weak combined alpha signals. Review individual signal quality and combination methodology.")
        
        # Turnover analysis
        avg_turnover = performance_df['turnover'].mean()
        if avg_turnover > 2.0:
            recommendations.append("High turnover detected in combination methods. Factor in transaction costs and consider lower frequency rebalancing.")
        
        # Method diversity
        if len(performance_df) > 1:
            ic_range = performance_df['information_coefficient'].max() - performance_df['information_coefficient'].min()
            if ic_range > 0.02:
                recommendations.append("Significant performance differences between methods. Consider ensemble of top-performing methods.")
        
        # Stability analysis
        if 'ml_random_forest' in combination_results:
            ml_result = combination_results['ml_random_forest']
            weight_stability = ml_result.performance_metrics.get('weight_stability', 0)
            if weight_stability < 0.5:
                recommendations.append("ML combination shows unstable weights. Consider more regularization or simpler methods.")
        
        return recommendations
    
    def _generate_implementation_notes(self, best_method: str, 
                                     best_result: Optional[CombinedAlphaSignal]) -> List[str]:
        """Generate implementation notes for best combination method."""
        notes = []
        
        if best_result is None:
            notes.append("No valid combination result available for implementation.")
            return notes
        
        metadata = best_result.combination_metadata
        
        # Dynamic weights
        if metadata.get('dynamic_weights', False):
            notes.append("Selected method uses dynamic weights. Implement rebalancing mechanism.")
        
        # Optimization requirements
        if metadata.get('optimization_required', False):
            notes.append("Method requires optimization. Ensure sufficient computational resources and consider optimization frequency.")
        
        # ML approach
        if metadata.get('ml_approach', False):
            retraining_freq = metadata.get('model_retraining_frequency', 30)
            notes.append(f"ML-based method requires model retraining every {retraining_freq} periods.")
        
        # Risk considerations
        if 'risk_budget' in best_method:
            notes.append("Risk-based method selected. Monitor risk contributions and ensure diversification benefits.")
        
        # Performance expectations
        perf_metrics = best_result.performance_metrics
        expected_ic = perf_metrics.get('information_coefficient', 0)
        notes.append(f"Expected Information Coefficient: {expected_ic:.3f}")
        
        return notes

