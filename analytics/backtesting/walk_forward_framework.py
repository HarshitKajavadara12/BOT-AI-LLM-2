"""
Walk-Forward Analysis Framework
Advanced walk-forward optimization and validation for trading strategies
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
import concurrent.futures
from itertools import product
import copy
import pickle
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error
from scipy import optimize


class OptimizationMetric(Enum):
    """Optimization metrics for walk-forward analysis"""
    SHARPE_RATIO = "SHARPE_RATIO"
    TOTAL_RETURN = "TOTAL_RETURN"
    CALMAR_RATIO = "CALMAR_RATIO"
    SORTINO_RATIO = "SORTINO_RATIO"
    MAX_DRAWDOWN = "MAX_DRAWDOWN"
    WIN_RATE = "WIN_RATE"
    PROFIT_FACTOR = "PROFIT_FACTOR"
    CUSTOM = "CUSTOM"


class RebalanceFrequency(Enum):
    """Rebalancing frequency options"""
    DAILY = "DAILY"
    WEEKLY = "WEEKLY"
    MONTHLY = "MONTHLY"
    QUARTERLY = "QUARTERLY"
    SEMI_ANNUAL = "SEMI_ANNUAL"
    ANNUAL = "ANNUAL"


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward analysis"""
    
    # Window settings
    training_window_days: int = 252         # Training period length
    validation_window_days: int = 63        # Out-of-sample validation period
    step_size_days: int = 21               # Step size between windows
    
    # Optimization settings
    optimization_metric: OptimizationMetric = OptimizationMetric.SHARPE_RATIO
    rebalance_frequency: RebalanceFrequency = RebalanceFrequency.MONTHLY
    min_observations: int = 100             # Minimum observations required
    
    # Strategy settings
    initial_capital: float = 100000
    transaction_costs: float = 0.001
    position_sizing_method: str = "equal_weight"  # equal_weight, volatility_target, etc.
    max_position_size: float = 0.1          # Max position size as fraction of capital
    
    # Performance settings
    benchmark_symbol: Optional[str] = None   # Benchmark for comparison
    risk_free_rate: float = 0.02           # Annual risk-free rate
    
    # Parallelization
    n_jobs: int = 1                        # Number of parallel jobs
    random_state: int = 42


@dataclass
class ParameterSet:
    """Set of parameters for strategy optimization"""
    parameters: Dict[str, Any]
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    equity_curve: Optional[pd.Series] = None
    trades: Optional[pd.DataFrame] = None
    
    def __hash__(self):
        """Make parameter set hashable"""
        return hash(tuple(sorted(self.parameters.items())))


@dataclass
class WalkForwardResult:
    """Results from walk-forward analysis"""
    
    # Overall results
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    volatility: float = 0.0
    calmar_ratio: float = 0.0
    
    # Walk-forward specific metrics
    consistency_ratio: float = 0.0          # Fraction of profitable periods
    parameter_stability: float = 0.0       # Stability of optimal parameters
    out_of_sample_correlation: float = 0.0 # IS vs OOS correlation
    
    # Detailed results
    period_results: List[Dict[str, Any]] = field(default_factory=list) 
    optimal_parameters_history: List[Dict[str, Any]] = field(default_factory=list)
    equity_curve: Optional[pd.Series] = None
    
    # Benchmark comparison
    excess_return: float = 0.0
    information_ratio: float = 0.0
    beta: float = 0.0
    alpha: float = 0.0


class TradingStrategy(ABC):
    """Abstract base class for trading strategies"""
    
    @abstractmethod
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """Set strategy parameters"""
        pass
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals"""
        pass
    
    @abstractmethod
    def get_parameter_space(self) -> Dict[str, List[Any]]:
        """Get parameter space for optimization"""
        pass
    
    @abstractmethod
    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters"""
        pass


class MovingAverageCrossoverStrategy(TradingStrategy):
    """Simple moving average crossover strategy for testing"""
    
    def __init__(self):
        self.short_window = 10
        self.long_window = 30
        self.signal_threshold = 0.01
        
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """Set strategy parameters"""
        self.short_window = parameters.get('short_window', self.short_window)
        self.long_window = parameters.get('long_window', self.long_window)
        self.signal_threshold = parameters.get('signal_threshold', self.signal_threshold)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate MA crossover signals"""
        
        signals = pd.DataFrame(index=data.index)
        
        # Get price columns
        price_cols = [col for col in data.columns if col.endswith('_close') or col == 'close']
        
        for col in price_cols:
            symbol = col.replace('_close', '') if '_close' in col else 'asset'
            
            if len(data) < self.long_window:
                signals[f'{symbol}_signal'] = 0.0
                continue
            
            # Calculate moving averages
            short_ma = data[col].rolling(window=self.short_window).mean()
            long_ma = data[col].rolling(window=self.long_window).mean()
            
            # Generate signals
            signal = np.where(short_ma > long_ma * (1 + self.signal_threshold), 1.0,
                            np.where(short_ma < long_ma * (1 - self.signal_threshold), -1.0, 0.0))
            
            signals[f'{symbol}_signal'] = signal
        
        return signals
    
    def get_parameter_space(self) -> Dict[str, List[Any]]:
        """Get parameter space for optimization"""
        return {
            'short_window': [5, 10, 15, 20],
            'long_window': [20, 30, 50, 100],
            'signal_threshold': [0.0, 0.005, 0.01, 0.02]
        }
    
    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters"""
        return {
            'short_window': 10,
            'long_window': 30,
            'signal_threshold': 0.01
        }


class MomentumStrategy(TradingStrategy):
    """Momentum strategy for testing"""
    
    def __init__(self):
        self.lookback_window = 20
        self.holding_period = 5
        self.momentum_threshold = 0.05
        
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """Set strategy parameters"""
        self.lookback_window = parameters.get('lookback_window', self.lookback_window)
        self.holding_period = parameters.get('holding_period', self.holding_period)
        self.momentum_threshold = parameters.get('momentum_threshold', self.momentum_threshold)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate momentum signals"""
        
        signals = pd.DataFrame(index=data.index)
        
        # Get price columns
        price_cols = [col for col in data.columns if col.endswith('_close') or col == 'close']
        
        for col in price_cols:
            symbol = col.replace('_close', '') if '_close' in col else 'asset'
            
            if len(data) < self.lookback_window + self.holding_period:
                signals[f'{symbol}_signal'] = 0.0
                continue
            
            # Calculate momentum
            momentum = data[col].pct_change(self.lookback_window)
            
            # Generate signals with holding period
            raw_signals = np.where(momentum > self.momentum_threshold, 1.0,
                                 np.where(momentum < -self.momentum_threshold, -1.0, 0.0))
            
            # Apply holding period (simplified)
            held_signals = pd.Series(raw_signals, index=data.index).fillna(0)
            signals[f'{symbol}_signal'] = held_signals
        
        return signals
    
    def get_parameter_space(self) -> Dict[str, List[Any]]:
        """Get parameter space for optimization"""
        return {
            'lookback_window': [10, 20, 30, 60],
            'holding_period': [1, 5, 10, 20],
            'momentum_threshold': [0.02, 0.05, 0.1, 0.15]
        }
    
    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters"""
        return {
            'lookback_window': 20,
            'holding_period': 5,
            'momentum_threshold': 0.05
        }


class PerformanceCalculator:
    """Calculate performance metrics for walk-forward analysis"""
    
    @staticmethod
    def calculate_returns(equity_curve: pd.Series) -> pd.Series:
        """Calculate returns from equity curve"""
        return equity_curve.pct_change().fillna(0)
    
    @staticmethod
    def calculate_metrics(equity_curve: pd.Series, 
                         benchmark: Optional[pd.Series] = None,
                         risk_free_rate: float = 0.02) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        
        if len(equity_curve) < 2:
            return {}
        
        returns = PerformanceCalculator.calculate_returns(equity_curve)
        
        # Basic metrics
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1)
        
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe_ratio = (np.mean(returns) - risk_free_rate/252) / np.std(returns) * np.sqrt(252)
            volatility = np.std(returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
            volatility = 0.0
        
        # Drawdown
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Downside deviation for Sortino ratio
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0:
            downside_deviation = np.std(negative_returns) * np.sqrt(252)
            sortino_ratio = (np.mean(returns) - risk_free_rate/252) / np.std(negative_returns) * np.sqrt(252)
        else:
            downside_deviation = 0.0
            sortino_ratio = 0.0
        
        # Win rate
        win_rate = np.sum(returns > 0) / len(returns)
        
        # Profit factor
        winning_returns = returns[returns > 0]
        losing_returns = returns[returns < 0]
        profit_factor = (np.sum(winning_returns) / abs(np.sum(losing_returns)) 
                        if len(losing_returns) > 0 and np.sum(losing_returns) != 0 else 0)
        
        metrics = {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'downside_deviation': downside_deviation
        }
        
        # Benchmark comparison
        if benchmark is not None and len(benchmark) == len(equity_curve):
            benchmark_returns = PerformanceCalculator.calculate_returns(benchmark)
            
            if len(benchmark_returns) > 0 and np.std(benchmark_returns) > 0:
                # Excess return
                excess_returns = returns - benchmark_returns
                excess_return = np.mean(excess_returns) * 252
                
                # Information ratio
                information_ratio = (np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) 
                                   if np.std(excess_returns) > 0 else 0)
                
                # Beta and Alpha
                covariance = np.cov(returns, benchmark_returns)[0, 1]
                benchmark_variance = np.var(benchmark_returns)
                beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
                alpha = (np.mean(returns) - beta * np.mean(benchmark_returns)) * 252
                
                metrics.update({
                    'excess_return': excess_return,
                    'information_ratio': information_ratio,
                    'beta': beta,
                    'alpha': alpha
                })
        
        return metrics


class BacktestEngine:
    """Simple backtesting engine for walk-forward analysis"""
    
    def __init__(self, initial_capital: float = 100000, 
                 transaction_costs: float = 0.001):
        self.initial_capital = initial_capital
        self.transaction_costs = transaction_costs
    
    def run_backtest(self, data: pd.DataFrame, signals: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Run backtest given data and signals
        
        Returns:
            equity_curve: Portfolio value over time
            trades: DataFrame of executed trades
        """
        
        if data.empty or signals.empty:
            return pd.Series([self.initial_capital]), pd.DataFrame()
        
        # Align data and signals
        common_index = data.index.intersection(signals.index)
        if len(common_index) < 2:
            return pd.Series([self.initial_capital]), pd.DataFrame()
        
        data = data.loc[common_index]
        signals = signals.loc[common_index]
        
        # Get price and signal columns
        price_cols = [col for col in data.columns if col.endswith('_close') or col == 'close']
        signal_cols = [col for col in signals.columns if col.endswith('_signal')]
        
        if not price_cols or not signal_cols:
            return pd.Series([self.initial_capital]), pd.DataFrame()
        
        # Initialize portfolio
        portfolio_value = self.initial_capital
        cash = self.initial_capital
        positions = {col.replace('_close', ''): 0.0 for col in price_cols}
        
        equity_curve = []
        trades = []
        
        for i, (date, row) in enumerate(data.iterrows()):
            
            # Get current prices and signals
            current_prices = {}
            current_signals = {}
            
            for price_col in price_cols:
                symbol = price_col.replace('_close', '') if '_close' in price_col else 'asset'
                price = row[price_col]
                
                if pd.notna(price) and price > 0:
                    current_prices[symbol] = price
                
                # Find corresponding signal
                signal_col = f'{symbol}_signal'
                if signal_col in signals.columns:
                    signal = signals.loc[date, signal_col]
                    if pd.notna(signal):
                        current_signals[symbol] = signal
            
            # Execute trades
            for symbol in current_prices:
                if symbol in current_signals and symbol in positions:
                    
                    current_price = current_prices[symbol]
                    target_signal = current_signals[symbol]
                    current_position = positions[symbol]
                    
                    # Simple position sizing: equal weight across signals
                    n_active_signals = sum(1 for s in current_signals.values() if abs(s) > 0.01)
                    
                    if n_active_signals > 0:
                        position_size_per_signal = portfolio_value * 0.8 / n_active_signals  # 80% invested
                        target_position_value = target_signal * position_size_per_signal
                        target_shares = target_position_value / current_price
                    else:
                        target_shares = 0.0
                    
                    # Calculate trade
                    trade_shares = target_shares - current_position
                    
                    if abs(trade_shares) > 0.01:  # Minimum trade size
                        trade_value = trade_shares * current_price
                        trade_cost = abs(trade_value) * self.transaction_costs
                        
                        # Check cash availability
                        if trade_value + trade_cost <= cash:
                            # Execute trade
                            positions[symbol] = target_shares
                            cash -= trade_value + trade_cost
                            
                            # Record trade
                            trades.append({
                                'date': date,
                                'symbol': symbol,
                                'shares': trade_shares,
                                'price': current_price,
                                'value': trade_value,
                                'cost': trade_cost
                            })
            
            # Calculate portfolio value
            positions_value = sum(positions[symbol] * current_prices.get(symbol, 0) 
                                for symbol in positions)
            portfolio_value = cash + positions_value
            
            equity_curve.append(portfolio_value)
        
        equity_series = pd.Series(equity_curve, index=common_index)
        trades_df = pd.DataFrame(trades)
        
        return equity_series, trades_df


class WalkForwardAnalyzer:
    """
    Main walk-forward analysis framework
    """
    
    def __init__(self, strategy: TradingStrategy, config: WalkForwardConfig):
        self.strategy = strategy
        self.config = config
        self.backtest_engine = BacktestEngine(
            initial_capital=config.initial_capital,
            transaction_costs=config.transaction_costs
        )
        
        # Results storage
        self.results: Optional[WalkForwardResult] = None
        self.optimization_history: List[Dict[str, Any]] = []
    
    def run_analysis(self, data: pd.DataFrame, 
                    benchmark_data: Optional[pd.DataFrame] = None) -> WalkForwardResult:
        """Run complete walk-forward analysis"""
        
        if data.empty:
            raise ValueError("Data cannot be empty")
        
        print(f"Starting walk-forward analysis...")
        print(f"  Data period: {data.index[0]} to {data.index[-1]}")
        print(f"  Training window: {self.config.training_window_days} days")
        print(f"  Validation window: {self.config.validation_window_days} days")
        print(f"  Step size: {self.config.step_size_days} days")
        
        # Generate walk-forward windows
        windows = self._generate_windows(data)
        print(f"  Generated {len(windows)} walk-forward windows")
        
        if len(windows) == 0:
            raise ValueError("No valid walk-forward windows generated")
        
        # Run optimization for each window
        period_results = []
        optimal_parameters_history = []
        out_of_sample_equity_curves = []
        
        for i, (train_start, train_end, val_start, val_end) in enumerate(windows):
            print(f"\nProcessing window {i+1}/{len(windows)}")
            print(f"  Training: {train_start} to {train_end}")
            print(f"  Validation: {val_start} to {val_end}")
            
            try:
                # Run single window analysis
                window_result = self._analyze_window(
                    data, train_start, train_end, val_start, val_end
                )
                
                period_results.append(window_result)
                optimal_parameters_history.append(window_result['optimal_parameters'])
                
                if 'oos_equity_curve' in window_result:
                    out_of_sample_equity_curves.append(window_result['oos_equity_curve'])
                
                print(f"  IS {self.config.optimization_metric.value}: {window_result['is_performance']:.4f}")
                print(f"  OOS {self.config.optimization_metric.value}: {window_result['oos_performance']:.4f}")
                
            except Exception as e:
                print(f"  Window {i+1} failed: {e}")
                continue
        
        if not period_results:
            raise RuntimeError("No successful walk-forward windows")
        
        print(f"\nCompleted {len(period_results)} windows successfully")
        
        # Combine out-of-sample results
        combined_equity_curve = self._combine_equity_curves(out_of_sample_equity_curves)
        
        # Calculate overall metrics
        overall_metrics = PerformanceCalculator.calculate_metrics(
            combined_equity_curve, 
            benchmark=benchmark_data,
            risk_free_rate=self.config.risk_free_rate
        )
        
        # Calculate walk-forward specific metrics
        wf_metrics = self._calculate_walkforward_metrics(period_results, optimal_parameters_history)
        
        # Create result object
        self.results = WalkForwardResult(
            total_return=overall_metrics.get('total_return', 0.0),
            sharpe_ratio=overall_metrics.get('sharpe_ratio', 0.0),
            max_drawdown=overall_metrics.get('max_drawdown', 0.0),
            volatility=overall_metrics.get('volatility', 0.0),
            calmar_ratio=overall_metrics.get('calmar_ratio', 0.0),
            consistency_ratio=wf_metrics.get('consistency_ratio', 0.0),
            parameter_stability=wf_metrics.get('parameter_stability', 0.0),
            out_of_sample_correlation=wf_metrics.get('oos_correlation', 0.0),
            period_results=period_results,
            optimal_parameters_history=optimal_parameters_history,
            equity_curve=combined_equity_curve,
            excess_return=overall_metrics.get('excess_return', 0.0),
            information_ratio=overall_metrics.get('information_ratio', 0.0),
            beta=overall_metrics.get('beta', 0.0),
            alpha=overall_metrics.get('alpha', 0.0)
        )
        
        print(f"\nWalk-Forward Analysis Results:")
        print(f"  Total Return: {self.results.total_return:.2%}")
        print(f"  Sharpe Ratio: {self.results.sharpe_ratio:.2f}")
        print(f"  Max Drawdown: {self.results.max_drawdown:.2%}")
        print(f"  Consistency Ratio: {self.results.consistency_ratio:.2%}")
        print(f"  Parameter Stability: {self.results.parameter_stability:.2f}")
        print(f"  IS/OOS Correlation: {self.results.out_of_sample_correlation:.2f}")
        
        return self.results
    
    def _generate_windows(self, data: pd.DataFrame) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """Generate walk-forward windows"""
        
        windows = []
        
        start_date = data.index[0]
        end_date = data.index[-1]
        
        current_start = start_date
        
        while True:
            # Training window
            train_start = current_start
            train_end_idx = data.index.get_loc(train_start) + self.config.training_window_days - 1
            
            if train_end_idx >= len(data):
                break
            
            train_end = data.index[train_end_idx]
            
            # Validation window
            val_start_idx = train_end_idx + 1
            val_end_idx = val_start_idx + self.config.validation_window_days - 1
            
            if val_end_idx >= len(data):
                break
            
            val_start = data.index[val_start_idx]
            val_end = data.index[val_end_idx]
            
            windows.append((train_start, train_end, val_start, val_end))
            
            # Move to next window
            next_start_idx = data.index.get_loc(current_start) + self.config.step_size_days
            
            if next_start_idx >= len(data) - self.config.training_window_days - self.config.validation_window_days:
                break
            
            current_start = data.index[next_start_idx]
        
        return windows
    
    def _analyze_window(self, data: pd.DataFrame, 
                       train_start: pd.Timestamp, train_end: pd.Timestamp,
                       val_start: pd.Timestamp, val_end: pd.Timestamp) -> Dict[str, Any]:
        """Analyze single walk-forward window"""
        
        # Split data
        train_data = data.loc[train_start:train_end]
        val_data = data.loc[val_start:val_end]
        
        if len(train_data) < self.config.min_observations:
            raise ValueError(f"Insufficient training data: {len(train_data)} < {self.config.min_observations}")
        
        # Optimize parameters on training data
        optimal_params, is_metrics = self._optimize_parameters(train_data)
        
        # Test on validation data
        self.strategy.set_parameters(optimal_params)
        val_signals = self.strategy.generate_signals(val_data)
        oos_equity_curve, oos_trades = self.backtest_engine.run_backtest(val_data, val_signals)
        
        # Calculate out-of-sample metrics
        oos_metrics = PerformanceCalculator.calculate_metrics(
            oos_equity_curve, risk_free_rate=self.config.risk_free_rate
        )
        
        # Extract optimization metric value
        metric_name = self.config.optimization_metric.value.lower()
        is_performance = is_metrics.get(metric_name, 0.0)
        oos_performance = oos_metrics.get(metric_name, 0.0)
        
        return {
            'train_start': train_start,
            'train_end': train_end,
            'val_start': val_start,
            'val_end': val_end,
            'optimal_parameters': optimal_params,
            'is_performance': is_performance,
            'oos_performance': oos_performance,
            'is_metrics': is_metrics,
            'oos_metrics': oos_metrics,
            'oos_equity_curve': oos_equity_curve,
            'oos_trades': oos_trades
        }
    
    def _optimize_parameters(self, train_data: pd.DataFrame) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """Optimize strategy parameters on training data"""
        
        # Get parameter space
        param_space = self.strategy.get_parameter_space()
        
        if not param_space:
            # No parameters to optimize
            default_params = self.strategy.get_default_parameters()
            self.strategy.set_parameters(default_params)
            
            signals = self.strategy.generate_signals(train_data)
            equity_curve, _ = self.backtest_engine.run_backtest(train_data, signals)
            metrics = PerformanceCalculator.calculate_metrics(
                equity_curve, risk_free_rate=self.config.risk_free_rate
            )
            
            return default_params, metrics
        
        # Generate parameter combinations
        param_grid = list(ParameterGrid(param_space))
        
        if len(param_grid) > 1000:  # Limit parameter combinations
            # Deterministic subsample of the parameter grid: evenly spaced selection
            step = max(1, len(param_grid) // 1000)
            param_grid = [param_grid[i] for i in range(0, len(param_grid), step)][:1000]
        
        print(f"    Optimizing over {len(param_grid)} parameter combinations...")
        
        # Optimize parameters
        best_params = None
        best_score = -np.inf
        best_metrics = {}
        
        # Use parallel processing if configured
        if self.config.n_jobs > 1:
            results = self._optimize_parallel(train_data, param_grid)
        else:
            results = []
            for params in param_grid:
                try:
                    result = self._evaluate_parameters(train_data, params)
                    results.append(result)
                except Exception as e:
                    continue
        
        # Find best parameters
        metric_name = self.config.optimization_metric.value.lower()
        
        for params, metrics in results:
            score = metrics.get(metric_name, -np.inf)
            
            # Handle max drawdown (lower is better)
            if self.config.optimization_metric == OptimizationMetric.MAX_DRAWDOWN:
                score = -score  # Convert to maximization problem
            
            if score > best_score:
                best_score = score
                best_params = params
                best_metrics = metrics
        
        if best_params is None:
            # Fallback to default parameters
            best_params = self.strategy.get_default_parameters()
            self.strategy.set_parameters(best_params)
            
            signals = self.strategy.generate_signals(train_data)
            equity_curve, _ = self.backtest_engine.run_backtest(train_data, signals)
            best_metrics = PerformanceCalculator.calculate_metrics(
                equity_curve, risk_free_rate=self.config.risk_free_rate
            )
        
        return best_params, best_metrics
    
    def _optimize_parallel(self, train_data: pd.DataFrame, 
                          param_grid: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Dict[str, float]]]:
        """Optimize parameters in parallel"""
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.config.n_jobs) as executor:
            futures = [executor.submit(self._evaluate_parameters, train_data, params) 
                      for params in param_grid]
            
            results = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result(timeout=60)  # 60 second timeout
                    results.append(result)
                except Exception as e:
                    continue
        
        return results
    
    def _evaluate_parameters(self, train_data: pd.DataFrame, 
                           parameters: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """Evaluate parameter set on training data"""
        
        # Create a copy of strategy to avoid threading issues
        strategy_copy = copy.deepcopy(self.strategy)
        strategy_copy.set_parameters(parameters)
        
        # Generate signals
        signals = strategy_copy.generate_signals(train_data)
        
        # Run backtest
        equity_curve, _ = self.backtest_engine.run_backtest(train_data, signals)
        
        # Calculate metrics
        metrics = PerformanceCalculator.calculate_metrics(
            equity_curve, risk_free_rate=self.config.risk_free_rate
        )
        
        return parameters, metrics
    
    def _combine_equity_curves(self, equity_curves: List[pd.Series]) -> pd.Series:
        """Combine out-of-sample equity curves"""
        
        if not equity_curves:
            return pd.Series([self.config.initial_capital])
        
        # Concatenate curves, scaling each to start where previous ended
        combined_values = []
        combined_dates = []
        current_value = self.config.initial_capital
        
        for curve in equity_curves:
            if len(curve) > 0:
                # Scale curve to start at current value
                curve_returns = curve.pct_change().fillna(0)
                scaled_curve = current_value * (1 + curve_returns).cumprod()
                
                combined_values.extend(scaled_curve.values)
                combined_dates.extend(curve.index)
                
                current_value = scaled_curve.iloc[-1]
        
        if not combined_values:
            return pd.Series([self.config.initial_capital])
        
        return pd.Series(combined_values, index=combined_dates)
    
    def _calculate_walkforward_metrics(self, period_results: List[Dict[str, Any]], 
                                     parameter_history: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate walk-forward specific metrics"""
        
        metrics = {}
        
        # Consistency ratio (fraction of profitable periods)
        oos_returns = []
        for result in period_results:
            if 'oos_metrics' in result and 'total_return' in result['oos_metrics']:
                oos_returns.append(result['oos_metrics']['total_return'])
        
        if oos_returns:
            profitable_periods = sum(1 for r in oos_returns if r > 0)
            metrics['consistency_ratio'] = profitable_periods / len(oos_returns)
        else:
            metrics['consistency_ratio'] = 0.0
        
        # Parameter stability
        if len(parameter_history) > 1:
            stability_scores = []
            
            for param_name in parameter_history[0].keys():
                param_values = [params.get(param_name) for params in parameter_history]
                param_values = [v for v in param_values if v is not None]
                
                if len(param_values) > 1 and all(isinstance(v, (int, float)) for v in param_values):
                    # Coefficient of variation (lower is more stable)
                    cv = np.std(param_values) / np.mean(param_values) if np.mean(param_values) != 0 else 1
                    stability_scores.append(1 - min(cv, 1))  # Convert to stability score
            
            if stability_scores:
                metrics['parameter_stability'] = np.mean(stability_scores)
            else:
                metrics['parameter_stability'] = 0.0
        else:
            metrics['parameter_stability'] = 1.0
        
        # In-sample vs Out-of-sample correlation
        is_scores = []
        oos_scores = []
        
        metric_name = self.config.optimization_metric.value.lower()
        
        for result in period_results:
            if ('is_metrics' in result and 'oos_metrics' in result and 
                metric_name in result['is_metrics'] and metric_name in result['oos_metrics']):
                
                is_scores.append(result['is_metrics'][metric_name])
                oos_scores.append(result['oos_metrics'][metric_name])
        
        if len(is_scores) > 1 and len(oos_scores) > 1:
            correlation = np.corrcoef(is_scores, oos_scores)[0, 1]
            metrics['oos_correlation'] = correlation if not np.isnan(correlation) else 0.0
        else:
            metrics['oos_correlation'] = 0.0
        
        return metrics
    
    def generate_report(self) -> str:
        """Generate detailed walk-forward analysis report"""
        
        if self.results is None:
            return "No results available. Run analysis first."
        
        report = []
        report.append("="*60)
        report.append("WALK-FORWARD ANALYSIS REPORT")
        report.append("="*60)
        
        # Overall performance
        report.append("\nOVERALL PERFORMANCE:")
        report.append(f"  Total Return: {self.results.total_return:.2%}")
        report.append(f"  Sharpe Ratio: {self.results.sharpe_ratio:.2f}")
        report.append(f"  Calmar Ratio: {self.results.calmar_ratio:.2f}")
        report.append(f"  Maximum Drawdown: {self.results.max_drawdown:.2%}")
        report.append(f"  Volatility: {self.results.volatility:.2%}")
        
        # Walk-forward specific metrics
        report.append("\nWALK-FORWARD METRICS:")
        report.append(f"  Consistency Ratio: {self.results.consistency_ratio:.2%}")
        report.append(f"  Parameter Stability: {self.results.parameter_stability:.2f}")
        report.append(f"  IS/OOS Correlation: {self.results.out_of_sample_correlation:.2f}")
        
        # Benchmark comparison
        if self.results.excess_return != 0:
            report.append("\nBENCHMARK COMPARISON:")
            report.append(f"  Excess Return: {self.results.excess_return:.2%}")
            report.append(f"  Information Ratio: {self.results.information_ratio:.2f}")
            report.append(f"  Beta: {self.results.beta:.2f}")
            report.append(f"  Alpha: {self.results.alpha:.2%}")
        
        # Period-by-period results
        report.append(f"\nPERIOD RESULTS ({len(self.results.period_results)} periods):")
        report.append("  Period          IS Perf    OOS Perf   Difference")
        report.append("  " + "-"*50)
        
        for i, result in enumerate(self.results.period_results[-10:]):  # Show last 10
            period_str = f"{result['val_start'].strftime('%Y-%m')} to {result['val_end'].strftime('%Y-%m')}"
            is_perf = result['is_performance']
            oos_perf = result['oos_performance']
            diff = oos_perf - is_perf
            
            report.append(f"  {period_str:<15} {is_perf:>8.3f}  {oos_perf:>8.3f}  {diff:>+8.3f}")
        
        # Parameter stability analysis
        if len(self.results.optimal_parameters_history) > 1:
            report.append("\nPARAMETER STABILITY:")
            
            param_names = list(self.results.optimal_parameters_history[0].keys())
            
            for param_name in param_names:
                param_values = [params.get(param_name) for params in self.results.optimal_parameters_history]
                param_values = [v for v in param_values if v is not None]
                
                if len(param_values) > 1 and all(isinstance(v, (int, float)) for v in param_values):
                    mean_val = np.mean(param_values)
                    std_val = np.std(param_values)
                    min_val = np.min(param_values)
                    max_val = np.max(param_values)
                    
                    report.append(f"  {param_name}: Mean={mean_val:.3f}, Std={std_val:.3f}, Range=[{min_val:.3f}, {max_val:.3f}]")
        
        return "\n".join(report)
