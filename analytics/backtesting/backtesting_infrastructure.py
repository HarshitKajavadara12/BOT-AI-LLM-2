"""
Backtesting Infrastructure for QUANTUM-FORGE
Implements sophisticated backtesting framework with realistic market simulation,
transaction costs, slippage modeling, and comprehensive performance analysis.
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import warnings
from dataclasses import dataclass, field
from enum import Enum
import time
from datetime import datetime, timedelta
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
from abc import ABC, abstractmethod
warnings.filterwarnings('ignore')

class OrderType(Enum):
    """Order types for backtesting."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    MARKET_ON_CLOSE = "market_on_close"
    LIMIT_ON_CLOSE = "limit_on_close"

class OrderSide(Enum):
    """Order sides."""
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

class SlippageModel(Enum):
    """Slippage modeling approaches."""
    FIXED_BASIS_POINTS = "fixed_bps"
    VOLUME_PARTICIPATION = "volume_participation"
    PRICE_IMPACT = "price_impact"
    SQUARE_ROOT = "square_root"
    LINEAR = "linear"

@dataclass
class Order:
    """Trading order representation."""
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType
    timestamp: pd.Timestamp
    price: Optional[float] = None
    stop_price: Optional[float] = None
    order_id: str = field(default_factory=lambda: str(time.time_ns()))
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: Optional[float] = None
    commission: float = 0.0
    slippage: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Trade:
    """Executed trade representation."""
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: pd.Timestamp
    commission: float = 0.0
    slippage: float = 0.0
    order_id: str = ""
    trade_id: str = field(default_factory=lambda: str(time.time_ns()))
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Position:
    """Portfolio position representation."""
    symbol: str
    quantity: float
    average_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float = 0.0
    first_trade_time: Optional[pd.Timestamp] = None
    last_trade_time: Optional[pd.Timestamp] = None

@dataclass
class PortfolioSnapshot:
    """Portfolio state snapshot."""
    timestamp: pd.Timestamp
    cash: float
    total_value: float
    positions: Dict[str, Position]
    daily_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

class CommissionModel(ABC):
    """Abstract commission model."""
    
    @abstractmethod
    def calculate_commission(self, trade: Trade) -> float:
        """Calculate commission for a trade."""
        pass

class FixedCommissionModel(CommissionModel):
    """Fixed commission per trade."""
    
    def __init__(self, commission_per_trade: float):
        """Initialize fixed commission model."""
        self.commission_per_trade = commission_per_trade
    
    def calculate_commission(self, trade: Trade) -> float:
        """Calculate fixed commission."""
        return self.commission_per_trade

class PercentageCommissionModel(CommissionModel):
    """Percentage-based commission."""
    
    def __init__(self, commission_rate: float, min_commission: float = 0.0):
        """Initialize percentage commission model."""
        self.commission_rate = commission_rate
        self.min_commission = min_commission
    
    def calculate_commission(self, trade: Trade) -> float:
        """Calculate percentage-based commission."""
        commission = abs(trade.quantity * trade.price * self.commission_rate)
        return max(commission, self.min_commission)

class TieredCommissionModel(CommissionModel):
    """Tiered commission structure."""
    
    def __init__(self, tiers: List[Tuple[float, float]]):
        """
        Initialize tiered commission model.
        
        Args:
            tiers: List of (volume_threshold, commission_rate) tuples
        """
        self.tiers = sorted(tiers, key=lambda x: x[0])
    
    def calculate_commission(self, trade: Trade) -> float:
        """Calculate tiered commission."""
        volume = abs(trade.quantity * trade.price)
        
        for threshold, rate in reversed(self.tiers):
            if volume >= threshold:
                return volume * rate
        
        # Fallback to highest rate if volume is below all thresholds
        return volume * self.tiers[0][1]

class SlippageEngine:
    """Slippage calculation engine."""
    
    def __init__(self, model: SlippageModel = SlippageModel.FIXED_BASIS_POINTS,
                 parameters: Dict[str, float] = None):
        """Initialize slippage engine."""
        self.model = model
        self.parameters = parameters or {}
        
        # Default parameters
        default_params = {
            'fixed_bps': 2.0,  # 2 basis points
            'participation_rate': 0.1,  # 10% participation
            'impact_coefficient': 0.5,
            'sqrt_coefficient': 0.1,
            'linear_coefficient': 0.001
        }
        
        for key, value in default_params.items():
            if key not in self.parameters:
                self.parameters[key] = value
    
    def calculate_slippage(self, order: Order, market_data: Dict[str, Any]) -> float:
        """Calculate slippage for an order."""
        
        if self.model == SlippageModel.FIXED_BASIS_POINTS:
            return self._fixed_bps_slippage(order)
        
        elif self.model == SlippageModel.VOLUME_PARTICIPATION:
            return self._volume_participation_slippage(order, market_data)
        
        elif self.model == SlippageModel.PRICE_IMPACT:
            return self._price_impact_slippage(order, market_data)
        
        elif self.model == SlippageModel.SQUARE_ROOT:
            return self._square_root_slippage(order, market_data)
        
        elif self.model == SlippageModel.LINEAR:
            return self._linear_slippage(order, market_data)
        
        else:
            return 0.0
    
    def _fixed_bps_slippage(self, order: Order) -> float:
        """Fixed basis points slippage."""
        bps = self.parameters.get('fixed_bps', 2.0)
        multiplier = 1 if order.side == OrderSide.BUY else -1
        return multiplier * bps / 10000.0
    
    def _volume_participation_slippage(self, order: Order, 
                                     market_data: Dict[str, Any]) -> float:
        """Volume participation based slippage."""
        volume = market_data.get('volume', 1000000)
        participation_rate = self.parameters.get('participation_rate', 0.1)
        
        # Order size as fraction of volume
        order_fraction = abs(order.quantity) / volume
        
        # Slippage increases with participation rate
        base_slippage = order_fraction / participation_rate * 10  # basis points
        
        multiplier = 1 if order.side == OrderSide.BUY else -1
        return multiplier * base_slippage / 10000.0
    
    def _price_impact_slippage(self, order: Order, 
                             market_data: Dict[str, Any]) -> float:
        """Price impact based slippage."""
        volume = market_data.get('volume', 1000000)
        volatility = market_data.get('volatility', 0.02)
        
        # Market impact model
        order_size_usd = abs(order.quantity * order.price) if order.price else abs(order.quantity * 100)
        avg_volume_usd = volume * market_data.get('price', 100)
        
        # Square root impact
        impact_coefficient = self.parameters.get('impact_coefficient', 0.5)
        impact = impact_coefficient * volatility * np.sqrt(order_size_usd / avg_volume_usd)
        
        multiplier = 1 if order.side == OrderSide.BUY else -1
        return multiplier * impact
    
    def _square_root_slippage(self, order: Order, 
                            market_data: Dict[str, Any]) -> float:
        """Square root slippage model."""
        volume = market_data.get('volume', 1000000)
        coefficient = self.parameters.get('sqrt_coefficient', 0.1)
        
        # Square root of order size relative to volume
        order_fraction = abs(order.quantity) / volume
        slippage_bps = coefficient * np.sqrt(order_fraction) * 100
        
        multiplier = 1 if order.side == OrderSide.BUY else -1
        return multiplier * slippage_bps / 10000.0
    
    def _linear_slippage(self, order: Order, 
                       market_data: Dict[str, Any]) -> float:
        """Linear slippage model."""
        volume = market_data.get('volume', 1000000)
        coefficient = self.parameters.get('linear_coefficient', 0.001)
        
        # Linear in order size
        order_fraction = abs(order.quantity) / volume
        slippage_rate = coefficient * order_fraction
        
        multiplier = 1 if order.side == OrderSide.BUY else -1
        return multiplier * slippage_rate

class MarketSimulator:
    """Market data simulation for backtesting."""
    
    def __init__(self, data: pd.DataFrame):
        """Initialize market simulator with OHLCV data."""
        self.data = data
        self.current_index = 0
        
        # Ensure required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")
    
    def get_current_data(self) -> Optional[pd.Series]:
        """Get current market data."""
        if self.current_index >= len(self.data):
            return None
        
        return self.data.iloc[self.current_index]
    
    def advance(self) -> bool:
        """Advance to next time step."""
        self.current_index += 1
        return self.current_index < len(self.data)
    
    def get_market_price(self, order_type: OrderType) -> float:
        """Get market price for order execution."""
        current_data = self.get_current_data()
        
        if current_data is None:
            return 0.0
        
        if order_type in [OrderType.MARKET, OrderType.STOP]:
            return current_data['open']  # Assume execution at next open
        elif order_type in [OrderType.MARKET_ON_CLOSE, OrderType.LIMIT_ON_CLOSE]:
            return current_data['close']
        else:
            return current_data['close']  # Default to close price
    
    def can_execute_limit_order(self, order: Order) -> bool:
        """Check if limit order can be executed."""
        current_data = self.get_current_data()
        
        if current_data is None or order.price is None:
            return False
        
        if order.side == OrderSide.BUY:
            # Buy limit order executes if market goes at or below limit price
            return current_data['low'] <= order.price
        else:
            # Sell limit order executes if market goes at or above limit price
            return current_data['high'] >= order.price
    
    def can_execute_stop_order(self, order: Order) -> bool:
        """Check if stop order can be executed."""
        current_data = self.get_current_data()
        
        if current_data is None or order.stop_price is None:
            return False
        
        if order.side == OrderSide.BUY:
            # Buy stop order executes if market goes at or above stop price
            return current_data['high'] >= order.stop_price
        else:
            # Sell stop order executes if market goes at or below stop price
            return current_data['low'] <= order.stop_price

class Portfolio:
    """Portfolio management for backtesting."""
    
    def __init__(self, initial_cash: float = 100000.0):
        """Initialize portfolio."""
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, Position] = {}
        self.history: List[PortfolioSnapshot] = []
        self.trades: List[Trade] = []
        
    def get_position(self, symbol: str) -> Position:
        """Get position for symbol."""
        if symbol not in self.positions:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=0.0,
                average_price=0.0,
                market_value=0.0,
                unrealized_pnl=0.0
            )
        
        return self.positions[symbol]
    
    def update_position(self, trade: Trade, current_price: float):
        """Update position after trade execution."""
        position = self.get_position(trade.symbol)
        
        if trade.side == OrderSide.BUY:
            # Buy trade
            if position.quantity >= 0:
                # Adding to long position or opening long
                new_quantity = position.quantity + trade.quantity
                if new_quantity > 0:
                    position.average_price = (
                        (position.quantity * position.average_price + 
                         trade.quantity * trade.price) / new_quantity
                    )
                position.quantity = new_quantity
            else:
                # Covering short position
                if abs(position.quantity) >= trade.quantity:
                    # Partial or full cover
                    position.realized_pnl += (position.average_price - trade.price) * trade.quantity
                    position.quantity += trade.quantity
                else:
                    # Cover short and go long
                    cover_quantity = abs(position.quantity)
                    long_quantity = trade.quantity - cover_quantity
                    
                    # Realize P&L from short cover
                    position.realized_pnl += (position.average_price - trade.price) * cover_quantity
                    
                    # New long position
                    position.quantity = long_quantity
                    position.average_price = trade.price
        
        else:  # SELL
            # Sell trade
            if position.quantity <= 0:
                # Adding to short position or opening short
                new_quantity = position.quantity - trade.quantity
                if new_quantity < 0:
                    position.average_price = (
                        (abs(position.quantity) * position.average_price + 
                         trade.quantity * trade.price) / abs(new_quantity)
                    )
                position.quantity = new_quantity
            else:
                # Selling long position
                if position.quantity >= trade.quantity:
                    # Partial or full sale
                    position.realized_pnl += (trade.price - position.average_price) * trade.quantity
                    position.quantity -= trade.quantity
                else:
                    # Sell long and go short
                    sell_quantity = position.quantity
                    short_quantity = trade.quantity - sell_quantity
                    
                    # Realize P&L from long sale
                    position.realized_pnl += (trade.price - position.average_price) * sell_quantity
                    
                    # New short position
                    position.quantity = -short_quantity
                    position.average_price = trade.price
        
        # Update market value and unrealized P&L
        position.market_value = position.quantity * current_price
        
        if position.quantity != 0:
            if position.quantity > 0:
                position.unrealized_pnl = (current_price - position.average_price) * position.quantity
            else:
                position.unrealized_pnl = (position.average_price - current_price) * abs(position.quantity)
        else:
            position.unrealized_pnl = 0.0
        
        # Update timestamps
        if position.first_trade_time is None:
            position.first_trade_time = trade.timestamp
        position.last_trade_time = trade.timestamp
        
        # Update cash
        cash_flow = -trade.quantity * trade.price if trade.side == OrderSide.BUY else trade.quantity * trade.price
        self.cash += cash_flow - trade.commission
    
    def get_total_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value."""
        total_value = self.cash
        
        for symbol, position in self.positions.items():
            if position.quantity != 0:
                current_price = current_prices.get(symbol, position.average_price)
                total_value += position.quantity * current_price
        
        return total_value
    
    def take_snapshot(self, timestamp: pd.Timestamp, 
                     current_prices: Dict[str, float]) -> PortfolioSnapshot:
        """Take portfolio snapshot."""
        
        # Update all positions with current prices
        for symbol, position in self.positions.items():
            if position.quantity != 0:
                current_price = current_prices.get(symbol, position.average_price)
                position.market_value = position.quantity * current_price
                
                if position.quantity > 0:
                    position.unrealized_pnl = (current_price - position.average_price) * position.quantity
                else:
                    position.unrealized_pnl = (position.average_price - current_price) * abs(position.quantity)
        
        total_value = self.get_total_value(current_prices)
        
        # Calculate daily P&L
        daily_pnl = 0.0
        if self.history:
            daily_pnl = total_value - self.history[-1].total_value
        
        # Calculate total unrealized and realized P&L
        unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        realized_pnl = sum(pos.realized_pnl for pos in self.positions.values())
        
        snapshot = PortfolioSnapshot(
            timestamp=timestamp,
            cash=self.cash,
            total_value=total_value,
            positions=self.positions.copy(),
            daily_pnl=daily_pnl,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl
        )
        
        self.history.append(snapshot)
        return snapshot

class BacktestEngine:
    """Main backtesting engine."""
    
    def __init__(self, initial_cash: float = 100000.0,
                 commission_model: Optional[CommissionModel] = None,
                 slippage_engine: Optional[SlippageEngine] = None):
        """Initialize backtest engine."""
        
        self.portfolio = Portfolio(initial_cash)
        self.commission_model = commission_model or FixedCommissionModel(1.0)
        self.slippage_engine = slippage_engine or SlippageEngine()
        
        self.pending_orders: List[Order] = []
        self.filled_orders: List[Order] = []
        self.rejected_orders: List[Order] = []
        
        self.market_simulators: Dict[str, MarketSimulator] = {}
        self.current_timestamp: Optional[pd.Timestamp] = None
        
        # Performance tracking
        self.performance_metrics = {}
        self.benchmark_data: Optional[pd.Series] = None
        
    def add_data(self, symbol: str, data: pd.DataFrame):
        """Add market data for a symbol."""
        self.market_simulators[symbol] = MarketSimulator(data)
    
    def set_benchmark(self, benchmark_data: pd.Series):
        """Set benchmark for performance comparison."""
        self.benchmark_data = benchmark_data
    
    def submit_order(self, order: Order):
        """Submit an order for execution."""
        order.timestamp = self.current_timestamp or pd.Timestamp.now()
        self.pending_orders.append(order)
    
    def create_market_order(self, symbol: str, side: OrderSide, 
                          quantity: float) -> Order:
        """Create a market order."""
        return Order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=OrderType.MARKET,
            timestamp=self.current_timestamp or pd.Timestamp.now()
        )
    
    def create_limit_order(self, symbol: str, side: OrderSide,
                         quantity: float, price: float) -> Order:
        """Create a limit order."""
        return Order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=OrderType.LIMIT,
            price=price,
            timestamp=self.current_timestamp or pd.Timestamp.now()
        )
    
    def create_stop_order(self, symbol: str, side: OrderSide,
                        quantity: float, stop_price: float) -> Order:
        """Create a stop order."""
        return Order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=OrderType.STOP,
            stop_price=stop_price,
            timestamp=self.current_timestamp or pd.Timestamp.now()
        )
    
    def process_orders(self):
        """Process pending orders."""
        orders_to_remove = []
        
        for i, order in enumerate(self.pending_orders):
            simulator = self.market_simulators.get(order.symbol)
            
            if simulator is None:
                order.status = OrderStatus.REJECTED
                self.rejected_orders.append(order)
                orders_to_remove.append(i)
                continue
            
            current_data = simulator.get_current_data()
            
            if current_data is None:
                continue
            
            # Check if order can be executed
            can_execute = False
            execution_price = 0.0
            
            if order.order_type == OrderType.MARKET:
                can_execute = True
                execution_price = simulator.get_market_price(order.order_type)
                
            elif order.order_type == OrderType.LIMIT:
                can_execute = simulator.can_execute_limit_order(order)
                if can_execute:
                    execution_price = order.price
                    
            elif order.order_type == OrderType.STOP:
                can_execute = simulator.can_execute_stop_order(order)
                if can_execute:
                    execution_price = simulator.get_market_price(OrderType.MARKET)
                    
            elif order.order_type in [OrderType.MARKET_ON_CLOSE, OrderType.LIMIT_ON_CLOSE]:
                can_execute = True
                execution_price = simulator.get_market_price(order.order_type)
            
            if can_execute and execution_price > 0:
                # Execute order
                self._execute_order(order, execution_price, current_data)
                orders_to_remove.append(i)
        
        # Remove processed orders
        for i in reversed(orders_to_remove):
            self.pending_orders.pop(i)
    
    def _execute_order(self, order: Order, execution_price: float, 
                      market_data: pd.Series):
        """Execute an order."""
        
        # Calculate slippage
        market_data_dict = {
            'price': execution_price,
            'volume': market_data.get('volume', 1000000),
            'volatility': 0.02  # Default volatility
        }
        
        slippage_rate = self.slippage_engine.calculate_slippage(order, market_data_dict)
        slippage_adjusted_price = execution_price * (1 + slippage_rate)
        
        # Create trade
        trade = Trade(
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=slippage_adjusted_price,
            timestamp=self.current_timestamp,
            order_id=order.order_id,
            slippage=slippage_rate * execution_price
        )
        
        # Calculate commission
        trade.commission = self.commission_model.calculate_commission(trade)
        
        # Update order status
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.filled_price = slippage_adjusted_price
        order.commission = trade.commission
        order.slippage = trade.slippage
        
        # Update portfolio
        self.portfolio.update_position(trade, slippage_adjusted_price)
        
        # Record trade
        self.portfolio.trades.append(trade)
        self.filled_orders.append(order)
    
    def run_backtest(self, start_date: Optional[pd.Timestamp] = None,
                    end_date: Optional[pd.Timestamp] = None,
                    strategy_function: Optional[Callable] = None) -> Dict[str, Any]:
        """Run backtest simulation."""
        
        if not self.market_simulators:
            raise ValueError("No market data provided")
        
        # Get common date range
        all_dates = set()
        for simulator in self.market_simulators.values():
            all_dates.update(simulator.data.index)
        
        all_dates = sorted(all_dates)
        
        if start_date:
            all_dates = [d for d in all_dates if d >= start_date]
        if end_date:
            all_dates = [d for d in all_dates if d <= end_date]
        
        # Initialize simulators
        for simulator in self.market_simulators.values():
            simulator.current_index = 0
        
        # Run simulation
        for date in all_dates:
            self.current_timestamp = date
            
            # Advance all simulators to current date
            current_prices = {}
            
            for symbol, simulator in self.market_simulators.items():
                # Find the appropriate data point
                if date in simulator.data.index:
                    simulator.current_index = simulator.data.index.get_loc(date)
                    current_data = simulator.get_current_data()
                    if current_data is not None:
                        current_prices[symbol] = current_data['close']
            
            # Process pending orders
            self.process_orders()
            
            # Run strategy if provided
            if strategy_function:
                try:
                    strategy_function(self, date, current_prices)
                except Exception as e:
                    print(f"Strategy error on {date}: {e}")
            
            # Take portfolio snapshot
            if current_prices:  # Only if we have price data
                self.portfolio.take_snapshot(date, current_prices)
        
        # Calculate performance metrics
        self._calculate_performance_metrics()
        
        # Return backtest results
        return self._generate_backtest_report()
    
    def _calculate_performance_metrics(self):
        """Calculate performance metrics from backtest results."""
        
        if not self.portfolio.history:
            return
        
        # Extract time series data
        dates = [snapshot.timestamp for snapshot in self.portfolio.history]
        values = [snapshot.total_value for snapshot in self.portfolio.history]
        
        portfolio_series = pd.Series(values, index=dates)
        returns = portfolio_series.pct_change().dropna()
        
        # Basic metrics
        total_return = (portfolio_series.iloc[-1] / portfolio_series.iloc[0]) - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        
        # Drawdown analysis
        cumulative = portfolio_series
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        positive_returns = returns[returns > 0]
        win_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0
        
        # Trade analysis
        if self.portfolio.trades:
            trade_pnl = []
            for trade in self.portfolio.trades:
                # Simplified P&L calculation
                if trade.side == OrderSide.BUY:
                    trade_pnl.append(-trade.quantity * trade.price - trade.commission)
                else:
                    trade_pnl.append(trade.quantity * trade.price - trade.commission)
            
            avg_trade = np.mean(trade_pnl) if trade_pnl else 0
            trade_win_rate = len([pnl for pnl in trade_pnl if pnl > 0]) / len(trade_pnl) if trade_pnl else 0
        else:
            avg_trade = 0
            trade_win_rate = 0
        
        # Benchmark comparison
        benchmark_metrics = {}
        if self.benchmark_data is not None:
            # Align benchmark with portfolio dates
            benchmark_aligned = self.benchmark_data.reindex(portfolio_series.index, method='ffill')
            benchmark_returns = benchmark_aligned.pct_change().dropna()
            
            if len(benchmark_returns) > 0:
                benchmark_total_return = (benchmark_aligned.iloc[-1] / benchmark_aligned.iloc[0]) - 1
                benchmark_volatility = benchmark_returns.std() * np.sqrt(252)
                
                # Align returns for correlation and tracking error
                common_dates = returns.index.intersection(benchmark_returns.index)
                if len(common_dates) > 0:
                    aligned_portfolio = returns.loc[common_dates]
                    aligned_benchmark = benchmark_returns.loc[common_dates]
                    
                    correlation = aligned_portfolio.corr(aligned_benchmark)
                    tracking_error = (aligned_portfolio - aligned_benchmark).std() * np.sqrt(252)
                    information_ratio = (aligned_portfolio.mean() - aligned_benchmark.mean()) / (aligned_portfolio - aligned_benchmark).std() * np.sqrt(252)
                    
                    benchmark_metrics = {
                        'benchmark_total_return': benchmark_total_return,
                        'benchmark_volatility': benchmark_volatility,
                        'correlation': correlation,
                        'tracking_error': tracking_error,
                        'information_ratio': information_ratio,
                        'excess_return': annualized_return - (benchmark_total_return * (252 / len(benchmark_returns)))
                    }
        
        self.performance_metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(self.portfolio.trades),
            'avg_trade_pnl': avg_trade,
            'trade_win_rate': trade_win_rate,
            **benchmark_metrics
        }
    
    def _generate_backtest_report(self) -> Dict[str, Any]:
        """Generate comprehensive backtest report."""
        
        return {
            'backtest_period': {
                'start_date': self.portfolio.history[0].timestamp.isoformat() if self.portfolio.history else None,
                'end_date': self.portfolio.history[-1].timestamp.isoformat() if self.portfolio.history else None,
                'total_days': len(self.portfolio.history)
            },
            'portfolio_summary': {
                'initial_capital': self.portfolio.initial_cash,
                'final_value': self.portfolio.history[-1].total_value if self.portfolio.history else self.portfolio.initial_cash,
                'final_cash': self.portfolio.history[-1].cash if self.portfolio.history else self.portfolio.cash,
                'total_realized_pnl': sum(pos.realized_pnl for pos in self.portfolio.positions.values()),
                'total_unrealized_pnl': sum(pos.unrealized_pnl for pos in self.portfolio.positions.values())
            },
            'performance_metrics': self.performance_metrics,
            'trading_summary': {
                'total_orders': len(self.filled_orders) + len(self.rejected_orders),
                'filled_orders': len(self.filled_orders),
                'rejected_orders': len(self.rejected_orders),
                'total_trades': len(self.portfolio.trades),
                'total_commission': sum(trade.commission for trade in self.portfolio.trades),
                'total_slippage': sum(abs(trade.slippage) for trade in self.portfolio.trades)
            },
            'positions': {
                symbol: {
                    'quantity': pos.quantity,
                    'average_price': pos.average_price,
                    'market_value': pos.market_value,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'realized_pnl': pos.realized_pnl
                }
                for symbol, pos in self.portfolio.positions.items()
                if pos.quantity != 0
            },
            'equity_curve': [
                {
                    'date': snapshot.timestamp.isoformat(),
                    'total_value': snapshot.total_value,
                    'cash': snapshot.cash,
                    'daily_pnl': snapshot.daily_pnl
                }
                for snapshot in self.portfolio.history[-100:]  # Last 100 days
            ]
        }

# Example strategy for testing
def simple_moving_average_strategy(engine: BacktestEngine, 
                                 current_date: pd.Timestamp,
                                 current_prices: Dict[str, float]):
    """Simple moving average crossover strategy."""
    
    symbol = 'TEST'  # Assume single symbol for simplicity
    
    if symbol not in current_prices:
        return
    
    # Get historical data
    simulator = engine.market_simulators.get(symbol)
    if not simulator:
        return
    
    # Calculate moving averages (simplified)
    if simulator.current_index < 20:  # Need at least 20 periods
        return
    
    # Get recent data
    recent_data = simulator.data.iloc[max(0, simulator.current_index-19):simulator.current_index+1]
    
    if len(recent_data) < 20:
        return
    
    short_ma = recent_data['close'].tail(5).mean()
    long_ma = recent_data['close'].tail(20).mean()
    
    current_position = engine.portfolio.get_position(symbol)
    current_price = current_prices[symbol]
    
    # Simple crossover logic
    if short_ma > long_ma and current_position.quantity <= 0:
        # Buy signal
        order_quantity = 100  # Fixed quantity
        if engine.portfolio.cash > order_quantity * current_price:
            order = engine.create_market_order(symbol, OrderSide.BUY, order_quantity)
            engine.submit_order(order)
    
    elif short_ma < long_ma and current_position.quantity >= 0:
        # Sell signal
        if current_position.quantity > 0:
            order = engine.create_market_order(symbol, OrderSide.SELL, current_position.quantity)
            engine.submit_order(order)
    import sys
    import os
    import time
    # Add project root to path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
    from data.ingestion.realtime_data_cache import RealTimeDataCache

    print("Fetching real-time data for Backtesting Infrastructure...")
    cache = RealTimeDataCache(["BTCUSDT"])
    cache.start()
    time.sleep(2)
    
    df = cache.get_history("BTCUSDT", limit=252)
    
    if df is not None and not df.empty:
        market_data = df
        print(f"Fetched {len(market_data)} days of real market data for BTCUSDT")
        print(f"Price range: ${market_data['low'].min():.2f} - ${market_data['high'].max():.2f}")
    else:
        # Do not generate synthetic market data here. Require real-time data.
        logger.error("Could not fetch real-time market data from RealTimeDataCache.")
        logger.error("Synthetic data generation has been disabled - please provide real market data or ensure the ingestion service is running.")
        raise RuntimeError("Real-time market data unavailable - synthetic fallback disabled.")
    
    print("\n=== Commission Models Testing ===")
    
    # Test commission models
    test_trade = Trade('TEST', OrderSide.BUY, 100, 50.0, pd.Timestamp.now())
    
    fixed_commission = FixedCommissionModel(5.0)
    pct_commission = PercentageCommissionModel(0.001, min_commission=1.0)
    tiered_commission = TieredCommissionModel([(0, 0.003), (10000, 0.002), (50000, 0.001)])
    
    print(f"Fixed commission: ${fixed_commission.calculate_commission(test_trade):.2f}")
    print(f"Percentage commission: ${pct_commission.calculate_commission(test_trade):.2f}")
    print(f"Tiered commission: ${tiered_commission.calculate_commission(test_trade):.2f}")
    
    print("\n=== Slippage Models Testing ===")
    
    # Test slippage models
    test_order = Order('TEST', OrderSide.BUY, 100, OrderType.MARKET, pd.Timestamp.now(), price=50.0)
    test_market_data = {'price': 50.0, 'volume': 100000, 'volatility': 0.02}
    
    for model in SlippageModel:
        slippage_engine = SlippageEngine(model)
        slippage = slippage_engine.calculate_slippage(test_order, test_market_data)
        print(f"{model.value}: {slippage:.6f} ({slippage*50:.4f} per share)")
    
    print("\n=== Backtest Execution ===")
    
    # Initialize backtest engine
    engine = BacktestEngine(
        initial_cash=100000.0,
        commission_model=PercentageCommissionModel(0.001, min_commission=1.0),
        slippage_engine=SlippageEngine(SlippageModel.FIXED_BASIS_POINTS, {'fixed_bps': 2.0})
    )
    
    # Add market data
    engine.add_data('TEST', market_data)
    
    # Set benchmark (market returns)
    benchmark = market_data['close'] / market_data['close'].iloc[0]
    engine.set_benchmark(benchmark)
    
    # Run backtest with simple strategy
    print("Running backtest with moving average strategy...")
    
    start_time = time.time()
    results = engine.run_backtest(strategy_function=simple_moving_average_strategy)
    backtest_time = time.time() - start_time
    
    print(f"Backtest completed in {backtest_time:.3f} seconds")
    
    print("\n=== Backtest Results ===")
    
    # Print key results
    performance = results['performance_metrics']
    portfolio = results['portfolio_summary']
    trading = results['trading_summary']
    
    print(f"Initial Capital: ${portfolio['initial_capital']:,.2f}")
    print(f"Final Value: ${portfolio['final_value']:,.2f}")
    print(f"Total Return: {performance['total_return']:.2%}")
    print(f"Annualized Return: {performance['annualized_return']:.2%}")
    print(f"Volatility: {performance['volatility']:.2%}")
    print(f"Sharpe Ratio: {performance['sharpe_ratio']:.3f}")
    print(f"Max Drawdown: {performance['max_drawdown']:.2%}")
    print(f"Win Rate: {performance['win_rate']:.2%}")
    
    print(f"\nTrading Statistics:")
    print(f"Total Trades: {trading['total_trades']}")
    print(f"Total Commission: ${trading['total_commission']:.2f}")
    print(f"Total Slippage: ${trading['total_slippage']:.2f}")
    
    if 'benchmark_total_return' in performance:
        print(f"\nBenchmark Comparison:")
        print(f"Benchmark Return: {performance['benchmark_total_return']:.2%}")
        print(f"Excess Return: {performance['excess_return']:.2%}")
        print(f"Tracking Error: {performance['tracking_error']:.2%}")
        print(f"Information Ratio: {performance['information_ratio']:.3f}")
    
    print("\n=== Order Types Testing ===")
    
    # Test different order types
    test_engine = BacktestEngine(initial_cash=100000.0)
    test_engine.add_data('TEST', market_data.head(50))  # Smaller dataset for testing
    
    # Submit various order types
    current_price = market_data['close'].iloc[10]
    
    market_order = test_engine.create_market_order('TEST', OrderSide.BUY, 100)
    limit_order = test_engine.create_limit_order('TEST', OrderSide.BUY, 100, current_price * 0.95)
    stop_order = test_engine.create_stop_order('TEST', OrderSide.SELL, 50, current_price * 1.05)
    
    test_engine.submit_order(market_order)
    test_engine.submit_order(limit_order)
    test_engine.submit_order(stop_order)
    
    print(f"Submitted {len(test_engine.pending_orders)} test orders")
    
    # Run a few steps to see order execution
    test_results = test_engine.run_backtest(
        start_date=market_data.index[0],
        end_date=market_data.index[20]
    )
    
    print(f"Test orders processed:")
    print(f"  Filled: {len(test_engine.filled_orders)}")
    print(f"  Rejected: {len(test_engine.rejected_orders)}")
    print(f"  Pending: {len(test_engine.pending_orders)}")
    
    print("\n=== Performance Analysis ===")
    
    # Analyze performance characteristics
    if engine.portfolio.history:
        equity_curve = pd.Series(
            [s.total_value for s in engine.portfolio.history],
            index=[s.timestamp for s in engine.portfolio.history]
        )
        
        returns = equity_curve.pct_change().dropna()
        
        print(f"Return Statistics:")
        print(f"  Mean Daily Return: {returns.mean():.4f}")
        print(f"  Std Daily Return: {returns.std():.4f}")
        print(f"  Skewness: {returns.skew():.3f}")
        print(f"  Kurtosis: {returns.kurtosis():.3f}")
        print(f"  Best Day: {returns.max():.2%}")
        print(f"  Worst Day: {returns.min():.2%}")
    
    # Trade analysis
    if engine.portfolio.trades:
        trade_sizes = [abs(trade.quantity * trade.price) for trade in engine.portfolio.trades]
        commissions = [trade.commission for trade in engine.portfolio.trades]
        slippages = [abs(trade.slippage) for trade in engine.portfolio.trades]
        
        print(f"\nTrade Analysis:")
        print(f"  Average Trade Size: ${np.mean(trade_sizes):,.2f}")
        print(f"  Average Commission: ${np.mean(commissions):.2f}")
        print(f"  Average Slippage: ${np.mean(slippages):.2f}")
        print(f"  Commission as % of Trade Size: {np.mean(commissions)/np.mean(trade_sizes):.3%}")
        print(f"  Slippage as % of Trade Size: {np.mean(slippages)/np.mean(trade_sizes):.3%}")
    
    print("\n=== Risk Analysis ===")
    
    # Risk metrics
    if performance:
        calmar_ratio = performance['annualized_return'] / abs(performance['max_drawdown']) if performance['max_drawdown'] != 0 else 0
        sortino_ratio = performance['annualized_return'] / (returns[returns < 0].std() * np.sqrt(252)) if len(returns[returns < 0]) > 0 else 0
        
        print(f"Risk-Adjusted Metrics:")
        print(f"  Sharpe Ratio: {performance['sharpe_ratio']:.3f}")
        print(f"  Calmar Ratio: {calmar_ratio:.3f}")
        print(f"  Sortino Ratio: {sortino_ratio:.3f}")
        
        # Value at Risk
        if len(returns) > 0:
            var_95 = returns.quantile(0.05)
            var_99 = returns.quantile(0.01)
            
            print(f"  VaR (95%): {var_95:.2%}")
            print(f"  VaR (99%): {var_99:.2%}")
    
    print("\nBacktesting infrastructure testing completed successfully!")
    print(f"Processed {len(market_data)} days of market data")
    print(f"Executed {len(engine.portfolio.trades)} trades")
    print(f"Generated comprehensive performance analytics")
    print(f"Tested multiple order types and execution models")