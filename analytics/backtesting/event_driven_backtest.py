"""
Event-Driven Backtesting Framework
Advanced event-driven backtesting system for quantitative strategies
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
import warnings
import math
from collections import deque, defaultdict
import concurrent.futures
import heapq


class EventType(Enum):
    """Types of events in the backtesting system"""
    MARKET_DATA = "MARKET_DATA"             # Market data update
    SIGNAL = "SIGNAL"                       # Trading signal
    ORDER = "ORDER"                         # Order placement
    FILL = "FILL"                           # Order fill
    REBALANCE = "REBALANCE"                 # Portfolio rebalance
    RISK_CHECK = "RISK_CHECK"               # Risk monitoring
    EOD = "EOD"                            # End of day processing
    CUSTOM = "CUSTOM"                       # Custom event


class OrderType(Enum):
    """Order types"""
    MARKET = "MARKET"                       # Market order
    LIMIT = "LIMIT"                         # Limit order
    STOP = "STOP"                          # Stop order
    STOP_LIMIT = "STOP_LIMIT"              # Stop-limit order


class OrderStatus(Enum):
    """Order status"""
    PENDING = "PENDING"                     # Order pending
    FILLED = "FILLED"                       # Order filled
    PARTIAL = "PARTIAL"                     # Partially filled
    CANCELLED = "CANCELLED"                 # Order cancelled
    REJECTED = "REJECTED"                   # Order rejected


@dataclass
class Event:
    """Base event class"""
    event_type: EventType
    timestamp: datetime
    data: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0                       # Higher priority processed first
    
    def __lt__(self, other):
        """For priority queue sorting"""
        return (self.timestamp, -self.priority) < (other.timestamp, -other.priority)


@dataclass
class MarketDataEvent(Event):
    """Market data event"""
    symbol: str = ""
    price: float = 0.0
    volume: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    
    def __post_init__(self):
        self.event_type = EventType.MARKET_DATA


@dataclass
class SignalEvent(Event):
    """Trading signal event"""
    strategy_name: str = ""
    signals: Dict[str, float] = field(default_factory=dict)  # symbol -> signal strength
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.event_type = EventType.SIGNAL


@dataclass
class OrderEvent(Event):
    """Order event"""
    order_id: str = ""
    symbol: str = ""
    quantity: float = 0.0
    order_type: OrderType = OrderType.MARKET
    price: Optional[float] = None
    stop_price: Optional[float] = None
    strategy_name: str = ""
    
    def __post_init__(self):
        self.event_type = EventType.ORDER


@dataclass
class FillEvent(Event):
    """Fill event"""
    order_id: str = ""
    symbol: str = ""
    quantity: float = 0.0
    fill_price: float = 0.0
    commission: float = 0.0
    strategy_name: str = ""
    
    def __post_init__(self):
        self.event_type = EventType.FILL


@dataclass
class Position:
    """Portfolio position"""
    symbol: str
    quantity: float = 0.0
    average_price: float = 0.0
    market_price: float = 0.0
    
    @property
    def market_value(self) -> float:
        return self.quantity * self.market_price
    
    @property
    def unrealized_pnl(self) -> float:
        return self.quantity * (self.market_price - self.average_price)
    
    @property
    def is_long(self) -> bool:
        return self.quantity > 0
    
    @property
    def is_short(self) -> bool:
        return self.quantity < 0


@dataclass
class Order:
    """Order representation"""
    order_id: str
    symbol: str
    quantity: float
    order_type: OrderType
    timestamp: datetime
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    strategy_name: str = ""


class DataHandler(ABC):
    """Abstract data handler for backtesting"""
    
    @abstractmethod
    def get_latest_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get latest market data for symbol"""
        pass
    
    @abstractmethod
    def get_historical_data(self, symbol: str, start_date: datetime, 
                          end_date: datetime) -> pd.DataFrame:
        """Get historical data for symbol"""
        pass
    
    @abstractmethod
    def update_data(self) -> List[MarketDataEvent]:
        """Update data and return new market data events"""
        pass


class SimpleDataHandler(DataHandler):
    """Simple data handler using pandas DataFrame"""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize with DataFrame containing OHLCV data
        Index: MultiIndex with (timestamp, symbol) or DatetimeIndex with symbol columns
        """
        self.data = data
        self.current_index = 0
        self.current_data: Dict[str, Dict[str, Any]] = {}
        
        # Prepare data structure
        if isinstance(data.index, pd.MultiIndex):
            self.data_format = "multi_index"
            self.timestamps = sorted(data.index.get_level_values(0).unique())
            self.symbols = sorted(data.index.get_level_values(1).unique())
        else:
            self.data_format = "columns"
            self.timestamps = data.index.tolist()
            # Assume columns are like 'AAPL_close', 'AAPL_volume', etc.
            self.symbols = list(set([col.split('_')[0] for col in data.columns if '_' in col]))
    
    def get_latest_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get latest market data for symbol"""
        return self.current_data.get(symbol)
    
    def get_historical_data(self, symbol: str, start_date: datetime, 
                          end_date: datetime) -> pd.DataFrame:
        """Get historical data for symbol"""
        
        if self.data_format == "multi_index":
            try:
                symbol_data = self.data.xs(symbol, level=1)
                return symbol_data[(symbol_data.index >= start_date) & 
                                 (symbol_data.index <= end_date)]
            except KeyError:
                return pd.DataFrame()
        else:
            # Column format
            symbol_cols = [col for col in self.data.columns if col.startswith(f"{symbol}_")]
            if symbol_cols:
                symbol_data = self.data[symbol_cols]
                # Rename columns to remove symbol prefix
                symbol_data.columns = [col.replace(f"{symbol}_", "") for col in symbol_data.columns]
                return symbol_data[(symbol_data.index >= start_date) & 
                                 (symbol_data.index <= end_date)]
            return pd.DataFrame()
    
    def update_data(self) -> List[MarketDataEvent]:
        """Update data and return new market data events"""
        
        if self.current_index >= len(self.timestamps):
            return []  # No more data
        
        current_timestamp = self.timestamps[self.current_index]
        events = []
        
        # Get data for current timestamp
        if self.data_format == "multi_index":
            try:
                timestamp_data = self.data.xs(current_timestamp, level=0)
                
                for symbol in self.symbols:
                    if symbol in timestamp_data.index:
                        row = timestamp_data.loc[symbol]
                        
                        # Update current data
                        self.current_data[symbol] = {
                            'timestamp': current_timestamp,
                            'open': row.get('open', 0),
                            'high': row.get('high', 0),
                            'low': row.get('low', 0),
                            'close': row.get('close', 0),
                            'volume': row.get('volume', 0)
                        }
                        
                        # Create market data event
                        event = MarketDataEvent(
                            event_type=EventType.MARKET_DATA,
                            timestamp=current_timestamp,
                            symbol=symbol,
                            price=row.get('close', 0),
                            volume=row.get('volume', 0)
                        )
                        events.append(event)
                        
            except KeyError:
                pass  # No data for this timestamp
        
        else:
            # Column format
            if current_timestamp in self.data.index:
                row = self.data.loc[current_timestamp]
                
                for symbol in self.symbols:
                    close_col = f"{symbol}_close"
                    volume_col = f"{symbol}_volume"
                    
                    if close_col in row.index and not pd.isna(row[close_col]):
                        
                        # Update current data
                        self.current_data[symbol] = {
                            'timestamp': current_timestamp,
                            'open': row.get(f"{symbol}_open", row[close_col]),
                            'high': row.get(f"{symbol}_high", row[close_col]),
                            'low': row.get(f"{symbol}_low", row[close_col]),
                            'close': row[close_col],
                            'volume': row.get(volume_col, 0)
                        }
                        
                        # Create market data event
                        event = MarketDataEvent(
                            event_type=EventType.MARKET_DATA,
                            timestamp=current_timestamp,
                            symbol=symbol,
                            price=row[close_col],
                            volume=row.get(volume_col, 0)
                        )
                        events.append(event)
        
        self.current_index += 1
        return events


class Strategy(ABC):
    """Abstract strategy class"""
    
    def __init__(self, name: str):
        self.name = name
        self.portfolio_value = 0.0
        self.positions: Dict[str, Position] = {}
        
    @abstractmethod
    def calculate_signals(self, event: MarketDataEvent) -> List[SignalEvent]:
        """Calculate trading signals based on market data"""
        pass
    
    @abstractmethod
    def generate_orders(self, event: SignalEvent) -> List[OrderEvent]:
        """Generate orders based on signals"""
        pass
    
    def update_portfolio(self, event: FillEvent):
        """Update portfolio based on fill"""
        
        symbol = event.symbol
        quantity = event.quantity
        price = event.fill_price
        
        # Update position
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol)
        
        position = self.positions[symbol]
        
        # Calculate new average price
        if (position.quantity > 0 and quantity > 0) or (position.quantity < 0 and quantity < 0):
            # Same direction - update average price
            total_cost = position.quantity * position.average_price + quantity * price
            position.quantity += quantity
            if position.quantity != 0:
                position.average_price = total_cost / position.quantity
        else:
            # Opposite direction or new position
            if abs(quantity) >= abs(position.quantity):
                # Closes current position and potentially opens new one
                remaining_quantity = quantity + position.quantity
                position.quantity = remaining_quantity
                position.average_price = price if remaining_quantity != 0 else 0
            else:
                # Partial close
                position.quantity += quantity
                # Keep same average price
        
        # Remove position if quantity is zero
        if abs(position.quantity) < 1e-6:
            del self.positions[symbol]


class SimpleMovingAverageStrategy(Strategy):
    """Simple moving average crossover strategy"""
    
    def __init__(self, name: str = "SMA_Strategy", short_window: int = 10, long_window: int = 30):
        super().__init__(name)
        self.short_window = short_window
        self.long_window = long_window
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.long_window))
        
    def calculate_signals(self, event: MarketDataEvent) -> List[SignalEvent]:
        """Calculate SMA crossover signals"""
        
        symbol = event.symbol
        price = event.price
        
        # Update price history
        self.price_history[symbol].append(price)
        
        # Need enough data for long MA
        if len(self.price_history[symbol]) < self.long_window:
            return []
        
        # Calculate moving averages
        prices = list(self.price_history[symbol])
        short_ma = np.mean(prices[-self.short_window:])
        long_ma = np.mean(prices[-self.long_window:])
        
        # Previous MAs for crossover detection
        if len(prices) > 1:
            prev_short_ma = np.mean(prices[-(self.short_window+1):-1])
            prev_long_ma = np.mean(prices[-(self.long_window+1):-1])
            
            # Generate signal
            signal_strength = 0.0
            
            # Bullish crossover
            if short_ma > long_ma and prev_short_ma <= prev_long_ma:
                signal_strength = 1.0
            # Bearish crossover  
            elif short_ma < long_ma and prev_short_ma >= prev_long_ma:
                signal_strength = -1.0
            
            if signal_strength != 0:
                signal_event = SignalEvent(
                    event_type=EventType.SIGNAL,
                    timestamp=event.timestamp,
                    strategy_name=self.name,
                    signals={symbol: signal_strength},
                    metadata={
                        'short_ma': short_ma,
                        'long_ma': long_ma,
                        'price': price
                    }
                )
                return [signal_event]
        
        return []
    
    def generate_orders(self, event: SignalEvent) -> List[OrderEvent]:
        """Generate orders based on SMA signals"""
        
        orders = []
        
        for symbol, signal_strength in event.signals.items():
            if abs(signal_strength) < 0.5:  # Minimum signal threshold
                continue
            
            # Simple position sizing: fixed dollar amount
            target_value = 10000 * signal_strength  # $10k per unit signal
            current_price = event.metadata.get('price', 100)  # Fallback price
            
            if current_price > 0:
                target_quantity = target_value / current_price
                
                order = OrderEvent(
                    event_type=EventType.ORDER,
                    timestamp=event.timestamp,
                    order_id=f"{self.name}_{symbol}_{event.timestamp.strftime('%Y%m%d_%H%M%S')}",
                    symbol=symbol,
                    quantity=target_quantity,
                    order_type=OrderType.MARKET,
                    strategy_name=self.name
                )
                orders.append(order)
        
        return orders


class ExecutionHandler(ABC):
    """Abstract execution handler"""
    
    @abstractmethod
    def execute_order(self, order_event: OrderEvent) -> List[FillEvent]:
        """Execute order and return fill events"""
        pass


class SimpleExecutionHandler(ExecutionHandler):
    """Simple execution handler with immediate fills"""
    
    def __init__(self, commission_rate: float = 0.001, slippage_rate: float = 0.0005):
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.orders: Dict[str, Order] = {}
        
    def execute_order(self, order_event: OrderEvent) -> List[FillEvent]:
        """Execute order with simple fill logic"""
        
        # Create order record
        order = Order(
            order_id=order_event.order_id,
            symbol=order_event.symbol,
            quantity=order_event.quantity,
            order_type=order_event.order_type,
            timestamp=order_event.timestamp,
            price=order_event.price,
            stop_price=order_event.stop_price,
            strategy_name=order_event.strategy_name
        )
        
        self.orders[order.order_id] = order
        
        # Simple execution: assume immediate fill at market price with slippage
        if order_event.order_type == OrderType.MARKET:
            
            # Simulate slippage
            market_price = order_event.data.get('market_price', 100)  # Should be passed in
            
            if order_event.quantity > 0:  # Buy order
                fill_price = market_price * (1 + self.slippage_rate)
            else:  # Sell order
                fill_price = market_price * (1 - self.slippage_rate)
            
            # Calculate commission
            commission = abs(order_event.quantity * fill_price * self.commission_rate)
            
            # Update order status
            order.status = OrderStatus.FILLED
            order.filled_quantity = order_event.quantity
            order.average_fill_price = fill_price
            
            # Create fill event
            fill_event = FillEvent(
                event_type=EventType.FILL,
                timestamp=order_event.timestamp,
                order_id=order_event.order_id,
                symbol=order_event.symbol,
                quantity=order_event.quantity,
                fill_price=fill_price,
                commission=commission,
                strategy_name=order_event.strategy_name
            )
            
            return [fill_event]
        
        return []  # Other order types not implemented


class Portfolio:
    """Portfolio manager for backtesting"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions: Dict[str, Position] = {}
        self.cash = initial_capital
        
        # Performance tracking
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.trade_log: List[Dict[str, Any]] = []
        
    def update_fill(self, fill_event: FillEvent):
        """Update portfolio based on fill"""
        
        symbol = fill_event.symbol
        quantity = fill_event.quantity
        fill_price = fill_event.fill_price
        commission = fill_event.commission
        
        # Update cash
        self.cash -= quantity * fill_price + commission
        
        # Update position
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol)
        
        position = self.positions[symbol]
        
        # Log trade
        trade = {
            'timestamp': fill_event.timestamp,
            'symbol': symbol,
            'quantity': quantity,
            'price': fill_price,
            'commission': commission,
            'strategy': fill_event.strategy_name
        }
        self.trade_log.append(trade)
        
        # Update position
        if (position.quantity >= 0 and quantity > 0) or (position.quantity <= 0 and quantity < 0):
            # Same direction
            total_cost = position.quantity * position.average_price + quantity * fill_price
            position.quantity += quantity
            if position.quantity != 0:
                position.average_price = total_cost / position.quantity
        else:
            # Opposite direction
            if abs(quantity) >= abs(position.quantity):
                # Full close and potentially reverse
                remaining_quantity = position.quantity + quantity
                position.quantity = remaining_quantity
                if remaining_quantity != 0:
                    position.average_price = fill_price
            else:
                # Partial close
                position.quantity += quantity
        
        # Remove zero positions
        if abs(position.quantity) < 1e-6:
            del self.positions[symbol]
    
    def update_market_value(self, market_data: Dict[str, float]):
        """Update portfolio market value with current prices"""
        
        for symbol, position in self.positions.items():
            if symbol in market_data:
                position.market_price = market_data[symbol]
        
        # Calculate total portfolio value
        positions_value = sum(pos.market_value for pos in self.positions.values())
        total_value = self.cash + positions_value
        
        return total_value
    
    def get_portfolio_stats(self) -> Dict[str, Any]:
        """Get portfolio statistics"""
        
        if not self.equity_curve:
            return {}
        
        equity_values = [value for _, value in self.equity_curve]
        
        if len(equity_values) < 2:
            return {'total_return': 0.0, 'equity_values': equity_values}
        
        # Calculate returns
        returns = np.diff(equity_values) / equity_values[:-1]
        
        # Performance metrics
        total_return = (equity_values[-1] - self.initial_capital) / self.initial_capital
        
        if len(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            
            # Maximum drawdown
            peak = np.maximum.accumulate(equity_values)
            drawdowns = (np.array(equity_values) - peak) / peak
            max_drawdown = np.min(drawdowns)
            
            volatility = np.std(returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
            max_drawdown = 0.0
            volatility = 0.0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'n_trades': len(self.trade_log),
            'equity_values': equity_values
        }


class EventDrivenBacktester:
    """
    Main event-driven backtesting engine
    """
    
    def __init__(self, data_handler: DataHandler, strategy: Strategy, 
                 execution_handler: ExecutionHandler, portfolio: Portfolio):
        
        self.data_handler = data_handler
        self.strategy = strategy
        self.execution_handler = execution_handler
        self.portfolio = portfolio
        
        # Event queue (priority queue)
        self.event_queue: List[Event] = []
        
        # Current market data
        self.current_market_data: Dict[str, float] = {}
        
        # Performance tracking
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        
    def run_backtest(self) -> Dict[str, Any]:
        """Run the backtest"""
        
        self.start_time = datetime.now()
        
        try:
            while True:
                # Get new market data
                market_events = self.data_handler.update_data()
                
                if not market_events:
                    break  # No more data
                
                # Add market events to queue
                for event in market_events:
                    heapq.heappush(self.event_queue, event)
                
                # Process all events for current timestamp
                self._process_events()
            
            self.end_time = datetime.now()
            
            # Generate final results
            return self._generate_results()
            
        except Exception as e:
            self.end_time = datetime.now()
            raise RuntimeError(f"Backtest failed: {e}")
    
    def _process_events(self):
        """Process all events in the queue"""
        
        while self.event_queue:
            # Get highest priority event
            current_event = heapq.heappop(self.event_queue)
            
            # Process based on event type
            if current_event.event_type == EventType.MARKET_DATA:
                self._handle_market_data(current_event)
                
            elif current_event.event_type == EventType.SIGNAL:
                self._handle_signal(current_event)
                
            elif current_event.event_type == EventType.ORDER:
                self._handle_order(current_event)
                
            elif current_event.event_type == EventType.FILL:
                self._handle_fill(current_event)
    
    def _handle_market_data(self, event: MarketDataEvent):
        """Handle market data event"""
        
        # Update current market data
        self.current_market_data[event.symbol] = event.price
        
        # Update portfolio market values
        portfolio_value = self.portfolio.update_market_value(self.current_market_data)
        self.portfolio.equity_curve.append((event.timestamp, portfolio_value))
        
        # Generate signals from strategy
        signal_events = self.strategy.calculate_signals(event)
        
        # Add signal events to queue
        for signal_event in signal_events:
            heapq.heappush(self.event_queue, signal_event)
    
    def _handle_signal(self, event: SignalEvent):
        """Handle signal event"""
        
        # Generate orders from signals
        order_events = self.strategy.generate_orders(event)
        
        # Add market price data to orders
        for order_event in order_events:
            order_event.data['market_price'] = self.current_market_data.get(order_event.symbol, 100)
        
        # Add order events to queue
        for order_event in order_events:
            heapq.heappush(self.event_queue, order_event)
    
    def _handle_order(self, event: OrderEvent):
        """Handle order event"""
        
        # Execute order
        fill_events = self.execution_handler.execute_order(event)
        
        # Add fill events to queue
        for fill_event in fill_events:
            heapq.heappush(self.event_queue, fill_event)
    
    def _handle_fill(self, event: FillEvent):
        """Handle fill event"""
        
        # Update portfolio
        self.portfolio.update_fill(event)
        
        # Update strategy positions
        self.strategy.update_portfolio(event)
    
    def _generate_results(self) -> Dict[str, Any]:
        """Generate backtest results"""
        
        portfolio_stats = self.portfolio.get_portfolio_stats()
        
        results = {
            'start_time': self.start_time,
            'end_time': self.end_time,
            'runtime_seconds': (self.end_time - self.start_time).total_seconds() if self.end_time and self.start_time else 0,
            'initial_capital': self.portfolio.initial_capital,
            'final_value': self.portfolio.equity_curve[-1][1] if self.portfolio.equity_curve else self.portfolio.initial_capital,
            'portfolio_stats': portfolio_stats,
            'positions': {symbol: {'quantity': pos.quantity, 'market_value': pos.market_value} 
                         for symbol, pos in self.portfolio.positions.items()},
            'cash': self.portfolio.cash,
            'n_trades': len(self.portfolio.trade_log),
            'equity_curve': self.portfolio.equity_curve
        }
        
        return results
