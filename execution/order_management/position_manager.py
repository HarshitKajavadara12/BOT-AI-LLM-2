"""
Position Manager
Real-time position tracking and risk management
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable, Any, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
import warnings
from collections import defaultdict, deque
import json


class PositionType(Enum):
    """Position type classification"""
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class RiskStatus(Enum):
    """Risk status levels"""
    GREEN = "GREEN"      # Normal
    YELLOW = "YELLOW"    # Warning
    RED = "RED"          # Breach
    BLACK = "BLACK"      # Emergency


@dataclass
class Position:
    """Individual position record"""
    symbol: str
    quantity: float  # Positive = long, negative = short
    average_price: float
    market_price: float = 0.0
    
    # Cost basis and P&L
    cost_basis: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    
    # Risk metrics
    var_1day: float = 0.0      # 1-day Value at Risk
    var_10day: float = 0.0     # 10-day Value at Risk
    beta: float = 1.0          # Market beta
    
    # Timestamps
    first_trade_time: Optional[datetime] = None
    last_trade_time: Optional[datetime] = None
    last_update_time: datetime = field(default_factory=datetime.now)
    
    @property
    def position_type(self) -> PositionType:
        """Get position type"""
        if self.quantity > 0:
            return PositionType.LONG
        elif self.quantity < 0:
            return PositionType.SHORT
        else:
            return PositionType.FLAT
    
    @property
    def market_value(self) -> float:
        """Current market value of position"""
        return self.quantity * self.market_price
    
    @property
    def notional_value(self) -> float:
        """Absolute notional value"""
        return abs(self.market_value)
    
    @property
    def pnl_percent(self) -> float:
        """P&L as percentage of cost basis"""
        if self.cost_basis != 0:
            return self.unrealized_pnl / abs(self.cost_basis)
        return 0.0
    
    def update_market_price(self, new_price: float) -> None:
        """Update market price and recalculate P&L"""
        self.market_price = new_price
        self.unrealized_pnl = (new_price - self.average_price) * self.quantity
        self.last_update_time = datetime.now()


@dataclass
class PortfolioSummary:
    """Portfolio-level summary statistics"""
    total_long_value: float = 0.0
    total_short_value: float = 0.0
    net_value: float = 0.0
    gross_value: float = 0.0
    
    # P&L
    total_unrealized_pnl: float = 0.0
    total_realized_pnl: float = 0.0
    daily_pnl: float = 0.0
    
    # Risk metrics
    portfolio_var: float = 0.0
    portfolio_beta: float = 0.0
    max_leverage: float = 0.0
    
    # Counts
    long_positions: int = 0
    short_positions: int = 0
    total_positions: int = 0
    
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RiskLimit:
    """Risk limit definition"""
    limit_type: str  # 'position_size', 'notional', 'var', 'pnl', etc.
    symbol: Optional[str] = None  # None for portfolio-level limits
    soft_limit: float = 0.0
    hard_limit: float = 0.0
    current_value: float = 0.0
    breach_count: int = 0
    last_breach_time: Optional[datetime] = None
    
    @property
    def soft_breach(self) -> bool:
        """Check if soft limit is breached"""
        return abs(self.current_value) > self.soft_limit
    
    @property
    def hard_breach(self) -> bool:
        """Check if hard limit is breached"""
        return abs(self.current_value) > self.hard_limit
    
    @property
    def utilization(self) -> float:
        """Limit utilization percentage"""
        if self.soft_limit > 0:
            return abs(self.current_value) / self.soft_limit
        return 0.0


class PositionManager:
    """
    Real-time position tracking and risk management system
    """
    
    def __init__(self, name: str = "PositionManager"):
        self.name = name
        
        # Position storage
        self.positions: Dict[str, Position] = {}
        self.portfolio_summary = PortfolioSummary()
        
        # Risk management
        self.risk_limits: Dict[str, RiskLimit] = {}
        self.risk_status = RiskStatus.GREEN
        
        # Market data
        self.market_prices: Dict[str, float] = {}
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=252))  # 1 year
        
        # P&L tracking
        self.daily_pnl_history = deque(maxlen=252)
        self.pnl_attribution = defaultdict(float)
        
        # Event handling
        self.position_event_handlers = []
        self.risk_event_handlers = []
        
        # Threading
        self.lock = threading.RLock()
        self.is_running = False
        self.update_thread = None
        
        # Performance metrics
        self.metrics = {
            'total_trades_processed': 0,
            'position_updates': 0,
            'risk_breaches': 0,
            'last_update_time': datetime.now()
        }
    
    def start(self) -> None:
        """Start position manager"""
        self.is_running = True
        
        # Start update thread
        self.update_thread = threading.Thread(
            target=self._update_loop,
            name="PositionUpdateLoop"
        )
        self.update_thread.start()
        
        print(f"Position Manager {self.name} started")
    
    def stop(self) -> None:
        """Stop position manager"""
        self.is_running = False
        
        if self.update_thread:
            self.update_thread.join(timeout=5.0)
        
        print(f"Position Manager {self.name} stopped")
    
    def process_trade(
        self,
        symbol: str,
        quantity: float,  # Positive = buy, negative = sell
        price: float,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Process a trade and update positions
        
        Args:
            symbol: Trading symbol
            quantity: Trade quantity (signed)
            price: Execution price
            timestamp: Trade timestamp
        """
        
        if timestamp is None:
            timestamp = datetime.now()
        
        with self.lock:
            # Get or create position
            if symbol not in self.positions:
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=0.0,
                    average_price=0.0,
                    first_trade_time=timestamp
                )
            
            position = self.positions[symbol]
            
            # Calculate new position
            old_quantity = position.quantity
            old_cost_basis = position.cost_basis
            
            new_quantity = old_quantity + quantity
            trade_value = quantity * price
            
            # Handle position updates
            if old_quantity == 0:
                # Opening new position
                position.quantity = new_quantity
                position.average_price = price
                position.cost_basis = abs(trade_value)
                
            elif (old_quantity > 0 and quantity > 0) or (old_quantity < 0 and quantity < 0):
                # Adding to existing position
                total_cost = old_cost_basis + abs(trade_value)
                position.quantity = new_quantity
                position.cost_basis = total_cost
                
                if new_quantity != 0:
                    position.average_price = total_cost / abs(new_quantity)
                
            else:
                # Reducing or reversing position
                if abs(quantity) <= abs(old_quantity):
                    # Partial reduction
                    realized_pnl = self._calculate_realized_pnl(position, quantity, price)
                    position.realized_pnl += realized_pnl
                    
                    position.quantity = new_quantity
                    position.cost_basis = position.cost_basis * (abs(new_quantity) / abs(old_quantity))
                    
                else:
                    # Position reversal
                    closing_quantity = -old_quantity
                    opening_quantity = quantity + old_quantity
                    
                    # Realize P&L on closed portion
                    realized_pnl = self._calculate_realized_pnl(position, closing_quantity, price)
                    position.realized_pnl += realized_pnl
                    
                    # Set new position
                    position.quantity = opening_quantity
                    position.average_price = price
                    position.cost_basis = abs(opening_quantity * price)
            
            # Update timestamps
            position.last_trade_time = timestamp
            position.last_update_time = timestamp
            
            # Update market price if available
            if symbol in self.market_prices:
                position.update_market_price(self.market_prices[symbol])
            else:
                position.market_price = price
                position.unrealized_pnl = 0.0
            
            # Update metrics
            self.metrics['total_trades_processed'] += 1
            
            # Emit position event
            self._emit_position_event("TRADE_PROCESSED", position, {
                'trade_quantity': quantity,
                'trade_price': price,
                'timestamp': timestamp
            })
            
            # Update portfolio summary
            self._update_portfolio_summary()
            
            # Check risk limits
            self._check_risk_limits()
    
    def update_market_price(self, symbol: str, price: float) -> None:
        """
        Update market price for a symbol
        
        Args:
            symbol: Symbol to update
            price: New market price
        """
        
        with self.lock:
            self.market_prices[symbol] = price
            self.price_history[symbol].append(price)
            
            # Update position if exists
            if symbol in self.positions:
                position = self.positions[symbol]
                old_pnl = position.unrealized_pnl
                
                position.update_market_price(price)
                
                # Update P&L attribution
                pnl_change = position.unrealized_pnl - old_pnl
                self.pnl_attribution[symbol] += pnl_change
                
                # Emit price update event
                self._emit_position_event("PRICE_UPDATE", position, {
                    'old_price': position.market_price,
                    'new_price': price,
                    'pnl_change': pnl_change
                })
            
            self.metrics['position_updates'] += 1
            self.metrics['last_update_time'] = datetime.now()
            
            # Update portfolio summary
            self._update_portfolio_summary()
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol"""
        with self.lock:
            return self.positions.get(symbol)
    
    def get_all_positions(self) -> Dict[str, Position]:
        """Get all positions"""
        with self.lock:
            return self.positions.copy()
    
    def get_portfolio_summary(self) -> PortfolioSummary:
        """Get portfolio summary"""
        with self.lock:
            return self.portfolio_summary
    
    def get_positions_by_type(self, position_type: PositionType) -> List[Position]:
        """Get positions by type"""
        with self.lock:
            return [pos for pos in self.positions.values() 
                   if pos.position_type == position_type]
    
    def add_risk_limit(self, limit_id: str, risk_limit: RiskLimit) -> None:
        """Add risk limit"""
        with self.lock:
            self.risk_limits[limit_id] = risk_limit
    
    def remove_risk_limit(self, limit_id: str) -> None:
        """Remove risk limit"""
        with self.lock:
            if limit_id in self.risk_limits:
                del self.risk_limits[limit_id]
    
    def get_risk_status(self) -> Tuple[RiskStatus, List[str]]:
        """
        Get current risk status
        
        Returns:
            (risk_status, list_of_breach_messages)
        """
        
        with self.lock:
            breach_messages = []
            max_status = RiskStatus.GREEN
            
            for limit_id, limit in self.risk_limits.items():
                if limit.hard_breach:
                    breach_messages.append(f"HARD BREACH: {limit_id}")
                    max_status = RiskStatus.RED
                elif limit.soft_breach:
                    breach_messages.append(f"Soft breach: {limit_id}")
                    if max_status == RiskStatus.GREEN:
                        max_status = RiskStatus.YELLOW
            
            return max_status, breach_messages
    
    def close_position(self, symbol: str, price: Optional[float] = None) -> bool:
        """
        Close position completely
        
        Args:
            symbol: Symbol to close
            price: Closing price (uses market price if None)
        
        Returns:
            True if position was closed
        """
        
        with self.lock:
            if symbol not in self.positions:
                return False
            
            position = self.positions[symbol]
            
            if position.quantity == 0:
                return True  # Already flat
            
            # Use market price if not provided
            if price is None:
                price = self.market_prices.get(symbol, position.average_price)
            
            # Create closing trade
            closing_quantity = -position.quantity
            
            self.process_trade(symbol, closing_quantity, price)
            
            self._emit_position_event("POSITION_CLOSED", position, {
                'closing_price': price,
                'closing_quantity': closing_quantity
            })
            
            return True
    
    def calculate_var(self, confidence: float = 0.05, days: int = 1) -> float:
        """
        Calculate portfolio Value at Risk
        
        Args:
            confidence: Confidence level (e.g., 0.05 for 95% VaR)
            days: Time horizon in days
        
        Returns:
            VaR value
        """
        
        with self.lock:
            portfolio_returns = []
            
            # Calculate historical returns for each position
            for symbol, position in self.positions.items():
                if symbol in self.price_history and len(self.price_history[symbol]) > days:
                    prices = list(self.price_history[symbol])
                    returns = np.diff(prices) / prices[:-1]
                    
                    # Scale by position size
                    position_returns = returns * position.notional_value
                    portfolio_returns.extend(position_returns)
            
            if not portfolio_returns:
                return 0.0
            
            # Calculate VaR
            portfolio_returns = np.array(portfolio_returns)
            var_value = np.percentile(portfolio_returns, confidence * 100)
            
            # Scale for time horizon
            if days > 1:
                var_value *= np.sqrt(days)
            
            return abs(var_value)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        
        with self.lock:
            total_positions = len(self.positions)
            active_positions = len([p for p in self.positions.values() if p.quantity != 0])
            
            # Risk metrics
            current_risk_status, breach_messages = self.get_risk_status()
            
            # P&L metrics
            total_unrealized = sum(p.unrealized_pnl for p in self.positions.values())
            total_realized = sum(p.realized_pnl for p in self.positions.values())
            
            return {
                'total_positions': total_positions,
                'active_positions': active_positions,
                'total_unrealized_pnl': total_unrealized,
                'total_realized_pnl': total_realized,
                'portfolio_value': self.portfolio_summary.net_value,
                'gross_exposure': self.portfolio_summary.gross_value,
                'risk_status': current_risk_status.value,
                'risk_breaches': len(breach_messages),
                'trades_processed': self.metrics['total_trades_processed'],
                'position_updates': self.metrics['position_updates'],
                'last_update': self.metrics['last_update_time']
            }
    
    def add_position_event_handler(self, handler: Callable) -> None:
        """Add position event handler"""
        self.position_event_handlers.append(handler)
    
    def add_risk_event_handler(self, handler: Callable) -> None:
        """Add risk event handler"""
        self.risk_event_handlers.append(handler)
    
    def _calculate_realized_pnl(self, position: Position, quantity: float, price: float) -> float:
        """Calculate realized P&L for a trade"""
        
        if position.quantity == 0 or quantity == 0:
            return 0.0
        
        # P&L per share
        pnl_per_share = price - position.average_price
        
        # Adjust sign based on position direction
        if position.quantity > 0:  # Long position
            realized_pnl = pnl_per_share * abs(quantity)
        else:  # Short position
            realized_pnl = -pnl_per_share * abs(quantity)
        
        return realized_pnl
    
    def _update_portfolio_summary(self) -> None:
        """Update portfolio-level summary"""
        
        summary = PortfolioSummary()
        
        for position in self.positions.values():
            if position.quantity > 0:  # Long position
                summary.total_long_value += position.market_value
                summary.long_positions += 1
            elif position.quantity < 0:  # Short position
                summary.total_short_value += abs(position.market_value)
                summary.short_positions += 1
            
            summary.total_unrealized_pnl += position.unrealized_pnl
            summary.total_realized_pnl += position.realized_pnl
        
        summary.net_value = summary.total_long_value - summary.total_short_value
        summary.gross_value = summary.total_long_value + summary.total_short_value
        summary.total_positions = len(self.positions)
        summary.daily_pnl = summary.total_unrealized_pnl  # Simplified
        
        self.portfolio_summary = summary
    
    def _check_risk_limits(self) -> None:
        """Check all risk limits"""
        
        breaches = []
        
        for limit_id, limit in self.risk_limits.items():
            # Update current value based on limit type
            if limit.limit_type == 'position_size':
                if limit.symbol and limit.symbol in self.positions:
                    limit.current_value = abs(self.positions[limit.symbol].quantity)
                else:
                    limit.current_value = 0.0
                    
            elif limit.limit_type == 'notional':
                if limit.symbol and limit.symbol in self.positions:
                    limit.current_value = self.positions[limit.symbol].notional_value
                else:
                    limit.current_value = self.portfolio_summary.gross_value
                    
            elif limit.limit_type == 'pnl':
                if limit.symbol and limit.symbol in self.positions:
                    limit.current_value = self.positions[limit.symbol].unrealized_pnl
                else:
                    limit.current_value = self.portfolio_summary.total_unrealized_pnl
                    
            elif limit.limit_type == 'var':
                limit.current_value = self.calculate_var()
            
            # Check for breaches
            if limit.hard_breach or limit.soft_breach:
                limit.breach_count += 1
                limit.last_breach_time = datetime.now()
                
                breaches.append({
                    'limit_id': limit_id,
                    'limit': limit,
                    'is_hard_breach': limit.hard_breach
                })
                
                self.metrics['risk_breaches'] += 1
        
        # Update risk status
        if any(b['is_hard_breach'] for b in breaches):
            self.risk_status = RiskStatus.RED
        elif breaches:
            self.risk_status = RiskStatus.YELLOW
        else:
            self.risk_status = RiskStatus.GREEN
        
        # Emit risk events
        for breach in breaches:
            self._emit_risk_event("RISK_BREACH", breach['limit'], breach['is_hard_breach'])
    
    def _update_loop(self) -> None:
        """Main update loop"""
        
        while self.is_running:
            try:
                with self.lock:
                    # Update portfolio summary
                    self._update_portfolio_summary()
                    
                    # Check risk limits
                    self._check_risk_limits()
                
                # Sleep briefly
                threading.Event().wait(1.0)
                
            except Exception as e:
                print(f"Error in position update loop: {e}")
    
    def _emit_position_event(self, event_type: str, position: Position, data: Dict) -> None:
        """Emit position event"""
        
        event = {
            'type': event_type,
            'position': position,
            'data': data,
            'timestamp': datetime.now()
        }
        
        for handler in self.position_event_handlers:
            try:
                handler(event)
            except Exception as e:
                print(f"Error in position event handler: {e}")
    
    def _emit_risk_event(self, event_type: str, risk_limit: RiskLimit, is_hard: bool) -> None:
        """Emit risk event"""
        
        event = {
            'type': event_type,
            'risk_limit': risk_limit,
            'is_hard_breach': is_hard,
            'timestamp': datetime.now()
        }
        
        for handler in self.risk_event_handlers:
            try:
                handler(event)
            except Exception as e:
                print(f"Error in risk event handler: {e}")


if __name__ == "__main__":
    import time
    
    # Example usage and testing
    print("Testing Position Manager...")
    
    # Create position manager
    pm = PositionManager("TestPM")
    
    # Add event handlers
    def position_event_handler(event):
        pos = event['position']
        print(f"Position Event: {event['type']} - {pos.symbol} "
              f"Qty: {pos.quantity}, P&L: {pos.unrealized_pnl:.2f}")
    
    def risk_event_handler(event):
        limit = event['risk_limit']
        print(f"Risk Event: {event['type']} - {limit.limit_type} "
              f"Value: {limit.current_value:.2f}, Limit: {limit.soft_limit}")
    
    pm.add_position_event_handler(position_event_handler)
    pm.add_risk_event_handler(risk_event_handler)
    
    # Add risk limits
    pm.add_risk_limit("max_position_AAPL", RiskLimit(
        limit_type="position_size",
        symbol="AAPL",
        soft_limit=5000,
        hard_limit=10000
    ))
    
    pm.add_risk_limit("max_portfolio_var", RiskLimit(
        limit_type="var",
        soft_limit=100000,
        hard_limit=200000
    ))
    
    # Start position manager
    pm.start()
    
    try:
        # Simulate trading activity
        print("\nSimulating trades...")
        
        # Initial position
        pm.process_trade("AAPL", 1000, 150.00)
        pm.update_market_price("AAPL", 152.00)
        
        time.sleep(1)
        
        # Add to position
        pm.process_trade("AAPL", 500, 151.50)
        pm.update_market_price("AAPL", 153.00)
        
        # Different symbol
        pm.process_trade("MSFT", -800, 300.00)
        pm.update_market_price("MSFT", 298.00)
        
        time.sleep(1)
        
        # Partial close
        pm.process_trade("AAPL", -600, 154.00)
        
        time.sleep(1)
        
        # Check positions
        print(f"\nCurrent Positions:")
        for symbol, position in pm.get_all_positions().items():
            print(f"  {symbol}: Qty={position.quantity}, "
                  f"Avg Price={position.average_price:.2f}, "
                  f"Market Price={position.market_price:.2f}, "
                  f"P&L={position.unrealized_pnl:.2f}")
        
        # Portfolio summary
        summary = pm.get_portfolio_summary()
        print(f"\nPortfolio Summary:")
        print(f"  Long Value: ${summary.total_long_value:.2f}")
        print(f"  Short Value: ${summary.total_short_value:.2f}")
        print(f"  Net Value: ${summary.net_value:.2f}")
        print(f"  Gross Value: ${summary.gross_value:.2f}")
        print(f"  Unrealized P&L: ${summary.total_unrealized_pnl:.2f}")
        print(f"  Total Positions: {summary.total_positions}")
        
        # Risk status
        risk_status, breach_messages = pm.get_risk_status()
        print(f"\nRisk Status: {risk_status.value}")
        for msg in breach_messages:
            print(f"  {msg}")
        
        # Test large position (should trigger risk limit)
        print(f"\nTesting risk limits...")
        pm.process_trade("AAPL", 4500, 155.00)  # Should trigger soft limit
        
        time.sleep(2)
        
        # Performance metrics
        print(f"\nPerformance Metrics:")
        metrics = pm.get_performance_metrics()
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        
        # Test position closing
        print(f"\nClosing MSFT position...")
        pm.close_position("MSFT", 299.50)
        
        time.sleep(1)
        
        # Final positions
        print(f"\nFinal Positions:")
        active_positions = [p for p in pm.get_all_positions().values() if p.quantity != 0]
        for position in active_positions:
            print(f"  {position.symbol}: Qty={position.quantity}, "
                  f"P&L={position.unrealized_pnl:.2f}")
        
    finally:
        # Stop position manager
        pm.stop()
    
    print("\nPosition Manager testing completed!")