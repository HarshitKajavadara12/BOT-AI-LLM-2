"""
Transaction Cost Model Framework
Advanced transaction cost modeling for realistic backtesting
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
from collections import defaultdict
import concurrent.futures


class OrderSide(Enum):
    """Order side"""
    BUY = "BUY"
    SELL = "SELL"


class VenueType(Enum):
    """Trading venue types"""
    EXCHANGE = "EXCHANGE"              # Primary exchange
    DARK_POOL = "DARK_POOL"           # Dark pool
    ECN = "ECN"                       # Electronic Communication Network
    MARKET_MAKER = "MARKET_MAKER"     # Market maker


@dataclass
class MarketData:
    """Market data snapshot"""
    symbol: str
    timestamp: datetime
    price: float
    bid: float = 0.0
    ask: float = 0.0
    bid_size: float = 0.0
    ask_size: float = 0.0
    volume: float = 0.0
    volatility: float = 0.0
    spread: float = 0.0
    
    def __post_init__(self):
        if self.spread == 0.0 and self.bid > 0 and self.ask > 0:
            self.spread = self.ask - self.bid


@dataclass
class OrderInfo:
    """Order information for cost calculation"""
    symbol: str
    side: OrderSide  
    quantity: float
    order_type: str = "MARKET"  # MARKET, LIMIT, etc.
    urgency: float = 1.0        # 0-1, higher = more urgent
    timestamp: datetime = field(default_factory=datetime.now)
    strategy_name: str = ""
    
    @property
    def notional_value(self) -> float:
        """Get notional value (needs price)"""
        return 0.0  # Will be calculated with market price


@dataclass
class TransactionCost:
    """Transaction cost breakdown"""
    symbol: str
    side: OrderSide
    quantity: float
    reference_price: float
    execution_price: float
    
    # Cost components (in dollars)
    commission: float = 0.0
    spread_cost: float = 0.0
    market_impact: float = 0.0
    timing_cost: float = 0.0
    opportunity_cost: float = 0.0
    
    # Additional fees
    exchange_fees: float = 0.0
    regulatory_fees: float = 0.0
    other_fees: float = 0.0
    
    @property
    def total_cost(self) -> float:
        """Total transaction cost in dollars"""
        return (self.commission + self.spread_cost + self.market_impact + 
                self.timing_cost + self.opportunity_cost + self.exchange_fees + 
                self.regulatory_fees + self.other_fees)
    
    @property
    def total_cost_bps(self) -> float:
        """Total transaction cost in basis points"""
        notional = abs(self.quantity * self.reference_price)
        if notional > 0:
            return (self.total_cost / notional) * 10000
        return 0.0
    
    @property
    def slippage(self) -> float:
        """Price slippage in dollars per share"""
        if self.side == OrderSide.BUY:
            return max(0, self.execution_price - self.reference_price)
        else:
            return max(0, self.reference_price - self.execution_price)
    
    @property
    def slippage_bps(self) -> float:
        """Price slippage in basis points"""
        if self.reference_price > 0:
            return (self.slippage / self.reference_price) * 10000
        return 0.0


class TransactionCostModel(ABC):
    """Abstract base class for transaction cost models"""
    
    @abstractmethod
    def calculate_cost(self, order: OrderInfo, market_data: MarketData) -> TransactionCost:
        """Calculate transaction costs for an order"""
        pass
    
    @abstractmethod
    def estimate_market_impact(self, order: OrderInfo, market_data: MarketData) -> float:
        """Estimate market impact in dollars per share"""
        pass


class SimpleTransactionCostModel(TransactionCostModel):
    """Simple transaction cost model with fixed rates"""
    
    def __init__(self, 
                 commission_per_share: float = 0.005,
                 commission_min: float = 1.0,
                 spread_capture_rate: float = 0.5,
                 market_impact_rate: float = 0.001):
        """
        Initialize simple cost model
        
        Args:
            commission_per_share: Commission per share
            commission_min: Minimum commission per order
            spread_capture_rate: Fraction of spread paid (0-1)
            market_impact_rate: Market impact as fraction of price
        """
        self.commission_per_share = commission_per_share
        self.commission_min = commission_min
        self.spread_capture_rate = spread_capture_rate
        self.market_impact_rate = market_impact_rate
    
    def calculate_cost(self, order: OrderInfo, market_data: MarketData) -> TransactionCost:
        """Calculate simple transaction costs"""
        
        # Reference price (mid price or last price)
        if market_data.bid > 0 and market_data.ask > 0:
            reference_price = (market_data.bid + market_data.ask) / 2
        else:
            reference_price = market_data.price
        
        # Commission
        commission = max(self.commission_min, 
                        abs(order.quantity) * self.commission_per_share)
        
        # Spread cost
        spread = market_data.spread if market_data.spread > 0 else reference_price * 0.001
        spread_cost = abs(order.quantity) * spread * self.spread_capture_rate
        
        # Market impact
        market_impact_per_share = self.estimate_market_impact(order, market_data)
        market_impact = abs(order.quantity) * market_impact_per_share
        
        # Execution price (includes spread and impact)
        if order.side == OrderSide.BUY:
            execution_price = reference_price + spread * self.spread_capture_rate + market_impact_per_share
        else:
            execution_price = reference_price - spread * self.spread_capture_rate - market_impact_per_share
        
        return TransactionCost(
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            reference_price=reference_price,
            execution_price=execution_price,
            commission=commission,
            spread_cost=spread_cost,
            market_impact=market_impact
        )
    
    def estimate_market_impact(self, order: OrderInfo, market_data: MarketData) -> float:
        """Estimate linear market impact"""
        return abs(order.quantity) * market_data.price * self.market_impact_rate


class AlmgrenChrissCostModel(TransactionCostModel):
    """
    Almgren-Chriss transaction cost model
    Implements optimal execution with temporary and permanent impact
    """
    
    def __init__(self,
                 commission_rate: float = 0.001,
                 spread_rate: float = 0.0005,
                 temp_impact_coeff: float = 0.1,
                 perm_impact_coeff: float = 0.01,
                 volatility_scaling: float = 0.5):
        """
        Initialize Almgren-Chriss model
        
        Args:
            commission_rate: Commission as fraction of notional
            spread_rate: Spread cost as fraction of notional  
            temp_impact_coeff: Temporary impact coefficient
            perm_impact_coeff: Permanent impact coefficient
            volatility_scaling: Volatility scaling factor
        """
        self.commission_rate = commission_rate
        self.spread_rate = spread_rate
        self.temp_impact_coeff = temp_impact_coeff
        self.perm_impact_coeff = perm_impact_coeff
        self.volatility_scaling = volatility_scaling
    
    def calculate_cost(self, order: OrderInfo, market_data: MarketData) -> TransactionCost:
        """Calculate Almgren-Chriss transaction costs"""
        
        reference_price = market_data.price
        notional = abs(order.quantity * reference_price)
        
        # Commission
        commission = notional * self.commission_rate
        
        # Spread cost
        spread_cost = notional * self.spread_rate
        
        # Market impact components
        temp_impact, perm_impact = self._calculate_market_impact(order, market_data)
        total_impact = temp_impact + perm_impact
        
        # Execution price
        impact_per_share = total_impact / abs(order.quantity) if order.quantity != 0 else 0
        
        if order.side == OrderSide.BUY:
            execution_price = reference_price + impact_per_share
        else:
            execution_price = reference_price - impact_per_share
        
        return TransactionCost(
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            reference_price=reference_price,
            execution_price=execution_price,
            commission=commission,
            spread_cost=spread_cost,
            market_impact=total_impact
        )
    
    def estimate_market_impact(self, order: OrderInfo, market_data: MarketData) -> float:
        """Estimate total market impact per share"""
        temp_impact, perm_impact = self._calculate_market_impact(order, market_data)
        return (temp_impact + perm_impact) / abs(order.quantity) if order.quantity != 0 else 0
    
    def _calculate_market_impact(self, order: OrderInfo, market_data: MarketData) -> Tuple[float, float]:
        """Calculate temporary and permanent market impact"""
        
        price = market_data.price
        volume = max(market_data.volume, 1000)  # Minimum volume assumption
        volatility = max(market_data.volatility, 0.01)  # Minimum volatility
        
        # Participation rate (fraction of volume)
        participation_rate = abs(order.quantity) / volume
        participation_rate = min(participation_rate, 1.0)  # Cap at 100%
        
        # Temporary impact: proportional to participation rate and volatility
        temp_impact = (self.temp_impact_coeff * 
                      participation_rate * 
                      volatility * price * 
                      abs(order.quantity) * 
                      order.urgency)
        
        # Permanent impact: square root of participation rate
        perm_impact = (self.perm_impact_coeff * 
                      np.sqrt(participation_rate) * 
                      volatility * price * 
                      abs(order.quantity))
        
        return temp_impact, perm_impact


class BerrasCostModel(TransactionCostModel):
    """
    Barra transaction cost model with regime-dependent costs
    """
    
    def __init__(self):
        """Initialize Barra cost model"""
        
        # Model parameters (calibrated to US equity markets)
        self.base_commission_rate = 0.0005
        self.base_spread_rate = 0.0003
        
        # Market impact parameters
        self.alpha = 0.6          # Power law exponent
        self.beta = 0.4           # Volatility sensitivity
        self.gamma = 0.3          # Volume sensitivity
        
        # Regime parameters
        self.stress_multiplier = 2.0    # Cost multiplier in stress
        self.volatility_threshold = 0.3  # Stress threshold
        
        # Currency and asset class adjustments
        self.currency_adjustments = {
            'USD': 1.0,
            'EUR': 1.1,
            'GBP': 1.2,
            'JPY': 1.3,
            'OTHER': 1.5
        }
        
        self.asset_class_adjustments = {
            'LARGE_CAP': 1.0,
            'MID_CAP': 1.3,
            'SMALL_CAP': 1.8,
            'EMERGING': 2.5
        }
    
    def calculate_cost(self, order: OrderInfo, market_data: MarketData) -> TransactionCost:
        """Calculate Barra transaction costs"""
        
        reference_price = market_data.price
        notional = abs(order.quantity * reference_price)
        
        # Detect market regime
        regime_multiplier = self._get_regime_multiplier(market_data)
        
        # Asset classification (simplified)
        asset_multiplier = self._get_asset_multiplier(order.symbol, reference_price)
        
        # Base costs
        commission = notional * self.base_commission_rate * asset_multiplier
        spread_cost = notional * self.base_spread_rate * asset_multiplier * regime_multiplier
        
        # Market impact
        market_impact = self._calculate_barra_market_impact(order, market_data, 
                                                          regime_multiplier, asset_multiplier)
        
        # Execution price
        impact_per_share = market_impact / abs(order.quantity) if order.quantity != 0 else 0
        
        if order.side == OrderSide.BUY:
            execution_price = reference_price + impact_per_share
        else:
            execution_price = reference_price - impact_per_share
        
        # Additional fees based on notional
        exchange_fees = notional * 0.00001  # 0.1 bps
        regulatory_fees = notional * 0.000005  # 0.05 bps
        
        return TransactionCost(
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            reference_price=reference_price,
            execution_price=execution_price,
            commission=commission,
            spread_cost=spread_cost,
            market_impact=market_impact,
            exchange_fees=exchange_fees,
            regulatory_fees=regulatory_fees
        )
    
    def estimate_market_impact(self, order: OrderInfo, market_data: MarketData) -> float:
        """Estimate Barra market impact per share"""
        regime_multiplier = self._get_regime_multiplier(market_data)
        asset_multiplier = self._get_asset_multiplier(order.symbol, market_data.price)
        
        impact = self._calculate_barra_market_impact(order, market_data, 
                                                   regime_multiplier, asset_multiplier)
        return impact / abs(order.quantity) if order.quantity != 0 else 0
    
    def _get_regime_multiplier(self, market_data: MarketData) -> float:
        """Get regime-dependent cost multiplier"""
        volatility = market_data.volatility
        
        if volatility > self.volatility_threshold:
            return self.stress_multiplier
        else:
            # Smooth transition
            stress_factor = volatility / self.volatility_threshold
            return 1.0 + (self.stress_multiplier - 1.0) * stress_factor
    
    def _get_asset_multiplier(self, symbol: str, price: float) -> float:
        """Get asset class multiplier (simplified classification)"""
        
        # Simple heuristic based on price (in practice, use market cap)
        if price > 100:
            return self.asset_class_adjustments['LARGE_CAP']
        elif price > 20:
            return self.asset_class_adjustments['MID_CAP']
        else:
            return self.asset_class_adjustments['SMALL_CAP']
    
    def _calculate_barra_market_impact(self, order: OrderInfo, market_data: MarketData,
                                     regime_multiplier: float, asset_multiplier: float) -> float:
        """Calculate market impact using Barra model"""
        
        price = market_data.price
        volume = max(market_data.volume, 1000)
        volatility = max(market_data.volatility, 0.01)
        
        # Participation rate
        participation_rate = abs(order.quantity) / volume
        participation_rate = min(participation_rate, 1.0)
        
        # Power law impact with volatility and volume adjustments
        base_impact = (participation_rate ** self.alpha * 
                      volatility ** self.beta * 
                      (1 / np.log10(volume + 1)) ** self.gamma)
        
        # Scale by price and quantity
        market_impact = (base_impact * price * abs(order.quantity) * 
                        regime_multiplier * asset_multiplier * order.urgency)
        
        return market_impact


class ITGCostModel(TransactionCostModel):
    """
    ITG ACE transaction cost model
    """
    
    def __init__(self):
        """Initialize ITG cost model"""
        
        self.base_spread_capture = 0.3      # Fraction of spread captured
        self.min_spread_bps = 5.0           # Minimum spread in bps
        
        # Market impact parameters
        self.impact_coefficients = {
            'LARGE_CAP': {'temp': 0.05, 'perm': 0.01},
            'MID_CAP': {'temp': 0.08, 'perm': 0.02},
            'SMALL_CAP': {'temp': 0.15, 'perm': 0.04}
        }
        
        # Timing risk parameters
        self.timing_risk_factor = 0.1       # Timing risk coefficient
        
    def calculate_cost(self, order: OrderInfo, market_data: MarketData) -> TransactionCost:
        """Calculate ITG transaction costs"""
        
        reference_price = market_data.price
        notional = abs(order.quantity * reference_price)
        
        # Commission (assumed to be included in spread capture)
        commission = 0.0
        
        # Spread cost
        effective_spread = max(market_data.spread, 
                              reference_price * self.min_spread_bps / 10000)
        spread_cost = abs(order.quantity) * effective_spread * self.base_spread_capture
        
        # Market impact
        market_impact = self._calculate_itg_market_impact(order, market_data)
        
        # Timing cost (opportunity cost of not trading immediately)
        timing_cost = self._calculate_timing_cost(order, market_data)
        
        # Execution price
        total_impact_per_share = ((market_impact + timing_cost) / abs(order.quantity) 
                                if order.quantity != 0 else 0)
        
        if order.side == OrderSide.BUY:
            execution_price = (reference_price + effective_spread * self.base_spread_capture + 
                             total_impact_per_share)
        else:
            execution_price = (reference_price - effective_spread * self.base_spread_capture - 
                             total_impact_per_share)
        
        return TransactionCost(
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            reference_price=reference_price,
            execution_price=execution_price,
            commission=commission,
            spread_cost=spread_cost,
            market_impact=market_impact,
            timing_cost=timing_cost
        )
    
    def estimate_market_impact(self, order: OrderInfo, market_data: MarketData) -> float:
        """Estimate ITG market impact per share"""
        impact = self._calculate_itg_market_impact(order, market_data)
        return impact / abs(order.quantity) if order.quantity != 0 else 0
    
    def _calculate_itg_market_impact(self, order: OrderInfo, market_data: MarketData) -> float:
        """Calculate ITG market impact"""
        
        price = market_data.price
        volume = max(market_data.volume, 1000)
        volatility = max(market_data.volatility, 0.01)
        
        # Classify stock (simplified)
        if price > 50:
            stock_class = 'LARGE_CAP'
        elif price > 10:
            stock_class = 'MID_CAP'
        else:
            stock_class = 'SMALL_CAP'
        
        coeffs = self.impact_coefficients[stock_class]
        
        # Participation rate
        participation_rate = min(abs(order.quantity) / volume, 1.0)
        
        # Temporary impact
        temp_impact = (coeffs['temp'] * 
                      participation_rate * 
                      volatility * price * 
                      abs(order.quantity))
        
        # Permanent impact
        perm_impact = (coeffs['perm'] * 
                      np.sqrt(participation_rate) * 
                      volatility * price * 
                      abs(order.quantity))
        
        return temp_impact + perm_impact
    
    def _calculate_timing_cost(self, order: OrderInfo, market_data: MarketData) -> float:
        """Calculate timing/opportunity cost"""
        
        # Simple model: proportional to volatility and order size
        volatility = max(market_data.volatility, 0.01)
        
        timing_cost = (self.timing_risk_factor * 
                      volatility * 
                      market_data.price * 
                      abs(order.quantity) * 
                      (1 - order.urgency))  # Higher urgency = lower timing cost
        
        return timing_cost


class MultiVenueCostModel(TransactionCostModel):
    """
    Multi-venue transaction cost model considering venue selection
    """
    
    def __init__(self):
        """Initialize multi-venue cost model"""
        
        # Venue characteristics
        self.venue_costs = {
            VenueType.EXCHANGE: {
                'commission_rate': 0.0005,
                'spread_capture': 0.5,
                'market_impact': 1.0,
                'fill_probability': 0.9
            },
            VenueType.DARK_POOL: {
                'commission_rate': 0.0003,
                'spread_capture': 0.1,  # Better price improvement
                'market_impact': 0.3,   # Lower market impact
                'fill_probability': 0.6  # Lower fill rate
            },
            VenueType.ECN: {
                'commission_rate': 0.0004,
                'spread_capture': 0.3,
                'market_impact': 0.7,
                'fill_probability': 0.8
            },
            VenueType.MARKET_MAKER: {
                'commission_rate': 0.0008,
                'spread_capture': 0.6,
                'market_impact': 1.2,
                'fill_probability': 0.95
            }
        }
    
    def calculate_cost(self, order: OrderInfo, market_data: MarketData) -> TransactionCost:
        """Calculate multi-venue transaction costs"""
        
        # Select optimal venue
        optimal_venue = self._select_optimal_venue(order, market_data)
        venue_params = self.venue_costs[optimal_venue]
        
        reference_price = market_data.price
        notional = abs(order.quantity * reference_price)
        
        # Calculate costs based on selected venue
        commission = notional * venue_params['commission_rate']
        
        effective_spread = max(market_data.spread, reference_price * 0.0001)
        spread_cost = (abs(order.quantity) * effective_spread * 
                      venue_params['spread_capture'])
        
        # Market impact adjusted for venue
        base_impact = abs(order.quantity) * reference_price * 0.001
        market_impact = base_impact * venue_params['market_impact']
        
        # Opportunity cost for potential non-fill
        opportunity_cost = self._calculate_opportunity_cost(order, market_data, 
                                                          venue_params['fill_probability'])
        
        # Execution price
        impact_per_share = market_impact / abs(order.quantity) if order.quantity != 0 else 0
        
        if order.side == OrderSide.BUY:
            execution_price = (reference_price + 
                             effective_spread * venue_params['spread_capture'] + 
                             impact_per_share)
        else:
            execution_price = (reference_price - 
                             effective_spread * venue_params['spread_capture'] - 
                             impact_per_share)
        
        return TransactionCost(
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            reference_price=reference_price,
            execution_price=execution_price,
            commission=commission,
            spread_cost=spread_cost,
            market_impact=market_impact,
            opportunity_cost=opportunity_cost
        )
    
    def estimate_market_impact(self, order: OrderInfo, market_data: MarketData) -> float:
        """Estimate market impact per share"""
        optimal_venue = self._select_optimal_venue(order, market_data)
        venue_params = self.venue_costs[optimal_venue]
        
        base_impact = market_data.price * 0.001
        return base_impact * venue_params['market_impact']
    
    def _select_optimal_venue(self, order: OrderInfo, market_data: MarketData) -> VenueType:
        """Select optimal venue based on order characteristics"""
        
        # Simple venue selection logic
        notional = abs(order.quantity * market_data.price)
        
        # Large orders prefer dark pools
        if notional > 100000 and order.urgency < 0.5:
            return VenueType.DARK_POOL
        
        # Urgent small orders go to exchange
        elif notional < 10000 or order.urgency > 0.8:
            return VenueType.EXCHANGE
        
        # Medium orders use ECN
        else:
            return VenueType.ECN
    
    def _calculate_opportunity_cost(self, order: OrderInfo, market_data: MarketData, 
                                  fill_probability: float) -> float:
        """Calculate opportunity cost of potential non-fill"""
        
        # Cost if order doesn't fill and market moves
        non_fill_probability = 1 - fill_probability
        volatility = max(market_data.volatility, 0.01)
        
        # Expected cost of delayed execution
        opportunity_cost = (non_fill_probability * 
                          volatility * 
                          market_data.price * 
                          abs(order.quantity) * 
                          0.5)  # Assume 50% adverse selection
        
        return opportunity_cost


class AdaptiveTransactionCostModel(TransactionCostModel):
    """
    Adaptive transaction cost model that learns from execution data
    """
    
    def __init__(self, learning_rate: float = 0.1):
        """Initialize adaptive cost model"""
        
        self.learning_rate = learning_rate
        
        # Base model parameters
        self.base_commission_rate = 0.0005
        self.base_spread_rate = 0.0003
        self.base_impact_coeff = 0.001
        
        # Adaptive parameters (updated based on execution history)
        self.adaptive_params = {
            'commission_adjustment': 1.0,
            'spread_adjustment': 1.0,
            'impact_adjustment': 1.0
        }
        
        # Execution history for learning
        self.execution_history: List[Dict[str, Any]] = []
        
    def calculate_cost(self, order: OrderInfo, market_data: MarketData) -> TransactionCost:
        """Calculate adaptive transaction costs"""
        
        reference_price = market_data.price
        notional = abs(order.quantity * reference_price)
        
        # Apply adaptive adjustments
        commission = (notional * self.base_commission_rate * 
                     self.adaptive_params['commission_adjustment'])
        
        effective_spread = max(market_data.spread, reference_price * 0.0001)
        spread_cost = (abs(order.quantity) * effective_spread * 
                      self.base_spread_rate * 
                      self.adaptive_params['spread_adjustment'])
        
        market_impact = (abs(order.quantity) * reference_price * 
                        self.base_impact_coeff * 
                        self.adaptive_params['impact_adjustment'])
        
        # Execution price
        impact_per_share = market_impact / abs(order.quantity) if order.quantity != 0 else 0
        
        if order.side == OrderSide.BUY:
            execution_price = reference_price + impact_per_share
        else:
            execution_price = reference_price - impact_per_share
        
        return TransactionCost(
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            reference_price=reference_price,
            execution_price=execution_price,
            commission=commission,
            spread_cost=spread_cost,
            market_impact=market_impact
        )
    
    def estimate_market_impact(self, order: OrderInfo, market_data: MarketData) -> float:
        """Estimate adaptive market impact per share"""
        base_impact = market_data.price * self.base_impact_coeff
        return base_impact * self.adaptive_params['impact_adjustment']
    
    def update_model(self, predicted_cost: TransactionCost, actual_cost: TransactionCost):
        """Update model parameters based on actual execution results"""
        
        # Calculate prediction errors
        commission_error = actual_cost.commission - predicted_cost.commission
        spread_error = actual_cost.spread_cost - predicted_cost.spread_cost  
        impact_error = actual_cost.market_impact - predicted_cost.market_impact
        
        # Update adaptive parameters using simple gradient descent
        if predicted_cost.commission > 0:
            self.adaptive_params['commission_adjustment'] += (
                self.learning_rate * commission_error / predicted_cost.commission)
        
        if predicted_cost.spread_cost > 0:
            self.adaptive_params['spread_adjustment'] += (
                self.learning_rate * spread_error / predicted_cost.spread_cost)
        
        if predicted_cost.market_impact > 0:  
            self.adaptive_params['impact_adjustment'] += (
                self.learning_rate * impact_error / predicted_cost.market_impact)
        
        # Bound adjustments to reasonable ranges
        for param in self.adaptive_params:
            self.adaptive_params[param] = np.clip(self.adaptive_params[param], 0.1, 3.0)
        
        # Store execution for future analysis
        execution_record = {
            'timestamp': datetime.now(),
            'symbol': actual_cost.symbol,
            'predicted_total_cost_bps': predicted_cost.total_cost_bps,
            'actual_total_cost_bps': actual_cost.total_cost_bps,
            'commission_adjustment': self.adaptive_params['commission_adjustment'],
            'spread_adjustment': self.adaptive_params['spread_adjustment'],
            'impact_adjustment': self.adaptive_params['impact_adjustment']
        }
        self.execution_history.append(execution_record)


class TransactionCostAnalyzer:
    """
    Analyzer for transaction cost performance and attribution
    """
    
    def __init__(self):
        """Initialize cost analyzer"""
        self.cost_history: List[TransactionCost] = []
        
    def add_transaction(self, transaction_cost: TransactionCost):
        """Add transaction cost to history"""
        self.cost_history.append(transaction_cost)
    
    def analyze_costs(self, start_date: Optional[datetime] = None, 
                     end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Analyze transaction costs over specified period"""
        
        if not self.cost_history:
            return {}
        
        # Filter by date if specified (assuming timestamp in cost object)
        costs = self.cost_history
        
        # Aggregate statistics
        total_costs = sum(cost.total_cost for cost in costs)
        total_notional = sum(abs(cost.quantity * cost.reference_price) for cost in costs)
        
        if total_notional == 0:
            return {'error': 'No notional value'}
        
        # Cost breakdown
        commission_total = sum(cost.commission for cost in costs)
        spread_total = sum(cost.spread_cost for cost in costs)
        impact_total = sum(cost.market_impact for cost in costs)
        timing_total = sum(cost.timing_cost for cost in costs)
        
        # Statistics by symbol
        symbol_stats = defaultdict(lambda: {
            'count': 0, 'total_cost': 0, 'total_notional': 0, 'avg_cost_bps': 0
        })
        
        for cost in costs:
            symbol = cost.symbol
            notional = abs(cost.quantity * cost.reference_price)
            
            symbol_stats[symbol]['count'] += 1
            symbol_stats[symbol]['total_cost'] += cost.total_cost
            symbol_stats[symbol]['total_notional'] += notional
        
        # Calculate average cost bps by symbol
        for symbol in symbol_stats:
            stats = symbol_stats[symbol]
            if stats['total_notional'] > 0:
                stats['avg_cost_bps'] = (stats['total_cost'] / stats['total_notional']) * 10000
        
        # Overall statistics
        analysis = {
            'period_summary': {
                'total_transactions': len(costs),
                'total_cost_dollars': total_costs,
                'total_notional': total_notional,
                'average_cost_bps': (total_costs / total_notional) * 10000
            },
            'cost_breakdown': {
                'commission_pct': (commission_total / total_costs) * 100 if total_costs > 0 else 0,
                'spread_pct': (spread_total / total_costs) * 100 if total_costs > 0 else 0,
                'market_impact_pct': (impact_total / total_costs) * 100 if total_costs > 0 else 0,
                'timing_pct': (timing_total / total_costs) * 100 if total_costs > 0 else 0
            },
            'symbol_analysis': dict(symbol_stats)
        }
        
        # Cost statistics
        cost_bps = [(cost.total_cost / abs(cost.quantity * cost.reference_price)) * 10000 
                   for cost in costs if cost.quantity * cost.reference_price != 0]
        
        if cost_bps:
            analysis['cost_distribution'] = {
                'mean_bps': np.mean(cost_bps),
                'median_bps': np.median(cost_bps),
                'std_bps': np.std(cost_bps),
                'min_bps': np.min(cost_bps),
                'max_bps': np.max(cost_bps),
                'percentiles': {
                    '25th': np.percentile(cost_bps, 25),
                    '75th': np.percentile(cost_bps, 75),
                    '90th': np.percentile(cost_bps, 90),
                    '95th': np.percentile(cost_bps, 95)
                }
            }
        
        return analysis
    
    def benchmark_model(self, model: TransactionCostModel, 
                       test_orders: List[Tuple[OrderInfo, MarketData]]) -> Dict[str, Any]:
        """Benchmark a transaction cost model against test data"""
        
        predictions = []
        actuals = []
        
        for order_info, market_data in test_orders:
            predicted_cost = model.calculate_cost(order_info, market_data)
            
            # Find corresponding actual cost (simplified lookup)
            actual_cost = None
            for cost in self.cost_history:
                if (cost.symbol == order_info.symbol and 
                    abs(cost.quantity - order_info.quantity) < 0.01):
                    actual_cost = cost
                    break
            
            if actual_cost:
                predictions.append(predicted_cost.total_cost_bps)
                actuals.append(actual_cost.total_cost_bps) 
        
        if not predictions:
            return {'error': 'No matching predictions found'}
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Calculate metrics
        mae = np.mean(np.abs(predictions - actuals))
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        
        correlation = np.corrcoef(predictions, actuals)[0, 1] if len(predictions) > 1 else 0
        
        bias = np.mean(predictions - actuals)
        
        return {
            'n_predictions': len(predictions),
            'mae_bps': mae,
            'rmse_bps': rmse,
            'correlation': correlation,
            'bias_bps': bias,
            'mean_predicted_bps': np.mean(predictions),
            'mean_actual_bps': np.mean(actuals)
        }
