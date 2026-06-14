"""
Liquidity Models for QUANTUM-FORGE
Implements sophisticated models for market liquidity analysis and prediction.
"""

import numpy as np
import pandas as pd
from scipy import stats, optimize
from scipy.integrate import quad
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional, Union, Callable
import warnings
from numba import jit, prange
from dataclasses import dataclass
from enum import Enum
from collections import deque
import networkx as nx
warnings.filterwarnings('ignore')

class LiquidityRegime(Enum):
    """Market liquidity regimes."""
    HIGH_LIQUIDITY = "high"
    NORMAL_LIQUIDITY = "normal" 
    LOW_LIQUIDITY = "low"
    STRESS_LIQUIDITY = "stress"
    CRISIS_LIQUIDITY = "crisis"

@dataclass
class LiquidityMetrics:
    """Comprehensive liquidity metrics."""
    timestamp: float
    bid_ask_spread: float
    effective_spread: float
    realized_spread: float
    price_impact: float
    market_depth: float
    resilience: float
    turnover_rate: float
    amihud_illiquidity: float
    volume_weighted_spread: float
    quoted_depth: float
    hidden_liquidity: float

@dataclass
class LiquidityEvent:
    """Liquidity event representation."""
    timestamp: float
    event_type: str
    severity: float
    duration: float
    affected_instruments: List[str]
    contributing_factors: Dict[str, float]

class AmihudModel:
    """Amihud (2002) illiquidity measure and extensions."""
    
    def __init__(self, lookback_days: int = 252):
        """
        Initialize Amihud illiquidity model.
        
        Args:
            lookback_days: Number of days for rolling calculation
        """
        self.lookback_days = lookback_days
        self.price_history = deque(maxlen=lookback_days)
        self.volume_history = deque(maxlen=lookback_days) 
        self.return_history = deque(maxlen=lookback_days)
        self.illiquidity_history = deque(maxlen=lookback_days)
    
    def add_observation(self, price: float, volume: float, timestamp: float):
        """Add price/volume observation."""
        if len(self.price_history) > 0:
            prev_price = self.price_history[-1]
            log_return = np.log(price / prev_price) if prev_price > 0 else 0
            self.return_history.append(log_return)
        
        self.price_history.append(price)
        self.volume_history.append(volume)
    
    def calculate_illiquidity(self) -> float:
        """
        Calculate Amihud illiquidity measure.
        
        Returns:
            Illiquidity measure (higher = less liquid)
        """
        if len(self.return_history) < 20 or len(self.volume_history) < 20:
            return 0.0
        
        # Calculate |return| / volume for each observation
        illiq_components = []
        
        for ret, vol in zip(self.return_history, self.volume_history):
            if vol > 0:
                illiq_components.append(abs(ret) / vol)
        
        if not illiq_components:
            return 0.0
        
        # Average over the period
        illiquidity = np.mean(illiq_components)
        self.illiquidity_history.append(illiquidity)
        
        return illiquidity
    
    def calculate_liquidity_beta(self, market_illiquidity: List[float]) -> float:
        """
        Calculate liquidity beta (sensitivity to market liquidity).
        
        Args:
            market_illiquidity: Market-wide illiquidity time series
        
        Returns:
            Liquidity beta coefficient
        """
        if len(self.illiquidity_history) < 50 or len(market_illiquidity) < 50:
            return 0.0
        
        # Align time series
        min_len = min(len(self.illiquidity_history), len(market_illiquidity))
        stock_illiq = list(self.illiquidity_history)[-min_len:]
        market_illiq = market_illiquidity[-min_len:]
        
        # Calculate changes in illiquidity
        stock_changes = np.diff(stock_illiq)
        market_changes = np.diff(market_illiq)
        
        if len(stock_changes) < 10:
            return 0.0
        
        # Regress stock illiquidity changes on market illiquidity changes
        try:
            slope, _, _, _, _ = stats.linregress(market_changes, stock_changes)
            return slope
        except:
            return 0.0

class KyleModel:
    """Kyle (1985) model of market microstructure and liquidity."""
    
    def __init__(self):
        """Initialize Kyle model parameters."""
        self.sigma_v = 1.0      # Volatility of asset value
        self.sigma_u = 1.0      # Volatility of noise trading
        self.beta = 1.0         # Price impact parameter (lambda)
        self.mu = 1.0           # Informed trader intensity
        
        self.fitted = False
        
    def fit(self, prices: np.ndarray, volumes: np.ndarray, order_flows: np.ndarray):
        """
        Fit Kyle model parameters using MLE or method of moments.
        
        Args:
            prices: Price time series
            volumes: Volume time series
            order_flows: Net order flow (signed volume)
        """
        if len(prices) < 50:
            return False
        
        # Calculate returns
        returns = np.diff(np.log(prices))
        
        # Estimate parameters using method of moments
        try:
            # Price impact (lambda) - regression of returns on order flow
            if len(order_flows) > len(returns):
                order_flows = order_flows[1:]  # Align with returns
            
            valid_idx = ~(np.isnan(returns) | np.isnan(order_flows) | np.isinf(returns) | np.isinf(order_flows))
            
            if np.sum(valid_idx) < 10:
                return False
            
            clean_returns = returns[valid_idx]
            clean_flows = order_flows[valid_idx]
            
            # Regress returns on order flow to get lambda (price impact)
            slope, _, _, _, _ = stats.linregress(clean_flows, clean_returns)
            self.beta = abs(slope)  # Price impact should be positive
            
            # Estimate noise trading volatility
            residuals = clean_returns - slope * clean_flows
            self.sigma_u = np.std(residuals) if len(residuals) > 1 else 1.0
            
            # Estimate value volatility (assuming equilibrium)
            self.sigma_v = np.std(clean_returns) if len(clean_returns) > 1 else 1.0
            
            # Informed trading intensity (simplified)
            self.mu = self.beta * self.sigma_u / self.sigma_v if self.sigma_v > 0 else 1.0
            
            self.fitted = True
            return True
            
        except Exception as e:
            return False
    
    def calculate_market_depth(self) -> float:
        """Calculate market depth (1/lambda)."""
        return 1.0 / max(self.beta, 1e-6)
    
    def calculate_adverse_selection_cost(self) -> float:
        """Calculate adverse selection component of spread."""
        if not self.fitted:
            return 0.0
        
        return self.beta * self.sigma_v / (2 * self.mu) if self.mu > 0 else 0.0
    
    def calculate_bid_ask_spread(self) -> float:
        """Calculate theoretical bid-ask spread."""
        if not self.fitted:
            return 0.0
        
        adverse_selection = self.calculate_adverse_selection_cost()
        inventory_cost = self.sigma_u / np.sqrt(2)  # Simplified inventory cost
        
        return 2 * (adverse_selection + inventory_cost)

class GlosstenMilgromModel:
    """Glosten-Milgrom (1985) sequential trade model."""
    
    def __init__(self, prior_prob: float = 0.5):
        """
        Initialize Glosten-Milgrom model.
        
        Args:
            prior_prob: Prior probability that trader is informed
        """
        self.prior_prob = prior_prob
        self.current_prob = prior_prob
        self.value_high = 1.0
        self.value_low = -1.0
        
        self.trade_history = []
        self.belief_history = deque(maxlen=1000)
    
    def update_beliefs(self, trade_direction: str) -> float:
        """
        Update beliefs about informed trading probability.
        
        Args:
            trade_direction: 'buy' or 'sell'
        
        Returns:
            Updated probability of informed trading
        """
        # Bayesian updating based on trade direction
        if trade_direction == 'buy':
            # Buy order received - update belief about high value
            likelihood_informed = 1.0  # Informed trader always buys when value is high
            likelihood_uninformed = 0.5  # Uninformed trader buys randomly
        else:  # sell
            # Sell order received - update belief about low value  
            likelihood_informed = 1.0  # Informed trader always sells when value is low
            likelihood_uninformed = 0.5  # Uninformed trader sells randomly
        
        # Bayes' rule
        numerator = likelihood_informed * self.current_prob
        denominator = (likelihood_informed * self.current_prob + 
                      likelihood_uninformed * (1 - self.current_prob))
        
        if denominator > 0:
            self.current_prob = numerator / denominator
        
        self.belief_history.append(self.current_prob)
        self.trade_history.append(trade_direction)
        
        return self.current_prob
    
    def calculate_bid_ask_quotes(self) -> Tuple[float, float]:
        """
        Calculate optimal bid and ask quotes.
        
        Returns:
            Tuple of (bid_price, ask_price)
        """
        expected_value = (self.current_prob * self.value_high + 
                         (1 - self.current_prob) * self.value_low)
        
        # Bid quote (for potential sell order)
        prob_informed_sell = self.current_prob if expected_value < 0 else (1 - self.current_prob)
        bid = (prob_informed_sell * self.value_low + 
               (1 - prob_informed_sell) * expected_value)
        
        # Ask quote (for potential buy order)
        prob_informed_buy = self.current_prob if expected_value > 0 else (1 - self.current_prob)
        ask = (prob_informed_buy * self.value_high + 
               (1 - prob_informed_buy) * expected_value)
        
        return bid, ask
    
    def calculate_spread(self) -> float:
        """Calculate bid-ask spread."""
        bid, ask = self.calculate_bid_ask_quotes()
        return ask - bid

class LiquidityProvider:
    """Sophisticated liquidity provision model with optimal pricing."""
    
    def __init__(self, inventory_capacity: float = 1000, risk_aversion: float = 0.01):
        """
        Initialize liquidity provider.
        
        Args:
            inventory_capacity: Maximum inventory position
            risk_aversion: Risk aversion parameter
        """
        self.inventory_capacity = inventory_capacity
        self.risk_aversion = risk_aversion
        self.current_inventory = 0.0
        
        # Model parameters
        self.arrival_rate_buy = 1.0
        self.arrival_rate_sell = 1.0
        self.volatility = 0.02
        self.tick_size = 0.01
        
        # History tracking
        self.pnl_history = deque(maxlen=10000)
        self.inventory_history = deque(maxlen=10000)
        self.quote_history = deque(maxlen=10000)
    
    def calculate_optimal_quotes(self, mid_price: float, time_to_horizon: float = 1.0) -> Tuple[float, float]:
        """
        Calculate optimal bid/ask quotes using Avellaneda-Stoikov framework.
        
        Args:
            mid_price: Current mid price
            time_to_horizon: Time remaining until liquidation
        
        Returns:
            Tuple of (optimal_bid, optimal_ask)
        """
        if time_to_horizon <= 0:
            return mid_price, mid_price
        
        # Risk adjustment for inventory
        inventory_penalty = self.risk_aversion * self.volatility**2 * time_to_horizon
        inventory_adjustment = inventory_penalty * self.current_inventory
        
        # Optimal spread calculation
        gamma = self.risk_aversion
        sigma = self.volatility
        T = time_to_horizon
        q = self.current_inventory
        
        # Reservation price
        reservation_price = mid_price - inventory_adjustment
        
        # Optimal half-spread
        half_spread = (gamma * sigma**2 * T / 2 + 
                      np.log(1 + gamma / self.arrival_rate_buy) / gamma)
        
        # Asymmetric quotes based on inventory
        bid_adjustment = gamma * sigma**2 * T * q / 2
        ask_adjustment = gamma * sigma**2 * T * q / 2
        
        optimal_bid = reservation_price - half_spread - bid_adjustment
        optimal_ask = reservation_price + half_spread - ask_adjustment
        
        # Round to tick size
        optimal_bid = np.floor(optimal_bid / self.tick_size) * self.tick_size
        optimal_ask = np.ceil(optimal_ask / self.tick_size) * self.tick_size
        
        return optimal_bid, optimal_ask
    
    def process_trade(self, side: str, quantity: int, price: float):
        """
        Process an incoming trade.
        
        Args:
            side: 'buy' or 'sell' from market maker perspective
            quantity: Trade quantity
            price: Execution price
        """
        if side == 'buy':
            # Market maker sells
            self.current_inventory -= quantity
            pnl = quantity * price
        else:
            # Market maker buys
            self.current_inventory += quantity
            pnl = -quantity * price
        
        self.pnl_history.append(pnl)
        self.inventory_history.append(self.current_inventory)
        
        # Check inventory limits
        if abs(self.current_inventory) > self.inventory_capacity:
            # Need to adjust quotes more aggressively
            self.risk_aversion *= 1.1  # Increase risk aversion
    
    def calculate_sharpe_ratio(self, lookback: int = 1000) -> float:
        """Calculate Sharpe ratio of market making strategy."""
        if len(self.pnl_history) < lookback:
            return 0.0
        
        recent_pnl = list(self.pnl_history)[-lookback:]
        
        if len(recent_pnl) < 2:
            return 0.0
        
        mean_pnl = np.mean(recent_pnl)
        std_pnl = np.std(recent_pnl)
        
        return mean_pnl / std_pnl if std_pnl > 0 else 0.0

class LiquidityAnalyzer:
    """Comprehensive liquidity analysis and measurement."""
    
    def __init__(self):
        """Initialize liquidity analyzer."""
        self.models = {
            'amihud': AmihudModel(),
            'kyle': KyleModel(),
            'glosten_milgrom': GlosstenMilgromModel()
        }
        
        self.regime_thresholds = {
            'spread_high': 0.01,
            'spread_crisis': 0.05,
            'depth_low': 1000,
            'depth_crisis': 100,
            'turnover_low': 0.1,
            'turnover_high': 2.0
        }
    
    def calculate_comprehensive_metrics(self, 
                                      prices: np.ndarray,
                                      volumes: np.ndarray, 
                                      bid_prices: np.ndarray,
                                      ask_prices: np.ndarray,
                                      bid_sizes: np.ndarray,
                                      ask_sizes: np.ndarray) -> LiquidityMetrics:
        """
        Calculate comprehensive liquidity metrics.
        
        Args:
            prices: Transaction prices
            volumes: Transaction volumes
            bid_prices: Best bid prices
            ask_prices: Best ask prices  
            bid_sizes: Bid sizes
            ask_sizes: Ask sizes
        
        Returns:
            LiquidityMetrics object
        """
        timestamp = len(prices)  # Simplified timestamp
        
        # Basic spread measures
        spreads = ask_prices - bid_prices
        bid_ask_spread = np.mean(spreads) if len(spreads) > 0 else 0.0
        
        # Effective spread (for trades)
        effective_spreads = []
        for i, (price, volume) in enumerate(zip(prices, volumes)):
            if i < len(bid_prices) and i < len(ask_prices):
                mid_price = (bid_prices[i] + ask_prices[i]) / 2
                # Assume buy if price > mid, sell if price < mid
                if price > mid_price:
                    effective_spread = 2 * (price - mid_price)
                else:
                    effective_spread = 2 * (mid_price - price)
                effective_spreads.append(effective_spread)
        
        effective_spread = np.mean(effective_spreads) if effective_spreads else 0.0
        
        # Market depth
        total_depth = np.mean(bid_sizes + ask_sizes) if len(bid_sizes) > 0 else 0.0
        quoted_depth = np.mean(np.minimum(bid_sizes, ask_sizes)) if len(bid_sizes) > 0 else 0.0
        
        # Price impact estimation
        if len(prices) > 1:
            returns = np.diff(prices) / prices[:-1]
            volume_changes = np.diff(volumes) if len(volumes) > 1 else np.array([0])
            
            if len(returns) == len(volume_changes) and len(returns) > 10:
                try:
                    # Simple price impact regression
                    slope, _, _, _, _ = stats.linregress(volume_changes, returns)
                    price_impact = abs(slope)
                except:
                    price_impact = 0.0
            else:
                price_impact = 0.0
        else:
            price_impact = 0.0
        
        # Amihud illiquidity
        amihud_illiquidity = 0.0
        if len(prices) > 1 and len(volumes) > 1:
            for price, volume in zip(prices[-50:], volumes[-50:]):  # Recent data
                self.models['amihud'].add_observation(price, volume, timestamp)
            amihud_illiquidity = self.models['amihud'].calculate_illiquidity()
        
        # Turnover rate (simplified)
        total_volume = np.sum(volumes) if len(volumes) > 0 else 0
        avg_price = np.mean(prices) if len(prices) > 0 else 1
        market_cap_proxy = avg_price * 1000000  # Simplified market cap
        turnover_rate = total_volume * avg_price / market_cap_proxy if market_cap_proxy > 0 else 0
        
        # Volume weighted spread
        total_traded_volume = np.sum(volumes) if len(volumes) > 0 else 1
        volume_weighted_spread = np.sum(effective_spreads * volumes[:len(effective_spreads)]) / total_traded_volume if effective_spreads and total_traded_volume > 0 else 0
        
        # Resilience (simplified - would need time series analysis)
        resilience = max(0, 1 - bid_ask_spread / 0.1)  # Higher spread = lower resilience
        
        # Realized spread (simplified)
        realized_spread = effective_spread * 0.7  # Assume 70% is temporary impact
        
        return LiquidityMetrics(
            timestamp=timestamp,
            bid_ask_spread=bid_ask_spread,
            effective_spread=effective_spread,
            realized_spread=realized_spread,
            price_impact=price_impact,
            market_depth=total_depth,
            resilience=resilience,
            turnover_rate=turnover_rate,
            amihud_illiquidity=amihud_illiquidity,
            volume_weighted_spread=volume_weighted_spread,
            quoted_depth=quoted_depth,
            hidden_liquidity=0.0  # Would need order book data
        )
    
    def classify_liquidity_regime(self, metrics: LiquidityMetrics) -> LiquidityRegime:
        """
        Classify current liquidity regime.
        
        Args:
            metrics: Current liquidity metrics
        
        Returns:
            Classified liquidity regime
        """
        # Crisis conditions
        if (metrics.bid_ask_spread > self.regime_thresholds['spread_crisis'] or
            metrics.market_depth < self.regime_thresholds['depth_crisis']):
            return LiquidityRegime.CRISIS_LIQUIDITY
        
        # Stress conditions
        if (metrics.bid_ask_spread > self.regime_thresholds['spread_high'] or
            metrics.market_depth < self.regime_thresholds['depth_low'] or
            metrics.price_impact > 0.01):
            return LiquidityRegime.STRESS_LIQUIDITY
        
        # Low liquidity
        if (metrics.turnover_rate < self.regime_thresholds['turnover_low'] or
            metrics.amihud_illiquidity > 0.001):
            return LiquidityRegime.LOW_LIQUIDITY
        
        # High liquidity
        if (metrics.turnover_rate > self.regime_thresholds['turnover_high'] and
            metrics.bid_ask_spread < 0.001 and
            metrics.market_depth > 10000):
            return LiquidityRegime.HIGH_LIQUIDITY
        
        # Normal liquidity
        return LiquidityRegime.NORMAL_LIQUIDITY
    
    def detect_liquidity_events(self, metrics_history: List[LiquidityMetrics], 
                               threshold_multiplier: float = 3.0) -> List[LiquidityEvent]:
        """
        Detect significant liquidity events.
        
        Args:
            metrics_history: Historical liquidity metrics
            threshold_multiplier: Standard deviation multiplier for event detection
        
        Returns:
            List of detected liquidity events
        """
        if len(metrics_history) < 50:
            return []
        
        events = []
        
        # Extract time series for each metric
        spreads = [m.bid_ask_spread for m in metrics_history]
        depths = [m.market_depth for m in metrics_history]
        impacts = [m.price_impact for m in metrics_history]
        
        # Calculate rolling statistics
        window = 20
        for i in range(window, len(metrics_history)):
            current_metrics = metrics_history[i]
            
            # Recent window statistics
            recent_spreads = spreads[i-window:i]
            recent_depths = depths[i-window:i]
            recent_impacts = impacts[i-window:i]
            
            spread_mean = np.mean(recent_spreads)
            spread_std = np.std(recent_spreads)
            depth_mean = np.mean(recent_depths)
            depth_std = np.std(recent_depths)
            impact_mean = np.mean(recent_impacts)
            impact_std = np.std(recent_impacts)
            
            # Detect anomalies
            contributing_factors = {}
            
            # Spread spike
            if (spread_std > 0 and 
                current_metrics.bid_ask_spread > spread_mean + threshold_multiplier * spread_std):
                contributing_factors['spread_spike'] = (current_metrics.bid_ask_spread - spread_mean) / spread_std
            
            # Depth drop
            if (depth_std > 0 and 
                current_metrics.market_depth < depth_mean - threshold_multiplier * depth_std):
                contributing_factors['depth_drop'] = (depth_mean - current_metrics.market_depth) / depth_std
            
            # Impact increase
            if (impact_std > 0 and 
                current_metrics.price_impact > impact_mean + threshold_multiplier * impact_std):
                contributing_factors['impact_increase'] = (current_metrics.price_impact - impact_mean) / impact_std
            
            # Create event if significant factors detected
            if contributing_factors:
                severity = max(contributing_factors.values())
                
                event = LiquidityEvent(
                    timestamp=current_metrics.timestamp,
                    event_type='liquidity_stress',
                    severity=severity,
                    duration=1.0,  # Would need to track duration properly
                    affected_instruments=['primary'],  # Would expand for multi-asset
                    contributing_factors=contributing_factors
                )
                
                events.append(event)
        
        return events

