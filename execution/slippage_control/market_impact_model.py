"""
Market Impact Model
Advanced models for predicting and measuring market impact of trades
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


class MarketRegime(Enum):
    """Market regime classification"""
    NORMAL = "NORMAL"             # Normal market conditions
    STRESSED = "STRESSED"         # Stressed/volatile conditions
    ILLIQUID = "ILLIQUID"         # Low liquidity conditions
    MOMENTUM = "MOMENTUM"         # Strong directional momentum
    MEAN_REVERTING = "MEAN_REVERTING" # Mean reverting conditions


class ImpactComponent(Enum):
    """Components of market impact"""
    PERMANENT = "PERMANENT"       # Permanent price impact
    TEMPORARY = "TEMPORARY"       # Temporary price impact
    TIMING = "TIMING"            # Timing impact (opportunity cost)
    SPREAD = "SPREAD"            # Bid-ask spread cost
    DELAY = "DELAY"              # Delay/queue impact


@dataclass
class MarketState:
    """Current market state snapshot"""
    symbol: str
    timestamp: datetime
    
    # Price and volume
    mid_price: float
    bid_price: float
    ask_price: float
    last_price: float
    
    # Liquidity metrics
    bid_size: float
    ask_size: float
    spread_bps: float
    depth_ratio: float  # Available liquidity vs normal
    
    # Volatility and momentum
    realized_volatility: float  # Recent realized volatility
    momentum_score: float       # -1 to 1, directional momentum
    
    # Market structure
    venue_count: int           # Number of active venues
    fragmentation_ratio: float # Liquidity fragmentation
    
    # Derived metrics
    market_cap_usd: Optional[float] = None
    avg_daily_volume_usd: Optional[float] = None
    
    @property
    def spread_dollars(self) -> float:
        """Spread in dollars"""
        return (self.spread_bps / 10000) * self.mid_price
    
    @property
    def total_top_size(self) -> float:
        """Total size at top of book"""
        return self.bid_size + self.ask_size


@dataclass
class TradeVector:
    """Trade execution parameters"""
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: float
    urgency: float  # 0 to 1, execution urgency
    
    # Execution constraints
    max_participation_rate: float = 0.20  # Max % of volume
    max_market_impact_bps: float = 50.0   # Max acceptable impact
    time_horizon_minutes: float = 60.0    # Time to complete
    
    # Strategy preferences
    allow_dark_pools: bool = True
    allow_crossing_networks: bool = True
    preferred_venues: Optional[List[str]] = None
    
    @property
    def is_buy(self) -> bool:
        """Check if trade is a buy"""
        return self.side.upper() == 'BUY'
    
    @property
    def signed_quantity(self) -> float:
        """Signed quantity (positive for buy, negative for sell)"""
        return self.quantity if self.is_buy else -self.quantity


@dataclass
class ImpactEstimate:
    """Market impact estimation"""
    symbol: str
    trade_vector: TradeVector
    market_state: MarketState
    
    # Impact components (in basis points)
    permanent_impact_bps: float = 0.0
    temporary_impact_bps: float = 0.0
    timing_impact_bps: float = 0.0
    spread_impact_bps: float = 0.0
    delay_impact_bps: float = 0.0
    
    # Total impact
    total_impact_bps: float = 0.0
    
    # Confidence and risk metrics
    confidence_level: float = 0.0  # 0 to 1
    impact_volatility_bps: float = 0.0  # Expected volatility of impact
    
    # Model metadata
    model_name: str = "Unknown"
    estimation_time: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Calculate total impact"""
        self.total_impact_bps = (
            self.permanent_impact_bps +
            self.temporary_impact_bps +
            self.timing_impact_bps +
            self.spread_impact_bps +
            self.delay_impact_bps
        )
    
    @property
    def total_cost_dollars(self) -> float:
        """Total estimated cost in dollars"""
        notional = self.trade_vector.quantity * self.market_state.mid_price
        return (self.total_impact_bps / 10000) * notional
    
    @property
    def impact_summary(self) -> Dict[str, float]:
        """Summary of impact components"""
        return {
            'permanent_bps': self.permanent_impact_bps,
            'temporary_bps': self.temporary_impact_bps,
            'timing_bps': self.timing_impact_bps,
            'spread_bps': self.spread_impact_bps,
            'delay_bps': self.delay_impact_bps,
            'total_bps': self.total_impact_bps,
            'total_dollars': self.total_cost_dollars
        }


class MarketImpactModel(ABC):
    """
    Abstract base class for market impact models
    """
    
    def __init__(self, name: str):
        self.name = name
        self.calibration_data: Optional[pd.DataFrame] = None
        self.model_parameters: Dict[str, Any] = {}
        self.last_calibration: Optional[datetime] = None
    
    @abstractmethod
    def estimate_impact(
        self,
        trade_vector: TradeVector,
        market_state: MarketState
    ) -> ImpactEstimate:
        """
        Estimate market impact for trade
        
        Args:
            trade_vector: Trade parameters
            market_state: Current market state
        
        Returns:
            Impact estimate
        """
        pass
    
    @abstractmethod
    def calibrate(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calibrate model parameters
        
        Args:
            historical_data: Historical trade and market data
        
        Returns:
            Calibration results
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'name': self.name,
            'parameters': self.model_parameters.copy(),
            'last_calibration': self.last_calibration,
            'calibration_data_points': len(self.calibration_data) if self.calibration_data is not None else 0
        }


class AlmgrenChrisskModel(MarketImpactModel):
    """
    Almgren-Chriss model for market impact
    Based on "Optimal Execution of Portfolio Transactions"
    """
    
    def __init__(self):
        super().__init__("Almgren-Chriss")
        
        # Model parameters
        self.model_parameters = {
            'lambda': 1e-6,     # Permanent impact parameter  
            'eta': 2.5e-7,      # Temporary impact parameter
            'gamma': 2.5e-7,    # Risk aversion parameter
            'sigma': 0.95       # Volatility parameter (annual)
        }
    
    def estimate_impact(
        self,
        trade_vector: TradeVector,
        market_state: MarketState
    ) -> ImpactEstimate:
        """Estimate impact using Almgren-Chriss model"""
        
        # Extract parameters
        lambda_param = self.model_parameters['lambda']
        eta_param = self.model_parameters['eta']
        sigma = self.model_parameters['sigma']
        
        # Trade parameters
        X = abs(trade_vector.signed_quantity)  # Total shares to trade
        T = trade_vector.time_horizon_minutes / (252 * 24 * 60)  # Time in years
        S = market_state.mid_price
        
        # Daily volume estimate (use market_state if available)
        if market_state.avg_daily_volume_usd:
            V = market_state.avg_daily_volume_usd / S  # Daily volume in shares
        else:
            # Fallback estimate
            V = X * 20  # Assume trade is 5% of daily volume
        
        # Participation rate
        participation_rate = min(
            trade_vector.max_participation_rate,
            X / (V * T) if T > 0 else 1.0
        )
        
        # Permanent impact (linear in trade size)
        permanent_impact = lambda_param * (X / V) * S
        permanent_impact_bps = (permanent_impact / S) * 10000
        
        # Temporary impact (depends on trading rate)
        trading_rate = X / T if T > 0 else X
        temporary_impact = eta_param * (trading_rate / V) * S
        temporary_impact_bps = (temporary_impact / S) * 10000
        
        # Timing cost (opportunity cost of not trading immediately)
        volatility_cost = 0.5 * sigma * np.sqrt(T) * S
        timing_impact_bps = (volatility_cost / S) * 10000
        
        # Spread cost
        spread_impact_bps = market_state.spread_bps / 2  # Half spread
        
        # Create impact estimate
        estimate = ImpactEstimate(
            symbol=trade_vector.symbol,
            trade_vector=trade_vector,
            market_state=market_state,
            permanent_impact_bps=permanent_impact_bps,
            temporary_impact_bps=temporary_impact_bps,
            timing_impact_bps=timing_impact_bps,
            spread_impact_bps=spread_impact_bps,
            delay_impact_bps=0.0,  # Not modeled in basic AC
            confidence_level=0.75,  # Moderate confidence
            impact_volatility_bps=permanent_impact_bps * 0.3,  # 30% uncertainty
            model_name=self.name
        )
        
        return estimate
    
    def calibrate(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Calibrate Almgren-Chriss parameters"""
        
        required_columns = ['symbol', 'quantity', 'price', 'market_impact_bps', 'daily_volume']
        
        if not all(col in historical_data.columns for col in required_columns):
            raise ValueError(f"Historical data must contain columns: {required_columns}")
        
        self.calibration_data = historical_data.copy()
        
        # Calibrate lambda (permanent impact parameter)
        # Regress market_impact_bps against (quantity / daily_volume)
        
        data = historical_data.dropna()
        
        if len(data) < 10:
            warnings.warn("Insufficient data for reliable calibration")
            return {'status': 'insufficient_data'}
        
        # Calculate trade size relative to daily volume
        data['trade_ratio'] = data['quantity'] / data['daily_volume']
        
        # Simple linear regression for permanent impact
        X = data['trade_ratio'].values.reshape(-1, 1)
        y = data['market_impact_bps'].values
        
        # Use numpy for simple linear regression
        X_with_intercept = np.column_stack([np.ones(len(X)), X.flatten()])
        
        try:
            coefficients = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
            intercept, slope = coefficients
            
            # Update lambda parameter
            self.model_parameters['lambda'] = slope / 10000  # Convert from bps
            
            # Calculate R-squared
            y_pred = X_with_intercept @ coefficients
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            self.last_calibration = datetime.now()
            
            return {
                'status': 'success',
                'lambda': self.model_parameters['lambda'],
                'r_squared': r_squared,
                'data_points': len(data),
                'intercept': intercept,
                'slope': slope
            }
            
        except np.linalg.LinAlgError:
            return {'status': 'calibration_failed', 'error': 'Linear algebra error'}


class BertsimaskModel(MarketImpactModel):
    """
    Bertsimas-Lo model with market microstructure effects
    """
    
    def __init__(self):
        super().__init__("Bertsimas-Lo")
        
        self.model_parameters = {
            'alpha': 0.6,       # Participation rate exponent
            'beta': 0.4,        # Size impact exponent  
            'delta': 2.0,       # Urgency impact multiplier
            'kappa': 0.1,       # Liquidity adjustment
            'theta': 0.05       # Volatility impact scaling
        }
    
    def estimate_impact(
        self,
        trade_vector: TradeVector,
        market_state: MarketState
    ) -> ImpactEstimate:
        """Estimate impact using Bertsimas-Lo model"""
        
        # Model parameters
        alpha = self.model_parameters['alpha']
        beta = self.model_parameters['beta']
        delta = self.model_parameters['delta']
        kappa = self.model_parameters['kappa']
        theta = self.model_parameters['theta']
        
        # Trade metrics
        X = abs(trade_vector.signed_quantity)
        S = market_state.mid_price
        
        # Market metrics
        total_book_size = market_state.total_top_size
        volatility = market_state.realized_volatility
        
        # Estimate daily volume
        if market_state.avg_daily_volume_usd:
            daily_volume = market_state.avg_daily_volume_usd / S
        else:
            daily_volume = X * 50  # Conservative estimate
        
        # Participation rate impact
        participation_rate = min(
            trade_vector.max_participation_rate,
            X / daily_volume
        )
        
        # Size-based impact
        size_impact = (X / daily_volume) ** beta
        
        # Urgency impact
        urgency_multiplier = 1 + (delta - 1) * trade_vector.urgency
        
        # Liquidity adjustment
        liquidity_ratio = total_book_size / max(daily_volume * 0.001, 1)  # Book size vs expected
        liquidity_adjustment = max(kappa / liquidity_ratio, 1.0)
        
        # Permanent impact
        permanent_impact_bps = (
            size_impact * urgency_multiplier * liquidity_adjustment * 100  # Base 100 bps
        )
        
        # Temporary impact (function of participation rate)
        temporary_impact_bps = (
            (participation_rate ** alpha) * urgency_multiplier * 50  # Base 50 bps
        )
        
        # Timing impact (volatility-based)
        timing_minutes = trade_vector.time_horizon_minutes
        timing_impact_bps = (
            theta * volatility * np.sqrt(timing_minutes / 60) * 10000  # Convert to bps
        )
        
        # Spread impact
        spread_impact_bps = market_state.spread_bps / 2
        
        # Delay impact (queue/latency effects)
        delay_impact_bps = (1 - market_state.depth_ratio) * 10  # Up to 10 bps for poor liquidity
        
        estimate = ImpactEstimate(
            symbol=trade_vector.symbol,
            trade_vector=trade_vector,
            market_state=market_state,
            permanent_impact_bps=permanent_impact_bps,
            temporary_impact_bps=temporary_impact_bps,
            timing_impact_bps=timing_impact_bps,
            spread_impact_bps=spread_impact_bps,
            delay_impact_bps=delay_impact_bps,
            confidence_level=0.8,
            impact_volatility_bps=permanent_impact_bps * 0.4,
            model_name=self.name
        )
        
        return estimate
    
    def calibrate(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Calibrate Bertsimas-Lo parameters using historical data"""
        
        # This is a simplified calibration - real implementation would use
        # maximum likelihood estimation or other advanced techniques
        
        if len(historical_data) < 20:
            return {'status': 'insufficient_data'}
        
        self.calibration_data = historical_data.copy()
        
        # Basic parameter adjustment based on observed impacts
        data = historical_data.dropna()
        
        if 'market_impact_bps' in data.columns and 'quantity' in data.columns:
            # Adjust beta based on size-impact relationship
            avg_impact = data['market_impact_bps'].mean()
            
            if avg_impact > 100:  # High impact environment
                self.model_parameters['beta'] = 0.6
                self.model_parameters['delta'] = 2.5
            elif avg_impact < 30:  # Low impact environment
                self.model_parameters['beta'] = 0.3
                self.model_parameters['delta'] = 1.5
            
            self.last_calibration = datetime.now()
            
            return {
                'status': 'success',
                'average_impact_bps': avg_impact,
                'adjusted_beta': self.model_parameters['beta'],
                'adjusted_delta': self.model_parameters['delta'],
                'data_points': len(data)
            }
        
        return {'status': 'insufficient_columns'}


class JPMorganModel(MarketImpactModel):
    """
    JPMorgan market impact model with regime awareness
    """
    
    def __init__(self):
        super().__init__("JPMorgan")
        
        self.model_parameters = {
            'base_impact': 25.0,        # Base impact in bps per 1% of daily volume
            'volatility_scaling': 1.5,   # Volatility impact multiplier
            'urgency_power': 0.8,       # Urgency impact exponent
            'liquidity_beta': -0.3,     # Liquidity impact (negative = more liquidity = less impact)
            'momentum_adjustment': 0.2,  # Momentum impact adjustment
            'regime_multipliers': {     # Regime-specific multipliers
                MarketRegime.NORMAL: 1.0,
                MarketRegime.STRESSED: 1.8,
                MarketRegime.ILLIQUID: 2.5,
                MarketRegime.MOMENTUM: 1.3,
                MarketRegime.MEAN_REVERTING: 0.8
            }
        }
    
    def estimate_impact(
        self,
        trade_vector: TradeVector,
        market_state: MarketState
    ) -> ImpactEstimate:
        """Estimate impact using JPMorgan model"""
        
        # Determine market regime
        regime = self._classify_market_regime(market_state)
        regime_multiplier = self.model_parameters['regime_multipliers'][regime]
        
        # Trade metrics
        X = abs(trade_vector.signed_quantity)
        S = market_state.mid_price
        
        # Estimate participation rate
        if market_state.avg_daily_volume_usd:
            daily_volume = market_state.avg_daily_volume_usd / S
            participation_rate = X / daily_volume
        else:
            participation_rate = trade_vector.max_participation_rate
        
        # Base impact (linear in participation rate)
        base_impact_bps = (
            self.model_parameters['base_impact'] * participation_rate * 100
        )
        
        # Volatility adjustment
        vol_adjustment = (
            1 + self.model_parameters['volatility_scaling'] * 
            (market_state.realized_volatility - 0.2)  # Assume 20% base volatility
        )
        
        # Urgency adjustment
        urgency_adjustment = (
            trade_vector.urgency ** self.model_parameters['urgency_power']
        )
        
        # Liquidity adjustment
        liquidity_adjustment = (
            market_state.depth_ratio ** self.model_parameters['liquidity_beta']
        )
        
        # Momentum adjustment
        momentum_adjustment = (
            1 + self.model_parameters['momentum_adjustment'] * 
            abs(market_state.momentum_score)
        )
        
        # Calculate permanent impact
        permanent_impact_bps = (
            base_impact_bps * vol_adjustment * urgency_adjustment * 
            liquidity_adjustment * momentum_adjustment * regime_multiplier * 0.6  # 60% permanent
        )
        
        # Temporary impact (40% of total impact)
        temporary_impact_bps = permanent_impact_bps * (0.4 / 0.6)
        
        # Timing impact based on volatility and time horizon
        timing_impact_bps = (
            market_state.realized_volatility * 
            np.sqrt(trade_vector.time_horizon_minutes / 60) * 10000 * 0.1
        )
        
        # Spread impact
        spread_impact_bps = market_state.spread_bps / 2
        
        # Delay impact (higher in stressed regimes)
        delay_multiplier = 2.0 if regime == MarketRegime.STRESSED else 1.0
        delay_impact_bps = (
            (1 - market_state.depth_ratio) * 15 * delay_multiplier
        )
        
        # Confidence level based on regime
        confidence_levels = {
            MarketRegime.NORMAL: 0.85,
            MarketRegime.STRESSED: 0.65,
            MarketRegime.ILLIQUID: 0.60,
            MarketRegime.MOMENTUM: 0.70,
            MarketRegime.MEAN_REVERTING: 0.80
        }
        
        estimate = ImpactEstimate(
            symbol=trade_vector.symbol,
            trade_vector=trade_vector,
            market_state=market_state,
            permanent_impact_bps=permanent_impact_bps,
            temporary_impact_bps=temporary_impact_bps,
            timing_impact_bps=timing_impact_bps,
            spread_impact_bps=spread_impact_bps,
            delay_impact_bps=delay_impact_bps,
            confidence_level=confidence_levels.get(regime, 0.70),
            impact_volatility_bps=permanent_impact_bps * 0.35,
            model_name=f"{self.name}-{regime.value}"
        )
        
        return estimate
    
    def _classify_market_regime(self, market_state: MarketState) -> MarketRegime:
        """Classify current market regime"""
        
        # Simple regime classification based on market metrics
        
        # Check for illiquid conditions
        if market_state.depth_ratio < 0.5 or market_state.spread_bps > 50:
            return MarketRegime.ILLIQUID
        
        # Check for stressed conditions (high volatility)
        if market_state.realized_volatility > 0.4:  # 40% annual volatility
            return MarketRegime.STRESSED
        
        # Check for momentum conditions
        if abs(market_state.momentum_score) > 0.7:
            return MarketRegime.MOMENTUM
        
        # Check for mean reverting conditions
        if (abs(market_state.momentum_score) < 0.2 and 
            market_state.realized_volatility < 0.15):
            return MarketRegime.MEAN_REVERTING
        
        # Default to normal
        return MarketRegime.NORMAL
    
    def calibrate(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Calibrate JPMorgan model parameters"""
        
        if len(historical_data) < 50:
            return {'status': 'insufficient_data'}
        
        self.calibration_data = historical_data.copy()
        
        # Regime-based calibration
        calibration_results = {}
        
        for regime in MarketRegime:
            regime_data = self._filter_by_regime(historical_data, regime)
            
            if len(regime_data) > 10:
                avg_impact = regime_data['market_impact_bps'].mean()
                regime_multiplier = avg_impact / self.model_parameters['base_impact']
                
                self.model_parameters['regime_multipliers'][regime] = max(0.5, regime_multiplier)
                calibration_results[regime.value] = {
                    'data_points': len(regime_data),
                    'average_impact': avg_impact,
                    'multiplier': regime_multiplier
                }
        
        self.last_calibration = datetime.now()
        
        return {
            'status': 'success',
            'total_data_points': len(historical_data),
            'regime_results': calibration_results
        }
    
    def _filter_by_regime(self, data: pd.DataFrame, regime: MarketRegime) -> pd.DataFrame:
        """Filter data by market regime (simplified)"""
        
        # This is a simplified regime filter - real implementation would be more sophisticated
        
        if regime == MarketRegime.NORMAL:
            return data[
                (data['realized_volatility'] <= 0.3) & 
                (data['spread_bps'] <= 20) &
                (abs(data.get('momentum_score', 0)) <= 0.5)
            ]
        elif regime == MarketRegime.STRESSED:
            return data[data['realized_volatility'] > 0.4]
        elif regime == MarketRegime.ILLIQUID:
            return data[
                (data['spread_bps'] > 30) | 
                (data.get('depth_ratio', 1.0) < 0.6)
            ]
        elif regime == MarketRegime.MOMENTUM:
            return data[abs(data.get('momentum_score', 0)) > 0.7]
        else:  # MEAN_REVERTING
            return data[
                (data['realized_volatility'] < 0.15) &
                (abs(data.get('momentum_score', 0)) < 0.3)
            ]


class MarketImpactEngine:
    """
    Main market impact estimation engine
    Coordinates multiple models and provides ensemble estimates
    """
    
    def __init__(self):
        self.models: Dict[str, MarketImpactModel] = {}
        
        # Add default models
        self.add_model(AlmgrenChrisskModel())
        self.add_model(BertsimaskModel())
        self.add_model(JPMorganModel())
        
        # Model weights for ensemble
        self.model_weights = {
            'Almgren-Chriss': 0.3,
            'Bertsimas-Lo': 0.3,
            'JPMorgan': 0.4
        }
        
        # Performance tracking
        self.estimation_count = 0
        self.model_performance: Dict[str, Dict[str, float]] = {}
    
    def add_model(self, model: MarketImpactModel) -> None:
        """Add impact model to engine"""
        self.models[model.name] = model
        
        if model.name not in self.model_performance:
            self.model_performance[model.name] = {
                'estimations': 0,
                'average_impact': 0.0,
                'calibration_score': 0.0
            }
    
    def estimate_impact(
        self,
        trade_vector: TradeVector,
        market_state: MarketState,
        model_names: Optional[List[str]] = None
    ) -> Dict[str, ImpactEstimate]:
        """
        Estimate market impact using specified models
        
        Args:
            trade_vector: Trade parameters
            market_state: Market state
            model_names: Models to use (None for all)
        
        Returns:
            Dictionary of impact estimates by model name
        """
        
        if model_names is None:
            model_names = list(self.models.keys())
        
        estimates = {}
        
        for model_name in model_names:
            if model_name in self.models:
                try:
                    estimate = self.models[model_name].estimate_impact(
                        trade_vector, market_state
                    )
                    estimates[model_name] = estimate
                    
                    # Update performance tracking
                    perf = self.model_performance[model_name]
                    perf['estimations'] += 1
                    
                    # Update running average of impact
                    current_avg = perf['average_impact']
                    new_avg = (
                        (current_avg * (perf['estimations'] - 1) + estimate.total_impact_bps) /
                        perf['estimations']
                    )
                    perf['average_impact'] = new_avg
                    
                except Exception as e:
                    print(f"Error estimating impact with {model_name}: {e}")
        
        self.estimation_count += 1
        return estimates
    
    def get_ensemble_estimate(
        self,
        trade_vector: TradeVector,
        market_state: MarketState
    ) -> ImpactEstimate:
        """
        Get ensemble (weighted average) impact estimate
        
        Args:
            trade_vector: Trade parameters
            market_state: Market state
        
        Returns:
            Ensemble impact estimate
        """
        
        # Get individual model estimates
        individual_estimates = self.estimate_impact(trade_vector, market_state)
        
        if not individual_estimates:
            raise ValueError("No valid impact estimates available")
        
        # Calculate weighted averages
        total_weight = 0.0
        weighted_components = {
            'permanent_impact_bps': 0.0,
            'temporary_impact_bps': 0.0,
            'timing_impact_bps': 0.0,
            'spread_impact_bps': 0.0,
            'delay_impact_bps': 0.0
        }
        
        weighted_confidence = 0.0
        weighted_volatility = 0.0
        
        for model_name, estimate in individual_estimates.items():
            weight = self.model_weights.get(model_name, 1.0 / len(individual_estimates))
            total_weight += weight
            
            # Weight each component
            for component in weighted_components:
                weighted_components[component] += weight * getattr(estimate, component)
            
            weighted_confidence += weight * estimate.confidence_level
            weighted_volatility += weight * estimate.impact_volatility_bps
        
        # Normalize by total weight
        if total_weight > 0:
            for component in weighted_components:
                weighted_components[component] /= total_weight
            
            weighted_confidence /= total_weight
            weighted_volatility /= total_weight
        
        # Create ensemble estimate
        ensemble_estimate = ImpactEstimate(
            symbol=trade_vector.symbol,
            trade_vector=trade_vector,
            market_state=market_state,
            permanent_impact_bps=weighted_components['permanent_impact_bps'],
            temporary_impact_bps=weighted_components['temporary_impact_bps'],
            timing_impact_bps=weighted_components['timing_impact_bps'],
            spread_impact_bps=weighted_components['spread_impact_bps'],
            delay_impact_bps=weighted_components['delay_impact_bps'],
            confidence_level=weighted_confidence,
            impact_volatility_bps=weighted_volatility,
            model_name="Ensemble"
        )
        
        return ensemble_estimate
    
    def calibrate_models(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Calibrate all models with historical data"""
        
        calibration_results = {}
        
        for model_name, model in self.models.items():
            try:
                result = model.calibrate(historical_data)
                calibration_results[model_name] = result
                
                # Update performance score
                if result.get('status') == 'success':
                    self.model_performance[model_name]['calibration_score'] = (
                        result.get('r_squared', 0.5)
                    )
                
            except Exception as e:
                calibration_results[model_name] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        return calibration_results
    
    def get_model_performance(self) -> Dict[str, Dict[str, float]]:
        """Get model performance statistics"""
        return self.model_performance.copy()
    
    def set_model_weights(self, weights: Dict[str, float]) -> None:
        """Set model weights for ensemble"""
        
        # Normalize weights
        total_weight = sum(weights.values())
        
        if total_weight > 0:
            self.model_weights = {
                model: weight / total_weight 
                for model, weight in weights.items()
            }
    
    def get_impact_distribution(
        self,
        trade_vector: TradeVector,
        market_state: MarketState,
        confidence_intervals: List[float] = [0.5, 0.75, 0.9, 0.95]
    ) -> Dict[str, float]:
        """
        Get impact distribution with confidence intervals
        
        Args:
            trade_vector: Trade parameters
            market_state: Market state
            confidence_intervals: Confidence levels to calculate
        
        Returns:
            Impact distribution statistics
        """
        
        # Get ensemble estimate
        ensemble = self.get_ensemble_estimate(trade_vector, market_state)
        
        # Assume normal distribution of impact
        mean_impact = ensemble.total_impact_bps
        impact_std = ensemble.impact_volatility_bps
        
        distribution = {
            'mean_impact_bps': mean_impact,
            'std_impact_bps': impact_std,
            'expected_cost_dollars': ensemble.total_cost_dollars
        }
        
        # Calculate confidence intervals
        from scipy import stats
        
        for ci in confidence_intervals:
            # Two-tailed confidence interval
            alpha = 1 - ci
            z_score = stats.norm.ppf(1 - alpha/2)
            
            margin = z_score * impact_std
            
            distribution[f'ci_{int(ci*100)}_lower_bps'] = mean_impact - margin
            distribution[f'ci_{int(ci*100)}_upper_bps'] = mean_impact + margin
            
            # Cost versions
            lower_cost = ((mean_impact - margin) / 10000) * trade_vector.quantity * market_state.mid_price
            upper_cost = ((mean_impact + margin) / 10000) * trade_vector.quantity * market_state.mid_price
            
            distribution[f'ci_{int(ci*100)}_lower_cost'] = lower_cost
            distribution[f'ci_{int(ci*100)}_upper_cost'] = upper_cost
        
        return distribution


if __name__ == "__main__":
    import random
    from datetime import datetime, timedelta
    
    # Example usage and testing
    print("Testing Market Impact Models...")
    
    # Create market impact engine
    engine = MarketImpactEngine()
    
    # Create sample market state
    market_state = MarketState(
        symbol="AAPL",
        timestamp=datetime.now(),
        mid_price=150.00,
        bid_price=149.98,
        ask_price=150.02,
        last_price=150.01,
        bid_size=1000,
        ask_size=1200,
        spread_bps=2.67,  # (150.02 - 149.98) / 150.00 * 10000
        depth_ratio=0.8,
        realized_volatility=0.25,
        momentum_score=0.3,
        venue_count=12,
        fragmentation_ratio=0.6,
        avg_daily_volume_usd=2_000_000_000  # $2B daily volume
    )
    
    # Create sample trade vectors
    trade_vectors = [
        TradeVector(
            symbol="AAPL",
            side="BUY",
            quantity=10_000,  # $1.5M trade
            urgency=0.7,
            max_participation_rate=0.15,
            time_horizon_minutes=30
        ),
        TradeVector(
            symbol="AAPL", 
            side="SELL",
            quantity=50_000,  # $7.5M trade
            urgency=0.3,
            max_participation_rate=0.10,
            time_horizon_minutes=120
        ),
        TradeVector(
            symbol="AAPL",
            side="BUY", 
            quantity=100_000,  # $15M trade
            urgency=0.9,
            max_participation_rate=0.25,
            time_horizon_minutes=15
        )
    ]
    
    print(f"\nMarket State:")
    print(f"  Symbol: {market_state.symbol}")
    print(f"  Mid Price: ${market_state.mid_price:.2f}")
    print(f"  Spread: {market_state.spread_bps:.1f} bps")
    print(f"  Liquidity Ratio: {market_state.depth_ratio:.1%}")
    print(f"  Volatility: {market_state.realized_volatility:.1%}")
    print(f"  Daily Volume: ${market_state.avg_daily_volume_usd/1e9:.1f}B")
    
    # Test individual models
    print(f"\nTesting Individual Models:")
    
    for i, trade_vector in enumerate(trade_vectors):
        print(f"\nTrade {i+1}: {trade_vector.side} {trade_vector.quantity:,} shares "
              f"(${trade_vector.quantity * market_state.mid_price/1e6:.1f}M)")
        print(f"  Urgency: {trade_vector.urgency:.1%}, "
              f"Time Horizon: {trade_vector.time_horizon_minutes:.0f} min")
        
        # Get estimates from all models
        estimates = engine.estimate_impact(trade_vector, market_state)
        
        for model_name, estimate in estimates.items():
            print(f"  {model_name}:")
            print(f"    Total Impact: {estimate.total_impact_bps:.1f} bps "
                  f"(${estimate.total_cost_dollars:,.0f})")
            print(f"    Permanent: {estimate.permanent_impact_bps:.1f} bps, "
                  f"Temporary: {estimate.temporary_impact_bps:.1f} bps")
            print(f"    Confidence: {estimate.confidence_level:.1%}")
    
    # Test ensemble estimates
    print(f"\nTesting Ensemble Estimates:")
    
    for i, trade_vector in enumerate(trade_vectors):
        print(f"\nTrade {i+1} Ensemble Estimate:")
        
        ensemble_estimate = engine.get_ensemble_estimate(trade_vector, market_state)
        
        print(f"  Total Impact: {ensemble_estimate.total_impact_bps:.1f} bps")
        print(f"  Estimated Cost: ${ensemble_estimate.total_cost_dollars:,.0f}")
        
        # Get impact breakdown
        breakdown = ensemble_estimate.impact_summary
        print(f"  Breakdown:")
        
        for component, value in breakdown.items():
            if 'bps' in component and component != 'total_bps':
                print(f"    {component.replace('_', ' ').title()}: {value:.1f} bps")
        
        # Get confidence intervals
        distribution = engine.get_impact_distribution(trade_vector, market_state)
        print(f"  95% Confidence Interval: "
              f"{distribution['ci_95_lower_bps']:.1f} - {distribution['ci_95_upper_bps']:.1f} bps")
        print(f"  Cost Range (95% CI): "
              f"${distribution['ci_95_lower_cost']:,.0f} - ${distribution['ci_95_upper_cost']:,.0f}")
    
    # Test model calibration with synthetic data
    print(f"\nTesting Model Calibration:")
    
    # Generate synthetic historical data
    np.random.seed(42)
    n_samples = 1000
    
    synthetic_data = pd.DataFrame({
        'symbol': ['AAPL'] * n_samples,
        'quantity': np.random.lognormal(9, 1, n_samples),  # Log-normal distribution
        'price': np.random.normal(150, 5, n_samples),
        'daily_volume': np.random.normal(13_000_000, 2_000_000, n_samples),  # Shares
        'realized_volatility': np.random.gamma(2, 0.1, n_samples),
        'spread_bps': np.random.gamma(1.5, 2, n_samples),
        'depth_ratio': np.random.beta(8, 2, n_samples),
        'momentum_score': np.random.normal(0, 0.3, n_samples)
    })
    
    # Add synthetic market impact (based on simple model)
    synthetic_data['participation_rate'] = synthetic_data['quantity'] / synthetic_data['daily_volume']
    synthetic_data['market_impact_bps'] = (
        20 * synthetic_data['participation_rate'] * 100 +  # Base impact
        5 * synthetic_data['realized_volatility'] * 100 +   # Volatility impact
        synthetic_data['spread_bps'] * 0.5 +                # Spread impact
        np.random.normal(0, 10, n_samples)                  # Noise
    )
    
    # Calibrate models
    calibration_results = engine.calibrate_models(synthetic_data)
    
    print(f"Calibration Results:")
    for model_name, result in calibration_results.items():
        print(f"  {model_name}: {result.get('status', 'unknown')}")
        
        if result.get('status') == 'success':
            if 'r_squared' in result:
                print(f"    R-squared: {result['r_squared']:.3f}")
            if 'data_points' in result:
                print(f"    Data Points: {result['data_points']}")
    
    # Test model performance
    print(f"\nModel Performance:")
    performance = engine.get_model_performance()
    
    for model_name, perf in performance.items():
        print(f"  {model_name}:")
        print(f"    Estimations: {perf['estimations']}")
        print(f"    Average Impact: {perf['average_impact']:.1f} bps")
        print(f"    Calibration Score: {perf['calibration_score']:.3f}")
    
    # Test different market regimes
    print(f"\nTesting Market Regime Effects:")
    
    # Create different market states
    regime_states = {
        'Normal': MarketState(
            symbol="AAPL", timestamp=datetime.now(),
            mid_price=150.00, bid_price=149.98, ask_price=150.02,
            last_price=150.01, bid_size=1000, ask_size=1200,
            spread_bps=2.7, depth_ratio=0.9, realized_volatility=0.15,
            momentum_score=0.1, venue_count=12, fragmentation_ratio=0.6,
            avg_daily_volume_usd=2_000_000_000
        ),
        'Stressed': MarketState(
            symbol="AAPL", timestamp=datetime.now(),
            mid_price=150.00, bid_price=149.95, ask_price=150.05,
            last_price=150.02, bid_size=500, ask_size=600,
            spread_bps=6.7, depth_ratio=0.4, realized_volatility=0.45,
            momentum_score=-0.2, venue_count=8, fragmentation_ratio=0.8,
            avg_daily_volume_usd=2_000_000_000
        ),
        'Illiquid': MarketState(
            symbol="AAPL", timestamp=datetime.now(),
            mid_price=150.00, bid_price=149.92, ask_price=150.08,
            last_price=150.03, bid_size=200, ask_size=300,
            spread_bps=10.7, depth_ratio=0.3, realized_volatility=0.25,
            momentum_score=0.0, venue_count=5, fragmentation_ratio=0.9,
            avg_daily_volume_usd=500_000_000  # Lower volume
        )
    }
    
    test_trade = TradeVector(
        symbol="AAPL", side="BUY", quantity=25_000,
        urgency=0.5, max_participation_rate=0.15, time_horizon_minutes=60
    )
    
    for regime_name, regime_state in regime_states.items():
        print(f"\n{regime_name} Market Conditions:")
        
        ensemble = engine.get_ensemble_estimate(test_trade, regime_state)
        
        print(f"  Total Impact: {ensemble.total_impact_bps:.1f} bps")
        print(f"  Estimated Cost: ${ensemble.total_cost_dollars:,.0f}")
        print(f"  Confidence: {ensemble.confidence_level:.1%}")
    
    print("\nMarket Impact Model testing completed!")