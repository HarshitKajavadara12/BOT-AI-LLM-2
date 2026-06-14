"""
Risk Assessor
Pre-trade risk analysis and assessment for order execution
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


class RiskType(Enum):
    """Types of execution risks"""
    MARKET_RISK = "MARKET_RISK"                 # Price movement risk
    LIQUIDITY_RISK = "LIQUIDITY_RISK"           # Liquidity shortage risk
    TIMING_RISK = "TIMING_RISK"                 # Execution timing risk
    VENUE_RISK = "VENUE_RISK"                   # Venue-specific risks
    OPERATIONAL_RISK = "OPERATIONAL_RISK"       # Operational failures
    INFORMATION_LEAKAGE = "INFORMATION_LEAKAGE" # Information leakage risk
    CONCENTRATION_RISK = "CONCENTRATION_RISK"   # Position concentration risk
    REGULATORY_RISK = "REGULATORY_RISK"         # Regulatory compliance risk


class RiskSeverity(Enum):
    """Risk severity levels"""
    LOW = "LOW"                    # Minor impact expected
    MEDIUM = "MEDIUM"              # Moderate impact expected
    HIGH = "HIGH"                  # Significant impact expected
    CRITICAL = "CRITICAL"          # Severe impact expected


class RiskLikelihood(Enum):
    """Risk likelihood levels"""
    RARE = "RARE"                  # Very unlikely to occur (< 5%)
    UNLIKELY = "UNLIKELY"          # Unlikely to occur (5-25%)
    POSSIBLE = "POSSIBLE"          # May occur (25-50%)
    LIKELY = "LIKELY"              # Likely to occur (50-75%)
    ALMOST_CERTAIN = "ALMOST_CERTAIN"  # Very likely to occur (> 75%)


@dataclass
class RiskFactor:
    """Individual risk factor"""
    risk_id: str
    risk_type: RiskType
    severity: RiskSeverity
    likelihood: RiskLikelihood
    
    # Risk description
    description: str = ""
    impact_description: str = ""
    
    # Quantitative measures
    expected_cost_impact_bps: float = 0.0    # Expected cost impact
    max_cost_impact_bps: float = 0.0         # Maximum cost impact
    probability: float = 0.0                 # Probability (0-1)
    
    # Risk scoring
    risk_score: float = 0.0                  # Overall risk score (0-100)
    
    # Mitigation
    mitigation_strategies: List[str] = field(default_factory=list)
    is_mitigatable: bool = True
    
    # Context
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Calculate risk score"""
        if self.risk_score == 0.0:
            self.risk_score = self._calculate_risk_score()
    
    def _calculate_risk_score(self) -> float:
        """Calculate overall risk score"""
        
        # Severity weights
        severity_weights = {
            RiskSeverity.LOW: 1.0,
            RiskSeverity.MEDIUM: 2.5,
            RiskSeverity.HIGH: 5.0,
            RiskSeverity.CRITICAL: 10.0
        }
        
        # Likelihood weights
        likelihood_weights = {
            RiskLikelihood.RARE: 0.05,
            RiskLikelihood.UNLIKELY: 0.15,
            RiskLikelihood.POSSIBLE: 0.375,
            RiskLikelihood.LIKELY: 0.625,
            RiskLikelihood.ALMOST_CERTAIN: 0.875
        }
        
        severity_weight = severity_weights.get(self.severity, 1.0)
        likelihood_weight = likelihood_weights.get(self.likelihood, 0.5)
        
        # Risk score = Severity * Likelihood * 10 (scale to 0-100)
        base_score = severity_weight * likelihood_weight * 10
        
        # Adjust for cost impact
        if self.expected_cost_impact_bps > 0:
            cost_multiplier = 1.0 + min(1.0, self.expected_cost_impact_bps / 50.0)
            base_score *= cost_multiplier
        
        return min(100.0, base_score)


@dataclass
class MarketConditions:
    """Market conditions for risk assessment"""
    
    # Price and volatility
    current_price: float = 0.0
    price_volatility: float = 0.30           # Annual volatility
    intraday_volatility: float = 0.02        # Intraday volatility
    recent_price_change_pct: float = 0.0     # Recent price change
    
    # Volume and liquidity
    current_volume: float = 0.0
    average_daily_volume: float = 1000000
    volume_ratio: float = 1.0                # Current vs average volume
    bid_ask_spread_bps: float = 10.0
    market_depth: float = 50000              # Total market depth
    
    # Market structure
    trading_session: str = "REGULAR"         # REGULAR, PRE, POST, CLOSED
    time_to_close_minutes: float = 240.0     # Minutes until market close
    is_expiration_day: bool = False          # Options/futures expiration
    
    # News and events
    news_sentiment: float = 0.0              # -1 to +1
    earnings_proximity_days: float = 30.0    # Days to earnings
    has_pending_announcements: bool = False
    
    # Market regime
    market_stress_level: float = 0.0         # 0-1 market stress
    correlation_breakdown: bool = False       # Correlation breakdown indicator
    liquidity_stress: bool = False           # Liquidity stress indicator
    
    @property
    def is_high_volatility_period(self) -> bool:
        """Check if in high volatility period"""
        return self.price_volatility > 0.4 or self.intraday_volatility > 0.04
    
    @property
    def is_low_liquidity_period(self) -> bool:
        """Check if in low liquidity period"""
        return (
            self.volume_ratio < 0.5 or
            self.bid_ask_spread_bps > 30 or
            self.market_depth < 10000
        )
    
    @property
    def market_stress_score(self) -> float:
        """Calculate overall market stress score"""
        stress_factors = [
            self.market_stress_level,
            min(1.0, self.price_volatility / 0.6),
            min(1.0, self.bid_ask_spread_bps / 50.0),
            1.0 - min(1.0, self.volume_ratio),
            abs(self.recent_price_change_pct) / 0.05
        ]
        
        return min(1.0, sum(stress_factors) / len(stress_factors))


@dataclass
class OrderProfile:
    """Order profile for risk assessment"""
    
    # Basic order details
    symbol: str = ""
    quantity: float = 0.0
    side: str = "buy"                        # "buy" or "sell"
    order_value_usd: float = 0.0
    
    # Size characteristics
    percentage_of_adv: float = 0.0           # % of average daily volume
    market_cap_percentage: float = 0.0       # % of market cap
    is_block_order: bool = False
    
    # Execution characteristics
    execution_style: str = "balanced"        # "aggressive", "passive", "balanced"
    time_horizon_minutes: float = 60.0       # Execution time horizon
    urgency_level: float = 0.5               # 0-1 urgency scale
    
    # Portfolio context
    current_position: float = 0.0            # Current position size
    position_limit: float = 0.0              # Position limit
    sector_exposure: float = 0.0             # Sector exposure
    
    # Client and regulatory
    client_type: str = "institutional"       # "retail", "institutional", "prop"
    is_restricted_security: bool = False
    requires_best_execution: bool = True
    
    @property
    def is_large_order(self) -> bool:
        """Check if this is a large order"""
        return (
            self.is_block_order or
            self.percentage_of_adv > 0.05 or  # > 5% of ADV
            self.order_value_usd > 10_000_000  # > $10M
        )
    
    @property
    def concentration_risk_level(self) -> float:
        """Calculate concentration risk level"""
        if self.position_limit == 0:
            return 0.0
        
        new_position = abs(self.current_position + 
                          (self.quantity if self.side == "buy" else -self.quantity))
        
        concentration = new_position / self.position_limit if self.position_limit > 0 else 0.0
        
        return min(1.0, concentration)


class RiskAnalyzer:
    """
    Risk analysis engine for pre-trade assessment
    """
    
    def __init__(self):
        
        # Risk assessment parameters
        self.risk_thresholds = {
            'high_volatility': 0.40,          # 40% annual volatility
            'wide_spread_bps': 30.0,          # 30 bps spread
            'low_volume_ratio': 0.5,          # 50% of average volume
            'large_order_adv': 0.10,          # 10% of ADV
            'concentration_limit': 0.80       # 80% of position limit
        }
        
        # Risk factor templates
        self.risk_templates = self._initialize_risk_templates()
        
        # Historical risk data
        self.risk_history: deque = deque(maxlen=1000)
        
        # Risk model parameters
        self.market_impact_alpha = 0.6       # Market impact power law
        self.liquidity_decay_rate = 0.1      # Liquidity decay rate
        self.volatility_shock_threshold = 2.0  # Volatility shock threshold
        
    def _initialize_risk_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize risk factor templates"""
        
        templates = {
            'high_volatility': {
                'risk_type': RiskType.MARKET_RISK,
                'base_severity': RiskSeverity.MEDIUM,
                'description': 'High market volatility increases execution uncertainty',
                'impact_description': 'Increased price movement during execution',
                'mitigation_strategies': [
                    'Use smaller order sizes',
                    'Increase execution time horizon',
                    'Consider dark pools for stealth'
                ]
            },
            'wide_spreads': {
                'risk_type': RiskType.LIQUIDITY_RISK,
                'base_severity': RiskSeverity.MEDIUM,
                'description': 'Wide bid-ask spreads increase execution costs',
                'impact_description': 'Higher crossing costs and market impact',
                'mitigation_strategies': [
                    'Use limit orders instead of market orders',
                    'Split execution across multiple venues',
                    'Wait for spread compression'
                ]
            },
            'low_liquidity': {
                'risk_type': RiskType.LIQUIDITY_RISK,
                'base_severity': RiskSeverity.HIGH,
                'description': 'Low market liquidity may prevent full execution',
                'impact_description': 'Partial fills and increased market impact',
                'mitigation_strategies': [
                    'Extend execution timeframe',
                    'Use iceberg orders',
                    'Consider crossing networks'
                ]
            },
            'large_order_size': {
                'risk_type': RiskType.MARKET_RISK,
                'base_severity': RiskSeverity.HIGH,
                'description': 'Large order size relative to market may cause significant impact',
                'impact_description': 'High market impact and information leakage',
                'mitigation_strategies': [
                    'Break into smaller parcels',
                    'Use dark pools',
                    'Implement over longer timeframe'
                ]
            },
            'market_close_proximity': {
                'risk_type': RiskType.TIMING_RISK,
                'base_severity': RiskSeverity.MEDIUM,
                'description': 'Proximity to market close increases execution pressure',
                'impact_description': 'Forced execution at unfavorable prices',
                'mitigation_strategies': [
                    'Execute earlier in session',
                    'Consider extended hours trading',
                    'Use more aggressive execution'
                ]
            },
            'earnings_proximity': {
                'risk_type': RiskType.MARKET_RISK,
                'base_severity': RiskSeverity.HIGH,
                'description': 'Proximity to earnings announcement increases volatility risk',
                'impact_description': 'Unpredictable price movements during execution',
                'mitigation_strategies': [
                    'Execute before announcement',
                    'Reduce order size',
                    'Increase risk controls'
                ]
            },
            'concentration_risk': {
                'risk_type': RiskType.CONCENTRATION_RISK,
                'base_severity': RiskSeverity.HIGH,
                'description': 'Order creates excessive position concentration',
                'impact_description': 'Increased portfolio risk and potential regulatory issues',
                'mitigation_strategies': [
                    'Reduce order size',
                    'Diversify across related securities',
                    'Implement position limits'
                ]
            },
            'venue_concentration': {
                'risk_type': RiskType.VENUE_RISK,
                'base_severity': RiskSeverity.MEDIUM,
                'description': 'Over-reliance on single venue creates execution risk',
                'impact_description': 'Venue outage could prevent execution',
                'mitigation_strategies': [
                    'Diversify across multiple venues',
                    'Have backup execution plans',
                    'Monitor venue performance'
                ]
            },
            'information_leakage': {
                'risk_type': RiskType.INFORMATION_LEAKAGE,
                'base_severity': RiskSeverity.MEDIUM,
                'description': 'Large order may signal investment intentions',
                'impact_description': 'Adverse price movement before completion',
                'mitigation_strategies': [
                    'Use dark pools',
                    'Randomize execution timing',
                    'Use multiple execution styles'
                ]
            },
            'operational_risk': {
                'risk_type': RiskType.OPERATIONAL_RISK,
                'base_severity': RiskSeverity.LOW,
                'description': 'Technology or operational failures during execution',
                'impact_description': 'Execution delays or failures',
                'mitigation_strategies': [
                    'Have backup systems',
                    'Monitor system health',
                    'Implement failover procedures'
                ]
            }
        }
        
        return templates
    
    def assess_pre_trade_risks(
        self,
        order_profile: OrderProfile,
        market_conditions: MarketConditions
    ) -> List[RiskFactor]:
        """
        Perform comprehensive pre-trade risk assessment
        
        Args:
            order_profile: Order characteristics
            market_conditions: Current market conditions
            
        Returns:
            List of identified risk factors
        """
        
        risk_factors = []
        
        # Market risk assessment
        market_risks = self._assess_market_risks(order_profile, market_conditions)
        risk_factors.extend(market_risks)
        
        # Liquidity risk assessment
        liquidity_risks = self._assess_liquidity_risks(order_profile, market_conditions)
        risk_factors.extend(liquidity_risks)
        
        # Timing risk assessment
        timing_risks = self._assess_timing_risks(order_profile, market_conditions)
        risk_factors.extend(timing_risks)
        
        # Size and concentration risks
        size_risks = self._assess_size_risks(order_profile, market_conditions)
        risk_factors.extend(size_risks)
        
        # Information leakage risks
        leakage_risks = self._assess_information_leakage_risks(order_profile, market_conditions)
        risk_factors.extend(leakage_risks)
        
        # Operational risks
        operational_risks = self._assess_operational_risks(order_profile, market_conditions)
        risk_factors.extend(operational_risks)
        
        # Regulatory risks
        regulatory_risks = self._assess_regulatory_risks(order_profile, market_conditions)
        risk_factors.extend(regulatory_risks)
        
        # Sort by risk score (highest first)
        risk_factors.sort(key=lambda x: x.risk_score, reverse=True)
        
        return risk_factors
    
    def _assess_market_risks(
        self,
        order_profile: OrderProfile,
        market_conditions: MarketConditions
    ) -> List[RiskFactor]:
        """Assess market-related risks"""
        
        risks = []
        
        # High volatility risk
        if market_conditions.is_high_volatility_period:
            
            # Calculate likelihood based on volatility level
            vol_ratio = market_conditions.price_volatility / 0.30  # Normalize to 30%
            
            if vol_ratio > 2.0:
                likelihood = RiskLikelihood.ALMOST_CERTAIN
                severity = RiskSeverity.HIGH
            elif vol_ratio > 1.5:
                likelihood = RiskLikelihood.LIKELY
                severity = RiskSeverity.MEDIUM
            else:
                likelihood = RiskLikelihood.POSSIBLE
                severity = RiskSeverity.MEDIUM
            
            # Estimate cost impact
            expected_impact = market_conditions.intraday_volatility * 10000 * 0.5  # 50% of intraday vol
            max_impact = expected_impact * 2.0
            
            risk = RiskFactor(
                risk_id="market_volatility",
                risk_type=RiskType.MARKET_RISK,
                severity=severity,
                likelihood=likelihood,
                description=f"High volatility ({market_conditions.price_volatility:.1%}) increases execution uncertainty",
                impact_description="Price may move significantly during execution period",
                expected_cost_impact_bps=expected_impact,
                max_cost_impact_bps=max_impact,
                probability=min(0.9, vol_ratio / 2.0),
                mitigation_strategies=[
                    "Use smaller order slices",
                    "Increase execution time horizon", 
                    "Consider volatility-adjusted participation rates"
                ]
            )
            
            risks.append(risk)
        
        # Directional price movement risk
        if abs(market_conditions.recent_price_change_pct) > 0.02:  # 2% recent move
            
            # Adverse if moving against order direction
            is_adverse = (
                (order_profile.side == "buy" and market_conditions.recent_price_change_pct > 0) or
                (order_profile.side == "sell" and market_conditions.recent_price_change_pct < 0)
            )
            
            if is_adverse:
                magnitude = abs(market_conditions.recent_price_change_pct)
                
                if magnitude > 0.05:  # 5% move
                    severity = RiskSeverity.HIGH
                    likelihood = RiskLikelihood.LIKELY
                elif magnitude > 0.03:  # 3% move
                    severity = RiskSeverity.MEDIUM
                    likelihood = RiskLikelihood.POSSIBLE
                else:
                    severity = RiskSeverity.LOW
                    likelihood = RiskLikelihood.UNLIKELY
                
                risk = RiskFactor(
                    risk_id="adverse_price_movement",
                    risk_type=RiskType.MARKET_RISK,
                    severity=severity,
                    likelihood=likelihood,
                    description=f"Recent {magnitude:.1%} price move against order direction",
                    impact_description="Continued adverse movement increases execution cost",
                    expected_cost_impact_bps=magnitude * 5000,  # 50% of move as cost
                    max_cost_impact_bps=magnitude * 10000,
                    probability=0.4,  # 40% chance trend continues
                    mitigation_strategies=[
                        "Execute more aggressively to minimize timing risk",
                        "Consider waiting for price reversion",
                        "Use limit orders to control worst-case cost"
                    ]
                )
                
                risks.append(risk)
        
        # Earnings proximity risk
        if market_conditions.earnings_proximity_days < 7:  # Within a week
            
            days_to_earnings = market_conditions.earnings_proximity_days
            
            if days_to_earnings < 1:
                severity = RiskSeverity.CRITICAL
                likelihood = RiskLikelihood.ALMOST_CERTAIN
            elif days_to_earnings < 3:
                severity = RiskSeverity.HIGH
                likelihood = RiskLikelihood.LIKELY
            else:
                severity = RiskSeverity.MEDIUM
                likelihood = RiskLikelihood.POSSIBLE
            
            # Estimate volatility spike impact
            vol_multiplier = max(1.5, 3.0 - days_to_earnings * 0.3)
            expected_impact = market_conditions.intraday_volatility * vol_multiplier * 10000
            
            risk = RiskFactor(
                risk_id="earnings_proximity",
                risk_type=RiskType.MARKET_RISK,
                severity=severity,
                likelihood=likelihood,
                description=f"Earnings announcement in {days_to_earnings:.1f} days",
                impact_description="Volatility spike around earnings may cause execution issues",
                expected_cost_impact_bps=expected_impact * 0.6,
                max_cost_impact_bps=expected_impact * 1.5,
                probability=0.8 if days_to_earnings < 3 else 0.5,
                mitigation_strategies=[
                    "Execute before earnings announcement",
                    "Reduce order size",
                    "Increase risk controls and limits"
                ]
            )
            
            risks.append(risk)
        
        return risks
    
    def _assess_liquidity_risks(
        self,
        order_profile: OrderProfile,
        market_conditions: MarketConditions
    ) -> List[RiskFactor]:
        """Assess liquidity-related risks"""
        
        risks = []
        
        # Low liquidity risk
        if market_conditions.is_low_liquidity_period:
            
            # Determine severity based on liquidity metrics
            liquidity_factors = [
                market_conditions.volume_ratio,
                max(0, 1.0 - market_conditions.bid_ask_spread_bps / 50.0),
                min(1.0, market_conditions.market_depth / 100000)
            ]
            
            liquidity_score = sum(liquidity_factors) / len(liquidity_factors)
            
            if liquidity_score < 0.3:
                severity = RiskSeverity.CRITICAL
                likelihood = RiskLikelihood.LIKELY
            elif liquidity_score < 0.5:
                severity = RiskSeverity.HIGH
                likelihood = RiskLikelihood.POSSIBLE
            else:
                severity = RiskSeverity.MEDIUM
                likelihood = RiskLikelihood.UNLIKELY
            
            # Calculate impact based on order size vs available liquidity
            if order_profile.percentage_of_adv > 0:
                participation_impact = min(50.0, order_profile.percentage_of_adv * 100 * 0.5)
            else:
                participation_impact = 10.0
            
            risk = RiskFactor(
                risk_id="low_liquidity",
                risk_type=RiskType.LIQUIDITY_RISK,
                severity=severity,
                likelihood=likelihood,
                description=f"Low liquidity conditions (score: {liquidity_score:.2f})",
                impact_description="Difficult to execute without significant market impact",
                expected_cost_impact_bps=participation_impact,
                max_cost_impact_bps=participation_impact * 2.0,
                probability=1.0 - liquidity_score,
                mitigation_strategies=[
                    "Extend execution timeframe",
                    "Use dark pools and crossing networks",
                    "Reduce order size or split into smaller parcels"
                ]
            )
            
            risks.append(risk)
        
        # Wide spread risk
        if market_conditions.bid_ask_spread_bps > self.risk_thresholds['wide_spread_bps']:
            
            spread_bps = market_conditions.bid_ask_spread_bps
            
            if spread_bps > 50:
                severity = RiskSeverity.HIGH
                likelihood = RiskLikelihood.ALMOST_CERTAIN
            elif spread_bps > 30:
                severity = RiskSeverity.MEDIUM
                likelihood = RiskLikelihood.LIKELY
            else:
                severity = RiskSeverity.LOW
                likelihood = RiskLikelihood.POSSIBLE
            
            # Spread cost impact
            expected_spread_cost = spread_bps * 0.5  # Half spread
            
            risk = RiskFactor(
                risk_id="wide_spreads",
                risk_type=RiskType.LIQUIDITY_RISK,
                severity=severity,
                likelihood=likelihood,
                description=f"Wide bid-ask spread ({spread_bps:.1f} bps)",
                impact_description="High crossing costs for market orders",
                expected_cost_impact_bps=expected_spread_cost,
                max_cost_impact_bps=spread_bps,  # Full spread worst case
                probability=0.9,  # Spread cost is almost certain
                mitigation_strategies=[
                    "Use limit orders to avoid crossing spread",
                    "Wait for spread compression",
                    "Use venues with better spreads"
                ]
            )
            
            risks.append(risk)
        
        return risks
    
    def _assess_timing_risks(
        self,
        order_profile: OrderProfile,
        market_conditions: MarketConditions
    ) -> List[RiskFactor]:
        """Assess timing-related risks"""
        
        risks = []
        
        # Market close proximity risk
        if market_conditions.time_to_close_minutes < 60:  # Less than 1 hour to close
            
            time_remaining = market_conditions.time_to_close_minutes
            
            if time_remaining < 15:
                severity = RiskSeverity.CRITICAL
                likelihood = RiskLikelihood.ALMOST_CERTAIN
            elif time_remaining < 30:
                severity = RiskSeverity.HIGH
                likelihood = RiskLikelihood.LIKELY
            else:
                severity = RiskSeverity.MEDIUM
                likelihood = RiskLikelihood.POSSIBLE
            
            # Time pressure increases execution costs
            time_pressure_multiplier = max(1.0, 60 / time_remaining)
            base_cost = 5.0  # 5 bps base time pressure cost
            expected_impact = base_cost * time_pressure_multiplier
            
            risk = RiskFactor(
                risk_id="market_close_proximity",
                risk_type=RiskType.TIMING_RISK,
                severity=severity,
                likelihood=likelihood,
                description=f"Only {time_remaining:.0f} minutes until market close",
                impact_description="Time pressure may force execution at unfavorable prices",
                expected_cost_impact_bps=expected_impact,
                max_cost_impact_bps=expected_impact * 2.0,
                probability=min(0.9, (60 - time_remaining) / 60),
                mitigation_strategies=[
                    "Execute more aggressively",
                    "Consider extended hours trading",
                    "Prioritize completion over cost optimization"
                ]
            )
            
            risks.append(risk)
        
        # Extended execution horizon risk
        if order_profile.time_horizon_minutes > 240:  # More than 4 hours
            
            horizon_hours = order_profile.time_horizon_minutes / 60
            
            if horizon_hours > 24:  # Multi-day
                severity = RiskSeverity.HIGH
                likelihood = RiskLikelihood.LIKELY
            elif horizon_hours > 8:
                severity = RiskSeverity.MEDIUM
                likelihood = RiskLikelihood.POSSIBLE
            else:
                severity = RiskSeverity.LOW
                likelihood = RiskLikelihood.UNLIKELY
            
            # Longer horizon increases timing risk
            vol_cost = market_conditions.price_volatility * math.sqrt(horizon_hours / 24) * 10000
            expected_timing_cost = vol_cost * 0.3  # 30% of volatility risk
            
            risk = RiskFactor(
                risk_id="extended_execution_horizon",
                risk_type=RiskType.TIMING_RISK,
                severity=severity,
                likelihood=likelihood,
                description=f"Extended execution horizon ({horizon_hours:.1f} hours)",
                impact_description="Longer execution period increases market movement risk",
                expected_cost_impact_bps=expected_timing_cost,
                max_cost_impact_bps=expected_timing_cost * 2.0,
                probability=min(0.7, horizon_hours / 24),
                mitigation_strategies=[
                    "Break into shorter execution periods",
                    "Use more adaptive execution strategies",
                    "Implement stop-loss mechanisms"
                ]
            )
            
            risks.append(risk)
        
        return risks
    
    def _assess_size_risks(
        self,
        order_profile: OrderProfile,
        market_conditions: MarketConditions
    ) -> List[RiskFactor]:
        """Assess order size and concentration risks"""
        
        risks = []
        
        # Large order size risk
        if order_profile.percentage_of_adv > self.risk_thresholds['large_order_adv']:
            
            adv_percentage = order_profile.percentage_of_adv * 100
            
            if adv_percentage > 25:  # More than 25% of ADV
                severity = RiskSeverity.CRITICAL
                likelihood = RiskLikelihood.ALMOST_CERTAIN
            elif adv_percentage > 15:
                severity = RiskSeverity.HIGH
                likelihood = RiskLikelihood.LIKELY
            else:
                severity = RiskSeverity.MEDIUM
                likelihood = RiskLikelihood.POSSIBLE
            
            # Market impact increases non-linearly with size
            impact_coefficient = 0.5  # Square root law coefficient
            market_impact = impact_coefficient * (adv_percentage ** self.market_impact_alpha)
            
            risk = RiskFactor(
                risk_id="large_order_size",
                risk_type=RiskType.MARKET_RISK,
                severity=severity,
                likelihood=likelihood,
                description=f"Order size is {adv_percentage:.1f}% of average daily volume",
                impact_description="Large size creates significant market impact",
                expected_cost_impact_bps=market_impact,
                max_cost_impact_bps=market_impact * 1.8,
                probability=0.95,  # Market impact is almost certain for large orders
                mitigation_strategies=[
                    "Break order into smaller parcels",
                    "Use dark pools to hide order size",
                    "Extend execution timeframe",
                    "Use iceberg orders"
                ]
            )
            
            risks.append(risk)
        
        # Position concentration risk
        concentration_level = order_profile.concentration_risk_level
        
        if concentration_level > self.risk_thresholds['concentration_limit']:
            
            if concentration_level > 0.95:
                severity = RiskSeverity.CRITICAL
                likelihood = RiskLikelihood.ALMOST_CERTAIN
            elif concentration_level > 0.90:
                severity = RiskSeverity.HIGH
                likelihood = RiskLikelihood.LIKELY
            else:
                severity = RiskSeverity.MEDIUM
                likelihood = RiskLikelihood.POSSIBLE
            
            risk = RiskFactor(
                risk_id="position_concentration",
                risk_type=RiskType.CONCENTRATION_RISK,
                severity=severity,
                likelihood=likelihood,
                description=f"Order creates {concentration_level:.1%} position concentration",
                impact_description="High concentration increases portfolio risk",
                expected_cost_impact_bps=0.0,  # No direct cost impact
                max_cost_impact_bps=0.0,
                probability=1.0,  # Concentration is certain
                mitigation_strategies=[
                    "Reduce order size",
                    "Diversify across related securities",
                    "Review position limits",
                    "Consider hedging strategies"
                ]
            )
            
            risks.append(risk)
        
        return risks
    
    def _assess_information_leakage_risks(
        self,
        order_profile: OrderProfile,
        market_conditions: MarketConditions
    ) -> List[RiskFactor]:
        """Assess information leakage risks"""
        
        risks = []
        
        # Large order information leakage
        if order_profile.is_large_order:
            
            # Leakage risk increases with order size and decreases with liquidity
            size_factor = min(1.0, order_profile.percentage_of_adv * 10)  # Normalize
            liquidity_factor = max(0.3, 1.0 - market_conditions.volume_ratio)
            
            leakage_score = size_factor * liquidity_factor
            
            if leakage_score > 0.7:
                severity = RiskSeverity.HIGH
                likelihood = RiskLikelihood.LIKELY
            elif leakage_score > 0.4:
                severity = RiskSeverity.MEDIUM
                likelihood = RiskLikelihood.POSSIBLE
            else:
                severity = RiskSeverity.LOW
                likelihood = RiskLikelihood.UNLIKELY
            
            # Information leakage can cause adverse price movement
            expected_leakage_cost = leakage_score * 15.0  # Up to 15 bps
            
            risk = RiskFactor(
                risk_id="information_leakage",
                risk_type=RiskType.INFORMATION_LEAKAGE,
                severity=severity,
                likelihood=likelihood,
                description="Large order size may signal trading intentions",
                impact_description="Market participants may front-run the order",
                expected_cost_impact_bps=expected_leakage_cost,
                max_cost_impact_bps=expected_leakage_cost * 2.0,
                probability=leakage_score,
                mitigation_strategies=[
                    "Use dark pools for stealth execution",
                    "Randomize execution timing and sizing",
                    "Use multiple execution algorithms",
                    "Consider iceberg orders"
                ]
            )
            
            risks.append(risk)
        
        return risks
    
    def _assess_operational_risks(
        self,
        order_profile: OrderProfile,
        market_conditions: MarketConditions
    ) -> List[RiskFactor]:
        """Assess operational risks"""
        
        risks = []
        
        # Basic operational risk (always present at low level)
        base_operational_risk = RiskFactor(
            risk_id="operational_risk",
            risk_type=RiskType.OPERATIONAL_RISK,
            severity=RiskSeverity.LOW,
            likelihood=RiskLikelihood.UNLIKELY,
            description="General operational and technology risks",
            impact_description="System failures or connectivity issues during execution",
            expected_cost_impact_bps=2.0,
            max_cost_impact_bps=20.0,
            probability=0.05,  # 5% chance of operational issues
            mitigation_strategies=[
                "Have backup execution systems",
                "Monitor system health continuously",
                "Implement failover procedures",
                "Maintain multiple venue connections"
            ]
        )
        
        risks.append(base_operational_risk)
        
        # Extended hours operational risk
        if market_conditions.trading_session != "REGULAR":
            
            extended_hours_risk = RiskFactor(
                risk_id="extended_hours_operational",
                risk_type=RiskType.OPERATIONAL_RISK,
                severity=RiskSeverity.MEDIUM,
                likelihood=RiskLikelihood.POSSIBLE,
                description="Extended hours trading has higher operational risk",
                impact_description="Limited support and liquidity during extended hours",
                expected_cost_impact_bps=5.0,
                max_cost_impact_bps=25.0,
                probability=0.15,
                mitigation_strategies=[
                    "Increase monitoring during extended hours",
                    "Have specialized support available",
                    "Use only reliable venues"
                ]
            )
            
            risks.append(extended_hours_risk)
        
        return risks
    
    def _assess_regulatory_risks(
        self,
        order_profile: OrderProfile,
        market_conditions: MarketConditions
    ) -> List[RiskFactor]:
        """Assess regulatory and compliance risks"""
        
        risks = []
        
        # Best execution requirements
        if order_profile.requires_best_execution:
            
            # Risk increases with order size and market conditions
            if order_profile.is_large_order or market_conditions.is_low_liquidity_period:
                severity = RiskSeverity.MEDIUM
                likelihood = RiskLikelihood.POSSIBLE
            else:
                severity = RiskSeverity.LOW
                likelihood = RiskLikelihood.UNLIKELY
            
            best_execution_risk = RiskFactor(
                risk_id="best_execution_compliance",
                risk_type=RiskType.REGULATORY_RISK,
                severity=severity,
                likelihood=likelihood,
                description="Best execution regulatory requirements",
                impact_description="Potential regulatory scrutiny of execution quality",
                expected_cost_impact_bps=0.0,  # No direct cost but compliance cost
                max_cost_impact_bps=0.0,
                probability=0.1 if order_profile.is_large_order else 0.02,
                mitigation_strategies=[
                    "Document execution decision rationale",
                    "Use TCA to demonstrate best execution",
                    "Consider multiple venue execution",
                    "Maintain execution audit trail"
                ]
            )
            
            risks.append(best_execution_risk)
        
        # Position limit compliance
        if order_profile.concentration_risk_level > 0.8:
            
            position_limit_risk = RiskFactor(
                risk_id="position_limit_compliance",
                risk_type=RiskType.REGULATORY_RISK,
                severity=RiskSeverity.HIGH,
                likelihood=RiskLikelihood.LIKELY,
                description="Order may violate position limits",
                impact_description="Regulatory violations and potential fines",
                expected_cost_impact_bps=0.0,
                max_cost_impact_bps=0.0,
                probability=0.8 if order_profile.concentration_risk_level > 0.9 else 0.3,
                mitigation_strategies=[
                    "Verify position limits before execution",
                    "Reduce order size if necessary",
                    "Consider cross-hedging strategies"
                ]
            )
            
            risks.append(position_limit_risk)
        
        return risks
    
    def calculate_overall_risk_score(self, risk_factors: List[RiskFactor]) -> float:
        """Calculate overall risk score from individual risk factors"""
        
        if not risk_factors:
            return 0.0
        
        # Weighted combination of risk scores
        total_weighted_score = 0.0
        total_weights = 0.0
        
        # Risk type weights
        risk_type_weights = {
            RiskType.MARKET_RISK: 1.0,
            RiskType.LIQUIDITY_RISK: 0.9,
            RiskType.TIMING_RISK: 0.8,
            RiskType.VENUE_RISK: 0.6,
            RiskType.OPERATIONAL_RISK: 0.5,
            RiskType.INFORMATION_LEAKAGE: 0.7,
            RiskType.CONCENTRATION_RISK: 0.8,
            RiskType.REGULATORY_RISK: 0.9
        }
        
        for risk_factor in risk_factors:
            weight = risk_type_weights.get(risk_factor.risk_type, 1.0)
            total_weighted_score += risk_factor.risk_score * weight
            total_weights += weight
        
        overall_score = total_weighted_score / total_weights if total_weights > 0 else 0.0
        
        return min(100.0, overall_score)
    
    def generate_risk_report(
        self,
        order_profile: OrderProfile,
        market_conditions: MarketConditions
    ) -> Dict[str, Any]:
        """Generate comprehensive risk assessment report"""
        
        # Perform risk assessment
        risk_factors = self.assess_pre_trade_risks(order_profile, market_conditions)
        
        # Calculate overall risk score
        overall_risk_score = self.calculate_overall_risk_score(risk_factors)
        
        # Categorize risks by type
        risks_by_type = defaultdict(list)
        for risk in risk_factors:
            risks_by_type[risk.risk_type.value].append(risk)
        
        # Identify top risks
        top_risks = sorted(risk_factors, key=lambda x: x.risk_score, reverse=True)[:5]
        
        # Calculate expected cost impact
        total_expected_cost = sum(risk.expected_cost_impact_bps for risk in risk_factors)
        total_max_cost = sum(risk.max_cost_impact_bps for risk in risk_factors)
        
        # Risk level classification
        if overall_risk_score > 70:
            risk_level = "HIGH"
        elif overall_risk_score > 40:
            risk_level = "MEDIUM"
        elif overall_risk_score > 15:
            risk_level = "LOW"
        else:
            risk_level = "MINIMAL"
        
        # Recommendation
        if overall_risk_score > 80:
            recommendation = "RECONSIDER - High risk execution"
        elif overall_risk_score > 60:
            recommendation = "PROCEED WITH CAUTION - Implement risk mitigations"
        elif overall_risk_score > 30:
            recommendation = "PROCEED - Monitor risks during execution"
        else:
            recommendation = "PROCEED - Low risk execution"
        
        report = {
            'timestamp': datetime.now(),
            'symbol': order_profile.symbol,
            'order_summary': {
                'quantity': order_profile.quantity,
                'side': order_profile.side,
                'order_value_usd': order_profile.order_value_usd,
                'percentage_of_adv': order_profile.percentage_of_adv * 100,
                'execution_style': order_profile.execution_style,
                'time_horizon_minutes': order_profile.time_horizon_minutes
            },
            'risk_assessment': {
                'overall_risk_score': overall_risk_score,
                'risk_level': risk_level,
                'recommendation': recommendation,
                'total_risk_factors': len(risk_factors),
                'expected_cost_impact_bps': total_expected_cost,
                'max_cost_impact_bps': total_max_cost
            },
            'top_risks': [
                {
                    'risk_id': risk.risk_id,
                    'risk_type': risk.risk_type.value,
                    'severity': risk.severity.value,
                    'likelihood': risk.likelihood.value,
                    'risk_score': risk.risk_score,
                    'description': risk.description,
                    'impact_description': risk.impact_description,
                    'expected_cost_impact_bps': risk.expected_cost_impact_bps,
                    'probability': risk.probability,
                    'mitigation_strategies': risk.mitigation_strategies
                }
                for risk in top_risks
            ],
            'risks_by_type': {
                risk_type: len(risks) for risk_type, risks in risks_by_type.items()
            },
            'market_conditions_summary': {
                'volatility': market_conditions.price_volatility,
                'liquidity_score': 1.0 - market_conditions.market_stress_score,
                'spread_bps': market_conditions.bid_ask_spread_bps,
                'volume_ratio': market_conditions.volume_ratio,
                'time_to_close_minutes': market_conditions.time_to_close_minutes,
                'market_stress_score': market_conditions.market_stress_score
            },
            'mitigation_summary': {
                'total_mitigatable_risks': sum(1 for risk in risk_factors if risk.is_mitigatable),
                'key_mitigation_themes': self._extract_mitigation_themes(risk_factors)
            }
        }
        
        return report
    
    def _extract_mitigation_themes(self, risk_factors: List[RiskFactor]) -> List[str]:
        """Extract common mitigation themes from risk factors"""
        
        mitigation_counts = defaultdict(int)
        
        for risk in risk_factors:
            for mitigation in risk.mitigation_strategies:
                # Normalize and count similar mitigations
                if "dark pool" in mitigation.lower():
                    mitigation_counts["Use dark pools"] += 1
                elif "limit order" in mitigation.lower():
                    mitigation_counts["Use limit orders"] += 1
                elif "smaller" in mitigation.lower() or "reduce" in mitigation.lower():
                    mitigation_counts["Reduce order size"] += 1
                elif "time" in mitigation.lower() and ("extend" in mitigation.lower() or "longer" in mitigation.lower()):
                    mitigation_counts["Extend execution timeframe"] += 1
                elif "venue" in mitigation.lower() and "multiple" in mitigation.lower():
                    mitigation_counts["Diversify venues"] += 1
                elif "monitor" in mitigation.lower():
                    mitigation_counts["Increase monitoring"] += 1
        
        # Return most common themes
        sorted_themes = sorted(mitigation_counts.items(), key=lambda x: x[1], reverse=True)
        return [theme for theme, count in sorted_themes[:5]]


if __name__ == "__main__":
    import random
    
    # Example usage and testing
    print("Testing Risk Assessor...")
    
    # Create risk analyzer
    analyzer = RiskAnalyzer()
    
    # Sample market conditions
    market_conditions = MarketConditions(
        current_price=150.0,
        price_volatility=0.45,  # High volatility
        intraday_volatility=0.035,
        recent_price_change_pct=0.025,  # 2.5% recent move
        current_volume=800000,
        average_daily_volume=1500000,
        volume_ratio=0.53,  # Low volume
        bid_ask_spread_bps=25.0,  # Wide spread
        market_depth=15000,
        trading_session="REGULAR",
        time_to_close_minutes=45,  # Close to market close
        earnings_proximity_days=2.5,  # Earnings soon
        news_sentiment=-0.3,
        market_stress_level=0.6
    )
    
    # Test different order scenarios
    test_orders = [
        {
            'name': 'Small Retail Order',
            'profile': OrderProfile(
                symbol="AAPL",
                quantity=500,
                side="buy",
                order_value_usd=75000,
                percentage_of_adv=500 / 1500000,  # Small percentage
                execution_style="balanced",
                time_horizon_minutes=30,
                urgency_level=0.6,
                current_position=0,
                position_limit=10000,
                client_type="retail"
            )
        },
        {
            'name': 'Large Institutional Block',
            'profile': OrderProfile(
                symbol="AAPL",
                quantity=75000,
                side="sell",
                order_value_usd=11250000,
                percentage_of_adv=75000 / 1500000,  # 5% of ADV
                is_block_order=True,
                execution_style="passive",
                time_horizon_minutes=180,
                urgency_level=0.3,
                current_position=50000,
                position_limit=100000,
                sector_exposure=0.4,
                client_type="institutional"
            )
        },
        {
            'name': 'Urgent Prop Trade',
            'profile': OrderProfile(
                symbol="AAPL",
                quantity=25000,
                side="buy",
                order_value_usd=3750000,
                percentage_of_adv=25000 / 1500000,
                execution_style="aggressive",
                time_horizon_minutes=10,
                urgency_level=0.9,
                current_position=80000,
                position_limit=100000,  # Near limit
                client_type="prop"
            )
        }
    ]
    
    print(f"\nRisk Assessment Results:")
    print(f"Market Conditions:")
    print(f"  Volatility: {market_conditions.price_volatility:.1%}")
    print(f"  Volume Ratio: {market_conditions.volume_ratio:.1%}")
    print(f"  Spread: {market_conditions.bid_ask_spread_bps:.1f} bps")
    print(f"  Time to Close: {market_conditions.time_to_close_minutes:.0f} minutes")
    print(f"  Earnings in: {market_conditions.earnings_proximity_days:.1f} days")
    print(f"  Market Stress: {market_conditions.market_stress_score:.1%}")
    
    for test_case in test_orders:
        print(f"\n{test_case['name']}:")
        profile = test_case['profile']
        
        print(f"  Order: {profile.quantity:,} shares {profile.side}")
        print(f"  Value: ${profile.order_value_usd:,.0f}")
        print(f"  % of ADV: {profile.percentage_of_adv * 100:.2f}%")
        print(f"  Urgency: {profile.urgency_level:.1f}")
        
        # Perform risk assessment
        risk_factors = analyzer.assess_pre_trade_risks(profile, market_conditions)
        overall_risk = analyzer.calculate_overall_risk_score(risk_factors)
        
        print(f"  Overall Risk Score: {overall_risk:.1f}/100")
        
        if overall_risk > 70:
            risk_level = "HIGH"
        elif overall_risk > 40:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        print(f"  Risk Level: {risk_level}")
        print(f"  Risk Factors Identified: {len(risk_factors)}")
        
        # Show top 3 risks
        top_risks = sorted(risk_factors, key=lambda x: x.risk_score, reverse=True)[:3]
        
        if top_risks:
            print(f"  Top Risks:")
            for i, risk in enumerate(top_risks, 1):
                print(f"    {i}. {risk.description}")
                print(f"       Severity: {risk.severity.value}, Likelihood: {risk.likelihood.value}")
                print(f"       Score: {risk.risk_score:.1f}, Expected Impact: {risk.expected_cost_impact_bps:.1f} bps")
                if risk.mitigation_strategies:
                    print(f"       Key Mitigation: {risk.mitigation_strategies[0]}")
    
    # Generate comprehensive report
    print(f"\nGenerating Comprehensive Risk Report...")
    
    sample_order = test_orders[1]['profile']  # Large institutional block
    
    report = analyzer.generate_risk_report(sample_order, market_conditions)
    
    print(f"Risk Assessment Report:")
    print(f"  Symbol: {report['symbol']}")
    print(f"  Order: {report['order_summary']['quantity']:,} shares {report['order_summary']['side']}")
    print(f"  Value: ${report['order_summary']['order_value_usd']:,.0f}")
    
    print(f"  Risk Assessment:")
    assessment = report['risk_assessment']
    print(f"    Overall Risk Score: {assessment['overall_risk_score']:.1f}/100")
    print(f"    Risk Level: {assessment['risk_level']}")
    print(f"    Recommendation: {assessment['recommendation']}")
    print(f"    Total Risk Factors: {assessment['total_risk_factors']}")
    print(f"    Expected Cost Impact: {assessment['expected_cost_impact_bps']:.1f} bps")
    print(f"    Max Cost Impact: {assessment['max_cost_impact_bps']:.1f} bps")
    
    print(f"  Top Risks:")
    for i, risk in enumerate(report['top_risks'][:3], 1):
        print(f"    {i}. {risk['description']}")
        print(f"       Type: {risk['risk_type']}, Score: {risk['risk_score']:.1f}")
        print(f"       Impact: {risk['expected_cost_impact_bps']:.1f} bps")
    
    print(f"  Risk Distribution:")
    for risk_type, count in report['risks_by_type'].items():
        print(f"    {risk_type}: {count}")
    
    print(f"  Key Mitigation Themes:")
    for theme in report['mitigation_summary']['key_mitigation_themes']:
        print(f"    - {theme}")
    
    print("\nRisk assessor testing completed!")