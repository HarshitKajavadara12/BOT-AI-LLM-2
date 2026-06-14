"""
Adaptive Execution Engine
Dynamic execution adjustment based on real-time market conditions
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
import asyncio
from collections import deque, defaultdict
import warnings


class AdaptiveSignal(Enum):
    """Adaptive execution signals"""
    ACCELERATE = "ACCELERATE"       # Speed up execution
    DECELERATE = "DECELERATE"       # Slow down execution  
    MAINTAIN = "MAINTAIN"           # Keep current pace
    PAUSE = "PAUSE"                 # Pause execution temporarily
    OPPORTUNISTIC = "OPPORTUNISTIC" # Wait for better conditions
    EMERGENCY = "EMERGENCY"         # Emergency liquidation mode


class MarketRegime(Enum):
    """Market regime classification"""
    NORMAL = "NORMAL"               # Normal market conditions
    VOLATILE = "VOLATILE"           # High volatility period
    TRENDING = "TRENDING"           # Strong directional movement
    CHOPPY = "CHOPPY"              # Range-bound, choppy conditions
    STRESSED = "STRESSED"           # Market stress/crisis conditions
    ILLIQUID = "ILLIQUID"          # Low liquidity conditions


class AdaptationTrigger(Enum):
    """Triggers for execution adaptation"""
    PRICE_MOVEMENT = "PRICE_MOVEMENT"           # Significant price change
    VOLUME_SPIKE = "VOLUME_SPIKE"               # Volume anomaly
    VOLATILITY_CHANGE = "VOLATILITY_CHANGE"     # Volatility regime change
    LIQUIDITY_CHANGE = "LIQUIDITY_CHANGE"       # Liquidity conditions change
    NEWS_EVENT = "NEWS_EVENT"                   # News or announcement
    EXECUTION_PERFORMANCE = "EXECUTION_PERFORMANCE"  # Poor execution metrics
    TIME_PRESSURE = "TIME_PRESSURE"             # Running out of time
    OPPORTUNITY = "OPPORTUNITY"                 # Favorable execution opportunity


@dataclass
class MarketConditions:
    """Real-time market conditions"""
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Price data
    current_price: float = 0.0
    previous_price: float = 0.0
    price_change: float = 0.0
    price_change_pct: float = 0.0
    
    # Volume data
    current_volume: float = 0.0
    average_volume: float = 0.0
    volume_ratio: float = 1.0
    
    # Volatility measures
    realized_volatility: float = 0.0
    implied_volatility: Optional[float] = None
    volatility_percentile: float = 50.0
    
    # Liquidity measures
    bid_ask_spread: float = 0.0
    spread_bps: float = 0.0
    market_depth: float = 0.0
    liquidity_score: float = 0.5  # 0-1 scale
    
    # Market microstructure
    tick_direction: int = 0  # +1 uptick, -1 downtick, 0 no change
    order_flow_imbalance: float = 0.0
    short_term_momentum: float = 0.0
    
    # Regime classification
    market_regime: MarketRegime = MarketRegime.NORMAL
    regime_confidence: float = 0.0
    
    @property
    def is_favorable_condition(self) -> bool:
        """Check if conditions are favorable for execution"""
        return (
            self.liquidity_score > 0.6 and
            self.volatility_percentile < 80 and
            abs(self.price_change_pct) < 0.02 and
            self.spread_bps < 20
        )
    
    @property
    def stress_level(self) -> float:
        """Calculate market stress level (0-1)"""
        stress_factors = [
            min(1.0, self.volatility_percentile / 100),
            min(1.0, abs(self.price_change_pct) / 0.05),
            min(1.0, max(0, self.spread_bps - 5) / 20),
            1.0 - self.liquidity_score
        ]
        
        return sum(stress_factors) / len(stress_factors)


@dataclass
class AdaptationRule:
    """Rule for execution adaptation"""
    name: str
    trigger: AdaptationTrigger
    condition: Callable[[MarketConditions, Dict[str, Any]], bool]
    action: AdaptiveSignal
    priority: int = 1  # Higher number = higher priority
    cooldown_seconds: float = 60.0  # Minimum time between triggers
    
    # Rule state
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0
    
    def can_trigger(self) -> bool:
        """Check if rule can be triggered (not in cooldown)"""
        if self.last_triggered is None:
            return True
        
        elapsed = (datetime.now() - self.last_triggered).total_seconds()
        return elapsed >= self.cooldown_seconds
    
    def evaluate(self, market_conditions: MarketConditions, context: Dict[str, Any]) -> bool:
        """Evaluate if rule should trigger"""
        if not self.can_trigger():
            return False
        
        try:
            should_trigger = self.condition(market_conditions, context)
            
            if should_trigger:
                self.last_triggered = datetime.now()
                self.trigger_count += 1
                
            return should_trigger
            
        except Exception as e:
            warnings.warn(f"Rule {self.name} evaluation failed: {e}")
            return False


@dataclass
class ExecutionPace:
    """Current execution pace parameters"""
    target_participation_rate: float = 0.10
    current_participation_rate: float = 0.10
    slice_interval_minutes: float = 10.0
    
    # Pace modifiers
    acceleration_factor: float = 1.0  # > 1 = faster, < 1 = slower
    aggression_level: float = 0.5     # 0-1 scale
    
    # Constraints
    max_participation_rate: float = 0.25
    min_participation_rate: float = 0.01
    max_slice_size: Optional[float] = None
    
    def apply_signal(self, signal: AdaptiveSignal, strength: float = 1.0) -> None:
        """Apply adaptive signal to pace"""
        
        if signal == AdaptiveSignal.ACCELERATE:
            # Increase participation rate and aggression
            self.acceleration_factor = min(3.0, self.acceleration_factor * (1 + 0.5 * strength))
            self.aggression_level = min(1.0, self.aggression_level + 0.2 * strength)
            self.slice_interval_minutes = max(2.0, self.slice_interval_minutes * 0.8)
            
        elif signal == AdaptiveSignal.DECELERATE:
            # Decrease participation rate and aggression
            self.acceleration_factor = max(0.3, self.acceleration_factor * (1 - 0.3 * strength))
            self.aggression_level = max(0.1, self.aggression_level - 0.2 * strength)
            self.slice_interval_minutes = min(30.0, self.slice_interval_minutes * 1.5)
            
        elif signal == AdaptiveSignal.PAUSE:
            # Pause execution
            self.acceleration_factor = 0.1
            self.aggression_level = 0.0
            self.slice_interval_minutes = min(60.0, self.slice_interval_minutes * 2.0)
            
        elif signal == AdaptiveSignal.OPPORTUNISTIC:
            # Wait for better conditions
            self.acceleration_factor = max(0.1, self.acceleration_factor * 0.5)
            self.aggression_level = max(0.0, self.aggression_level - 0.3)
            self.slice_interval_minutes = min(45.0, self.slice_interval_minutes * 1.8)
            
        elif signal == AdaptiveSignal.EMERGENCY:
            # Emergency mode - execute quickly
            self.acceleration_factor = 5.0
            self.aggression_level = 1.0
            self.slice_interval_minutes = 1.0
        
        # Apply constraints
        effective_participation = self.target_participation_rate * self.acceleration_factor
        self.current_participation_rate = max(
            self.min_participation_rate,
            min(self.max_participation_rate, effective_participation)
        )
    
    @property
    def is_paused(self) -> bool:
        """Check if execution is effectively paused"""
        return self.acceleration_factor < 0.2 or self.current_participation_rate < 0.005


@dataclass
class AdaptationEvent:
    """Record of adaptation event"""
    timestamp: datetime
    trigger: AdaptationTrigger
    signal: AdaptiveSignal
    rule_name: str
    market_conditions: MarketConditions
    strength: float
    context: Dict[str, Any]
    
    def __post_init__(self):
        if not hasattr(self, 'timestamp'):
            self.timestamp = datetime.now()


class MarketRegimeDetector:
    """
    Real-time market regime detection
    """
    
    def __init__(self, lookback_periods: int = 50):
        self.lookback_periods = lookback_periods
        
        # Price and volume history
        self.price_history = deque(maxlen=lookback_periods)
        self.volume_history = deque(maxlen=lookback_periods)
        self.returns_history = deque(maxlen=lookback_periods)
        
        # Regime thresholds
        self.volatility_thresholds = {
            'low': 0.15,      # Below 15th percentile = low vol
            'high': 0.85      # Above 85th percentile = high vol
        }
        
        self.trend_threshold = 0.02  # 2% directional movement
        self.volume_spike_threshold = 2.0  # 2x average volume
        
    def update(self, price: float, volume: float, timestamp: datetime = None) -> MarketConditions:
        """Update market conditions and detect regime"""
        
        if timestamp is None:
            timestamp = datetime.now()
        
        # Update histories
        self.price_history.append(price)
        self.volume_history.append(volume)
        
        if len(self.price_history) >= 2:
            return_pct = (price - self.price_history[-2]) / self.price_history[-2]
            self.returns_history.append(return_pct)
        
        # Calculate market conditions
        conditions = self._calculate_conditions(price, volume, timestamp)
        
        # Detect regime
        regime, confidence = self._detect_regime(conditions)
        conditions.market_regime = regime
        conditions.regime_confidence = confidence
        
        return conditions
    
    def _calculate_conditions(self, price: float, volume: float, timestamp: datetime) -> MarketConditions:
        """Calculate current market conditions"""
        
        conditions = MarketConditions(timestamp=timestamp, current_price=price)
        
        if len(self.price_history) < 2:
            return conditions
        
        # Price metrics
        conditions.previous_price = self.price_history[-2]
        conditions.price_change = price - conditions.previous_price
        conditions.price_change_pct = conditions.price_change / conditions.previous_price
        
        # Volume metrics
        conditions.current_volume = volume
        if len(self.volume_history) > 1:
            conditions.average_volume = sum(self.volume_history) / len(self.volume_history)
            conditions.volume_ratio = volume / conditions.average_volume if conditions.average_volume > 0 else 1.0
        
        # Volatility calculation
        if len(self.returns_history) >= 10:
            returns_array = np.array(list(self.returns_history))
            conditions.realized_volatility = np.std(returns_array) * np.sqrt(252)  # Annualized
            
            # Volatility percentile
            vol_history = []
            for i in range(10, len(self.returns_history)):
                period_returns = returns_array[i-10:i]
                vol_history.append(np.std(period_returns))
            
            if vol_history:
                current_vol = np.std(returns_array[-10:])
                conditions.volatility_percentile = (
                    sum(v < current_vol for v in vol_history) / len(vol_history) * 100
                )
        
        # Liquidity estimation (simplified)
        if conditions.volume_ratio > 0:
            conditions.liquidity_score = min(1.0, conditions.volume_ratio / 2.0)
        
        # Market microstructure
        if len(self.price_history) >= 3:
            recent_prices = list(self.price_history)[-3:]
            if recent_prices[-1] > recent_prices[-2]:
                conditions.tick_direction = 1
            elif recent_prices[-1] < recent_prices[-2]:
                conditions.tick_direction = -1
            else:
                conditions.tick_direction = 0
        
        # Short-term momentum
        if len(self.returns_history) >= 5:
            recent_returns = list(self.returns_history)[-5:]
            conditions.short_term_momentum = sum(recent_returns)
        
        # Spread estimation (simplified - would use actual bid/ask in practice)
        base_spread = 0.001  # 10 bps base
        vol_adjustment = conditions.realized_volatility / 0.3  # Scale by 30% vol
        liquidity_adjustment = 2.0 - conditions.liquidity_score  # Higher spread for low liquidity
        
        conditions.spread_bps = base_spread * vol_adjustment * liquidity_adjustment * 10000
        conditions.bid_ask_spread = conditions.spread_bps / 10000 * price
        
        return conditions
    
    def _detect_regime(self, conditions: MarketConditions) -> Tuple[MarketRegime, float]:
        """Detect current market regime"""
        
        if len(self.returns_history) < 10:
            return MarketRegime.NORMAL, 0.5
        
        # Calculate regime indicators
        vol_percentile = conditions.volatility_percentile
        price_change_pct = abs(conditions.price_change_pct)
        volume_ratio = conditions.volume_ratio
        momentum = abs(conditions.short_term_momentum)
        
        # Regime detection logic
        regime_scores = {
            MarketRegime.VOLATILE: 0.0,
            MarketRegime.TRENDING: 0.0,
            MarketRegime.CHOPPY: 0.0,
            MarketRegime.STRESSED: 0.0,
            MarketRegime.ILLIQUID: 0.0,
            MarketRegime.NORMAL: 0.0
        }
        
        # Volatile regime
        if vol_percentile > 80:
            regime_scores[MarketRegime.VOLATILE] += (vol_percentile - 80) / 20
        
        # Trending regime
        if momentum > 0.05:  # 5% cumulative movement
            regime_scores[MarketRegime.TRENDING] += min(1.0, momentum / 0.10)
        
        # Choppy regime (high vol but low momentum)
        if vol_percentile > 60 and momentum < 0.02:
            regime_scores[MarketRegime.CHOPPY] += 0.7
        
        # Stressed regime (multiple stress indicators)
        stress_indicators = [
            vol_percentile > 90,
            price_change_pct > 0.03,
            volume_ratio > 3.0,
            conditions.spread_bps > 50
        ]
        stress_score = sum(stress_indicators) / len(stress_indicators)
        regime_scores[MarketRegime.STRESSED] = stress_score
        
        # Illiquid regime
        if conditions.liquidity_score < 0.3:
            regime_scores[MarketRegime.ILLIQUID] += (0.3 - conditions.liquidity_score) / 0.3
        
        # Normal regime (baseline)
        regime_scores[MarketRegime.NORMAL] = max(0.0, 1.0 - sum(regime_scores.values()))
        
        # Select regime with highest score
        best_regime = max(regime_scores.items(), key=lambda x: x[1])
        
        return best_regime[0], best_regime[1]


class AdaptiveExecutionEngine:
    """
    Main adaptive execution engine
    """
    
    def __init__(self, name: str = "AdaptiveEngine"):
        self.name = name
        
        # Core components
        self.regime_detector = MarketRegimeDetector()
        self.execution_pace = ExecutionPace()
        
        # Adaptation rules
        self.adaptation_rules: List[AdaptationRule] = []
        self.rule_priorities = defaultdict(list)
        
        # Event tracking
        self.adaptation_history: List[AdaptationEvent] = []
        self.performance_metrics = defaultdict(float)
        
        # State management
        self.is_active = False
        self.last_adaptation_time = None
        self.adaptation_lock = threading.Lock()
        
        # Initialize default rules
        self._initialize_default_rules()
        
    def _initialize_default_rules(self) -> None:
        """Initialize default adaptation rules"""
        
        # Price movement rules
        self.add_adaptation_rule(
            name="Large Adverse Price Movement",
            trigger=AdaptationTrigger.PRICE_MOVEMENT,
            condition=lambda mc, ctx: (
                abs(mc.price_change_pct) > 0.02 and
                np.sign(mc.price_change_pct) != np.sign(ctx.get('order_side', 1))
            ),
            action=AdaptiveSignal.ACCELERATE,
            priority=3,
            cooldown_seconds=30
        )
        
        self.add_adaptation_rule(
            name="Favorable Price Movement",
            trigger=AdaptationTrigger.PRICE_MOVEMENT,
            condition=lambda mc, ctx: (
                abs(mc.price_change_pct) > 0.015 and
                np.sign(mc.price_change_pct) == np.sign(ctx.get('order_side', 1))
            ),
            action=AdaptiveSignal.DECELERATE,
            priority=2,
            cooldown_seconds=60
        )
        
        # Volatility rules
        self.add_adaptation_rule(
            name="High Volatility Regime",
            trigger=AdaptationTrigger.VOLATILITY_CHANGE,
            condition=lambda mc, ctx: mc.volatility_percentile > 85,
            action=AdaptiveSignal.DECELERATE,
            priority=2,
            cooldown_seconds=120
        )
        
        self.add_adaptation_rule(
            name="Low Volatility Opportunity",
            trigger=AdaptationTrigger.VOLATILITY_CHANGE,
            condition=lambda mc, ctx: (
                mc.volatility_percentile < 20 and
                mc.is_favorable_condition
            ),
            action=AdaptiveSignal.ACCELERATE,
            priority=1,
            cooldown_seconds=180
        )
        
        # Liquidity rules
        self.add_adaptation_rule(
            name="Poor Liquidity Conditions",
            trigger=AdaptationTrigger.LIQUIDITY_CHANGE,
            condition=lambda mc, ctx: mc.liquidity_score < 0.3,
            action=AdaptiveSignal.DECELERATE,
            priority=2,
            cooldown_seconds=90
        )
        
        self.add_adaptation_rule(
            name="Excellent Liquidity",
            trigger=AdaptationTrigger.LIQUIDITY_CHANGE,
            condition=lambda mc, ctx: (
                mc.liquidity_score > 0.8 and
                mc.spread_bps < 10
            ),
            action=AdaptiveSignal.ACCELERATE,
            priority=1,
            cooldown_seconds=120
        )
        
        # Volume spike rules
        self.add_adaptation_rule(
            name="Volume Spike",
            trigger=AdaptationTrigger.VOLUME_SPIKE,
            condition=lambda mc, ctx: mc.volume_ratio > 2.5,
            action=AdaptiveSignal.OPPORTUNISTIC,
            priority=1,
            cooldown_seconds=60
        )
        
        # Stress conditions
        self.add_adaptation_rule(
            name="Market Stress",
            trigger=AdaptationTrigger.EXECUTION_PERFORMANCE,
            condition=lambda mc, ctx: mc.stress_level > 0.7,
            action=AdaptiveSignal.PAUSE,
            priority=4,
            cooldown_seconds=300
        )
        
        # Emergency conditions
        self.add_adaptation_rule(
            name="Emergency Liquidation",
            trigger=AdaptationTrigger.TIME_PRESSURE,
            condition=lambda mc, ctx: (
                ctx.get('time_remaining_pct', 1.0) < 0.1 and
                ctx.get('execution_progress', 0.0) < 0.8
            ),
            action=AdaptiveSignal.EMERGENCY,
            priority=5,
            cooldown_seconds=60
        )
        
        # Build priority index
        self._rebuild_priority_index()
    
    def add_adaptation_rule(
        self,
        name: str,
        trigger: AdaptationTrigger,
        condition: Callable[[MarketConditions, Dict[str, Any]], bool],
        action: AdaptiveSignal,
        priority: int = 1,
        cooldown_seconds: float = 60.0
    ) -> None:
        """Add a new adaptation rule"""
        
        rule = AdaptationRule(
            name=name,
            trigger=trigger,
            condition=condition,
            action=action,
            priority=priority,
            cooldown_seconds=cooldown_seconds
        )
        
        self.adaptation_rules.append(rule)
        self._rebuild_priority_index()
    
    def _rebuild_priority_index(self) -> None:
        """Rebuild priority index for efficient rule lookup"""
        self.rule_priorities.clear()
        
        for rule in self.adaptation_rules:
            self.rule_priorities[rule.priority].append(rule)
        
        # Sort each priority level by creation order
        for priority_list in self.rule_priorities.values():
            priority_list.sort(key=lambda r: r.trigger_count)
    
    def start(self) -> None:
        """Start adaptive execution"""
        self.is_active = True
        print(f"Adaptive execution engine '{self.name}' started")
    
    def stop(self) -> None:
        """Stop adaptive execution"""
        self.is_active = False
        print(f"Adaptive execution engine '{self.name}' stopped")
    
    def update_market_conditions(
        self,
        price: float,
        volume: float,
        execution_context: Dict[str, Any],
        timestamp: datetime = None
    ) -> Optional[AdaptiveSignal]:
        """
        Update market conditions and evaluate adaptation rules
        
        Args:
            price: Current price
            volume: Current volume
            execution_context: Current execution context
            timestamp: Current timestamp
        
        Returns:
            Adaptive signal if triggered, None otherwise
        """
        
        if not self.is_active:
            return None
        
        # Update market conditions
        market_conditions = self.regime_detector.update(price, volume, timestamp)
        
        # Evaluate adaptation rules
        with self.adaptation_lock:
            signal = self._evaluate_adaptation_rules(market_conditions, execution_context)
        
        return signal
    
    def _evaluate_adaptation_rules(
        self,
        market_conditions: MarketConditions,
        context: Dict[str, Any]
    ) -> Optional[AdaptiveSignal]:
        """Evaluate adaptation rules in priority order"""
        
        triggered_rules = []
        
        # Evaluate rules by priority (highest first)
        for priority in sorted(self.rule_priorities.keys(), reverse=True):
            for rule in self.rule_priorities[priority]:
                
                if rule.evaluate(market_conditions, context):
                    triggered_rules.append((rule, priority))
                    
                    # For highest priority rules, trigger immediately
                    if priority >= 4:
                        break
        
        if not triggered_rules:
            return None
        
        # Select the highest priority triggered rule
        best_rule, best_priority = max(triggered_rules, key=lambda x: x[1])
        
        # Calculate signal strength based on conditions
        strength = self._calculate_signal_strength(
            best_rule, market_conditions, context
        )
        
        # Apply the signal
        self.execution_pace.apply_signal(best_rule.action, strength)
        
        # Record adaptation event
        event = AdaptationEvent(
            timestamp=datetime.now(),
            trigger=best_rule.trigger,
            signal=best_rule.action,
            rule_name=best_rule.name,
            market_conditions=market_conditions,
            strength=strength,
            context=context.copy()
        )
        
        self.adaptation_history.append(event)
        self.last_adaptation_time = datetime.now()
        
        # Update performance metrics
        self.performance_metrics['total_adaptations'] += 1
        self.performance_metrics[f'signal_{best_rule.action.value}'] += 1
        
        print(f"Adaptive signal: {best_rule.action.value} (strength: {strength:.2f}) "
              f"triggered by '{best_rule.name}'")
        
        return best_rule.action
    
    def _calculate_signal_strength(
        self,
        rule: AdaptationRule,
        conditions: MarketConditions,
        context: Dict[str, Any]
    ) -> float:
        """Calculate signal strength based on market conditions"""
        
        base_strength = 1.0
        
        # Adjust based on market stress
        stress_multiplier = 1.0 + conditions.stress_level
        
        # Adjust based on execution progress
        execution_progress = context.get('execution_progress', 0.0)
        
        if execution_progress > 0.8:
            # Near completion - reduce adaptation
            progress_multiplier = 0.5
        elif execution_progress < 0.2:
            # Early stage - allow more adaptation
            progress_multiplier = 1.5
        else:
            progress_multiplier = 1.0
        
        # Adjust based on time pressure
        time_remaining_pct = context.get('time_remaining_pct', 1.0)
        
        if time_remaining_pct < 0.2:
            time_multiplier = 2.0  # Urgent
        elif time_remaining_pct > 0.8:
            time_multiplier = 0.8  # Plenty of time
        else:
            time_multiplier = 1.0
        
        # Combine factors
        total_strength = (
            base_strength * stress_multiplier * 
            progress_multiplier * time_multiplier
        )
        
        return max(0.1, min(3.0, total_strength))  # Clamp to reasonable range
    
    def get_current_execution_parameters(self) -> Dict[str, Any]:
        """Get current execution parameters"""
        
        return {
            'participation_rate': self.execution_pace.current_participation_rate,
            'slice_interval_minutes': self.execution_pace.slice_interval_minutes,
            'acceleration_factor': self.execution_pace.acceleration_factor,
            'aggression_level': self.execution_pace.aggression_level,
            'is_paused': self.execution_pace.is_paused,
            'last_adaptation': self.last_adaptation_time
        }
    
    def get_adaptation_statistics(self) -> Dict[str, Any]:
        """Get adaptation performance statistics"""
        
        if not self.adaptation_history:
            return {'status': 'no_adaptations'}
        
        # Calculate statistics
        recent_events = self.adaptation_history[-50:]  # Last 50 events
        
        signal_counts = defaultdict(int)
        trigger_counts = defaultdict(int)
        
        for event in recent_events:
            signal_counts[event.signal.value] += 1
            trigger_counts[event.trigger.value] += 1
        
        # Average adaptation frequency
        if len(recent_events) > 1:
            time_span = (recent_events[-1].timestamp - recent_events[0].timestamp).total_seconds()
            adaptation_frequency = len(recent_events) / (time_span / 3600)  # Per hour
        else:
            adaptation_frequency = 0.0
        
        # Rule effectiveness (placeholder - would need more sophisticated analysis)
        rule_effectiveness = {}
        for rule in self.adaptation_rules:
            if rule.trigger_count > 0:
                # Simple effectiveness measure
                rule_effectiveness[rule.name] = min(1.0, rule.trigger_count / 10)
        
        stats = {
            'total_adaptations': len(self.adaptation_history),
            'recent_adaptations': len(recent_events),
            'adaptation_frequency_per_hour': adaptation_frequency,
            'signal_distribution': dict(signal_counts),
            'trigger_distribution': dict(trigger_counts),
            'rule_effectiveness': rule_effectiveness,
            'performance_metrics': dict(self.performance_metrics)
        }
        
        return stats
    
    def reset_adaptation_state(self) -> None:
        """Reset adaptation state to defaults"""
        
        with self.adaptation_lock:
            self.execution_pace = ExecutionPace()
            self.last_adaptation_time = None
            
            # Reset rule states
            for rule in self.adaptation_rules:
                rule.last_triggered = None
                rule.trigger_count = 0
        
        print(f"Adaptation state reset for engine '{self.name}'")
    
    def generate_adaptation_report(self) -> Dict[str, Any]:
        """Generate comprehensive adaptation report"""
        
        if not self.adaptation_history:
            return {'status': 'no_data'}
        
        # Recent performance
        recent_events = self.adaptation_history[-100:]
        
        # Signal effectiveness analysis
        signal_outcomes = defaultdict(list)
        
        for i, event in enumerate(recent_events[:-1]):
            # Look at next few events to gauge effectiveness
            next_events = recent_events[i+1:i+6]  # Next 5 events
            
            # Simple effectiveness metric: did conditions improve?
            if next_events:
                avg_stress_before = event.market_conditions.stress_level
                avg_stress_after = sum(e.market_conditions.stress_level for e in next_events) / len(next_events)
                effectiveness = max(0, avg_stress_before - avg_stress_after)
                signal_outcomes[event.signal.value].append(effectiveness)
        
        # Calculate average effectiveness per signal
        signal_effectiveness = {}
        for signal, outcomes in signal_outcomes.items():
            if outcomes:
                signal_effectiveness[signal] = sum(outcomes) / len(outcomes)
        
        # Market regime analysis
        regime_adaptations = defaultdict(int)
        for event in recent_events:
            regime_adaptations[event.market_conditions.market_regime.value] += 1
        
        # Timing analysis
        adaptation_intervals = []
        for i in range(1, len(recent_events)):
            interval = (recent_events[i].timestamp - recent_events[i-1].timestamp).total_seconds()
            adaptation_intervals.append(interval)
        
        report = {
            'summary': {
                'total_adaptations': len(self.adaptation_history),
                'recent_period_adaptations': len(recent_events),
                'average_adaptation_interval_seconds': sum(adaptation_intervals) / len(adaptation_intervals) if adaptation_intervals else 0,
                'most_common_signal': max(signal_outcomes.keys(), key=lambda k: len(signal_outcomes[k])) if signal_outcomes else None
            },
            'signal_effectiveness': signal_effectiveness,
            'regime_analysis': dict(regime_adaptations),
            'current_state': {
                'execution_pace': self.get_current_execution_parameters(),
                'market_regime': recent_events[-1].market_conditions.market_regime.value if recent_events else None,
                'stress_level': recent_events[-1].market_conditions.stress_level if recent_events else None
            },
            'rule_performance': {
                rule.name: {
                    'trigger_count': rule.trigger_count,
                    'last_triggered': rule.last_triggered.isoformat() if rule.last_triggered else None,
                    'cooldown_seconds': rule.cooldown_seconds
                }
                for rule in self.adaptation_rules
            }
        }
        
        return report


if __name__ == "__main__":
    import random
    import matplotlib.pyplot as plt
    
    # Example usage and testing
    print("Testing Adaptive Execution Engine...")
    
    # Create adaptive engine
    engine = AdaptiveExecutionEngine("TestEngine")
    engine.start()
    
    # Simulate market data and execution
    print(f"\nSimulating market conditions and adaptive execution...")
    
    base_price = 100.0
    base_volume = 1000000
    
    prices = []
    volumes = []
    signals = []
    participation_rates = []
    market_regimes = []
    stress_levels = []
    
    # Simulation parameters
    simulation_minutes = 240  # 4 hours
    update_frequency = 1      # Every minute
    
    for minute in range(simulation_minutes):
        
        # Generate market data with different regimes
        if minute < 60:
            # Normal market
            price_change = random.gauss(0, 0.001)
            volume_multiplier = random.uniform(0.8, 1.2)
            
        elif minute < 120:
            # Volatile market
            price_change = random.gauss(0, 0.003)
            volume_multiplier = random.uniform(1.5, 3.0)
            
        elif minute < 180:
            # Trending market (upward)
            price_change = random.gauss(0.0005, 0.002)
            volume_multiplier = random.uniform(1.2, 2.0)
            
        else:
            # Stressed market
            price_change = random.gauss(0, 0.005)
            volume_multiplier = random.uniform(0.5, 4.0)
            
            # Add some large moves
            if random.random() < 0.1:  # 10% chance of large move
                price_change += random.choice([-0.02, 0.02])
        
        # Update price and volume
        base_price *= (1 + price_change)
        current_volume = base_volume * volume_multiplier
        
        prices.append(base_price)
        volumes.append(current_volume)
        
        # Create execution context
        execution_progress = minute / simulation_minutes
        time_remaining_pct = 1.0 - execution_progress
        
        execution_context = {
            'order_side': 1,  # Buy order
            'execution_progress': execution_progress,
            'time_remaining_pct': time_remaining_pct,
            'original_quantity': 100000,
            'executed_quantity': execution_progress * 100000
        }
        
        # Update engine with market conditions
        signal = engine.update_market_conditions(
            price=base_price,
            volume=current_volume,
            execution_context=execution_context
        )
        
        signals.append(signal.value if signal else 'MAINTAIN')
        
        # Record current state
        params = engine.get_current_execution_parameters()
        participation_rates.append(params['participation_rate'])
        
        # Get market conditions for analysis
        if engine.regime_detector.price_history:
            latest_conditions = engine.regime_detector._calculate_conditions(
                base_price, current_volume, datetime.now()
            )
            market_regimes.append(latest_conditions.market_regime.value)
            stress_levels.append(latest_conditions.stress_level)
        else:
            market_regimes.append('NORMAL')
            stress_levels.append(0.0)
    
    # Analysis and reporting
    print(f"\nSimulation completed. Analyzing results...")
    
    # Get adaptation statistics
    stats = engine.get_adaptation_statistics()
    print(f"\nAdaptation Statistics:")
    print(f"  Total Adaptations: {stats['total_adaptations']}")
    print(f"  Adaptation Frequency: {stats['adaptation_frequency_per_hour']:.1f} per hour")
    
    if 'signal_distribution' in stats:
        print(f"  Signal Distribution:")
        for signal, count in stats['signal_distribution'].items():
            print(f"    {signal}: {count}")
    
    if 'trigger_distribution' in stats:
        print(f"  Trigger Distribution:")
        for trigger, count in stats['trigger_distribution'].items():
            print(f"    {trigger}: {count}")
    
    # Generate adaptation report
    report = engine.generate_adaptation_report()
    
    print(f"\nAdaptation Report Summary:")
    if 'summary' in report:
        summary = report['summary']
        print(f"  Recent Adaptations: {summary['recent_period_adaptations']}")
        print(f"  Average Interval: {summary['average_adaptation_interval_seconds']:.0f} seconds")
        if summary['most_common_signal']:
            print(f"  Most Common Signal: {summary['most_common_signal']}")
    
    if 'signal_effectiveness' in report and report['signal_effectiveness']:
        print(f"  Signal Effectiveness:")
        for signal, effectiveness in report['signal_effectiveness'].items():
            print(f"    {signal}: {effectiveness:.3f}")
    
    # Show final execution parameters
    final_params = engine.get_current_execution_parameters()
    print(f"\nFinal Execution Parameters:")
    print(f"  Participation Rate: {final_params['participation_rate']:.1%}")
    print(f"  Slice Interval: {final_params['slice_interval_minutes']:.1f} minutes")
    print(f"  Acceleration Factor: {final_params['acceleration_factor']:.2f}")
    print(f"  Aggression Level: {final_params['aggression_level']:.1%}")
    print(f"  Is Paused: {final_params['is_paused']}")
    
    # Simple visualization (if matplotlib available)
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        time_axis = list(range(len(prices)))
        
        # Price chart
        ax1.plot(time_axis, prices, 'b-', label='Price')
        ax1.set_title('Price Movement')
        ax1.set_ylabel('Price')
        ax1.grid(True)
        
        # Participation rate
        ax2.plot(time_axis, participation_rates, 'g-', label='Participation Rate')
        ax2.set_title('Adaptive Participation Rate')
        ax2.set_ylabel('Participation Rate')
        ax2.grid(True)
        
        # Stress level
        ax3.plot(time_axis, stress_levels, 'r-', label='Market Stress')
        ax3.set_title('Market Stress Level')
        ax3.set_ylabel('Stress Level')
        ax3.set_xlabel('Time (minutes)')
        ax3.grid(True)
        
        # Signal frequency
        signal_changes = [1 if s != 'MAINTAIN' else 0 for s in signals]
        ax4.plot(time_axis, np.cumsum(signal_changes), 'm-', label='Cumulative Adaptations')
        ax4.set_title('Cumulative Adaptations')
        ax4.set_ylabel('Adaptation Count')
        ax4.set_xlabel('Time (minutes)')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig('adaptive_execution_test.png', dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved as 'adaptive_execution_test.png'")
        
    except ImportError:
        print(f"\nMatplotlib not available for visualization")
    
    engine.stop()
    print("\nAdaptive execution engine testing completed!")