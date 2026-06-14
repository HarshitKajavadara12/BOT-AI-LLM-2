"""
QUANTUM-FORGE: Quantum Signal Strategy
=========================================
Wraps the SignalGenerator + MLEnsemble pipeline as an IStrategy
so it can be registered with the StrategyMultiplexer.

This allows running the existing quantum pipeline alongside other
strategies (e.g., MomentumStrategy) through the multiplexer,
with shadow tracking for strategies under evaluation.
"""

import logging
from typing import Dict, Any, List
from collections import deque
from datetime import datetime

from core.strategy_interface import IStrategy, StrategySignal, StrategyState
from core.signal_generator import SignalGenerator
from core.ml_ensemble import MLEnsembleEngine

logger = logging.getLogger("QuantumStrategy")


class QuantumSignalStrategy(IStrategy):
    """
    Wraps the full SignalGenerator + MLEnsemble pipeline as an IStrategy.
    
    This is the primary strategy in the Quantum-Forge system.
    It uses 7 mathematical signal sources + 20-model ML ensemble
    to generate BUY/SELL/HOLD signals.
    """
    
    def __init__(
        self,
        strategy_id: str = "quantum_core_v1",
        signal_threshold: float = 0.25,
        ml_weight: float = 0.3,
        math_weight: float = 0.7,
        enable_ml: bool = True,
    ):
        self._id = strategy_id
        self.signal_threshold = signal_threshold
        self.ml_weight = ml_weight
        self.math_weight = math_weight
        
        # Capital
        self.capital = 0.0
        self.active = True
        
        # Signal Generator (7 math sources)
        self.signal_generator = SignalGenerator(
            min_history=30,
            signal_threshold=signal_threshold,
        )
        
        # ML Ensemble (20+ models)
        if enable_ml:
            self.ml_ensemble = MLEnsembleEngine(feature_dim=20, enable_training=False)
        else:
            self.ml_ensemble = None
        
        # State tracking
        self.last_signal = "HOLD"
        self._trade_count = 0
        self._win_count = 0
        self._position_count = 0
        self._exposure = 0.0
        self._max_drawdown = 0.0
        self._peak_capital = 0.0
        self._price_history: deque = deque(maxlen=200)
        
        logger.info(f"QuantumSignalStrategy '{strategy_id}' initialized")
    
    @property
    def strategy_id(self) -> str:
        return self._id
    
    def set_capital_allocation(self, amount: float):
        self.capital = amount
        if amount > self._peak_capital:
            self._peak_capital = amount
    
    def on_market_data(self, data: Dict[str, Any]) -> List[StrategySignal]:
        """
        Process market data through the full quantum pipeline.
        
        Expected data format:
            {
                'symbol': 'BTCUSDT',
                'price' or 'close': 50000.0,
                'volume': 1234.5,
                'timestamp': datetime,
                'open': ..., 'high': ..., 'low': ...,
            }
        """
        if not self.active or self.capital <= 0:
            return []
        
        price = float(data.get('close', data.get('price', 0.0)))
        symbol = data.get('symbol', 'UNKNOWN')
        timestamp = data.get('timestamp', datetime.now())
        volume = float(data.get('volume', 0.0))
        
        if price <= 0:
            return []
        
        self._price_history.append(price)
        
        # ===  Math Signal ===
        math_signal = self.signal_generator.generate_signal(symbol, price, volume)
        
        if math_signal is None:
            return []
        
        math_direction = 1.0 if math_signal.signal_type.value == "BUY" else (
            -1.0 if math_signal.signal_type.value == "SELL" else 0.0
        )
        math_score = math_direction * math_signal.strength
        
        # === ML Signal ===
        ml_score = 0.0
        if self.ml_ensemble and len(self._price_history) >= 20:
            prices_array = list(self._price_history)
            ml_prediction = self.ml_ensemble.predict(prices_array, volume)
            if ml_prediction:
                ml_score = ml_prediction.consensus
        
        # === Fuse signals ===
        fused = self.math_weight * math_score + self.ml_weight * ml_score
        strength = abs(fused)
        
        if fused > 0.2:
            signal_type = "BUY"
        elif fused < -0.2:
            signal_type = "SELL"
        else:
            signal_type = "HOLD"
        
        self.last_signal = signal_type
        
        if signal_type == "HOLD":
            return []
        
        signal = StrategySignal(
            strategy_id=self._id,
            symbol=symbol,
            signal_type=signal_type,
            signal_strength=min(strength, 1.0),
            timestamp=timestamp if isinstance(timestamp, datetime) else datetime.now(),
            metadata={
                'math_signal': math_signal.signal_type.value,
                'math_strength': math_signal.strength,
                'math_sources': math_signal.sources,
                'ml_consensus': ml_score,
                'fused_score': fused,
            },
        )
        
        return [signal]
    
    def get_state(self) -> StrategyState:
        """Report strategy health/state."""
        # Calculate drawdown
        if self._peak_capital > 0:
            drawdown = (self._peak_capital - self.capital) / self._peak_capital
        else:
            drawdown = 0.0
        
        return StrategyState(
            strategy_id=self._id,
            is_active=self.active,
            current_drawdown=drawdown,
            current_exposure=self._exposure,
            open_positions=self._position_count,
            last_update=datetime.now(),
        )
