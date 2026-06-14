"""
Cognitive Risk Dampener
Phase 4C Component

This module implements the logic to convert LLM insights (Regime, Confidence)
into a safe risk multiplier (0.0 - 1.0).

CRITICAL INVARIANT:
    The output multiplier must NEVER exceed 1.0.
    This ensures the LLM can only REDUCE risk, never INCREASE it.
"""

from typing import Dict, Optional
from dataclasses import dataclass

@dataclass
class CognitiveRiskConfig:
    """Configuration for risk dampening logic."""
    base_confidence_threshold: float = 0.7
    min_multiplier: float = 0.2  # Never go below 20% size even in bad regimes
    regime_penalties: Dict[str, float] = None

    def __post_init__(self):
        if self.regime_penalties is None:
            self.regime_penalties = {
                "HIGH_VOLATILITY": 0.5,
                "MARKET_CRASH": 0.2,
                "LIQUIDITY_CRISIS": 0.3,
                "UNKNOWN": 0.8,
                "STABLE": 1.0,
                "BULL_TREND": 1.0,
                "BEAR_TREND": 1.0
            }

class CognitiveDampener:
    """Calculates the risk multiplier based on cognitive inputs."""
    
    def __init__(self, config: CognitiveRiskConfig = None):
        self.config = config or CognitiveRiskConfig()
        
    def calculate_multiplier(self, regime: str, confidence: float) -> float:
        """
        Derives the position sizing multiplier.
        
        Logic:
        1. Start with the regime-based penalty.
        2. If LLM confidence is low, apply an additional penalty (uncertainty).
        3. Clamp the result between min_multiplier and 1.0.
        """
        # 1. Regime Penalty
        # Default to UNKNOWN if regime not found
        raw_multiplier = self.config.regime_penalties.get(regime, self.config.regime_penalties["UNKNOWN"])
        
        # 2. Confidence Adjustment
        # If confidence is below threshold, we reduce the multiplier linearly
        if confidence < self.config.base_confidence_threshold:
            # Scale factor: 0.5 confidence -> 0.5/0.7 = ~0.71 factor
            confidence_factor = confidence / self.config.base_confidence_threshold
            raw_multiplier *= confidence_factor
            
        # 3. Safety Clamping
        # Ensure we never exceed 1.0 (No leverage increase)
        # Ensure we never go below min_multiplier (unless 0.0 is explicitly desired, but usually we want some exposure)
        final_multiplier = max(self.config.min_multiplier, min(1.0, raw_multiplier))
        
        return float(final_multiplier)

# Global instance
_dampener = CognitiveDampener()

def get_cognitive_dampener() -> CognitiveDampener:
    return _dampener
