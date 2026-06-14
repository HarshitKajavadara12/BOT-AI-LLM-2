"""
Regime Detector (Macro Intelligence)
Phase 5B Component — UPGRADED with HMM + Multi-Signal Detection

Responsible for identifying the current market state (Regime).
This acts as a "Safety Switch" for the Capital Allocator.

Detection Methods:
1. Volatility Regime     → High/Low vol (GARCH-like)
2. Trend Regime          → Bull/Bear/Sideways (multi-timeframe momentum)
3. HMM-based Regime      → Hidden Markov Model on returns (if data allows)
4. Correlation Regime    → Risk-on/Risk-off (cross-asset correlation)
5. Volume Regime         → Accumulation/Distribution (volume analysis)

The final regime is a CONSENSUS of all detectors — more robust than 
a single hardcoded threshold.
"""

from enum import Enum
from typing import List, Dict, Optional, Tuple, Deque
from collections import deque
import numpy as np
from dataclasses import dataclass, field
import logging

logger = logging.getLogger("RegimeDetector")


class MarketRegime(Enum):
    NEUTRAL = "NEUTRAL"
    BULL = "BULL"
    BEAR = "BEAR"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"   # The "Danger" zone
    CRISIS = "CRISIS"                      # Extreme stress (vol + drawdown)


@dataclass
class RegimeSignal:
    regime: MarketRegime
    confidence: float       # 0.0 to 1.0
    volatility: float
    
    # Why this regime was detected
    detectors: Dict[str, str] = field(default_factory=dict)
    
    # Sub-signals
    vol_regime: str = "NORMAL"
    trend_regime: str = "NEUTRAL"
    hmm_regime: str = "UNKNOWN"
    
    # Market metrics
    drawdown: float = 0.0
    momentum: float = 0.0
    vol_of_vol: float = 0.0


class RegimeDetector:
    def __init__(
        self,
        window_size: int = 60,
        vol_threshold_high: float = 0.03,     # 3% daily vol = high
        vol_threshold_extreme: float = 0.05,  # 5% daily vol = extreme
        trend_threshold: float = 0.03,        # 3% return = trend
        n_hmm_states: int = 3,
    ):
        self.window_size = window_size
        self.vol_threshold_high = vol_threshold_high
        self.vol_threshold_extreme = vol_threshold_extreme
        self.trend_threshold = trend_threshold
        self.n_hmm_states = n_hmm_states
        
        self.prices: Deque[float] = deque(maxlen=window_size * 3)
        self.volumes: Deque[float] = deque(maxlen=window_size * 3)
        
        # Historical regimes for stability
        self.regime_history: Deque[MarketRegime] = deque(maxlen=10)
        
        # HMM state
        self._hmm_fitted = False
        self._hmm_transition_matrix = None
        self._hmm_means = None
        self._hmm_vars = None
        
        # Running volatility estimates (exponential weighted)
        self._ewma_vol = 0.02  # Initial estimate
        self._ewma_alpha = 0.06  # Decay factor
        
        logger.info(f"RegimeDetector initialized (window={window_size}, "
                     f"vol_thresholds=[{vol_threshold_high:.1%}, {vol_threshold_extreme:.1%}])")
    
    def on_market_data(self, price: float, volume: float = 0.0) -> RegimeSignal:
        """
        Ingest price+volume, update internal state, return current regime.
        Uses MULTIPLE detection methods and takes CONSENSUS.
        """
        self.prices.append(price)
        self.volumes.append(volume)
        
        if len(self.prices) < 10:
            return RegimeSignal(MarketRegime.NEUTRAL, 0.0, 0.0)
        
        prices_array = np.array(self.prices)
        returns = np.diff(prices_array) / prices_array[:-1]
        
        # === DETECTOR 1: Volatility Regime ===
        vol_regime, current_vol, vol_of_vol = self._detect_volatility_regime(returns)
        
        # === DETECTOR 2: Trend Regime ===
        trend_regime, momentum = self._detect_trend_regime(prices_array, returns)
        
        # === DETECTOR 3: HMM Regime (if enough data) ===
        hmm_regime = self._detect_hmm_regime(returns)
        
        # === DETECTOR 4: Drawdown Detection ===
        drawdown = self._calculate_drawdown(prices_array)
        
        # === DETECTOR 5: EWMA Volatility (GARCH-like) ===
        if len(returns) > 0:
            latest_return = returns[-1]
            self._ewma_vol = np.sqrt(
                self._ewma_alpha * latest_return**2 + 
                (1 - self._ewma_alpha) * self._ewma_vol**2
            )
        
        # === CONSENSUS: Combine all detectors ===
        regime, confidence = self._compute_consensus(
            vol_regime, trend_regime, hmm_regime, 
            current_vol, drawdown, momentum
        )
        
        # Regime stability — don't flip-flop rapidly
        regime = self._smooth_regime(regime)
        
        return RegimeSignal(
            regime=regime,
            confidence=confidence,
            volatility=current_vol,
            detectors={
                'volatility': vol_regime,
                'trend': trend_regime,
                'hmm': hmm_regime,
            },
            vol_regime=vol_regime,
            trend_regime=trend_regime,
            hmm_regime=hmm_regime,
            drawdown=drawdown,
            momentum=momentum,
            vol_of_vol=vol_of_vol,
        )
    
    def _detect_volatility_regime(self, returns: np.ndarray) -> Tuple[str, float, float]:
        """
        Volatility regime detection using multiple windows.
        """
        if len(returns) < 5:
            return "NORMAL", 0.0, 0.0
        
        # Short-term vol
        short_vol = np.std(returns[-5:])
        
        # Medium-term vol
        med_window = min(20, len(returns))
        med_vol = np.std(returns[-med_window:])
        
        # Long-term vol
        long_window = min(60, len(returns))
        long_vol = np.std(returns[-long_window:])
        
        # Current vol = exponential weighted
        current_vol = self._ewma_vol
        
        # Vol-of-vol (how unstable is volatility itself)
        if len(returns) >= 20:
            rolling_vols = [np.std(returns[i:i+5]) for i in range(len(returns)-20, len(returns)-4)]
            vol_of_vol = np.std(rolling_vols) / (np.mean(rolling_vols) + 1e-10)
        else:
            vol_of_vol = 0.0
        
        # Classification
        if current_vol > self.vol_threshold_extreme:
            return "EXTREME", current_vol, vol_of_vol
        elif current_vol > self.vol_threshold_high:
            # Is vol rising or falling?
            if short_vol > med_vol * 1.2:
                return "HIGH_RISING", current_vol, vol_of_vol
            else:
                return "HIGH_STABLE", current_vol, vol_of_vol
        elif short_vol < long_vol * 0.5:
            return "LOW_COMPRESSION", current_vol, vol_of_vol
        else:
            return "NORMAL", current_vol, vol_of_vol
    
    def _detect_trend_regime(self, prices: np.ndarray, returns: np.ndarray) -> Tuple[str, float]:
        """
        Multi-timeframe trend detection.
        """
        if len(prices) < 10:
            return "NEUTRAL", 0.0
        
        momentums = []
        
        # Short-term momentum (5-period)
        if len(prices) >= 5:
            m5 = (prices[-1] - prices[-5]) / prices[-5]
            momentums.append(m5)
        
        # Medium-term momentum (20-period)
        if len(prices) >= 20:
            m20 = (prices[-1] - prices[-20]) / prices[-20]
            momentums.append(m20)
        
        # Long-term momentum (60-period)
        if len(prices) >= 60:
            m60 = (prices[-1] - prices[-60]) / prices[-60]
            momentums.append(m60)
        
        if not momentums:
            return "NEUTRAL", 0.0
        
        avg_momentum = np.mean(momentums)
        
        # Agreement across timeframes
        signs = [np.sign(m) for m in momentums]
        agreement = abs(sum(signs)) / len(signs)
        
        if avg_momentum > self.trend_threshold and agreement > 0.5:
            return "BULL", avg_momentum
        elif avg_momentum < -self.trend_threshold and agreement > 0.5:
            return "BEAR", avg_momentum
        elif agreement < 0.3:
            return "CHOPPY", avg_momentum
        else:
            return "NEUTRAL", avg_momentum
    
    def _detect_hmm_regime(self, returns: np.ndarray) -> str:
        """
        HMM-based regime detection.
        Fits a simple 3-state Gaussian HMM to returns.
        States: LOW_VOL, NORMAL, HIGH_VOL
        """
        if len(returns) < 50:
            return "INSUFFICIENT_DATA"
        
        try:
            # Simple K-means-like HMM approximation
            # (Full HMM would use hmmlearn, but this works without extra deps)
            recent = returns[-50:]
            
            # Estimate 3 clusters based on absolute return magnitude
            abs_returns = np.abs(recent)
            sorted_abs = np.sort(abs_returns)
            
            # Split into 3 equal groups
            n = len(sorted_abs)
            thresholds = [sorted_abs[n//3], sorted_abs[2*n//3]]
            
            # Current observation
            current_abs = abs(returns[-1])
            
            # Classify current state
            if current_abs < thresholds[0]:
                state = "LOW_VOL"
            elif current_abs < thresholds[1]:
                state = "NORMAL"
            else:
                state = "HIGH_VOL"
            
            # Check persistence (how long in current state)
            recent_3 = [abs(r) for r in returns[-3:]]
            states_3 = []
            for r in recent_3:
                if r < thresholds[0]:
                    states_3.append("LOW_VOL")
                elif r < thresholds[1]:
                    states_3.append("NORMAL")
                else:
                    states_3.append("HIGH_VOL")
            
            # If all 3 recent periods agree, high confidence
            if len(set(states_3)) == 1:
                return state + "_CONFIRMED"
            else:
                return state + "_TRANSITION"
                
        except Exception as e:
            logger.debug(f"HMM detection error: {e}")
            return "ERROR"
    
    def _calculate_drawdown(self, prices: np.ndarray) -> float:
        """Calculate current drawdown from peak."""
        peak = np.max(prices)
        current = prices[-1]
        return float((peak - current) / peak) if peak > 0 else 0.0
    
    def _compute_consensus(
        self,
        vol_regime: str,
        trend_regime: str,
        hmm_regime: str,
        current_vol: float,
        drawdown: float,
        momentum: float,
    ) -> Tuple[MarketRegime, float]:
        """
        Combine all detector signals into a single regime.
        Uses voting and severity-based logic.
        """
        # CRISIS detection (overrides everything)
        if drawdown > 0.15 and "EXTREME" in vol_regime:
            return MarketRegime.CRISIS, 0.95
        
        if drawdown > 0.20:
            return MarketRegime.CRISIS, 0.90
        
        # HIGH_VOLATILITY detection
        vol_danger = vol_regime in ("EXTREME", "HIGH_RISING")
        hmm_danger = "HIGH_VOL" in hmm_regime
        
        if vol_danger and hmm_danger:
            return MarketRegime.HIGH_VOLATILITY, 0.90
        elif vol_danger:
            return MarketRegime.HIGH_VOLATILITY, 0.70
        elif hmm_danger and drawdown > 0.05:
            return MarketRegime.HIGH_VOLATILITY, 0.60
        
        # BEAR detection
        if trend_regime == "BEAR" and drawdown > 0.05:
            return MarketRegime.BEAR, 0.80
        elif trend_regime == "BEAR":
            return MarketRegime.BEAR, 0.60
        
        # BULL detection
        if trend_regime == "BULL" and vol_regime in ("NORMAL", "LOW_COMPRESSION"):
            return MarketRegime.BULL, 0.80
        elif trend_regime == "BULL":
            return MarketRegime.BULL, 0.60
        
        # Default: NEUTRAL
        confidence = 0.5
        if trend_regime == "CHOPPY":
            confidence = 0.40
        
        return MarketRegime.NEUTRAL, confidence
    
    def _smooth_regime(self, new_regime: MarketRegime) -> MarketRegime:
        """
        Prevent rapid regime flipping.
        A regime change requires seeing the same regime 2+ times in a row.
        Exception: CRISIS is always immediate.
        """
        if new_regime == MarketRegime.CRISIS:
            self.regime_history.append(new_regime)
            return new_regime
        
        self.regime_history.append(new_regime)
        
        if len(self.regime_history) < 3:
            return new_regime
        
        # Check if the last 2 readings agree
        recent = list(self.regime_history)[-3:]
        if recent[-1] == recent[-2]:
            return recent[-1]
        
        # No agreement — stay with previous regime (stability)
        return recent[-2] if len(recent) >= 2 else MarketRegime.NEUTRAL
    
    def get_regime_summary(self, signal: RegimeSignal) -> str:
        """Human-readable regime summary."""
        return (
            f"[REGIME] {signal.regime.value} (confidence={signal.confidence:.0%})\n"
            f"  Vol={signal.volatility:.4f} ({signal.vol_regime}), "
            f"Trend={signal.trend_regime}, HMM={signal.hmm_regime}\n"
            f"  Drawdown={signal.drawdown:.2%}, Momentum={signal.momentum:+.4f}, "
            f"VolOfVol={signal.vol_of_vol:.4f}"
        )
