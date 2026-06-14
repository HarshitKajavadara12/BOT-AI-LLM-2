"""
QUANTUM-FORGE: Cross-Asset Alpha Engine
=========================================
Generates cross-asset signals — uses correlations and lead-lag
relationships between the 7 crypto pairs instead of treating them
independently.

Fixes Missing Concept 2.4 (Cross-Asset Alphas).

Key signals:
1. BTC-leading: BTC moves often lead ALT moves by 1-5 minutes
2. ETH-BTC ratio: ETH/BTC trend signals risk-on/risk-off
3. Correlation breakdown: When usual correlations break, it signals regime shift
4. Cross-momentum: Strongest movers predict sector rotation
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger("CrossAssetAlpha")


@dataclass
class CrossAssetSignal:
    """Signal derived from cross-asset relationships."""
    symbol: str
    signal: float  # -1 to +1
    source: str
    confidence: float
    lead_symbol: str = ""


class CrossAssetAlphaEngine:
    """
    Generates alpha signals from cross-asset relationships.
    
    The 7 crypto pairs are correlated — this engine exploits:
    1. Lead-lag effects (BTC leads ALTs)
    2. Correlation regime shifts
    3. Relative strength rotation
    4. Pair spread mean-reversion
    """
    
    BTC = "BTCUSDT"
    ETH = "ETHUSDT"
    
    def __init__(
        self,
        symbols: List[str],
        lookback: int = 60,
        lead_lag_window: int = 10,
    ):
        self.symbols = symbols
        self.lookback = lookback
        self.lead_lag_window = lead_lag_window
        
        # Price history per symbol
        self.prices: Dict[str, deque] = {s: deque(maxlen=500) for s in symbols}
        self.returns: Dict[str, deque] = {s: deque(maxlen=500) for s in symbols}
        
        # Correlation state
        self._corr_matrix: Optional[np.ndarray] = None
        self._baseline_corr: Optional[np.ndarray] = None
    
    def update(self, symbol: str, price: float):
        """Feed new price for a symbol."""
        self.prices[symbol].append(price)
        if len(self.prices[symbol]) >= 2:
            prev = self.prices[symbol][-2]
            ret = (price - prev) / prev
            self.returns[symbol].append(ret)
    
    def generate_signals(self) -> List[CrossAssetSignal]:
        """Generate all cross-asset signals."""
        signals = []
        
        signals.extend(self._btc_leading_signals())
        signals.extend(self._correlation_breakdown_signals())
        signals.extend(self._relative_strength_signals())
        signals.extend(self._pair_spread_signals())
        
        return signals
    
    def get_signal_for_symbol(self, symbol: str) -> float:
        """Get aggregated cross-asset signal for a specific symbol."""
        signals = [s for s in self.generate_signals() if s.symbol == symbol]
        if not signals:
            return 0.0
        
        # Confidence-weighted average
        total_weight = sum(s.confidence for s in signals)
        if total_weight < 1e-10:
            return 0.0
        
        weighted = sum(s.signal * s.confidence for s in signals) / total_weight
        return float(np.clip(weighted, -1, 1))
    
    def _btc_leading_signals(self) -> List[CrossAssetSignal]:
        """
        BTC-leading alpha: BTC moves often lead ALT moves by a few ticks.
        If BTC just moved up strongly, ALTs may follow.
        """
        signals = []
        btc_returns = list(self.returns.get(self.BTC, []))
        
        if len(btc_returns) < self.lead_lag_window:
            return signals
        
        # Recent BTC momentum
        btc_momentum = np.mean(btc_returns[-self.lead_lag_window:])
        btc_strength = abs(btc_momentum) / (np.std(btc_returns[-30:]) + 1e-10)
        
        for symbol in self.symbols:
            if symbol == self.BTC:
                continue
            
            sym_returns = list(self.returns.get(symbol, []))
            if len(sym_returns) < self.lead_lag_window:
                continue
            
            # Check if this ALT typically follows BTC
            min_len = min(len(btc_returns), len(sym_returns))
            if min_len < 20:
                continue
            
            # Lead-lag correlation (BTC t-1 vs ALT t)
            btc_lagged = np.array(btc_returns[-min_len:-1])
            alt_current = np.array(sym_returns[-min_len + 1:])
            
            if len(btc_lagged) < 10:
                continue
            
            corr = np.corrcoef(btc_lagged, alt_current)[0, 1]
            
            if abs(corr) > 0.3:  # Significant lead-lag
                signal = float(np.clip(btc_momentum * corr * 10, -1, 1))
                signals.append(CrossAssetSignal(
                    symbol=symbol,
                    signal=signal,
                    source="btc_leading",
                    confidence=min(abs(corr), 1.0) * min(btc_strength, 1.0),
                    lead_symbol=self.BTC,
                ))
        
        return signals
    
    def _correlation_breakdown_signals(self) -> List[CrossAssetSignal]:
        """
        When cross-asset correlations deviate from baseline,
        it signals a regime shift.
        """
        signals = []
        
        # Need enough data
        sym_list = [s for s in self.symbols if len(self.returns.get(s, [])) >= 30]
        if len(sym_list) < 3:
            return signals
        
        # Build returns matrix
        min_len = min(len(self.returns[s]) for s in sym_list)
        if min_len < 30:
            return signals
        
        returns_matrix = np.array([list(self.returns[s])[-min_len:] for s in sym_list])
        
        # Current correlation
        current_corr = np.corrcoef(returns_matrix)
        
        # Baseline (earlier period)
        if min_len >= 60:
            baseline = np.array([list(self.returns[s])[-min_len:-30] for s in sym_list])
            baseline_corr = np.corrcoef(baseline)
        else:
            baseline_corr = current_corr
        
        # Find symbols where correlation broke down
        for i, sym in enumerate(sym_list):
            # Average correlation change for this symbol
            corr_changes = []
            for j in range(len(sym_list)):
                if i != j:
                    change = current_corr[i, j] - baseline_corr[i, j]
                    corr_changes.append(change)
            
            avg_change = np.mean(corr_changes)
            
            if abs(avg_change) > 0.15:  # Significant correlation shift
                # Decorrelation → potential mean reversion opportunity
                signal = float(-np.clip(avg_change * 2, -1, 1))
                signals.append(CrossAssetSignal(
                    symbol=sym,
                    signal=signal,
                    source="correlation_breakdown",
                    confidence=min(abs(avg_change), 1.0),
                ))
        
        return signals
    
    def _relative_strength_signals(self) -> List[CrossAssetSignal]:
        """
        Cross-momentum: rank symbols by recent performance,
        go long strongest, short weakest.
        """
        signals = []
        
        # Compute recent returns per symbol
        perf = {}
        for sym in self.symbols:
            rets = list(self.returns.get(sym, []))
            if len(rets) >= 10:
                perf[sym] = np.sum(rets[-10:])
        
        if len(perf) < 3:
            return signals
        
        # Rank
        sorted_syms = sorted(perf.keys(), key=lambda s: perf[s], reverse=True)
        n = len(sorted_syms)
        
        for i, sym in enumerate(sorted_syms):
            # Normalize rank to [-1, 1]
            rank_signal = 1.0 - 2.0 * (i / (n - 1)) if n > 1 else 0.0
            rank_signal *= 0.5  # Scale down (cross-momentum is one signal among many)
            
            signals.append(CrossAssetSignal(
                symbol=sym,
                signal=float(rank_signal),
                source="relative_strength",
                confidence=0.4,
            ))
        
        return signals
    
    def _pair_spread_signals(self) -> List[CrossAssetSignal]:
        """
        Pair spread mean-reversion: when ETH/BTC ratio deviates
        from its mean, expect reversion.
        """
        signals = []
        
        eth_prices = list(self.prices.get(self.ETH, []))
        btc_prices = list(self.prices.get(self.BTC, []))
        
        if len(eth_prices) < 30 or len(btc_prices) < 30:
            return signals
        
        min_len = min(len(eth_prices), len(btc_prices))
        eth_arr = np.array(eth_prices[-min_len:])
        btc_arr = np.array(btc_prices[-min_len:])
        
        # ETH/BTC ratio
        ratio = eth_arr / btc_arr
        
        if len(ratio) < 20:
            return signals
        
        mean_ratio = np.mean(ratio[-20:])
        std_ratio = np.std(ratio[-20:])
        
        if std_ratio < 1e-10:
            return signals
        
        z_score = (ratio[-1] - mean_ratio) / std_ratio
        
        # If ETH is overvalued vs BTC (z > 1) → sell ETH, buy BTC
        if abs(z_score) > 1.0:
            eth_signal = float(-np.clip(z_score / 3.0, -1, 1))
            btc_signal = float(np.clip(z_score / 3.0, -1, 1))
            
            signals.append(CrossAssetSignal(
                symbol=self.ETH,
                signal=eth_signal,
                source="pair_spread_eth_btc",
                confidence=min(abs(z_score) / 3.0, 1.0),
            ))
            signals.append(CrossAssetSignal(
                symbol=self.BTC,
                signal=btc_signal,
                source="pair_spread_eth_btc",
                confidence=min(abs(z_score) / 3.0, 1.0),
            ))
        
        return signals
