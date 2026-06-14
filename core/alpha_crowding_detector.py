"""
Alpha Crowding Detector — Monitors whether the system's alpha signals
are being crowded out by other market participants.

Missing Concept 2.6: "Alpha Crowding Detection"

Detection methods:
    1. Volume-weighted alpha decay: if signal was profitable but recent
       volume surged while edge decayed → crowding.
    2. Correlation with market-wide flow: if our signal direction matches
       aggregate order flow more than expected → crowded trade.
    3. Slippage increase: rising execution slippage indicates crowding.
    4. Return autocorrelation collapse: crowded alphas lose predictability.

Pipeline integration:
    QuantumCoreOrchestrator._post_trade_analysis() →
        crowding_detector.update(symbol, signal, fill, …)
        crowding_detector.is_crowded(symbol) → bool
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CrowdingMetrics:
    """Per-symbol crowding diagnostics."""
    symbol: str
    crowding_score: float       # [0, 1] — 0 = uncrowded, 1 = fully crowded
    is_crowded: bool
    avg_slippage_bps: float
    signal_return_corr: float   # autocorrelation of signal returns
    volume_ratio: float         # recent / baseline volume
    edge_decay_rate: float      # slope of rolling alpha profitability
    n_observations: int


class AlphaCrowdingDetector:
    """
    Tracks multiple indicators of alpha crowding and raises flags when
    the system's edge is likely being arbitraged away.
    """

    def __init__(
        self,
        symbols: List[str],
        window: int = 200,
        crowding_threshold: float = 0.6,
        slippage_baseline_bps: float = 2.0,
        volume_surge_ratio: float = 2.0,
    ):
        self.symbols = symbols
        self.window = window
        self.threshold = crowding_threshold
        self.slippage_baseline = slippage_baseline_bps
        self.volume_surge = volume_surge_ratio

        # Per-symbol rolling buffers
        self._signals: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self._returns: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self._slippages: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self._volumes: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self._baseline_vol: Dict[str, float] = defaultdict(float)
        self._metrics: Dict[str, CrowdingMetrics] = {}

    # ── Public API ───────────────────────────────────────────────────

    def update(
        self,
        symbol: str,
        signal: float,
        realised_return: float,
        slippage_bps: float = 0.0,
        volume: float = 0.0,
    ) -> CrowdingMetrics:
        """
        Feed one observation. Call after each trade or bar.

        Args:
            signal: The alpha signal that was acted on.
            realised_return: Actual return achieved.
            slippage_bps: Execution slippage in basis points.
            volume: Bar volume.
        """
        self._signals[symbol].append(signal)
        self._returns[symbol].append(realised_return)
        self._slippages[symbol].append(slippage_bps)
        self._volumes[symbol].append(volume)

        # Maintain exponential baseline volume
        if self._baseline_vol[symbol] == 0:
            self._baseline_vol[symbol] = volume if volume > 0 else 1.0
        else:
            self._baseline_vol[symbol] = (
                0.99 * self._baseline_vol[symbol] + 0.01 * volume
            )

        return self._compute(symbol)

    def is_crowded(self, symbol: str) -> bool:
        m = self._metrics.get(symbol)
        return m.is_crowded if m else False

    def get_metrics(self, symbol: str) -> CrowdingMetrics:
        return self._metrics.get(
            symbol,
            CrowdingMetrics(symbol=symbol, crowding_score=0.0, is_crowded=False,
                            avg_slippage_bps=0.0, signal_return_corr=0.0,
                            volume_ratio=1.0, edge_decay_rate=0.0, n_observations=0),
        )

    def get_all_metrics(self) -> Dict[str, CrowdingMetrics]:
        return dict(self._metrics)

    # ── Internals ────────────────────────────────────────────────────

    def _compute(self, sym: str) -> CrowdingMetrics:
        sigs = np.array(self._signals[sym])
        rets = np.array(self._returns[sym])
        slips = np.array(self._slippages[sym])
        vols = np.array(self._volumes[sym])
        n = len(sigs)

        # 1) Signal-return correlation (should be positive if alpha works)
        if n >= 20:
            corr = float(np.corrcoef(sigs[-50:], rets[-50:])[0, 1])
            if np.isnan(corr):
                corr = 0.0
        else:
            corr = 0.0

        # 2) Average slippage
        avg_slip = float(np.mean(slips[-50:])) if n >= 5 else 0.0

        # 3) Volume surge ratio
        baseline = max(self._baseline_vol[sym], 1e-12)
        recent_vol = float(np.mean(vols[-20:])) if n >= 20 else baseline
        vol_ratio = recent_vol / baseline

        # 4) Edge decay: slope of rolling signal * return product
        if n >= 40:
            profit_curve = sigs[-40:] * rets[-40:]
            x = np.arange(len(profit_curve))
            try:
                slope = float(np.polyfit(x, profit_curve, 1)[0])
            except Exception:
                slope = 0.0
        else:
            slope = 0.0

        # ── Composite crowding score ─────────────────────────────────
        # Low correlation → alpha lost → crowded
        corr_component = max(0, 1.0 - max(corr, 0) / 0.3)  # 0 when corr >= 0.3

        # High slippage → crowded
        slip_component = min(avg_slip / max(self.slippage_baseline * 3, 1), 1.0)

        # Volume surge → crowded
        vol_component = min(max(vol_ratio / self.volume_surge - 1, 0), 1.0)

        # Negative slope → edge decaying → crowded
        decay_component = min(max(-slope * 1000, 0), 1.0)

        crowding = float(np.clip(
            0.35 * corr_component +
            0.25 * slip_component +
            0.20 * vol_component +
            0.20 * decay_component,
            0, 1
        ))

        m = CrowdingMetrics(
            symbol=sym,
            crowding_score=crowding,
            is_crowded=crowding >= self.threshold,
            avg_slippage_bps=avg_slip,
            signal_return_corr=corr,
            volume_ratio=vol_ratio,
            edge_decay_rate=slope,
            n_observations=n,
        )
        self._metrics[sym] = m

        if m.is_crowded:
            logger.warning(
                "CROWDING ALERT %s: score=%.2f corr=%.3f slip=%.1fbps vol_ratio=%.1f",
                sym, crowding, corr, avg_slip, vol_ratio,
            )

        return m
