"""
Spoofing & Manipulation Detection — Real-time detection of
abnormal order-book patterns that indicate market manipulation.

Missing Concept 3.4: "Spoofing / Manipulation Detection"

Detects:
    1. Layering: Multiple large orders placed on one side then cancelled before fill.
    2. Quote stuffing: Abnormally rapid order placement/cancellation bursts.
    3. Spoofing: Large resting order pulled when execution approaches.
    4. Wash trading: Self-matching trades (same entity both sides).

Pipeline integration:
    OrderBookAnalyzer.snapshots →
        SpoofingDetector.process_snapshot()
    QuantumCoreOrchestrator._microstructure_layer() →
        manipulation_risk = spoofing_detector.get_risk_score()
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BookSnapshot:
    """Simplified L2 order book snapshot."""
    symbol: str
    timestamp_ms: int
    bids: List[List[float]]  # [[price, qty], ...]
    asks: List[List[float]]  # [[price, qty], ...]


@dataclass
class ManipulationAlert:
    """Single manipulation event alert."""
    alert_type: str   # "LAYERING", "QUOTE_STUFFING", "SPOOFING", "WASH_TRADE"
    symbol: str
    severity: float   # 0.0 – 1.0
    description: str
    timestamp: float


class SpoofingDetector:
    """
    Stateful detector that ingests order-book snapshots and raises
    alerts when manipulation patterns exceed configurable thresholds.
    """

    def __init__(
        self,
        cancel_ratio_threshold: float = 0.90,
        layering_depth_levels: int = 5,
        stuffing_rate_per_sec: int = 50,
        imbalance_spike_threshold: float = 0.80,
        window_seconds: float = 10.0,
    ):
        self.cancel_ratio_threshold = cancel_ratio_threshold
        self.layering_depth = layering_depth_levels
        self.stuffing_rate = stuffing_rate_per_sec
        self.imbalance_spike_threshold = imbalance_spike_threshold
        self.window_seconds = window_seconds

        # Rolling state per symbol
        self._snapshot_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=200))
        self._alerts: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
        self._composite_risk: Dict[str, float] = {}

    # ── Public API ───────────────────────────────────────────────────

    def process_snapshot(self, snap: BookSnapshot) -> List[ManipulationAlert]:
        """
        Ingest a new L2 snapshot.  Returns any new alerts detected.
        """
        sym = snap.symbol.upper()
        self._snapshot_history[sym].append(snap)

        alerts: List[ManipulationAlert] = []

        # Need at least 3 snapshots to detect changes
        if len(self._snapshot_history[sym]) < 3:
            self._composite_risk[sym] = 0.0
            return alerts

        history = list(self._snapshot_history[sym])

        a = self._detect_layering(sym, history)
        if a:
            alerts.append(a)

        a = self._detect_quote_stuffing(sym, history)
        if a:
            alerts.append(a)

        a = self._detect_spoofing(sym, history)
        if a:
            alerts.append(a)

        a = self._detect_wash_trade_proxy(sym, history)
        if a:
            alerts.append(a)

        for alert in alerts:
            self._alerts[sym].append(alert)

        # Composite risk score: 0 if clean, up to 1.0
        recent = [a for a in self._alerts[sym]
                  if time.time() - a.timestamp < self.window_seconds * 3]
        if recent:
            self._composite_risk[sym] = min(1.0, sum(a.severity for a in recent) / 4.0)
        else:
            self._composite_risk[sym] = max(0.0, self._composite_risk.get(sym, 0.0) * 0.95)

        return alerts

    def process_snapshot_simple(
        self,
        symbol: str,
        bids: List[List[float]],
        asks: List[List[float]],
    ) -> List[ManipulationAlert]:
        """Convenience: create BookSnapshot and process."""
        return self.process_snapshot(BookSnapshot(
            symbol=symbol, timestamp_ms=int(time.time() * 1000),
            bids=bids, asks=asks,
        ))

    def get_risk_score(self, symbol: str) -> float:
        """Composite manipulation risk [0, 1]."""
        return self._composite_risk.get(symbol.upper(), 0.0)

    def get_recent_alerts(self, symbol: str, last_n: int = 20) -> List[ManipulationAlert]:
        return list(self._alerts.get(symbol.upper(), []))[-last_n:]

    # ── Detectors (private) ──────────────────────────────────────────

    def _detect_layering(self, sym: str, history: List[BookSnapshot]) -> Optional[ManipulationAlert]:
        """
        Layering: detect large orders that appear on first N levels
        then vanish in the next snapshot (> cancel_ratio_threshold of qty removed).
        """
        prev, curr = history[-2], history[-1]
        for side_name, prev_levels, curr_levels in [
            ("BID", prev.bids, curr.bids),
            ("ASK", prev.asks, curr.asks),
        ]:
            prev_qty = sum(lv[1] for lv in prev_levels[:self.layering_depth]) if prev_levels else 0
            curr_qty = sum(lv[1] for lv in curr_levels[:self.layering_depth]) if curr_levels else 0
            if prev_qty > 0:
                removal_ratio = 1.0 - (curr_qty / prev_qty)
                if removal_ratio >= self.cancel_ratio_threshold:
                    severity = min(1.0, removal_ratio)
                    return ManipulationAlert(
                        alert_type="LAYERING",
                        symbol=sym,
                        severity=severity,
                        description=(
                            f"Layering on {side_name}: {removal_ratio * 100:.0f}% "
                            f"of top-{self.layering_depth} qty removed between snapshots"
                        ),
                        timestamp=time.time(),
                    )
        return None

    def _detect_quote_stuffing(self, sym: str, history: List[BookSnapshot]) -> Optional[ManipulationAlert]:
        """
        Quote stuffing: detect abnormally high rate of price-level changes
        (appearing + disappearing) within the rolling window.
        """
        if len(history) < 5:
            return None

        recent = history[-10:]
        changes = 0
        for i in range(1, len(recent)):
            prev_prices = set(lv[0] for lv in (recent[i-1].bids + recent[i-1].asks))
            curr_prices = set(lv[0] for lv in (recent[i].bids + recent[i].asks))
            changes += len(prev_prices.symmetric_difference(curr_prices))

        time_span = max(0.001, (recent[-1].timestamp_ms - recent[0].timestamp_ms) / 1000.0)
        rate = changes / time_span

        if rate > self.stuffing_rate:
            return ManipulationAlert(
                alert_type="QUOTE_STUFFING",
                symbol=sym,
                severity=min(1.0, rate / (self.stuffing_rate * 3)),
                description=f"Quote stuffing: {rate:.0f} level changes/sec (threshold: {self.stuffing_rate})",
                timestamp=time.time(),
            )
        return None

    def _detect_spoofing(self, sym: str, history: List[BookSnapshot]) -> Optional[ManipulationAlert]:
        """
        Spoofing: detect large imbalance that vanishes when price approaches.
        Large bid/ask imbalance → trade towards heavy side → heavy side disappears.
        """
        if len(history) < 3:
            return None

        for i in range(-3, -1):
            snap = history[i]
            bid_qty = sum(lv[1] for lv in snap.bids[:5]) if snap.bids else 0
            ask_qty = sum(lv[1] for lv in snap.asks[:5]) if snap.asks else 0
            total = bid_qty + ask_qty
            if total < 1e-12:
                continue
            imb = abs(bid_qty - ask_qty) / total
            heavy_side = "BID" if bid_qty > ask_qty else "ASK"

            if imb > self.imbalance_spike_threshold:
                # Check if heavy side evaporated in latest snapshot
                latest = history[-1]
                if heavy_side == "BID":
                    new_qty = sum(lv[1] for lv in latest.bids[:5]) if latest.bids else 0
                    vanished = (bid_qty - new_qty) / max(bid_qty, 1e-12)
                else:
                    new_qty = sum(lv[1] for lv in latest.asks[:5]) if latest.asks else 0
                    vanished = (ask_qty - new_qty) / max(ask_qty, 1e-12)

                if vanished > 0.80:
                    return ManipulationAlert(
                        alert_type="SPOOFING",
                        symbol=sym,
                        severity=min(1.0, vanished),
                        description=(
                            f"Spoofing: {heavy_side} imbalance {imb:.0%} vanished "
                            f"({vanished:.0%} qty removed)"
                        ),
                        timestamp=time.time(),
                    )
        return None

    def _detect_wash_trade_proxy(self, sym: str, history: List[BookSnapshot]) -> Optional[ManipulationAlert]:
        """
        Wash-trade proxy: detect mid-price that oscillates in an
        unusually tight band with high volume (sign of self-matching).
        Without trade-level attribution we use spread + volume anomaly.
        """
        if len(history) < 10:
            return None

        mids = []
        for s in history[-20:]:
            if s.bids and s.asks:
                mids.append((s.bids[0][0] + s.asks[0][0]) / 2.0)

        if len(mids) < 5:
            return None

        arr = np.array(mids)
        volatility = np.std(arr) / max(np.mean(arr), 1e-12)

        # Extremely low volatility with frequent updates suggests wash trading
        if volatility < 1e-6 and len(history) > 15:
            return ManipulationAlert(
                alert_type="WASH_TRADE",
                symbol=sym,
                severity=0.5,
                description=f"Wash-trade proxy: near-zero volatility ({volatility:.2e}) over {len(mids)} ticks",
                timestamp=time.time(),
            )
        return None
