"""
Market Impact Measurement — Post-trade analysis of how orders moved
the market, with feedback to the execution manager.

Missing Concept 3.3: "Market Impact Measurement"

Measures:
    1. Temporary impact: price deviation during execution vs pre-trade mid.
    2. Permanent impact: price level after execution settles (T+5 bars).
    3. Realised vs expected slippage.
    4. Implementation shortfall decomposition (timing, market, execution).

Pipeline integration:
    ExecutionManager.on_fill(fill) →
        impact_tracker.record_fill(fill)
    QuantumCoreOrchestrator._post_trade() →
        impact_tracker.compute_impact(trade_id)
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FillRecord:
    """Single execution fill."""
    trade_id: str
    symbol: str
    side: str             # "BUY" or "SELL"
    size: float
    fill_price: float
    mid_price_at_decision: float
    mid_price_at_fill: float
    timestamp_decision_ms: int
    timestamp_fill_ms: int
    algo: str = "MARKET"


@dataclass
class ImpactReport:
    """Post-trade market impact analysis for one trade."""
    trade_id: str
    symbol: str
    side: str

    # basis points
    temporary_impact_bps: float
    permanent_impact_bps: float
    realised_slippage_bps: float
    expected_slippage_bps: float

    # Implementation shortfall components (bps)
    timing_cost_bps: float
    market_cost_bps: float
    execution_cost_bps: float
    total_is_bps: float

    execution_time_ms: int
    algo: str
    timestamp: str


class MarketImpactTracker:
    """
    Records execution fills and post-trade price data to measure
    market impact and implementation shortfall.
    """

    def __init__(
        self,
        settle_bars: int = 5,
        expected_slippage_bps: float = 2.0,
    ):
        self.settle_bars = settle_bars
        self.expected_slippage = expected_slippage_bps

        # Pending fills waiting for settlement prices
        self._pending: Dict[str, FillRecord] = {}
        self._post_fill_prices: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))

        # Completed impact reports
        self._reports: Dict[str, ImpactReport] = {}
        self._symbol_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=200))

    # ── Record fills ─────────────────────────────────────────────────

    def record_fill(self, fill: FillRecord) -> None:
        """Record a new execution fill for impact tracking."""
        self._pending[fill.trade_id] = fill
        self._post_fill_prices[fill.trade_id] = deque(maxlen=self.settle_bars + 5)
        logger.debug("Impact tracker: recording fill %s", fill.trade_id)

    def record_fill_simple(
        self,
        trade_id: str,
        symbol: str,
        side: str,
        size: float,
        fill_price: float,
        mid_at_decision: float,
        mid_at_fill: float,
        algo: str = "MARKET",
    ) -> None:
        """Convenience wrapper."""
        self.record_fill(FillRecord(
            trade_id=trade_id, symbol=symbol, side=side, size=size,
            fill_price=fill_price, mid_price_at_decision=mid_at_decision,
            mid_price_at_fill=mid_at_fill,
            timestamp_decision_ms=int(time.time() * 1000),
            timestamp_fill_ms=int(time.time() * 1000),
            algo=algo,
        ))

    # ── Price feed for settlement tracking ───────────────────────────

    def on_price_update(self, symbol: str, mid_price: float) -> None:
        """
        Feed post-trade prices so we can compute permanent impact once
        enough bars have settled.
        """
        settled_ids = []
        for tid, fill in list(self._pending.items()):
            if fill.symbol.upper() == symbol.upper():
                self._post_fill_prices[tid].append(mid_price)

                if len(self._post_fill_prices[tid]) >= self.settle_bars:
                    report = self._compute_impact(fill, list(self._post_fill_prices[tid]))
                    self._reports[tid] = report
                    self._symbol_history[fill.symbol].append(report)
                    settled_ids.append(tid)

        for tid in settled_ids:
            del self._pending[tid]
            del self._post_fill_prices[tid]

    # ── Queries ──────────────────────────────────────────────────────

    def get_report(self, trade_id: str) -> Optional[ImpactReport]:
        return self._reports.get(trade_id)

    def get_avg_impact(self, symbol: str, last_n: int = 50) -> Dict[str, float]:
        """Average impact metrics over recent trades for *symbol*."""
        history = list(self._symbol_history.get(symbol, []))[-last_n:]
        if not history:
            return {"temporary_bps": 0, "permanent_bps": 0, "slippage_bps": 0, "is_bps": 0}
        return {
            "temporary_bps": float(np.mean([r.temporary_impact_bps for r in history])),
            "permanent_bps": float(np.mean([r.permanent_impact_bps for r in history])),
            "slippage_bps": float(np.mean([r.realised_slippage_bps for r in history])),
            "is_bps": float(np.mean([r.total_is_bps for r in history])),
        }

    def pending_count(self) -> int:
        return len(self._pending)

    # ── Computation ──────────────────────────────────────────────────

    def _compute_impact(self, fill: FillRecord, post_prices: List[float]) -> ImpactReport:
        mid_decision = fill.mid_price_at_decision
        mid_fill = fill.mid_price_at_fill
        fill_px = fill.fill_price
        settle_px = float(np.mean(post_prices[-self.settle_bars:]))
        sign = 1.0 if fill.side.upper() == "BUY" else -1.0

        # All in bps relative to decision mid
        def bps(px_a: float, px_b: float) -> float:
            return sign * (px_a - px_b) / max(abs(mid_decision), 1e-12) * 10_000

        # Temporary impact: fill price vs decision mid
        temp_impact = bps(fill_px, mid_decision)

        # Permanent impact: settlement price vs decision mid
        perm_impact = bps(settle_px, mid_decision)

        # Realised slippage: fill vs mid at fill
        slippage = bps(fill_px, mid_fill)

        # Implementation shortfall decomposition
        timing_cost = bps(mid_fill, mid_decision)        # market moved before we executed
        market_cost = bps(settle_px, mid_fill)            # market moved after execution
        execution_cost = bps(fill_px, mid_fill)           # our execution vs mid at fill
        total_is = timing_cost + execution_cost           # IS = timing + execution

        exec_ms = fill.timestamp_fill_ms - fill.timestamp_decision_ms

        return ImpactReport(
            trade_id=fill.trade_id,
            symbol=fill.symbol,
            side=fill.side,
            temporary_impact_bps=float(temp_impact),
            permanent_impact_bps=float(perm_impact),
            realised_slippage_bps=float(slippage),
            expected_slippage_bps=self.expected_slippage,
            timing_cost_bps=float(timing_cost),
            market_cost_bps=float(market_cost),
            execution_cost_bps=float(execution_cost),
            total_is_bps=float(total_is),
            execution_time_ms=int(exec_ms),
            algo=fill.algo,
            timestamp=datetime.utcnow().isoformat(),
        )
