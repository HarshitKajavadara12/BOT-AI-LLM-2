"""
Queue Position Estimation — Estimate our resting limit order's
position in the order-book queue and implied fill probability.

Missing Concept 3.5: "Queue Position Estimation"

Approach:
    1. Track quantity at our price level before and after our order.
    2. Model queue consumption via observed trades (match-engine proxy).
    3. Estimate fill probability using a simple Bernoulli model:
       P(fill) = trades_through_level / qty_ahead_of_us + ε

Pipeline integration:
    on_trade() and on_book_update() events feed into this estimator.
    ExecutionManager queries fill_probability() before deciding whether
    to stay passive or cross the spread aggressively.
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
class RestingOrder:
    """Represents one of our resting limit orders."""
    order_id: str
    symbol: str
    side: str          # "BUY" or "SELL"
    price: float
    size: float
    placed_at_ms: int
    queue_qty_ahead: float = 0.0   # estimated qty ahead of us


@dataclass
class QueueEstimate:
    """Fill probability estimate for a resting order."""
    order_id: str
    symbol: str
    side: str
    price: float
    queue_position_qty: float    # estimated qty ahead
    queue_position_pct: float    # our position as % of full level qty
    fill_probability: float      # 0-1
    estimated_wait_ms: int       # estimated time to fill
    recommendation: str          # "STAY_PASSIVE" | "CROSS_SPREAD" | "CANCEL"


class QueuePositionEstimator:
    """
    Estimates our position in the limit order queue by tracking
    order-book level changes and trade-through events.
    """

    def __init__(
        self,
        cross_threshold: float = 0.15,    # if fill_prob < this → cross the spread
        cancel_threshold: float = 0.05,   # if fill_prob < this → cancel
        decay_rate: float = 0.001,        # per-second fill-prob decay
    ):
        self.cross_threshold = cross_threshold
        self.cancel_threshold = cancel_threshold
        self.decay_rate = decay_rate

        self._orders: Dict[str, RestingOrder] = {}
        self._level_qty_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=200))
        self._trade_through: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))

    # ── Order management ─────────────────────────────────────────────

    def register_order(
        self,
        order_id: str,
        symbol: str,
        side: str,
        price: float,
        size: float,
        total_level_qty: float,
    ) -> None:
        """
        Register a newly placed limit order.
        *total_level_qty* is the visible qty at the price level at placement time.
        We assume FIFO → we are at the back of the queue.
        """
        self._orders[order_id] = RestingOrder(
            order_id=order_id, symbol=symbol.upper(), side=side.upper(),
            price=price, size=size, placed_at_ms=int(time.time() * 1000),
            queue_qty_ahead=max(0.0, total_level_qty - size),
        )
        logger.debug("Queue estimator: registered order %s  ahead=%.4f", order_id, total_level_qty - size)

    def cancel_order(self, order_id: str) -> None:
        self._orders.pop(order_id, None)

    # ── Feed events ──────────────────────────────────────────────────

    def on_book_update(self, symbol: str, price: float, new_qty: float) -> None:
        """
        When the order book level qty changes, update queue estimates.
        If qty at our level decreased → trades consumed queue ahead of us.
        """
        sym = symbol.upper()
        key = f"{sym}:{price}"
        self._level_qty_history[key].append((time.time(), new_qty))

        for oid, order in list(self._orders.items()):
            if order.symbol == sym and abs(order.price - price) < 1e-12:
                history = self._level_qty_history[key]
                if len(history) >= 2:
                    prev_qty = history[-2][1]
                    delta = prev_qty - new_qty
                    if delta > 0:
                        # Queue consumed — move our position forward
                        order.queue_qty_ahead = max(0.0, order.queue_qty_ahead - delta)

    def on_trade(self, symbol: str, price: float, qty: float, side: str) -> None:
        """
        When a trade occurs at or through our price level,
        reduce queue ahead by the traded quantity.
        """
        sym = symbol.upper()
        for oid, order in list(self._orders.items()):
            if order.symbol != sym:
                continue

            is_through = False
            if order.side == "BUY" and price <= order.price:
                is_through = True
            elif order.side == "SELL" and price >= order.price:
                is_through = True

            if is_through:
                order.queue_qty_ahead = max(0.0, order.queue_qty_ahead - qty)
                self._trade_through[oid].append((time.time(), qty))

    # ── Estimation ───────────────────────────────────────────────────

    def estimate(self, order_id: str) -> Optional[QueueEstimate]:
        """Compute current queue position and fill probability."""
        order = self._orders.get(order_id)
        if order is None:
            return None

        key = f"{order.symbol}:{order.price}"
        history = self._level_qty_history.get(key, deque())

        # Current total qty at level
        total_level_qty = history[-1][1] if history else order.queue_qty_ahead + order.size

        # Position as % of level
        pos_pct = order.queue_qty_ahead / max(total_level_qty, 1e-12)

        # Fill probability model
        # Base: inverse of position fraction
        p_base = max(0.0, 1.0 - pos_pct)

        # Trade-through rate adjustment
        trades = self._trade_through.get(order_id, deque())
        recent_trades = [(t, q) for (t, q) in trades if time.time() - t < 30.0]
        consumed_rate = sum(q for _, q in recent_trades) / 30.0 if recent_trades else 0.0
        time_to_fill_est = (order.queue_qty_ahead / consumed_rate) if consumed_rate > 1e-12 else float('inf')

        # Time decay penalty — longer we wait without fills, probability decays
        elapsed_sec = (time.time() * 1000 - order.placed_at_ms) / 1000.0
        time_decay = max(0.0, 1.0 - self.decay_rate * elapsed_sec)

        fill_prob = min(1.0, p_base * time_decay)

        # Bounded wait estimate (cap at 10 min)
        wait_ms = min(600_000, int(time_to_fill_est * 1000)) if time_to_fill_est != float('inf') else 600_000

        # Recommendation
        if fill_prob < self.cancel_threshold:
            rec = "CANCEL"
        elif fill_prob < self.cross_threshold:
            rec = "CROSS_SPREAD"
        else:
            rec = "STAY_PASSIVE"

        return QueueEstimate(
            order_id=order_id,
            symbol=order.symbol,
            side=order.side,
            price=order.price,
            queue_position_qty=order.queue_qty_ahead,
            queue_position_pct=pos_pct,
            fill_probability=fill_prob,
            estimated_wait_ms=wait_ms,
            recommendation=rec,
        )

    def estimate_all(self) -> Dict[str, QueueEstimate]:
        """Estimate queue position for all tracked orders."""
        return {oid: est for oid in list(self._orders) if (est := self.estimate(oid)) is not None}

    def get_aggressive_candidates(self) -> List[str]:
        """Return order IDs that should cross the spread or be cancelled."""
        return [
            oid for oid, est in self.estimate_all().items()
            if est.recommendation in ("CROSS_SPREAD", "CANCEL")
        ]
