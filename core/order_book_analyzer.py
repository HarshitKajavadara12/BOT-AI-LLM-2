"""
QUANTUM-FORGE: Order Book & Trade Flow Analysis
=================================================
Consumes Binance partial depth stream (L2) and trade stream to produce:
  1. Order-book imbalance (bid/ask volume ratio)
  2. Spread analytics (bid-ask spread in bps)
  3. Trade flow toxicity (VPIN-like metric)
  4. Aggressive buyer/seller pressure

All metrics are exposed as a dict for the main pipeline to consume.
"""

import numpy as np
import logging
import threading
import time
import json
from collections import deque
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger("OrderBookFeed")


class OrderBookAnalyzer:
    """
    Real-time order book and trade flow analysis.
    
    Subscribes to:
      - Binance partial depth stream (@depth20@100ms)
      - Binance aggTrade stream (@aggTrade)
    
    Produces per-symbol metrics dict consumed by quantum_core.
    """

    def __init__(
        self,
        symbols: List[str],
        depth_levels: int = 20,
        trade_window: int = 200,
    ):
        self.symbols = [s.lower() for s in symbols]
        self.depth_levels = depth_levels
        self.trade_window = trade_window

        # Per-symbol state
        self._book: Dict[str, Dict] = {}       # latest order book snapshot
        self._trades: Dict[str, deque] = {}     # recent trades
        self._metrics: Dict[str, Dict] = {}     # computed metrics
        self._lock = threading.Lock()

        for s in self.symbols:
            self._book[s] = {"bids": [], "asks": []}
            self._trades[s] = deque(maxlen=trade_window)
            self._metrics[s] = self._empty_metrics()

    # ─── Public API ──────────────────────────────────────────────────

    def on_depth_update(self, symbol: str, data: dict):
        """
        Handle a partial book depth update.
        
        Expected data format (Binance @depth20):
            {"bids": [[price, qty], ...], "asks": [[price, qty], ...]}
        """
        sym = symbol.lower()
        if sym not in self._book:
            return

        bids = [[float(p), float(q)] for p, q in data.get("bids", [])]
        asks = [[float(p), float(q)] for p, q in data.get("asks", [])]

        with self._lock:
            self._book[sym] = {"bids": bids, "asks": asks}

        self._recompute_metrics(sym)

    def on_trade(self, symbol: str, data: dict):
        """
        Handle an aggTrade update.
        
        Expected data format:
            {"p": price, "q": qty, "m": is_maker}
        """
        sym = symbol.lower()
        if sym not in self._trades:
            return

        trade = {
            "price": float(data.get("p", 0)),
            "qty": float(data.get("q", 0)),
            "is_buyer_maker": bool(data.get("m", False)),
            "time": data.get("T", int(time.time() * 1000)),
        }

        with self._lock:
            self._trades[sym].append(trade)

        self._recompute_trade_flow(sym)

    def get_metrics(self, symbol: str) -> Dict:
        """Get current order book + trade flow metrics for a symbol."""
        sym = symbol.lower()
        with self._lock:
            return dict(self._metrics.get(sym, self._empty_metrics()))

    def get_imbalance(self, symbol: str) -> float:
        """Get bid/ask volume imbalance in [-1, 1]. Positive = bid-heavy."""
        return self.get_metrics(symbol).get("imbalance", 0.0)

    def get_spread_bps(self, symbol: str) -> float:
        """Get current bid-ask spread in basis points."""
        return self.get_metrics(symbol).get("spread_bps", 0.0)

    def get_toxicity(self, symbol: str) -> float:
        """Get VPIN-like trade flow toxicity in [0, 1]."""
        return self.get_metrics(symbol).get("toxicity", 0.0)

    # ─── Internal ────────────────────────────────────────────────────

    def _recompute_metrics(self, sym: str):
        """Recompute order-book-based metrics."""
        with self._lock:
            book = self._book[sym]

        bids = book["bids"]
        asks = book["asks"]

        if not bids or not asks:
            return

        # Bid/Ask volume
        bid_vol = sum(q for _, q in bids[:self.depth_levels])
        ask_vol = sum(q for _, q in asks[:self.depth_levels])
        total = bid_vol + ask_vol

        imbalance = (bid_vol - ask_vol) / total if total > 0 else 0.0

        # Spread
        best_bid = bids[0][0]
        best_ask = asks[0][0]
        mid = (best_bid + best_ask) / 2
        spread_bps = ((best_ask - best_bid) / mid * 10000) if mid > 0 else 0.0

        # Weighted mid price (microprice)
        microprice = (best_bid * ask_vol + best_ask * bid_vol) / total if total > 0 else mid

        # Book depth (total $ within first 5 levels)
        bid_depth_usd = sum(p * q for p, q in bids[:5])
        ask_depth_usd = sum(p * q for p, q in asks[:5])

        with self._lock:
            self._metrics[sym].update({
                "imbalance": float(np.clip(imbalance, -1, 1)),
                "spread_bps": float(spread_bps),
                "microprice": float(microprice),
                "best_bid": float(best_bid),
                "best_ask": float(best_ask),
                "bid_volume": float(bid_vol),
                "ask_volume": float(ask_vol),
                "bid_depth_usd": float(bid_depth_usd),
                "ask_depth_usd": float(ask_depth_usd),
                "updated_at": datetime.now().isoformat(),
            })

    def _recompute_trade_flow(self, sym: str):
        """Recompute trade-flow-based metrics (VPIN-like)."""
        with self._lock:
            trades = list(self._trades[sym])

        if len(trades) < 10:
            return

        # Buyer vs seller classification (is_buyer_maker=True means seller-initiated)
        buy_volume = sum(
            t["qty"] for t in trades if not t["is_buyer_maker"]
        )
        sell_volume = sum(
            t["qty"] for t in trades if t["is_buyer_maker"]
        )
        total_volume = buy_volume + sell_volume

        if total_volume < 1e-10:
            return

        # VPIN approximation: |buy - sell| / total
        toxicity = abs(buy_volume - sell_volume) / total_volume

        # Aggressiveness: fraction of volume that's market orders (taker)
        buy_pct = buy_volume / total_volume
        pressure = buy_pct * 2 - 1  # [-1, 1]: -1=all sellers, +1=all buyers

        # Trade intensity (trades per second, recent)
        if len(trades) >= 2:
            time_span = (trades[-1]["time"] - trades[0]["time"]) / 1000  # seconds
            intensity = len(trades) / max(time_span, 0.1)
        else:
            intensity = 0.0

        with self._lock:
            self._metrics[sym].update({
                "toxicity": float(np.clip(toxicity, 0, 1)),
                "buy_pressure": float(np.clip(pressure, -1, 1)),
                "buy_volume": float(buy_volume),
                "sell_volume": float(sell_volume),
                "trade_intensity": float(intensity),
            })

    @staticmethod
    def _empty_metrics() -> Dict:
        return {
            "imbalance": 0.0,
            "spread_bps": 0.0,
            "microprice": 0.0,
            "best_bid": 0.0,
            "best_ask": 0.0,
            "bid_volume": 0.0,
            "ask_volume": 0.0,
            "bid_depth_usd": 0.0,
            "ask_depth_usd": 0.0,
            "toxicity": 0.0,
            "buy_pressure": 0.0,
            "buy_volume": 0.0,
            "sell_volume": 0.0,
            "trade_intensity": 0.0,
            "updated_at": None,
        }
