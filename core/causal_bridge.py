"""
QUANTUM-FORGE: Causal Discovery Bridge
========================================
Lightweight bridge that runs CausalDiscoveryEnsemble on cross-asset
returns and exposes the causal graph for signal gating.

The graph tells us: "Which asset returns *cause* which other returns?"
This lets us:
  - Boost signals when a causal parent confirms the direction
  - Suppress signals when the causal parent contradicts
  - Detect regime changes (graph structure shifts)
"""

import numpy as np
import logging
import threading
from typing import Dict, List, Optional, Tuple
from collections import deque
from datetime import datetime

logger = logging.getLogger("CausalBridge")


class CausalBridge:
    """
    Periodically runs CausalDiscoveryEnsemble on multi-asset returns.
    
    Thread-safe: discovery runs in background, last result is always available.
    """

    def __init__(
        self,
        symbols: List[str],
        lookback: int = 200,
        update_every_n: int = 50,
    ):
        self.symbols = symbols
        self.lookback = lookback
        self.update_every_n = update_every_n

        # Returns buffer: {symbol: deque of returns}
        self._returns: Dict[str, deque] = {
            s: deque(maxlen=lookback) for s in symbols
        }
        self._prices: Dict[str, Optional[float]] = {s: None for s in symbols}
        self._update_counter = 0

        # Causal graph result (thread-safe via lock)
        self._lock = threading.Lock()
        self._causal_strengths: Dict[Tuple[str, str], float] = {}
        self._last_updated: Optional[datetime] = None
        self._graph_edges: List[Tuple[str, str, float]] = []

        # Try to load causal discovery
        self._discovery = None
        try:
            from intelligence.feature_learning.causal_discovery import CausalDiscoveryEnsemble
            self._discovery = CausalDiscoveryEnsemble(voting_threshold=0.5)
            logger.info("CausalBridge: CausalDiscoveryEnsemble loaded")
        except ImportError:
            logger.warning("CausalBridge: causal_discovery not available — will be a no-op")

    def update(self, symbol: str, price: float):
        """Feed a new price tick for a symbol."""
        if symbol not in self._returns:
            return

        prev = self._prices[symbol]
        self._prices[symbol] = price

        if prev is not None and prev > 0:
            ret = (price - prev) / prev
            self._returns[symbol].append(ret)

        self._update_counter += 1

        # Periodically re-run discovery (don't run every tick)
        if self._update_counter >= self.update_every_n:
            self._update_counter = 0
            self._run_discovery()

    def _run_discovery(self):
        """Run causal discovery on accumulated returns (lightweight)."""
        if self._discovery is None:
            return

        # Build returns matrix: each column = symbol returns
        min_len = min(len(self._returns[s]) for s in self.symbols)
        if min_len < 30:
            return  # Need enough data

        try:
            X = np.column_stack([
                np.array(list(self._returns[s]))[-min_len:]
                for s in self.symbols
            ])

            graph = self._discovery.discover_graph(
                X, variable_names=self.symbols, time_series=True
            )

            # Extract causal strengths
            new_strengths: Dict[Tuple[str, str], float] = {}
            new_edges: List[Tuple[str, str, float]] = []

            for u, v, data in graph.edges(data=True):
                strength = data.get("weight", data.get("causal_strength", 0.5))
                new_strengths[(u, v)] = float(strength)
                new_edges.append((u, v, float(strength)))

            with self._lock:
                self._causal_strengths = new_strengths
                self._graph_edges = new_edges
                self._last_updated = datetime.now()

            if new_edges:
                logger.debug(
                    f"CausalBridge updated: {len(new_edges)} edges. "
                    f"Top: {sorted(new_edges, key=lambda x: -x[2])[:3]}"
                )

        except Exception as e:
            logger.debug(f"CausalBridge discovery failed: {e}")

    def get_causal_boost(self, target_symbol: str) -> float:
        """
        Get a [-1, 1] causal boost for a target symbol.

        If causal parents of `target_symbol` have moved in a direction
        that historically causes `target_symbol` to follow, return positive.
        If they moved opposite, return negative.
        """
        with self._lock:
            if not self._causal_strengths:
                return 0.0

            boost = 0.0
            count = 0

            for (parent, child), strength in self._causal_strengths.items():
                if child != target_symbol:
                    continue

                # Check if parent has recent positive or negative return
                parent_returns = list(self._returns.get(parent, []))
                if len(parent_returns) < 3:
                    continue

                recent_parent_ret = np.mean(parent_returns[-3:])
                # If parent moved positively → causal strength says target should follow
                boost += np.sign(recent_parent_ret) * strength
                count += 1

            if count == 0:
                return 0.0

            return float(np.clip(boost / count, -1, 1))

    def get_graph_summary(self) -> Dict:
        """Return a summary of the current causal graph."""
        with self._lock:
            return {
                "edges": self._graph_edges,
                "n_edges": len(self._graph_edges),
                "last_updated": self._last_updated.isoformat() if self._last_updated else None,
                "symbols": self.symbols,
            }
