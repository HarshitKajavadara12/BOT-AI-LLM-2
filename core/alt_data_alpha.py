"""
Alternative Data Alpha Bridge — Wires the existing 964-line
AlternativeDataLoader (Twitter/Reddit/News/Economic) into the live
trading pipeline to produce sentiment-based alpha signals.

Missing Concept 2.3: "Alternative Data Alphas"

Data flow:
    AlternativeDataLoader (background) →
        sentiment scores persisted in alternative_data.db →
            AltDataAlphaEngine reads recent scores →
                Produces per-symbol sentiment signal [-1, +1]

Pipeline integration:
    QuantumCoreOrchestrator._process_symbol() →
        alt_data_engine.get_sentiment_signal(symbol)
"""

from __future__ import annotations

import logging
import sqlite3
import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Try importing the real loader
try:
    from data.ingestion.alternative_data_loader import (
        AlternativeDataLoader,
        SentimentAnalyzer,
    )
    _HAS_ALT_LOADER = True
except Exception:
    _HAS_ALT_LOADER = False
    logger.info("AlternativeDataLoader not importable — using DB-only mode")


@dataclass
class SentimentSignal:
    """Aggregated sentiment signal for a symbol."""
    symbol: str
    score: float           # [-1, +1]  negative = bearish
    confidence: float      # [0, 1]
    n_sources: int
    n_datapoints: int
    freshness_minutes: float
    dominant_source: str
    timestamp: str


# ── Symbol mapping: crypto tickers → alt-data search terms ──────────
_CRYPTO_ALT_MAP = {
    "BTCUSDT": ["BTC", "bitcoin"],
    "ETHUSDT": ["ETH", "ethereum"],
    "BNBUSDT": ["BNB", "binance"],
    "SOLUSDT": ["SOL", "solana"],
    "ADAUSDT": ["ADA", "cardano"],
    "DOGEUSDT": ["DOGE", "dogecoin"],
    "XRPUSDT": ["XRP", "ripple"],
}


class AltDataAlphaEngine:
    """
    Produces sentiment-based alpha signals from alternative data.
    
    Two modes:
        1. Full mode: Starts AlternativeDataLoader background ingestion
           and reads fresh data.
        2. DB-only mode: Reads from existing alternative_data.db
           (works even without API keys).
    """

    def __init__(
        self,
        symbols: List[str],
        db_path: str = "alternative_data.db",
        lookback_hours: float = 24.0,
        update_interval_s: int = 300,
        decay_halflife_hours: float = 6.0,
    ):
        self.symbols = symbols
        self.db_path = db_path
        self.lookback_hours = lookback_hours
        self.update_interval_s = update_interval_s
        self.decay_halflife = decay_halflife_hours

        # Cached signals
        self._signals: Dict[str, SentimentSignal] = {}
        self._lock = threading.Lock()

        # Background loader
        self._loader: Optional[AlternativeDataLoader] = None
        self._bg_thread: Optional[threading.Thread] = None
        self._running = False

        # In-memory sentiment analyzer fallback
        self._analyzer = None
        if _HAS_ALT_LOADER:
            try:
                self._analyzer = SentimentAnalyzer()
            except Exception:
                pass

        self._init_db()

    # ── Lifecycle ────────────────────────────────────────────────────

    def start(self):
        """Start background ingestion (if AlternativeDataLoader available)."""
        if _HAS_ALT_LOADER:
            try:
                self._loader = AlternativeDataLoader()
                self._loader.start()
                logger.info("AltDataAlphaEngine: background loader started")
            except Exception as e:
                logger.warning("AltDataAlphaEngine: loader start failed: %s", e)

        self._running = True
        self._bg_thread = threading.Thread(
            target=self._refresh_loop, daemon=True, name="AltDataRefresh"
        )
        self._bg_thread.start()

    def stop(self):
        self._running = False
        if self._loader is not None:
            try:
                self._loader.stop()
            except Exception:
                pass

    # ── Public API ───────────────────────────────────────────────────

    def get_sentiment_signal(self, symbol: str) -> SentimentSignal:
        """Return latest aggregated sentiment signal for *symbol*."""
        with self._lock:
            sig = self._signals.get(symbol)
        if sig is not None:
            return sig
        return SentimentSignal(
            symbol=symbol, score=0.0, confidence=0.0,
            n_sources=0, n_datapoints=0, freshness_minutes=999.0,
            dominant_source="none", timestamp=datetime.utcnow().isoformat(),
        )

    def get_all_signals(self) -> Dict[str, SentimentSignal]:
        with self._lock:
            return dict(self._signals)

    # ── Internal ─────────────────────────────────────────────────────

    def _init_db(self):
        """Ensure the DB and table exist."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alternative_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT,
                    source_type TEXT,
                    symbol TEXT,
                    timestamp DATETIME,
                    content TEXT,
                    sentiment_score REAL,
                    sentiment_polarity INTEGER,
                    confidence_score REAL,
                    relevance_score REAL,
                    data_quality_score REAL,
                    metadata TEXT
                )
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning("AltData DB init: %s", e)

    def _refresh_loop(self):
        """Periodically re-aggregate sentiment from DB."""
        while self._running:
            try:
                self._refresh_signals()
            except Exception as e:
                logger.warning("AltData refresh error: %s", e)
            time.sleep(self.update_interval_s)

    def _refresh_signals(self):
        """Query DB for recent data and aggregate per symbol."""
        cutoff = (datetime.utcnow() - timedelta(hours=self.lookback_hours)).isoformat()

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
        except Exception:
            return

        for symbol in self.symbols:
            alt_keys = _CRYPTO_ALT_MAP.get(symbol, [symbol])
            placeholders = ",".join("?" for _ in alt_keys)

            try:
                cursor.execute(
                    f"""SELECT sentiment_score, confidence_score, source, timestamp
                        FROM alternative_data
                        WHERE symbol IN ({placeholders})
                          AND timestamp >= ?
                        ORDER BY timestamp DESC
                        LIMIT 200""",
                    (*alt_keys, cutoff),
                )
                rows = cursor.fetchall()
            except Exception:
                rows = []

            if not rows:
                continue

            # Exponential decay weighting
            now = datetime.utcnow()
            weighted_scores: List[float] = []
            weights: List[float] = []
            sources: Dict[str, int] = defaultdict(int)

            for sent, conf, src, ts_str in rows:
                try:
                    ts = datetime.fromisoformat(ts_str)
                    age_h = max((now - ts).total_seconds() / 3600, 0.01)
                except Exception:
                    age_h = self.lookback_hours

                w = np.exp(-np.log(2) * age_h / self.decay_halflife) * max(conf or 0.5, 0.1)
                weighted_scores.append((sent or 0.0) * w)
                weights.append(w)
                sources[src or "unknown"] += 1

            total_w = sum(weights)
            if total_w < 1e-9:
                continue

            agg_score = float(np.clip(sum(weighted_scores) / total_w, -1, 1))
            agg_conf = float(np.clip(total_w / len(rows), 0, 1))
            dominant = max(sources, key=sources.get) if sources else "unknown"

            # Freshness: age of newest record
            try:
                newest_ts = datetime.fromisoformat(rows[0][3])
                freshness = (now - newest_ts).total_seconds() / 60
            except Exception:
                freshness = 999.0

            sig = SentimentSignal(
                symbol=symbol,
                score=agg_score,
                confidence=agg_conf,
                n_sources=len(sources),
                n_datapoints=len(rows),
                freshness_minutes=freshness,
                dominant_source=dominant,
                timestamp=now.isoformat(),
            )
            with self._lock:
                self._signals[symbol] = sig

        conn.close()
