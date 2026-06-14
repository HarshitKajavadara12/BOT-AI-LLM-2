"""
Storage Coordinator — Unified Storage Layer

Wires together all 5 storage backends into the trading pipeline:
  - ParquetWriter: Always-on local columnar storage (trades, signals, market data)
  - RedisCache: Optional sub-ms tick cache + feature hot path
  - TimescaleDBManager: Optional time-series database for historical queries
  - FeatureStore: Optional versioned feature serving for ML
  - HistoricalDataManager: Catalog + retrieval layer

Graceful degradation: Parquet always works; Redis/TimescaleDB degrade to no-ops
if services are unavailable.
"""

import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
import pandas as pd
import numpy as np
import traceback

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Immutable record of a trade execution."""
    timestamp: str
    symbol: str
    side: str  # BUY or SELL
    quantity: float
    price: float
    fill_price: float
    slippage_bps: float
    fees: float
    algorithm: str
    pnl: Optional[float] = None
    signal_strength: Optional[float] = None
    regime: Optional[str] = None


@dataclass
class SignalRecord:
    """Record of a computed signal at a point in time."""
    timestamp: str
    symbol: str
    math_signal: float
    ml_signal: float
    fused_signal: float
    regime: str
    confidence: float
    components: Dict[str, float] = field(default_factory=dict)


class StorageCoordinator:
    """
    Unified storage layer that routes data to appropriate backends.
    
    Usage:
        sc = StorageCoordinator()
        sc.store_trade(trade_record)
        sc.store_signal(signal_record)
        sc.store_market_tick(symbol, price, volume, timestamp)
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._parquet = None
        self._redis = None
        self._timescale = None
        self._feature_store = None
        self._historical = None
        
        # Counters
        self._trades_stored = 0
        self._signals_stored = 0
        self._ticks_stored = 0
        self._errors = 0
        
        self._init_backends()

    def _init_backends(self):
        """Initialize all storage backends with graceful fallback."""
        # Parquet — always available (local filesystem)
        try:
            from data.storage.parquet_writer import ParquetWriter
            parquet_path = self.config.get('parquet_path', './data/parquet')
            self._parquet = ParquetWriter(base_path=parquet_path)
            logger.info("ParquetWriter initialized at %s", parquet_path)
        except Exception as e:
            logger.error("ParquetWriter init failed: %s", e)

        # Redis — optional
        try:
            from data.storage.redis_cache import RedisCache
            redis_cfg = self.config.get('redis', {})
            self._redis = RedisCache(
                host=redis_cfg.get('host', 'localhost'),
                port=redis_cfg.get('port', 6379),
                db=redis_cfg.get('db', 0),
            )
            logger.info("RedisCache initialized")
        except Exception as e:
            logger.warning("RedisCache unavailable (degraded mode): %s", e)
            self._redis = None

        # TimescaleDB — optional
        try:
            from data.storage.timescaledb_manager import TimescaleDBManager
            ts_cfg = self.config.get('timescaledb', {})
            if ts_cfg:
                self._timescale = TimescaleDBManager(connection_config=ts_cfg)
                self._timescale.start()
                logger.info("TimescaleDB initialized")
            else:
                logger.info("TimescaleDB config not provided — skipping")
        except Exception as e:
            logger.warning("TimescaleDB unavailable (degraded mode): %s", e)
            self._timescale = None

        # Feature Store — optional (needs Redis)
        try:
            from data.storage.feature_store import FeatureStore
            fs_cfg = self.config.get('feature_store', {})
            self._feature_store = FeatureStore(
                redis_host=fs_cfg.get('redis_host', 'localhost'),
                redis_port=fs_cfg.get('redis_port', 6379),
                redis_db=fs_cfg.get('redis_db', 1),
                parquet_path=fs_cfg.get('parquet_path', './data/features'),
            )
            logger.info("FeatureStore initialized")
        except Exception as e:
            logger.warning("FeatureStore unavailable (degraded mode): %s", e)
            self._feature_store = None

        # Historical Data Manager — always available (SQLite + local files)
        try:
            from data.storage.historical_data import HistoricalDataManager
            self._historical = HistoricalDataManager()
            logger.info("HistoricalDataManager initialized")
        except Exception as e:
            logger.warning("HistoricalDataManager init failed: %s", e)

    # ========================================================================
    # Trade Storage
    # ========================================================================

    def store_trade(self, trade: TradeRecord) -> bool:
        """Store a trade execution record to all available backends."""
        stored = False
        try:
            # Always write to Parquet
            if self._parquet:
                df = pd.DataFrame([asdict(trade)])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['date'] = df['timestamp'].dt.date.astype(str)
                self._parquet.append_data(df, table_name='trades')
                stored = True
        except Exception as e:
            logger.error("Parquet trade write failed: %s", e)
            self._errors += 1

        try:
            # Optional: TimescaleDB
            if self._timescale:
                self._timescale.insert_tick_data({
                    'symbol': trade.symbol,
                    'timestamp': trade.timestamp,
                    'price': trade.fill_price,
                    'volume': trade.quantity,
                    'exchange': 'binance',
                    'side': trade.side.lower(),
                })
        except Exception as e:
            logger.debug("TimescaleDB trade write failed: %s", e)

        if stored:
            self._trades_stored += 1
        return stored

    # ========================================================================
    # Signal Storage
    # ========================================================================

    def store_signal(self, signal: SignalRecord) -> bool:
        """Store a computed signal to Parquet and optionally FeatureStore."""
        stored = False
        try:
            if self._parquet:
                record = asdict(signal)
                # Flatten components dict into columns
                components = record.pop('components', {})
                for k, v in components.items():
                    record[f'component_{k}'] = v
                df = pd.DataFrame([record])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['date'] = df['timestamp'].dt.date.astype(str)
                self._parquet.append_data(df, table_name='signals')
                stored = True
        except Exception as e:
            logger.error("Parquet signal write failed: %s", e)
            self._errors += 1

        try:
            if self._feature_store:
                ts = datetime.fromisoformat(signal.timestamp).timestamp()
                self._feature_store.write_features_batch([
                    (f'math_signal', signal.symbol, signal.math_signal, ts),
                    (f'ml_signal', signal.symbol, signal.ml_signal, ts),
                    (f'fused_signal', signal.symbol, signal.fused_signal, ts),
                    (f'regime', signal.symbol, signal.regime, ts),
                ])
        except Exception as e:
            logger.debug("FeatureStore signal write failed: %s", e)

        if stored:
            self._signals_stored += 1
        return stored

    # ========================================================================
    # Market Data Storage
    # ========================================================================

    def store_market_tick(self, symbol: str, price: float, volume: float,
                          bid: float = 0.0, ask: float = 0.0,
                          timestamp: Optional[str] = None) -> bool:
        """Store a market tick to Redis (hot) and optionally TimescaleDB (cold)."""
        stored = False
        ts = timestamp or datetime.now(timezone.utc).isoformat()

        try:
            if self._redis:
                from data.storage.redis_cache import MarketTick
                tick = MarketTick(
                    symbol=symbol,
                    price=price,
                    volume=volume,
                    bid=bid,
                    ask=ask,
                    timestamp=datetime.fromisoformat(ts).timestamp(),
                )
                self._redis.store_tick(tick)
                stored = True
        except Exception as e:
            logger.debug("Redis tick write failed: %s", e)

        try:
            if self._timescale:
                self._timescale.insert_tick_data({
                    'symbol': symbol,
                    'timestamp': ts,
                    'price': price,
                    'volume': volume,
                    'bid': bid,
                    'ask': ask,
                    'exchange': 'binance',
                })
        except Exception as e:
            logger.debug("TimescaleDB tick write failed: %s", e)

        if stored:
            self._ticks_stored += 1
        return stored

    # ========================================================================
    # Feature Storage
    # ========================================================================

    def store_features(self, symbol: str, features: Dict[str, float],
                       timestamp: Optional[str] = None) -> bool:
        """Store computed features for a symbol."""
        ts_val = datetime.fromisoformat(
            timestamp or datetime.now(timezone.utc).isoformat()
        ).timestamp()
        
        try:
            if self._feature_store:
                batch = [
                    (name, symbol, value, ts_val)
                    for name, value in features.items()
                ]
                self._feature_store.write_features_batch(batch)
                return True
        except Exception as e:
            logger.debug("FeatureStore feature write failed: %s", e)

        try:
            if self._parquet:
                record = {'timestamp': timestamp or datetime.now(timezone.utc).isoformat(),
                          'symbol': symbol, 'date': datetime.now(timezone.utc).strftime('%Y-%m-%d')}
                record.update(features)
                df = pd.DataFrame([record])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                self._parquet.append_data(df, table_name='features')
                return True
        except Exception as e:
            logger.error("Parquet feature write failed: %s", e)
            self._errors += 1

        return False

    # ========================================================================
    # Read Methods
    # ========================================================================

    def get_trades(self, symbol: Optional[str] = None,
                   start_date: Optional[str] = None,
                   end_date: Optional[str] = None) -> pd.DataFrame:
        """Read trade history from Parquet."""
        try:
            if self._parquet:
                symbols = [symbol] if symbol else None
                return self._parquet.read_data(
                    table_name='trades',
                    symbols=symbols,
                    start_date=start_date,
                    end_date=end_date,
                )
        except Exception as e:
            logger.error("Trade read failed: %s", e)
        return pd.DataFrame()

    def get_signals(self, symbol: Optional[str] = None,
                    start_date: Optional[str] = None) -> pd.DataFrame:
        """Read signal history from Parquet."""
        try:
            if self._parquet:
                symbols = [symbol] if symbol else None
                return self._parquet.read_data(
                    table_name='signals',
                    symbols=symbols,
                    start_date=start_date,
                )
        except Exception as e:
            logger.error("Signal read failed: %s", e)
        return pd.DataFrame()

    def get_latest_tick(self, symbol: str) -> Optional[Dict]:
        """Get latest tick from Redis cache."""
        try:
            if self._redis:
                ticks = self._redis.get_latest_ticks(symbol, count=1)
                if ticks:
                    return {'price': ticks[0].price, 'volume': ticks[0].volume,
                            'bid': ticks[0].bid, 'ask': ticks[0].ask,
                            'timestamp': ticks[0].timestamp}
        except Exception as e:
            logger.debug("Redis tick read failed: %s", e)
        return None

    # ========================================================================
    # Status & Lifecycle
    # ========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Return storage coordinator statistics."""
        backends = {
            'parquet': self._parquet is not None,
            'redis': self._redis is not None,
            'timescaledb': self._timescale is not None,
            'feature_store': self._feature_store is not None,
            'historical': self._historical is not None,
        }
        return {
            'trades_stored': self._trades_stored,
            'signals_stored': self._signals_stored,
            'ticks_stored': self._ticks_stored,
            'errors': self._errors,
            'backends_active': sum(backends.values()),
            'backends': backends,
        }

    def close(self):
        """Shutdown all backends."""
        for name, backend in [
            ('redis', self._redis),
            ('timescale', self._timescale),
            ('feature_store', self._feature_store),
            ('parquet', self._parquet),
        ]:
            if backend and hasattr(backend, 'close'):
                try:
                    backend.close()
                except Exception as e:
                    logger.error("Error closing %s: %s", name, e)
            elif backend and hasattr(backend, 'stop'):
                try:
                    backend.stop()
                except Exception as e:
                    logger.error("Error stopping %s: %s", name, e)
        logger.info("StorageCoordinator shutdown complete")
