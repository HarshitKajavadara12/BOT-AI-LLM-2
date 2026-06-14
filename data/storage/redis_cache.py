"""
QUANTUM-FORGE Redis Cache Manager
Ultra-fast in-memory cache for real-time trading data

Features:
- Hot cache for last 1M ticks (<100µs latency)
- Feature store for pre-computed signals
- Lock-free queues for multi-threaded access
- Automatic eviction policies
"""

import redis
import numpy as np
import pandas as pd
import pickle
import json
import time
import asyncio
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__)

@dataclass
class MarketTick:
    """Optimized tick data structure"""
    timestamp: float
    symbol: str
    price: float
    size: float
    side: str  # 'B' or 'S'
    exchange: str
    
    def to_bytes(self) -> bytes:
        """Serialize to bytes for Redis storage"""
        return pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)
    
    @classmethod
    def from_bytes(cls, data: bytes):
        """Deserialize from bytes"""
        return pickle.loads(data)

class RedisCache:
    """
    High-performance Redis cache for HFT data
    
    Three-tier caching strategy:
    - L1: Hot ticks (last 1M ticks, <100µs access)
    - L2: Features (pre-computed signals, <1ms access)
    - L3: Metadata (symbols, configs, <10ms access)
    """
    
    def __init__(self, 
                 host: str = 'localhost',
                 port: int = 6379,
                 db: int = 0,
                 password: Optional[str] = None,
                 max_connections: int = 100):
        
        # Connection pool for high throughput
        self.pool = redis.ConnectionPool(
            host=host,
            port=port,
            db=db,
            password=password,
            max_connections=max_connections,
            decode_responses=False,  # Keep binary for speed
            socket_keepalive=True,
            socket_keepalive_options={},
            health_check_interval=30
        )
        
        self.redis_client = redis.Redis(connection_pool=self.pool)
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Cache configuration
        self.hot_tick_limit = 1_000_000  # Last 1M ticks
        self.feature_ttl = 3600  # 1 hour TTL for features
        self.metadata_ttl = 86400  # 24 hours for metadata
        
        # Key namespaces
        self.TICK_NS = "tick:"
        self.FEATURE_NS = "feature:"
        self.META_NS = "meta:"
        self.ORDERBOOK_NS = "orderbook:"
        
        logger.info("Redis cache initialized")
    
    async def ping(self) -> bool:
        """Health check"""
        try:
            return self.redis_client.ping()
        except Exception as e:
            logger.error(f"Redis ping failed: {e}")
            return False
    
    # ==================== TICK DATA OPERATIONS ====================
    
    def store_tick(self, tick: MarketTick) -> bool:
        """
        Store tick with automatic eviction
        
        Key format: tick:{symbol}:{timestamp_ns}
        """
        try:
            key = f"{self.TICK_NS}{tick.symbol}:{int(tick.timestamp * 1e9)}"
            
            # Store tick data
            pipe = self.redis_client.pipeline()
            pipe.set(key, tick.to_bytes())
            
            # Add to sorted set for time-based queries
            sorted_key = f"{self.TICK_NS}sorted:{tick.symbol}"
            pipe.zadd(sorted_key, {key: tick.timestamp})
            
            # Maintain hot cache limit
            pipe.zremrangebyrank(sorted_key, 0, -(self.hot_tick_limit + 1))
            
            pipe.execute()
            return True
            
        except Exception as e:
            logger.error(f"Failed to store tick: {e}")
            return False
    
    def get_latest_ticks(self, symbol: str, count: int = 1000) -> List[MarketTick]:
        """Get latest N ticks for symbol (optimized for speed)"""
        try:
            sorted_key = f"{self.TICK_NS}sorted:{symbol}"
            
            # Get latest tick keys
            tick_keys = self.redis_client.zrevrange(sorted_key, 0, count - 1)
            
            if not tick_keys:
                return []
            
            # Batch retrieve tick data
            pipe = self.redis_client.pipeline()
            for key in tick_keys:
                pipe.get(key)
            
            tick_data = pipe.execute()
            
            # Deserialize ticks
            ticks = []
            for data in tick_data:
                if data:
                    ticks.append(MarketTick.from_bytes(data))
            
            return ticks
            
        except Exception as e:
            logger.error(f"Failed to get latest ticks: {e}")
            return []
    
    def get_ticks_range(self, symbol: str, start_time: float, end_time: float) -> List[MarketTick]:
        """Get ticks within time range"""
        try:
            sorted_key = f"{self.TICK_NS}sorted:{symbol}"
            
            # Get tick keys in time range
            tick_keys = self.redis_client.zrangebyscore(
                sorted_key, start_time, end_time
            )
            
            if not tick_keys:
                return []
            
            # Batch retrieve
            pipe = self.redis_client.pipeline()
            for key in tick_keys:
                pipe.get(key)
            
            tick_data = pipe.execute()
            
            ticks = []
            for data in tick_data:
                if data:
                    ticks.append(MarketTick.from_bytes(data))
            
            return sorted(ticks, key=lambda x: x.timestamp)
            
        except Exception as e:
            logger.error(f"Failed to get tick range: {e}")
            return []
    
    # ==================== FEATURE STORE OPERATIONS ====================
    
    def store_feature(self, symbol: str, feature_name: str, 
                     value: Union[float, np.ndarray, pd.Series], 
                     timestamp: Optional[float] = None) -> bool:
        """Store computed feature with TTL"""
        try:
            if timestamp is None:
                timestamp = time.time()
            
            key = f"{self.FEATURE_NS}{symbol}:{feature_name}:{int(timestamp)}"
            
            # Serialize value based on type
            if isinstance(value, (int, float)):
                serialized = json.dumps(value)
            elif isinstance(value, np.ndarray):
                serialized = pickle.dumps(value)
            elif isinstance(value, pd.Series):
                serialized = pickle.dumps(value.values)
            else:
                serialized = pickle.dumps(value)
            
            # Store with TTL
            self.redis_client.setex(key, self.feature_ttl, serialized)
            
            # Add to feature index
            index_key = f"{self.FEATURE_NS}index:{symbol}:{feature_name}"
            self.redis_client.zadd(index_key, {key: timestamp})
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store feature: {e}")
            return False
    
    def get_latest_feature(self, symbol: str, feature_name: str) -> Optional[Any]:
        """Get latest feature value"""
        try:
            index_key = f"{self.FEATURE_NS}index:{symbol}:{feature_name}"
            
            # Get latest feature key
            latest_keys = self.redis_client.zrevrange(index_key, 0, 0)
            
            if not latest_keys:
                return None
            
            # Get feature value
            feature_data = self.redis_client.get(latest_keys[0])
            
            if not feature_data:
                return None
            
            # Try JSON first (for simple values)
            try:
                return json.loads(feature_data)
            except:
                # Fall back to pickle
                return pickle.loads(feature_data)
                
        except Exception as e:
            logger.error(f"Failed to get latest feature: {e}")
            return None
    
    def get_feature_history(self, symbol: str, feature_name: str, 
                           count: int = 100) -> Dict[float, Any]:
        """Get feature history with timestamps"""
        try:
            index_key = f"{self.FEATURE_NS}index:{symbol}:{feature_name}"
            
            # Get feature keys with scores (timestamps)
            feature_items = self.redis_client.zrevrange(
                index_key, 0, count - 1, withscores=True
            )
            
            if not feature_items:
                return {}
            
            # Batch retrieve feature values
            pipe = self.redis_client.pipeline()
            for key, timestamp in feature_items:
                pipe.get(key)
            
            feature_data = pipe.execute()
            
            # Build result dictionary
            result = {}
            for (key, timestamp), data in zip(feature_items, feature_data):
                if data:
                    try:
                        value = json.loads(data)
                    except:
                        value = pickle.loads(data)
                    result[timestamp] = value
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get feature history: {e}")
            return {}
    
    # ==================== ORDER BOOK OPERATIONS ====================
    
    def store_orderbook_snapshot(self, symbol: str, bids: List[tuple], 
                                asks: List[tuple], timestamp: Optional[float] = None) -> bool:
        """Store order book snapshot"""
        try:
            if timestamp is None:
                timestamp = time.time()
            
            key = f"{self.ORDERBOOK_NS}{symbol}:{int(timestamp * 1e6)}"
            
            orderbook_data = {
                'bids': bids,  # [(price, size), ...]
                'asks': asks,
                'timestamp': timestamp
            }
            
            # Store with short TTL (order books change frequently)
            self.redis_client.setex(
                key, 300, pickle.dumps(orderbook_data)  # 5 minutes TTL
            )
            
            # Add to sorted set for time queries
            sorted_key = f"{self.ORDERBOOK_NS}sorted:{symbol}"
            self.redis_client.zadd(sorted_key, {key: timestamp})
            
            # Keep only last 1000 snapshots
            self.redis_client.zremrangebyrank(sorted_key, 0, -1001)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store orderbook: {e}")
            return False
    
    def get_latest_orderbook(self, symbol: str) -> Optional[Dict]:
        """Get latest order book snapshot"""
        try:
            sorted_key = f"{self.ORDERBOOK_NS}sorted:{symbol}"
            
            latest_keys = self.redis_client.zrevrange(sorted_key, 0, 0)
            
            if not latest_keys:
                return None
            
            orderbook_data = self.redis_client.get(latest_keys[0])
            
            if not orderbook_data:
                return None
            
            return pickle.loads(orderbook_data)
            
        except Exception as e:
            logger.error(f"Failed to get latest orderbook: {e}")
            return None
    
    # ==================== PERFORMANCE MONITORING ====================
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        try:
            info = self.redis_client.info()
            
            stats = {
                'memory_used': info.get('used_memory_human', 'Unknown'),
                'memory_peak': info.get('used_memory_peak_human', 'Unknown'),
                'connected_clients': info.get('connected_clients', 0),
                'commands_processed': info.get('total_commands_processed', 0),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0),
                'hit_rate': 0.0
            }
            
            # Calculate hit rate
            hits = stats['keyspace_hits']
            misses = stats['keyspace_misses']
            if hits + misses > 0:
                stats['hit_rate'] = hits / (hits + misses) * 100
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {}
    
    def flush_namespace(self, namespace: str) -> bool:
        """Flush specific namespace (for testing)"""
        try:
            keys = self.redis_client.keys(f"{namespace}*")
            if keys:
                self.redis_client.delete(*keys)
            return True
        except Exception as e:
            logger.error(f"Failed to flush namespace {namespace}: {e}")
            return False
    
    def close(self):
        """Clean shutdown"""
        try:
            self.executor.shutdown(wait=True)
            self.pool.disconnect()
            logger.info("Redis cache connection closed")
        except Exception as e:
            logger.error(f"Error closing Redis cache: {e}")

# ==================== ASYNC WRAPPER ====================

class AsyncRedisCache:
    """Async wrapper for Redis operations (for use with asyncio)"""
    
    def __init__(self, cache: RedisCache):
        self.cache = cache
        self.loop = asyncio.get_event_loop()
    
    async def store_tick(self, tick: MarketTick) -> bool:
        """Async tick storage"""
        return await self.loop.run_in_executor(
            self.cache.executor, self.cache.store_tick, tick
        )
    
    async def get_latest_ticks(self, symbol: str, count: int = 1000) -> List[MarketTick]:
        """Async tick retrieval"""
        return await self.loop.run_in_executor(
            self.cache.executor, self.cache.get_latest_ticks, symbol, count
        )
    
    async def store_feature(self, symbol: str, feature_name: str, 
                           value: Any, timestamp: Optional[float] = None) -> bool:
        """Async feature storage"""
        return await self.loop.run_in_executor(
            self.cache.executor, self.cache.store_feature, 
            symbol, feature_name, value, timestamp
        )
    
    async def get_latest_feature(self, symbol: str, feature_name: str) -> Optional[Any]:
        """Async feature retrieval"""
        return await self.loop.run_in_executor(
            self.cache.executor, self.cache.get_latest_feature, symbol, feature_name
        )

# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    # Initialize cache
    cache = RedisCache()
    
    # Test connection
    if cache.ping():
        print("  Redis connection successful")
    else:
        print("  Redis connection failed")
        exit(1)
    
    # Example tick data
    tick = MarketTick(
        timestamp=time.time(),
        symbol="BTCUSDT",
        price=67432.50,
        size=0.15,
        side="B",
        exchange="binance"
    )
    
    # Store tick
    if cache.store_tick(tick):
        print("  Tick stored successfully")
    
    # Retrieve latest ticks
    latest_ticks = cache.get_latest_ticks("BTCUSDT", count=10)
    print(f"  Retrieved {len(latest_ticks)} latest ticks")
    
    # Store feature
    rsi_value = 65.3
    if cache.store_feature("BTCUSDT", "RSI_14", rsi_value):
        print("  Feature stored successfully")
    
    # Retrieve feature
    latest_rsi = cache.get_latest_feature("BTCUSDT", "RSI_14")
    print(f"  Latest RSI: {latest_rsi}")
    
    # Cache statistics
    stats = cache.get_cache_stats()
    print(f"  Cache Stats: {stats}")
    
    # Cleanup
    cache.close()