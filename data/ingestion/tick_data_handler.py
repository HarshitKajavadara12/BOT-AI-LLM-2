"""
Tick Data Handler
================

High-performance tick data ingestion and processing system for real-time market data.
Handles multiple data sources, normalization, and real-time streaming capabilities.

Features:
- Multi-source tick data ingestion (exchanges, market data vendors)
- Real-time data normalization and validation
- High-frequency data processing with microsecond precision
- Data quality monitoring and anomaly detection
- Configurable buffering and batching strategies
- WebSocket and FIX protocol support
- Failover and recovery mechanisms
- Performance monitoring and latency tracking

Author: Quantum Forge Data Team
Date: November 2025
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
import json
import websocket
import threading
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor
import queue
import yaml
import redis
from pathlib import Path
import sys

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataSource(Enum):
    """Supported data sources."""
    BINANCE = "binance"
    COINBASE = "coinbase"
    KRAKEN = "kraken"
    ALPACA = "alpaca"
    IEX = "iex"
    POLYGON = "polygon"
    BLOOMBERG = "bloomberg"
    REFINITIV = "refinitiv"

class MessageType(Enum):
    """Message types for tick data."""
    TRADE = "trade"
    QUOTE = "quote"
    ORDER_BOOK = "orderbook"
    BAR = "bar"
    NEWS = "news"
    STATUS = "status"

@dataclass
class TickData:
    """Standardized tick data structure."""
    symbol: str
    exchange: str
    timestamp: datetime
    message_type: MessageType
    price: Optional[float] = None
    size: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_size: Optional[float] = None
    ask_size: Optional[float] = None
    trade_id: Optional[str] = None
    sequence: Optional[int] = None
    raw_data: Dict[str, Any] = field(default_factory=dict)
    received_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'symbol': self.symbol,
            'exchange': self.exchange,
            'timestamp': self.timestamp.isoformat(),
            'message_type': self.message_type.value,
            'price': self.price,
            'size': self.size,
            'bid': self.bid,
            'ask': self.ask,
            'bid_size': self.bid_size,
            'ask_size': self.ask_size,
            'trade_id': self.trade_id,
            'sequence': self.sequence,
            'received_at': self.received_at.isoformat(),
            'raw_data': self.raw_data
        }

@dataclass
class DataSourceConfig:
    """Configuration for data source connections."""
    name: str
    url: str
    api_key: Optional[str] = None
    secret_key: Optional[str] = None
    symbols: List[str] = field(default_factory=list)
    channels: List[str] = field(default_factory=list)
    max_reconnects: int = 10
    reconnect_delay: float = 5.0
    heartbeat_interval: float = 30.0
    buffer_size: int = 10000

class DataQualityMonitor:
    """Monitor data quality and detect anomalies."""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.message_counts = defaultdict(lambda: deque(maxlen=window_size))
        self.latencies = defaultdict(lambda: deque(maxlen=window_size))
        self.price_history = defaultdict(lambda: deque(maxlen=window_size))
        self.last_update = defaultdict(datetime)
    
    def update(self, tick: TickData):
        """Update quality metrics with new tick data."""
        key = f"{tick.exchange}:{tick.symbol}"
        
        # Update message count
        self.message_counts[key].append(tick.timestamp)
        
        # Calculate latency
        latency = (tick.received_at - tick.timestamp).total_seconds() * 1000  # ms
        self.latencies[key].append(latency)
        
        # Update price history for gap detection
        if tick.price is not None:
            self.price_history[key].append(tick.price)
        
        self.last_update[key] = datetime.now()
    
    def get_metrics(self, symbol: str, exchange: str) -> Dict[str, Any]:
        """Get quality metrics for a symbol."""
        key = f"{exchange}:{symbol}"
        
        if key not in self.message_counts:
            return {}
        
        # Message rate (per second)
        timestamps = list(self.message_counts[key])
        if len(timestamps) > 1:
            time_span = (timestamps[-1] - timestamps[0]).total_seconds()
            message_rate = len(timestamps) / max(time_span, 1)
        else:
            message_rate = 0
        
        # Latency statistics
        latencies = list(self.latencies[key])
        latency_stats = {
            'mean': np.mean(latencies) if latencies else 0,
            'p95': np.percentile(latencies, 95) if latencies else 0,
            'p99': np.percentile(latencies, 99) if latencies else 0
        }
        
        # Price gap detection
        prices = list(self.price_history[key])
        price_gaps = []
        if len(prices) > 1:
            price_diffs = np.diff(prices)
            price_gaps = [abs(diff) for diff in price_diffs if abs(diff) > np.std(prices) * 3]
        
        # Data freshness
        last_update = self.last_update.get(key, datetime.min)
        staleness_seconds = (datetime.now() - last_update).total_seconds()
        
        return {
            'message_rate': message_rate,
            'latency': latency_stats,
            'price_gaps': len(price_gaps),
            'staleness_seconds': staleness_seconds,
            'total_messages': len(timestamps)
        }

class WebSocketHandler:
    """WebSocket connection handler for real-time data feeds."""
    
    def __init__(self, config: DataSourceConfig, message_processor: Callable[[Dict], None]):
        self.config = config
        self.message_processor = message_processor
        self.ws = None
        self.is_connected = False
        self.reconnect_count = 0
        self.last_heartbeat = datetime.now()
        
        # Threading
        self.worker_thread = None
        self.should_stop = threading.Event()
        
    def connect(self):
        """Establish WebSocket connection."""
        try:
            logger.info(f"Connecting to {self.config.name} at {self.config.url}")
            
            self.ws = websocket.WebSocketApp(
                self.config.url,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
                on_open=self._on_open
            )
            
            # Start WebSocket in separate thread
            self.worker_thread = threading.Thread(target=self.ws.run_forever)
            self.worker_thread.daemon = True
            self.worker_thread.start()
            
        except Exception as e:
            logger.error(f"Failed to connect to {self.config.name}: {str(e)}")
            self._handle_reconnect()
    
    def _on_open(self, ws):
        """Handle WebSocket connection opened."""
        logger.info(f"Connected to {self.config.name}")
        self.is_connected = True
        self.reconnect_count = 0
        self.last_heartbeat = datetime.now()
        
        # Subscribe to channels
        self._subscribe_channels()
        
        # Start heartbeat timer
        self._start_heartbeat()
    
    def _on_message(self, ws, message):
        """Handle incoming WebSocket message."""
        try:
            # Update heartbeat
            self.last_heartbeat = datetime.now()
            
            # Parse and process message
            data = json.loads(message)
            self.message_processor(data)
            
        except Exception as e:
            logger.error(f"Error processing message from {self.config.name}: {str(e)}")
    
    def _on_error(self, ws, error):
        """Handle WebSocket error."""
        logger.error(f"WebSocket error for {self.config.name}: {str(error)}")
    
    def _on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket connection closed."""
        logger.warning(f"Connection to {self.config.name} closed: {close_msg}")
        self.is_connected = False
        
        if not self.should_stop.is_set():
            self._handle_reconnect()
    
    def _subscribe_channels(self):
        """Subscribe to configured channels."""
        if not self.config.channels or not self.config.symbols:
            return
        
        # Generic subscription format - customize per exchange
        subscription = {
            "method": "SUBSCRIBE",
            "params": [f"{symbol}@{channel}" for symbol in self.config.symbols 
                      for channel in self.config.channels],
            "id": int(time.time())
        }
        
        self.ws.send(json.dumps(subscription))
        logger.info(f"Subscribed to {len(subscription['params'])} channels on {self.config.name}")
    
    def _start_heartbeat(self):
        """Start heartbeat monitoring."""
        def heartbeat_monitor():
            while not self.should_stop.is_set() and self.is_connected:
                time.sleep(self.config.heartbeat_interval)
                
                # Check if we've received messages recently
                time_since_last = (datetime.now() - self.last_heartbeat).total_seconds()
                if time_since_last > self.config.heartbeat_interval * 2:
                    logger.warning(f"No heartbeat from {self.config.name} for {time_since_last}s")
                    self.ws.close()
                    break
        
        heartbeat_thread = threading.Thread(target=heartbeat_monitor)
        heartbeat_thread.daemon = True
        heartbeat_thread.start()
    
    def _handle_reconnect(self):
        """Handle reconnection logic."""
        if self.reconnect_count >= self.config.max_reconnects:
            logger.error(f"Max reconnects reached for {self.config.name}")
            return
        
        self.reconnect_count += 1
        wait_time = self.config.reconnect_delay * (2 ** (self.reconnect_count - 1))  # Exponential backoff
        
        logger.info(f"Reconnecting to {self.config.name} in {wait_time}s (attempt {self.reconnect_count})")
        time.sleep(wait_time)
        
        if not self.should_stop.is_set():
            self.connect()
    
    def disconnect(self):
        """Disconnect WebSocket."""
        self.should_stop.set()
        self.is_connected = False
        
        if self.ws:
            self.ws.close()
        
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)

class TickDataHandler:
    """
    Main tick data handler orchestrating data ingestion from multiple sources.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize tick data handler."""
        self.config = self._load_config(config_path)
        self.data_sources = {}
        self.ws_handlers = {}
        self.quality_monitor = DataQualityMonitor()
        
        # Data buffering
        self.tick_buffer = queue.Queue(maxsize=100000)
        self.batch_buffer = []
        self.batch_size = self.config.get('batch_size', 1000)
        self.batch_timeout = self.config.get('batch_timeout', 5.0)
        self.last_batch_time = time.time()
        
        # Redis connection for caching
        self.redis_client = None
        if self.config.get('redis_enabled', False):
            self._setup_redis()
        
        # Processing threads
        self.processor_pool = ThreadPoolExecutor(max_workers=4)
        self.is_running = False
        
        # Statistics
        self.stats = {
            'messages_processed': 0,
            'messages_dropped': 0,
            'last_reset': datetime.now()
        }
        
        # Data callbacks
        self.callbacks = []
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        
        # Default configuration
        return {
            'batch_size': 1000,
            'batch_timeout': 5.0,
            'redis_enabled': False,
            'redis_host': 'localhost',
            'redis_port': 6379,
            'redis_db': 0,
            'data_sources': []
        }
    
    def _setup_redis(self):
        """Setup Redis connection."""
        try:
            self.redis_client = redis.Redis(
                host=self.config.get('redis_host', 'localhost'),
                port=self.config.get('redis_port', 6379),
                db=self.config.get('redis_db', 0),
                decode_responses=True
            )
            self.redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            self.redis_client = None
    
    def add_data_source(self, config: DataSourceConfig):
        """Add a new data source."""
        self.data_sources[config.name] = config
        logger.info(f"Added data source: {config.name}")
    
    def add_callback(self, callback: Callable[[List[TickData]], None]):
        """Add callback for processed tick data."""
        self.callbacks.append(callback)
    
    def start(self):
        """Start tick data ingestion."""
        if self.is_running:
            logger.warning("Tick data handler is already running")
            return
        
        self.is_running = True
        logger.info("Starting tick data handler")
        
        # Start WebSocket connections
        for name, config in self.data_sources.items():
            handler = WebSocketHandler(config, self._process_raw_message)
            self.ws_handlers[name] = handler
            handler.connect()
        
        # Start processing threads
        self.processor_pool.submit(self._process_tick_buffer)
        self.processor_pool.submit(self._batch_processor)
        self.processor_pool.submit(self._stats_reporter)
        
        logger.info(f"Started {len(self.data_sources)} data source connections")
    
    def stop(self):
        """Stop tick data ingestion."""
        if not self.is_running:
            return
        
        logger.info("Stopping tick data handler")
        self.is_running = False
        
        # Disconnect WebSocket handlers
        for handler in self.ws_handlers.values():
            handler.disconnect()
        
        # Shutdown thread pool
        self.processor_pool.shutdown(wait=True)
        
        # Process remaining buffer
        self._flush_buffers()
        
        logger.info("Tick data handler stopped")
    
    def _process_raw_message(self, raw_data: Dict[str, Any]):
        """Process raw message from WebSocket."""
        try:
            # Normalize message based on source
            tick_data = self._normalize_message(raw_data)
            
            if tick_data:
                # Add to buffer
                if not self.tick_buffer.full():
                    self.tick_buffer.put(tick_data)
                    self.stats['messages_processed'] += 1
                else:
                    self.stats['messages_dropped'] += 1
                    logger.warning("Tick buffer full, dropping message")
                
                # Update quality monitor
                self.quality_monitor.update(tick_data)
                
                # Cache in Redis if enabled
                if self.redis_client:
                    self._cache_tick_data(tick_data)
        
        except Exception as e:
            logger.error(f"Error processing raw message: {str(e)}")
    
    def _normalize_message(self, raw_data: Dict[str, Any]) -> Optional[TickData]:
        """Normalize raw message to standard TickData format."""
        try:
            # This is a generic implementation - customize per exchange
            
            # Extract common fields
            symbol = raw_data.get('s') or raw_data.get('symbol', 'UNKNOWN')
            exchange = raw_data.get('exchange', 'unknown')
            
            # Parse timestamp
            timestamp_ms = raw_data.get('T') or raw_data.get('timestamp', int(time.time() * 1000))
            timestamp = datetime.fromtimestamp(timestamp_ms / 1000)
            
            # Determine message type
            if 'p' in raw_data or 'price' in raw_data:
                message_type = MessageType.TRADE
                price = float(raw_data.get('p') or raw_data.get('price', 0))
                size = float(raw_data.get('q') or raw_data.get('size', 0))
                
                return TickData(
                    symbol=symbol,
                    exchange=exchange,
                    timestamp=timestamp,
                    message_type=message_type,
                    price=price,
                    size=size,
                    trade_id=raw_data.get('t') or raw_data.get('trade_id'),
                    raw_data=raw_data
                )
            
            elif 'b' in raw_data or 'bid' in raw_data:
                message_type = MessageType.QUOTE
                bid = float(raw_data.get('b') or raw_data.get('bid', 0))
                ask = float(raw_data.get('a') or raw_data.get('ask', 0))
                bid_size = float(raw_data.get('B') or raw_data.get('bid_size', 0))
                ask_size = float(raw_data.get('A') or raw_data.get('ask_size', 0))
                
                return TickData(
                    symbol=symbol,
                    exchange=exchange,
                    timestamp=timestamp,
                    message_type=message_type,
                    bid=bid,
                    ask=ask,
                    bid_size=bid_size,
                    ask_size=ask_size,
                    raw_data=raw_data
                )
            
            # Skip unrecognized message types
            return None
            
        except Exception as e:
            logger.error(f"Error normalizing message: {str(e)}")
            return None
    
    def _cache_tick_data(self, tick: TickData):
        """Cache tick data in Redis."""
        try:
            if not self.redis_client:
                return
            
            key = f"tick:{tick.exchange}:{tick.symbol}:latest"
            value = json.dumps(tick.to_dict())
            
            # Set with expiration
            self.redis_client.setex(key, 300, value)  # 5 minutes
            
        except Exception as e:
            logger.error(f"Error caching tick data: {str(e)}")
    
    def _process_tick_buffer(self):
        """Process tick data from buffer."""
        while self.is_running:
            try:
                # Get tick from buffer with timeout
                tick = self.tick_buffer.get(timeout=1.0)
                
                # Add to batch
                self.batch_buffer.append(tick)
                
                # Check if batch is ready
                current_time = time.time()
                if (len(self.batch_buffer) >= self.batch_size or 
                    current_time - self.last_batch_time >= self.batch_timeout):
                    self._process_batch()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing tick buffer: {str(e)}")
    
    def _process_batch(self):
        """Process a batch of tick data."""
        if not self.batch_buffer:
            return
        
        try:
            batch = self.batch_buffer.copy()
            self.batch_buffer.clear()
            self.last_batch_time = time.time()
            
            # Call registered callbacks
            for callback in self.callbacks:
                try:
                    callback(batch)
                except Exception as e:
                    logger.error(f"Error in callback: {str(e)}")
            
            logger.debug(f"Processed batch of {len(batch)} ticks")
            
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
    
    def _batch_processor(self):
        """Periodic batch processor for timeout handling."""
        while self.is_running:
            time.sleep(1.0)
            
            if self.batch_buffer:
                current_time = time.time()
                if current_time - self.last_batch_time >= self.batch_timeout:
                    self._process_batch()
    
    def _stats_reporter(self):
        """Periodic statistics reporter."""
        while self.is_running:
            time.sleep(60.0)  # Report every minute
            
            current_time = datetime.now()
            elapsed = (current_time - self.stats['last_reset']).total_seconds()
            
            if elapsed > 0:
                msg_rate = self.stats['messages_processed'] / elapsed
                drop_rate = self.stats['messages_dropped'] / elapsed
                
                logger.info(f"Stats - Messages/sec: {msg_rate:.1f}, Drops/sec: {drop_rate:.1f}")
                
                # Reset stats
                self.stats['messages_processed'] = 0
                self.stats['messages_dropped'] = 0
                self.stats['last_reset'] = current_time
    
    def _flush_buffers(self):
        """Flush remaining data in buffers."""
        logger.info("Flushing remaining buffers")
        
        # Process remaining items in tick buffer
        while not self.tick_buffer.empty():
            try:
                tick = self.tick_buffer.get_nowait()
                self.batch_buffer.append(tick)
            except queue.Empty:
                break
        
        # Process final batch
        if self.batch_buffer:
            self._process_batch()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current statistics."""
        stats = self.stats.copy()
        
        # Add connection status
        stats['connections'] = {
            name: handler.is_connected 
            for name, handler in self.ws_handlers.items()
        }
        
        # Add quality metrics
        stats['quality_metrics'] = {}
        for source_config in self.data_sources.values():
            for symbol in source_config.symbols:
                metrics = self.quality_monitor.get_metrics(symbol, source_config.name)
                if metrics:
                    stats['quality_metrics'][f"{source_config.name}:{symbol}"] = metrics
        
        return stats
    
    def get_latest_tick(self, symbol: str, exchange: str) -> Optional[TickData]:
        """Get latest tick data for a symbol."""
        if not self.redis_client:
            return None
        
        try:
            key = f"tick:{exchange}:{symbol}:latest"
            data = self.redis_client.get(key)
            
            if data:
                tick_dict = json.loads(data)
                # Reconstruct TickData object (simplified)
                return TickData(
                    symbol=tick_dict['symbol'],
                    exchange=tick_dict['exchange'],
                    timestamp=datetime.fromisoformat(tick_dict['timestamp']),
                    message_type=MessageType(tick_dict['message_type']),
                    price=tick_dict.get('price'),
                    size=tick_dict.get('size'),
                    bid=tick_dict.get('bid'),
                    ask=tick_dict.get('ask'),
                    bid_size=tick_dict.get('bid_size'),
                    ask_size=tick_dict.get('ask_size'),
                    trade_id=tick_dict.get('trade_id'),
                    sequence=tick_dict.get('sequence'),
                    raw_data=tick_dict.get('raw_data', {})
                )
        
        except Exception as e:
            logger.error(f"Error getting latest tick: {str(e)}")
        
        return None

def example_callback(ticks: List[TickData]):
    """Example callback function for processing tick data."""
    if not ticks:
        return
    
    # Group by symbol
    by_symbol = defaultdict(list)
    for tick in ticks:
        by_symbol[tick.symbol].append(tick)
    
    # Process each symbol
    for symbol, symbol_ticks in by_symbol.items():
        trades = [t for t in symbol_ticks if t.message_type == MessageType.TRADE]
        quotes = [t for t in symbol_ticks if t.message_type == MessageType.QUOTE]
        
        logger.info(f"{symbol}: {len(trades)} trades, {len(quotes)} quotes")
        
        if trades:
            prices = [t.price for t in trades if t.price]
            if prices:
                logger.info(f"{symbol} price range: {min(prices):.4f} - {max(prices):.4f}")

def main():
    """Example usage of TickDataHandler."""
    # Create handler
    handler = TickDataHandler()
    
    # Add example data source (Binance)
    binance_config = DataSourceConfig(
        name="binance",
        url="wss://stream.binance.com:9443/ws/btcusdt@trade/btcusdt@bookTicker",
        symbols=["BTCUSDT", "ETHUSDT"],
        channels=["trade", "bookTicker"]
    )
    handler.add_data_source(binance_config)
    
    # Add callback
    handler.add_callback(example_callback)
    
    try:
        # Start processing
        handler.start()
        
        # Run for some time
        time.sleep(30)
        
        # Print statistics
        stats = handler.get_statistics()
        print(f"Statistics: {json.dumps(stats, indent=2, default=str)}")
        
    finally:
        # Clean shutdown
        handler.stop()

if __name__ == "__main__":
    main()