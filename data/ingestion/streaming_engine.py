"""
Real-time Data Streaming Engine for QUANTUM-FORGE
High-performance streaming data processing with multiple data source integration.
"""

import asyncio
import websockets
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Any, Optional, Callable, Union
import threading
import queue
import time
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import sqlite3
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, Float, String, DateTime
from sqlalchemy.orm import sessionmaker
import redis
import warnings
warnings.filterwarnings('ignore')

@dataclass
class MarketDataPoint:
    """Market data point structure."""
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    last: float
    volume: int
    open_price: float
    high: float
    low: float
    close: float
    vwap: float = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

@dataclass
class TradeData:
    """Trade data structure."""
    symbol: str
    timestamp: datetime
    price: float
    volume: int
    side: str  # 'buy' or 'sell'
    trade_id: str
    venue: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

class DataStreamManager:
    """Real-time data streaming management system."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize data stream manager."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.config = config or self._default_config()
        self.subscribers = {}
        self.data_cache = {}
        self.running = False
        
        # Initialize storage systems
        self._init_storage()
        
        # Initialize data sources
        self.data_sources = {
            'market_data': MarketDataStream(self.config),
            'trade_data': TradeDataStream(self.config),
            'news_feed': NewsFeedStream(self.config),
            'economic_data': EconomicDataStream(self.config)
        }
        
        # Message queues for different data types
        self.data_queues = {
            'market_data': queue.Queue(maxsize=10000),
            'trade_data': queue.Queue(maxsize=10000),
            'news_feed': queue.Queue(maxsize=1000),
            'economic_data': queue.Queue(maxsize=1000)
        }
        
        # Processing threads
        self.processor_threads = {}
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration."""
        return {
            'redis_host': 'localhost',
            'redis_port': 6379,
            'database_url': 'sqlite:///quantum_forge_data.db',
            'buffer_size': 10000,
            'batch_size': 100,
            'flush_interval': 5,
            'data_retention_days': 365,
            'sources': {
                'alpha_vantage_key': 'demo',
                'iex_token': 'demo',
                'news_api_key': 'demo'
            }
        }
    
    def _init_storage(self):
        """Initialize storage systems."""
        try:
            # Redis for real-time caching
            self.redis_client = redis.Redis(
                host=self.config['redis_host'],
                port=self.config['redis_port'],
                decode_responses=True
            )
            
            # SQLAlchemy for persistent storage
            self.engine = create_engine(self.config['database_url'])
            self.Session = sessionmaker(bind=self.engine)
            
            # Create tables if they don't exist
            self._create_tables()
            
            self.logger.info("[OK] Storage systems initialized")
            
        except Exception as e:
            self.logger.error(f"[ERR] Storage initialization failed: {e}")
            # Fallback to in-memory storage
            self.redis_client = None
            self.engine = None
    
    def _create_tables(self):
        """Create database tables."""
        metadata = MetaData()
        
        # Market data table
        market_data_table = Table(
            'market_data', metadata,
            Column('id', Integer, primary_key=True),
            Column('symbol', String(10)),
            Column('timestamp', DateTime),
            Column('bid', Float),
            Column('ask', Float),
            Column('last', Float),
            Column('volume', Integer),
            Column('open_price', Float),
            Column('high', Float),
            Column('low', Float),
            Column('close', Float),
            Column('vwap', Float)
        )
        
        # Trade data table
        trade_data_table = Table(
            'trade_data', metadata,
            Column('id', Integer, primary_key=True),
            Column('symbol', String(10)),
            Column('timestamp', DateTime),
            Column('price', Float),
            Column('volume', Integer),
            Column('side', String(4)),
            Column('trade_id', String(50)),
            Column('venue', String(20))
        )
        
        if self.engine:
            metadata.create_all(self.engine)
    
    async def subscribe(self, data_type: str, callback: Callable):
        """Subscribe to data stream."""
        if data_type not in self.subscribers:
            self.subscribers[data_type] = []
        
        self.subscribers[data_type].append(callback)
        self.logger.info(f"[OK] Subscribed to {data_type} stream")
    
    async def unsubscribe(self, data_type: str, callback: Callable):
        """Unsubscribe from data stream."""
        if data_type in self.subscribers:
            if callback in self.subscribers[data_type]:
                self.subscribers[data_type].remove(callback)
                self.logger.info(f"[OK] Unsubscribed from {data_type} stream")
    
    def start_streaming(self):
        """Start all data streams."""
        if self.running:
            self.logger.warning("[WARN] Data streaming already running")
            return
        
        self.running = True
        self.logger.info("[INIT] Starting QUANTUM-FORGE data streaming...")
        
        # Start data source streams
        for source_name, source in self.data_sources.items():
            thread = threading.Thread(
                target=self._run_data_source,
                args=(source_name, source),
                daemon=True
            )
            thread.start()
            self.processor_threads[source_name] = thread
        
        # Start data processors
        for data_type in self.data_queues.keys():
            thread = threading.Thread(
                target=self._process_data_queue,
                args=(data_type,),
                daemon=True
            )
            thread.start()
            self.processor_threads[f"{data_type}_processor"] = thread
        
        # Start storage flush thread
        flush_thread = threading.Thread(
            target=self._flush_storage,
            daemon=True
        )
        flush_thread.start()
        self.processor_threads['flush'] = flush_thread
        
        self.logger.info("[OK] All data streams started successfully")
    
    def stop_streaming(self):
        """Stop all data streams."""
        self.running = False
        self.logger.info("[STOP] Stopping data streaming...")
        
        # Wait for threads to finish
        for thread in self.processor_threads.values():
            if thread.is_alive():
                thread.join(timeout=5)
        
        self.logger.info("[OK] Data streaming stopped")
    
    def _run_data_source(self, source_name: str, source):
        """Run individual data source."""
        while self.running:
            try:
                data = source.get_data()
                if data:
                    self.data_queues[source_name].put(data, timeout=1)
                time.sleep(source.update_interval)
                
            except Exception as e:
                self.logger.error(f"[ERR] Error in {source_name}: {e}")
                time.sleep(5)  # Wait before retry
    
    def _process_data_queue(self, data_type: str):
        """Process data from queue."""
        batch = []
        last_flush = time.time()
        
        while self.running:
            try:
                # Get data from queue with timeout
                try:
                    data = self.data_queues[data_type].get(timeout=1)
                    batch.append(data)
                except queue.Empty:
                    continue
                
                # Process batch if it's full or time to flush
                current_time = time.time()
                if (len(batch) >= self.config['batch_size'] or 
                    current_time - last_flush > self.config['flush_interval']):
                    
                    if batch:
                        self._notify_subscribers(data_type, batch)
                        self._store_batch(data_type, batch)
                        batch = []
                        last_flush = current_time
                
            except Exception as e:
                self.logger.error(f"[ERR] Error processing {data_type}: {e}")
    
    def _notify_subscribers(self, data_type: str, batch: List[Any]):
        """Notify subscribers of new data."""
        if data_type in self.subscribers:
            for callback in self.subscribers[data_type]:
                try:
                    callback(batch)
                except Exception as e:
                    self.logger.error(f"[ERR] Subscriber callback error: {e}")
    
    def _store_batch(self, data_type: str, batch: List[Any]):
        """Store data batch."""
        try:
            # Store in Redis cache
            if self.redis_client:
                for data_point in batch[-100:]:  # Keep last 100 points in cache
                    key = f"{data_type}:{data_point.symbol if hasattr(data_point, 'symbol') else 'general'}"
                    self.redis_client.lpush(key, json.dumps(data_point.to_dict(), default=str))
                    self.redis_client.ltrim(key, 0, 999)  # Keep only last 1000 points
            
            # Store in persistent database
            if self.engine and data_type in ['market_data', 'trade_data']:
                self._store_in_database(data_type, batch)
                
        except Exception as e:
            self.logger.error(f"  Storage error for {data_type}: {e}")
    
    def _store_in_database(self, data_type: str, batch: List[Any]):
        """Store batch in database."""
        try:
            session = self.Session()
            
            for data_point in batch:
                if data_type == 'market_data':
                    # Insert market data
                    session.execute(
                        "INSERT INTO market_data (symbol, timestamp, bid, ask, last, volume, "
                        "open_price, high, low, close, vwap) VALUES "
                        "(:symbol, :timestamp, :bid, :ask, :last, :volume, :open_price, "
                        ":high, :low, :close, :vwap)",
                        data_point.to_dict()
                    )
                elif data_type == 'trade_data':
                    # Insert trade data
                    session.execute(
                        "INSERT INTO trade_data (symbol, timestamp, price, volume, side, "
                        "trade_id, venue) VALUES "
                        "(:symbol, :timestamp, :price, :volume, :side, :trade_id, :venue)",
                        data_point.to_dict()
                    )
            
            session.commit()
            session.close()
            
        except Exception as e:
            self.logger.error(f"  Database storage error: {e}")
            if 'session' in locals():
                session.rollback()
                session.close()
    
    def _flush_storage(self):
        """Periodic storage maintenance."""
        while self.running:
            try:
                # Clean old data
                cutoff_date = datetime.now() - timedelta(days=self.config['data_retention_days'])
                
                if self.engine:
                    session = self.Session()
                    session.execute(
                        "DELETE FROM market_data WHERE timestamp < :cutoff",
                        {'cutoff': cutoff_date}
                    )
                    session.execute(
                        "DELETE FROM trade_data WHERE timestamp < :cutoff",
                        {'cutoff': cutoff_date}
                    )
                    session.commit()
                    session.close()
                
                # Trim Redis caches
                if self.redis_client:
                    for key in self.redis_client.scan_iter(match="*:*"):
                        self.redis_client.ltrim(key, 0, 999)
                
                self.logger.info("  Storage maintenance completed")
                
            except Exception as e:
                self.logger.error(f"  Storage maintenance error: {e}")
            
            # Wait 1 hour before next maintenance
            time.sleep(3600)
    
    def get_latest_data(self, data_type: str, symbol: str = None, limit: int = 100) -> List[Dict]:
        """Get latest data from cache."""
        try:
            if self.redis_client:
                key = f"{data_type}:{symbol or 'general'}"
                data = self.redis_client.lrange(key, 0, limit - 1)
                return [json.loads(item) for item in data]
            else:
                # Fallback to in-memory cache
                cache_key = f"{data_type}:{symbol or 'general'}"
                return self.data_cache.get(cache_key, [])[-limit:]
                
        except Exception as e:
            self.logger.error(f"  Error retrieving data: {e}")
            return []
    
    def get_historical_data(self, data_type: str, symbol: str, 
                           start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get historical data from database."""
        try:
            if not self.engine or data_type not in ['market_data', 'trade_data']:
                return pd.DataFrame()
            
            query = f"""
                SELECT * FROM {data_type} 
                WHERE symbol = :symbol 
                AND timestamp BETWEEN :start_date AND :end_date
                ORDER BY timestamp
            """
            
            return pd.read_sql(
                query, 
                self.engine,
                params={
                    'symbol': symbol,
                    'start_date': start_date,
                    'end_date': end_date
                }
            )
            
        except Exception as e:
            self.logger.error(f"  Error retrieving historical data: {e}")
            return pd.DataFrame()

class MarketDataStream:
    """Market data streaming source."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize market data stream."""
        self.config = config
        self.update_interval = 1  # 1 second
        self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA']
        
    def get_data(self) -> MarketDataPoint:
        """Generate synthetic market data."""
        symbol = np.random.choice(self.symbols)
        base_price = np.random.uniform(100, 300)
        
        # Generate realistic market data
        bid = base_price - np.random.uniform(0.01, 0.05)
        ask = base_price + np.random.uniform(0.01, 0.05)
        last = np.random.uniform(bid, ask)
        volume = np.random.randint(100, 10000)
        
        return MarketDataPoint(
            symbol=symbol,
            timestamp=datetime.now(),
            bid=bid,
            ask=ask,
            last=last,
            volume=volume,
            open_price=base_price * np.random.uniform(0.99, 1.01),
            high=base_price * np.random.uniform(1.00, 1.03),
            low=base_price * np.random.uniform(0.97, 1.00),
            close=last,
            vwap=base_price * np.random.uniform(0.995, 1.005)
        )

class TradeDataStream:
    """Trade data streaming source."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize trade data stream."""
        self.config = config
        self.update_interval = 0.5  # 500ms
        self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA']
        self.venues = ['NYSE', 'NASDAQ', 'BATS', 'EDGX']
        
    def get_data(self) -> TradeData:
        """Generate synthetic trade data."""
        symbol = np.random.choice(self.symbols)
        
        return TradeData(
            symbol=symbol,
            timestamp=datetime.now(),
            price=np.random.uniform(100, 300),
            volume=np.random.randint(100, 5000),
            side=np.random.choice(['buy', 'sell']),
            trade_id=f"T{int(time.time()*1000)}{np.random.randint(100, 999)}",
            venue=np.random.choice(self.venues)
        )

class NewsFeedStream:
    """News feed streaming source."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize news feed stream."""
        self.config = config
        self.update_interval = 30  # 30 seconds
        
    def get_data(self) -> Dict[str, Any]:
        """Generate synthetic news data."""
        headlines = [
            "Market reaches new highs amid strong earnings",
            "Fed considers interest rate adjustments",
            "Technology sector shows resilience",
            "Energy stocks surge on commodity prices",
            "Global markets respond to economic data"
        ]
        
        return {
            'timestamp': datetime.now(),
            'headline': np.random.choice(headlines),
            'sentiment': np.random.uniform(-1, 1),
            'relevance': np.random.uniform(0, 1),
            'source': 'Financial News Wire'
        }

class EconomicDataStream:
    """Economic data streaming source."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize economic data stream."""
        self.config = config
        self.update_interval = 300  # 5 minutes
        
    def get_data(self) -> Dict[str, Any]:
        """Generate synthetic economic data."""
        indicators = ['GDP', 'CPI', 'Unemployment', 'Consumer Confidence', 'PMI']
        
        return {
            'timestamp': datetime.now(),
            'indicator': np.random.choice(indicators),
            'value': np.random.uniform(0, 100),
            'previous': np.random.uniform(0, 100),
            'forecast': np.random.uniform(0, 100),
            'impact': np.random.choice(['High', 'Medium', 'Low'])
        }

# Example usage and testing
if __name__ == "__main__":
    # Initialize data stream manager
    manager = DataStreamManager()
    
    # Example subscriber callback
    def market_data_callback(data_batch):
        print(f"  Received {len(data_batch)} market data points")
        for data in data_batch[-3:]:  # Show last 3 points
            print(f"   {data.symbol}: ${data.last:.2f} @ {data.timestamp}")
    
    def trade_data_callback(data_batch):
        print(f"  Received {len(data_batch)} trades")
        for trade in data_batch[-2:]:  # Show last 2 trades
            print(f"   {trade.symbol}: {trade.volume} @ ${trade.price:.2f} ({trade.side})")
    
    # Subscribe to data streams
    async def setup_subscriptions():
        await manager.subscribe('market_data', market_data_callback)
        await manager.subscribe('trade_data', trade_data_callback)
    
    # Run setup
    asyncio.run(setup_subscriptions())
    
    # Start streaming
    manager.start_streaming()
    
    try:
        # Let it run for demonstration
        time.sleep(30)
        
        # Test data retrieval
        print("\n  Latest market data:")
        latest = manager.get_latest_data('market_data', 'AAPL', 5)
        for data in latest:
            print(f"   AAPL: ${data['last']:.2f} @ {data['timestamp']}")
        
    except KeyboardInterrupt:
        print("\n  Stopping data streaming...")
    finally:
        manager.stop_streaming()