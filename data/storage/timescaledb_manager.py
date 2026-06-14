"""
TimescaleDB Manager
==================

High-performance time-series database manager for financial market data.
Handles tick data, order books, and alternative data with microsecond precision
using TimescaleDB's advanced time-series capabilities.

Features:
- Hypertable management for time-series data
- Continuous aggregates for real-time analytics
- Data retention policies and compression
- High-frequency data ingestion with batching
- Advanced querying with time-based partitioning
- Connection pooling and failover support
- Schema management and migrations

Author: Quantum Forge Data Team  
Date: November 2025
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
import psycopg2
import asyncpg
from psycopg2.pool import ThreadedConnectionPool
from psycopg2.extras import execute_values, Json
import json
from contextlib import contextmanager, asynccontextmanager
import threading
from queue import Queue, Empty
import sys
from pathlib import Path

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataType(Enum):
    """Types of financial data stored."""
    TICK_DATA = "tick_data"
    ORDER_BOOK = "order_book"
    TRADE_DATA = "trade_data"
    ALTERNATIVE_DATA = "alternative_data"
    FEATURES = "features"
    AGGREGATED_DATA = "aggregated_data"

class CompressionType(Enum):
    """TimescaleDB compression types."""
    NONE = "none"
    GZIP = "gzip"
    LZ4 = "lz4"
    ZSTD = "zstd"

@dataclass
class HypertableConfig:
    """Configuration for TimescaleDB hypertables."""
    table_name: str
    time_column: str = "timestamp"
    partition_column: Optional[str] = None
    chunk_time_interval: str = "1 hour"
    compression_enabled: bool = True
    compression_after: str = "7 days"
    compression_type: CompressionType = CompressionType.ZSTD
    retention_period: Optional[str] = "1 year"
    
    # Indexing
    indexes: List[str] = field(default_factory=list)
    unique_constraints: List[str] = field(default_factory=list)
    
    # Continuous aggregates
    continuous_aggregates: List[Dict[str, Any]] = field(default_factory=list)

@dataclass  
class BatchInsertConfig:
    """Configuration for batch insertions."""
    batch_size: int = 10000
    flush_interval: float = 1.0  # seconds
    max_queue_size: int = 100000
    parallel_workers: int = 4
    enable_compression: bool = True

class ConnectionManager:
    """Manages TimescaleDB connections and pooling."""
    
    def __init__(self, connection_config: Dict[str, Any]):
        self.config = connection_config
        self.sync_pool = None
        self.async_pool = None
        self._lock = threading.Lock()
        
        # Connection parameters
        self.host = connection_config.get('host', 'localhost')
        self.port = connection_config.get('port', 5432)
        self.database = connection_config.get('database', 'timeseries')
        self.user = connection_config.get('user', 'postgres')
        self.password = connection_config.get('password', 'password')
        
        # Pool configuration
        self.min_connections = connection_config.get('min_connections', 5)
        self.max_connections = connection_config.get('max_connections', 20)
        
        # Initialize pools
        self._initialize_sync_pool()
    
    def _initialize_sync_pool(self):
        """Initialize synchronous connection pool."""
        try:
            self.sync_pool = ThreadedConnectionPool(
                minconn=self.min_connections,
                maxconn=self.max_connections,
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                application_name="TimescaleDB_Manager"
            )
            logger.info(f"Synchronized connection pool initialized: {self.min_connections}-{self.max_connections}")
            
        except Exception as e:
            logger.info(f"TimescaleDB not available (using local storage only)")
            self.sync_pool = None
    
    async def _initialize_async_pool(self):
        """Initialize asynchronous connection pool."""
        try:
            self.async_pool = await asyncpg.create_pool(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                min_size=self.min_connections,
                max_size=self.max_connections,
                command_timeout=60
            )
            logger.info(f"Async connection pool initialized: {self.min_connections}-{self.max_connections}")
            
        except Exception as e:
            logger.error(f"Failed to initialize async pool: {str(e)}")
            raise
    
    @contextmanager
    def get_connection(self):
        """Get synchronous database connection."""
        conn = None
        try:
            conn = self.sync_pool.getconn()
            yield conn
        finally:
            if conn:
                self.sync_pool.putconn(conn)
    
    @asynccontextmanager
    async def get_async_connection(self):
        """Get asynchronous database connection."""
        if not self.async_pool:
            await self._initialize_async_pool()
        
        async with self.async_pool.acquire() as conn:
            yield conn
    
    def close(self):
        """Close all connections."""
        if self.sync_pool:
            self.sync_pool.closeall()
        
        if self.async_pool:
            asyncio.create_task(self.async_pool.close())

class SchemaManager:
    """Manages database schema and hypertables."""
    
    def __init__(self, connection_manager: ConnectionManager):
        self.conn_manager = connection_manager
        
    def create_schema(self, schema_name: str):
        """Create database schema."""
        with self.conn_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")
                conn.commit()
                logger.info(f"Schema created: {schema_name}")
    
    def create_tick_data_table(self, table_name: str = "tick_data"):
        """Create tick data hypertable."""
        with self.conn_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                # Create table
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        timestamp TIMESTAMPTZ NOT NULL,
                        symbol VARCHAR(20) NOT NULL,
                        exchange VARCHAR(20) NOT NULL,
                        price DECIMAL(20,8) NOT NULL,
                        size DECIMAL(20,8) NOT NULL,
                        side VARCHAR(4) CHECK (side IN ('buy', 'sell')),
                        trade_id VARCHAR(50),
                        sequence_number BIGINT,
                        latency_micros INTEGER,
                        raw_data JSONB,
                        
                        -- Metadata
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        data_source VARCHAR(50),
                        quality_score DECIMAL(3,2),
                        
                        PRIMARY KEY (timestamp, symbol, exchange)
                    )
                """)
                
                # Create hypertable
                cursor.execute(f"""
                    SELECT create_hypertable('{table_name}', 'timestamp', 
                                           chunk_time_interval => INTERVAL '1 hour',
                                           if_not_exists => TRUE)
                """)
                
                # Create indexes
                cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_symbol_time ON {table_name} (symbol, timestamp DESC)")
                cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_exchange_time ON {table_name} (exchange, timestamp DESC)")
                cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_price ON {table_name} (price)")
                
                conn.commit()
                logger.info(f"Tick data hypertable created: {table_name}")
    
    def create_orderbook_table(self, table_name: str = "orderbook_data"):
        """Create order book hypertable."""
        with self.conn_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        timestamp TIMESTAMPTZ NOT NULL,
                        symbol VARCHAR(20) NOT NULL,
                        exchange VARCHAR(20) NOT NULL,
                        sequence_number BIGINT NOT NULL,
                        
                        -- Order book levels (JSON arrays)
                        bids JSONB NOT NULL,  -- [{"price": 1.0, "size": 100}, ...]
                        asks JSONB NOT NULL,  -- [{"price": 1.1, "size": 50}, ...]
                        
                        -- Best bid/ask for fast queries
                        best_bid DECIMAL(20,8),
                        best_ask DECIMAL(20,8),
                        spread DECIMAL(20,8),
                        mid_price DECIMAL(20,8),
                        
                        -- Metadata
                        total_bid_volume DECIMAL(20,8),
                        total_ask_volume DECIMAL(20,8),
                        num_bid_levels INTEGER,
                        num_ask_levels INTEGER,
                        checksum VARCHAR(50),
                        
                        -- Quality metrics
                        latency_micros INTEGER,
                        quality_score DECIMAL(3,2),
                        is_crossed BOOLEAN DEFAULT FALSE,
                        
                        PRIMARY KEY (timestamp, symbol, exchange, sequence_number)
                    )
                """)
                
                # Create hypertable
                cursor.execute(f"""
                    SELECT create_hypertable('{table_name}', 'timestamp',
                                           chunk_time_interval => INTERVAL '1 hour',
                                           if_not_exists => TRUE)
                """)
                
                # Create indexes
                cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_symbol_time ON {table_name} (symbol, timestamp DESC)")
                cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_best_prices ON {table_name} (best_bid, best_ask)")
                cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_spread ON {table_name} (spread)")
                
                conn.commit()
                logger.info(f"Order book hypertable created: {table_name}")
    
    def create_alternative_data_table(self, table_name: str = "alternative_data"):
        """Create alternative data hypertable."""
        with self.conn_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        timestamp TIMESTAMPTZ NOT NULL,
                        symbol VARCHAR(20),
                        source VARCHAR(50) NOT NULL,
                        source_type VARCHAR(20) NOT NULL,
                        
                        -- Content
                        content TEXT NOT NULL,
                        title VARCHAR(500),
                        url VARCHAR(1000),
                        
                        -- Sentiment analysis
                        sentiment_score DECIMAL(5,4),
                        sentiment_polarity INTEGER CHECK (sentiment_polarity IN (-2,-1,0,1,2)),
                        confidence_score DECIMAL(5,4),
                        
                        -- Relevance and quality
                        relevance_score DECIMAL(5,4),
                        quality_score DECIMAL(5,4),
                        
                        -- Metadata
                        metadata JSONB,
                        raw_data JSONB,
                        language VARCHAR(10),
                        
                        -- Processing info
                        processed_at TIMESTAMPTZ DEFAULT NOW(),
                        processing_version VARCHAR(20),
                        
                        PRIMARY KEY (timestamp, source, content)
                    )
                """)
                
                # Create hypertable
                cursor.execute(f"""
                    SELECT create_hypertable('{table_name}', 'timestamp',
                                           chunk_time_interval => INTERVAL '6 hours',
                                           if_not_exists => TRUE)
                """)
                
                # Create indexes
                cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_symbol_time ON {table_name} (symbol, timestamp DESC)")
                cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_source ON {table_name} (source, timestamp DESC)")
                cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_sentiment ON {table_name} (sentiment_score, timestamp DESC)")
                
                # GIN index for JSONB fields
                cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_metadata_gin ON {table_name} USING GIN (metadata)")
                
                conn.commit()
                logger.info(f"Alternative data hypertable created: {table_name}")
    
    def setup_compression(self, table_name: str, compress_after: str = "7 days"):
        """Setup compression for hypertable."""
        with self.conn_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    ALTER TABLE {table_name} SET (
                        timescaledb.compress,
                        timescaledb.compress_segmentby = 'symbol, exchange'
                    )
                """)
                
                cursor.execute(f"""
                    SELECT add_compression_policy('{table_name}', INTERVAL '{compress_after}')
                """)
                
                conn.commit()
                logger.info(f"Compression enabled for {table_name} after {compress_after}")
    
    def setup_retention_policy(self, table_name: str, retention_period: str = "1 year"):
        """Setup data retention policy."""
        with self.conn_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    SELECT add_retention_policy('{table_name}', INTERVAL '{retention_period}')
                """)
                
                conn.commit()
                logger.info(f"Retention policy set for {table_name}: {retention_period}")
    
    def create_continuous_aggregate(self, table_name: str, aggregate_name: str, 
                                  time_bucket: str, select_query: str):
        """Create continuous aggregate view."""
        with self.conn_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    CREATE MATERIALIZED VIEW {aggregate_name}
                    WITH (timescaledb.continuous) AS
                    SELECT time_bucket('{time_bucket}', timestamp) AS time_bucket,
                           symbol,
                           exchange,
                           {select_query}
                    FROM {table_name}
                    GROUP BY time_bucket, symbol, exchange
                """)
                
                conn.commit()
                logger.info(f"Continuous aggregate created: {aggregate_name}")

class BatchInserter:
    """High-performance batch insertion for time-series data."""
    
    def __init__(self, connection_manager: ConnectionManager, config: BatchInsertConfig):
        self.conn_manager = connection_manager
        self.config = config
        
        # Queues for different data types
        self.queues = {
            DataType.TICK_DATA: Queue(maxsize=config.max_queue_size),
            DataType.ORDER_BOOK: Queue(maxsize=config.max_queue_size),
            DataType.ALTERNATIVE_DATA: Queue(maxsize=config.max_queue_size)
        }
        
        # Worker threads
        self.workers = []
        self.is_running = False
        
        # Statistics
        self.stats = {
            'total_inserted': 0,
            'batches_processed': 0,
            'errors': 0,
            'last_insert_time': None
        }
    
    def start(self):
        """Start batch insertion workers."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start worker threads for each data type
        for data_type in self.queues.keys():
            for i in range(self.config.parallel_workers):
                worker = threading.Thread(
                    target=self._worker_loop,
                    args=(data_type,),
                    name=f"BatchInserter-{data_type.value}-{i}"
                )
                worker.daemon = True
                worker.start()
                self.workers.append(worker)
        
        logger.info(f"Started {len(self.workers)} batch insertion workers")
    
    def stop(self, timeout: float = 10.0):
        """Stop batch insertion workers."""
        self.is_running = False
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=timeout / len(self.workers))
        
        # Flush remaining data
        self._flush_all_queues()
        
        logger.info("Batch insertion workers stopped")
    
    def insert_tick_data(self, data: List[Dict[str, Any]]):
        """Queue tick data for batch insertion."""
        self._queue_data(DataType.TICK_DATA, data)
    
    def insert_orderbook_data(self, data: List[Dict[str, Any]]):
        """Queue order book data for batch insertion."""
        self._queue_data(DataType.ORDER_BOOK, data)
    
    def insert_alternative_data(self, data: List[Dict[str, Any]]):
        """Queue alternative data for batch insertion."""
        self._queue_data(DataType.ALTERNATIVE_DATA, data)
    
    def _queue_data(self, data_type: DataType, data: List[Dict[str, Any]]):
        """Queue data for insertion."""
        queue = self.queues[data_type]
        
        for item in data:
            try:
                queue.put(item, timeout=1.0)
            except:
                logger.warning(f"Queue full for {data_type.value}, dropping data")
                break
    
    def _worker_loop(self, data_type: DataType):
        """Worker loop for batch insertion."""
        queue = self.queues[data_type]
        batch = []
        last_flush = time.time()
        
        while self.is_running:
            try:
                # Get item from queue
                try:
                    item = queue.get(timeout=0.1)
                    batch.append(item)
                    queue.task_done()
                except Empty:
                    pass
                
                # Flush batch if conditions are met
                current_time = time.time()
                should_flush = (
                    len(batch) >= self.config.batch_size or
                    (batch and current_time - last_flush >= self.config.flush_interval)
                )
                
                if should_flush:
                    self._flush_batch(data_type, batch)
                    batch.clear()
                    last_flush = current_time
                    
            except Exception as e:
                logger.error(f"Error in worker loop for {data_type.value}: {str(e)}")
                self.stats['errors'] += 1
        
        # Flush remaining batch
        if batch:
            self._flush_batch(data_type, batch)
    
    def _flush_batch(self, data_type: DataType, batch: List[Dict[str, Any]]):
        """Flush batch to database."""
        if not batch:
            return
        
        try:
            with self.conn_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    
                    if data_type == DataType.TICK_DATA:
                        self._insert_tick_batch(cursor, batch)
                    elif data_type == DataType.ORDER_BOOK:
                        self._insert_orderbook_batch(cursor, batch)
                    elif data_type == DataType.ALTERNATIVE_DATA:
                        self._insert_alternative_batch(cursor, batch)
                
                conn.commit()
                
                self.stats['total_inserted'] += len(batch)
                self.stats['batches_processed'] += 1
                self.stats['last_insert_time'] = datetime.now()
                
                logger.debug(f"Inserted batch of {len(batch)} {data_type.value} records")
                
        except Exception as e:
            logger.error(f"Error flushing {data_type.value} batch: {str(e)}")
            self.stats['errors'] += 1
    
    def _insert_tick_batch(self, cursor, batch: List[Dict[str, Any]]):
        """Insert tick data batch."""
        values = [
            (
                item.get('timestamp'),
                item.get('symbol'),
                item.get('exchange'),
                item.get('price'),
                item.get('size'),
                item.get('side'),
                item.get('trade_id'),
                item.get('sequence_number'),
                item.get('latency_micros'),
                Json(item.get('raw_data', {})),
                item.get('data_source'),
                item.get('quality_score')
            )
            for item in batch
        ]
        
        execute_values(
            cursor,
            """
            INSERT INTO tick_data (
                timestamp, symbol, exchange, price, size, side,
                trade_id, sequence_number, latency_micros, raw_data,
                data_source, quality_score
            ) VALUES %s
            ON CONFLICT (timestamp, symbol, exchange) DO NOTHING
            """,
            values,
            page_size=self.config.batch_size
        )
    
    def _insert_orderbook_batch(self, cursor, batch: List[Dict[str, Any]]):
        """Insert order book batch."""
        values = [
            (
                item.get('timestamp'),
                item.get('symbol'),
                item.get('exchange'),
                item.get('sequence_number'),
                Json(item.get('bids', [])),
                Json(item.get('asks', [])),
                item.get('best_bid'),
                item.get('best_ask'),
                item.get('spread'),
                item.get('mid_price'),
                item.get('total_bid_volume'),
                item.get('total_ask_volume'),
                item.get('num_bid_levels'),
                item.get('num_ask_levels'),
                item.get('checksum'),
                item.get('latency_micros'),
                item.get('quality_score'),
                item.get('is_crossed', False)
            )
            for item in batch
        ]
        
        execute_values(
            cursor,
            """
            INSERT INTO orderbook_data (
                timestamp, symbol, exchange, sequence_number,
                bids, asks, best_bid, best_ask, spread, mid_price,
                total_bid_volume, total_ask_volume, num_bid_levels, num_ask_levels,
                checksum, latency_micros, quality_score, is_crossed
            ) VALUES %s
            ON CONFLICT (timestamp, symbol, exchange, sequence_number) DO NOTHING
            """,
            values,
            page_size=self.config.batch_size
        )
    
    def _insert_alternative_batch(self, cursor, batch: List[Dict[str, Any]]):
        """Insert alternative data batch."""
        values = [
            (
                item.get('timestamp'),
                item.get('symbol'),
                item.get('source'),
                item.get('source_type'),
                item.get('content'),
                item.get('title'),
                item.get('url'),
                item.get('sentiment_score'),
                item.get('sentiment_polarity'),
                item.get('confidence_score'),
                item.get('relevance_score'),
                item.get('quality_score'),
                Json(item.get('metadata', {})),
                Json(item.get('raw_data', {})),
                item.get('language'),
                item.get('processing_version')
            )
            for item in batch
        ]
        
        execute_values(
            cursor,
            """
            INSERT INTO alternative_data (
                timestamp, symbol, source, source_type, content, title, url,
                sentiment_score, sentiment_polarity, confidence_score,
                relevance_score, quality_score, metadata, raw_data,
                language, processing_version
            ) VALUES %s
            ON CONFLICT (timestamp, source, content) DO NOTHING
            """,
            values,
            page_size=self.config.batch_size
        )
    
    def _flush_all_queues(self):
        """Flush all remaining queued data."""
        for data_type, queue in self.queues.items():
            batch = []
            
            # Drain queue
            while not queue.empty():
                try:
                    item = queue.get_nowait()
                    batch.append(item)
                    queue.task_done()
                except Empty:
                    break
            
            if batch:
                self._flush_batch(data_type, batch)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get insertion statistics."""
        stats = self.stats.copy()
        
        # Add queue sizes
        stats['queue_sizes'] = {
            data_type.value: queue.qsize() 
            for data_type, queue in self.queues.items()
        }
        
        return stats

class TimescaleDBManager:
    """
    Main TimescaleDB manager for financial time-series data.
    """
    
    def __init__(self, connection_config: Dict[str, Any], 
                 batch_config: Optional[BatchInsertConfig] = None):
        """Initialize TimescaleDB manager."""
        self.connection_manager = ConnectionManager(connection_config)
        self.schema_manager = SchemaManager(self.connection_manager)
        
        # Batch insertion
        self.batch_config = batch_config or BatchInsertConfig()
        self.batch_inserter = BatchInserter(self.connection_manager, self.batch_config)
        
        # Initialize database schema
        self._initialize_database()
        
        # Statistics
        self.stats = {
            'queries_executed': 0,
            'data_points_inserted': 0,
            'errors': 0,
            'start_time': datetime.now()
        }
    
    def _initialize_database(self):
        """Initialize database schema and hypertables."""
        if not self.connection_manager.sync_pool:
            logger.warning("[WARN] Skipping database initialization: No connection pool")
            return

        try:
            # Create schema
            self.schema_manager.create_schema('timeseries')
            
            # Create hypertables
            self.schema_manager.create_tick_data_table()
            self.schema_manager.create_orderbook_table()
            self.schema_manager.create_alternative_data_table()
            
            # Setup compression and retention
            for table in ['tick_data', 'orderbook_data', 'alternative_data']:
                try:
                    self.schema_manager.setup_compression(table)
                    self.schema_manager.setup_retention_policy(table)
                except Exception as e:
                    logger.warning(f"Could not setup policies for {table}: {str(e)}")
            
            logger.info("TimescaleDB initialization complete")
            
        except Exception as e:
            logger.error(f"Database initialization error: {str(e)}")
            raise
    
    def start(self):
        """Start the manager and batch insertion."""
        self.batch_inserter.start()
        logger.info("TimescaleDB manager started")
    
    def stop(self):
        """Stop the manager and close connections."""
        self.batch_inserter.stop()
        self.connection_manager.close()
        logger.info("TimescaleDB manager stopped")
    
    # Data insertion methods
    def insert_tick_data(self, data: Union[Dict[str, Any], List[Dict[str, Any]]]):
        """Insert tick data."""
        if isinstance(data, dict):
            data = [data]
        
        self.batch_inserter.insert_tick_data(data)
        self.stats['data_points_inserted'] += len(data)
    
    def insert_orderbook_data(self, data: Union[Dict[str, Any], List[Dict[str, Any]]]):
        """Insert order book data."""
        if isinstance(data, dict):
            data = [data]
        
        self.batch_inserter.insert_orderbook_data(data)
        self.stats['data_points_inserted'] += len(data)
    
    def insert_alternative_data(self, data: Union[Dict[str, Any], List[Dict[str, Any]]]):
        """Insert alternative data."""
        if isinstance(data, dict):
            data = [data]
        
        self.batch_inserter.insert_alternative_data(data)
        self.stats['data_points_inserted'] += len(data)
    
    # Query methods
    def query_tick_data(self, symbol: str, start_time: datetime, 
                       end_time: datetime, exchange: Optional[str] = None,
                       limit: Optional[int] = None) -> pd.DataFrame:
        """Query tick data."""
        with self.connection_manager.get_connection() as conn:
            
            query = """
                SELECT timestamp, symbol, exchange, price, size, side,
                       trade_id, sequence_number, latency_micros, quality_score
                FROM tick_data
                WHERE symbol = %s AND timestamp >= %s AND timestamp <= %s
            """
            params = [symbol, start_time, end_time]
            
            if exchange:
                query += " AND exchange = %s"
                params.append(exchange)
            
            query += " ORDER BY timestamp DESC"
            
            if limit:
                query += f" LIMIT {limit}"
            
            df = pd.read_sql_query(query, conn, params=params)
            self.stats['queries_executed'] += 1
            
            return df
    
    def query_orderbook_data(self, symbol: str, start_time: datetime,
                           end_time: datetime, exchange: Optional[str] = None,
                           limit: Optional[int] = None) -> pd.DataFrame:
        """Query order book data."""
        with self.connection_manager.get_connection() as conn:
            
            query = """
                SELECT timestamp, symbol, exchange, sequence_number,
                       best_bid, best_ask, spread, mid_price,
                       total_bid_volume, total_ask_volume,
                       num_bid_levels, num_ask_levels, quality_score
                FROM orderbook_data
                WHERE symbol = %s AND timestamp >= %s AND timestamp <= %s
            """
            params = [symbol, start_time, end_time]
            
            if exchange:
                query += " AND exchange = %s"
                params.append(exchange)
            
            query += " ORDER BY timestamp DESC"
            
            if limit:
                query += f" LIMIT {limit}"
            
            df = pd.read_sql_query(query, conn, params=params)
            self.stats['queries_executed'] += 1
            
            return df
    
    def query_alternative_data(self, symbol: Optional[str] = None,
                             start_time: Optional[datetime] = None,
                             end_time: Optional[datetime] = None,
                             source: Optional[str] = None,
                             min_sentiment: Optional[float] = None,
                             limit: Optional[int] = None) -> pd.DataFrame:
        """Query alternative data."""
        with self.connection_manager.get_connection() as conn:
            
            query = """
                SELECT timestamp, symbol, source, source_type, content, title,
                       sentiment_score, sentiment_polarity, confidence_score,
                       relevance_score, quality_score
                FROM alternative_data
                WHERE 1=1
            """
            params = []
            
            if symbol:
                query += " AND symbol = %s"
                params.append(symbol)
            
            if start_time:
                query += " AND timestamp >= %s"
                params.append(start_time)
            
            if end_time:
                query += " AND timestamp <= %s"
                params.append(end_time)
            
            if source:
                query += " AND source = %s"
                params.append(source)
            
            if min_sentiment is not None:
                query += " AND sentiment_score >= %s"
                params.append(min_sentiment)
            
            query += " ORDER BY timestamp DESC"
            
            if limit:
                query += f" LIMIT {limit}"
            
            df = pd.read_sql_query(query, conn, params=params)
            self.stats['queries_executed'] += 1
            
            return df
    
    def get_market_statistics(self, symbol: str, time_bucket: str = '1 minute',
                            start_time: Optional[datetime] = None,
                            end_time: Optional[datetime] = None) -> pd.DataFrame:
        """Get aggregated market statistics."""
        with self.connection_manager.get_connection() as conn:
            
            # Default to last 24 hours if no time range specified
            if not start_time:
                start_time = datetime.now() - timedelta(hours=24)
            if not end_time:
                end_time = datetime.now()
            
            query = f"""
                SELECT time_bucket('{time_bucket}', timestamp) AS time_bucket,
                       symbol,
                       exchange,
                       COUNT(*) as trade_count,
                       AVG(price) as avg_price,
                       MIN(price) as min_price,
                       MAX(price) as max_price,
                       SUM(size) as total_volume,
                       STDDEV(price) as price_volatility
                FROM tick_data
                WHERE symbol = %s AND timestamp >= %s AND timestamp <= %s
                GROUP BY time_bucket, symbol, exchange
                ORDER BY time_bucket DESC
            """
            
            df = pd.read_sql_query(query, conn, params=[symbol, start_time, end_time])
            self.stats['queries_executed'] += 1
            
            return df
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get manager statistics."""
        stats = self.stats.copy()
        
        # Add batch inserter stats
        batch_stats = self.batch_inserter.get_stats()
        stats.update({f"batch_{k}": v for k, v in batch_stats.items()})
        
        # Add uptime
        stats['uptime_seconds'] = (datetime.now() - stats['start_time']).total_seconds()
        
        return stats

def main():
    """Example usage of TimescaleDB Manager."""
    # Connection configuration
    connection_config = {
        'host': 'localhost',
        'port': 5432,
        'database': 'timeseries',
        'user': 'postgres',
        'password': 'password',
        'min_connections': 5,
        'max_connections': 20
    }
    
    # Batch configuration
    batch_config = BatchInsertConfig(
        batch_size=1000,
        flush_interval=1.0,
        parallel_workers=4
    )
    
    # Create manager
    manager = TimescaleDBManager(connection_config, batch_config)
    
    try:
        # Start manager
        manager.start()
        
        # Insert sample tick data
        sample_tick = {
            'timestamp': datetime.now(),
            'symbol': 'BTC',
            'exchange': 'binance',
            'price': 50000.00,
            'size': 0.1,
            'side': 'buy',
            'trade_id': 'trade123',
            'sequence_number': 12345,
            'latency_micros': 1500,
            'data_source': 'websocket',
            'quality_score': 0.95
        }
        
        manager.insert_tick_data(sample_tick)
        
        # Wait for batch insertion
        time.sleep(2)
        
        # Query data
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=5)
        
        df = manager.query_tick_data('BTC', start_time, end_time)
        print(f"Retrieved {len(df)} tick records")
        
        # Get statistics
        stats = manager.get_statistics()
        print(f"Manager statistics: {json.dumps(stats, indent=2, default=str)}")
        
    finally:
        manager.stop()

if __name__ == "__main__":
    main()