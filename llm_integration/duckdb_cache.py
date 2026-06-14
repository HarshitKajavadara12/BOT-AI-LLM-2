
"""DuckDB Fast Cache for QUANTUM-FORGE Trading Data

This module provides an in-memory analytical database for ultra-fast queries on
trading data. It maintains a synchronized copy of portfolio state, market data,
and trading history optimized for sub-millisecond analytical queries.

Responsibilities:
    - Maintain in-memory copy of trading data for fast analytics
    - Provide SQL query interface for complex aggregations
    - Support historical trade analysis
    - Enable real-time performance metrics calculation
    - Store trading signals for pattern analysis

Inputs:
    - Portfolio state synced from DynamicPortfolioTracker
    - Trade executions from core trading engine
    - Market data snapshots from data feeds
    - Trading signals from strategy modules
    - Position updates and P&L calculations

Outputs:
    - SQL query results (pandas DataFrames)
    - Aggregated statistics (win rate, P&L, Sharpe ratio, etc.)
    - Historical trade summaries
    - Position history and performance metrics
    - Signal analysis and backtesting data

CRITICAL CONSTRAINT:
    No AI/LLM is allowed to modify execution or risk decisions here.
    
    This cache is a READ-ONLY analytical layer. It stores copies of data
    for fast querying but has ZERO influence on trading operations.
    
    Queries against this cache cannot:
    - Execute trades
    - Modify positions or orders
    - Change risk limits or parameters
    - Trigger strategy actions
    - Override portfolio decisions
    
    All data is historical or snapshot-based for analysis only.

Performance:
    - Query latency: <1ms for simple aggregations
    - Indexing: Optimized for time-series and symbol lookups
    - Memory: In-memory database for maximum speed
    - Persistence: Optional disk backing for durability

Schema:
    - market_data: Price, volume, spread snapshots
    - trades: Execution history with P&L
    - positions: Current and historical positions
    - signals: Trading signals and confidence levels

Thread Safety:
    All database operations are protected by threading locks.
"""

import duckdb
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
import threading


class TradingDataCache:
    """DuckDB-powered fast cache for QUANTUM-FORGE data"""
    
    def __init__(self, db_path=':memory:'):
        """
        Initialize DuckDB cache
        
        Args:
            db_path: Database path (':memory:' for in-memory, or file path)
        """
        self.conn = duckdb.connect(db_path)
        self.lock = threading.Lock()
        self._setup_schema()
        print("[INFO] DuckDB cache initialized")
    
    def _setup_schema(self):
        """Create optimized tables for trading data"""
        with self.lock:
            # Market data table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    symbol VARCHAR,
                    timestamp TIMESTAMP,
                    price DOUBLE,
                    volume DOUBLE,
                    bid DOUBLE,
                    ask DOUBLE,
                    spread DOUBLE,
                    PRIMARY KEY (symbol, timestamp)
                )
            """)
            
            # Trades table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id VARCHAR PRIMARY KEY,
                    symbol VARCHAR,
                    side VARCHAR,
                    quantity DOUBLE,
                    price DOUBLE,
                    timestamp TIMESTAMP,
                    pnl DOUBLE,
                    entry_price DOUBLE,
                    metadata VARCHAR
                )
            """)
            
            # Positions table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    symbol VARCHAR PRIMARY KEY,
                    quantity DOUBLE,
                    entry_price DOUBLE,
                    current_value DOUBLE,
                    unrealized_pnl DOUBLE,
                    realized_pnl DOUBLE,
                    last_update TIMESTAMP
                )
            """)
            
            # Signals table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    signal_id VARCHAR PRIMARY KEY,
                    symbol VARCHAR,
                    signal_type VARCHAR,
                    confidence DOUBLE,
                    timestamp TIMESTAMP,
                    metadata VARCHAR
                )
            """)
            
            # Create indexes for fast queries
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp)")
    
    def sync_from_tracker(self, tracker):
        """
        Sync data from QUANTUM-FORGE tracker
        
        Args:
            tracker: DynamicPortfolioTracker instance
        """
        with self.lock:
            try:
                # Sync positions
                positions = tracker.get_positions()
                for symbol, pos in positions.items():
                    current_price = getattr(pos, 'current_price', 0)
                    if current_price == 0:
                        # Try to get from tracker's cache if available
                        try:
                            current_price = pos.entry_price
                        except:
                            current_price = 0
                    
                    self.conn.execute("""
                        INSERT OR REPLACE INTO positions VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, [
                        symbol,
                        pos.amount,
                        pos.entry_price,
                        pos.amount * current_price,
                        getattr(pos, 'unrealized_pnl', 0),
                        getattr(pos, 'realized_pnl', 0),
                        datetime.now()
                    ])
                
                # Sync trades from trade history
                if hasattr(tracker, 'trade_history') and tracker.trade_history:
                    # Get last synced trade ID
                    result = self.conn.execute("SELECT MAX(trade_id) FROM trades").fetchone()
                    last_synced_count = len(self.conn.execute("SELECT COUNT(*) FROM trades").fetchone())
                    
                    # Only sync new trades
                    for i, trade in enumerate(tracker.trade_history):
                        if i < last_synced_count:
                            continue
                        
                        trade_id = f"TRADE_{int(trade.timestamp.timestamp() * 1000)}_{i}"
                        
                        self.conn.execute("""
                            INSERT OR IGNORE INTO trades VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, [
                            trade_id,
                            trade.symbol,
                            trade.side,
                            trade.quantity,
                            trade.price,
                            trade.timestamp,
                            getattr(trade, 'pnl', 0),
                            getattr(trade, 'entry_price', trade.price),
                            '{}'
                        ])
                
            except Exception as e:
                print(f" ️ Sync error: {e}")
    
    def sync_market_data(self, symbol: str, price: float, volume: float, bid: float, ask: float):
        """Sync real-time market data"""
        with self.lock:
            try:
                spread = ask - bid if ask > bid else 0
                self.conn.execute("""
                    INSERT OR REPLACE INTO market_data VALUES (?, ?, ?, ?, ?, ?, ?)
                """, [symbol, datetime.now(), price, volume, bid, ask, spread])
            except Exception as e:
                print(f" ️ Market data sync error: {e}")
    
    def add_signal(self, symbol: str, signal_type: str, confidence: float, metadata: str = '{}'):
        """Add trading signal"""
        with self.lock:
            signal_id = f"SIG_{symbol}_{int(datetime.now().timestamp() * 1000)}"
            self.conn.execute("""
                INSERT INTO signals VALUES (?, ?, ?, ?, ?, ?)
            """, [signal_id, symbol, signal_type, confidence, datetime.now(), metadata])
    
    def get_symbol_summary(self, symbol: str) -> pd.DataFrame:
        """Get aggregated summary for a symbol"""
        with self.lock:
            return self.conn.execute("""
                SELECT 
                    symbol,
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN side = 'BUY' THEN 1 ELSE 0 END) as buy_count,
                    SUM(CASE WHEN side = 'SELL' THEN 1 ELSE 0 END) as sell_count,
                    SUM(pnl) as total_pnl,
                    AVG(pnl) as avg_pnl,
                    STDDEV(pnl) as pnl_std,
                    MIN(timestamp) as first_trade,
                    MAX(timestamp) as last_trade
                FROM trades
                WHERE symbol = ?
                GROUP BY symbol
            """, [symbol]).fetchdf()
    
    def get_recent_trades(self, symbol: str = None, limit: int = 100) -> pd.DataFrame:
        """Get recent trades"""
        with self.lock:
            if symbol:
                return self.conn.execute("""
                    SELECT * FROM trades 
                    WHERE symbol = ?
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, [symbol, limit]).fetchdf()
            else:
                return self.conn.execute("""
                    SELECT * FROM trades 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, [limit]).fetchdf()
    
    def get_recent_signals(self, symbol: str = None, limit: int = 50) -> pd.DataFrame:
        """Get recent trading signals"""
        with self.lock:
            if symbol:
                return self.conn.execute("""
                    SELECT * FROM signals 
                    WHERE symbol = ?
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, [symbol, limit]).fetchdf()
            else:
                return self.conn.execute("""
                    SELECT * FROM signals 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, [limit]).fetchdf()
    
    def get_position_performance(self) -> pd.DataFrame:
        """Get current position performance"""
        with self.lock:
            return self.conn.execute("""
                SELECT 
                    symbol,
                    quantity,
                    entry_price,
                    current_value,
                    unrealized_pnl,
                    realized_pnl,
                    (unrealized_pnl + realized_pnl) as total_pnl,
                    ((current_value - entry_price * quantity) / (entry_price * quantity) * 100) as pnl_percent
                FROM positions
                ORDER BY total_pnl DESC
            """).fetchdf()
    
    def get_analytics_context(self, symbol: str = None) -> Dict:
        """Get comprehensive analytics context for RAG"""
        with self.lock:
            context = {}
            
            # Overall statistics
            stats = self.conn.execute("""
                SELECT 
                    COUNT(DISTINCT symbol) as active_symbols,
                    COUNT(*) as total_trades,
                    SUM(pnl) as total_pnl,
                    AVG(pnl) as avg_pnl
                FROM trades
            """).fetchdf()
            context['overall_stats'] = stats.to_dict('records')[0] if not stats.empty else {}
            
            # Symbol-specific
            if symbol:
                symbol_stats = self.conn.execute("""
                    SELECT 
                        COUNT(*) as trade_count,
                        SUM(pnl) as symbol_pnl,
                        AVG(pnl) as avg_trade_pnl,
                        MAX(pnl) as max_win,
                        MIN(pnl) as max_loss
                    FROM trades
                    WHERE symbol = ?
                """, [symbol]).fetchdf()
                context['symbol_stats'] = symbol_stats.to_dict('records')[0] if not symbol_stats.empty else {}
                
                # Recent trades for symbol
                context['recent_trades'] = self.get_recent_trades(symbol, limit=5).to_dict('records')
            
            # Position performance
            positions = self.get_position_performance()
            context['positions'] = positions.to_dict('records') if not positions.empty else []
            
            return context
    
    def fast_query(self, query: str) -> pd.DataFrame:
        """Execute arbitrary SQL query"""
        with self.lock:
            return self.conn.execute(query).fetchdf()
    
    def close(self):
        """Close database connection"""
        self.conn.close()


# Singleton instance
_cache_instance = None
_cache_lock = threading.Lock()


def get_trading_cache(db_path=':memory:'):
    """Get or create singleton cache instance"""
    global _cache_instance
    
    with _cache_lock:
        if _cache_instance is None:
            _cache_instance = TradingDataCache(db_path)
        return _cache_instance


# Test the cache
if __name__ == "__main__":
    import time
    
    cache = TradingDataCache()
    print("  DuckDB cache initialized")
    
    # Test query performance
    start = time.perf_counter()
    result = cache.conn.execute("SELECT 1").fetchall()
    latency = (time.perf_counter() - start) * 1_000_000
    print(f"  Query latency: {latency:.2f} μs")
    
    # Test adding data
    cache.sync_market_data('BTCUSDT', 85000.0, 1000000, 84999, 85001)
    cache.add_signal('BTCUSDT', 'BUY', 0.85)
    
    print(f"  Market data and signals added")
    print(f"  Cache ready for real-time operations")
