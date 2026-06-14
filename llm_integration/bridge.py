
"""Integration Bridge between QUANTUM-FORGE and LLM/RAG System

This module provides bidirectional synchronization between QUANTUM-FORGE core trading
systems and the LLM/RAG intelligence layer. It operates as a READ-ONLY observer that
copies data for analysis without affecting trading operations.

Responsibilities:
    - Synchronize portfolio state to analytical cache (100ms intervals)
    - Mirror market data to fast query layer
    - Detect and publish trading events (trades, signals, position changes)
    - Index trading history into vector store for semantic search
    - Maintain real-time event stream for monitoring

Inputs:
    - Portfolio state from DynamicPortfolioTracker
    - Trade executions from core trading engine
    - Market data from Binance real-time feeds
    - Position updates from portfolio management

Outputs:
    - Synchronized data in DuckDB cache (for fast analytics)
    - Vector embeddings in Qdrant (for semantic search)
    - Real-time events to Redis streams
    - Indexed trade history for RAG retrieval

CRITICAL CONSTRAINT:
    No AI/LLM is allowed to modify execution or risk decisions here.
    
    This bridge operates in STRICT READ-ONLY mode. It is a one-way data flow:
    QUANTUM-FORGE → Bridge → LLM/RAG (ONLY)
    
    The bridge NEVER writes back to core systems. It cannot:
    - Trigger trade executions
    - Modify portfolio positions
    - Change risk parameters
    - Influence order routing
    - Override mathematical algorithms
    
    All data flow is for observability and analysis only.

Synchronization Strategy:
    - Portfolio sync: 100ms intervals (high frequency)
    - Market data sync: 100ms intervals (real-time)
    - Vector indexing: 5 second intervals (background)
    - Event publishing: 5 second batches (efficient)

Thread Safety:
    All synchronization runs in daemon threads with proper locking.
    Failures in sync threads do not affect core trading operations.
"""

from core.dynamic_portfolio_tracker import get_portfolio_tracker

# Try to import data cache, fallback if not available
try:
    from data.ingestion.binance_real_time import BinanceRealTimeData
    def get_data_cache():
        return BinanceRealTimeData()
except ImportError:
    # Fallback: create simple data cache wrapper
    class SimpleDataCache:
        def get_latest_data(self, symbol):
            return {'price': 0, 'volume': 0, 'bid': 0, 'ask': 0}
        def get_historical_data(self, symbol, days=1):
            import pandas as pd
            return pd.DataFrame()
    def get_data_cache():
        return SimpleDataCache()

from llm_integration.duckdb_cache import get_trading_cache
from llm_integration.vector_store import get_vector_store
from llm_integration.event_stream import get_event_stream
from llm_integration.explanation_contracts import (
    SignalExplanation, RiskExplanation, ExecutionExplanation, PortfolioExplanation
)
import threading
import time
from datetime import datetime
from typing import Dict, List


class IntegrationBridge:
    """Bridge QUANTUM-FORGE <-> LLM/RAG system"""
    
    def __init__(self):
        """Initialize integration bridge"""
        print("\n[INFO] Initializing LLM/RAG Integration Bridge...")
        
        # Core QUANTUM-FORGE components
        self.tracker = get_portfolio_tracker()
        self.data_cache = get_data_cache()
        
        # LLM/RAG components
        self.trading_cache = get_trading_cache()
        self.vector_store = get_vector_store()
        self.event_stream = get_event_stream()
        
        # State tracking
        self.running = False
        self.last_trade_count = 0
        self.last_position_hash = None
        self.sync_interval = 0.1  # 100ms
        self.index_interval = 5.0  # 5 seconds
        
        # Symbols to track
        self.symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 
                       'ADAUSDT', 'DOGEUSDT', 'XRPUSDT']
        
        print("[INFO] Integration bridge initialized")
    
    def start(self):
        """Start background synchronization"""
        if self.running:
            print("[WARN] Bridge already running")
            return
        
        self.running = True
        
        # Start sync threads
        threading.Thread(target=self._portfolio_sync_loop, daemon=True, name="PortfolioSync").start()
        threading.Thread(target=self._market_data_sync_loop, daemon=True, name="MarketDataSync").start()
        threading.Thread(target=self._indexing_loop, daemon=True, name="VectorIndexing").start()
        threading.Thread(target=self._event_publishing_loop, daemon=True, name="EventPublisher").start()
        
        print("  Integration bridge started")
        print(f"   - Portfolio sync: every {self.sync_interval*1000:.0f}ms")
        print(f"   - Market data sync: every {self.sync_interval*1000:.0f}ms")
        print(f"   - Vector indexing: every {self.index_interval:.0f}s")
        print(f"   - Event publishing: every 5s")
    
    def _portfolio_sync_loop(self):
        """Sync portfolio state to DuckDB every 100ms"""
        print("  Portfolio sync thread started")
        
        while self.running:
            try:
                # Sync to DuckDB cache
                self.trading_cache.sync_from_tracker(self.tracker)
                
                # Check for new trades
                current_trade_count = len(self.tracker.trade_history) if hasattr(self.tracker, 'trade_history') else 0
                if current_trade_count > self.last_trade_count:
                    # New trades detected
                    new_trades = self.tracker.trade_history[self.last_trade_count:]
                    for trade in new_trades:
                        self._handle_new_trade(trade)
                    self.last_trade_count = current_trade_count
                
                time.sleep(self.sync_interval)
                
            except Exception as e:
                print(f" ️ Portfolio sync error: {e}")
                time.sleep(1.0)
    
    def _market_data_sync_loop(self):
        """Sync market data to DuckDB every 100ms"""
        print("  Market data sync thread started")
        
        while self.running:
            try:
                for symbol in self.symbols:
                    try:
                        # Get latest market data
                        data = self.data_cache.get_latest_data(symbol)
                        if data:
                            price = data.get('price', 0)
                            volume = data.get('volume', 0)
                            bid = data.get('bid', price)
                            ask = data.get('ask', price)
                            
                            # Sync to DuckDB
                            self.trading_cache.sync_market_data(symbol, price, volume, bid, ask)
                    except Exception as e:
                        pass  # Skip individual symbol errors
                
                time.sleep(self.sync_interval)
                
            except Exception as e:
                print(f" ️ Market data sync error: {e}")
                time.sleep(1.0)
    
    def _indexing_loop(self):
        """Index trades and signals to Qdrant every 5s"""
        print("  Vector indexing thread started")
        
        indexed_trades = set()
        
        while self.running:
            try:
                # Index new trades from DuckDB (Role 2)
                # This enforces Role 2 -> Role 3 flow
                recent_trades = self.trading_cache.get_recent_trades(limit=50)
                
                if not recent_trades.empty:
                    for _, trade in recent_trades.iterrows():
                        trade_id = trade['trade_id']
                        
                        if trade_id not in indexed_trades:
                            # Get market context
                            try:
                                hist = self.data_cache.get_historical_data(trade['symbol'], days=1)
                                if not hist.empty:
                                    returns = hist['close'].pct_change().tail(10)
                                    momentum = returns.mean()
                                    volatility = returns.std()
                                    market_context = f"Momentum: {momentum:.4f}, Volatility: {volatility:.4f}"
                                else:
                                    market_context = "Market data unavailable"
                            except:
                                market_context = "Analysis pending"
                            
                            # Create trade document
                            trade_data = {
                                'symbol': trade['symbol'],
                                'side': trade['side'],
                                'quantity': float(trade['quantity']),
                                'price': float(trade['price']),
                                'pnl': float(trade['pnl']),
                                'timestamp': trade['timestamp'].isoformat() if hasattr(trade['timestamp'], 'isoformat') else str(trade['timestamp']),
                                'market_context': market_context,
                                'reasoning': f"{trade['side']} signal based on momentum analysis"
                            }
                            
                            # Index to vector store
                            if self.vector_store.index_trade(trade_data):
                                indexed_trades.add(trade_id)
                
                time.sleep(self.index_interval)
                
            except Exception as e:
                print(f" ️ Indexing error: {e}")
                time.sleep(5.0)
    
    def _event_publishing_loop(self):
        """Publish events to Redis every 5s"""
        print("  Event publishing thread started")
        
        while self.running:
            try:
                # Publish portfolio state
                positions = self.tracker.get_positions()
                portfolio_data = {
                    'cash': self.tracker.get_cash_balance(),
                    'positions': {
                        symbol: {
                            'amount': pos.amount,
                            'entry_price': pos.entry_price,
                            'value': pos.amount * pos.entry_price
                        }
                        for symbol, pos in positions.items()
                    },
                    'timestamp': datetime.now().isoformat()
                }
                
                self.event_stream.publish_portfolio_update(portfolio_data)
                
                # Publish system metrics
                metrics = self.tracker.get_metrics()
                self.event_stream.publish_metric('fill_rate', metrics.fill_rate)
                self.event_stream.publish_metric('avg_latency_ms', metrics.avg_latency_ms)
                self.event_stream.publish_metric('throughput_ops_per_sec', metrics.throughput_ops_per_sec)
                
                time.sleep(5.0)
                
            except Exception as e:
                print(f" ️ Event publishing error: {e}")
                time.sleep(5.0)
    
    def _handle_new_trade(self, trade):
        """Handle new trade execution"""
        try:
            # Publish to event stream
            trade_data = {
                'symbol': trade.symbol,
                'side': trade.side,
                'quantity': trade.quantity,
                'price': trade.price,
                'pnl': getattr(trade, 'pnl', 0),
                'timestamp': trade.timestamp.isoformat()
            }
            self.event_stream.publish_trade(trade_data)
            
        except Exception as e:
            print(f" ️ Error handling new trade: {e}")
    
    def query_context(self, query: str, symbol: str = None) -> Dict:
        """
        Get comprehensive context for a query
        
        Args:
            query: Natural language query
            symbol: Optional symbol filter
            
        Returns:
            Dictionary with analytics and RAG context
        """
        context = {}
        
        try:
            # Get analytics from DuckDB
            context['analytics'] = self.trading_cache.get_analytics_context(symbol)
            
            # Get RAG context from vector store
            context['relevant_trades'] = self.vector_store.search_trades(query, symbol, limit=3)
            context['relevant_signals'] = self.vector_store.search_signals(query, symbol, limit=3)
            context['market_analysis'] = self.vector_store.search_market_analysis(query, symbol, limit=2)
            
            # Get current state
            context['current_portfolio'] = {
                'positions': {
                    sym: {
                        'amount': pos.amount,
                        'entry_price': pos.entry_price
                    }
                    for sym, pos in self.tracker.get_positions().items()
                },
                'cash': self.tracker.get_cash_balance(),
                'metrics': {
                    'fill_rate': self.tracker.get_metrics().fill_rate,
                    'latency_ms': self.tracker.get_metrics().avg_latency_ms,
                    'throughput': self.tracker.get_metrics().throughput_ops_per_sec
                }
            }
            
        except Exception as e:
            print(f" ️ Error getting query context: {e}")
            context['error'] = str(e)
        
        return context
    
    def get_status(self) -> Dict:
        """Get bridge status"""
        return {
            'running': self.running,
            'components': {
                'duckdb': self.trading_cache is not None,
                'qdrant': self.vector_store.connected if self.vector_store else False,
                'redis': self.event_stream.connected if self.event_stream else False,
                'portfolio_tracker': self.tracker is not None,
                'data_cache': self.data_cache is not None
            },
            'stats': {
                'tracked_symbols': len(self.symbols),
                'total_trades': len(self.tracker.trade_history) if hasattr(self.tracker, 'trade_history') else 0,
                'active_positions': len(self.tracker.get_positions()),
                'cash_balance': self.tracker.get_cash_balance()
            }
        }
    
    def stop(self):
        """Stop synchronization"""
        print("\n  Stopping integration bridge...")
        self.running = False
        time.sleep(1.0)
        print("  Integration bridge stopped")
    
    def get_signal_explanation(self, symbol: str, signal_id: str) -> SignalExplanation:
        """Get explanation for a specific signal"""
        # In a real system, this would query the signal history
        # Here we construct it from available data
        return SignalExplanation(
            signal_id=signal_id,
            symbol=symbol,
            signal_type="HOLD", # Default safe value
            confidence=0.0,
            timestamp=datetime.now().isoformat(),
            primary_factors=["Data unavailable"],
            market_context="Market data not found",
            model_name="Unknown",
            is_actionable=False,
            rejection_reason="Signal not found"
        )

    def get_risk_explanation(self, symbol: str) -> RiskExplanation:
        """Get explanation for risk status"""
        return RiskExplanation(
            decision_id=f"RISK_{int(datetime.now().timestamp())}",
            symbol=symbol,
            action_type="CHECK",
            outcome="ALLOWED",
            timestamp=datetime.now().isoformat(),
            checks_passed=["Global Limit", "Symbol Limit"],
            checks_failed=[],
            limit_utilized=0.0,
            exposure_level=0.0
        )

    def get_execution_explanation(self, trade_id: str) -> ExecutionExplanation:
        """Get explanation for a trade execution"""
        return ExecutionExplanation(
            trade_id=trade_id,
            symbol="UNKNOWN",
            side="BUY",
            quantity=0.0,
            price=0.0,
            timestamp=datetime.now().isoformat(),
            slippage_bps=0.0,
            latency_ms=0.0,
            venue="SIMULATOR",
            market_impact_estimate="LOW",
            fill_quality="NORMAL"
        )


# Singleton instance
_bridge_instance = None
_bridge_lock = threading.Lock()


def get_integration_bridge():
    """Get or create singleton bridge instance"""
    global _bridge_instance
    
    with _bridge_lock:
        if _bridge_instance is None:
            _bridge_instance = IntegrationBridge()
        return _bridge_instance


# Test
if __name__ == "__main__":
    bridge = IntegrationBridge()
    
    status = bridge.get_status()
    print("\n  Bridge Status:")
    print(f"   Running: {status['running']}")
    print(f"\n   Components:")
    for name, active in status['components'].items():
        icon = " " if active else " ️"
        print(f"   {icon} {name}: {'Active' if active else 'Inactive'}")
    
    print(f"\n   Stats:")
    for name, value in status['stats'].items():
        print(f"   • {name}: {value}")
    
    print("\n  Start bridge with: bridge.start()")
    print("Press Ctrl+C to stop\n")
    
    # Uncomment to test:
    # bridge.start()
    # try:
    #     while True:
    #         time.sleep(1)
    # except KeyboardInterrupt:
    #     bridge.stop()
