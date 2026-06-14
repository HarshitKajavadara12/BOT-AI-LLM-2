"""
Real-Time Data Cache for Cryptocurrency Market Data
Caches live data from Binance API and WebSocket streams
"""

import time
import threading
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import deque
import pandas as pd
import numpy as np

from data.ingestion.binance_client import BinanceAPIClient
from data.ingestion.binance_websocket import MultiStreamWebSocket


class RealTimeDataCache:
    """
    Centralized cache for real-time cryptocurrency market data
    Combines REST API and WebSocket streams for optimal data freshness
    """
    
    def __init__(self, symbols: List[str], api_key: Optional[str] = None, 
                 api_secret: Optional[str] = None, testnet: bool = False):
        """
        Initialize real-time data cache
        
        Args:
            symbols: List of trading pairs to monitor
            api_key: Binance API key (optional)
            api_secret: Binance API secret (optional)
            testnet: Use testnet
        """
        self.symbols = symbols
        self.testnet = testnet
        
        # Initialize clients
        self.api_client = BinanceAPIClient(api_key, api_secret, testnet)
        self.ws_client = None
        
        # Data storage
        self.prices = {}
        self.order_books = {}
        self.tickers = {}
        self.historical_data = {}
        self.trade_history = {symbol: deque(maxlen=1000) for symbol in symbols}
        
        # Cache metadata
        self.last_update = {}
        self.cache_duration = 60  # seconds
        
        # Threading
        self.lock = threading.Lock()
        self.running = False
        self.update_thread = None
        
    def start(self, enable_websocket: bool = True):
        """
        Start real-time data streaming
        
        Args:
            enable_websocket: Enable WebSocket streaming for real-time updates
        """
        print(f"Starting real-time data cache for {len(self.symbols)} symbols...")
        
        # Initial data load
        self._initialize_data()
        
        # Start WebSocket streaming
        if enable_websocket:
            self.ws_client = MultiStreamWebSocket(self.symbols, self.testnet)
            self.ws_client.start()
            print("WebSocket streaming enabled")
        
        # Start background update thread
        self.running = True
        self.update_thread = threading.Thread(target=self._background_update)
        self.update_thread.daemon = True
        self.update_thread.start()
        
        print("Real-time data cache started successfully")
    
    def stop(self):
        """Stop real-time data streaming"""
        print("Stopping real-time data cache...")
        self.running = False
        
        if self.ws_client:
            self.ws_client.stop()
        
        if self.update_thread:
            self.update_thread.join(timeout=5)
        
        print("Real-time data cache stopped")
    
    def _initialize_data(self):
        """Load initial data from REST API"""
        print("Loading initial market data...")
        
        with self.lock:
            # Get current prices
            prices = self.api_client.get_current_prices(self.symbols)
            for symbol, price in prices.items():
                self.prices[symbol] = price
                self.last_update[symbol] = time.time()
            
            # Get 24h tickers
            for symbol in self.symbols:
                try:
                    ticker = self.api_client.get_24h_ticker(symbol)
                    self.tickers[symbol] = ticker
                except Exception as e:
                    print(f"Failed to get ticker for {symbol}: {e}")
            
            # Get order books
            for symbol in self.symbols:
                try:
                    order_book = self.api_client.get_order_book(symbol, limit=20)
                    self.order_books[symbol] = order_book
                except Exception as e:
                    print(f"Failed to get order book for {symbol}: {e}")
            
            # Get historical data (last 7 days, 1h candles)
            for symbol in self.symbols:
                try:
                    df = self.api_client.get_historical_data(symbol, '1h', days=7)
                    self.historical_data[symbol] = df
                except Exception as e:
                    print(f"Failed to get historical data for {symbol}: {e}")
        
        print("Initial data loaded successfully")
    
    def _background_update(self):
        """Background thread to periodically update data"""
        while self.running:
            try:
                # Update from WebSocket if available
                if self.ws_client:
                    self._update_from_websocket()
                
                # Periodically refresh from REST API
                current_time = time.time()
                for symbol in self.symbols:
                    if current_time - self.last_update.get(symbol, 0) > self.cache_duration:
                        self._refresh_symbol_data(symbol)
                
                time.sleep(0.01)  # Check every 10ms for high-frequency updates
                
            except Exception as e:
                print(f"Error in background update: {e}")
                time.sleep(5)
    
    def _update_from_websocket(self):
        """Update cache from WebSocket data"""
        if not self.ws_client:
            return
        
        with self.lock:
            # Update prices from WebSocket
            ws_prices = self.ws_client.get_all_prices()
            if ws_prices:  # Only update if we got data
                for symbol, price in ws_prices.items():
                    if price > 0:
                        self.prices[symbol] = price
                        self.last_update[symbol] = time.time()
            
            # Update order books and tickers
            for symbol in self.symbols:
                market_data = self.ws_client.get_market_data(symbol)
                
                if 'ticker' in market_data:
                    self.tickers[symbol] = market_data['ticker']
                
                if 'order_book' in market_data:
                    self.order_books[symbol] = market_data['order_book']
                
                if 'last_trade' in market_data:
                    self.trade_history[symbol].append(market_data['last_trade'])
    
    def _refresh_symbol_data(self, symbol: str):
        """Refresh data for specific symbol from REST API"""
        try:
            with self.lock:
                # Update price
                price = self.api_client.get_current_price(symbol)
                self.prices[symbol] = price
                
                # Update ticker
                ticker = self.api_client.get_24h_ticker(symbol)
                self.tickers[symbol] = ticker
                
                # Update order book
                order_book = self.api_client.get_order_book(symbol, limit=20)
                self.order_books[symbol] = order_book
                
                self.last_update[symbol] = time.time()
                
        except Exception as e:
            print(f"Failed to refresh data for {symbol}: {e}")
    
    # =================
    # Public API
    # =================
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for symbol"""
        with self.lock:
            return self.prices.get(symbol)
    
    def get_current_prices(self, symbols: Optional[List[str]] = None) -> Dict[str, float]:
        """Get current prices for multiple symbols"""
        with self.lock:
            if symbols is None:
                return self.prices.copy()
            return {s: self.prices.get(s, 0.0) for s in symbols if s in self.prices}
    
    def get_ticker(self, symbol: str) -> Optional[Dict]:
        """Get 24h ticker data for symbol"""
        with self.lock:
            return self.tickers.get(symbol)
    
    def get_order_book(self, symbol: str) -> Optional[Dict]:
        """Get order book for symbol"""
        with self.lock:
            return self.order_books.get(symbol)
    
    def get_history(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """
        Get historical data (alias for get_historical_data with different signature)
        Used by dashboards expecting get_history(symbol, limit)
        """
        # Map limit to days roughly (assuming hourly data)
        days = max(1, limit // 24)
        return self.get_historical_data(symbol, interval='1h', days=days)

    def get_trades(self, symbol: str, limit: int = 50) -> List[Dict]:
        """Alias for get_recent_trades"""
        return self.get_recent_trades(symbol, limit)

    def get_historical_data(self, symbol: str, interval: str = '1h', 
                          days: int = 7) -> pd.DataFrame:
        """
        Get historical OHLCV data
        
        Args:
            symbol: Trading pair
            interval: Candle interval
            days: Number of days
            
        Returns:
            DataFrame with OHLCV data
        """
        # Check cache first
        with self.lock:
            if symbol in self.historical_data and not self.historical_data[symbol].empty:
                cached_df = self.historical_data[symbol]
                
                # Check if cache is recent enough AND covers the requested period
                start_date = datetime.now() - timedelta(days=days)
                cache_start = cached_df.index[0]
                
                # If cache starts before or at requested start (with 1 day buffer) and is recent
                if (datetime.now() - cached_df.index[-1]).total_seconds() < 3600 and \
                   cache_start <= start_date + timedelta(days=1):
                    return cached_df[cached_df.index >= start_date]
        
        # Fetch fresh data
        df = self.api_client.get_historical_data(symbol, interval, days)
        
        # Update cache
        with self.lock:
            self.historical_data[symbol] = df
        
        return df
    
    def get_recent_trades(self, symbol: str, limit: int = 50) -> List[Dict]:
        """Get recent trades for symbol"""
        with self.lock:
            trades = list(self.trade_history.get(symbol, []))
            return trades[-limit:]
    
    def get_market_data(self, symbol: str) -> Dict:
        """
        Get comprehensive market data for symbol
        
        Returns:
            Dict with price, ticker, order_book, recent_trades
        """
        with self.lock:
            return {
                'symbol': symbol,
                'price': self.prices.get(symbol),
                'ticker': self.tickers.get(symbol),
                'order_book': self.order_books.get(symbol),
                'recent_trades': list(self.trade_history.get(symbol, []))[-10:],
                'timestamp': datetime.now().isoformat()
            }
    
    def get_ohlcv_dataframe(self, symbol: str, interval: str = '1h', 
                           days: int = 7) -> pd.DataFrame:
        """
        Get OHLCV data as DataFrame (for backtesting/analysis)
        
        Returns:
            DataFrame with columns: open, high, low, close, volume
        """
        return self.get_historical_data(symbol, interval, days)
    
    def get_returns(self, symbols: Optional[List[str]] = None, 
                   days: int = 30) -> pd.DataFrame:
        """
        Calculate returns for multiple symbols
        
        Args:
            symbols: List of symbols (None for all)
            days: Number of days of history
            
        Returns:
            DataFrame with returns for each symbol
        """
        if symbols is None:
            symbols = self.symbols
        
        returns_dict = {}
        
        for symbol in symbols:
            df = self.get_historical_data(symbol, '1h', days)
            if not df.empty:
                # Calculate hourly returns
                returns = df['close'].pct_change().dropna()
                returns_dict[symbol] = returns
        
        if returns_dict:
            # Combine into DataFrame
            returns_df = pd.DataFrame(returns_dict)
            return returns_df
        
        return pd.DataFrame()
    
    def get_correlation_matrix(self, symbols: Optional[List[str]] = None, 
                               days: int = 30) -> pd.DataFrame:
        """
        Calculate correlation matrix between symbols
        
        Args:
            symbols: List of symbols (None for all)
            days: Number of days of history
            
        Returns:
            Correlation matrix DataFrame
        """
        returns_df = self.get_returns(symbols, days)
        if not returns_df.empty:
            return returns_df.corr()
        return pd.DataFrame()
    
    def get_volatility(self, symbol: str, days: int = 30, 
                      annualized: bool = True) -> float:
        """
        Calculate historical volatility
        
        Args:
            symbol: Trading pair
            days: Number of days
            annualized: Return annualized volatility
            
        Returns:
            Volatility (standard deviation of returns)
        """
        df = self.get_historical_data(symbol, '1h', days)
        if df.empty:
            return 0.0
        
        returns = df['close'].pct_change().dropna()
        volatility = returns.std()
        
        if annualized:
            # Annualize hourly volatility (24 * 365 trading hours)
            volatility *= np.sqrt(24 * 365)
        
        return volatility
    
    def get_portfolio_data(self, symbols: Optional[List[str]] = None) -> Dict:
        """
        Get comprehensive data for portfolio analysis
        
        Returns:
            Dict with prices, returns, correlations, volatilities
        """
        if symbols is None:
            symbols = self.symbols
        
        prices = self.get_current_prices(symbols)
        returns = self.get_returns(symbols, days=30)
        correlations = self.get_correlation_matrix(symbols, days=30)
        volatilities = {s: self.get_volatility(s, days=30) for s in symbols}
        
        return {
            'prices': prices,
            'returns': returns,
            'correlations': correlations,
            'volatilities': volatilities,
            'timestamp': datetime.now().isoformat()
        }
    
    def is_healthy(self) -> bool:
        """Check if cache is healthy and receiving updates"""
        current_time = time.time()
        
        # Check if we have recent data
        for symbol in self.symbols:
            if symbol not in self.last_update:
                return False
            
            if current_time - self.last_update[symbol] > 120:  # 2 minutes stale
                return False
        
        return True
    
    def get_status(self) -> Dict:
        """Get cache status and statistics"""
        current_time = time.time()
        
        status = {
            'running': self.running,
            'symbols': len(self.symbols),
            'websocket_enabled': self.ws_client is not None,
            'cache_healthy': self.is_healthy(),
            'symbol_status': {}
        }
        
        for symbol in self.symbols:
            last_update = self.last_update.get(symbol, 0)
            age = current_time - last_update if last_update > 0 else None
            
            status['symbol_status'][symbol] = {
                'has_price': symbol in self.prices,
                'has_ticker': symbol in self.tickers,
                'has_order_book': symbol in self.order_books,
                'has_historical': symbol in self.historical_data,
                'last_update_age_seconds': age,
                'trade_count': len(self.trade_history.get(symbol, []))
            }
        
        return status


# Global cache instance
_global_cache = None


def get_cache(symbols: Optional[List[str]] = None, 
             api_key: Optional[str] = None, 
             api_secret: Optional[str] = None,
             testnet: bool = False) -> RealTimeDataCache:
    """
    Get or create global data cache instance
    
    Args:
        symbols: List of trading pairs (required for first call)
        api_key: Binance API key
        api_secret: Binance API secret
        testnet: Use testnet
        
    Returns:
        Global RealTimeDataCache instance
    """
    global _global_cache
    
    if _global_cache is None:
        if symbols is None:
            # Default crypto pairs
            symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 
                      'ADAUSDT', 'DOGEUSDT', 'XRPUSDT']
        
        _global_cache = RealTimeDataCache(symbols, api_key, api_secret, testnet)
        _global_cache.start()
    
    return _global_cache


# Example usage
if __name__ == "__main__":
    import time
    
    # Define crypto pairs
    crypto_pairs = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']
    
    print("="*60)
    print("Real-Time Data Cache Demo")
    print("="*60)
    print()
    
    # Initialize cache
    cache = RealTimeDataCache(crypto_pairs)
    cache.start(enable_websocket=True)
    
    # Let it warm up
    print("Warming up cache (5 seconds)...")
    time.sleep(5)
    print()
    
    # Get current prices
    print("="*60)
    print("Current Prices")
    print("="*60)
    prices = cache.get_current_prices()
    for symbol, price in prices.items():
        print(f"{symbol:12} ${price:>12,.2f}")
    print()
    
    # Get market data
    print("="*60)
    print("Market Data for BTCUSDT")
    print("="*60)
    btc_data = cache.get_market_data('BTCUSDT')
    if btc_data['ticker']:
        ticker = btc_data['ticker']
        print(f"Price: ${ticker['close']:,.2f}")
        print(f"24h Change: {ticker['price_change_percent']:.2f}%")
        print(f"24h Volume: {ticker['volume']:,.2f} BTC")
        print(f"24h High: ${ticker['high']:,.2f}")
        print(f"24h Low: ${ticker['low']:,.2f}")
    print()
    
    # Get historical data
    print("="*60)
    print("Historical Data (Last 24 Hours)")
    print("="*60)
    hist_data = cache.get_historical_data('BTCUSDT', '1h', days=1)
    print(hist_data[['open', 'high', 'low', 'close', 'volume']].tail(10))
    print()
    
    # Calculate returns and volatility
    print("="*60)
    print("Returns & Volatility Analysis")
    print("="*60)
    for symbol in crypto_pairs:
        vol = cache.get_volatility(symbol, days=7, annualized=True)
        print(f"{symbol:12} Annualized Volatility: {vol*100:.2f}%")
    print()
    
    # Get cache status
    print("="*60)
    print("Cache Status")
    print("="*60)
    status = cache.get_status()
    print(f"Running: {status['running']}")
    print(f"WebSocket Enabled: {status['websocket_enabled']}")
    print(f"Cache Healthy: {status['cache_healthy']}")
    print(f"Symbols: {status['symbols']}")
    print()
    
    # Stop cache
    cache.stop()
    
    print("="*60)
    print("Real-time data cache demo complete!")
    print("All data was fetched live from Binance!")
    print("="*60)
