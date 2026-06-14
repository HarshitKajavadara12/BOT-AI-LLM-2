"""
Binance Real-Time Data Client
Fetches live cryptocurrency market data from Binance exchange
"""

import os
import time
import hmac
import hashlib
import requests
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class BinanceAPIClient:
    """Real-time Binance API client for fetching cryptocurrency market data"""
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, testnet: bool = False):
        """
        Initialize Binance API client
        
        Args:
            api_key: Binance API key (optional for public endpoints)
            api_secret: Binance API secret (optional for public endpoints)
            testnet: Use Binance testnet instead of production
        """
        self.api_key = api_key or os.getenv('BINANCE_API_KEY')
        self.api_secret = api_secret or os.getenv('BINANCE_SECRET_KEY')
        self.testnet = testnet or os.getenv('BINANCE_TESTNET', 'False').lower() == 'true'
        
        # Base URLs
        if self.testnet:
            self.base_url = "https://testnet.binance.vision"
        else:
            self.base_url = "https://api.binance.com"
        
        # Setup session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Cache for rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with API key"""
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['X-MBX-APIKEY'] = self.api_key
        return headers
    
    def _generate_signature(self, params: Dict) -> str:
        """Generate HMAC SHA256 signature for authenticated requests"""
        if not self.api_secret:
            raise ValueError("API secret required for authenticated requests")
        
        query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def _rate_limit(self):
        """Implement simple rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None, 
                     authenticated: bool = False) -> Dict:
        """Make HTTP request to Binance API"""
        self._rate_limit()
        
        url = f"{self.base_url}{endpoint}"
        headers = self._get_headers()
        
        if params is None:
            params = {}
        
        if authenticated:
            params['timestamp'] = int(time.time() * 1000)
            params['signature'] = self._generate_signature(params)
        
        try:
            response = self.session.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Binance API request failed: {e}")
            raise
    
    # =================
    # Market Data
    # =================
    
    def get_server_time(self) -> int:
        """Get Binance server time (milliseconds)"""
        data = self._make_request("/api/v3/time")
        return data['serverTime']
    
    def get_exchange_info(self, symbol: Optional[str] = None) -> Dict:
        """Get exchange trading rules and symbol information"""
        params = {}
        if symbol:
            params['symbol'] = symbol
        return self._make_request("/api/v3/exchangeInfo", params)
    
    def get_current_price(self, symbol: str) -> float:
        """
        Get current price for a symbol
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            
        Returns:
            Current price as float
        """
        data = self._make_request("/api/v3/ticker/price", {'symbol': symbol})
        return float(data['price'])
    
    def get_current_prices(self, symbols: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Get current prices for multiple symbols
        
        Args:
            symbols: List of trading pairs, None for all pairs
            
        Returns:
            Dict mapping symbol to price
        """
        data = self._make_request("/api/v3/ticker/price")
        
        prices = {item['symbol']: float(item['price']) for item in data}
        
        if symbols:
            prices = {k: v for k, v in prices.items() if k in symbols}
        
        return prices
    
    def get_24h_ticker(self, symbol: str) -> Dict:
        """
        Get 24-hour price change statistics
        
        Returns:
            Dict with open, high, low, close, volume, etc.
        """
        data = self._make_request("/api/v3/ticker/24hr", {'symbol': symbol})
        return {
            'symbol': data['symbol'],
            'price_change': float(data['priceChange']),
            'price_change_percent': float(data['priceChangePercent']),
            'open': float(data['openPrice']),
            'high': float(data['highPrice']),
            'low': float(data['lowPrice']),
            'close': float(data['lastPrice']),
            'volume': float(data['volume']),
            'quote_volume': float(data['quoteVolume']),
            'open_time': data['openTime'],
            'close_time': data['closeTime'],
            'trades': data['count']
        }
    
    def get_order_book(self, symbol: str, limit: int = 100) -> Dict:
        """
        Get order book (market depth)
        
        Args:
            symbol: Trading pair
            limit: Number of price levels (5, 10, 20, 50, 100, 500, 1000, 5000)
            
        Returns:
            Dict with 'bids' and 'asks' lists
        """
        data = self._make_request("/api/v3/depth", {'symbol': symbol, 'limit': limit})
        
        return {
            'last_update_id': data['lastUpdateId'],
            'bids': [[float(price), float(qty)] for price, qty in data['bids']],
            'asks': [[float(price), float(qty)] for price, qty in data['asks']],
            'timestamp': int(time.time() * 1000)
        }
    
    def get_recent_trades(self, symbol: str, limit: int = 500) -> List[Dict]:
        """
        Get recent trades
        
        Args:
            symbol: Trading pair
            limit: Number of trades (max 1000)
            
        Returns:
            List of trade dicts
        """
        data = self._make_request("/api/v3/trades", {'symbol': symbol, 'limit': limit})
        
        return [{
            'id': trade['id'],
            'price': float(trade['price']),
            'qty': float(trade['qty']),
            'quote_qty': float(trade['quoteQty']),
            'time': trade['time'],
            'is_buyer_maker': trade['isBuyerMaker']
        } for trade in data]
    
    def get_klines(self, symbol: str, interval: str, limit: int = 500, 
                   start_time: Optional[int] = None, end_time: Optional[int] = None) -> pd.DataFrame:
        """
        Get candlestick/kline data
        
        Args:
            symbol: Trading pair
            interval: Candle interval (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M)
            limit: Number of candles (max 1000)
            start_time: Start time in milliseconds
            end_time: End time in milliseconds
            
        Returns:
            DataFrame with OHLCV data
        """
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time
        
        data = self._make_request("/api/v3/klines", params)
        
        # Convert to DataFrame
        df = pd.DataFrame(data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Convert types
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 
                    'taker_buy_base', 'taker_buy_quote']:
            df[col] = df[col].astype(float)
        
        df['trades'] = df['trades'].astype(int)
        df = df.drop('ignore', axis=1)
        df = df.set_index('open_time')
        
        return df
    
    def get_historical_data(self, symbol: str, interval: str, days: int = 30) -> pd.DataFrame:
        """
        Get historical OHLCV data for specified number of days
        
        Args:
            symbol: Trading pair
            interval: Candle interval
            days: Number of days of history
            
        Returns:
            DataFrame with historical OHLCV data
        """
        end_time = int(time.time() * 1000)
        start_time = end_time - (days * 24 * 60 * 60 * 1000)
        
        all_data = []
        current_start = start_time
        
        # Binance limits to 1000 candles per request
        while current_start < end_time:
            df = self.get_klines(symbol, interval, limit=1000, 
                               start_time=current_start, end_time=end_time)
            
            if df.empty:
                break
            
            all_data.append(df)
            current_start = int(df['close_time'].iloc[-1].timestamp() * 1000) + 1
        
        if not all_data:
            return pd.DataFrame()
        
        # Combine all data
        combined_df = pd.concat(all_data)
        combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
        combined_df = combined_df.sort_index()
        
        return combined_df
    
    def get_avg_price(self, symbol: str) -> float:
        """Get current average price"""
        data = self._make_request("/api/v3/avgPrice", {'symbol': symbol})
        return float(data['price'])
    
    # =================
    # Account Data (requires authentication)
    # =================
    
    def get_account_info(self) -> Dict:
        """Get account information (requires API key)"""
        return self._make_request("/api/v3/account", authenticated=True)
    
    def get_account_balances(self) -> Dict[str, Dict[str, float]]:
        """
        Get all account balances
        
        Returns:
            Dict mapping asset to {free, locked, total}
        """
        account_info = self.get_account_info()
        balances = {}
        
        for balance in account_info['balances']:
            asset = balance['asset']
            free = float(balance['free'])
            locked = float(balance['locked'])
            
            if free > 0 or locked > 0:
                balances[asset] = {
                    'free': free,
                    'locked': locked,
                    'total': free + locked
                }
        
        return balances
    
    # =================
    # Trading (requires authentication)
    # =================
    
    def place_order(self, symbol: str, side: str, order_type: str, 
                   quantity: float, price: Optional[float] = None, 
                   time_in_force: str = 'GTC') -> Dict:
        """
        Place a new order
        
        Args:
            symbol: Trading pair
            side: 'BUY' or 'SELL'
            order_type: 'LIMIT', 'MARKET', 'STOP_LOSS_LIMIT', etc.
            quantity: Order quantity
            price: Order price (required for LIMIT orders)
            time_in_force: GTC, IOC, FOK
            
        Returns:
            Order response dict
        """
        params = {
            'symbol': symbol,
            'side': side.upper(),
            'type': order_type.upper(),
            'quantity': quantity
        }
        
        if order_type.upper() == 'LIMIT':
            if price is None:
                raise ValueError("Price required for LIMIT orders")
            params['price'] = price
            params['timeInForce'] = time_in_force
        
        return self._make_request("/api/v3/order", params, authenticated=True)
    
    def cancel_order(self, symbol: str, order_id: int) -> Dict:
        """Cancel an active order"""
        params = {'symbol': symbol, 'orderId': order_id}
        return self._make_request("/api/v3/order", params, authenticated=True)
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get all open orders"""
        params = {}
        if symbol:
            params['symbol'] = symbol
        return self._make_request("/api/v3/openOrders", params, authenticated=True)
    
    # =================
    # Utility Methods
    # =================
    
    def get_multi_asset_data(self, symbols: List[str], interval: str = '1h', 
                            days: int = 30) -> Dict[str, pd.DataFrame]:
        """
        Get historical data for multiple assets
        
        Args:
            symbols: List of trading pairs
            interval: Candle interval
            days: Number of days
            
        Returns:
            Dict mapping symbol to DataFrame
        """
        data = {}
        for symbol in symbols:
            try:
                df = self.get_historical_data(symbol, interval, days)
                data[symbol] = df
                time.sleep(0.2)  # Rate limiting
            except Exception as e:
                print(f"Failed to get data for {symbol}: {e}")
        
        return data
    
    def get_portfolio_value(self, prices: Optional[Dict[str, float]] = None) -> Dict:
        """
        Calculate total portfolio value in USDT
        
        Args:
            prices: Optional dict of current prices, will fetch if not provided
            
        Returns:
            Dict with total value and breakdown by asset
        """
        balances = self.get_account_balances()
        
        if prices is None:
            # Get prices for all assets we hold
            assets_to_price = [f"{asset}USDT" for asset in balances.keys() 
                             if asset != 'USDT']
            prices = self.get_current_prices(assets_to_price)
        
        total_value = balances.get('USDT', {}).get('total', 0.0)
        breakdown = {'USDT': total_value}
        
        for asset, balance in balances.items():
            if asset == 'USDT':
                continue
            
            symbol = f"{asset}USDT"
            if symbol in prices:
                value = balance['total'] * prices[symbol]
                total_value += value
                breakdown[asset] = value
        
        return {
            'total_usdt': total_value,
            'breakdown': breakdown,
            'timestamp': datetime.now().isoformat()
        }


def check_api_status() -> Dict:
    """Check Binance API connectivity"""
    try:
        client = BinanceAPIClient()
        server_time = client.get_server_time()
        local_time = int(time.time() * 1000)
        time_diff = abs(server_time - local_time)
        
        return {
            'status': 'connected',
            'server_time': server_time,
            'local_time': local_time,
            'time_diff_ms': time_diff,
            'time_sync_ok': time_diff < 1000
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }


# Example usage
if __name__ == "__main__":
    # Initialize client (no API key needed for public endpoints)
    client = BinanceAPIClient()
    
    # Check API status
    print("="*60)
    print("Binance API Status Check")
    print("="*60)
    status = check_api_status()
    print(f"Status: {status['status']}")
    if status['status'] == 'connected':
        print(f"Time sync: {'OK' if status['time_sync_ok'] else 'WARNING'}")
        print(f"Time difference: {status['time_diff_ms']}ms")
    print()
    
    # Get current prices for crypto pairs
    print("="*60)
    print("Current Cryptocurrency Prices")
    print("="*60)
    crypto_pairs = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 
                    'ADAUSDT', 'DOGEUSDT', 'XRPUSDT']
    
    prices = client.get_current_prices(crypto_pairs)
    for symbol, price in prices.items():
        print(f"{symbol:12} ${price:>12,.2f}")
    print()
    
    # Get 24h ticker for BTC
    print("="*60)
    print("BTC 24-Hour Statistics")
    print("="*60)
    ticker = client.get_24h_ticker('BTCUSDT')
    print(f"Open:    ${ticker['open']:>12,.2f}")
    print(f"High:    ${ticker['high']:>12,.2f}")
    print(f"Low:     ${ticker['low']:>12,.2f}")
    print(f"Close:   ${ticker['close']:>12,.2f}")
    print(f"Change:  {ticker['price_change_percent']:>11,.2f}%")
    print(f"Volume:  {ticker['volume']:>12,.2f} BTC")
    print(f"Trades:  {ticker['trades']:>12,}")
    print()
    
    # Get order book
    print("="*60)
    print("BTC Order Book (Top 5 Levels)")
    print("="*60)
    order_book = client.get_order_book('BTCUSDT', limit=5)
    print("\nAsks (Sell Orders):")
    for price, qty in reversed(order_book['asks']):
        print(f"  ${price:>10,.2f}  {qty:>8,.4f} BTC")
    print("\nBids (Buy Orders):")
    for price, qty in order_book['bids']:
        print(f"  ${price:>10,.2f}  {qty:>8,.4f} BTC")
    print()
    
    # Get recent historical data
    print("="*60)
    print("BTC Recent 1-Hour Candles (Last 10)")
    print("="*60)
    klines = client.get_klines('BTCUSDT', '1h', limit=10)
    print(klines[['open', 'high', 'low', 'close', 'volume']].tail(10))
    print()
    
    print("="*60)
    print("Real-time data integration complete!")
    print("All data is fetched live from Binance exchange")
    print("="*60)
