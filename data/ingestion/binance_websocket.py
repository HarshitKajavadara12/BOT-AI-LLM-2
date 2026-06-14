"""
Binance WebSocket Client for Real-Time Streaming Data
Streams live market data: trades, order book, klines, ticker
"""

import json
import time
import logging
import threading
from typing import Dict, List, Callable, Optional
from datetime import datetime
import websocket
from collections import deque, defaultdict
import pandas as pd

logger = logging.getLogger("BinanceWebSocket")


class BinanceWebSocket:
    """Real-time WebSocket client for Binance market data streams"""
    
    def __init__(self, testnet: bool = False):
        """
        Initialize Binance WebSocket client
        
        Args:
            testnet: Use testnet endpoint
        """
        self.testnet = testnet
        
        if testnet:
            self.base_url = "wss://testnet.binance.vision/ws"
        else:
            self.base_url = "wss://stream.binance.com:9443/ws"
        
        self.ws = None
        self.thread = None
        self.running = False
        
        # Data storage
        self.latest_prices = {}
        self.order_books = {}
        self.recent_trades = defaultdict(lambda: deque(maxlen=100))
        self.klines = defaultdict(dict)
        self.tickers = {}
        
        # Callbacks
        self.callbacks = {
            'trade': [],
            'depth': [],
            'kline': [],
            'ticker': [],
            'aggTrade': []
        }
        
        # Connection management
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.reconnect_delay = 2  # initial delay (seconds)
        self.last_streams: List[str] = []  # stored for reconnection
        self._last_pong = time.time()
        
    def _on_message(self, ws, message):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(message)
            
            # Check if this is a combined streams message (has 'stream' and 'data' fields)
            if 'stream' in data and 'data' in data:
                # Extract the actual event data
                data = data['data']
            
            # Debug: print first few messages
            if not hasattr(self, '_msg_count'):
                self._msg_count = 0
            self._msg_count += 1
            if self._msg_count <= 5:
                print(f"[DEBUG] WebSocket message #{self._msg_count}: {data.get('e', 'unknown')} for {data.get('s', 'N/A')}")
            
            # Handle different message types
            if 'e' in data:
                event_type = data['e']
                
                if event_type == 'trade':
                    self._handle_trade(data)
                elif event_type == 'depthUpdate':
                    self._handle_depth_update(data)
                elif event_type == 'kline':
                    self._handle_kline(data)
                elif event_type == '24hrTicker':
                    self._handle_ticker(data)
                elif event_type == 'aggTrade':
                    self._handle_agg_trade(data)
                    
        except json.JSONDecodeError as e:
            print(f"Failed to decode WebSocket message: {e}")
        except Exception as e:
            print(f"Error handling WebSocket message: {e}")
    
    def _handle_trade(self, data: Dict):
        """Handle trade updates"""
        symbol = data['s']
        trade_data = {
            'symbol': symbol,
            'trade_id': data['t'],
            'price': float(data['p']),
            'quantity': float(data['q']),
            'time': data['T'],
            'is_buyer_maker': data['m']
        }
        
        # Update latest price
        self.latest_prices[symbol] = trade_data['price']
        
        # Store recent trade
        self.recent_trades[symbol].append(trade_data)
        
        # Trigger callbacks
        for callback in self.callbacks['trade']:
            callback(trade_data)
    
    def _handle_agg_trade(self, data: Dict):
        """Handle aggregate trade updates"""
        symbol = data['s']
        trade_data = {
            'symbol': symbol,
            'agg_trade_id': data['a'],
            'price': float(data['p']),
            'quantity': float(data['q']),
            'first_trade_id': data['f'],
            'last_trade_id': data['l'],
            'time': data['T'],
            'is_buyer_maker': data['m']
        }
        
        # Update latest price
        self.latest_prices[symbol] = trade_data['price']
        
        # Trigger callbacks
        for callback in self.callbacks['aggTrade']:
            callback(trade_data)
    
    def _handle_depth_update(self, data: Dict):
        """Handle order book depth updates"""
        symbol = data['s']
        
        # Initialize order book if needed
        if symbol not in self.order_books:
            self.order_books[symbol] = {
                'bids': {},
                'asks': {},
                'last_update_id': 0
            }
        
        # Update bids
        for bid in data['b']:
            price, qty = float(bid[0]), float(bid[1])
            if qty == 0:
                self.order_books[symbol]['bids'].pop(price, None)
            else:
                self.order_books[symbol]['bids'][price] = qty
        
        # Update asks
        for ask in data['a']:
            price, qty = float(ask[0]), float(ask[1])
            if qty == 0:
                self.order_books[symbol]['asks'].pop(price, None)
            else:
                self.order_books[symbol]['asks'][price] = qty
        
        self.order_books[symbol]['last_update_id'] = data['u']
        
        # Trigger callbacks
        for callback in self.callbacks['depth']:
            callback({
                'symbol': symbol,
                'bids': list(self.order_books[symbol]['bids'].items()),
                'asks': list(self.order_books[symbol]['asks'].items())
            })
    
    def _handle_kline(self, data: Dict):
        """Handle kline/candlestick updates"""
        symbol = data['s']
        kline_data = data['k']
        
        interval = kline_data['i']
        kline_info = {
            'symbol': symbol,
            'interval': interval,
            'open_time': kline_data['t'],
            'close_time': kline_data['T'],
            'open': float(kline_data['o']),
            'high': float(kline_data['h']),
            'low': float(kline_data['l']),
            'close': float(kline_data['c']),
            'volume': float(kline_data['v']),
            'quote_volume': float(kline_data['q']),
            'trades': kline_data['n'],
            'is_closed': kline_data['x']
        }
        
        # Store kline
        if symbol not in self.klines:
            self.klines[symbol] = {}
        self.klines[symbol][interval] = kline_info
        
        # Update latest price
        self.latest_prices[symbol] = kline_info['close']
        
        # Trigger callbacks
        for callback in self.callbacks['kline']:
            callback(kline_info)
    
    def _handle_ticker(self, data: Dict):
        """Handle 24hr ticker updates"""
        symbol = data['s']
        ticker_data = {
            'symbol': symbol,
            'price_change': float(data['p']),
            'price_change_percent': float(data['P']),
            'weighted_avg_price': float(data['w']),
            'prev_close': float(data['x']),
            'last_price': float(data['c']),
            'last_qty': float(data['Q']),
            'bid': float(data['b']),
            'bid_qty': float(data['B']),
            'ask': float(data['a']),
            'ask_qty': float(data['A']),
            'open': float(data['o']),
            'high': float(data['h']),
            'low': float(data['l']),
            'volume': float(data['v']),
            'quote_volume': float(data['q']),
            'open_time': data['O'],
            'close_time': data['C'],
            'trades': data['n']
        }
        
        # Update ticker
        self.tickers[symbol] = ticker_data
        
        # Update latest price
        self.latest_prices[symbol] = ticker_data['last_price']
        
        # Trigger callbacks
        for callback in self.callbacks['ticker']:
            callback(ticker_data)
    
    def _on_error(self, ws, error):
        """Handle WebSocket errors"""
        logger.error(f"WebSocket error: {error}")
    
    def _on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close"""
        logger.warning(f"WebSocket closed: {close_status_code} - {close_msg}")
        self.running = False
        
        # Attempt reconnection with exponential backoff
        if self.reconnect_attempts < self.max_reconnect_attempts:
            self.reconnect_attempts += 1
            delay = min(self.reconnect_delay * (2 ** (self.reconnect_attempts - 1)), 120)
            logger.info(
                f"Reconnecting in {delay:.0f}s ... "
                f"Attempt {self.reconnect_attempts}/{self.max_reconnect_attempts}"
            )
            time.sleep(delay)
            self.reconnect()
        else:
            logger.error("Max reconnection attempts reached — giving up")
    
    def _on_open(self, ws):
        """Handle WebSocket open"""
        logger.info("WebSocket connection opened")
        self.reconnect_attempts = 0
        self._last_pong = time.time()
    
    def _build_stream_url(self, streams: List[str]) -> str:
        """Build WebSocket URL with multiple streams"""
        if len(streams) == 1:
            return f"{self.base_url}/{streams[0]}"
        else:
            stream_names = '/'.join(streams)
            return f"{self.base_url.replace('/ws', '/stream')}?streams={stream_names}"
    
    def subscribe_ticker(self, symbols: List[str]):
        """Subscribe to 24hr ticker updates"""
        streams = [f"{symbol.lower()}@ticker" for symbol in symbols]
        self._subscribe(streams)
    
    def subscribe_trades(self, symbols: List[str]):
        """Subscribe to individual trade updates"""
        streams = [f"{symbol.lower()}@trade" for symbol in symbols]
        self._subscribe(streams)
    
    def subscribe_agg_trades(self, symbols: List[str]):
        """Subscribe to aggregate trade updates"""
        streams = [f"{symbol.lower()}@aggTrade" for symbol in symbols]
        self._subscribe(streams)
    
    def subscribe_orderbook(self, symbols: List[str], depth: int = 20, update_speed: str = '100ms'):
        """
        Subscribe to order book depth updates
        
        Args:
            symbols: List of trading pairs
            depth: Order book depth (5, 10, 20)
            update_speed: Update speed ('100ms' or '1000ms')
        """
        if update_speed == '100ms':
            stream_suffix = f"@depth{depth}@100ms"
        else:
            stream_suffix = f"@depth{depth}"
        
        streams = [f"{symbol.lower()}{stream_suffix}" for symbol in symbols]
        self._subscribe(streams)
    
    def subscribe_klines(self, symbols: List[str], interval: str = '1m'):
        """
        Subscribe to kline/candlestick updates
        
        Args:
            symbols: List of trading pairs
            interval: Kline interval (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M)
        """
        streams = [f"{symbol.lower()}@kline_{interval}" for symbol in symbols]
        self._subscribe(streams)
    
    def _subscribe(self, streams: List[str]):
        """Subscribe to WebSocket streams"""
        self.last_streams = streams  # persist for reconnection
        if not self.running:
            url = self._build_stream_url(streams)
            self.ws = websocket.WebSocketApp(
                url,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
                on_open=self._on_open
            )
            
            self.running = True
            self.thread = threading.Thread(target=self.ws.run_forever, kwargs={"ping_interval": 20, "ping_timeout": 10})
            self.thread.daemon = True
            self.thread.start()
    
    def add_callback(self, event_type: str, callback: Callable):
        """
        Add callback for specific event type
        
        Args:
            event_type: Type of event ('trade', 'depth', 'kline', 'ticker', 'aggTrade')
            callback: Callback function that takes data dict as argument
        """
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
    
    def reconnect(self):
        """Reconnect WebSocket using stored streams."""
        if self.ws:
            try:
                self.ws.close()
            except Exception:
                pass
        self.running = False
        
        if self.last_streams:
            self._subscribe(self.last_streams)
        else:
            logger.warning("Cannot reconnect — no previous streams stored")
    
    def start(self):
        """Start WebSocket connection (kept for compatibility)"""
        pass  # Connection starts automatically when subscribing
    
    def stop(self):
        """Stop WebSocket connection"""
        self.running = False
        if self.ws:
            self.ws.close()
        if self.thread:
            self.thread.join(timeout=2)
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price for symbol"""
        return self.latest_prices.get(symbol)
    
    def get_latest_prices(self) -> Dict[str, float]:
        """Get all latest prices"""
        return self.latest_prices.copy()
    
    def get_order_book(self, symbol: str, limit: int = 20) -> Optional[Dict]:
        """
        Get current order book snapshot
        
        Args:
            symbol: Trading pair
            limit: Number of levels to return
            
        Returns:
            Dict with bids and asks
        """
        if symbol not in self.order_books:
            return None
        
        ob = self.order_books[symbol]
        
        # Sort and limit
        bids = sorted(ob['bids'].items(), key=lambda x: x[0], reverse=True)[:limit]
        asks = sorted(ob['asks'].items(), key=lambda x: x[0])[:limit]
        
        return {
            'symbol': symbol,
            'bids': bids,
            'asks': asks,
            'timestamp': int(time.time() * 1000)
        }
    
    def get_recent_trades(self, symbol: str, limit: int = 50) -> List[Dict]:
        """Get recent trades for symbol"""
        trades = list(self.recent_trades.get(symbol, []))
        return trades[-limit:]
    
    def get_ticker(self, symbol: str) -> Optional[Dict]:
        """Get latest 24hr ticker data"""
        return self.tickers.get(symbol)
    
    def get_kline(self, symbol: str, interval: str = '1m') -> Optional[Dict]:
        """Get latest kline data"""
        if symbol in self.klines and interval in self.klines[symbol]:
            return self.klines[symbol][interval]
        return None


class MultiStreamWebSocket:
    """Manage multiple WebSocket connections for different data types"""
    
    def __init__(self, symbols: List[str], testnet: bool = False):
        """
        Initialize multi-stream WebSocket manager
        
        Args:
            symbols: List of trading pairs to monitor
            testnet: Use testnet
        """
        self.symbols = symbols
        self.testnet = testnet
        
        # Create separate WebSocket clients for different data types
        self.ticker_ws = BinanceWebSocket(testnet)
        self.trade_ws = BinanceWebSocket(testnet)
        self.depth_ws = BinanceWebSocket(testnet)
        self.kline_ws = BinanceWebSocket(testnet)
        
        # Aggregated data
        self.market_data = defaultdict(dict)
    
    def start(self):
        """Start all WebSocket streams"""
        print(f"Starting WebSocket streams for {len(self.symbols)} symbols...")
        
        # Subscribe to different data types
        self.ticker_ws.subscribe_ticker(self.symbols)
        self.trade_ws.subscribe_agg_trades(self.symbols)
        self.depth_ws.subscribe_orderbook(self.symbols, depth=20, update_speed='100ms')
        self.kline_ws.subscribe_klines(self.symbols, interval='1m')
        
        # Add callbacks to aggregate data
        self.ticker_ws.add_callback('ticker', self._on_ticker)
        self.trade_ws.add_callback('aggTrade', self._on_trade)
        self.depth_ws.add_callback('depth', self._on_depth)
        self.kline_ws.add_callback('kline', self._on_kline)
        
        print("WebSocket streams started successfully")
    
    def _on_ticker(self, data: Dict):
        """Aggregate ticker data"""
        symbol = data['symbol']
        self.market_data[symbol]['ticker'] = data
        self.market_data[symbol]['last_price'] = data['last_price']  # Update last_price from ticker too
        self.market_data[symbol]['last_update'] = time.time()
    
    def _on_trade(self, data: Dict):
        """Aggregate trade data - REAL-TIME price updates"""
        symbol = data['symbol']
        self.market_data[symbol]['last_trade'] = data
        self.market_data[symbol]['last_price'] = data['price']  # Update price from trades
        self.market_data[symbol]['last_update'] = time.time()  # Track update time
    
    def _on_depth(self, data: Dict):
        """Aggregate depth data"""
        symbol = data['symbol']
        self.market_data[symbol]['order_book'] = data
    
    def _on_kline(self, data: Dict):
        """Aggregate kline data"""
        symbol = data['symbol']
        self.market_data[symbol]['kline'] = data
    
    def get_market_data(self, symbol: str) -> Dict:
        """Get all aggregated market data for a symbol"""
        return self.market_data.get(symbol, {})
    
    def get_all_prices(self) -> Dict[str, float]:
        """Get latest prices for all symbols"""
        prices = {}
        for symbol, data in self.market_data.items():
            # Try to get price from ticker first (most accurate)
            if 'ticker' in data and 'last_price' in data['ticker']:
                prices[symbol] = data['ticker']['last_price']
            # Fallback to last_price field
            elif 'last_price' in data:
                prices[symbol] = data['last_price']
        return prices
    
    def stop(self):
        """Stop all WebSocket connections"""
        print("Stopping WebSocket streams...")
        self.ticker_ws.stop()
        self.trade_ws.stop()
        self.depth_ws.stop()
        self.kline_ws.stop()
        print("WebSocket streams stopped")


# Example usage
if __name__ == "__main__":
    import time
    
    # Define crypto pairs to monitor
    crypto_pairs = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']
    
    print("="*60)
    print("Binance Real-Time WebSocket Streaming")
    print("="*60)
    print(f"Monitoring: {', '.join(crypto_pairs)}")
    print()
    
    # Initialize multi-stream WebSocket
    ws = MultiStreamWebSocket(crypto_pairs)
    ws.start()
    
    # Let it run for 30 seconds
    print("Streaming live data for 30 seconds...")
    print("="*60)
    print()
    
    for i in range(30):
        time.sleep(1)
        
        # Display real-time prices
        prices = ws.get_all_prices()
        print(f"\rTime: {i+1}s | Prices: ", end='')
        for symbol, price in prices.items():
            print(f"{symbol}: ${price:,.2f} | ", end='')
    
    print("\n\n" + "="*60)
    print("Final Market Data Summary")
    print("="*60)
    
    for symbol in crypto_pairs:
        data = ws.get_market_data(symbol)
        if 'ticker' in data:
            ticker = data['ticker']
            print(f"\n{symbol}:")
            print(f"  Price: ${ticker['last_price']:,.2f}")
            print(f"  24h Change: {ticker['price_change_percent']:.2f}%")
            print(f"  24h Volume: {ticker['volume']:,.2f}")
            print(f"  High: ${ticker['high']:,.2f}")
            print(f"  Low: ${ticker['low']:,.2f}")
    
    ws.stop()
    print("\n" + "="*60)
    print("Real-time WebSocket streaming demo complete!")
    print("="*60)
