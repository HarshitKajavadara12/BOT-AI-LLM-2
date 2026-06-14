"""
QUANTUM-FORGE: Execution Manager
===================================
P2 — Wires VWAP, TWAP, and Implementation Shortfall algorithms
into the live trading pipeline.

This module selects the optimal execution algorithm based on:
  - Order size (large orders → VWAP/TWAP, small → market)
  - Market conditions (volatility, spread, liquidity)
  - Urgency (signal strength, regime momentum)
  
It also connects to the Binance order API for real execution
when live_mode=True.
"""

import os
import sys
import time
import logging
import hmac
import hashlib
import requests
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger("ExecutionManager")


class ExecutionMode(Enum):
    PAPER = "PAPER"         # Internal bookkeeping only (default)
    LIVE = "LIVE"           # Real orders on Binance


class AlgoType(Enum):
    MARKET = "MARKET"       # Immediate execution
    TWAP = "TWAP"           # Time-weighted slicing
    VWAP = "VWAP"           # Volume-weighted slicing
    IS = "IS"               # Implementation Shortfall (minimize impact)


@dataclass
class ExecutionOrder:
    """An order to be executed."""
    symbol: str
    side: str           # "BUY" or "SELL"
    quantity: float
    target_price: float
    algo: AlgoType
    urgency: float      # 0-1, higher = more aggressive
    
    # Filled by execution
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    fees_paid: float = 0.0
    slippage: float = 0.0
    status: str = "PENDING"
    exchange_order_id: Optional[str] = None


class ExecutionManager:
    """
    Manages trade execution with algorithm selection and exchange connectivity.
    
    Paper mode (default): Updates internal state, no real orders.
    Live mode: Places real orders on Binance via REST API.
    """
    
    FEE_RATE = 0.001  # 0.1% Binance spot fee
    
    def __init__(
        self,
        mode: ExecutionMode = ExecutionMode.PAPER,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = False,
    ):
        self.mode = mode
        self.api_key = api_key or os.environ.get('BINANCE_API_KEY')
        self.api_secret = api_secret or os.environ.get('BINANCE_SECRET_KEY')
        self.testnet = testnet
        
        if testnet:
            self.base_url = "https://testnet.binance.vision/api/v3"
        else:
            self.base_url = "https://api.binance.com/api/v3"
        
        # Initialize execution algorithms
        self.twap_algo = None
        self.vwap_algo = None
        self.is_algo = None
        self._init_algorithms()
        
        # Execution log
        self.execution_log: List[Dict] = []
        
        # Symbol info cache (lot sizes, tick sizes)
        self._symbol_info: Dict[str, Dict] = {}
        
        logger.info(f"ExecutionManager initialized: mode={mode.value}, testnet={testnet}")
    
    def _init_algorithms(self):
        """Initialize execution algorithms."""
        try:
            from core.execution_algorithms.twap_algorithm import TWAPAlgorithm, TWAPParameters
            self.twap_cls = TWAPAlgorithm
            self.twap_params_cls = TWAPParameters
            logger.info("  [OK] TWAP algorithm loaded")
        except Exception as e:
            logger.warning(f"  [SKIP] TWAP: {e}")
            self.twap_cls = None
        
        try:
            from core.execution_algorithms.vwap_algorithm import VWAPAlgorithm, VWAPParameters
            self.vwap_cls = VWAPAlgorithm
            self.vwap_params_cls = VWAPParameters
            logger.info("  [OK] VWAP algorithm loaded")
        except Exception as e:
            logger.warning(f"  [SKIP] VWAP: {e}")
            self.vwap_cls = None
        
        try:
            from core.execution_algorithms.implementation_shortfall import (
                ImplementationShortfallAlgorithm, ISParameters
            )
            self.is_cls = ImplementationShortfallAlgorithm
            self.is_params_cls = ISParameters
            logger.info("  [OK] Implementation Shortfall algorithm loaded")
        except Exception as e:
            logger.warning(f"  [SKIP] IS: {e}")
            self.is_cls = None
    
    def select_algorithm(
        self,
        order_value_usd: float,
        volatility: float,
        signal_strength: float,
    ) -> AlgoType:
        """
        Select the optimal execution algorithm.
        
        Rules:
          - Small orders (<$500): MARKET (immediate)
          - Medium orders ($500-$5000): TWAP if available
          - Large orders (>$5000): VWAP if available
          - High urgency (strength>0.8): MARKET (don't wait)
          - High volatility: IS (minimize market impact)
        """
        # High urgency → immediate execution
        if signal_strength > 0.8:
            return AlgoType.MARKET
        
        # Small orders → not worth slicing
        if order_value_usd < 500:
            return AlgoType.MARKET
        
        # High volatility → minimize impact
        if volatility > 0.05 and self.is_cls is not None:
            return AlgoType.IS
        
        # Large orders → VWAP for volume participation
        if order_value_usd > 5000 and self.vwap_cls is not None:
            return AlgoType.VWAP
        
        # Medium orders → TWAP for time-spreading
        if self.twap_cls is not None:
            return AlgoType.TWAP
        
        return AlgoType.MARKET
    
    def execute(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        signal_strength: float = 0.5,
        volatility: float = 0.02,
    ) -> ExecutionOrder:
        """
        Execute a trade using the optimal algorithm.
        
        Args:
            symbol: Trading pair
            side: "BUY" or "SELL"
            quantity: Amount to trade
            price: Current market price
            signal_strength: Signal strength for urgency
            volatility: Current volatility for algo selection
        
        Returns:
            ExecutionOrder with fill details
        """
        order_value = quantity * price
        algo = self.select_algorithm(order_value, volatility, signal_strength)
        
        order = ExecutionOrder(
            symbol=symbol,
            side=side,
            quantity=quantity,
            target_price=price,
            algo=algo,
            urgency=signal_strength,
        )
        
        logger.info(f"[EXEC] {side} {quantity:.6f} {symbol} @ ${price:,.2f} via {algo.value} ({self.mode.value})")
        
        if self.mode == ExecutionMode.PAPER:
            order = self._execute_paper(order)
        elif self.mode == ExecutionMode.LIVE:
            order = self._execute_live(order)
        
        # Log execution
        self.execution_log.append({
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'target_price': price,
            'fill_price': order.avg_fill_price,
            'fees': order.fees_paid,
            'slippage': order.slippage,
            'algo': algo.value,
            'mode': self.mode.value,
            'status': order.status,
        })
        
        return order
    
    def _execute_paper(self, order: ExecutionOrder) -> ExecutionOrder:
        """
        Paper trade execution with realistic slippage simulation.
        """
        # Simulate slippage based on order size and algo
        if order.algo == AlgoType.MARKET:
            # Market order: ~2-5 bps slippage
            slippage_bps = np.random.uniform(2, 5) * 0.0001
        elif order.algo == AlgoType.TWAP:
            # TWAP: ~1-3 bps
            slippage_bps = np.random.uniform(1, 3) * 0.0001
        elif order.algo == AlgoType.VWAP:
            # VWAP: ~0.5-2 bps
            slippage_bps = np.random.uniform(0.5, 2) * 0.0001
        elif order.algo == AlgoType.IS:
            # IS: ~0.5-1.5 bps
            slippage_bps = np.random.uniform(0.5, 1.5) * 0.0001
        else:
            slippage_bps = 0.0002
        
        # Apply slippage
        if order.side == "BUY":
            fill_price = order.target_price * (1 + slippage_bps)
        else:
            fill_price = order.target_price * (1 - slippage_bps)
        
        # Calculate fees
        fill_value = order.quantity * fill_price
        fees = fill_value * self.FEE_RATE
        
        order.filled_quantity = order.quantity
        order.avg_fill_price = fill_price
        order.fees_paid = fees
        order.slippage = (fill_price - order.target_price) / order.target_price
        order.status = "FILLED"
        
        return order
    
    def _execute_live(self, order: ExecutionOrder) -> ExecutionOrder:
        """
        Execute a real order on Binance with retry + fill confirmation.
        
        For MARKET orders: Single market order via REST API.
        For TWAP/VWAP/IS: Currently falls back to LIMIT with tolerance.
        Retries up to 3 times on transient failures.
        """
        if not self.api_key or not self.api_secret:
            logger.error("Cannot execute live order: API keys not configured")
            logger.error("Set BINANCE_API_KEY and BINANCE_SECRET_KEY environment variables")
            order.status = "REJECTED"
            return order
        
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                result_order = self._send_order(order)
                
                # If order is NEW/PARTIALLY_FILLED, poll for confirmation
                if result_order.status in ("NEW", "PARTIALLY_FILLED"):
                    result_order = self._poll_order_status(result_order, timeout_seconds=30)
                
                if result_order.status == "FILLED":
                    return result_order
                
                if result_order.status == "REJECTED":
                    return result_order  # Don't retry rejections
                    
                # Transient failure — retry
                if attempt < max_retries:
                    delay = 2 ** attempt
                    logger.warning(
                        f"[LIVE] Attempt {attempt}/{max_retries} status={result_order.status} "
                        f"— retrying in {delay}s"
                    )
                    time.sleep(delay)
                else:
                    return result_order
                    
            except Exception as e:
                logger.error(f"[LIVE] Attempt {attempt}/{max_retries} failed: {e}")
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                else:
                    order.status = "FAILED"
        
        return order
    
    def _send_order(self, order: ExecutionOrder) -> ExecutionOrder:
        """Send a single order to Binance REST API."""
        
        try:
            # Get lot size for the symbol
            lot_size = self._get_lot_size(order.symbol)
            adjusted_qty = self._adjust_quantity(order.quantity, lot_size)
            
            if adjusted_qty <= 0:
                logger.error(f"Quantity {order.quantity} below minimum lot size")
                order.status = "REJECTED"
                return order
            
            # Place order
            params = {
                'symbol': order.symbol,
                'side': order.side.upper(),
                'type': 'MARKET',
                'quantity': f"{adjusted_qty:.8f}".rstrip('0').rstrip('.'),
                'timestamp': str(int(time.time() * 1000)),
                'recvWindow': '5000',
            }
            
            # Add limit price for non-market algos
            if order.algo != AlgoType.MARKET:
                # For advanced algos, use LIMIT order with tolerance
                if order.side == "BUY":
                    limit_price = order.target_price * 1.002  # 0.2% above
                else:
                    limit_price = order.target_price * 0.998  # 0.2% below
                
                tick_size = self._get_tick_size(order.symbol)
                limit_price = round(limit_price / tick_size) * tick_size
                
                params['type'] = 'LIMIT'
                params['price'] = f"{limit_price:.8f}".rstrip('0').rstrip('.')
                params['timeInForce'] = 'GTC'
            
            # Sign request
            query_string = '&'.join(f"{k}={v}" for k, v in params.items())
            signature = hmac.new(
                self.api_secret.encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256,
            ).hexdigest()
            params['signature'] = signature
            
            headers = {'X-MBX-APIKEY': self.api_key}
            
            response = requests.post(
                f"{self.base_url}/order",
                params=params,
                headers=headers,
                timeout=10,
            )
            
            result = response.json()
            
            if response.status_code == 200:
                order.exchange_order_id = str(result.get('orderId', ''))
                order.status = result.get('status', 'FILLED')
                
                # Parse fills
                fills = result.get('fills', [])
                if fills:
                    total_qty = sum(float(f['qty']) for f in fills)
                    total_cost = sum(float(f['qty']) * float(f['price']) for f in fills)
                    total_fees = sum(float(f['commission']) for f in fills)
                    
                    order.filled_quantity = total_qty
                    order.avg_fill_price = total_cost / total_qty if total_qty > 0 else order.target_price
                    order.fees_paid = total_fees
                else:
                    order.filled_quantity = float(result.get('executedQty', order.quantity))
                    order.avg_fill_price = float(result.get('price', order.target_price))
                    order.fees_paid = order.filled_quantity * order.avg_fill_price * self.FEE_RATE
                
                order.slippage = (order.avg_fill_price - order.target_price) / order.target_price
                
                logger.info(
                    f"[LIVE] Order filled: {order.filled_quantity:.6f} {order.symbol} "
                    f"@ ${order.avg_fill_price:,.2f} (slippage: {order.slippage*10000:.1f}bps)"
                )
            else:
                error_msg = result.get('msg', 'Unknown error')
                logger.error(f"[LIVE] Order rejected: {error_msg}")
                order.status = "REJECTED"
                
        except Exception as e:
            logger.error(f"[LIVE] Order execution failed: {e}")
            order.status = "FAILED"
        
        return order
    
    def _poll_order_status(
        self, order: ExecutionOrder, timeout_seconds: float = 30, poll_interval: float = 2.0
    ) -> ExecutionOrder:
        """Poll Binance for order status until filled, cancelled, or timeout."""
        if not order.exchange_order_id:
            return order
        
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            try:
                params = {
                    'symbol': order.symbol,
                    'orderId': order.exchange_order_id,
                    'timestamp': str(int(time.time() * 1000)),
                    'recvWindow': '5000',
                }
                query_string = '&'.join(f"{k}={v}" for k, v in params.items())
                sig = hmac.new(
                    self.api_secret.encode('utf-8'),
                    query_string.encode('utf-8'),
                    hashlib.sha256,
                ).hexdigest()
                params['signature'] = sig
                headers = {'X-MBX-APIKEY': self.api_key}
                
                resp = requests.get(
                    f"{self.base_url}/order", params=params, headers=headers, timeout=10,
                )
                data = resp.json()
                status = data.get('status', order.status)
                order.status = status
                
                if status == "FILLED":
                    order.filled_quantity = float(data.get('executedQty', order.quantity))
                    cum_quote = float(data.get('cummulativeQuoteQty', 0))
                    if order.filled_quantity > 0 and cum_quote > 0:
                        order.avg_fill_price = cum_quote / order.filled_quantity
                    order.fees_paid = order.filled_quantity * order.avg_fill_price * self.FEE_RATE
                    order.slippage = (order.avg_fill_price - order.target_price) / order.target_price
                    logger.info(f"[LIVE] Order {order.exchange_order_id} confirmed FILLED")
                    return order
                
                if status in ("CANCELED", "EXPIRED", "REJECTED"):
                    logger.warning(f"[LIVE] Order {order.exchange_order_id} terminal: {status}")
                    return order
                
            except Exception as e:
                logger.debug(f"Poll error: {e}")
            
            time.sleep(poll_interval)
        
        logger.warning(f"[LIVE] Order {order.exchange_order_id} poll timeout ({timeout_seconds}s)")
        return order
    
    def reconcile_positions(self) -> Dict[str, Dict]:
        """
        Fetch real account balances from Binance for reconciliation.
        Returns: {asset: {free, locked, total}} for non-zero balances.
        """
        if not self.api_key or not self.api_secret:
            logger.warning("Cannot reconcile: no API keys")
            return {}
        
        try:
            params = {
                'timestamp': str(int(time.time() * 1000)),
                'recvWindow': '5000',
            }
            query_string = '&'.join(f"{k}={v}" for k, v in params.items())
            sig = hmac.new(
                self.api_secret.encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256,
            ).hexdigest()
            params['signature'] = sig
            headers = {'X-MBX-APIKEY': self.api_key}
            
            resp = requests.get(
                f"{self.base_url}/account", params=params, headers=headers, timeout=10,
            )
            data = resp.json()
            
            balances = {}
            for b in data.get('balances', []):
                free, locked = float(b['free']), float(b['locked'])
                if free + locked > 0:
                    balances[b['asset']] = {'free': free, 'locked': locked, 'total': free + locked}
            
            logger.info(f"[RECONCILE] {len(balances)} non-zero balances fetched")
            return balances
            
        except Exception as e:
            logger.error(f"[RECONCILE] Failed: {e}")
            return {}
    
    def _get_lot_size(self, symbol: str) -> float:
        """Get minimum lot size (step size) for a symbol."""
        if symbol not in self._symbol_info:
            self._load_symbol_info(symbol)
        return self._symbol_info.get(symbol, {}).get('lot_size', 0.00001)
    
    def _get_tick_size(self, symbol: str) -> float:
        """Get minimum price increment for a symbol."""
        if symbol not in self._symbol_info:
            self._load_symbol_info(symbol)
        return self._symbol_info.get(symbol, {}).get('tick_size', 0.01)
    
    def _load_symbol_info(self, symbol: str):
        """Load symbol trading rules from Binance."""
        try:
            response = requests.get(
                f"https://api.binance.com/api/v3/exchangeInfo?symbol={symbol}",
                timeout=10,
            )
            data = response.json()
            
            for s in data.get('symbols', []):
                if s['symbol'] == symbol:
                    for f in s.get('filters', []):
                        if f['filterType'] == 'LOT_SIZE':
                            self._symbol_info.setdefault(symbol, {})['lot_size'] = float(f['stepSize'])
                        elif f['filterType'] == 'PRICE_FILTER':
                            self._symbol_info.setdefault(symbol, {})['tick_size'] = float(f['tickSize'])
                    break
        except:
            self._symbol_info[symbol] = {'lot_size': 0.00001, 'tick_size': 0.01}
    
    def _adjust_quantity(self, quantity: float, lot_size: float) -> float:
        """Round quantity to valid lot size."""
        if lot_size <= 0:
            return quantity
        return float(int(quantity / lot_size) * lot_size)
    
    def get_execution_stats(self) -> Dict:
        """Get execution statistics."""
        if not self.execution_log:
            return {'total_orders': 0}
        
        total = len(self.execution_log)
        filled = sum(1 for o in self.execution_log if o['status'] == 'FILLED')
        total_fees = sum(o.get('fees', 0) for o in self.execution_log)
        avg_slippage = np.mean([abs(o.get('slippage', 0)) for o in self.execution_log])
        
        algo_counts = {}
        for o in self.execution_log:
            algo = o.get('algo', 'UNKNOWN')
            algo_counts[algo] = algo_counts.get(algo, 0) + 1
        
        return {
            'total_orders': total,
            'filled': filled,
            'fill_rate': filled / total if total > 0 else 0,
            'total_fees': total_fees,
            'avg_slippage_bps': avg_slippage * 10000,
            'algo_distribution': algo_counts,
            'mode': self.mode.value,
        }
