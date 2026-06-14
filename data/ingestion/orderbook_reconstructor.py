"""
Order Book Reconstructor
=======================

Advanced order book reconstruction system for building accurate market depth
from tick-by-tick data feeds. Handles multiple exchanges, latency correction,
and real-time order book maintenance.

Features:
- Real-time order book reconstruction from L2/L3 data
- Multi-exchange order book aggregation
- Latency correction and time synchronization
- Order book validation and integrity checks
- Snapshot and incremental update handling
- Cross-validation between data sources
- Performance optimization for high-frequency updates
- Historical order book replay capabilities

Author: Quantum Forge Data Team
Date: November 2025
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, DefaultDict
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, OrderedDict
import pandas as pd
import numpy as np
import json
import threading
from concurrent.futures import ThreadPoolExecutor
import bisect
import heapq
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

class BookSide(Enum):
    """Order book sides."""
    BID = "bid"
    ASK = "ask"

class UpdateType(Enum):
    """Order book update types."""
    SNAPSHOT = "snapshot"
    INCREMENTAL = "incremental"
    TRADE = "trade"

@dataclass
class PriceLevel:
    """Individual price level in order book."""
    price: float
    size: float
    count: int = 1
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        # Ensure positive values
        self.price = max(0.0, self.price)
        self.size = max(0.0, self.size)
        self.count = max(0, self.count)

@dataclass
class OrderBookUpdate:
    """Order book update message."""
    symbol: str
    exchange: str
    timestamp: datetime
    update_type: UpdateType
    sequence: Optional[int] = None
    bids: List[Tuple[float, float]] = field(default_factory=list)  # [(price, size), ...]
    asks: List[Tuple[float, float]] = field(default_factory=list)  # [(price, size), ...]
    trades: List[Dict[str, Any]] = field(default_factory=list)
    received_at: datetime = field(default_factory=datetime.now)

@dataclass
class OrderBookSnapshot:
    """Complete order book snapshot."""
    symbol: str
    exchange: str
    timestamp: datetime
    sequence: Optional[int] = None
    bids: OrderedDict = field(default_factory=OrderedDict)  # price -> PriceLevel
    asks: OrderedDict = field(default_factory=OrderedDict)  # price -> PriceLevel
    last_update: datetime = field(default_factory=datetime.now)
    
    def get_best_bid(self) -> Optional[PriceLevel]:
        """Get best bid (highest price)."""
        if self.bids:
            price = max(self.bids.keys())
            return self.bids[price]
        return None
    
    def get_best_ask(self) -> Optional[PriceLevel]:
        """Get best ask (lowest price)."""
        if self.asks:
            price = min(self.asks.keys())
            return self.asks[price]
        return None
    
    def get_spread(self) -> Optional[float]:
        """Get bid-ask spread."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid and best_ask:
            return best_ask.price - best_bid.price
        return None
    
    def get_mid_price(self) -> Optional[float]:
        """Get mid price."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid and best_ask:
            return (best_bid.price + best_ask.price) / 2
        return None
    
    def get_depth(self, levels: int = 10) -> Dict[str, List[Tuple[float, float]]]:
        """Get order book depth."""
        # Sort bids descending (highest first)
        sorted_bids = sorted(self.bids.items(), key=lambda x: x[0], reverse=True)
        bid_levels = [(price, level.size) for price, level in sorted_bids[:levels]]
        
        # Sort asks ascending (lowest first)
        sorted_asks = sorted(self.asks.items(), key=lambda x: x[0])
        ask_levels = [(price, level.size) for price, level in sorted_asks[:levels]]
        
        return {
            'bids': bid_levels,
            'asks': ask_levels
        }
    
    def calculate_volume_weighted_price(self, side: BookSide, volume: float) -> Optional[float]:
        """Calculate volume-weighted average price for given volume."""
        levels = self.bids if side == BookSide.BID else self.asks
        
        if not levels:
            return None
        
        # Sort levels appropriately
        if side == BookSide.BID:
            sorted_levels = sorted(levels.items(), key=lambda x: x[0], reverse=True)
        else:
            sorted_levels = sorted(levels.items(), key=lambda x: x[0])
        
        total_volume = 0
        weighted_sum = 0
        
        for price, level in sorted_levels:
            available_volume = min(level.size, volume - total_volume)
            
            if available_volume <= 0:
                break
            
            weighted_sum += price * available_volume
            total_volume += available_volume
            
            if total_volume >= volume:
                break
        
        return weighted_sum / total_volume if total_volume > 0 else None

class OrderBookValidator:
    """Validate order book integrity and detect anomalies."""
    
    def __init__(self, max_spread_pct: float = 10.0, max_level_count: int = 1000):
        self.max_spread_pct = max_spread_pct
        self.max_level_count = max_level_count
        self.validation_stats = defaultdict(int)
    
    def validate_snapshot(self, snapshot: OrderBookSnapshot) -> List[str]:
        """Validate order book snapshot and return issues."""
        issues = []
        
        # Check if book is empty
        if not snapshot.bids and not snapshot.asks:
            issues.append("Empty order book")
            self.validation_stats['empty_book'] += 1
            return issues
        
        # Check bid/ask ordering
        if snapshot.bids:
            bid_prices = list(snapshot.bids.keys())
            if bid_prices != sorted(bid_prices, reverse=True):
                issues.append("Bids not properly ordered")
                self.validation_stats['bid_ordering'] += 1
        
        if snapshot.asks:
            ask_prices = list(snapshot.asks.keys())
            if ask_prices != sorted(ask_prices):
                issues.append("Asks not properly ordered")
                self.validation_stats['ask_ordering'] += 1
        
        # Check for crossed book
        best_bid = snapshot.get_best_bid()
        best_ask = snapshot.get_best_ask()
        
        if best_bid and best_ask and best_bid.price >= best_ask.price:
            issues.append(f"Crossed book: bid {best_bid.price} >= ask {best_ask.price}")
            self.validation_stats['crossed_book'] += 1
        
        # Check spread reasonableness
        if best_bid and best_ask:
            spread_pct = ((best_ask.price - best_bid.price) / best_bid.price) * 100
            if spread_pct > self.max_spread_pct:
                issues.append(f"Excessive spread: {spread_pct:.2f}%")
                self.validation_stats['excessive_spread'] += 1
        
        # Check for zero sizes
        zero_bid_levels = sum(1 for level in snapshot.bids.values() if level.size <= 0)
        zero_ask_levels = sum(1 for level in snapshot.asks.values() if level.size <= 0)
        
        if zero_bid_levels > 0:
            issues.append(f"{zero_bid_levels} bid levels with zero size")
            self.validation_stats['zero_bid_size'] += 1
        
        if zero_ask_levels > 0:
            issues.append(f"{zero_ask_levels} ask levels with zero size")
            self.validation_stats['zero_ask_size'] += 1
        
        # Check level count
        total_levels = len(snapshot.bids) + len(snapshot.asks)
        if total_levels > self.max_level_count:
            issues.append(f"Too many levels: {total_levels}")
            self.validation_stats['excessive_levels'] += 1
        
        return issues
    
    def get_validation_stats(self) -> Dict[str, int]:
        """Get validation statistics."""
        return dict(self.validation_stats)

class LatencyCorrector:
    """Correct for latency and synchronize timestamps."""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.latency_samples = defaultdict(lambda: [])
        self.clock_offsets = defaultdict(float)
    
    def update_latency(self, exchange: str, sent_time: datetime, received_time: datetime):
        """Update latency measurements."""
        latency_ms = (received_time - sent_time).total_seconds() * 1000
        
        self.latency_samples[exchange].append(latency_ms)
        
        # Keep only recent samples
        if len(self.latency_samples[exchange]) > self.window_size:
            self.latency_samples[exchange] = self.latency_samples[exchange][-self.window_size:]
    
    def get_corrected_timestamp(self, exchange: str, timestamp: datetime, received_time: datetime) -> datetime:
        """Get latency-corrected timestamp."""
        if exchange not in self.latency_samples or not self.latency_samples[exchange]:
            return timestamp
        
        # Use median latency for correction
        median_latency_ms = np.median(self.latency_samples[exchange])
        correction = timedelta(milliseconds=median_latency_ms / 2)  # Assume half roundtrip
        
        return timestamp + correction
    
    def synchronize_timestamps(self, updates: List[OrderBookUpdate]) -> List[OrderBookUpdate]:
        """Synchronize timestamps across multiple exchanges."""
        corrected_updates = []
        
        for update in updates:
            corrected_timestamp = self.get_corrected_timestamp(
                update.exchange, 
                update.timestamp, 
                update.received_at
            )
            
            # Create corrected update
            corrected_update = OrderBookUpdate(
                symbol=update.symbol,
                exchange=update.exchange,
                timestamp=corrected_timestamp,
                update_type=update.update_type,
                sequence=update.sequence,
                bids=update.bids,
                asks=update.asks,
                trades=update.trades,
                received_at=update.received_at
            )
            
            corrected_updates.append(corrected_update)
        
        # Sort by corrected timestamp
        corrected_updates.sort(key=lambda x: x.timestamp)
        
        return corrected_updates

class OrderBookReconstructor:
    """
    Main order book reconstruction engine.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize order book reconstructor."""
        self.config = config or {}
        
        # Order books by symbol and exchange
        self.order_books: Dict[str, Dict[str, OrderBookSnapshot]] = defaultdict(dict)
        
        # Update queues for sequencing
        self.update_queues: Dict[str, List[OrderBookUpdate]] = defaultdict(list)
        self.expected_sequences: Dict[str, int] = defaultdict(int)
        
        # Components
        self.validator = OrderBookValidator(
            max_spread_pct=self.config.get('max_spread_pct', 10.0),
            max_level_count=self.config.get('max_level_count', 1000)
        )
        self.latency_corrector = LatencyCorrector(
            window_size=self.config.get('latency_window_size', 1000)
        )
        
        # Processing
        self.is_running = False
        self.processing_thread = None
        self.update_buffer = []
        self.buffer_lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'updates_processed': 0,
            'snapshots_processed': 0,
            'validation_errors': 0,
            'sequence_gaps': 0,
            'last_reset': datetime.now()
        }
        
        # Callbacks
        self.book_update_callbacks = []
        self.validation_error_callbacks = []
    
    def add_book_update_callback(self, callback):
        """Add callback for order book updates."""
        self.book_update_callbacks.append(callback)
    
    def add_validation_error_callback(self, callback):
        """Add callback for validation errors."""
        self.validation_error_callbacks.append(callback)
    
    def start(self):
        """Start order book reconstruction."""
        if self.is_running:
            return
        
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._process_updates)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        logger.info("Order book reconstructor started")
    
    def stop(self):
        """Stop order book reconstruction."""
        self.is_running = False
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5)
        
        logger.info("Order book reconstructor stopped")
    
    def process_update(self, update: OrderBookUpdate):
        """Process order book update."""
        with self.buffer_lock:
            self.update_buffer.append(update)
    
    def _process_updates(self):
        """Main processing loop."""
        while self.is_running:
            try:
                # Get updates from buffer
                with self.buffer_lock:
                    updates = self.update_buffer.copy()
                    self.update_buffer.clear()
                
                if not updates:
                    time.sleep(0.001)  # 1ms sleep
                    continue
                
                # Correct latency
                corrected_updates = self.latency_corrector.synchronize_timestamps(updates)
                
                # Process each update
                for update in corrected_updates:
                    self._process_single_update(update)
                
            except Exception as e:
                logger.error(f"Error in update processing loop: {str(e)}")
                time.sleep(0.1)
    
    def _process_single_update(self, update: OrderBookUpdate):
        """Process a single update."""
        try:
            key = f"{update.exchange}:{update.symbol}"
            
            if update.update_type == UpdateType.SNAPSHOT:
                self._process_snapshot(update)
                self.stats['snapshots_processed'] += 1
            
            elif update.update_type == UpdateType.INCREMENTAL:
                self._process_incremental_update(update)
                self.stats['updates_processed'] += 1
            
            # Update latency measurements
            self.latency_corrector.update_latency(
                update.exchange, 
                update.timestamp, 
                update.received_at
            )
            
            # Validate order book after update
            if key in self.order_books and update.exchange in self.order_books[key]:
                book = self.order_books[key][update.exchange]
                issues = self.validator.validate_snapshot(book)
                
                if issues:
                    self.stats['validation_errors'] += 1
                    for callback in self.validation_error_callbacks:
                        callback(book, issues)
                else:
                    # Notify callbacks of successful update
                    for callback in self.book_update_callbacks:
                        callback(book, update)
        
        except Exception as e:
            logger.error(f"Error processing update: {str(e)}")
    
    def _process_snapshot(self, update: OrderBookUpdate):
        """Process snapshot update."""
        key = f"{update.exchange}:{update.symbol}"
        
        # Create new order book
        book = OrderBookSnapshot(
            symbol=update.symbol,
            exchange=update.exchange,
            timestamp=update.timestamp,
            sequence=update.sequence
        )
        
        # Add bid levels
        for price, size in update.bids:
            if size > 0:
                book.bids[price] = PriceLevel(price=price, size=size, timestamp=update.timestamp)
        
        # Add ask levels
        for price, size in update.asks:
            if size > 0:
                book.asks[price] = PriceLevel(price=price, size=size, timestamp=update.timestamp)
        
        # Store order book
        self.order_books[key][update.exchange] = book
        
        # Reset expected sequence
        if update.sequence is not None:
            self.expected_sequences[key] = update.sequence + 1
        
        logger.debug(f"Processed snapshot for {key}: {len(book.bids)} bids, {len(book.asks)} asks")
    
    def _process_incremental_update(self, update: OrderBookUpdate):
        """Process incremental update."""
        key = f"{update.exchange}:{update.symbol}"
        
        # Check if we have a base snapshot
        if key not in self.order_books or update.exchange not in self.order_books[key]:
            logger.warning(f"No snapshot for {key}, skipping incremental update")
            return
        
        book = self.order_books[key][update.exchange]
        
        # Check sequence number
        if update.sequence is not None:
            expected_seq = self.expected_sequences.get(key, 0)
            if expected_seq > 0 and update.sequence != expected_seq:
                logger.warning(f"Sequence gap for {key}: expected {expected_seq}, got {update.sequence}")
                self.stats['sequence_gaps'] += 1
                # Could request snapshot here
            
            self.expected_sequences[key] = update.sequence + 1
        
        # Apply bid updates
        for price, size in update.bids:
            if size > 0:
                book.bids[price] = PriceLevel(price=price, size=size, timestamp=update.timestamp)
            elif price in book.bids:
                del book.bids[price]
        
        # Apply ask updates
        for price, size in update.asks:
            if size > 0:
                book.asks[price] = PriceLevel(price=price, size=size, timestamp=update.timestamp)
            elif price in book.asks:
                del book.asks[price]
        
        # Update timestamp
        book.last_update = update.timestamp
        
        logger.debug(f"Applied incremental update to {key}")
    
    def get_order_book(self, symbol: str, exchange: str) -> Optional[OrderBookSnapshot]:
        """Get current order book for symbol and exchange."""
        key = f"{exchange}:{symbol}"
        return self.order_books.get(key, {}).get(exchange)
    
    def get_aggregated_book(self, symbol: str, exchanges: List[str] = None) -> Optional[OrderBookSnapshot]:
        """Get aggregated order book across exchanges."""
        if exchanges is None:
            # Get all available exchanges for symbol
            exchanges = []
            for key, books in self.order_books.items():
                if key.endswith(f":{symbol}"):
                    exchanges.extend(books.keys())
            exchanges = list(set(exchanges))
        
        if not exchanges:
            return None
        
        # Get all order books
        books = []
        for exchange in exchanges:
            book = self.get_order_book(symbol, exchange)
            if book:
                books.append(book)
        
        if not books:
            return None
        
        # Create aggregated book
        aggregated = OrderBookSnapshot(
            symbol=symbol,
            exchange="aggregated",
            timestamp=max(book.timestamp for book in books)
        )
        
        # Combine all bids and asks
        all_bids = {}
        all_asks = {}
        
        for book in books:
            for price, level in book.bids.items():
                if price not in all_bids:
                    all_bids[price] = PriceLevel(price=price, size=0, timestamp=level.timestamp)
                all_bids[price].size += level.size
                all_bids[price].count += level.count
            
            for price, level in book.asks.items():
                if price not in all_asks:
                    all_asks[price] = PriceLevel(price=price, size=0, timestamp=level.timestamp)
                all_asks[price].size += level.size
                all_asks[price].count += level.count
        
        aggregated.bids = OrderedDict(sorted(all_bids.items(), reverse=True))
        aggregated.asks = OrderedDict(sorted(all_asks.items()))
        
        return aggregated
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get reconstruction statistics."""
        stats = self.stats.copy()
        
        # Add current book counts
        stats['active_books'] = sum(len(books) for books in self.order_books.values())
        stats['unique_symbols'] = len(self.order_books)
        
        # Add validation stats
        stats['validation_stats'] = self.validator.get_validation_stats()
        
        # Add buffer size
        with self.buffer_lock:
            stats['buffer_size'] = len(self.update_buffer)
        
        return stats
    
    def replay_historical_data(self, updates: List[OrderBookUpdate], 
                              callback: Optional[callable] = None) -> Dict[str, OrderBookSnapshot]:
        """Replay historical order book updates."""
        # Save current state
        original_books = self.order_books.copy()
        original_sequences = self.expected_sequences.copy()
        
        # Reset state for replay
        self.order_books.clear()
        self.expected_sequences.clear()
        
        try:
            # Sort updates by timestamp
            sorted_updates = sorted(updates, key=lambda x: x.timestamp)
            
            # Process updates
            for update in sorted_updates:
                self._process_single_update(update)
                
                if callback:
                    callback(update, self.order_books)
            
            # Return final state
            return self.order_books.copy()
        
        finally:
            # Restore original state
            self.order_books = original_books
            self.expected_sequences = original_sequences

def example_book_callback(book: OrderBookSnapshot, update: OrderBookUpdate):
    """Example callback for order book updates."""
    spread = book.get_spread()
    mid_price = book.get_mid_price()
    
    logger.info(f"{book.exchange}:{book.symbol} - Mid: {mid_price:.4f}, Spread: {spread:.4f}")

def example_validation_callback(book: OrderBookSnapshot, issues: List[str]):
    """Example callback for validation errors."""
    logger.warning(f"Validation issues for {book.exchange}:{book.symbol}: {issues}")

def main():
    """Example usage of OrderBookReconstructor."""
    # Create reconstructor
    reconstructor = OrderBookReconstructor({
        'max_spread_pct': 5.0,
        'max_level_count': 500
    })
    
    # Add callbacks
    reconstructor.add_book_update_callback(example_book_callback)
    reconstructor.add_validation_error_callback(example_validation_callback)
    
    # Start reconstructor
    reconstructor.start()
    
    try:
        # Simulate some updates
        # Snapshot
        snapshot_update = OrderBookUpdate(
            symbol="BTCUSD",
            exchange="test",
            timestamp=datetime.now(),
            update_type=UpdateType.SNAPSHOT,
            sequence=1,
            bids=[(50000.0, 1.5), (49999.0, 2.0), (49998.0, 0.5)],
            asks=[(50001.0, 1.2), (50002.0, 1.8), (50003.0, 2.5)]
        )
        
        reconstructor.process_update(snapshot_update)
        
        # Incremental update
        time.sleep(0.1)
        incremental_update = OrderBookUpdate(
            symbol="BTCUSD",
            exchange="test",
            timestamp=datetime.now(),
            update_type=UpdateType.INCREMENTAL,
            sequence=2,
            bids=[(50000.0, 2.0)],  # Update existing level
            asks=[(50001.0, 0.0)]   # Remove level
        )
        
        reconstructor.process_update(incremental_update)
        
        # Wait for processing
        time.sleep(1.0)
        
        # Get order book
        book = reconstructor.get_order_book("BTCUSD", "test")
        if book:
            depth = book.get_depth(5)
            print(f"Order book depth: {json.dumps(depth, indent=2)}")
        
        # Print statistics
        stats = reconstructor.get_statistics()
        print(f"Statistics: {json.dumps(stats, indent=2, default=str)}")
        
    finally:
        reconstructor.stop()

if __name__ == "__main__":
    main()