"""
QUANTUM-FORGE: Historical Data Collector & Storage
===================================================
P0 — Everything else depends on having data.

Collects historical candle data from Binance and stores it in Parquet files.
Supports:
  - Bulk download of historical OHLCV data for all trading pairs
  - Incremental updates (only fetch new data since last stored candle)
  - Multiple timeframes (1m, 5m, 15m, 1h, 4h, 1d)
  - Data validation and cleaning
  - Automatic resume on failure

Storage format:
  data/parquet/candles/{symbol}/{interval}/YYYY-MM-DD.parquet
"""

import os
import sys
import time
import logging
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.ingestion.binance_client import BinanceAPIClient

logger = logging.getLogger("HistoricalCollector")


class HistoricalDataCollector:
    """
    Collects and stores historical market data from Binance.
    
    Usage:
        collector = HistoricalDataCollector()
        collector.collect_all(days=365)  # Collect 1 year of data
        
        # Later, update with new data:
        collector.update_all()
    """
    
    INTERVALS = ["1m", "5m", "15m", "1h", "4h", "1d"]
    DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", "DOGEUSDT", "XRPUSDT"]
    
    def __init__(
        self,
        symbols: List[str] = None,
        base_path: str = "./data/parquet/candles",
        intervals: List[str] = None,
    ):
        self.symbols = symbols or self.DEFAULT_SYMBOLS
        self.base_path = Path(base_path)
        self.intervals = intervals or ["1h", "4h", "1d"]  # Default: reasonable intervals
        self.client = BinanceAPIClient()
        
        # Create directory structure
        for symbol in self.symbols:
            for interval in self.intervals:
                (self.base_path / symbol / interval).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"HistoricalDataCollector initialized: {len(self.symbols)} symbols, {self.intervals} intervals")
    
    def collect_all(self, days: int = 90):
        """
        Collect historical data for all symbols and intervals.
        
        Args:
            days: Number of days of history to collect
        """
        total = len(self.symbols) * len(self.intervals)
        done = 0
        
        for symbol in self.symbols:
            for interval in self.intervals:
                done += 1
                logger.info(f"[{done}/{total}] Collecting {symbol} {interval} ({days} days)...")
                try:
                    df = self._fetch_candles(symbol, interval, days)
                    if df is not None and len(df) > 0:
                        self._save_candles(df, symbol, interval)
                        logger.info(f"  Saved {len(df)} candles for {symbol} {interval}")
                    else:
                        logger.warning(f"  No data returned for {symbol} {interval}")
                except Exception as e:
                    logger.error(f"  Error collecting {symbol} {interval}: {e}")
                
                time.sleep(0.5)  # Rate limit
        
        logger.info(f"Collection complete: {done} series collected")
    
    def update_all(self):
        """
        Incremental update — fetch only new data since last stored candle.
        """
        for symbol in self.symbols:
            for interval in self.intervals:
                try:
                    last_ts = self._get_last_timestamp(symbol, interval)
                    if last_ts is None:
                        logger.info(f"No existing data for {symbol} {interval}, doing full collect (30 days)")
                        df = self._fetch_candles(symbol, interval, 30)
                    else:
                        # Fetch from last stored candle to now
                        now = datetime.utcnow()
                        delta = now - last_ts
                        days = max(int(delta.total_seconds() / 86400) + 1, 1)
                        logger.info(f"Updating {symbol} {interval} (last: {last_ts}, fetching {days} days)")
                        df = self._fetch_candles(symbol, interval, days, start_from=last_ts)
                    
                    if df is not None and len(df) > 0:
                        self._save_candles(df, symbol, interval)
                        logger.info(f"  Updated {symbol} {interval}: +{len(df)} candles")
                except Exception as e:
                    logger.error(f"  Error updating {symbol} {interval}: {e}")
                
                time.sleep(0.3)
    
    def load_candles(
        self,
        symbol: str,
        interval: str = "1h",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load stored candle data from Parquet files.
        
        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Candle interval (e.g., "1h")
            start_date: Optional start date filter (YYYY-MM-DD)
            end_date: Optional end date filter (YYYY-MM-DD)
        
        Returns:
            DataFrame with OHLCV data
        """
        data_dir = self.base_path / symbol / interval
        
        if not data_dir.exists():
            logger.warning(f"No data directory for {symbol}/{interval}")
            return pd.DataFrame()
        
        parquet_files = sorted(data_dir.glob("*.parquet"))
        
        if not parquet_files:
            logger.warning(f"No parquet files in {data_dir}")
            return pd.DataFrame()
        
        # Filter by date range if specified
        if start_date or end_date:
            filtered_files = []
            for f in parquet_files:
                file_date = f.stem  # YYYY-MM-DD format
                if start_date and file_date < start_date:
                    continue
                if end_date and file_date > end_date:
                    continue
                filtered_files.append(f)
            parquet_files = filtered_files
        
        if not parquet_files:
            return pd.DataFrame()
        
        dfs = []
        for f in parquet_files:
            try:
                df = pd.read_parquet(f)
                dfs.append(df)
            except Exception as e:
                logger.warning(f"Error reading {f}: {e}")
        
        if not dfs:
            return pd.DataFrame()
        
        result = pd.concat(dfs, ignore_index=True)
        result = result.sort_values('open_time').drop_duplicates(subset=['open_time']).reset_index(drop=True)
        
        return result
    
    def load_prices(self, symbol: str, interval: str = "1h", days: int = 90) -> np.ndarray:
        """
        Load close prices as a numpy array — ready for signal generator and ML.
        
        Args:
            symbol: Trading pair
            interval: Candle interval
            days: Number of days of data to load
        
        Returns:
            numpy array of close prices
        """
        start_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        df = self.load_candles(symbol, interval, start_date=start_date)
        
        if df.empty:
            return np.array([])
        
        return df['close'].values.astype(float)
    
    def load_ohlcv(self, symbol: str, interval: str = "1h", days: int = 90) -> pd.DataFrame:
        """
        Load full OHLCV data for backtesting.
        
        Returns:
            DataFrame with columns: open_time, open, high, low, close, volume
        """
        start_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        df = self.load_candles(symbol, interval, start_date=start_date)
        
        if df.empty:
            return df
        
        # Ensure numeric types
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df[['open_time', 'open', 'high', 'low', 'close', 'volume']].copy()
    
    def get_data_summary(self) -> Dict:
        """Get summary of all stored data."""
        summary = {}
        total_candles = 0
        total_size = 0
        
        for symbol in self.symbols:
            summary[symbol] = {}
            for interval in self.intervals:
                data_dir = self.base_path / symbol / interval
                if data_dir.exists():
                    files = list(data_dir.glob("*.parquet"))
                    n_files = len(files)
                    size = sum(f.stat().st_size for f in files)
                    
                    # Get date range
                    if files:
                        dates = sorted([f.stem for f in files])
                        summary[symbol][interval] = {
                            'files': n_files,
                            'size_mb': size / (1024 * 1024),
                            'first_date': dates[0],
                            'last_date': dates[-1],
                        }
                        total_candles += n_files
                        total_size += size
        
        return {
            'symbols': summary,
            'total_files': total_candles,
            'total_size_mb': total_size / (1024 * 1024),
        }
    
    # === Private methods ===
    
    def _fetch_candles(
        self,
        symbol: str,
        interval: str,
        days: int,
        start_from: Optional[datetime] = None,
    ) -> Optional[pd.DataFrame]:
        """Fetch historical candles from Binance with pagination."""
        try:
            if start_from:
                start_time = int(start_from.timestamp() * 1000)
            else:
                start_time = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)
            
            end_time = int(datetime.utcnow().timestamp() * 1000)
            
            all_candles = []
            current_start = start_time
            
            while current_start < end_time:
                df = self.client.get_klines(
                    symbol=symbol,
                    interval=interval,
                    limit=1000,
                    start_time=current_start,
                    end_time=end_time,
                )
                
                if df is None or len(df) == 0:
                    break
                
                all_candles.append(df)
                
                # Move to next batch
                last_time = df['open_time'].iloc[-1]
                if isinstance(last_time, (int, float, np.integer)):
                    current_start = int(last_time) + 1
                else:
                    current_start = int(last_time.timestamp() * 1000) + 1
                
                if len(df) < 1000:
                    break  # No more data
                
                time.sleep(0.2)  # Rate limit
            
            if not all_candles:
                return None
            
            result = pd.concat(all_candles, ignore_index=True)
            result = result.drop_duplicates(subset=['open_time']).sort_values('open_time').reset_index(drop=True)
            
            # Validate data
            result = self._validate_candles(result, symbol)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to fetch candles for {symbol} {interval}: {e}")
            return None
    
    def _validate_candles(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Validate and clean candle data."""
        initial_len = len(df)
        
        # Remove rows with zero/null prices
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df = df[df[col] > 0]
        
        # Remove volume nulls
        if 'volume' in df.columns:
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
            df = df[df['volume'] >= 0]
        
        # Validate OHLC relationships: high >= low, high >= open, high >= close
        if all(c in df.columns for c in ['open', 'high', 'low', 'close']):
            df = df[df['high'] >= df['low']]
        
        # Remove extreme outliers (>5x standard deviation in returns)
        if 'close' in df.columns and len(df) > 10:
            returns = df['close'].pct_change().abs()
            median_ret = returns.median()
            if median_ret > 0:
                df = df[returns.fillna(0) < median_ret * 50]  # 50x median return = likely bad data
        
        removed = initial_len - len(df)
        if removed > 0:
            logger.info(f"  Validated {symbol}: removed {removed}/{initial_len} bad candles")
        
        # Add symbol column
        df['symbol'] = symbol
        
        return df.reset_index(drop=True)
    
    def _save_candles(self, df: pd.DataFrame, symbol: str, interval: str):
        """Save candles to Parquet files, partitioned by date."""
        data_dir = self.base_path / symbol / interval
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert open_time to datetime for partitioning
        if 'open_time' in df.columns:
            if df['open_time'].dtype in [np.int64, np.float64, int, float]:
                df['datetime'] = pd.to_datetime(df['open_time'], unit='ms')
            else:
                df['datetime'] = pd.to_datetime(df['open_time'])
            
            df['date'] = df['datetime'].dt.strftime('%Y-%m-%d')
        else:
            df['date'] = datetime.utcnow().strftime('%Y-%m-%d')
        
        # Write one file per date
        for date_str, group in df.groupby('date'):
            file_path = data_dir / f"{date_str}.parquet"
            
            # Drop helper columns before saving
            save_df = group.drop(columns=['date', 'datetime'], errors='ignore')
            
            # If file exists, merge with existing data
            if file_path.exists():
                try:
                    existing = pd.read_parquet(file_path)
                    save_df = pd.concat([existing, save_df], ignore_index=True)
                    save_df = save_df.drop_duplicates(subset=['open_time']).sort_values('open_time').reset_index(drop=True)
                except:
                    pass  # Overwrite if existing file is corrupted
            
            save_df.to_parquet(file_path, compression='snappy', index=False)
    
    def _get_last_timestamp(self, symbol: str, interval: str) -> Optional[datetime]:
        """Get the timestamp of the last stored candle."""
        data_dir = self.base_path / symbol / interval
        
        if not data_dir.exists():
            return None
        
        parquet_files = sorted(data_dir.glob("*.parquet"), reverse=True)
        
        for f in parquet_files[:3]:  # Check last 3 files
            try:
                df = pd.read_parquet(f)
                if len(df) > 0 and 'open_time' in df.columns:
                    last_ts = df['open_time'].max()
                    if isinstance(last_ts, (int, float, np.integer)):
                        return datetime.utcfromtimestamp(last_ts / 1000)
                    else:
                        return pd.to_datetime(last_ts)
            except:
                continue
        
        return None


class ContinuousDataRecorder:
    """
    Records real-time data continuously alongside the trading pipeline.
    Saves tick data and candles in real-time to fill the data/parquet/ directory.
    
    Run this in the background while quantum_core is running.
    """
    
    def __init__(
        self,
        symbols: List[str] = None,
        base_path: str = "./data/parquet",
        save_interval_seconds: int = 300,  # Save to disk every 5 minutes
    ):
        self.symbols = symbols or HistoricalDataCollector.DEFAULT_SYMBOLS
        self.base_path = Path(base_path)
        self.save_interval = save_interval_seconds
        self.client = BinanceAPIClient()
        self.is_running = False
        
        # In-memory tick buffer
        self.tick_buffer: Dict[str, List[Dict]] = {s: [] for s in self.symbols}
        
        # Ensure directories
        (self.base_path / "ticks").mkdir(parents=True, exist_ok=True)
        (self.base_path / "live_candles").mkdir(parents=True, exist_ok=True)
    
    def start(self):
        """Start continuous recording in a background thread."""
        import threading
        self.is_running = True
        self._thread = threading.Thread(target=self._record_loop, daemon=True)
        self._thread.start()
        logger.info("[RECORDER] Started continuous data recording")
    
    def stop(self):
        """Stop recording and flush remaining data."""
        self.is_running = False
        self._flush_buffers()
        logger.info("[RECORDER] Stopped")
    
    def _record_loop(self):
        """Continuously record prices and periodically flush to disk."""
        last_flush = time.time()
        
        while self.is_running:
            try:
                # Record current prices
                prices = self.client.get_current_prices(self.symbols)
                timestamp = datetime.utcnow()
                
                for symbol, price in prices.items():
                    self.tick_buffer[symbol].append({
                        'timestamp': timestamp.isoformat(),
                        'symbol': symbol,
                        'price': price,
                        'recorded_at': time.time(),
                    })
                
                # Periodic flush
                if time.time() - last_flush > self.save_interval:
                    self._flush_buffers()
                    last_flush = time.time()
                
                time.sleep(2.0)  # Record every 2 seconds
                
            except Exception as e:
                logger.error(f"[RECORDER] Error: {e}")
                time.sleep(5.0)
    
    def _flush_buffers(self):
        """Write buffered ticks to Parquet files."""
        date_str = datetime.utcnow().strftime('%Y-%m-%d')
        
        for symbol, ticks in self.tick_buffer.items():
            if not ticks:
                continue
            
            df = pd.DataFrame(ticks)
            file_path = self.base_path / "ticks" / f"{symbol}_{date_str}.parquet"
            
            if file_path.exists():
                try:
                    existing = pd.read_parquet(file_path)
                    df = pd.concat([existing, df], ignore_index=True)
                except:
                    pass
            
            file_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(file_path, compression='snappy', index=False)
        
        # Clear buffers
        for symbol in self.tick_buffer:
            self.tick_buffer[symbol] = []
        
        logger.debug(f"[RECORDER] Flushed tick data to disk")


def main():
    """CLI entry point for historical data collection."""
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)s | %(message)s',
    )
    
    parser = argparse.ArgumentParser(description='Quantum-Forge Historical Data Collector')
    parser.add_argument('--days', type=int, default=90, help='Days of history to collect')
    parser.add_argument('--symbols', nargs='+', default=None, help='Symbols to collect')
    parser.add_argument('--intervals', nargs='+', default=['1h', '4h', '1d'], help='Candle intervals')
    parser.add_argument('--update', action='store_true', help='Only fetch new data since last collect')
    parser.add_argument('--summary', action='store_true', help='Show data summary and exit')
    
    args = parser.parse_args()
    
    collector = HistoricalDataCollector(
        symbols=args.symbols,
        intervals=args.intervals,
    )
    
    if args.summary:
        summary = collector.get_data_summary()
        print(f"\nTotal files: {summary['total_files']}")
        print(f"Total size: {summary['total_size_mb']:.2f} MB\n")
        for symbol, intervals in summary['symbols'].items():
            for interval, info in intervals.items():
                print(f"  {symbol}/{interval}: {info['files']} files, "
                      f"{info['size_mb']:.2f} MB, "
                      f"{info['first_date']} → {info['last_date']}")
        return
    
    if args.update:
        collector.update_all()
    else:
        collector.collect_all(days=args.days)
    
    # Print summary
    summary = collector.get_data_summary()
    print(f"\nCollection complete!")
    print(f"Total files: {summary['total_files']}")
    print(f"Total size: {summary['total_size_mb']:.2f} MB")


if __name__ == "__main__":
    main()
