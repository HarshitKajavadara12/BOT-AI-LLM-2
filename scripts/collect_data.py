#!/usr/bin/env python3
"""
Quantum-Forge: Collect Historical Data
========================================
Downloads OHLCV candles from Binance and stores them as Parquet files.

Usage:
    python scripts/collect_data.py
    python scripts/collect_data.py --symbols BTCUSDT ETHUSDT --days 180 --interval 1h
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-18s | %(levelname)-7s | %(message)s",
)

def main():
    parser = argparse.ArgumentParser(description="Collect Binance historical data")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"])
    parser.add_argument("--days", type=int, default=90, help="Days of history to download")
    parser.add_argument("--interval", type=str, default="1h", help="Candle interval (1m,5m,15m,1h,4h,1d)")
    args = parser.parse_args()

    from data.historical_collector import HistoricalDataCollector

    collector = HistoricalDataCollector(symbols=args.symbols, intervals=[args.interval])
    collector.collect_all(days=args.days)
    print(f"\nDone — {len(args.symbols)} symbols × {args.days} days collected.")


if __name__ == "__main__":
    main()
