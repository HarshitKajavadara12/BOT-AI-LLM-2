#!/usr/bin/env python3
"""
Quantum-Forge: Run Backtest
=============================
Runs the real backtester over historical data.

Usage:
    python scripts/run_backtest.py
    python scripts/run_backtest.py --symbols BTCUSDT ETHUSDT --days 60
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
    parser = argparse.ArgumentParser(description="Run Quantum-Forge backtester")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "BNBUSDT"])
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--interval", type=str, default="1h")
    parser.add_argument("--capital", type=float, default=100000.0)
    args = parser.parse_args()

    from core.real_backtester import RealBacktester

    bt = RealBacktester(
        initial_capital=args.capital,
        fee_rate=0.001,
        enable_ml=False,
    )

    results = {}
    for symbol in args.symbols:
        print(f"\n{'='*60}")
        print(f"  Backtesting {symbol} — {args.days} days, {args.interval}")
        print(f"{'='*60}")
        result = bt.run(symbol=symbol, interval=args.interval, days=args.days)
        results[symbol] = result
        print(f"  Return: {result.total_return_pct:+.2f}%")
        print(f"  Sharpe: {result.sharpe_ratio:.3f}")
        print(f"  MaxDD:  {result.max_drawdown_pct:.2f}%")
        print(f"  Trades: {result.total_trades}")

    print(f"\n{'='*60}")
    print("BACKTEST SUMMARY")
    print(f"{'='*60}")
    for sym, r in results.items():
        print(f"  {sym:12s} | Return: {r.total_return_pct:+.2f}% | Sharpe: {r.sharpe_ratio:.3f} | Trades: {r.total_trades}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
