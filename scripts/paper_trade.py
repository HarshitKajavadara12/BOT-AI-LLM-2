#!/usr/bin/env python3
"""
Quantum-Forge: Paper Trading
===============================
Runs the full pipeline in PAPER mode against live Binance prices.
No real orders are placed.

Usage:
    python scripts/paper_trade.py
    python scripts/paper_trade.py --capital 50000 --symbols BTCUSDT ETHUSDT
"""
import sys, os, signal, time
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
    parser = argparse.ArgumentParser(description="Run Quantum-Forge in paper-trading mode")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"])
    parser.add_argument("--capital", type=float, default=100000.0)
    parser.add_argument("--ml", action="store_true", default=False, help="Enable ML ensemble")
    parser.add_argument("--threshold", type=float, default=0.25, help="Signal threshold")
    args = parser.parse_args()

    from core.quantum_core import QuantumCoreOrchestrator

    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║     QUANTUM-FORGE — PAPER TRADING MODE                  ║
    ║     No real orders. Press Ctrl+C to stop.               ║
    ╚══════════════════════════════════════════════════════════╝
    """)

    orchestrator = QuantumCoreOrchestrator(
        symbols=args.symbols,
        initial_capital=args.capital,
        enable_ml=args.ml,
        enable_llm=False,
        signal_threshold=args.threshold,
    )

    def _shutdown(signum, frame):
        print("\nShutting down gracefully...")
        orchestrator.stop()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    orchestrator.start()

    try:
        while orchestrator.is_running:
            time.sleep(1)
    except KeyboardInterrupt:
        orchestrator.stop()


if __name__ == "__main__":
    main()
