#!/usr/bin/env python3
"""
Quantum-Forge: Train ML Models
================================
Runs walk-forward training for all ML models using historical data.

Usage:
    python scripts/train_models.py
    python scripts/train_models.py --symbols BTCUSDT ETHUSDT --days 90 --epochs 50
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
    parser = argparse.ArgumentParser(description="Train Quantum-Forge ML models")
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--interval", type=str, default="1h")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    from intelligence.training_pipeline import TrainingPipeline

    pipeline = TrainingPipeline()
    results = pipeline.run(
        symbols=args.symbols,
        days=args.days,
        interval=args.interval,
        n_folds=args.folds,
        epochs=args.epochs,
    )

    # Summary
    ok = sum(1 for r in results.values() if "error" not in r)
    fail = sum(1 for r in results.values() if "error" in r)
    print(f"\nTraining complete — {ok} succeeded, {fail} failed.")


if __name__ == "__main__":
    main()
