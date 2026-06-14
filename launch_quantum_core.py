"""
QUANTUM-FORGE: Direct Quantum Core Launcher
=============================================
Run this to start the REAL pipeline directly without the full system overhead.
Lightweight, fast startup, focused on the core trading engine.

Usage:
    python launch_quantum_core.py
    python launch_quantum_core.py --symbols BTCUSDT,ETHUSDT --capital 50000
"""

import sys
import os
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser(description="Quantum-Forge Core Pipeline")
    parser.add_argument(
        '--symbols', 
        type=str, 
        default='BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT',
        help='Comma-separated list of trading symbols'
    )
    parser.add_argument(
        '--capital', 
        type=float, 
        default=100000.0,
        help='Initial capital in USD'
    )
    parser.add_argument(
        '--no-ml', 
        action='store_true',
        help='Disable ML ensemble (math-only mode)'
    )
    parser.add_argument(
        '--llm', 
        action='store_true',
        help='Enable LLM integration (read-only)'
    )
    parser.add_argument(
        '--threshold', 
        type=float, 
        default=0.25,
        help='Signal strength threshold (0.0-1.0)'
    )
    
    args = parser.parse_args()
    
    symbols = [s.strip() for s in args.symbols.split(',')]
    
    from core.quantum_core import QuantumCoreOrchestrator
    
    orchestrator = QuantumCoreOrchestrator(
        symbols=symbols,
        initial_capital=args.capital,
        enable_ml=not args.no_ml,
        enable_llm=args.llm,
        signal_threshold=args.threshold,
    )
    
    orchestrator.start()
    
    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        orchestrator.stop()


if __name__ == "__main__":
    main()
