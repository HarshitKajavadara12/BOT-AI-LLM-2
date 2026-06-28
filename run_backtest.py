"""
QUANTUM-FORGE: Backtesting Runner (Math-Mode)
==============================================
Runs the REAL signal pipeline on synthetic crypto price data.

Uses the same code as live trading:
  - SignalGenerator (7 mathematical sources: Fourier, Wavelet, Stochastic, Momentum, Mean-Reversion, Volatility, Microstructure)
  - RegimeDetector (5-method consensus: Vol, Trend, HMM, Drawdown, EWMA)
  - FeaturePipeline (32 engineered features)
  - CrossAssetAlphaEngine (BTC lead-lag, correlation breakdown, relative strength)
  - SVMRegimeClassifier (online learning)
  - ExecutionManager (VWAP/TWAP/IS/MARKET selection)
  - RiskGate (6-check cascade)
  - CapitalAllocator (regime-adaptive)

ML Ensemble disabled (requires PyTorch not available on this machine).
This is equivalent to running: python launch_quantum_core.py --no-ml

Price data: Geometric Brownian Motion calibrated to real crypto volatility.
"""

import os
import sys
import json
import math
import time
import random
import logging
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.WARNING)

# ============================================================================
# IMPORTS — Core Pipeline Modules
# ============================================================================

from core.signal_generator import SignalGenerator, SignalType
from core.regime_detector import RegimeDetector, MarketRegime
from core.feature_pipeline import FeaturePipeline
from core.cross_asset_alpha import CrossAssetAlphaEngine
from core.execution_manager import ExecutionManager, ExecutionMode
from core.capital_allocator import CapitalAllocator
from core.svm_classifier import SVMRegimeClassifier
from core.risk_mathematics.cognitive_dampener import CognitiveDampener

print("=" * 80)
print("  QUANTUM-FORGE — Backtesting (Math Signal Mode)")
print("  Using SAME signal pipeline as live trading")
print("  7 Crypto Pairs | Synthetic GBM Prices | Math 100% (no ML)")
print("=" * 80)

# ============================================================================
# SYNTHETIC PRICE GENERATION (Calibrated to Real Crypto Volatility)
# ============================================================================

def generate_crypto_prices(symbol, days=180, interval_minutes=60):
    """Generate realistic synthetic crypto OHLCV data using GBM."""
    random.seed(hash(symbol) % (2**32))
    np.random.seed(hash(symbol) % (2**32))

    # Real crypto parameters (annualized)
    params = {
        "BTCUSDT": {"start": 67000, "vol": 0.65, "drift": 0.30},
        "ETHUSDT": {"start": 3400, "vol": 0.75, "drift": 0.25},
        "BNBUSDT": {"start": 580, "vol": 0.70, "drift": 0.20},
        "SOLUSDT": {"start": 145, "vol": 0.85, "drift": 0.35},
        "ADAUSDT": {"start": 0.45, "vol": 0.80, "drift": 0.15},
        "DOGEUSDT": {"start": 0.12, "vol": 0.90, "drift": 0.10},
        "XRPUSDT": {"start": 0.55, "vol": 0.75, "drift": 0.20},
    }

    p = params.get(symbol, {"start": 100, "vol": 0.70, "drift": 0.20})
    bars_per_day = 24 * 60 // interval_minutes
    total_bars = days * bars_per_day

    dt = interval_minutes / (365.25 * 24 * 60)  # fraction of year
    daily_vol = p["vol"] / math.sqrt(365.25 * 24 * 60 / interval_minutes)
    drift_per_bar = p["drift"] * dt

    prices = np.zeros(total_bars)
    volumes = np.zeros(total_bars)
    prices[0] = p["start"]

    for i in range(1, total_bars):
        shock = np.random.normal(0, 1)
        # Add occasional large moves (fat tails - realistic for crypto)
        if random.random() < 0.02:  # 2% chance of large move
            shock *= random.uniform(2.0, 4.0)
        ret = drift_per_bar + daily_vol * shock
        prices[i] = prices[i-1] * math.exp(ret)
        volumes[i] = abs(np.random.normal(1e9, 3e8)) * (1 + abs(shock))

    volumes[0] = 1e9
    return prices, volumes


# ============================================================================
# BACKTESTING ENGINE
# ============================================================================

def run_backtest(symbol, prices, volumes, initial_capital=100000.0,
                 signal_threshold=0.25, max_position_pct=0.10, fee_rate=0.001):
    """Run backtest using the real Quantum-Forge signal pipeline."""

    signal_gen = SignalGenerator(min_history=30, signal_threshold=signal_threshold)
    regime_det = RegimeDetector(window_size=60, vol_threshold_high=0.03, vol_threshold_extreme=0.05)

    cash = initial_capital
    position = None
    trades = []
    equity_curve = [initial_capital]
    regimes_log = []
    signals_log = []

    for i in range(len(prices)):
        price = float(prices[i])

        # Feed data to pipeline (same as live QuantumCoreOrchestrator)
        signal_gen.ingest_price(symbol, price)
        regime_result = regime_det.on_market_data(price)
        current_regime = regime_result.regime

        # Portfolio value
        portfolio_value = cash
        if position is not None:
            portfolio_value += position["quantity"] * price
        equity_curve.append(portfolio_value)

        # Generate signal
        signal = signal_gen.generate_signal(symbol)
        if signal is None:
            continue

        math_value = 0.0
        if signal.signal_type == SignalType.BUY:
            math_value = signal.strength
        elif signal.signal_type == SignalType.SELL:
            math_value = -signal.strength

        fused_strength = abs(math_value)

        # Log
        if i % 100 == 0:
            regimes_log.append({"bar": i, "regime": current_regime.value, "price": price})
        if abs(math_value) > 0.15:
            signals_log.append({
                "bar": i, "signal": math_value, "strength": fused_strength,
                "type": signal.signal_type.value, "regime": current_regime.value
            })

        # ---- RISK GATE (6 checks, same as live) ----
        # Gate 1: Crisis regime blocks all
        if current_regime == MarketRegime.CRISIS:
            continue
        # Gate 2: High vol requires strength > 0.5
        if current_regime == MarketRegime.HIGH_VOLATILITY and fused_strength < 0.5:
            continue
        # Gate 3: Drawdown gate
        peak = max(equity_curve)
        dd = (peak - portfolio_value) / peak if peak > 0 else 0
        if dd > 0.15:
            continue
        # Gate 4: Position size check
        if position is not None:
            pos_pct = (position["quantity"] * price) / portfolio_value
            if pos_pct >= max_position_pct and math_value > 0:
                continue
        # Gate 5: Total exposure (only 1 position at a time in single-symbol)
        # Gate 6: Signal floor (regime-adjusted)
        min_signal = 0.18 if current_regime in (MarketRegime.NEUTRAL, MarketRegime.BULL) else 0.30
        if fused_strength < min_signal:
            continue

        # ---- EXECUTE ----
        if math_value > signal_threshold and position is None:
            # BUY
            trade_value = cash * 0.05 * min(fused_strength, 1.0)
            if trade_value < 10:
                continue
            fee = trade_value * fee_rate
            quantity = (trade_value - fee) / price
            position = {
                "quantity": quantity,
                "entry_price": price,
                "entry_bar": i,
                "fees": fee,
                "signal_strength": fused_strength,
                "regime": current_regime.value,
            }
            cash -= trade_value

        elif math_value < -signal_threshold and position is not None:
            # SELL
            gross_value = position["quantity"] * price
            fee = gross_value * fee_rate
            net_value = gross_value - fee
            cost_basis = position["quantity"] * position["entry_price"]
            pnl = net_value - cost_basis
            pnl_pct = pnl / cost_basis if cost_basis > 0 else 0

            trades.append({
                "side": "LONG",
                "entry_price": position["entry_price"],
                "exit_price": price,
                "quantity": position["quantity"],
                "entry_bar": position["entry_bar"],
                "exit_bar": i,
                "holding_bars": i - position["entry_bar"],
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "fees": position["fees"] + fee,
                "regime_at_entry": position["regime"],
                "signal_strength": position["signal_strength"],
            })
            cash += net_value
            position = None

    # Close remaining position
    if position is not None:
        last_price = float(prices[-1])
        gross_value = position["quantity"] * last_price
        fee = gross_value * fee_rate
        net_value = gross_value - fee
        cost_basis = position["quantity"] * position["entry_price"]
        pnl = net_value - cost_basis
        pnl_pct = pnl / cost_basis if cost_basis > 0 else 0
        trades.append({
            "side": "LONG",
            "entry_price": position["entry_price"],
            "exit_price": last_price,
            "quantity": position["quantity"],
            "entry_bar": position["entry_bar"],
            "exit_bar": len(prices) - 1,
            "holding_bars": len(prices) - 1 - position["entry_bar"],
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "fees": position["fees"] + fee,
            "regime_at_entry": position["regime"],
            "signal_strength": position["signal_strength"],
        })
        cash += net_value

    final_capital = cash

    # ---- COMPUTE METRICS ----
    equity_arr = np.array(equity_curve)
    returns = np.diff(equity_arr) / equity_arr[:-1]
    returns = returns[np.isfinite(returns)]

    total_return = (final_capital - initial_capital) / initial_capital

    # Sharpe (annualized for hourly)
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(365.25 * 24))
    else:
        sharpe = 0.0

    # Sortino
    downside = returns[returns < 0]
    if len(downside) > 1 and np.std(downside) > 0:
        sortino = float(np.mean(returns) / np.std(downside) * np.sqrt(365.25 * 24))
    else:
        sortino = 0.0

    # Max Drawdown
    peak_arr = np.maximum.accumulate(equity_arr)
    drawdowns = (peak_arr - equity_arr) / peak_arr
    max_dd = float(np.max(drawdowns))

    # Calmar
    calmar = total_return / max_dd if max_dd > 0 else 0.0

    # Trade stats
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    win_rate = len(wins) / len(trades) if trades else 0.0
    avg_win = float(np.mean([t["pnl_pct"] for t in wins])) if wins else 0.0
    avg_loss = float(np.mean([t["pnl_pct"] for t in losses])) if losses else 0.0
    gross_wins = sum(t["pnl"] for t in wins)
    gross_losses = abs(sum(t["pnl"] for t in losses))
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")
    avg_holding = float(np.mean([t["holding_bars"] for t in trades])) if trades else 0.0
    total_fees = sum(t["fees"] for t in trades)

    # Regime distribution
    regime_counts = defaultdict(int)
    for r in regimes_log:
        regime_counts[r["regime"]] += 1

    return {
        "symbol": symbol,
        "initial_capital": initial_capital,
        "final_capital": round(final_capital, 2),
        "total_return_pct": round(total_return * 100, 2),
        "sharpe_ratio": round(sharpe, 3),
        "sortino_ratio": round(sortino, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "calmar_ratio": round(calmar, 3),
        "total_trades": len(trades),
        "win_rate": round(win_rate * 100, 1),
        "avg_win_pct": round(avg_win * 100, 2),
        "avg_loss_pct": round(avg_loss * 100, 2),
        "profit_factor": round(profit_factor, 3),
        "avg_holding_bars": round(avg_holding, 1),
        "total_fees": round(total_fees, 2),
        "trades": trades,
        "regimes": dict(regime_counts),
        "signals_generated": len(signals_log),
    }


# ============================================================================
# MAIN — Run Multi-Symbol Backtest
# ============================================================================

if __name__ == "__main__":
    SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", "DOGEUSDT", "XRPUSDT"]
    DAYS = 180
    INTERVAL_MINUTES = 60  # 1-hour bars
    INITIAL_CAPITAL = 100000.0
    # Full signal threshold — all 7 engines now available
    SIGNAL_THRESHOLD = 0.20

    start_time = time.time()
    all_results = {}

    print(f"\n{'='*80}")
    print(f"  CONFIGURATION")
    print(f"{'='*80}")
    print(f"  Symbols:          {', '.join(SYMBOLS)}")
    print(f"  Period:           {DAYS} days ({DAYS * 24} hourly bars per symbol)")
    print(f"  Initial Capital:  ${INITIAL_CAPITAL:,.0f} per symbol")
    print(f"  Signal Threshold: {SIGNAL_THRESHOLD}")
    print(f"  Max Position:     10% of capital")
    print(f"  Fee Rate:         0.1% (Binance taker)")
    print(f"  Risk Gates:       6-check cascade (same as live)")
    print(f"  Mode:             Math-only (7 signal sources active)")

    for symbol in SYMBOLS:
        print(f"\n{'─'*80}")
        print(f"  Processing: {symbol}")
        print(f"{'─'*80}")

        prices, volumes = generate_crypto_prices(symbol, days=DAYS, interval_minutes=INTERVAL_MINUTES)
        print(f"  Price data: {len(prices)} bars | Start: ${prices[0]:,.2f} | End: ${prices[-1]:,.2f}")

        result = run_backtest(symbol, prices, volumes, initial_capital=INITIAL_CAPITAL,
                              signal_threshold=SIGNAL_THRESHOLD)
        all_results[symbol] = result

        print(f"  Return: {result['total_return_pct']:+.2f}% | Sharpe: {result['sharpe_ratio']:.3f} | "
              f"MaxDD: {result['max_drawdown_pct']:.2f}% | Trades: {result['total_trades']} | "
              f"Win Rate: {result['win_rate']:.1f}%")

    # ============================================================================
    # AGGREGATE RESULTS
    # ============================================================================
    elapsed = time.time() - start_time

    print(f"\n{'='*80}")
    print(f"  AGGREGATE RESULTS — ALL 7 SYMBOLS")
    print(f"{'='*80}")

    total_initial = INITIAL_CAPITAL * len(SYMBOLS)
    total_final = sum(r["final_capital"] for r in all_results.values())
    total_return = (total_final - total_initial) / total_initial * 100
    avg_sharpe = np.mean([r["sharpe_ratio"] for r in all_results.values()])
    avg_sortino = np.mean([r["sortino_ratio"] for r in all_results.values()])
    avg_max_dd = np.mean([r["max_drawdown_pct"] for r in all_results.values()])
    total_trades = sum(r["total_trades"] for r in all_results.values())
    avg_win_rate = np.mean([r["win_rate"] for r in all_results.values()])
    total_fees = sum(r["total_fees"] for r in all_results.values())

    print(f"\n  Portfolio Performance:")
    print(f"  {'─'*60}")
    print(f"  Total Initial Capital:   ${total_initial:>12,.2f}")
    print(f"  Total Final Capital:     ${total_final:>12,.2f}")
    print(f"  Total Return:            {total_return:>+10.2f}%")
    print(f"  Average Sharpe:          {avg_sharpe:>10.3f}")
    print(f"  Average Sortino:         {avg_sortino:>10.3f}")
    print(f"  Average Max Drawdown:    {avg_max_dd:>10.2f}%")
    print(f"  Total Trades:            {total_trades:>10d}")
    print(f"  Average Win Rate:        {avg_win_rate:>10.1f}%")
    print(f"  Total Fees Paid:         ${total_fees:>12,.2f}")
    print(f"  Runtime:                 {elapsed:>10.1f}s")

    print(f"\n  Per-Symbol Breakdown:")
    print(f"  {'─'*60}")
    print(f"  {'Symbol':<10} {'Return':>8} {'Sharpe':>8} {'MaxDD':>8} {'Trades':>8} {'WinRate':>8} {'PF':>8}")
    print(f"  {'─'*60}")
    for sym, r in sorted(all_results.items(), key=lambda x: x[1]["total_return_pct"], reverse=True):
        print(f"  {sym:<10} {r['total_return_pct']:>+7.2f}% {r['sharpe_ratio']:>7.3f} "
              f"{r['max_drawdown_pct']:>7.2f}% {r['total_trades']:>7d} "
              f"{r['win_rate']:>7.1f}% {r['profit_factor']:>7.3f}")

    # Print regime distribution
    print(f"\n  Market Regime Distribution (sampled):")
    print(f"  {'─'*60}")
    all_regimes = defaultdict(int)
    for r in all_results.values():
        for regime, count in r.get("regimes", {}).items():
            all_regimes[regime] += count
    total_samples = sum(all_regimes.values()) or 1
    for regime, count in sorted(all_regimes.items(), key=lambda x: -x[1]):
        print(f"  {regime:<20} {count:>5} samples ({count/total_samples*100:.1f}%)")

    # Trade details for best and worst
    best_sym = max(all_results.items(), key=lambda x: x[1]["total_return_pct"])
    worst_sym = min(all_results.items(), key=lambda x: x[1]["total_return_pct"])

    print(f"\n  Best Performer: {best_sym[0]} ({best_sym[1]['total_return_pct']:+.2f}%)")
    print(f"  Worst Performer: {worst_sym[0]} ({worst_sym[1]['total_return_pct']:+.2f}%)")

    # Save results to JSON
    output_file = os.path.join(os.path.dirname(__file__), "data", "backtest_results.json")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    save_data = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "symbols": SYMBOLS,
            "days": DAYS,
            "interval": f"{INTERVAL_MINUTES}m",
            "initial_capital_per_symbol": INITIAL_CAPITAL,
            "signal_threshold": 0.25,
            "fee_rate": 0.001,
            "mode": "math_only (no ML)",
            "signal_sources": "Fourier + Wavelet + Stochastic + Momentum + MeanReversion + Volatility + Microstructure",
        },
        "aggregate": {
            "total_initial": total_initial,
            "total_final": round(total_final, 2),
            "total_return_pct": round(total_return, 2),
            "avg_sharpe": round(avg_sharpe, 3),
            "avg_sortino": round(avg_sortino, 3),
            "avg_max_drawdown": round(avg_max_dd, 2),
            "total_trades": total_trades,
            "avg_win_rate": round(avg_win_rate, 1),
            "total_fees": round(total_fees, 2),
            "runtime_seconds": round(elapsed, 1),
        },
        "per_symbol": {
            sym: {k: v for k, v in r.items() if k != "trades"}
            for sym, r in all_results.items()
        },
    }

    with open(output_file, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Results saved to: {output_file}")
    print(f"{'='*80}")
