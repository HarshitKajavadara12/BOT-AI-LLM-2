# QUANTUM-FORGE — Complete System Explanation

## What This System Is

This is an **institutional-grade cryptocurrency algorithmic trading platform** designed for Binance. It runs 24/7, trading 7 crypto pairs using a fusion of mathematical signal analysis (Fourier, wavelets, stochastic calculus), a 9-model ML ensemble (LSTM, Transformer, PPO, SAC), and cross-asset alpha detection — all governed by a 6-gate risk management system.

**135+ modules | 166 validated components | 10 Streamlit dashboards | 7 crypto pairs | 2-second trading loop**

---

## How to Run It (Input → Output)

### Starting the System

```bash
# Primary method — direct pipeline (recommended)
python launch_quantum_core.py --symbols BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT --capital 100000

# With ML disabled (math-only mode)
python launch_quantum_core.py --symbols BTCUSDT --capital 50000 --no-ml

# With LLM research assistant enabled
python launch_quantum_core.py --symbols BTCUSDT,ETHUSDT --capital 100000 --llm

# Full system validation (loads all 135+ modules)
python run_full_system.py

# Streamlit dashboard (10 pages)
streamlit run app.py
```

### What You Give It (Input)

| Input | Source | Description |
|-------|--------|-------------|
| **Live Price Data** | Binance WebSocket (automatic) | Real-time ticks, 1-min OHLCV candles, orderbook depth |
| **Configuration** | `config/binance_config.yaml` | Pairs, position limits, risk thresholds |
| **Capital** | CLI argument | Starting paper trading capital (default $100,000) |
| **API Keys** | `.env` file | Binance API key/secret (optional for paper mode) |

You don't manually feed it news or text — it connects to **Binance WebSocket** and processes live market data automatically, 24/7.

### What You Get Back (Output)

Every 2 seconds, for each of the 7 symbols, the system produces:

```
Trading Decision:
  Signal Direction:     BUY / SELL / HOLD
  Signal Strength:      -1.0 to +1.0 (negative = sell, positive = buy)
  Signal Components:
    Math Signal:        -1.0 to +1.0 (from 7 mathematical lenses)
    ML Signal:          -1.0 to +1.0 (from 9 models)
    Cross-Asset Signal: -1.0 to +1.0 (from intermarket analysis)
  Market Regime:        BULL / BEAR / NEUTRAL / HIGH_VOLATILITY / CRISIS
  Regime Confidence:    0.0 to 1.0
  Risk Decision:        PASS / BLOCKED (with reason)
  Execution Algorithm:  VWAP / TWAP / IS / MARKET
  Position Size:        $ amount based on signal strength

Continuous Output:
  Portfolio P&L (real-time)
  Equity curve
  Trade history
  Hash-chained audit trail (tamper-proof)
  Prometheus metrics (latency, throughput, fill rate)
  Telegram/Email alerts on regime changes
```

### Example Terminal Output (Live)
```
[14:30:02] BTCUSDT | Price: $67,234.50 | Signal: +0.42 (BUY)
           Math: +0.38 | ML: +0.51 | CrossAsset: +0.33
           Regime: BULL (0.80) | Risk: PASS
           → EXECUTE: BUY $3,361 via TWAP | Position: 2.1% of capital

[14:30:04] ETHUSDT | Price: $3,456.20 | Signal: -0.12 (HOLD)
           Math: -0.15 | ML: -0.08 | CrossAsset: -0.14
           Regime: NEUTRAL (0.55) | Risk: PASS (below threshold)
           → NO TRADE: signal below minimum (0.25)

[14:30:06] SOLUSDT | Price: $145.80 | Signal: +0.73 (BUY)
           Math: +0.65 | ML: +0.82 | CrossAsset: +0.71
           Regime: BULL (0.85) | Risk: PASS
           → EXECUTE: BUY $5,290 via VWAP | Position: 5.3% of capital
```

---

## What Happens Inside (The Full 10-Step Process)

The brain is `QuantumCoreOrchestrator` — it runs a **10-step loop every 2 seconds** for each of the 7 trading pairs:

### Step 1: Fetch Real Market Data

```
Binance WebSocket (primary) → REST API (fallback)
  ├── Current price (real-time tick)
  ├── 100 × 1-minute OHLCV candles (price history)
  ├── Volume data
  └── Feed to shadow strategies (comparison tracking)
```

### Step 2: Mathematical Signal Generation (7 Lenses)

The `SignalGenerator` analyzes price from 7 different mathematical perspectives:

| Source | Weight | Math Used | What It Detects |
|--------|--------|-----------|-----------------|
| **Fourier** | 15% | FFT → power spectrum → dominant frequency → phase | Price cycles (at trough = BUY, at peak = SELL) |
| **Wavelet** | 15% | Multi-scale moving average at scales [4,8,16,32] | Trend direction across multiple timeframes |
| **Stochastic** | 20% | Ornstein-Uhlenbeck process (θ, μ, σ estimation via OLS) | Mean-reversion speed & z-score deviation |
| **Momentum** | 15% | Multi-timeframe Rate of Change [5,10,20,50] | Trending strength and direction |
| **Mean Reversion** | 15% | Z-score + Hurst exponent (R/S analysis) | How far price deviated from mean |
| **Volatility** | 10% | Short-term/long-term vol ratio | Vol spikes (danger) or compression (opportunity) |
| **Microstructure** | 10% | Support/resistance from local min/max | Key price levels |

**Key Formulas:**

Fourier Signal:
```
FFT_result = np.fft.rfft(prices - mean_price)
dominant_frequency = argmax(|FFT|²)
cycle_position = cos(2π × freq × t + phase)
signal = -cycle_position × spectral_ratio × 3   (at trough → BUY)
```

Stochastic (Ornstein-Uhlenbeck):
```
dX = θ(μ - X)dt + σdW

θ (mean-reversion speed) = -ln(|b|) / dt     (from OLS: X[t] = a + b×X[t-1])
μ (long-term mean) = a / (1 - b)
σ (volatility) = residual_std / √dt

If θ > 0.01: signal = -z_score / 2   where z = (price - μ) / (σ × √(1/(2θ)))
```

Hurst Exponent (for mean-reversion confidence):
```
H < 0.45 → mean-reverting (boost signal ×1.5)
H = 0.50 → random walk (no edge)
H > 0.55 → trending (suppress mean-reversion ×0.3, use momentum)
```

### Step 3: Regime Detection (5 Methods + Consensus)

The `RegimeDetector` determines what "mode" the market is in:

| Method | How It Works |
|--------|-------------|
| Volatility Regime | EWMA vol with α=0.06; short/medium/long windows |
| Trend Regime | Multi-timeframe momentum [5, 20, 60] periods |
| HMM Regime | 3-state Gaussian Hidden Markov Model on |returns| |
| Drawdown Detection | (peak - current) / peak |
| EWMA Vol (GARCH-like) | √(α×r² + (1-α)×σ²) |

**Consensus Logic:**
```
drawdown > 15% AND vol = EXTREME  → CRISIS (confidence 0.95)
drawdown > 20%                    → CRISIS (confidence 0.90)
vol_danger AND hmm_danger         → HIGH_VOLATILITY (0.90)
trend = BEAR AND drawdown > 5%    → BEAR (0.80)
trend = BULL AND vol = NORMAL     → BULL (0.80)
Default                           → NEUTRAL (0.50)
```

Regime must be confirmed 2+ consecutive times (except CRISIS = immediate).

### Step 4: ML Ensemble (9 Models)

The `MLEnsembleEngine` runs 9 machine learning models on 32 engineered features:

| Model | Type | Architecture |
|-------|------|-------------|
| **LSTM** | Deep Learning | input=32, hidden=64, layers=2, output=1 |
| **GRU** | Deep Learning | input=32, hidden=64, layers=2, output=1 |
| **Transformer** | Deep Learning | input=32, hidden=64, heads=4, layers=2 |
| **TCN** | Deep Learning | Temporal Convolutional Network |
| **PPO** | Reinforcement Learning | Proximal Policy Optimization |
| **SAC** | Reinforcement Learning | Soft Actor-Critic |
| **Gaussian Process** | Probabilistic | Non-parametric with uncertainty |
| **Linear Momentum** | Statistical | Lightweight trend following |
| **Vol Predictor** | Statistical | Volatility forecasting |

**How predictions combine:**
```python
final_ml = Σ(model_prediction × model_weight) × (0.5 + 0.5 × consensus)
# consensus = how many models agree on direction (0 → 1)
# Weights adapt online based on each model's trailing accuracy
```

### Step 5: Signal Fusion (Triple-Source Blending)

```
Final Signal = Math_Signal × 0.50 + ML_Signal × 0.30 + CrossAsset_Signal × 0.20
```

If `|final_signal| > threshold (default 0.25)`: → BUY or SELL  
Otherwise: → HOLD (no trade)

### Step 6: Risk Gate (6 Safety Checks)

Every signal must pass ALL 6 gates before execution:

| # | Gate | Rule | If Failed |
|---|------|------|-----------|
| 1 | Crisis Regime | Is market in CRISIS? | BLOCK ALL trading |
| 2 | High Vol | Is volatility extreme? | Require signal > 0.7 |
| 3 | Drawdown | Portfolio drawdown > 15%? | BLOCK ALL trading |
| 4 | Position Size | Symbol position ≥ 10% of capital? | BLOCK BUY for that symbol |
| 5 | Total Exposure | Total invested ≥ 80% of capital? | BLOCK all new BUYs |
| 6 | Signal Floor | Is signal strong enough? | NEUTRAL: min 0.3; BEAR: min 0.5 |

Plus: Circuit Breaker (5 consecutive failures → 60s cooldown)

### Step 7: Execute Trade

The `ExecutionManager` auto-selects the best algorithm:

| Condition | Algorithm | Why |
|-----------|-----------|-----|
| Signal > 0.8 (very strong) | MARKET | Need immediate fill |
| Order < $500 | MARKET | Too small for algo |
| Volatility > 5% | Implementation Shortfall | Minimize market impact |
| Order > $5000 | VWAP | Large order, blend with volume |
| Default | TWAP | Spread evenly over time |

Position sizing: `capital × 5% × signal_strength`  
Fee model: 0.1% taker / 0.075% maker

### Step 8: Post-Trade Analysis
- Market Impact Tracker: measures fill quality
- Alert system: Telegram/email notification

### Step 9: Audit Trail
- Hash-chained JSONL log (tamper-proof: each entry contains hash of previous)
- Records: market_state, signal_state, risk_state, decision

### Step 10: Feedback & Adaptation
- ML model weights updated based on realized P&L
- Models that predicted correctly get boosted
- State saved every 50 iterations

---

## The 32 Features (What ML Models See)

| Category | Features |
|----------|----------|
| **Returns** (10) | Mean/Std (5-day, 20-day), skewness, kurtosis, last return, max/min return, cumulative 20-day |
| **Technical** (10) | RSI-14, MACD (line/signal/histogram), Bollinger %B, ATR-14, ROC (5/10/20), Z-score-20 |
| **Spectral** (5) | Dominant frequency, dominant power, spectral entropy, centroid, bandwidth |
| **Microstructure** (5) | Spread proxy, autocorrelation(1), price clustering, vol ratio, tick direction |
| **Volume** (2) | Volume ratio (5/20 day), coefficient of variation |

---

## Cross-Asset Alpha (Intermarket Signals)

The `CrossAssetAlphaEngine` finds signals from relationships between crypto pairs:

| Signal Type | Logic |
|-------------|-------|
| **BTC-Leading** | If BTC moved first, altcoins follow (lead-lag correlation) |
| **Correlation Breakdown** | When usual correlations break → mean-reversion opportunity |
| **Relative Strength** | Buy strongest / sell weakest (sector rotation) |
| **Pair Spread** | When pairs deviate from typical spread → expect reversion |

---

## Data Sources

| Source | Connection | Data |
|--------|-----------|------|
| **Binance WebSocket** | `wss://stream.binance.com` | Real-time ticks, klines, depth, trades |
| **Binance REST API** | `api.binance.com/api/v3` | Historical klines, account info, order placement |
| **Alternative Data** | `AlternativeDataLoader` | Social sentiment, on-chain metrics |
| **Redis** | Localhost:6379 | Three-tier cache (<100µs hot ticks) |
| **TimescaleDB** | PostgreSQL extension | Time-series storage for historical analysis |
| **DuckDB** | In-process | Fast analytical queries for LLM/RAG |
| **Qdrant** | Vector DB | Semantic search over trade history |

---

## Why These Technologies (And Not Others)

| Technology | Why Chosen | Why Not The Alternative |
|------------|-----------|----------------------|
| **PyTorch** (LSTM, GRU, Transformer) | Dynamic computation graphs, ideal for time-series where sequence length varies. Strong RL ecosystem (PPO, SAC). GPU acceleration for inference. | TensorFlow: static graphs are harder for RL. JAX: smaller ecosystem for trading. |
| **Fourier/FFT** | Detects hidden cycles in price data (e.g., 4-hour cycle, daily cycle). Fast O(n log n). Mathematically rigorous. | Moving averages: only detect trends, miss cycles. Visual pattern matching: subjective. |
| **Ornstein-Uhlenbeck** (stochastic calculus) | Proper mathematical framework for mean-reversion. Gives you exact parameters: speed (θ), mean (μ), volatility (σ). Can calculate optimal entry/exit. | Simple z-score: works but doesn't tell you HOW FAST it reverts. No theoretical framework. |
| **Wavelets** | Multi-scale trend detection — see both short-term noise and long-term trend simultaneously. Better than single-timeframe indicators. | MACD: only 2 timeframes. Fourier: loses time information. |
| **HMM** (Hidden Markov Model) | Markets have hidden states (bull/bear/crisis) that you can't directly observe. HMM infers these states from returns. Theoretically sound for regime detection. | Simple threshold-based: arbitrary cutoffs. K-means: doesn't model sequential transitions. |
| **PPO + SAC** (Reinforcement Learning) | Learn optimal trading policy through trial-and-error. SAC handles continuous action spaces. PPO is stable for financial environments. | DQN: discrete actions only (buy/sell/hold). A3C: less stable. |
| **Gaussian Process** | Gives prediction WITH uncertainty bounds. When GP says "I don't know," you can reduce position size. Few other models give calibrated uncertainty. | Neural networks: point predictions only. Bayesian NNs: expensive. |
| **VWAP/TWAP/IS** (execution algorithms) | Institutional-standard execution. VWAP blends with market volume (less impact). TWAP for predictability. IS minimizes total cost. | Market orders only: high slippage on large orders. Limit orders only: may not fill. |
| **Redis** (cache) | <100µs access for hot tick data. Three-tier caching (hot/warm/cold). Standard in HFT systems. | Database: too slow for real-time ticks. Files: no atomicity. |
| **TimescaleDB** | PostgreSQL + time-series optimization. Hypertables auto-partition by time. Standard SQL interface. | InfluxDB: limited query capabilities. MongoDB: no time partitioning. |
| **Streamlit** | 10 interactive dashboards in Python — no frontend build tools needed. Real-time refresh. Perfect for quant research. | React: overkill for internal tools. Dash: more verbose. Grafana: limited customization. |
| **Hash-chained JSONL** (audit) | Tamper-proof audit trail. Each entry contains SHA-256 hash of previous. Can detect if ANY entry was modified. Required for institutional compliance. | Simple logging: can be edited. Database: can be altered. |
| **DuckDB** (LLM cache) | In-process analytical database. No server needed. 10-100× faster than SQLite for OLAP queries. Perfect for LLM/RAG analytical queries. | PostgreSQL: needs server. SQLite: slow for analytics. Pandas: memory-limited. |
| **Llama 3.2 8B** (LLM) | Local inference (no API costs). Fast (50-200ms). Privacy — trade data never leaves machine. GGUF quantization for CPU. | GPT-4: expensive API calls, latency, data privacy risk. Claude: same issues. |
| **Qdrant** (vector store) | Purpose-built for semantic search. Fast similarity matching for "find similar trades." Better than generic databases. | Pinecone: cloud-only. ChromaDB: less performant. FAISS: no persistence. |

---

## The Creator's Thinking: Why This System Exists

### The Core Problem

> "Most crypto trading bots use ONE signal source — either technical indicators OR machine learning.  
> Both fail alone. Indicators lag. ML overfits.  
> What if you FUSE mathematical models, ML models, AND cross-asset signals, then GATE everything through institutional risk management?"

### The Intellectual Evolution

```
Step 1: "Simple indicators (RSI, MACD) are too slow for crypto's 4-8% daily moves"
Step 2: "What if I use Fourier analysis to find hidden cycles in crypto prices?"
Step 3: "Add stochastic calculus (Ornstein-Uhlenbeck) for mean-reversion math"
Step 4: "Add wavelets for multi-timeframe trend detection"
Step 5: "Now add ML — LSTM, Transformer, RL — but they can overfit"
Step 6: "Solution: FUSE math (50%) + ML (30%) + cross-asset (20%) — math anchors ML"
Step 7: "Add regime detection — don't trade in CRISIS, require stronger signals in BEAR"
Step 8: "Add institutional execution (VWAP, TWAP) — minimize market impact"
Step 9: "Add LLM but with ZERO execution authority — it can only observe and explain"
Step 10: "Make everything auditable with hash-chained logs — institutional compliance"
```

### Key Design Decisions

1. **Math gets 50% weight (not ML)** — Because mathematical models are deterministic, interpretable, and don't overfit. ML is powerful but unreliable alone.

2. **LLM has ZERO trading authority** — The LLM can explain signals, answer questions, and analyze performance. But it CANNOT place trades. If the LLM crashes, trading continues. This prevents AI hallucination from causing financial loss.

3. **6-gate risk management** — Every trade must pass 6 independent safety checks. Even if signals are wrong, risk gates prevent catastrophic loss (max 15% drawdown, max 10% per position, max 80% total exposure).

4. **2-second loop** — Crypto markets move fast (4-8% daily). A 2-second processing cycle means the system can react within seconds to price movements. Not HFT (milliseconds) but fast enough for the signal quality produced.

5. **Paper trading default** — The system runs in simulated mode by default. Live trading requires explicit API keys and mode switch. This prevents accidental real-money trading.

6. **Graceful degradation everywhere** — If Redis dies → in-memory cache. If Qdrant dies → skip vector search. If LLM dies → trading continues. If WebSocket drops → REST fallback. Nothing is a single point of failure.

---

## Risk Mathematics (Advanced)

### Extreme Value Theory (EVT)
Models the **tail risk** — what's the probability of a -25% crash?
```
GEV Distribution: F(x) = exp(-(1 + ξ(x-μ)/σ)^(-1/ξ))
  ξ > 0 (Fréchet): Heavy tails (crypto markets)
  ξ = 0 (Gumbel): Light tails (stocks)
  ξ < 0 (Weibull): Bounded tails (rare)

Metrics computed: VaR(95/99/99.9%), Expected Shortfall, Return Levels (10yr, 100yr)
```

### Copula Models
Models how assets crash **together** (joint tail risk):
```
Types: Gaussian, t-Copula, Clayton, Gumbel, Frank, Joe

Gaussian Copula: ρ_Gaussian = 2×sin(π/6 × ρ_Spearman)
t-Copula: Models heavier tail dependence than Gaussian

Key metric: Tail Dependence Coefficient — probability of ETH crashing
           given that BTC is already crashing
```

### Optimal Stopping (When to Exit)
```
American Option Pricing via binomial tree:
  u = exp(σ√dt)    (up factor)
  d = 1/u          (down factor)  
  p = (exp(rdt) - d) / (u - d)   (risk-neutral probability)
  
Backward induction: At each node, compare "hold" vs "exercise" value
→ Tells you the optimal moment to close a position
```

### Cognitive Dampener
```
Position Multiplier = regime_penalty × confidence_factor  (clamped to [0.2, 1.0])

HIGH_VOLATILITY: × 0.5    (cut position in half)
MARKET_CRASH:    × 0.2    (cut to 20%)
LIQUIDITY_CRISIS: × 0.3   (cut to 30%)
STABLE:          × 1.0    (full position)
```

---

## Configuration & Parameters

### Trading Parameters
| Parameter | Value | Meaning |
|-----------|-------|---------|
| Loop interval | 2 seconds | How often to process each symbol |
| Capital | $100,000 (default) | Starting paper trading balance |
| Position size per trade | 5% × signal_strength | How much capital per trade |
| Max position per symbol | 10% | Never hold >$10K in one coin |
| Max total exposure | 80% | Keep 20% in cash always |
| Max drawdown | 15% | Stop trading if portfolio drops 15% |
| Signal threshold | 0.25 | Minimum signal to trade |
| Fee rate | 0.1% taker | Binance standard fee |
| Stop loss | 5% | Exit if position drops 5% |
| Take profit | 10% | Exit at 10% profit |
| Trailing stop | Activate at 3%, trail 2% | Lock in profits |
| Max leverage | 3.0× | Not used in paper mode |

### Per-Pair Risk Limits
| Pair | Max Position | Expected Daily Vol |
|------|-------------|-------------------|
| BTCUSDT | $100,000 | 4% |
| ETHUSDT | $80,000 | 5% |
| BNBUSDT | $50,000 | 5% |
| SOLUSDT | $50,000 | 6% |
| ADAUSDT | $30,000 | 6% |
| DOGEUSDT | $30,000 | 8% |
| XRPUSDT | $30,000 | 5% |

### Stress Test Scenarios (Built-In)
| Scenario | BTC | ETH | SOL | DOGE |
|----------|-----|-----|-----|------|
| Flash Crash | -25% | -30% | -35% | -40% |

---

## LLM Integration (Read-Only Research Assistant)

The system includes a **Llama 3.2 8B** local LLM that acts as a research assistant:

```
User: "What's my portfolio status?"
LLM:  "You have 3 open positions: BTC +2.3%, ETH -0.5%, SOL +4.1%
       Total unrealized P&L: +$1,240. Sharpe ratio: 1.85."

User: "Why did the system sell ETH yesterday?"
LLM:  "Signal turned negative (-0.63) due to: Fourier cycle peak detected,
       ML ensemble bearish consensus (7/9 models), and BTC-leading lag
       correlation broke down. Regime: HIGH_VOLATILITY required strength > 0.7."
```

**CRITICAL CONSTRAINT**: The LLM has **ZERO execution authority**. It cannot:
- Place trades
- Modify parameters
- Override risk gates
- Change allocations

It can ONLY read data and explain. If `LLM_ENABLED=false`, the trading system works exactly the same.

---

## Validation Results

**166 total checks | 164 passed | 2 failed (optional packages) | 98.8% pass rate**

| Category | Passed | Total | Status |
|----------|--------|-------|--------|
| Data Ingestion | 8 | 8 | 100% |
| Deep Learning Models | 8 | 9 | 89% (GNN needs torch_geometric) |
| Reinforcement Learning | 4 | 4 | 100% |
| Feature Learning | 3 | 4 | 75% (UMAP optional) |
| Math Engine | 10 | 10 | 100% |
| Risk Mathematics | 5 | 5 | 100% |
| Signal & Alpha | 14 | 14 | 100% |
| Execution Algorithms | 4 | 4 | 100% |
| Execution Infrastructure | 10 | 10 | 100% |
| Backtesting | 6 | 6 | 100% |
| Analytics | 24 | 24 | 100% |
| LLM Integration | 4 | 4 | 100% |
| Core System | 9 | 9 | 100% |
| Infrastructure | 5 | 5 | 100% |
| Strategies | 4 | 4 | 100% |
| Pipeline Integration | 12 | 12 | 100% |

The 2 failures are optional libraries (`torch_geometric` for GNN, `umap` for manifold learning) — both have fallback implementations.

---

## Backtesting System

### How It Validates Results

```python
# Uses the SAME code as live trading (no separate backtest logic)
backtester = RealBacktester(
    signal_generator=SignalGenerator(),   # same instance
    ml_ensemble=MLEnsembleEngine(),       # same instance
    regime_detector=RegimeDetector(),     # same instance
    fee_rate=0.001,                       # realistic fees
)
```

### Walk-Forward Analysis
```
Training window:   252 days (1 year)
Validation window: 63 days (3 months)
Step size:         21 days (monthly)
→ Train on history, validate on unseen future, step forward, repeat
```

### Performance Metrics Computed
| Metric | Description |
|--------|-------------|
| Sharpe Ratio | Risk-adjusted return (target: >1.5) |
| Sortino Ratio | Downside risk-adjusted |
| Max Drawdown | Worst peak-to-trough (target: <15%) |
| Calmar Ratio | Return / Max Drawdown |
| Win Rate | % profitable trades (target: >55%) |
| Profit Factor | Gross profit / Gross loss |
| Equity Curve | Capital over time |

---

## Dashboard (10 Interactive Pages)

| # | Page | Shows |
|---|------|-------|
| 1 | **Main Dashboard** | System health, module status, quick overview |
| 2 | **Trading** | Live orders, executions, signal history |
| 3 | **Risk** | Drawdown, VaR, regime, exposure |
| 4 | **Portfolio** | Holdings, P&L, allocation |
| 5 | **Analytics** | Backtest results, performance attribution |
| 6 | **Research** | Alpha discovery, factor analysis |
| 7 | **Execution** | VWAP/TWAP fill analysis, slippage |
| 8 | **Microstructure** | Orderbook depth, spread, toxicity |
| 9 | **Configuration** | System settings, pair configs |
| 10 | **Investor Portal** | External-facing performance summary |

---

## File Structure

```
QUANTUM-FORGE/
├── launch_quantum_core.py       # Primary entry point
├── launch_pipeline.py           # Pipeline launcher
├── run_full_system.py           # Full module validation
├── app.py                       # Streamlit dashboard (10 pages)
├── config/
│   ├── system.yaml              # Master config (DB, cache, messaging)
│   ├── binance_config.yaml      # Per-pair trading config
│   ├── strategies/              # Strategy parameter files
│   └── firms/                   # Firm-level risk profiles
├── core/
│   ├── quantum_core.py          # THE BRAIN: 10-step orchestrator
│   ├── signal_generator.py      # 7-source mathematical signal fusion
│   ├── ml_ensemble.py           # 9-model ML ensemble with online learning
│   ├── regime_detector.py       # 5-method regime detection + consensus
│   ├── feature_pipeline.py      # 32 engineered features
│   ├── cross_asset_alpha.py     # Intermarket signal generation
│   ├── execution_manager.py     # VWAP/TWAP/IS algorithm selection
│   ├── capital_allocator.py     # Regime-adaptive capital allocation
│   ├── math_engine/             # Fourier, stochastic, wavelets, Kalman
│   ├── risk_mathematics/        # EVT, copulas, optimal stopping, drawdown
│   ├── execution_algorithms/    # VWAP, TWAP, IS, arrival price
│   ├── strategies/              # Momentum, QuantumSignal (shadow)
│   └── market_microstructure/   # Orderbook, price formation, liquidity
├── intelligence/
│   ├── deep_learning/           # LSTM, GRU, Transformer, TCN, CNN
│   ├── reinforcement_learning/  # PPO, SAC, MBPO, MarketEnvironment
│   ├── probabilistic_ml/        # GP, Bayesian, VI, Conformal
│   ├── meta_learning/           # MAML, few-shot, transfer, distillation
│   └── feature_learning/        # Representation, causal discovery
├── data/
│   ├── ingestion/               # Binance WebSocket, REST, ticks, orderbook
│   ├── preprocessing/           # Clean, normalize, align
│   ├── storage/                 # Redis, TimescaleDB, Parquet, FeatureStore
│   └── parquet/                 # Columnar storage for historical data
├── execution/
│   ├── order_management/        # OMS, smart router, position manager
│   ├── slippage_control/        # Adaptive execution, impact model
│   ├── pre_trade_analytics/     # Cost estimation, liquidity, risk
│   └── latency_critical/        # Ultra-low latency execution
├── analytics/
│   ├── backtesting/             # Walk-forward, event-driven, regime-aware
│   ├── performance_attribution/ # P&L decomposition, risk-adjusted metrics
│   ├── alpha_research/          # Discovery, validation, decay analysis
│   ├── factor_research/         # Factor construction, selection, combination
│   └── market_regime/           # HMM, GARCH, changepoint, correlation
├── risk_management/             # Portfolio risk, limits, VaR
├── llm_integration/             # Llama 3.2 LLM, Qdrant, DuckDB, API
├── pages/                       # 10 Streamlit dashboard pages
├── research/                    # Research notebooks and experiments
└── tests/                       # Test suite
```

---

## What Makes This Different From Simple Trading Bots

| Simple Bot | QUANTUM-FORGE |
|-----------|---------------|
| One indicator (RSI > 70 = sell) | 7 mathematical lenses fused with 9 ML models |
| No risk management | 6-gate risk cascade + circuit breaker + cognitive dampener |
| One timeframe | Multi-scale analysis (4, 8, 16, 32 bars + Fourier cycles) |
| No regime awareness | 5-method regime detection (won't trade in CRISIS) |
| Market orders only | Auto-selects VWAP/TWAP/IS based on size and volatility |
| No audit trail | Hash-chained tamper-proof JSONL |
| ML OR math | Math (50%) + ML (30%) + Cross-Asset (20%) — math anchors ML |
| LLM controls trading | LLM has ZERO execution authority — read-only |
| Single asset | 7 crypto pairs simultaneously + cross-asset signals |
| Fixed position size | Regime-adaptive: cut to 20% in crash, full in stable |
| No backtesting | Walk-forward with realistic costs (252d train, 63d validate) |
| Breaks when dependency fails | Every external dependency has in-memory fallback |

---

## Technical Requirements

```
Python >= 3.8

Core:
  torch >= 1.10       (deep learning)
  numpy >= 1.21       (numerical computing)
  pandas >= 1.3       (data manipulation)
  scipy >= 1.7        (scientific computing)
  scikit-learn >= 1.0  (ML models)
  statsmodels >= 0.13  (statistical models)

Exchange:
  websockets >= 10.1   (Binance WebSocket)
  websocket-client     (alternative WS)
  requests             (REST API)

Infrastructure:
  redis >= 7.2         (caching)
  sqlalchemy >= 1.4    (database ORM)
  psycopg2-binary      (PostgreSQL)
  fastapi >= 0.75      (REST API)
  streamlit >= 1.20    (dashboards)

ML/AI:
  hmmlearn >= 0.3      (Hidden Markov Models)
  xgboost >= 1.5       (gradient boosting)
  lightgbm >= 3.3      (gradient boosting)
  arch >= 5.0          (GARCH models)

LLM (optional):
  llama-cpp-python     (local LLM inference)
  qdrant-client        (vector search)
  sentence-transformers (embeddings)
  duckdb               (analytical cache)

Monitoring:
  prometheus-client    (metrics)
  psutil               (system monitoring)
```

---

## Summary

QUANTUM-FORGE is a production-grade crypto trading system that fuses:
- **7 mathematical signal sources** (Fourier, wavelets, OU processes, momentum, mean-reversion, volatility, microstructure)
- **9 ML models** (LSTM, GRU, Transformer, TCN, PPO, SAC, GP, momentum, vol)
- **Cross-asset alpha** (lead-lag, correlation breakdown, relative strength, spread reversion)

Into a single fused signal (50% math / 30% ML / 20% cross-asset) that must pass 6 independent risk gates before execution via institutional algorithms (VWAP/TWAP/IS).

The creator's core insight: **No single signal source is reliable in crypto. But if you fuse math (stable, interpretable) with ML (adaptive, pattern-finding) with cross-asset (contextual), and anchor the fusion with math at 50% weight, you get a robust system that doesn't overfit to any one approach.**

The LLM layer (Llama 3.2 8B) provides research intelligence without any risk of AI-caused trades — a strict "observe and explain" role with zero execution authority.
