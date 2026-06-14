# QUANTUM-FORGE — Operational Workflow Document

> **Project:** QUANTUM-FORGE Institutional Trading Platform  
> **Purpose:** Step-by-step operational workflows showing how every subsystem coordinates during live trading, backtesting, research, deployment, and monitoring.  
> **Generated:** 2026-02-24

---

## 1. LIVE TRADING WORKFLOW — Primary Loop

This is the **main workflow** running in `QuantumCoreOrchestrator._run_loop()`:

```
STARTUP
  │
  ├── 1. Load configuration (symbols, capital, thresholds)
  ├── 2. Initialize all modules:
  │      SignalGenerator, MLEnsembleEngine, RegimeDetector,
  │      CapitalAllocator, StrategyMultiplexer, ShadowTracker,
  │      ExecutionManager, PortfolioRiskManager, AlertSystem,
  │      StorageCoordinator, FeaturePipeline, CrossAssetAlphaEngine,
  │      SVMRegimeClassifier, AlphaResearchScheduler, CausalBridge,
  │      OrderBookAnalyzer, GPPredictionBridge, LLMOutputParser,
  │      AltDataAlphaEngine, AlphaCrowdingDetector, MarketImpactTracker,
  │      SpoofingDetector, QueuePositionEstimator, FinancialLLMFineTuner,
  │      VariationalInferenceBridge
  ├── 3. _restore_state() — load persisted state from disk
  ├── 4. Start WebSocket feeds (one per symbol)
  ├── 5. Register SIGINT/SIGTERM handlers for graceful shutdown
  │
  MAIN LOOP (every ~2 seconds)
  │
  ├── FOR EACH SYMBOL in [BTCUSDT, ETHUSDT, SOLUSDT, DOGEUSDT, ...]
  │     │
  │     ├── Step 1: FETCH MARKET DATA
  │     │   ├── Try WebSocket cache first
  │     │   ├── Fallback to REST API (/api/v3/ticker/price)
  │     │   ├── Fetch 100-bar klines (1-minute)
  │     │   ├── Feed to Strategy Multiplexer (shadow tracking)
  │     │   └── Build DataFrame for feature engineering
  │     │
  │     ├── Step 2: GENERATE MATH SIGNAL
  │     │   ├── SignalGenerator.generate()
  │     │   │   ├── Fourier spectral analysis (dominant frequencies)
  │     │   │   ├── Stochastic process modelling (mean-reversion)
  │     │   │   └── Wavelet decomposition (multi-scale trends)
  │     │   └── Result: math_signal ∈ [-1.0, +1.0]
  │     │
  │     ├── Step 3: DETECT REGIME
  │     │   ├── RegimeDetector.detect() — HMM-based
  │     │   ├── Possible: BULL, BEAR, NEUTRAL, HIGH_VOL, CRISIS
  │     │   ├── CapitalAllocator adjusts positions per regime
  │     │   ├── If regime changed → AlertSystem notification
  │     │   └── SVMRegimeClassifier (online learning, updates every 100 ticks)
  │     │
  │     ├── Step 4: ML ENSEMBLE (if enable_ml=True)
  │     │   ├── FeaturePipeline.compute_features() — 60+ features
  │     │   ├── MLEnsembleEngine.predict()
  │     │   │   ├── LSTM prediction
  │     │   │   ├── Transformer prediction
  │     │   │   ├── PPO action (RL agent)
  │     │   │   └── SAC action (RL agent)
  │     │   ├── CrossAssetAlphaEngine.compute_alpha()
  │     │   ├── CausalBridge.compute_causal_alpha()
  │     │   ├── GPPredictionBridge (Gaussian Process w/ uncertainty)
  │     │   ├── AltDataAlphaEngine (alternative data signals)
  │     │   ├── AlphaCrowdingDetector (crowded trade warning)
  │     │   ├── SpoofingDetector (manipulation detection)
  │     │   ├── QueuePositionEstimator (limit order queue)
  │     │   ├── VariationalInferenceBridge (Bayesian streaming)
  │     │   └── LLMOutputParser (if LLM enabled, read-only signal)
  │     │
  │     ├── Step 5: FUSE SIGNALS
  │     │   ├── Math signal × 0.50
  │     │   ├── ML signal × 0.30
  │     │   ├── Cross-asset × 0.20
  │     │   ├── Dynamic reweighting if any source unavailable
  │     │   └── Result: final_signal ∈ [-1.0, +1.0]
  │     │
  │     ├── Step 6: RISK GATE (6-check cascade)
  │     │   ├── Check 1: Regime gate — CRISIS = BLOCK ALL
  │     │   ├── Check 2: High-vol gate — require |signal| > 0.7
  │     │   ├── Check 3: Drawdown gate — portfolio drawdown < 15%
  │     │   ├── Check 4: Position gate — symbol exposure < 10%
  │     │   ├── Check 5: Total exposure gate — total < 80%
  │     │   ├── Check 6: Signal floor — |signal| > threshold (regime-adjusted)
  │     │   ├── PortfolioRiskManager validates limits
  │     │   ├── Circuit Breaker check (5-failure cooldown)
  │     │   └── Result: BLOCK or PASS
  │     │
  │     ├── Step 7: EXECUTE (if PASS)
  │     │   ├── ExecutionManager.execute_signal()
  │     │   │   ├── Algorithm selection:
  │     │   │   │   ├── size > 1000 & vol < 3% → VWAP
  │     │   │   │   ├── vol > 5% → Market (fast exit)
  │     │   │   │   └── else → TWAP
  │     │   │   ├── Calculate order size from signal × capital fraction
  │     │   │   ├── Apply fee model (0.1% taker / 0.075% maker)
  │     │   │   └── Update position tracker + P&L
  │     │   └── MarketImpactTracker (post-trade fill analysis)
  │     │
  │     ├── Step 8: AUDIT LOG
  │     │   ├── Append to hash-chained JSONL
  │     │   └── Fields: timestamp, symbol, price, signal, regime,
  │     │              risk_decision, execution_result, prev_hash
  │     │
  │     └── Step 9: FEEDBACK & ADAPTATION
  │         ├── Track per-symbol returns
  │         ├── ML ensemble weight adaptation (online learning)
  │         ├── Every 50 symbols: auto-save state
  │         ├── Every 100 ticks: SVM online update
  │         ├── Periodic: EVT tail-risk analysis
  │         ├── Periodic: Copula cross-asset dependence
  │         └── Periodic: Shadow strategy comparison
  │
  └── CONTINUE LOOP (asyncio.sleep(2))

SHUTDOWN
  ├── Triggered by SIGINT / SIGTERM / KeyboardInterrupt
  ├── _save_state() — persist all state to disk
  ├── Close WebSocket connections
  ├── Flush storage buffers
  └── Log final statistics
```

---

## 2. SIGNAL GENERATION WORKFLOW

```
Raw Market Data (OHLCV + Volume + Orderbook)
  │
  ├──────────────────────────────────────────────────┐
  │                                                   │
  ▼                                                   ▼
MATH TRACK                                     ML TRACK
  │                                                   │
  ├── Fourier Analysis                                ├── Feature Pipeline (60+ features)
  │   └── FFT → dominant cycles                       │   ├── Returns (1m, 5m, 15m, 1h)
  │                                                   │   ├── Volatility (rolling, GARCH)
  ├── Stochastic Calculus                             │   ├── Momentum (RSI, MACD)
  │   └── Ornstein-Uhlenbeck mean reversion           │   ├── Volume profile
  │                                                   │   ├── Order flow imbalance
  ├── Wavelet Decomposition                           │   └── Cross-asset correlations
  │   └── Multi-scale trend extraction                │
  │                                                   ├── LSTM Prediction
  ├── Kalman Filter                                   ├── Transformer Attention
  │   └── State estimation + smoothing                ├── PPO Action (RL)
  │                                                   ├── SAC Action (RL)
  └── MATH SIGNAL [-1, +1]                           ├── GP Posterior (uncertainty)
        weight: 50%                                    ├── SVM Hyperplane
                                                       └── ML SIGNAL [-1, +1]
                                                             weight: 30%

CROSS-ASSET TRACK
  │
  ├── CrossAssetAlphaEngine
  │   └── correlation regime, contagion detection
  ├── CausalBridge
  │   └── Granger causality, information flow
  └── CROSS-ASSET SIGNAL [-1, +1]
        weight: 20%

                    ┌──────────────┐
ALL 3 SIGNALS ────► │ SIGNAL FUSION │ ────► FINAL SIGNAL [-1, +1]
                    └──────────────┘
                    Math=50% ML=30% Cross=20%
```

---

## 3. RISK MANAGEMENT WORKFLOW

```
FINAL SIGNAL ARRIVES
  │
  ▼
┌──────────────────────────────────────────────────┐
│                 RISK GATE (6 checks)              │
│                                                    │
│  1. REGIME CHECK                                   │
│     ├── Current regime from RegimeDetector         │
│     ├── CRISIS → BLOCK (immediate return)          │
│     └── BEAR → raise threshold to 0.5              │
│                                                    │
│  2. HIGH VOLATILITY CHECK                          │
│     ├── regime == HIGH_VOLATILITY?                 │
│     └── |signal| < 0.7 → BLOCK                    │
│                                                    │
│  3. DRAWDOWN CHECK                                 │
│     ├── current portfolio drawdown                 │
│     └── > 15% → BLOCK                             │
│                                                    │
│  4. POSITION SIZE CHECK                            │
│     ├── symbol_exposure / total_capital            │
│     └── > 10% → BLOCK                             │
│                                                    │
│  5. TOTAL EXPOSURE CHECK                           │
│     ├── sum(all positions) / total_capital         │
│     └── > 80% → BLOCK                             │
│                                                    │
│  6. SIGNAL STRENGTH CHECK                          │
│     ├── |final_signal| vs regime-adjusted threshold│
│     └── below threshold → BLOCK                    │
│                                                    │
│  CIRCUIT BREAKER                                   │
│     ├── Recent failures > 5 → cooldown             │
│     └── Cooldown period = configurable             │
│                                                    │
│  PORTFOLIO RISK MANAGER                            │
│     ├── Max position value limits                  │
│     ├── Portfolio-level VaR                        │
│     └── Concentration limits                       │
│                                                    │
│  COGNITIVE DAMPENER (if loaded)                    │
│     ├── Regime-aware signal adjustment             │
│     └── Prevents overtrading in uncertain regimes  │
│                                                    │
│  Result: BLOCK or PASS                             │
└──────────────────────────────────────────────────┘
```

---

## 4. EXECUTION WORKFLOW

```
Signal PASSES Risk Gate
  │
  ▼
┌─────────────────────────────────┐
│   EXECUTION MANAGER             │
│                                  │
│   1. Calculate position size     │
│      └── signal × capital       │
│         × per-symbol fraction    │
│                                  │
│   2. Select algo based on:       │
│      ├── order_size > 1000       │
│      │   & volatility < 3%      │
│      │   → VWAP                  │
│      ├── volatility > 5%         │
│      │   → MARKET ORDER          │
│      └── else → TWAP             │
│                                  │
│   3. Execute via chosen algo     │
│      ├── PAPER → simulated fill  │
│      └── LIVE → Binance API      │
│                                  │
│   4. Fee accounting              │
│      ├── Taker: 0.10%           │
│      └── Maker: 0.075%          │
│                                  │
│   5. Slippage model              │
│   6. Update position + P&L      │
│   7. Alert notification          │
└─────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────┐
│   POST-TRADE ANALYTICS          │
│   ├── MarketImpactTracker       │
│   ├── Fill quality analysis     │
│   ├── Execution cost comparison │
│   └── Hash-chained audit entry  │
└─────────────────────────────────┘
```

---

## 5. REGIME DETECTION WORKFLOW

```
Market Data (price, volume, volatility)
  │
  ▼
┌───────────────────────────────────────────┐
│  RegimeDetector.detect()                   │
│                                             │
│  1. HMM (Hidden Markov Model)               │
│     ├── Emission probabilities from returns │
│     ├── Transition matrix                    │
│     └── Viterbi decoding → regime state     │
│                                             │
│  2. Multi-Signal Analysis                    │
│     ├── Volatility clustering (GARCH)       │
│     ├── Correlation regime shifts            │
│     ├── Volume anomalies                     │
│     └── Trend strength indicators            │
│                                             │
│  3. SVM Regime Classifier                    │
│     ├── Online learning (libsvm)            │
│     ├── Hyperplane separation               │
│     └── Updated every 100 new data points   │
│                                             │
│  4. Analytics Layer (parallel)               │
│     ├── HMMRegimeDetector                    │
│     ├── GARCHVolatilityAnalyzer              │
│     ├── CorrelationRegimeDetector            │
│     └── ChangePointDetector                  │
│                                             │
│  Output:                                     │
│  ┌─────────────────────────────────┐        │
│  │ BULL     │ trending up strongly │        │
│  │ BEAR     │ trending down        │        │
│  │ NEUTRAL  │ range-bound          │        │
│  │ HIGH_VOL │ elevated volatility  │        │
│  │ CRISIS   │ extreme stress       │        │
│  └─────────────────────────────────┘        │
│                                             │
│  Actions on regime change:                   │
│  ├── AlertSystem.alert()                     │
│  ├── CapitalAllocator.adjust()               │
│  └── Risk thresholds updated                 │
└───────────────────────────────────────────┘
```

---

## 6. BACKTESTING WORKFLOW

```
Historical Data (CSV / Parquet / TimescaleDB)
  │
  ▼
┌───────────────────────────────────────┐
│  1. Load historical OHLCV data         │
│  2. Select strategy:                    │
│     ├── Momentum Strategy               │
│     ├── Quantum Signal Strategy          │
│     └── Custom strategy via config      │
│  3. Configure backtest:                  │
│     ├── Date range                       │
│     ├── Initial capital                  │
│     ├── Commission model                 │
│     ├── Slippage model                   │
│     └── Regime filter                    │
│  4. Run via BacktestEngine               │
│     ├── EventDrivenBacktester            │
│     ├── WalkForwardAnalyzer              │
│     └── RegimeAwareBacktester            │
│  5. Transaction cost model applied       │
│  6. Generate results:                    │
│     ├── Equity curve                     │
│     ├── Sharpe ratio, Sortino            │
│     ├── Max drawdown                     │
│     ├── Win rate, profit factor          │
│     ├── Regime-specific performance      │
│     └── Trade-by-trade analysis          │
│  7. Performance Attribution:             │
│     ├── PnLAnalyzer                      │
│     ├── RiskAttribution                  │
│     └── Factor decomposition             │
└───────────────────────────────────────┘
```

---

## 7. ALPHA RESEARCH WORKFLOW

```
Hypothesis: "Cross-asset momentum alpha decays after 3 days"
  │
  ▼
┌───────────────────────────────────────┐
│  1. AlphaDiscovery                     │
│     └── Systematic alpha search        │
│  2. AlphaValidator                     │
│     ├── Statistical significance test  │
│     ├── Out-of-sample validation       │
│     └── Regime robustness check        │
│  3. AlphaCombiner                      │
│     └── Combine validated alphas       │
│  4. AlphaDecayStudy                    │
│     └── Half-life, decay rate analysis │
│  5. AlphaResearchScheduler             │
│     └── Scheduled refresh + monitoring │
│  6. AlphaStore                         │
│     └── Persist validated alphas       │
│  7. AlphaCrowdingDetector              │
│     └── Monitor for crowded positions  │
└───────────────────────────────────────┘
```

---

## 8. LLM RESEARCH TRACK WORKFLOW (Read-Only)

```
                ┌──────────────────────────────────────┐
                │     LIVE MARKET DATA (read-only)      │
                │  prices, volumes, regime, signals     │
                └──────────────┬───────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   DuckDB Cache      │
                    │   (TradingDataCache) │
                    │   Analytical queries │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Integration Bridge  │
                    │  (READ-ONLY)        │
                    │  ZERO exec authority │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Vector Store      │
                    │   (embeddings)      │
                    │   RAG retrieval     │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │   LLM Engine        │
                    │   (QuantumForgeLLM) │
                    │   Inference only    │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  LLMOutputParser    │
                    │  Type-safe parsing  │
                    │  Explanation only   │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Research Output     │
                    │  ├── Market insights │
                    │  ├── Regime context  │
                    │  └── Strategy notes  │
                    │                      │
                    │  *** CANNOT ***      │
                    │  place orders        │
                    │  modify positions    │
                    │  change risk params  │
                    └─────────────────────┘
```

---

## 9. DASHBOARD WORKFLOW

```
Browser → http://localhost:8501
  │
  ▼
app.py (Streamlit Multi-Page)
  │
  ├── 1_Main_Dashboard.py
  │   └── System overview, P&L summary, regime status
  │
  ├── 2_Trading_Dashboard.py
  │   └── Live orders, position manager, signal visualization
  │
  ├── 3_Risk_Dashboard.py
  │   └── VaR display, drawdown charts, exposure heatmap
  │
  ├── 4_Portfolio_Dashboard.py
  │   └── Holdings, allocation, rebalancing
  │
  ├── 5_Analytics_Dashboard.py
  │   └── Backtesting results, Sharpe, attribution
  │
  ├── 6_Research_Dashboard.py
  │   └── Alpha research, factor analysis, regime study
  │
  ├── 7_Execution_Dashboard.py
  │   └── Order flow, fill quality, algorithm comparison
  │
  ├── 8_Market_Microstructure.py
  │   └── Orderbook visualization, liquidity, toxicity
  │
  ├── 9_Configuration.py
  │   └── System parameters, API keys, strategy settings
  │
  └── 10_Investor_Portal.py
      └── NAV curve, risk report, investor-grade analytics
```

---

## 10. DEPLOYMENT WORKFLOW

```
1. CONFIGURATION
   ├── Edit config/ YAML files (symbols, API keys, thresholds)
   ├── Set EXECUTION_MODE (PAPER or LIVE)
   └── Set enable_ml, enable_llm flags

2. INFRASTRUCTURE
   ├── docker-compose up -d
   │   ├── TimescaleDB : port 5432
   │   ├── Redis       : port 6379
   │   └── Grafana     : port 3000
   └── Verify connectivity

3. STARTUP (choose one)
   ├── Full system:    python run_full_system.py
   ├── Pipeline only:  python launch_pipeline.py
   ├── Core only:      python launch_quantum_core.py --symbols BTCUSDT,ETHUSDT
   ├── Dashboard only: streamlit run app.py
   └── Visual launch:  python launch.py

4. MONITORING
   ├── Streamlit dashboards (10 pages)
   ├── Grafana (Docker)
   ├── Prometheus metrics server
   ├── HealthMonitor heartbeats
   ├── AlertSystem notifications
   └── Hash-chained audit log (JSONL)

5. GRACEFUL SHUTDOWN
   ├── Ctrl+C or SIGTERM
   ├── State auto-saved
   ├── WebSocket connections closed
   ├── Storage buffers flushed
   └── Final stats logged
```

---

## 11. STATE MANAGEMENT WORKFLOW

```
During Operation:
  │
  ├── Every 50 trade iterations: _save_state()
  │   ├── positions (dict)
  │   ├── portfolio_value (float)
  │   ├── trade_count (int)
  │   ├── regime per symbol
  │   └── Serialized to disk (JSON/pickle)
  │
  ├── On shutdown: _save_state()
  │
  └── On startup: _restore_state()
      ├── Load persisted state
      ├── Resume from last known positions
      └── Continue trading seamlessly
```

---

## 12. FEEDBACK & ADAPTATION WORKFLOW

```
After Each Trade Cycle:
  │
  ├── Per-symbol return tracking
  │   └── Track recent trade P&L
  │
  ├── ML Ensemble Weight Adaptation
  │   ├── Track which models predicted correctly
  │   ├── Increase weight of accurate models
  │   └── Decrease weight of poor models
  │
  ├── SVM Online Learning (every 100 ticks)
  │   └── Update hyperplane with new data
  │
  ├── EVT Tail Risk (periodic)
  │   └── Extreme Value Theory re-estimation
  │
  ├── Copula Dependence (periodic)
  │   └── Cross-asset correlation update
  │
  └── Shadow Strategy Comparison (periodic)
      ├── Compare live strategy vs shadow strategies
      ├── Track which would have performed better
      └── Inform strategy selection
```

---

*Document auto-generated from codebase analysis of QUANTUM-FORGE/*
