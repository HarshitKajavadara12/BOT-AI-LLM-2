# QUANTUM-FORGE — EXPECTED RESULTS DOCUMENT

## Complete Reference: What Happens When You Run This System

**Document Version:** 1.0  
**System:** QUANTUM-FORGE v2.0.0 — Institutional-Grade Cryptocurrency Quantitative Trading Platform  
**Architecture:** AI/ML-Driven Multi-Symbol Crypto Trading with LLM Research Integration  
**Default Mode:** Paper Trading with $100,000 Simulated Capital  
**Exchange:** Binance (7 Cryptocurrency Pairs, 24/7)  
**Total Modules:** 135+  

---

## TABLE OF CONTENTS

1. [System Overview](#1-system-overview)
2. [Prerequisites & Environment Setup](#2-prerequisites--environment-setup)
3. [Startup Sequence — Terminal Output](#3-startup-sequence--terminal-output)
4. [Boot Phase — Full System Initialization](#4-boot-phase--full-system-initialization)
5. [Quantum Core Orchestrator — The Brain](#5-quantum-core-orchestrator--the-brain)
6. [Live Trading Loop — Per-Symbol Pipeline](#6-live-trading-loop--per-symbol-pipeline)
7. [Signal Generation — 7-Source Mathematical Fusion](#7-signal-generation--7-source-mathematical-fusion)
8. [ML Ensemble — 9+ Model Predictions](#8-ml-ensemble--9-model-predictions)
9. [Feature Pipeline — 32 Engineered Features](#9-feature-pipeline--32-engineered-features)
10. [Regime Detection — 5-Detector Consensus](#10-regime-detection--5-detector-consensus)
11. [Signal Fusion — Triple-Source Blending](#11-signal-fusion--triple-source-blending)
12. [Risk Gate — 6-Check Cascade](#12-risk-gate--6-check-cascade)
13. [Circuit Breaker System](#13-circuit-breaker-system)
14. [Capital Allocation — Regime-Adaptive](#14-capital-allocation--regime-adaptive)
15. [Execution Engine — Professional Algorithms](#15-execution-engine--professional-algorithms)
16. [Cross-Asset Alpha Engine](#16-cross-asset-alpha-engine)
17. [Shadow Strategy Tracking](#17-shadow-strategy-tracking)
18. [Periodic Analysis — Portfolio Summary](#18-periodic-analysis--portfolio-summary)
19. [Streamlit Dashboard — 10 Interactive Pages](#19-streamlit-dashboard--10-interactive-pages)
20. [LLM/RAG Integration — AI Research Assistant](#20-llmrag-integration--ai-research-assistant)
21. [Prometheus Metrics & FastAPI Endpoints](#21-prometheus-metrics--fastapi-endpoints)
22. [Alert System — Telegram & Email Notifications](#22-alert-system--telegram--email-notifications)
23. [Health Monitoring System](#23-health-monitoring-system)
24. [Audit Trail — Hash-Chained JSONL](#24-audit-trail--hash-chained-jsonl)
25. [Log Files Generated](#25-log-files-generated)
26. [Database Records Created](#26-database-records-created)
27. [Backtesting Results](#27-backtesting-results)
28. [Advanced Risk Mathematics Output](#28-advanced-risk-mathematics-output)
29. [Shutdown Sequence & Final Statistics](#29-shutdown-sequence--final-statistics)
30. [Docker Deployment — Full Stack Output](#30-docker-deployment--full-stack-output)
31. [Paper Trading vs Live Trading Differences](#31-paper-trading-vs-live-trading-differences)
32. [Performance Benchmarks — Expected Numbers](#32-performance-benchmarks--expected-numbers)
33. [Error Scenarios & Fallback Behaviors](#33-error-scenarios--fallback-behaviors)
34. [Complete File Output Reference](#34-complete-file-output-reference)

---

## 1. SYSTEM OVERVIEW

QUANTUM-FORGE is an institutional-grade cryptocurrency quantitative trading platform designed specifically for Binance markets. It fuses mathematical signal generation (Fourier, Stochastic, Wavelets, Kalman), an ML ensemble (LSTM, Transformer, PPO, SAC, Gaussian Process), and cross-asset alpha into a unified trading pipeline running on a 2-second loop across 7 crypto pairs.

### Core Architecture
```
┌──────────────────────────────────────────────────────────────────────────┐
│                     QUANTUM-FORGE v2.0.0                                │
│                                                                          │
│   BINANCE WebSocket ──▶ DATA LAYER ──▶ PREPROCESSING ──▶ FEATURE STORE  │
│                              │                                           │
│           ┌──────────────────┼──────────────────┐                        │
│           ▼                  ▼                   ▼                        │
│   ┌──────────────┐  ┌──────────────┐   ┌──────────────────┐             │
│   │ MATH ENGINE  │  │ ML ENSEMBLE  │   │ CROSS-ASSET      │             │
│   │ 7 Sources    │  │ 9+ Models    │   │ ALPHA ENGINE     │             │
│   │ Weight: 50%  │  │ Weight: 30%  │   │ Weight: 20%      │             │
│   └──────┬───────┘  └──────┬───────┘   └────────┬─────────┘             │
│          └─────────────────┼────────────────────┘                        │
│                            ▼                                             │
│              ┌──────────────────────┐                                    │
│              │   SIGNAL FUSION      │── BUY / SELL / HOLD                │
│              └──────────┬───────────┘                                    │
│                         ▼                                                │
│              ┌──────────────────────┐                                    │
│              │   RISK GATE (6)      │── Block or Approve                 │
│              │ + Circuit Breaker    │                                    │
│              │ + Portfolio Risk Mgr │                                    │
│              └──────────┬───────────┘                                    │
│                         ▼                                                │
│              ┌──────────────────────┐                                    │
│              │   EXECUTION ENGINE   │── VWAP / TWAP / IS / MARKET        │
│              │   + Market Impact    │                                    │
│              └──────────┬───────────┘                                    │
│                         ▼                                                │
│              ┌──────────────────────┐                                    │
│              │   AUDIT + FEEDBACK   │── Hash-chain + ML adaptation       │
│              └──────────────────────┘                                    │
│                                                                          │
│  PARALLEL: LLM/RAG (Read-Only) │ 10 Dashboards │ Prometheus │ Alerts   │
├──────────────────────────────────────────────────────────────────────────┤
│  7 Pairs: BTCUSDT │ ETHUSDT │ BNBUSDT │ SOLUSDT │ ADAUSDT │ DOGEUSDT  │
│           XRPUSDT │  Capital: $100,000 │  Loop: 2s │  Mode: PAPER      │
└──────────────────────────────────────────────────────────────────────────┘
```

### Key Design Principles
- **LLM has ZERO execution authority** — read-only research track
- **System works fully with `LLM_ENABLED=false`** — LLM crash does not affect trading
- **Authority flows downward only** — Deterministic Core controls execution; Cognitive Layer observes
- **Graceful degradation** — every external dependency has in-memory fallback

---

## 2. PREREQUISITES & ENVIRONMENT SETUP

### Required Software
| Component | Version | Purpose |
|-----------|---------|---------|
| Python | 3.8+ | Core runtime |
| PostgreSQL + TimescaleDB | 15+ | Time-series storage |
| Redis | 7.0+ | Three-tier caching (<100µs hot ticks) |
| Docker | 20.10+ | Full stack deployment |
| R | 4.0+ (optional) | Advanced statistical models |
| Qdrant | 1.0+ (optional) | Vector similarity search for LLM/RAG |

### Key Python Dependencies
```
torch >= 2.0              # Deep learning (LSTM, GRU, Transformer)
numpy >= 1.24             # Numerical computing
pandas >= 2.0             # Data manipulation
scipy >= 1.11             # Scientific computing
scikit-learn >= 1.3       # ML models (SVM, Random Forest)
statsmodels >= 0.14       # Statistical models
streamlit >= 1.28         # Dashboard UI (10 pages)
fastapi >= 0.100          # REST API + Prometheus metrics
websockets >= 12.0        # Binance WebSocket
prometheus-client >= 0.17 # Metrics export
uvicorn >= 0.23           # ASGI server
hmmlearn >= 0.3           # Hidden Markov Models
llama-cpp-python >= 0.2   # Local LLM inference (optional)
qdrant-client >= 1.6      # Vector store (optional)
sentence-transformers     # Embeddings (optional)
```

### Installation Output
```bash
$ pip install -r requirements.txt

# Expected:
Successfully installed torch-2.x numpy-1.24.x pandas-2.x scipy-1.11.x ...
Installing collected packages: 60+ packages
Successfully installed 60+ packages

$ pip install -e .
# Expected: Successfully installed quantum-forge-2.0.0
```

### Environment Variables (.env.example)
```bash
# Binance API (required for live trading, optional for paper)
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret

# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=quantum_forge
DB_USER=quantum_user
DB_PASSWORD=your_password

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# LLM (optional)
LLM_ENABLED=false
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Alerts (optional)
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_CHAT_ID=your_chat_id
ALERT_EMAIL_FROM=alerts@domain.com
ALERT_EMAIL_TO=admin@domain.com
ALERT_SMTP_HOST=smtp.gmail.com
```

---

## 3. STARTUP SEQUENCE — TERMINAL OUTPUT

### Three Ways to Launch

```bash
# Option 1: Full System (all 135+ modules + UI launcher)
python launch.py

# Option 2: Direct Quantum Core (lightweight, recommended)
python launch_quantum_core.py --symbols BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT --capital 100000

# Option 3: Pipeline mode
python launch_pipeline.py

# Option 4: Full system with all modules validated
python run_full_system.py
```

### Command Line Arguments (launch_quantum_core.py)
| Argument | Default | Description |
|----------|---------|-------------|
| `--symbols` | `BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT` | Comma-separated trading pairs |
| `--capital` | `100000.0` | Initial capital in USD |
| `--no-ml` | `false` | Disable ML ensemble (math-only mode) |
| `--llm` | `false` | Enable LLM integration (read-only) |
| `--threshold` | `0.25` | Signal strength threshold (0.0-1.0) |

### launch.py Banner Output
```
═══════════════════════════════════════════════════════════════════════
 SYSTEM STATUS - 100% ACTIVATION
═══════════════════════════════════════════════════════════════════════

    Data Layer          [                                        ]  17 files
        Real-time ingestion, storage, preprocessing

    Intelligence        [                                        ]  24 files
        20+ AI/ML models (LSTM, GRU, SAC, etc.)

    Analytics           [                                        ]  24 files
        Backtesting, attribution, regime detection

    Execution           [                                        ]  18 files
        VWAP, TWAP, smart routing, HFT

    Core Math           [                                        ]  17 files
        Stochastic calculus, Fourier, microstructure

    Risk Management     [                                        ]   1 files
        EVT, copulas, optimal stopping

    Interface           [                                        ]  24 files
        9 dashboards (multi-page app)

    Infrastructure      [                                        ]   6 files
        Monitoring, backup, deployment

──────────────────────────────────────────────────────────────────────
  TOTAL: 131 files active (100% of system)
═══════════════════════════════════════════════════════════════════════

═══════════════════════════════════════════════════════════════════════
 KEY FEATURES ACTIVATED
═══════════════════════════════════════════════════════════════════════

  9 Interactive Dashboards (Streamlit multi-page app)
  20+ AI Models running in ensemble
  Professional execution algorithms (VWAP, TWAP, IS)
  Complete data infrastructure (TimescaleDB, Redis, Parquet)
  Market regime detection (HMM, changepoint)
  Advanced risk mathematics (EVT, Copulas)
  Comprehensive backtesting framework
  Performance attribution & P&L decomposition
  High-frequency trading infrastructure
  Multi-level risk management system

═══════════════════════════════════════════════════════════════════════

═══════════════════════════════════════════════════════════════════════
 AVAILABLE DASHBOARDS
═══════════════════════════════════════════════════════════════════════

  📊  Main Dashboard             - Unified system overview
  📈  Trading Dashboard          - Order execution & management
  🛡️  Risk Dashboard             - Risk analytics & monitoring
  💼  Portfolio Dashboard         - Holdings & performance tracking
  🔬  Analytics Dashboard        - Backtesting & attribution
  🧪  Research Dashboard         - Strategy development tools
  ⚡  Execution Dashboard        - Order flow analysis
  🔍  Market Microstructure      - Orderbook visualization
  ⚙️  Configuration              - System settings & parameters

═══════════════════════════════════════════════════════════════════════
```

---

## 4. BOOT PHASE — FULL SYSTEM INITIALIZATION

When `run_full_system.py` is launched, the `QuantumForgeFullSystem` class initializes 7 subsystem layers.

### Boot Sequence Output
```
════════════════════════════════════════════════════════════════════════════════
INITIALIZING QUANTUM-FORGE FULL SYSTEM - ALL MODULES
════════════════════════════════════════════════════════════════════════════════

[INIT] Initializing Data Layer (Storage + Cache + Processing)...
  [OK] Redis cache connected
  [OK] TimescaleDB connected
  [OK] Parquet writer: data/parquet
  [OK] Feature store initialized
  [OK] Data cleaner ready
  [OK] Normalizer ready
  [OK] Time-series aligner ready
[OK] Data Layer Ready

[INIT] Initializing Intelligence Layer (All ML Models)...
  [OK] LSTM model: input=10, hidden=64, layers=2
  [OK] GRU model: input=10, hidden=64, layers=2
  [OK] Transformer: input=10, hidden=64, heads=8, layers=4
  [OK] GNN: input=10, hidden=64 (or None if torch_geometric missing)
  [OK] TCN: input=10, output=1
  [OK] PPO agent: state=10, actions=3
  [OK] SAC agent: state=10, actions=3
  [OK] Model-based RL: MBPO
  [OK] Autoencoder feature learner
  [OK] Representation learner
  [OK] Manifold learner (or None if umap missing)
  [OK] Causal discovery ensemble
  [OK] MAML meta-learner
  [OK] Few-shot learner
  [OK] Transfer learner
  [OK] Ensemble distillation
  [OK] Gaussian Process regressor
  [OK] Bayesian optimizer
  [OK] Variational inference
  [OK] Conformal predictor
[OK] Intelligence Layer Ready (20+ Models)

[INIT] Initializing Analytics Layer (Backtesting + Attribution + Research)...
  [OK] Backtest engine
  [OK] Event-driven backtester
  [OK] Regime-aware backtester
  [OK] Transaction cost model
  [OK] Walk-forward optimizer
  [OK] Performance analyzer
  [OK] P&L decomposer
  [OK] Risk-adjusted metrics calculator
  [OK] Sharpe calculator
  [OK] Drawdown analyzer
  [OK] Risk attributor
  [OK] Market regime detector (HMM + 5 detectors)
  [OK] Alpha discovery + validation + combination + decay
  [OK] Factor construction + selection + combination + decay
[OK] Analytics Layer Ready (30+ Tools)

[INIT] Initializing Risk Layer (Advanced Mathematics)...
  [OK] Portfolio risk manager
  [OK] Risk limit: max_position_weight = 25%
  [OK] Risk limit: max_leverage = 2.0x
  [OK] EVT analyzer (Extreme Value Theory)
  [OK] Copula model (dependence analysis)
  [OK] Drawdown mathematics
  [OK] Optimal stopping theory
[OK] Risk Layer Ready

[INIT] Initializing Execution Layer (Pro Algorithms)...
  [OK] Order Management System (OMS)
  [OK] Smart Order Router (4 venue types)
  [OK] Position Manager
  [OK] Fill Manager
  [OK] Optimal execution engine
  [OK] TWAP algorithm
  [OK] VWAP algorithm
  [OK] Arrival Price algorithm
  [OK] Implementation Shortfall algorithm
  [OK] Adaptive execution engine
  [OK] Market impact model (Almgren-Chriss)
  [OK] HFT executor
  [OK] Latency optimizer
  [OK] Ultra-low-latency router
[OK] Execution Layer Ready

[INIT] Initializing Core Mathematics Engine...
  [OK] Mathematical engine (master)
  [OK] Stochastic calculus (OU, GBM, Jump-Diffusion)
  [OK] Fourier analyzer (FFT spectral)
  [OK] Kalman filter / signal processor
  [OK] Model Predictive Control
  [OK] Numerical optimizer
  [OK] Statistical tests (ADF, KPSS)
  [OK] Linear algebra engine
  [OK] Order book dynamics
  [OK] Price formation model
  [OK] Liquidity model
  [OK] Toxicity detector
[OK] Mathematics Engine Ready

[INIT] Initializing Infrastructure Layer...
  [OK] System monitor (5s interval, CPU/Memory/Disk thresholds)
  [OK] Backup manager (30-day retention, SHA-256 checksums)
  [OK] Performance benchmark suite
  [OK] Deployment manager
[OK] Infrastructure Ready

[OK] FULL SYSTEM INITIALIZED - 100% MODULES ACTIVE
```

### Boot Fallback Behavior
| Component | If Unavailable | Fallback |
|-----------|----------------|----------|
| Redis | Connection refused | In-memory dict (non-persistent) |
| TimescaleDB | Connection failed | SQLite / in-memory |
| Qdrant | Not running | In-memory vector store |
| torch_geometric (GNN) | Not installed | GNN = None, skipped |
| umap (ManifoldLearner) | Not installed | ManifoldLearner = None, skipped |
| Binance WebSocket | Network error | REST API fallback (`api.binance.com/api/v3/ticker/price`) |
| R runtime | Not installed | Python statsmodels |
| Llama model file | Not present | Template-based responses |
| Telegram/Email | No tokens configured | Alerts logged to console only |

---

## 5. QUANTUM CORE ORCHESTRATOR — THE BRAIN

The `QuantumCoreOrchestrator` is the actual trading engine. It is the only component with execution authority.

### Internal Module Initialization
```
[INIT] Starting REAL Quantum Core Orchestrator...

  Symbols:          BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT, XRPUSDT
  Initial Capital:  $100,000.00
  ML Enabled:       YES (9+ models)
  LLM Enabled:      NO (read-only, off by default)
  Signal Threshold: 0.25

  Module Initialization:
  ┌───────────────────────────────────┬────────────────────────────────────┐
  │ Module                            │ Key Parameters                     │
  ├───────────────────────────────────┼────────────────────────────────────┤
  │ SignalGenerator                   │ min_history=30, threshold=0.25     │
  │ MLEnsembleEngine                  │ feature_dim=20, 9 models           │
  │ FeaturePipeline                   │ lookback=60, output_dim=32         │
  │ RegimeDetector                    │ window=60, vol_high=3%, extreme=5% │
  │ RiskGate                          │ max_pos=10%, max_exp=80%, DD=15%   │
  │ CapitalAllocator                  │ method=HYBRID, capital=$100K       │
  │ ExecutionManager                  │ mode=PAPER, fee=0.1%               │
  │ CircuitBreaker                    │ max_fails=5, cooldown=60s          │
  │ CrossAssetAlphaEngine             │ 5 symbols                          │
  │ SVMRegimeClassifier               │ online learning                    │
  │ CausalBridge                      │ lookback=200, update every 50      │
  │ OrderBookAnalyzer                 │ depth=20 levels, window=200        │
  │ GPPredictionBridge                │ feature_dim=32, max_points=500     │
  │ MarketImpactTracker               │ settle=5 bars, expected=2.0 bps    │
  │ VariationalInferenceBridge        │ input=32, hidden=[64,32]           │
  │ PortfolioRiskManager              │ max_pos=$10K, max_DD=15%           │
  │ AlertSystem                       │ rate_limit: 10/min, 5s min gap     │
  │ StorageCoordinator                │ Parquet + Redis + TimescaleDB      │
  │ AlphaResearchScheduler            │ run every 4 hours                  │
  │ ShadowTracker + Multiplexer       │ shadow strategies $100K virtual    │
  └───────────────────────────────────┴────────────────────────────────────┘

  Logging: logs/quantum_core.log (10 MB × 5 rotations)
  Format:  %(asctime)s | %(name)-18s | %(levelname)-7s | %(message)s
```

### System State Machine
```
    ┌─────────────┐
    │  STARTING   │
    └──────┬──────┘
           │ Modules initialized
    ┌──────▼──────┐
    │  RUNNING    │ ← Main loop (2s per iteration)
    └──────┬──────┘
           │
    ┌──────▼──────┐      Circuit Breaker
    │  DEGRADED   │ ◄─── or component failure
    └──────┬──────┘
           │ CRISIS regime or manual stop
    ┌──────▼──────┐
    │  STOPPED    │ ← Graceful shutdown (Ctrl+C / SIGTERM)
    └─────────────┘
```

---

## 6. LIVE TRADING LOOP — PER-SYMBOL PIPELINE

The main loop runs every **2 seconds**, iterating over all 5-7 symbols.

### 10-Step Pipeline Per Symbol
```
[TICK] Symbol: BTCUSDT ──────────────────────────────────────────

  STEP 1:  FETCH DATA
           → WebSocket price: $67,234.50
           → REST fallback if WS unavailable
           → OHLCV: O=$67,200 H=$67,280 L=$67,180 C=$67,234 V=12.5

  STEP 2:  SIGNAL GENERATOR (7 math sources)
           → Fourier: +0.23 (dominant period: 45 bars)
           → Wavelet: +0.15 (multi-scale trend agreement)
           → Stochastic: -0.08 (OU process: mean-reverting)
           → Momentum: +0.31 (ROC at 5/10/20/50)
           → Mean Reversion: -0.12 (z-score: -0.67, Hurst: 0.42)
           → Volatility: +0.05 (short/long vol ratio: 0.85)
           → Microstructure: +0.18 (near support level)
           → Fused math signal: +0.21 (weighted sum)
           → Signal: BUY (strength: 0.21)

  STEP 3:  REGIME DETECTION
           → Volatility regime: NORMAL (EWMA vol: 2.1%)
           → Trend regime: BULL (momentum agreement: 0.73)
           → HMM regime: NORMAL (state 1)
           → Drawdown: -0.8%
           → Consensus: BULL (confidence: 0.80)

  STEP 4:  ML ENSEMBLE (9 models)
           → Feature extraction: 32 features → 20 used
           → LSTM: +0.12 (latency: 3ms)
           → GRU: +0.08 (latency: 2ms)
           → Transformer: +0.18 (latency: 5ms)
           → TCN: +0.15 (latency: 3ms)
           → PPO: action=BUY (latency: 1ms)
           → SAC: action=BUY (latency: 1ms)
           → GP: +0.09 (latency: 4ms)
           → Linear Momentum: +0.22 (latency: <1ms)
           → Vol Predictor: -0.05 (latency: <1ms)
           → Consensus: 0.78 (7/9 agree on BUY)
           → Ensemble signal: BUY (strength: 0.28)

  STEP 5:  SIGNAL FUSION
           → Math: +0.21 × 50% = +0.105
           → ML:   +0.28 × 30% = +0.084
           → Cross: +0.15 × 20% = +0.030
           → Fused: +0.219
           → Direction: BUY (> 0.20 threshold)

  STEP 6:  RISK GATE (6 checks)
           ✅ CRISIS check: regime=BULL (not CRISIS)
           ✅ HIGH_VOL check: regime=BULL (skip vol gate)
           ✅ Drawdown check: -0.8% (< 15% limit)
           ✅ Position size: 3.2% (< 10% limit)
           ✅ Total exposure: 45% (< 80% limit)
           ✅ Signal floor: 0.219 > 0.30 floor? → adjusted for BULL
           → APPROVED

  STEP 7:  EXECUTION
           → Algorithm selected: TWAP (order < $5K, low urgency)
           → Position size: $100,000 × 5% × 0.219 = $1,095
           → BUY 0.0163 BTC @ $67,234.50
           → Slippage: 1.8 bps (TWAP range: 1-3 bps)
           → Fee: $1.10 (0.1% Binance)
           → Fill price: $67,246.60

  STEP 8:  POST-TRADE
           → Market impact recorded: 1.8 bps
           → Trade alert sent: 🟢 BUY BTCUSDT 0.0163 @ $67,246.60

  STEP 9:  AUDIT LOG
           → Hash-chained JSONL entry written
           → Contains: market_state + signal_state + risk_state + decision

  STEP 10: FEEDBACK
           → ML ensemble weight update (direction correct → +1.0)
           → Per-symbol return tracked
           → Storage: signal + trade saved to Parquet/Redis/TimescaleDB
```

### Per-Symbol Log Output (Every 2 Seconds)
```
2026-02-27 14:30:22.345 | QF_CORE            | INFO    | [BTCUSDT] $67,234.50 | Math: BUY 0.21 | ML: BUY 0.28 | Fused: BUY 0.22 | Regime: BULL
2026-02-27 14:30:22.456 | QF_CORE            | INFO    | [BTCUSDT] TRADE EXECUTED: BUY 0.0163 @ $67,246.60 | fee=$1.10 | slip=1.8bps | algo=TWAP
2026-02-27 14:30:24.567 | QF_CORE            | INFO    | [ETHUSDT] $3,456.78 | Math: HOLD 0.08 | ML: HOLD 0.05 | Fused: HOLD 0.06 | Regime: BULL
2026-02-27 14:30:26.789 | QF_CORE            | INFO    | [BNBUSDT] $312.45 | Math: SELL -0.34 | ML: SELL -0.22 | Fused: SELL -0.28 | Regime: NEUTRAL
```

Note: HOLD signals are **not logged** by default to reduce noise — only BUY/SELL signals appear.

---

## 7. SIGNAL GENERATION — 7-SOURCE MATHEMATICAL FUSION

### Signal Sources and Weights
| Source | Weight | Algorithm | Output Range |
|--------|--------|-----------|-------------|
| **Fourier** | 15% | FFT spectral analysis → dominant frequency/period → phase → trough=BUY/peak=SELL | [-1, +1] |
| **Wavelet** | 15% | Multi-scale MA decomposition (scales 4,8,16,32) → trend agreement | [-1, +1] |
| **Stochastic** | 20% | OU process fit via OLS → if mean-reverting: z-score from long-term mean | [-1, +1] |
| **Momentum** | 15% | ROC at lookbacks [5,10,20,50], exponentially weighted (recent > older) | [-1, +1] |
| **Mean Reversion** | 15% | Z-score (window=20), boosted if Hurst < 0.45 (×1.5) | [-1, +1] |
| **Volatility** | 10% | short_vol/long_vol ratio: >2.0 → -0.8, >1.5 → -0.4, <0.5 → +0.3 | [-1, +1] |
| **Microstructure** | 10% | Local support/resistance detection (30-bar), proximity scoring | [-1, +1] |

### Signal Output Structure
```python
QuantumSignal:
  symbol:                "BTCUSDT"
  signal_type:           BUY | SELL | HOLD
  strength:              0.34          # [0-1], absolute magnitude
  timestamp:             2026-02-27 14:30:22
  sources:               {fourier: 0.23, wavelet: 0.15, stochastic: -0.08, ...}
  fourier_dominant_period: 45          # bars
  wavelet_trend:         +0.15
  stochastic_drift:      +0.0012
  stochastic_vol:        0.023
  mean_reversion_speed:  0.045
  stationarity_pvalue:   0.03         # ADF test
  regime:                "BULL"
  regime_confidence:     0.80
```

### Signal Threshold Logic
- `fused_signal > +signal_threshold` → **BUY**
- `fused_signal < -signal_threshold` → **SELL**
- otherwise → **HOLD**
- Default `signal_threshold = 0.3` (signal generator) / `0.25` (quantum core)
- **Non-stationary adjustment:** if ADF p-value > 0.05 → strength × 0.7

### Minimum History Requirement
- **50 ticks** minimum before first signal generated (configurable, 30 in quantum core)
- Prices stored in `deque(maxlen=500)`

---

## 8. ML ENSEMBLE — 9+ MODEL PREDICTIONS

### Models and Architecture
| Model | Type | Architecture | Output |
|-------|------|-------------|--------|
| **LSTM** | Deep Learning | input=20, hidden=64, layers=2 | tanh(pred) ∈ [-1,1] |
| **GRU** | Deep Learning | input=20, hidden=64, layers=2 | tanh(pred) ∈ [-1,1] |
| **Transformer** | Deep Learning | input=20, hidden=64, heads=4, layers=2 | tanh(pred) ∈ [-1,1] |
| **TCN** | Deep Learning | input_channels=20, output=1 | tanh(pred) ∈ [-1,1] |
| **PPO** | Reinforcement Learning | state=20, actions=3 (BUY/HOLD/SELL) | argmax - 1 |
| **SAC** | Reinforcement Learning | state=20, actions=3 | argmax - 1 |
| **Gaussian Process** | Probabilistic | non-parametric | prediction ∈ ℝ |
| **Linear Momentum** | Statistical | Exponentially-weighted recent returns × 10 | ∈ [-1,1] |
| **Vol Predictor** | Statistical | short/long vol ratio → -(ratio-1)×2 | ∈ [-1,1] |

### Ensemble Prediction Output
```python
EnsemblePrediction:
  signal:         "BUY"             # BUY / SELL / HOLD
  strength:       0.28              # [0-1]
  consensus:      0.78              # fraction of models agreeing
  predictions:    {lstm: 0.12, gru: 0.08, transformer: 0.18, ...}
  weights:        {lstm: 0.15, gru: 0.12, transformer: 0.18, ...}
  model_count:    9
  total_latency_ms: 20.3
```

### Model Weight Adaptation (Online Learning)
```
[ML ENSEMBLE] Weight Update After Trade:
  Actual return: +0.3%
  Scoring: correct direction = 1.0, wrong = 0.0, neutral = 0.5
  
  Updated weights:
    lstm:          0.142 → 0.155 (correct)
    transformer:   0.178 → 0.192 (correct)
    ppo:           0.112 → 0.098 (wrong)
    vol_predictor: 0.095 → 0.075 (wrong)
  
  Models below 30% accuracy get weight = 0
  If all models fail → reset to equal weights (1/N)
```

### Trained Model Loading
The ensemble checks `intelligence/trained_models/{name}.pt` for pretrained PyTorch state dicts at startup. If found, weights are loaded; otherwise, models run with random initialization.

---

## 9. FEATURE PIPELINE — 32 ENGINEERED FEATURES

### Feature Vector Breakdown
```
[FEATURE PIPELINE] Extracting 32 features for BTCUSDT:

  RETURNS-BASED (10 features):
  ┌────────────────────┬────────────────┐
  │ ret_mean_20        │ +0.000234      │
  │ ret_std_20         │ 0.0189         │
  │ ret_mean_5         │ +0.000567      │
  │ ret_std_5          │ 0.0145         │
  │ ret_skew_20        │ -0.34          │
  │ ret_kurt_20        │ 3.45           │
  │ ret_last           │ +0.0012        │
  │ ret_max_20         │ +0.045         │
  │ ret_min_20         │ -0.032         │
  │ ret_cum_20         │ +0.0047        │
  └────────────────────┴────────────────┘

  TECHNICAL (10 features):
  ┌────────────────────┬────────────────┐
  │ rsi_14             │ 58.3           │
  │ macd_line          │ +0.0023        │
  │ macd_signal        │ +0.0018        │
  │ macd_hist          │ +0.0005        │
  │ bollinger_pct_b    │ 0.62           │
  │ atr_14             │ 0.0234         │
  │ roc_5              │ +0.0089        │
  │ roc_10             │ +0.0167        │
  │ roc_20             │ +0.0234        │
  │ z_score_20         │ -0.45          │
  └────────────────────┴────────────────┘

  SPECTRAL (5 features):
  ┌────────────────────┬────────────────┐
  │ spec_dominant_freq │ 0.022          │  (≈ 45-bar cycle)
  │ spec_dominant_power│ 0.234          │
  │ spec_entropy       │ 3.45           │  (bits, log₂)
  │ spec_centroid      │ 0.089          │
  │ spec_bandwidth     │ 0.034          │
  └────────────────────┴────────────────┘

  MICROSTRUCTURE (5 features):
  ┌────────────────────┬────────────────┐
  │ micro_spread_proxy │ 0.0012         │  (Roll estimator)
  │ micro_autocorr_1   │ -0.08          │
  │ micro_price_cluster│ 0.45           │  (mod 10 clustering)
  │ micro_vol_ratio    │ 1.12           │
  │ micro_tick_dir     │ +0.23          │  (up/down imbalance)
  └────────────────────┴────────────────┘

  VOLUME (2 features):
  ┌────────────────────┬────────────────┐
  │ vol_ratio_5_20     │ 1.18           │
  │ vol_cv_20          │ 0.34           │  (coefficient of variation)
  └────────────────────┴────────────────┘

  Minimum data needed: 20 prices
  NaN replacement: 0.0
```

---

## 10. REGIME DETECTION — 5-DETECTOR CONSENSUS

### Detection Output
```
[REGIME] Detection for current market state:

  Detector 1 — VOLATILITY:
    EWMA vol: 2.1% (threshold: high=3%, extreme=5%)
    Short vol vs Medium vol: ratio = 0.92 (not rising)
    Result: NORMAL

  Detector 2 — TREND:
    Momentum (5-period):  +1.2%
    Momentum (20-period): +3.5%
    Momentum (60-period): +8.2%
    Agreement score: 0.73 (all positive, > 0.50)
    Result: BULL (momentum > 3% threshold)

  Detector 3 — HMM:
    3-state K-means on |returns|:
    State 0: LOW_VOL (|ret| < 0.01)
    State 1: NORMAL  (0.01 ≤ |ret| < 0.03)
    State 2: HIGH_VOL (|ret| ≥ 0.03)
    Current: State 1 (NORMAL_CONFIRMED)

  Detector 4 — DRAWDOWN:
    Peak capital: $100,234
    Current:      $99,432
    Drawdown:     -0.8%

  Detector 5 — EWMA VOLATILITY:
    GARCH-like: σ² = α × r² + (1-α) × σ²_prev
    α = 0.06, current σ = 2.1%

  ──────────────────────────────────
  CONSENSUS: BULL (confidence: 0.80)
  ──────────────────────────────────
```

### Regime Transition Rules
| Condition | Result | Confidence |
|-----------|--------|-----------|
| drawdown > 15% + EXTREME vol | **CRISIS** | 0.95 |
| drawdown > 20% | **CRISIS** | 0.90 |
| vol_danger + hmm_danger | **HIGH_VOLATILITY** | 0.90 |
| vol_danger only | **HIGH_VOLATILITY** | 0.70 |
| BEAR trend + drawdown > 5% | **BEAR** | 0.80 |
| BEAR trend | **BEAR** | 0.60 |
| BULL + normal/low vol | **BULL** | 0.80 |
| CHOPPY trend | **NEUTRAL** | 0.40 |
| Default | **NEUTRAL** | 0.50 |

### Regime Change Alert
```
[ALERT] 📊 REGIME CHANGE
  Old: NEUTRAL (confidence: 0.50)
  New: BULL (confidence: 0.80)
  → Capital allocator adjusting weights
  → Signal thresholds updated
```

### Smoothing
- **CRISIS** transitions are always **immediate** (no smoothing)
- All other transitions require **2 consecutive readings** to agree before switching

---

## 11. SIGNAL FUSION — TRIPLE-SOURCE BLENDING

### Fusion Weights
```
SIGNAL FUSION:
  Math signal:       50% weight (signal_generator.py — 7 sources)
  ML ensemble:       30% weight (ml_ensemble.py — 9 models)
  Cross-asset alpha: 20% weight (cross_asset_alpha.py + causal_bridge)

  Dynamic Reweighting:
    If ML unavailable:  Math 65% + CrossAsset 35%
    If CrossAsset unavailable: Math 60% + ML 40%
    If only Math available: Math 100%

  Final Fusion: weighted_sum → clip to [-1, +1]
  Threshold:    > +0.20 → BUY, < -0.20 → SELL, else HOLD
```

---

## 12. RISK GATE — 6-CHECK CASCADE

Every signal must pass through 6 sequential risk checks before execution.

### Gate Cascade
```
[RISK GATE] Evaluating signal: BUY BTCUSDT (strength: 0.34)

  Gate 1: CRISIS REGIME CHECK
    Current regime: BULL
    ✅ PASS (not CRISIS → proceed)

  Gate 2: HIGH VOLATILITY CHECK
    Current regime: BULL
    ✅ PASS (not HIGH_VOL → skip vol gate)
    [If HIGH_VOL: requires signal_strength > 0.70]

  Gate 3: DRAWDOWN CHECK
    Current drawdown: -0.8%
    Max allowed: 15%
    ✅ PASS (-0.8% < 15%)

  Gate 4: POSITION SIZE CHECK
    Current BTCUSDT exposure: $3,200 (3.2% of capital)
    Max allowed: 10% ($10,000)
    ✅ PASS (3.2% < 10%)
    [Only blocks BUY if at limit]

  Gate 5: TOTAL EXPOSURE CHECK
    Total positions: $45,000 (45% of capital)
    Max allowed: 80% ($80,000)
    ✅ PASS (45% < 80%)

  Gate 6: SIGNAL FLOOR CHECK
    Signal strength: 0.34
    Base floor: 0.30
    Regime adjustment: BEAR → floor raised to 0.50
    Current floor: 0.30 (BULL regime)
    ✅ PASS (0.34 > 0.30)

  ══════════════════════════════════
  RESULT: APPROVED ✅
  ══════════════════════════════════
```

### Risk Gate Block Example
```
[RISK GATE] ❌ BLOCKED: BUY SOLUSDT (strength: 0.28)
  Gate 6 FAILED: Signal floor = 0.50 (BEAR regime)
  Signal strength 0.28 < 0.50
  Reason: "Signal too weak for BEAR regime (need > 0.50)"

  Stats: total_signals=234, blocked=47, block_rate=20.1%
```

---

## 13. CIRCUIT BREAKER SYSTEM

### Per-Symbol Circuit Breaker
```
[CIRCUIT BREAKER] Status:
  ┌────────────┬──────────┬────────────┐
  │ Symbol     │ Failures │ Status     │
  ├────────────┼──────────┼────────────┤
  │ BTCUSDT    │ 0/5      │ 🟢 CLOSED │
  │ ETHUSDT    │ 2/5      │ 🟡 2 fails │
  │ BNBUSDT    │ 0/5      │ 🟢 CLOSED │
  │ SOLUSDT    │ 5/5      │ 🔴 OPEN   │ ← cooldown 60s
  │ XRPUSDT    │ 0/5      │ 🟢 CLOSED │
  └────────────┴──────────┴────────────┘
```

### Circuit Breaker Trigger
```
[CIRCUIT] SOLUSDT breaker OPEN after 5 consecutive failures
  → Symbol skipped for 60 seconds
  → After cooldown: half-open (reset count, allow one attempt)
  → Success resets breaker; failure re-opens
```

---

## 14. CAPITAL ALLOCATION — REGIME-ADAPTIVE

### Allocation Output
```
[CAPITAL ALLOCATOR] Current State:
  Method:         HYBRID (60% performance + 40% risk parity)
  Total Capital:  $100,000
  Current Regime: BULL (multiplier: 1.00)

  Strategy Weights:
  ┌──────────────────────┬────────┬──────────┬────────┬──────────┐
  │ Strategy             │ Weight │ Sharpe   │ DD     │ Vol      │
  ├──────────────────────┼────────┼──────────┼────────┼──────────┤
  │ quantum_signal_v1    │ 45.2%  │ 1.45     │ -2.3%  │ 12.4%   │
  │ momentum_strategy    │ 32.8%  │ 0.89     │ -4.1%  │ 18.2%   │
  │ (cash reserve)       │ 22.0%  │    —     │   —    │   —     │
  └──────────────────────┴────────┴──────────┴────────┴──────────┘

  Constraints: min 5% per strategy, max 40% per strategy
```

### Regime Multipliers
| Regime | Capital Multiplier | Effect |
|--------|-------------------|--------|
| BULL | 1.00 | Full allocation |
| NEUTRAL | 0.85 | 15% reduced |
| BEAR | 0.60 | 40% reduced |
| HIGH_VOLATILITY | 0.30 | 70% reduced |
| CRISIS | 0.00 | ALL positions halted |

---

## 15. EXECUTION ENGINE — PROFESSIONAL ALGORITHMS

### Algorithm Auto-Selection Rules
| Condition | Selected Algorithm |
|-----------|-------------------|
| `signal_strength > 0.8` | **MARKET** (immediate) |
| `order_value < $500` | **MARKET** (too small for algo) |
| `volatility > 5%` + IS available | **Implementation Shortfall** |
| `order_value > $5,000` + VWAP available | **VWAP** (minimize impact) |
| TWAP available | **TWAP** (default algo) |
| Fallback | **MARKET** |

### Paper Trading Slippage Model
| Algorithm | Simulated Slippage (bps) |
|-----------|-------------------------|
| MARKET | 2 - 5 |
| TWAP | 1 - 3 |
| VWAP | 0.5 - 2 |
| Implementation Shortfall | 0.5 - 1.5 |

### Trade Execution Output
```
[EXECUTION] Order Filled:
  Symbol:     BTCUSDT
  Side:       BUY
  Quantity:   0.0163 BTC
  Target:     $67,234.50
  Fill:       $67,246.60
  Slippage:   1.8 bps
  Fee:        $1.10 (0.1% Binance)
  Algorithm:  TWAP
  Status:     FILLED
  Mode:       PAPER

[EXECUTION] Stats:
  Total orders:      47
  Filled:            45
  Fill rate:         95.7%
  Total fees:        $52.30
  Avg slippage:      2.1 bps
  Algo distribution: {MARKET: 12, TWAP: 28, VWAP: 5, IS: 2}
```

### Live Trading Execution
```
[EXECUTION] LIVE MODE — Binance API:
  → HMAC-SHA256 signed request to api.binance.com/api/v3/order
  → Order type: MARKET (high urgency) or LIMIT (±0.2% tolerance, GTC)
  → recvWindow: 5000ms
  → Retry: up to 3 attempts, exponential backoff (2^n seconds)
  → Poll order status every 2s, timeout 30s
  → LOT_SIZE and PRICE_FILTER from exchangeInfo

[EXECUTION] Position Reconciliation:
  → Fetches /account balances from Binance
  → Compares with internal position tracker
  → Logs discrepancies
```

### Position Sizing
```
Position size = cash × 5% × min(signal_strength, 1.0)

Example: $100,000 × 5% × 0.34 = $1,700
Minimum trade: $10

BUY:  Opens or increases position
SELL: Sells up to 50% of existing position (risk management)
```

---

## 16. CROSS-ASSET ALPHA ENGINE

### Cross-Asset Signal Output
```
[CROSS-ASSET] Update:
  BTC-ETH correlation:   0.87 (strong positive)
  BTC-SOL correlation:   0.72
  BTC-BNB correlation:   0.65
  ETH-SOL correlation:   0.78

  Lead-lag relationships detected:
    BTC leads ETH by ~3 minutes (Granger causality p=0.02)
    ETH leads SOL by ~5 minutes (Granger causality p=0.04)

  Cross-asset alpha for ETHUSDT: +0.15
    → BTC momentum spill-over signal
```

### Causal Discovery Bridge
```
[CAUSAL] Causal boost for ETHUSDT:
  Discovery method: multivariate analysis (lookback=200)
  BTC→ETH causal strength: 0.23
  Blend: 70% cross-asset + 30% causal
  Final cross-asset signal: +0.18
  Update: every 50 iterations
```

---

## 17. SHADOW STRATEGY TRACKING

### Shadow Strategies Output
```
[SHADOW TRACKER] Performance Comparison:

  LIVE STRATEGIES:
  ┌────────────────────────┬──────────┬───────────┬─────────┐
  │ Strategy               │ Equity   │ Return    │ Trades  │
  ├────────────────────────┼──────────┼───────────┼─────────┤
  │ quantum_signal_v1      │ $102,345 │ +2.35%    │ 34      │
  │ momentum_strategy      │ $99,876  │ -0.12%    │ 18      │
  └────────────────────────┴──────────┴───────────┴─────────┘

  SHADOW STRATEGIES (virtual $100K):
  ┌────────────────────────┬──────────┬───────────┬─────────┐
  │ Strategy               │ Equity   │ Return    │ Trades  │
  ├────────────────────────┼──────────┼───────────┼─────────┤
  │ shadow_momentum_v2     │ $103,200 │ +3.20%    │ 42      │
  └────────────────────────┴──────────┴───────────┴─────────┘

  Promotion Criteria: return > 2%, trades > 20, max 1 promotion
```

### Shadow Promotion
```
[MULTIPLEXER] PROMOTED shadow→live: shadow_momentum_v2
  Return: +3.20% (> 2% threshold)
  Trades: 42 (> 20 minimum)
  → Now receiving real capital allocation
```

---

## 18. PERIODIC ANALYSIS — PORTFOLIO SUMMARY

Every **20 iterations** (~40 seconds), the system prints a comprehensive portfolio analysis.

### Periodic Analysis Output
```
════════════════════════════════════════════════════════════════════════
 PERIODIC ANALYSIS (Iteration 100)
════════════════════════════════════════════════════════════════════════

 PORTFOLIO:
  Value:      $100,456.78
  P&L:        +$456.78 (+0.46%)
  Cash:       $55,230.00
  Positions:  3 open
  Trades:     23 total
  Win Rate:   60.9% (14/23)

 RISK GATE STATS:
  Total signals:  234
  Blocked:        47
  Block rate:     20.1%
  Drawdown:       -0.8%
  Exposure:       44.8%

 ML ENSEMBLE:
  Active models: 9
  Weights: {lstm: 0.155, transformer: 0.192, gru: 0.112, tcn: 0.098, ...}

 EXECUTION STATS:
  Orders:       23
  Fill rate:    95.7%
  Avg slippage: 2.1 bps
  Total fees:   $25.30
  Algorithms:   {MARKET: 5, TWAP: 15, VWAP: 3}

 OPEN POSITIONS:
  ┌────────────┬──────────┬──────────┬───────────┬──────────┐
  │ Symbol     │ Entry    │ Current  │ Unrealized│ P&L %    │
  ├────────────┼──────────┼──────────┼───────────┼──────────┤
  │ BTCUSDT    │ $67,100  │ $67,250  │ +$48.90   │ +0.22%   │
  │ ETHUSDT    │ $3,420   │ $3,456   │ +$72.00   │ +1.05%   │
  │ SOLUSDT    │ $98.50   │ $97.20   │ -$32.50   │ -1.32%   │
  └────────────┴──────────┴──────────┴───────────┴──────────┘

 PORTFOLIO RISK:
  Status:       NORMAL
  Active alerts: 0
  VaR (95%):    $2,340

 ADVANCED ANALYTICS (if 50+ returns):
  EVT tail VaR(99%): $4,560
  Copula: cross-asset dependence updated

 SHADOW STRATEGIES:
  shadow_momentum_v2: equity=$103,200 (+3.20%)

 STORAGE:
  Signals saved:  234
  Trades saved:   23
  State saved:    auto-save at iteration 100

════════════════════════════════════════════════════════════════════════
```

---

## 19. STREAMLIT DASHBOARD — 10 INTERACTIVE PAGES

**Access:** `http://localhost:8501`  
**Auto-refresh:** Every **30 seconds**

### Landing Page
```
┌──────────────────────────────────────────────────────────┐
│  🏛️  QUANTUM-FORGE Trading Platform                     │
│  Complete Institutional-Grade Trading System             │
│                                                          │
│  ℹ️  Select a dashboard from the sidebar to begin       │
│                                                          │
│  System Status: ✅ ONLINE    All modules active          │
│  Dashboards:    9 Active     100% operational            │
│  Real-Time:     7 Symbols    Binance WebSocket           │
│                                                          │
│  ✅ Full system initialized - All 135+ modules active!   │
└──────────────────────────────────────────────────────────┘
```

### Sidebar (All Pages)
Every page shows a health check against `http://localhost:8000/health`:
- 🟢 **System Online** — API healthy
- 🟡 **Degraded** — some components unhealthy
- 🔴 **Status unavailable** — API unreachable

### Page 1: 📊 Main Dashboard
| Section | What You See |
|---------|-------------|
| System Overview | Portfolio value, P&L, cash, open positions count |
| Market Snapshot | Real-time prices for all 7 symbols with sparklines |
| Regime Indicator | Current market regime (BULL/BEAR/NEUTRAL/HIGH_VOL/CRISIS) with confidence |
| Signal Summary | Latest signals per symbol with strength bars |
| Performance Chart | Cumulative portfolio returns line chart |
| Key Metrics | Sharpe, Sortino, Max Drawdown, Win Rate gauges |

### Page 2: 📈 Trading Dashboard
| Section | What You See |
|---------|-------------|
| Live Orders | Active orders table: symbol, side, quantity, price, status |
| Trade History | All executed trades with P&L per trade |
| Order Entry | Manual order form (symbol, side, quantity, algo) |
| Execution Quality | Fill rate, avg slippage, algo distribution pie chart |
| Real-Time Feed | Streaming price updates (WebSocket) |

### Page 3: 🛡️ Risk Dashboard
| Section | What You See |
|---------|-------------|
| Risk Status | GREEN/YELLOW/RED/BLACK status cards |
| Risk Limits | Utilization bars for position, leverage, VaR, drawdown |
| VaR Analysis | Historical, Parametric, Monte Carlo VaR comparison |
| Stress Testing | Scenario results: BTC -25%, DOGE -40% crash |
| Drawdown Chart | Underwater drawdown area chart |
| Correlation Matrix | 7×7 heatmap of crypto pair correlations |

### Page 4: 💼 Portfolio Dashboard
| Section | What You See |
|---------|-------------|
| Holdings Table | All positions: symbol, quantity, entry, current, P&L, weight |
| Allocation Pie | Portfolio allocation donut chart |
| P&L Timeline | Daily P&L bar chart |
| Position History | Historical position changes |
| Cash Flow | Cash inflows/outflows from trades |

### Page 5: 🔬 Analytics Dashboard
| Section | What You See |
|---------|-------------|
| Backtest Results | Total return, Sharpe, max DD, win rate cards |
| Performance Chart | Equity curve with drawdown overlay |
| Monthly Heatmap | Monthly returns (Red-Yellow-Green) |
| Factor Attribution | Factor contribution breakdown |
| Walk-Forward | Out-of-sample performance comparison |

### Page 6: 🧪 Research Dashboard
| Section | What You See |
|---------|-------------|
| Alpha Discovery | Discovered alpha signals with IC and Sharpe |
| Alpha Decay | Decay curves showing signal half-life |
| Regime Analysis | HMM state diagram and transition matrix |
| Factor Research | Factor IC, turnover, capacity analysis |
| Strategy Lab | Parameter sensitivity heat maps |

### Page 7: ⚡ Execution Dashboard
| Section | What You See |
|---------|-------------|
| Execution Timeline | Order lifecycle with timestamps |
| Algo Performance | Per-algorithm slippage and fill rate comparison |
| Implementation Shortfall | IS decomposition: timing, impact, spread |
| Latency Distribution | Execution latency histogram |
| Cost Analysis | Transaction cost breakdown pie chart |

### Page 8: 🔍 Market Microstructure
| Section | What You See |
|---------|-------------|
| Order Book Depth | Green bids / Red asks depth chart |
| Spread Analysis | Bid-ask spread time series |
| Toxicity Score | VPIN-style flow toxicity gauge |
| Price Formation | Microprice vs midprice comparison |
| Liquidity Score | Real-time liquidity heatmap |

### Page 9: ⚙️ Configuration
| Section | What You See |
|---------|-------------|
| System Config | Editable parameters (read from system.yaml) |
| Risk Limits | Position limits, drawdown limits, leverage |
| Strategy Parameters | Signal thresholds, ML weights |
| Execution Settings | Default algorithm, slippage tolerance |
| Alert Settings | Telegram/Email configuration |

### Page 10: 🏛️ Investor Portal
| Section | What You See |
|---------|-------------|
| Fund Overview | AUM, returns, Sharpe, max drawdown |
| Monthly Statement | Downloadable performance reports |
| Risk Report | Current risk exposure summary |
| Audit Trail | Recent system decisions and rationale |

---

## 20. LLM/RAG INTEGRATION — AI RESEARCH ASSISTANT

**CRITICAL: LLM has ZERO execution authority. Read-only research track.**

### Architecture
```
Trading Core ──(read-only sync)──▶ DuckDB Cache ──▶ Vector Store (Qdrant)
                                       │                    │
                                       ▼                    ▼
                                  LLM Engine ◄──── RAG Context
                                  (Llama 3.2 8B)
                                       │
                                       ▼
                                  FastAPI REST
                                  (informational only)
```

### LLM Engine
- **Model:** Llama 3.2 8B (GGUF format) at `llm_integration/models/llama-3.2-8b.gguf`
- **Context:** 4096 tokens, 8 threads, 35 GPU layers
- **Temperature:** 0.7, top_p: 0.9
- **Target latency:** 50-200ms per query
- **Fallback:** If model file not present → template-based responses (keyword matching)

### Input Validation
- Max query length: 1000 characters
- Max tokens: 1024
- **Forbidden patterns** (blocked instantly):
  - `"ignore previous instructions"`
  - `"system prompt"`
  - `"execute trade"`
  - `"delete database"`
  - `"drop table"`

### Example Queries and Responses
```
Query: "What's my portfolio status?"
Response: "Portfolio value is $100,456.78 (+0.46% from initial $100,000).
          You have 3 open positions: BTCUSDT (+0.22%), ETHUSDT (+1.05%),
          SOLUSDT (-1.32%). Cash available: $55,230. Win rate: 60.9%."

Query: "Analyze my performance this week"
Response: "This week: 23 trades, 14 wins (60.9%), total P&L +$456.78.
          Sharpe ratio (30d): 1.45. Max drawdown: -0.8%.
          Best performer: ETHUSDT (+1.05%). Worst: SOLUSDT (-1.32%)."

Query: "System health check"
Response: "All systems nominal. WebSocket connected to Binance.
          ML ensemble: 9 models active. Latency: avg 23ms.
          Risk status: GREEN. No circuit breakers triggered."
```

### Vector Store (Qdrant)
- **Embedding model:** all-MiniLM-L6-v2 (384 dimensions)
- **Distance metric:** Cosine
- **Index:** HNSW
- **Collections:** trades, signals, market_analysis, strategies
- **Search latency:** <10ms
- **Fallback:** In-memory Qdrant if server unavailable

### REST API Endpoints
| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/` | Root with endpoint listing |
| `GET` | `/health` | Health check |
| `POST` | `/api/v1/query` | RAG-enhanced LLM query |
| `GET` | `/api/v1/portfolio` | Read-only portfolio state |
| `GET` | `/api/v1/status` | Component health status |
| `POST` | `/api/v1/signal` | Signal analysis (advisory only) |

---

## 21. PROMETHEUS METRICS & FASTAPI ENDPOINTS

### Prometheus Metrics (Port 8000, `/metrics`)
```prometheus
# HELP qf_trades_total Total trade executions
# TYPE qf_trades_total counter
qf_trades_total{symbol="BTCUSDT",side="BUY",status="FILLED"} 12

# HELP qf_signals_total Total signals generated
# TYPE qf_signals_total counter
qf_signals_total{symbol="BTCUSDT",direction="BUY"} 45

# HELP qf_portfolio_value_usd Current portfolio value
# TYPE qf_portfolio_value_usd gauge
qf_portfolio_value_usd 100456.78

# HELP qf_open_positions Open position count
# TYPE qf_open_positions gauge
qf_open_positions 3

# HELP qf_iteration_duration_seconds Pipeline iteration latency
# TYPE qf_iteration_duration_seconds histogram
qf_iteration_duration_seconds_bucket{le="0.1"} 89
qf_iteration_duration_seconds_bucket{le="0.5"} 98
qf_iteration_duration_seconds_bucket{le="2.0"} 100

# HELP qf_risk_blocked_total Signals blocked by risk gate
# TYPE qf_risk_blocked_total counter
qf_risk_blocked_total 47

# HELP qf_circuit_breaker_open Circuit breaker status (1=open)
# TYPE qf_circuit_breaker_open gauge
qf_circuit_breaker_open{symbol="SOLUSDT"} 1

# HELP qf_ws_reconnects_total WebSocket reconnections
# TYPE qf_ws_reconnects_total counter
qf_ws_reconnects_total 2

# HELP qf_ml_inference_seconds ML ensemble inference latency
# TYPE qf_ml_inference_seconds summary
qf_ml_inference_seconds{quantile="0.5"} 0.018
qf_ml_inference_seconds{quantile="0.9"} 0.032
```

### FastAPI Health Endpoint
```json
GET http://localhost:8000/health
{
  "status": "running",
  "uptime_seconds": 3456,
  "trading_mode": "PAPER"
}
```

### FastAPI Status Endpoint
```json
GET http://localhost:8000/api/status
{
  "is_running": true,
  "iteration": 234,
  "symbols": ["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT"],
  "portfolio_value": 100456.78,
  "cash": 55230.00,
  "open_positions": 3,
  "total_trades": 23,
  "win_rate": 0.609,
  "regime": "BULL",
  "execution_mode": "PAPER",
  "uptime_seconds": 3456
}
```

---

## 22. ALERT SYSTEM — TELEGRAM & EMAIL NOTIFICATIONS

### Alert Types and Channels
| Alert Type | Emoji | Default Level | Telegram | Email |
|------------|-------|---------------|----------|-------|
| Trade | 🟢 BUY / 🔴 SELL | INFO | ✅ | ❌ |
| Risk | ⚠️ / 🚨 | WARNING+ | ✅ | ✅ |
| Regime Change | 📊 | INFO | ✅ | ❌ |
| Portfolio | 💼 | INFO | ✅ | ❌ |
| System | ⚙️ / 🔧 | WARNING+ | ✅ | ✅ |

### Rate Limiting
- Max **10 alerts per minute**
- Min **5 seconds** between same type+level combo
- Background daemon thread polls queue every 1 second

### Trade Alert Example
```
🟢 TRADE: BUY BTCUSDT
  Quantity: 0.0163 BTC
  Price: $67,246.60
  P&L: +$48.90
  Algorithm: TWAP
  Fee: $1.10

  Sent via: Telegram ✅ | Email: skipped (INFO level)
```

### Risk Alert Example
```
🚨 RISK ALERT [CRITICAL]
  Message: Drawdown approaching limit: -13.2% (max: -15%)
  
  Sent via: Telegram ✅ | Email ✅
```

### Alert Stats Output
```
Alert Statistics:
  Total alerts: 67
  By level: {INFO: 45, WARNING: 18, CRITICAL: 3, EMERGENCY: 1}
  By type: {TRADE: 23, RISK: 12, REGIME: 8, PORTFOLIO: 15, SYSTEM: 9}
  Channels: {telegram: true, email: false}
```

---

## 23. HEALTH MONITORING SYSTEM

### Monitoring Thresholds
| Metric | Warning | Critical | Direction |
|--------|---------|----------|-----------|
| `pipeline_latency_ms` | 500ms | 2000ms | above |
| `model_latency_ms` | 200ms | 1000ms | above |
| `error_rate_pct` | 5% | 20% | above |
| `memory_pct` | 80% | 95% | above |
| `signal_rate_per_min` | < 0.5 | < 0.1 | below |
| `ws_reconnects` | 3 | 10 | above |

### Health Status Output
```
[HEALTH] System Status:
  CPU:            45.2% (OK)
  Memory:         62.3% (OK)
  Disk:           34.5% (OK)
  Pipeline Lat:   23ms (OK — threshold: 500ms)
  Model Lat:      18ms (OK — threshold: 200ms)
  Error Rate:     1.2% (OK — threshold: 5%)
  Signal Rate:    2.3/min (OK — threshold: 0.5/min)
  WS Reconnects:  1 (OK — threshold: 3)

  Alert cooldown: 300 seconds per metric
  Check interval: 30 seconds (monitoring), 5 seconds (infra)
```

### Health Alert
```
[WARNING] memory_pct = 82.3% breached threshold (warn: 80%)
  → Alert logged, cooldown 5 minutes
```

---

## 24. AUDIT TRAIL — HASH-CHAINED JSONL

Every trading decision is logged to a tamper-proof hash-chained JSONL file.

### Audit Entry Structure
```json
{
  "iteration": 234,
  "timestamp": "2026-02-27T14:30:22.345Z",
  "symbol": "BTCUSDT",
  "market_state": {
    "price": 67234.50,
    "volume": 12.5,
    "regime": "BULL",
    "regime_confidence": 0.80
  },
  "signal_state": {
    "math_signal": 0.21,
    "ml_signal": 0.28,
    "cross_asset_signal": 0.15,
    "fused_signal": 0.219,
    "direction": "BUY"
  },
  "risk_state": {
    "approved": true,
    "drawdown": -0.008,
    "exposure": 0.45,
    "position_pct": 0.032
  },
  "decision": {
    "action": "BUY",
    "quantity": 0.0163,
    "price": 67246.60,
    "algo": "TWAP",
    "fee": 1.10,
    "slippage_bps": 1.8
  },
  "prev_hash": "a3f4b2c1d5e6...",
  "hash": "7b8c9d0e1f2a..."
}
```

### Hash Chain Integrity
- Each entry includes SHA-256 hash of previous entry (`prev_hash`)
- First entry has `prev_hash = "genesis"`
- Tampering any entry breaks the chain → detectable
- File: JSONL format (one JSON object per line)

---

## 25. LOG FILES GENERATED

### Log File Structure
```
logs/
├── quantum_core.log              # Main orchestrator log (10 MB × 5 rotations)
├── system_full.log               # Full system runner log
├── audit/
│   └── decisions_YYYYMMDD.jsonl  # Hash-chained audit trail
└── archive/                      # Rotated logs
```

### Main Log Format
```
2026-02-27 14:30:22.345 | QF_CORE            | INFO    | [BTCUSDT] $67,234.50 | Math: BUY 0.21 | ML: BUY 0.28
2026-02-27 14:30:22.456 | QF_EXEC            | INFO    | TRADE EXECUTED: BUY 0.0163 @ $67,246.60 | algo=TWAP
2026-02-27 14:30:22.457 | QF_RISK            | INFO    | Risk gate: APPROVED | drawdown=-0.8% | exposure=45%
2026-02-27 14:30:22.458 | QF_ALERT           | INFO    | 🟢 Trade alert sent: BUY BTCUSDT
2026-02-27 14:32:45.123 | QF_REGIME          | INFO    | Regime change: NEUTRAL → BULL (conf=0.80)
2026-02-27 14:35:00.234 | QF_CIRCUIT         | WARNING | SOLUSDT breaker OPEN (5/5 failures)
```

### Log Rotation
- **Size:** 10 MB per file
- **Backups:** 5 rotated files kept
- **Format:** `%(asctime)s | %(name)-18s | %(levelname)-7s | %(message)s`

---

## 26. DATABASE RECORDS CREATED

### TimescaleDB Hypertables

#### tick_data (Hypertable — 1-hour chunks)
```sql
timestamp TIMESTAMPTZ,     -- Microsecond precision
symbol VARCHAR(20),
exchange VARCHAR(50),
price DECIMAL(20,8),
size DECIMAL(20,8),
side VARCHAR(4),           -- 'buy' or 'sell'
trade_id BIGINT,
latency_micros INTEGER,
quality_score DECIMAL(5,4),
raw_data JSONB
```
**Retention:** 1 year (configurable)

#### orderbook_data (Hypertable)
```sql
timestamp TIMESTAMPTZ,
symbol VARCHAR(20),
bids JSONB,               -- Array of [price, size] pairs
asks JSONB,
best_bid DECIMAL(20,8),
best_ask DECIMAL(20,8),
spread DECIMAL(20,8),
mid_price DECIMAL(20,8),
is_crossed BOOLEAN,       -- Check: best_bid < best_ask
num_bid_levels INTEGER,
num_ask_levels INTEGER
```

### Data Volume Estimates
| Table | Records/Day (approx) |
|-------|---------------------|
| `tick_data` | 500,000+ (7 symbols × 24h) |
| `orderbook_data` | 100,000+ |
| `features` | 50,000+ |
| `signals` | 5,000-20,000 |
| `trades` | 10-100 |
| `audit_log` | 5,000-20,000 |

### Redis Cache Tiers
| Tier | Latency | TTL | Content |
|------|---------|-----|---------|
| L1 Hot Ticks | <100µs | Last 1M entries | Raw market ticks |
| L2 Features | <1ms | 1 hour | Computed features |
| L3 Metadata | <10ms | 24 hours | Strategy state, configs |

### Storage Coordinator
All signals, trades, and portfolio snapshots are written to three backends simultaneously:
1. **Parquet** — columnar files in `data/parquet/`
2. **Redis** — hot cache for real-time access
3. **TimescaleDB** — persistent time-series

Auto-save of complete system state: **every 50 iterations**

---

## 27. BACKTESTING RESULTS

### Running a Backtest
```bash
python -c "from analytics.backtesting.event_driven_backtest import EventDrivenBacktester; ..."
```

### Event-Driven Backtest Events
| Event Type | Priority | Description |
|------------|----------|-------------|
| MARKET_DATA | 1 | New price tick |
| SIGNAL | 2 | Strategy generates signal |
| ORDER | 3 | Order submitted |
| FILL | 4 | Order filled |
| RISK_CHECK | 5 | Risk limits checked |
| REBALANCE | 6 | Portfolio rebalanced |
| EOD | 7 | End-of-day settlement |

### Available Backtest Modes
| Mode | Engine | Description |
|------|--------|-------------|
| Standard | `BacktestEngine` | Basic bar-by-bar simulation |
| Event-Driven | `EventDrivenBacktester` | Priority-queue event processing |
| Regime-Aware | `RegimeAwareBacktester` | HMM regime overlay |
| Walk-Forward | `WalkForwardOptimizer` | In-sample/out-of-sample split |

### Expected Benchmark Results (From README)
| Metric | Value |
|--------|-------|
| Order Execution Latency | < 1ms |
| Market Data Processing | 100K+ msg/sec |
| Risk Calculation Speed | < 10ms |
| Model Inference Time | < 5ms |
| System Uptime | 99.99% |
| Sharpe Ratio (Backtest) | 2.5+ |

---

## 28. ADVANCED RISK MATHEMATICS OUTPUT

### Extreme Value Theory (EVT)
```
[EVT] Tail Risk Analysis (50+ returns):
  Distribution: Generalized Extreme Value (GEV)
  Parameters:
    Location (μ): -0.0089
    Scale (σ):    0.0234
    Shape (ξ):    0.18 (heavy tail — Fréchet class)

  Tail Risk Metrics:
    VaR 95%:   -$2,340 (1-day)
    VaR 99%:   -$4,560 (1-day)
    VaR 99.9%: -$7,890 (1-day)
    ES 95%:    -$3,120
    ES 99%:    -$5,890
    ES 99.9%:  -$9,450
    
  Return Levels:
    10-year event: -28.4%
    100-year event: -45.2%
    
  Tail index (Hill estimator): 1/ξ = 5.56
```

### Copula Models
```
[COPULA] Cross-Asset Dependence Analysis:
  Type: Gaussian Copula (5-dimension)
  Method: Pearson correlation on normal-transformed margins

  Dependence Metrics (BTC-ETH):
    Pearson:         0.87
    Spearman:        0.82
    Kendall's Tau:   0.68
    Upper tail dep:  0.23
    Lower tail dep:  0.31 ← higher tail dependence in crashes
    
  Correlation Matrix (5×5):
         BTC    ETH    BNB    SOL    XRP
  BTC   1.00   0.87   0.65   0.72   0.58
  ETH   0.87   1.00   0.71   0.78   0.63
  BNB   0.65   0.71   1.00   0.55   0.48
  SOL   0.72   0.78   0.55   1.00   0.52
  XRP   0.58   0.63   0.48   0.52   1.00
```

### Optimal Stopping
```
[OPTIMAL STOPPING] Analysis:
  Method: American put option framework
  Current position: BTCUSDT long +0.0163 @ $67,100
  Optimal exit boundary: $66,200 (1.34% below entry)
  Expected optimal time: 4.2 hours
  Continuation value: $234.56
  Stopping value: $0.00 (close now)
  → Recommendation: CONTINUE (value of waiting > stopping)
```

### Portfolio Risk Manager
```
[PORTFOLIO RISK] Assessment:
  Status:     NORMAL
  Alerts:     0 active
  
  Risk Limits:
  ┌──────────────────────┬────────┬────────┬──────────┐
  │ Limit                │ Value  │ Limit  │ Util %   │
  ├──────────────────────┼────────┼────────┼──────────┤
  │ Max position weight  │ 8.2%   │ 25%    │ 32.8%    │
  │ Max leverage         │ 0.45x  │ 2.0x   │ 22.5%    │
  │ Max drawdown         │ -0.8%  │ -15%   │ 5.3%     │
  │ Daily loss limit     │ -$120  │ -$2,000│ 6.0%     │
  └──────────────────────┴────────┴────────┴──────────┘
  
  Warning threshold: 80% utilization
  Critical threshold: 95% utilization
```

---

## 29. SHUTDOWN SEQUENCE & FINAL STATISTICS

### Graceful Shutdown (Ctrl+C or SIGTERM)
```
[STOP] Stopping system...

  → Stopping Quantum Core Orchestrator
  → Closing open positions (paper mode: no real orders)
  → Stopping data cache
  → Stopping WebSocket connections
  → Stopping system monitor
  → Terminating Streamlit UI process
  → Flushing storage buffers

════════════════════════════════════════════════════════════════════════
                    QUANTUM-FORGE FINAL STATISTICS
════════════════════════════════════════════════════════════════════════

  SESSION PERFORMANCE:
    Initial Capital:     $100,000.00
    Final Value:         $100,456.78
    Total Return:        +0.46%
    Total Trades:        23
    Win Rate:            60.9% (14W / 9L)
    Iterations:          1,234

  RISK METRICS:
    Sharpe Ratio:        1.45
    Max Drawdown:        -0.8%
    Annualized Vol:      12.4%
    Signals Blocked:     47 (20.1% block rate)

  ML ENSEMBLE:
    Model Weights:       {lstm: 0.155, transformer: 0.192, gru: 0.112, ...}
    Average Accuracy:    61.2%

  EXECUTION:
    Total Fees:          $25.30
    Avg Slippage:        2.1 bps
    Fill Rate:           95.7%

════════════════════════════════════════════════════════════════════════

[OK] System Shutdown Complete
```

---

## 30. DOCKER DEPLOYMENT — FULL STACK OUTPUT

### Starting with Docker Compose
```bash
docker-compose up -d
```

### Services Started (10+ Containers)
```
Creating network "quantum-forge-network"
Creating quantum-forge-timescaledb    ... done  (port 5432)
Creating quantum-forge-redis          ... done  (port 6379)
Creating quantum-forge-app            ... done  (port 8000 + 8501)
Creating quantum-forge-dashboard      ... done  (port 8502)
Creating quantum-forge-nginx          ... done  (port 80, 443)
Creating quantum-forge-prometheus     ... done  (port 9090)
Creating quantum-forge-grafana        ... done  (port 3000)
Creating quantum-forge-jaeger         ... done  (port 16686)
```

### Dev Profile (Additional Services)
```bash
docker-compose --profile dev up -d

Creating quantum-forge-redis-insight  ... done  (port 8001)
Creating quantum-forge-pgadmin        ... done  (port 8080)
Creating quantum-forge-jupyter        ... done  (port 8888)
```

### Port Map
| Port | Service | Purpose |
|------|---------|---------|
| 80 / 443 | Nginx | Reverse proxy (HTTP/HTTPS) |
| 3000 | Grafana | Monitoring dashboards (admin/admin) |
| 5432 | TimescaleDB/PostgreSQL | Time-series database |
| 6379 | Redis | Three-tier cache (8GB max, LRU) |
| 8000 | FastAPI | Prometheus metrics + REST API |
| 8001 | Redis Insight (dev) | Redis GUI |
| 8080 | pgAdmin (dev) | Database GUI |
| 8501 | Streamlit | Trading dashboards (10 pages) |
| 8888 | Jupyter (dev) | Research notebooks |
| 9090 | Prometheus | Metric scraping (15s interval, 90-day retention) |
| 16686 | Jaeger | Distributed tracing UI |

### Resource Limits (Docker)
| Service | CPU | Memory |
|---------|-----|--------|
| TimescaleDB | 2 cores | 2 GB |
| Redis | 1 core | 2 GB |
| Quantum Forge App | 4 cores | 8 GB |

---

## 31. PAPER TRADING vs LIVE TRADING DIFFERENCES

| Aspect | Paper Trading (Default) | Live Trading |
|--------|------------------------|-------------|
| Capital | Simulated $100,000 | Real funds |
| Orders | Never sent to Binance | Signed API requests to `api.binance.com/api/v3/order` |
| Fills | Simulated (mid + slippage model) | Real exchange fills |
| Slippage | Modeled (1-5 bps by algo) | Real market slippage |
| Fees | Simulated 0.1% | Real Binance fees (0.1% maker/taker) |
| API Keys | Not required | Required (`BINANCE_API_KEY`, `BINANCE_API_SECRET`) |
| Financial Risk | None | Real money at risk |
| Activation | Default | `--live` flag or code change |
| Circuit Breaker | Active (logs only) | Active (blocks real trades) |
| Retry Logic | N/A | 3 retries, exponential backoff |
| Order Polling | N/A | Check every 2s, timeout 30s |
| Reconciliation | N/A | Fetches `/account` balances from Binance |

---

## 32. PERFORMANCE BENCHMARKS — EXPECTED NUMBERS

### Latency Benchmarks
| Component | Target | Expected |
|-----------|--------|----------|
| Main loop iteration | 2s | 1.5-2.5s |
| Signal generation | <50ms | 10-30ms |
| ML ensemble inference | <200ms | 15-40ms |
| Feature extraction | <50ms | 5-15ms |
| Risk gate check | <5ms | 1-3ms |
| Execution (paper) | <10ms | 2-5ms |
| WebSocket tick | <5ms | 1-3ms |
| Prometheus scrape | <100ms | 10-50ms |
| LLM query | <200ms | 50-200ms |
| Vector search | <10ms | 2-8ms |

### Resource Usage
| Resource | Expected | Limit |
|----------|----------|-------|
| CPU | 30-50% (4 cores) | 16 threads configured |
| RAM | 2-6 GB | 16 GB max in config |
| Disk (logs) | 50-200 MB/day | 10 MB × 5 rotation |
| Disk (database) | 2-10 GB/day | 1-year retention |
| Network | 5-15 Mbps | Binance WS + API |
| Redis | 100-500 MB | 8 GB max, LRU eviction |

### Daily Trading Statistics
| Metric | Expected Range |
|--------|---------------|
| Trades per day | 10-100 |
| Signals generated | 5,000-20,000 |
| Signals blocked by risk | 15-25% |
| Win rate | 50-65% |
| Daily P&L range | -$500 to +$1,000 |
| Max daily loss cap | -$2,000 (2%) |
| Positions open | 1-5 |
| Avg trade hold time | 1-24 hours |
| Circuit breaker triggers | 0-3 per day |

### Binance-Specific Limits
| Limit | Default |
|-------|---------|
| Max positions | 7 (one per symbol) |
| Max position size per symbol | BTC: $100K, ETH: $80K, BNB: $60K, SOL: $50K, ADA: $40K, XRP: $35K, DOGE: $30K |
| Max leverage | 3.0x |
| Total exposure | $500K USDT |
| API rate limit | 1200 orders/min, 6000 requests/min |
| Stop loss per trade | 5% |
| Take profit per trade | 10% (scaled: 50% at +3%, rest at +5%) |
| Trailing stop | Activate at +3%, trail 2% |
| Expected daily volatility | BTC 4%, ETH 5%, BNB 4.5%, SOL 6%, ADA 5.5%, DOGE 8%, XRP 5% |

---

## 33. ERROR SCENARIOS & FALLBACK BEHAVIORS

### Binance WebSocket Failure
```
[ERROR] WebSocket connection lost
[WARN]  Reconnect attempt 1/10 (delay: 2s)
[WARN]  Reconnect attempt 2/10 (delay: 4s)
...
[WARN]  Max reconnect attempts reached (10)
[INFO]  Falling back to REST API: api.binance.com/api/v3/ticker/price
[INFO]  System continues with REST polling (higher latency)
```

### Redis Failure
```
[WARN] Redis unavailable: Connection refused
[INFO] Falling back to in-memory dict cache (non-persistent)
[WARN] L1 hot tick cache disabled — higher latency expected
```

### TimescaleDB Failure
```
[WARN] TimescaleDB connection failed
[INFO] Using Parquet-only storage (local files)
[WARN] Historical queries will be slower
```

### Qdrant Failure (LLM/RAG)
```
[WARN] Qdrant server unavailable at localhost:6333
[INFO] Falling back to in-memory vector store
[WARN] Vector search non-persistent — reindexing on restart
[INFO] Trading continues normally (LLM is read-only)
```

### ML Model Loading Failure
```
[WARN] Failed to load LSTM state dict from intelligence/trained_models/lstm.pt
[INFO] Model running with random initialization (untrained)
[INFO] Ensemble falls back to statistical models
```

### Llama Model Missing
```
[WARN] Model file not found: llm_integration/models/llama-3.2-8b.gguf
[INFO] LLM Engine falling back to Template Mode
[INFO] Natural language queries will return keyword-based responses
[INFO] Trading pipeline completely unaffected
```

### Circuit Breaker Triggered
```
[CIRCUIT] SOLUSDT breaker OPEN after 5 consecutive failures
  → Symbol skipped for 60s cooldown
  → Other symbols continue normally
  → After cooldown: half-open (allow one attempt)
  → 1 success → CLOSED; 1 failure → reopen
```

### Risk Gate — All Signals Blocked
```
[RISK] CRISIS regime detected — ALL signals blocked
  → No new trades for any symbol
  → Existing positions maintained (no panic selling)
  → Waiting for regime change to exit CRISIS
  → Capital allocator multiplier: 0.00
```

### Complete System Recovery
```
[WARN] Component failure detected — entering DEGRADED mode
  → Failed: Redis, Qdrant
  → Active: Binance WS, PostgreSQL, ML Ensemble, Math Engine
  → Trading continues with degraded caching
  → Dashboard may show stale data
  → Alerts: SYSTEM WARNING sent
```

---

## 34. COMPLETE FILE OUTPUT REFERENCE

### Files Generated During a Session
| File/Path | Type | Description |
|-----------|------|-------------|
| `logs/quantum_core.log` | Log | Main orchestrator log (10 MB × 5) |
| `logs/system_full.log` | Log | Full system runner log |
| `logs/audit/decisions_YYYYMMDD.jsonl` | Audit | Hash-chained decision trail |
| `data/parquet/*.parquet` | Data | Columnar data snapshots |
| `data/state/*.json` | State | Auto-saved system state |
| `backups/` | Backup | Full/incremental backups (SHA-256) |

### Network Endpoints Active
| Endpoint | Protocol | Purpose |
|----------|----------|---------|
| `wss://stream.binance.com:9443/ws` | WebSocket | Real-time market data |
| `https://api.binance.com/api/v3/*` | HTTPS | REST fallback + trading |
| `http://localhost:8000/health` | HTTP | K8s/Docker health probe |
| `http://localhost:8000/metrics` | HTTP | Prometheus scrape |
| `http://localhost:8000/api/status` | HTTP | Full system status JSON |
| `http://localhost:8501` | HTTP | Streamlit dashboards (10 pages) |
| `http://localhost:3000` | HTTP | Grafana (admin/admin) |
| `http://localhost:9090` | HTTP | Prometheus |
| `http://localhost:16686` | HTTP | Jaeger tracing |
| `localhost:6379` | TCP | Redis |
| `localhost:5432` | TCP | TimescaleDB |
| `localhost:6333` | HTTP | Qdrant vector store |

### Verified Module Count (from Validation)
| Category | Modules | Status |
|----------|---------|--------|
| Data Ingestion | 8 | ✅ PASS |
| Preprocessing | 3 | ✅ PASS |
| Storage | 5 | ✅ PASS |
| Deep Learning | 8/9 | ✅ (1 optional: GNN) |
| Reinforcement Learning | 4 | ✅ PASS |
| Feature Learning | 3/4 | ✅ (1 optional: Manifold) |
| Meta Learning | 4 | ✅ PASS |
| Probabilistic ML | 4 | ✅ PASS |
| Math Engine | 10 | ✅ PASS |
| Microstructure | 4 | ✅ PASS |
| Risk Mathematics | 5 | ✅ PASS |
| Signal & Alpha | 14 | ✅ PASS |
| Execution (Core) | 14 | ✅ PASS |
| Execution (Layer) | 15 | ✅ PASS |
| Analytics | 24 | ✅ PASS |
| Risk Management | 1 | ✅ PASS |
| LLM Integration | 4 | ✅ PASS |
| Infrastructure | 14 | ✅ PASS |
| Strategies | 4 | ✅ PASS |
| **TOTAL** | **164/166** | **98.8% (100% effective)** |

---

## APPENDIX A: QUICK START COMMAND REFERENCE

```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install -e .

# 2a. Quick start (lightweight, recommended)
python launch_quantum_core.py

# 2b. Full system with UI launcher
python launch.py

# 2c. Full system with all 135+ modules validated
python run_full_system.py

# 2d. Pipeline mode
python launch_pipeline.py

# 3. Custom configuration
python launch_quantum_core.py --symbols BTCUSDT,ETHUSDT --capital 50000
python launch_quantum_core.py --no-ml          # Math-only mode
python launch_quantum_core.py --llm            # Enable LLM research
python launch_quantum_core.py --threshold 0.35 # Higher signal bar

# 4. Open dashboard
# Navigate to http://localhost:8501

# 5. Check API status
curl http://localhost:8000/api/status

# 6. Check Prometheus metrics
curl http://localhost:8000/metrics

# 7. Docker full stack
docker-compose up -d
docker-compose --profile dev up -d  # With pgAdmin, Redis Insight, Jupyter

# 8. Run tests
pytest tests/ -v --cov=core --cov=intelligence --cov=analytics

# 9. Stop
Ctrl+C (graceful shutdown with final stats)
docker-compose down
```

---

## APPENDIX B: CONFIGURATION REFERENCE

### Key Configurable Parameters
| Parameter | Default | Source | Effect |
|-----------|---------|--------|--------|
| `initial_capital` | $100,000 | CLI / code | Starting portfolio value |
| `symbols` | 5-7 pairs | CLI / config | Trading universe |
| `signal_threshold` | 0.25 | CLI | Minimum signal for trade |
| `enable_ml` | True | CLI | ML ensemble on/off |
| `enable_llm` | False | CLI | LLM research track on/off |
| `max_position_pct` | 10% | RiskGate | Max per-symbol exposure |
| `max_total_exposure` | 80% | RiskGate | Max total portfolio exposure |
| `max_drawdown_pct` | 15% | RiskGate | Drawdown halt threshold |
| `circuit_breaker_fails` | 5 | CircuitBreaker | Failures before cooldown |
| `circuit_breaker_cooldown` | 60s | CircuitBreaker | Cooldown duration |
| `fee_rate` | 0.1% | ExecutionManager | Binance spot fee |
| `loop_interval` | 2s | QuantumCore | Main loop sleep |
| `periodic_analysis` | Every 20 iter | QuantumCore | Summary frequency |
| `auto_save` | Every 50 iter | QuantumCore | State persistence frequency |
| `allocation_method` | HYBRID | CapitalAllocator | 60% perf + 40% risk parity |
| `max_single_allocation` | 40% | CapitalAllocator | Max per-strategy weight |
| `min_single_allocation` | 5% | CapitalAllocator | Min per-strategy weight |
| Signal fusion: Math | 50% | QuantumCore | Math signal weight |
| Signal fusion: ML | 30% | QuantumCore | ML ensemble weight |
| Signal fusion: Cross | 20% | QuantumCore | Cross-asset weight |
| Trade size | 5% of cash × strength | QuantumCore | Position sizing |
| Sell size | 50% of position | QuantumCore | Partial exit |
| Min trade value | $10 | QuantumCore | Minimum order |
| Max leverage | 2.0x | Risk limits | Portfolio leverage cap |
| VaR confidence | 95% | Risk config | VaR calculation level |
| Stop loss | 5% | Binance config | Per-trade stop |
| Take profit | 10% (scaled) | Binance config | Per-trade target |
| Trailing stop | Activate +3%, trail 2% | Binance config | Trailing exit |

---

## APPENDIX C: EXPECTED FIRST-RUN EXPERIENCE

### For a First-Time User with Only Python Installed

**Scenario:** User has Python 3.8+, no Docker, no Redis, no databases.

**What happens:**
1. `pip install -r requirements.txt` — installs ~60+ Python packages including PyTorch
2. `python launch_quantum_core.py` — starts with defaults
3. System boots with graceful fallbacks:
   - Redis → **not available** → in-memory dict cache
   - TimescaleDB → **not available** → Parquet-only storage
   - Qdrant → **not available** → in-memory vector store
   - Llama model → **not present** → template responses (LLM disabled by default anyway)
   - R → **not installed** → Python statsmodels
4. Binance WebSocket connects (no API key needed for public market data)
5. Real-time prices stream in for 5 symbols (BTCUSDT, ETHUSDT, BNB, SOL, XRP)
6. First 30+ ticks: warmup phase (signals generated but minimum history needed)
7. After ~60 seconds: first signals appear
8. Risk gate filters cautiously — many signals blocked initially
9. First trades execute in paper mode with simulated fills
10. Every ~40 seconds: periodic portfolio summary printed
11. `Ctrl+C` → graceful shutdown with final statistics

**First-run terminal output will show:**
- Several WARNING messages about missing infrastructure (Redis, TimescaleDB)
- Real Binance WebSocket market data streaming
- Math signals (Fourier, Wavelet, etc.) generating immediately
- ML ensemble running (untrained models → lower confidence initially)
- Risk gate blocking many initial signals (expected — conservative)
- Periodic portfolio analysis with P&L near $0
- Clean shutdown with comprehensive session statistics

**Expected P&L in first 1-hour run:** Near $0 to ±$100 (untrained ML models, conservative risk gates, paper mode)

**To get better results:**
- Train ML models on historical data first
- Run for longer periods (the system learns and adapts online)
- Download historical data with `scripts/download_data.py`

---

*This document covers every expected output, visual, log entry, database record, metric, alert, and behavior that occurs when running the QUANTUM-FORGE trading system. All values, thresholds, and defaults are extracted directly from the source code.*

**Total Source Modules:** 135+ (164/166 verified, 98.8%)  
**Total AI/ML Models:** 20+ (9 in active ensemble)  
**Total Math Signal Sources:** 7  
**Total Features per Tick:** 32  
**Total Prometheus Metrics:** 9  
**Total Dashboard Pages:** 10  
**Total Risk Gate Checks:** 6  
**Total Execution Algorithms:** 4 + MARKET  
**Exchange:** Binance (24/7 cryptocurrency)  
**LLM Authority:** ZERO (read-only research only)  
