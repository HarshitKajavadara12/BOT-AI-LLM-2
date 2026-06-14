# QUANTUM-FORGE — Complete Pipeline Architecture Document

> **Project:** QUANTUM-FORGE Institutional Trading Platform  
> **Purpose:** Fully automated, AI/ML-driven multi-symbol crypto trading system with 135+ modules, 20+ ML models, professional execution algorithms, advanced risk mathematics, LLM research integration, and 10 interactive dashboards.  
> **Generated:** 2026-02-24

---

## 1. PURPOSE — Why This System Was Built

QUANTUM-FORGE is an **institutional-grade quantitative trading platform** designed to:

| Goal | How It's Achieved |
|---|---|
| **Real-time market data** | Binance WebSocket + REST (klines, aggTrades, orderbook), alternative data feeds |
| **AI/ML intelligence** | 20+ models: LSTM, GRU, Transformer, TCN, PPO, SAC, GNN, GP, Bayesian, MAML, Autoencoders |
| **Mathematical signal generation** | Fourier analysis, stochastic calculus, wavelet transforms, Kalman filters |
| **Regime detection** | HMM, changepoint detection, GARCH volatility clustering, correlation regimes |
| **Alpha research** | Alpha discovery, validation, combination, decay analysis, factor research |
| **Advanced risk math** | Extreme Value Theory, copula models, optimal stopping, cognitive dampener |
| **Professional execution** | VWAP, TWAP, Implementation Shortfall, Arrival Price, smart routing, HFT |
| **Portfolio management** | Dynamic portfolio tracking, capital allocation, cross-asset risk |
| **LLM integration** | Read-only research track: vector store, DuckDB cache, LLM engine (zero execution authority) |
| **Interactive dashboards** | 10 Streamlit dashboards (trading, risk, portfolio, analytics, research, execution, microstructure) |
| **Tamper-proof audit** | Hash-chained JSONL audit trail for every decision |

---

## 2. SYSTEM PIPELINE — End-to-End Data Flow

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                        QUANTUM-FORGE MASTER PIPELINE                                  │
│                                                                                      │
│                        ┌─────────────────────┐                                       │
│                        │    BINANCE MARKET    │                                       │
│                        │   WebSocket + REST   │                                       │
│                        └──────────┬──────────┘                                       │
│                                   │                                                  │
│                    ┌──────────────▼──────────────┐                                   │
│                    │      DATA INGESTION LAYER    │                                   │
│                    │  WS → Tick → Stream → Cache  │                                   │
│                    └──────────────┬──────────────┘                                   │
│                                   │                                                  │
│                    ┌──────────────▼──────────────┐                                   │
│                    │    PREPROCESSING & STORAGE   │                                   │
│                    │  Clean → Normalize → Align   │                                   │
│                    │  → FeatureStore → Parquet     │                                   │
│                    └──────────────┬──────────────┘                                   │
│                                   │                                                  │
│              ┌────────────────────┼────────────────────┐                             │
│              ▼                    ▼                     ▼                             │
│   ┌──────────────────┐ ┌─────────────────┐  ┌──────────────────┐                    │
│   │  MATH ENGINE     │ │  ML ENSEMBLE    │  │ CROSS-ASSET      │                    │
│   │  Fourier+Stoch   │ │  LSTM+Trans+PPO │  │ ALPHA ENGINE     │                    │
│   │  +Wavelet+Kalman │ │  +SAC+GP+SVM    │  │ + Causal Bridge  │                    │
│   └────────┬─────────┘ └───────┬─────────┘  └────────┬─────────┘                    │
│            │                    │                      │                              │
│            └────────────┬───────┘──────────────────────┘                             │
│                         ▼                                                            │
│              ┌──────────────────────┐                                                │
│              │   SIGNAL FUSION      │                                                │
│              │  Math 50% + ML 30%   │                                                │
│              │  + CrossAsset 20%    │                                                │
│              └──────────┬───────────┘                                                │
│                         │                                                            │
│              ┌──────────▼───────────┐                                                │
│              │   REGIME DETECTION   │                                                │
│              │   HMM + Multi-Signal │                                                │
│              │   + Vol-of-Vol       │                                                │
│              └──────────┬───────────┘                                                │
│                         │                                                            │
│         ┌───────────────▼───────────────┐                                            │
│         │        RISK GATE              │                                            │
│         │  Regime → Drawdown → Exposure │                                            │
│         │  → Position → Signal Floor    │                                            │
│         │  + Portfolio Risk Manager     │                                            │
│         │  + Circuit Breaker            │                                            │
│         │  + Cognitive Dampener         │                                            │
│         └───────────────┬───────────────┘                                            │
│                         │ (approved only)                                            │
│              ┌──────────▼───────────┐                                                │
│              │   EXECUTION ENGINE   │                                                │
│              │  Auto-select algo:   │                                                │
│              │  VWAP/TWAP/IS/Market │                                                │
│              │  + Fee + Slippage    │                                                │
│              └──────────┬───────────┘                                                │
│                         │                                                            │
│         ┌───────────────▼───────────────┐                                            │
│         │     AUDIT + FEEDBACK          │                                            │
│         │  Hash-chained JSONL           │                                            │
│         │  ML weight adaptation         │                                            │
│         │  Storage (Parquet/Redis/TS)   │                                            │
│         └───────────────────────────────┘                                            │
│                                                                                      │
│  ┌────────────────────────────────────────────────────────────────────────────────┐  │
│  │  PARALLEL TRACKS                                                                │  │
│  │                                                                                  │  │
│  │  LLM RESEARCH (Read-Only)         ANALYTICS & DASHBOARDS                       │  │
│  │  DuckDB → Bridge → VectorStore    Backtesting, Attribution, Alpha Research      │  │
│  │  → LLM Engine (ZERO exec auth)    10 Streamlit Dashboards                      │  │
│  └────────────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. QUANTUM CORE — The Real Pipeline (10-Step Loop)

The `QuantumCoreOrchestrator` in `core/quantum_core.py` is the **actual brain** that wires all modules together:

```
PER-SYMBOL PROCESSING (every 2 seconds, per symbol):

STEP 1:  Fetch REAL market data (WebSocket → REST fallback)
           → price, OHLCV, volume
           → Feed to Strategy Multiplexer (shadow strategies)

STEP 2:  Signal Generator — REAL math
           → Fourier spectral analysis
           → Stochastic process modelling
           → Wavelet decomposition
           → Multi-source fusion

STEP 3:  Regime Detection (HMM + Multi-Signal)
           → BULL / BEAR / NEUTRAL / HIGH_VOLATILITY / CRISIS
           → Capital allocator adjusts per regime
           → Alert on regime change

STEP 4:  ML Ensemble — REAL models, REAL features
           → Feature Pipeline: 60+ engineered features
           → LSTM, Transformer, PPO, SAC predictions
           → SVM hyperplane classifier (online learning)
           → Cross-Asset Alpha Engine
           → Causal Discovery Bridge
           → Alt Data Alpha, Crowding Detection
           → Spoofing Detection, Queue Estimation
           → Variational Inference (Bayesian streaming)

STEP 5:  Signal Fusion
           → Math signal: 50% weight
           → ML ensemble: 30% weight
           → Cross-asset: 20% weight
           → Dynamic reweighting if sources unavailable

STEP 6:  Risk Gate (6 checks)
           → Regime gate (CRISIS = halt all)
           → High-vol gate (require strength > 0.7)
           → Drawdown gate (max 15%)
           → Position size check (max 10% per symbol)
           → Total exposure check (max 80%)
           → Signal strength floor (regime-adjusted)
           + Portfolio Risk Manager (limits + alerts)
           + Circuit Breaker (5-failure cooldown)

STEP 7:  Execution via ExecutionManager
           → Auto-select algorithm: VWAP / TWAP / IS / MARKET
           → Based on: order size, volatility, urgency
           → Fee accounting + slippage model
           → Position update + P&L calculation

STEP 8:  Post-Trade
           → Market Impact Tracker (fill analysis)
           → Trade alert (notification system)

STEP 9:  Audit Log
           → Hash-chained JSONL (tamper-proof)
           → market_state + signal_state + risk_state + decision

STEP 10: Feedback & Adaptation
           → ML ensemble weight adaptation (online learning)
           → Per-symbol return tracking
           → EVT tail risk analysis (periodic)
           → Copula cross-asset dependence (periodic)
           → Shadow strategy comparison (periodic)
           → State persistence (auto-save every 50 iterations)
```

---

## 4. COMPONENT INVENTORY (135+ Modules)

### 4.1 Data Ingestion Layer (`data/ingestion/`)

| File | Component | Purpose |
|---|---|---|
| `binance_client.py` | `BinanceAPIClient` | REST API client for Binance |
| `binance_websocket.py` | `BinanceWebSocket` / `MultiStreamWebSocket` | Real-time WebSocket streams |
| `tick_data_handler.py` | `TickDataHandler` | Raw tick processing |
| `streaming_engine.py` | `DataStreamManager` | Multi-stream data distribution |
| `realtime_data_cache.py` | `RealTimeDataCache` | In-memory market data cache |
| `orderbook_reconstructor.py` | `OrderBookReconstructor` | L2 orderbook reconstruction |
| `alternative_data_loader.py` | `AlternativeDataLoader` | Alternative data feeds |

### 4.2 Data Preprocessing (`data/preprocessing/`)

| File | Component | Purpose |
|---|---|---|
| `data_cleaner.py` | `DataCleaner` | Missing data, outlier handling |
| `normalization.py` | `DataNormalizer` | Feature normalization/scaling |
| `alignment.py` | `DataAligner` | Time series alignment |

### 4.3 Data Storage (`data/storage/`)

| File | Component | Purpose |
|---|---|---|
| `redis_cache.py` | `RedisCache` | High-speed key-value cache |
| `timescaledb_manager.py` | `TimescaleDBManager` | Time-series database |
| `parquet_writer.py` | `ParquetWriter` | Columnar storage for analytics |
| `feature_store.py` | `FeatureStore` | Centralised feature storage |
| `historical_data.py` | `HistoricalDataManager` | Historical data management |

### 4.4 Intelligence — Deep Learning (`intelligence/deep_learning/`)

| File | Component | Purpose |
|---|---|---|
| `deep_learning_models.py` | `LSTMModel`, `GRUModel`, `TransformerModel`, `CNNModel` | Core DL architectures |
| `attention_mechanisms.py` | `MultiHeadAttention`, `SelfAttention` | Attention layers |
| `graph_networks.py` | `CorrelationGraphNet` | Graph neural network for asset correlations |
| `temporal_models.py` | `TemporalConvNet` | Temporal convolutional networks |
| `state_space_models.py` | `StateSpaceModel` | State-space models |

### 4.5 Intelligence — Reinforcement Learning (`intelligence/reinforcement_learning/`)

| File | Component | Purpose |
|---|---|---|
| `ppo_agent.py` | `PPOAgent` | Proximal Policy Optimization |
| `soft_actor_critic.py` | `SACAgent` | Soft Actor-Critic |
| `model_based_rl.py` | `MBPO` | Model-Based Policy Optimization |
| `market_env.py` | `MarketEnvironment` | RL training environment |

### 4.6 Intelligence — Feature Learning (`intelligence/feature_learning/`)

| File | Component | Purpose |
|---|---|---|
| `feature_learning.py` | `FeatureLearningEngine` | Autoencoder feature extraction |
| `representation_learning.py` | `RepresentationLearner` | Learned representations |
| `manifold_learning.py` | `ManifoldLearner` | Manifold dimensionality reduction |
| `causal_discovery.py` | `CausalDiscoveryEnsemble` | Causal structure discovery |

### 4.7 Intelligence — Meta Learning (`intelligence/meta_learning/`)

| File | Component | Purpose |
|---|---|---|
| `meta_learning.py` | `MAML` | Model-Agnostic Meta-Learning |
| `few_shot_learning.py` | `FewShotLearner` | Few-shot regime adaptation |
| `transfer_learning.py` | `TransferLearner` | Cross-domain transfer |
| `ensemble_distillation.py` | `EnsembleDistiller` | Knowledge distillation |

### 4.8 Intelligence — Probabilistic ML (`intelligence/probabilistic_ml/`)

| File | Component | Purpose |
|---|---|---|
| `gaussian_processes.py` | `FinancialGaussianProcess` | GP regression with uncertainty |
| `bayesian_optimization.py` | `BayesianOptimizer` | Hyperparameter optimization |
| `variational_inference.py` | `StochasticVariationalInference` | VI for posterior estimation |
| `conformal_prediction.py` | `ConformalPredictor` | Distribution-free prediction intervals |

### 4.9 Core — Math Engine (`core/math_engine/`)

| File | Component | Purpose |
|---|---|---|
| `mathematical_engine.py` | `MathematicalEngine` | Master math dispatcher |
| `fourier_analysis.py` | `FourierAnalyzer` | Spectral analysis / cycle detection |
| `stochastic_calculus.py` | `StochasticProcesses` | Itô calculus, GBM, Ornstein-Uhlenbeck |
| `signal_processing.py` | `KalmanFilter` / `WaveletAnalysis` | Signal filtering + denoising |
| `optimal_control.py` | `ModelPredictiveControl` / `StochasticControl` | Optimal trading control |
| `numerical_methods.py` | `OptimizationMethods` | Numerical optimization |
| `statistical_tests.py` | `UnitRootTests` | ADF, KPSS, stationarity tests |
| `linear_algebra.py` | `LinearAlgebraEngine` | Matrix operations, PCA, SVD |

### 4.10 Core — Market Microstructure (`core/market_microstructure/`)

| File | Component | Purpose |
|---|---|---|
| `orderbook_dynamics.py` | `OrderBookAnalyzer` | Orderbook imbalance, pressure |
| `price_formation.py` | `PriceFormationAnalyzer` | Price impact models |
| `liquidity_models.py` | `LiquidityAnalyzer` | Liquidity estimation |
| `toxicity_detection.py` | `ToxicityDetector` | VPIN / toxic flow detection |

### 4.11 Core — Risk Mathematics (`core/risk_mathematics/`)

| File | Component | Purpose |
|---|---|---|
| `extreme_value_theory.py` | `EVTAnalyzer` | Tail risk estimation (GPD) |
| `copula_models.py` | `CopulaAnalyzer` | Cross-asset dependence |
| `drawdown_mathematics.py` | `DrawdownAnalyzer` | Drawdown distribution modelling |
| `optimal_stopping.py` | `OptimalStoppingAnalyzer` | When to exit positions |
| `cognitive_dampener.py` | `CognitiveDampener` | Regime-aware LLM risk layer |

### 4.12 Core — Signal & Alpha (`core/`)

| File | Component | Purpose |
|---|---|---|
| `signal_generator.py` | `SignalGenerator` | Math-based signal generation |
| `ml_ensemble.py` | `MLEnsembleEngine` | ML model ensemble with weight adaptation |
| `regime_detector.py` | `RegimeDetector` | HMM + multi-signal regime detection |
| `feature_pipeline.py` | `FeaturePipeline` | 60+ engineered features |
| `cross_asset_alpha.py` | `CrossAssetAlphaEngine` | Cross-asset alpha signals |
| `alpha_bridge.py` | `AlphaResearchScheduler` / `AlphaStore` | Alpha lifecycle management |
| `causal_bridge.py` | `CausalBridge` | Causal discovery for signal boost |
| `svm_classifier.py` | `SVMRegimeClassifier` | SVM hyperplane regime classification |
| `alt_data_alpha.py` | `AltDataAlphaEngine` | Alternative data alpha |
| `alpha_crowding_detector.py` | `AlphaCrowdingDetector` | Crowded trade detection |
| `order_book_analyzer.py` | `OrderBookAnalyzer` | Order flow analysis |
| `spoofing_detector.py` | `SpoofingDetector` | Manipulation detection |
| `queue_position_estimator.py` | `QueuePositionEstimator` | Limit order queue estimation |

### 4.13 Core — Execution & Management (`core/`)

| File | Component | Purpose |
|---|---|---|
| `execution_manager.py` | `ExecutionManager` | Auto-select algo, fee/slippage accounting |
| `capital_allocator.py` | `CapitalAllocator` | Dynamic capital allocation |
| `strategy_multiplexer.py` | `StrategyMultiplexer` | Live + shadow strategy management |
| `shadow_tracker.py` | `ShadowTracker` | Paper-trade shadow strategies |
| `market_impact_tracker.py` | `MarketImpactTracker` | Post-trade impact measurement |

### 4.14 Core — Execution Algorithms (`core/execution_algorithms/`)

| File | Component | Purpose |
|---|---|---|
| `vwap_algorithm.py` | `VWAPAlgorithm` | Volume-Weighted Average Price |
| `twap_algorithm.py` | `TWAPAlgorithm` | Time-Weighted Average Price |
| `implementation_shortfall.py` | `ImplementationShortfallAlgorithm` | Minimize execution shortfall |
| `arrival_price.py` | `ArrivalPriceAlgorithm` | Arrival price benchmark |

### 4.15 Execution Layer (`execution/`)

| Subdirectory | Key Components | Purpose |
|---|---|---|
| `order_management/` | `OrderManagementSystem`, `SmartOrderRouter`, `PositionManager`, `FillManager`, `OptimalExecution` | Full OMS |
| `pre_trade_analytics/` | `ExecutionCostEstimator`, `LiquidityAnalyzer`, `RiskAnalyzer`, `VenueSelector` | Pre-trade analytics |
| `slippage_control/` | `AdaptiveExecutionEngine`, `MarketImpactModel`, `ImplementationShortfallOptimizer` | Slippage management |
| `latency_critical/` | `HighFrequencyExecutor`, `LatencyOptimizer`, `UltraLowLatencyRouter` | HFT infrastructure |

### 4.16 Analytics Layer (`analytics/`)

| Subdirectory | Key Components | Purpose |
|---|---|---|
| `backtesting/` | `BacktestEngine`, `EventDrivenBacktester`, `RegimeAwareBacktester`, `TransactionCostModel`, `WalkForwardAnalyzer` | Backtesting suite |
| `performance_attribution/` | `PerformanceAnalytics`, `PnLAnalyzer`, `RiskAdjustedMetrics`, `SharpeCalculator`, `DrawdownAnalyzer`, `RiskAttributionEngine` | Performance decomposition |
| `market_regime/` | `MarketRegimeDetector`, `HMMRegimeDetector`, `GARCHVolatilityAnalyzer`, `CorrelationRegimeDetector`, `ChangePointDetector` | Regime analysis |
| `alpha_research/` | `AlphaDiscovery`, `AlphaValidator`, `AlphaCombiner`, `AlphaDecayStudy` | Alpha lifecycle |
| `factor_research/` | `FactorConstructor`, `FactorSelector`, `FactorCombiner`, `FactorDecayAnalyzer` | Factor models |

### 4.17 Risk Management (`risk_management/`)

| File | Component | Purpose |
|---|---|---|
| `portfolio_risk_manager.py` | `PortfolioRiskManager` | Position limits, VaR, leverage, drawdown |

### 4.18 LLM Integration (`llm_integration/`) — READ-ONLY TRACK

| File | Component | Purpose |
|---|---|---|
| `duckdb_cache.py` | `TradingDataCache` | DuckDB analytical cache |
| `bridge.py` | `IntegrationBridge` | Read-only bridge (ZERO execution authority) |
| `vector_store.py` | `VectorStore` | Embedding storage for RAG |
| `llm_engine.py` | `QuantumForgeLLM` | LLM inference engine |
| `api.py` | API | LLM API endpoints |
| `event_stream.py` | EventStream | Real-time event streaming to LLM |
| `explanation_contracts.py` | ExplanationContracts | Structured LLM output contracts |

### 4.19 Core — Infrastructure & Audit (`core/`)

| File | Component | Purpose |
|---|---|---|
| `audit.py` | Hash-chained audit logger | Tamper-proof decision trail |
| `state_persistence.py` | `StatePersistence` | Save/restore system state |
| `storage_coordinator.py` | `StorageCoordinator` | Multi-backend storage (Parquet + Redis + CSV) |
| `analytics.py` | `AnalyticsEngine` | Performance analytics |
| `replay_engine.py` | `ReplayEngine` | Trade replay and analysis |
| `dynamic_portfolio_tracker.py` | `DynamicPortfolioTracker` | Live portfolio tracking |
| `alert_system.py` | `AlertSystem` | Regime change, trade, risk alerts |
| `health_monitor.py` | `HealthMonitor` | System health monitoring |
| `metrics_server.py` | `MetricsServer` | Prometheus metrics |
| `vi_bridge.py` | `VariationalInferenceBridge` | Bayesian streaming updates |
| `gp_bridge.py` | `GPPredictionBridge` | Gaussian process live predictions |
| `financial_llm_finetuner.py` | `FinancialLLMFineTuner` | LoRA/QLoRA fine-tuning scaffolding |
| `llm_structured_output.py` | `LLMOutputParser` | Type-safe LLM output parsing |

### 4.20 Infrastructure (`infrastructure/`)

| Subdirectory | Key Components | Purpose |
|---|---|---|
| `config/` | `ConfigurationManager` | YAML-based configuration |
| `scripts/` | `SystemMonitor`, `BackupManager`, `BenchmarkSuite`, `SystemDeployer` | DevOps tools |
| `deployment/` | Deployment configs | Production deployment |
| `monitoring/` | Monitoring setup | System monitoring |

### 4.21 Interface — 10 Dashboards (`pages/`)

| Dashboard | Purpose |
|---|---|
| Main Dashboard | Unified system overview |
| Trading Dashboard | Order execution & management |
| Risk Dashboard | Risk analytics & monitoring |
| Portfolio Dashboard | Holdings & performance tracking |
| Analytics Dashboard | Backtesting & attribution |
| Research Dashboard | Strategy development tools |
| Execution Dashboard | Order flow analysis |
| Market Microstructure | Orderbook visualization |
| Configuration | System settings & parameters |
| Investor Portal | Investor-facing reports |

---

## 5. DUAL-TRACK ARCHITECTURE

```
┌──────────────────────────────────────────────────────────────────────┐
│                    AUTHORITY BOUNDARY                                  │
│                                                                       │
│  LIVE EXECUTION TRACK                 RESEARCH / LLM TRACK           │
│  ═══════════════════                  ═══════════════════             │
│                                                                       │
│  Math Signal Generator                DuckDB Cache                    │
│  ML Ensemble                          Integration Bridge              │
│  Regime Detector                      Vector Store                    │
│  Risk Gate                            LLM Engine                      │
│  Execution Manager                    Explanation Contracts           │
│  Audit Logger                                                         │
│                                       ┌─────────────────┐            │
│  ─── EXECUTES TRADES ───             │  ZERO EXECUTION  │            │
│                                       │    AUTHORITY     │            │
│                                       └─────────────────┘            │
│                                                                       │
│  Authority flows DOWNWARD ONLY                                       │
│  LLM has READ-ONLY access to market data                             │
│  System MUST work with LLM_ENABLED=false                             │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 6. ENTRY POINTS

| Entry Point | File | Purpose |
|---|---|---|
| **Full System** | `run_full_system.py` | 135+ modules + Streamlit UI + Quantum Core |
| **Pipeline Only** | `launch_pipeline.py` | Core pipeline without full system overhead |
| **Quantum Core** | `launch_quantum_core.py` | Lightweight real pipeline (CLI flags) |
| **Dashboard Only** | `app.py` | Streamlit multi-page app (10 dashboards) |
| **Visual Launcher** | `launch.py` | Banner + status display → `run_full_system.py` |

---

## 7. INFRASTRUCTURE DEPENDENCIES

```
Docker Compose Stack:
  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
  │ TimescaleDB   │  │    Redis     │  │   DuckDB     │  │   Grafana    │
  │ (PostgreSQL)  │  │   (Cache)    │  │  (Analytics) │  │ (Dashboards) │
  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘

External APIs:
  - Binance WebSocket (wss://stream.binance.com:9443)
  - Binance REST API (https://api.binance.com/api/v3)
  - Alternative data feeds (configurable)
```

---

## 8. OPERATING MODES

| Mode | Config | Behaviour |
|---|---|---|
| **PAPER** | `ExecutionMode.PAPER` | Simulated fills with fee/slippage modelling |
| **LIVE** | `ExecutionMode.LIVE` | Real Binance API orders (requires API keys) |
| **ML ON** | `enable_ml=True` | Full ML ensemble (LSTM + Transformer + PPO + SAC + GP) |
| **ML OFF** | `enable_ml=False` / `--no-ml` | Math-only mode (Fourier + Stochastic + Wavelet) |
| **LLM ON** | `enable_llm=True` / `--llm` | LLM research track active (read-only) |
| **LLM OFF** | `enable_llm=False` | System works fully without LLM |

---

*Document auto-generated from codebase analysis of QUANTUM-FORGE/*
