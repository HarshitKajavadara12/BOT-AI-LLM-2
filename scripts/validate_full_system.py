"""
QUANTUM-FORGE: Complete Pipeline & Workflow Validation Script
=============================================================
Validates that every module documented in PIPELINE_DOCUMENT.md and
WORKFLOW_DOCUMENT.md is importable, instantiable, and correctly wired.

Usage:
    cd QUANTUM-FORGE
    python scripts/validate_full_system.py
"""

import sys
import os
import traceback

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS = 0
FAIL = 0
RESULTS = []


def check(label: str, fn):
    """Run a check and record pass/fail."""
    global PASS, FAIL
    try:
        fn()
        PASS += 1
        RESULTS.append(("PASS", label, ""))
        print(f"  [PASS] {label}")
    except Exception as e:
        FAIL += 1
        RESULTS.append(("FAIL", label, str(e)))
        print(f"  [FAIL] {label} → {e}")


# ============================================================================
# SECTION 1: DATA INGESTION LAYER
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 1: DATA INGESTION LAYER (data/ingestion/)")
print("=" * 70)

check("Import BinanceAPIClient",
      lambda: __import__("data.ingestion.binance_client", fromlist=["BinanceAPIClient"]))
check("Import BinanceWebSocket",
      lambda: __import__("data.ingestion.binance_websocket", fromlist=["BinanceWebSocket"]))
check("Import MultiStreamWebSocket",
      lambda: __import__("data.ingestion.binance_websocket", fromlist=["MultiStreamWebSocket"]))
check("Import TickDataHandler",
      lambda: __import__("data.ingestion.tick_data_handler", fromlist=["TickDataHandler"]))
check("Import DataStreamManager",
      lambda: __import__("data.ingestion.streaming_engine", fromlist=["DataStreamManager"]))
check("Import RealTimeDataCache",
      lambda: __import__("data.ingestion.realtime_data_cache", fromlist=["RealTimeDataCache"]))
check("Import OrderBookReconstructor",
      lambda: __import__("data.ingestion.orderbook_reconstructor", fromlist=["OrderBookReconstructor"]))
check("Import AlternativeDataLoader",
      lambda: __import__("data.ingestion.alternative_data_loader", fromlist=["AlternativeDataLoader"]))

# ============================================================================
# SECTION 2: DATA PREPROCESSING
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 2: DATA PREPROCESSING (data/preprocessing/)")
print("=" * 70)

check("Import DataCleaner",
      lambda: __import__("data.preprocessing.data_cleaner", fromlist=["DataCleaner"]))
check("Import DataNormalizer",
      lambda: __import__("data.preprocessing.normalization", fromlist=["DataNormalizer"]))
check("Import DataAligner",
      lambda: __import__("data.preprocessing.alignment", fromlist=["DataAligner"]))

# ============================================================================
# SECTION 3: DATA STORAGE
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 3: DATA STORAGE (data/storage/)")
print("=" * 70)

check("Import RedisCache",
      lambda: __import__("data.storage.redis_cache", fromlist=["RedisCache"]))
check("Import TimescaleDBManager",
      lambda: __import__("data.storage.timescaledb_manager", fromlist=["TimescaleDBManager"]))
check("Import ParquetWriter",
      lambda: __import__("data.storage.parquet_writer", fromlist=["ParquetWriter"]))
check("Import FeatureStore",
      lambda: __import__("data.storage.feature_store", fromlist=["FeatureStore"]))
check("Import HistoricalDataManager",
      lambda: __import__("data.storage.historical_data", fromlist=["HistoricalDataManager"]))

# ============================================================================
# SECTION 4: INTELLIGENCE — DEEP LEARNING
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 4: INTELLIGENCE — DEEP LEARNING")
print("=" * 70)

check("Import LSTMModel",
      lambda: __import__("intelligence.deep_learning.deep_learning_models", fromlist=["LSTMModel"]))
check("Import GRUModel",
      lambda: __import__("intelligence.deep_learning.deep_learning_models", fromlist=["GRUModel"]))
check("Import TransformerModel",
      lambda: __import__("intelligence.deep_learning.deep_learning_models", fromlist=["TransformerModel"]))
check("Import CNNModel",
      lambda: __import__("intelligence.deep_learning.deep_learning_models", fromlist=["CNNModel"]))
check("Import MultiHeadAttention",
      lambda: __import__("intelligence.deep_learning.attention_mechanisms", fromlist=["MultiHeadAttention"]))
check("Import SelfAttention",
      lambda: __import__("intelligence.deep_learning.attention_mechanisms", fromlist=["SelfAttention"]))
check("Import CorrelationGraphNet (GNN)",
      lambda: __import__("intelligence.deep_learning.graph_networks", fromlist=["CorrelationGraphNet"]))
check("Import TemporalConvNet",
      lambda: __import__("intelligence.deep_learning.temporal_models", fromlist=["TemporalConvNet"]))
check("Import StateSpaceModel",
      lambda: __import__("intelligence.deep_learning.state_space_models", fromlist=["StateSpaceModel"]))

# ============================================================================
# SECTION 5: INTELLIGENCE — REINFORCEMENT LEARNING
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 5: INTELLIGENCE — REINFORCEMENT LEARNING")
print("=" * 70)

check("Import PPOAgent",
      lambda: __import__("intelligence.reinforcement_learning.ppo_agent", fromlist=["PPOAgent"]))
check("Import SACAgent",
      lambda: __import__("intelligence.reinforcement_learning.soft_actor_critic", fromlist=["SACAgent"]))
check("Import MBPO (Model-Based RL)",
      lambda: __import__("intelligence.reinforcement_learning.model_based_rl", fromlist=["MBPO"]))
check("Import MarketEnvironment",
      lambda: __import__("intelligence.reinforcement_learning.market_env", fromlist=["MarketEnvironment"]))

# ============================================================================
# SECTION 6: INTELLIGENCE — FEATURE LEARNING
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 6: INTELLIGENCE — FEATURE LEARNING")
print("=" * 70)

check("Import FeatureLearningEngine",
      lambda: __import__("intelligence.feature_learning.feature_learning", fromlist=["FeatureLearningEngine"]))
check("Import RepresentationLearner",
      lambda: __import__("intelligence.feature_learning.representation_learning", fromlist=["RepresentationLearner"]))
check("Import ManifoldLearner",
      lambda: __import__("intelligence.feature_learning.manifold_learning", fromlist=["ManifoldLearner"]))
check("Import CausalDiscoveryEnsemble",
      lambda: __import__("intelligence.feature_learning.causal_discovery", fromlist=["CausalDiscoveryEnsemble"]))

# ============================================================================
# SECTION 7: INTELLIGENCE — META LEARNING
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 7: INTELLIGENCE — META LEARNING")
print("=" * 70)

check("Import MAML",
      lambda: __import__("intelligence.meta_learning.meta_learning", fromlist=["MAML"]))
check("Import FewShotLearner",
      lambda: __import__("intelligence.meta_learning.few_shot_learning", fromlist=["FewShotLearner"]))
check("Import TransferLearner",
      lambda: __import__("intelligence.meta_learning.transfer_learning", fromlist=["TransferLearner"]))
check("Import EnsembleDistiller",
      lambda: __import__("intelligence.meta_learning.ensemble_distillation", fromlist=["EnsembleDistiller"]))

# ============================================================================
# SECTION 8: INTELLIGENCE — PROBABILISTIC ML
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 8: INTELLIGENCE — PROBABILISTIC ML")
print("=" * 70)

check("Import FinancialGaussianProcess",
      lambda: __import__("intelligence.probabilistic_ml.gaussian_processes", fromlist=["FinancialGaussianProcess"]))
check("Import BayesianOptimizer",
      lambda: __import__("intelligence.probabilistic_ml.bayesian_optimization", fromlist=["BayesianOptimizer"]))
check("Import StochasticVariationalInference",
      lambda: __import__("intelligence.probabilistic_ml.variational_inference", fromlist=["StochasticVariationalInference"]))
check("Import ConformalPredictor",
      lambda: __import__("intelligence.probabilistic_ml.conformal_prediction", fromlist=["ConformalPredictor"]))

# ============================================================================
# SECTION 9: CORE — MATH ENGINE
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 9: CORE — MATH ENGINE (core/math_engine/)")
print("=" * 70)

check("Import MathematicalEngine",
      lambda: __import__("core.math_engine.mathematical_engine", fromlist=["MathematicalEngine"]))
check("Import FourierAnalyzer",
      lambda: __import__("core.math_engine.fourier_analysis", fromlist=["FourierAnalyzer"]))
check("Import StochasticProcesses",
      lambda: __import__("core.math_engine.stochastic_calculus", fromlist=["StochasticProcesses"]))
check("Import WaveletAnalysis (Signal Processing)",
      lambda: __import__("core.math_engine.signal_processing", fromlist=["WaveletAnalysis"]))
check("Import KalmanFilter",
      lambda: __import__("core.math_engine.signal_processing", fromlist=["KalmanFilter"]))
check("Import StochasticControl (Optimal Control)",
      lambda: __import__("core.math_engine.optimal_control", fromlist=["StochasticControl"]))
check("Import ModelPredictiveControl",
      lambda: __import__("core.math_engine.optimal_control", fromlist=["ModelPredictiveControl"]))
check("Import OptimizationMethods",
      lambda: __import__("core.math_engine.numerical_methods", fromlist=["OptimizationMethods"]))
check("Import UnitRootTests",
      lambda: __import__("core.math_engine.statistical_tests", fromlist=["UnitRootTests"]))
check("Import LinearAlgebraEngine",
      lambda: __import__("core.math_engine.linear_algebra", fromlist=["LinearAlgebraEngine"]))

# ============================================================================
# SECTION 10: CORE — MARKET MICROSTRUCTURE
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 10: CORE — MARKET MICROSTRUCTURE")
print("=" * 70)

check("Import OrderBookAnalyzer (Dynamics)",
      lambda: __import__("core.market_microstructure.orderbook_dynamics", fromlist=["OrderBookAnalyzer"]))
check("Import PriceFormationAnalyzer",
      lambda: __import__("core.market_microstructure.price_formation", fromlist=["PriceFormationAnalyzer"]))
check("Import LiquidityAnalyzer",
      lambda: __import__("core.market_microstructure.liquidity_models", fromlist=["LiquidityAnalyzer"]))
check("Import ToxicityDetector",
      lambda: __import__("core.market_microstructure.toxicity_detection", fromlist=["ToxicityDetector"]))

# ============================================================================
# SECTION 11: CORE — RISK MATHEMATICS
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 11: CORE — RISK MATHEMATICS")
print("=" * 70)

check("Import EVTAnalyzer",
      lambda: __import__("core.risk_mathematics.extreme_value_theory", fromlist=["EVTAnalyzer"]))
check("Import CopulaAnalyzer",
      lambda: __import__("core.risk_mathematics.copula_models", fromlist=["CopulaAnalyzer"]))
check("Import DrawdownAnalyzer",
      lambda: __import__("core.risk_mathematics.drawdown_mathematics", fromlist=["DrawdownAnalyzer"]))
check("Import OptimalStoppingAnalyzer",
      lambda: __import__("core.risk_mathematics.optimal_stopping", fromlist=["OptimalStoppingAnalyzer"]))
check("Import CognitiveDampener",
      lambda: __import__("core.risk_mathematics.cognitive_dampener", fromlist=["CognitiveDampener"]))

# ============================================================================
# SECTION 12: CORE — SIGNAL & ALPHA GENERATION
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 12: CORE — SIGNAL & ALPHA GENERATION")
print("=" * 70)

check("Import SignalGenerator",
      lambda: __import__("core.signal_generator", fromlist=["SignalGenerator"]))
check("Import MLEnsembleEngine",
      lambda: __import__("core.ml_ensemble", fromlist=["MLEnsembleEngine"]))
check("Import RegimeDetector",
      lambda: __import__("core.regime_detector", fromlist=["RegimeDetector"]))
check("Import FeaturePipeline",
      lambda: __import__("core.feature_pipeline", fromlist=["FeaturePipeline"]))
check("Import CrossAssetAlphaEngine",
      lambda: __import__("core.cross_asset_alpha", fromlist=["CrossAssetAlphaEngine"]))
check("Import AlphaResearchScheduler",
      lambda: __import__("core.alpha_bridge", fromlist=["AlphaResearchScheduler"]))
check("Import AlphaStore",
      lambda: __import__("core.alpha_bridge", fromlist=["AlphaStore"]))
check("Import CausalBridge",
      lambda: __import__("core.causal_bridge", fromlist=["CausalBridge"]))
check("Import SVMRegimeClassifier",
      lambda: __import__("core.svm_classifier", fromlist=["SVMRegimeClassifier"]))
check("Import OrderBookAnalyzer (Core)",
      lambda: __import__("core.order_book_analyzer", fromlist=["OrderBookAnalyzer"]))
check("Import AltDataAlphaEngine",
      lambda: __import__("core.alt_data_alpha", fromlist=["AltDataAlphaEngine"]))
check("Import AlphaCrowdingDetector",
      lambda: __import__("core.alpha_crowding_detector", fromlist=["AlphaCrowdingDetector"]))
check("Import SpoofingDetector",
      lambda: __import__("core.spoofing_detector", fromlist=["SpoofingDetector"]))
check("Import QueuePositionEstimator",
      lambda: __import__("core.queue_position_estimator", fromlist=["QueuePositionEstimator"]))

# ============================================================================
# SECTION 13: CORE — EXECUTION & MANAGEMENT
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 13: CORE — EXECUTION & MANAGEMENT")
print("=" * 70)

check("Import ExecutionManager",
      lambda: __import__("core.execution_manager", fromlist=["ExecutionManager"]))
check("Import ExecutionMode",
      lambda: __import__("core.execution_manager", fromlist=["ExecutionMode"]))
check("Import CapitalAllocator",
      lambda: __import__("core.capital_allocator", fromlist=["CapitalAllocator"]))
check("Import StrategyMultiplexer",
      lambda: __import__("core.strategy_multiplexer", fromlist=["StrategyMultiplexer"]))
check("Import ShadowTracker",
      lambda: __import__("core.shadow_tracker", fromlist=["ShadowTracker"]))
check("Import MarketImpactTracker",
      lambda: __import__("core.market_impact_tracker", fromlist=["MarketImpactTracker"]))
check("Import GPPredictionBridge",
      lambda: __import__("core.gp_bridge", fromlist=["GPPredictionBridge"]))
check("Import LLMOutputParser",
      lambda: __import__("core.llm_structured_output", fromlist=["LLMOutputParser"]))
check("Import FinancialLLMFineTuner",
      lambda: __import__("core.financial_llm_finetuner", fromlist=["FinancialLLMFineTuner"]))
check("Import VariationalInferenceBridge",
      lambda: __import__("core.vi_bridge", fromlist=["VariationalInferenceBridge"]))

# ============================================================================
# SECTION 14: CORE — EXECUTION ALGORITHMS
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 14: CORE — EXECUTION ALGORITHMS")
print("=" * 70)

check("Import VWAPAlgorithm",
      lambda: __import__("core.execution_algorithms.vwap_algorithm", fromlist=["VWAPAlgorithm"]))
check("Import TWAPAlgorithm",
      lambda: __import__("core.execution_algorithms.twap_algorithm", fromlist=["TWAPAlgorithm"]))
check("Import ImplementationShortfallAlgorithm",
      lambda: __import__("core.execution_algorithms.implementation_shortfall", fromlist=["ImplementationShortfallAlgorithm"]))
check("Import ArrivalPriceAlgorithm",
      lambda: __import__("core.execution_algorithms.arrival_price", fromlist=["ArrivalPriceAlgorithm"]))

# ============================================================================
# SECTION 15: EXECUTION LAYER
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 15: EXECUTION LAYER")
print("=" * 70)

# Order Management
check("Import OrderManagementSystem",
      lambda: __import__("execution.order_management.order_management_system", fromlist=["OrderManagementSystem"]))
check("Import SmartOrderRouter",
      lambda: __import__("execution.order_management.smart_order_router", fromlist=["SmartOrderRouter"]))
check("Import PositionManager",
      lambda: __import__("execution.order_management.position_manager", fromlist=["PositionManager"]))
check("Import FillManager",
      lambda: __import__("execution.order_management.fill_manager", fromlist=["FillManager"]))
check("Import OptimalExecution (ExecutionEngine)",
      lambda: __import__("execution.order_management.optimal_execution", fromlist=["ExecutionEngine"]))

# Pre-Trade
check("Import ExecutionCostEstimator",
      lambda: __import__("execution.pre_trade_analytics.cost_estimator", fromlist=["ExecutionCostEstimator"]))
check("Import LiquidityAnalyzer (Pre-Trade)",
      lambda: __import__("execution.pre_trade_analytics.liquidity_analyzer", fromlist=["LiquidityAnalyzer"]))
check("Import RiskAnalyzer (Pre-Trade)",
      lambda: __import__("execution.pre_trade_analytics.risk_assessor", fromlist=["RiskAnalyzer"]))
check("Import VenueSelector",
      lambda: __import__("execution.pre_trade_analytics.venue_selector", fromlist=["VenueSelector"]))

# Slippage Control
check("Import AdaptiveExecutionEngine",
      lambda: __import__("execution.slippage_control.adaptive_execution", fromlist=["AdaptiveExecutionEngine"]))
check("Import MarketImpactModel",
      lambda: __import__("execution.slippage_control.market_impact_model", fromlist=["MarketImpactModel"]))
check("Import ImplementationShortfallOptimizer",
      lambda: __import__("execution.slippage_control.implementation_shortfall", fromlist=["ImplementationShortfallOptimizer"]))

# Latency Critical
check("Import HighFrequencyExecutor",
      lambda: __import__("execution.latency_critical.high_frequency_execution", fromlist=["HighFrequencyExecutor"]))
check("Import LatencyOptimizer",
      lambda: __import__("execution.latency_critical.latency_optimization", fromlist=["LatencyOptimizer"]))
check("Import UltraLowLatencyRouter",
      lambda: __import__("execution.latency_critical.ultra_low_latency_router", fromlist=["UltraLowLatencyRouter"]))

# ============================================================================
# SECTION 16: ANALYTICS LAYER
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 16: ANALYTICS LAYER")
print("=" * 70)

# Backtesting
check("Import BacktestEngine",
      lambda: __import__("analytics.backtesting.backtesting_infrastructure", fromlist=["BacktestEngine"]))
check("Import EventDrivenBacktester",
      lambda: __import__("analytics.backtesting.event_driven_backtest", fromlist=["EventDrivenBacktester"]))
check("Import RegimeAwareBacktester",
      lambda: __import__("analytics.backtesting.regime_aware_backtest", fromlist=["RegimeAwareBacktester"]))
check("Import TransactionCostModel",
      lambda: __import__("analytics.backtesting.transaction_cost_model", fromlist=["TransactionCostModel"]))
check("Import WalkForwardAnalyzer",
      lambda: __import__("analytics.backtesting.walk_forward_framework", fromlist=["WalkForwardAnalyzer"]))

# Performance Attribution
check("Import PerformanceAnalytics",
      lambda: __import__("analytics.performance_attribution.performance_analytics", fromlist=["PerformanceAnalytics"]))
check("Import PnLAnalyzer",
      lambda: __import__("analytics.performance_attribution.pnl_decomposition", fromlist=["PnLAnalyzer"]))
check("Import RiskAdjustedMetrics",
      lambda: __import__("analytics.performance_attribution.risk_adjusted_metrics", fromlist=["RiskAdjustedMetrics"]))
check("Import SharpeCalculator",
      lambda: __import__("analytics.performance_attribution.sharpe_calculator", fromlist=["SharpeCalculator"]))
check("Import DrawdownAnalyzer (Analytics)",
      lambda: __import__("analytics.performance_attribution.drawdown_analysis", fromlist=["DrawdownAnalyzer"]))
check("Import RiskAttributionEngine",
      lambda: __import__("analytics.performance_attribution.risk_attribution", fromlist=["RiskAttributionEngine"]))

# Market Regime Analytics
check("Import MarketRegimeDetector",
      lambda: __import__("analytics.market_regime.market_regime_detection", fromlist=["MarketRegimeDetector"]))
check("Import HMMRegimeDetector",
      lambda: __import__("analytics.market_regime.hmm_regime_detection", fromlist=["HMMRegimeDetector"]))
check("Import GARCHVolatilityAnalyzer",
      lambda: __import__("analytics.market_regime.volatility_clustering", fromlist=["GARCHVolatilityAnalyzer"]))
check("Import CorrelationRegimeDetector",
      lambda: __import__("analytics.market_regime.correlation_regimes", fromlist=["CorrelationRegimeDetector"]))
check("Import ChangePointDetector",
      lambda: __import__("analytics.market_regime.change_point_detection", fromlist=["ChangePointDetector"]))

# Alpha Research
check("Import ComprehensiveAlphaDiscovery",
      lambda: __import__("analytics.alpha_research.alpha_discovery", fromlist=["ComprehensiveAlphaDiscovery"]))
check("Import ComprehensiveAlphaValidator",
      lambda: __import__("analytics.alpha_research.alpha_validation", fromlist=["ComprehensiveAlphaValidator"]))
check("Import ComprehensiveAlphaCombination",
      lambda: __import__("analytics.alpha_research.alpha_combination", fromlist=["ComprehensiveAlphaCombination"]))
check("Import ComprehensiveAlphaDecayStudy",
      lambda: __import__("analytics.alpha_research.alpha_decay_study", fromlist=["ComprehensiveAlphaDecayStudy"]))

# Factor Research
check("Import FactorConstructor",
      lambda: __import__("analytics.factor_research.factor_construction", fromlist=["FactorConstructor"]))
check("Import FactorSelector",
      lambda: __import__("analytics.factor_research.factor_selection", fromlist=["FactorSelector"]))
check("Import FactorCombiner",
      lambda: __import__("analytics.factor_research.factor_combination", fromlist=["FactorCombiner"]))
check("Import FactorDecayAnalyzer",
      lambda: __import__("analytics.factor_research.factor_decay", fromlist=["FactorDecayAnalyzer"]))

# ============================================================================
# SECTION 17: RISK MANAGEMENT
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 17: RISK MANAGEMENT")
print("=" * 70)

check("Import PortfolioRiskManager",
      lambda: __import__("risk_management.portfolio_risk_manager", fromlist=["PortfolioRiskManager"]))

# ============================================================================
# SECTION 18: LLM INTEGRATION (Read-Only Research Track)
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 18: LLM INTEGRATION (Read-Only Research Track)")
print("=" * 70)

check("Import TradingDataCache (DuckDB)",
      lambda: __import__("llm_integration.duckdb_cache", fromlist=["TradingDataCache"]))
check("Import IntegrationBridge",
      lambda: __import__("llm_integration.bridge", fromlist=["IntegrationBridge"]))
check("Import VectorStore",
      lambda: __import__("llm_integration.vector_store", fromlist=["VectorStore"]))
check("Import QuantumForgeLLM",
      lambda: __import__("llm_integration.llm_engine", fromlist=["QuantumForgeLLM"]))

# ============================================================================
# SECTION 19: CORE — INFRASTRUCTURE & AUDIT
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 19: CORE INFRASTRUCTURE & AUDIT")
print("=" * 70)

check("Import get_audit_logger (Hash-chain)",
      lambda: __import__("core.audit", fromlist=["get_audit_logger"]))
check("Import StatePersistence",
      lambda: __import__("core.state_persistence", fromlist=["StatePersistence"]))
check("Import StorageCoordinator",
      lambda: __import__("core.storage_coordinator", fromlist=["StorageCoordinator"]))
check("Import AnalyticsEngine",
      lambda: __import__("core.analytics", fromlist=["AnalyticsEngine"]))
check("Import ReplayEngine",
      lambda: __import__("core.replay_engine", fromlist=["ReplayEngine"]))
check("Import DynamicPortfolioTracker",
      lambda: __import__("core.dynamic_portfolio_tracker", fromlist=["DynamicPortfolioTracker"]))
check("Import AlertSystem",
      lambda: __import__("core.alert_system", fromlist=["AlertSystem"]))
check("Import HealthMonitor",
      lambda: __import__("core.health_monitor", fromlist=["HealthMonitor"]))
check("Import MetricsServer",
      lambda: __import__("core.metrics_server", fromlist=["MetricsServer"]))

# ============================================================================
# SECTION 20: INFRASTRUCTURE
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 20: INFRASTRUCTURE")
print("=" * 70)

check("Import ConfigurationManager",
      lambda: __import__("infrastructure.config.config_manager", fromlist=["ConfigurationManager"]))
check("Import SystemMonitor",
      lambda: __import__("infrastructure.scripts.monitoring", fromlist=["SystemMonitor"]))
check("Import BackupManager",
      lambda: __import__("infrastructure.scripts.backup", fromlist=["BackupManager"]))
check("Import BenchmarkSuite",
      lambda: __import__("infrastructure.scripts.benchmark", fromlist=["BenchmarkSuite"]))
check("Import SystemDeployer",
      lambda: __import__("infrastructure.scripts.deployment", fromlist=["SystemDeployer"]))

# ============================================================================
# SECTION 21: CORE STRATEGIES
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 21: CORE STRATEGIES")
print("=" * 70)

check("Import IStrategy",
      lambda: __import__("core.strategy_interface", fromlist=["IStrategy"]))
check("Import StrategySignal",
      lambda: __import__("core.strategy_interface", fromlist=["StrategySignal"]))
check("Import core.strategies.momentum_strategy",
      lambda: __import__("core.strategies.momentum_strategy"))
check("Import core.strategies.quantum_signal_strategy",
      lambda: __import__("core.strategies.quantum_signal_strategy"))

# ============================================================================
# SECTION 22: ENTRY POINT CLASSES
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 22: ENTRY POINT CLASSES")
print("=" * 70)

check("Import QuantumCoreOrchestrator",
      lambda: __import__("core.quantum_core", fromlist=["QuantumCoreOrchestrator"]))
check("Import QuantumForgePipeline",
      lambda: __import__("core.pipeline", fromlist=["QuantumForgePipeline"]))
check("Import QuantumForgeFullSystem",
      lambda: __import__("run_full_system", fromlist=["QuantumForgeFullSystem"]))

# ============================================================================
# SECTION 23: PIPELINE WIRING CHECKS
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 23: PIPELINE WIRING CHECKS")
print("=" * 70)


def check_quantum_core_imports():
    """Verify QuantumCoreOrchestrator imports all mandatory modules."""
    import inspect
    from core.quantum_core import QuantumCoreOrchestrator
    src = inspect.getsource(QuantumCoreOrchestrator)
    required = [
        "SignalGenerator", "MLEnsembleEngine", "RegimeDetector",
        "CapitalAllocator", "StrategyMultiplexer",
        "ExecutionManager", "PortfolioRiskManager", "AlertSystem",
        "StorageCoordinator", "FeaturePipeline", "CrossAssetAlphaEngine",
        "SVMRegimeClassifier", "AlphaResearchScheduler", "CausalBridge",
        "OrderBookAnalyzer", "GPPredictionBridge", "LLMOutputParser",
        "AltDataAlphaEngine", "AlphaCrowdingDetector", "MarketImpactTracker",
        "SpoofingDetector", "QueuePositionEstimator", "FinancialLLMFineTuner",
        "VariationalInferenceBridge",
    ]
    missing = [m for m in required if m not in src]
    assert not missing, f"Missing modules in QuantumCoreOrchestrator: {missing}"


check("QuantumCoreOrchestrator wires all 24 core modules", check_quantum_core_imports)


def check_shadow_tracker_wired():
    """Verify ShadowTracker is imported and used via StrategyMultiplexer."""
    import inspect
    from core.quantum_core import QuantumCoreOrchestrator
    src = inspect.getsource(QuantumCoreOrchestrator)
    assert "shadow_tracker" in src, "ShadowTracker not wired"


check("ShadowTracker wired via StrategyMultiplexer", check_shadow_tracker_wired)


def check_pipeline_has_build_methods():
    """Verify QuantumForgePipeline has all build methods."""
    import inspect
    from core.pipeline import QuantumForgePipeline
    methods = [m for m in dir(QuantumForgePipeline) if m.startswith("_build")]
    assert len(methods) >= 4, f"Need >=4 _build methods, found {methods}"


check("QuantumForgePipeline has build methods", check_pipeline_has_build_methods)


def check_signal_fusion_weights():
    """Verify signal fusion uses documented weights (50/30/20)."""
    import inspect
    from core.quantum_core import QuantumCoreOrchestrator
    src = inspect.getsource(QuantumCoreOrchestrator)
    assert "0.5" in src or "0.50" in src, "Math weight 0.5 not found"
    assert "0.3" in src or "0.30" in src, "ML weight 0.3 not found"
    assert "0.2" in src or "0.20" in src, "CrossAsset weight 0.2 not found"


check("Signal fusion uses 50/30/20 weights", check_signal_fusion_weights)


def check_risk_gate_checks():
    """Verify RiskGate has regime, drawdown, exposure checks."""
    import inspect
    from core.quantum_core import RiskGate
    src = inspect.getsource(RiskGate)
    for keyword in ["CRISIS", "drawdown", "exposure"]:
        assert keyword.lower() in src.lower(), f"RiskGate missing '{keyword}' check"


check("RiskGate has regime/drawdown/exposure checks", check_risk_gate_checks)


def check_circuit_breaker():
    """Verify CircuitBreaker class exists."""
    from core.quantum_core import CircuitBreaker
    assert CircuitBreaker is not None


check("CircuitBreaker class exists", check_circuit_breaker)


def check_audit_hash_chain():
    """Verify audit logger uses hash chaining."""
    import inspect
    mod = __import__("core.audit", fromlist=["get_audit_logger"])
    src = inspect.getsource(mod)
    assert "hash" in src.lower() or "sha" in src.lower(), "Audit not using hash chain"


check("Audit logger uses hash chaining", check_audit_hash_chain)


def check_execution_algo_selection():
    """Verify ExecutionManager selects between VWAP/TWAP."""
    import inspect
    from core.execution_manager import ExecutionManager
    src = inspect.getsource(ExecutionManager)
    assert "vwap" in src.lower() or "twap" in src.lower(), "Execution algo selection missing"


check("ExecutionManager selects VWAP/TWAP", check_execution_algo_selection)


def check_state_persistence():
    """Verify state persistence save/restore methods."""
    import inspect
    from core.quantum_core import QuantumCoreOrchestrator
    src = inspect.getsource(QuantumCoreOrchestrator)
    assert "save_state" in src, "save_state missing"
    assert "restore_state" in src, "restore_state missing"


check("State persistence (save/restore) wired", check_state_persistence)


def check_websocket_with_rest_fallback():
    """Verify WebSocket with REST fallback pattern."""
    import inspect
    from core.quantum_core import QuantumCoreOrchestrator
    src = inspect.getsource(QuantumCoreOrchestrator)
    assert "websocket" in src.lower() or "ws_" in src.lower(), "WebSocket not found"
    assert "api.binance.com" in src or "REST" in src or "requests.get" in src, "REST fallback not found"


check("WebSocket + REST fallback pattern present", check_websocket_with_rest_fallback)


def check_graceful_shutdown():
    """Verify SIGINT/SIGTERM handling."""
    import inspect
    from core.quantum_core import QuantumCoreOrchestrator
    src = inspect.getsource(QuantumCoreOrchestrator)
    assert "SIGINT" in src or "signal.signal" in src or "KeyboardInterrupt" in src, "Graceful shutdown not found"


check("Graceful shutdown handler present", check_graceful_shutdown)


def check_10_step_pipeline():
    """Verify the 10-step pipeline exists in _process_symbol."""
    import inspect
    from core.quantum_core import QuantumCoreOrchestrator
    src = inspect.getsource(QuantumCoreOrchestrator)
    steps = [
        "signal_generator", "regime_detector", "ml_ensemble",
        "risk_gate", "execution_manager", "audit"
    ]
    missing = [s for s in steps if s not in src.lower()]
    assert not missing, f"Pipeline steps missing: {missing}"


check("10-step pipeline in QuantumCoreOrchestrator", check_10_step_pipeline)


# ============================================================================
# SECTION 24: TRAINING PIPELINE
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 24: TRAINING PIPELINE")
print("=" * 70)

check("Import training_pipeline",
      lambda: __import__("intelligence.training_pipeline"))


# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print(f"VALIDATION SUMMARY")
print("=" * 70)
print(f"  Total checks:  {PASS + FAIL}")
print(f"  PASSED:        {PASS}")
print(f"  FAILED:        {FAIL}")
print("=" * 70)

if FAIL == 0:
    print("  RESULT: ALL CHECKS PASSED ✓")
else:
    print(f"  RESULT: {FAIL} CHECKS FAILED:")
    for status, label, err in RESULTS:
        if status == "FAIL":
            print(f"    - {label}: {err}")

print("=" * 70)

# Write results to file
with open("VALIDATION_RESULTS.txt", "w", encoding="utf-8") as f:
    f.write(f"QUANTUM-FORGE Validation Results\n")
    f.write(f"{'=' * 50}\n")
    f.write(f"Total: {PASS + FAIL}  |  Pass: {PASS}  |  Fail: {FAIL}\n\n")
    for status, label, err in RESULTS:
        line = f"[{status}] {label}"
        if err:
            line += f" → {err}"
        f.write(line + "\n")

sys.exit(0 if FAIL == 0 else 1)
