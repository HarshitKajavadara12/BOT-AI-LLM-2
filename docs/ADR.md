# Architecture Decision Records (ADRs)

## ADR-001: LLM Read-Only Architecture (Kill-Switch Invariant)

**Date:** 2026-01-15  
**Status:** Accepted  
**Context:** The system integrates an LLM (Llama 3.2 8B) for natural-language explanations and RAG queries. A key question is whether the LLM should be able to influence trading decisions.  
**Decision:** The LLM is strictly **read-only**. It can observe system state but has ZERO execution authority. A `CognitiveDampener` clamps LLM output to [0.2, 1.0] — it can only reduce risk, never increase it. The system MUST function identically with `LLM_ENABLED=false` (kill-switch invariant).  
**Rationale:** LLM hallucinations could cause catastrophic financial loss. A read-only constraint eliminates this risk class entirely while preserving explanatory value.  
**Consequences:** LLM cannot learn to improve execution in real-time; all alpha must come from the math/ML pipeline.

---

## ADR-002: Why Llama 3.2 8B Over GPT-4 / Claude

**Date:** 2026-01-15  
**Status:** Accepted  
**Context:** Multiple LLM options exist including proprietary APIs (OpenAI GPT-4, Anthropic Claude) and open-source models (Llama, Mistral, Gemma).  
**Decision:** Use Llama 3.2 8B (GGUF quantized) running locally via `llama-cpp-python`.  
**Rationale:**  
1. **Latency:** Local inference = 50-200ms vs 500-2000ms for API calls.  
2. **Cost:** Zero per-token cost; unlimited queries.  
3. **Reliability:** No external dependency. Network outages don't disable explanations.  
4. **Privacy:** Trading data never leaves the machine.  
5. **Determinism:** Controlled temperature and seed for reproducible outputs.  
**Consequences:** Lower capability than GPT-4 for complex reasoning. Mitigated by ExpandedRAGContext providing structured data alongside the query. Multi-model fallback chain added (Llama → Mistral → Template).

---

## ADR-003: Why Qdrant Over Pinecone / Weaviate / ChromaDB

**Date:** 2026-01-15  
**Status:** Accepted  
**Context:** Need a vector database for RAG semantic search across trades, signals, and market analysis.  
**Decision:** Qdrant (local mode, no server required).  
**Rationale:**  
1. **Local-first:** Runs in-process, no separate server. Aligns with our self-contained architecture.  
2. **Performance:** Sub-10ms queries on 384-dim vectors (all-MiniLM-L6-v2 embeddings).  
3. **Filtering:** Native payload filtering (by symbol, timestamp, signal_type).  
4. **Persistence:** On-disk persistence with WAL for crash recovery.  
5. **Python-native:** First-class Python client with typed models.  
**Consequences:** Cannot scale to billions of vectors like Pinecone. Acceptable since our collections have <100K documents each.

---

## ADR-004: Math-First Signal Generation (60/40 Math/ML Split)

**Date:** 2026-01-20  
**Status:** Accepted  
**Context:** System has both mathematical models (stochastic calculus, Fourier, wavelets, EVT) and ML models (LSTM, GRU, Transformer, RL). How should signals be combined?  
**Decision:** Mathematical signals get 50-60% base weight; ML gets 30%; cross-asset gets 20%. Dynamic re-weighting when components are unavailable.  
**Rationale:**  
1. Mathematical models are interpretable, deterministic, and don't need training data.  
2. ML models initially run with random weights until training pipeline executes.  
3. Math models degrade gracefully; ML can catastrophically fail.  
**Consequences:** System is conservative by default. ML weight increases as models prove themselves via walk-forward validation.

---

## ADR-005: Seven Database Backends with Graceful Degradation

**Date:** 2026-01-20  
**Status:** Accepted  
**Context:** The system needs time-series storage, hot caching, analytical queries, vector search, archival, and feature versioning.  
**Decision:** Use 7 specialised backends: TimescaleDB, Redis 7, DuckDB, Qdrant, Parquet, SQLite, Feature Store. Each is optional — system continues trading with Parquet alone.  
**Rationale:** Each backend excels at its specific access pattern. Parquet is always available (file-based, no server). The storage coordinator provides a unified API.  
**Consequences:** Complex deployment. Mitigated by docker-compose definitions and graceful fallback chains.

---

## ADR-006: Walk-Forward Training Over Static Train/Test

**Date:** 2026-02-01  
**Status:** Accepted  
**Context:** ML models need training. Standard train/test split causes lookahead bias in financial data.  
**Decision:** All ML training uses walk-forward validation: train on window [0, T], validate on [T, T+Δ], slide forward.  
**Rationale:** Prevents lookahead bias. Models are only evaluated on truly out-of-sample data. Matches how models would be deployed in production.  
**Consequences:** Slower training (multiple windows). More realistic performance estimates.

---

## ADR-007: 32-Feature Pipeline Over Raw Returns

**Date:** 2026-02-15  
**Status:** Accepted  
**Context:** Original ML ensemble received only last-10-returns as features. System has 100+ feature engineering modules.  
**Decision:** Build `FeaturePipeline` extracting 32 curated features: returns statistics (10), technical indicators (10), spectral features (5), microstructure features (5), volume features (2).  
**Rationale:** Balanced dimensionality — enough signal diversity without curse of dimensionality. Each feature group captures different market dynamics.  
**Consequences:** All model input dimensions must align to 32. MLEnsemble, TrainingPipeline, and SVMClassifier updated accordingly.

---

## ADR-008: Shiryaev-Roberts for Trade Exit Over Percentage Stops

**Date:** 2026-02-15  
**Status:** Accepted  
**Context:** Original exit logic used simple percentage-based stop-loss/take-profit. Sub-optimal from an information-theoretic perspective.  
**Decision:** Use Shiryaev-Roberts change-point detection + continuation-value estimation for optimal stopping. Hard stops retained as safety boundaries.  
**Rationale:** SR is the minimax-optimal sequential detection procedure. Combined with LSM-inspired continuation value, it produces exits that maximise expected holding PnL.  
**Consequences:** More complex exit logic. SR parameters (threshold, mu_0, mu_1) need calibration per symbol.

---

## ADR-009: Authority Flows Downward Only

**Date:** 2026-01-15  
**Status:** Accepted  
**Context:** System has multiple layers: Math Engine → Signal Generator → ML Ensemble → Risk Manager → Execution Manager → LLM (read-only).  
**Decision:** Authority flows strictly downward. Lower layers (execution) can never override upper layers (risk). LLM has zero authority.  
**Rationale:** Prevents dangerous feedback loops where execution urgency overrides risk limits. Kill-switch at any layer stops everything below.  
**Consequences:** System may miss opportunities when lower layers have valid information that upper layers don't. Acceptable trade-off for safety.

---

*Last updated: 2026-02-17*
