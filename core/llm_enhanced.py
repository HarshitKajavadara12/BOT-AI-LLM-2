"""
QUANTUM-FORGE: LLM Enhancement Module
=======================================
P2 5.1 — Multi-model LLM support (fallback chain)
P2 5.2 — Fine-tuning interface  
P2 5.3 — LLM-driven alpha hypothesis generation
P2 5.5 — RAG expansion (more retrieval sources)

Architecture invariant: LLM is ALWAYS read-only.
These enhancements improve LLM *insight quality* without granting execution authority.
"""

import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("LLMEnhanced")


class MultiModelLLM:
    """
    Multi-model LLM with fallback chain.
    
    Tries models in priority order:
    1. Local Llama 3.2 8B (GGUF) — primary
    2. Mistral 7B — fallback
    3. Template engine — always works
    
    All models are read-only: they produce insights, never execute trades.
    """

    def __init__(self):
        self._models: List[Dict] = []
        self._active_model: Optional[str] = None
        self._initialize()

    def _initialize(self):
        """Try to load models in priority order."""
        # Model 1: Llama (primary)
        try:
            from llm_integration.llm_engine import get_llm_engine
            engine = get_llm_engine()
            if engine and engine.connected:
                self._models.append({
                    "name": "llama-3.2-8b",
                    "engine": engine,
                    "type": "local",
                })
                self._active_model = "llama-3.2-8b"
                logger.info("MultiModelLLM: Llama 3.2 available (primary)")
        except Exception as e:
            logger.debug(f"Llama not available: {e}")

        # Model 2: Future Mistral slot
        # self._models.append({...}) when Mistral GGUF is available

        # Model 3: Template fallback (always available)
        self._models.append({
            "name": "template",
            "engine": TemplateLLM(),
            "type": "template",
        })
        logger.info(f"MultiModelLLM: {len(self._models)} models in chain")

    def generate_insight(self, query: str, context: Dict) -> Dict:
        """Generate insight using fallback chain."""
        for model_info in self._models:
            try:
                engine = model_info["engine"]
                if model_info["type"] == "template":
                    return engine.generate(query, context)
                else:
                    result = engine.generate_trading_insight(
                        query=query,
                        rag_context=context,
                        portfolio_state=context.get("portfolio", {}),
                    )
                    if result:
                        return {
                            "model": model_info["name"],
                            "insight": result,
                            "timestamp": datetime.now().isoformat(),
                        }
            except Exception as e:
                logger.debug(f"Model {model_info['name']} failed: {e}")
                continue

        return {"model": "none", "insight": "All models unavailable", "timestamp": datetime.now().isoformat()}

    def get_active_model(self) -> str:
        return self._active_model or "template"


class TemplateLLM:
    """Template-based LLM fallback that always works."""

    def generate(self, query: str, context: Dict) -> Dict:
        """Generate template-based response."""
        regime = context.get("regime", "UNKNOWN")
        price = context.get("price", 0)
        symbol = context.get("symbol", "N/A")

        insight = (
            f"[Template Analysis] {symbol} at ${price:,.2f}. "
            f"Current regime: {regime}. "
            f"Query: {query[:100]}. "
            f"Recommendation: Follow quantitative signals. "
            f"Note: LLM models unavailable — using template fallback."
        )

        return {
            "model": "template",
            "insight": insight,
            "timestamp": datetime.now().isoformat(),
        }


# ─── LLM Alpha Hypothesis Generator (5.3) ───────────────────────────────

class AlphaHypothesisGenerator:
    """
    Uses LLM to generate alpha hypotheses from market data.
    
    These hypotheses are SUGGESTIONS only — they must pass through
    the alpha research pipeline for validation before going live.
    """

    PROMPT_TEMPLATE = """You are a quantitative researcher analyzing cryptocurrency markets.

Given the following market data:
- Symbol: {symbol}
- Current regime: {regime}  
- Recent returns: {returns}
- Volatility (20d): {volatility:.4f}
- RSI: {rsi:.2f}
- Correlation with BTC: {btc_corr:.3f}

Generate 3 potential alpha hypotheses that could be tested.
Each hypothesis should include:
1. A clear, testable statement
2. The expected holding period
3. The market condition under which it would work

Format: JSON array of objects with keys: hypothesis, holding_period, market_condition
"""

    def __init__(self, llm_engine=None):
        self._llm = llm_engine

    def generate_hypotheses(self, market_data: Dict) -> List[Dict]:
        """Generate alpha hypotheses from market data."""
        if self._llm is None:
            return self._default_hypotheses(market_data)

        prompt = self.PROMPT_TEMPLATE.format(
            symbol=market_data.get("symbol", "BTCUSDT"),
            regime=market_data.get("regime", "UNKNOWN"),
            returns=market_data.get("recent_returns", "N/A"),
            volatility=market_data.get("volatility", 0.02),
            rsi=market_data.get("rsi", 50),
            btc_corr=market_data.get("btc_correlation", 1.0),
        )

        try:
            if hasattr(self._llm, "generate_trading_insight"):
                result = self._llm.generate_trading_insight(
                    query=prompt,
                    rag_context={},
                    portfolio_state={},
                )
                return self._parse_hypotheses(result)
        except Exception as e:
            logger.debug(f"Alpha hypothesis generation failed: {e}")

        return self._default_hypotheses(market_data)

    def _parse_hypotheses(self, raw_text: str) -> List[Dict]:
        """Parse LLM output into structured hypotheses."""
        try:
            import re
            json_match = re.search(r'\[.*\]', raw_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except (json.JSONDecodeError, AttributeError):
            pass
        return self._default_hypotheses({})

    @staticmethod
    def _default_hypotheses(market_data: Dict) -> List[Dict]:
        """Default hypotheses when LLM is unavailable."""
        regime = market_data.get("regime", "UNKNOWN")
        return [
            {
                "hypothesis": "Mean-reversion in RSI extremes (>70 or <30) produces excess returns",
                "holding_period": "4-24 hours",
                "market_condition": "Range-bound markets",
            },
            {
                "hypothesis": f"Momentum factor strengthens in {regime} regime",
                "holding_period": "1-7 days",
                "market_condition": f"Current regime: {regime}",
            },
            {
                "hypothesis": "Cross-asset correlation breakdown signals upcoming volatility",
                "holding_period": "12-48 hours",
                "market_condition": "Regime transition periods",
            },
        ]


# ─── RAG Expansion (5.5) ────────────────────────────────────────────────

class ExpandedRAGContext:
    """
    Expanded RAG context provider that includes more data sources.
    
    Existing RAG: trades, signals, market analysis
    New sources:
      - Alpha store (active alphas and their performance)
      - Causal graph (current inter-asset relationships)
      - Health metrics (system status)
      - Feature pipeline stats
    """

    def __init__(self):
        self._sources: Dict[str, Any] = {}

    def register_source(self, name: str, provider: Any):
        """Register a data source for RAG context."""
        self._sources[name] = provider

    def get_expanded_context(self, query: str, symbol: str = "") -> Dict:
        """Build expanded RAG context from all sources."""
        context = {}

        for name, provider in self._sources.items():
            try:
                if hasattr(provider, "get_status"):
                    context[name] = provider.get_status()
                elif hasattr(provider, "get_summary"):
                    context[name] = provider.get_summary()
                elif hasattr(provider, "get_graph_summary"):
                    context[name] = provider.get_graph_summary()
                elif callable(provider):
                    context[name] = provider(query, symbol)
            except Exception as e:
                context[name] = {"error": str(e)}

        return context
