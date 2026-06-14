"""
QUANTUM-FORGE: Structured LLM Output Parser
=============================================
Ensures LLM responses conform to Pydantic schemas with validated fields.

The LLM produces free text → this module parses it into structured objects.
Falls back gracefully to template values if parsing fails.
"""

import json
import re
import logging
from typing import Dict, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum

logger = logging.getLogger("LLMStructuredOutput")


class LLMSentiment(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class StructuredSignalAnalysis:
    """Structured output from LLM signal analysis."""
    sentiment: LLMSentiment = LLMSentiment.NEUTRAL
    confidence: float = 0.5          # [0, 1]
    key_factors: list = field(default_factory=list)   # Up to 5 factors
    risk_assessment: str = "MODERATE"  # LOW, MODERATE, HIGH, EXTREME
    recommendation: str = "HOLD"       # BUY, SELL, HOLD
    reasoning: str = ""
    time_horizon: str = "SHORT"        # SHORT, MEDIUM, LONG
    parsed_successfully: bool = False

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["sentiment"] = self.sentiment.value
        return d


@dataclass
class StructuredMarketThought:
    """Structured output from LLM market thought generation."""
    summary: str = ""
    regime_assessment: str = "NEUTRAL"
    top_opportunities: list = field(default_factory=list)
    top_risks: list = field(default_factory=list)
    parsed_successfully: bool = False

    def to_dict(self) -> Dict:
        return asdict(self)


class LLMOutputParser:
    """
    Parses raw LLM text into structured Pydantic-like dataclass objects.
    
    Uses multiple parsing strategies:
    1. JSON extraction (if LLM outputs JSON)
    2. Key-value parsing (e.g., "Sentiment: BULLISH")
    3. Heuristic text analysis (fallback)
    """

    # ─── Signal Analysis Parsing ─────────────────────────────────

    @staticmethod
    def parse_signal_analysis(
        raw_text: str,
        fallback_data: Optional[Dict] = None,
    ) -> StructuredSignalAnalysis:
        """
        Parse LLM signal analysis output into structured format.
        
        Args:
            raw_text: Raw LLM output text
            fallback_data: Fallback values from the signal generator
        """
        result = StructuredSignalAnalysis()
        fallback = fallback_data or {}

        if not raw_text or not raw_text.strip():
            result.reasoning = "No LLM output available"
            result.confidence = fallback.get("confidence", 0.5)
            result.recommendation = fallback.get("signal_type", "HOLD")
            return result

        # Strategy 1: Try JSON extraction
        json_result = LLMOutputParser._try_json_parse(raw_text)
        if json_result:
            return LLMOutputParser._fill_from_dict(json_result, fallback)

        # Strategy 2: Key-value parsing
        kv = LLMOutputParser._extract_key_values(raw_text)
        if kv:
            result = LLMOutputParser._fill_from_kv(kv, fallback)
            result.reasoning = raw_text[:500]
            result.parsed_successfully = True
            return result

        # Strategy 3: Heuristic text analysis
        result = LLMOutputParser._heuristic_parse(raw_text, fallback)
        return result

    # ─── Market Thought Parsing ──────────────────────────────────

    @staticmethod
    def parse_market_thought(raw_text: str) -> StructuredMarketThought:
        """Parse LLM market thought into structured format."""
        result = StructuredMarketThought()

        if not raw_text or not raw_text.strip():
            return result

        result.summary = raw_text[:500]

        # Extract regime assessment
        text_lower = raw_text.lower()
        if any(w in text_lower for w in ["bull", "uptrend", "rally", "positive"]):
            result.regime_assessment = "BULLISH"
        elif any(w in text_lower for w in ["bear", "downtrend", "crash", "negative"]):
            result.regime_assessment = "BEARISH"
        elif any(w in text_lower for w in ["volatile", "uncertain", "choppy"]):
            result.regime_assessment = "HIGH_VOLATILITY"
        else:
            result.regime_assessment = "NEUTRAL"

        # Extract opportunities/risks (simple sentence extraction)
        sentences = re.split(r'[.!?]', raw_text)
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            s_lower = s.lower()
            if any(w in s_lower for w in ["opportunity", "bullish", "support", "buy"]):
                if len(result.top_opportunities) < 3:
                    result.top_opportunities.append(s[:200])
            elif any(w in s_lower for w in ["risk", "bearish", "resistance", "sell", "danger"]):
                if len(result.top_risks) < 3:
                    result.top_risks.append(s[:200])

        result.parsed_successfully = True
        return result

    # ─── Internal Parsing Strategies ─────────────────────────────

    @staticmethod
    def _try_json_parse(text: str) -> Optional[Dict]:
        """Try to extract JSON from LLM output."""
        # Look for JSON block
        json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Try whole text as JSON
        try:
            return json.loads(text.strip())
        except (json.JSONDecodeError, ValueError):
            pass

        return None

    @staticmethod
    def _extract_key_values(text: str) -> Dict[str, str]:
        """Extract key: value pairs from text."""
        kv = {}
        patterns = [
            r'(?i)sentiment[:\s]+(\w+)',
            r'(?i)confidence[:\s]+([\d.]+)',
            r'(?i)recommendation[:\s]+(\w+)',
            r'(?i)risk[:\s]+(\w+)',
            r'(?i)time.?horizon[:\s]+(\w+)',
        ]
        keys = ["sentiment", "confidence", "recommendation", "risk", "time_horizon"]

        for pattern, key in zip(patterns, keys):
            match = re.search(pattern, text)
            if match:
                kv[key] = match.group(1)

        return kv

    @staticmethod
    def _fill_from_dict(d: Dict, fallback: Dict) -> StructuredSignalAnalysis:
        """Fill StructuredSignalAnalysis from a parsed dict."""
        result = StructuredSignalAnalysis()

        # Sentiment
        sent = d.get("sentiment", "").upper()
        if sent in ("BULLISH", "BEARISH", "NEUTRAL", "UNCERTAIN"):
            result.sentiment = LLMSentiment(sent)

        # Confidence
        try:
            result.confidence = float(d.get("confidence", fallback.get("confidence", 0.5)))
            result.confidence = max(0.0, min(1.0, result.confidence))
        except (ValueError, TypeError):
            result.confidence = 0.5

        # Recommendation
        rec = d.get("recommendation", fallback.get("signal_type", "HOLD")).upper()
        if rec in ("BUY", "SELL", "HOLD"):
            result.recommendation = rec

        # Risk
        risk = d.get("risk_assessment", d.get("risk", "MODERATE")).upper()
        if risk in ("LOW", "MODERATE", "HIGH", "EXTREME"):
            result.risk_assessment = risk

        # Key factors
        factors = d.get("key_factors", d.get("factors", []))
        if isinstance(factors, list):
            result.key_factors = [str(f)[:200] for f in factors[:5]]

        # Reasoning
        result.reasoning = d.get("reasoning", d.get("analysis", ""))[:500]
        result.parsed_successfully = True

        return result

    @staticmethod
    def _fill_from_kv(kv: Dict, fallback: Dict) -> StructuredSignalAnalysis:
        """Fill from extracted key-value pairs."""
        result = StructuredSignalAnalysis()

        sent = kv.get("sentiment", "").upper()
        if sent in ("BULLISH", "BEARISH", "NEUTRAL", "UNCERTAIN"):
            result.sentiment = LLMSentiment(sent)

        try:
            result.confidence = float(kv.get("confidence", fallback.get("confidence", 0.5)))
        except ValueError:
            result.confidence = 0.5

        rec = kv.get("recommendation", fallback.get("signal_type", "HOLD")).upper()
        if rec in ("BUY", "SELL", "HOLD"):
            result.recommendation = rec

        risk = kv.get("risk", "MODERATE").upper()
        if risk in ("LOW", "MODERATE", "HIGH", "EXTREME"):
            result.risk_assessment = risk

        return result

    @staticmethod
    def _heuristic_parse(text: str, fallback: Dict) -> StructuredSignalAnalysis:
        """Heuristic text analysis as last resort."""
        result = StructuredSignalAnalysis()
        text_lower = text.lower()

        # Sentiment from word frequencies
        bull_words = sum(1 for w in ["bullish", "buy", "long", "upside", "positive", "support"]
                         if w in text_lower)
        bear_words = sum(1 for w in ["bearish", "sell", "short", "downside", "negative", "resistance"]
                         if w in text_lower)

        if bull_words > bear_words + 1:
            result.sentiment = LLMSentiment.BULLISH
            result.recommendation = "BUY"
        elif bear_words > bull_words + 1:
            result.sentiment = LLMSentiment.BEARISH
            result.recommendation = "SELL"
        else:
            result.sentiment = LLMSentiment.NEUTRAL
            result.recommendation = fallback.get("signal_type", "HOLD")

        result.confidence = fallback.get("confidence", 0.5)
        result.reasoning = text[:500]
        result.parsed_successfully = True

        return result
