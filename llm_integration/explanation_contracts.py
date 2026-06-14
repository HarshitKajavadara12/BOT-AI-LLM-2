"""
Explanation Contracts for QUANTUM-FORGE
Defines strict, read-only schemas for system explainability.

CRITICAL CONSTRAINT:
    These schemas are for OBSERVABILITY ONLY.
    They must NEVER be used to drive execution or modify system state.
    All fields must be derived from immutable historical data or current snapshots.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union
from datetime import datetime

class SignalExplanation(BaseModel):
    """Schema for explaining WHY a signal was generated."""
    signal_id: str = Field(..., description="Unique identifier of the signal")
    symbol: str = Field(..., description="Trading symbol")
    signal_type: str = Field(..., description="Type of signal (BUY/SELL/HOLD)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence score")
    timestamp: str = Field(..., description="ISO 8601 timestamp of generation")
    
    # Explanation fields (Read-Only)
    primary_factors: List[str] = Field(..., description="List of key drivers (e.g., 'RSI Oversold', 'Volume Spike')")
    market_context: str = Field(..., description="Summary of market conditions at time of signal")
    model_name: str = Field(..., description="Name of the model that generated the signal")
    
    # Guardrails
    is_actionable: bool = Field(False, description="Whether this signal passed initial validity checks")
    rejection_reason: Optional[str] = Field(None, description="If rejected, why (e.g., 'Low Confidence')")

class RiskExplanation(BaseModel):
    """Schema for explaining WHY a risk decision was made."""
    decision_id: str = Field(..., description="Unique identifier of the risk check")
    symbol: str = Field(..., description="Trading symbol")
    action_type: str = Field(..., description="Proposed action (e.g., 'OPEN_POSITION')")
    outcome: str = Field(..., description="Result (ALLOWED/BLOCKED/MODIFIED)")
    timestamp: str = Field(..., description="ISO 8601 timestamp of decision")
    
    # Explanation fields (Read-Only)
    checks_passed: List[str] = Field(..., description="List of risk checks that passed")
    checks_failed: List[str] = Field(..., description="List of risk checks that failed")
    limit_utilized: float = Field(..., description="Percentage of risk limit used")
    exposure_level: float = Field(..., description="Current exposure amount")

class ExecutionExplanation(BaseModel):
    """Schema for explaining WHY execution happened this way."""
    trade_id: str = Field(..., description="Unique identifier of the trade")
    snapshot_id: Optional[str] = Field(None, description="ID of the deterministic snapshot for audit replay")
    symbol: str = Field(..., description="Trading symbol")
    side: str = Field(..., description="Trade side (BUY/SELL)")
    quantity: float = Field(..., description="Executed quantity")
    price: float = Field(..., description="Executed price")
    timestamp: str = Field(..., description="ISO 8601 timestamp of execution")
    
    # Explanation fields (Read-Only)
    slippage_bps: float = Field(..., description="Slippage in basis points")
    latency_ms: float = Field(..., description="Execution latency in milliseconds")
    venue: str = Field(..., description="Execution venue or route")
    market_impact_estimate: str = Field(..., description="Estimated impact (LOW/MEDIUM/HIGH)")
    fill_quality: str = Field(..., description="Assessment of fill quality")

class PortfolioExplanation(BaseModel):
    """Schema for explaining WHY portfolio state changed."""
    timestamp: str = Field(..., description="ISO 8601 timestamp of snapshot")
    total_value: float = Field(..., description="Total portfolio value")
    cash_balance: float = Field(..., description="Available cash")
    
    # Explanation fields (Read-Only)
    active_positions: int = Field(..., description="Number of active positions")
    exposure_distribution: Dict[str, float] = Field(..., description="Exposure by asset class or symbol")
    daily_pnl_attribution: Dict[str, float] = Field(..., description="PnL attribution by source")
    risk_utilization: float = Field(..., description="Overall portfolio risk utilization %")

class SystemHealthExplanation(BaseModel):
    """Schema for explaining system health state."""
    status: str = Field(..., description="Overall status (HEALTHY/DEGRADED/DOWN)")
    timestamp: str = Field(..., description="ISO 8601 timestamp")
    
    # Explanation fields (Read-Only)
    component_status: Dict[str, str] = Field(..., description="Status of individual components")
    active_alerts: List[str] = Field(..., description="List of active system alerts")
    resource_usage: Dict[str, float] = Field(..., description="Resource usage metrics (CPU, RAM)")

class CognitiveRiskAssessment(BaseModel):
    """
    Schema for LLM-based risk assessment.
    Used to DAMPEN risk, never to increase it.
    """
    assessment_id: str = Field(..., description="Unique identifier")
    timestamp: str = Field(..., description="ISO 8601 timestamp")
    
    # Inputs
    regime_tag: str = Field(..., description="Detected market regime (e.g., 'HIGH_VOLATILITY')")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="LLM confidence in this assessment")
    
    # Output (The Dampener)
    risk_multiplier: float = Field(..., ge=0.0, le=1.0, description="Multiplier for position sizing (Max 1.0)")
    reasoning: str = Field(..., description="Why this multiplier was chosen")
