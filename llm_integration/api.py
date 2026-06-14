"""FastAPI Service for QUANTUM-FORGE LLM/RAG Integration

This module provides a REST API service for natural language queries and AI-augmented
trading intelligence. It serves as the HTTP interface layer for the LLM/RAG system.

Responsibilities:
    - Expose REST API endpoints for natural language queries
    - Handle portfolio status requests
    - Provide system health monitoring
    - Validate and route signal analysis requests
    - Manage CORS and API middleware

Inputs:
    - HTTP POST /query: Natural language questions about trading state
    - HTTP GET /portfolio: Portfolio status requests
    - HTTP GET /health: System health checks
    - HTTP POST /analyze-signal: Signal validation requests (informational)

Outputs:
    - JSON responses with AI-generated insights and context
    - Portfolio state snapshots (read-only)
    - System health status
    - Signal analysis recommendations (advisory only, not executable)

CRITICAL CONSTRAINT:
    No AI/LLM is allowed to modify execution or risk decisions here.
    
    This API provides INFORMATIONAL responses only. All endpoints operate in
    READ-ONLY mode with respect to trading systems. No endpoint can:
    - Execute trades
    - Modify positions
    - Change risk parameters
    - Place or cancel orders
    - Override portfolio management decisions
    
    Signal analysis responses are purely advisory and require human or
    deterministic algorithm approval before any action is taken.

Performance:
    - Target latency: <200ms for LLM queries
    - Concurrent requests: Supports multiple simultaneous queries
    - Rate limiting: Configurable per-client limits

Dependencies:
    - bridge.py: Access to synchronized trading data
    - llm_engine.py: LLM inference for response generation
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
from datetime import datetime
import uvicorn
import time

from llm_integration.bridge import get_integration_bridge
from llm_integration.llm_engine import get_llm_engine
from llm_integration.explanation_contracts import (
    SignalExplanation, RiskExplanation, ExecutionExplanation, PortfolioExplanation
)
from core.analytics import get_analytics_engine, PerformanceMetrics


# Pydantic models
class QueryRequest(BaseModel):
    """Natural language query request"""
    query: str = Field(..., description="Natural language query", min_length=1)
    symbol: Optional[str] = Field(None, description="Optional symbol filter (e.g., BTCUSDT)")
    max_tokens: int = Field(512, description="Maximum response tokens", ge=64, le=2048)


class QueryResponse(BaseModel):
    """Query response with AI insight"""
    query: str
    response: str
    context: Dict
    latency_ms: float
    timestamp: str
    model: str = "Llama-3.2-8B"


class PortfolioResponse(BaseModel):
    """Portfolio status response"""
    cash: float
    positions: Dict[str, Dict]
    metrics: Dict
    total_value: float
    timestamp: str


class SystemStatus(BaseModel):
    """System health status"""
    status: str
    components: Dict[str, bool]
    stats: Dict
    timestamp: str


class SignalRequest(BaseModel):
    """Trading signal analysis request"""
    symbol: str
    signal_type: str  # BUY, SELL, HOLD
    confidence: float = Field(..., ge=0.0, le=1.0)
    indicators: Optional[Dict] = None


class SignalResponse(BaseModel):
    """Signal analysis response"""
    symbol: str
    valid: bool
    confidence: float
    reasoning: str
    recommendation: str
    timestamp: str


class AnalyticsRequest(BaseModel):
    """Request for calculating performance metrics"""
    returns: List[float] = Field(..., description="List of period returns (e.g., 0.01 for 1%)")
    risk_free_rate: float = Field(0.0, description="Annualized risk-free rate")


# Initialize FastAPI
app = FastAPI(
    title="Quantum Forge LLM/RAG API",
    description="Trading intelligence API with AI-powered insights",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global components
bridge = None
llm = None


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    global bridge, llm
    
    print("\n  Starting Quantum Forge API...")
    
    # Initialize bridge
    bridge = get_integration_bridge()
    bridge.start()
    
    # Initialize LLM
    llm = get_llm_engine()
    
    print("  Quantum Forge API ready")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("\n  Shutting down Quantum Forge API...")
    
    if bridge:
        bridge.stop()
    
    print("  Shutdown complete")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Quantum Forge LLM/RAG API",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs",
        "endpoints": {
            "query": "POST /api/v1/query",
            "portfolio": "GET /api/v1/portfolio",
            "status": "GET /api/v1/status",
            "signal": "POST /api/v1/signal"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/v1/query", response_model=QueryResponse)
async def query_ai(request: QueryRequest):
    """
    Natural language query endpoint
    
    Query the AI about trading, portfolio, market conditions, etc.
    
    Example queries:
    - "What's my current portfolio status?"
    - "Show me recent Bitcoin trades"
    - "Analyze my performance this week"
    - "What's the market sentiment for ETH?"
    """
    if not bridge or not llm:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    start_time = time.perf_counter()
    
    try:
        # Get context (5-30ms)
        rag_context = bridge.query_context(request.query, request.symbol)
        
        # Get portfolio state (<1ms)
        portfolio_state = rag_context.get('current_portfolio', {})
        
        # Generate AI response (50-200ms)
        ai_response = llm.generate_trading_insight(
            request.query,
            rag_context,
            portfolio_state,
            max_tokens=request.max_tokens
        )
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        return QueryResponse(
            query=request.query,
            response=ai_response,
            context={
                'analytics': rag_context.get('analytics', {}),
                'relevant_items': {
                    'trades': len(rag_context.get('relevant_trades', [])),
                    'signals': len(rag_context.get('relevant_signals', [])),
                    'analyses': len(rag_context.get('market_analysis', []))
                }
            },
            latency_ms=round(latency_ms, 2),
            timestamp=datetime.now().isoformat(),
            model="Llama-3.2-8B" if llm.connected else "Template"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing error: {str(e)}")


@app.get("/api/v1/portfolio", response_model=PortfolioResponse)
async def get_portfolio():
    """
    Get current portfolio state
    
    Returns positions, cash balance, and metrics
    """
    if not bridge:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        # Get portfolio from bridge
        context = bridge.query_context("portfolio", None)
        portfolio = context.get('current_portfolio', {})
        
        # Calculate total value
        positions = portfolio.get('positions', {})
        cash = portfolio.get('cash', 0)
        
        total_value = cash
        for symbol, pos in positions.items():
            total_value += pos['amount'] * pos['entry_price']
        
        return PortfolioResponse(
            cash=cash,
            positions=positions,
            metrics=portfolio.get('metrics', {}),
            total_value=round(total_value, 2),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Portfolio retrieval error: {str(e)}")


@app.get("/api/v1/status", response_model=SystemStatus)
async def get_status():
    """
    Get system health status
    
    Returns component status and statistics
    """
    if not bridge:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        status = bridge.get_status()
        
        return SystemStatus(
            status="operational" if status['running'] else "stopped",
            components=status['components'],
            stats=status['stats'],
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status retrieval error: {str(e)}")


@app.post("/api/v1/signal", response_model=SignalResponse)
async def analyze_signal(request: SignalRequest):
    """
    Analyze a trading signal
    
    Get AI analysis of a trading signal before execution
    """
    if not llm:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        # Analyze signal
        analysis = llm.analyze_signal(
            request.symbol,
            {
                'signal_type': request.signal_type,
                'confidence': request.confidence,
                'indicators': request.indicators or {}
            }
        )
        
        return SignalResponse(
            symbol=request.symbol,
            valid=analysis['valid'],
            confidence=analysis['confidence'],
            reasoning=analysis['reasoning'],
            recommendation=analysis['recommendation'],
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Signal analysis error: {str(e)}")


@app.get("/api/v1/analytics")
async def get_analytics(symbol: Optional[str] = None):
    """
    Get trading analytics
    
    Returns performance metrics and statistics
    """
    if not bridge:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        context = bridge.query_context("analytics", symbol)
        
        return {
            "analytics": context.get('analytics', {}),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analytics retrieval error: {str(e)}")


@app.post("/api/v1/analytics/calculate", response_model=Dict[str, float])
async def calculate_analytics(request: AnalyticsRequest):
    """
    Calculate standard performance metrics.
    
    This endpoint exposes the EXACT SAME logic used by the core system,
    ensuring parity between Research notebooks and Live reporting.
    """
    try:
        engine = get_analytics_engine()
        metrics = engine.calculate_metrics(request.returns, request.risk_free_rate)
        
        # Convert dataclass to dict
        return {
            "total_return": metrics.total_return,
            "sharpe_ratio": metrics.sharpe_ratio,
            "sortino_ratio": metrics.sortino_ratio,
            "max_drawdown": metrics.max_drawdown,
            "volatility": metrics.volatility,
            "win_rate": metrics.win_rate,
            "profit_factor": metrics.profit_factor
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Calculation error: {str(e)}")


@app.get("/api/v1/explain/signal/{signal_id}", response_model=SignalExplanation)
async def explain_signal(signal_id: str, symbol: str):
    """
    Explain WHY a signal was generated
    
    Returns structured explanation of signal drivers and context.
    """
    if not bridge:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        return bridge.get_signal_explanation(symbol, signal_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation error: {str(e)}")


@app.get("/api/v1/explain/risk/{symbol}", response_model=RiskExplanation)
async def explain_risk(symbol: str):
    """
    Explain WHY a risk decision was made
    
    Returns structured explanation of risk checks and limits.
    """
    if not bridge:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        return bridge.get_risk_explanation(symbol)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation error: {str(e)}")


@app.get("/api/v1/explain/execution/{trade_id}", response_model=ExecutionExplanation)
async def explain_execution(trade_id: str):
    """
    Explain WHY execution happened this way
    
    Returns structured explanation of slippage, latency, and routing.
    """
    if not bridge:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        return bridge.get_execution_explanation(trade_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation error: {str(e)}")


# Run server
if __name__ == "__main__":
    print("\n" + "="*80)
    print("QUANTUM FORGE LLM/RAG API")
    print("="*80)
    print("\n  Starting FastAPI server...")
    print("   • Docs: http://localhost:8000/docs")
    print("   • API: http://localhost:8000/api/v1/")
    print("\n  Endpoints:")
    print("   POST /api/v1/query     - Natural language queries")
    print("   GET  /api/v1/portfolio - Portfolio status")
    print("   GET  /api/v1/status    - System health")
    print("   POST /api/v1/signal    - Signal analysis")
    print("   GET  /api/v1/analytics - Trading analytics")
    print("\n" + "="*80)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
