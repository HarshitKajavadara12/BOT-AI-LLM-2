"""
Quantum-Forge: Metrics & Health API
=====================================
FastAPI application that exposes:
  - /health            — JSON health status (for Docker / K8s probes)
  - /metrics           — Prometheus text metrics
  - /api/status        — System overview JSON

Start standalone:
    uvicorn core.metrics_server:app --host 0.0.0.0 --port 8000
"""

import os
import time
import logging
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse, JSONResponse

try:
    from prometheus_client import (
        Counter, Gauge, Histogram, Summary,
        generate_latest, CONTENT_TYPE_LATEST,
    )
    PROM_AVAILABLE = True
except ImportError:
    PROM_AVAILABLE = False

logger = logging.getLogger("MetricsServer")

app = FastAPI(title="Quantum-Forge Metrics", version="1.0.0")

# ── Prometheus metrics ────────────────────────────────────────────
if PROM_AVAILABLE:
    TRADE_COUNTER = Counter(
        "qf_trades_total", "Total trade executions", ["symbol", "side", "status"]
    )
    SIGNAL_COUNTER = Counter(
        "qf_signals_total", "Total signals generated", ["symbol", "direction"]
    )
    PORTFOLIO_VALUE = Gauge(
        "qf_portfolio_value_usd", "Current portfolio value in USD"
    )
    POSITION_COUNT = Gauge(
        "qf_open_positions", "Number of open positions"
    )
    ITERATION_DURATION = Histogram(
        "qf_iteration_duration_seconds", "Time per main-loop iteration",
        buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
    )
    RISK_BLOCKED = Counter(
        "qf_risk_blocked_total", "Signals blocked by risk gate"
    )
    CIRCUIT_OPEN = Gauge(
        "qf_circuit_breaker_open", "Circuit breaker open (1) or closed (0)", ["symbol"]
    )
    WS_RECONNECTS = Counter(
        "qf_ws_reconnects_total", "WebSocket reconnection attempts"
    )
    ML_INFERENCE_LATENCY = Summary(
        "qf_ml_inference_seconds", "ML ensemble inference latency"
    )

# ── Shared state (set by QuantumCoreOrchestrator) ─────────────────
_orchestrator = None
_start_time = time.time()


def set_orchestrator(orch):
    """Called by QuantumCoreOrchestrator to expose state to the API."""
    global _orchestrator
    _orchestrator = orch


# ── Endpoints ─────────────────────────────────────────────────────
@app.get("/health")
async def health():
    """Docker / K8s health-check probe."""
    ok = _orchestrator is not None and getattr(_orchestrator, "is_running", False)
    status_code = 200 if ok else 503
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "healthy" if ok else "unhealthy",
            "uptime_seconds": int(time.time() - _start_time),
            "trading_mode": getattr(_orchestrator, "execution_manager", None)
                           and _orchestrator.execution_manager.mode.value or "UNKNOWN",
        },
    )


@app.get("/metrics")
async def metrics():
    """Prometheus scrape endpoint."""
    if not PROM_AVAILABLE:
        return PlainTextResponse("# prometheus_client not installed\n", status_code=501)
    
    # Push latest values from orchestrator into gauges
    if _orchestrator:
        try:
            cash = getattr(_orchestrator, "cash", 0)
            positions = getattr(_orchestrator, "positions", {})
            pv = cash + sum(
                p["quantity"] * p.get("current_price", p.get("entry_price", 0))
                for p in positions.values()
            )
            PORTFOLIO_VALUE.set(pv)
            POSITION_COUNT.set(len(positions))
        except Exception:
            pass
    
    return PlainTextResponse(
        generate_latest().decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.get("/api/status")
async def status():
    """Full system status JSON."""
    if not _orchestrator:
        return {"status": "not_initialized"}
    
    o = _orchestrator
    try:
        portfolio_value = o.cash + sum(
            p["quantity"] * p.get("current_price", p.get("entry_price", 0))
            for p in o.positions.values()
        )
    except Exception:
        portfolio_value = 0
    
    return {
        "is_running": o.is_running,
        "iteration": o.iteration,
        "symbols": o.symbols,
        "portfolio_value": round(portfolio_value, 2),
        "cash": round(o.cash, 2),
        "open_positions": len(o.positions),
        "total_trades": o._trade_count,
        "win_rate": round(o._win_count / o._trade_count, 3) if o._trade_count else 0,
        "regime": o.current_regime.value,
        "execution_mode": o.execution_manager.mode.value,
        "uptime_seconds": int(time.time() - _start_time),
    }
