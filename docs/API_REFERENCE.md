# QUANTUM-FORGE — API Reference

## Base URL

```
http://localhost:8000
```

All endpoints return JSON. The API is built with FastAPI and includes automatic OpenAPI docs at `/docs` and `/redoc`.

---

## Authentication

Currently no authentication is required. In production, add a `Bearer` token header or IP whitelist.

---

## Endpoints

### `GET /`

Root endpoint — system information.

**Response 200:**
```json
{
  "system": "QUANTUM-FORGE",
  "version": "2.0.0",
  "status": "operational",
  "mode": "paper_trading",
  "llm_enabled": true,
  "endpoints": ["/api/v1/query", "/api/v1/portfolio", "/api/v1/status", ...]
}
```

---

### `GET /health`

Health check for load balancers and monitoring.

**Response 200:**
```json
{
  "status": "healthy",
  "timestamp": "2026-02-17T12:00:00Z"
}
```

---

### `POST /api/v1/query`

Send a natural-language question to the LLM with RAG context.

**Request Body:**
```json
{
  "query": "Why did the system take a short position on ETH?",
  "max_tokens": 512,
  "temperature": 0.7,
  "include_context": true
}
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| query | string | ✅ | — | Natural-language question (max 1000 chars) |
| max_tokens | int | ❌ | 512 | Maximum response tokens |
| temperature | float | ❌ | 0.7 | Sampling temperature [0, 1] |
| include_context | bool | ❌ | true | Include RAG context in response |

**Response 200 (QueryResponse):**
```json
{
  "answer": "The system shorted ETH due to ...",
  "confidence": 0.85,
  "sources": ["signals", "market_analysis"],
  "processing_time_ms": 142.5,
  "model": "llama-3.2-8b",
  "context_used": true
}
```

**Errors:**
| Code | Reason |
|---|---|
| 400 | Query too long (> 1000 chars) or prompt injection detected |
| 503 | LLM engine unavailable |

---

### `GET /api/v1/portfolio`

Get current portfolio state.

**Response 200 (PortfolioResponse):**
```json
{
  "positions": [
    {
      "symbol": "BTCUSDT",
      "side": "long",
      "size": 0.001,
      "entry_price": 68500.0,
      "unrealised_pnl": 0.00012,
      "regime": "low_vol"
    }
  ],
  "total_equity": 1000.0,
  "utilisation": 0.15,
  "timestamp": "2026-02-17T12:00:00Z"
}
```

---

### `GET /api/v1/status`

System operational status.

**Response 200 (SystemStatus):**
```json
{
  "pipeline_running": true,
  "websocket_connected": true,
  "llm_loaded": true,
  "models_trained": false,
  "symbols_active": ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", "DOGEUSDT", "XRPUSDT"],
  "uptime_seconds": 3600
}
```

---

### `POST /api/v1/signal`

Request a trading signal for a specific symbol.

**Request Body:**
```json
{
  "symbol": "BTCUSDT",
  "timeframe": "1m"
}
```

**Response 200 (SignalResponse):**
```json
{
  "symbol": "BTCUSDT",
  "signal": 0.42,
  "direction": "long",
  "confidence": 0.78,
  "components": {
    "math_signal": 0.35,
    "ml_signal": 0.50,
    "cross_asset_signal": 0.40
  },
  "regime": "trending_up",
  "timestamp": "2026-02-17T12:00:00Z"
}
```

---

### `GET /api/v1/analytics`

Get available analytics metrics.

**Response 200:**
```json
{
  "available_metrics": ["sharpe", "sortino", "max_drawdown", "calmar", "win_rate", "profit_factor"],
  "last_updated": "2026-02-17T12:00:00Z"
}
```

---

### `POST /api/v1/analytics/calculate`

Calculate specific analytics metrics.

**Request Body:**
```json
{
  "metrics": ["sharpe", "max_drawdown"],
  "window": 30
}
```

**Response 200:**
```json
{
  "sharpe": 1.42,
  "max_drawdown": -0.045
}
```

---

### `GET /api/v1/explain/signal/{signal_id}`

Get LLM-generated explanation of a trading signal.

**Path Parameters:** `signal_id` (string)

**Response 200 (SignalExplanation):**
```json
{
  "signal_id": "sig_abc123",
  "explanation": "The long signal was generated because ...",
  "factors": [
    {"name": "fourier_dominant_cycle", "weight": 0.3, "value": 0.6},
    {"name": "stochastic_vol", "weight": 0.25, "value": -0.1}
  ],
  "confidence": 0.82
}
```

---

### `GET /api/v1/explain/risk/{symbol}`

Get LLM-generated risk explanation for a symbol.

**Path Parameters:** `symbol` (string, e.g. "BTCUSDT")

**Response 200 (RiskExplanation):**
```json
{
  "symbol": "BTCUSDT",
  "risk_level": "moderate",
  "explanation": "Current VaR at 95% is ...",
  "evt_tail_risk": 0.02,
  "regime": "low_vol"
}
```

---

### `GET /api/v1/explain/execution/{trade_id}`

Get LLM-generated explanation of a trade execution.

**Path Parameters:** `trade_id` (string)

**Response 200 (ExecutionExplanation):**
```json
{
  "trade_id": "trade_abc123",
  "algo_used": "TWAP",
  "explanation": "TWAP was selected because ...",
  "slippage_bps": 2.1,
  "market_impact_bps": 0.5
}
```

---

## Error Format

All errors follow this format:
```json
{
  "detail": "Description of what went wrong"
}
```

## Rate Limits

No explicit rate limits currently enforced. The LLM endpoint is naturally bottlenecked by inference time (~50-200ms).

---

*Generated for QUANTUM-FORGE v2.0.0*
