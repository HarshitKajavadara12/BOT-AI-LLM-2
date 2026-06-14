# QUANTUM-FORGE Architecture & Authority Boundaries

##   ARCHITECTURE FROZEN (v1.0)
**Status:** LOCKED
**Date:** January 4, 2026

The architectural boundaries defined in this document are **non-negotiable**. Any change to these boundaries requires a formal architectural review.

### 1. The Prime Directive: Authority Flow
Authority flows **downwards only**.
- **Deterministic Core (Roles 1-3)**: Has full authority over execution and risk.
- **Cognitive Layer (Role 5)**: Has **ZERO** authority. It is a read-only observer.

### 2. Strict Boundaries
| Layer | Can Import | CANNOT Import |
|-------|------------|---------------|
| **Execution** | Core, Math, Data | LLM, API, Bridge |
| **Risk** | Core, Math | LLM, API, Bridge |
| **LLM** | Core (Read-Only) | Execution (Write) |

### 3. The "Kill-Switch" Invariant
The system must be fully functional with `LLM_ENABLED=false`.
- If the LLM layer crashes, trading continues.
- If the Vector Store fails, trading continues.
- If the API is down, trading continues.

### 4. Research Protocol
All LLM-assisted research must follow the **Safe Research Loop**:
1.  **Read** deterministic metrics.
2.  **Explain** via LLM (Read-Only).
3.  **Human** decision to implement changes manually.
4.  **Never** auto-deploy code or configs from LLM output.

### 5. Infrastructure Fallbacks
The system is designed to be self-healing. If external dependencies are missing, it degrades gracefully:
- **Qdrant Missing?** -> Falls back to **In-Memory Vector Store** (Non-persistent).
- **Redis Missing?** -> Falls back to **In-Memory Mock Stream** (Non-persistent).

This allows the system to run in "Development Mode" without a full Docker stack.

---
*This document serves as the source of truth for QUANTUM-FORGE architectural integrity.*
