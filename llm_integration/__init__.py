"""
LLM/RAG Integration Module for QUANTUM-FORGE

This module provides AI-augmented trading intelligence through natural language interfaces
and retrieval-augmented generation (RAG) capabilities.

Responsibilities:
    - Bridge QUANTUM-FORGE core systems with LLM/RAG components
    - Provide natural language query interface for trading data
    - Enable semantic search across trading history and market analysis
    - Real-time event streaming for AI-augmented insights
    - Fast in-memory caching for sub-millisecond analytics

Inputs:
    - Natural language queries from users/interfaces
    - Trading events from core QUANTUM-FORGE systems
    - Market data from data ingestion pipeline
    - Portfolio state from dynamic portfolio tracker

Outputs:
    - AI-generated insights and explanations (informational only)
    - Semantic search results from trading history
    - Real-time event notifications
    - Analytics and statistics from cached data

CRITICAL CONSTRAINT:
    No AI/LLM is allowed to modify execution or risk decisions here.
    
    This module operates in READ-ONLY mode for all critical trading systems.
    All trading decisions, risk management, and order execution remain under
    exclusive control of deterministic mathematical algorithms in core/.
    
    LLM outputs are INFORMATIONAL ONLY and never directly trigger:
    - Trade executions
    - Position sizing decisions
    - Risk limit modifications
    - Portfolio rebalancing
    - Order placement or cancellation

Architecture:
    - api.py: REST API endpoints for queries
    - bridge.py: Synchronization layer with core systems
    - llm_engine.py: Local LLM inference engine
    - vector_store.py: Semantic search via vector embeddings
    - duckdb_cache.py: Fast analytical queries
    - event_stream.py: Real-time event publishing
"""

__version__ = "1.0.0"
__author__ = "QUANTUM-FORGE Team"
