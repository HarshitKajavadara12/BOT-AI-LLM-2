"""
Market Microstructure Package Init for QUANTUM-FORGE
Exposes key classes and functions for market microstructure analysis.
"""

from .orderbook_dynamics import (
    LimitOrderBook,
    OrderBookAnalyzer,
    HawkesProcessOrderBook,
    Order,
    Trade,
    OrderType,
    OrderSide
)

from .toxicity_detection import (
    ToxicityDetector,
    VPINCalculator,
    PINModel,
    NetworkToxicityDetector,
    ToxicityLevel,
    ToxicitySignal,
    TradeMetrics
)

from .liquidity_models import (
    AmihudModel,
    KyleModel,
    GlosstenMilgromModel,
    LiquidityProvider,
    LiquidityAnalyzer,
    LiquidityRegime,
    LiquidityMetrics,
    LiquidityEvent
)

from .price_formation import (
    HasbrouckModel,
    GloStenHarrisModel,
    StructuralVARModel,
    NonlinearPriceModel,
    PriceFormationAnalyzer,
    PriceDiscoveryMechanism,
    PriceComponent,
    PriceDiscoveryMetrics
)

__all__ = [
    # Order Book Dynamics
    'LimitOrderBook',
    'OrderBookAnalyzer', 
    'HawkesProcessOrderBook',
    'Order',
    'Trade',
    'OrderType',
    'OrderSide',
    
    # Toxicity Detection
    'ToxicityDetector',
    'VPINCalculator',
    'PINModel',
    'NetworkToxicityDetector',
    'ToxicityLevel',
    'ToxicitySignal',
    'TradeMetrics',
    
    # Liquidity Models
    'AmihudModel',
    'KyleModel',
    'GlosstenMilgromModel',
    'LiquidityProvider',
    'LiquidityAnalyzer',
    'LiquidityRegime',
    'LiquidityMetrics',
    'LiquidityEvent',
    
    # Price Formation
    'HasbrouckModel',
    'GloStenHarrisModel',
    'StructuralVARModel',
    'NonlinearPriceModel',
    'PriceFormationAnalyzer',
    'PriceDiscoveryMechanism',
    'PriceComponent',
    'PriceDiscoveryMetrics'
]

# Package metadata
__version__ = "1.0.0"
__author__ = "QUANTUM-FORGE Team"
__description__ = "Advanced market microstructure analysis and modeling toolkit"