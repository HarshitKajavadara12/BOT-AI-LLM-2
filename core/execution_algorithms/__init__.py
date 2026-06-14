"""
Core Execution Algorithms Module
Initialization file for execution algorithms package
"""

from .twap_algorithm import TWAPAlgorithm, TWAPParameters, TWAPScheduler, AdaptiveTWAP, MarketData
from .vwap_algorithm import VWAPAlgorithm, VWAPParameters, VWAPCalculator, VolumeProfile
from .implementation_shortfall import (
    ImplementationShortfallAlgorithm,
    ISParameters,
    RiskModel,
    MarketImpactModel
)
from .arrival_price import ArrivalPriceAlgorithm, ArrivalPriceParameters

__all__ = [
    # TWAP Algorithm
    'TWAPAlgorithm',
    'TWAPParameters', 
    'TWAPScheduler',
    'AdaptiveTWAP',
    
    # VWAP Algorithm
    'VWAPAlgorithm',
    'VWAPParameters',
    'VWAPCalculator',
    'VolumeProfile',
    
    # Implementation Shortfall
    'ImplementationShortfallAlgorithm',
    'ISParameters',
    'RiskModel',
    'MarketImpactModel',
    
    # Arrival Price
    'ArrivalPriceAlgorithm',
    'ArrivalPriceParameters',
    
    # Shared
    'MarketData'
]

__version__ = "1.0.0"
__author__ = "QUANTUM-FORGE Team"
__description__ = "Advanced execution algorithms for institutional trading"