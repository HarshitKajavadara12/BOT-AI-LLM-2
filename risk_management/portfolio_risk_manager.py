"""
Portfolio Risk Manager
Real-time portfolio risk monitoring and management system
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
import queue
import time
from collections import deque, defaultdict
import warnings
import math
import asyncio
from concurrent.futures import ThreadPoolExecutor
from core.risk_mathematics.cognitive_dampener import get_cognitive_dampener


class RiskMeasureType(Enum):
    """Types of risk measures"""
    VAR_PARAMETRIC = "VAR_PARAMETRIC"       # Parametric Value at Risk
    VAR_HISTORICAL = "VAR_HISTORICAL"       # Historical Value at Risk
    VAR_MONTE_CARLO = "VAR_MONTE_CARLO"     # Monte Carlo Value at Risk
    EXPECTED_SHORTFALL = "EXPECTED_SHORTFALL"  # Expected Shortfall (CVaR)
    VOLATILITY = "VOLATILITY"               # Portfolio volatility
    BETA = "BETA"                           # Portfolio beta
    TRACKING_ERROR = "TRACKING_ERROR"       # Tracking error vs benchmark
    MAX_DRAWDOWN = "MAX_DRAWDOWN"          # Maximum drawdown
    CORRELATION = "CORRELATION"             # Portfolio correlations
    CONCENTRATION = "CONCENTRATION"         # Concentration risk


class RiskLimitType(Enum):
    """Types of risk limits"""
    POSITION_LIMIT = "POSITION_LIMIT"       # Position size limits
    SECTOR_LIMIT = "SECTOR_LIMIT"           # Sector concentration limits
    VAR_LIMIT = "VAR_LIMIT"                 # VaR limits
    VOLATILITY_LIMIT = "VOLATILITY_LIMIT"   # Volatility limits
    DRAWDOWN_LIMIT = "DRAWDOWN_LIMIT"       # Drawdown limits
    BETA_LIMIT = "BETA_LIMIT"               # Beta limits
    CORRELATION_LIMIT = "CORRELATION_LIMIT"  # Correlation limits
    LEVERAGE_LIMIT = "LEVERAGE_LIMIT"       # Leverage limits


class RiskAlertLevel(Enum):
    """Risk alert severity levels"""
    INFO = "INFO"                           # Informational
    WARNING = "WARNING"                     # Warning level
    CRITICAL = "CRITICAL"                   # Critical level
    BREACH = "BREACH"                       # Limit breach


@dataclass
class Position:
    """Portfolio position"""
    symbol: str
    quantity: float
    average_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    
    # Risk characteristics
    beta: float = 1.0
    volatility: float = 0.30
    sector: str = "Unknown"
    country: str = "US"
    currency: str = "USD"
    
    # Metadata
    last_updated: datetime = field(default_factory=datetime.now)
    
    @property
    def notional_value(self) -> float:
        """Absolute notional value"""
        return abs(self.market_value)
    
    @property
    def weight_pct(self) -> float:
        """Position weight as percentage (to be set by portfolio)"""
        return 0.0  # Will be calculated by portfolio manager
    
    @property
    def is_long(self) -> bool:
        """Check if position is long"""
        return self.quantity > 0
    
    @property
    def is_short(self) -> bool:
        """Check if position is short"""
        return self.quantity < 0


@dataclass
class RiskLimit:
    """Risk limit definition"""
    limit_id: str
    limit_type: RiskLimitType
    limit_value: float
    
    # Limit scope
    scope: str = "PORTFOLIO"                # "PORTFOLIO", "SECTOR", "POSITION"
    scope_filter: Optional[str] = None      # Specific filter (e.g., sector name)
    
    # Alert thresholds (as percentage of limit)
    warning_threshold: float = 0.80         # 80% of limit
    critical_threshold: float = 0.95        # 95% of limit
    
    # Limit properties
    is_active: bool = True
    description: str = ""
    
    # Temporal properties
    effective_date: datetime = field(default_factory=datetime.now)
    expiry_date: Optional[datetime] = None
    
    def check_breach(self, current_value: float) -> RiskAlertLevel:
        """Check if current value breaches limits"""
        
        if not self.is_active:
            return RiskAlertLevel.INFO
        
        utilization = abs(current_value) / self.limit_value if self.limit_value != 0 else 0
        
        if utilization >= 1.0:
            return RiskAlertLevel.BREACH
        elif utilization >= self.critical_threshold:
            return RiskAlertLevel.CRITICAL
        elif utilization >= self.warning_threshold:
            return RiskAlertLevel.WARNING
        else:
            return RiskAlertLevel.INFO


@dataclass
class RiskAlert:
    """Risk alert/notification"""
    alert_id: str
    alert_level: RiskAlertLevel
    limit_id: str
    
    # Alert details
    message: str
    current_value: float
    limit_value: float
    utilization_pct: float
    
    # Context
    scope: str = "PORTFOLIO"
    scope_filter: Optional[str] = None
    
    # Timing
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None


@dataclass
class VaRResult:
    """Value at Risk calculation results"""
    var_amount: float                       # VaR amount in base currency
    confidence_level: float                 # Confidence level (e.g., 0.95)
    time_horizon_days: int                  # Time horizon in days
    method: str                             # Calculation method
    
    # Additional metrics
    expected_shortfall: Optional[float] = None  # Expected Shortfall (CVaR)
    portfolio_volatility: Optional[float] = None
    worst_case_scenario: Optional[float] = None
    
    # Metadata
    calculation_time: datetime = field(default_factory=datetime.now)
    positions_count: int = 0
    portfolio_value: float = 0.0
    
    @property
    def var_percentage(self) -> float:
        """VaR as percentage of portfolio value"""
        if self.portfolio_value != 0:
            return abs(self.var_amount) / self.portfolio_value
        return 0.0


class RiskCalculator:
    """
    Risk calculation engine
    """
    
    def __init__(self):
        
        # Historical data for calculations
        self.price_history: Dict[str, pd.DataFrame] = {}
        self.return_history: Dict[str, pd.DataFrame] = {}
        
        # Risk model parameters
        self.default_confidence_levels = [0.95, 0.99]
        self.default_time_horizons = [1, 10, 21]  # 1 day, 2 weeks, 1 month
        self.decay_factor = 0.94                   # EWMA decay factor
        
        # Monte Carlo parameters
        self.mc_simulations = 10000
        self.random_seed = 42
        
    def calculate_portfolio_var(
        self,
        positions: List[Position],
        confidence_level: float = 0.95,
        time_horizon_days: int = 1,
        method: str = "parametric"
    ) -> VaRResult:
        """
        Calculate portfolio Value at Risk
        
        Args:
            positions: List of portfolio positions
            confidence_level: Confidence level (e.g., 0.95 for 95% VaR)
            time_horizon_days: Time horizon in days
            method: Calculation method ("parametric", "historical", "monte_carlo")
            
        Returns:
            VaR calculation results
        """
        
        if not positions:
            return VaRResult(
                var_amount=0.0,
                confidence_level=confidence_level,
                time_horizon_days=time_horizon_days,
                method=method
            )
        
        # Calculate portfolio value
        portfolio_value = sum(pos.market_value for pos in positions)
        
        if method.lower() == "parametric":
            var_result = self._calculate_parametric_var(
                positions, confidence_level, time_horizon_days
            )
        elif method.lower() == "historical":
            var_result = self._calculate_historical_var(
                positions, confidence_level, time_horizon_days
            )
        elif method.lower() == "monte_carlo":
            var_result = self._calculate_monte_carlo_var(
                positions, confidence_level, time_horizon_days
            )
        else:
            raise ValueError(f"Unknown VaR method: {method}")
        
        var_result.positions_count = len(positions)
        var_result.portfolio_value = portfolio_value
        
        return var_result
    
    def _calculate_parametric_var(
        self,
        positions: List[Position],
        confidence_level: float,
        time_horizon_days: int
    ) -> VaRResult:
        """Calculate parametric (delta-normal) VaR"""
        
        # Build portfolio vector
        symbols = [pos.symbol for pos in positions]
        weights = np.array([pos.market_value for pos in positions])
        portfolio_value = np.sum(np.abs(weights))
        
        if portfolio_value == 0:
            return VaRResult(
                var_amount=0.0,
                confidence_level=confidence_level,
                time_horizon_days=time_horizon_days,
                method="parametric"
            )
        
        # Normalize weights
        weights = weights / portfolio_value
        
        # Get or estimate covariance matrix
        cov_matrix = self._get_covariance_matrix(symbols)
        
        if cov_matrix is None:
            # Fallback to individual position VaR sum (assumes no correlation)
            individual_vars = []
            z_score = self._get_z_score(confidence_level)
            
            for pos in positions:
                position_var = abs(pos.market_value) * pos.volatility * z_score * math.sqrt(time_horizon_days)
                individual_vars.append(position_var)
            
            # Conservative estimate: sum of individual VaRs
            total_var = sum(individual_vars)
            portfolio_vol = math.sqrt(sum((w * pos.volatility) ** 2 for w, pos in zip(weights, positions)))
            
        else:
            # Full portfolio VaR calculation
            portfolio_vol = math.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            z_score = self._get_z_score(confidence_level)
            total_var = portfolio_value * portfolio_vol * z_score * math.sqrt(time_horizon_days)
        
        # Calculate Expected Shortfall (approximate)
        es_multiplier = self._get_expected_shortfall_multiplier(confidence_level)
        expected_shortfall = total_var * es_multiplier
        
        return VaRResult(
            var_amount=total_var,
            confidence_level=confidence_level,
            time_horizon_days=time_horizon_days,
            method="parametric",
            expected_shortfall=expected_shortfall,
            portfolio_volatility=portfolio_vol
        )
    
    def _calculate_historical_var(
        self,
        positions: List[Position],
        confidence_level: float,
        time_horizon_days: int
    ) -> VaRResult:
        """Calculate historical simulation VaR"""
        
        symbols = [pos.symbol for pos in positions]
        
        # Get historical returns
        returns_data = self._get_historical_returns(symbols, lookback_days=252 * 2)  # 2 years
        
        if returns_data is None or returns_data.empty:
            # Fallback to parametric method
            return self._calculate_parametric_var(positions, confidence_level, time_horizon_days)
        
        # Calculate portfolio returns for each historical period
        portfolio_returns = []
        
        for _, returns_row in returns_data.iterrows():
            portfolio_return = 0.0
            portfolio_value = sum(abs(pos.market_value) for pos in positions)
            
            for pos in positions:
                if pos.symbol in returns_row:
                    weight = pos.market_value / portfolio_value if portfolio_value > 0 else 0
                    portfolio_return += weight * returns_row[pos.symbol]
            
            portfolio_returns.append(portfolio_return)
        
        portfolio_returns = np.array(portfolio_returns)
        
        # Scale for time horizon
        if time_horizon_days > 1:
            portfolio_returns = portfolio_returns * math.sqrt(time_horizon_days)
        
        # Calculate VaR as percentile
        var_percentile = (1 - confidence_level) * 100
        var_return = np.percentile(portfolio_returns, var_percentile)
        
        portfolio_value = sum(abs(pos.market_value) for pos in positions)
        var_amount = abs(var_return * portfolio_value)
        
        # Calculate Expected Shortfall
        worse_returns = portfolio_returns[portfolio_returns <= var_return]
        expected_shortfall = abs(np.mean(worse_returns) * portfolio_value) if len(worse_returns) > 0 else var_amount
        
        # Portfolio volatility
        portfolio_vol = np.std(portfolio_returns)
        
        return VaRResult(
            var_amount=var_amount,
            confidence_level=confidence_level,
            time_horizon_days=time_horizon_days,
            method="historical",
            expected_shortfall=expected_shortfall,
            portfolio_volatility=portfolio_vol,
            worst_case_scenario=abs(np.min(portfolio_returns) * portfolio_value)
        )
    
    def _calculate_monte_carlo_var(
        self,
        positions: List[Position],
        confidence_level: float,
        time_horizon_days: int
    ) -> VaRResult:
        """Calculate Monte Carlo simulation VaR"""
        
        # Keep seed call for reproducibility if provided, but use deterministic sampling below
        try:
            if hasattr(self, 'random_seed') and self.random_seed is not None:
                np.random.seed(self.random_seed)
        except Exception:
            pass
        
        symbols = [pos.symbol for pos in positions]
        portfolio_value = sum(abs(pos.market_value) for pos in positions)
        
        if portfolio_value == 0:
            return VaRResult(
                var_amount=0.0,
                confidence_level=confidence_level,
                time_horizon_days=time_horizon_days,
                method="monte_carlo"
            )
        
        # Get correlation matrix
        corr_matrix = self._get_correlation_matrix(symbols)
        
        if corr_matrix is None:
            # Use independent simulations
            simulated_returns = []
            
            for i in range(self.mc_simulations):
                portfolio_return = 0.0
                
                for idx, pos in enumerate(positions):
                    weight = pos.market_value / portfolio_value
                    # Deterministic pseudo-sample (sinusoidal low-discrepancy surrogate)
                    z = math.sin((i + 1) * (idx + 1) * 0.123)
                    random_return = z * pos.volatility * math.sqrt(time_horizon_days)
                    portfolio_return += weight * random_return
                
                simulated_returns.append(portfolio_return)
        
        else:
            # Correlated simulation
            simulated_returns = self._simulate_correlated_returns(
                positions, corr_matrix, time_horizon_days
            )
        
        simulated_returns = np.array(simulated_returns)
        
        # Calculate VaR
        var_percentile = (1 - confidence_level) * 100
        var_return = np.percentile(simulated_returns, var_percentile)
        var_amount = abs(var_return * portfolio_value)
        
        # Calculate Expected Shortfall
        worse_returns = simulated_returns[simulated_returns <= var_return]
        expected_shortfall = abs(np.mean(worse_returns) * portfolio_value) if len(worse_returns) > 0 else var_amount
        
        # Portfolio volatility
        portfolio_vol = np.std(simulated_returns)
        
        return VaRResult(
            var_amount=var_amount,
            confidence_level=confidence_level,
            time_horizon_days=time_horizon_days,
            method="monte_carlo",
            expected_shortfall=expected_shortfall,
            portfolio_volatility=portfolio_vol,
            worst_case_scenario=abs(np.min(simulated_returns) * portfolio_value)
        )
    
    def _simulate_correlated_returns(
        self,
        positions: List[Position],
        corr_matrix: np.ndarray,
        time_horizon_days: int
    ) -> List[float]:
        """Simulate correlated returns using Cholesky decomposition"""
        
        try:
            # Cholesky decomposition for correlation
            L = np.linalg.cholesky(corr_matrix)
            
            simulated_returns = []
            volatilities = np.array([pos.volatility for pos in positions])
            weights = np.array([pos.market_value for pos in positions])
            portfolio_value = np.sum(np.abs(weights))
            weights = weights / portfolio_value if portfolio_value > 0 else weights
            
            for i in range(self.mc_simulations):
                # Deterministic low-discrepancy surrogate for standard normals
                z = np.array([math.sin((i + 1) * (j + 1) * 0.123) for j in range(len(positions))])

                # Apply correlation structure
                correlated_z = np.dot(L, z)

                # Scale by volatility and time horizon
                asset_returns = correlated_z * volatilities * math.sqrt(time_horizon_days)

                # Calculate portfolio return
                portfolio_return = np.dot(weights, asset_returns)
                simulated_returns.append(portfolio_return)
            
            return simulated_returns
            
        except np.linalg.LinAlgError:
            # Fallback to independent simulation if correlation matrix is not positive definite
            simulated_returns = []
            
            for i in range(self.mc_simulations):
                portfolio_return = 0.0
                portfolio_value = sum(abs(pos.market_value) for pos in positions)
                
                for pos in positions:
                    weight = pos.market_value / portfolio_value if portfolio_value > 0 else 0
                    # Deterministic pseudo-sample for fallback correlated simulation
                    z = math.sin((i + 1) * (idx + 1) * 0.123)
                    random_return = z * pos.volatility * math.sqrt(time_horizon_days)
                    portfolio_return += weight * random_return
                
                simulated_returns.append(portfolio_return)
            
            return simulated_returns
    
    def calculate_portfolio_volatility(self, positions: List[Position]) -> float:
        """Calculate portfolio volatility"""
        
        if not positions:
            return 0.0
        
        symbols = [pos.symbol for pos in positions]
        weights = np.array([pos.market_value for pos in positions])
        portfolio_value = np.sum(np.abs(weights))
        
        if portfolio_value == 0:
            return 0.0
        
        weights = weights / portfolio_value
        
        # Get covariance matrix
        cov_matrix = self._get_covariance_matrix(symbols)
        
        if cov_matrix is not None:
            # Full portfolio volatility calculation
            portfolio_vol = math.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
        else:
            # Fallback: weighted average of individual volatilities (assumes no correlation)
            portfolio_vol = math.sqrt(sum((w * pos.volatility) ** 2 for w, pos in zip(weights, positions)))
        
        return portfolio_vol
    
    def calculate_portfolio_beta(self, positions: List[Position], benchmark_beta: float = 1.0) -> float:
        """Calculate portfolio beta"""
        
        if not positions:
            return 0.0
        
        weights = np.array([pos.market_value for pos in positions])
        portfolio_value = np.sum(np.abs(weights))
        
        if portfolio_value == 0:
            return 0.0
        
        weights = weights / portfolio_value
        betas = np.array([pos.beta for pos in positions])
        
        portfolio_beta = np.dot(weights, betas)
        
        return portfolio_beta
    
    def calculate_concentration_risk(self, positions: List[Position]) -> Dict[str, float]:
        """Calculate various concentration risk metrics"""
        
        if not positions:
            return {"max_position_weight": 0.0, "top_5_concentration": 0.0, "herfindahl_index": 0.0}
        
        # Calculate position weights
        weights = np.array([abs(pos.market_value) for pos in positions])
        total_value = np.sum(weights)
        
        if total_value == 0:
            return {"max_position_weight": 0.0, "top_5_concentration": 0.0, "herfindahl_index": 0.0}
        
        weights = weights / total_value
        
        # Max position weight
        max_weight = np.max(weights)
        
        # Top 5 concentration
        top_5_weights = np.sort(weights)[-5:] if len(weights) >= 5 else weights
        top_5_concentration = np.sum(top_5_weights)
        
        # Herfindahl-Hirschman Index
        hhi = np.sum(weights ** 2)
        
        return {
            "max_position_weight": max_weight,
            "top_5_concentration": top_5_concentration,
            "herfindahl_index": hhi
        }
    
    def calculate_sector_concentration(self, positions: List[Position]) -> Dict[str, float]:
        """Calculate sector concentration"""
        
        if not positions:
            return {}
        
        # Group by sector
        sector_values = defaultdict(float)
        total_value = 0.0
        
        for pos in positions:
            sector_values[pos.sector] += abs(pos.market_value)
            total_value += abs(pos.market_value)
        
        if total_value == 0:
            return {}
        
        # Calculate sector weights
        sector_weights = {sector: value / total_value for sector, value in sector_values.items()}
        
        return sector_weights
    
    def _get_covariance_matrix(self, symbols: List[str]) -> Optional[np.ndarray]:
        """Get or estimate covariance matrix for symbols"""
        
        # Try to get from historical data
        returns_data = self._get_historical_returns(symbols)
        
        if returns_data is not None and not returns_data.empty:
            # Calculate sample covariance matrix
            cov_matrix = returns_data.cov().values
            return cov_matrix
        
        # Return None if no data available
        return None
    
    def _get_correlation_matrix(self, symbols: List[str]) -> Optional[np.ndarray]:
        """Get or estimate correlation matrix for symbols"""
        
        # Try to get from historical data  
        returns_data = self._get_historical_returns(symbols)
        
        if returns_data is not None and not returns_data.empty:
            # Calculate sample correlation matrix
            corr_matrix = returns_data.corr().values
            
            # Ensure positive definite
            eigenvals = np.linalg.eigvals(corr_matrix)
            if np.min(eigenvals) < 1e-8:
                # Add small value to diagonal for numerical stability
                corr_matrix += np.eye(len(symbols)) * 1e-6
            
            return corr_matrix
        
        # Return None if no data available
        return None
    
    def _get_historical_returns(self, symbols: List[str], lookback_days: int = 252) -> Optional[pd.DataFrame]:
        """Get historical returns data"""
        
        # This would typically fetch from database or data provider
        # For now, generate deterministic synthetic returns
        if not hasattr(self, '_simulated_returns') or self._simulated_returns is None:
            dates = pd.date_range(end=datetime.now(), periods=lookback_days, freq='D')
            returns_data = {}

            t = np.linspace(0, 2 * math.pi, lookback_days)
            for idx, symbol in enumerate(symbols):
                # Deterministic seasonal return pattern with small asset-specific offset
                offset = (idx - len(symbols) / 2) * 0.00005
                returns = 0.0005 + 0.0005 * np.sin(t + idx * 0.3) + offset
                returns_data[symbol] = returns

            self._simulated_returns = pd.DataFrame(returns_data, index=dates)
        
        # Return only requested symbols
        available_symbols = [s for s in symbols if s in self._simulated_returns.columns]
        
        if available_symbols:
            return self._simulated_returns[available_symbols].tail(lookback_days)
        
        return None
    
    def _get_z_score(self, confidence_level: float) -> float:
        """Get z-score for given confidence level"""
        
        from scipy.stats import norm
        return norm.ppf(confidence_level)
    
    def _get_expected_shortfall_multiplier(self, confidence_level: float) -> float:
        """Get Expected Shortfall multiplier relative to VaR"""
        
        # Approximate multipliers for normal distribution
        if confidence_level >= 0.99:
            return 1.27
        elif confidence_level >= 0.95:
            return 1.25
        elif confidence_level >= 0.90:
            return 1.22
        else:
            return 1.20


class PortfolioRiskManager:
    """
    Manages portfolio-level risk
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize risk manager
        
        Args:
            config: Risk configuration
        """
        self.config = config
        self.positions: Dict[str, Position] = {}
        self.limits: Dict[str, RiskLimit] = {}
        self.alerts: List[RiskAlert] = []
        
        # Cognitive Risk Dampener (Phase 4C)
        self.cognitive_dampener = get_cognitive_dampener()
        self.current_cognitive_multiplier = 1.0
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Initialize limits
        self._init_limits()
    
    def update_cognitive_state(self, regime: str, confidence: float):
        """
        Updates the global risk multiplier based on LLM insights.
        This is the ONLY way the LLM influences risk.
        """
        with self.lock:
            new_multiplier = self.cognitive_dampener.calculate_multiplier(regime, confidence)
            self.current_cognitive_multiplier = new_multiplier
            print(f"  Cognitive Risk Update: Regime={regime}, Conf={confidence:.2f} -> Multiplier={new_multiplier:.2f}")

    def get_effective_position_limit(self, symbol: str) -> float:
        """
        Returns the position limit adjusted by the cognitive dampener.
        """
        base_limit = self._get_base_limit(symbol)
        return base_limit * self.current_cognitive_multiplier

    def _get_base_limit(self, symbol: str) -> float:
        # Placeholder for actual limit retrieval logic
        # In a real system, this would look up self.limits
        return self.config.get("max_position_size", 100000.0)
    
    def _init_limits(self):
        """Initialize risk limits from config"""
        
        with self.lock:
            # Example: max 10% of portfolio in a single position by default
            default_limit = RiskLimit(
                limit_id="default_position_limit",
                limit_type=RiskLimitType.POSITION_LIMIT,
                limit_value=self.config.get("max_position_size", 100000.0),
                description="Default maximum position size"
            )
            
            self.limits["default_position_limit"] = default_limit
    
    def update_positions(self, positions: List[Position]):
        """Update portfolio positions"""
        
        with self.lock:
            self.positions = {pos.symbol: pos for pos in positions}
            
            # Clear alerts for updated positions
            self.alerts = [alert for alert in self.alerts if alert.limit_id != "position_limit_breach"]
    
    def check_position_limits(self) -> List[RiskAlert]:
        """Check position limits and generate alerts if breached"""
        
        with self.lock:
            current_alerts = []
            
            for pos in self.positions.values():
                if not pos.is_long:  # Only check long positions for limits
                    continue
                
                limit = self.get_effective_position_limit(pos.symbol)
                
                if pos.notional_value > limit:
                    # Breach detected
                    alert = RiskAlert(
                        alert_id=f"position_limit_breach_{pos.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        alert_level=RiskAlertLevel.CRITICAL,
                        limit_id="position_limit_breach",
                        message=f"Position limit breached for {pos.symbol}: {pos.notional_value:.2f} > {limit:.2f}",
                        current_value=pos.notional_value,
                        limit_value=limit,
                        utilization_pct=(pos.notional_value / limit) * 100,
                        scope="POSITION",
                        scope_filter=pos.symbol
                    )
                    
                    current_alerts.append(alert)
            
            # Update active alerts
            self.alerts = current_alerts
            
            return current_alerts
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get high-level risk summary"""
        
        with self.lock:
            alerts = self.check_position_limits()
            
            # Count alerts by level
            alert_counts = {level.value: 0 for level in RiskAlertLevel}
            for alert in alerts:
                alert_counts[alert.alert_level.value] += 1
            
            portfolio_summary = {
                'total_positions': len(self.positions),
                'gross_market_value': sum(pos.market_value for pos in self.positions.values()),
                'net_market_value': sum(pos.market_value for pos in self.positions.values() if pos.is_long),
                'leverage': sum(pos.notional_value for pos in self.positions.values() if pos.is_long) / sum(abs(pos.market_value) for pos in self.positions.values() if pos.is_long) if sum(abs(pos.market_value) for pos in self.positions.values() if pos.is_long) != 0 else 0,
                'long_positions': sum(1 for p in self.positions.values() if p.is_long),
                'short_positions': sum(1 for p in self.positions.values() if p.is_short),
                'long_value': sum(p.market_value for p in self.positions.values() if p.is_long),
                'short_value': sum(p.market_value for p in self.positions.values() if p.is_short)
            }
            
            # Get primary VaR metric
            var_95_1d = None
            if 'var' in self.config:
                var_95_1d = self.config['var'].get('var_95_1d_parametric', {})
            
            summary = {
                'timestamp': datetime.now(),
                'portfolio_value': portfolio_summary.get('gross_market_value', 0),
                'net_exposure': portfolio_summary.get('net_market_value', 0),
                'leverage': portfolio_summary.get('leverage', 0),
                'positions_count': portfolio_summary.get('total_positions', 0),
                'portfolio_volatility': portfolio_summary.get('annualized_volatility', 0),
                'portfolio_beta': portfolio_summary.get('beta', 0),
                'var_95_1d': var_95_1d.get('var_amount', 0) if var_95_1d else 0,
                'var_95_1d_pct': var_95_1d.get('var_percentage', 0) * 100 if var_95_1d else 0,
                'expected_shortfall': var_95_1d.get('expected_shortfall', 0) if var_95_1d else 0,
                'alerts': {
                    'total': len(alerts),
                    'by_level': alert_counts,
                    'breach_count': alert_counts.get('BREACH', 0),
                    'critical_count': alert_counts.get('CRITICAL', 0)
                },
                'risk_status': self._determine_risk_status(alerts)
            }
            
            return summary
    
    def _determine_risk_status(self, alerts: List[RiskAlert]) -> str:
        """Determine overall risk status"""
        
        if any(alert.alert_level == RiskAlertLevel.BREACH for alert in alerts):
            return "BREACH"
        elif any(alert.alert_level == RiskAlertLevel.CRITICAL for alert in alerts):
            return "CRITICAL"
        elif any(alert.alert_level == RiskAlertLevel.WARNING for alert in alerts):
            return "WARNING"
        else:
            return "NORMAL"


