"""
Market Impact Models for QUANTUM-FORGE
Implements advanced market impact modeling including linear, square-root, and nonlinear models.
"""

import numpy as np
import pandas as pd
from scipy import optimize, stats, interpolate
from typing import Dict, List, Tuple, Optional, Union, Callable
import warnings
from numba import jit, prange
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

class ImpactType(Enum):
    """Types of market impact."""
    TEMPORARY = "temporary"
    PERMANENT = "permanent"
    TOTAL = "total"

class ImpactModel(Enum):
    """Market impact model types."""
    LINEAR = "linear"
    SQUARE_ROOT = "square_root"
    POWER_LAW = "power_law"
    NONLINEAR = "nonlinear"
    ALMGREN_CHRISS = "almgren_chriss"
    BERTSIMAS_LO = "bertsimas_lo"
    OBIZHAEVA_WANG = "obizhaeva_wang"

class TradeDirection(Enum):
    """Trade direction for impact calculation."""
    BUY = "buy"
    SELL = "sell"

@dataclass
class MarketState:
    """Current market state for impact calculation."""
    bid_price: float
    ask_price: float
    mid_price: float
    spread: float
    bid_size: float
    ask_size: float
    volume_rate: float
    volatility: float
    momentum: float
    order_book_depth: float

@dataclass
class TradeParameters:
    """Parameters for trade execution."""
    quantity: float
    direction: TradeDirection
    urgency: float  # 0 = patient, 1 = urgent
    execution_time: float  # seconds
    participation_rate: float
    order_size: float
    total_position: float

@dataclass
class ImpactPrediction:
    """Market impact prediction result."""
    temporary_impact: float
    permanent_impact: float
    total_impact: float
    temporary_impact_bps: float
    permanent_impact_bps: float
    total_impact_bps: float
    confidence_interval: Tuple[float, float]
    model_used: ImpactModel
    parameters: Dict

class BaseImpactModel(ABC):
    """Abstract base class for market impact models."""
    
    def __init__(self, name: str):
        """Initialize base impact model."""
        self.name = name
        self.parameters = {}
        self.fitted = False
        
    @abstractmethod
    def fit(self, trade_data: pd.DataFrame) -> bool:
        """Fit model to historical trade data."""
        pass
    
    @abstractmethod
    def predict_impact(self, trade_params: TradeParameters, 
                      market_state: MarketState) -> ImpactPrediction:
        """Predict market impact for given trade."""
        pass
    
    def _calculate_participation_rate(self, quantity: float, 
                                    execution_time: float,
                                    volume_rate: float) -> float:
        """Calculate participation rate."""
        if volume_rate <= 0 or execution_time <= 0:
            return 0.0
        
        total_market_volume = volume_rate * execution_time
        return min(1.0, quantity / total_market_volume)

class LinearImpactModel(BaseImpactModel):
    """Linear market impact model."""
    
    def __init__(self):
        """Initialize linear impact model."""
        super().__init__("Linear Impact Model")
        self.temp_coeff = 0.0
        self.perm_coeff = 0.0
        
    def fit(self, trade_data: pd.DataFrame) -> bool:
        """
        Fit linear impact model to historical data.
        
        Expected columns: 'quantity', 'volume_rate', 'temporary_impact', 'permanent_impact'
        """
        required_columns = ['quantity', 'volume_rate', 'temporary_impact', 'permanent_impact']
        
        if not all(col in trade_data.columns for col in required_columns):
            return False
        
        # Calculate participation rates
        participation_rates = trade_data['quantity'] / trade_data['volume_rate']
        
        try:
            # Fit temporary impact: temp_impact = temp_coeff * participation_rate
            temp_model = stats.linregress(participation_rates, trade_data['temporary_impact'])
            self.temp_coeff = temp_model.slope
            
            # Fit permanent impact: perm_impact = perm_coeff * participation_rate
            perm_model = stats.linregress(participation_rates, trade_data['permanent_impact'])
            self.perm_coeff = perm_model.slope
            
            self.parameters = {
                'temp_coeff': self.temp_coeff,
                'perm_coeff': self.perm_coeff,
                'temp_r_squared': temp_model.rvalue**2,
                'perm_r_squared': perm_model.rvalue**2
            }
            
            self.fitted = True
            return True
            
        except Exception as e:
            return False
    
    def predict_impact(self, trade_params: TradeParameters, 
                      market_state: MarketState) -> ImpactPrediction:
        """Predict linear market impact."""
        
        if not self.fitted:
            # Use default coefficients
            self.temp_coeff = 0.5
            self.perm_coeff = 0.1
        
        participation_rate = self._calculate_participation_rate(
            trade_params.quantity, trade_params.execution_time, market_state.volume_rate
        )
        
        # Linear impact calculation
        temporary_impact = self.temp_coeff * participation_rate
        permanent_impact = self.perm_coeff * participation_rate
        total_impact = temporary_impact + permanent_impact
        
        # Convert to basis points
        temporary_impact_bps = temporary_impact * 10000
        permanent_impact_bps = permanent_impact * 10000
        total_impact_bps = total_impact * 10000
        
        # Simple confidence interval (±20%)
        confidence_interval = (total_impact * 0.8, total_impact * 1.2)
        
        return ImpactPrediction(
            temporary_impact=temporary_impact,
            permanent_impact=permanent_impact,
            total_impact=total_impact,
            temporary_impact_bps=temporary_impact_bps,
            permanent_impact_bps=permanent_impact_bps,
            total_impact_bps=total_impact_bps,
            confidence_interval=confidence_interval,
            model_used=ImpactModel.LINEAR,
            parameters=self.parameters
        )

class SquareRootImpactModel(BaseImpactModel):
    """Square-root law market impact model."""
    
    def __init__(self):
        """Initialize square-root impact model."""
        super().__init__("Square Root Impact Model")
        self.temp_coeff = 0.0
        self.perm_coeff = 0.0
        self.volatility_scaling = 0.0
        
    def fit(self, trade_data: pd.DataFrame) -> bool:
        """Fit square-root impact model."""
        required_columns = ['quantity', 'volume_rate', 'volatility', 'temporary_impact', 'permanent_impact']
        
        if not all(col in trade_data.columns for col in required_columns):
            return False
        
        try:
            # Calculate normalized quantities (quantity / sqrt(volume_rate))
            normalized_quantities = (trade_data['quantity'] / 
                                   np.sqrt(trade_data['volume_rate'] + 1e-6))
            
            # Scale by volatility
            volatility_scaled_quantities = normalized_quantities * trade_data['volatility']
            
            # Fit temporary impact
            temp_model = stats.linregress(volatility_scaled_quantities, trade_data['temporary_impact'])
            self.temp_coeff = temp_model.slope
            
            # Fit permanent impact
            perm_model = stats.linregress(volatility_scaled_quantities, trade_data['permanent_impact'])
            self.perm_coeff = perm_model.slope
            
            # Volatility scaling factor
            vol_model = stats.linregress(trade_data['volatility'], 
                                       trade_data['temporary_impact'] + trade_data['permanent_impact'])
            self.volatility_scaling = vol_model.slope
            
            self.parameters = {
                'temp_coeff': self.temp_coeff,
                'perm_coeff': self.perm_coeff,
                'volatility_scaling': self.volatility_scaling,
                'temp_r_squared': temp_model.rvalue**2,
                'perm_r_squared': perm_model.rvalue**2
            }
            
            self.fitted = True
            return True
            
        except Exception as e:
            return False
    
    def predict_impact(self, trade_params: TradeParameters, 
                      market_state: MarketState) -> ImpactPrediction:
        """Predict square-root market impact."""
        
        if not self.fitted:
            # Default coefficients
            self.temp_coeff = 0.3
            self.perm_coeff = 0.05
            self.volatility_scaling = 0.8
        
        # Square-root law: impact ~ sqrt(quantity / volume_rate) * volatility
        normalized_quantity = np.sqrt(trade_params.quantity / max(market_state.volume_rate, 1e-6))
        volatility_factor = market_state.volatility * self.volatility_scaling
        
        # Calculate impacts
        temporary_impact = self.temp_coeff * normalized_quantity * volatility_factor
        permanent_impact = self.perm_coeff * normalized_quantity * volatility_factor
        total_impact = temporary_impact + permanent_impact
        
        # Convert to basis points
        temporary_impact_bps = temporary_impact * 10000
        permanent_impact_bps = permanent_impact * 10000
        total_impact_bps = total_impact * 10000
        
        # Confidence interval based on volatility
        uncertainty = market_state.volatility * 0.5
        confidence_interval = (
            total_impact * (1 - uncertainty),
            total_impact * (1 + uncertainty)
        )
        
        return ImpactPrediction(
            temporary_impact=temporary_impact,
            permanent_impact=permanent_impact,
            total_impact=total_impact,
            temporary_impact_bps=temporary_impact_bps,
            permanent_impact_bps=permanent_impact_bps,
            total_impact_bps=total_impact_bps,
            confidence_interval=confidence_interval,
            model_used=ImpactModel.SQUARE_ROOT,
            parameters=self.parameters
        )

class PowerLawImpactModel(BaseImpactModel):
    """Power-law market impact model."""
    
    def __init__(self):
        """Initialize power-law impact model."""
        super().__init__("Power Law Impact Model")
        self.temp_coeff = 0.0
        self.perm_coeff = 0.0
        self.temp_exponent = 0.5
        self.perm_exponent = 0.3
        
    def fit(self, trade_data: pd.DataFrame) -> bool:
        """Fit power-law impact model."""
        required_columns = ['quantity', 'volume_rate', 'temporary_impact', 'permanent_impact']
        
        if not all(col in trade_data.columns for col in required_columns):
            return False
        
        try:
            participation_rates = trade_data['quantity'] / trade_data['volume_rate']
            
            # Filter out zero or negative values
            valid_mask = (participation_rates > 0) & (trade_data['temporary_impact'] > 0)
            valid_participation = participation_rates[valid_mask]
            valid_temp_impact = trade_data['temporary_impact'][valid_mask]
            valid_perm_impact = trade_data['permanent_impact'][valid_mask]
            
            if len(valid_participation) < 10:
                return False
            
            # Fit temporary impact: log(temp_impact) = log(temp_coeff) + temp_exp * log(participation)
            log_participation = np.log(valid_participation)
            log_temp_impact = np.log(valid_temp_impact)
            
            temp_model = stats.linregress(log_participation, log_temp_impact)
            self.temp_coeff = np.exp(temp_model.intercept)
            self.temp_exponent = temp_model.slope
            
            # Fit permanent impact
            valid_perm_mask = valid_perm_impact > 0
            if np.sum(valid_perm_mask) > 5:
                log_perm_impact = np.log(valid_perm_impact[valid_perm_mask])
                log_participation_perm = log_participation[valid_perm_mask]
                
                perm_model = stats.linregress(log_participation_perm, log_perm_impact)
                self.perm_coeff = np.exp(perm_model.intercept)
                self.perm_exponent = perm_model.slope
            else:
                self.perm_coeff = 0.1
                self.perm_exponent = 0.3
            
            self.parameters = {
                'temp_coeff': self.temp_coeff,
                'perm_coeff': self.perm_coeff,
                'temp_exponent': self.temp_exponent,
                'perm_exponent': self.perm_exponent,
                'temp_r_squared': temp_model.rvalue**2 if 'temp_model' in locals() else 0
            }
            
            self.fitted = True
            return True
            
        except Exception as e:
            return False
    
    def predict_impact(self, trade_params: TradeParameters, 
                      market_state: MarketState) -> ImpactPrediction:
        """Predict power-law market impact."""
        
        if not self.fitted:
            # Default parameters
            self.temp_coeff = 0.2
            self.perm_coeff = 0.05
            self.temp_exponent = 0.6
            self.perm_exponent = 0.4
        
        participation_rate = self._calculate_participation_rate(
            trade_params.quantity, trade_params.execution_time, market_state.volume_rate
        )
        
        # Power-law impact: impact = coeff * participation_rate^exponent
        temporary_impact = self.temp_coeff * (participation_rate ** self.temp_exponent)
        permanent_impact = self.perm_coeff * (participation_rate ** self.perm_exponent)
        total_impact = temporary_impact + permanent_impact
        
        # Convert to basis points
        temporary_impact_bps = temporary_impact * 10000
        permanent_impact_bps = permanent_impact * 10000
        total_impact_bps = total_impact * 10000
        
        # Confidence interval
        confidence_interval = (total_impact * 0.7, total_impact * 1.3)
        
        return ImpactPrediction(
            temporary_impact=temporary_impact,
            permanent_impact=permanent_impact,
            total_impact=total_impact,
            temporary_impact_bps=temporary_impact_bps,
            permanent_impact_bps=permanent_impact_bps,
            total_impact_bps=total_impact_bps,
            confidence_interval=confidence_interval,
            model_used=ImpactModel.POWER_LAW,
            parameters=self.parameters
        )

class AlmgrenChrissModel(BaseImpactModel):
    """Almgren-Chriss market impact model."""
    
    def __init__(self):
        """Initialize Almgren-Chriss model."""
        super().__init__("Almgren-Chriss Impact Model")
        self.gamma = 0.0  # Permanent impact coefficient
        self.eta = 0.0    # Temporary impact coefficient
        self.sigma = 0.0  # Volatility parameter
        
    def fit(self, trade_data: pd.DataFrame) -> bool:
        """Fit Almgren-Chriss model."""
        required_columns = ['quantity', 'volume_rate', 'volatility', 'temporary_impact', 'permanent_impact']
        
        if not all(col in trade_data.columns for col in required_columns):
            return False
        
        try:
            # Calculate trading rates (quantity per unit time per volume)
            trading_rates = trade_data['quantity'] / (trade_data['volume_rate'] + 1e-6)
            
            # Fit gamma (permanent impact)
            # In A-C model: permanent_impact = gamma * quantity
            gamma_model = stats.linregress(trade_data['quantity'], trade_data['permanent_impact'])
            self.gamma = gamma_model.slope
            
            # Fit eta (temporary impact)
            # In A-C model: temporary_impact = eta * trading_rate
            eta_model = stats.linregress(trading_rates, trade_data['temporary_impact'])
            self.eta = eta_model.slope
            
            # Estimate volatility parameter
            self.sigma = np.mean(trade_data['volatility'])
            
            self.parameters = {
                'gamma': self.gamma,
                'eta': self.eta,
                'sigma': self.sigma,
                'gamma_r_squared': gamma_model.rvalue**2,
                'eta_r_squared': eta_model.rvalue**2
            }
            
            self.fitted = True
            return True
            
        except Exception as e:
            return False
    
    def predict_impact(self, trade_params: TradeParameters, 
                      market_state: MarketState) -> ImpactPrediction:
        """Predict impact using Almgren-Chriss model."""
        
        if not self.fitted:
            # Default A-C parameters
            self.gamma = 0.1 * market_state.volatility
            self.eta = 0.5 * market_state.spread
            self.sigma = market_state.volatility
        
        # Almgren-Chriss formulation
        # Permanent impact = gamma * quantity
        permanent_impact = self.gamma * trade_params.quantity / market_state.volume_rate
        
        # Temporary impact = eta * trading_rate
        trading_rate = trade_params.quantity / max(trade_params.execution_time, 1e-6)
        temporary_impact = self.eta * trading_rate / market_state.volume_rate
        
        total_impact = temporary_impact + permanent_impact
        
        # Scale by volatility
        volatility_scaling = market_state.volatility / max(self.sigma, 1e-6)
        temporary_impact *= volatility_scaling
        permanent_impact *= volatility_scaling
        total_impact *= volatility_scaling
        
        # Convert to basis points
        temporary_impact_bps = temporary_impact * 10000
        permanent_impact_bps = permanent_impact * 10000
        total_impact_bps = total_impact * 10000
        
        # Confidence interval based on model uncertainty
        model_uncertainty = 0.3  # 30% uncertainty
        confidence_interval = (
            total_impact * (1 - model_uncertainty),
            total_impact * (1 + model_uncertainty)
        )
        
        return ImpactPrediction(
            temporary_impact=temporary_impact,
            permanent_impact=permanent_impact,
            total_impact=total_impact,
            temporary_impact_bps=temporary_impact_bps,
            permanent_impact_bps=permanent_impact_bps,
            total_impact_bps=total_impact_bps,
            confidence_interval=confidence_interval,
            model_used=ImpactModel.ALMGREN_CHRISS,
            parameters=self.parameters
        )

class NonlinearImpactModel(BaseImpactModel):
    """Nonlinear market impact model with regime detection."""
    
    def __init__(self):
        """Initialize nonlinear impact model."""
        super().__init__("Nonlinear Impact Model")
        self.regimes = {}
        self.regime_classifier = None
        
    def fit(self, trade_data: pd.DataFrame) -> bool:
        """Fit nonlinear impact model with regime detection."""
        required_columns = ['quantity', 'volume_rate', 'volatility', 'spread', 
                          'temporary_impact', 'permanent_impact']
        
        if not all(col in trade_data.columns for col in required_columns):
            return False
        
        try:
            # Classify market regimes based on volatility and spread
            vol_thresh = np.percentile(trade_data['volatility'], 67)
            spread_thresh = np.percentile(trade_data['spread'], 67)
            
            # Three regimes: Normal, Volatile, Stressed
            regimes = []
            for _, row in trade_data.iterrows():
                if row['volatility'] > vol_thresh and row['spread'] > spread_thresh:
                    regimes.append('stressed')
                elif row['volatility'] > vol_thresh or row['spread'] > spread_thresh:
                    regimes.append('volatile')
                else:
                    regimes.append('normal')
            
            trade_data['regime'] = regimes
            
            # Fit models for each regime
            for regime in ['normal', 'volatile', 'stressed']:
                regime_data = trade_data[trade_data['regime'] == regime]
                
                if len(regime_data) < 10:
                    continue
                
                # Fit power-law model for this regime
                participation_rates = regime_data['quantity'] / regime_data['volume_rate']
                
                # Use robust fitting (median-based)
                temp_impacts = regime_data['temporary_impact']
                perm_impacts = regime_data['permanent_impact']
                
                # Fit using quantile regression or robust methods
                if len(participation_rates) > 5:
                    # Simple percentile-based fitting
                    p_rates_sorted = np.sort(participation_rates)
                    temp_sorted = np.sort(temp_impacts)
                    
                    # Use interquartile range for robust fitting
                    q25_idx = int(len(p_rates_sorted) * 0.25)
                    q75_idx = int(len(p_rates_sorted) * 0.75)
                    
                    temp_coeff = np.median(temp_sorted) / np.median(p_rates_sorted)**0.6
                    perm_coeff = np.median(perm_impacts) / np.median(p_rates_sorted)**0.4
                    
                    self.regimes[regime] = {
                        'temp_coeff': temp_coeff,
                        'perm_coeff': perm_coeff,
                        'temp_exponent': 0.6,
                        'perm_exponent': 0.4,
                        'vol_threshold': vol_thresh,
                        'spread_threshold': spread_thresh,
                        'sample_count': len(regime_data)
                    }
            
            self.parameters = {
                'regimes': list(self.regimes.keys()),
                'vol_threshold': vol_thresh,
                'spread_threshold': spread_thresh
            }
            
            self.fitted = len(self.regimes) > 0
            return self.fitted
            
        except Exception as e:
            return False
    
    def _classify_regime(self, market_state: MarketState) -> str:
        """Classify current market regime."""
        if not self.fitted or not self.parameters:
            return 'normal'
        
        vol_thresh = self.parameters.get('vol_threshold', 0.02)
        spread_thresh = self.parameters.get('spread_threshold', 0.001)
        
        if (market_state.volatility > vol_thresh and 
            market_state.spread > spread_thresh):
            return 'stressed'
        elif (market_state.volatility > vol_thresh or 
              market_state.spread > spread_thresh):
            return 'volatile'
        else:
            return 'normal'
    
    def predict_impact(self, trade_params: TradeParameters, 
                      market_state: MarketState) -> ImpactPrediction:
        """Predict nonlinear market impact."""
        
        current_regime = self._classify_regime(market_state)
        
        if not self.fitted or current_regime not in self.regimes:
            # Default to power-law with regime adjustments
            base_temp_coeff = 0.2
            base_perm_coeff = 0.05
            temp_exponent = 0.6
            perm_exponent = 0.4
            
            # Regime adjustments
            if current_regime == 'volatile':
                base_temp_coeff *= 1.5
                base_perm_coeff *= 1.2
            elif current_regime == 'stressed':
                base_temp_coeff *= 2.0
                base_perm_coeff *= 1.5
                temp_exponent = 0.8  # More nonlinear in stressed conditions
        else:
            regime_params = self.regimes[current_regime]
            base_temp_coeff = regime_params['temp_coeff']
            base_perm_coeff = regime_params['perm_coeff']
            temp_exponent = regime_params['temp_exponent']
            perm_exponent = regime_params['perm_exponent']
        
        participation_rate = self._calculate_participation_rate(
            trade_params.quantity, trade_params.execution_time, market_state.volume_rate
        )
        
        # Nonlinear impact calculation
        temporary_impact = base_temp_coeff * (participation_rate ** temp_exponent)
        permanent_impact = base_perm_coeff * (participation_rate ** perm_exponent)
        
        # Additional nonlinear adjustments
        urgency_factor = 1 + (trade_params.urgency - 0.5) * 0.5
        volatility_factor = 1 + market_state.volatility * 2
        
        temporary_impact *= urgency_factor * volatility_factor
        permanent_impact *= np.sqrt(volatility_factor)
        
        total_impact = temporary_impact + permanent_impact
        
        # Convert to basis points
        temporary_impact_bps = temporary_impact * 10000
        permanent_impact_bps = permanent_impact * 10000
        total_impact_bps = total_impact * 10000
        
        # Wider confidence intervals for nonlinear model
        uncertainty = 0.4 if current_regime == 'stressed' else 0.25
        confidence_interval = (
            total_impact * (1 - uncertainty),
            total_impact * (1 + uncertainty)
        )
        
        return ImpactPrediction(
            temporary_impact=temporary_impact,
            permanent_impact=permanent_impact,
            total_impact=total_impact,
            temporary_impact_bps=temporary_impact_bps,
            permanent_impact_bps=permanent_impact_bps,
            total_impact_bps=total_impact_bps,
            confidence_interval=confidence_interval,
            model_used=ImpactModel.NONLINEAR,
            parameters={
                'regime': current_regime,
                'temp_coeff': base_temp_coeff,
                'perm_coeff': base_perm_coeff,
                'temp_exponent': temp_exponent,
                'perm_exponent': perm_exponent
            }
        )

class MarketImpactEngine:
    """Main market impact modeling engine."""
    
    def __init__(self):
        """Initialize market impact engine."""
        self.models = {
            ImpactModel.LINEAR: LinearImpactModel(),
            ImpactModel.SQUARE_ROOT: SquareRootImpactModel(),
            ImpactModel.POWER_LAW: PowerLawImpactModel(),
            ImpactModel.ALMGREN_CHRISS: AlmgrenChrissModel(),
            ImpactModel.NONLINEAR: NonlinearImpactModel()
        }
        self.default_model = ImpactModel.SQUARE_ROOT
        self.model_performance = {}
        
    def fit_models(self, trade_data: pd.DataFrame) -> Dict[ImpactModel, bool]:
        """Fit all available models to historical data."""
        fitting_results = {}
        
        for model_type, model in self.models.items():
            success = model.fit(trade_data)
            fitting_results[model_type] = success
            
            if success:
                print(f"Successfully fitted {model_type.value} model")
            else:
                print(f"Failed to fit {model_type.value} model")
        
        return fitting_results
    
    def predict_impact(self, trade_params: TradeParameters,
                      market_state: MarketState,
                      model_type: Optional[ImpactModel] = None) -> ImpactPrediction:
        """Predict market impact using specified or default model."""
        
        if model_type is None:
            model_type = self.default_model
        
        if model_type not in self.models:
            model_type = self.default_model
        
        model = self.models[model_type]
        return model.predict_impact(trade_params, market_state)
    
    def compare_models(self, trade_params: TradeParameters,
                      market_state: MarketState) -> Dict[ImpactModel, ImpactPrediction]:
        """Compare predictions across all models."""
        
        predictions = {}
        
        for model_type, model in self.models.items():
            try:
                prediction = model.predict_impact(trade_params, market_state)
                predictions[model_type] = prediction
            except Exception as e:
                print(f"Error with {model_type.value}: {e}")
        
        return predictions
    
    def select_best_model(self, trade_params: TradeParameters,
                         market_state: MarketState,
                         historical_performance: Optional[Dict] = None) -> ImpactModel:
        """Select best model based on market conditions and historical performance."""
        
        # Default model selection logic
        if market_state.volatility > 0.04:  # High volatility
            return ImpactModel.NONLINEAR
        elif market_state.volume_rate < 1000:  # Low liquidity
            return ImpactModel.POWER_LAW
        elif trade_params.quantity > 50000:  # Large orders
            return ImpactModel.ALMGREN_CHRISS
        else:
            return ImpactModel.SQUARE_ROOT
    
    def calculate_execution_cost(self, trade_params: TradeParameters,
                               market_state: MarketState,
                               model_type: Optional[ImpactModel] = None) -> Dict:
        """Calculate total execution cost including impact."""
        
        prediction = self.predict_impact(trade_params, market_state, model_type)
        
        # Base execution cost
        notional_value = trade_params.quantity * market_state.mid_price
        
        # Impact costs
        temporary_cost = prediction.temporary_impact * notional_value
        permanent_cost = prediction.permanent_impact * notional_value
        total_impact_cost = temporary_cost + permanent_cost
        
        # Spread cost (half-spread for market orders)
        spread_cost = (market_state.spread / 2) * notional_value
        
        # Commission (simplified)
        commission_rate = 0.0005  # 5 basis points
        commission_cost = notional_value * commission_rate
        
        total_cost = total_impact_cost + spread_cost + commission_cost
        
        return {
            'notional_value': notional_value,
            'temporary_impact_cost': temporary_cost,
            'permanent_impact_cost': permanent_cost,
            'total_impact_cost': total_impact_cost,
            'spread_cost': spread_cost,
            'commission_cost': commission_cost,
            'total_execution_cost': total_cost,
            'cost_breakdown_bps': {
                'temporary_impact': prediction.temporary_impact_bps,
                'permanent_impact': prediction.permanent_impact_bps,
                'spread': (spread_cost / notional_value) * 10000,
                'commission': commission_rate * 10000,
                'total': (total_cost / notional_value) * 10000
            },
            'model_used': prediction.model_used.value
        }

# Example usage and testing
if __name__ == "__main__":
    print("Testing Market Impact Models...")
    
    # Generate synthetic trade data
    np.random.seed(42)
    n_trades = 1000
    
    # Create realistic trade dataset
    trade_data = pd.DataFrame({
        'quantity': np.random.lognormal(mean=8, sigma=1.5, size=n_trades),  # Log-normal distribution
        'volume_rate': np.random.gamma(shape=2, scale=1000, size=n_trades),  # Gamma distribution
        'volatility': np.random.gamma(shape=2, scale=0.01, size=n_trades),  # Volatility
        'spread': np.random.exponential(scale=0.001, size=n_trades),  # Spread
    })
    
    # Generate synthetic impact data based on realistic relationships
    participation_rates = trade_data['quantity'] / trade_data['volume_rate']
    
    # Temporary impact ~ sqrt(participation_rate) * volatility
    trade_data['temporary_impact'] = (
        0.3 * np.sqrt(participation_rates) * trade_data['volatility'] +
        np.random.normal(0, 0.001, n_trades)  # Noise
    )
    
    # Permanent impact ~ participation_rate^0.6 * volatility * 0.2
    trade_data['permanent_impact'] = (
        0.05 * (participation_rates ** 0.6) * trade_data['volatility'] +
        np.random.normal(0, 0.0005, n_trades)  # Noise
    )
    
    # Ensure non-negative impacts
    trade_data['temporary_impact'] = np.maximum(trade_data['temporary_impact'], 0)
    trade_data['permanent_impact'] = np.maximum(trade_data['permanent_impact'], 0)
    
    print(f"Generated {n_trades} synthetic trades")
    print(f"Quantity range: {trade_data['quantity'].min():.0f} - {trade_data['quantity'].max():.0f}")
    print(f"Volume rate range: {trade_data['volume_rate'].min():.0f} - {trade_data['volume_rate'].max():.0f}")
    print(f"Participation rate range: {participation_rates.min():.4f} - {participation_rates.max():.4f}")
    
    # Initialize impact engine
    impact_engine = MarketImpactEngine()
    
    # Fit models
    print(f"\nFitting market impact models...")
    fitting_results = impact_engine.fit_models(trade_data)
    
    successful_models = [model for model, success in fitting_results.items() if success]
    print(f"Successfully fitted {len(successful_models)} models: {[m.value for m in successful_models]}")
    
    # Test predictions
    print(f"\nTesting impact predictions...")
    
    # Sample market state
    market_state = MarketState(
        bid_price=99.95,
        ask_price=100.05,
        mid_price=100.00,
        spread=0.10,
        bid_size=1000,
        ask_size=1200,
        volume_rate=2000,
        volatility=0.025,
        momentum=0.001,
        order_book_depth=5000
    )
    
    # Sample trade parameters
    test_trades = [
        TradeParameters(
            quantity=5000,
            direction=TradeDirection.BUY,
            urgency=0.3,
            execution_time=300,  # 5 minutes
            participation_rate=0.1,
            order_size=500,
            total_position=5000
        ),
        TradeParameters(
            quantity=50000,
            direction=TradeDirection.SELL,
            urgency=0.8,
            execution_time=60,   # 1 minute
            participation_rate=0.4,
            order_size=2000,
            total_position=50000
        ),
        TradeParameters(
            quantity=1000,
            direction=TradeDirection.BUY,
            urgency=0.5,
            execution_time=600,  # 10 minutes
            participation_rate=0.02,
            order_size=100,
            total_position=1000
        )
    ]
    
    for i, trade_params in enumerate(test_trades, 1):
        print(f"\n--- Trade Scenario {i} ---")
        print(f"Quantity: {trade_params.quantity:,}, Direction: {trade_params.direction.value}")
        print(f"Urgency: {trade_params.urgency:.1f}, Execution time: {trade_params.execution_time}s")
        
        # Compare all models
        predictions = impact_engine.compare_models(trade_params, market_state)
        
        print(f"{'Model':<15} {'Temp (bp)':<10} {'Perm (bp)':<10} {'Total (bp)':<10} {'Total Cost'}")
        print("-" * 65)
        
        for model_type, prediction in predictions.items():
            total_cost = impact_engine.calculate_execution_cost(
                trade_params, market_state, model_type
            )
            
            print(f"{model_type.value[:14]:<15} "
                  f"{prediction.temporary_impact_bps:<10.2f} "
                  f"{prediction.permanent_impact_bps:<10.2f} "
                  f"{prediction.total_impact_bps:<10.2f} "
                  f"${total_cost['total_execution_cost']:,.0f}")
        
        # Select best model
        best_model = impact_engine.select_best_model(trade_params, market_state)
        print(f"Recommended model: {best_model.value}")
    
    # Test model comparison across different market conditions
    print(f"\n--- Market Condition Sensitivity ---")
    
    market_conditions = [
        ("Normal", MarketState(99.95, 100.05, 100.00, 0.10, 1000, 1200, 2000, 0.015, 0.001, 5000)),
        ("Volatile", MarketState(99.90, 100.10, 100.00, 0.20, 800, 900, 1500, 0.045, 0.005, 3000)),
        ("Illiquid", MarketState(99.98, 100.02, 100.00, 0.04, 200, 250, 500, 0.020, 0.002, 1000)),
        ("Stressed", MarketState(99.85, 100.15, 100.00, 0.30, 300, 400, 800, 0.060, 0.010, 2000))
    ]
    
    base_trade = TradeParameters(10000, TradeDirection.BUY, 0.5, 180, 0.15, 1000, 10000)
    
    print(f"Impact predictions for 10,000 share buy order across market conditions:")
    print(f"{'Condition':<10} {'Model':<15} {'Temp (bp)':<10} {'Perm (bp)':<10} {'Total (bp)':<10}")
    print("-" * 60)
    
    for condition_name, market_state in market_conditions:
        best_model = impact_engine.select_best_model(base_trade, market_state)
        prediction = impact_engine.predict_impact(base_trade, market_state, best_model)
        
        print(f"{condition_name:<10} "
              f"{best_model.value[:14]:<15} "
              f"{prediction.temporary_impact_bps:<10.2f} "
              f"{prediction.permanent_impact_bps:<10.2f} "
              f"{prediction.total_impact_bps:<10.2f}")
    
    print("\nMarket impact models testing completed successfully!")