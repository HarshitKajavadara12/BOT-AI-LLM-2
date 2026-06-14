"""
PnL Decomposition Framework
Advanced profit and loss attribution analysis for trading strategies
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import warnings
import math
from collections import defaultdict, deque
import concurrent.futures
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns


class AttributionMethod(Enum):
    """Methods for PnL attribution"""
    BRINSON = "BRINSON"                     # Brinson attribution
    FACTOR_MODEL = "FACTOR_MODEL"           # Factor-based attribution
    HOLDINGS_BASED = "HOLDINGS_BASED"       # Holdings-based attribution
    TRANSACTION_BASED = "TRANSACTION_BASED" # Transaction-based attribution
    RISK_ADJUSTED = "RISK_ADJUSTED"         # Risk-adjusted attribution
    SECTOR_INDUSTRY = "SECTOR_INDUSTRY"     # Sector/industry attribution


class TimePeriod(Enum):
    """Time periods for attribution analysis"""
    DAILY = "DAILY"
    WEEKLY = "WEEKLY"
    MONTHLY = "MONTHLY"
    QUARTERLY = "QUARTERLY"
    ANNUAL = "ANNUAL"
    INCEPTION_TO_DATE = "INCEPTION_TO_DATE"


@dataclass
class Attribution:
    """Attribution result"""
    source: str                             # Attribution source (factor, sector, etc.)
    contribution: float                     # Contribution to total return
    allocation_effect: float = 0.0         # Allocation effect
    selection_effect: float = 0.0          # Selection effect
    interaction_effect: float = 0.0        # Interaction effect
    confidence_interval: Tuple[float, float] = (0.0, 0.0)  # Confidence interval


@dataclass
class PnLBreakdown:
    """Complete PnL breakdown"""
    total_pnl: float
    explained_pnl: float
    unexplained_pnl: float
    attributions: List[Attribution] = field(default_factory=list)
    r_squared: float = 0.0                 # Goodness of fit
    time_period: str = ""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None


@dataclass
class PortfolioHolding:
    """Portfolio holding information"""
    symbol: str
    weight: float                          # Portfolio weight
    return_contribution: float             # Contribution to portfolio return
    sector: Optional[str] = None
    industry: Optional[str] = None
    market_cap: Optional[float] = None
    beta: Optional[float] = None


@dataclass
class TransactionRecord:
    """Transaction record for attribution"""
    timestamp: datetime
    symbol: str
    quantity: float
    price: float
    transaction_cost: float = 0.0
    strategy: Optional[str] = None
    alpha_source: Optional[str] = None


class PnLAttributor(ABC):
    """Abstract base class for PnL attribution"""
    
    @abstractmethod
    def calculate_attribution(self, 
                            portfolio_returns: pd.Series,
                            benchmark_returns: pd.Series,
                            holdings: Optional[pd.DataFrame] = None,
                            factor_exposures: Optional[pd.DataFrame] = None) -> PnLBreakdown:
        """Calculate PnL attribution"""
        pass


class BrinsonAttributor(PnLAttributor):
    """
    Brinson attribution model
    Decomposes excess return into allocation, selection, and interaction effects
    """
    
    def __init__(self, sectors: Optional[List[str]] = None):
        self.sectors = sectors or []
        
    def calculate_attribution(self, 
                            portfolio_returns: pd.Series,
                            benchmark_returns: pd.Series,
                            holdings: Optional[pd.DataFrame] = None,
                            factor_exposures: Optional[pd.DataFrame] = None) -> PnLBreakdown:
        """Calculate Brinson attribution"""
        
        if holdings is None:
            raise ValueError("Holdings data required for Brinson attribution")
        
        # Calculate excess returns
        excess_returns = portfolio_returns - benchmark_returns
        total_excess = excess_returns.sum()
        
        attributions = []
        explained_pnl = 0.0
        
        # Get unique sectors
        if 'sector' not in holdings.columns:
            # Create single "Equity" sector if no sector data
            holdings['sector'] = 'Equity'
        
        sectors = holdings['sector'].unique()
        
        for sector in sectors:
            # Get sector holdings
            sector_holdings = holdings[holdings['sector'] == sector]
            
            if sector_holdings.empty:
                continue
            
            # Calculate sector attribution
            sector_attribution = self._calculate_sector_attribution(
                sector_holdings, portfolio_returns, benchmark_returns
            )
            
            attributions.append(sector_attribution)
            explained_pnl += sector_attribution.contribution
        
        unexplained_pnl = total_excess - explained_pnl
        
        return PnLBreakdown(
            total_pnl=total_excess,
            explained_pnl=explained_pnl,
            unexplained_pnl=unexplained_pnl,
            attributions=attributions,
            r_squared=explained_pnl / total_excess if total_excess != 0 else 0,
            time_period="Custom"
        )
    
    def _calculate_sector_attribution(self, 
                                    sector_holdings: pd.DataFrame,
                                    portfolio_returns: pd.Series,
                                    benchmark_returns: pd.Series) -> Attribution:
        """Calculate attribution for a single sector"""
        
        sector_name = sector_holdings['sector'].iloc[0]
        
        # Portfolio weight in sector
        wp = sector_holdings['weight'].sum()
        
        # Benchmark weight in sector (assume equal weight for simplicity)
        wb = 1.0 / len(sector_holdings)
        
        # Portfolio return in sector
        sector_returns = []
        for _, holding in sector_holdings.iterrows():
            symbol = holding['symbol']
            weight = holding['weight']
            
            # Get symbol returns (simplified - in practice would need asset returns)
            if symbol in portfolio_returns.index:
                symbol_return = portfolio_returns[symbol] if isinstance(portfolio_returns, pd.DataFrame) else 0.0
            else:
                symbol_return = 0.0
            
            sector_returns.append(weight * symbol_return)
        
        rp = sum(sector_returns) / wp if wp > 0 else 0.0
        
        # Benchmark return in sector (simplified)
        rb = benchmark_returns.mean() if len(benchmark_returns) > 0 else 0.0
        
        # Brinson attribution components
        allocation_effect = (wp - wb) * rb
        selection_effect = wb * (rp - rb)
        interaction_effect = (wp - wb) * (rp - rb)
        
        total_contribution = allocation_effect + selection_effect + interaction_effect
        
        return Attribution(
            source=sector_name,
            contribution=total_contribution,
            allocation_effect=allocation_effect,
            selection_effect=selection_effect,
            interaction_effect=interaction_effect
        )


class FactorAttributor(PnLAttributor):
    """
    Factor-based attribution using factor models
    """
    
    def __init__(self, factors: Optional[List[str]] = None, 
                 use_pca: bool = False, n_components: int = 5):
        self.factors = factors or ['Market', 'Size', 'Value', 'Momentum', 'Quality']
        self.use_pca = use_pca
        self.n_components = n_components
        self.factor_model = None
        self.pca_model = None
        
    def calculate_attribution(self, 
                            portfolio_returns: pd.Series,
                            benchmark_returns: pd.Series,
                            holdings: Optional[pd.DataFrame] = None,
                            factor_exposures: Optional[pd.DataFrame] = None) -> PnLBreakdown:
        """Calculate factor-based attribution"""
        
        if factor_exposures is None:
            # Generate synthetic factor exposures if not provided
            factor_exposures = self._generate_synthetic_exposures(portfolio_returns)
        
        # Calculate excess returns
        excess_returns = portfolio_returns - benchmark_returns.reindex(portfolio_returns.index, method='ffill')
        
        # Fit factor model
        self._fit_factor_model(excess_returns, factor_exposures)
        
        # Calculate factor contributions
        attributions = self._calculate_factor_contributions(excess_returns, factor_exposures)
        
        # Calculate explained variance
        predicted_returns = self._predict_returns(factor_exposures)
        explained_pnl = predicted_returns.sum()
        total_pnl = excess_returns.sum()
        unexplained_pnl = total_pnl - explained_pnl
        
        r_squared = 1 - (np.sum((excess_returns - predicted_returns) ** 2) / 
                        np.sum((excess_returns - excess_returns.mean()) ** 2))
        
        return PnLBreakdown(
            total_pnl=total_pnl,
            explained_pnl=explained_pnl,
            unexplained_pnl=unexplained_pnl,
            attributions=attributions,
            r_squared=max(0, r_squared),
            time_period="Custom"
        )
    
    def _generate_synthetic_exposures(self, returns: pd.Series) -> pd.DataFrame:
        """
        DEPRECATED: Generate synthetic factor exposures.
        
        WARNING: This generates FAKE data and should NOT be used in production.
        Use real factor exposures calculated from actual portfolio holdings.
        """
        raise NotImplementedError(
            "Synthetic exposure generation is DISABLED. "
            "Calculate real factor exposures from actual portfolio positions and factor models."
        )
    
    def _fit_factor_model(self, returns: pd.Series, factor_exposures: pd.DataFrame):
        """Fit factor model to returns"""
        
        # Align data
        common_index = returns.index.intersection(factor_exposures.index)
        if len(common_index) < 10:
            raise ValueError("Insufficient overlapping data for factor model")
        
        y = returns.loc[common_index].values
        X = factor_exposures.loc[common_index].values
        
        # Handle missing values
        valid_mask = ~(np.isnan(y) | np.any(np.isnan(X), axis=1))
        if np.sum(valid_mask) < 5:
            raise ValueError("Insufficient valid data for factor model")
        
        y = y[valid_mask]
        X = X[valid_mask]
        
        # Apply PCA if requested
        if self.use_pca:
            self.pca_model = PCA(n_components=min(self.n_components, X.shape[1]))
            X = self.pca_model.fit_transform(X)
        
        # Fit linear regression
        self.factor_model = LinearRegression()
        self.factor_model.fit(X, y)
    
    def _predict_returns(self, factor_exposures: pd.DataFrame) -> pd.Series:
        """Predict returns using factor model"""
        
        if self.factor_model is None:
            return pd.Series(0.0, index=factor_exposures.index)
        
        X = factor_exposures.values
        
        # Handle missing values
        valid_mask = ~np.any(np.isnan(X), axis=1)
        predictions = np.zeros(len(X))
        
        if np.any(valid_mask):
            # Apply same transformations as in fitting
            X_valid = X[valid_mask]
            
            if self.pca_model is not None:
                X_valid = self.pca_model.transform(X_valid)
            
            predictions[valid_mask] = self.factor_model.predict(X_valid)
        
        return pd.Series(predictions, index=factor_exposures.index)
    
    def _calculate_factor_contributions(self, returns: pd.Series, 
                                      factor_exposures: pd.DataFrame) -> List[Attribution]:
        """Calculate individual factor contributions"""
        
        if self.factor_model is None:
            return []
        
        attributions = []
        
        # Get factor coefficients
        if self.use_pca and self.pca_model is not None:
            # For PCA, need to transform back to original factor space
            factor_loadings = self.pca_model.components_.T @ self.factor_model.coef_
        else:
            factor_loadings = self.factor_model.coef_
        
        # Calculate contribution of each factor
        for i, factor in enumerate(self.factors):
            if i < len(factor_loadings):
                # Factor contribution = loading * average exposure * periods
                avg_exposure = factor_exposures[factor].mean()
                contribution = factor_loadings[i] * avg_exposure * len(returns)
                
                # Calculate confidence interval (simplified)
                std_error = np.std(factor_exposures[factor]) / np.sqrt(len(factor_exposures))
                ci_lower = contribution - 1.96 * std_error
                ci_upper = contribution + 1.96 * std_error
                
                attribution = Attribution(
                    source=factor,
                    contribution=contribution,
                    confidence_interval=(ci_lower, ci_upper)
                )
                attributions.append(attribution)
        
        return attributions


class HoldingsBasedAttributor(PnLAttributor):
    """
    Holdings-based attribution analyzing individual position contributions
    """
    
    def __init__(self, rebalance_frequency: str = 'monthly'):
        self.rebalance_frequency = rebalance_frequency
        
    def calculate_attribution(self, 
                            portfolio_returns: pd.Series,
                            benchmark_returns: pd.Series,
                            holdings: Optional[pd.DataFrame] = None,
                            factor_exposures: Optional[pd.DataFrame] = None) -> PnLBreakdown:
        """Calculate holdings-based attribution"""
        
        if holdings is None:
            raise ValueError("Holdings data required for holdings-based attribution")
        
        # Calculate total excess return
        excess_returns = portfolio_returns - benchmark_returns.reindex(portfolio_returns.index, method='ffill')
        total_pnl = excess_returns.sum()
        
        # Calculate individual holding contributions
        attributions = []
        explained_pnl = 0.0
        
        for _, holding in holdings.iterrows():
            symbol = holding['symbol']
            weight = holding['weight']
            
            # Calculate holding contribution
            holding_attribution = self._calculate_holding_contribution(
                symbol, weight, portfolio_returns, benchmark_returns
            )
            
            attributions.append(holding_attribution)
            explained_pnl += holding_attribution.contribution
        
        unexplained_pnl = total_pnl - explained_pnl
        r_squared = explained_pnl / total_pnl if total_pnl != 0 else 0
        
        return PnLBreakdown(
            total_pnl=total_pnl,
            explained_pnl=explained_pnl,
            unexplained_pnl=unexplained_pnl,
            attributions=attributions,
            r_squared=max(0, r_squared),
            time_period="Custom"
        )
    
    def _calculate_holding_contribution(self, symbol: str, weight: float,
                                      portfolio_returns: pd.Series,
                                      benchmark_returns: pd.Series) -> Attribution:
        """Calculate contribution of individual holding"""
        
        # Simplified calculation - in practice would need individual asset returns
        # Using portfolio return as proxy for individual asset performance
        avg_portfolio_return = portfolio_returns.mean()
        avg_benchmark_return = benchmark_returns.mean()
        
        # Contribution = weight * (asset return - benchmark return) * periods
        excess_return = avg_portfolio_return - avg_benchmark_return
        contribution = weight * excess_return * len(portfolio_returns)
        
        return Attribution(
            source=symbol,
            contribution=contribution,
            allocation_effect=weight * avg_benchmark_return * len(portfolio_returns),
            selection_effect=weight * excess_return * len(portfolio_returns)
        )


class TransactionBasedAttributor(PnLAttributor):
    """
    Transaction-based attribution analyzing trading decisions
    """
    
    def __init__(self, transactions: List[TransactionRecord]):
        self.transactions = transactions
        
    def calculate_attribution(self, 
                            portfolio_returns: pd.Series,
                            benchmark_returns: pd.Series,
                            holdings: Optional[pd.DataFrame] = None,
                            factor_exposures: Optional[pd.DataFrame] = None) -> PnLBreakdown:
        """Calculate transaction-based attribution"""
        
        if not self.transactions:
            return PnLBreakdown(
                total_pnl=0.0,
                explained_pnl=0.0,
                unexplained_pnl=0.0,
                attributions=[],
                r_squared=0.0
            )
        
        # Group transactions by strategy/alpha source
        strategy_groups = defaultdict(list)
        for txn in self.transactions:
            key = txn.strategy or txn.alpha_source or 'Unknown'
            strategy_groups[key].append(txn)
        
        # Calculate attribution for each strategy
        attributions = []
        total_contribution = 0.0
        
        for strategy, txns in strategy_groups.items():
            contribution = self._calculate_strategy_contribution(txns, portfolio_returns)
            
            attribution = Attribution(
                source=strategy,
                contribution=contribution
            )
            attributions.append(attribution)
            total_contribution += contribution
        
        # Calculate transaction costs
        total_costs = sum(txn.transaction_cost for txn in self.transactions)
        cost_attribution = Attribution(
            source='Transaction Costs',
            contribution=-total_costs
        )
        attributions.append(cost_attribution)
        total_contribution -= total_costs
        
        total_pnl = portfolio_returns.sum() - benchmark_returns.sum()
        unexplained_pnl = total_pnl - total_contribution
        
        return PnLBreakdown(
            total_pnl=total_pnl,
            explained_pnl=total_contribution,
            unexplained_pnl=unexplained_pnl,
            attributions=attributions,
            r_squared=total_contribution / total_pnl if total_pnl != 0 else 0,
            time_period="Custom"
        )
    
    def _calculate_strategy_contribution(self, transactions: List[TransactionRecord],
                                       portfolio_returns: pd.Series) -> float:
        """Calculate contribution of a trading strategy"""
        
        # Simplified calculation based on transaction timing and size
        total_contribution = 0.0
        
        for txn in transactions:
            # Find closest return date
            txn_date = txn.timestamp
            closest_dates = [d for d in portfolio_returns.index if d >= txn_date]
            
            if closest_dates:
                closest_date = min(closest_dates)
                if closest_date in portfolio_returns.index:
                    # Contribution = position size * subsequent return
                    position_value = txn.quantity * txn.price
                    subsequent_return = portfolio_returns[closest_date]
                    contribution = position_value * subsequent_return
                    total_contribution += contribution
        
        return total_contribution


class RiskAdjustedAttributor(PnLAttributor):
    """
    Risk-adjusted attribution considering volatility and correlation
    """
    
    def __init__(self, risk_model: str = 'historical'):
        self.risk_model = risk_model
        
    def calculate_attribution(self, 
                            portfolio_returns: pd.Series,
                            benchmark_returns: pd.Series,
                            holdings: Optional[pd.DataFrame] = None,
                            factor_exposures: Optional[pd.DataFrame] = None) -> PnLBreakdown:
        """Calculate risk-adjusted attribution"""
        
        # Calculate excess returns
        excess_returns = portfolio_returns - benchmark_returns.reindex(portfolio_returns.index, method='ffill')
        
        # Calculate risk metrics
        portfolio_vol = portfolio_returns.std() * np.sqrt(252)
        benchmark_vol = benchmark_returns.std() * np.sqrt(252)
        
        # Risk-adjusted components
        attributions = []
        
        # Alpha component (risk-adjusted excess return)
        beta = self._calculate_beta(portfolio_returns, benchmark_returns)
        alpha = excess_returns.mean() * 252 - beta * (benchmark_returns.mean() * 252 - 0.02)  # Assume 2% risk-free rate
        
        alpha_attribution = Attribution(
            source='Alpha',
            contribution=alpha * len(excess_returns) / 252
        )
        attributions.append(alpha_attribution)
        
        # Beta component
        beta_attribution = Attribution(
            source='Beta',
            contribution=beta * benchmark_returns.sum()
        )
        attributions.append(beta_attribution)
        
        # Volatility component
        vol_diff = portfolio_vol - benchmark_vol
        vol_attribution = Attribution(
            source='Volatility Effect',
            contribution=vol_diff * excess_returns.mean() * len(excess_returns) / 252
        )
        attributions.append(vol_attribution)
        
        # Specific risk component
        specific_risk = self._calculate_specific_risk(portfolio_returns, benchmark_returns)
        specific_attribution = Attribution(
            source='Specific Risk',
            contribution=specific_risk
        )
        attributions.append(specific_attribution)
        
        explained_pnl = sum(attr.contribution for attr in attributions)
        total_pnl = excess_returns.sum()
        unexplained_pnl = total_pnl - explained_pnl
        
        return PnLBreakdown(
            total_pnl=total_pnl,
            explained_pnl=explained_pnl,
            unexplained_pnl=unexplained_pnl,
            attributions=attributions,
            r_squared=explained_pnl / total_pnl if total_pnl != 0 else 0,
            time_period="Custom"
        )
    
    def _calculate_beta(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate portfolio beta"""
        
        # Align returns
        aligned_data = pd.concat([portfolio_returns, benchmark_returns], axis=1, join='inner')
        if len(aligned_data) < 10:
            return 1.0  # Default beta
        
        portfolio_aligned = aligned_data.iloc[:, 0]
        benchmark_aligned = aligned_data.iloc[:, 1]
        
        # Calculate beta
        covariance = np.cov(portfolio_aligned, benchmark_aligned)[0, 1]
        benchmark_variance = np.var(benchmark_aligned)
        
        return covariance / benchmark_variance if benchmark_variance > 0 else 1.0
    
    def _calculate_specific_risk(self, portfolio_returns: pd.Series, 
                               benchmark_returns: pd.Series) -> float:
        """Calculate specific risk contribution"""
        
        # Calculate tracking error
        excess_returns = portfolio_returns - benchmark_returns.reindex(portfolio_returns.index, method='ffill')
        tracking_error = excess_returns.std() * np.sqrt(252)
        
        # Specific risk approximation
        beta = self._calculate_beta(portfolio_returns, benchmark_returns)
        systematic_risk = beta * benchmark_returns.std() * np.sqrt(252)
        total_risk = portfolio_returns.std() * np.sqrt(252)
        
        specific_risk_vol = np.sqrt(max(0, total_risk**2 - systematic_risk**2))
        
        # Convert to PnL contribution (simplified)
        return specific_risk_vol * excess_returns.mean() * len(excess_returns) / 252


class PnLAnalyzer:
    """
    Main PnL analysis and decomposition framework
    """
    
    def __init__(self):
        self.attributors = {
            AttributionMethod.BRINSON: BrinsonAttributor(),
            AttributionMethod.FACTOR_MODEL: FactorAttributor(),
            AttributionMethod.HOLDINGS_BASED: HoldingsBasedAttributor(),
            AttributionMethod.RISK_ADJUSTED: RiskAdjustedAttributor()
        }
        
    def analyze_pnl(self, 
                   portfolio_returns: pd.Series,
                   benchmark_returns: pd.Series,
                   method: AttributionMethod = AttributionMethod.FACTOR_MODEL,
                   holdings: Optional[pd.DataFrame] = None,
                   factor_exposures: Optional[pd.DataFrame] = None,
                   transactions: Optional[List[TransactionRecord]] = None) -> PnLBreakdown:
        """
        Perform comprehensive PnL analysis
        
        Args:
            portfolio_returns: Portfolio returns time series
            benchmark_returns: Benchmark returns time series  
            method: Attribution method to use
            holdings: Portfolio holdings data
            factor_exposures: Factor exposures data
            transactions: Transaction records
            
        Returns:
            PnLBreakdown with detailed attribution
        """
        
        # Handle transaction-based attribution
        if method == AttributionMethod.TRANSACTION_BASED:
            if transactions is None:
                raise ValueError("Transactions required for transaction-based attribution")
            attributor = TransactionBasedAttributor(transactions)
        else:
            attributor = self.attributors.get(method)
            if attributor is None:
                raise ValueError(f"Unsupported attribution method: {method}")
        
        # Calculate attribution
        breakdown = attributor.calculate_attribution(
            portfolio_returns=portfolio_returns,
            benchmark_returns=benchmark_returns,
            holdings=holdings,
            factor_exposures=factor_exposures
        )
        
        # Add timing information
        breakdown.start_date = portfolio_returns.index[0] if len(portfolio_returns) > 0 else None
        breakdown.end_date = portfolio_returns.index[-1] if len(portfolio_returns) > 0 else None
        
        return breakdown
    
    def multi_method_analysis(self, 
                            portfolio_returns: pd.Series,
                            benchmark_returns: pd.Series,
                            methods: List[AttributionMethod] = None,
                            **kwargs) -> Dict[AttributionMethod, PnLBreakdown]:
        """Run multiple attribution methods and compare results"""
        
        if methods is None:
            methods = [AttributionMethod.FACTOR_MODEL, AttributionMethod.RISK_ADJUSTED, 
                      AttributionMethod.HOLDINGS_BASED]
        
        results = {}
        
        for method in methods:
            try:
                breakdown = self.analyze_pnl(
                    portfolio_returns=portfolio_returns,
                    benchmark_returns=benchmark_returns,
                    method=method,
                    **kwargs
                )
                results[method] = breakdown
            except Exception as e:
                print(f"Attribution method {method.value} failed: {e}")
                continue
        
        return results
    
    def time_series_attribution(self,
                              portfolio_returns: pd.Series,
                              benchmark_returns: pd.Series,
                              period: TimePeriod = TimePeriod.MONTHLY,
                              method: AttributionMethod = AttributionMethod.FACTOR_MODEL,
                              **kwargs) -> List[PnLBreakdown]:
        """Perform attribution analysis over multiple time periods"""
        
        # Define period frequencies
        freq_map = {
            TimePeriod.DAILY: 'D',
            TimePeriod.WEEKLY: 'W',
            TimePeriod.MONTHLY: 'M',
            TimePeriod.QUARTERLY: 'Q',
            TimePeriod.ANNUAL: 'A'
        }
        
        if period not in freq_map:
            # Return single period analysis
            return [self.analyze_pnl(portfolio_returns, benchmark_returns, method, **kwargs)]
        
        # Group returns by period
        freq = freq_map[period]
        portfolio_grouped = portfolio_returns.groupby(pd.Grouper(freq=freq))
        benchmark_grouped = benchmark_returns.groupby(pd.Grouper(freq=freq))
        
        results = []
        
        for (period_start, portfolio_period), (_, benchmark_period) in zip(
            portfolio_grouped, benchmark_grouped):
            
            if len(portfolio_period) > 0 and len(benchmark_period) > 0:
                try:
                    breakdown = self.analyze_pnl(
                        portfolio_period, benchmark_period, method, **kwargs
                    )
                    breakdown.time_period = f"{period.value}_{period_start.strftime('%Y-%m')}"
                    results.append(breakdown)
                except Exception as e:
                    continue
        
        return results
    
    def generate_attribution_report(self, breakdown: PnLBreakdown) -> str:
        """Generate detailed attribution report"""
        
        report = []
        report.append("="*60)
        report.append("PnL ATTRIBUTION ANALYSIS REPORT")
        report.append("="*60)
        
        # Summary
        report.append(f"\nSUMMARY:")
        report.append(f"  Period: {breakdown.start_date} to {breakdown.end_date}" if breakdown.start_date else "  Period: Custom")
        report.append(f"  Total PnL: {breakdown.total_pnl:,.2f}")
        report.append(f"  Explained PnL: {breakdown.explained_pnl:,.2f}")
        report.append(f"  Unexplained PnL: {breakdown.unexplained_pnl:,.2f}")
        report.append(f"  R-Squared: {breakdown.r_squared:.2%}")
        
        # Attribution breakdown
        if breakdown.attributions:
            report.append(f"\nATTRIBUTION BREAKDOWN:")
            report.append(f"  {'Source':<20} {'Contribution':<15} {'Allocation':<12} {'Selection':<12} {'Interaction':<12}")
            report.append("  " + "-"*70)
            
            for attr in sorted(breakdown.attributions, key=lambda x: abs(x.contribution), reverse=True):
                report.append(f"  {attr.source:<20} {attr.contribution:>14.2f} "
                            f"{attr.allocation_effect:>11.2f} {attr.selection_effect:>11.2f} "
                            f"{attr.interaction_effect:>11.2f}")
        
        # Confidence intervals
        ci_attributions = [attr for attr in breakdown.attributions if attr.confidence_interval != (0.0, 0.0)]
        if ci_attributions:
            report.append(f"\nCONFIDENCE INTERVALS (95%):")
            for attr in ci_attributions:
                ci_lower, ci_upper = attr.confidence_interval
                report.append(f"  {attr.source}: [{ci_lower:.2f}, {ci_upper:.2f}]")
        
        return "\n".join(report)
    
    def create_attribution_visualization(self, breakdown: PnLBreakdown, 
                                       save_path: Optional[str] = None):
        """Create visualization of attribution results"""
        
        if not breakdown.attributions:
            print("No attributions to visualize")
            return
        
        # Prepare data
        sources = [attr.source for attr in breakdown.attributions]
        contributions = [attr.contribution for attr in breakdown.attributions]
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Contribution waterfall chart
        cumulative = [0]
        for contrib in contributions:
            cumulative.append(cumulative[-1] + contrib)
        
        colors = ['green' if c > 0 else 'red' for c in contributions]
        ax1.bar(range(len(contributions)), contributions, color=colors, alpha=0.7)
        ax1.set_title('Attribution Contributions')
        ax1.set_xticks(range(len(sources)))
        ax1.set_xticklabels(sources, rotation=45, ha='right')
        ax1.set_ylabel('Contribution')
        ax1.grid(True, alpha=0.3)
        
        # 2. Pie chart of positive contributions
        positive_contribs = [(s, c) for s, c in zip(sources, contributions) if c > 0]
        if positive_contribs:
            pos_sources, pos_values = zip(*positive_contribs)
            ax2.pie(pos_values, labels=pos_sources, autopct='%1.1f%%', startangle=90)
            ax2.set_title('Positive Contributions')
        
        # 3. Allocation vs Selection effects
        allocation_effects = [attr.allocation_effect for attr in breakdown.attributions if attr.allocation_effect != 0]
        selection_effects = [attr.selection_effect for attr in breakdown.attributions if attr.selection_effect != 0]
        
        if allocation_effects and selection_effects:
            x = np.arange(len(allocation_effects))
            width = 0.35
            
            ax3.bar(x - width/2, allocation_effects, width, label='Allocation', alpha=0.7)
            ax3.bar(x + width/2, selection_effects, width, label='Selection', alpha=0.7)
            ax3.set_title('Allocation vs Selection Effects')
            ax3.set_xlabel('Attribution Sources')
            ax3.set_ylabel('Effect')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Summary statistics
        ax4.axis('off')
        summary_text = f"""
        Total PnL: {breakdown.total_pnl:,.0f}
        Explained: {breakdown.explained_pnl:,.0f}
        Unexplained: {breakdown.unexplained_pnl:,.0f}
        R-Squared: {breakdown.r_squared:.1%}
        
        Top Contributors:
        """
        
        # Add top 3 contributors
        sorted_attrs = sorted(breakdown.attributions, key=lambda x: abs(x.contribution), reverse=True)
        for i, attr in enumerate(sorted_attrs[:3]):
            summary_text += f"\n{i+1}. {attr.source}: {attr.contribution:,.0f}"
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Attribution visualization saved to {save_path}")
        
        plt.show()
