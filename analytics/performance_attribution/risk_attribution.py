"""
Risk Attribution Engine for QUANTUM-FORGE
Implements comprehensive risk factor attribution, decomposition analysis,
and factor exposure measurement for quantitative trading portfolios.
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import warnings
from dataclasses import dataclass
from enum import Enum
import time
from datetime import datetime, timedelta
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.optimize import minimize
from scipy.linalg import LinAlgError
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
warnings.filterwarnings('ignore')

class RiskFactorType(Enum):
    """Types of risk factors."""
    MARKET = "market"
    SECTOR = "sector"
    STYLE = "style"
    COUNTRY = "country"
    CURRENCY = "currency"
    VOLATILITY = "volatility"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    CARRY = "carry"
    QUALITY = "quality"
    GROWTH = "growth"
    VALUE = "value"
    SIZE = "size"
    PROFITABILITY = "profitability"
    INVESTMENT = "investment"
    LEVERAGE = "leverage"
    LIQUIDITY = "liquidity"
    CREDIT = "credit"
    TERM_STRUCTURE = "term_structure"
    INFLATION = "inflation"
    CUSTOM = "custom"

class AttributionMethod(Enum):
    """Risk attribution methods."""
    FACTOR_REGRESSION = "factor_regression"
    PRINCIPAL_COMPONENT = "principal_component"
    RISK_MODEL = "risk_model"
    BRINSON_ATTRIBUTION = "brinson_attribution"
    STYLE_ATTRIBUTION = "style_attribution"
    SECTOR_ATTRIBUTION = "sector_attribution"

@dataclass
class RiskFactorExposure:
    """Risk factor exposure information."""
    factor_name: str
    factor_type: RiskFactorType
    exposure: float
    t_statistic: Optional[float] = None
    p_value: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    r_squared: Optional[float] = None

@dataclass
class AttributionResult:
    """Risk attribution result."""
    factor_name: str
    factor_type: RiskFactorType
    contribution: float
    exposure: float
    factor_return: float
    risk_contribution: float
    tracking_error_contribution: float
    active_weight: Optional[float] = None
    benchmark_exposure: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class RiskDecomposition:
    """Complete risk decomposition result."""
    total_risk: float
    systematic_risk: float
    idiosyncratic_risk: float
    factor_contributions: List[AttributionResult]
    explained_variance: float
    residual_variance: float
    diversification_ratio: float

class FamaFrenchFactors:
    """Fama-French factor model implementation."""
    
    def __init__(self):
        """Initialize Fama-French factor model."""
        self.factors = None
        self.factor_loadings = None
        
    def create_synthetic_factors(self, market_returns: pd.Series, 
                               n_portfolios: int = 25) -> pd.DataFrame:
        """
        DEPRECATED: Create synthetic Fama-French style factors.
        
        WARNING: This generates FAKE data and should NOT be used in production.
        Use real Fama-French factors from Kenneth French's data library instead.
        This method exists only for backwards compatibility with old test code.
        """
        raise NotImplementedError(
            "Synthetic factor generation is DISABLED. "
            "Use real Fama-French factors from data providers like Kenneth French Data Library, "
            "Bloomberg, or construct factors from real asset universe data."
        )

class RiskAttributionEngine:
    """Comprehensive risk attribution and factor analysis engine."""
    
    def __init__(self, risk_free_rate: float = 0.0):
        """Initialize risk attribution engine."""
        self.risk_free_rate = risk_free_rate
        self.factor_models = {}
        self.exposures_cache = {}
        self.covariance_matrices = {}
        
    def estimate_factor_exposures(self, returns: pd.Series, 
                                factors: pd.DataFrame,
                                method: str = 'ols',
                                rolling_window: Optional[int] = None) -> Dict[str, RiskFactorExposure]:
        """Estimate factor exposures using regression analysis."""
        
        # Align data
        aligned_data = pd.concat([returns, factors], axis=1, join='inner')
        
        if len(aligned_data) < len(factors.columns) + 10:
            raise ValueError("Insufficient data for factor exposure estimation")
        
        portfolio_returns = aligned_data.iloc[:, 0]
        factor_returns = aligned_data.iloc[:, 1:]
        
        exposures = {}
        
        if rolling_window is None:
            # Single period estimation
            exposures_result = self._estimate_exposures_single_period(
                portfolio_returns, factor_returns, method
            )
            
            for factor_name, result in exposures_result.items():
                exposures[factor_name] = result
                
        else:
            # Rolling window estimation
            rolling_exposures = self._estimate_exposures_rolling(
                portfolio_returns, factor_returns, rolling_window, method
            )
            
            # Use latest exposures
            for factor_name in factor_returns.columns:
                latest_exposure = rolling_exposures[factor_name].iloc[-1]
                
                exposures[factor_name] = RiskFactorExposure(
                    factor_name=factor_name,
                    factor_type=self._infer_factor_type(factor_name),
                    exposure=latest_exposure,
                    t_statistic=None,  # Would need more complex calculation for rolling
                    p_value=None
                )
        
        return exposures
    
    def _estimate_exposures_single_period(self, portfolio_returns: pd.Series,
                                        factor_returns: pd.DataFrame,
                                        method: str) -> Dict[str, RiskFactorExposure]:
        """Estimate factor exposures for single period."""
        
        X = factor_returns.values
        y = portfolio_returns.values
        
        # Add intercept
        X_with_intercept = np.column_stack([np.ones(len(X)), X])
        
        exposures = {}
        
        if method == 'ols':
            # Ordinary Least Squares
            try:
                coeffs = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
                
                # Calculate standard errors and t-statistics
                residuals = y - X_with_intercept @ coeffs
                mse = np.sum(residuals**2) / (len(y) - len(coeffs))
                
                try:
                    cov_matrix = mse * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
                    std_errors = np.sqrt(np.diag(cov_matrix))
                    t_stats = coeffs / std_errors
                    
                    # P-values (two-tailed)
                    df = len(y) - len(coeffs)
                    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df))
                    
                except LinAlgError:
                    std_errors = np.full_like(coeffs, np.nan)
                    t_stats = np.full_like(coeffs, np.nan)
                    p_values = np.full_like(coeffs, np.nan)
                
                # R-squared
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((y - np.mean(y))**2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
            except LinAlgError:
                # Fallback to pseudoinverse
                coeffs = np.linalg.pinv(X_with_intercept) @ y
                std_errors = np.full_like(coeffs, np.nan)
                t_stats = np.full_like(coeffs, np.nan)
                p_values = np.full_like(coeffs, np.nan)
                r_squared = 0
            
        elif method == 'ridge':
            # Ridge regression
            ridge = Ridge(alpha=0.1, fit_intercept=False)
            ridge.fit(X_with_intercept, y)
            coeffs = ridge.coef_
            
            # Standard errors not easily available for Ridge
            std_errors = np.full_like(coeffs, np.nan)
            t_stats = np.full_like(coeffs, np.nan)
            p_values = np.full_like(coeffs, np.nan)
            r_squared = ridge.score(X_with_intercept, y)
            
        elif method == 'lasso':
            # Lasso regression
            lasso = Lasso(alpha=0.01, fit_intercept=False)
            lasso.fit(X_with_intercept, y)
            coeffs = lasso.coef_
            
            # Standard errors not easily available for Lasso
            std_errors = np.full_like(coeffs, np.nan)
            t_stats = np.full_like(coeffs, np.nan)
            p_values = np.full_like(coeffs, np.nan)
            r_squared = lasso.score(X_with_intercept, y)
            
        else:
            raise ValueError(f"Unknown regression method: {method}")
        
        # Skip intercept (first coefficient)
        factor_coeffs = coeffs[1:]
        factor_std_errors = std_errors[1:]
        factor_t_stats = t_stats[1:]
        factor_p_values = p_values[1:]
        
        # Create exposure objects
        for i, factor_name in enumerate(factor_returns.columns):
            # Confidence intervals (95%)
            if not np.isnan(factor_std_errors[i]):
                ci_lower = factor_coeffs[i] - 1.96 * factor_std_errors[i]
                ci_upper = factor_coeffs[i] + 1.96 * factor_std_errors[i]
                ci = (ci_lower, ci_upper)
            else:
                ci = None
            
            exposures[factor_name] = RiskFactorExposure(
                factor_name=factor_name,
                factor_type=self._infer_factor_type(factor_name),
                exposure=factor_coeffs[i],
                t_statistic=factor_t_stats[i] if not np.isnan(factor_t_stats[i]) else None,
                p_value=factor_p_values[i] if not np.isnan(factor_p_values[i]) else None,
                confidence_interval=ci,
                r_squared=r_squared
            )
        
        return exposures
    
    def _estimate_exposures_rolling(self, portfolio_returns: pd.Series,
                                  factor_returns: pd.DataFrame,
                                  window: int, method: str) -> pd.DataFrame:
        """Estimate rolling factor exposures."""
        
        rolling_exposures = pd.DataFrame(index=portfolio_returns.index,
                                       columns=factor_returns.columns)
        
        for i in range(window, len(portfolio_returns)):
            window_portfolio = portfolio_returns.iloc[i-window:i]
            window_factors = factor_returns.iloc[i-window:i]
            
            try:
                exposures = self._estimate_exposures_single_period(
                    window_portfolio, window_factors, method
                )
                
                for factor_name, exposure_obj in exposures.items():
                    rolling_exposures.iloc[i, rolling_exposures.columns.get_loc(factor_name)] = exposure_obj.exposure
                    
            except (LinAlgError, ValueError):
                # Skip if estimation fails
                continue
        
        return rolling_exposures
    
    def _infer_factor_type(self, factor_name: str) -> RiskFactorType:
        """Infer factor type from factor name."""
        
        name_lower = factor_name.lower()
        
        if any(x in name_lower for x in ['market', 'mkt', 'rm-rf']):
            return RiskFactorType.MARKET
        elif any(x in name_lower for x in ['smb', 'size', 'small']):
            return RiskFactorType.SIZE
        elif any(x in name_lower for x in ['hml', 'value', 'book']):
            return RiskFactorType.VALUE
        elif any(x in name_lower for x in ['rmw', 'profit', 'robust']):
            return RiskFactorType.PROFITABILITY
        elif any(x in name_lower for x in ['cma', 'invest', 'conservative']):
            return RiskFactorType.INVESTMENT
        elif any(x in name_lower for x in ['mom', 'momentum']):
            return RiskFactorType.MOMENTUM
        elif any(x in name_lower for x in ['vol', 'volatility']):
            return RiskFactorType.VOLATILITY
        elif any(x in name_lower for x in ['quality', 'qual']):
            return RiskFactorType.QUALITY
        elif any(x in name_lower for x in ['growth']):
            return RiskFactorType.GROWTH
        elif any(x in name_lower for x in ['carry']):
            return RiskFactorType.CARRY
        elif any(x in name_lower for x in ['credit']):
            return RiskFactorType.CREDIT
        elif any(x in name_lower for x in ['currency', 'fx']):
            return RiskFactorType.CURRENCY
        elif any(x in name_lower for x in ['sector', 'industry']):
            return RiskFactorType.SECTOR
        elif any(x in name_lower for x in ['country', 'region']):
            return RiskFactorType.COUNTRY
        else:
            return RiskFactorType.CUSTOM
    
    def calculate_risk_attribution(self, portfolio_returns: pd.Series,
                                 factor_returns: pd.DataFrame,
                                 benchmark_returns: Optional[pd.Series] = None,
                                 factor_covariance: Optional[pd.DataFrame] = None) -> List[AttributionResult]:
        """Calculate comprehensive risk attribution."""
        
        # Estimate factor exposures
        exposures = self.estimate_factor_exposures(portfolio_returns, factor_returns)
        
        # Align data
        aligned_data = pd.concat([portfolio_returns, factor_returns], axis=1, join='inner')
        portfolio_ret = aligned_data.iloc[:, 0]
        factor_ret = aligned_data.iloc[:, 1:]
        
        # Calculate factor covariance matrix if not provided
        if factor_covariance is None:
            factor_covariance = factor_ret.cov()
        
        # Calculate benchmark exposures if benchmark provided
        benchmark_exposures = {}
        if benchmark_returns is not None:
            try:
                benchmark_exposures = self.estimate_factor_exposures(benchmark_returns, factor_returns)
            except:
                benchmark_exposures = {name: RiskFactorExposure(name, RiskFactorType.CUSTOM, 0.0) 
                                     for name in factor_returns.columns}
        
        attribution_results = []
        
        for factor_name in factor_returns.columns:
            exposure = exposures[factor_name]
            factor_return = factor_ret[factor_name].mean()
            
            # Factor contribution to return
            contribution = exposure.exposure * factor_return
            
            # Risk contribution
            factor_variance = factor_covariance.loc[factor_name, factor_name]
            risk_contribution = (exposure.exposure ** 2) * factor_variance
            
            # Tracking error contribution (if benchmark available)
            if benchmark_returns is not None and factor_name in benchmark_exposures:
                bench_exposure = benchmark_exposures[factor_name].exposure
                active_weight = exposure.exposure - bench_exposure
                te_contribution = (active_weight ** 2) * factor_variance
            else:
                active_weight = None
                te_contribution = risk_contribution
                bench_exposure = None
            
            attribution_results.append(AttributionResult(
                factor_name=factor_name,
                factor_type=exposure.factor_type,
                contribution=contribution,
                exposure=exposure.exposure,
                factor_return=factor_return,
                risk_contribution=risk_contribution,
                tracking_error_contribution=te_contribution,
                active_weight=active_weight,
                benchmark_exposure=bench_exposure
            ))
        
        return attribution_results
    
    def decompose_portfolio_risk(self, portfolio_returns: pd.Series,
                               factor_returns: pd.DataFrame,
                               factor_covariance: Optional[pd.DataFrame] = None) -> RiskDecomposition:
        """Decompose portfolio risk into systematic and idiosyncratic components."""
        
        # Estimate factor exposures
        exposures = self.estimate_factor_exposures(portfolio_returns, factor_returns)
        
        # Align data
        aligned_data = pd.concat([portfolio_returns, factor_returns], axis=1, join='inner')
        portfolio_ret = aligned_data.iloc[:, 0]
        factor_ret = aligned_data.iloc[:, 1:]
        
        # Calculate factor covariance matrix if not provided
        if factor_covariance is None:
            factor_covariance = factor_ret.cov()
        
        # Extract exposures as array
        exposure_vector = np.array([exposures[name].exposure for name in factor_returns.columns])
        
        # Calculate systematic risk
        systematic_variance = exposure_vector.T @ factor_covariance.values @ exposure_vector
        systematic_risk = np.sqrt(systematic_variance)
        
        # Calculate total portfolio risk
        total_variance = portfolio_ret.var()
        total_risk = np.sqrt(total_variance)
        
        # Calculate idiosyncratic risk
        idiosyncratic_variance = max(0, total_variance - systematic_variance)
        idiosyncratic_risk = np.sqrt(idiosyncratic_variance)
        
        # Explained variance ratio
        explained_variance = systematic_variance / total_variance if total_variance > 0 else 0
        residual_variance = 1 - explained_variance
        
        # Calculate individual factor contributions
        attribution_results = self.calculate_risk_attribution(
            portfolio_returns, factor_returns, factor_covariance=factor_covariance
        )
        
        # Diversification ratio
        # Sum of individual factor risks vs portfolio systematic risk
        individual_risks = np.sum([abs(exp.exposure) * np.sqrt(factor_covariance.loc[exp.factor_name, exp.factor_name]) 
                                 for exp in attribution_results])
        diversification_ratio = individual_risks / systematic_risk if systematic_risk > 0 else 1.0
        
        return RiskDecomposition(
            total_risk=total_risk,
            systematic_risk=systematic_risk,
            idiosyncratic_risk=idiosyncratic_risk,
            factor_contributions=attribution_results,
            explained_variance=explained_variance,
            residual_variance=residual_variance,
            diversification_ratio=diversification_ratio
        )
    
    def brinson_attribution(self, portfolio_weights: pd.Series,
                          portfolio_returns: pd.Series,
                          benchmark_weights: pd.Series,
                          benchmark_returns: pd.Series,
                          sector_mapping: Dict[str, str]) -> pd.DataFrame:
        """Perform Brinson attribution analysis."""
        
        # Align all data
        common_assets = portfolio_weights.index.intersection(benchmark_weights.index)
        common_assets = common_assets.intersection(portfolio_returns.index)
        common_assets = common_assets.intersection(benchmark_returns.index)
        
        if len(common_assets) == 0:
            raise ValueError("No common assets found for attribution")
        
        # Filter data to common assets
        pw = portfolio_weights.loc[common_assets]
        pr = portfolio_returns.loc[common_assets]
        bw = benchmark_weights.loc[common_assets]
        br = benchmark_returns.loc[common_assets]
        
        # Create sector mapping for common assets
        asset_sectors = {asset: sector_mapping.get(asset, 'Unknown') for asset in common_assets}
        
        # Calculate sector-level data
        sectors = list(set(asset_sectors.values()))
        attribution_data = []
        
        for sector in sectors:
            sector_assets = [asset for asset in common_assets if asset_sectors[asset] == sector]
            
            if not sector_assets:
                continue
            
            # Sector weights and returns
            sector_pw = pw.loc[sector_assets].sum()
            sector_bw = bw.loc[sector_assets].sum()
            
            # Weighted average returns
            if sector_pw > 0:
                sector_pr = (pw.loc[sector_assets] * pr.loc[sector_assets]).sum() / sector_pw
            else:
                sector_pr = 0
            
            if sector_bw > 0:
                sector_br = (bw.loc[sector_assets] * br.loc[sector_assets]).sum() / sector_bw
            else:
                sector_br = 0
            
            # Brinson attribution components
            allocation_effect = (sector_pw - sector_bw) * sector_br
            selection_effect = sector_bw * (sector_pr - sector_br)
            interaction_effect = (sector_pw - sector_bw) * (sector_pr - sector_br)
            
            attribution_data.append({
                'Sector': sector,
                'Portfolio_Weight': sector_pw,
                'Benchmark_Weight': sector_bw,
                'Portfolio_Return': sector_pr,
                'Benchmark_Return': sector_br,
                'Allocation_Effect': allocation_effect,
                'Selection_Effect': selection_effect,
                'Interaction_Effect': interaction_effect,
                'Total_Effect': allocation_effect + selection_effect + interaction_effect
            })
        
        return pd.DataFrame(attribution_data)
    
    def style_attribution(self, portfolio_returns: pd.Series,
                         style_factors: pd.DataFrame,
                         benchmark_returns: Optional[pd.Series] = None) -> Dict[str, float]:
        """Perform style-based attribution analysis."""
        
        # Estimate style exposures
        exposures = self.estimate_factor_exposures(portfolio_returns, style_factors)
        
        # Calculate style contributions
        aligned_data = pd.concat([portfolio_returns, style_factors], axis=1, join='inner')
        factor_ret = aligned_data.iloc[:, 1:]
        
        style_attribution = {}
        
        for factor_name in style_factors.columns:
            exposure = exposures[factor_name].exposure
            factor_return = factor_ret[factor_name].mean()
            
            contribution = exposure * factor_return
            style_attribution[factor_name] = contribution
        
        # Calculate benchmark style attribution if provided
        if benchmark_returns is not None:
            try:
                benchmark_exposures = self.estimate_factor_exposures(benchmark_returns, style_factors)
                benchmark_attribution = {}
                
                for factor_name in style_factors.columns:
                    bench_exposure = benchmark_exposures[factor_name].exposure
                    factor_return = factor_ret[factor_name].mean()
                    
                    bench_contribution = bench_exposure * factor_return
                    benchmark_attribution[factor_name] = bench_contribution
                
                # Calculate active style attribution
                active_attribution = {}
                for factor_name in style_factors.columns:
                    active_attribution[f'Active_{factor_name}'] = (
                        style_attribution[factor_name] - benchmark_attribution[factor_name]
                    )
                
                style_attribution.update(active_attribution)
                
            except:
                pass  # Skip if benchmark attribution fails
        
        return style_attribution
    
    def factor_timing_analysis(self, portfolio_returns: pd.Series,
                             factors: pd.DataFrame,
                             rolling_window: int = 252) -> pd.DataFrame:
        """Analyze factor timing ability."""
        
        # Calculate rolling factor exposures
        rolling_exposures = self._estimate_exposures_rolling(
            portfolio_returns, factors, rolling_window, 'ols'
        )
        
        # Calculate factor returns going forward
        factor_forward_returns = factors.shift(-1)  # Next period returns
        
        # Analyze correlation between exposures and future factor returns
        timing_results = []
        
        for factor_name in factors.columns:
            exposures_series = rolling_exposures[factor_name].dropna()
            forward_returns = factor_forward_returns[factor_name]
            
            # Align data
            common_index = exposures_series.index.intersection(forward_returns.index)
            
            if len(common_index) > 20:  # Minimum observations
                aligned_exposures = exposures_series.loc[common_index]
                aligned_forward = forward_returns.loc[common_index]
                
                # Calculate correlation
                timing_corr = aligned_exposures.corr(aligned_forward)
                
                # Calculate timing information coefficient
                # Rank correlation (Spearman)
                timing_ic = stats.spearmanr(aligned_exposures, aligned_forward)[0]
                
                # Hit rate (proportion of correct directional calls)
                exposure_direction = np.sign(aligned_exposures)
                return_direction = np.sign(aligned_forward)
                hit_rate = (exposure_direction == return_direction).mean()
                
                timing_results.append({
                    'Factor': factor_name,
                    'Timing_Correlation': timing_corr,
                    'Information_Coefficient': timing_ic,
                    'Hit_Rate': hit_rate,
                    'Observations': len(common_index)
                })
        
        return pd.DataFrame(timing_results)
    
    def generate_attribution_report(self, portfolio_returns: pd.Series,
                                  factors: pd.DataFrame,
                                  benchmark_returns: Optional[pd.Series] = None,
                                  factor_names: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Generate comprehensive attribution report."""
        
        report = {
            'generation_time': datetime.now().isoformat(),
            'analysis_period': {
                'start_date': portfolio_returns.index[0].isoformat(),
                'end_date': portfolio_returns.index[-1].isoformat(),
                'total_periods': len(portfolio_returns)
            }
        }
        
        # Risk decomposition
        try:
            risk_decomp = self.decompose_portfolio_risk(portfolio_returns, factors)
            
            report['risk_decomposition'] = {
                'total_risk': risk_decomp.total_risk,
                'systematic_risk': risk_decomp.systematic_risk,
                'idiosyncratic_risk': risk_decomp.idiosyncratic_risk,
                'explained_variance': risk_decomp.explained_variance,
                'diversification_ratio': risk_decomp.diversification_ratio
            }
            
            # Factor contributions
            factor_contributions = {}
            for contrib in risk_decomp.factor_contributions:
                factor_contributions[contrib.factor_name] = {
                    'exposure': contrib.exposure,
                    'return_contribution': contrib.contribution,
                    'risk_contribution': contrib.risk_contribution,
                    'factor_type': contrib.factor_type.value
                }
            
            report['factor_contributions'] = factor_contributions
            
        except Exception as e:
            report['risk_decomposition_error'] = str(e)
        
        # Factor exposures
        try:
            exposures = self.estimate_factor_exposures(portfolio_returns, factors)
            
            exposure_summary = {}
            for name, exposure in exposures.items():
                exposure_summary[name] = {
                    'exposure': exposure.exposure,
                    'factor_type': exposure.factor_type.value,
                    't_statistic': exposure.t_statistic,
                    'p_value': exposure.p_value,
                    'significant': exposure.p_value < 0.05 if exposure.p_value is not None else None
                }
            
            report['factor_exposures'] = exposure_summary
            
        except Exception as e:
            report['factor_exposures_error'] = str(e)
        
        # Performance attribution
        try:
            attribution = self.calculate_risk_attribution(portfolio_returns, factors, benchmark_returns)
            
            performance_attribution = {}
            for attr in attribution:
                performance_attribution[attr.factor_name] = {
                    'return_contribution': attr.contribution,
                    'exposure': attr.exposure,
                    'factor_return': attr.factor_return,
                    'active_weight': attr.active_weight
                }
            
            report['performance_attribution'] = performance_attribution
            
        except Exception as e:
            report['performance_attribution_error'] = str(e)
        
        # Style attribution
        try:
            style_attr = self.style_attribution(portfolio_returns, factors, benchmark_returns)
            report['style_attribution'] = style_attr
            
        except Exception as e:
            report['style_attribution_error'] = str(e)
        
        # Factor timing analysis
        try:
            timing_analysis = self.factor_timing_analysis(portfolio_returns, factors)
            
            timing_summary = {}
            for _, row in timing_analysis.iterrows():
                timing_summary[row['Factor']] = {
                    'timing_correlation': row['Timing_Correlation'],
                    'information_coefficient': row['Information_Coefficient'],
                    'hit_rate': row['Hit_Rate']
                }
            
            report['factor_timing'] = timing_summary
            
        except Exception as e:
            report['factor_timing_error'] = str(e)
        
        return report
