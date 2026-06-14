"""
Advanced Statistical Tests Engine for QUANTUM-FORGE
Implements sophisticated statistical testing for financial time series.
"""

import numpy as np
import scipy.stats as stats
from scipy.stats import jarque_bera, shapiro, anderson, kstest
from scipy.optimize import minimize
from statsmodels.tsa.stattools import adfuller, kpss, coint
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
import arch
from typing import Tuple, Dict, List, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class UnitRootTests:
    """Comprehensive unit root and stationarity testing."""
    
    @staticmethod
    def augmented_dickey_fuller(series: np.ndarray, maxlag: Optional[int] = None, 
                               regression: str = 'c') -> Dict[str, float]:
        """
        Augmented Dickey-Fuller test for unit roots.
        
        Args:
            series: Time series data
            maxlag: Maximum number of lags
            regression: 'c' (constant), 'ct' (constant+trend), 'ctt' (constant+linear+quadratic trend), 'nc' (no constant)
        
        Returns:
            Dictionary with test results
        """
        result = adfuller(series, maxlag=maxlag, regression=regression)
        
        return {
            'adf_statistic': result[0],
            'p_value': result[1],
            'n_lags': result[2],
            'n_obs': result[3],
            'critical_values': result[4],
            'ic_best': result[5],
            'is_stationary': result[1] < 0.05
        }
    
    @staticmethod
    def kpss_test(series: np.ndarray, regression: str = 'c', nlags: str = 'auto') -> Dict[str, float]:
        """
        KPSS test for stationarity (null hypothesis: stationary).
        
        Args:
            series: Time series data
            regression: 'c' (level stationary) or 'ct' (trend stationary)
            nlags: Number of lags or 'auto'
        
        Returns:
            Dictionary with test results
        """
        result = kpss(series, regression=regression, nlags=nlags)
        
        return {
            'kpss_statistic': result[0],
            'p_value': result[1],
            'n_lags': result[2],
            'critical_values': result[3],
            'is_stationary': result[1] > 0.05
        }
    
    @staticmethod
    def phillips_perron(series: np.ndarray, lags: Optional[int] = None) -> Dict[str, float]:
        """
        Phillips-Perron test for unit roots.
        
        Args:
            series: Time series data
            lags: Number of lags for Newey-West correction
        
        Returns:
            Dictionary with test results
        """
        n = len(series)
        if lags is None:
            lags = int(4 * (n/100)**(2/9))
        
        # First difference
        dy = np.diff(series)
        y_lag = series[:-1]
        
        # OLS regression
        X = np.column_stack([np.ones(len(y_lag)), y_lag])
        beta = np.linalg.lstsq(X, dy, rcond=None)[0]
        residuals = dy - X @ beta
        
        # Standard error with Newey-West correction
        gamma_0 = np.var(residuals, ddof=1)
        gamma_sum = 0
        
        for j in range(1, lags + 1):
            gamma_j = np.cov(residuals[j:], residuals[:-j], ddof=1)[0, 1]
            gamma_sum += 2 * (1 - j/(lags + 1)) * gamma_j
        
        sigma_sq = gamma_0 + gamma_sum
        
        # Test statistic
        pp_stat = (n * beta[1] - 1) / np.sqrt(sigma_sq / np.var(y_lag, ddof=1))
        
        # Critical values (approximate)
        critical_values = {'1%': -3.43, '5%': -2.86, '10%': -2.57}
        p_value = 0.05 if pp_stat > critical_values['5%'] else 0.01
        
        return {
            'pp_statistic': pp_stat,
            'p_value': p_value,
            'critical_values': critical_values,
            'is_stationary': pp_stat < critical_values['5%']
        }

class CointegrationTests:
    """Cointegration testing for multiple time series."""
    
    @staticmethod
    def engle_granger(y: np.ndarray, x: np.ndarray, trend: str = 'c') -> Dict[str, float]:
        """
        Engle-Granger two-step cointegration test.
        
        Args:
            y: Dependent variable
            x: Independent variable(s)
            trend: 'c' (constant), 'ct' (constant+trend), 'ctt' (constant+linear+quadratic trend), 'nc' (no constant)
        
        Returns:
            Dictionary with test results
        """
        result = coint(y, x, trend=trend)
        
        return {
            'eg_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[2],
            'is_cointegrated': result[1] < 0.05
        }
    
    @staticmethod
    def johansen_test(data: np.ndarray, det_order: int = 0, k_ar_diff: int = 1) -> Dict[str, any]:
        """
        Johansen cointegration test for multiple time series.
        
        Args:
            data: Matrix of time series (T x n)
            det_order: Deterministic components (0: no deterministic, 1: constant, 2: linear trend)
            k_ar_diff: Number of lagged differences
        
        Returns:
            Dictionary with test results
        """
        try:
            from statsmodels.tsa.vector_ar.vecm import coint_johansen
            result = coint_johansen(data, det_order=det_order, k_ar_diff=k_ar_diff)
            
            return {
                'trace_statistic': result.lr1,
                'max_eigenvalue_statistic': result.lr2,
                'critical_values_trace': result.cvt,
                'critical_values_max_eigenvalue': result.cvm,
                'eigenvalues': result.eig,
                'eigenvectors': result.evec,
                'n_cointegrating_relationships': np.sum(result.lr1 > result.cvt[:, 1])  # 5% level
            }
        except ImportError:
            return {'error': 'statsmodels version does not support Johansen test'}

class NormalityTests:
    """Comprehensive normality testing suite."""
    
    @staticmethod
    def jarque_bera_test(data: np.ndarray) -> Dict[str, float]:
        """Jarque-Bera test for normality."""
        statistic, p_value = jarque_bera(data)
        
        return {
            'jb_statistic': statistic,
            'p_value': p_value,
            'is_normal': p_value > 0.05,
            'skewness': stats.skew(data),
            'kurtosis': stats.kurtosis(data, fisher=False)
        }
    
    @staticmethod
    def shapiro_wilk_test(data: np.ndarray) -> Dict[str, float]:
        """Shapiro-Wilk test for normality."""
        if len(data) > 5000:
            # Deterministic subsample for large datasets: evenly-spaced indices
            n = len(data)
            indices = np.linspace(0, n - 1, 5000, dtype=int)
            data = np.asarray(data)[indices]
        
        statistic, p_value = shapiro(data)
        
        return {
            'sw_statistic': statistic,
            'p_value': p_value,
            'is_normal': p_value > 0.05
        }
    
    @staticmethod
    def anderson_darling_test(data: np.ndarray, dist: str = 'norm') -> Dict[str, float]:
        """Anderson-Darling test for normality."""
        result = anderson(data, dist=dist)
        
        # Check at 5% significance level
        critical_value_5pct = result.critical_values[2]  # Usually 5% level is at index 2
        is_normal = result.statistic < critical_value_5pct
        
        return {
            'ad_statistic': result.statistic,
            'critical_values': result.critical_values,
            'significance_levels': result.significance_level,
            'is_normal': is_normal
        }
    
    @staticmethod
    def kolmogorov_smirnov_test(data: np.ndarray) -> Dict[str, float]:
        """Kolmogorov-Smirnov test against normal distribution."""
        # Standardize data
        standardized_data = (data - np.mean(data)) / np.std(data, ddof=1)
        
        statistic, p_value = kstest(standardized_data, 'norm')
        
        return {
            'ks_statistic': statistic,
            'p_value': p_value,
            'is_normal': p_value > 0.05
        }

class SerialCorrelationTests:
    """Tests for serial correlation in time series."""
    
    @staticmethod
    def ljung_box_test(residuals: np.ndarray, lags: Optional[int] = None) -> Dict[str, float]:
        """Ljung-Box test for serial correlation."""
        if lags is None:
            lags = min(10, len(residuals) // 5)
        
        result = acorr_ljungbox(residuals, lags=lags, return_df=False)
        
        return {
            'lb_statistic': result['lb_stat'].iloc[-1],
            'p_value': result['lb_pvalue'].iloc[-1],
            'has_serial_correlation': result['lb_pvalue'].iloc[-1] < 0.05
        }
    
    @staticmethod
    def durbin_watson_test(residuals: np.ndarray) -> Dict[str, float]:
        """Durbin-Watson test for first-order serial correlation."""
        dw_statistic = durbin_watson(residuals)
        
        # Rough interpretation (exact critical values depend on sample size and regressors)
        interpretation = 'inconclusive'
        if dw_statistic < 1.5:
            interpretation = 'positive_correlation'
        elif dw_statistic > 2.5:
            interpretation = 'negative_correlation'
        elif 1.5 <= dw_statistic <= 2.5:
            interpretation = 'no_correlation'
        
        return {
            'dw_statistic': dw_statistic,
            'interpretation': interpretation
        }
    
    @staticmethod
    def breusch_godfrey_test(residuals: np.ndarray, lags: int = 1) -> Dict[str, float]:
        """Breusch-Godfrey test for higher-order serial correlation."""
        n = len(residuals)
        
        # Create lagged residuals matrix
        X = np.ones((n - lags, 1))  # Constant term
        for i in range(1, lags + 1):
            X = np.column_stack([X, residuals[lags-i:-i]])
        
        y = residuals[lags:]
        
        # OLS regression
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        fitted = X @ beta
        ssr_restricted = np.sum((y - np.mean(y))**2)
        ssr_unrestricted = np.sum((y - fitted)**2)
        
        # LM statistic
        n_effective = len(y)
        lm_statistic = n_effective * (1 - ssr_unrestricted / ssr_restricted)
        p_value = 1 - stats.chi2.cdf(lm_statistic, lags)
        
        return {
            'lm_statistic': lm_statistic,
            'p_value': p_value,
            'has_serial_correlation': p_value < 0.05
        }

class HeteroskedasticityTests:
    """Tests for heteroskedasticity in time series."""
    
    @staticmethod
    def arch_test(residuals: np.ndarray, lags: int = 5) -> Dict[str, float]:
        """ARCH test for conditional heteroskedasticity."""
        try:
            from arch.diagnostic import het_arch
            result = het_arch(residuals, lags=lags)
            
            return {
                'arch_statistic': result['statistic'],
                'p_value': result['pvalue'],
                'has_arch_effects': result['pvalue'] < 0.05
            }
        except ImportError:
            # Manual implementation
            n = len(residuals)
            squared_residuals = residuals**2
            
            # Create lagged squared residuals
            X = np.ones((n - lags, 1))
            for i in range(1, lags + 1):
                X = np.column_stack([X, squared_residuals[lags-i:-i]])
            
            y = squared_residuals[lags:]
            
            # OLS regression
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            fitted = X @ beta
            
            # R-squared
            ss_res = np.sum((y - fitted)**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r_squared = 1 - ss_res / ss_tot
            
            # LM statistic
            lm_statistic = (n - lags) * r_squared
            p_value = 1 - stats.chi2.cdf(lm_statistic, lags)
            
            return {
                'arch_statistic': lm_statistic,
                'p_value': p_value,
                'has_arch_effects': p_value < 0.05
            }
    
    @staticmethod
    def white_test(residuals: np.ndarray, fitted_values: np.ndarray) -> Dict[str, float]:
        """White test for heteroskedasticity."""
        n = len(residuals)
        squared_residuals = residuals**2
        
        # Auxiliary regression: squared residuals on fitted values and their squares
        X = np.column_stack([
            np.ones(n),
            fitted_values,
            fitted_values**2
        ])
        
        # OLS regression
        beta = np.linalg.lstsq(X, squared_residuals, rcond=None)[0]
        fitted_aux = X @ beta
        
        # R-squared
        ss_res = np.sum((squared_residuals - fitted_aux)**2)
        ss_tot = np.sum((squared_residuals - np.mean(squared_residuals))**2)
        r_squared = 1 - ss_res / ss_tot
        
        # Test statistic
        white_statistic = n * r_squared
        p_value = 1 - stats.chi2.cdf(white_statistic, 2)  # 2 restrictions
        
        return {
            'white_statistic': white_statistic,
            'p_value': p_value,
            'has_heteroskedasticity': p_value < 0.05
        }
    
    @staticmethod
    def breusch_pagan_test(residuals: np.ndarray, regressors: np.ndarray) -> Dict[str, float]:
        """Breusch-Pagan test for heteroskedasticity."""
        n = len(residuals)
        squared_residuals = residuals**2
        
        # Add constant to regressors if not present
        if regressors.ndim == 1:
            regressors = regressors.reshape(-1, 1)
        
        X = np.column_stack([np.ones(n), regressors])
        
        # Auxiliary regression
        beta = np.linalg.lstsq(X, squared_residuals, rcond=None)[0]
        fitted_aux = X @ beta
        
        # Sum of squared residuals from auxiliary regression
        ssr_aux = np.sum((squared_residuals - fitted_aux)**2)
        
        # Test statistic
        sigma_sq = np.mean(squared_residuals)
        bp_statistic = 0.5 * np.sum((squared_residuals - sigma_sq)**2) / sigma_sq**2
        
        df = X.shape[1] - 1  # Degrees of freedom
        p_value = 1 - stats.chi2.cdf(bp_statistic, df)
        
        return {
            'bp_statistic': bp_statistic,
            'p_value': p_value,
            'has_heteroskedasticity': p_value < 0.05
        }

class StructuralBreakTests:
    """Tests for structural breaks in time series."""
    
    @staticmethod
    def chow_test(y: np.ndarray, X: np.ndarray, break_point: int) -> Dict[str, float]:
        """Chow test for structural break at known break point."""
        n, k = X.shape
        
        # Split data
        y1, X1 = y[:break_point], X[:break_point]
        y2, X2 = y[break_point:], X[break_point:]
        
        # Full sample regression
        beta_full = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals_full = y - X @ beta_full
        ssr_full = np.sum(residuals_full**2)
        
        # Sub-sample regressions
        beta1 = np.linalg.lstsq(X1, y1, rcond=None)[0]
        beta2 = np.linalg.lstsq(X2, y2, rcond=None)[0]
        
        residuals1 = y1 - X1 @ beta1
        residuals2 = y2 - X2 @ beta2
        
        ssr1 = np.sum(residuals1**2)
        ssr2 = np.sum(residuals2**2)
        ssr_split = ssr1 + ssr2
        
        # Chow statistic
        chow_statistic = ((ssr_full - ssr_split) / k) / (ssr_split / (n - 2*k))
        p_value = 1 - stats.f.cdf(chow_statistic, k, n - 2*k)
        
        return {
            'chow_statistic': chow_statistic,
            'p_value': p_value,
            'has_structural_break': p_value < 0.05
        }
    
    @staticmethod
    def cusum_test(residuals: np.ndarray) -> Dict[str, float]:
        """CUSUM test for parameter stability."""
        n = len(residuals)
        sigma = np.std(residuals, ddof=1)
        
        # Standardized residuals
        standardized_residuals = residuals / sigma
        
        # CUSUM statistic
        cusum = np.cumsum(standardized_residuals) / np.sqrt(n)
        
        # Critical value at 5% level (approximate)
        critical_value = 0.948  # For 5% significance level
        
        # Test statistic (maximum absolute CUSUM)
        test_statistic = np.max(np.abs(cusum))
        
        return {
            'cusum_statistic': test_statistic,
            'critical_value': critical_value,
            'cusum_path': cusum,
            'has_structural_break': test_statistic > critical_value
        }

class NonlinearityTests:
    """Tests for nonlinearity in time series."""
    
    @staticmethod
    def bds_test(series: np.ndarray, m: int = 3, epsilon: Optional[float] = None) -> Dict[str, float]:
        """BDS test for nonlinear dependence."""
        n = len(series)
        
        if epsilon is None:
            epsilon = np.std(series) / 2
        
        # Compute correlation integrals
        def correlation_integral(data, m, eps):
            n = len(data)
            count = 0
            total_pairs = 0
            
            for i in range(n - m + 1):
                for j in range(i + 1, n - m + 1):
                    # Check if all m-dimensional vectors are within epsilon
                    max_diff = 0
                    for k in range(m):
                        diff = abs(data[i + k] - data[j + k])
                        max_diff = max(max_diff, diff)
                    
                    if max_diff < eps:
                        count += 1
                    total_pairs += 1
            
            return count / total_pairs if total_pairs > 0 else 0
        
        # Calculate correlation integrals for dimensions 1 and m
        c1 = correlation_integral(series, 1, epsilon)
        cm = correlation_integral(series, m, epsilon)
        
        # BDS statistic (simplified)
        if c1 > 0:
            bds_statistic = np.sqrt(n) * (cm - c1**m) / np.sqrt(c1**(2*m))
            p_value = 2 * (1 - stats.norm.cdf(abs(bds_statistic)))
        else:
            bds_statistic = np.nan
            p_value = np.nan
        
        return {
            'bds_statistic': bds_statistic,
            'p_value': p_value,
            'has_nonlinear_dependence': p_value < 0.05 if not np.isnan(p_value) else False
        }

# Example usage and testing
if __name__ == "__main__":
    # Generate test data
    np.random.seed(42)
    
    # Stationary series
    stationary_series = np.random.randn(1000)
    
    # Non-stationary series (random walk)
    non_stationary_series = np.cumsum(np.random.randn(1000) * 0.1)
    
    # Test unit root tests
    print("Testing Unit Root Tests...")
    adf_result = UnitRootTests.augmented_dickey_fuller(non_stationary_series)
    print(f"ADF test p-value: {adf_result['p_value']:.4f}, Stationary: {adf_result['is_stationary']}")
    
    kpss_result = UnitRootTests.kpss_test(stationary_series)
    print(f"KPSS test p-value: {kpss_result['p_value']:.4f}, Stationary: {kpss_result['is_stationary']}")
    
    # Test normality
    print("\nTesting Normality...")
    normal_data = np.random.randn(1000)
    non_normal_data = np.random.exponential(2, 1000)
    
    jb_normal = NormalityTests.jarque_bera_test(normal_data)
    jb_non_normal = NormalityTests.jarque_bera_test(non_normal_data)
    
    print(f"JB test (normal data): p-value = {jb_normal['p_value']:.4f}, Normal: {jb_normal['is_normal']}")
    print(f"JB test (non-normal data): p-value = {jb_non_normal['p_value']:.4f}, Normal: {jb_non_normal['is_normal']}")
    
    # Test ARCH effects
    print("\nTesting ARCH Effects...")
    # Generate series with ARCH effects
    n = 1000
    e = np.random.randn(n)
    h = np.zeros(n)
    h[0] = 1
    
    for t in range(1, n):
        h[t] = 0.1 + 0.8 * h[t-1] + 0.1 * e[t-1]**2
        e[t] = np.sqrt(h[t]) * np.random.randn()
    
    arch_result = HeteroskedasticityTests.arch_test(e)
    print(f"ARCH test p-value: {arch_result['p_value']:.4f}, Has ARCH: {arch_result['has_arch_effects']}")
    
    print("\nStatistical tests engine completed successfully!")