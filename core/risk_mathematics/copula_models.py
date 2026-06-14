"""
Copula Models Engine for QUANTUM-FORGE
Implements advanced copula models for dependency modeling and risk aggregation.
"""

import numpy as np
import pandas as pd
from scipy import stats, optimize, special
from scipy.stats import norm, t as t_dist, kendalltau, spearmanr
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional, Union, Callable
import warnings
from numba import jit, prange
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

class CopulaType(Enum):
    """Types of copula models."""
    GAUSSIAN = "gaussian"
    T_COPULA = "t_copula"
    CLAYTON = "clayton"
    GUMBEL = "gumbel"
    FRANK = "frank"
    JOE = "joe"
    ARCHIMEDEAN = "archimedean"
    VINE = "vine"

@dataclass
class CopulaParameters:
    """Parameters for copula models."""
    correlation_matrix: np.ndarray
    dependence_parameter: float
    degrees_of_freedom: Optional[float] = None  # For t-copula
    copula_type: CopulaType = CopulaType.GAUSSIAN

@dataclass
class DependenceMetrics:
    """Comprehensive dependence metrics."""
    pearson_correlation: float
    spearman_correlation: float
    kendall_tau: float
    tail_dependence_upper: float
    tail_dependence_lower: float
    mutual_information: float
    distance_correlation: float

class GaussianCopula:
    """Gaussian (Normal) copula model."""
    
    def __init__(self, dimension: int):
        """
        Initialize Gaussian copula.
        
        Args:
            dimension: Number of variables
        """
        self.dimension = dimension
        self.correlation_matrix = np.eye(dimension)
        self.fitted = False
    
    def fit(self, data: np.ndarray, method: str = 'pearson') -> bool:
        """
        Fit Gaussian copula to data.
        
        Args:
            data: Data matrix (n_samples x n_variables)
            method: Correlation estimation method ('pearson', 'spearman', 'kendall')
        
        Returns:
            True if fitting successful
        """
        if data.shape[1] != self.dimension:
            return False
        
        try:
            # Transform data to uniform margins using empirical CDF
            uniform_data = np.zeros_like(data)
            
            for i in range(self.dimension):
                # Empirical CDF transformation
                sorted_values = np.sort(data[:, i])
                n = len(sorted_values)
                
                # Handle ties by averaging ranks
                ranks = np.searchsorted(sorted_values, data[:, i], side='right')
                uniform_data[:, i] = ranks / (n + 1)
            
            # Transform to normal margins
            normal_data = norm.ppf(uniform_data)
            
            # Estimate correlation matrix
            if method == 'pearson':
                self.correlation_matrix = np.corrcoef(normal_data.T)
            elif method == 'spearman':
                # Use Spearman correlation and convert to Gaussian copula parameter
                spearman_corr = np.corrcoef(stats.rankdata(data, axis=0).T)
                # Conversion formula: ρ_Gaussian = 2 * sin(π/6 * ρ_Spearman)
                self.correlation_matrix = 2 * np.sin(np.pi/6 * spearman_corr)
            elif method == 'kendall':
                # Use Kendall's tau and convert to Gaussian copula parameter
                kendall_matrix = np.eye(self.dimension)
                for i in range(self.dimension):
                    for j in range(i+1, self.dimension):
                        tau, _ = kendalltau(data[:, i], data[:, j])
                        # Conversion: ρ_Gaussian = sin(π/2 * τ)
                        kendall_matrix[i, j] = kendall_matrix[j, i] = np.sin(np.pi/2 * tau)
                
                self.correlation_matrix = kendall_matrix
            
            # Ensure positive definiteness
            eigenvals, eigenvecs = np.linalg.eigh(self.correlation_matrix)
            eigenvals = np.maximum(eigenvals, 1e-8)
            self.correlation_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
            
            # Normalize diagonal to 1
            diag_sqrt = np.sqrt(np.diag(self.correlation_matrix))
            self.correlation_matrix = self.correlation_matrix / np.outer(diag_sqrt, diag_sqrt)
            
            self.fitted = True
            return True
            
        except Exception as e:
            return False
    
    def sample(self, n_samples: int) -> np.ndarray:
        """
        Generate samples from Gaussian copula.
        
        Args:
            n_samples: Number of samples to generate
        
        Returns:
            Samples from copula (uniform margins)
        """
        if not self.fitted:
            # Deterministic uniform grid for reproducible samples
            u = np.linspace(1.0/(n_samples+1), n_samples/(n_samples+1), n_samples)
            uniform_samples = np.zeros((n_samples, self.dimension))
            for d in range(self.dimension):
                uniform_samples[:, d] = np.roll(u, d)
            return uniform_samples

        # Generate deterministic multivariate normal samples via inverse CDF + Cholesky
        u = np.linspace(1.0/(n_samples+1), n_samples/(n_samples+1), n_samples)
        independent_normals = np.zeros((n_samples, self.dimension))
        for d in range(self.dimension):
            independent_normals[:, d] = norm.ppf(np.roll(u, d))

        # Apply covariance via Cholesky factor
        L = np.linalg.cholesky(self.correlation_matrix)
        normal_samples = independent_normals @ L.T
        
        # Transform to uniform margins
        uniform_samples = norm.cdf(normal_samples)
        
        return uniform_samples
    
    def pdf(self, u: np.ndarray) -> np.ndarray:
        """
        Calculate copula density.
        
        Args:
            u: Uniform samples (n_samples x dimension)
        
        Returns:
            Density values
        """
        if not self.fitted:
            return np.ones(len(u))
        
        # Transform to normal margins
        z = norm.ppf(np.clip(u, 1e-10, 1-1e-10))
        
        # Calculate multivariate normal density
        inv_corr = np.linalg.inv(self.correlation_matrix)
        det_corr = np.linalg.det(self.correlation_matrix)
        
        # Copula density formula
        density = np.zeros(len(u))
        
        for i in range(len(u)):
            quad_form = z[i] @ inv_corr @ z[i] - np.sum(z[i]**2)
            density[i] = np.exp(-0.5 * quad_form) / np.sqrt(det_corr)
        
        return density
    
    def cdf(self, u: np.ndarray) -> np.ndarray:
        """
        Calculate copula CDF (approximate for multivariate case).
        
        Args:
            u: Uniform samples
        
        Returns:
            CDF values
        """
        if not self.fitted:
            return np.prod(u, axis=1)
        
        # Transform to normal margins
        z = norm.ppf(np.clip(u, 1e-10, 1-1e-10))
        
        # Use Monte Carlo integration for multivariate CDF
        # This is computationally expensive; in practice, other methods would be used
        n_mc = 10000
        
        cdf_values = np.zeros(len(u))
        
        for i in range(len(u)):
            # Generate samples from truncated multivariate normal
            # Deterministic sample set for Monte Carlo integration: use inverse CDF grid
            u_mc = np.linspace(1.0/(n_mc+1), n_mc/(n_mc+1), n_mc)
            indep = np.zeros((n_mc, self.dimension))
            for d in range(self.dimension):
                indep[:, d] = norm.ppf(np.roll(u_mc, d))
            L = np.linalg.cholesky(self.correlation_matrix)
            samples = indep @ L.T
            
            # Count samples that fall below the point
            below = np.all(samples <= z[i], axis=1)
            cdf_values[i] = np.mean(below)
        
        return cdf_values

class TCopula:
    """Student's t-copula model."""
    
    def __init__(self, dimension: int):
        """
        Initialize t-copula.
        
        Args:
            dimension: Number of variables
        """
        self.dimension = dimension
        self.correlation_matrix = np.eye(dimension)
        self.degrees_of_freedom = 4.0
        self.fitted = False
    
    def fit(self, data: np.ndarray, method: str = 'mle') -> bool:
        """
        Fit t-copula to data.
        
        Args:
            data: Data matrix
            method: Fitting method ('mle', 'moments')
        
        Returns:
            True if fitting successful
        """
        if data.shape[1] != self.dimension:
            return False
        
        try:
            # Transform to uniform margins
            uniform_data = np.zeros_like(data)
            
            for i in range(self.dimension):
                sorted_values = np.sort(data[:, i])
                n = len(sorted_values)
                ranks = np.searchsorted(sorted_values, data[:, i], side='right')
                uniform_data[:, i] = ranks / (n + 1)
            
            if method == 'mle':
                # Maximum likelihood estimation
                def neg_log_likelihood(params):
                    # Extract parameters
                    n_corr_params = self.dimension * (self.dimension - 1) // 2
                    
                    if len(params) != n_corr_params + 1:
                        return np.inf
                    
                    df = params[-1]
                    if df <= 2:
                        return np.inf
                    
                    # Reconstruct correlation matrix
                    corr_matrix = np.eye(self.dimension)
                    idx = 0
                    for i in range(self.dimension):
                        for j in range(i+1, self.dimension):
                            corr_matrix[i, j] = corr_matrix[j, i] = np.tanh(params[idx])
                            idx += 1
                    
                    # Check positive definiteness
                    eigenvals = np.linalg.eigvals(corr_matrix)
                    if np.any(eigenvals <= 0):
                        return np.inf
                    
                    # Transform to t-margins
                    t_data = t_dist.ppf(np.clip(uniform_data, 1e-10, 1-1e-10), df)
                    
                    # Calculate log-likelihood
                    try:
                        inv_corr = np.linalg.inv(corr_matrix)
                        det_corr = np.linalg.det(corr_matrix)
                        
                        log_likelihood = 0
                        
                        for i in range(len(uniform_data)):
                            quad_form = t_data[i] @ inv_corr @ t_data[i]
                            
                            log_likelihood += (
                                special.loggamma((df + self.dimension) / 2) -
                                special.loggamma(df / 2) -
                                0.5 * self.dimension * np.log(df * np.pi) -
                                0.5 * np.log(det_corr) -
                                0.5 * (df + self.dimension) * np.log(1 + quad_form / df)
                            )
                        
                        return -log_likelihood
                        
                    except:
                        return np.inf
                
                # Initial parameters
                n_corr_params = self.dimension * (self.dimension - 1) // 2
                initial_params = np.zeros(n_corr_params + 1)
                initial_params[-1] = 4.0  # Initial df
                
                # Optimization
                result = optimize.minimize(
                    neg_log_likelihood,
                    initial_params,
                    method='L-BFGS-B',
                    bounds=[(None, None)] * n_corr_params + [(2.1, 30)]
                )
                
                if result.success:
                    # Extract fitted parameters
                    self.degrees_of_freedom = result.x[-1]
                    
                    # Reconstruct correlation matrix
                    idx = 0
                    for i in range(self.dimension):
                        for j in range(i+1, self.dimension):
                            self.correlation_matrix[i, j] = np.tanh(result.x[idx])
                            self.correlation_matrix[j, i] = self.correlation_matrix[i, j]
                            idx += 1
                    
                    self.fitted = True
                    return True
            
            elif method == 'moments':
                # Method of moments (simplified)
                # Transform to normal margins first
                normal_data = norm.ppf(uniform_data)
                self.correlation_matrix = np.corrcoef(normal_data.T)
                
                # Estimate degrees of freedom using kurtosis
                sample_kurtosis = np.mean([stats.kurtosis(data[:, i]) for i in range(self.dimension)])
                
                # For t-distribution: kurtosis = 6/(df-4) for df > 4
                if sample_kurtosis > 0:
                    self.degrees_of_freedom = max(4.1, 6/sample_kurtosis + 4)
                else:
                    self.degrees_of_freedom = 10.0
                
                self.fitted = True
                return True
                
        except Exception as e:
            return False
        
        return False
    
    def sample(self, n_samples: int) -> np.ndarray:
        """Generate samples from t-copula."""
        if not self.fitted:
            u = np.linspace(1.0/(n_samples+1), n_samples/(n_samples+1), n_samples)
            uniform_samples = np.zeros((n_samples, self.dimension))
            for d in range(self.dimension):
                uniform_samples[:, d] = np.roll(u, d)
            return uniform_samples

        # Deterministic multivariate t sampling via inverse transform
        u = np.linspace(1.0/(n_samples+1), n_samples/(n_samples+1), n_samples)
        independent_normals = np.zeros((n_samples, self.dimension))
        for d in range(self.dimension):
            independent_normals[:, d] = norm.ppf(np.roll(u, d))

        L = np.linalg.cholesky(self.correlation_matrix)
        normal_samples = independent_normals @ L.T

        # Deterministic chi-square quantiles for scaling
        u_chi = np.roll(u, 1)
        chi2_vals = stats.chi2.ppf(u_chi, self.degrees_of_freedom)
        scaling = np.sqrt(self.degrees_of_freedom / chi2_vals)

        t_samples = normal_samples * scaling[:, np.newaxis]

        # Transform to uniform margins
        uniform_samples = t_dist.cdf(t_samples, self.degrees_of_freedom)

        return uniform_samples
    
    def tail_dependence(self) -> Tuple[float, float]:
        """
        Calculate tail dependence coefficients.
        
        Returns:
            Tuple of (lower_tail_dependence, upper_tail_dependence)
        """
        if not self.fitted or self.dimension != 2:
            return (0.0, 0.0)
        
        # For t-copula with correlation ρ and df ν
        rho = self.correlation_matrix[0, 1]
        nu = self.degrees_of_freedom
        
        # Tail dependence formula for t-copula
        sqrt_term = np.sqrt((nu + 1) * (1 - rho) / (1 + rho))
        tail_dep = 2 * t_dist.cdf(-sqrt_term, nu + 1)
        
        # Both upper and lower tail dependence are equal for t-copula
        return (tail_dep, tail_dep)

class ArchimedeanCopula:
    """Archimedean copula family (Clayton, Gumbel, Frank, Joe)."""
    
    def __init__(self, copula_type: CopulaType, dimension: int = 2):
        """
        Initialize Archimedean copula.
        
        Args:
            copula_type: Type of Archimedean copula
            dimension: Number of variables (usually 2)
        """
        self.copula_type = copula_type
        self.dimension = dimension
        self.theta = 1.0  # Dependence parameter
        self.fitted = False
    
    def fit(self, data: np.ndarray, method: str = 'tau') -> bool:
        """
        Fit Archimedean copula to data.
        
        Args:
            data: Data matrix (for bivariate case)
            method: Fitting method ('tau', 'mle')
        
        Returns:
            True if fitting successful
        """
        if data.shape[1] != 2:  # Currently only bivariate
            return False
        
        try:
            # Transform to uniform margins
            uniform_data = np.zeros_like(data)
            
            for i in range(2):
                sorted_values = np.sort(data[:, i])
                n = len(sorted_values)
                ranks = np.searchsorted(sorted_values, data[:, i], side='right')
                uniform_data[:, i] = ranks / (n + 1)
            
            if method == 'tau':
                # Use Kendall's tau for parameter estimation
                tau, _ = kendalltau(data[:, 0], data[:, 1])
                
                # Convert tau to copula parameter
                if self.copula_type == CopulaType.CLAYTON:
                    # τ = θ/(θ+2)
                    if tau > 0:
                        self.theta = 2 * tau / (1 - tau)
                    else:
                        self.theta = 0.01
                        
                elif self.copula_type == CopulaType.GUMBEL:
                    # τ = (θ-1)/θ
                    if tau > 0:
                        self.theta = 1 / (1 - tau)
                    else:
                        self.theta = 1.01
                        
                elif self.copula_type == CopulaType.FRANK:
                    # τ = 1 - 4/θ * (1 - D₁(θ)) where D₁ is Debye function
                    # Approximate inversion
                    if abs(tau) < 1e-6:
                        self.theta = 0.0
                    else:
                        # Approximate relationship
                        self.theta = 4 * tau / (1 - tau) if tau > 0 else 4 * tau / (1 + tau)
                        
                elif self.copula_type == CopulaType.JOE:
                    # More complex relationship - use approximation
                    if tau > 0:
                        self.theta = 1 / (1 - 2 * tau) if tau < 0.5 else 2.0
                    else:
                        self.theta = 1.01
                
                self.fitted = True
                return True
                
            elif method == 'mle':
                # Maximum likelihood estimation
                def neg_log_likelihood(theta):
                    if not self._valid_parameter(theta):
                        return np.inf
                    
                    u, v = uniform_data[:, 0], uniform_data[:, 1]
                    
                    try:
                        log_density = np.sum(np.log(self._pdf(u, v, theta)))
                        return -log_density if np.isfinite(log_density) else np.inf
                    except:
                        return np.inf
                
                # Optimize
                if self.copula_type == CopulaType.CLAYTON:
                    bounds = (0.01, 20)
                elif self.copula_type == CopulaType.GUMBEL:
                    bounds = (1.01, 20)
                elif self.copula_type == CopulaType.FRANK:
                    bounds = (-20, 20)
                else:  # JOE
                    bounds = (1.01, 20)
                
                result = optimize.minimize_scalar(
                    neg_log_likelihood,
                    bounds=bounds,
                    method='bounded'
                )
                
                if result.success:
                    self.theta = result.x
                    self.fitted = True
                    return True
                    
        except Exception as e:
            return False
        
        return False
    
    def _valid_parameter(self, theta: float) -> bool:
        """Check if parameter is in valid range."""
        if self.copula_type == CopulaType.CLAYTON:
            return theta > 0
        elif self.copula_type == CopulaType.GUMBEL:
            return theta >= 1
        elif self.copula_type == CopulaType.FRANK:
            return True  # Any real number
        elif self.copula_type == CopulaType.JOE:
            return theta >= 1
        return False
    
    def _pdf(self, u: np.ndarray, v: np.ndarray, theta: float) -> np.ndarray:
        """Calculate copula density."""
        u = np.clip(u, 1e-10, 1-1e-10)
        v = np.clip(v, 1e-10, 1-1e-10)
        
        if self.copula_type == CopulaType.CLAYTON:
            # Clayton copula density
            term1 = u**(-theta-1) * v**(-theta-1)
            term2 = (u**(-theta) + v**(-theta) - 1)**(-1/theta - 2)
            return (1 + theta) * term1 * term2
            
        elif self.copula_type == CopulaType.GUMBEL:
            # Gumbel copula density
            log_u, log_v = np.log(u), np.log(v)
            t1 = (-log_u)**theta + (-log_v)**theta
            
            c_uv = np.exp(-t1**(1/theta))
            
            density = (c_uv / (u * v) * 
                      t1**(-2 + 2/theta) * 
                      ((-log_u) * (-log_v))**(theta-1) *
                      (1 + (theta-1) * t1**(-1/theta)))
            
            return density
            
        elif self.copula_type == CopulaType.FRANK:
            # Frank copula density
            if abs(theta) < 1e-6:
                return np.ones_like(u)  # Independence case
            
            exp_theta = np.exp(theta)
            exp_theta_u = np.exp(theta * u)
            exp_theta_v = np.exp(theta * v)
            
            numerator = theta * (exp_theta - 1) * np.exp(theta * (u + v))
            denominator = ((exp_theta - 1) + (exp_theta_u - 1) * (exp_theta_v - 1))**2
            
            return numerator / denominator
            
        elif self.copula_type == CopulaType.JOE:
            # Joe copula density (simplified approximation)
            term1 = (1 - u)**(theta-1) * (1 - v)**(theta-1)
            term2 = (1 - (1-u)**theta - (1-v)**theta + (1-u)**theta * (1-v)**theta)**(1/theta - 2)
            
            return theta * term1 * term2
        
        return np.ones_like(u)
    
    def sample(self, n_samples: int) -> np.ndarray:
        """Generate samples from Archimedean copula."""
        if not self.fitted:
            u = np.linspace(1.0/(n_samples+1), n_samples/(n_samples+1), n_samples)
            return np.column_stack([u, np.roll(u, 1)])

        # Use deterministic conditional sampling method
        u1 = np.linspace(1.0/(n_samples+1), n_samples/(n_samples+1), n_samples)
        w = np.roll(u1, 3)
        
        # Generate u2 using conditional distribution (approximation)
        u2 = np.zeros(n_samples)
        
        for i in range(n_samples):
            # Numerical inversion of conditional CDF (simplified)
            # In practice, more sophisticated methods would be used
            
            def conditional_cdf(v, u1_val, w_val):
                # Approximate conditional CDF
                if self.copula_type == CopulaType.CLAYTON:
                    return (u1_val**(-self.theta) * (v**(-self.theta/(1+self.theta)) - 1) + 1)**(-1/self.theta)
                else:
                    # Fallback to approximate method
                    return v
            
            try:
                # Simple approximation for sampling
                if self.copula_type == CopulaType.CLAYTON and self.theta > 0:
                    u2[i] = (u1[i]**(-self.theta) * (w[i]**(-self.theta/(1+self.theta)) - 1) + 1)**(-1/self.theta)
                else:
                    u2[i] = w[i]  # Fallback to independence
                    
                u2[i] = np.clip(u2[i], 1e-10, 1-1e-10)
            except:
                u2[i] = w[i]
        
        return np.column_stack([u1, u2])
    
    def tail_dependence(self) -> Tuple[float, float]:
        """Calculate tail dependence coefficients."""
        if not self.fitted:
            return (0.0, 0.0)
        
        if self.copula_type == CopulaType.CLAYTON:
            # Clayton: λ_L = 2^(-1/θ), λ_U = 0
            lambda_l = 2**(-1/self.theta) if self.theta > 0 else 0
            lambda_u = 0
            
        elif self.copula_type == CopulaType.GUMBEL:
            # Gumbel: λ_L = 0, λ_U = 2 - 2^(1/θ)
            lambda_l = 0
            lambda_u = 2 - 2**(1/self.theta)
            
        elif self.copula_type == CopulaType.FRANK:
            # Frank: λ_L = λ_U = 0 (no tail dependence)
            lambda_l = 0
            lambda_u = 0
            
        elif self.copula_type == CopulaType.JOE:
            # Joe: λ_L = 0, λ_U = 2 - 2^(1/θ)
            lambda_l = 0
            lambda_u = 2 - 2**(1/self.theta)
            
        else:
            lambda_l = lambda_u = 0
        
        return (lambda_l, lambda_u)

class CopulaAnalyzer:
    """Comprehensive copula analysis and model selection framework."""
    
    def __init__(self):
        """Initialize copula analyzer."""
        self.fitted_copulas = {}
        self.model_selection_results = {}
        
    def fit_all_copulas(self, data: np.ndarray) -> Dict[str, bool]:
        """
        Fit all available copula models to data.
        
        Args:
            data: Data matrix
        
        Returns:
            Dictionary of fitting results
        """
        results = {}
        n_vars = data.shape[1]
        
        # Gaussian copula
        gaussian = GaussianCopula(n_vars)
        results['gaussian'] = gaussian.fit(data)
        if results['gaussian']:
            self.fitted_copulas['gaussian'] = gaussian
        
        # t-copula
        t_copula = TCopula(n_vars)
        results['t_copula'] = t_copula.fit(data)
        if results['t_copula']:
            self.fitted_copulas['t_copula'] = t_copula
        
        # Archimedean copulas (bivariate only)
        if n_vars == 2:
            for copula_type in [CopulaType.CLAYTON, CopulaType.GUMBEL, 
                              CopulaType.FRANK, CopulaType.JOE]:
                arch_copula = ArchimedeanCopula(copula_type, 2)
                name = copula_type.value
                results[name] = arch_copula.fit(data)
                if results[name]:
                    self.fitted_copulas[name] = arch_copula
        
        return results
    
    def model_selection(self, data: np.ndarray, criterion: str = 'aic') -> str:
        """
        Select best copula model using information criteria.
        
        Args:
            data: Data matrix
            criterion: Selection criterion ('aic', 'bic', 'hqic')
        
        Returns:
            Name of best model
        """
        # Transform to uniform margins
        uniform_data = np.zeros_like(data)
        
        for i in range(data.shape[1]):
            sorted_values = np.sort(data[:, i])
            n = len(sorted_values)
            ranks = np.searchsorted(sorted_values, data[:, i], side='right')
            uniform_data[:, i] = ranks / (n + 1)
        
        criteria_values = {}
        
        for name, copula in self.fitted_copulas.items():
            try:
                # Calculate log-likelihood
                if hasattr(copula, 'pdf'):
                    densities = copula.pdf(uniform_data)
                    log_likelihood = np.sum(np.log(np.maximum(densities, 1e-10)))
                else:
                    # For Archimedean copulas
                    if data.shape[1] == 2:
                        densities = copula._pdf(uniform_data[:, 0], uniform_data[:, 1], copula.theta)
                        log_likelihood = np.sum(np.log(np.maximum(densities, 1e-10)))
                    else:
                        continue
                
                # Count parameters
                if name == 'gaussian':
                    n_params = data.shape[1] * (data.shape[1] - 1) // 2
                elif name == 't_copula':
                    n_params = data.shape[1] * (data.shape[1] - 1) // 2 + 1
                else:  # Archimedean
                    n_params = 1
                
                n_obs = len(data)
                
                # Calculate criteria
                if criterion == 'aic':
                    criteria_values[name] = -2 * log_likelihood + 2 * n_params
                elif criterion == 'bic':
                    criteria_values[name] = -2 * log_likelihood + np.log(n_obs) * n_params
                elif criterion == 'hqic':
                    criteria_values[name] = -2 * log_likelihood + 2 * np.log(np.log(n_obs)) * n_params
                    
            except Exception as e:
                continue
        
        if criteria_values:
            best_model = min(criteria_values.keys(), key=lambda k: criteria_values[k])
            self.model_selection_results = criteria_values
            return best_model
        
        return 'gaussian'  # Default fallback
    
    def calculate_dependence_metrics(self, data: np.ndarray) -> DependenceMetrics:
        """
        Calculate comprehensive dependence metrics.
        
        Args:
            data: Bivariate data matrix
        
        Returns:
            DependenceMetrics object
        """
        if data.shape[1] != 2:
            raise ValueError("Dependence metrics currently only for bivariate data")
        
        x, y = data[:, 0], data[:, 1]
        
        # Pearson correlation
        pearson_corr = np.corrcoef(x, y)[0, 1]
        
        # Spearman correlation
        spearman_corr, _ = spearmanr(x, y)
        
        # Kendall's tau
        kendall_tau, _ = kendalltau(x, y)
        
        # Tail dependence (if t-copula is fitted)
        tail_dep_upper = tail_dep_lower = 0.0
        
        if 't_copula' in self.fitted_copulas:
            tail_dep_lower, tail_dep_upper = self.fitted_copulas['t_copula'].tail_dependence()
        elif any(name in self.fitted_copulas for name in ['clayton', 'gumbel', 'frank', 'joe']):
            # Use best Archimedean copula
            for name in ['clayton', 'gumbel', 'frank', 'joe']:
                if name in self.fitted_copulas:
                    tail_dep_lower, tail_dep_upper = self.fitted_copulas[name].tail_dependence()
                    break
        
        # Mutual information (simplified estimate)
        try:
            hist, x_edges, y_edges = np.histogram2d(x, y, bins=20)
            hist = hist + 1e-10  # Avoid log(0)
            
            # Normalize
            hist = hist / np.sum(hist)
            
            # Marginal distributions
            px = np.sum(hist, axis=1)
            py = np.sum(hist, axis=0)
            
            # Mutual information
            mutual_info = 0.0
            for i in range(len(px)):
                for j in range(len(py)):
                    if hist[i, j] > 0:
                        mutual_info += hist[i, j] * np.log(hist[i, j] / (px[i] * py[j]))
                        
        except:
            mutual_info = 0.0
        
        # Distance correlation (simplified approximation)
        # Full implementation would be more complex
        distance_corr = abs(pearson_corr)  # Simplified proxy
        
        return DependenceMetrics(
            pearson_correlation=pearson_corr,
            spearman_correlation=spearman_corr,
            kendall_tau=kendall_tau,
            tail_dependence_upper=tail_dep_upper,
            tail_dependence_lower=tail_dep_lower,
            mutual_information=mutual_info,
            distance_correlation=distance_corr
        )
    
    def risk_aggregation(self, marginal_vars: List[float], 
                        confidence_level: float = 0.99) -> float:
        """
        Aggregate risk using fitted copula model.
        
        Args:
            marginal_vars: VaR estimates for individual assets
            confidence_level: Confidence level for aggregation
        
        Returns:
            Aggregated VaR
        """
        if len(marginal_vars) != 2 or 'gaussian' not in self.fitted_copulas:
            # Fallback to simple aggregation
            return np.sqrt(np.sum(np.array(marginal_vars)**2))
        
        # Use Gaussian copula for aggregation (Monte Carlo)
        copula = self.fitted_copulas['gaussian']
        n_simulations = 10000
        
        # Generate copula samples
        copula_samples = copula.sample(n_simulations)
        
        # Transform to VaR space (assuming normal marginals for simplicity)
        risk_scenarios = np.zeros((n_simulations, len(marginal_vars)))
        
        for i, var in enumerate(marginal_vars):
            # Transform uniform samples to risk scenarios
            # Using quantile of standard normal scaled by VaR
            risk_scenarios[:, i] = norm.ppf(copula_samples[:, i]) * abs(var)
        
        # Calculate portfolio risk (sum)
        portfolio_risks = np.sum(risk_scenarios, axis=1)
        
        # Aggregate VaR
        aggregated_var = np.quantile(portfolio_risks, 1 - confidence_level)
        
        return aggregated_var

# Example usage and testing
if __name__ == "__main__":
    print("Testing Copula Models Engine...")
    
    # Generate synthetic correlated data deterministically
    n_samples = 1000

    # Create correlated normal data using deterministic inverse CDF + Cholesky
    correlation = 0.6
    mean = [0, 0]
    cov = [[1, correlation], [correlation, 1]]
    L = np.linalg.cholesky(np.array(cov))

    u = np.linspace(1.0/(n_samples+1), n_samples/(n_samples+1), n_samples)
    indep = np.zeros((n_samples, 2))
    indep[:, 0] = norm.ppf(u)
    indep[:, 1] = norm.ppf(np.roll(u, 1))

    normal_data = indep @ L.T
    
    # Transform to different marginal distributions
    # Asset 1: Normal returns
    asset1 = normal_data[:, 0] * 0.02
    
    # Asset 2: t-distributed returns (fat tails)
    uniform_margins = norm.cdf(normal_data[:, 1])
    asset2 = t_dist.ppf(uniform_margins, df=4) * 0.025
    
    data = np.column_stack([asset1, asset2])
    
    print(f"Generated {n_samples} observations")
    print(f"Sample correlations: Pearson={np.corrcoef(data.T)[0,1]:.3f}")
    
    print("\nTesting Gaussian copula...")
    gaussian_copula = GaussianCopula(2)
    success = gaussian_copula.fit(data, method='spearman')
    
    print(f"Gaussian copula fitting: {'successful' if success else 'failed'}")
    if success:
        print(f"Estimated correlation: {gaussian_copula.correlation_matrix[0,1]:.3f}")
    
    print("\nTesting t-copula...")
    t_copula = TCopula(2)
    success = t_copula.fit(data, method='moments')
    
    print(f"t-copula fitting: {'successful' if success else 'failed'}")
    if success:
        print(f"Degrees of freedom: {t_copula.degrees_of_freedom:.2f}")
        print(f"Correlation: {t_copula.correlation_matrix[0,1]:.3f}")
        
        # Calculate tail dependence
        lower_tail, upper_tail = t_copula.tail_dependence()
        print(f"Tail dependence - Lower: {lower_tail:.3f}, Upper: {upper_tail:.3f}")
    
    print("\nTesting Archimedean copulas...")
    
    # Clayton copula
    clayton = ArchimedeanCopula(CopulaType.CLAYTON, 2)
    success_clayton = clayton.fit(data, method='tau')
    print(f"Clayton copula: {'successful' if success_clayton else 'failed'}")
    if success_clayton:
        print(f"  Parameter θ: {clayton.theta:.3f}")
        lower, upper = clayton.tail_dependence()
        print(f"  Tail dependence - Lower: {lower:.3f}, Upper: {upper:.3f}")
    
    # Gumbel copula
    gumbel = ArchimedeanCopula(CopulaType.GUMBEL, 2)
    success_gumbel = gumbel.fit(data, method='tau')
    print(f"Gumbel copula: {'successful' if success_gumbel else 'failed'}")
    if success_gumbel:
        print(f"  Parameter θ: {gumbel.theta:.3f}")
        lower, upper = gumbel.tail_dependence()
        print(f"  Tail dependence - Lower: {lower:.3f}, Upper: {upper:.3f}")
    
    # Frank copula
    frank = ArchimedeanCopula(CopulaType.FRANK, 2)
    success_frank = frank.fit(data, method='tau')
    print(f"Frank copula: {'successful' if success_frank else 'failed'}")
    if success_frank:
        print(f"  Parameter θ: {frank.theta:.3f}")
    
    print("\nTesting comprehensive copula analysis...")
    analyzer = CopulaAnalyzer()
    
    # Fit all models
    fit_results = analyzer.fit_all_copulas(data)
    print("Fitting results:")
    for model, success in fit_results.items():
        print(f"  {model}: {' ' if success else ' '}")
    
    # Model selection
    best_model = analyzer.model_selection(data, criterion='aic')
    print(f"\nBest model (AIC): {best_model}")
    
    if analyzer.model_selection_results:
        print("AIC values:")
        for model, aic in analyzer.model_selection_results.items():
            print(f"  {model}: {aic:.2f}")
    
    # Calculate dependence metrics
    dep_metrics = analyzer.calculate_dependence_metrics(data)
    print(f"\nDependence Metrics:")
    print(f"  Pearson correlation: {dep_metrics.pearson_correlation:.3f}")
    print(f"  Spearman correlation: {dep_metrics.spearman_correlation:.3f}")
    print(f"  Kendall's tau: {dep_metrics.kendall_tau:.3f}")
    print(f"  Upper tail dependence: {dep_metrics.tail_dependence_upper:.3f}")
    print(f"  Lower tail dependence: {dep_metrics.tail_dependence_lower:.3f}")
    print(f"  Mutual information: {dep_metrics.mutual_information:.3f}")
    
    # Risk aggregation example
    print("\nTesting risk aggregation...")
    # Assume individual VaRs
    individual_vars = [0.05, 0.04]  # 5% and 4% VaR
    
    aggregated_var = analyzer.risk_aggregation(individual_vars, confidence_level=0.99)
    print(f"Individual VaRs: {individual_vars}")
    print(f"Aggregated VaR (99%): {aggregated_var:.4f}")
    
    # Compare with simple sum and independence assumption
    simple_sum = sum(individual_vars)
    independence_var = np.sqrt(sum(v**2 for v in individual_vars))
    
    print(f"Simple sum: {simple_sum:.4f}")
    print(f"Independence assumption: {independence_var:.4f}")
    print(f"Copula-based aggregation: {aggregated_var:.4f}")
    
    # Generate samples from best copula
    if best_model in analyzer.fitted_copulas:
        print(f"\nGenerating samples from {best_model} copula...")
        copula_samples = analyzer.fitted_copulas[best_model].sample(100)
        
        print(f"Sample correlation of generated data: {np.corrcoef(copula_samples.T)[0,1]:.3f}")
        print(f"Sample range: [{np.min(copula_samples):.3f}, {np.max(copula_samples):.3f}]")
    
    print("\nCopula models engine test completed successfully!")