"""
Advanced Stochastic Calculus Engine for QUANTUM-FORGE
Implements sophisticated mathematical models for financial processes.
"""

import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize
from numba import jit, prange
from typing import Tuple, Optional, Union, List
import warnings
warnings.filterwarnings('ignore')

class StochasticProcesses:
    """Advanced stochastic process implementations for financial modeling."""
    
    @staticmethod
    @jit(nopython=True)
    def geometric_brownian_motion(S0: float, mu: float, sigma: float, 
                                 T: float, N: int, M: int = 10000) -> np.ndarray:
        """
        Generate GBM paths with enhanced numerical stability.
        
        Args:
            S0: Initial price
            mu: Drift parameter
            sigma: Volatility parameter
            T: Time horizon
            N: Number of time steps
            M: Number of simulation paths
        
        Returns:
            Array of shape (M, N+1) containing price paths
        """
        dt = T / N
        sqrt_dt = np.sqrt(dt)
        
        # Pre-allocate paths
        paths = np.zeros((M, N + 1))
        paths[:, 0] = S0
        
        # Generate random numbers in chunks for efficiency
        for i in prange(M):
            for j in range(N):
                dW = np.random.normal(0, 1) * sqrt_dt
                paths[i, j + 1] = paths[i, j] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)
        
        return paths
    
    @staticmethod
    @jit(nopython=True)
    def ornstein_uhlenbeck_process(X0: float, theta: float, mu: float, 
                                 sigma: float, T: float, N: int, M: int = 10000) -> np.ndarray:
        """
        Ornstein-Uhlenbeck mean-reverting process for interest rates and volatility.
        
        dX_t = θ(μ - X_t)dt + σdW_t
        """
        dt = T / N
        sqrt_dt = np.sqrt(dt)
        
        paths = np.zeros((M, N + 1))
        paths[:, 0] = X0
        
        for i in prange(M):
            for j in range(N):
                dW = np.random.normal(0, 1) * sqrt_dt
                paths[i, j + 1] = (paths[i, j] + theta * (mu - paths[i, j]) * dt + 
                                 sigma * dW)
        
        return paths
    
    @staticmethod
    @jit(nopython=True)
    def cox_ingersoll_ross(r0: float, kappa: float, theta: float, 
                          sigma: float, T: float, N: int, M: int = 10000) -> np.ndarray:
        """
        Cox-Ingersoll-Ross process for interest rate modeling.
        
        dr_t = κ(θ - r_t)dt + σ√r_t dW_t
        """
        dt = T / N
        sqrt_dt = np.sqrt(dt)
        
        paths = np.zeros((M, N + 1))
        paths[:, 0] = r0
        
        for i in prange(M):
            for j in range(N):
                dW = np.random.normal(0, 1) * sqrt_dt
                r_curr = max(paths[i, j], 1e-8)  # Prevent negative rates
                paths[i, j + 1] = (r_curr + kappa * (theta - r_curr) * dt + 
                                 sigma * np.sqrt(r_curr) * dW)
                paths[i, j + 1] = max(paths[i, j + 1], 0)  # Ensure non-negative
        
        return paths

class HestonModel:
    """Heston stochastic volatility model implementation."""
    
    def __init__(self, S0: float, v0: float, kappa: float, theta: float, 
                 sigma: float, rho: float, r: float = 0.0):
        """
        Initialize Heston model parameters.
        
        Args:
            S0: Initial stock price
            v0: Initial variance
            kappa: Mean reversion speed
            theta: Long-term variance
            sigma: Volatility of volatility
            rho: Correlation between price and volatility
            r: Risk-free rate
        """
        self.S0 = S0
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.r = r
    
    @jit(forceobj=True)
    def simulate_paths(self, T: float, N: int, M: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate Heston model paths using Euler-Maruyama scheme.
        
        Returns:
            Tuple of (price_paths, variance_paths)
        """
        dt = T / N
        sqrt_dt = np.sqrt(dt)
        
        # Initialize arrays
        S_paths = np.zeros((M, N + 1))
        v_paths = np.zeros((M, N + 1))
        
        S_paths[:, 0] = self.S0
        v_paths[:, 0] = self.v0
        
        # Correlated random numbers
        for i in range(M):
            for j in range(N):
                # Generate correlated Brownian motions
                dW1 = np.random.normal(0, 1)
                dW2 = self.rho * dW1 + np.sqrt(1 - self.rho**2) * np.random.normal(0, 1)
                
                # Current values
                S_curr = S_paths[i, j]
                v_curr = max(v_paths[i, j], 1e-8)  # Feller condition
                
                # Update variance (using reflection to ensure positivity)
                v_next = v_curr + self.kappa * (self.theta - v_curr) * dt + self.sigma * np.sqrt(v_curr) * dW2 * sqrt_dt
                v_paths[i, j + 1] = abs(v_next)  # Reflection scheme
                
                # Update stock price
                S_paths[i, j + 1] = S_curr * np.exp((self.r - 0.5 * v_curr) * dt + np.sqrt(v_curr) * dW1 * sqrt_dt)
        
        return S_paths, v_paths
    
    def option_price_monte_carlo(self, K: float, T: float, option_type: str = 'call', 
                               N: int = 1000, M: int = 100000) -> float:
        """Calculate option price using Monte Carlo simulation."""
        S_paths, _ = self.simulate_paths(T, N, M)
        S_T = S_paths[:, -1]
        
        if option_type.lower() == 'call':
            payoffs = np.maximum(S_T - K, 0)
        elif option_type.lower() == 'put':
            payoffs = np.maximum(K - S_T, 0)
        else:
            raise ValueError("option_type must be 'call' or 'put'")
        
        # Discount back to present value
        option_price = np.exp(-self.r * T) * np.mean(payoffs)
        return option_price

class JumpDiffusionModel:
    """Merton jump-diffusion model implementation."""
    
    def __init__(self, S0: float, mu: float, sigma: float, lambda_jump: float, 
                 mu_jump: float, sigma_jump: float):
        """
        Initialize jump-diffusion parameters.
        
        Args:
            S0: Initial stock price
            mu: Drift parameter
            sigma: Diffusion volatility
            lambda_jump: Jump intensity (jumps per unit time)
            mu_jump: Mean jump size (log scale)
            sigma_jump: Jump size volatility
        """
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        self.lambda_jump = lambda_jump
        self.mu_jump = mu_jump
        self.sigma_jump = sigma_jump
    
    def simulate_paths(self, T: float, N: int, M: int = 10000) -> np.ndarray:
        """Simulate jump-diffusion paths."""
        dt = T / N
        sqrt_dt = np.sqrt(dt)
        
        paths = np.zeros((M, N + 1))
        paths[:, 0] = self.S0
        
        for i in range(M):
            for j in range(N):
                # Brownian motion component
                dW = np.random.normal(0, 1) * sqrt_dt
                
                # Jump component (Poisson process)
                dN = np.random.poisson(self.lambda_jump * dt)
                if dN > 0:
                    # Jump sizes (log-normal)
                    jump_sizes = np.random.normal(self.mu_jump, self.sigma_jump, dN)
                    total_jump = np.sum(jump_sizes)
                else:
                    total_jump = 0
                
                # Update price
                paths[i, j + 1] = paths[i, j] * np.exp(
                    (self.mu - 0.5 * self.sigma**2) * dt + 
                    self.sigma * dW + total_jump
                )
        
        return paths

class FractionalBrownianMotion:
    """Fractional Brownian motion for long-memory processes."""
    
    @staticmethod
    def generate_fbm(H: float, T: float, N: int) -> np.ndarray:
        """
        Generate fractional Brownian motion using Cholesky decomposition.
        
        Args:
            H: Hurst parameter (0.5 = standard BM, >0.5 = persistent, <0.5 = anti-persistent)
            T: Time horizon
            N: Number of time steps
        
        Returns:
            FBM path
        """
        dt = T / N
        t = np.linspace(0, T, N + 1)
        
        # Covariance matrix for fBM
        def fbm_covariance(s, t, H):
            return 0.5 * (abs(t)**(2*H) + abs(s)**(2*H) - abs(t-s)**(2*H))
        
        # Build covariance matrix
        cov_matrix = np.zeros((N + 1, N + 1))
        for i in range(N + 1):
            for j in range(N + 1):
                cov_matrix[i, j] = fbm_covariance(t[i], t[j], H)
        
        # Cholesky decomposition
        L = np.linalg.cholesky(cov_matrix + 1e-10 * np.eye(N + 1))
        
        # Generate fBM
        Z = np.random.normal(0, 1, N + 1)
        fBM = L @ Z
        
        return fBM

class VolatilityModels:
    """Advanced volatility modeling techniques."""
    
    @staticmethod
    def garch_simulation(omega: float, alpha: float, beta: float, 
                        T: int, initial_vol: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate GARCH(1,1) process.
        
        Args:
            omega: Long-term variance
            alpha: ARCH coefficient
            beta: GARCH coefficient
            T: Number of time periods
            initial_vol: Initial volatility
        
        Returns:
            Tuple of (returns, volatilities)
        """
        if initial_vol is None:
            initial_vol = np.sqrt(omega / (1 - alpha - beta))
        
        returns = np.zeros(T)
        volatilities = np.zeros(T)
        volatilities[0] = initial_vol
        
        for t in range(1, T):
            # Generate standardized innovation
            epsilon = np.random.normal(0, 1)
            
            # Calculate return
            returns[t] = volatilities[t-1] * epsilon
            
            # Update volatility
            volatilities[t] = np.sqrt(
                omega + alpha * returns[t-1]**2 + beta * volatilities[t-1]**2
            )
        
        return returns, volatilities
    
    @staticmethod
    def local_volatility_dupire(S: np.ndarray, T: np.ndarray, 
                               option_prices: np.ndarray) -> np.ndarray:
        """
        Estimate local volatility using Dupire's formula.
        
        σ²(S,T) = (∂C/∂T + rS∂C/∂S) / (0.5 S² ∂²C/∂S²)
        """
        # Numerical derivatives (simplified implementation)
        dC_dT = np.gradient(option_prices, T, axis=1)
        dC_dS = np.gradient(option_prices, S, axis=0)
        d2C_dS2 = np.gradient(dC_dS, S, axis=0)
        
        # Avoid division by zero
        d2C_dS2 = np.where(np.abs(d2C_dS2) < 1e-10, 1e-10, d2C_dS2)
        
        # Dupire formula (assuming r=0 for simplicity)
        S_grid, T_grid = np.meshgrid(S, T, indexing='ij')
        local_var = 2 * dC_dT / (S_grid**2 * d2C_dS2)
        
        # Ensure non-negative variance
        local_var = np.maximum(local_var, 1e-6)
        local_vol = np.sqrt(local_var)
        
        return local_vol

class StochasticVolatilityJumps:
    """Bates model: Heston with jumps in both price and volatility."""
    
    def __init__(self, S0: float, v0: float, kappa: float, theta: float, 
                 sigma: float, rho: float, lambda_s: float, mu_s: float, 
                 sigma_s: float, lambda_v: float, mu_v: float, sigma_v: float):
        """
        Initialize Bates model parameters.
        
        Args:
            S0, v0, kappa, theta, sigma, rho: Heston parameters
            lambda_s, mu_s, sigma_s: Price jump parameters
            lambda_v, mu_v, sigma_v: Volatility jump parameters
        """
        self.S0 = S0
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.lambda_s = lambda_s
        self.mu_s = mu_s
        self.sigma_s = sigma_s
        self.lambda_v = lambda_v
        self.mu_v = mu_v
        self.sigma_v = sigma_v
    
    def simulate_paths(self, T: float, N: int, M: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate Bates model paths."""
        dt = T / N
        sqrt_dt = np.sqrt(dt)
        
        S_paths = np.zeros((M, N + 1))
        v_paths = np.zeros((M, N + 1))
        
        S_paths[:, 0] = self.S0
        v_paths[:, 0] = self.v0
        
        for i in range(M):
            for j in range(N):
                # Generate correlated Brownian motions
                dW1 = np.random.normal(0, 1)
                dW2 = self.rho * dW1 + np.sqrt(1 - self.rho**2) * np.random.normal(0, 1)
                
                # Current values
                S_curr = S_paths[i, j]
                v_curr = max(v_paths[i, j], 1e-8)
                
                # Price jumps
                dN_s = np.random.poisson(self.lambda_s * dt)
                J_s = 0
                if dN_s > 0:
                    jump_sizes = np.random.normal(self.mu_s, self.sigma_s, dN_s)
                    J_s = np.sum(jump_sizes)
                
                # Volatility jumps
                dN_v = np.random.poisson(self.lambda_v * dt)
                J_v = 0
                if dN_v > 0:
                    jump_sizes = np.random.normal(self.mu_v, self.sigma_v, dN_v)
                    J_v = np.sum(jump_sizes)
                
                # Update variance
                v_next = v_curr + self.kappa * (self.theta - v_curr) * dt + \
                        self.sigma * np.sqrt(v_curr) * dW2 * sqrt_dt + J_v
                v_paths[i, j + 1] = max(v_next, 1e-8)
                
                # Update stock price
                S_paths[i, j + 1] = S_curr * np.exp(
                    -0.5 * v_curr * dt + np.sqrt(v_curr) * dW1 * sqrt_dt + J_s
                )
        
        return S_paths, v_paths

# Example usage and testing
if __name__ == "__main__":
    # Test Heston model
    heston = HestonModel(S0=100, v0=0.04, kappa=2.0, theta=0.04, 
                        sigma=0.3, rho=-0.7, r=0.05)
    
    S_paths, v_paths = heston.simulate_paths(T=1.0, N=252, M=1000)
    print(f"Heston simulation completed. Final prices: {S_paths[:, -1][:5]}")
    
    # Test option pricing
    call_price = heston.option_price_monte_carlo(K=100, T=1.0, option_type='call')
    print(f"Heston call option price: {call_price:.4f}")
    
    # Test jump-diffusion
    jump_model = JumpDiffusionModel(S0=100, mu=0.1, sigma=0.2, 
                                   lambda_jump=0.1, mu_jump=0, sigma_jump=0.1)
    
    jump_paths = jump_model.simulate_paths(T=1.0, N=252, M=1000)
    print(f"Jump-diffusion simulation completed. Final prices: {jump_paths[:, -1][:5]}")
    
    # Test fractional Brownian motion
    fbm_path = FractionalBrownianMotion.generate_fbm(H=0.7, T=1.0, N=252)
    print(f"fBM generated. Final value: {fbm_path[-1]:.4f}")