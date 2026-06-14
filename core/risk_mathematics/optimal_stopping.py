"""
Optimal Stopping Theory Engine for QUANTUM-FORGE
Implements advanced optimal stopping models for timing decisions and option valuation.
"""

import numpy as np
import pandas as pd
from scipy import stats, optimize
from scipy.interpolate import interp1d, RectBivariateSpline
from typing import Dict, List, Tuple, Optional, Union, Callable
import warnings
from numba import jit, prange
from dataclasses import dataclass
from enum import Enum
from collections import deque
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

class StoppingCriterion(Enum):
    """Types of stopping criteria."""
    THRESHOLD = "threshold"
    OPTIMAL_VALUE = "optimal_value"
    TIME_LIMIT = "time_limit"
    BARRIER = "barrier"
    LOOKBACK = "lookback"

class ExerciseStyle(Enum):
    """Exercise styles for options."""
    AMERICAN = "american"
    BERMUDAN = "bermudan"
    EUROPEAN = "european"

@dataclass
class StoppingRule:
    """Optimal stopping rule specification."""
    criterion: StoppingCriterion
    threshold: float
    exercise_times: Optional[List[float]] = None
    time_limit: Optional[float] = None
    lookback_window: Optional[int] = None

@dataclass
class OptimalStoppingResult:
    """Results from optimal stopping problem."""
    optimal_value: float
    stopping_time: float
    continuation_value: float
    exercise_value: float
    optimal_policy: Callable
    convergence_info: Dict

class AmericanOptionPricer:
    """American option pricing using optimal stopping theory."""
    
    def __init__(self, option_type: str = 'call'):
        """
        Initialize American option pricer.
        
        Args:
            option_type: 'call' or 'put'
        """
        self.option_type = option_type.lower()
        self.fitted = False
        
        # Grid parameters
        self.S_grid = None
        self.t_grid = None
        self.value_grid = None
        self.continuation_grid = None
        
    def price_binomial(self, S0: float, K: float, T: float, r: float, 
                      sigma: float, n_steps: int = 100) -> Dict:
        """
        Price American option using binomial tree.
        
        Args:
            S0: Initial stock price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            n_steps: Number of time steps
        
        Returns:
            Dictionary with pricing results
        """
        dt = T / n_steps
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(r * dt) - d) / (u - d)
        
        # Initialize price tree
        price_tree = np.zeros((n_steps + 1, n_steps + 1))
        
        # Stock prices at each node
        for i in range(n_steps + 1):
            for j in range(i + 1):
                price_tree[j, i] = S0 * (u ** j) * (d ** (i - j))
        
        # Option values at maturity
        option_tree = np.zeros((n_steps + 1, n_steps + 1))
        
        for j in range(n_steps + 1):
            if self.option_type == 'call':
                option_tree[j, n_steps] = max(0, price_tree[j, n_steps] - K)
            else:  # put
                option_tree[j, n_steps] = max(0, K - price_tree[j, n_steps])
        
        # Backward induction
        exercise_boundary = []
        
        for i in range(n_steps - 1, -1, -1):
            for j in range(i + 1):
                # Continuation value
                continuation = np.exp(-r * dt) * (p * option_tree[j + 1, i + 1] + 
                                                (1 - p) * option_tree[j, i + 1])
                
                # Exercise value
                if self.option_type == 'call':
                    exercise = max(0, price_tree[j, i] - K)
                else:  # put
                    exercise = max(0, K - price_tree[j, i])
                
                # Optimal value
                option_tree[j, i] = max(continuation, exercise)
                
            # Find exercise boundary (approximate)
            if i < n_steps:
                boundary_price = 0
                for j in range(i + 1):
                    if self.option_type == 'call':
                        exercise_val = max(0, price_tree[j, i] - K)
                    else:
                        exercise_val = max(0, K - price_tree[j, i])
                    
                    if option_tree[j, i] <= exercise_val + 1e-6:  # Small tolerance
                        boundary_price = price_tree[j, i]
                        break
                
                exercise_boundary.append(boundary_price)
        
        # Calculate Greeks (finite difference)
        delta = (option_tree[1, 1] - option_tree[0, 1]) / (price_tree[1, 1] - price_tree[0, 1])
        
        return {
            'price': option_tree[0, 0],
            'delta': delta,
            'exercise_boundary': list(reversed(exercise_boundary)),
            'price_tree': price_tree,
            'option_tree': option_tree
        }
    
    def price_lsm(self, S0: float, K: float, T: float, r: float, 
                 sigma: float, n_paths: int = 10000, n_steps: int = 100) -> Dict:
        """
        Price American option using Longstaff-Schwartz method.
        
        Args:
            S0: Initial stock price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            n_paths: Number of Monte Carlo paths
            n_steps: Number of time steps
        
        Returns:
            Dictionary with pricing results
        """
        dt = T / n_steps
        discount = np.exp(-r * dt)
        
        # Generate price paths
        np.random.seed(42)
        Z = np.random.standard_normal((n_paths, n_steps))
        
        # Stock price paths
        S = np.zeros((n_paths, n_steps + 1))
        S[:, 0] = S0
        
        for i in range(n_steps):
            S[:, i + 1] = S[:, i] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, i])
        
        # Payoff at maturity
        if self.option_type == 'call':
            payoff = np.maximum(S - K, 0)
        else:  # put
            payoff = np.maximum(K - S, 0)
        
        # Value matrix
        V = np.zeros((n_paths, n_steps + 1))
        V[:, -1] = payoff[:, -1]
        
        # Backward induction with regression
        for i in range(n_steps - 1, 0, -1):
            # Find in-the-money paths
            exercise_value = payoff[:, i]
            in_money = exercise_value > 0
            
            if np.sum(in_money) == 0:
                V[:, i] = discount * V[:, i + 1]
                continue
            
            # Regression variables (polynomial basis)
            X = S[in_money, i]
            Y = discount * V[in_money, i + 1]
            
            # Fit polynomial regression (degree 3)
            try:
                if len(X) >= 4:  # Need enough points for degree 3
                    basis_matrix = np.column_stack([
                        np.ones_like(X),
                        X,
                        X**2,
                        X**3
                    ])
                    
                    coeffs = np.linalg.lstsq(basis_matrix, Y, rcond=None)[0]
                    
                    # Continuation value for all in-the-money paths
                    continuation_value = (coeffs[0] + coeffs[1] * X + 
                                       coeffs[2] * X**2 + coeffs[3] * X**3)
                    
                    # Exercise decision
                    exercise_decision = exercise_value[in_money] >= continuation_value
                    
                    # Update values
                    V[in_money, i] = np.where(exercise_decision, 
                                            exercise_value[in_money],
                                            discount * V[in_money, i + 1])
                    
                    # Out-of-money paths
                    V[~in_money, i] = discount * V[~in_money, i + 1]
                    
                else:
                    # Not enough points for regression
                    V[:, i] = discount * V[:, i + 1]
                    
            except:
                # Fallback: just discount
                V[:, i] = discount * V[:, i + 1]
        
        # Price at t=0
        option_price = np.mean(V[:, 0])
        
        # Calculate standard error
        std_error = np.std(V[:, 0]) / np.sqrt(n_paths)
        
        # Estimate optimal exercise boundary
        exercise_boundary = []
        for i in range(1, n_steps):
            exercise_prices = []
            for path in range(n_paths):
                if payoff[path, i] > 0:  # In the money
                    # Check if exercised at this step
                    if abs(V[path, i] - payoff[path, i]) < 1e-6:
                        exercise_prices.append(S[path, i])
            
            if exercise_prices:
                exercise_boundary.append(np.mean(exercise_prices))
            else:
                exercise_boundary.append(0)
        
        return {
            'price': option_price,
            'std_error': std_error,
            'confidence_interval': (option_price - 1.96 * std_error, 
                                  option_price + 1.96 * std_error),
            'exercise_boundary': exercise_boundary,
            'value_paths': V
        }

class OptimalStoppingProblem:
    """General optimal stopping problem solver."""
    
    def __init__(self, payoff_function: Callable, discount_rate: float = 0.05):
        """
        Initialize optimal stopping problem.
        
        Args:
            payoff_function: Function that returns payoff given state
            discount_rate: Discount rate for future payoffs
        """
        self.payoff_function = payoff_function
        self.discount_rate = discount_rate
        self.fitted = False
        
        # Solution grids
        self.state_grid = None
        self.time_grid = None
        self.value_function = None
        self.continuation_function = None
        self.stopping_region = None
    
    def solve_finite_horizon(self, state_bounds: Tuple[float, float], 
                           time_horizon: float, n_state: int = 100, 
                           n_time: int = 100) -> OptimalStoppingResult:
        """
        Solve finite horizon optimal stopping problem using dynamic programming.
        
        Args:
            state_bounds: (min_state, max_state)
            time_horizon: Maximum time horizon
            n_state: Number of state grid points
            n_time: Number of time grid points
        
        Returns:
            OptimalStoppingResult object
        """
        # Create grids
        self.state_grid = np.linspace(state_bounds[0], state_bounds[1], n_state)
        self.time_grid = np.linspace(0, time_horizon, n_time)
        dt = time_horizon / (n_time - 1)
        
        # Initialize value function
        self.value_function = np.zeros((n_state, n_time))
        self.continuation_function = np.zeros((n_state, n_time))
        self.stopping_region = np.zeros((n_state, n_time), dtype=bool)
        
        # Terminal condition
        for i, state in enumerate(self.state_grid):
            self.value_function[i, -1] = self.payoff_function(state, time_horizon)
        
        # Backward induction
        for j in range(n_time - 2, -1, -1):
            t = self.time_grid[j]
            
            for i, state in enumerate(self.state_grid):
                # Immediate payoff from stopping
                immediate_payoff = self.payoff_function(state, t)
                
                # Continuation value (simplified diffusion approximation)
                if j < n_time - 1:
                    # Simple expectation assuming Brownian motion
                    continuation = np.exp(-self.discount_rate * dt) * self.value_function[i, j + 1]
                    
                    # Add drift and diffusion terms (simplified)
                    if i > 0 and i < n_state - 1:
                        drift_term = 0  # Assume zero drift for simplicity
                        diffusion_term = 0.01 * (self.value_function[i + 1, j + 1] - 
                                               2 * self.value_function[i, j + 1] + 
                                               self.value_function[i - 1, j + 1])
                        
                        continuation += dt * (drift_term + 0.5 * diffusion_term)
                else:
                    continuation = 0
                
                self.continuation_function[i, j] = continuation
                
                # Optimal decision
                if immediate_payoff >= continuation:
                    self.value_function[i, j] = immediate_payoff
                    self.stopping_region[i, j] = True
                else:
                    self.value_function[i, j] = continuation
                    self.stopping_region[i, j] = False
        
        # Create optimal policy function
        def optimal_policy(state: float, time: float) -> bool:
            """Returns True if should stop, False if should continue."""
            if time >= time_horizon:
                return True
            
            # Interpolate on grid
            try:
                state_idx = np.searchsorted(self.state_grid, state)
                time_idx = np.searchsorted(self.time_grid, time)
                
                state_idx = np.clip(state_idx, 0, n_state - 1)
                time_idx = np.clip(time_idx, 0, n_time - 1)
                
                return self.stopping_region[state_idx, time_idx]
            except:
                return True
        
        # Calculate initial optimal value
        initial_state = (state_bounds[0] + state_bounds[1]) / 2
        initial_idx = np.searchsorted(self.state_grid, initial_state)
        initial_idx = np.clip(initial_idx, 0, n_state - 1)
        
        optimal_value = self.value_function[initial_idx, 0]
        
        self.fitted = True
        
        return OptimalStoppingResult(
            optimal_value=optimal_value,
            stopping_time=0.0,  # Would need simulation to estimate
            continuation_value=self.continuation_function[initial_idx, 0],
            exercise_value=self.payoff_function(initial_state, 0),
            optimal_policy=optimal_policy,
            convergence_info={'method': 'finite_difference', 'grid_size': (n_state, n_time)}
        )
    
    def solve_monte_carlo(self, initial_state: float, time_horizon: float,
                         n_paths: int = 10000, n_steps: int = 100) -> OptimalStoppingResult:
        """
        Solve using Monte Carlo simulation with dynamic programming.
        
        Args:
            initial_state: Initial state value
            time_horizon: Time horizon
            n_paths: Number of simulation paths
            n_steps: Number of time steps
        
        Returns:
            OptimalStoppingResult object
        """
        dt = time_horizon / n_steps
        
        # Generate state paths (assume geometric Brownian motion)
        np.random.seed(42)
        dW = np.random.normal(0, np.sqrt(dt), (n_paths, n_steps))
        
        states = np.zeros((n_paths, n_steps + 1))
        states[:, 0] = initial_state
        
        # Simple GBM evolution
        mu = 0.05  # Drift
        sigma = 0.2  # Volatility
        
        for i in range(n_steps):
            states[:, i + 1] = states[:, i] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW[:, i])
        
        # Payoffs at each time step
        payoffs = np.zeros((n_paths, n_steps + 1))
        for i in range(n_steps + 1):
            t = i * dt
            for path in range(n_paths):
                payoffs[path, i] = self.payoff_function(states[path, i], t)
        
        # Backward induction to find optimal stopping times
        values = np.zeros((n_paths, n_steps + 1))
        values[:, -1] = payoffs[:, -1]  # Terminal values
        
        optimal_stopping_times = np.full(n_paths, time_horizon)
        
        for i in range(n_steps - 1, -1, -1):
            t = i * dt
            
            # Continuation value (discounted future value)
            continuation_values = np.exp(-self.discount_rate * dt) * values[:, i + 1]
            
            # Exercise values
            exercise_values = payoffs[:, i]
            
            # Optimal decision
            exercise_now = exercise_values >= continuation_values
            
            values[:, i] = np.where(exercise_now, exercise_values, continuation_values)
            
            # Update stopping times
            optimal_stopping_times = np.where(
                (exercise_now) & (optimal_stopping_times == time_horizon),
                t,
                optimal_stopping_times
            )
        
        # Calculate results
        optimal_value = np.mean(values[:, 0])
        mean_stopping_time = np.mean(optimal_stopping_times)
        
        # Create simple policy based on simulation results
        def optimal_policy(state: float, time: float) -> bool:
            """Simple threshold-based policy."""
            payoff_now = self.payoff_function(state, time)
            # Rough heuristic: stop if payoff is above certain threshold
            threshold = np.percentile([self.payoff_function(s, time) for s in [state * 0.9, state, state * 1.1]], 75)
            return payoff_now >= threshold * 0.8
        
        return OptimalStoppingResult(
            optimal_value=optimal_value,
            stopping_time=mean_stopping_time,
            continuation_value=np.mean(continuation_values),
            exercise_value=np.mean(exercise_values),
            optimal_policy=optimal_policy,
            convergence_info={'method': 'monte_carlo', 'n_paths': n_paths, 'n_steps': n_steps}
        )

class BarrierOptionPricer:
    """Barrier option pricing with optimal stopping features."""
    
    def __init__(self, option_type: str, barrier_type: str):
        """
        Initialize barrier option pricer.
        
        Args:
            option_type: 'call' or 'put'
            barrier_type: 'up_and_out', 'up_and_in', 'down_and_out', 'down_and_in'
        """
        self.option_type = option_type.lower()
        self.barrier_type = barrier_type.lower()
    
    def price_analytical(self, S0: float, K: float, B: float, T: float, 
                        r: float, sigma: float, rebate: float = 0.0) -> Dict:
        """
        Price barrier option using analytical formulas when available.
        
        Args:
            S0: Initial stock price
            K: Strike price
            B: Barrier level
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            rebate: Rebate paid if barrier is hit
        
        Returns:
            Dictionary with pricing results
        """
        # Calculate parameters
        mu = (r - 0.5 * sigma**2) / sigma**2
        lambda_param = mu + 0.5
        
        x1 = np.log(S0 / K) / (sigma * np.sqrt(T)) + lambda_param * sigma * np.sqrt(T)
        x2 = np.log(S0 / B) / (sigma * np.sqrt(T)) + lambda_param * sigma * np.sqrt(T)
        
        y1 = np.log(B**2 / (S0 * K)) / (sigma * np.sqrt(T)) + lambda_param * sigma * np.sqrt(T)
        y2 = np.log(B / S0) / (sigma * np.sqrt(T)) + lambda_param * sigma * np.sqrt(T)
        
        # Standard Black-Scholes terms
        if self.option_type == 'call':
            vanilla_call = S0 * stats.norm.cdf(x1) - K * np.exp(-r * T) * stats.norm.cdf(x1 - sigma * np.sqrt(T))
            
            if self.barrier_type == 'up_and_out':
                if B <= K:
                    # Barrier below strike
                    price = vanilla_call
                else:
                    # Barrier above strike
                    term1 = S0 * stats.norm.cdf(x2) - K * np.exp(-r * T) * stats.norm.cdf(x2 - sigma * np.sqrt(T))
                    term2 = S0 * (B / S0)**(2 * lambda_param) * stats.norm.cdf(y2)
                    term3 = K * np.exp(-r * T) * (B / S0)**(2 * lambda_param - 2) * stats.norm.cdf(y2 - sigma * np.sqrt(T))
                    
                    price = vanilla_call - term1 + term2 - term3
                    
            elif self.barrier_type == 'up_and_in':
                up_and_out_price = self.price_analytical(S0, K, B, T, r, sigma, rebate)['price']
                price = vanilla_call - up_and_out_price
                
            elif self.barrier_type == 'down_and_out':
                if B >= K:
                    price = 0  # Always knocked out
                else:
                    # Complex formula for down-and-out call
                    eta = 1 if self.option_type == 'call' else -1
                    
                    term1 = S0 * stats.norm.cdf(eta * x1) - K * np.exp(-r * T) * stats.norm.cdf(eta * (x1 - sigma * np.sqrt(T)))
                    term2 = S0 * (B / S0)**(2 * lambda_param) * stats.norm.cdf(eta * y1)
                    term3 = K * np.exp(-r * T) * (B / S0)**(2 * lambda_param - 2) * stats.norm.cdf(eta * (y1 - sigma * np.sqrt(T)))
                    
                    price = term1 - term2 + term3
                    
            else:  # down_and_in
                down_and_out_price = self.price_analytical(S0, K, B, T, r, sigma, rebate)['price']
                price = vanilla_call - down_and_out_price
        
        else:  # put option
            vanilla_put = K * np.exp(-r * T) * stats.norm.cdf(-x1 + sigma * np.sqrt(T)) - S0 * stats.norm.cdf(-x1)
            
            # Similar calculations for put options (omitted for brevity)
            # In practice, use put-call parity or similar formulas
            price = vanilla_put  # Simplified
        
        # Add rebate if applicable
        if rebate > 0:
            # Probability of hitting barrier
            hit_prob = self._barrier_hit_probability(S0, B, T, r, sigma)
            rebate_value = rebate * np.exp(-r * T) * hit_prob
            price += rebate_value
        
        return {
            'price': max(0, price),
            'vanilla_price': vanilla_call if self.option_type == 'call' else vanilla_put,
            'barrier_adjustment': price - (vanilla_call if self.option_type == 'call' else vanilla_put)
        }
    
    def _barrier_hit_probability(self, S0: float, B: float, T: float, 
                               r: float, sigma: float) -> float:
        """Calculate probability of hitting barrier."""
        mu = r - 0.5 * sigma**2
        
        if self.barrier_type.startswith('up'):
            # Probability of max > B
            a = (np.log(B / S0) - mu * T) / (sigma * np.sqrt(T))
            b = (np.log(B / S0) + mu * T) / (sigma * np.sqrt(T))
            
            prob = stats.norm.cdf(-a) + (B / S0)**(2 * mu / sigma**2) * stats.norm.cdf(-b)
            
        else:  # down
            # Probability of min < B  
            a = (np.log(S0 / B) + mu * T) / (sigma * np.sqrt(T))
            b = (np.log(S0 / B) - mu * T) / (sigma * np.sqrt(T))
            
            prob = stats.norm.cdf(-a) + (S0 / B)**(2 * mu / sigma**2) * stats.norm.cdf(-b)
        
        return prob

class LookbackOptionPricer:
    """Lookback option pricing using optimal stopping theory."""
    
    def __init__(self, option_type: str, lookback_type: str):
        """
        Initialize lookback option pricer.
        
        Args:
            option_type: 'call' or 'put'
            lookback_type: 'floating' or 'fixed'
        """
        self.option_type = option_type.lower()
        self.lookback_type = lookback_type.lower()
    
    def price_analytical(self, S0: float, K: float, T: float, r: float, 
                        sigma: float, S_min: Optional[float] = None, 
                        S_max: Optional[float] = None) -> Dict:
        """
        Price lookback option using analytical formulas.
        
        Args:
            S0: Current stock price
            K: Strike price (for fixed strike lookback)
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            S_min: Current minimum (for floating strike calls)
            S_max: Current maximum (for floating strike puts)
        
        Returns:
            Dictionary with pricing results
        """
        # Parameters
        b = r  # Cost of carry (assume no dividends)
        
        a1 = (np.log(S0 / K) + (b + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        a2 = a1 - sigma * np.sqrt(T)
        a3 = (np.log(S0 / K) + (-b + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        
        if self.lookback_type == 'floating':
            if self.option_type == 'call':
                # Floating strike lookback call: max(S_T - min(S_t), 0)
                if S_min is None:
                    S_min = S0
                
                m = min(S_min, S0)
                
                if S0 > m:
                    term1 = S0 - m
                    term2 = S0 * stats.norm.cdf(a1) - m * np.exp(-r * T) * stats.norm.cdf(a2)
                    term3 = (S0 * sigma**2 / (2 * b)) * ((S0 / m)**(2 * b / sigma**2) * stats.norm.cdf(-a3) - np.exp(-r * T) * stats.norm.cdf(-a1))
                    
                    price = term1 + term2 - term3
                else:
                    price = S0 * stats.norm.cdf(a1) - S0 * np.exp(-r * T) * stats.norm.cdf(a2)
                    
            else:  # put
                # Floating strike lookback put: max(max(S_t) - S_T, 0)  
                if S_max is None:
                    S_max = S0
                
                M = max(S_max, S0)
                
                term1 = M - S0
                term2 = -S0 * stats.norm.cdf(-a1) + M * np.exp(-r * T) * stats.norm.cdf(-a2)
                term3 = (S0 * sigma**2 / (2 * b)) * ((S0 / M)**(2 * b / sigma**2) * stats.norm.cdf(a3) - np.exp(-r * T) * stats.norm.cdf(a1))
                
                price = term1 + term2 + term3
        
        else:  # fixed strike
            if self.option_type == 'call':
                # Fixed strike lookback call: max(max(S_t) - K, 0)
                if S_max is None:
                    S_max = S0
                
                M = max(S_max, S0)
                
                if K < M:
                    term1 = M - K
                    term2 = S0 * stats.norm.cdf(a1) - K * np.exp(-r * T) * stats.norm.cdf(a2)
                    term3 = (S0 * sigma**2 / (2 * b)) * ((S0 / M)**(2 * b / sigma**2) * stats.norm.cdf(-a3) - np.exp(-r * T) * stats.norm.cdf(-a1))
                    
                    price = term1 + term2 - term3
                else:
                    # Standard European call
                    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
                    d2 = d1 - sigma * np.sqrt(T)
                    
                    price = S0 * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
            
            else:  # put
                # Fixed strike lookback put: max(K - min(S_t), 0)
                if S_min is None:
                    S_min = S0
                
                m = min(S_min, S0)
                
                term1 = K - m
                term2 = -S0 * stats.norm.cdf(-a1) + K * np.exp(-r * T) * stats.norm.cdf(-a2)
                term3 = (S0 * sigma**2 / (2 * b)) * ((S0 / m)**(2 * b / sigma**2) * stats.norm.cdf(a3) - np.exp(-r * T) * stats.norm.cdf(a1))
                
                price = term1 + term2 + term3
        
        return {
            'price': max(0, price),
            'option_type': self.option_type,
            'lookback_type': self.lookback_type
        }

class OptimalStoppingAnalyzer:
    """Comprehensive optimal stopping analysis framework."""
    
    def __init__(self):
        """Initialize optimal stopping analyzer."""
        self.problems = {}
        self.solutions = {}
    
    def add_problem(self, name: str, payoff_function: Callable, 
                   discount_rate: float = 0.05):
        """Add optimal stopping problem."""
        self.problems[name] = OptimalStoppingProblem(payoff_function, discount_rate)
    
    def solve_all_problems(self, state_bounds: Tuple[float, float],
                          time_horizon: float, method: str = 'finite_horizon') -> Dict:
        """
        Solve all registered problems.
        
        Args:
            state_bounds: State space bounds
            time_horizon: Time horizon
            method: Solution method
        
        Returns:
            Dictionary of solutions
        """
        results = {}
        
        for name, problem in self.problems.items():
            try:
                if method == 'finite_horizon':
                    solution = problem.solve_finite_horizon(state_bounds, time_horizon)
                elif method == 'monte_carlo':
                    initial_state = (state_bounds[0] + state_bounds[1]) / 2
                    solution = problem.solve_monte_carlo(initial_state, time_horizon)
                else:
                    continue
                
                results[name] = solution
                self.solutions[name] = solution
                
            except Exception as e:
                results[name] = None
        
        return results
    
    def compare_solutions(self) -> pd.DataFrame:
        """Compare solutions across different problems."""
        comparison_data = []
        
        for name, solution in self.solutions.items():
            if solution is not None:
                comparison_data.append({
                    'Problem': name,
                    'Optimal_Value': solution.optimal_value,
                    'Stopping_Time': solution.stopping_time,
                    'Continuation_Value': solution.continuation_value,
                    'Exercise_Value': solution.exercise_value,
                    'Method': solution.convergence_info.get('method', 'unknown')
                })
        
        return pd.DataFrame(comparison_data)

# Example usage and testing
if __name__ == "__main__":
    print("Testing Optimal Stopping Theory Engine...")
    
    # Test American option pricing
    print("\nTesting American option pricing...")
    
    # Parameters
    S0 = 100  # Initial stock price
    K = 105   # Strike price
    T = 0.25  # 3 months
    r = 0.05  # Risk-free rate
    sigma = 0.2  # Volatility
    
    # American put option
    american_put = AmericanOptionPricer('put')
    
    # Binomial tree pricing
    binomial_result = american_put.price_binomial(S0, K, T, r, sigma, n_steps=100)
    print(f"American put (binomial): ${binomial_result['price']:.4f}")
    print(f"Delta: {binomial_result['delta']:.4f}")
    
    # Longstaff-Schwartz pricing
    lsm_result = american_put.price_lsm(S0, K, T, r, sigma, n_paths=5000, n_steps=50)
    print(f"American put (LSM): ${lsm_result['price']:.4f} ± {lsm_result['std_error']:.4f}")
    print(f"95% CI: [{lsm_result['confidence_interval'][0]:.4f}, {lsm_result['confidence_interval'][1]:.4f}]")
    
    # Compare with European put
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    european_put = K*np.exp(-r*T)*stats.norm.cdf(-d2) - S0*stats.norm.cdf(-d1)
    
    print(f"European put: ${european_put:.4f}")
    print(f"Early exercise premium: ${binomial_result['price'] - european_put:.4f}")
    
    # Test general optimal stopping problem
    print("\nTesting general optimal stopping problem...")
    
    # Define a simple payoff function: option to sell an asset
    def asset_payoff(state: float, time: float) -> float:
        """Payoff from selling asset at given state and time."""
        # Time decay reduces value
        decay_factor = np.exp(-0.1 * time)
        return max(0, state - 95) * decay_factor
    
    stopping_problem = OptimalStoppingProblem(asset_payoff, discount_rate=0.05)
    
    # Solve using finite horizon method
    result = stopping_problem.solve_finite_horizon(
        state_bounds=(80, 120),
        time_horizon=1.0,
        n_state=50,
        n_time=50
    )
    
    print(f"Optimal value: {result.optimal_value:.4f}")
    print(f"Continuation value: {result.continuation_value:.4f}")
    print(f"Exercise value: {result.exercise_value:.4f}")
    
    # Solve using Monte Carlo
    mc_result = stopping_problem.solve_monte_carlo(
        initial_state=100,
        time_horizon=1.0,
        n_paths=5000
    )
    
    print(f"Monte Carlo optimal value: {mc_result.optimal_value:.4f}")
    print(f"Average stopping time: {mc_result.stopping_time:.4f}")
    
    # Test barrier options
    print("\nTesting barrier options...")
    
    # Up-and-out call
    barrier_pricer = BarrierOptionPricer('call', 'up_and_out')
    B = 110  # Barrier level
    
    barrier_result = barrier_pricer.price_analytical(S0, K, B, T, r, sigma)
    print(f"Up-and-out call (barrier={B}): ${barrier_result['price']:.4f}")
    print(f"Vanilla call price: ${barrier_result['vanilla_price']:.4f}")
    print(f"Barrier adjustment: ${barrier_result['barrier_adjustment']:.4f}")
    
    # Test lookback options
    print("\nTesting lookback options...")
    
    # Floating strike lookback call
    lookback_pricer = LookbackOptionPricer('call', 'floating')
    S_min = 95  # Historical minimum
    
    lookback_result = lookback_pricer.price_analytical(S0, K, T, r, sigma, S_min=S_min)
    print(f"Floating strike lookback call: ${lookback_result['price']:.4f}")
    
    # Fixed strike lookback call
    fixed_lookback_pricer = LookbackOptionPricer('call', 'fixed')
    S_max = 108  # Historical maximum
    
    fixed_lookback_result = fixed_lookback_pricer.price_analytical(S0, K, T, r, sigma, S_max=S_max)
    print(f"Fixed strike lookback call: ${fixed_lookback_result['price']:.4f}")
    
    # Test comprehensive analyzer
    print("\nTesting comprehensive optimal stopping analyzer...")
    
    analyzer = OptimalStoppingAnalyzer()
    
    # Add multiple problems
    analyzer.add_problem('asset_sale', asset_payoff, discount_rate=0.05)
    
    # Problem 2: Investment timing
    def investment_payoff(state: float, time: float) -> float:
        """NPV of investment opportunity."""
        # Investment becomes less attractive over time
        time_factor = max(0, 1 - 0.5 * time)
        return max(0, (state - 100) * time_factor)
    
    analyzer.add_problem('investment_timing', investment_payoff, discount_rate=0.08)
    
    # Problem 3: Resource extraction
    def extraction_payoff(state: float, time: float) -> float:
        """Payoff from resource extraction."""
        # Resource price with extraction cost
        extraction_cost = 80
        return max(0, state - extraction_cost)
    
    analyzer.add_problem('resource_extraction', extraction_payoff, discount_rate=0.06)
    
    # Solve all problems
    solutions = analyzer.solve_all_problems(
        state_bounds=(70, 130),
        time_horizon=2.0,
        method='monte_carlo'
    )
    
    print("\nOptimal stopping solutions:")
    for name, solution in solutions.items():
        if solution:
            print(f"{name}:")
            print(f"  Optimal value: {solution.optimal_value:.4f}")
            print(f"  Expected stopping time: {solution.stopping_time:.4f}")
    
    # Compare solutions
    comparison_df = analyzer.compare_solutions()
    print("\nSolution comparison:")
    print(comparison_df.to_string(index=False))
    
    print("\nOptimal stopping theory engine test completed successfully!")