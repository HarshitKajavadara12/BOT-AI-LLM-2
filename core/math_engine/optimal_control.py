"""
Optimal Control Theory Engine for QUANTUM-FORGE
Implements advanced control theory for portfolio optimization and execution.
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.linalg import solve_continuous_are, expm
from scipy.integrate import solve_ivp
from numba import jit, prange
from typing import Tuple, Callable, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class LinearQuadraticRegulator:
    """Linear Quadratic Regulator for optimal control problems."""
    
    def __init__(self, A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray):
        """
        Initialize LQR controller.
        
        Args:
            A: State transition matrix
            B: Control input matrix  
            Q: State cost matrix
            R: Control cost matrix
        """
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        
        # Solve algebraic Riccati equation
        self.P = solve_continuous_are(A, B, Q, R)
        
        # Optimal gain matrix
        self.K = np.linalg.inv(R) @ B.T @ self.P
    
    def control_law(self, state: np.ndarray) -> np.ndarray:
        """
        Compute optimal control given current state.
        
        Args:
            state: Current state vector
        
        Returns:
            Optimal control vector
        """
        return -self.K @ state
    
    def simulate_closed_loop(self, x0: np.ndarray, T: float, dt: float = 0.01) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate closed-loop system.
        
        Args:
            x0: Initial state
            T: Time horizon
            dt: Time step
        
        Returns:
            Tuple of (time_points, states, controls)
        """
        t_span = (0, T)
        t_eval = np.arange(0, T + dt, dt)
        
        def closed_loop_dynamics(t, x):
            u = self.control_law(x)
            return (self.A - self.B @ self.K) @ x
        
        # Solve differential equation
        sol = solve_ivp(closed_loop_dynamics, t_span, x0, t_eval=t_eval, method='RK45')
        
        # Compute controls
        controls = np.array([self.control_law(x) for x in sol.y.T])
        
        return sol.t, sol.y.T, controls

class ModelPredictiveControl:
    """Model Predictive Control for constrained optimization."""
    
    def __init__(self, A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray, 
                 N: int, x_bounds: Optional[Tuple] = None, u_bounds: Optional[Tuple] = None):
        """
        Initialize MPC controller.
        
        Args:
            A: State transition matrix
            B: Control input matrix
            Q: State cost matrix
            R: Control cost matrix
            N: Prediction horizon
            x_bounds: State constraints (min, max)
            u_bounds: Control constraints (min, max)
        """
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.N = N
        self.x_bounds = x_bounds
        self.u_bounds = u_bounds
        
        self.n_states = A.shape[0]
        self.n_controls = B.shape[1]
    
    def mpc_cost(self, u_sequence: np.ndarray, x0: np.ndarray, 
                x_ref: np.ndarray) -> float:
        """
        Compute MPC cost function.
        
        Args:
            u_sequence: Control sequence (N * n_controls,)
            x0: Initial state
            x_ref: Reference trajectory
        
        Returns:
            Total cost
        """
        U = u_sequence.reshape(self.N, self.n_controls)
        
        # Forward simulation
        x = x0.copy()
        total_cost = 0.0
        
        for k in range(self.N):
            # State cost
            x_error = x - x_ref
            total_cost += x_error.T @ self.Q @ x_error
            
            # Control cost
            u = U[k]
            total_cost += u.T @ self.R @ u
            
            # Update state
            x = self.A @ x + self.B @ u
        
        # Terminal cost
        x_error = x - x_ref
        total_cost += x_error.T @ self.Q @ x_error
        
        return total_cost
    
    def solve_mpc(self, x0: np.ndarray, x_ref: np.ndarray) -> np.ndarray:
        """
        Solve MPC optimization problem.
        
        Args:
            x0: Current state
            x_ref: Reference state
        
        Returns:
            Optimal control sequence
        """
        # Initial guess
        u0 = np.zeros(self.N * self.n_controls)
        
        # Bounds
        bounds = []
        if self.u_bounds is not None:
            for _ in range(self.N):
                for j in range(self.n_controls):
                    bounds.append((self.u_bounds[0], self.u_bounds[1]))
        else:
            bounds = None
        
        # Solve optimization
        result = minimize(
            self.mpc_cost,
            u0,
            args=(x0, x_ref),
            method='SLSQP',
            bounds=bounds
        )
        
        return result.x.reshape(self.N, self.n_controls)

class HamiltonJacobiBellman:
    """Hamilton-Jacobi-Bellman equation solver for optimal control."""
    
    def __init__(self, state_bounds: Tuple[np.ndarray, np.ndarray], 
                 control_bounds: Tuple[np.ndarray, np.ndarray], 
                 discount_rate: float = 0.05):
        """
        Initialize HJB solver.
        
        Args:
            state_bounds: (min_state, max_state)
            control_bounds: (min_control, max_control)
            discount_rate: Discount rate for infinite horizon problems
        """
        self.state_bounds = state_bounds
        self.control_bounds = control_bounds
        self.discount_rate = discount_rate
    
    def value_iteration(self, dynamics_func: Callable, reward_func: Callable, 
                       state_grid: np.ndarray, control_grid: np.ndarray, 
                       dt: float = 0.01, max_iter: int = 1000, 
                       tolerance: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve HJB using value iteration.
        
        Args:
            dynamics_func: State dynamics function f(x, u)
            reward_func: Instantaneous reward function r(x, u)
            state_grid: Discretized state space
            control_grid: Discretized control space
            dt: Time step
            max_iter: Maximum iterations
            tolerance: Convergence tolerance
        
        Returns:
            Tuple of (value_function, optimal_policy)
        """
        # Initialize value function
        V = np.zeros(len(state_grid))
        V_new = np.zeros_like(V)
        policy = np.zeros(len(state_grid))
        
        for iteration in range(max_iter):
            for i, x in enumerate(state_grid):
                max_value = -np.inf
                best_control_idx = 0
                
                for j, u in enumerate(control_grid):
                    # Compute next state
                    x_next = x + dynamics_func(x, u) * dt
                    
                    # Interpolate value at next state
                    V_next = np.interp(x_next, state_grid, V)
                    
                    # Bellman equation
                    q_value = reward_func(x, u) * dt + np.exp(-self.discount_rate * dt) * V_next
                    
                    if q_value > max_value:
                        max_value = q_value
                        best_control_idx = j
                
                V_new[i] = max_value
                policy[i] = best_control_idx
            
            # Check convergence
            if np.max(np.abs(V_new - V)) < tolerance:
                print(f"Value iteration converged in {iteration + 1} iterations")
                break
            
            V = V_new.copy()
        
        return V, policy

class OptimalStopping:
    """Optimal stopping problems using dynamic programming."""
    
    @staticmethod
    def american_option_binomial(S0: float, K: float, r: float, sigma: float, 
                               T: float, N: int, option_type: str = 'call') -> Tuple[float, np.ndarray]:
        """
        American option pricing using binomial tree with optimal stopping.
        
        Args:
            S0: Initial stock price
            K: Strike price
            r: Risk-free rate
            sigma: Volatility
            T: Time to expiration
            N: Number of time steps
            option_type: 'call' or 'put'
        
        Returns:
            Tuple of (option_price, optimal_exercise_boundary)
        """
        dt = T / N
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(r * dt) - d) / (u - d)
        
        # Initialize price tree
        price_tree = np.zeros((N + 1, N + 1))
        for i in range(N + 1):
            for j in range(i + 1):
                price_tree[j, i] = S0 * (u ** j) * (d ** (i - j))
        
        # Initialize option value tree
        option_tree = np.zeros((N + 1, N + 1))
        exercise_boundary = np.zeros(N + 1)
        
        # Terminal condition
        for j in range(N + 1):
            if option_type == 'call':
                option_tree[j, N] = max(price_tree[j, N] - K, 0)
            else:  # put
                option_tree[j, N] = max(K - price_tree[j, N], 0)
        
        # Backward induction
        for i in range(N - 1, -1, -1):
            for j in range(i + 1):
                # Continuation value
                continuation = np.exp(-r * dt) * (p * option_tree[j + 1, i + 1] + 
                                                (1 - p) * option_tree[j, i + 1])
                
                # Exercise value
                if option_type == 'call':
                    exercise = max(price_tree[j, i] - K, 0)
                else:  # put
                    exercise = max(K - price_tree[j, i], 0)
                
                # Optimal decision
                option_tree[j, i] = max(continuation, exercise)
                
                # Record exercise boundary (for puts, find highest exercisable price)
                if option_type == 'put' and exercise > continuation and exercise > 0:
                    exercise_boundary[i] = max(exercise_boundary[i], price_tree[j, i])
        
        return option_tree[0, 0], exercise_boundary

class StochasticControl:
    """Stochastic optimal control using various numerical methods."""
    
    @staticmethod
    def merton_portfolio_problem(mu: float, sigma: float, r: float, gamma: float, 
                               T: float, W0: float = 1.0) -> Dict[str, Any]:
        """
        Solve Merton's portfolio optimization problem analytically.
        
        Args:
            mu: Expected return of risky asset
            sigma: Volatility of risky asset
            r: Risk-free rate
            gamma: Risk aversion parameter
            T: Investment horizon
            W0: Initial wealth
        
        Returns:
            Dictionary with optimal strategy and value function
        """
        # Market price of risk
        lambda_market = (mu - r) / sigma
        
        # Optimal portfolio weight in risky asset
        pi_optimal = lambda_market / (gamma * sigma)
        
        # Optimal drift and volatility of wealth
        mu_optimal = r + 0.5 * lambda_market**2 / gamma
        sigma_optimal = lambda_market / gamma
        
        # Value function
        if gamma != 1:
            A = (1 - gamma) * mu_optimal - 0.5 * gamma * (1 - gamma) * sigma_optimal**2
            value_function = lambda t, w: (w**(1 - gamma) * np.exp(A * (T - t))) / (1 - gamma)
        else:
            A = mu_optimal - 0.5 * sigma_optimal**2
            value_function = lambda t, w: np.log(w) + A * (T - t)
        
        return {
            'optimal_weight': pi_optimal,
            'optimal_drift': mu_optimal,
            'optimal_volatility': sigma_optimal,
            'value_function': value_function,
            'initial_value': value_function(0, W0)
        }
    
    @staticmethod
    def finite_difference_hjb(state_grid: np.ndarray, time_grid: np.ndarray,
                            dynamics_func: Callable, reward_func: Callable,
                            control_set: np.ndarray, 
                            terminal_condition: Callable) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve HJB PDE using finite difference methods.
        
        Args:
            state_grid: Spatial discretization
            time_grid: Time discretization
            dynamics_func: State dynamics μ(x,u) and σ²(x,u)
            reward_func: Instantaneous reward r(x,u)
            control_set: Set of admissible controls
            terminal_condition: Terminal payoff g(x)
        
        Returns:
            Tuple of (value_function_grid, optimal_control_grid)
        """
        nx = len(state_grid)
        nt = len(time_grid)
        dt = time_grid[1] - time_grid[0]
        dx = state_grid[1] - state_grid[0]
        
        # Initialize value function and policy
        V = np.zeros((nx, nt))
        policy = np.zeros((nx, nt))
        
        # Terminal condition
        for i in range(nx):
            V[i, -1] = terminal_condition(state_grid[i])
        
        # Backward time-stepping
        for t_idx in range(nt - 2, -1, -1):
            for x_idx in range(1, nx - 1):  # Interior points
                x = state_grid[x_idx]
                max_value = -np.inf
                best_control = control_set[0]
                
                for u in control_set:
                    # Get dynamics
                    mu, sigma_sq = dynamics_func(x, u)
                    
                    # Finite difference approximation
                    V_x = (V[x_idx + 1, t_idx + 1] - V[x_idx - 1, t_idx + 1]) / (2 * dx)
                    V_xx = (V[x_idx + 1, t_idx + 1] - 2 * V[x_idx, t_idx + 1] + V[x_idx - 1, t_idx + 1]) / (dx**2)
                    
                    # HJB equation
                    hjb_value = (reward_func(x, u) + mu * V_x + 0.5 * sigma_sq * V_xx)
                    
                    if hjb_value > max_value:
                        max_value = hjb_value
                        best_control = u
                
                V[x_idx, t_idx] = V[x_idx, t_idx + 1] + dt * max_value
                policy[x_idx, t_idx] = best_control
        
        return V, policy

class RobustControl:
    """Robust control for uncertain systems."""
    
    @staticmethod
    def h_infinity_control(A: np.ndarray, B1: np.ndarray, B2: np.ndarray,
                          C1: np.ndarray, D12: np.ndarray, 
                          gamma: float) -> Tuple[np.ndarray, bool]:
        """
        H-infinity control synthesis.
        
        Args:
            A: State matrix
            B1: Disturbance input matrix
            B2: Control input matrix
            C1: Performance output matrix
            D12: Direct feedthrough matrix
            gamma: Performance bound
        
        Returns:
            Tuple of (controller_gain, feasible)
        """
        n = A.shape[0]
        
        # Hamiltonian matrix for H-infinity control
        R_inv = np.linalg.inv(D12.T @ D12)
        H = np.block([
            [A - B2 @ R_inv @ D12.T @ C1, (1/gamma**2) * B1 @ B1.T - B2 @ R_inv @ B2.T],
            [-C1.T @ (np.eye(C1.shape[0]) - D12 @ R_inv @ D12.T) @ C1, -(A - B2 @ R_inv @ D12.T @ C1).T]
        ])
        
        # Solve Riccati equation
        try:
            eigenvals, eigenvecs = np.linalg.eig(H)
            
            # Select stable eigenvalues
            stable_idx = np.real(eigenvals) < 0
            
            if np.sum(stable_idx) != n:
                return None, False
            
            # Extract solution
            stable_eigenvecs = eigenvecs[:, stable_idx]
            X21 = stable_eigenvecs[n:2*n, :]
            X11 = stable_eigenvecs[:n, :]
            
            X = X21 @ np.linalg.inv(X11)
            
            # Controller gain
            K = -R_inv @ (D12.T @ C1 + B2.T @ X)
            
            return K, True
            
        except np.linalg.LinAlgError:
            return None, False

# Example usage and testing
if __name__ == "__main__":
    # Test LQR controller
    print("Testing Linear Quadratic Regulator...")
    
    # Simple double integrator system
    A = np.array([[0, 1], [0, 0]])
    B = np.array([[0], [1]])
    Q = np.array([[1, 0], [0, 1]])
    R = np.array([[1]])
    
    lqr = LinearQuadraticRegulator(A, B, Q, R)
    print(f"LQR gain matrix: {lqr.K}")
    
    # Test MPC
    print("\nTesting Model Predictive Control...")
    mpc = ModelPredictiveControl(A, B, Q, R, N=10, u_bounds=(-5, 5))
    
    x0 = np.array([1.0, 0.0])
    x_ref = np.array([0.0, 0.0])
    
    u_sequence = mpc.solve_mpc(x0, x_ref)
    print(f"MPC first control action: {u_sequence[0]}")
    
    # Test Merton portfolio problem
    print("\nTesting Merton Portfolio Problem...")
    merton_solution = StochasticControl.merton_portfolio_problem(
        mu=0.1, sigma=0.2, r=0.05, gamma=2.0, T=1.0
    )
    print(f"Optimal portfolio weight: {merton_solution['optimal_weight']:.4f}")
    print(f"Initial value function: {merton_solution['initial_value']:.4f}")
    
    # Test American option with optimal stopping
    print("\nTesting American Option Pricing...")
    option_price, exercise_boundary = OptimalStopping.american_option_binomial(
        S0=100, K=100, r=0.05, sigma=0.2, T=1.0, N=100, option_type='put'
    )
    print(f"American put option price: {option_price:.4f}")
    
    print("\nOptimal control engine test completed successfully!")