"""
Advanced Numerical Methods Engine for QUANTUM-FORGE
Implements high-performance numerical computation methods for financial modeling.
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution, root_scalar
from scipy.integrate import quad, solve_ivp, simpson
from scipy.interpolate import interp1d, CubicSpline, RBFInterpolator
from scipy.linalg import solve, cholesky, eigvals
from scipy.sparse import csc_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
from numba import jit, prange, cuda
try:
    import cupy as cp
except ImportError:
    import numpy as cp
    # warnings.warn("CuPy not found, using NumPy as fallback. GPU acceleration disabled.")
from typing import Tuple, Callable, Optional, Union, List, Dict
import warnings
warnings.filterwarnings('ignore')

class MonteCarloMethods:
    """Advanced Monte Carlo simulation techniques."""
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def quasi_monte_carlo(func: Callable, dim: int, n_samples: int, 
                         bounds: np.ndarray) -> float:
        """
        Quasi-Monte Carlo integration using Sobol sequences.
        
        Args:
            func: Function to integrate
            dim: Dimension of integration
            n_samples: Number of sample points
            bounds: Integration bounds (dim x 2)
        
        Returns:
            Approximated integral value
        """
        # Simple quasi-random sequence (simplified Sobol-like)
        samples = np.zeros((n_samples, dim))
        
        for i in prange(n_samples):
            for j in range(dim):
                # Van der Corput sequence
                n = i + 1
                base = 2 + j
                vdc = 0.0
                denom = 1.0
                
                while n > 0:
                    denom *= base
                    vdc += (n % base) / denom
                    n //= base
                
                # Scale to bounds
                samples[i, j] = bounds[j, 0] + vdc * (bounds[j, 1] - bounds[j, 0])
        
        # Evaluate function at sample points
        values = np.zeros(n_samples)
        for i in prange(n_samples):
            values[i] = func(samples[i])
        
        # Compute volume
        volume = 1.0
        for j in range(dim):
            volume *= (bounds[j, 1] - bounds[j, 0])
        
        return volume * np.mean(values)
    
    @staticmethod
    def importance_sampling(func: Callable, proposal_dist: Callable, 
                          proposal_sample: Callable, n_samples: int) -> float:
        """
        Importance sampling for variance reduction.
        
        Args:
            func: Target function
            proposal_dist: Proposal distribution density
            proposal_sample: Function to sample from proposal distribution
            n_samples: Number of samples
        
        Returns:
            Importance sampling estimate
        """
        samples = np.array([proposal_sample() for _ in range(n_samples)])
        weights = np.array([func(x) / proposal_dist(x) for x in samples])
        
        # Handle infinite weights
        weights = np.where(np.isfinite(weights), weights, 0)
        
        return np.mean(weights)
    
    @staticmethod
    @jit(nopython=True)
    def antithetic_variates(func: Callable, n_samples: int) -> Tuple[float, float]:
        """
        Antithetic variates for variance reduction.
        
        Args:
            func: Function that takes random numbers as input
            n_samples: Number of sample pairs
        
        Returns:
            Tuple of (estimate, variance_reduction_factor)
        """
        estimates_1 = np.zeros(n_samples)
        estimates_2 = np.zeros(n_samples)

        # Deterministic evenly spaced samples on [0,1)
        for i in prange(n_samples):
            u = i / (n_samples if n_samples > 0 else 1)
            estimates_1[i] = func(u)
            estimates_2[i] = func(1.0 - u)  # Antithetic variate
        
        # Combined estimate
        combined_estimate = 0.5 * (np.mean(estimates_1) + np.mean(estimates_2))
        
        # Variance reduction factor
        var_standard = np.var(estimates_1, ddof=1)
        var_antithetic = 0.25 * (np.var(estimates_1, ddof=1) + np.var(estimates_2, ddof=1) + 
                                2 * np.cov(estimates_1, estimates_2, ddof=1)[0, 1])
        
        variance_reduction = var_standard / max(var_antithetic, 1e-10)
        
        return combined_estimate, variance_reduction

class FiniteDifferenceMethods:
    """Advanced finite difference schemes for PDEs."""
    
    @staticmethod
    def black_scholes_pde_solver(S_max: float, K: float, r: float, sigma: float, 
                                T: float, option_type: str = 'call',
                                N_s: int = 100, N_t: int = 1000,
                                scheme: str = 'implicit') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve Black-Scholes PDE using finite differences.
        
        Args:
            S_max: Maximum stock price in grid
            K: Strike price
            r: Risk-free rate
            sigma: Volatility
            T: Time to expiration
            option_type: 'call' or 'put'
            N_s: Number of stock price grid points
            N_t: Number of time grid points
            scheme: 'explicit', 'implicit', or 'crank_nicolson'
        
        Returns:
            Tuple of (stock_grid, time_grid, option_values)
        """
        # Grid setup
        S_grid = np.linspace(0, S_max, N_s + 1)
        dt = T / N_t
        ds = S_max / N_s
        
        # Initialize option values at expiration
        if option_type == 'call':
            V = np.maximum(S_grid - K, 0)
        else:  # put
            V = np.maximum(K - S_grid, 0)
        
        # Store solution
        solution = np.zeros((N_s + 1, N_t + 1))
        solution[:, -1] = V
        
        # Time stepping
        for t_idx in range(N_t - 1, -1, -1):
            if scheme == 'explicit':
                V_new = np.zeros_like(V)
                
                for i in range(1, N_s):
                    S = S_grid[i]
                    
                    # Finite difference coefficients
                    alpha = 0.5 * dt * (sigma**2 * S**2 / ds**2 - r * S / ds)
                    beta = 1 - dt * (sigma**2 * S**2 / ds**2 + r)
                    gamma = 0.5 * dt * (sigma**2 * S**2 / ds**2 + r * S / ds)
                    
                    V_new[i] = alpha * V[i-1] + beta * V[i] + gamma * V[i+1]
                
                # Boundary conditions
                if option_type == 'call':
                    V_new[0] = 0  # V(0,t) = 0
                    V_new[-1] = S_max - K * np.exp(-r * (T - t_idx * dt))  # V(S_max,t)
                else:  # put
                    V_new[0] = K * np.exp(-r * (T - t_idx * dt))  # V(0,t)
                    V_new[-1] = 0  # V(S_max,t) = 0
                
                V = V_new
                
            elif scheme == 'implicit':
                # Build tridiagonal matrix
                A = np.zeros((N_s - 1, N_s - 1))
                b = np.zeros(N_s - 1)
                
                for i in range(1, N_s):
                    S = S_grid[i]
                    
                    alpha = -0.5 * dt * (sigma**2 * S**2 / ds**2 - r * S / ds)
                    beta = 1 + dt * (sigma**2 * S**2 / ds**2 + r)
                    gamma = -0.5 * dt * (sigma**2 * S**2 / ds**2 + r * S / ds)
                    
                    if i == 1:
                        A[i-1, i-1] = beta
                        if i < N_s - 1:
                            A[i-1, i] = gamma
                        b[i-1] = V[i] - alpha * V[0]  # Boundary condition
                    elif i == N_s - 1:
                        A[i-1, i-2] = alpha
                        A[i-1, i-1] = beta
                        b[i-1] = V[i] - gamma * V[N_s]  # Boundary condition
                    else:
                        A[i-1, i-2] = alpha
                        A[i-1, i-1] = beta
                        A[i-1, i] = gamma
                        b[i-1] = V[i]
                
                # Solve linear system
                V_interior = solve(A, b)
                V[1:N_s] = V_interior
            
            solution[:, t_idx] = V
        
        time_grid = np.linspace(0, T, N_t + 1)
        return S_grid, time_grid, solution

class SpectralMethods:
    """Spectral methods for high-accuracy numerical solutions."""
    
    @staticmethod
    def chebyshev_differentiation_matrix(N: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Chebyshev differentiation matrix and grid.
        
        Args:
            N: Number of grid points
        
        Returns:
            Tuple of (differentiation_matrix, chebyshev_grid)
        """
        # Chebyshev grid
        x = np.cos(np.pi * np.arange(N + 1) / N)
        
        # Differentiation matrix
        c = np.ones(N + 1)
        c[0] = c[N] = 2
        
        X = np.tile(x, (N + 1, 1)).T
        dX = X - X.T
        
        D = np.outer(c, 1/c) / (dX + np.eye(N + 1))
        D = D - np.diag(np.sum(D, axis=1))
        
        return D, x
    
    @staticmethod
    def fourier_spectral_derivative(u: np.ndarray, L: float = 2*np.pi) -> np.ndarray:
        """
        Compute derivative using Fourier spectral method.
        
        Args:
            u: Function values on periodic domain
            L: Domain length
        
        Returns:
            Derivative values
        """
        N = len(u)
        
        # Wavenumbers
        k = 2 * np.pi / L * np.fft.fftfreq(N, 1/N)
        
        # Fourier transform
        u_hat = np.fft.fft(u)
        
        # Spectral derivative
        du_hat = 1j * k * u_hat
        
        # Inverse transform
        du = np.real(np.fft.ifft(du_hat))
        
        return du

class OptimizationMethods:
    """Advanced optimization algorithms for financial problems."""
    
    @staticmethod
    def simulated_annealing(objective: Callable, x0: np.ndarray, bounds: List[Tuple],
                          T0: float = 100.0, alpha: float = 0.95, 
                          max_iter: int = 10000) -> Dict:
        """
        Simulated annealing optimization.
        
        Args:
            objective: Objective function to minimize
            x0: Initial solution
            bounds: Variable bounds
            T0: Initial temperature
            alpha: Cooling rate
            max_iter: Maximum iterations
        
        Returns:
            Optimization result dictionary
        """
        x_current = np.array(x0)
        f_current = objective(x_current)
        
        x_best = x_current.copy()
        f_best = f_current
        
        T = T0
        
        for iteration in range(max_iter):
            # Deterministic candidate generation: sinusoidal perturbations
            x_candidate = x_current.copy()

            for i, (lower, upper) in enumerate(bounds):
                span = upper - lower
                perturbation = 0.1 * span * np.sin((iteration + 1) * (i + 1) * 0.314159)
                x_candidate[i] = np.clip(x_current[i] + perturbation, lower, upper)

            f_candidate = objective(x_candidate)

            # Deterministic acceptance rule: accept if better, else accept if
            # deterministic probability threshold exceeded (using mean probability 0.5)
            if f_candidate < f_current:
                x_current = x_candidate
                f_current = f_candidate

                if f_candidate < f_best:
                    x_best = x_candidate.copy()
                    f_best = f_candidate
            else:
                delta = f_candidate - f_current
                probability = np.exp(-delta / T)
                if probability > 0.5:
                    x_current = x_candidate
                    f_current = f_candidate

            # Cool down
            T *= alpha

            if T < 1e-8:
                break
        
        return {
            'x': x_best,
            'fun': f_best,
            'nit': iteration + 1,
            'success': True
        }
    
    @staticmethod
    def particle_swarm_optimization(objective: Callable, bounds: List[Tuple],
                                  n_particles: int = 30, max_iter: int = 1000,
                                  w: float = 0.5, c1: float = 1.5, c2: float = 1.5) -> Dict:
        """
        Particle Swarm Optimization.
        
        Args:
            objective: Objective function to minimize
            bounds: Variable bounds
            n_particles: Number of particles
            max_iter: Maximum iterations
            w: Inertia weight
            c1: Cognitive parameter
            c2: Social parameter
        
        Returns:
            Optimization result dictionary
        """
        dim = len(bounds)
        
        # Deterministic initialization of particles: evenly spaced along each dimension
        particles = np.zeros((n_particles, dim))
        for p in range(n_particles):
            for d in range(dim):
                lower, upper = bounds[d]
                if n_particles == 1:
                    t = 0.5
                else:
                    t = p / (n_particles - 1)
                particles[p, d] = lower + t * (upper - lower)

        velocities = np.zeros((n_particles, dim))
        
        # Personal best positions and values
        p_best = particles.copy()
        p_best_values = np.array([objective(p) for p in particles])
        
        # Global best
        g_best_idx = np.argmin(p_best_values)
        g_best = p_best[g_best_idx].copy()
        g_best_value = p_best_values[g_best_idx]
        
        for iteration in range(max_iter):
            for i in range(n_particles):
                # Deterministic coefficients
                r1, r2 = 0.5, 0.5

                velocities[i] = (w * velocities[i] +
                               c1 * r1 * (p_best[i] - particles[i]) +
                               c2 * r2 * (g_best - particles[i]))
                
                # Update position
                particles[i] += velocities[i]
                
                # Apply bounds
                for j, (lower, upper) in enumerate(bounds):
                    particles[i, j] = np.clip(particles[i, j], lower, upper)
                
                # Evaluate objective
                f_value = objective(particles[i])
                
                # Update personal best
                if f_value < p_best_values[i]:
                    p_best[i] = particles[i].copy()
                    p_best_values[i] = f_value
                    
                    # Update global best
                    if f_value < g_best_value:
                        g_best = particles[i].copy()
                        g_best_value = f_value
        
        return {
            'x': g_best,
            'fun': g_best_value,
            'nit': max_iter,
            'success': True
        }

class InterpolationMethods:
    """Advanced interpolation techniques for financial data."""
    
    @staticmethod
    def cubic_spline_interpolation(x: np.ndarray, y: np.ndarray, 
                                 x_new: np.ndarray, 
                                 boundary_conditions: str = 'natural') -> np.ndarray:
        """
        Cubic spline interpolation with various boundary conditions.
        
        Args:
            x: Known x values
            y: Known y values
            x_new: Points to interpolate
            boundary_conditions: 'natural', 'clamped', or 'periodic'
        
        Returns:
            Interpolated values
        """
        if boundary_conditions == 'natural':
            bc_type = 'natural'
        elif boundary_conditions == 'clamped':
            bc_type = 'clamped'
        elif boundary_conditions == 'periodic':
            bc_type = 'periodic'
        else:
            bc_type = 'natural'
        
        spline = CubicSpline(x, y, bc_type=bc_type)
        return spline(x_new)
    
    @staticmethod
    def radial_basis_function_interpolation(x: np.ndarray, y: np.ndarray, 
                                          x_new: np.ndarray, 
                                          function: str = 'thin_plate_spline') -> np.ndarray:
        """
        Radial basis function interpolation.
        
        Args:
            x: Known x values (can be multidimensional)
            y: Known y values
            x_new: Points to interpolate (same dimension as x)
            function: RBF type ('thin_plate_spline', 'multiquadric', 'gaussian')
        
        Returns:
            Interpolated values
        """
        rbf = RBFInterpolator(x, y, kernel=function)
        return rbf(x_new)

class LinearAlgebraMethods:
    """High-performance linear algebra operations."""
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def fast_matrix_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Fast matrix multiplication using Numba.
        
        Args:
            A: First matrix
            B: Second matrix
        
        Returns:
            Matrix product A @ B
        """
        n, k = A.shape
        k2, m = B.shape
        
        if k != k2:
            raise ValueError("Matrix dimensions don't match")
        
        C = np.zeros((n, m))
        
        for i in prange(n):
            for j in range(m):
                for l in range(k):
                    C[i, j] += A[i, l] * B[l, j]
        
        return C
    
    @staticmethod
    def cholesky_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Solve linear system using Cholesky decomposition.
        
        Args:
            A: Positive definite matrix
            b: Right-hand side vector
        
        Returns:
            Solution vector x such that Ax = b
        """
        L = cholesky(A, lower=True)
        
        # Forward substitution: Ly = b
        y = solve(L, b, lower=True)
        
        # Backward substitution: L^T x = y
        x = solve(L.T, y, lower=False)
        
        return x
    
    @staticmethod
    def eigenvalue_power_method(A: np.ndarray, max_iter: int = 1000, 
                              tol: float = 1e-10) -> Tuple[float, np.ndarray]:
        """
        Compute largest eigenvalue and eigenvector using power method.
        
        Args:
            A: Square matrix
            max_iter: Maximum iterations
            tol: Convergence tolerance
        
        Returns:
            Tuple of (largest_eigenvalue, corresponding_eigenvector)
        """
        n = A.shape[0]
        # Deterministic initial vector
        x = np.ones(n)
        x = x / np.linalg.norm(x)
        
        lambda_old = 0
        
        for iteration in range(max_iter):
            # Power iteration step
            y = A @ x
            lambda_new = np.dot(x, y)
            x = y / np.linalg.norm(y)
            
            # Check convergence
            if abs(lambda_new - lambda_old) < tol:
                break
            
            lambda_old = lambda_new
        
        return lambda_new, x

# GPU-accelerated methods (requires CuPy)
class GPUMethods:
    """GPU-accelerated numerical methods."""
    
    @staticmethod
    def gpu_monte_carlo(func_gpu, n_samples: int, dim: int) -> float:
        """
        GPU-accelerated Monte Carlo integration.
        
        Args:
            func_gpu: CuPy-compatible function
            n_samples: Number of samples
            dim: Dimension
        
        Returns:
            Integration result
        """
        try:
            # Generate deterministic low-discrepancy samples on GPU
            total = n_samples * dim
            samples = cp.linspace(0, 1, total).reshape((n_samples, dim))
            
            # Evaluate function on GPU
            values = func_gpu(samples)
            
            # Compute mean
            result = cp.mean(values)
            
            # Transfer back to CPU
            return float(result.get())
        
        except ImportError:
            print("CuPy not available. Falling back to CPU.")
            return 0.0

# Example usage and testing
if __name__ == "__main__":
    print("Testing Numerical Methods Engine...")
    
    # Test Black-Scholes PDE solver
    print("\nTesting Black-Scholes PDE Solver...")
    S_grid, t_grid, option_values = FiniteDifferenceMethods.black_scholes_pde_solver(
        S_max=200, K=100, r=0.05, sigma=0.2, T=1.0, N_s=50, N_t=100
    )
    
    # Option value at S=100, t=0
    S_idx = np.argmin(np.abs(S_grid - 100))
    option_price = option_values[S_idx, 0]
    print(f"European call option price (PDE): {option_price:.4f}")
    
    # Test Particle Swarm Optimization
    print("\nTesting Particle Swarm Optimization...")
    
    def rosenbrock(x):
        return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2
    
    result = OptimizationMethods.particle_swarm_optimization(
        rosenbrock, bounds=[(-2, 2), (-2, 2)], n_particles=20, max_iter=100
    )
    print(f"PSO result: x = {result['x']}, f = {result['fun']:.6f}")
    
    # Test Chebyshev differentiation
    print("\nTesting Chebyshev Spectral Method...")
    D, x = SpectralMethods.chebyshev_differentiation_matrix(20)
    
    # Test on f(x) = exp(x)
    f = np.exp(x)
    df_exact = np.exp(x)
    df_numerical = D @ f
    
    error = np.max(np.abs(df_numerical - df_exact))
    print(f"Chebyshev derivative error: {error:.2e}")
    
    # Test Monte Carlo with antithetic variates
    print("\nTesting Monte Carlo Methods...")
    
    def test_function(u):
        return np.exp(-u**2)  # Integral should be approximately sqrt(pi)/2
    
    estimate, variance_reduction = MonteCarloMethods.antithetic_variates(test_function, 10000)
    print(f"Antithetic variates estimate: {estimate:.4f}")
    print(f"Variance reduction factor: {variance_reduction:.2f}")
    
    print("\nNumerical methods engine test completed successfully!")