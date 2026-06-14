"""
Mathematical Engine Core
Advanced mathematical operations and numerical methods for financial computations
"""

import numpy as np
import pandas as pd
import scipy.optimize as optimize
import scipy.integrate as integrate
import scipy.interpolate as interpolate
import scipy.special as special
import scipy.linalg as linalg
from typing import Optional, Tuple, Dict, Any, List, Union, Callable
from abc import ABC, abstractmethod
import numba
from numba import jit, njit
import warnings


class MathematicalEngine:
    """
    Core mathematical engine for financial computations
    """
    
    def __init__(self):
        self.precision = np.float64
        self.tolerance = 1e-12
        self.max_iterations = 10000
        
    @staticmethod
    @njit
    def fast_dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Fast dot product using Numba JIT compilation"""
        return np.dot(a, b)
    
    @staticmethod
    @njit
    def fast_sum(array: np.ndarray) -> float:
        """Fast sum using Numba JIT compilation"""
        return np.sum(array)
    
    @staticmethod
    @njit
    def fast_mean(array: np.ndarray) -> float:
        """Fast mean calculation using Numba JIT compilation"""
        return np.mean(array)
    
    @staticmethod
    @njit
    def fast_std(array: np.ndarray) -> float:
        """Fast standard deviation using Numba JIT compilation"""
        return np.std(array)
    
    def matrix_inverse(self, matrix: np.ndarray, method: str = 'lu') -> np.ndarray:
        """
        Compute matrix inverse using various methods
        
        Args:
            matrix: Input matrix
            method: Inversion method ('lu', 'cholesky', 'svd', 'moore_penrose')
        
        Returns:
            Inverse matrix
        """
        
        if method == 'lu':
            try:
                return linalg.inv(matrix)
            except linalg.LinAlgError:
                return linalg.pinv(matrix)
        
        elif method == 'cholesky':
            try:
                L = linalg.cholesky(matrix, lower=True)
                return linalg.cho_solve((L, True), np.eye(matrix.shape[0]))
            except linalg.LinAlgError:
                return linalg.pinv(matrix)
        
        elif method == 'svd':
            U, s, Vt = linalg.svd(matrix)
            # Filter out small singular values
            s_inv = np.where(s > self.tolerance, 1.0 / s, 0.0)
            return Vt.T @ np.diag(s_inv) @ U.T
        
        elif method == 'moore_penrose':
            return linalg.pinv(matrix)
        
        else:
            raise ValueError(f"Unknown inversion method: {method}")
    
    def solve_linear_system(
        self,
        A: np.ndarray,
        b: np.ndarray,
        method: str = 'lu'
    ) -> np.ndarray:
        """
        Solve linear system Ax = b
        
        Args:
            A: Coefficient matrix
            b: Right-hand side vector
            method: Solution method ('lu', 'cholesky', 'qr', 'svd')
        
        Returns:
            Solution vector x
        """
        
        if method == 'lu':
            return linalg.solve(A, b)
        
        elif method == 'cholesky':
            L = linalg.cholesky(A, lower=True)
            return linalg.solve_triangular(
                L.T,
                linalg.solve_triangular(L, b, lower=True),
                lower=False
            )
        
        elif method == 'qr':
            Q, R = linalg.qr(A)
            return linalg.solve_triangular(R, Q.T @ b, lower=False)
        
        elif method == 'svd':
            U, s, Vt = linalg.svd(A, full_matrices=False)
            s_inv = np.where(s > self.tolerance, 1.0 / s, 0.0)
            return Vt.T @ (s_inv * (U.T @ b))
        
        else:
            raise ValueError(f"Unknown solution method: {method}")
    
    def eigenvalue_decomposition(
        self,
        matrix: np.ndarray,
        eigvals_only: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Compute eigenvalue decomposition
        
        Args:
            matrix: Input matrix
            eigvals_only: If True, return only eigenvalues
        
        Returns:
            Eigenvalues or (eigenvalues, eigenvectors)
        """
        
        if eigvals_only:
            return linalg.eigvals(matrix)
        else:
            eigenvalues, eigenvectors = linalg.eig(matrix)
            
            # Sort by eigenvalue magnitude (descending)
            idx = np.argsort(np.abs(eigenvalues))[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            return eigenvalues, eigenvectors
    
    def singular_value_decomposition(
        self,
        matrix: np.ndarray,
        full_matrices: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute singular value decomposition
        
        Args:
            matrix: Input matrix
            full_matrices: Whether to compute full or reduced SVD
        
        Returns:
            U, s, Vt matrices
        """
        return linalg.svd(matrix, full_matrices=full_matrices)
    
    def condition_number(self, matrix: np.ndarray, p=None) -> float:
        """
        Compute condition number of a matrix
        
        Args:
            matrix: Input matrix
            p: Order of the norm (None, 1, -1, 2, -2, 'fro')
        
        Returns:
            Condition number
        """
        return linalg.norm(matrix, ord=p) * linalg.norm(self.matrix_inverse(matrix), ord=p)
    
    def matrix_exponential(self, matrix: np.ndarray) -> np.ndarray:
        """Compute matrix exponential using Padé approximation"""
        return linalg.expm(matrix)
    
    def matrix_logarithm(self, matrix: np.ndarray) -> np.ndarray:
        """Compute matrix logarithm"""
        return linalg.logm(matrix)
    
    def matrix_square_root(self, matrix: np.ndarray) -> np.ndarray:
        """Compute matrix square root"""
        return linalg.sqrtm(matrix)


class NumericalIntegration:
    """
    Numerical integration methods for financial applications
    """
    
    def __init__(self, tolerance: float = 1e-8, max_subdivisions: int = 50):
        self.tolerance = tolerance
        self.max_subdivisions = max_subdivisions
    
    def integrate_1d(
        self,
        func: Callable[[float], float],
        a: float,
        b: float,
        method: str = 'quad'
    ) -> Tuple[float, float]:
        """
        1D numerical integration
        
        Args:
            func: Function to integrate
            a: Lower bound
            b: Upper bound
            method: Integration method ('quad', 'simpson', 'romberg')
        
        Returns:
            (integral_value, error_estimate)
        """
        
        if method == 'quad':
            return integrate.quad(func, a, b, epsabs=self.tolerance)
        
        elif method == 'simpson':
            # Use scipy's simpson rule with adaptive subdivision
            n_points = 1001  # Odd number for Simpson's rule
            x = np.linspace(a, b, n_points)
            y = np.array([func(xi) for xi in x])
            result = integrate.simpson(y, x)
            return result, self.tolerance  # Approximate error
        
        elif method == 'romberg':
            return integrate.romberg(func, a, b, tol=self.tolerance)
        
        else:
            raise ValueError(f"Unknown integration method: {method}")
    
    def integrate_2d(
        self,
        func: Callable[[float, float], float],
        bounds: List[Tuple[float, float]]
    ) -> Tuple[float, float]:
        """
        2D numerical integration
        
        Args:
            func: Function to integrate f(x, y)
            bounds: [(x_min, x_max), (y_min, y_max)]
        
        Returns:
            (integral_value, error_estimate)
        """
        
        (x_min, x_max), (y_min, y_max) = bounds
        
        return integrate.dblquad(
            func,
            x_min, x_max,
            lambda x: y_min, lambda x: y_max,
            epsabs=self.tolerance
        )
    
    def monte_carlo_integration(
        self,
        func: Callable[..., float],
        bounds: List[Tuple[float, float]],
        n_samples: int = 100000
    ) -> Tuple[float, float]:
        """
        Monte Carlo integration for high-dimensional integrals
        
        Args:
            func: Function to integrate
            bounds: List of (min, max) bounds for each dimension
            n_samples: Number of Monte Carlo samples
        
        Returns:
            (integral_estimate, standard_error)
        """
        
        n_dims = len(bounds)
        
        # Generate random samples
        samples = np.random.uniform(size=(n_samples, n_dims))
        
        # Scale to bounds
        for i, (min_val, max_val) in enumerate(bounds):
            samples[:, i] = min_val + samples[:, i] * (max_val - min_val)
        
        # Evaluate function at sample points
        values = np.array([func(*sample) for sample in samples])
        
        # Compute integral estimate
        volume = np.prod([max_val - min_val for min_val, max_val in bounds])
        integral = volume * np.mean(values)
        
        # Standard error
        standard_error = volume * np.std(values) / np.sqrt(n_samples)
        
        return integral, standard_error


class NumericalOptimization:
    """
    Numerical optimization methods for financial problems
    """
    
    def __init__(self, tolerance: float = 1e-8, max_iterations: int = 1000):
        self.tolerance = tolerance
        self.max_iterations = max_iterations
    
    def minimize_scalar(
        self,
        func: Callable[[float], float],
        bounds: Tuple[float, float],
        method: str = 'golden'
    ) -> optimize.OptimizeResult:
        """
        Scalar function minimization
        
        Args:
            func: Objective function
            bounds: (min, max) bounds
            method: Optimization method ('golden', 'brent', 'bounded')
        
        Returns:
            Optimization result
        """
        
        if method == 'golden':
            return optimize.minimize_scalar(func, bounds=bounds, method='Golden')
        elif method == 'brent':
            return optimize.minimize_scalar(func, bounds=bounds, method='Brent')
        elif method == 'bounded':
            return optimize.minimize_scalar(func, bounds=bounds, method='Bounded')
        else:
            raise ValueError(f"Unknown scalar optimization method: {method}")
    
    def minimize_multivariate(
        self,
        func: Callable[[np.ndarray], float],
        x0: np.ndarray,
        bounds: Optional[List[Tuple[float, float]]] = None,
        constraints: Optional[List[Dict]] = None,
        method: str = 'L-BFGS-B',
        jac: Optional[Callable[[np.ndarray], np.ndarray]] = None
    ) -> optimize.OptimizeResult:
        """
        Multivariate function minimization
        
        Args:
            func: Objective function
            x0: Initial guess
            bounds: Variable bounds
            constraints: Optimization constraints
            method: Optimization method
            jac: Jacobian function (optional)
        
        Returns:
            Optimization result
        """
        
        options = {
            'maxiter': self.max_iterations,
            'ftol': self.tolerance
        }
        
        return optimize.minimize(
            func, x0,
            method=method,
            bounds=bounds,
            constraints=constraints,
            jac=jac,
            options=options
        )
    
    def least_squares(
        self,
        residual_func: Callable[[np.ndarray], np.ndarray],
        x0: np.ndarray,
        bounds: Tuple[np.ndarray, np.ndarray] = (-np.inf, np.inf),
        jac: Optional[Callable[[np.ndarray], np.ndarray]] = None
    ) -> optimize.OptimizeResult:
        """
        Non-linear least squares optimization
        
        Args:
            residual_func: Residual function
            x0: Initial guess
            bounds: Parameter bounds (lower, upper)
            jac: Jacobian of residual function
        
        Returns:
            Optimization result
        """
        
        return optimize.least_squares(
            residual_func, x0,
            bounds=bounds,
            jac=jac,
            ftol=self.tolerance,
            max_nfev=self.max_iterations
        )
    
    def root_finding(
        self,
        func: Callable[[float], float],
        x0: float,
        method: str = 'brentq',
        bracket: Optional[Tuple[float, float]] = None
    ) -> float:
        """
        Root finding for scalar functions
        
        Args:
            func: Function to find root of
            x0: Initial guess
            method: Root finding method ('brentq', 'newton', 'secant')
            bracket: Bracketing interval for some methods
        
        Returns:
            Root value
        """
        
        if method == 'brentq' and bracket is not None:
            return optimize.brentq(func, bracket[0], bracket[1], xtol=self.tolerance)
        elif method == 'newton':
            return optimize.newton(func, x0, tol=self.tolerance, maxiter=self.max_iterations)
        elif method == 'secant':
            return optimize.newton(func, x0, tol=self.tolerance, maxiter=self.max_iterations)
        else:
            # Fallback to fsolve
            result = optimize.fsolve(func, x0, xtol=self.tolerance)
            return result[0]


class SpecialFunctions:
    """
    Special mathematical functions used in finance
    """
    
    @staticmethod
    def normal_cdf(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Standard normal cumulative distribution function"""
        return 0.5 * (1 + special.erf(x / np.sqrt(2)))
    
    @staticmethod
    def normal_pdf(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Standard normal probability density function"""
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    
    @staticmethod
    def inverse_normal_cdf(p: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Inverse of standard normal CDF (quantile function)"""
        return special.ndtri(p)
    
    @staticmethod
    def gamma_function(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Gamma function"""
        return special.gamma(x)
    
    @staticmethod
    def log_gamma_function(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Logarithm of gamma function"""
        return special.loggamma(x)
    
    @staticmethod
    def beta_function(a: float, b: float) -> float:
        """Beta function"""
        return special.beta(a, b)
    
    @staticmethod
    def incomplete_gamma(a: float, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Lower incomplete gamma function"""
        return special.gammainc(a, x) * special.gamma(a)
    
    @staticmethod
    def bessel_j(n: int, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Bessel function of the first kind"""
        return special.jv(n, x)
    
    @staticmethod
    def bessel_y(n: int, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Bessel function of the second kind"""
        return special.yv(n, x)
    
    @staticmethod
    def modified_bessel_i(n: int, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Modified Bessel function of the first kind"""
        return special.iv(n, x)
    
    @staticmethod
    def modified_bessel_k(n: int, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Modified Bessel function of the second kind"""
        return special.kv(n, x)
    
    @staticmethod
    def hypergeometric_1f1(a: float, b: float, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Confluent hypergeometric function 1F1"""
        return special.hyp1f1(a, b, x)
    
    @staticmethod
    def hypergeometric_2f1(a: float, b: float, c: float, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Gauss hypergeometric function 2F1"""
        return special.hyp2f1(a, b, c, x)


class InterpolationMethods:
    """
    Interpolation methods for financial data
    """
    
    def __init__(self):
        self.tolerance = 1e-12
    
    def linear_interpolation(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_new: np.ndarray,
        extrapolate: bool = False
    ) -> np.ndarray:
        """Linear interpolation"""
        
        fill_value = 'extrapolate' if extrapolate else np.nan
        interp_func = interpolate.interp1d(x, y, kind='linear', 
                                          bounds_error=False, 
                                          fill_value=fill_value)
        return interp_func(x_new)
    
    def cubic_spline(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_new: np.ndarray,
        bc_type: str = 'natural'
    ) -> np.ndarray:
        """Cubic spline interpolation"""
        
        cs = interpolate.CubicSpline(x, y, bc_type=bc_type)
        return cs(x_new)
    
    def hermite_spline(
        self,
        x: np.ndarray,
        y: np.ndarray,
        dy: np.ndarray,
        x_new: np.ndarray
    ) -> np.ndarray:
        """Hermite spline interpolation with derivatives"""
        
        hs = interpolate.PchipInterpolator(x, y, dy)
        return hs(x_new)
    
    def rational_interpolation(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_new: np.ndarray,
        degree: int = 3
    ) -> np.ndarray:
        """Rational function interpolation"""
        
        # Use barycentric interpolation as approximation
        bi = interpolate.BarycentricInterpolator(x, y)
        return bi(x_new)
    
    def radial_basis_function(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_new: np.ndarray,
        function: str = 'multiquadric',
        epsilon: Optional[float] = None
    ) -> np.ndarray:
        """Radial basis function interpolation"""
        
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if x_new.ndim == 1:
            x_new = x_new.reshape(-1, 1)
        
        rbf = interpolate.Rbf(*x.T, y, function=function, epsilon=epsilon)
        return rbf(*x_new.T)
    
    def multidimensional_interpolation(
        self,
        points: np.ndarray,
        values: np.ndarray,
        xi: np.ndarray,
        method: str = 'linear'
    ) -> np.ndarray:
        """N-dimensional interpolation"""
        
        return interpolate.griddata(points, values, xi, method=method)


class FastFourierTransform:
    """
    Fast Fourier Transform operations for financial signal processing
    """
    
    def __init__(self):
        pass
    
    def fft(self, signal: np.ndarray) -> np.ndarray:
        """Forward FFT"""
        return np.fft.fft(signal)
    
    def ifft(self, spectrum: np.ndarray) -> np.ndarray:
        """Inverse FFT"""
        return np.fft.ifft(spectrum)
    
    def rfft(self, signal: np.ndarray) -> np.ndarray:
        """Real FFT (for real-valued signals)"""
        return np.fft.rfft(signal)
    
    def irfft(self, spectrum: np.ndarray) -> np.ndarray:
        """Inverse real FFT"""
        return np.fft.irfft(spectrum)
    
    def fft2(self, signal: np.ndarray) -> np.ndarray:
        """2D FFT"""
        return np.fft.fft2(signal)
    
    def ifft2(self, spectrum: np.ndarray) -> np.ndarray:
        """2D inverse FFT"""
        return np.fft.ifft2(spectrum)
    
    def fftfreq(self, n: int, d: float = 1.0) -> np.ndarray:
        """FFT frequency bins"""
        return np.fft.fftfreq(n, d)
    
    def power_spectrum(self, signal: np.ndarray) -> np.ndarray:
        """Power spectrum density"""
        spectrum = self.fft(signal)
        return np.abs(spectrum)**2
    
    def cross_spectrum(self, signal1: np.ndarray, signal2: np.ndarray) -> np.ndarray:
        """Cross power spectrum"""
        spectrum1 = self.fft(signal1)
        spectrum2 = self.fft(signal2)
        return spectrum1 * np.conj(spectrum2)
    
    def coherence(self, signal1: np.ndarray, signal2: np.ndarray) -> np.ndarray:
        """Coherence between two signals"""
        
        cross_spec = self.cross_spectrum(signal1, signal2)
        power1 = self.power_spectrum(signal1)
        power2 = self.power_spectrum(signal2)
        
        return np.abs(cross_spec)**2 / (power1 * power2)
    
    def convolution_fft(self, signal1: np.ndarray, signal2: np.ndarray) -> np.ndarray:
        """Convolution using FFT"""
        
        # Pad signals to avoid circular convolution
        n = len(signal1) + len(signal2) - 1
        padded_size = 2**int(np.ceil(np.log2(n)))  # Next power of 2
        
        fft1 = np.fft.fft(signal1, padded_size)
        fft2 = np.fft.fft(signal2, padded_size)
        
        conv_fft = fft1 * fft2
        conv_result = np.fft.ifft(conv_fft).real
        
        return conv_result[:n]
    
    def correlation_fft(self, signal1: np.ndarray, signal2: np.ndarray) -> np.ndarray:
        """Cross-correlation using FFT"""
        
        n = len(signal1) + len(signal2) - 1
        padded_size = 2**int(np.ceil(np.log2(n)))
        
        fft1 = np.fft.fft(signal1, padded_size)
        fft2 = np.fft.fft(signal2, padded_size)
        
        corr_fft = fft1 * np.conj(fft2)
        corr_result = np.fft.ifft(corr_fft).real
        
        return corr_result[:n]


if __name__ == "__main__":
    # Example usage and testing
    print("Testing Mathematical Engine...")
    
    # Test basic mathematical operations
    math_engine = MathematicalEngine()
    
    # Test matrix operations
    print("\nTesting matrix operations...")
    A = np.random.rand(5, 5)
    A = A @ A.T  # Make positive definite
    
    try:
        A_inv = math_engine.matrix_inverse(A, method='lu')
        print(f"Matrix inversion successful: {np.allclose(A @ A_inv, np.eye(5))}")
        
        eigenvals, eigenvecs = math_engine.eigenvalue_decomposition(A)
        print(f"Eigenvalue decomposition: {len(eigenvals)} eigenvalues computed")
        
        U, s, Vt = math_engine.singular_value_decomposition(A)
        print(f"SVD successful: U shape {U.shape}, s shape {s.shape}, Vt shape {Vt.shape}")
        
    except Exception as e:
        print(f"Matrix operations error: {e}")
    
    # Test numerical integration
    print("\nTesting numerical integration...")
    integrator = NumericalIntegration()
    
    try:
        # Simple polynomial integration
        def poly_func(x):
            return x**2 + 2*x + 1
        
        result, error = integrator.integrate_1d(poly_func, 0, 1)
        exact = 1/3 + 1 + 1  # Exact integral
        print(f"1D integration: computed={result:.6f}, exact={exact:.6f}, error={abs(result-exact):.2e}")
        
        # 2D integration
        def func_2d(x, y):
            return x**2 + y**2
        
        result_2d, error_2d = integrator.integrate_2d(func_2d, [(0, 1), (0, 1)])
        exact_2d = 2/3  # Exact integral
        print(f"2D integration: computed={result_2d:.6f}, exact={exact_2d:.6f}")
        
    except Exception as e:
        print(f"Integration error: {e}")
    
    # Test optimization
    print("\nTesting optimization...")
    optimizer = NumericalOptimization()
    
    try:
        # Scalar optimization
        def quadratic(x):
            return (x - 2)**2 + 1
        
        result = optimizer.minimize_scalar(quadratic, bounds=(0, 5))
        print(f"Scalar optimization: minimum at x={result.x:.6f}, f(x)={result.fun:.6f}")
        
        # Multivariate optimization (Rosenbrock function)
        def rosenbrock(x):
            return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2
        
        result_multi = optimizer.minimize_multivariate(rosenbrock, np.array([0, 0]))
        print(f"Multivariate optimization converged: {result_multi.success}")
        print(f"Solution: {result_multi.x}, Function value: {result_multi.fun:.2e}")
        
    except Exception as e:
        print(f"Optimization error: {e}")
    
    # Test special functions
    print("\nTesting special functions...")
    
    try:
        x_vals = np.array([-2, -1, 0, 1, 2])
        
        normal_cdf_vals = SpecialFunctions.normal_cdf(x_vals)
        normal_pdf_vals = SpecialFunctions.normal_pdf(x_vals)
        
        print(f"Normal CDF at {x_vals}: {normal_cdf_vals}")
        print(f"Normal PDF at {x_vals}: {normal_pdf_vals}")
        
        # Test inverse
        p_vals = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        inv_normal = SpecialFunctions.inverse_normal_cdf(p_vals)
        print(f"Inverse normal CDF at {p_vals}: {inv_normal}")
        
    except Exception as e:
        print(f"Special functions error: {e}")
    
    # Test interpolation
    print("\nTesting interpolation...")
    interp = InterpolationMethods()
    
    try:
        # Sample data
        x_data = np.array([0, 1, 2, 3, 4, 5])
        y_data = np.sin(x_data)
        x_new = np.linspace(0, 5, 20)
        
        # Linear interpolation
        y_linear = interp.linear_interpolation(x_data, y_data, x_new)
        print(f"Linear interpolation: {len(y_linear)} points interpolated")
        
        # Cubic spline
        y_cubic = interp.cubic_spline(x_data, y_data, x_new)
        print(f"Cubic spline interpolation: {len(y_cubic)} points interpolated")
        
    except Exception as e:
        print(f"Interpolation error: {e}")
    
    # Test FFT
    print("\nTesting FFT operations...")
    fft_engine = FastFourierTransform()
    
    try:
        # Generate test signal
        t = np.linspace(0, 1, 1000, endpoint=False)
        signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)
        
        # Forward FFT
        spectrum = fft_engine.fft(signal)
        
        # Inverse FFT
        reconstructed = fft_engine.ifft(spectrum)
        
        # Check reconstruction accuracy
        reconstruction_error = np.max(np.abs(signal - reconstructed.real))
        print(f"FFT reconstruction error: {reconstruction_error:.2e}")
        
        # Power spectrum
        power = fft_engine.power_spectrum(signal)
        print(f"Power spectrum computed: {len(power)} frequency bins")
        
    except Exception as e:
        print(f"FFT error: {e}")
    
    print("\nMathematical engine testing completed!")