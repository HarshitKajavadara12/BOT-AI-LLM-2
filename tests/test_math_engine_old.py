"""
Test Suite for Mathematical Engine Components

This module contains comprehensive unit tests for the mathematical engine,
covering linear algebra, numerical methods, stochastic calculus, signal processing,
optimal control, statistical tests, and Fourier analysis.
"""

import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_almost_equal
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.math_engine.linear_algebra import (
    matrix_decomposition,
    solve_linear_system,
    eigenvalue_analysis,
    matrix_operations
)
from core.math_engine.numerical_methods import (
    numerical_integration,
    root_finding,
    optimization,
    differential_equations
)
from core.math_engine.stochastic_calculus import (
    brownian_motion,
    geometric_brownian_motion,
    ornstein_uhlenbeck_process,
    ito_integral,
    heston_model
)
from core.math_engine.signal_processing import (
    filter_design,
    spectral_analysis,
    wavelet_transform,
    kalman_filter
)
from core.math_engine.statistical_tests import (
    normality_tests,
    stationarity_tests,
    correlation_tests,
    hypothesis_tests
)
from core.math_engine.fourier_analysis import (
    fft_transform,
    inverse_fft,
    power_spectrum,
    spectral_density
)


class TestLinearAlgebra(unittest.TestCase):
    """Test cases for linear algebra operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.matrix_a = np.array([[4, 2], [3, 1]])
        self.matrix_b = np.array([[1, 0], [0, 1]])
        self.vector = np.array([5, 3])
        
    def test_matrix_multiplication(self):
        """Test matrix multiplication."""
        result = matrix_operations.multiply(self.matrix_a, self.matrix_b)
        expected = self.matrix_a
        assert_array_almost_equal(result, expected)
        
    def test_matrix_inverse(self):
        """Test matrix inversion."""
        inv = matrix_operations.inverse(self.matrix_a)
        identity = np.dot(self.matrix_a, inv)
        assert_array_almost_equal(identity, np.eye(2), decimal=10)
        
    def test_lu_decomposition(self):
        """Test LU decomposition."""
        L, U = matrix_decomposition.lu_decompose(self.matrix_a)
        reconstructed = np.dot(L, U)
        assert_array_almost_equal(reconstructed, self.matrix_a, decimal=10)
        
    def test_qr_decomposition(self):
        """Test QR decomposition."""
        Q, R = matrix_decomposition.qr_decompose(self.matrix_a)
        # Q should be orthogonal
        identity = np.dot(Q.T, Q)
        assert_array_almost_equal(identity, np.eye(2), decimal=10)
        # Reconstruction
        reconstructed = np.dot(Q, R)
        assert_array_almost_equal(reconstructed, self.matrix_a, decimal=10)
        
    def test_svd_decomposition(self):
        """Test Singular Value Decomposition."""
        U, S, Vt = matrix_decomposition.svd(self.matrix_a)
        reconstructed = U @ np.diag(S) @ Vt
        assert_array_almost_equal(reconstructed, self.matrix_a, decimal=10)
        
    def test_eigenvalue_decomposition(self):
        """Test eigenvalue decomposition."""
        eigenvals, eigenvecs = eigenvalue_analysis.eigen_decompose(self.matrix_a)
        # Check Av = λv
        for i, eigenval in enumerate(eigenvals):
            left = np.dot(self.matrix_a, eigenvecs[:, i])
            right = eigenval * eigenvecs[:, i]
            assert_array_almost_equal(left, right, decimal=10)
            
    def test_linear_system_solver(self):
        """Test linear system solver."""
        solution = solve_linear_system.solve(self.matrix_a, self.vector)
        # Check Ax = b
        result = np.dot(self.matrix_a, solution)
        assert_array_almost_equal(result, self.vector, decimal=10)
        
    def test_matrix_determinant(self):
        """Test matrix determinant calculation."""
        det = matrix_operations.determinant(self.matrix_a)
        expected = np.linalg.det(self.matrix_a)
        assert_almost_equal(det, expected, decimal=10)
        
    def test_matrix_trace(self):
        """Test matrix trace calculation."""
        trace = matrix_operations.trace(self.matrix_a)
        expected = np.trace(self.matrix_a)
        assert_almost_equal(trace, expected, decimal=10)
        
    def test_cholesky_decomposition(self):
        """Test Cholesky decomposition for positive definite matrices."""
        # Create a positive definite matrix
        pd_matrix = np.array([[4, 2], [2, 3]])
        L = matrix_decomposition.cholesky(pd_matrix)
        reconstructed = np.dot(L, L.T)
        assert_array_almost_equal(reconstructed, pd_matrix, decimal=10)


class TestNumericalMethods(unittest.TestCase):
    """Test cases for numerical methods."""
    
    def test_trapezoidal_integration(self):
        """Test trapezoidal rule integration."""
        # Integrate x^2 from 0 to 1 (analytical result = 1/3)
        f = lambda x: x**2
        result = numerical_integration.trapezoidal(f, 0, 1, n=1000)
        assert_almost_equal(result, 1/3, decimal=3)
        
    def test_simpson_integration(self):
        """Test Simpson's rule integration."""
        f = lambda x: x**2
        result = numerical_integration.simpson(f, 0, 1, n=1000)
        assert_almost_equal(result, 1/3, decimal=5)
        
    def test_monte_carlo_integration(self):
        """Test Monte Carlo integration."""
        f = lambda x: x**2
        result = numerical_integration.monte_carlo(f, 0, 1, n_samples=100000)
        assert_almost_equal(result, 1/3, decimal=2)
        
    def test_newton_raphson_root_finding(self):
        """Test Newton-Raphson root finding."""
        # Find root of x^2 - 4 = 0 (root should be 2)
        f = lambda x: x**2 - 4
        df = lambda x: 2*x
        root = root_finding.newton_raphson(f, df, x0=1.5, tol=1e-10)
        assert_almost_equal(root, 2.0, decimal=10)
        
    def test_bisection_root_finding(self):
        """Test bisection method root finding."""
        f = lambda x: x**2 - 4
        root = root_finding.bisection(f, 0, 3, tol=1e-10)
        assert_almost_equal(root, 2.0, decimal=10)
        
    def test_gradient_descent(self):
        """Test gradient descent optimization."""
        # Minimize (x-3)^2 (minimum at x=3)
        f = lambda x: (x - 3)**2
        grad = lambda x: 2*(x - 3)
        result = optimization.gradient_descent(f, grad, x0=0, learning_rate=0.1, max_iter=1000)
        assert_almost_equal(result, 3.0, decimal=3)
        
    def test_bfgs_optimization(self):
        """Test BFGS quasi-Newton optimization."""
        # Minimize (x-3)^2 + (y-4)^2
        f = lambda x: (x[0] - 3)**2 + (x[1] - 4)**2
        result = optimization.bfgs(f, x0=np.array([0.0, 0.0]))
        expected = np.array([3.0, 4.0])
        assert_array_almost_equal(result, expected, decimal=3)
        
    def test_euler_method_ode(self):
        """Test Euler method for ODEs."""
        # Solve dy/dx = y with y(0) = 1 (solution: y = e^x)
        f = lambda x, y: y
        x_span = (0, 1)
        y0 = 1.0
        x, y = differential_equations.euler(f, x_span, y0, n=1000)
        expected = np.exp(1.0)
        assert_almost_equal(y[-1], expected, decimal=2)
        
    def test_runge_kutta_ode(self):
        """Test Runge-Kutta method for ODEs."""
        f = lambda x, y: y
        x_span = (0, 1)
        y0 = 1.0
        x, y = differential_equations.runge_kutta_4(f, x_span, y0, n=100)
        expected = np.exp(1.0)
        assert_almost_equal(y[-1], expected, decimal=5)


class TestStochasticCalculus(unittest.TestCase):
    """Test cases for stochastic calculus."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        
    def test_brownian_motion_properties(self):
        """Test properties of Brownian motion."""
        n_steps = 10000
        dt = 0.01
        bm = brownian_motion.generate(n_steps, dt)
        
        # Test mean is close to zero
        assert_almost_equal(np.mean(bm), 0.0, decimal=1)
        
        # Test variance grows linearly with time
        expected_variance = n_steps * dt
        assert_almost_equal(np.var(bm), expected_variance, decimal=0)
        
    def test_geometric_brownian_motion(self):
        """Test geometric Brownian motion."""
        S0 = 100
        mu = 0.05
        sigma = 0.2
        T = 1.0
        n_steps = 1000
        
        paths = geometric_brownian_motion.generate(S0, mu, sigma, T, n_steps, n_paths=10000)
        
        # Test expected value
        expected_mean = S0 * np.exp(mu * T)
        actual_mean = np.mean(paths[:, -1])
        # Allow 5% tolerance due to sampling
        assert actual_mean > expected_mean * 0.95
        assert actual_mean < expected_mean * 1.05
        
    def test_ornstein_uhlenbeck_process(self):
        """Test Ornstein-Uhlenbeck process mean reversion."""
        theta = 0.5  # Mean reversion speed
        mu = 10.0    # Long-term mean
        sigma = 0.3  # Volatility
        X0 = 5.0
        T = 10.0
        n_steps = 10000
        
        process = ornstein_uhlenbeck_process.generate(X0, theta, mu, sigma, T, n_steps)
        
        # Test mean reversion to long-term mean
        final_mean = np.mean(process[-1000:])  # Average of last 1000 steps
        assert_almost_equal(final_mean, mu, decimal=0)
        
    def test_ito_integral_properties(self):
        """Test properties of Ito integral."""
        n_steps = 10000
        dt = 0.01
        
        # Simple Ito integral: ∫W_t dW_t = (W_T^2 - T)/2
        W = brownian_motion.generate(n_steps, dt)
        integral = ito_integral.compute(W, W, dt)
        
        expected = (W[-1]**2 - n_steps * dt) / 2
        assert_almost_equal(integral[-1], expected, decimal=1)
        
    def test_heston_model(self):
        """Test Heston stochastic volatility model."""
        S0 = 100
        v0 = 0.04
        kappa = 2.0
        theta = 0.04
        sigma_v = 0.3
        rho = -0.7
        r = 0.05
        T = 1.0
        n_steps = 1000
        
        S, v = heston_model.simulate(S0, v0, kappa, theta, sigma_v, rho, r, T, n_steps)
        
        # Test that price is positive
        self.assertTrue(np.all(S > 0))
        
        # Test that variance is positive
        self.assertTrue(np.all(v > 0))
        
        # Test mean reversion of variance
        long_run_var = np.mean(v[-100:])
        assert_almost_equal(long_run_var, theta, decimal=1)


class TestSignalProcessing(unittest.TestCase):
    """Test cases for signal processing."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.fs = 1000  # Sampling frequency
        self.t = np.linspace(0, 1, self.fs)
        
    def test_lowpass_filter(self):
        """Test low-pass filter design and application."""
        # Create signal with low and high frequency components
        signal = np.sin(2 * np.pi * 5 * self.t) + np.sin(2 * np.pi * 100 * self.t)
        
        # Apply low-pass filter with cutoff at 50 Hz
        filtered = filter_design.lowpass(signal, cutoff=50, fs=self.fs, order=4)
        
        # High frequency component should be attenuated
        fft_original = np.fft.fft(signal)
        fft_filtered = np.fft.fft(filtered)
        freqs = np.fft.fftfreq(len(signal), 1/self.fs)
        
        # Check attenuation at 100 Hz
        idx_100hz = np.argmin(np.abs(freqs - 100))
        attenuation = np.abs(fft_filtered[idx_100hz]) / np.abs(fft_original[idx_100hz])
        self.assertLess(attenuation, 0.1)  # Should be strongly attenuated
        
    def test_highpass_filter(self):
        """Test high-pass filter."""
        signal = np.sin(2 * np.pi * 5 * self.t) + np.sin(2 * np.pi * 100 * self.t)
        filtered = filter_design.highpass(signal, cutoff=50, fs=self.fs, order=4)
        
        # Low frequency component should be attenuated
        fft_original = np.fft.fft(signal)
        fft_filtered = np.fft.fft(filtered)
        freqs = np.fft.fftfreq(len(signal), 1/self.fs)
        
        idx_5hz = np.argmin(np.abs(freqs - 5))
        attenuation = np.abs(fft_filtered[idx_5hz]) / np.abs(fft_original[idx_5hz])
        self.assertLess(attenuation, 0.1)
        
    def test_bandpass_filter(self):
        """Test band-pass filter."""
        signal = (np.sin(2 * np.pi * 5 * self.t) + 
                 np.sin(2 * np.pi * 50 * self.t) + 
                 np.sin(2 * np.pi * 150 * self.t))
        
        filtered = filter_design.bandpass(signal, lowcut=30, highcut=70, fs=self.fs, order=4)
        
        # Only middle frequency should pass
        fft_filtered = np.fft.fft(filtered)
        freqs = np.fft.fftfreq(len(signal), 1/self.fs)
        power = np.abs(fft_filtered)**2
        
        # Find peak frequency
        positive_freqs = freqs[:len(freqs)//2]
        positive_power = power[:len(power)//2]
        peak_freq = positive_freqs[np.argmax(positive_power)]
        
        assert_almost_equal(peak_freq, 50, decimal=0)
        
    def test_power_spectrum(self):
        """Test power spectrum calculation."""
        # Create signal with known frequencies
        freq1, freq2 = 10, 50
        signal = np.sin(2 * np.pi * freq1 * self.t) + 0.5 * np.sin(2 * np.pi * freq2 * self.t)
        
        freqs, power = spectral_analysis.power_spectrum(signal, self.fs)
        
        # Find peaks
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(power, height=np.max(power) * 0.1)
        peak_freqs = freqs[peaks]
        
        # Should have peaks near 10 and 50 Hz
        self.assertTrue(any(abs(peak_freqs - freq1) < 2))
        self.assertTrue(any(abs(peak_freqs - freq2) < 2))
        
    def test_wavelet_transform(self):
        """Test wavelet transform."""
        # Create signal with time-varying frequency
        signal = np.concatenate([
            np.sin(2 * np.pi * 10 * self.t[:len(self.t)//2]),
            np.sin(2 * np.pi * 50 * self.t[len(self.t)//2:])
        ])
        
        coeffs, freqs = wavelet_transform.cwt(signal, self.fs)
        
        # Check shape
        self.assertEqual(coeffs.shape[1], len(signal))
        self.assertTrue(len(freqs) > 0)
        
    def test_kalman_filter_constant(self):
        """Test Kalman filter on constant signal with noise."""
        true_value = 10.0
        noise_std = 1.0
        n_samples = 100
        
        # Noisy observations
        observations = true_value + np.random.normal(0, noise_std, n_samples)
        
        # Apply Kalman filter
        estimates = kalman_filter.filter_1d(observations, process_variance=0.01, 
                                            observation_variance=noise_std**2)
        
        # Filtered estimate should be closer to true value than raw observations
        final_error = abs(estimates[-1] - true_value)
        raw_error = abs(observations[-1] - true_value)
        
        # On average, filtered should be better (may not hold for single sample)
        mean_filtered_error = np.mean(np.abs(estimates - true_value))
        mean_raw_error = np.mean(np.abs(observations - true_value))
        self.assertLess(mean_filtered_error, mean_raw_error)


class TestStatisticalTests(unittest.TestCase):
    """Test cases for statistical tests."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        
    def test_jarque_bera_normal(self):
        """Test Jarque-Bera normality test on normal data."""
        data = np.random.normal(0, 1, 1000)
        statistic, p_value = normality_tests.jarque_bera(data)
        
        # Should not reject normality (p-value > 0.05)
        self.assertGreater(p_value, 0.05)
        
    def test_jarque_bera_nonnormal(self):
        """Test Jarque-Bera test on non-normal data."""
        data = np.random.exponential(1, 1000)  # Exponential is not normal
        statistic, p_value = normality_tests.jarque_bera(data)
        
        # Should reject normality (p-value < 0.05)
        self.assertLess(p_value, 0.05)
        
    def test_shapiro_wilk_normal(self):
        """Test Shapiro-Wilk normality test."""
        data = np.random.normal(0, 1, 100)
        statistic, p_value = normality_tests.shapiro_wilk(data)
        
        # Should not reject normality
        self.assertGreater(p_value, 0.05)
        
    def test_adf_stationary(self):
        """Test Augmented Dickey-Fuller test on stationary data."""
        data = np.random.normal(0, 1, 1000)
        statistic, p_value = stationarity_tests.adf(data)
        
        # Should reject unit root (data is stationary)
        self.assertLess(p_value, 0.05)
        
    def test_adf_nonstationary(self):
        """Test ADF test on non-stationary data."""
        data = np.cumsum(np.random.normal(0, 1, 1000))  # Random walk
        statistic, p_value = stationarity_tests.adf(data)
        
        # Should not reject unit root (data is non-stationary)
        self.assertGreater(p_value, 0.05)
        
    def test_kpss_stationary(self):
        """Test KPSS stationarity test."""
        data = np.random.normal(0, 1, 1000)
        statistic, p_value = stationarity_tests.kpss(data)
        
        # Should not reject stationarity
        self.assertGreater(p_value, 0.05)
        
    def test_ljung_box_white_noise(self):
        """Test Ljung-Box test on white noise."""
        data = np.random.normal(0, 1, 1000)
        statistic, p_value = correlation_tests.ljung_box(data, lags=10)
        
        # Should not reject independence
        self.assertGreater(p_value, 0.05)
        
    def test_ljung_box_autocorrelated(self):
        """Test Ljung-Box test on autocorrelated data."""
        # AR(1) process with high autocorrelation
        data = [0]
        for _ in range(999):
            data.append(0.9 * data[-1] + np.random.normal(0, 1))
        data = np.array(data)
        
        statistic, p_value = correlation_tests.ljung_box(data, lags=10)
        
        # Should reject independence
        self.assertLess(p_value, 0.05)
        
    def test_t_test(self):
        """Test t-test for mean."""
        # Sample from distribution with mean 5
        data = np.random.normal(5, 1, 100)
        
        # Test null hypothesis that mean = 5
        statistic, p_value = hypothesis_tests.t_test(data, popmean=5)
        self.assertGreater(p_value, 0.05)
        
        # Test null hypothesis that mean = 0
        statistic, p_value = hypothesis_tests.t_test(data, popmean=0)
        self.assertLess(p_value, 0.05)
        
    def test_chi_square_test(self):
        """Test chi-square goodness of fit."""
        # Generate data from uniform distribution
        data = np.random.uniform(0, 4, 1000)
        bins = np.array([0, 1, 2, 3, 4])
        observed, _ = np.histogram(data, bins=bins)
        
        # Expected frequencies for uniform distribution
        expected = np.array([250, 250, 250, 250])
        
        statistic, p_value = hypothesis_tests.chi_square(observed, expected)
        
        # Should not reject uniform distribution
        self.assertGreater(p_value, 0.05)


class TestFourierAnalysis(unittest.TestCase):
    """Test cases for Fourier analysis."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.fs = 1000
        self.t = np.linspace(0, 1, self.fs)
        
    def test_fft_basic(self):
        """Test basic FFT transformation."""
        freq = 50  # Hz
        signal = np.sin(2 * np.pi * freq * self.t)
        
        fft_result = fft_transform.compute(signal)
        freqs = np.fft.fftfreq(len(signal), 1/self.fs)
        
        # Find peak frequency
        positive_mask = freqs > 0
        peak_idx = np.argmax(np.abs(fft_result[positive_mask]))
        peak_freq = freqs[positive_mask][peak_idx]
        
        assert_almost_equal(peak_freq, freq, decimal=0)
        
    def test_inverse_fft(self):
        """Test inverse FFT."""
        signal = np.sin(2 * np.pi * 50 * self.t) + np.sin(2 * np.pi * 120 * self.t)
        
        fft_result = fft_transform.compute(signal)
        reconstructed = inverse_fft.compute(fft_result)
        
        assert_array_almost_equal(signal, reconstructed.real, decimal=10)
        
    def test_power_spectrum_parseval(self):
        """Test Parseval's theorem on power spectrum."""
        signal = np.random.normal(0, 1, 1000)
        
        # Time domain power
        time_power = np.sum(signal**2)
        
        # Frequency domain power
        freq_power = power_spectrum.compute(signal)
        total_freq_power = np.sum(freq_power) / len(signal)
        
        # Parseval's theorem
        assert_almost_equal(time_power, total_freq_power, decimal=1)
        
    def test_spectral_density_white_noise(self):
        """Test spectral density of white noise."""
        signal = np.random.normal(0, 1, 10000)
        
        freqs, psd = spectral_density.welch(signal, self.fs)
        
        # White noise should have flat spectrum
        # Check that variance across frequencies is small
        psd_variance = np.var(psd)
        psd_mean = np.mean(psd)
        
        # Coefficient of variation should be small for white noise
        cv = np.sqrt(psd_variance) / psd_mean
        self.assertLess(cv, 0.5)
        
    def test_windowing(self):
        """Test windowing functions."""
        signal = np.ones(1000)
        
        # Apply Hann window
        windowed = fft_transform.apply_window(signal, window='hann')
        
        # Check that edges are attenuated
        self.assertLess(windowed[0], 0.1)
        self.assertLess(windowed[-1], 0.1)
        
        # Check that center is near 1
        center_value = windowed[len(windowed)//2]
        assert_almost_equal(center_value, 1.0, decimal=1)


class TestOptimalControl(unittest.TestCase):
    """Test cases for optimal control methods."""
    
    def test_linear_quadratic_regulator(self):
        """Test LQR optimal control."""
        from core.math_engine.optimal_control import lqr_solver
        
        # Simple 1D system: dx/dt = u (control the derivative)
        A = np.array([[0.0]])
        B = np.array([[1.0]])
        Q = np.array([[1.0]])  # State cost
        R = np.array([[0.1]])  # Control cost
        
        K = lqr_solver.solve(A, B, Q, R)
        
        # Check that K is negative (feedback control)
        self.assertTrue(K[0, 0] < 0)
        
    def test_dynamic_programming(self):
        """Test dynamic programming for discrete problems."""
        from core.math_engine.optimal_control import dynamic_programming
        
        # Simple problem: minimize sum of squares reaching target
        def cost(state, action, next_state):
            return action**2 + (next_state - 10)**2
        
        states = np.arange(0, 11)
        actions = np.arange(-2, 3)
        
        policy = dynamic_programming.value_iteration(states, actions, cost, 
                                                     discount=0.9, max_iter=100)
        
        # Policy should drive state toward target (10)
        self.assertIsNotNone(policy)


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestLinearAlgebra))
    suite.addTests(loader.loadTestsFromTestCase(TestNumericalMethods))
    suite.addTests(loader.loadTestsFromTestCase(TestStochasticCalculus))
    suite.addTests(loader.loadTestsFromTestCase(TestSignalProcessing))
    suite.addTests(loader.loadTestsFromTestCase(TestStatisticalTests))
    suite.addTests(loader.loadTestsFromTestCase(TestFourierAnalysis))
    suite.addTests(loader.loadTestsFromTestCase(TestOptimalControl))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return success status
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
