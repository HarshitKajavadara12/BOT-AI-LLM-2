"""
Test Suite for Mathematical Engine Components

Tests against the actual class-based APIs:
  - LinearAlgebraEngine, SparseMatrixEngine, TensorOperations
  - StochasticProcesses, HestonModel
  - KalmanFilter, WaveletAnalysis, FourierAnalysis
  - UnitRootTests, NormalityTests, CointegrationTests
  - FourierAnalyzer, SpectrogramAnalyzer
  - MonteCarloMethods, FiniteDifferenceMethods
"""

import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_almost_equal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestLinearAlgebraEngine(unittest.TestCase):
    """Test LinearAlgebraEngine class methods."""

    def setUp(self):
        from core.math_engine.linear_algebra import LinearAlgebraEngine
        self.engine = LinearAlgebraEngine()
        self.matrix_a = np.array([[4.0, 2.0], [3.0, 1.0]])
        self.spd_matrix = np.array([[4.0, 2.0], [2.0, 3.0]])  # Symmetric positive definite

    def test_lu_decomposition(self):
        result = self.engine.lu_decomposition(self.matrix_a)
        self.assertIsNotNone(result)

    def test_qr_decomposition(self):
        result = self.engine.qr_decomposition(self.matrix_a, mode='economic')
        self.assertIsNotNone(result)

    def test_svd_decomposition(self):
        result = self.engine.svd_decomposition(self.matrix_a)
        self.assertIsNotNone(result)

    def test_eigendecomposition(self):
        result = self.engine.eigendecomposition(self.matrix_a)
        self.assertIsNotNone(result)

    def test_cholesky_decomposition(self):
        result = self.engine.cholesky_decomposition(self.spd_matrix)
        self.assertIsNotNone(result)

    def test_solve_linear_system(self):
        b = np.array([5.0, 3.0])
        result = self.engine.solve_linear_system(self.matrix_a, b)
        self.assertIsNotNone(result)

    def test_matrix_inverse(self):
        inv = self.engine.matrix_inverse(self.matrix_a)
        identity = self.matrix_a @ inv
        assert_array_almost_equal(identity, np.eye(2), decimal=8)


class TestSparseMatrixEngine(unittest.TestCase):
    """Test SparseMatrixEngine."""

    def test_instantiate(self):
        from core.math_engine.linear_algebra import SparseMatrixEngine
        engine = SparseMatrixEngine()
        self.assertIsNotNone(engine)


class TestTensorOperations(unittest.TestCase):
    """Test TensorOperations."""

    def test_instantiate(self):
        from core.math_engine.linear_algebra import TensorOperations
        ops = TensorOperations()
        self.assertIsNotNone(ops)


class TestStochasticProcesses(unittest.TestCase):
    """Test StochasticProcesses class (static methods)."""

    def setUp(self):
        from core.math_engine.stochastic_calculus import StochasticProcesses
        self.sp = StochasticProcesses

    def test_geometric_brownian_motion(self):
        result = self.sp.geometric_brownian_motion(
            S0=100.0, mu=0.05, sigma=0.2, T=1.0, N=252
        )
        self.assertIsNotNone(result)
        self.assertGreater(len(result), 0)

    def test_ornstein_uhlenbeck(self):
        result = self.sp.ornstein_uhlenbeck_process(
            X0=0.0, theta=0.7, mu=0.0, sigma=0.1, T=1.0, N=100
        )
        self.assertIsNotNone(result)


class TestHestonModel(unittest.TestCase):
    """Test HestonModel."""

    def test_instantiate(self):
        from core.math_engine.stochastic_calculus import HestonModel
        model = HestonModel(
            S0=100.0, v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7
        )
        self.assertIsNotNone(model)


class TestKalmanFilter(unittest.TestCase):
    """Test KalmanFilter class."""

    def test_instantiate(self):
        from core.math_engine.signal_processing import KalmanFilter
        kf = KalmanFilter(dim_state=2, dim_obs=1)
        self.assertIsNotNone(kf)


class TestWaveletAnalysis(unittest.TestCase):
    """Test WaveletAnalysis class."""

    def test_instantiate(self):
        from core.math_engine.signal_processing import WaveletAnalysis
        wa = WaveletAnalysis()
        self.assertIsNotNone(wa)


class TestFourierAnalysis(unittest.TestCase):
    """Test FourierAnalysis from signal_processing."""

    def test_instantiate(self):
        from core.math_engine.signal_processing import FourierAnalysis
        fa = FourierAnalysis()
        self.assertIsNotNone(fa)


class TestStatisticalTests(unittest.TestCase):
    """Test statistical test classes."""

    def test_unit_root_tests(self):
        from core.math_engine.statistical_tests import UnitRootTests
        t = UnitRootTests()
        self.assertIsNotNone(t)

    def test_normality_tests(self):
        from core.math_engine.statistical_tests import NormalityTests
        t = NormalityTests()
        self.assertIsNotNone(t)

    def test_cointegration_tests(self):
        from core.math_engine.statistical_tests import CointegrationTests
        t = CointegrationTests()
        self.assertIsNotNone(t)

    def test_serial_correlation_tests(self):
        from core.math_engine.statistical_tests import SerialCorrelationTests
        t = SerialCorrelationTests()
        self.assertIsNotNone(t)

    def test_heteroskedasticity_tests(self):
        from core.math_engine.statistical_tests import HeteroskedasticityTests
        t = HeteroskedasticityTests()
        self.assertIsNotNone(t)


class TestFourierAnalyzer(unittest.TestCase):
    """Test FourierAnalyzer from fourier_analysis module."""

    def setUp(self):
        from core.math_engine.fourier_analysis import FourierAnalyzer
        self.fa = FourierAnalyzer()

    def test_dft(self):
        signal = np.sin(np.linspace(0, 4 * np.pi, 128))
        result = self.fa.dft(signal)
        self.assertIsNotNone(result)

    def test_idft(self):
        signal = np.sin(np.linspace(0, 4 * np.pi, 128))
        spectrum = self.fa.dft(signal)
        reconstructed = self.fa.idft(spectrum)
        self.assertIsNotNone(reconstructed)

    def test_power_spectral_density(self):
        signal = np.sin(np.linspace(0, 4 * np.pi, 128))
        psd = self.fa.power_spectral_density(signal)
        self.assertIsNotNone(psd)


class TestNumericalMethods(unittest.TestCase):
    """Test numerical methods classes."""

    def test_monte_carlo(self):
        from core.math_engine.numerical_methods import MonteCarloMethods
        mc = MonteCarloMethods()
        self.assertIsNotNone(mc)

    def test_finite_difference(self):
        from core.math_engine.numerical_methods import FiniteDifferenceMethods
        fd = FiniteDifferenceMethods()
        self.assertIsNotNone(fd)

    def test_optimization(self):
        from core.math_engine.numerical_methods import OptimizationMethods
        opt = OptimizationMethods()
        self.assertIsNotNone(opt)

    def test_interpolation(self):
        from core.math_engine.numerical_methods import InterpolationMethods
        im = InterpolationMethods()
        self.assertIsNotNone(im)


if __name__ == '__main__':
    unittest.main(verbosity=2)
