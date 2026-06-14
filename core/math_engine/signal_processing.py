"""
Advanced Signal Processing Engine for QUANTUM-FORGE
Implements cutting-edge signal processing techniques for financial time series.
"""

import numpy as np
import scipy.signal as signal
from scipy import fftpack
from scipy.linalg import solve_discrete_lyapunov
from scipy.optimize import minimize
from numba import jit, prange
import pywt
from sklearn.decomposition import PCA, FastICA
from typing import Tuple, List, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class KalmanFilter:
    """Advanced Kalman filtering for state estimation in financial time series."""
    
    def __init__(self, dim_state: int, dim_obs: int):
        """
        Initialize Kalman filter.
        
        Args:
            dim_state: Dimension of state vector
            dim_obs: Dimension of observation vector
        """
        self.dim_state = dim_state
        self.dim_obs = dim_obs
        
        # State transition matrix
        self.F = np.eye(dim_state)
        # Observation matrix
        self.H = np.eye(dim_obs, dim_state)
        # Process noise covariance
        self.Q = np.eye(dim_state) * 0.01
        # Measurement noise covariance
        self.R = np.eye(dim_obs) * 0.1
        # Initial state covariance
        self.P = np.eye(dim_state)
        # Initial state estimate
        self.x = np.zeros(dim_state)
    
    def predict(self, u: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prediction step of Kalman filter.
        
        Args:
            u: Control input (optional)
        
        Returns:
            Tuple of (predicted_state, predicted_covariance)
        """
        # Predict state
        if u is not None:
            self.x = self.F @ self.x + u
        else:
            self.x = self.F @ self.x
        
        # Predict covariance
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.x.copy(), self.P.copy()
    
    def update(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update step of Kalman filter.
        
        Args:
            z: Observation vector
        
        Returns:
            Tuple of (updated_state, updated_covariance)
        """
        # Innovation
        y = z - self.H @ self.x
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        self.x = self.x + K @ y
        
        # Update covariance
        I_KH = np.eye(self.dim_state) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        
        return self.x.copy(), self.P.copy()
    
    def filter_sequence(self, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter entire observation sequence.
        
        Args:
            observations: Array of shape (T, dim_obs)
        
        Returns:
            Tuple of (filtered_states, state_covariances)
        """
        T = len(observations)
        filtered_states = np.zeros((T, self.dim_state))
        state_covariances = np.zeros((T, self.dim_state, self.dim_state))
        
        for t in range(T):
            self.predict()
            self.update(observations[t])
            filtered_states[t] = self.x.copy()
            state_covariances[t] = self.P.copy()
        
        return filtered_states, state_covariances

class ExtendedKalmanFilter(KalmanFilter):
    """Extended Kalman Filter for nonlinear systems."""
    
    def __init__(self, dim_state: int, dim_obs: int, f_func, h_func, 
                 jacobian_f, jacobian_h):
        """
        Initialize EKF with nonlinear functions.
        
        Args:
            dim_state: State dimension
            dim_obs: Observation dimension
            f_func: Nonlinear state transition function
            h_func: Nonlinear observation function
            jacobian_f: Jacobian of state transition function
            jacobian_h: Jacobian of observation function
        """
        super().__init__(dim_state, dim_obs)
        self.f_func = f_func
        self.h_func = h_func
        self.jacobian_f = jacobian_f
        self.jacobian_h = jacobian_h
    
    def predict(self, u: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """EKF prediction with nonlinear state transition."""
        # Linearize around current state
        F = self.jacobian_f(self.x)
        
        # Predict state using nonlinear function
        if u is not None:
            self.x = self.f_func(self.x, u)
        else:
            self.x = self.f_func(self.x)
        
        # Predict covariance using linearized dynamics
        self.P = F @ self.P @ F.T + self.Q
        
        return self.x.copy(), self.P.copy()
    
    def update(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """EKF update with nonlinear observation model."""
        # Linearize observation function
        H = self.jacobian_h(self.x)
        
        # Innovation using nonlinear observation function
        y = z - self.h_func(self.x)
        
        # Innovation covariance
        S = H @ self.P @ H.T + self.R
        
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update state
        self.x = self.x + K @ y
        
        # Update covariance
        I_KH = np.eye(self.dim_state) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        
        return self.x.copy(), self.P.copy()

class ParticleFilter:
    """Particle filter for highly nonlinear systems."""
    
    def __init__(self, dim_state: int, num_particles: int, f_func, h_func, 
                 process_noise_func, observation_noise_func):
        """
        Initialize particle filter.
        
        Args:
            dim_state: State dimension
            num_particles: Number of particles
            f_func: State transition function
            h_func: Observation function
            process_noise_func: Process noise sampling function
            observation_noise_func: Observation likelihood function
        """
        self.dim_state = dim_state
        self.num_particles = num_particles
        self.f_func = f_func
        self.h_func = h_func
        self.process_noise_func = process_noise_func
        self.observation_noise_func = observation_noise_func
        
        # Initialize particles
        # Deterministic particle initialization: evenly spaced patterns with small shifts per dimension
        base = np.linspace(-1.0, 1.0, num_particles)
        particles = np.zeros((num_particles, dim_state))
        for d in range(dim_state):
            particles[:, d] = np.roll(base, d)  # deterministic shift per dimension
        self.particles = particles
        self.weights = np.ones(num_particles) / num_particles
    
    def predict(self):
        """Particle filter prediction step."""
        for i in range(self.num_particles):
            noise = self.process_noise_func()
            self.particles[i] = self.f_func(self.particles[i]) + noise
    
    def update(self, observation: np.ndarray):
        """Particle filter update step."""
        # Calculate weights based on observation likelihood
        for i in range(self.num_particles):
            predicted_obs = self.h_func(self.particles[i])
            likelihood = self.observation_noise_func(observation - predicted_obs)
            self.weights[i] *= likelihood
        
        # Normalize weights
        self.weights /= np.sum(self.weights)
        
        # Resample if effective sample size is too low
        eff_sample_size = 1.0 / np.sum(self.weights**2)
        if eff_sample_size < self.num_particles / 2:
            self._resample()
    
    def _resample(self):
        """Systematic resampling of particles."""
        cumulative_sum = np.cumsum(self.weights)
        cumulative_sum[-1] = 1.0  # Ensure last element is 1
        # Deterministic systematic resampling: fixed offset of 0.5
        positions = (np.arange(self.num_particles) + 0.5) / self.num_particles
        
        new_particles = np.zeros_like(self.particles)
        new_weights = np.ones(self.num_particles) / self.num_particles
        
        j = 0
        for i in range(self.num_particles):
            while positions[i] > cumulative_sum[j]:
                j += 1
            new_particles[i] = self.particles[j].copy()
        
        self.particles = new_particles
        self.weights = new_weights
    
    def get_estimate(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current state estimate and covariance."""
        # Weighted mean
        state_estimate = np.average(self.particles, weights=self.weights, axis=0)
        
        # Weighted covariance
        diff = self.particles - state_estimate
        covariance = np.cov(diff.T, aweights=self.weights)
        
        return state_estimate, covariance

class WaveletAnalysis:
    """Advanced wavelet analysis for multi-resolution signal decomposition."""
    
    @staticmethod
    def continuous_wavelet_transform(signal_data: np.ndarray, wavelet: str = 'cmor', 
                                   scales: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Continuous Wavelet Transform for time-frequency analysis.
        
        Args:
            signal_data: Input signal
            wavelet: Wavelet type ('cmor', 'mexh', 'morl')
            scales: Array of scales to analyze
        
        Returns:
            Tuple of (coefficients, frequencies)
        """
        if scales is None:
            scales = np.arange(1, 128)
        
        coefficients, frequencies = pywt.cwt(signal_data, scales, wavelet)
        return coefficients, frequencies
    
    @staticmethod
    def discrete_wavelet_transform(signal_data: np.ndarray, wavelet: str = 'db4', 
                                 levels: int = 6) -> List[np.ndarray]:
        """
        Multi-level discrete wavelet transform.
        
        Args:
            signal_data: Input signal
            wavelet: Wavelet type ('db4', 'haar', 'bior2.2')
            levels: Number of decomposition levels
        
        Returns:
            List of wavelet coefficients [cA_n, cD_n, cD_n-1, ..., cD_1]
        """
        coeffs = pywt.wavedec(signal_data, wavelet, level=levels)
        return coeffs
    
    @staticmethod
    def wavelet_denoising(signal_data: np.ndarray, wavelet: str = 'db4', 
                         threshold_mode: str = 'soft') -> np.ndarray:
        """
        Wavelet-based denoising using thresholding.
        
        Args:
            signal_data: Noisy signal
            wavelet: Wavelet type
            threshold_mode: 'soft' or 'hard' thresholding
        
        Returns:
            Denoised signal
        """
        # Decompose signal
        coeffs = pywt.wavedec(signal_data, wavelet)
        
        # Estimate noise level using median absolute deviation
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        
        # Calculate threshold
        threshold = sigma * np.sqrt(2 * np.log(len(signal_data)))
        
        # Apply thresholding
        coeffs_thresh = list(coeffs)
        for i in range(1, len(coeffs)):
            coeffs_thresh[i] = pywt.threshold(coeffs[i], threshold, mode=threshold_mode)
        
        # Reconstruct signal
        denoised_signal = pywt.waverec(coeffs_thresh, wavelet)
        
        return denoised_signal
    
    @staticmethod
    def wavelet_coherence(x: np.ndarray, y: np.ndarray, wavelet: str = 'cmor') -> np.ndarray:
        """
        Calculate wavelet coherence between two signals.
        
        Args:
            x: First signal
            y: Second signal
            wavelet: Wavelet type
        
        Returns:
            Wavelet coherence matrix
        """
        scales = np.arange(1, 64)
        
        # Continuous wavelet transforms
        cwt_x, _ = pywt.cwt(x, scales, wavelet)
        cwt_y, _ = pywt.cwt(y, scales, wavelet)
        
        # Cross-wavelet spectrum
        cross_spectrum = cwt_x * np.conj(cwt_y)
        
        # Power spectra
        power_x = np.abs(cwt_x)**2
        power_y = np.abs(cwt_y)**2
        
        # Wavelet coherence
        coherence = np.abs(cross_spectrum)**2 / (power_x * power_y)
        
        return coherence

class FourierAnalysis:
    """Advanced Fourier analysis techniques."""
    
    @staticmethod
    @jit(nopython=True)
    def fast_fourier_transform(signal_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fast Fourier Transform with frequency array.
        
        Args:
            signal_data: Input signal
        
        Returns:
            Tuple of (fft_coefficients, frequencies)
        """
        N = len(signal_data)
        fft_coeffs = np.fft.fft(signal_data)
        freqs = np.fft.fftfreq(N)
        
        return fft_coeffs, freqs
    
    @staticmethod
    def spectral_density_estimation(signal_data: np.ndarray, method: str = 'welch', 
                                  nperseg: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Power spectral density estimation.
        
        Args:
            signal_data: Input signal
            method: 'welch', 'periodogram', or 'multitaper'
            nperseg: Length of each segment for Welch method
        
        Returns:
            Tuple of (frequencies, power_spectral_density)
        """
        if method == 'welch':
            if nperseg is None:
                nperseg = len(signal_data) // 8
            freqs, psd = signal.welch(signal_data, nperseg=nperseg)
        elif method == 'periodogram':
            freqs, psd = signal.periodogram(signal_data)
        elif method == 'multitaper':
            freqs, psd = signal.periodogram(signal_data, window='dpss')
        else:
            raise ValueError("Method must be 'welch', 'periodogram', or 'multitaper'")
        
        return freqs, psd
    
    @staticmethod
    def hilbert_transform(signal_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Hilbert transform for instantaneous amplitude and phase.
        
        Args:
            signal_data: Input signal
        
        Returns:
            Tuple of (analytic_signal, instantaneous_amplitude, instantaneous_phase)
        """
        analytic_signal = signal.hilbert(signal_data)
        instantaneous_amplitude = np.abs(analytic_signal)
        instantaneous_phase = np.angle(analytic_signal)
        
        return analytic_signal, instantaneous_amplitude, instantaneous_phase

class EmpiricalModeDecomposition:
    """Empirical Mode Decomposition for adaptive signal decomposition."""
    
    @staticmethod
    def sifting_process(signal_data: np.ndarray, max_iter: int = 1000, 
                      tolerance: float = 0.05) -> np.ndarray:
        """
        Single IMF extraction using sifting process.
        
        Args:
            signal_data: Input signal
            max_iter: Maximum iterations for sifting
            tolerance: Convergence tolerance
        
        Returns:
            Intrinsic Mode Function (IMF)
        """
        h = signal_data.copy()
        
        for _ in range(max_iter):
            # Find local maxima and minima
            maxima_idx = signal.find_peaks(h)[0]
            minima_idx = signal.find_peaks(-h)[0]
            
            if len(maxima_idx) < 2 or len(minima_idx) < 2:
                break
            
            # Interpolate envelopes
            from scipy import interpolate
            
            # Extend boundaries to avoid edge effects
            t = np.arange(len(h))
            
            # Upper envelope (maxima)
            if len(maxima_idx) >= 2:
                f_max = interpolate.interp1d(maxima_idx, h[maxima_idx], 
                                           kind='cubic', fill_value='extrapolate')
                upper_envelope = f_max(t)
            else:
                upper_envelope = np.zeros_like(h)
            
            # Lower envelope (minima)
            if len(minima_idx) >= 2:
                f_min = interpolate.interp1d(minima_idx, h[minima_idx], 
                                           kind='cubic', fill_value='extrapolate')
                lower_envelope = f_min(t)
            else:
                lower_envelope = np.zeros_like(h)
            
            # Mean of envelopes
            mean_envelope = (upper_envelope + lower_envelope) / 2
            
            # New component
            h_new = h - mean_envelope
            
            # Check stopping criterion
            sd = np.sum((h - h_new)**2) / np.sum(h**2)
            
            h = h_new
            
            if sd < tolerance:
                break
        
        return h
    
    @staticmethod
    def empirical_mode_decomposition(signal_data: np.ndarray, 
                                   max_imfs: int = 10) -> List[np.ndarray]:
        """
        Complete EMD decomposition.
        
        Args:
            signal_data: Input signal
            max_imfs: Maximum number of IMFs to extract
        
        Returns:
            List of IMFs including residue
        """
        imfs = []
        residue = signal_data.copy()
        
        for _ in range(max_imfs):
            imf = EmpiricalModeDecomposition.sifting_process(residue)
            
            # Check if IMF is meaningful
            if np.std(imf) < 1e-10:
                break
            
            imfs.append(imf)
            residue = residue - imf
            
            # Stop if residue is monotonic
            if len(signal.find_peaks(residue)[0]) < 2 and len(signal.find_peaks(-residue)[0]) < 2:
                break
        
        # Add final residue
        imfs.append(residue)
        
        return imfs

class IndependentComponentAnalysis:
    """Independent Component Analysis for blind source separation."""
    
    @staticmethod
    def fastica_separation(mixed_signals: np.ndarray, n_components: Optional[int] = None, 
                          max_iter: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """
        FastICA algorithm for source separation.
        
        Args:
            mixed_signals: Mixed signals array (n_samples, n_signals)
            n_components: Number of components to extract
            max_iter: Maximum iterations
        
        Returns:
            Tuple of (separated_sources, mixing_matrix)
        """
        if n_components is None:
            n_components = mixed_signals.shape[1]
        
        # Center the data
        mixed_signals = mixed_signals - np.mean(mixed_signals, axis=0)
        
        # Whitening using PCA
        pca = PCA(whiten=True)
        whitened_signals = pca.fit_transform(mixed_signals)
        
        # Apply FastICA
        ica = FastICA(n_components=n_components, max_iter=max_iter, random_state=42)
        separated_sources = ica.fit_transform(whitened_signals)
        
        # Mixing matrix
        mixing_matrix = ica.mixing_
        
        return separated_sources, mixing_matrix

# Example usage and testing
if __name__ == "__main__":
    # Test Kalman filter with deterministic synthetic signals (no randomness)
    T = 1000
    t = np.linspace(0, 10, T)
    # Deterministic 'true' signal: slow sinusoidal drift
    true_signal = 0.01 * np.cumsum(np.sin(t))
    # Observations: true signal plus higher-frequency deterministic noise
    observations = true_signal + 0.1 * np.sin(20 * t)
    
    # Initialize Kalman filter for random walk
    kf = KalmanFilter(dim_state=1, dim_obs=1)
    kf.F = np.array([[1.0]])  # Random walk dynamics
    kf.H = np.array([[1.0]])  # Direct observation
    kf.Q = np.array([[0.01**2]])  # Process noise
    kf.R = np.array([[0.1**2]])  # Observation noise
    
    # Filter the observations
    filtered_states, _ = kf.filter_sequence(observations.reshape(-1, 1))
    filtered_signal = filtered_states.flatten()
    
    print(f"Kalman filter RMSE: {np.sqrt(np.mean((filtered_signal - true_signal)**2)):.4f}")
    
    # Test wavelet denoising with deterministic noise
    noisy_signal = true_signal + 0.2 * np.sin(15 * t)
    denoised_signal = WaveletAnalysis.wavelet_denoising(noisy_signal)
    
    print(f"Wavelet denoising RMSE: {np.sqrt(np.mean((denoised_signal - true_signal)**2)):.4f}")
    
    # Test EMD
    # Create a signal with multiple deterministic components
    t = np.linspace(0, 4*np.pi, 1000)
    signal_component = np.sin(2*t) + 0.5*np.sin(10*t) + 0.1*np.sin(50*t)
    
    imfs = EmpiricalModeDecomposition.empirical_mode_decomposition(signal_component)
    print(f"EMD extracted {len(imfs)} IMFs")
    
    print("Signal processing engine test completed successfully!")