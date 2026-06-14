"""
Fourier Analysis Engine
Advanced Fourier analysis and spectral methods for financial time series
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, Any, List, Union, Callable
from abc import ABC, abstractmethod
import scipy.fft as fft
import scipy.signal as signal
from scipy.signal.windows import get_window
import warnings


class FourierAnalyzer:
    """
    Comprehensive Fourier analysis for financial time series
    """
    
    def __init__(self):
        self.sample_rate = 1.0  # Default sample rate
        self.window_functions = {
            'hann': signal.windows.hann,
            'hamming': signal.windows.hamming,
            'blackman': signal.windows.blackman,
            'bartlett': signal.windows.bartlett,
            'rectangular': lambda n: np.ones(n)
        }
    
    def dft(self, x: np.ndarray, normalized: bool = False) -> np.ndarray:
        """
        Discrete Fourier Transform
        
        Args:
            x: Input signal
            normalized: Whether to normalize by length
        
        Returns:
            DFT coefficients
        """
        X = fft.fft(x)
        
        if normalized:
            X = X / len(x)
        
        return X
    
    def idft(self, X: np.ndarray, normalized: bool = False) -> np.ndarray:
        """
        Inverse Discrete Fourier Transform
        
        Args:
            X: DFT coefficients
            normalized: Whether input was normalized
        
        Returns:
            Reconstructed signal
        """
        x = fft.ifft(X)
        
        if normalized:
            x = x * len(X)
        
        return x.real
    
    def power_spectral_density(
        self,
        x: np.ndarray,
        method: str = 'periodogram',
        window: str = 'hann',
        nperseg: Optional[int] = None,
        noverlap: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Power Spectral Density
        
        Args:
            x: Input signal
            method: Method ('periodogram', 'welch', 'multitaper')
            window: Window function name
            nperseg: Length of each segment (for Welch method)
            noverlap: Number of points to overlap between segments
        
        Returns:
            (frequencies, power_spectral_density)
        """
        
        if method == 'periodogram':
            freqs, psd = signal.periodogram(x, fs=self.sample_rate, window=window)
        
        elif method == 'welch':
            if nperseg is None:
                nperseg = min(len(x) // 4, 256)
            if noverlap is None:
                noverlap = nperseg // 2
            
            freqs, psd = signal.welch(
                x, fs=self.sample_rate, window=window,
                nperseg=nperseg, noverlap=noverlap
            )
        
        elif method == 'multitaper':
            # Use scipy's implementation
            freqs, psd = signal.periodogram(x, fs=self.sample_rate, window='dpss')
        
        else:
            raise ValueError(f"Unknown PSD method: {method}")
        
        return freqs, psd
    
    def cross_spectral_density(
        self,
        x: np.ndarray,
        y: np.ndarray,
        method: str = 'welch',
        window: str = 'hann',
        nperseg: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Cross Spectral Density between two signals
        
        Args:
            x, y: Input signals
            method: Method for estimation
            window: Window function
            nperseg: Segment length
        
        Returns:
            (frequencies, cross_spectral_density)
        """
        
        if method == 'welch':
            if nperseg is None:
                nperseg = min(len(x) // 4, 256)
            
            freqs, csd = signal.csd(
                x, y, fs=self.sample_rate, window=window, nperseg=nperseg
            )
        
        else:
            # Direct method
            X = fft.fft(x)
            Y = fft.fft(y)
            csd = X * np.conj(Y) / len(x)
            freqs = fft.fftfreq(len(x), 1/self.sample_rate)
        
        return freqs, csd
    
    def coherence(
        self,
        x: np.ndarray,
        y: np.ndarray,
        method: str = 'welch',
        nperseg: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute coherence between two signals
        
        Args:
            x, y: Input signals
            method: Estimation method
            nperseg: Segment length
        
        Returns:
            (frequencies, coherence)
        """
        
        if nperseg is None:
            nperseg = min(len(x) // 4, 256)
        
        freqs, coh = signal.coherence(x, y, fs=self.sample_rate, nperseg=nperseg)
        
        return freqs, coh
    
    def phase_spectrum(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute phase spectrum
        
        Args:
            x: Input signal
        
        Returns:
            (frequencies, phases)
        """
        
        X = fft.fft(x)
        freqs = fft.fftfreq(len(x), 1/self.sample_rate)
        phases = np.angle(X)
        
        return freqs, phases
    
    def group_delay(
        self,
        b: np.ndarray,
        a: Optional[np.ndarray] = None,
        w: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute group delay of a filter
        
        Args:
            b: Numerator coefficients
            a: Denominator coefficients
            w: Frequencies at which to evaluate
        
        Returns:
            (frequencies, group_delays)
        """
        
        if a is None:
            a = np.array([1.0])
        
        if w is None:
            w = np.linspace(0, np.pi, 512)
        
        w, gd = signal.group_delay((b, a), w)
        
        return w, gd


class SpectrogramAnalyzer:
    """
    Time-frequency analysis using spectrograms
    """
    
    def __init__(self, fs: float = 1.0):
        self.fs = fs
    
    def stft(
        self,
        x: np.ndarray,
        window: str = 'hann',
        nperseg: Optional[int] = None,
        noverlap: Optional[int] = None,
        nfft: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Short-Time Fourier Transform
        
        Args:
            x: Input signal
            window: Window function
            nperseg: Length of each segment
            noverlap: Number of points to overlap
            nfft: Length of FFT
        
        Returns:
            (frequencies, times, STFT_matrix)
        """
        
        if nperseg is None:
            nperseg = min(len(x) // 8, 256)
        
        if noverlap is None:
            noverlap = nperseg // 2
        
        freqs, times, Zxx = signal.stft(
            x, fs=self.fs, window=window,
            nperseg=nperseg, noverlap=noverlap, nfft=nfft
        )
        
        return freqs, times, Zxx
    
    def istft(
        self,
        Zxx: np.ndarray,
        window: str = 'hann',
        nperseg: Optional[int] = None,
        noverlap: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inverse Short-Time Fourier Transform
        
        Args:
            Zxx: STFT matrix
            window: Window function (must match forward STFT)
            nperseg: Segment length (must match forward STFT)
            noverlap: Overlap (must match forward STFT)
        
        Returns:
            (times, reconstructed_signal)
        """
        
        times, x_reconstructed = signal.istft(
            Zxx, fs=self.fs, window=window,
            nperseg=nperseg, noverlap=noverlap
        )
        
        return times, x_reconstructed
    
    def spectrogram(
        self,
        x: np.ndarray,
        window: str = 'hann',
        nperseg: Optional[int] = None,
        noverlap: Optional[int] = None,
        mode: str = 'psd'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute spectrogram
        
        Args:
            x: Input signal
            window: Window function
            nperseg: Length of each segment
            noverlap: Number of points to overlap
            mode: Type of spectrogram ('psd', 'magnitude', 'phase')
        
        Returns:
            (frequencies, times, spectrogram)
        """
        
        if nperseg is None:
            nperseg = min(len(x) // 8, 256)
        
        freqs, times, Sxx = signal.spectrogram(
            x, fs=self.fs, window=window,
            nperseg=nperseg, noverlap=noverlap, mode=mode
        )
        
        return freqs, times, Sxx
    
    def cross_spectrogram(
        self,
        x: np.ndarray,
        y: np.ndarray,
        window: str = 'hann',
        nperseg: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute cross-spectrogram between two signals
        
        Args:
            x, y: Input signals
            window: Window function
            nperseg: Segment length
        
        Returns:
            (frequencies, times, cross_spectrogram)
        """
        
        # Compute STFT of both signals
        freqs_x, times_x, Zxx = self.stft(x, window, nperseg)
        freqs_y, times_y, Zyy = self.stft(y, window, nperseg)
        
        # Cross-spectrogram
        cross_spec = Zxx * np.conj(Zyy)
        
        return freqs_x, times_x, cross_spec


class WaveletAnalyzer:
    """
    Wavelet analysis for time-frequency decomposition
    """
    
    def __init__(self):
        self.wavelet_functions = {
            'morlet': self._morlet_wavelet,
            'mexican_hat': self._mexican_hat_wavelet,
            'gabor': self._gabor_wavelet
        }
    
    def _morlet_wavelet(self, t: np.ndarray, f0: float = 1.0) -> np.ndarray:
        """Morlet wavelet"""
        return np.exp(2j * np.pi * f0 * t) * np.exp(-t**2 / 2)
    
    def _mexican_hat_wavelet(self, t: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """Mexican hat (Ricker) wavelet"""
        return (2 / (np.sqrt(3 * sigma) * np.pi**0.25)) * \
               (1 - (t / sigma)**2) * np.exp(-(t / sigma)**2 / 2)
    
    def _gabor_wavelet(self, t: np.ndarray, f0: float = 1.0, sigma: float = 1.0) -> np.ndarray:
        """Gabor wavelet"""
        return np.exp(2j * np.pi * f0 * t) * np.exp(-(t / sigma)**2)
    
    def continuous_wavelet_transform(
        self,
        x: np.ndarray,
        scales: np.ndarray,
        wavelet: str = 'morlet',
        dt: float = 1.0
    ) -> np.ndarray:
        """
        Continuous Wavelet Transform
        
        Args:
            x: Input signal
            scales: Scale parameters
            wavelet: Wavelet function name
            dt: Sampling interval
        
        Returns:
            CWT coefficients matrix (scales × time)
        """
        
        N = len(x)
        cwt_matrix = np.zeros((len(scales), N), dtype=complex)
        
        # Frequency domain representation of signal
        x_fft = fft.fft(x)
        freqs = fft.fftfreq(N, dt)
        
        # For each scale
        for i, scale in enumerate(scales):
            # Create wavelet in frequency domain
            if wavelet == 'morlet':
                # Morlet wavelet in frequency domain
                psi_fft = np.exp(-0.5 * (scale * freqs - 1)**2)
                psi_fft[freqs < 0] = 0  # Analytic wavelet
            
            elif wavelet == 'mexican_hat':
                # Mexican hat in frequency domain
                psi_fft = np.sqrt(scale) * (scale * freqs)**2 * \
                         np.exp(-0.5 * (scale * freqs)**2)
            
            else:
                raise ValueError(f"Wavelet {wavelet} not implemented")
            
            # Convolution in frequency domain
            cwt_fft = x_fft * np.conj(psi_fft)
            cwt_matrix[i, :] = fft.ifft(cwt_fft) / np.sqrt(scale)
        
        return cwt_matrix
    
    def wavelet_scalogram(
        self,
        cwt_coeffs: np.ndarray,
        method: str = 'power'
    ) -> np.ndarray:
        """
        Compute wavelet scalogram (time-scale energy distribution)
        
        Args:
            cwt_coeffs: CWT coefficients
            method: Type of scalogram ('power', 'magnitude')
        
        Returns:
            Scalogram matrix
        """
        
        if method == 'power':
            return np.abs(cwt_coeffs)**2
        elif method == 'magnitude':
            return np.abs(cwt_coeffs)
        else:
            raise ValueError(f"Unknown scalogram method: {method}")
    
    def ridge_extraction(
        self,
        cwt_coeffs: np.ndarray,
        scales: np.ndarray,
        threshold_ratio: float = 0.1
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Extract ridges from CWT coefficients
        
        Args:
            cwt_coeffs: CWT coefficients
            scales: Scale values
            threshold_ratio: Minimum ratio of peak to maximum
        
        Returns:
            List of (time_indices, scale_indices) for each ridge
        """
        
        scalogram = self.wavelet_scalogram(cwt_coeffs, method='power')
        max_val = np.max(scalogram)
        threshold = threshold_ratio * max_val
        
        # Find local maxima
        from scipy.ndimage import maximum_filter
        
        local_maxima = (scalogram == maximum_filter(scalogram, size=3)) & \
                      (scalogram > threshold)
        
        # Extract connected components as ridges
        ridge_points = np.where(local_maxima)
        
        # Simple ridge following (can be improved with more sophisticated algorithms)
        ridges = []
        used_points = set()
        
        for i, (scale_idx, time_idx) in enumerate(zip(ridge_points[0], ridge_points[1])):
            if (scale_idx, time_idx) in used_points:
                continue
            
            # Start a new ridge
            ridge_scales = [scale_idx]
            ridge_times = [time_idx]
            used_points.add((scale_idx, time_idx))
            
            # Follow ridge in both directions
            # (Simplified implementation)
            
            if len(ridge_scales) > 5:  # Minimum ridge length
                ridges.append((np.array(ridge_times), np.array(ridge_scales)))
        
        return ridges


class HilbertTransform:
    """
    Hilbert Transform for analytic signal analysis
    """
    
    def __init__(self):
        pass
    
    def hilbert_transform(self, x: np.ndarray) -> np.ndarray:
        """
        Compute Hilbert transform of signal
        
        Args:
            x: Real-valued input signal
        
        Returns:
            Hilbert transform
        """
        return signal.hilbert(x).imag
    
    def analytic_signal(self, x: np.ndarray) -> np.ndarray:
        """
        Compute analytic signal x(t) + j*H[x(t)]
        
        Args:
            x: Real-valued input signal
        
        Returns:
            Complex analytic signal
        """
        return signal.hilbert(x)
    
    def instantaneous_amplitude(self, x: np.ndarray) -> np.ndarray:
        """
        Compute instantaneous amplitude (envelope)
        
        Args:
            x: Input signal
        
        Returns:
            Instantaneous amplitude
        """
        analytic = self.analytic_signal(x)
        return np.abs(analytic)
    
    def instantaneous_phase(self, x: np.ndarray) -> np.ndarray:
        """
        Compute instantaneous phase
        
        Args:
            x: Input signal
        
        Returns:
            Instantaneous phase
        """
        analytic = self.analytic_signal(x)
        return np.angle(analytic)
    
    def instantaneous_frequency(
        self,
        x: np.ndarray,
        fs: float = 1.0,
        method: str = 'gradient'
    ) -> np.ndarray:
        """
        Compute instantaneous frequency
        
        Args:
            x: Input signal
            fs: Sampling frequency
            method: Method ('gradient', 'diff')
        
        Returns:
            Instantaneous frequency
        """
        
        phase = self.instantaneous_phase(x)
        
        if method == 'gradient':
            inst_freq = np.gradient(phase) * fs / (2 * np.pi)
        elif method == 'diff':
            inst_freq = np.diff(phase) * fs / (2 * np.pi)
            # Pad to maintain original length
            inst_freq = np.concatenate([[inst_freq[0]], inst_freq])
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return inst_freq
    
    def empirical_mode_decomposition(
        self,
        x: np.ndarray,
        max_imfs: int = 10,
        tolerance: float = 0.01
    ) -> List[np.ndarray]:
        """
        Empirical Mode Decomposition (simplified implementation)
        
        Args:
            x: Input signal
            max_imfs: Maximum number of IMFs
            tolerance: Stopping criterion tolerance
        
        Returns:
            List of Intrinsic Mode Functions (IMFs)
        """
        
        imfs = []
        residue = x.copy()
        
        for _ in range(max_imfs):
            # Sifting process
            h = residue.copy()
            
            for sift_iter in range(100):  # Maximum sifting iterations
                # Find local extrema
                from scipy.signal import find_peaks
                
                max_peaks, _ = find_peaks(h)
                min_peaks, _ = find_peaks(-h)
                
                if len(max_peaks) < 2 or len(min_peaks) < 2:
                    break
                
                # Interpolate envelopes
                try:
                    from scipy.interpolate import interp1d
                    
                    # Upper envelope
                    max_interp = interp1d(max_peaks, h[max_peaks], 
                                        kind='cubic', fill_value='extrapolate')
                    upper_env = max_interp(np.arange(len(h)))
                    
                    # Lower envelope
                    min_interp = interp1d(min_peaks, h[min_peaks], 
                                        kind='cubic', fill_value='extrapolate')
                    lower_env = min_interp(np.arange(len(h)))
                    
                    # Mean envelope
                    mean_env = (upper_env + lower_env) / 2
                    
                    # Update h
                    h_new = h - mean_env
                    
                    # Check stopping criterion
                    if np.sum((h - h_new)**2) / np.sum(h**2) < tolerance:
                        break
                    
                    h = h_new
                    
                except Exception:
                    # If interpolation fails, break
                    break
            
            # Add IMF
            imfs.append(h)
            
            # Update residue
            residue = residue - h
            
            # Check if residue is monotonic
            diff_residue = np.diff(residue)
            if np.all(diff_residue >= 0) or np.all(diff_residue <= 0):
                break
        
        # Add final residue
        imfs.append(residue)
        
        return imfs


class SpectralEstimation:
    """
    Advanced spectral estimation methods
    """
    
    def __init__(self):
        pass
    
    def ar_spectrum(
        self,
        x: np.ndarray,
        order: int,
        fs: float = 1.0,
        nfft: int = 512
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Autoregressive (AR) spectral estimation
        
        Args:
            x: Input signal
            order: AR model order
            fs: Sampling frequency
            nfft: Number of FFT points
        
        Returns:
            (frequencies, power_spectrum)
        """
        
        # Estimate AR parameters using Yule-Walker equations
        from scipy.linalg import toeplitz, solve
        
        # Compute autocorrelation
        autocorr = np.correlate(x, x, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]  # Normalize
        
        # Set up Yule-Walker equations
        R = toeplitz(autocorr[:order])
        r = autocorr[1:order+1]
        
        # Solve for AR coefficients
        try:
            ar_coeffs = solve(R, r)
            
            # Estimate noise variance
            sigma2 = autocorr[0] - np.dot(ar_coeffs, autocorr[1:order+1])
            
            # Compute spectrum
            freqs = np.linspace(0, fs/2, nfft//2 + 1)
            omega = 2 * np.pi * freqs / fs
            
            # Transfer function
            H = 1 / (1 - np.sum(ar_coeffs[:, None] * 
                               np.exp(-1j * omega * np.arange(1, order+1)[:, None]), axis=0))
            
            # Power spectrum
            psd = sigma2 * np.abs(H)**2
            
            return freqs, psd
            
        except Exception:
            # Fallback to periodogram
            return signal.periodogram(x, fs=fs, nfft=nfft)[:2]
    
    def music_spectrum(
        self,
        x: np.ndarray,
        n_signals: int,
        fs: float = 1.0,
        nfft: int = 512
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        MUSIC (Multiple Signal Classification) spectrum estimation
        
        Args:
            x: Input signal (assumed to be complex)
            n_signals: Number of signals
            fs: Sampling frequency
            nfft: Number of frequency points
        
        Returns:
            (frequencies, pseudo_spectrum)
        """
        
        # Form correlation matrix (simplified for 1D case)
        N = len(x)
        L = min(N//3, 50)  # Correlation matrix size
        
        # Create data matrix
        X = np.zeros((L, N-L+1), dtype=complex)
        for i in range(N-L+1):
            X[:, i] = x[i:i+L]
        
        # Correlation matrix
        R = X @ X.conj().T / (N-L+1)
        
        # Eigendecomposition
        eigenvals, eigenvecs = np.linalg.eigh(R)
        
        # Sort eigenvalues in descending order
        idx = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        
        # Noise subspace (smallest eigenvalues)
        noise_subspace = eigenvecs[:, n_signals:]
        
        # MUSIC pseudo-spectrum
        freqs = np.linspace(0, fs/2, nfft//2 + 1)
        omega = 2 * np.pi * freqs / fs
        
        # Steering vectors
        a_vectors = np.exp(1j * omega[:, None] * np.arange(L))
        
        # Pseudo-spectrum
        pseudo_spectrum = np.zeros(len(freqs))
        for i, a in enumerate(a_vectors):
            denominator = np.real(a.conj().T @ noise_subspace @ noise_subspace.conj().T @ a)
            pseudo_spectrum[i] = 1 / (denominator + 1e-12)  # Avoid division by zero
        
        return freqs, pseudo_spectrum


if __name__ == "__main__":
    # Example usage and testing
    print("Testing Fourier Analysis Engine...")
    
    # Create test signal
    fs = 1000  # Sampling frequency
    T = 1.0    # Duration
    t = np.linspace(0, T, int(fs * T), endpoint=False)
    
    # Multi-component signal
    f1, f2, f3 = 50, 120, 200
    signal_test = (np.sin(2 * np.pi * f1 * t) + 
                  0.5 * np.sin(2 * np.pi * f2 * t) + 
                  0.3 * np.sin(2 * np.pi * f3 * t) +
                  0.1 * np.random.randn(len(t)))  # Add noise
    
    # Test Fourier Analyzer
    print("\nTesting Fourier Analysis...")
    fourier = FourierAnalyzer()
    fourier.sample_rate = fs
    
    try:
        # Power spectral density
        freqs, psd = fourier.power_spectral_density(signal_test, method='welch')
        
        # Find peaks
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(psd, height=np.max(psd) * 0.1)
        peak_freqs = freqs[peaks]
        
        print(f"Detected peaks at frequencies: {peak_freqs[peak_freqs < fs/2][:3]}")
        print(f"Expected frequencies: {[f1, f2, f3]}")
        
        # Phase spectrum
        freqs_phase, phases = fourier.phase_spectrum(signal_test)
        print(f"Phase spectrum computed: {len(phases)} points")
        
    except Exception as e:
        print(f"Fourier analysis error: {e}")
    
    # Test Spectrogram
    print("\nTesting Spectrogram Analysis...")
    spectrogram = SpectrogramAnalyzer(fs=fs)
    
    try:
        # Create time-varying signal
        t_long = np.linspace(0, 2, int(fs * 2))
        chirp_signal = signal.chirp(t_long, f0=20, f1=200, t1=2, method='linear')
        
        freqs_spec, times_spec, Sxx = spectrogram.spectrogram(chirp_signal)
        print(f"Spectrogram shape: {Sxx.shape} (freq × time)")
        
        # STFT
        freqs_stft, times_stft, Zxx = spectrogram.stft(chirp_signal)
        print(f"STFT shape: {Zxx.shape}")
        
        # Inverse STFT
        times_reconstructed, signal_reconstructed = spectrogram.istft(Zxx)
        reconstruction_error = np.mean((chirp_signal - signal_reconstructed[:len(chirp_signal)])**2)
        print(f"STFT reconstruction MSE: {reconstruction_error:.2e}")
        
    except Exception as e:
        print(f"Spectrogram analysis error: {e}")
    
    # Test Wavelet Analysis
    print("\nTesting Wavelet Analysis...")
    wavelet = WaveletAnalyzer()
    
    try:
        # Scales for CWT
        scales = np.logspace(0, 3, 50)  # Log-spaced scales
        
        # CWT
        cwt_coeffs = wavelet.continuous_wavelet_transform(
            signal_test, scales, wavelet='morlet', dt=1/fs
        )
        print(f"CWT coefficients shape: {cwt_coeffs.shape}")
        
        # Scalogram
        scalogram = wavelet.wavelet_scalogram(cwt_coeffs, method='power')
        print(f"Scalogram computed: max power = {np.max(scalogram):.2e}")
        
    except Exception as e:
        print(f"Wavelet analysis error: {e}")
    
    # Test Hilbert Transform
    print("\nTesting Hilbert Transform...")
    hilbert = HilbertTransform()
    
    try:
        # Test with amplitude modulated signal
        t_am = np.linspace(0, 1, 1000)
        carrier_freq = 100
        mod_freq = 10
        am_signal = (1 + 0.5 * np.cos(2 * np.pi * mod_freq * t_am)) * \
                   np.cos(2 * np.pi * carrier_freq * t_am)
        
        # Instantaneous amplitude (envelope)
        envelope = hilbert.instantaneous_amplitude(am_signal)
        
        # Theoretical envelope
        theoretical_envelope = 1 + 0.5 * np.cos(2 * np.pi * mod_freq * t_am)
        
        envelope_error = np.mean((envelope - theoretical_envelope)**2)
        print(f"Envelope extraction MSE: {envelope_error:.4f}")
        
        # Instantaneous frequency
        inst_freq = hilbert.instantaneous_frequency(am_signal, fs=1000)
        print(f"Mean instantaneous frequency: {np.mean(inst_freq):.1f} Hz (expected ~{carrier_freq} Hz)")
        
    except Exception as e:
        print(f"Hilbert transform error: {e}")
    
    # Test Spectral Estimation
    print("\nTesting Spectral Estimation...")
    spectral = SpectralEstimation()
    
    try:
        # AR spectrum
        freqs_ar, psd_ar = spectral.ar_spectrum(signal_test, order=20, fs=fs)
        
        # Find peaks in AR spectrum
        peaks_ar, _ = find_peaks(psd_ar, height=np.max(psd_ar) * 0.1)
        peak_freqs_ar = freqs_ar[peaks_ar]
        
        print(f"AR spectrum peaks: {peak_freqs_ar[:3]}")
        
        # Compare with periodogram
        freqs_per, psd_per = signal.periodogram(signal_test, fs=fs)
        resolution_improvement = len(freqs_ar) / len(freqs_per)
        print(f"AR method resolution improvement: {resolution_improvement:.1f}x")
        
    except Exception as e:
        print(f"Spectral estimation error: {e}")
    
    print("\nFourier analysis engine testing completed!")