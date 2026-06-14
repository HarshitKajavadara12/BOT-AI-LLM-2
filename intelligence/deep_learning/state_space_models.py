"""
State Space Models for Financial Time Series
Advanced state space architectures including Mamba, S4, and custom financial models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np
import math
from typing import Optional, Tuple, Dict, Any, Union
from scipy import signal
from scipy.linalg import solve_continuous_are


class StateSpaceModel(nn.Module):
    """
    Base class for state space models
    Implements the continuous-time state space equation:
    dx/dt = Ax + Bu
    y = Cx + Du
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int,
        dropout: float = 0.1,
        discretization: str = 'zoh'  # zero-order hold, bilinear, euler
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.discretization = discretization
        
        # Learnable state space matrices
        self.A = Parameter(torch.randn(d_state, d_state) * 0.1)
        self.B = Parameter(torch.randn(d_state, d_model) * 0.1)
        self.C = Parameter(torch.randn(d_model, d_state) * 0.1)
        self.D = Parameter(torch.randn(d_model, d_model) * 0.1)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def discretize(self, dt: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Discretize continuous state space model"""
        A, B = self.A, self.B
        
        if self.discretization == 'zoh':  # Zero-order hold
            # Matrix exponential approach
            # [A_d, B_d] = expm([[A, B], [0, 0]]) * dt
            
            # Approximation using series expansion for efficiency
            I = torch.eye(A.size(0), device=A.device, dtype=A.dtype)
            A_d = I + A * dt + 0.5 * torch.mm(A, A) * dt**2
            B_d = B * dt + 0.5 * torch.mm(A, B) * dt**2
            
        elif self.discretization == 'bilinear':  # Tustin's method
            I = torch.eye(A.size(0), device=A.device, dtype=A.dtype)
            temp = I - A * dt / 2
            A_d = torch.solve(I + A * dt / 2, temp)[0]
            B_d = torch.solve(B * dt / 2, temp)[0]
            
        elif self.discretization == 'euler':  # Forward Euler
            I = torch.eye(A.size(0), device=A.device, dtype=A.dtype)
            A_d = I + A * dt
            B_d = B * dt
            
        else:
            raise ValueError(f"Unknown discretization method: {self.discretization}")
        
        return A_d, B_d
    
    def forward(
        self,
        u: torch.Tensor,
        dt: float = 1.0,
        initial_state: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through state space model
        
        Args:
            u: Input sequence [batch_size, seq_len, d_model]
            dt: Time step for discretization
            initial_state: Initial state [batch_size, d_state]
        
        Returns:
            Output sequence [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = u.shape
        
        # Discretize the system
        A_d, B_d = self.discretize(dt)
        
        # Initialize state
        if initial_state is None:
            h = torch.zeros(batch_size, self.d_state, device=u.device, dtype=u.dtype)
        else:
            h = initial_state
        
        outputs = []
        for t in range(seq_len):
            # State update: h[t+1] = A_d * h[t] + B_d * u[t]
            h = torch.mm(h, A_d.T) + torch.mm(u[:, t, :], B_d.T)
            
            # Output: y[t] = C * h[t] + D * u[t]
            y = torch.mm(h, self.C.T) + torch.mm(u[:, t, :], self.D.T)
            outputs.append(y)
        
        output = torch.stack(outputs, dim=1)
        output = self.dropout(output)
        
        # Residual connection and layer norm
        return self.layer_norm(output + u)


class S4Layer(nn.Module):
    """
    Structured State Space (S4) Layer
    Based on "Efficiently Modeling Long Sequences with Structured State Spaces"
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        dropout: float = 0.1,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        lr_dt: float = 1e-3
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # HiPPO matrix initialization for A
        A = self._init_hippo_matrix(d_state)
        self.register_buffer('A', A)
        
        # Learnable parameters
        self.B = Parameter(torch.randn(d_state, d_model))
        self.C = Parameter(torch.randn(d_model, d_state))
        self.D = Parameter(torch.randn(d_model))
        
        # Time step parameter
        self.log_dt = Parameter(
            torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        
        # Normalization and dropout
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def _init_hippo_matrix(self, N: int) -> torch.Tensor:
        """Initialize HiPPO matrix for long-range dependencies"""
        A = torch.zeros(N, N)
        
        for i in range(N):
            for j in range(N):
                if i > j:
                    A[i, j] = math.sqrt(2 * i + 1) * math.sqrt(2 * j + 1)
                elif i == j:
                    A[i, j] = i + 1
                else:
                    A[i, j] = -math.sqrt(2 * i + 1) * math.sqrt(2 * j + 1)
        
        return -A
    
    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through S4 layer
        
        Args:
            u: Input sequence [batch_size, seq_len, d_model]
        
        Returns:
            Output sequence [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = u.shape
        dt = torch.exp(self.log_dt)  # [d_model]
        
        # Discretization using bilinear transform
        A_discrete, B_discrete = self._discretize_s4(dt)
        
        # Compute convolution kernel
        kernel = self._compute_kernel(A_discrete, B_discrete, seq_len)
        
        # Apply convolution
        u_fft = torch.fft.rfft(u.transpose(-1, -2), n=2*seq_len, dim=-1)
        kernel_fft = torch.fft.rfft(kernel, n=2*seq_len, dim=-1)
        
        output_fft = u_fft * kernel_fft
        output = torch.fft.irfft(output_fft, n=2*seq_len, dim=-1)[..., :seq_len]
        output = output.transpose(-1, -2)
        
        # Add skip connection
        output = output + u * self.D
        
        # Normalization and dropout
        output = self.dropout(output)
        return self.layer_norm(output + u)
    
    def _discretize_s4(self, dt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Discretize S4 system using bilinear transform"""
        # Bilinear transform
        I = torch.eye(self.d_state, device=self.A.device)
        
        # Vectorized discretization for each dt
        A_discrete = []
        B_discrete = []
        
        for i in range(self.d_model):
            dt_i = dt[i]
            temp = I - self.A * dt_i / 2
            A_d = torch.linalg.solve(temp, I + self.A * dt_i / 2)
            B_d = torch.linalg.solve(temp, self.B * dt_i)
            
            A_discrete.append(A_d)
            B_discrete.append(B_d[:, i:i+1])
        
        return A_discrete, B_discrete
    
    def _compute_kernel(
        self,
        A_discrete: list,
        B_discrete: list,
        seq_len: int
    ) -> torch.Tensor:
        """Compute convolution kernel for each channel"""
        kernels = []
        
        for i in range(self.d_model):
            A_d = A_discrete[i]
            B_d = B_discrete[i].squeeze(-1)
            C_i = self.C[i, :]
            
            # Compute powers of A_d
            A_powers = [torch.eye(self.d_state, device=A_d.device)]
            for _ in range(seq_len - 1):
                A_powers.append(A_powers[-1] @ A_d)
            
            # Compute kernel: C * A^k * B for k = 0, 1, ..., seq_len-1
            kernel_i = torch.stack([C_i @ A_k @ B_d for A_k in A_powers])
            kernels.append(kernel_i)
        
        return torch.stack(kernels, dim=0)  # [d_model, seq_len]


class MambaBlock(nn.Module):
    """
    Mamba block for selective state space modeling
    Based on "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Union[int, str] = 'auto',
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == 'auto' else dt_rank
        
        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # Convolution for local dependencies
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=True,
            groups=self.d_inner,
            padding=d_conv - 1
        )
        
        # Selective scan parameters
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        # State space parameters
        A_log = torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1))
        self.A_log = Parameter(A_log)
        self.D = Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Mamba block
        
        Args:
            x: Input sequence [batch_size, seq_len, d_model]
        
        Returns:
            Output sequence [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        
        # Input projection and gating
        x_and_res = self.in_proj(x)  # [batch, seq_len, 2 * d_inner]
        x_proj, res = x_and_res.split(self.d_inner, dim=-1)
        
        # Convolution for short-range dependencies
        x_conv = self.conv1d(x_proj.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
        x_conv = F.silu(x_conv)
        
        # Selective scan
        x_ssm = self.selective_scan(x_conv)
        
        # Gating and output projection
        y = x_ssm * F.silu(res)
        output = self.out_proj(y)
        
        return self.dropout(output)
    
    def selective_scan(self, x: torch.Tensor) -> torch.Tensor:
        """Apply selective state space scan"""
        batch_size, seq_len, d_inner = x.shape
        
        # Generate selective parameters
        x_dbl = self.x_proj(x)  # [batch, seq_len, dt_rank + 2*d_state]
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        # Time step
        dt = F.softplus(self.dt_proj(dt))  # [batch, seq_len, d_inner]
        
        # State space matrices
        A = -torch.exp(self.A_log.float())  # [d_inner, d_state]
        
        # Selective scan implementation
        h = torch.zeros(batch_size, d_inner, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []
        
        for t in range(seq_len):
            dt_t = dt[:, t, :].unsqueeze(-1)  # [batch, d_inner, 1]
            B_t = B[:, t, :].unsqueeze(1)     # [batch, 1, d_state]
            C_t = C[:, t, :].unsqueeze(1)     # [batch, 1, d_state]
            x_t = x[:, t, :].unsqueeze(-1)    # [batch, d_inner, 1]
            
            # Discretize
            dA = torch.exp(dt_t * A)  # [batch, d_inner, d_state]
            dB = dt_t * B_t           # [batch, d_inner, d_state]
            
            # State update
            h = h * dA + dB * x_t
            
            # Output
            y_t = torch.sum(h * C_t, dim=-1)  # [batch, d_inner]
            outputs.append(y_t)
        
        y = torch.stack(outputs, dim=1)  # [batch, seq_len, d_inner]
        
        # Add skip connection
        y = y + x * self.D
        
        return y


class LinearStateSpaceModel(nn.Module):
    """
    Linear State Space Model with learnable dynamics
    Optimized for financial time series with regime changes
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int,
        num_regimes: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.num_regimes = num_regimes
        
        # Regime-specific state space matrices
        self.A_matrices = Parameter(torch.randn(num_regimes, d_state, d_state) * 0.1)
        self.B_matrices = Parameter(torch.randn(num_regimes, d_state, d_model) * 0.1)
        self.C_matrices = Parameter(torch.randn(num_regimes, d_model, d_state) * 0.1)
        
        # Regime classifier
        self.regime_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_regimes)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Classify regime for each time step
        regime_logits = self.regime_classifier(x)  # [batch, seq_len, num_regimes]
        regime_probs = F.softmax(regime_logits, dim=-1)
        
        # Initialize state
        h = torch.zeros(batch_size, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []
        
        for t in range(seq_len):
            # Weighted combination of regime-specific matrices
            A_t = torch.sum(
                regime_probs[:, t, :].unsqueeze(-1).unsqueeze(-1) * self.A_matrices,
                dim=1
            )  # [batch, d_state, d_state]
            
            B_t = torch.sum(
                regime_probs[:, t, :].unsqueeze(-1).unsqueeze(-1) * self.B_matrices,
                dim=1
            )  # [batch, d_state, d_model]
            
            C_t = torch.sum(
                regime_probs[:, t, :].unsqueeze(-1).unsqueeze(-1) * self.C_matrices,
                dim=1
            )  # [batch, d_model, d_state]
            
            # State update
            h = torch.bmm(h.unsqueeze(1), A_t).squeeze(1) + \
                torch.bmm(x[:, t, :].unsqueeze(1), B_t.transpose(1, 2)).squeeze(1)
            
            # Output
            y_t = torch.bmm(h.unsqueeze(1), C_t.transpose(1, 2)).squeeze(1)
            outputs.append(y_t)
        
        output = torch.stack(outputs, dim=1)
        output = self.dropout(output)
        
        return self.layer_norm(output + x)


class KalmanFilterLayer(nn.Module):
    """
    Differentiable Kalman Filter for state estimation
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_obs: Optional[int] = None,
        learnable_noise: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_obs = d_obs or d_model
        
        # State transition model
        self.F = Parameter(torch.eye(d_state) + torch.randn(d_state, d_state) * 0.1)
        self.B = Parameter(torch.randn(d_state, d_model) * 0.1)
        
        # Observation model
        self.H = Parameter(torch.randn(self.d_obs, d_state) * 0.1)
        
        # Noise covariances (learnable if specified)
        if learnable_noise:
            self.Q = Parameter(torch.eye(d_state) * 0.1)  # Process noise
            self.R = Parameter(torch.eye(self.d_obs) * 0.1)  # Observation noise
        else:
            self.register_buffer('Q', torch.eye(d_state) * 0.1)
            self.register_buffer('R', torch.eye(self.d_obs) * 0.1)
        
        # Initial state and covariance
        self.x0 = Parameter(torch.zeros(d_state))
        self.P0 = Parameter(torch.eye(d_state))
        
    def forward(
        self,
        observations: torch.Tensor,
        control_inputs: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through Kalman filter
        
        Args:
            observations: Observation sequence [batch_size, seq_len, d_obs]
            control_inputs: Control inputs [batch_size, seq_len, d_model]
        
        Returns:
            state_estimates: Filtered state estimates [batch_size, seq_len, d_state]
            covariances: State covariance matrices [batch_size, seq_len, d_state, d_state]
        """
        batch_size, seq_len, _ = observations.shape
        
        # Initialize
        x = self.x0.unsqueeze(0).expand(batch_size, -1)  # [batch, d_state]
        P = self.P0.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, d_state, d_state]
        
        state_estimates = []
        covariances = []
        
        for t in range(seq_len):
            # Prediction step
            if control_inputs is not None:
                u_t = control_inputs[:, t, :]
                x_pred = torch.mm(x, self.F.T) + torch.mm(u_t, self.B.T)
            else:
                x_pred = torch.mm(x, self.F.T)
            
            P_pred = torch.bmm(
                torch.bmm(self.F.unsqueeze(0).expand(batch_size, -1, -1), P),
                self.F.T.unsqueeze(0).expand(batch_size, -1, -1)
            ) + self.Q.unsqueeze(0).expand(batch_size, -1, -1)
            
            # Update step
            z_t = observations[:, t, :]  # Current observation
            
            # Innovation
            y_t = z_t - torch.mm(x_pred, self.H.T)
            
            # Innovation covariance
            S_t = torch.bmm(
                torch.bmm(
                    self.H.unsqueeze(0).expand(batch_size, -1, -1),
                    P_pred
                ),
                self.H.T.unsqueeze(0).expand(batch_size, -1, -1)
            ) + self.R.unsqueeze(0).expand(batch_size, -1, -1)
            
            # Kalman gain
            K_t = torch.bmm(
                torch.bmm(P_pred, self.H.T.unsqueeze(0).expand(batch_size, -1, -1)),
                torch.linalg.inv(S_t)
            )
            
            # State update
            x = x_pred + torch.bmm(y_t.unsqueeze(1), K_t.transpose(1, 2)).squeeze(1)
            
            # Covariance update
            I_KH = torch.eye(self.d_state, device=P.device).unsqueeze(0).expand(batch_size, -1, -1) - \
                   torch.bmm(K_t, self.H.unsqueeze(0).expand(batch_size, -1, -1))
            P = torch.bmm(I_KH, P_pred)
            
            state_estimates.append(x)
            covariances.append(P)
        
        return torch.stack(state_estimates, dim=1), torch.stack(covariances, dim=1)


def create_state_space_model(
    model_type: str,
    d_model: int,
    **kwargs
) -> nn.Module:
    """
    Factory function to create state space models
    
    Args:
        model_type: Type of model ('basic', 's4', 'mamba', 'linear_ssm', 'kalman')
        d_model: Model dimension
        **kwargs: Additional model-specific parameters
    
    Returns:
        Configured state space model
    """
    
    if model_type == 'basic':
        return StateSpaceModel(d_model, **kwargs)
    elif model_type == 's4':
        return S4Layer(d_model, **kwargs)
    elif model_type == 'mamba':
        return MambaBlock(d_model, **kwargs)
    elif model_type == 'linear_ssm':
        return LinearStateSpaceModel(d_model, **kwargs)
    elif model_type == 'kalman':
        return KalmanFilterLayer(d_model, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Example usage
    batch_size, seq_len, d_model = 32, 100, 128
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Test different state space models
    models = {
        'basic': StateSpaceModel(d_model, d_state=64),
        's4': S4Layer(d_model, d_state=64),
        'mamba': MambaBlock(d_model, d_state=16),
        'linear_ssm': LinearStateSpaceModel(d_model, d_state=32)
    }
    
    for name, model in models.items():
        output = model(x)
        print(f"{name}: Input {x.shape} -> Output {output.shape}")
    
    # Test Kalman filter
    kalman = KalmanFilterLayer(d_model, d_state=32)
    states, covariances = kalman(x)
    print(f"Kalman: Input {x.shape} -> States {states.shape}, Covariances {covariances.shape}")
