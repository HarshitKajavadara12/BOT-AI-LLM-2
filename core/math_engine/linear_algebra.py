"""
Linear Algebra Engine
Advanced linear algebra operations for quantitative finance
"""

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from typing import Optional, Tuple, Dict, Any, List, Union, Callable
from abc import ABC, abstractmethod
import warnings
from numba import jit, prange
import matplotlib.pyplot as plt


class LinearAlgebraEngine:
    """
    Comprehensive linear algebra engine for financial computations
    """
    
    def __init__(self):
        self.precision_threshold = 1e-12
        self.max_iterations = 1000
        self.convergence_tolerance = 1e-10
    
    # Matrix Decompositions
    def lu_decomposition(
        self,
        A: np.ndarray,
        permute_l: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        LU decomposition with partial pivoting
        
        Args:
            A: Input matrix
            permute_l: Whether to permute L matrix
        
        Returns:
            (P, L, U) or (L, U) depending on permute_l
        """
        
        if permute_l:
            L, U = la.lu(A, permute_l=True)
            return L, U, None
        else:
            P, L, U = la.lu(A, permute_l=False)
            return P, L, U
    
    def qr_decomposition(
        self,
        A: np.ndarray,
        mode: str = 'reduced',
        pivoting: bool = False
    ) -> Union[Tuple[np.ndarray, np.ndarray], 
               Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        QR decomposition
        
        Args:
            A: Input matrix
            mode: 'reduced', 'complete', or 'economic'
            pivoting: Whether to use column pivoting
        
        Returns:
            (Q, R) or (Q, R, P) if pivoting
        """
        
        if pivoting:
            Q, R, P = la.qr(A, pivoting=True, mode=mode)
            return Q, R, P
        else:
            Q, R = la.qr(A, mode=mode)
            return Q, R
    
    def svd_decomposition(
        self,
        A: np.ndarray,
        full_matrices: bool = True,
        compute_uv: bool = True
    ) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
        """
        Singular Value Decomposition
        
        Args:
            A: Input matrix
            full_matrices: Whether to compute full or reduced matrices
            compute_uv: Whether to compute U and Vt matrices
        
        Returns:
            (U, s, Vt) or s depending on compute_uv
        """
        
        return la.svd(A, full_matrices=full_matrices, compute_uv=compute_uv)
    
    def eigendecomposition(
        self,
        A: np.ndarray,
        hermitian: bool = False,
        eigvals_only: bool = False
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        Eigenvalue decomposition
        
        Args:
            A: Input matrix
            hermitian: Whether matrix is Hermitian/symmetric
            eigvals_only: Whether to compute only eigenvalues
        
        Returns:
            (eigenvalues, eigenvectors) or eigenvalues only
        """
        
        if hermitian:
            if eigvals_only:
                return la.eigvalsh(A)
            else:
                return la.eigh(A)
        else:
            if eigvals_only:
                return la.eigvals(A)
            else:
                return la.eig(A)
    
    def cholesky_decomposition(
        self,
        A: np.ndarray,
        lower: bool = True,
        check_finite: bool = True
    ) -> np.ndarray:
        """
        Cholesky decomposition for positive definite matrices
        
        Args:
            A: Positive definite matrix
            lower: Whether to compute lower triangular matrix
            check_finite: Whether to check for finite values
        
        Returns:
            Cholesky factor
        """
        
        try:
            return la.cholesky(A, lower=lower, check_finite=check_finite)
        except la.LinAlgError as e:
            # Try regularization if not positive definite
            eigenvals = la.eigvalsh(A)
            min_eigenval = np.min(eigenvals)
            
            if min_eigenval <= 0:
                reg_factor = np.abs(min_eigenval) + 1e-6
                A_reg = A + reg_factor * np.eye(A.shape[0])
                warnings.warn(f"Matrix regularized with factor {reg_factor}")
                return la.cholesky(A_reg, lower=lower, check_finite=check_finite)
            else:
                raise e
    
    def schur_decomposition(
        self,
        A: np.ndarray,
        output: str = 'real'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Schur decomposition
        
        Args:
            A: Input matrix
            output: 'real' or 'complex'
        
        Returns:
            (T, Z) where A = Z @ T @ Z.T
        """
        
        return la.schur(A, output=output)
    
    # Matrix Operations
    def matrix_inverse(
        self,
        A: np.ndarray,
        method: str = 'auto',
        check_finite: bool = True
    ) -> np.ndarray:
        """
        Matrix inversion with multiple methods
        
        Args:
            A: Input matrix
            method: 'auto', 'lu', 'cholesky', 'svd'
            check_finite: Whether to check for finite values
        
        Returns:
            Inverse matrix
        """
        
        if method == 'auto':
            # Automatically choose best method
            if self.is_positive_definite(A):
                method = 'cholesky'
            elif A.shape[0] == A.shape[1] and np.linalg.cond(A) < 1e12:
                method = 'lu'
            else:
                method = 'svd'
        
        if method == 'lu':
            return la.inv(A, check_finite=check_finite)
        
        elif method == 'cholesky':
            try:
                L = self.cholesky_decomposition(A, lower=True)
                return la.cho_solve((L, True), np.eye(A.shape[0]))
            except:
                # Fallback to LU
                return la.inv(A, check_finite=check_finite)
        
        elif method == 'svd':
            U, s, Vt = self.svd_decomposition(A)
            # Pseudo-inverse using SVD
            s_inv = np.where(s > self.precision_threshold, 1/s, 0)
            return Vt.T @ np.diag(s_inv) @ U.T
        
        else:
            raise ValueError(f"Unknown inversion method: {method}")
    
    def matrix_sqrt(
        self,
        A: np.ndarray,
        method: str = 'eigen'
    ) -> np.ndarray:
        """
        Matrix square root
        
        Args:
            A: Positive semidefinite matrix
            method: 'eigen', 'cholesky', 'schur'
        
        Returns:
            Matrix square root
        """
        
        if method == 'eigen':
            eigenvals, eigenvecs = self.eigendecomposition(A, hermitian=True)
            sqrt_eigenvals = np.maximum(eigenvals, 0) ** 0.5
            return eigenvecs @ np.diag(sqrt_eigenvals) @ eigenvecs.T
        
        elif method == 'cholesky':
            try:
                return self.cholesky_decomposition(A, lower=True)
            except:
                # Fallback to eigendecomposition
                return self.matrix_sqrt(A, method='eigen')
        
        elif method == 'schur':
            return la.sqrtm(A)
        
        else:
            raise ValueError(f"Unknown matrix sqrt method: {method}")
    
    def matrix_exponential(
        self,
        A: np.ndarray,
        method: str = 'pade'
    ) -> np.ndarray:
        """
        Matrix exponential
        
        Args:
            A: Input matrix
            method: 'pade', 'eigen', 'taylor'
        
        Returns:
            Matrix exponential exp(A)
        """
        
        if method == 'pade':
            return la.expm(A)
        
        elif method == 'eigen':
            eigenvals, eigenvecs = self.eigendecomposition(A)
            exp_eigenvals = np.exp(eigenvals)
            return eigenvecs @ np.diag(exp_eigenvals) @ la.inv(eigenvecs)
        
        elif method == 'taylor':
            # Taylor series approximation
            n = A.shape[0]
            result = np.eye(n)
            term = np.eye(n)
            
            for k in range(1, 50):  # Sufficient for most cases
                term = term @ A / k
                result += term
                
                if np.max(np.abs(term)) < self.precision_threshold:
                    break
            
            return result
        
        else:
            raise ValueError(f"Unknown matrix exponential method: {method}")
    
    def matrix_logarithm(
        self,
        A: np.ndarray,
        method: str = 'schur'
    ) -> np.ndarray:
        """
        Matrix logarithm
        
        Args:
            A: Input matrix (should be non-singular)
            method: 'schur', 'eigen', 'pade'
        
        Returns:
            Matrix logarithm log(A)
        """
        
        if method == 'schur':
            return la.logm(A)
        
        elif method == 'eigen':
            eigenvals, eigenvecs = self.eigendecomposition(A)
            log_eigenvals = np.log(eigenvals)
            return eigenvecs @ np.diag(log_eigenvals) @ la.inv(eigenvecs)
        
        else:
            return la.logm(A)
    
    def matrix_power(
        self,
        A: np.ndarray,
        p: float,
        method: str = 'eigen'
    ) -> np.ndarray:
        """
        Matrix power A^p
        
        Args:
            A: Input matrix
            p: Power (can be fractional)
            method: 'eigen', 'schur'
        
        Returns:
            Matrix power A^p
        """
        
        if method == 'eigen':
            eigenvals, eigenvecs = self.eigendecomposition(A)
            power_eigenvals = eigenvals ** p
            return eigenvecs @ np.diag(power_eigenvals) @ la.inv(eigenvecs)
        
        elif method == 'schur':
            if p == int(p):
                # Integer power - use repeated multiplication
                result = np.eye(A.shape[0])
                base = A.copy()
                p_int = int(abs(p))
                
                while p_int > 0:
                    if p_int % 2 == 1:
                        result = result @ base
                    base = base @ base
                    p_int //= 2
                
                if p < 0:
                    result = self.matrix_inverse(result)
                
                return result
            else:
                # Fractional power using matrix functions
                return self.matrix_exponential(p * self.matrix_logarithm(A))
        
        else:
            raise ValueError(f"Unknown matrix power method: {method}")
    
    # Linear Systems
    def solve_linear_system(
        self,
        A: np.ndarray,
        b: np.ndarray,
        method: str = 'auto',
        assume_a: str = 'gen'
    ) -> np.ndarray:
        """
        Solve linear system Ax = b
        
        Args:
            A: Coefficient matrix
            b: Right-hand side
            method: 'auto', 'lu', 'cholesky', 'qr', 'svd'
            assume_a: Assumption about A ('gen', 'sym', 'pos', 'her')
        
        Returns:
            Solution vector x
        """
        
        if method == 'auto':
            if assume_a in ['pos', 'sym'] or self.is_positive_definite(A):
                method = 'cholesky'
            elif A.shape[0] == A.shape[1]:
                method = 'lu'
            else:
                method = 'qr'
        
        if method == 'lu':
            return la.solve(A, b, assume_a=assume_a)
        
        elif method == 'cholesky':
            try:
                return la.solve(A, b, assume_a='pos')
            except:
                # Fallback to LU
                return la.solve(A, b)
        
        elif method == 'qr':
            Q, R = self.qr_decomposition(A, mode='reduced')
            return la.solve_triangular(R, Q.T @ b)
        
        elif method == 'svd':
            U, s, Vt = self.svd_decomposition(A)
            # Pseudo-inverse solution
            s_inv = np.where(s > self.precision_threshold, 1/s, 0)
            return Vt.T @ (s_inv * (U.T @ b))
        
        else:
            raise ValueError(f"Unknown solver method: {method}")
    
    def solve_least_squares(
        self,
        A: np.ndarray,
        b: np.ndarray,
        method: str = 'qr',
        rcond: Optional[float] = None
    ) -> Tuple[np.ndarray, float, int, np.ndarray]:
        """
        Solve least squares problem min ||Ax - b||^2
        
        Args:
            A: Design matrix
            b: Target vector
            method: 'qr', 'svd', 'normal'
            rcond: Cutoff for small singular values
        
        Returns:
            (solution, residuals, rank, singular_values)
        """
        
        if method == 'svd':
            return la.lstsq(A, b, rcond=rcond)
        
        elif method == 'qr':
            Q, R = self.qr_decomposition(A, mode='reduced')
            x = la.solve_triangular(R, Q.T @ b)
            residuals = np.linalg.norm(b - A @ x) ** 2
            rank = np.linalg.matrix_rank(A)
            return x, residuals, rank, la.svdvals(A)
        
        elif method == 'normal':
            # Normal equations (less stable but faster)
            AtA = A.T @ A
            Atb = A.T @ b
            x = self.solve_linear_system(AtA, Atb, method='cholesky')
            residuals = np.linalg.norm(b - A @ x) ** 2
            rank = np.linalg.matrix_rank(A)
            return x, residuals, rank, la.svdvals(A)
        
        else:
            raise ValueError(f"Unknown least squares method: {method}")
    
    # Matrix Properties
    def is_positive_definite(self, A: np.ndarray, tol: float = 1e-8) -> bool:
        """Check if matrix is positive definite"""
        try:
            eigenvals = la.eigvalsh(A)
            return np.all(eigenvals > tol)
        except:
            return False
    
    def is_positive_semidefinite(self, A: np.ndarray, tol: float = 1e-8) -> bool:
        """Check if matrix is positive semidefinite"""
        try:
            eigenvals = la.eigvalsh(A)
            return np.all(eigenvals >= -tol)
        except:
            return False
    
    def matrix_rank(self, A: np.ndarray, tol: Optional[float] = None) -> int:
        """Compute matrix rank"""
        return np.linalg.matrix_rank(A, tol=tol)
    
    def condition_number(
        self,
        A: np.ndarray,
        p: Union[None, int, str] = None
    ) -> float:
        """Compute condition number"""
        return np.linalg.cond(A, p=p)
    
    def matrix_norm(
        self,
        A: np.ndarray,
        ord: Union[None, int, str] = 'fro'
    ) -> float:
        """Compute matrix norm"""
        return np.linalg.norm(A, ord=ord)
    
    def trace(self, A: np.ndarray) -> float:
        """Compute matrix trace"""
        return np.trace(A)
    
    def determinant(self, A: np.ndarray) -> float:
        """Compute determinant"""
        return la.det(A)
    
    # Iterative Methods
    def conjugate_gradient(
        self,
        A: np.ndarray,
        b: np.ndarray,
        x0: Optional[np.ndarray] = None,
        tol: float = 1e-6,
        maxiter: Optional[int] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Conjugate Gradient method for positive definite systems
        
        Args:
            A: Positive definite matrix
            b: Right-hand side
            x0: Initial guess
            tol: Convergence tolerance
            maxiter: Maximum iterations
        
        Returns:
            (solution, num_iterations)
        """
        
        n = len(b)
        if x0 is None:
            x = np.zeros(n)
        else:
            x = x0.copy()
        
        if maxiter is None:
            maxiter = n
        
        r = b - A @ x
        p = r.copy()
        rsold = r @ r
        
        for i in range(maxiter):
            Ap = A @ p
            alpha = rsold / (p @ Ap)
            x = x + alpha * p
            r = r - alpha * Ap
            rsnew = r @ r
            
            if np.sqrt(rsnew) < tol:
                return x, i + 1
            
            beta = rsnew / rsold
            p = r + beta * p
            rsold = rsnew
        
        return x, maxiter
    
    def gmres(
        self,
        A: np.ndarray,
        b: np.ndarray,
        x0: Optional[np.ndarray] = None,
        tol: float = 1e-6,
        maxiter: Optional[int] = None,
        restart: Optional[int] = None
    ) -> Tuple[np.ndarray, int]:
        """
        GMRES method for general linear systems
        
        Args:
            A: Coefficient matrix
            b: Right-hand side
            x0: Initial guess
            tol: Convergence tolerance
            maxiter: Maximum iterations
            restart: Restart parameter
        
        Returns:
            (solution, info)
        """
        
        # Use scipy's GMRES implementation
        x, info = spla.gmres(
            A, b, x0=x0, tol=tol, maxiter=maxiter, restart=restart
        )
        
        return x, info
    
    def bicgstab(
        self,
        A: np.ndarray,
        b: np.ndarray,
        x0: Optional[np.ndarray] = None,
        tol: float = 1e-6,
        maxiter: Optional[int] = None
    ) -> Tuple[np.ndarray, int]:
        """
        BiCGSTAB method for general linear systems
        
        Args:
            A: Coefficient matrix
            b: Right-hand side
            x0: Initial guess
            tol: Convergence tolerance
            maxiter: Maximum iterations
        
        Returns:
            (solution, info)
        """
        
        x, info = spla.bicgstab(A, b, x0=x0, tol=tol, maxiter=maxiter)
        
        return x, info


class SparseMatrixEngine:
    """
    Engine for sparse matrix operations
    """
    
    def __init__(self):
        self.sparse_formats = ['csr', 'csc', 'coo', 'lil', 'dok']
    
    def create_sparse_matrix(
        self,
        data: Union[np.ndarray, List],
        format: str = 'csr',
        shape: Optional[Tuple[int, int]] = None
    ) -> sp.spmatrix:
        """
        Create sparse matrix from data
        
        Args:
            data: Matrix data (dense array or (data, (row, col)) for COO)
            format: Sparse format
            shape: Matrix shape (if not inferrable)
        
        Returns:
            Sparse matrix
        """
        
        if format == 'csr':
            if isinstance(data, (list, tuple)) and len(data) == 2:
                # COO format input
                values, (rows, cols) = data
                return sp.csr_matrix((values, (rows, cols)), shape=shape)
            else:
                return sp.csr_matrix(data)
        
        elif format == 'csc':
            if isinstance(data, (list, tuple)) and len(data) == 2:
                values, (rows, cols) = data
                return sp.csc_matrix((values, (rows, cols)), shape=shape)
            else:
                return sp.csc_matrix(data)
        
        elif format == 'coo':
            if isinstance(data, (list, tuple)) and len(data) == 2:
                values, (rows, cols) = data
                return sp.coo_matrix((values, (rows, cols)), shape=shape)
            else:
                return sp.coo_matrix(data)
        
        else:
            raise ValueError(f"Unsupported sparse format: {format}")
    
    def sparse_solve(
        self,
        A: sp.spmatrix,
        b: np.ndarray,
        method: str = 'spsolve'
    ) -> np.ndarray:
        """
        Solve sparse linear system
        
        Args:
            A: Sparse coefficient matrix
            b: Right-hand side
            method: Solution method
        
        Returns:
            Solution vector
        """
        
        if method == 'spsolve':
            return spla.spsolve(A, b)
        elif method == 'spsolve_triangular':
            return spla.spsolve_triangular(A, b)
        elif method == 'factorized':
            solve = spla.factorized(A)
            return solve(b)
        else:
            raise ValueError(f"Unknown sparse solver: {method}")
    
    def sparse_eigenvalues(
        self,
        A: sp.spmatrix,
        k: int = 6,
        which: str = 'LM',
        return_eigenvectors: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Compute sparse matrix eigenvalues/eigenvectors
        
        Args:
            A: Sparse matrix
            k: Number of eigenvalues to compute
            which: Which eigenvalues ('LM', 'SM', 'LR', 'SR', etc.)
            return_eigenvectors: Whether to return eigenvectors
        
        Returns:
            Eigenvalues or (eigenvalues, eigenvectors)
        """
        
        if return_eigenvectors:
            return spla.eigs(A, k=k, which=which, return_eigenvectors=True)
        else:
            return spla.eigs(A, k=k, which=which, return_eigenvectors=False)
    
    def sparse_svd(
        self,
        A: sp.spmatrix,
        k: int = 6,
        which: str = 'LM'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute sparse SVD
        
        Args:
            A: Sparse matrix
            k: Number of singular values
            which: Which singular values
        
        Returns:
            (U, s, Vt)
        """
        
        return spla.svds(A, k=k, which=which)


class TensorOperations:
    """
    Tensor operations for multi-dimensional arrays
    """
    
    def __init__(self):
        pass
    
    def tensor_product(
        self,
        a: np.ndarray,
        b: np.ndarray,
        axes: Optional[Union[int, Tuple]] = None
    ) -> np.ndarray:
        """
        Tensor product (outer product generalization)
        
        Args:
            a, b: Input tensors
            axes: Axes to contract over
        
        Returns:
            Tensor product
        """
        
        if axes is None:
            # Outer product
            return np.outer(a.flatten(), b.flatten()).reshape(a.shape + b.shape)
        else:
            # Tensor contraction
            return np.tensordot(a, b, axes=axes)
    
    def tensor_contraction(
        self,
        tensor: np.ndarray,
        axes: Tuple[int, int]
    ) -> np.ndarray:
        """
        Contract tensor over specified axes
        
        Args:
            tensor: Input tensor
            axes: Pair of axes to contract
        
        Returns:
            Contracted tensor
        """
        
        return np.trace(tensor, axis1=axes[0], axis2=axes[1])
    
    def tensor_decomposition_cp(
        self,
        tensor: np.ndarray,
        rank: int,
        max_iter: int = 100,
        tol: float = 1e-6
    ) -> List[np.ndarray]:
        """
        CANDECOMP/PARAFAC (CP) tensor decomposition
        Simplified implementation
        
        Args:
            tensor: Input tensor
            rank: Target rank
            max_iter: Maximum iterations
            tol: Convergence tolerance
        
        Returns:
            List of factor matrices
        """
        
        # Simplified CP decomposition using ALS
        ndim = tensor.ndim
        shape = tensor.shape
        
        # Initialize factor matrices deterministically using SVD when possible
        factors = []
        for mode in range(ndim):
            unfolded = self._unfold_tensor(tensor, mode)
            try:
                # Use SVD on the unfolded tensor for a deterministic, data-driven init
                U, svals, Vt = la.svd(unfolded, full_matrices=False)
                f = U[:, :rank]

                # If SVD returned fewer components than rank, pad deterministically
                if f.shape[1] < rank:
                    pad_cols = rank - f.shape[1]
                    pad = np.zeros((shape[mode], pad_cols))
                    base = np.linspace(0, 1, shape[mode])
                    for j in range(pad_cols):
                        vec = np.roll(base, j)
                        vec = vec - np.mean(vec)
                        vec = vec / (np.linalg.norm(vec) + 1e-12)
                        pad[:, j] = vec
                    f = np.hstack([f, pad])

            except Exception:
                # Fallback deterministic pattern when SVD fails
                base = np.linspace(0, 1, shape[mode])
                f = np.vstack([np.roll(base, j) for j in range(rank)]).T

            factors.append(f)
        
        for iteration in range(max_iter):
            factors_old = [f.copy() for f in factors]
            
            # Alternating least squares
            for mode in range(ndim):
                # Compute Khatri-Rao product of all factors except current mode
                kr_product = factors[0] if mode != 0 else factors[1]
                for i in range(1 if mode == 0 else 2, ndim):
                    if i != mode:
                        kr_product = self._khatri_rao(kr_product, factors[i])
                
                # Unfold tensor along current mode
                unfolded = self._unfold_tensor(tensor, mode)
                
                # Solve least squares problem
                factors[mode] = la.lstsq(kr_product.T, unfolded.T)[0].T
            
            # Check convergence
            converged = True
            for i in range(ndim):
                if np.linalg.norm(factors[i] - factors_old[i]) > tol:
                    converged = False
                    break
            
            if converged:
                break
        
        return factors
    
    def _khatri_rao(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Khatri-Rao product"""
        return np.concatenate([np.kron(A[:, i:i+1], B[:, i:i+1]) 
                              for i in range(A.shape[1])], axis=1)
    
    def _unfold_tensor(self, tensor: np.ndarray, mode: int) -> np.ndarray:
        """Unfold tensor along specified mode"""
        shape = tensor.shape
        return tensor.reshape(shape[mode], -1, order='F')


