#--------------------------Library--------------------------# 
from functools import lru_cache
from scipy.integrate import quad
import numpy as np
from scipy.special import gamma as gamma_function
from math import pow, gamma 
from joblib import Parallel, delayed
from numba import njit, prange

FAST_MATH = True # Set to False to disable Numba optimizations for debugging purposes

### ----------------------------------------------------------------------------------------------------------------------###
# ---------- Annex I : Numba functions -------------------------------------------------------------------------------------#
### ----------------------------------------------------------------------------------------------------------------------###

@njit(fastmath=FAST_MATH)
def calculate_1_G_sparse_manual_numba(G_slice, L_compact, D_B_int, V_compact, dt, H, N, Z, tol, delta_buf, Z_l):
    """
    Computes a single column of the G matrix using a low-rank compact representation.
    """
    r = len(delta_buf)
    
    for l in range(1, N + 1):
        N_l = N - l + 2
        min_r = min(r, N_l - 1)
        
        # Reset the buffer without reallocating memory
        delta_buf.fill(0.0)
        delta = delta_buf[:min_r]
        
        # 1. Forward substitution for triangular resolution using compact L
        for i in range(min_r):
            s = V_compact[i]
            for j in range(i):
                s -= L_compact[i, j] * delta[j]
            delta[i] = s / L_compact[i, i]
        
        # 2. Compute kappa (orthogonal component weight)
        k_squared = dt
        for i in range(r):
            val = delta_buf[i] if i < min_r else 0.0
            k_squared -= val * val
        kappa = np.sqrt(k_squared) if k_squared > 0 else 0.0
        
        # Copy noise increments into the local buffer
        for i in range(N_l):
            Z_l[i] = Z[l - 1 + i]
            
        # 3. Apply the Volterra history weights
        for i in range(N_l - 1):
            for j in range(min_r):
                G_slice[l + i - 1, l - 1] += L_compact[i, j] * Z_l[j]
        
        # 4. Final entry of the column (standard Brownian part)
        dot_product = 0.0
        for i in range(min_r):
            dot_product += delta_buf[i] * Z_l[i]
        
        G_slice[N, l - 1] = dot_product + kappa * Z_l[N_l - 1]
    return 0

@njit(parallel=True, fastmath=FAST_MATH)
def calculate_all_G_sparse_manual_numba(L_B_int, D_B_int, V, dt, H, N, Z_all, tol=1e-4):
    """
    Parallel execution of the G matrix construction.
    Uses memory-efficient slicing and uninitialized buffers.
    """
    size_sample = Z_all.shape[0]
    G_all = np.zeros((size_sample, N + 1, N))
    
    # 1. Determine effective rank r based on tolerance
    r = len(D_B_int)
    for i in range(len(D_B_int)):
        if D_B_int[i] < tol:
            r = i
            break
    
    # 2. Memory optimization: Compact slicing
    L_compact = (L_B_int[:, :r]) * (dt ** H)
    V_compact = V[:r]
    
    # 3. Parallel loop over trajectories
    for i in prange(size_sample):
        local_delta_buf = np.empty(r)
        local_Z_l = np.empty(N + 1)
        
        G_all[i].fill(0.0)
        
        _ = calculate_1_G_sparse_manual_numba(
            G_all[i], L_compact, D_B_int, V_compact, dt, H, N, Z_all[i], 
            tol, local_delta_buf, local_Z_l
        )
        
    return G_all

@njit(parallel=True, fastmath=FAST_MATH)
def calculate_all_G_sparse_manual_numba_prealloc(L_B_int, D_B_int, V, dt, H, N, Z_all, G_buffer, tol=1e-4):
    """
    Version using a pre-allocated global buffer to minimize system memory calls.
    """
    size_sample = Z_all.shape[0]
    
    r = len(D_B_int)
    for i in range(len(D_B_int)):
        if D_B_int[i] < tol:
            r = i
            break
            
    L_compact = (L_B_int[:, :r]) * (dt ** H)
    V_compact = V[:r]
    
    for i in prange(size_sample):
        local_delta_buf = np.empty(r)
        local_Z_l = np.empty(N + 1)
        
        G_buffer[i].fill(0.0) 
        
        _ = calculate_1_G_sparse_manual_numba(
            G_buffer[i], L_compact, D_B_int, V_compact, dt, H, N, Z_all[i], 
            tol, local_delta_buf, local_Z_l
        )
    return G_buffer

@njit(fastmath=FAST_MATH)
def sig_numba(y, a, b, c, mode):
    if mode == 2:
        return np.sqrt(a * (y - b)**2 + c)
    return np.sqrt(max(y, 0.0)) 

@njit(parallel=True, fastmath=FAST_MATH)
def simulate_all_trajectories_Elephant_numba(size_sample, N, xi0, H, dt, eta, list_G, mu, lam, a, b, c, mode_sig=2):
    """
    Optimized simulation for Goldfish (Y) and Elephant (X) processes.
    Replaced dynamic array slicing with explicit loops for zero-allocation Numba execution.
    """
    Y_all = np.empty((size_sample, N + 1))
    X_all = np.empty((size_sample, N + 1))
    
    inv_g_h_plus_half = 1.0 / gamma(H + 0.5)
    g_3_2_h = gamma(1.5 - H)
    alpha = H + 0.5
    
    Y_all[:, 0] = 0.0
        
    kernel_det = np.zeros(N + 1)
    drift_frac = np.zeros(N + 1)
    
    for n in range(1, N + 1):
        kernel_det[n] = (pow(n, alpha) - pow(n - 1, alpha)) * pow(dt, alpha) / alpha
        drift_frac[n] = (xi0 / g_3_2_h) * (pow(n * dt, 0.5 - H) - pow((n - 1) * dt, 0.5 - H))
            
    for i in prange(size_sample):
        sig_y_cache = np.empty(N + 1)
        drift_y_cache = np.empty(N + 1)
        
        X_all[i, 0] = xi0
        
        sig_y_cache[0] = sig_numba(0.0, a, b, c, mode_sig)
        drift_y_cache[0] = mu - lam * 0.0
        
        G_traj = list_G[i]
        dW = G_traj[N, :] 
        
        for n in range(1, N + 1):
            Y_all[i, n] = Y_all[i, n-1] + drift_frac[n] + dt * drift_y_cache[n-1] + eta * sig_y_cache[n-1] * dW[n-1]
            
            sig_y_cache[n] = sig_numba(Y_all[i, n], a, b, c, mode_sig)
            drift_y_cache[n] = mu - lam * Y_all[i, n]
            
            sum_s1 = 0.0
            sum_s2 = 0.0
            
            for k in range(n):
                idx_rev = n - k
                sum_s1 += drift_y_cache[k] * kernel_det[idx_rev]
                sum_s2 += sig_y_cache[k] * G_traj[n - 1, k]
                
            sum_s2 *= eta
            
            X_all[i, n] = xi0 + (sum_s1 + sum_s2) * inv_g_h_plus_half
                
    return Y_all, X_all


### ----------------------------------------------------------------------------------------------------------------------###
# ---------- Annex II : Construction of the matrix B_int -------------------------------------------------------------------#
### ----------------------------------------------------------------------------------------------------------------------###

def integrand(v, i, j, alpha):
    """Integrand function used to calculate the coefficients of matrix B."""
    return np.power(i + v, alpha, dtype=np.float64) * np.power(j + v, alpha, dtype=np.float64)

def b_ij(i, j, alpha):
    """Calculates the exact coefficient (B_ij) using adaptive quadrature for singularities."""
    if i == j: 
        return (1.0 / (2.0*alpha + 1.0)) * (np.power(i + 1.0, 2.0*alpha + 1.0, dtype=np.float64) - np.power(float(i), 2.0*alpha + 1.0, dtype=np.float64))
    else:
        result, error = quad(integrand, 0, 1, args=(i, j, alpha))
        return result
    
@njit(fastmath=FAST_MATH)
def extended_cholesky_numba(C, tol=1e-14):
    """
    Extended Cholesky decomposition (LDLᵀ) – Numba version.
    """
    n = C.shape[0]
    T = np.eye(n)
    D = np.zeros(n)
    for k in range(n):
        tmp = 0.0
        for s in range(k):
            tmp += T[k, s] * T[k, s] * D[s]
        D[k] = C[k, k] - tmp
        if D[k] < tol: D[k] = 0.0 
        for i in range(k + 1, n):
            tmp2 = 0.0
            for s in range(k):
                tmp2 += T[i, s] * T[k, s] * D[s]
            T[i, k] = (C[i, k] - tmp2) / D[k] if D[k] > tol else 0.0
    return T, D

def fill_b_int(N, H, N_para=6):
    """Constructs the matrix B_int using parallel high-precision integration."""
    B_int = np.zeros((N, N))
    alpha = H - 0.5

    def compute_b_ij(i, j):
        return b_ij(i, j, alpha)

    tasks = [(i, j) for i in range(N) for j in range(i, N)]
    
    results = Parallel(n_jobs=N_para)(delayed(compute_b_ij)(i, j) for i, j in tasks)

    idx = 0
    for i in range(N):
        for j in range(i, N):
            B_int[i, j] = results[idx]
            idx += 1
            
    B_int = B_int + B_int.T - np.diag(np.diag(B_int))
    return B_int

@lru_cache(maxsize=100)
def get_cholesky_B_int(N, H, extended, eps=1e-8):
    """
    Returns the Cholesky decomposition of the B_int matrix of size N for a given H. 
    Uses memoization to avoid recomputation. Numba optimizations are applied by default.
    """
    B_int = fill_b_int(N, H) 
    
    if extended: 
        T_B_int, D_B_int = extended_cholesky_numba(B_int)
        sqrt_D = np.sqrt(D_B_int)
        L_B_int = T_B_int * sqrt_D
        return L_B_int, D_B_int        
            
    else: 
        B_int += eps * np.eye(N)
        try:
            L_B_int  = np.linalg.cholesky(B_int)
        except np.linalg.LinAlgError as e:
            print("Error during the compute of Cholesky decomposition of the matrix:", str(e))
            raise ValueError  

        return L_B_int, np.ones(N) 


### ----------------------------------------------------------------------------------------------------------------------###
# ---------- Part II: Functions for Monte Carlo errors analysis ------------------------------------------------------------#
### ----------------------------------------------------------------------------------------------------------------------###

# --------------------------------------------------------------------------------- #
# 1. Precomputations and General Tools
# --------------------------------------------------------------------------------- #

@njit(fastmath=FAST_MATH)
def compute_L_l_numba(T, N, l, k, alpha):
    """ Compute the Cholesky matrix L_l (2,2) for the covariance structure (used for arbitrary k). """
    L_l = np.zeros((2, 2))
    
    dt = T / (2 * N)
    squared_sigma_1 = dt
    
    squared_sigma_2 = (T / (2 * alpha - 1.0)) * (pow(k - l, 2 * alpha - 1.0) - pow(k - (l + 1), 2 * alpha - 1.0)) / pow(2 * N, 2 * alpha - 1.0)
    squared_sigma_2 = squared_sigma_2 / pow(gamma(alpha), 2.0)
    
    rho = (T / alpha) * (pow(k - l, alpha) - pow(k - (l + 1), alpha)) / (np.sqrt(squared_sigma_1) * np.sqrt(squared_sigma_2))
    rho = rho / pow(2 * N, alpha)
    rho = rho / gamma(alpha)  
    
    L_l[0, 0] = np.sqrt(squared_sigma_1)
    L_l[1, 0] = rho * np.sqrt(squared_sigma_2)
    L_l[1, 1] = np.sqrt(squared_sigma_2) * np.sqrt(1.0 - rho**2)
    
    return L_l

@njit(fastmath=FAST_MATH)
def compute_2n_G_numba(T, N, k, alpha, Z_gaussian):
    """ Compute the 2*k vectors G of size 2 (used for arbitrary k). """
    G = np.zeros((2, k))
    for i in range(k):
        L_l = compute_L_l_numba(T, N, i, k, alpha)
        G[0, i] = L_l[0, 0] * Z_gaussian[0, i] + L_l[0, 1] * Z_gaussian[1, i]
        G[1, i] = L_l[1, 0] * Z_gaussian[0, i] + L_l[1, 1] * Z_gaussian[1, i]
    return G

def precompute_G_matrices(T, N, alpha, Z_all):
    """
    Precomputes G for all trajectories using vectorized NumPy.
    Z_all has shape (size_sample, 2, 2N).
    This replaces the iterative compute_2n_G for the specific case k = 2N.
    """
    k_val = 2 * N
    dt = T / (2 * N)
    l_arr = np.arange(k_val)
    
    sqrt_sigma_1 = np.sqrt(dt)
    
    # Vectorized computation of variance and covariance
    term_sigma_2 = (np.power(k_val - l_arr, 2*alpha - 1) - np.power(k_val - l_arr - 1, 2*alpha - 1))
    squared_sigma_2 = (T / (2 * alpha - 1)) * term_sigma_2 / np.power(2 * N, 2 * alpha - 1) / (gamma_function(alpha)**2)
    sqrt_sigma_2 = np.sqrt(squared_sigma_2)
    
    term_rho = (np.power(k_val - l_arr, alpha) - np.power(k_val - l_arr - 1, alpha))
    rho = (T / alpha) * term_rho / (sqrt_sigma_1 * sqrt_sigma_2) / np.power(2 * N, alpha) / gamma_function(alpha)
    
    L00 = sqrt_sigma_1
    L10 = rho * sqrt_sigma_2
    L11 = sqrt_sigma_2 * np.sqrt(1 - rho**2)
    
    # Immediate application to all trajectories at once
    G_0 = L00 * Z_all[:, 0, :]
    G_1 = L10 * Z_all[:, 0, :] + L11 * Z_all[:, 1, :]
    
    return G_0, G_1

# --------------------------------------------------------------------------------- #
# 2. End Point Error Computation
# --------------------------------------------------------------------------------- #

@njit(fastmath=FAST_MATH)
def compute_endPoint_N_numba_opti(xi0, mu, lam, a, b, c, mode_sig, N, T, eta, alpha, dW_2N, G_2N, drift_frac_2N, drift_frac_N, kernel_end_2N, kernel_end_N):
    """ Purely arithmetic loop: 0 allocation, 0 complex 'pow' function calls. """
    dW_N = np.empty(N)
    G_N = np.empty(N)
    for l in range(N):
        dW_N[l] = dW_2N[2*l] + dW_2N[2*l+1]
        G_N[l] = G_2N[2*l] + G_2N[2*l+1]
   
    Y_2N = np.zeros(2 * N + 1)
    Y_N = np.zeros(N + 1)
    
    dt_2N = T / (2 * N)
    dt_N = T / N
    
    i = 0
    for k in range(2 * N):
        drift_2N = dt_2N * (mu - lam * Y_2N[k])
        diffusion_2N = eta * sig_numba(Y_2N[k], a, b, c, mode_sig) * dW_2N[k]
        Y_2N[k+1] = Y_2N[k] + drift_frac_2N[k] + drift_2N + diffusion_2N
        
        if k % 2 == 0:
            drift_N = dt_N * (mu - lam * Y_N[i])
            diffusion_N = eta * sig_numba(Y_N[i], a, b, c, mode_sig) * dW_N[i] 
            Y_N[i+1] = Y_N[i] + drift_frac_N[i] + drift_N + diffusion_N
            i += 1
            
    # Elephant process
    sum_1_2N = 0.0
    sum_2_2N = 0.0
    for l in range(2 * N):
        sum_1_2N += kernel_end_2N[l] * (mu - lam * Y_2N[l])
        sum_2_2N += G_2N[l] * sig_numba(Y_2N[l], a, b, c, mode_sig)
    
    X_2N_endPoint = xi0 + (sum_1_2N * (pow(T, alpha) / alpha)) / gamma(alpha) + eta * sum_2_2N 
    
    sum_1_N = 0.0
    sum_2_N = 0.0
    for l in range(N):
        sum_1_N += kernel_end_N[l] * (mu - lam * Y_N[l])
        sum_2_N += G_N[l] * sig_numba(Y_N[l], a, b, c, mode_sig)
        
    X_N_endPoint = xi0 + (sum_1_N * (pow(T, alpha) / alpha)) / gamma(alpha) + eta * sum_2_N      
    
    return X_N_endPoint, X_2N_endPoint, Y_N, Y_2N

@njit(parallel=True, fastmath=FAST_MATH)
def compute_m_endPoint_N_numba_parallel(xi0, mu, lam, a, b, c, mode_sig, N, T, alpha, eta, size_sample, dW_all_2N, G_all_2N, drift_frac_2N, drift_frac_N, kernel_end_2N, kernel_end_N):
    List_X_N = np.empty(size_sample)
    List_X_2N = np.empty(size_sample)
    List_Y_N = np.empty((size_sample, N + 1))
    List_Y_2N = np.empty((size_sample, 2 * N + 1))
    
    for m in prange(size_sample):
        x_n, x_2n, y_n, y_2n = compute_endPoint_N_numba_opti(
            xi0, mu, lam, a, b, c, mode_sig, N, T, eta, alpha, 
            dW_all_2N[m], G_all_2N[m], drift_frac_2N, drift_frac_N, kernel_end_2N, kernel_end_N
        )
        List_X_N[m] = x_n
        List_X_2N[m] = x_2n
        List_Y_N[m, :] = y_n
        List_Y_2N[m, :] = y_2n
        
    return List_X_N, List_X_2N, List_Y_N, List_Y_2N

def Compute_m_endPoint_N(xi0, para_b, para_sig, mode_sig, N, T, alpha, eta, size_sample, batch_size=2000):
    """
    Computes endpoint error variables in batches to handle very large size_sample (e.g., 50,000+).
    Memory is preallocated to avoid dynamic list resizing overhead.
    """
    mu, lam = para_b['mu'], para_b['lam']
    a, b, c = para_sig['a'], para_sig['b'], para_sig['c']
    H = alpha - 0.5
    dt_2N, dt_N = T / (2 * N), T / N
    
    # 1. NumPy Precomputations
    gamma_val = gamma_function(1.5 - H)
    k_arr_2N = np.arange(2*N)
    drift_frac_2N = (xi0 / gamma_val) * (np.power((k_arr_2N + 1) * dt_2N, 0.5 - H) - np.power(k_arr_2N * dt_2N, 0.5 - H))
    i_arr_N = np.arange(N)
    drift_frac_N = (xi0 / gamma_val) * (np.power((i_arr_N + 1) * dt_N, 0.5 - H) - np.power(i_arr_N * dt_N, 0.5 - H))
    
    kernel_end_2N = np.power(1.0 - k_arr_2N/(2*N), alpha) - np.power(1.0 - (k_arr_2N+1)/(2*N), alpha)
    kernel_end_N = np.power(1.0 - i_arr_N/N, alpha) - np.power(1.0 - (i_arr_N+1)/N, alpha)
    
    # 2. Memory Preallocation for the entire sample size
    final_X_N = np.empty(size_sample, dtype=np.float64)
    final_X_2N = np.empty(size_sample, dtype=np.float64)
    final_Y_N = np.empty((size_sample, N + 1), dtype=np.float64)
    final_Y_2N = np.empty((size_sample, 2 * N + 1), dtype=np.float64)
    
    # 3. Batch Processing
    # For EndPoint, the G matrix is very small (2, 2N), so we can use a large batch_size (e.g., 2000).
    for start_idx in range(0, size_sample, batch_size):
        end_idx = min(start_idx + batch_size, size_sample)
        current_batch_size = end_idx - start_idx
        
        Z_batch = np.random.standard_normal(size=(current_batch_size, 2, 2 * N))
        dW_all_2N, G_all_2N = precompute_G_matrices(T, N, alpha, Z_batch)
        
        X_N, X_2N, Y_N, Y_2N = compute_m_endPoint_N_numba_parallel(
            xi0, mu, lam, a, b, c, mode_sig, N, T, alpha, eta, current_batch_size, 
            dW_all_2N, G_all_2N, drift_frac_2N, drift_frac_N, kernel_end_2N, kernel_end_N
        )
        
        # Inject results directly into the preallocated arrays
        final_X_N[start_idx:end_idx] = X_N
        final_X_2N[start_idx:end_idx] = X_2N
        final_Y_N[start_idx:end_idx, :] = Y_N
        final_Y_2N[start_idx:end_idx, :] = Y_2N

    return final_X_N, final_X_2N, final_Y_N, final_Y_2N

# --------------------------------------------------------------------------------- #
# 3. Max Error Computation (X_N vs X_2N)
# --------------------------------------------------------------------------------- #

@njit(fastmath=FAST_MATH)
def compute_one_X_2N_N_numba_opti(xi0, mu, lam, a, b, c, mode_sig, N, T, eta, alpha, Z_2N, G_2N, drift_frac_2N, drift_frac_N, conv_kernel_2N, conv_kernel_N):
    dt_2N, dt_N = T / (2 * N), T / N
    dW_2N = Z_2N * np.sqrt(dt_2N)
    
    dW_N = np.empty(N)
    for l in range(N):
        dW_N[l] = dW_2N[2*l] + dW_2N[2*l+1]
        
    # 1. Compute Y trajectories (Goldfish)
    Y_2N = np.zeros(2 * N + 1)
    Y_N = np.zeros(N + 1)
    i = 0
    for k in range(2 * N):
        Y_2N[k+1] = Y_2N[k] + drift_frac_2N[k] + dt_2N * (mu - lam * Y_2N[k]) + eta * sig_numba(Y_2N[k], a, b, c, mode_sig) * dW_2N[k]
        if k % 2 == 0:
            Y_N[i+1] = Y_N[i] + drift_frac_N[i] + dt_N * (mu - lam * Y_N[i]) + eta * sig_numba(Y_N[i], a, b, c, mode_sig) * dW_N[i] 
            i += 1
            
    drift_Y_2N = np.empty(2 * N + 1)
    sig_Y_2N = np.empty(2 * N + 1)
    for l in range(2 * N + 1):
        drift_Y_2N[l] = mu - lam * Y_2N[l]
        sig_Y_2N[l] = sig_numba(Y_2N[l], a, b, c, mode_sig)
        
    drift_Y_N = np.empty(N + 1)
    sig_Y_N = np.empty(N + 1)
    for l in range(N + 1):
        drift_Y_N[l] = mu - lam * Y_N[l]
        sig_Y_N[l] = sig_numba(Y_N[l], a, b, c, mode_sig)

    # 3. Optimized Euler simulation for X (Elephant)
    X_N = np.zeros(N + 1)
    X_2N = np.zeros(2 * N + 1)
    X_N[0] = xi0
    X_2N[0] = xi0

    H = alpha - 0.5
    gamma_funcH05 = gamma(H + 0.5)

    i = 0
    for k in range(2 * N):
        sum_1_2N = 0.0
        sum_2_2N = 0.0
        
        for l in range(k + 1):
            term = conv_kernel_2N[k - l] 
            sum_1_2N += term * drift_Y_2N[l]
            sum_2_2N += G_2N[k, l] * sig_Y_2N[l]
            
        X_2N[k + 1] = xi0 + sum_1_2N / gamma_funcH05 + eta * sum_2_2N

        if k % 2 == 0:
            sum_1_N = 0.0
            sum_2_N = 0.0
            for l in range(i + 1):
                term = conv_kernel_N[i - l]
                sum_1_N += term * drift_Y_N[l]
                sum_2_N += (G_2N[k + 1, 2 * l] + G_2N[k + 1, 2 * l + 1]) * sig_Y_N[l]

            X_N[i + 1] = xi0 + sum_1_N / gamma_funcH05 + eta * sum_2_N
            i += 1

    return X_N, X_2N, Y_N, Y_2N

@njit(parallel=True, fastmath=FAST_MATH)
def compute_all_X_2N_N_numba_parallel(xi0, mu, lam, a, b, c, mode_sig, N, T, eta, alpha, Z_all, G_all_2N, drift_frac_2N, drift_frac_N, conv_kernel_2N, conv_kernel_N):
    size_sample = Z_all.shape[0]
    List_X_N = np.empty((size_sample, N + 1))
    List_X_2N = np.empty((size_sample, 2 * N + 1))
    List_Y_N = np.empty((size_sample, N + 1))
    List_Y_2N = np.empty((size_sample, 2 * N + 1))

    for m in prange(size_sample):
        x_n, x_2n, y_n, y_2n = compute_one_X_2N_N_numba_opti(
            xi0, mu, lam, a, b, c, mode_sig, N, T, eta, alpha, 
            Z_all[m], G_all_2N[m], drift_frac_2N, drift_frac_N, conv_kernel_2N, conv_kernel_N
        )
        List_X_N[m, :] = x_n
        List_X_2N[m, :] = x_2n
        List_Y_N[m, :] = y_n
        List_Y_2N[m, :] = y_2n

    return List_X_N, List_X_2N, List_Y_N, List_Y_2N

def Compute_m_XN_X2N(xi0, para_b, para_sig, mode_sig, N, T, alpha, eta, size_sample, tol=1e-4, batch_size=50):
    """
    Computes full trajectory variables in batches to handle very large size_sample.
    Batch size must remain small (e.g., 50) because the full history G matrix is extremely large (2N x 2N).
    """
    mu, lam = para_b['mu'], para_b['lam']
    a, b, c = para_sig['a'], para_sig['b'], para_sig['c']
    H = alpha - 0.5
    dt_2N, dt_N = T / (2 * N), T / N
    
    # 1. NumPy Precomputations
    gamma_val = gamma_function(1.5 - H)
    k_arr_2N = np.arange(2*N)
    drift_frac_2N = (xi0 / gamma_val) * (np.power((k_arr_2N + 1) * dt_2N, 0.5 - H) - np.power(k_arr_2N * dt_2N, 0.5 - H))
    i_arr_N = np.arange(N)
    drift_frac_N = (xi0 / gamma_val) * (np.power((i_arr_N + 1) * dt_N, 0.5 - H) - np.power(i_arr_N * dt_N, 0.5 - H))
    
    j_arr_2N = np.arange(2*N + 1)
    conv_kernel_2N = (np.power(j_arr_2N + 1, alpha) - np.power(j_arr_2N, alpha)) * np.power(dt_2N, alpha) / alpha
    j_arr_N = np.arange(N + 1)
    conv_kernel_N = (np.power(j_arr_N + 1, alpha) - np.power(j_arr_N, alpha)) * np.power(dt_N, alpha) / alpha

    L_B_int_2N, D_B_int_2N = get_cholesky_B_int(2 * N, H, extended=True)
    V_2N = np.power(dt_2N, alpha) * (1 / alpha) * ((k_arr_2N + 1)**alpha - k_arr_2N**alpha)
    
    # 2. Memory Preallocation for the entire sample size
    final_X_N = np.empty((size_sample, N + 1), dtype=np.float64)
    final_X_2N = np.empty((size_sample, 2 * N + 1), dtype=np.float64)
    final_Y_N = np.empty((size_sample, N + 1), dtype=np.float64)
    final_Y_2N = np.empty((size_sample, 2 * N + 1), dtype=np.float64)

    # 3. Batch Processing
    # For Max Error, G is a dense (2N, 2N) matrix, meaning high RAM usage. 
    # Batch size is kept small (e.g., 50) to cap RAM usage at ~1-2 GB.
    for start_idx in range(0, size_sample, batch_size):
        end_idx = min(start_idx + batch_size, size_sample)
        current_batch_size = end_idx - start_idx
        
        Z_2N_batch = np.random.standard_normal(size=(current_batch_size, 2 * N + 1))
        G_batch_2N = calculate_all_G_sparse_manual_numba(L_B_int_2N, D_B_int_2N, V_2N, dt_2N, H, 2 * N, Z_2N_batch, tol)

        X_N, X_2N, Y_N, Y_2N = compute_all_X_2N_N_numba_parallel(
            xi0, mu, lam, a, b, c, mode_sig, N, T, eta, alpha, 
            Z_2N_batch, G_batch_2N, drift_frac_2N, drift_frac_N, conv_kernel_2N, conv_kernel_N
        )
        
        # Inject results directly into the preallocated arrays
        final_X_N[start_idx:end_idx, :] = X_N
        final_X_2N[start_idx:end_idx, :] = X_2N
        final_Y_N[start_idx:end_idx, :] = Y_N
        final_Y_2N[start_idx:end_idx, :] = Y_2N

    return final_X_2N, final_X_N, final_Y_2N, final_Y_N

# --------------------------------------------------------------------------------- #
# 4. Holder Continuity Functions (X_tk)
# --------------------------------------------------------------------------------- #

@njit(fastmath=FAST_MATH)
def compute_Y_N_numba(xi0, mu, lam, a, b, c, mode_sig, N, T, k, eta, alpha, dW_N): 
    Y_N = np.zeros(k + 1)
    Y_N[0] = xi0
    
    dt_N = T / N
    H = alpha - 0.5
    gamma_val = gamma(1.5 - H)

    for l in range(k):    
        term1_N = (xi0 / gamma_val) * (pow((l + 1) * dt_N, 0.5 - H) - pow(l * dt_N, 0.5 - H))
        drift_N = dt_N * (mu - lam * Y_N[l])
        diffusion_N = eta * sig_numba(Y_N[l], a, b, c, mode_sig) * dW_N[l] 
        Y_N[l+1] = Y_N[l] + term1_N + drift_N + diffusion_N
    return Y_N

@njit(fastmath=FAST_MATH)
def compute_X_tk_numba(xi0, mu, lam, a, b, c, mode_sig, N, T, k, eta, H, Z_k):
    alpha = H + 0.5
    G = compute_2n_G_numba(T, N, k, alpha, Z_k) 
    
    dW_N = G[0, :] 
    G_N = G[1, :] 
    
    dt_N = T / N
    t_k = k * dt_N
   
    Y_N = compute_Y_N_numba(xi0, mu, lam, a, b, c, mode_sig, N, T, k, eta, alpha, dW_N) 
            
    sum_1_N = 0.0
    sum_2_N = 0.0
    for l in range(k):
        term = pow(t_k - l * dt_N, alpha) - pow(t_k - (l + 1) * dt_N, alpha)
        sum_1_N += term * (mu - lam * Y_N[l])
        sum_2_N += G_N[l] * sig_numba(Y_N[l], a, b, c, mode_sig)
        
    X_N_k = xi0 + (sum_1_N / alpha) / gamma(alpha) + eta * sum_2_N      
    return X_N_k, Y_N

@njit(parallel=True, fastmath=FAST_MATH)
def compute_all_X_tk_numba(xi0, mu, lam, a, b, c, mode_sig, N, T, k, eta, H, Z_k_all):
    size_sample = Z_k_all.shape[0]
    Val_X_tk = np.empty(size_sample)
    for m in prange(size_sample):
        x_k, _ = compute_X_tk_numba(xi0, mu, lam, a, b, c, mode_sig, N, T, k, eta, H, Z_k_all[m])
        Val_X_tk[m] = x_k
    return Val_X_tk

def compute_m_X_tk(xi0, para_b, para_sig, mode_sig, N, T, k, eta, H, size_sample):
    """ Generate size_sample value of the process X at time t_k """
    mu, lam = para_b['mu'], para_b['lam']
    a, b, c = para_sig['a'], para_sig['b'], para_sig['c']
    Z_k_all = np.random.standard_normal(size=(size_sample, 2, k))
    return compute_all_X_tk_numba(xi0, mu, lam, a, b, c, mode_sig, N, T, k, eta, H, Z_k_all)

### ----------------------------------------------------------------------------------------------------------------------###
# ---------- Part III: Functions for the others schemes --------------------------------------------------------------------#
### ----------------------------------------------------------------------------------------------------------------------###

@njit(fastmath=FAST_MATH)
def K_Gamma_numba(t, H):
    """ Implementation of the Gamma fractional kernel inside Numba """
    if t == 0.0:
        return 0.0
    alpha = H + 0.5
    return pow(t, alpha - 1.0) / gamma(alpha)

@njit(parallel=True, fastmath=FAST_MATH)
def Simu_mpath_X_FullEuler_numba(eta, x, mu, lam, a, b, c, mode_sig, H, N, T, Xi_m, dW_m): 
    m_samples = Xi_m.shape[0]
    X_m = np.zeros((m_samples, N + 1)) 
    dt = T / N 
    
    for Nth_path in prange(m_samples):
        X = np.zeros(N + 1)
        X[0] = x
        Xi = Xi_m[Nth_path]
        dW = dW_m[Nth_path]
        
        for k in range(1, N + 1): 
            tk = k * dt
            sum_drift = 0.0
            sum_diff = 0.0
            
            for i in range(k):
                ti = i * dt
                incr_t = tk - ti
                kernel_val = K_Gamma_numba(incr_t, H)
                
                drift_val = mu - lam * Xi[i]
                sig_val = sig_numba(Xi[i], a, b, c, mode_sig)
                
                sum_drift += kernel_val * drift_val * dt
                sum_diff += kernel_val * sig_val * dW[i]
                
            X[k] = x + sum_drift + eta * sum_diff
            
        X_m[Nth_path, :] = X
    return X_m

def Simu_mpath_X_FullEuler(eta, x, para_b, para_sig, mode_sig, H, N, T, Xi_m, dW_m):
    """ Python wrapper for parallel Numba full Euler simulation """
    mu, lam = para_b['mu'], para_b['lam']
    a, b, c = para_sig['a'], para_sig['b'], para_sig['c']
    return Simu_mpath_X_FullEuler_numba(eta, x, mu, lam, a, b, c, mode_sig, H, N, T, Xi_m, dW_m)

@njit(fastmath=FAST_MATH)
def simulate_one_trajectory_model_bis_numba(eta, mu, lam, a, b, c, mode_sig, H, T, xi0, N, dW): 
    Z = np.zeros(N + 1)
    Y = np.zeros(N + 1)
    Z[0] = xi0
    
    dt = T / N
    alpha = H + 0.5
    gamma_3_2_H = gamma(1.5 - H)
    gamma_H_plus_half = gamma(H + 0.5)
    
    for k in range(N):
        t_k = (k + 1) * dt
        
        term1 = (xi0 / gamma_3_2_H) * (pow((k + 1) * dt, 0.5 - H) - pow(k * dt, 0.5 - H))
        drift = dt * (mu - lam * Y[k])
        volatility = sig_numba(Y[k], a, b, c, mode_sig)
        diffusion = eta * volatility * dW[k] 
        Y[k + 1] = Y[k] + term1 + drift + diffusion
        
        s1_s2_sum = 0.0
        for l in range(k + 1):
            term_s1 = (pow(t_k - l * dt, alpha) - pow(t_k - (l + 1) * dt, alpha)) / alpha
            term_s2 = (mu - lam * Y[l]) + eta * (N / T) * sig_numba(Y[l], a, b, c, mode_sig) * dW[l]
            s1_s2_sum += term_s1 * term_s2
            
        Z[k + 1] = xi0 + s1_s2_sum / gamma_H_plus_half
        
    return Z, Y

@njit(parallel=True, fastmath=FAST_MATH)
def simulate_m_path_bis_numba(eta, mu, lam, a, b, c, mode_sig, H, T, xi0, N, dW_m):
    size_sample = dW_m.shape[0]
    Z_m = np.empty((size_sample, N + 1))
    Y_m = np.empty((size_sample, N + 1))
    
    for i in prange(size_sample):
        z, y = simulate_one_trajectory_model_bis_numba(eta, mu, lam, a, b, c, mode_sig, H, T, xi0, N, dW_m[i])
        Z_m[i, :] = z
        Y_m[i, :] = y
        
    return Z_m, Y_m

def simulate_m_path_bis(eta, para_b, para_sig, mode_sig, H, T, xi0, N, dW_m):
    """ Python wrapper for running the bis model in parallel """
    mu, lam = para_b['mu'], para_b['lam']
    a, b, c = para_sig['a'], para_sig['b'], para_sig['c']
    return simulate_m_path_bis_numba(eta, mu, lam, a, b, c, mode_sig, H, T, xi0, N, dW_m)