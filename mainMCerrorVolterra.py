'''
Goal: study numerically the strong convergence order of the Smart Euler scheme in the fractional case, which is theoretically 1/2. 
We will study two types of error: the end point error and the max error. 

This file is for compilation purposes, especially adapted from compilation of calculus servers. 
'''
#%%
# --- Library ---
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import time
from scipy.stats import linregress
from tqdm import tqdm

# Default theme for better plots
sns.set_theme()

# --- For random compute ---
rng = np.random.default_rng()

# --- Library file with all the required functions ---
import LibrairySimuVol_errorMC as lib

### ------------------- L2 error Monte Carlo functions ------------------- ###

def L2error_MonteCarlo_list(X, X_ref): 
    """
    Calculate the strong L2 error for 1D arrays (e.g., endpoint arrays).
    """
    assert np.shape(X) == np.shape(X_ref)
    S = (X - X_ref)**2
    squared_error = np.mean(S, axis=0, dtype=np.float64)
    return np.sqrt(squared_error)

def L2error_MonteCarlo_numpy(X, X_ref): 
    """
    Calculate the strong L2 error between simulated paths and reference paths over the entire trajectory (max error).
    """  
    assert X.shape == X_ref.shape, f"Shape mismatch: {X.shape} != {X_ref.shape}"
    squared_errors = (X - X_ref)**2  
    mean_squared_errors = np.mean(squared_errors, axis=0, dtype=np.float64) 
    return np.max(np.sqrt(mean_squared_errors))

def L2error_MonteCarlo_endPoint(X, X_ref): 
    """
    Calculate the strong L2 error between simulated paths and reference paths at the final time step.
    """  
    assert np.shape(X) == np.shape(X_ref)
    S = np.power(X[:, -1] - X_ref[:, -1], 2)
    squared_error = np.mean(S, dtype=np.float64)
    return np.sqrt(squared_error)


### ------------------- Plotting and Regression Helper ------------------- ###

def _plot_and_print_regression(N_array, errors, title, ylabel, label_err, slope_ref=0.5):
    """
    Computes the log-log regression, prints the estimated slope, and plots the results.
    """
    log_N = np.log(N_array)
    log_err = np.log(errors)
    slope_fit, intercept_fit, r_value, p_value, std_err = linregress(log_N, log_err)
    
    print(f"\n--- {title} ---")
    print(f"Estimated slope (log-log) : {-slope_fit:.4f}")
    print(f"Difference with {slope_ref}: {abs(-slope_fit - slope_ref):.4e}")
    
    ref_line = [errors[0] * (N_array[0] / N)**slope_ref for N in N_array]
    
    plt.figure(figsize=(8, 6))
    plt.loglog(N_array, errors, 'o-', label=label_err)
    plt.loglog(N_array, ref_line, 'k--', label=f"Slope {slope_ref} (ref.)")
    plt.xlabel(r'$N$')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which='both', ls='--')
    plt.xticks(N_array, labels=[str(v) for v in N_array])
    plt.legend()
    plt.show()


### ------------------- Main Analysis Functions ------------------- ###

def analyze_endpoint_errors(H, Val_loop, T, eta, xi0, para_b, para_sig, size_sample):
    """
    Runs the endpoint error simulations for both Goldfish and Elephant processes,
    for both sigma modes, computes metrics, and plots the results.
    """
    print(f"\n==================================================================")
    print(f"   STARTING ENDPOINT ERROR SIMULATION FOR H={H}")
    print(f"==================================================================\n")
    
    alpha = 0.5 + H
    errors_X_1 = np.zeros(len(Val_loop))
    errors_Y_1 = np.zeros(len(Val_loop))
    errors_X_2 = np.zeros(len(Val_loop))
    errors_Y_2 = np.zeros(len(Val_loop))
    
    start_time = time.time()
    for i, n in enumerate(tqdm(Val_loop, desc=f"Endpoint Errors (H={H})")):
        # mode_sig = 1
        List_X_N_1, List_X_2N_1, List_Y_N_1, List_Y_2N_1 = lib.Compute_m_endPoint_N(
            xi0, para_b, para_sig, 1, n, T, alpha, eta, size_sample
        )
        errors_X_1[i] = L2error_MonteCarlo_list(List_X_N_1, List_X_2N_1)
        errors_Y_1[i] = L2error_MonteCarlo_endPoint(np.array(List_Y_N_1), np.array(List_Y_2N_1)[:, ::2])
        
        # mode_sig = 2
        List_X_N_2, List_X_2N_2, List_Y_N_2, List_Y_2N_2 = lib.Compute_m_endPoint_N(
            xi0, para_b, para_sig, 2, n, T, alpha, eta, size_sample
        )
        errors_X_2[i] = L2error_MonteCarlo_list(List_X_N_2, List_X_2N_2)
        errors_Y_2[i] = L2error_MonteCarlo_endPoint(np.array(List_Y_N_2), np.array(List_Y_2N_2)[:, ::2])
        
    end_time = time.time()
    print(f"\nSimulation completed in {end_time - start_time:.2f} seconds.")

    # Plot Elephant Process
    label_X = r" $ \sqrt{E[|\overline{X}_T^{2N}-\overline{X}_T^N|^2]}$"
    _plot_and_print_regression(Val_loop, errors_X_1, f"Elephant EndPoint Error (H={H}, $\sigma_1$)", r"$L^2$ Error with $\sigma_1$", label_X)
    _plot_and_print_regression(Val_loop, errors_X_2, f"Elephant EndPoint Error (H={H}, $\sigma_2$)", r"$L^2$ Error with $\sigma_2$", label_X)

    # Plot Goldfish Process
    label_Y = r" $ \sqrt{E[|\overline{Y}_T^{2N}-\overline{Y}_T^N|^2]}$"
    _plot_and_print_regression(Val_loop, errors_Y_1, f"Goldfish EndPoint Error (H={H}, $\sigma_1$)", r"$L^2$ Error with $\sigma_1$", label_Y)
    _plot_and_print_regression(Val_loop, errors_Y_2, f"Goldfish EndPoint Error (H={H}, $\sigma_2$)", r"$L^2$ Error with $\sigma_2$", label_Y)


def analyze_max_errors(H, Val_loop, T, eta, xi0, para_b, para_sig, size_sample):
    """
    Runs the max error simulations across the entire trajectory for both 
    Goldfish and Elephant processes, computes metrics, and plots the results.
    """
    print(f"\n==================================================================")
    print(f"   STARTING MAX ERROR SIMULATION FOR H={H}")
    print(f"==================================================================\n")
    
    alpha = 0.5 + H
    errors_X_1 = np.zeros(len(Val_loop))
    errors_Y_1 = np.zeros(len(Val_loop))
    errors_X_2 = np.zeros(len(Val_loop))
    errors_Y_2 = np.zeros(len(Val_loop))
    
    start_time = time.time()
    for i, n in enumerate(tqdm(Val_loop, desc=f"Max Errors (H={H})")):
        # mode_sig = 1
        List_X_2N_1, List_X_N_1, List_Y_2N_1, List_Y_N_1 = lib.Compute_m_XN_X2N(
            xi0, para_b, para_sig, 1, n, T, alpha, eta, size_sample
        )
        errors_X_1[i] = L2error_MonteCarlo_numpy(np.array(List_X_N_1), np.array(List_X_2N_1)[:, ::2])
        errors_Y_1[i] = L2error_MonteCarlo_numpy(np.array(List_Y_N_1), np.array(List_Y_2N_1)[:, ::2])
        
        # mode_sig = 2
        List_X_2N_2, List_X_N_2, List_Y_2N_2, List_Y_N_2 = lib.Compute_m_XN_X2N(
            xi0, para_b, para_sig, 2, n, T, alpha, eta, size_sample
        )
        errors_X_2[i] = L2error_MonteCarlo_numpy(np.array(List_X_N_2), np.array(List_X_2N_2)[:, ::2])
        errors_Y_2[i] = L2error_MonteCarlo_numpy(np.array(List_Y_N_2), np.array(List_Y_2N_2)[:, ::2])
        
    end_time = time.time()
    print(f"\nSimulation completed in {end_time - start_time:.2f} seconds.")

    # Plot Elephant Process
    label_X = r" $ \max_k\sqrt{E[|\overline{X}_{t_k}^{2N}-\overline{X}_{t_k}^N|^2]}$"
    _plot_and_print_regression(Val_loop, errors_X_1, f"Elephant Max Error (H={H}, $\sigma_1$)", r"$L^2$ Error with $\sigma_1$", label_X)
    _plot_and_print_regression(Val_loop, errors_X_2, f"Elephant Max Error (H={H}, $\sigma_2$)", r"$L^2$ Error with $\sigma_2$", label_X)

    # Plot Goldfish Process
    label_Y = r" $ \max_k\sqrt{E[|\overline{Y}_{t_k}^{2N}-\overline{Y}_{t_k}^N|^2]}$"
    _plot_and_print_regression(Val_loop, errors_Y_1, f"Goldfish Max Error (H={H}, $\sigma_1$)", r"$L^2$ Error with $\sigma_1$", label_Y)
    _plot_and_print_regression(Val_loop, errors_Y_2, f"Goldfish Max Error (H={H}, $\sigma_2$)", r"$L^2$ Error with $\sigma_2$", label_Y)


#%%
#### --------------- Global Parameters Initialization --------------- ####

a = 0.384
b = 0.095
c = 0.0025
C_val = 1.2
para_sig1 = {'C': C_val, 'a': a, 'b': b, 'c': c}

mu = 2
lamb = 1.2
para_b1 = {'mu': mu, 'lam': lamb}

xi0 = 0
T = 1
size_sample = 1000
eta = 1

H_1 = 0.1
H_2 = 0.2
H_3 = 0.3
H_4 = 0.4

Val_loop = np.array([2**i for i in range(4, 11)])
# N values: 16, 32, 64, 128, 256, 512, 1024
print("Grid of N values:", Val_loop)

#%%
### ----------------------------------------------------------------------------------------------------------------------------###
# -------------------------- Main Execution Block --------------------------------------------------------------------------------#
### ----------------------------------------------------------------------------------------------------------------------------###

# Part I: Endpoint Errors
analyze_endpoint_errors(H_1, Val_loop, T, eta, xi0, para_b1, para_sig1, size_sample)
analyze_endpoint_errors(H_2, Val_loop, T, eta, xi0, para_b1, para_sig1, size_sample)
analyze_endpoint_errors(H_3, Val_loop, T, eta, xi0, para_b1, para_sig1, size_sample)
analyze_endpoint_errors(H_4, Val_loop, T, eta, xi0, para_b1, para_sig1, size_sample)

# Part II: Max Errors
analyze_max_errors(H_1, Val_loop, T, eta, xi0, para_b1, para_sig1, size_sample)
analyze_max_errors(H_2, Val_loop, T, eta, xi0, para_b1, para_sig1, size_sample)
analyze_max_errors(H_3, Val_loop, T, eta, xi0, para_b1, para_sig1, size_sample)
analyze_max_errors(H_4, Val_loop, T, eta, xi0, para_b1, para_sig1, size_sample)

print("\n--- All simulations and plots completed successfully. ---")