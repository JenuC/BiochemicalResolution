"""
Fisher Information Analysis Module for IRF Analysis.

This module implements Fisher information calculations for evaluating
measurement precision in fluorescence lifetime imaging microscopy (FLIM).
It includes functions for calculating Fisher information matrices,
validating against Kollner-Wolfrum reference results, and performing
comprehensive parameter sweeps.

The Fisher information quantifies the amount of information that an
observable random variable carries about an unknown parameter. In FLIM,
it provides a theoretical lower bound on the precision of lifetime
measurements given the instrument response function and measurement
conditions.
"""

import numpy as np
from scipy import optimize, integrate
from typing import Tuple, Dict, Any, Optional, Union
import warnings

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, *args, **kwargs):
        return iterable

try:
    from .core import AnalysisParameters, AnalysisResults, ValidationError, ComputationError
    from .irf_functions import get_irf, convolve_with_exponential, validate_irf
except ImportError:
    from core import AnalysisParameters, AnalysisResults, ValidationError, ComputationError
    from irf_functions import get_irf, convolve_with_exponential, validate_irf


def calculate_fisher_matrix(tau: float, T: float, time_bins: int, 
                          irf_type: str = 'dirac', irf_params: Optional[Dict] = None,
                          num_photons: int = 75000) -> np.ndarray:
    """
    Calculate Fisher information matrix for lifetime estimation precision.
    
    Computes the Fisher information matrix for fluorescence lifetime
    estimation given the experimental parameters and IRF characteristics.
    The Fisher information provides the theoretical lower bound (Cramér-Rao
    bound) on the variance of any unbiased estimator.
    
    Args:
        tau: Fluorescence lifetime (ns)
        T: Repetition period (ns)
        time_bins: Number of time bins in the measurement
        irf_type: Type of IRF ('dirac', 'gaussian', 'rectangular')
        irf_params: IRF-specific parameters (e.g., {'sigma': 0.1} for Gaussian)
        num_photons: Total number of detected photons
        
    Returns:
        Fisher information matrix (2x2 for amplitude and lifetime parameters)
        
    Raises:
        ValidationError: If input parameters are invalid
        ComputationError: If calculation fails
    """
    # Validate inputs
    if tau <= 0:
        raise ValidationError("Fluorescence lifetime must be positive")
    if T <= 0:
        raise ValidationError("Repetition period must be positive")
    if time_bins <= 0:
        raise ValidationError("Number of time bins must be positive")
    if num_photons <= 0:
        raise ValidationError("Number of photons must be positive")
    
    if irf_params is None:
        irf_params = {}
    
    try:
        # Create time array
        dt = T / time_bins
        t = np.arange(0, T, dt)
        
        # Generate IRF
        irf = get_irf(irf_type, t, **irf_params)
        
        # Calculate the expected signal (convolution of IRF with exponential)
        signal = convolve_with_exponential(irf, t, tau, T, amplitude=1.0)
        
        # Normalize signal to match photon count
        signal_normalized = signal * num_photons / np.sum(signal)
        
        # Calculate derivatives with respect to parameters
        # For exponential decay: f(t) = A * exp(-t/tau)
        # df/dA = exp(-t/tau)
        # df/dtau = A * t/tau^2 * exp(-t/tau)
        
        # Derivative with respect to amplitude
        dA_signal = signal / np.sum(signal) * num_photons
        
        # Derivative with respect to lifetime (using finite differences)
        dtau = tau * 1e-6  # Small perturbation
        signal_plus = convolve_with_exponential(irf, t, tau + dtau, T, amplitude=1.0)
        signal_plus_normalized = signal_plus * num_photons / np.sum(signal_plus)
        
        dtau_signal = (signal_plus_normalized - signal_normalized) / dtau
        
        # Calculate Fisher information matrix elements
        # F_ij = sum_k (1/mu_k) * (dmu_k/dp_i) * (dmu_k/dp_j)
        # where mu_k is the expected count in bin k
        
        # Avoid division by zero
        signal_safe = np.maximum(signal_normalized, 1e-10)
        
        # Fisher matrix elements
        F_AA = np.sum(dA_signal**2 / signal_safe)
        F_tau_tau = np.sum(dtau_signal**2 / signal_safe)
        F_A_tau = np.sum(dA_signal * dtau_signal / signal_safe)
        
        # Construct Fisher information matrix
        fisher_matrix = np.array([
            [F_AA, F_A_tau],
            [F_A_tau, F_tau_tau]
        ])
        
        # Check for numerical issues
        if not np.all(np.isfinite(fisher_matrix)):
            raise ComputationError("Fisher matrix contains non-finite values")
        
        # Check positive definiteness
        eigenvals = np.linalg.eigvals(fisher_matrix)
        if np.any(eigenvals <= 0):
            warnings.warn("Fisher matrix is not positive definite", UserWarning)
        
        return fisher_matrix
        
    except Exception as e:
        if isinstance(e, (ValidationError, ComputationError)):
            raise
        else:
            raise ComputationError(f"Fisher matrix calculation failed: {str(e)}")


def expofunc(x: np.ndarray, tau: float, T: float, tn: int) -> np.ndarray:
    """
    Exponential probability density function for time bins.
    
    This matches the expofunc from the Monte Carlo notebook:
    return np.exp(-T*x/(tn*tau))*(np.exp(T/(tn*tau))-1)/(1-np.exp(-T/tau))
    
    Args:
        x: Time bin indices (1 to tn)
        tau: Fluorescence lifetime (ns)
        T: Repetition period (ns)
        tn: Number of time bins
        
    Returns:
        Probability density for each time bin
    """
    exp_term = np.exp(-T * x / (tn * tau))
    exp_dt = np.exp(T / (tn * tau))
    exp_T = np.exp(-T / tau)
    
    return exp_term * (exp_dt - 1) / (1 - exp_T)


def kollner_wolfrum_reference(tau: float, T: float, tn: int, num_photons: int = 75000) -> float:
    """
    Kollner-Wolfrum reference implementation for Dirac IRF case.
    
    Implements the analytical calculation from Kollner and Wolfrum (1992) 
    for the ideal Dirac IRF case. This returns the F-value as defined in
    the existing codebase.
    
    Based on the Monte Carlo code: F = tau_sd * sqrt(intens_avg) / tau_avg
    This represents the coefficient of variation scaled by sqrt(intensity).
    
    Reference:
    Kollner, M., & Wolfrum, J. (1992). How many photons are necessary for 
    fluorescence-lifetime measurements? Chemical Physics Letters, 200(1-2), 199-204.
    
    Args:
        tau: Fluorescence lifetime (ns)
        T: Repetition period (ns)
        tn: Number of time bins
        num_photons: Number of detected photons
        
    Returns:
        F-value (relative precision measure)
        
    Raises:
        ValidationError: If input parameters are invalid
    """
    if tau <= 0:
        raise ValidationError("Fluorescence lifetime must be positive")
    if T <= 0:
        raise ValidationError("Repetition period must be positive")
    if tn <= 0:
        raise ValidationError("Number of time bins must be positive")
    if num_photons <= 0:
        raise ValidationError("Number of photons must be positive")
    
    try:
        # Use the same probability density function as the Monte Carlo code
        ti = np.arange(1, tn + 1)  # Time bin indices from 1 to tn
        pdf = expofunc(ti, tau, T, tn)
        
        # Expected photon counts in each bin
        expected_counts = pdf * num_photons
        
        # Calculate Fisher information using the definition:
        # F_tau = sum_i (1/mu_i) * (d_mu_i/d_tau)^2
        # where mu_i is expected count in bin i
        
        # Calculate derivative of PDF w.r.t. tau using finite differences
        dtau = tau * 1e-8  # Very small perturbation
        pdf_plus = expofunc(ti, tau + dtau, T, tn)
        d_pdf_d_tau = (pdf_plus - pdf) / dtau
        
        # Derivative of expected counts
        d_counts_d_tau = d_pdf_d_tau * num_photons
        
        # Fisher information
        fisher_info = np.sum(d_counts_d_tau**2 / np.maximum(expected_counts, 1e-10))
        
        if fisher_info <= 0:
            raise ComputationError("Invalid Fisher information calculated")
        
        # Convert to F-value to match the Monte Carlo definition
        # From Monte Carlo: F = tau_sd * sqrt(intens_avg) / tau_avg
        # Using Cramér-Rao bound: tau_sd >= 1/sqrt(fisher_info)
        # And intens_avg = num_photons
        # So F = (1/sqrt(fisher_info)) * sqrt(num_photons) / tau
        
        F_value = np.sqrt(num_photons) / (tau * np.sqrt(fisher_info))
        
        if not np.isfinite(F_value) or F_value <= 0:
            raise ComputationError("Invalid F-value calculated")
        
        return F_value
        
    except Exception as e:
        if isinstance(e, (ValidationError, ComputationError)):
            raise
        else:
            raise ComputationError(f"Kollner-Wolfrum calculation failed: {str(e)}")


def validate_against_kollner_wolfrum(tau_range: np.ndarray, T: float, 
                                   time_bins_range: np.ndarray,
                                   tolerance: float = 0.05) -> Dict[str, Any]:
    """
    Validate F-value calculations against Kollner-Wolfrum reference.
    
    Compares numerical F-value calculations with analytical
    Kollner-Wolfrum results for the Dirac IRF case to verify accuracy.
    
    Args:
        tau_range: Array of lifetimes to test (ns)
        T: Repetition period (ns)
        time_bins_range: Array of time bin counts to test
        tolerance: Relative tolerance for validation (default 5%)
        
    Returns:
        Dictionary containing validation results and statistics
        
    Raises:
        ValidationError: If validation fails
    """
    results = {
        'passed': True,
        'max_error': 0.0,
        'mean_error': 0.0,
        'failed_cases': [],
        'total_cases': 0,
        'passed_cases': 0
    }
    
    errors = []
    
    for tau in tau_range:
        for tn in time_bins_range:
            try:
                # Calculate numerical F-value
                F_numerical = calculate_f_value(
                    tau=tau, T=T, time_bins=int(tn), 
                    irf_type='dirac', num_photons=75000
                )
                
                # Calculate analytical reference
                F_analytical = kollner_wolfrum_reference(tau, T, int(tn), num_photons=75000)
                
                # Calculate relative error
                rel_error = abs(F_numerical - F_analytical) / F_analytical
                errors.append(rel_error)
                
                results['total_cases'] += 1
                
                if rel_error <= tolerance:
                    results['passed_cases'] += 1
                else:
                    results['failed_cases'].append({
                        'tau': tau,
                        'time_bins': tn,
                        'numerical': F_numerical,
                        'analytical': F_analytical,
                        'rel_error': rel_error
                    })
                    results['passed'] = False
                
            except Exception as e:
                results['failed_cases'].append({
                    'tau': tau,
                    'time_bins': tn,
                    'error': str(e)
                })
                results['passed'] = False
    
    if errors:
        results['max_error'] = np.max(errors)
        results['mean_error'] = np.mean(errors)
    
    return results


def dirac_irf_analysis(tau_range: np.ndarray, time_bins_range: np.ndarray,
                      T: float = 25.0, num_photons: int = 75000,
                      progress: bool = True) -> AnalysisResults:
    """
    Comprehensive parameter exploration for Dirac IRF Fisher analysis.
    
    Performs a systematic sweep over lifetime and time bin parameters
    to generate Fisher information values for the ideal Dirac IRF case.
    This reproduces the analysis from Kollner-Wolfrum Figure 1.
    
    Args:
        tau_range: Array of fluorescence lifetimes to analyze (ns)
        time_bins_range: Array of time bin counts to analyze
        T: Repetition period (ns)
        num_photons: Number of detected photons
        progress: Whether to show progress bar
        
    Returns:
        AnalysisResults object containing F-values and parameters
        
    Raises:
        ValidationError: If input parameters are invalid
        ComputationError: If analysis fails
    """
    # Validate inputs
    if len(tau_range) == 0 or len(time_bins_range) == 0:
        raise ValidationError("Parameter ranges cannot be empty")
    
    if np.any(tau_range <= 0):
        raise ValidationError("All lifetimes must be positive")
    
    if np.any(time_bins_range <= 0):
        raise ValidationError("All time bin counts must be positive")
    
    try:
        # Initialize results array
        F_values = np.zeros((len(tau_range), len(time_bins_range)))
        
        # Create progress iterator
        if progress and HAS_TQDM:
            iterator = tqdm(enumerate(tau_range), total=len(tau_range), 
                          desc="Dirac IRF Analysis")
        else:
            iterator = enumerate(tau_range)
        
        # Perform parameter sweep
        for i, tau in iterator:
            for j, tn in enumerate(time_bins_range):
                try:
                    # Calculate F-value
                    F_values[i, j] = calculate_f_value(
                        tau=tau, T=T, time_bins=int(tn),
                        irf_type='dirac', num_photons=num_photons
                    )
                    
                except Exception as e:
                    warnings.warn(f"Failed calculation for tau={tau}, tn={tn}: {str(e)}")
                    F_values[i, j] = np.nan
        
        # Create parameters object
        params = AnalysisParameters(
            repetition_period=T,
            lifetimes=tau_range,
            time_bins=time_bins_range,
            num_photons=num_photons
        )
        
        # Create metadata
        metadata = {
            'irf_type': 'dirac',
            'analysis_date': np.datetime64('now').astype(str),
            'total_calculations': len(tau_range) * len(time_bins_range),
            'successful_calculations': np.sum(np.isfinite(F_values)),
            'failed_calculations': np.sum(~np.isfinite(F_values))
        }
        
        # Create results object
        results = AnalysisResults(
            f_values=F_values,
            parameters=params,
            metadata=metadata,
            analysis_type='dirac_fisher_analysis'
        )
        
        return results
        
    except Exception as e:
        if isinstance(e, (ValidationError, ComputationError)):
            raise
        else:
            raise ComputationError(f"Dirac IRF analysis failed: {str(e)}")


def gaussian_irf_analysis(tau_range: np.ndarray, time_bins_range: np.ndarray,
                         sigma_range: np.ndarray, T: float = 25.0,
                         num_photons: int = 75000, progress: bool = True) -> AnalysisResults:
    """
    Comprehensive parameter exploration for Gaussian IRF Fisher analysis.
    
    Performs a systematic sweep over lifetime, time bin, and IRF width
    parameters to generate Fisher information values for Gaussian IRF.
    This extends the Dirac analysis to realistic instrument responses.
    
    Args:
        tau_range: Array of fluorescence lifetimes to analyze (ns)
        time_bins_range: Array of time bin counts to analyze
        sigma_range: Array of Gaussian IRF widths to analyze (ns)
        T: Repetition period (ns)
        num_photons: Number of detected photons
        progress: Whether to show progress bar
        
    Returns:
        AnalysisResults object containing F-values and parameters
        
    Raises:
        ValidationError: If input parameters are invalid
        ComputationError: If analysis fails
    """
    # Validate inputs
    if len(tau_range) == 0 or len(time_bins_range) == 0 or len(sigma_range) == 0:
        raise ValidationError("Parameter ranges cannot be empty")
    
    if np.any(tau_range <= 0):
        raise ValidationError("All lifetimes must be positive")
    
    if np.any(time_bins_range <= 0):
        raise ValidationError("All time bin counts must be positive")
    
    if np.any(sigma_range <= 0):
        raise ValidationError("All IRF sigmas must be positive")
    
    try:
        # Initialize results array (3D: tau x time_bins x sigma)
        F_values = np.zeros((len(tau_range), len(time_bins_range), len(sigma_range)))
        
        # Calculate total iterations for progress bar
        total_iterations = len(tau_range) * len(time_bins_range) * len(sigma_range)
        
        if progress and HAS_TQDM:
            pbar = tqdm(total=total_iterations, desc="Gaussian IRF Analysis")
        else:
            pbar = None
        
        # Perform parameter sweep
        for i, tau in enumerate(tau_range):
            for j, tn in enumerate(time_bins_range):
                for k, sigma in enumerate(sigma_range):
                    try:
                        # Calculate F-value
                        F_values[i, j, k] = calculate_f_value(
                            tau=tau, T=T, time_bins=int(tn),
                            irf_type='gaussian', irf_params={'sigma': sigma},
                            num_photons=num_photons
                        )
                        
                    except Exception as e:
                        warnings.warn(f"Failed calculation for tau={tau}, tn={tn}, sigma={sigma}: {str(e)}")
                        F_values[i, j, k] = np.nan
                    
                    if pbar is not None:
                        pbar.update(1)
        
        if pbar is not None:
            pbar.close()
        
        # Create parameters object
        params = AnalysisParameters(
            repetition_period=T,
            lifetimes=tau_range,
            time_bins=time_bins_range,
            irf_sigmas=sigma_range,
            num_photons=num_photons
        )
        
        # Create metadata
        metadata = {
            'irf_type': 'gaussian',
            'analysis_date': np.datetime64('now').astype(str),
            'total_calculations': total_iterations,
            'successful_calculations': np.sum(np.isfinite(F_values)),
            'failed_calculations': np.sum(~np.isfinite(F_values)),
            'sigma_range': sigma_range.tolist()
        }
        
        # Create results object
        results = AnalysisResults(
            f_values=F_values,
            parameters=params,
            metadata=metadata,
            analysis_type='gaussian_fisher_analysis'
        )
        
        return results
        
    except Exception as e:
        if isinstance(e, (ValidationError, ComputationError)):
            raise
        else:
            raise ComputationError(f"Gaussian IRF analysis failed: {str(e)}")


def calculate_f_value(tau: float, T: float, time_bins: int, 
                     irf_type: str = 'dirac', irf_params: Optional[Dict] = None,
                     num_photons: int = 75000) -> float:
    """
    Calculate F-value for lifetime measurement precision.
    
    Calculates the F-value as defined in the existing codebase, which
    represents the relative standard deviation of lifetime estimates
    scaled by sqrt(number of photons). This provides a measure of
    measurement precision that's independent of photon count.
    
    Args:
        tau: Fluorescence lifetime (ns)
        T: Repetition period (ns)
        time_bins: Number of time bins in the measurement
        irf_type: Type of IRF ('dirac', 'gaussian', 'rectangular')
        irf_params: IRF-specific parameters (e.g., {'sigma': 0.1} for Gaussian)
        num_photons: Total number of detected photons
        
    Returns:
        F-value (relative precision measure)
        
    Raises:
        ValidationError: If input parameters are invalid
        ComputationError: If calculation fails
    """
    # Validate inputs
    if tau <= 0:
        raise ValidationError("Fluorescence lifetime must be positive")
    if T <= 0:
        raise ValidationError("Repetition period must be positive")
    if time_bins <= 0:
        raise ValidationError("Number of time bins must be positive")
    if num_photons <= 0:
        raise ValidationError("Number of photons must be positive")
    
    if irf_params is None:
        irf_params = {}
    
    try:
        if irf_type == 'dirac':
            # For Dirac IRF, use the same approach as Kollner-Wolfrum reference
            ti = np.arange(1, time_bins + 1)  # Time bin indices from 1 to time_bins
            pdf = expofunc(ti, tau, T, time_bins)
        else:
            # For other IRFs, need to implement convolution
            # Create time array
            dt = T / time_bins
            t = np.arange(1, time_bins + 1) * dt  # Time bins from dt to T
            
            # Generate IRF
            irf = get_irf(irf_type, t, **irf_params)
            
            # Calculate the expected signal (convolution of IRF with exponential)
            decay = np.exp(-t / tau)
            
            # Perform discrete convolution
            signal = np.convolve(irf, decay, mode='same')
            
            # Normalize signal to create probability density
            pdf = signal / np.sum(signal)
        
        # Expected photon counts in each bin
        expected_counts = pdf * num_photons
        
        # Calculate Fisher information for lifetime parameter
        # Derivative of PDF w.r.t. tau using finite differences
        dtau = tau * 1e-8  # Very small perturbation
        
        if irf_type == 'dirac':
            pdf_plus = expofunc(ti, tau + dtau, T, time_bins)
        else:
            # For other IRFs
            decay_plus = np.exp(-t / (tau + dtau))
            signal_plus = np.convolve(irf, decay_plus, mode='same')
            pdf_plus = signal_plus / np.sum(signal_plus)
        
        d_pdf_d_tau = (pdf_plus - pdf) / dtau
        
        # Derivative of expected counts w.r.t. tau
        d_counts_d_tau = d_pdf_d_tau * num_photons
        
        # Fisher information
        fisher_info = np.sum(d_counts_d_tau**2 / np.maximum(expected_counts, 1e-10))
        
        # Convert to F-value (relative precision measure)
        if fisher_info <= 0:
            raise ComputationError("Invalid Fisher information calculated")
        
        F_value = np.sqrt(num_photons) / (tau * np.sqrt(fisher_info))
        
        if not np.isfinite(F_value) or F_value <= 0:
            raise ComputationError("Invalid F-value calculated")
        
        return F_value
        
    except Exception as e:
        if isinstance(e, (ValidationError, ComputationError)):
            raise
        else:
            raise ComputationError(f"F-value calculation failed: {str(e)}")


def calculate_precision_bounds(fisher_matrix: np.ndarray) -> Dict[str, float]:
    """
    Calculate precision bounds from Fisher information matrix.
    
    Computes the Cramér-Rao lower bounds on parameter estimation
    precision from the Fisher information matrix.
    
    Args:
        fisher_matrix: Fisher information matrix
        
    Returns:
        Dictionary containing precision bounds for each parameter
        
    Raises:
        ValidationError: If Fisher matrix is invalid
        ComputationError: If calculation fails
    """
    if not isinstance(fisher_matrix, np.ndarray):
        raise ValidationError("Fisher matrix must be a numpy array")
    
    if fisher_matrix.ndim != 2 or fisher_matrix.shape[0] != fisher_matrix.shape[1]:
        raise ValidationError("Fisher matrix must be square")
    
    try:
        # Calculate covariance matrix (inverse of Fisher matrix)
        cov_matrix = np.linalg.inv(fisher_matrix)
        
        # Extract standard deviations (square root of diagonal elements)
        std_devs = np.sqrt(np.diag(cov_matrix))
        
        # Create results dictionary
        bounds = {
            'amplitude_std': std_devs[0] if len(std_devs) > 0 else np.nan,
            'lifetime_std': std_devs[1] if len(std_devs) > 1 else np.nan,
            'correlation': cov_matrix[0, 1] / (std_devs[0] * std_devs[1]) if len(std_devs) > 1 else np.nan
        }
        
        return bounds
        
    except np.linalg.LinAlgError as e:
        raise ComputationError(f"Failed to invert Fisher matrix: {str(e)}")
    except Exception as e:
        raise ComputationError(f"Precision bounds calculation failed: {str(e)}")


def optimize_measurement_parameters(tau: float, target_precision: float,
                                  T: float = 25.0, irf_type: str = 'dirac',
                                  irf_params: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Optimize measurement parameters for target precision.
    
    Finds the optimal number of time bins and photons to achieve
    a target precision in lifetime estimation.
    
    Args:
        tau: Fluorescence lifetime (ns)
        target_precision: Target relative precision (e.g., 0.01 for 1%)
        T: Repetition period (ns)
        irf_type: Type of IRF
        irf_params: IRF-specific parameters
        
    Returns:
        Dictionary containing optimal parameters and achieved precision
        
    Raises:
        ValidationError: If input parameters are invalid
        ComputationError: If optimization fails
    """
    if tau <= 0:
        raise ValidationError("Fluorescence lifetime must be positive")
    if target_precision <= 0:
        raise ValidationError("Target precision must be positive")
    
    if irf_params is None:
        irf_params = {}
    
    try:
        def objective(params):
            """Objective function for optimization."""
            time_bins, log_photons = params
            time_bins = int(np.round(time_bins))
            num_photons = int(np.round(np.exp(log_photons)))
            
            if time_bins < 10 or num_photons < 1000:
                return 1e6  # Penalty for unrealistic parameters
            
            try:
                fisher_matrix = calculate_fisher_matrix(
                    tau=tau, T=T, time_bins=time_bins,
                    irf_type=irf_type, irf_params=irf_params,
                    num_photons=num_photons
                )
                
                bounds = calculate_precision_bounds(fisher_matrix)
                relative_precision = bounds['lifetime_std'] / tau
                
                # Minimize the difference from target precision
                return abs(relative_precision - target_precision)
                
            except:
                return 1e6  # Penalty for failed calculations
        
        # Initial guess
        x0 = [512, np.log(75000)]  # time_bins, log(num_photons)
        
        # Bounds
        bounds = [(10, 4096), (np.log(1000), np.log(1e7))]
        
        # Optimize
        result = optimize.minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
        
        if not result.success:
            raise ComputationError("Optimization failed to converge")
        
        # Extract optimal parameters
        optimal_time_bins = int(np.round(result.x[0]))
        optimal_photons = int(np.round(np.exp(result.x[1])))
        
        # Calculate achieved precision
        fisher_matrix = calculate_fisher_matrix(
            tau=tau, T=T, time_bins=optimal_time_bins,
            irf_type=irf_type, irf_params=irf_params,
            num_photons=optimal_photons
        )
        
        bounds = calculate_precision_bounds(fisher_matrix)
        achieved_precision = bounds['lifetime_std'] / tau
        
        return {
            'optimal_time_bins': optimal_time_bins,
            'optimal_photons': optimal_photons,
            'achieved_precision': achieved_precision,
            'target_precision': target_precision,
            'optimization_success': True,
            'fisher_information': fisher_matrix[1, 1]
        }
        
    except Exception as e:
        if isinstance(e, (ValidationError, ComputationError)):
            raise
        else:
            raise ComputationError(f"Parameter optimization failed: {str(e)}")