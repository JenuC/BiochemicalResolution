"""
Monte Carlo Simulation Module for IRF Analysis.

This module implements Monte Carlo simulations for validating Fisher information
analysis and performing extended analysis of instrument response functions in
fluorescence lifetime imaging microscopy (FLIM). It includes photon data
generation, curve fitting routines, and comprehensive analysis pipelines.

The Monte Carlo approach provides empirical validation of theoretical Fisher
information predictions and enables analysis of realistic experimental
conditions including noise, fitting convergence, and systematic errors.
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy import signal
from typing import Tuple, Dict, Any, Optional, Union, Callable
import warnings
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, *args, **kwargs):
        return iterable

try:
    from .core import AnalysisParameters, AnalysisResults, ValidationError, ComputationError, ConvergenceError
    from .irf_functions import get_irf, convolve_with_exponential, exponential_decay
except ImportError:
    from core import AnalysisParameters, AnalysisResults, ValidationError, ComputationError, ConvergenceError
    from irf_functions import get_irf, convolve_with_exponential, exponential_decay


def expofunc(x: np.ndarray, tau: float, T: float, tn: int) -> np.ndarray:
    """
    Exponential probability density function for time bins.
    
    This matches the expofunc from the existing Monte Carlo notebook:
    return np.exp(-T*x/(tn*tau))*(np.exp(T/(tn*tau))-1)/(1-np.exp(-T/tau))
    
    Args:
        x: Time bin indices (1 to tn)
        tau: Fluorescence lifetime (ns)
        T: Repetition period (ns)
        tn: Number of time bins
        
    Returns:
        Probability density for each time bin
    """
    if tau <= 0:
        raise ValidationError("Fluorescence lifetime must be positive")
    if T <= 0:
        raise ValidationError("Repetition period must be positive")
    if tn <= 0:
        raise ValidationError("Number of time bins must be positive")
    
    # Avoid division by zero and overflow
    exp_arg1 = -T * x / (tn * tau)
    exp_arg2 = T / (tn * tau)
    exp_arg3 = -T / tau
    
    # Check for potential overflow
    if np.any(exp_arg2 > 700):  # exp(700) is near overflow
        warnings.warn("Large exponential arguments detected, results may be inaccurate")
    
    exp_term = np.exp(exp_arg1)
    exp_dt = np.exp(exp_arg2)
    exp_T = np.exp(exp_arg3)
    
    # Handle case where exp_T approaches 1 (very long lifetimes)
    denominator = 1 - exp_T
    if abs(denominator) < 1e-15:
        # Use Taylor expansion for small arguments
        denominator = T / tau
    
    return exp_term * (exp_dt - 1) / denominator


def generate_photon_data(pdf: np.ndarray, num_photons: int, iterations: int = 1,
                        random_seed: Optional[int] = None) -> np.ndarray:
    """
    Generate synthetic photon data with Poisson statistics.
    
    Creates synthetic photon count data by sampling from a Poisson distribution
    with expected values determined by the probability density function and
    total photon count. This simulates the shot noise inherent in photon
    counting measurements.
    
    Args:
        pdf: Probability density function for photon arrival times
        num_photons: Total number of photons to simulate
        iterations: Number of independent measurements to generate
        random_seed: Random seed for reproducible results
        
    Returns:
        Array of photon counts with shape (iterations, len(pdf))
        
    Raises:
        ValidationError: If input parameters are invalid
        ComputationError: If data generation fails
    """
    # Validate inputs
    if not isinstance(pdf, np.ndarray):
        pdf = np.array(pdf)
    
    if len(pdf) == 0:
        raise ValidationError("PDF array cannot be empty")
    
    if np.any(pdf < 0):
        raise ValidationError("PDF contains negative values")
    
    if np.sum(pdf) == 0:
        raise ValidationError("PDF sums to zero")
    
    if num_photons <= 0:
        raise ValidationError("Number of photons must be positive")
    
    if iterations <= 0:
        raise ValidationError("Number of iterations must be positive")
    
    try:
        # Normalize PDF to ensure it sums to 1
        pdf_normalized = pdf / np.sum(pdf)
        
        # Calculate expected photon counts in each bin
        expected_counts = pdf_normalized * num_photons
        
        # Set up random number generator with seed for reproducibility
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
        else:
            rng = np.random.default_rng()
        
        # Generate Poisson-distributed photon counts
        photon_data = rng.poisson(expected_counts, (iterations, len(pdf)))
        
        # Validate generated data
        if not np.all(np.isfinite(photon_data)):
            raise ComputationError("Generated data contains non-finite values")
        
        if np.any(photon_data < 0):
            raise ComputationError("Generated data contains negative values")
        
        return photon_data
        
    except Exception as e:
        if isinstance(e, (ValidationError, ComputationError)):
            raise
        else:
            raise ComputationError(f"Photon data generation failed: {str(e)}")


def calculate_pdf_dirac(time_bins: np.ndarray, tau: float, T: float, 
                       tn: int) -> np.ndarray:
    """
    Calculate probability density function for Dirac IRF case.
    
    Computes the PDF for photon arrival times assuming an ideal Dirac
    delta IRF. This matches the expofunc implementation from the existing
    Monte Carlo notebook.
    
    Args:
        time_bins: Time bin indices (typically 1 to tn)
        tau: Fluorescence lifetime (ns)
        T: Repetition period (ns)
        tn: Total number of time bins
        
    Returns:
        Normalized probability density function
        
    Raises:
        ValidationError: If input parameters are invalid
    """
    if not isinstance(time_bins, np.ndarray):
        time_bins = np.array(time_bins)
    
    if len(time_bins) == 0:
        raise ValidationError("Time bins array cannot be empty")
    
    # Calculate PDF using the existing expofunc
    pdf = expofunc(time_bins, tau, T, tn)
    
    # Ensure normalization
    pdf_sum = np.sum(pdf)
    if pdf_sum == 0:
        raise ComputationError("PDF sums to zero")
    
    return pdf / pdf_sum


def calculate_pdf_gaussian(time_bins: np.ndarray, tau: float, T: float,
                          tn: int, sigma: float, t0: Optional[int] = None) -> np.ndarray:
    """
    Calculate probability density function for Gaussian IRF case.
    
    Computes the PDF for photon arrival times with a Gaussian IRF by
    convolving the exponential decay with a Gaussian instrument response.
    This matches the expogausfunc implementation from the existing notebook.
    
    Args:
        time_bins: Time bin indices (typically 1 to tn)
        tau: Fluorescence lifetime (ns)
        T: Repetition period (ns)
        tn: Total number of time bins
        sigma: Gaussian IRF standard deviation (ns)
        t0: IRF center position (default: tn//2)
        
    Returns:
        Normalized probability density function
        
    Raises:
        ValidationError: If input parameters are invalid
        ComputationError: If convolution fails
    """
    if not isinstance(time_bins, np.ndarray):
        time_bins = np.array(time_bins)
    
    if len(time_bins) == 0:
        raise ValidationError("Time bins array cannot be empty")
    
    if sigma <= 0:
        raise ValidationError("Gaussian sigma must be positive")
    
    if t0 is None:
        t0 = tn // 2
    
    try:
        # Calculate time width per bin
        twidth = T / tn
        
        # Create Gaussian IRF
        def irfgaus(x):
            """Gaussian IRF function matching the notebook implementation."""
            sigma_bins = sigma / twidth  # Convert to bin units
            normfact = np.sum((1/np.sqrt(2*math.pi*sigma_bins**2)) * 
                             np.exp(-(x-t0)**2/(2*sigma_bins**2)))
            return ((1/np.sqrt(2*math.pi*sigma_bins**2)) * 
                   np.exp(-(x-t0)**2/(2*sigma_bins**2))) / normfact
        
        # Calculate exponential decay PDF
        expo_pdf = expofunc(time_bins, tau, T, tn)
        
        # Calculate Gaussian IRF
        irf = irfgaus(time_bins)
        
        # Perform convolution
        pdf = np.convolve(expo_pdf, irf, mode='full')
        
        # Extract the relevant portion (matching notebook implementation)
        pdf = pdf[(t0-1):(3*t0-1)]
        
        # Ensure we have the right length
        if len(pdf) != len(time_bins):
            # Pad or truncate to match input length
            if len(pdf) < len(time_bins):
                pdf = np.pad(pdf, (0, len(time_bins) - len(pdf)), 'constant')
            else:
                pdf = pdf[:len(time_bins)]
        
        # Normalize
        pdf_sum = np.sum(pdf)
        if pdf_sum == 0:
            raise ComputationError("Convolved PDF sums to zero")
        
        return pdf / pdf_sum
        
    except Exception as e:
        if isinstance(e, (ValidationError, ComputationError)):
            raise
        else:
            raise ComputationError(f"Gaussian PDF calculation failed: {str(e)}")


def calculate_pdf_rectangular(time_bins: np.ndarray, tau: float, T: float,
                             tn: int, width: float, t0: Optional[int] = None) -> np.ndarray:
    """
    Calculate probability density function for rectangular IRF case.
    
    Computes the PDF for photon arrival times with a rectangular (square wave)
    IRF by convolving the exponential decay with a rectangular instrument response.
    This matches the exposqrfunc implementation from the existing notebook.
    
    Args:
        time_bins: Time bin indices (typically 1 to tn)
        tau: Fluorescence lifetime (ns)
        T: Repetition period (ns)
        tn: Total number of time bins
        width: Rectangular IRF width (ns) - uses FWHM convention
        t0: IRF center position (default: tn//2)
        
    Returns:
        Normalized probability density function
        
    Raises:
        ValidationError: If input parameters are invalid
        ComputationError: If convolution fails
    """
    if not isinstance(time_bins, np.ndarray):
        time_bins = np.array(time_bins)
    
    if len(time_bins) == 0:
        raise ValidationError("Time bins array cannot be empty")
    
    if width <= 0:
        raise ValidationError("Rectangular width must be positive")
    
    if t0 is None:
        t0 = tn // 2
    
    try:
        # Calculate time width per bin
        twidth = T / tn
        
        # Create rectangular IRF (matching notebook implementation)
        def irfsqr(x):
            """Rectangular IRF function matching the notebook implementation."""
            # Convert FWHM to half-width in bin units (using 2.355 factor from notebook)
            half_width_bins = (2.355 * width) / (2 * twidth)
            mask = np.abs(x - t0) <= half_width_bins
            normfact = np.sum(mask)
            return mask / normfact if normfact > 0 else mask
        
        # Calculate exponential decay PDF
        expo_pdf = expofunc(time_bins, tau, T, tn)
        
        # Calculate rectangular IRF
        irf = irfsqr(time_bins)
        
        # Perform convolution
        pdf = np.convolve(expo_pdf, irf, mode='full')
        
        # Extract the relevant portion (matching notebook implementation)
        pdf = pdf[(t0-1):(3*t0-1)]
        
        # Ensure we have the right length
        if len(pdf) != len(time_bins):
            # Pad or truncate to match input length
            if len(pdf) < len(time_bins):
                pdf = np.pad(pdf, (0, len(time_bins) - len(pdf)), 'constant')
            else:
                pdf = pdf[:len(time_bins)]
        
        # Normalize
        pdf_sum = np.sum(pdf)
        if pdf_sum == 0:
            raise ComputationError("Convolved PDF sums to zero")
        
        return pdf / pdf_sum
        
    except Exception as e:
        if isinstance(e, (ValidationError, ComputationError)):
            raise
        else:
            raise ComputationError(f"Rectangular PDF calculation failed: {str(e)}")


def calculate_pdf_for_irf(time_bins: np.ndarray, tau: float, T: float, tn: int,
                         irf_type: str = 'dirac', irf_params: Optional[Dict] = None) -> np.ndarray:
    """
    Calculate probability density function for specified IRF type.
    
    Factory function that calculates the appropriate PDF based on the
    IRF type and parameters. Supports all IRF types implemented in
    the existing Monte Carlo notebook.
    
    Args:
        time_bins: Time bin indices (typically 1 to tn)
        tau: Fluorescence lifetime (ns)
        T: Repetition period (ns)
        tn: Total number of time bins
        irf_type: Type of IRF ('dirac', 'gaussian', 'rectangular')
        irf_params: IRF-specific parameters
        
    Returns:
        Normalized probability density function
        
    Raises:
        ValidationError: If IRF type is unknown or parameters invalid
    """
    if irf_params is None:
        irf_params = {}
    
    irf_type = irf_type.lower()
    
    if irf_type == 'dirac':
        return calculate_pdf_dirac(time_bins, tau, T, tn)
    
    elif irf_type == 'gaussian':
        sigma = irf_params.get('sigma', 0.1)
        t0 = irf_params.get('t0', None)
        return calculate_pdf_gaussian(time_bins, tau, T, tn, sigma, t0)
    
    elif irf_type == 'rectangular':
        width = irf_params.get('width', 0.2)
        t0 = irf_params.get('t0', None)
        return calculate_pdf_rectangular(time_bins, tau, T, tn, width, t0)
    
    else:
        raise ValidationError(f"Unknown IRF type: {irf_type}")


def generate_measurement_data(tau: float, T: float, tn: int, num_photons: int,
                             iterations: int, irf_type: str = 'dirac',
                             irf_params: Optional[Dict] = None,
                             random_seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate complete measurement data for Monte Carlo analysis.
    
    Creates synthetic photon count data for a complete measurement scenario,
    including the probability density function and corresponding photon counts
    with Poisson noise. This provides the foundation for curve fitting and
    precision analysis.
    
    Args:
        tau: Fluorescence lifetime (ns)
        T: Repetition period (ns)
        tn: Number of time bins
        num_photons: Total number of photons per measurement
        iterations: Number of independent measurements
        irf_type: Type of IRF ('dirac', 'gaussian', 'rectangular')
        irf_params: IRF-specific parameters
        random_seed: Random seed for reproducible results
        
    Returns:
        Tuple of (photon_data, pdf) where:
        - photon_data: Array of shape (iterations, tn) with photon counts
        - pdf: Probability density function used for generation
        
    Raises:
        ValidationError: If input parameters are invalid
        ComputationError: If data generation fails
    """
    # Validate inputs
    if tau <= 0:
        raise ValidationError("Fluorescence lifetime must be positive")
    if T <= 0:
        raise ValidationError("Repetition period must be positive")
    if tn <= 0:
        raise ValidationError("Number of time bins must be positive")
    if num_photons <= 0:
        raise ValidationError("Number of photons must be positive")
    if iterations <= 0:
        raise ValidationError("Number of iterations must be positive")
    
    if irf_params is None:
        irf_params = {}
    
    try:
        # Create time bin indices (1 to tn, matching notebook convention)
        time_bins = np.arange(1, tn + 1)
        
        # Calculate probability density function
        pdf = calculate_pdf_for_irf(time_bins, tau, T, tn, irf_type, irf_params)
        
        # Generate photon data with Poisson statistics
        photon_data = generate_photon_data(pdf, num_photons, iterations, random_seed)
        
        return photon_data, pdf
        
    except Exception as e:
        if isinstance(e, (ValidationError, ComputationError)):
            raise
        else:
            raise ComputationError(f"Measurement data generation failed: {str(e)}")


def validate_photon_data(photon_data: np.ndarray, expected_total: int,
                        tolerance: float = 0.1) -> Dict[str, Any]:
    """
    Validate generated photon data for consistency and realism.
    
    Performs comprehensive validation of generated photon count data,
    checking for statistical consistency, realistic count levels,
    and absence of numerical artifacts.
    
    Args:
        photon_data: Array of photon counts with shape (iterations, time_bins)
        expected_total: Expected total photon count per measurement
        tolerance: Relative tolerance for total count validation
        
    Returns:
        Dictionary containing validation results and statistics
        
    Raises:
        ValidationError: If data fails basic validation
    """
    if not isinstance(photon_data, np.ndarray):
        raise ValidationError("Photon data must be a numpy array")
    
    if photon_data.ndim != 2:
        raise ValidationError("Photon data must be 2D array (iterations, time_bins)")
    
    if photon_data.size == 0:
        raise ValidationError("Photon data array is empty")
    
    results = {
        'valid': True,
        'warnings': [],
        'statistics': {}
    }
    
    try:
        # Check for non-negative integer values
        if np.any(photon_data < 0):
            results['valid'] = False
            results['warnings'].append("Data contains negative values")
        
        if not np.all(np.isfinite(photon_data)):
            results['valid'] = False
            results['warnings'].append("Data contains non-finite values")
        
        # Check total photon counts
        total_counts = np.sum(photon_data, axis=1)
        mean_total = np.mean(total_counts)
        std_total = np.std(total_counts)
        
        results['statistics']['mean_total_counts'] = mean_total
        results['statistics']['std_total_counts'] = std_total
        results['statistics']['expected_total'] = expected_total
        
        # Validate total count consistency
        rel_error = abs(mean_total - expected_total) / expected_total
        if rel_error > tolerance:
            results['warnings'].append(
                f"Mean total count ({mean_total:.1f}) differs from expected "
                f"({expected_total}) by {rel_error:.1%}"
            )
        
        # Check for reasonable Poisson statistics
        # For Poisson distribution, variance should approximately equal mean
        bin_means = np.mean(photon_data, axis=0)
        bin_vars = np.var(photon_data, axis=0)
        
        # Only check bins with reasonable counts
        valid_bins = bin_means > 5
        if np.any(valid_bins):
            poisson_ratios = bin_vars[valid_bins] / bin_means[valid_bins]
            mean_ratio = np.mean(poisson_ratios)
            
            results['statistics']['poisson_ratio'] = mean_ratio
            
            if abs(mean_ratio - 1.0) > 0.2:  # Allow 20% deviation
                results['warnings'].append(
                    f"Variance-to-mean ratio ({mean_ratio:.2f}) suggests "
                    "non-Poisson statistics"
                )
        
        # Check for zero counts (which can cause fitting problems)
        zero_fraction = np.mean(photon_data == 0)
        results['statistics']['zero_fraction'] = zero_fraction
        
        if zero_fraction > 0.5:
            results['warnings'].append(
                f"High fraction of zero counts ({zero_fraction:.1%}) "
                "may cause fitting difficulties"
            )
        
        # Summary statistics
        results['statistics']['min_count'] = np.min(photon_data)
        results['statistics']['max_count'] = np.max(photon_data)
        results['statistics']['mean_count'] = np.mean(photon_data)
        results['statistics']['std_count'] = np.std(photon_data)
        
        return results
        
    except Exception as e:
        results['valid'] = False
        results['warnings'].append(f"Validation failed: {str(e)}")
        return results

def fit_exponential_decay(time_bins: np.ndarray, photon_counts: np.ndarray,
                         tau_initial: Optional[float] = None, T: float = 25.0,
                         tn: int = None, irf_type: str = 'dirac',
                         irf_params: Optional[Dict] = None,
                         fit_irf: bool = False, max_iterations: int = 50000,
                         tolerance: float = 1e-10) -> Dict[str, Any]:
    """
    Fit exponential decay to photon count data with robust error handling.
    
    Performs curve fitting of exponential decay models to photon count data,
    supporting different IRF types and both known and unknown IRF parameter
    fitting scenarios. Includes comprehensive error handling and convergence
    checking.
    
    Args:
        time_bins: Time bin indices (typically 1 to tn)
        photon_counts: Observed photon counts in each time bin
        tau_initial: Initial guess for lifetime (ns). If None, estimated from data
        T: Repetition period (ns)
        tn: Number of time bins (inferred from data if None)
        irf_type: Type of IRF ('dirac', 'gaussian', 'rectangular')
        irf_params: IRF-specific parameters
        fit_irf: Whether to fit IRF parameters as well as lifetime
        max_iterations: Maximum number of fitting iterations
        tolerance: Convergence tolerance for fitting
        
    Returns:
        Dictionary containing fit results:
        - 'tau': Fitted lifetime (ns)
        - 'tau_error': Standard error of lifetime fit
        - 'irf_params': Fitted IRF parameters (if fit_irf=True)
        - 'success': Whether fit converged successfully
        - 'chi_squared': Chi-squared goodness of fit
        - 'residuals': Fit residuals
        - 'covariance': Parameter covariance matrix
        
    Raises:
        ValidationError: If input parameters are invalid
        ConvergenceError: If fitting fails to converge
    """
    # Validate inputs
    if not isinstance(time_bins, np.ndarray):
        time_bins = np.array(time_bins)
    
    if not isinstance(photon_counts, np.ndarray):
        photon_counts = np.array(photon_counts)
    
    if len(time_bins) != len(photon_counts):
        raise ValidationError("Time bins and photon counts must have same length")
    
    if len(time_bins) == 0:
        raise ValidationError("Input arrays cannot be empty")
    
    if np.any(photon_counts < 0):
        raise ValidationError("Photon counts cannot be negative")
    
    if T <= 0:
        raise ValidationError("Repetition period must be positive")
    
    if tn is None:
        tn = len(time_bins)
    
    if irf_params is None:
        irf_params = {}
    
    # Initialize result dictionary
    result = {
        'tau': np.nan,
        'tau_error': np.nan,
        'irf_params': irf_params.copy(),
        'success': False,
        'chi_squared': np.nan,
        'residuals': np.array([]),
        'covariance': np.array([]),
        'message': '',
        'iterations': 0
    }
    
    try:
        # Normalize photon counts to create probability distribution
        total_counts = np.sum(photon_counts)
        if total_counts == 0:
            raise ValidationError("Total photon count is zero")
        
        normalized_counts = photon_counts / total_counts
        
        # Estimate initial lifetime if not provided
        if tau_initial is None:
            # Use intensity-weighted mean time as initial guess
            t = time_bins * (T / tn)  # Convert to actual time
            tau_initial = np.sum(normalized_counts * t) / np.sum(normalized_counts)
            
            # Ensure reasonable bounds
            tau_initial = max(0.1, min(tau_initial, T/2))
        
        # Calculate weights for fitting (inverse variance for Poisson statistics)
        # Add 1 to avoid division by zero
        weights = 1.0 / np.sqrt(photon_counts + 1)
        
        # Define fitting function based on IRF type
        if irf_type.lower() == 'dirac':
            def fit_func(x, tau_fit):
                """Dirac IRF fitting function."""
                return expofunc(x, tau_fit, T, tn)
            
            # Perform fit
            try:
                popt, pcov = curve_fit(
                    fit_func, time_bins, normalized_counts,
                    p0=[tau_initial],
                    sigma=weights,
                    maxfev=max_iterations,
                    ftol=tolerance,
                    xtol=tolerance,
                    gtol=tolerance
                )
                
                result['tau'] = popt[0]
                result['tau_error'] = np.sqrt(pcov[0, 0]) if pcov.size > 0 else np.nan
                result['covariance'] = pcov
                result['success'] = True
                
            except RuntimeError as e:
                result['message'] = f"Dirac fit failed: {str(e)}"
                raise ConvergenceError(f"Dirac IRF fitting failed: {str(e)}")
        
        elif irf_type.lower() == 'gaussian':
            sigma = irf_params.get('sigma', 0.1)
            t0 = irf_params.get('t0', tn // 2)
            
            if fit_irf:
                # Fit both lifetime and IRF sigma
                def fit_func(x, tau_fit, sigma_fit):
                    """Gaussian IRF fitting function with unknown sigma."""
                    return calculate_pdf_gaussian(x, tau_fit, T, tn, sigma_fit, t0)
                
                initial_params = [tau_initial, sigma]
                
                try:
                    popt, pcov = curve_fit(
                        fit_func, time_bins, normalized_counts,
                        p0=initial_params,
                        sigma=weights,
                        maxfev=max_iterations,
                        ftol=tolerance,
                        xtol=tolerance,
                        gtol=tolerance
                    )
                    
                    result['tau'] = popt[0]
                    result['tau_error'] = np.sqrt(pcov[0, 0]) if pcov.size > 0 else np.nan
                    result['irf_params']['sigma'] = popt[1]
                    result['covariance'] = pcov
                    result['success'] = True
                    
                except RuntimeError as e:
                    result['message'] = f"Gaussian fit (unknown IRF) failed: {str(e)}"
                    raise ConvergenceError(f"Gaussian IRF fitting (unknown) failed: {str(e)}")
            
            else:
                # Fit only lifetime with known IRF
                def fit_func(x, tau_fit):
                    """Gaussian IRF fitting function with known sigma."""
                    return calculate_pdf_gaussian(x, tau_fit, T, tn, sigma, t0)
                
                try:
                    popt, pcov = curve_fit(
                        fit_func, time_bins, normalized_counts,
                        p0=[tau_initial],
                        sigma=weights,
                        maxfev=max_iterations,
                        ftol=tolerance,
                        xtol=tolerance,
                        gtol=tolerance
                    )
                    
                    result['tau'] = popt[0]
                    result['tau_error'] = np.sqrt(pcov[0, 0]) if pcov.size > 0 else np.nan
                    result['covariance'] = pcov
                    result['success'] = True
                    
                except RuntimeError as e:
                    result['message'] = f"Gaussian fit (known IRF) failed: {str(e)}"
                    raise ConvergenceError(f"Gaussian IRF fitting (known) failed: {str(e)}")
        
        elif irf_type.lower() == 'rectangular':
            width = irf_params.get('width', 0.2)
            t0 = irf_params.get('t0', tn // 2)
            
            # For rectangular IRF, typically only fit with known IRF parameters
            def fit_func(x, tau_fit):
                """Rectangular IRF fitting function."""
                return calculate_pdf_rectangular(x, tau_fit, T, tn, width, t0)
            
            try:
                popt, pcov = curve_fit(
                    fit_func, time_bins, normalized_counts,
                    p0=[tau_initial],
                    sigma=weights,
                    maxfev=max_iterations,
                    ftol=tolerance,
                    xtol=tolerance,
                    gtol=tolerance
                )
                
                result['tau'] = popt[0]
                result['tau_error'] = np.sqrt(pcov[0, 0]) if pcov.size > 0 else np.nan
                result['covariance'] = pcov
                result['success'] = True
                
            except RuntimeError as e:
                result['message'] = f"Rectangular fit failed: {str(e)}"
                raise ConvergenceError(f"Rectangular IRF fitting failed: {str(e)}")
        
        else:
            raise ValidationError(f"Unknown IRF type: {irf_type}")
        
        # Calculate goodness of fit metrics if successful
        if result['success']:
            # Calculate fitted curve
            if irf_type.lower() == 'dirac':
                fitted_curve = expofunc(time_bins, result['tau'], T, tn)
            elif irf_type.lower() == 'gaussian':
                fitted_curve = calculate_pdf_gaussian(
                    time_bins, result['tau'], T, tn, 
                    result['irf_params']['sigma'], t0
                )
            elif irf_type.lower() == 'rectangular':
                fitted_curve = calculate_pdf_rectangular(
                    time_bins, result['tau'], T, tn, width, t0
                )
            
            # Calculate residuals and chi-squared
            result['residuals'] = normalized_counts - fitted_curve
            
            # Chi-squared calculation (weighted by photon count uncertainties)
            chi_squared = np.sum((result['residuals'] / weights)**2)
            degrees_of_freedom = len(time_bins) - len(popt)
            result['chi_squared'] = chi_squared / degrees_of_freedom if degrees_of_freedom > 0 else chi_squared
        
        return result
        
    except Exception as e:
        if isinstance(e, (ValidationError, ConvergenceError)):
            raise
        else:
            result['message'] = f"Fitting failed: {str(e)}"
            return result


def validate_fit_results(fit_result: Dict[str, Any], tau_true: float,
                        tolerance: float = 0.1) -> Dict[str, Any]:
    """
    Validate curve fitting results for accuracy and convergence.
    
    Performs comprehensive validation of curve fitting results, checking
    for convergence, parameter accuracy, and goodness of fit metrics.
    
    Args:
        fit_result: Dictionary containing fit results from fit_exponential_decay
        tau_true: True lifetime value for accuracy assessment
        tolerance: Relative tolerance for accuracy validation
        
    Returns:
        Dictionary containing validation results and diagnostics
    """
    validation = {
        'valid': True,
        'warnings': [],
        'accuracy_metrics': {},
        'convergence_metrics': {}
    }
    
    try:
        # Check if fit was successful
        if not fit_result.get('success', False):
            validation['valid'] = False
            validation['warnings'].append("Fit did not converge successfully")
            return validation
        
        # Check parameter validity
        tau_fitted = fit_result.get('tau', np.nan)
        if not np.isfinite(tau_fitted) or tau_fitted <= 0:
            validation['valid'] = False
            validation['warnings'].append("Fitted lifetime is invalid")
            return validation
        
        # Calculate accuracy metrics
        rel_error = abs(tau_fitted - tau_true) / tau_true
        validation['accuracy_metrics']['relative_error'] = rel_error
        validation['accuracy_metrics']['absolute_error'] = abs(tau_fitted - tau_true)
        validation['accuracy_metrics']['tau_fitted'] = tau_fitted
        validation['accuracy_metrics']['tau_true'] = tau_true
        
        if rel_error > tolerance:
            validation['warnings'].append(
                f"Fitted lifetime ({tau_fitted:.3f}) differs from true value "
                f"({tau_true:.3f}) by {rel_error:.1%}"
            )
        
        # Check parameter uncertainty
        tau_error = fit_result.get('tau_error', np.nan)
        if np.isfinite(tau_error) and tau_error > 0:
            validation['accuracy_metrics']['tau_error'] = tau_error
            validation['accuracy_metrics']['relative_uncertainty'] = tau_error / tau_fitted
            
            # Check if error estimate is reasonable
            if tau_error / tau_fitted > 0.5:  # 50% uncertainty seems excessive
                validation['warnings'].append(
                    f"Large parameter uncertainty ({tau_error/tau_fitted:.1%})"
                )
        else:
            validation['warnings'].append("Parameter uncertainty not available or invalid")
        
        # Check goodness of fit
        chi_squared = fit_result.get('chi_squared', np.nan)
        if np.isfinite(chi_squared):
            validation['convergence_metrics']['chi_squared'] = chi_squared
            
            # Chi-squared should be close to 1 for good fits
            if chi_squared > 3:
                validation['warnings'].append(
                    f"Poor goodness of fit (χ² = {chi_squared:.2f})"
                )
            elif chi_squared < 0.3:
                validation['warnings'].append(
                    f"Suspiciously good fit (χ² = {chi_squared:.2f}) - possible overfitting"
                )
        
        # Check residuals
        residuals = fit_result.get('residuals', np.array([]))
        if len(residuals) > 0:
            residual_std = np.std(residuals)
            residual_mean = np.mean(residuals)
            
            validation['convergence_metrics']['residual_std'] = residual_std
            validation['convergence_metrics']['residual_mean'] = residual_mean
            
            # Mean should be close to zero for unbiased fits
            if abs(residual_mean) > 3 * residual_std / np.sqrt(len(residuals)):
                validation['warnings'].append("Systematic bias detected in residuals")
        
        return validation
        
    except Exception as e:
        validation['valid'] = False
        validation['warnings'].append(f"Validation failed: {str(e)}")
        return validation


def batch_fit_data(photon_data: np.ndarray, time_bins: np.ndarray,
                  tau_true: float, T: float, tn: int,
                  irf_type: str = 'dirac', irf_params: Optional[Dict] = None,
                  fit_irf: bool = False, progress: bool = True,
                  parallel: bool = False, n_jobs: Optional[int] = None) -> Dict[str, Any]:
    """
    Perform batch curve fitting on multiple measurements.
    
    Fits exponential decay models to multiple independent measurements,
    collecting statistics on fitted parameters and convergence rates.
    Supports parallel processing for improved performance.
    
    Args:
        photon_data: Array of shape (iterations, time_bins) with photon counts
        time_bins: Time bin indices
        tau_true: True lifetime value for validation
        T: Repetition period (ns)
        tn: Number of time bins
        irf_type: Type of IRF ('dirac', 'gaussian', 'rectangular')
        irf_params: IRF-specific parameters
        fit_irf: Whether to fit IRF parameters as well as lifetime
        progress: Whether to show progress bar
        parallel: Whether to use parallel processing
        n_jobs: Number of parallel jobs (None for auto-detection)
        
    Returns:
        Dictionary containing batch fitting results and statistics
        
    Raises:
        ValidationError: If input parameters are invalid
        ComputationError: If batch fitting fails
    """
    # Validate inputs
    if not isinstance(photon_data, np.ndarray):
        raise ValidationError("Photon data must be a numpy array")
    
    if photon_data.ndim != 2:
        raise ValidationError("Photon data must be 2D array (iterations, time_bins)")
    
    if not isinstance(time_bins, np.ndarray):
        time_bins = np.array(time_bins)
    
    if photon_data.shape[1] != len(time_bins):
        raise ValidationError("Photon data and time bins must have consistent dimensions")
    
    if tau_true <= 0:
        raise ValidationError("True lifetime must be positive")
    
    if irf_params is None:
        irf_params = {}
    
    iterations = photon_data.shape[0]
    
    # Initialize results storage
    results = {
        'tau_fits': np.full(iterations, np.nan),
        'tau_errors': np.full(iterations, np.nan),
        'chi_squared': np.full(iterations, np.nan),
        'success_flags': np.zeros(iterations, dtype=bool),
        'irf_params_fits': [],
        'convergence_rate': 0.0,
        'statistics': {},
        'validation_summary': {}
    }
    
    try:
        def fit_single_measurement(i):
            """Fit a single measurement - for parallel processing."""
            try:
                fit_result = fit_exponential_decay(
                    time_bins, photon_data[i, :],
                    tau_initial=None, T=T, tn=tn,
                    irf_type=irf_type, irf_params=irf_params,
                    fit_irf=fit_irf
                )
                
                return i, fit_result
                
            except Exception as e:
                # Return failed result
                failed_result = {
                    'tau': np.nan,
                    'tau_error': np.nan,
                    'chi_squared': np.nan,
                    'success': False,
                    'message': str(e),
                    'irf_params': irf_params.copy()
                }
                return i, failed_result
        
        # Perform fitting (parallel or sequential)
        if parallel and iterations > 10:  # Only use parallel for larger datasets
            if n_jobs is None:
                n_jobs = min(mp.cpu_count(), iterations)
            
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                if progress and HAS_TQDM:
                    futures = {executor.submit(fit_single_measurement, i): i 
                              for i in range(iterations)}
                    
                    for future in tqdm(as_completed(futures), total=iterations,
                                     desc="Batch fitting"):
                        i, fit_result = future.result()
                        _store_fit_result(results, i, fit_result)
                else:
                    futures = {executor.submit(fit_single_measurement, i): i 
                              for i in range(iterations)}
                    
                    for future in as_completed(futures):
                        i, fit_result = future.result()
                        _store_fit_result(results, i, fit_result)
        
        else:
            # Sequential processing
            if progress and HAS_TQDM:
                iterator = tqdm(range(iterations), desc="Batch fitting")
            else:
                iterator = range(iterations)
            
            for i in iterator:
                _, fit_result = fit_single_measurement(i)
                _store_fit_result(results, i, fit_result)
        
        # Calculate statistics
        successful_fits = results['success_flags']
        results['convergence_rate'] = np.mean(successful_fits)
        
        if np.any(successful_fits):
            valid_taus = results['tau_fits'][successful_fits]
            valid_errors = results['tau_errors'][successful_fits]
            valid_chi2 = results['chi_squared'][successful_fits]
            
            results['statistics'] = {
                'tau_mean': np.mean(valid_taus),
                'tau_std': np.std(valid_taus),
                'tau_median': np.median(valid_taus),
                'tau_error_mean': np.mean(valid_errors[np.isfinite(valid_errors)]),
                'chi_squared_mean': np.mean(valid_chi2[np.isfinite(valid_chi2)]),
                'relative_error_mean': np.mean(np.abs(valid_taus - tau_true) / tau_true),
                'relative_error_std': np.std(np.abs(valid_taus - tau_true) / tau_true)
            }
            
            # Calculate F-value (matching Monte Carlo notebook definition)
            # F = tau_sd * sqrt(intens_avg) / tau_avg
            intens_avg = np.mean(np.sum(photon_data[successful_fits, :], axis=1))
            results['statistics']['F_value'] = (
                results['statistics']['tau_std'] * 
                np.sqrt(intens_avg) / 
                results['statistics']['tau_mean']
            )
        
        # Validation summary
        if np.any(successful_fits):
            validation_results = []
            for i in range(iterations):
                if successful_fits[i]:
                    fit_result = {
                        'tau': results['tau_fits'][i],
                        'tau_error': results['tau_errors'][i],
                        'chi_squared': results['chi_squared'][i],
                        'success': True
                    }
                    validation = validate_fit_results(fit_result, tau_true)
                    validation_results.append(validation)
            
            if validation_results:
                results['validation_summary'] = {
                    'valid_fraction': np.mean([v['valid'] for v in validation_results]),
                    'common_warnings': _summarize_warnings([v['warnings'] for v in validation_results])
                }
        
        return results
        
    except Exception as e:
        if isinstance(e, (ValidationError, ComputationError)):
            raise
        else:
            raise ComputationError(f"Batch fitting failed: {str(e)}")


def _store_fit_result(results: Dict, i: int, fit_result: Dict) -> None:
    """Helper function to store individual fit results."""
    results['tau_fits'][i] = fit_result.get('tau', np.nan)
    results['tau_errors'][i] = fit_result.get('tau_error', np.nan)
    results['chi_squared'][i] = fit_result.get('chi_squared', np.nan)
    results['success_flags'][i] = fit_result.get('success', False)
    results['irf_params_fits'].append(fit_result.get('irf_params', {}))


def _summarize_warnings(warning_lists: list) -> Dict[str, int]:
    """Helper function to summarize common warnings."""
    warning_counts = {}
    for warnings in warning_lists:
        for warning in warnings:
            warning_counts[warning] = warning_counts.get(warning, 0) + 1
    
    # Return only warnings that occurred in >10% of cases
    total_cases = len(warning_lists)
    return {warning: count for warning, count in warning_counts.items() 
            if count > 0.1 * total_cases}


def estimate_initial_lifetime(time_bins: np.ndarray, photon_counts: np.ndarray,
                             T: float, tn: int) -> float:
    """
    Estimate initial lifetime from photon count data.
    
    Provides a reasonable initial guess for lifetime fitting based on
    the intensity-weighted mean arrival time of photons.
    
    Args:
        time_bins: Time bin indices
        photon_counts: Photon counts in each bin
        T: Repetition period (ns)
        tn: Number of time bins
        
    Returns:
        Estimated lifetime (ns)
        
    Raises:
        ValidationError: If input data is invalid
    """
    if not isinstance(time_bins, np.ndarray):
        time_bins = np.array(time_bins)
    
    if not isinstance(photon_counts, np.ndarray):
        photon_counts = np.array(photon_counts)
    
    if len(time_bins) != len(photon_counts):
        raise ValidationError("Time bins and photon counts must have same length")
    
    total_counts = np.sum(photon_counts)
    if total_counts == 0:
        raise ValidationError("Total photon count is zero")
    
    # Convert bin indices to actual time
    t = time_bins * (T / tn)
    
    # Calculate intensity-weighted mean time
    mean_time = np.sum(photon_counts * t) / total_counts
    
    # Use mean time as lifetime estimate, with reasonable bounds
    tau_estimate = max(0.1, min(mean_time, T/2))
    
    return tau_estimate


def monte_carlo_analysis(tau_range: np.ndarray, time_bins_range: np.ndarray,
                        irf_type: str = 'dirac', irf_params_range: Optional[Dict] = None,
                        T: float = 25.0, num_photons: int = 75000,
                        iterations: int = 5000, progress: bool = True,
                        parallel: bool = True, n_jobs: Optional[int] = None,
                        save_intermediate: bool = True,
                        output_dir: Optional[str] = None,
                        random_seed: Optional[int] = None) -> AnalysisResults:
    """
    Comprehensive Monte Carlo analysis pipeline for full parameter sweep.
    
    Performs a complete Monte Carlo analysis across specified parameter ranges,
    including photon data generation, curve fitting, and statistical analysis.
    Supports parallel processing for computational efficiency and intermediate
    result saving for long-running analyses.
    
    Args:
        tau_range: Array of fluorescence lifetimes to analyze (ns)
        time_bins_range: Array of time bin counts to analyze
        irf_type: Type of IRF ('dirac', 'gaussian', 'rectangular')
        irf_params_range: Dictionary of IRF parameter ranges (e.g., {'sigma': [0.1, 0.5, 1.0]})
        T: Repetition period (ns)
        num_photons: Number of photons per measurement
        iterations: Number of Monte Carlo iterations per parameter combination
        progress: Whether to show progress bars
        parallel: Whether to use parallel processing
        n_jobs: Number of parallel jobs (None for auto-detection)
        save_intermediate: Whether to save intermediate results
        output_dir: Directory for intermediate result files
        random_seed: Random seed for reproducible results
        
    Returns:
        AnalysisResults object containing F-values and comprehensive statistics
        
    Raises:
        ValidationError: If input parameters are invalid
        ComputationError: If analysis fails
    """
    # Validate inputs
    if not isinstance(tau_range, np.ndarray):
        tau_range = np.array(tau_range)
    
    if not isinstance(time_bins_range, np.ndarray):
        time_bins_range = np.array(time_bins_range)
    
    if len(tau_range) == 0 or len(time_bins_range) == 0:
        raise ValidationError("Parameter ranges cannot be empty")
    
    if np.any(tau_range <= 0):
        raise ValidationError("All lifetimes must be positive")
    
    if np.any(time_bins_range <= 0):
        raise ValidationError("All time bin counts must be positive")
    
    if T <= 0:
        raise ValidationError("Repetition period must be positive")
    
    if num_photons <= 0:
        raise ValidationError("Number of photons must be positive")
    
    if iterations <= 0:
        raise ValidationError("Number of iterations must be positive")
    
    # Handle IRF parameter ranges
    if irf_params_range is None:
        if irf_type.lower() == 'gaussian':
            irf_params_range = {'sigma': [0.1]}
        elif irf_type.lower() == 'rectangular':
            irf_params_range = {'width': [0.2]}
        else:
            irf_params_range = {}
    
    # Create parameter combinations
    param_combinations = _create_parameter_combinations(
        tau_range, time_bins_range, irf_type, irf_params_range
    )
    
    total_combinations = len(param_combinations)
    
    try:
        # Initialize results storage
        if irf_type.lower() == 'dirac':
            # 2D array: tau x time_bins
            F_values = np.full((len(tau_range), len(time_bins_range)), np.nan)
            results_shape = (len(tau_range), len(time_bins_range))
        else:
            # 3D array: tau x time_bins x irf_param
            irf_param_values = list(irf_params_range.values())[0]  # Assume single parameter for now
            F_values = np.full((len(tau_range), len(time_bins_range), len(irf_param_values)), np.nan)
            results_shape = (len(tau_range), len(time_bins_range), len(irf_param_values))
        
        # Storage for detailed statistics
        detailed_stats = {
            'convergence_rates': np.full(results_shape, np.nan),
            'tau_means': np.full(results_shape, np.nan),
            'tau_stds': np.full(results_shape, np.nan),
            'relative_errors': np.full(results_shape, np.nan),
            'chi_squared_means': np.full(results_shape, np.nan)
        }
        
        # Set up progress tracking
        if progress and HAS_TQDM:
            pbar = tqdm(total=total_combinations, desc="Monte Carlo Analysis")
        else:
            pbar = None
        
        # Process each parameter combination
        for combo_idx, (tau, tn, irf_params, result_indices) in enumerate(param_combinations):
            try:
                if progress and not HAS_TQDM:
                    print(f"Processing combination {combo_idx + 1}/{total_combinations}: "
                          f"tau={tau:.2f}, tn={int(tn)}, irf_params={irf_params}")
                
                # Generate measurement data
                photon_data, pdf = generate_measurement_data(
                    tau=tau, T=T, tn=int(tn), num_photons=num_photons,
                    iterations=iterations, irf_type=irf_type,
                    irf_params=irf_params, random_seed=random_seed
                )
                
                # Create time bins
                time_bins = np.arange(1, int(tn) + 1)
                
                # Perform batch fitting
                batch_results = batch_fit_data(
                    photon_data=photon_data, time_bins=time_bins,
                    tau_true=tau, T=T, tn=int(tn),
                    irf_type=irf_type, irf_params=irf_params,
                    fit_irf=False, progress=False,  # Disable inner progress bar
                    parallel=parallel, n_jobs=n_jobs
                )
                
                # Store results
                if 'F_value' in batch_results['statistics']:
                    F_values[result_indices] = batch_results['statistics']['F_value']
                
                # Store detailed statistics
                detailed_stats['convergence_rates'][result_indices] = batch_results['convergence_rate']
                
                if batch_results['statistics']:
                    detailed_stats['tau_means'][result_indices] = batch_results['statistics'].get('tau_mean', np.nan)
                    detailed_stats['tau_stds'][result_indices] = batch_results['statistics'].get('tau_std', np.nan)
                    detailed_stats['relative_errors'][result_indices] = batch_results['statistics'].get('relative_error_mean', np.nan)
                    detailed_stats['chi_squared_means'][result_indices] = batch_results['statistics'].get('chi_squared_mean', np.nan)
                
                # Save intermediate results if requested
                if save_intermediate and output_dir is not None:
                    _save_intermediate_results(
                        output_dir, combo_idx, tau, tn, irf_params,
                        batch_results, F_values, detailed_stats
                    )
                
            except Exception as e:
                warnings.warn(f"Failed analysis for tau={tau:.2f}, tn={int(tn)}, "
                            f"irf_params={irf_params}: {str(e)}")
                # Results remain NaN for this combination
            
            if pbar is not None:
                pbar.update(1)
        
        if pbar is not None:
            pbar.close()
        
        # Create analysis parameters
        if irf_type.lower() == 'dirac':
            irf_sigmas = np.array([0.0])  # Placeholder for dirac
        else:
            irf_sigmas = np.array(list(irf_params_range.values())[0])
        
        params = AnalysisParameters(
            repetition_period=T,
            lifetimes=tau_range,
            time_bins=time_bins_range,
            irf_sigmas=irf_sigmas,
            num_photons=num_photons,
            iterations=iterations
        )
        
        # Create comprehensive metadata
        metadata = {
            'irf_type': irf_type,
            'irf_params_range': irf_params_range,
            'analysis_date': np.datetime64('now').astype(str),
            'total_combinations': total_combinations,
            'successful_combinations': np.sum(np.isfinite(F_values)),
            'failed_combinations': np.sum(~np.isfinite(F_values)),
            'parallel_processing': parallel,
            'random_seed': random_seed,
            'detailed_statistics': detailed_stats,
            'convergence_summary': {
                'mean_convergence_rate': np.nanmean(detailed_stats['convergence_rates']),
                'min_convergence_rate': np.nanmin(detailed_stats['convergence_rates']),
                'combinations_with_poor_convergence': np.sum(detailed_stats['convergence_rates'] < 0.8)
            }
        }
        
        # Create results object
        results = AnalysisResults(
            f_values=F_values,
            parameters=params,
            metadata=metadata,
            analysis_type=f'{irf_type}_monte_carlo_analysis'
        )
        
        return results
        
    except Exception as e:
        if isinstance(e, (ValidationError, ComputationError)):
            raise
        else:
            raise ComputationError(f"Monte Carlo analysis failed: {str(e)}")


def _create_parameter_combinations(tau_range: np.ndarray, time_bins_range: np.ndarray,
                                 irf_type: str, irf_params_range: Dict) -> list:
    """
    Create all parameter combinations for Monte Carlo analysis.
    
    Args:
        tau_range: Array of lifetimes
        time_bins_range: Array of time bin counts
        irf_type: IRF type
        irf_params_range: Dictionary of IRF parameter ranges
        
    Returns:
        List of tuples (tau, tn, irf_params, result_indices)
    """
    combinations = []
    
    if irf_type.lower() == 'dirac':
        # Simple 2D case
        for i, tau in enumerate(tau_range):
            for j, tn in enumerate(time_bins_range):
                combinations.append((tau, tn, {}, (i, j)))
    
    else:
        # 3D case with IRF parameters
        irf_param_name = list(irf_params_range.keys())[0]
        irf_param_values = irf_params_range[irf_param_name]
        
        for i, tau in enumerate(tau_range):
            for j, tn in enumerate(time_bins_range):
                for k, param_value in enumerate(irf_param_values):
                    irf_params = {irf_param_name: param_value}
                    combinations.append((tau, tn, irf_params, (i, j, k)))
    
    return combinations


def _save_intermediate_results(output_dir: str, combo_idx: int, tau: float,
                             tn: int, irf_params: Dict, batch_results: Dict,
                             F_values: np.ndarray, detailed_stats: Dict) -> None:
    """Save intermediate results during long-running analysis."""
    import os
    from pathlib import Path
    
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save current F_values
        np.save(output_path / f"F_values_intermediate_{combo_idx:04d}.npy", F_values)
        
        # Save detailed stats
        np.savez(
            output_path / f"detailed_stats_{combo_idx:04d}.npz",
            **detailed_stats
        )
        
        # Save batch results for this combination
        import pickle
        with open(output_path / f"batch_results_{combo_idx:04d}.pkl", 'wb') as f:
            pickle.dump({
                'tau': tau,
                'tn': tn,
                'irf_params': irf_params,
                'batch_results': batch_results
            }, f)
        
    except Exception as e:
        warnings.warn(f"Failed to save intermediate results: {str(e)}")


def dirac_monte_carlo_analysis(tau_range: np.ndarray, time_bins_range: np.ndarray,
                              T: float = 25.0, num_photons: int = 75000,
                              iterations: int = 5000, progress: bool = True,
                              parallel: bool = True, random_seed: Optional[int] = None) -> AnalysisResults:
    """
    Monte Carlo analysis for Dirac IRF case.
    
    Specialized function for Dirac IRF Monte Carlo analysis, matching the
    implementation from the existing notebook. This reproduces the F_dirac
    results from the original code.
    
    Args:
        tau_range: Array of fluorescence lifetimes to analyze (ns)
        time_bins_range: Array of time bin counts to analyze
        T: Repetition period (ns)
        num_photons: Number of photons per measurement
        iterations: Number of Monte Carlo iterations
        progress: Whether to show progress bars
        parallel: Whether to use parallel processing
        random_seed: Random seed for reproducible results
        
    Returns:
        AnalysisResults object containing F-values and statistics
    """
    return monte_carlo_analysis(
        tau_range=tau_range,
        time_bins_range=time_bins_range,
        irf_type='dirac',
        irf_params_range={},
        T=T,
        num_photons=num_photons,
        iterations=iterations,
        progress=progress,
        parallel=parallel,
        random_seed=random_seed
    )


def gaussian_monte_carlo_analysis(tau_range: np.ndarray, time_bins_range: np.ndarray,
                                 sigma_range: np.ndarray, T: float = 25.0,
                                 num_photons: int = 75000, iterations: int = 5000,
                                 fit_irf: bool = False, progress: bool = True,
                                 parallel: bool = True, random_seed: Optional[int] = None) -> Dict[str, AnalysisResults]:
    """
    Monte Carlo analysis for Gaussian IRF case.
    
    Specialized function for Gaussian IRF Monte Carlo analysis, supporting
    both known and unknown IRF parameter fitting scenarios. This reproduces
    the F_known and F_unknown results from the original notebook.
    
    Args:
        tau_range: Array of fluorescence lifetimes to analyze (ns)
        time_bins_range: Array of time bin counts to analyze
        sigma_range: Array of Gaussian IRF widths to analyze (ns)
        T: Repetition period (ns)
        num_photons: Number of photons per measurement
        iterations: Number of Monte Carlo iterations
        fit_irf: Whether to fit IRF parameters (unknown) or use known values
        progress: Whether to show progress bars
        parallel: Whether to use parallel processing
        random_seed: Random seed for reproducible results
        
    Returns:
        Dictionary containing AnalysisResults for known and/or unknown IRF cases
    """
    results = {}
    
    if fit_irf:
        # Unknown IRF case - fit both lifetime and IRF width
        results['unknown'] = monte_carlo_analysis(
            tau_range=tau_range,
            time_bins_range=time_bins_range,
            irf_type='gaussian',
            irf_params_range={'sigma': sigma_range.tolist()},
            T=T,
            num_photons=num_photons,
            iterations=iterations,
            progress=progress,
            parallel=parallel,
            random_seed=random_seed
        )
    else:
        # Known IRF case - fit only lifetime with known IRF width
        results['known'] = monte_carlo_analysis(
            tau_range=tau_range,
            time_bins_range=time_bins_range,
            irf_type='gaussian',
            irf_params_range={'sigma': sigma_range.tolist()},
            T=T,
            num_photons=num_photons,
            iterations=iterations,
            progress=progress,
            parallel=parallel,
            random_seed=random_seed
        )
    
    return results


def rectangular_monte_carlo_analysis(tau_range: np.ndarray, time_bins_range: np.ndarray,
                                   width_range: np.ndarray, T: float = 25.0,
                                   num_photons: int = 75000, iterations: int = 5000,
                                   progress: bool = True, parallel: bool = True,
                                   random_seed: Optional[int] = None) -> AnalysisResults:
    """
    Monte Carlo analysis for rectangular IRF case.
    
    Specialized function for rectangular (square wave) IRF Monte Carlo analysis.
    This reproduces the F_knownsq results from the original notebook.
    
    Args:
        tau_range: Array of fluorescence lifetimes to analyze (ns)
        time_bins_range: Array of time bin counts to analyze
        width_range: Array of rectangular IRF widths to analyze (ns)
        T: Repetition period (ns)
        num_photons: Number of photons per measurement
        iterations: Number of Monte Carlo iterations
        progress: Whether to show progress bars
        parallel: Whether to use parallel processing
        random_seed: Random seed for reproducible results
        
    Returns:
        AnalysisResults object containing F-values and statistics
    """
    return monte_carlo_analysis(
        tau_range=tau_range,
        time_bins_range=time_bins_range,
        irf_type='rectangular',
        irf_params_range={'width': width_range.tolist()},
        T=T,
        num_photons=num_photons,
        iterations=iterations,
        progress=progress,
        parallel=parallel,
        random_seed=random_seed
    )


def compare_monte_carlo_fisher(monte_carlo_results: AnalysisResults,
                              fisher_results: AnalysisResults,
                              tolerance: float = 0.2) -> Dict[str, Any]:
    """
    Compare Monte Carlo results with Fisher information predictions.
    
    Validates Monte Carlo simulation results against theoretical Fisher
    information predictions to verify the accuracy of both approaches.
    
    Args:
        monte_carlo_results: Results from Monte Carlo analysis
        fisher_results: Results from Fisher information analysis
        tolerance: Relative tolerance for comparison (default 20%)
        
    Returns:
        Dictionary containing comparison results and statistics
        
    Raises:
        ValidationError: If results are incompatible for comparison
    """
    # Validate inputs
    if not isinstance(monte_carlo_results, AnalysisResults):
        raise ValidationError("Monte Carlo results must be AnalysisResults object")
    
    if not isinstance(fisher_results, AnalysisResults):
        raise ValidationError("Fisher results must be AnalysisResults object")
    
    # Check compatibility
    mc_shape = monte_carlo_results.f_values.shape
    fisher_shape = fisher_results.f_values.shape
    
    if mc_shape != fisher_shape:
        raise ValidationError(f"Result shapes incompatible: MC {mc_shape} vs Fisher {fisher_shape}")
    
    try:
        # Extract F-values
        mc_f_values = monte_carlo_results.f_values
        fisher_f_values = fisher_results.f_values
        
        # Find valid (finite) values in both datasets
        valid_mask = np.isfinite(mc_f_values) & np.isfinite(fisher_f_values)
        
        if not np.any(valid_mask):
            raise ValidationError("No valid data points for comparison")
        
        mc_valid = mc_f_values[valid_mask]
        fisher_valid = fisher_f_values[valid_mask]
        
        # Calculate comparison metrics
        relative_errors = np.abs(mc_valid - fisher_valid) / fisher_valid
        absolute_errors = np.abs(mc_valid - fisher_valid)
        
        # Statistical measures
        correlation = np.corrcoef(mc_valid, fisher_valid)[0, 1]
        
        # Agreement statistics
        within_tolerance = relative_errors <= tolerance
        agreement_fraction = np.mean(within_tolerance)
        
        # Create comparison results
        comparison = {
            'agreement_fraction': agreement_fraction,
            'correlation': correlation,
            'relative_error_mean': np.mean(relative_errors),
            'relative_error_std': np.std(relative_errors),
            'relative_error_median': np.median(relative_errors),
            'absolute_error_mean': np.mean(absolute_errors),
            'max_relative_error': np.max(relative_errors),
            'min_relative_error': np.min(relative_errors),
            'valid_points': np.sum(valid_mask),
            'total_points': valid_mask.size,
            'tolerance': tolerance,
            'summary': {}
        }
        
        # Generate summary
        if agreement_fraction > 0.9:
            comparison['summary']['overall'] = "Excellent agreement"
        elif agreement_fraction > 0.7:
            comparison['summary']['overall'] = "Good agreement"
        elif agreement_fraction > 0.5:
            comparison['summary']['overall'] = "Moderate agreement"
        else:
            comparison['summary']['overall'] = "Poor agreement"
        
        if correlation > 0.95:
            comparison['summary']['correlation'] = "Very strong correlation"
        elif correlation > 0.8:
            comparison['summary']['correlation'] = "Strong correlation"
        elif correlation > 0.6:
            comparison['summary']['correlation'] = "Moderate correlation"
        else:
            comparison['summary']['correlation'] = "Weak correlation"
        
        # Identify problematic parameter regions
        if mc_shape == (len(monte_carlo_results.parameters.lifetimes),
                       len(monte_carlo_results.parameters.time_bins)):
            # 2D case (Dirac IRF)
            error_map = np.full(mc_shape, np.nan)
            error_map[valid_mask] = relative_errors
            
            # Find worst-performing parameter combinations
            worst_indices = np.unravel_index(np.nanargmax(error_map), mc_shape)
            comparison['worst_case'] = {
                'tau': monte_carlo_results.parameters.lifetimes[worst_indices[0]],
                'time_bins': monte_carlo_results.parameters.time_bins[worst_indices[1]],
                'relative_error': error_map[worst_indices]
            }
        
        return comparison
        
    except Exception as e:
        if isinstance(e, ValidationError):
            raise
        else:
            raise ComputationError(f"Comparison failed: {str(e)}")


def save_monte_carlo_results(results: AnalysisResults, output_dir: str,
                           file_prefix: str = "monte_carlo") -> Dict[str, str]:
    """
    Save Monte Carlo results in multiple formats.
    
    Saves results in formats compatible with the original notebook outputs,
    including .npy files for F-values and comprehensive .npz archives.
    
    Args:
        results: AnalysisResults object to save
        output_dir: Output directory path
        file_prefix: Prefix for output files
        
    Returns:
        Dictionary mapping format names to saved file paths
        
    Raises:
        ComputationError: If saving fails
    """
    from pathlib import Path
    
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # Save F-values in .npy format (compatible with original notebook)
        f_values_file = output_path / f"{file_prefix}_F_values.npy"
        np.save(f_values_file, results.f_values)
        saved_files['f_values_npy'] = str(f_values_file)
        
        # Save complete results in .npz format
        results_file = output_path / f"{file_prefix}_results.npz"
        results.save(str(results_file))
        saved_files['complete_npz'] = str(results_file)
        
        # Save metadata as JSON
        import json
        metadata_file = output_path / f"{file_prefix}_metadata.json"
        
        # Convert numpy types to Python types for JSON serialization
        metadata_serializable = {}
        for key, value in results.metadata.items():
            if isinstance(value, np.integer):
                metadata_serializable[key] = int(value)
            elif isinstance(value, np.floating):
                metadata_serializable[key] = float(value)
            elif isinstance(value, np.ndarray):
                metadata_serializable[key] = value.tolist()
            elif isinstance(value, dict):
                # Handle nested dictionaries
                metadata_serializable[key] = _convert_dict_for_json(value)
            else:
                metadata_serializable[key] = value
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata_serializable, f, indent=2)
        saved_files['metadata_json'] = str(metadata_file)
        
        return saved_files
        
    except Exception as e:
        raise ComputationError(f"Failed to save Monte Carlo results: {str(e)}")


def _convert_dict_for_json(d: Dict) -> Dict:
    """Recursively convert dictionary values for JSON serialization."""
    result = {}
    for key, value in d.items():
        if isinstance(value, np.integer):
            result[key] = int(value)
        elif isinstance(value, np.floating):
            result[key] = float(value)
        elif isinstance(value, np.ndarray):
            result[key] = value.tolist()
        elif isinstance(value, dict):
            result[key] = _convert_dict_for_json(value)
        else:
            result[key] = value
    return result


def load_monte_carlo_results(file_path: str) -> AnalysisResults:
    """
    Load Monte Carlo results from saved files.
    
    Args:
        file_path: Path to saved results file (.npz or .pkl)
        
    Returns:
        AnalysisResults object
        
    Raises:
        ComputationError: If loading fails
    """
    try:
        return AnalysisResults.load(file_path)
    except Exception as e:
        raise ComputationError(f"Failed to load Monte Carlo results: {str(e)}")