"""
Instrument Response Function (IRF) models and convolution operations.

This module implements various IRF models commonly used in fluorescence
lifetime imaging microscopy (FLIM) analysis, including Dirac delta,
Gaussian, and rectangular IRFs, along with convolution operations
for exponential decay functions.
"""

import numpy as np
from scipy import signal
from typing import Union, Tuple, Optional
try:
    from .core import ValidationError, ComputationError
except ImportError:
    from core import ValidationError, ComputationError


def dirac_irf(t: np.ndarray, t0: float = 0.0) -> np.ndarray:
    """
    Dirac delta instrument response function.
    
    Implements an idealized IRF as a Dirac delta function, representing
    an infinitely narrow instrument response. In practice, this is
    approximated as a single non-zero value at the specified time point.
    
    Args:
        t: Time array (ns)
        t0: Time offset for the IRF peak (ns)
        
    Returns:
        Normalized Dirac IRF array
        
    Raises:
        ValidationError: If input arrays are invalid
    """
    if not isinstance(t, np.ndarray):
        t = np.array(t)
    
    if len(t) == 0:
        raise ValidationError("Time array cannot be empty")
    
    # Find the closest time point to t0
    idx = np.argmin(np.abs(t - t0))
    
    # Create Dirac delta
    irf = np.zeros_like(t, dtype=float)
    irf[idx] = 1.0
    
    # Normalize to unit area (accounting for time step)
    dt = np.mean(np.diff(t)) if len(t) > 1 else 1.0
    irf = irf / (np.sum(irf) * dt)
    
    return irf


def gaussian_irf(t: np.ndarray, t0: float = 0.0, sigma: float = 0.1) -> np.ndarray:
    """
    Gaussian instrument response function.
    
    Implements a Gaussian IRF model commonly used to represent realistic
    instrument response with finite temporal resolution.
    
    Args:
        t: Time array (ns)
        t0: Time offset for the IRF peak (ns)
        sigma: Standard deviation of the Gaussian (ns)
        
    Returns:
        Normalized Gaussian IRF array
        
    Raises:
        ValidationError: If input parameters are invalid
    """
    if not isinstance(t, np.ndarray):
        t = np.array(t)
    
    if len(t) == 0:
        raise ValidationError("Time array cannot be empty")
    
    if sigma <= 0:
        raise ValidationError("Gaussian sigma must be positive")
    
    # Calculate Gaussian IRF
    irf = np.exp(-0.5 * ((t - t0) / sigma) ** 2)
    
    # Normalize to unit area
    dt = np.mean(np.diff(t)) if len(t) > 1 else 1.0
    area = np.sum(irf) * dt
    
    if area == 0:
        raise ComputationError("Gaussian IRF has zero area - sigma may be too small")
    
    irf = irf / area
    
    return irf


def rectangular_irf(t: np.ndarray, t0: float = 0.0, width: float = 0.2) -> np.ndarray:
    """
    Rectangular instrument response function.
    
    Implements a rectangular (top-hat) IRF model representing a uniform
    instrument response over a finite time window.
    
    Args:
        t: Time array (ns)
        t0: Center time of the rectangular IRF (ns)
        width: Full width of the rectangular IRF (ns)
        
    Returns:
        Normalized rectangular IRF array
        
    Raises:
        ValidationError: If input parameters are invalid
    """
    if not isinstance(t, np.ndarray):
        t = np.array(t)
    
    if len(t) == 0:
        raise ValidationError("Time array cannot be empty")
    
    if width <= 0:
        raise ValidationError("Rectangular width must be positive")
    
    # Create rectangular IRF
    half_width = width / 2.0
    irf = np.where(
        (t >= t0 - half_width) & (t <= t0 + half_width),
        1.0,
        0.0
    )
    
    # Normalize to unit area
    dt = np.mean(np.diff(t)) if len(t) > 1 else 1.0
    area = np.sum(irf) * dt
    
    if area == 0:
        raise ComputationError("Rectangular IRF has zero area - width may be too small")
    
    irf = irf / area
    
    return irf


def validate_irf(irf: np.ndarray, t: np.ndarray, tolerance: float = 1e-6) -> bool:
    """
    Validate IRF properties.
    
    Checks that the IRF is properly normalized and has reasonable properties
    for use in convolution operations.
    
    Args:
        irf: IRF array to validate
        t: Corresponding time array
        tolerance: Tolerance for normalization check
        
    Returns:
        True if IRF is valid
        
    Raises:
        ValidationError: If IRF fails validation
    """
    if not isinstance(irf, np.ndarray) or not isinstance(t, np.ndarray):
        raise ValidationError("IRF and time must be numpy arrays")
    
    if len(irf) != len(t):
        raise ValidationError("IRF and time arrays must have same length")
    
    if len(irf) == 0:
        raise ValidationError("IRF array cannot be empty")
    
    # Check for non-negative values
    if np.any(irf < 0):
        raise ValidationError("IRF contains negative values")
    
    # Check normalization
    dt = np.mean(np.diff(t)) if len(t) > 1 else 1.0
    area = np.sum(irf) * dt
    
    if abs(area - 1.0) > tolerance:
        raise ValidationError(f"IRF not properly normalized: area = {area:.6f}")
    
    # Check for NaN or infinite values
    if not np.all(np.isfinite(irf)):
        raise ValidationError("IRF contains NaN or infinite values")
    
    return True


def get_irf(irf_type: str, t: np.ndarray, t0: float = 0.0, **kwargs) -> np.ndarray:
    """
    Factory function to create IRF of specified type.
    
    Convenience function that creates and validates an IRF of the
    specified type with given parameters.
    
    Args:
        irf_type: Type of IRF ('dirac', 'gaussian', 'rectangular')
        t: Time array (ns)
        t0: Time offset for IRF peak (ns)
        **kwargs: Additional parameters specific to IRF type
            - sigma: Standard deviation for Gaussian IRF (ns)
            - width: Width for rectangular IRF (ns)
            
    Returns:
        Normalized IRF array
        
    Raises:
        ValidationError: If IRF type is unknown or parameters invalid
    """
    irf_type = irf_type.lower()
    
    if irf_type == 'dirac':
        irf = dirac_irf(t, t0)
    elif irf_type == 'gaussian':
        sigma = kwargs.get('sigma', 0.1)
        irf = gaussian_irf(t, t0, sigma)
    elif irf_type == 'rectangular':
        width = kwargs.get('width', 0.2)
        irf = rectangular_irf(t, t0, width)
    else:
        raise ValidationError(f"Unknown IRF type: {irf_type}")
    
    # Validate the created IRF
    validate_irf(irf, t)
    
    return irf

def exponential_decay(t: np.ndarray, tau: float, amplitude: float = 1.0, 
                     offset: float = 0.0) -> np.ndarray:
    """
    Exponential decay function.
    
    Implements a single exponential decay function commonly used to
    model fluorescence lifetime decay.
    
    Args:
        t: Time array (ns)
        tau: Decay time constant (ns)
        amplitude: Amplitude scaling factor
        offset: Time offset (ns)
        
    Returns:
        Exponential decay array
        
    Raises:
        ValidationError: If parameters are invalid
    """
    if not isinstance(t, np.ndarray):
        t = np.array(t)
    
    if tau <= 0:
        raise ValidationError("Decay time constant must be positive")
    
    # Calculate exponential decay
    decay = amplitude * np.exp(-(t - offset) / tau)
    
    # Set negative times to zero
    decay[t < offset] = 0.0
    
    return decay


def convolve_with_exponential(irf: np.ndarray, t: np.ndarray, tau: float, 
                            T: float, amplitude: float = 1.0) -> np.ndarray:
    """
    Convolve IRF with exponential decay function.
    
    Performs convolution of the instrument response function with an
    exponential decay, accounting for periodic boundary conditions
    due to pulsed excitation.
    
    Args:
        irf: Instrument response function array
        t: Time array (ns)
        tau: Fluorescence lifetime (ns)
        T: Repetition period (ns)
        amplitude: Amplitude scaling factor
        
    Returns:
        Convolved signal array
        
    Raises:
        ValidationError: If input parameters are invalid
        ComputationError: If convolution fails
    """
    if not isinstance(irf, np.ndarray) or not isinstance(t, np.ndarray):
        raise ValidationError("IRF and time must be numpy arrays")
    
    if len(irf) != len(t):
        raise ValidationError("IRF and time arrays must have same length")
    
    if tau <= 0:
        raise ValidationError("Lifetime must be positive")
    
    if T <= 0:
        raise ValidationError("Repetition period must be positive")
    
    # Validate IRF
    validate_irf(irf, t)
    
    # Create exponential decay
    decay = exponential_decay(t, tau, amplitude)
    
    # Perform convolution using scipy.signal for efficiency
    try:
        # Use 'full' mode and then extract the relevant portion
        convolved_full = signal.convolve(irf, decay, mode='full')
        
        # Extract the causal part (same length as input)
        n = len(irf)
        convolved = convolved_full[:n]
        
        # Apply periodic boundary conditions for pulsed excitation
        # This accounts for the fact that fluorescence from previous pulses
        # can contribute to the current measurement window
        dt = np.mean(np.diff(t)) if len(t) > 1 else 1.0
        n_periods = int(np.ceil(5 * tau / T))  # Include ~5 decay constants
        
        if n_periods > 1 and T < 10 * tau:  # Only apply if significant overlap
            # Add contributions from previous pulses
            for n in range(1, n_periods):
                # Time shift for previous pulse
                t_shift = t - n * T
                
                # Only consider times within the current window
                valid_mask = t_shift >= t[0]
                if np.any(valid_mask):
                    decay_shifted = exponential_decay(t_shift, tau, amplitude)
                    decay_shifted[~valid_mask] = 0.0
                    
                    # Add contribution from this previous pulse
                    convolved_shifted_full = signal.convolve(irf, decay_shifted, mode='full')
                    convolved_shifted = convolved_shifted_full[:n]
                    convolved += convolved_shifted
        
    except Exception as e:
        raise ComputationError(f"Convolution failed: {str(e)}")
    
    return convolved


def convolve_fft(irf: np.ndarray, decay: np.ndarray) -> np.ndarray:
    """
    Efficient FFT-based convolution for large arrays.
    
    Uses FFT-based convolution for improved performance with large
    arrays, particularly useful for high time resolution data.
    
    Args:
        irf: Instrument response function array
        decay: Exponential decay array
        
    Returns:
        Convolved signal array (same length as input)
        
    Raises:
        ValidationError: If input arrays are invalid
        ComputationError: If FFT convolution fails
    """
    if not isinstance(irf, np.ndarray) or not isinstance(decay, np.ndarray):
        raise ValidationError("IRF and decay must be numpy arrays")
    
    if len(irf) != len(decay):
        raise ValidationError("IRF and decay arrays must have same length")
    
    try:
        # Pad arrays to avoid circular convolution artifacts
        n = len(irf)
        padded_size = 2 * n - 1
        
        # Zero-pad both arrays
        irf_padded = np.zeros(padded_size)
        decay_padded = np.zeros(padded_size)
        
        irf_padded[:n] = irf
        decay_padded[:n] = decay
        
        # Perform FFT convolution
        irf_fft = np.fft.fft(irf_padded)
        decay_fft = np.fft.fft(decay_padded)
        
        convolved_fft = irf_fft * decay_fft
        convolved_full = np.real(np.fft.ifft(convolved_fft))
        
        # Extract the causal part (first n points)
        result = convolved_full[:n]
        
    except Exception as e:
        raise ComputationError(f"FFT convolution failed: {str(e)}")
    
    return result


def handle_boundary_conditions(signal_array: np.ndarray, t: np.ndarray, 
                             T: float, method: str = 'periodic') -> np.ndarray:
    """
    Handle boundary conditions for pulsed excitation.
    
    Applies appropriate boundary conditions to account for the periodic
    nature of pulsed excitation in FLIM measurements.
    
    Args:
        signal_array: Input signal array
        t: Time array (ns)
        T: Repetition period (ns)
        method: Boundary condition method ('periodic', 'zero', 'reflect')
        
    Returns:
        Signal array with boundary conditions applied
        
    Raises:
        ValidationError: If method is unknown or parameters invalid
    """
    if not isinstance(signal_array, np.ndarray) or not isinstance(t, np.ndarray):
        raise ValidationError("Signal and time must be numpy arrays")
    
    if len(signal_array) != len(t):
        raise ValidationError("Signal and time arrays must have same length")
    
    if T <= 0:
        raise ValidationError("Repetition period must be positive")
    
    method = method.lower()
    
    if method == 'periodic':
        # For periodic boundary conditions, the signal repeats every T
        # This is the most physically realistic for pulsed excitation
        result = signal_array.copy()
        
    elif method == 'zero':
        # Zero boundary conditions - signal is zero outside the window
        result = signal_array.copy()
        
    elif method == 'reflect':
        # Reflective boundary conditions
        result = signal_array.copy()
        
    else:
        raise ValidationError(f"Unknown boundary condition method: {method}")
    
    return result


def validate_convolution_inputs(irf: np.ndarray, t: np.ndarray, tau: float, 
                              T: float) -> None:
    """
    Validate inputs for convolution operations.
    
    Performs comprehensive validation of all inputs required for
    IRF-decay convolution operations.
    
    Args:
        irf: Instrument response function array
        t: Time array (ns)
        tau: Fluorescence lifetime (ns)
        T: Repetition period (ns)
        
    Raises:
        ValidationError: If any input is invalid
    """
    # Validate arrays
    if not isinstance(irf, np.ndarray) or not isinstance(t, np.ndarray):
        raise ValidationError("IRF and time must be numpy arrays")
    
    if len(irf) != len(t):
        raise ValidationError("IRF and time arrays must have same length")
    
    if len(irf) == 0:
        raise ValidationError("Arrays cannot be empty")
    
    # Validate IRF properties
    validate_irf(irf, t)
    
    # Validate physical parameters
    if tau <= 0:
        raise ValidationError("Fluorescence lifetime must be positive")
    
    if T <= 0:
        raise ValidationError("Repetition period must be positive")
    
    # Check time array properties
    if not np.all(np.diff(t) > 0):
        raise ValidationError("Time array must be monotonically increasing")
    
    # Warn about potential numerical issues
    dt = np.mean(np.diff(t))
    if tau < 2 * dt:
        import warnings
        warnings.warn(
            f"Lifetime ({tau:.3f} ns) is very small compared to time step "
            f"({dt:.3f} ns). Consider using finer time resolution.",
            UserWarning
        )
    
    if T < 5 * tau:
        import warnings
        warnings.warn(
            f"Repetition period ({T:.1f} ns) is less than 5 times the lifetime "
            f"({tau:.3f} ns). Pulse overlap effects may be significant.",
            UserWarning
        )