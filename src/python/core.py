"""
Core data structures and configuration classes for IRF analysis.

This module provides the fundamental data structures used throughout
the IRF analysis package, including parameter configurations and
result containers.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import numpy as np
import pickle
import json
from pathlib import Path


@dataclass
class AnalysisParameters:
    """
    Configuration parameters for IRF analysis.
    
    Contains default FLIM parameters commonly used in fluorescence
    lifetime imaging microscopy analysis.
    
    Attributes:
        repetition_period: Laser repetition period in nanoseconds
        lifetimes: Array of fluorescence lifetimes to analyze (ns)
        time_bins: Array of time bin counts for analysis
        irf_sigmas: Array of IRF Gaussian widths to test (ns)
        num_photons: Number of photons for Monte Carlo simulations
        iterations: Number of Monte Carlo iterations
    """
    repetition_period: float = 25.0  # ns
    lifetimes: np.ndarray = field(default_factory=lambda: np.arange(0.2, 15, 0.4))
    time_bins: np.ndarray = field(default_factory=lambda: 2**(np.arange(9)+2))
    irf_sigmas: np.ndarray = field(default_factory=lambda: np.array([0.01, 0.1, 0.25, 0.5, 1, 2]))
    num_photons: int = 75000
    iterations: int = 5000
    
    def validate(self) -> None:
        """
        Validate parameter ranges and consistency.
        
        Raises:
            ValidationError: If parameters are invalid
        """
        if self.repetition_period <= 0:
            raise ValidationError("Repetition period must be positive")
        
        if np.any(self.lifetimes <= 0):
            raise ValidationError("All lifetimes must be positive")
            
        if np.any(self.time_bins <= 0):
            raise ValidationError("All time bins must be positive")
            
        if np.any(self.irf_sigmas <= 0):
            raise ValidationError("All IRF sigmas must be positive")
            
        if self.num_photons <= 0:
            raise ValidationError("Number of photons must be positive")
            
        if self.iterations <= 0:
            raise ValidationError("Number of iterations must be positive")


@dataclass 
class AnalysisResults:
    """
    Container for analysis results with save/load functionality.
    
    Stores analysis results along with metadata and parameters
    used to generate them.
    
    Attributes:
        f_values: Fisher information values or analysis results
        parameters: Analysis parameters used
        metadata: Additional metadata about the analysis
        analysis_type: Type of analysis performed
    """
    f_values: np.ndarray
    parameters: AnalysisParameters
    metadata: Dict[str, Any]
    analysis_type: str
    
    def save(self, filepath: str) -> None:
        """
        Save results to file.
        
        Supports both pickle (.pkl) and numpy (.npz) formats.
        
        Args:
            filepath: Path to save file
        """
        filepath = Path(filepath)
        
        if filepath.suffix == '.pkl':
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
        elif filepath.suffix == '.npz':
            # Convert numpy types to Python types for JSON serialization
            metadata_serializable = {}
            for key, value in self.metadata.items():
                if isinstance(value, np.integer):
                    metadata_serializable[key] = int(value)
                elif isinstance(value, np.floating):
                    metadata_serializable[key] = float(value)
                elif isinstance(value, np.ndarray):
                    metadata_serializable[key] = value.tolist()
                else:
                    metadata_serializable[key] = value
            
            # Save as numpy archive
            np.savez(
                filepath,
                f_values=self.f_values,
                lifetimes=self.parameters.lifetimes,
                time_bins=self.parameters.time_bins,
                irf_sigmas=self.parameters.irf_sigmas,
                repetition_period=self.parameters.repetition_period,
                num_photons=self.parameters.num_photons,
                iterations=self.parameters.iterations,
                analysis_type=self.analysis_type,
                metadata=json.dumps(metadata_serializable)
            )
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    @classmethod
    def load(cls, filepath: str) -> 'AnalysisResults':
        """
        Load results from file.
        
        Args:
            filepath: Path to load file
            
        Returns:
            AnalysisResults object
        """
        filepath = Path(filepath)
        
        if filepath.suffix == '.pkl':
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        elif filepath.suffix == '.npz':
            data = np.load(filepath, allow_pickle=True)
            
            # Reconstruct parameters
            params = AnalysisParameters(
                repetition_period=float(data['repetition_period']),
                lifetimes=data['lifetimes'],
                time_bins=data['time_bins'],
                irf_sigmas=data['irf_sigmas'],
                num_photons=int(data['num_photons']),
                iterations=int(data['iterations'])
            )
            
            # Reconstruct metadata
            metadata = json.loads(str(data['metadata']))
            
            return cls(
                f_values=data['f_values'],
                parameters=params,
                metadata=metadata,
                analysis_type=str(data['analysis_type'])
            )
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")


# Base exception classes for error handling
class IRFAnalysisError(Exception):
    """Base exception for IRF analysis errors."""
    pass


class ValidationError(IRFAnalysisError):
    """Raised when input validation fails."""
    pass


class ConvergenceError(IRFAnalysisError):
    """Raised when iterative algorithms fail to converge."""
    pass


class ComputationError(IRFAnalysisError):
    """Raised when numerical computations fail."""
    pass