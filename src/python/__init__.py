"""
InstrumentResponseFunction Analysis Package

A Python package for analyzing instrument response functions (IRF) in 
fluorescence lifetime imaging microscopy (FLIM) using Fisher information 
analysis and Monte Carlo simulations.
"""

__version__ = "0.1.0"
__author__ = "Research Team"

# Import main modules for easy access
try:
    from . import fisher_information
except ImportError:
    pass
from . import irf_functions
try:
    from . import monte_carlo
except ImportError:
    pass
try:
    from . import visualization
except ImportError:
    pass

# Import core data structures
from .core import AnalysisParameters, AnalysisResults

__all__ = [
    'irf_functions',
    'fisher_information',
    'monte_carlo',
    'visualization',
    'AnalysisParameters',
    'AnalysisResults'
]