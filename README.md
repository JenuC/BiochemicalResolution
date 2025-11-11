# Instrument Response Function Analysis

A Python package for analyzing instrument response functions (IRF) in fluorescence lifetime imaging microscopy (FLIM) using Fisher information analysis and Monte Carlo simulations.

## Overview

This package converts and enhances Mathematica notebooks for IRF analysis, providing:

- **Fisher Information Analysis**: Calculate measurement precision for different IRF shapes
- **Monte Carlo Simulations**: Validate theoretical predictions with synthetic data
- **Comprehensive Visualization**: Generate publication-ready plots and analysis
- **Modular Design**: Reusable components for custom analysis workflows

## Installation

### Prerequisites

- Python 3.8 or higher
- pip or uv package manager

### Using uv (Recommended)

```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt

# Install package in development mode
uv pip install -e .
```

### Using pip

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Quick Start

### Basic Usage

```python
import numpy as np
from src.python import AnalysisParameters, fisher_information

# Set up analysis parameters
params = AnalysisParameters(
    repetition_period=25.0,  # ns
    lifetimes=np.arange(0.2, 15, 0.4),
    time_bins=2**(np.arange(9)+2)
)

# Run Dirac IRF Fisher analysis
results = fisher_information.dirac_irf_analysis(
    params.lifetimes, 
    params.time_bins
)
```

### Jupyter Notebooks

The package includes comprehensive Jupyter notebooks:

1. **01_dirac_irf_fisher_analysis.ipynb**: Dirac IRF Fisher information analysis
2. **02_gaussian_irf_fisher_analysis.ipynb**: Gaussian IRF analysis  
3. **03_monte_carlo_simulations.ipynb**: Monte Carlo validation
4. **04_visualization_and_analysis.ipynb**: Comprehensive plotting and analysis

```bash
# Start Jupyter
jupyter notebook notebooks/
```

## Package Structure

```
BiochemicalResolution/
├── src/
│   └── python/                  # Core package modules
│       ├── __init__.py
│       ├── core.py              # Data structures and configuration
│       ├── fisher_information.py # Fisher information calculations
│       ├── irf_functions.py     # IRF models and convolution
│       ├── monte_carlo.py       # Monte Carlo simulations
│       └── visualization.py     # Plotting and visualization
├── notebooks/                   # Jupyter notebooks (converted from Mathematica)
├── mathematica/                 # Original Mathematica files
│   ├── notebooks/               # Original .nb files
│   ├── docs/                    # PDF documentation
│   └── data/                    # Original data files (.npy, .mat)
├── output/                      # Generated outputs
│   ├── plots/                   # Generated plots and figures
│   ├── demos/                   # Demo and test scripts
│   └── validation/              # Validation reports
├── data/                        # Runtime data
│   ├── generated/               # Generated analysis results
│   └── reference/               # Reference data for validation
├── requirements.txt             # Package dependencies
├── setup.py                     # Package installation
└── README.md                    # This file
```

## Key Features

### Fisher Information Analysis
- Dirac IRF analysis reproducing Kollner-Wolfrum results
- Gaussian IRF analysis with configurable widths
- Parameter sweep capabilities for comprehensive analysis

### Monte Carlo Simulations
- Synthetic photon data generation with Poisson statistics
- Curve fitting with robust error handling
- Validation against Fisher information predictions

### Visualization
- Consistent styling across all plots
- Publication-ready figure generation
- Interactive analysis capabilities

## Dependencies

Core dependencies:
- NumPy (≥1.21.0): Numerical computations
- SciPy (≥1.7.0): Scientific computing and optimization
- Matplotlib (≥3.5.0): Plotting and visualization
- Jupyter (≥1.0.0): Notebook environment

See `requirements.txt` for complete dependency list.

## Development

### Running Tests

```bash
# Install development dependencies
pip install -e .[dev]

# Run tests
pytest

# Run with coverage
pytest --cov=src/python
```

### Code Formatting

```bash
# Format code
black src/python/ notebooks/

# Check style
flake8 src/python/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

- Kollner, M. and Wolfrum, J. "Chemical Physics Letters" (1992) - Fisher information theory for FLIM
- Original Mathematica implementations for validation

## Documentation

### Quick Links

- **[Usage Examples](USAGE_EXAMPLES.md)**: Practical examples and parameter guidelines
- **[Troubleshooting Guide](TROUBLESHOOTING.md)**: Common issues and solutions
- **[Validation Report](output/validation/VALIDATION_REPORT.md)**: Numerical accuracy validation
- **[Plot Customization](PLOT_CUSTOMIZATION_GUIDE.md)**: Advanced plotting options
- **[Specialized Plots](SPECIALIZED_PLOTS_DOCUMENTATION.md)**: Specialized analysis plots

### Example Usage Scenarios

#### Scenario 1: Quick Fisher Information Analysis
```python
from src.python import AnalysisParameters, fisher_information
import numpy as np

params = AnalysisParameters()
results = fisher_information.dirac_irf_analysis(
    params.lifetimes, 
    params.time_bins
)
```

#### Scenario 2: Comparing IRF Effects
```python
from src.python.visualization import plot_irf_comparison

# Compare Dirac vs Gaussian IRF
results = compare_irf_types(
    tau=2.5, 
    irf_widths=[0.01, 0.1, 0.5]
)
plot_irf_comparison(results)
```

#### Scenario 3: Monte Carlo Validation
```python
from src.python.monte_carlo import monte_carlo_analysis

# Validate Fisher predictions empirically
mc_results = monte_carlo_analysis(
    tau_range=[2.5],
    iterations=1000,
    num_photons=50000
)
```

See [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md) for more detailed examples.

## Troubleshooting

Common issues and solutions:

- **Installation problems**: See [TROUBLESHOOTING.md](TROUBLESHOOTING.md#installation-issues)
- **Notebook execution errors**: See [TROUBLESHOOTING.md](TROUBLESHOOTING.md#notebook-execution-issues)
- **Numerical issues**: See [TROUBLESHOOTING.md](TROUBLESHOOTING.md#numerical-issues)
- **Performance problems**: See [TROUBLESHOOTING.md](TROUBLESHOOTING.md#performance-issues)

## Support

For questions or issues:
1. Check the [Troubleshooting Guide](TROUBLESHOOTING.md)
2. Review [Usage Examples](USAGE_EXAMPLES.md)
3. Check the documentation in the notebooks
4. Review the [Validation Report](output/validation/VALIDATION_REPORT.md)
5. Open an issue on the project repository