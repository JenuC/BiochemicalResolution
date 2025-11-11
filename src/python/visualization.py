"""
Visualization Module for IRF Analysis.

This module provides standardized plotting functions for all analysis types
in the IRF analysis package. It implements consistent styling and formatting
across all plots, with support for interactive plotting capabilities and
parameter exploration.

The module includes functions for Fisher information analysis plots,
Monte Carlo simulation visualizations, IRF comparisons, and specialized
analysis plots for biochemical resolution and resolving power analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from matplotlib.ticker import LogLocator, LogFormatter
from typing import Dict, Any, Optional, Union, List, Tuple
import warnings

try:
    from .core import AnalysisResults, AnalysisParameters, ValidationError
    from .fisher_information import calculate_f_value
    from .irf_functions import get_irf
except ImportError:
    from core import AnalysisResults, AnalysisParameters, ValidationError
    from fisher_information import calculate_f_value
    from irf_functions import get_irf

# Default color scheme and styling
DEFAULT_COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e', 
    'accent': '#2ca02c',
    'warning': '#d62728',
    'info': '#9467bd',
    'success': '#8c564b',
    'neutral': '#7f7f7f'
}

# Additional color schemes for different use cases
COLOR_SCHEMES = {
    'default': DEFAULT_COLORS,
    'viridis': {
        'primary': '#440154',
        'secondary': '#31688e',
        'accent': '#35b779',
        'warning': '#fde724',
        'info': '#21918c',
        'success': '#5ec962',
        'neutral': '#3b528b'
    },
    'publication': {
        'primary': '#000000',
        'secondary': '#e41a1c',
        'accent': '#377eb8',
        'warning': '#ff7f00',
        'info': '#984ea3',
        'success': '#4daf4a',
        'neutral': '#999999'
    },
    'colorblind': {
        'primary': '#0173b2',
        'secondary': '#de8f05',
        'accent': '#029e73',
        'warning': '#cc78bc',
        'info': '#ca9161',
        'success': '#949494',
        'neutral': '#ece133'
    },
    'grayscale': {
        'primary': '#000000',
        'secondary': '#404040',
        'accent': '#808080',
        'warning': '#a0a0a0',
        'info': '#c0c0c0',
        'success': '#606060',
        'neutral': '#d0d0d0'
    }
}

DEFAULT_STYLE = {
    'figure_size': (10, 8),
    'dpi': 100,
    'font_size': 12,
    'title_size': 14,
    'label_size': 12,
    'tick_size': 10,
    'line_width': 2,
    'marker_size': 6,
    'grid_alpha': 0.3,
    'legend_frameon': True,
    'legend_fancybox': True,
    'legend_shadow': True
}

# Export format configurations
EXPORT_FORMATS = {
    'pdf': {'format': 'pdf', 'dpi': 300, 'bbox_inches': 'tight', 'transparent': False},
    'png': {'format': 'png', 'dpi': 300, 'bbox_inches': 'tight', 'transparent': False},
    'svg': {'format': 'svg', 'bbox_inches': 'tight', 'transparent': False},
    'eps': {'format': 'eps', 'dpi': 300, 'bbox_inches': 'tight'},
    'jpg': {'format': 'jpg', 'dpi': 300, 'bbox_inches': 'tight', 'quality': 95},
    'tiff': {'format': 'tiff', 'dpi': 300, 'bbox_inches': 'tight'}
}


def setup_plot_style(style_params: Optional[Dict] = None) -> None:
    """
    Set up consistent plot styling across all visualizations.
    
    Configures matplotlib parameters for consistent appearance
    across all plots in the analysis package.
    
    Args:
        style_params: Optional dictionary to override default style parameters
    """
    # Merge with default style
    style = DEFAULT_STYLE.copy()
    if style_params:
        style.update(style_params)
    
    # Configure matplotlib parameters
    plt.rcParams.update({
        'figure.figsize': style['figure_size'],
        'figure.dpi': style['dpi'],
        'font.size': style['font_size'],
        'axes.titlesize': style['title_size'],
        'axes.labelsize': style['label_size'],
        'xtick.labelsize': style['tick_size'],
        'ytick.labelsize': style['tick_size'],
        'lines.linewidth': style['line_width'],
        'lines.markersize': style['marker_size'],
        'grid.alpha': style['grid_alpha'],
        'legend.frameon': style['legend_frameon'],
        'legend.fancybox': style['legend_fancybox'],
        'legend.shadow': style['legend_shadow']
    })


def get_color_scheme(scheme_name: str = 'default') -> Dict[str, str]:
    """
    Get a color scheme by name.
    
    Provides access to predefined color schemes optimized for different
    use cases including publication, colorblind-friendly, and grayscale.
    
    Args:
        scheme_name: Name of the color scheme ('default', 'viridis', 
                    'publication', 'colorblind', 'grayscale')
    
    Returns:
        Dictionary mapping color roles to hex color codes
        
    Raises:
        ValidationError: If scheme_name is not recognized
        
    Examples:
        >>> colors = get_color_scheme('publication')
        >>> plt.plot(x, y, color=colors['primary'])
    """
    if scheme_name not in COLOR_SCHEMES:
        available = ', '.join(COLOR_SCHEMES.keys())
        raise ValidationError(f"Unknown color scheme '{scheme_name}'. Available: {available}")
    
    return COLOR_SCHEMES[scheme_name].copy()


def apply_color_scheme(scheme_name: str = 'default') -> None:
    """
    Apply a color scheme globally to all subsequent plots.
    
    Updates the DEFAULT_COLORS dictionary to use the specified color scheme.
    This affects all plotting functions that use DEFAULT_COLORS.
    
    Args:
        scheme_name: Name of the color scheme to apply
        
    Raises:
        ValidationError: If scheme_name is not recognized
        
    Examples:
        >>> apply_color_scheme('publication')
        >>> fig = plot_fisher_analysis(results)  # Uses publication colors
    """
    global DEFAULT_COLORS
    DEFAULT_COLORS = get_color_scheme(scheme_name)


def export_figure(fig: plt.Figure, 
                 filepath: str,
                 formats: Optional[Union[str, List[str]]] = None,
                 dpi: int = 300,
                 transparent: bool = False,
                 bbox_inches: str = 'tight',
                 **kwargs) -> List[str]:
    """
    Export a matplotlib figure in one or multiple formats.
    
    Provides a unified interface for exporting figures in various formats
    with consistent quality settings. Supports batch export to multiple
    formats simultaneously.
    
    Args:
        fig: Matplotlib Figure object to export
        filepath: Base filepath without extension (e.g., 'figures/plot')
        formats: Single format string or list of formats 
                ('pdf', 'png', 'svg', 'eps', 'jpg', 'tiff')
                If None, exports as PNG
        dpi: Resolution in dots per inch (default: 300)
        transparent: Whether to use transparent background (default: False)
        bbox_inches: Bounding box setting (default: 'tight')
        **kwargs: Additional arguments passed to fig.savefig()
        
    Returns:
        List of saved file paths
        
    Raises:
        ValidationError: If format is not supported
        IOError: If file cannot be saved
        
    Examples:
        >>> fig, ax = plt.subplots()
        >>> ax.plot([1, 2, 3], [1, 4, 9])
        >>> export_figure(fig, 'myplot', formats=['pdf', 'png'])
        ['myplot.pdf', 'myplot.png']
        
        >>> export_figure(fig, 'myplot', formats='svg', transparent=True)
        ['myplot.svg']
    """
    if not isinstance(fig, plt.Figure):
        raise ValidationError("First argument must be a matplotlib Figure object")
    
    # Handle formats parameter
    if formats is None:
        formats = ['png']
    elif isinstance(formats, str):
        formats = [formats]
    elif not isinstance(formats, list):
        raise ValidationError("formats must be a string or list of strings")
    
    # Validate formats
    for fmt in formats:
        if fmt not in EXPORT_FORMATS:
            available = ', '.join(EXPORT_FORMATS.keys())
            raise ValidationError(f"Unsupported format '{fmt}'. Available: {available}")
    
    saved_files = []
    
    # Export to each format
    for fmt in formats:
        # Get format-specific settings
        export_params = EXPORT_FORMATS[fmt].copy()
        
        # Override with user-provided parameters
        export_params['dpi'] = dpi
        export_params['transparent'] = transparent
        export_params['bbox_inches'] = bbox_inches
        export_params.update(kwargs)
        
        # Construct full filepath
        full_path = f"{filepath}.{fmt}"
        
        try:
            # Save figure
            fig.savefig(full_path, **export_params)
            saved_files.append(full_path)
        except Exception as e:
            warnings.warn(f"Failed to save {full_path}: {str(e)}")
    
    return saved_files


def add_text_annotation(ax: plt.Axes,
                       text: str,
                       position: Tuple[float, float],
                       coords: str = 'data',
                       fontsize: Optional[int] = None,
                       color: str = 'black',
                       bbox_style: Optional[Dict] = None,
                       arrow_props: Optional[Dict] = None,
                       **kwargs) -> plt.Annotation:
    """
    Add a text annotation to a plot with optional styling and arrow.
    
    Provides a convenient interface for adding annotations with consistent
    styling. Supports both data coordinates and axes/figure coordinates.
    
    Args:
        ax: Matplotlib Axes object to annotate
        text: Text content of the annotation
        position: (x, y) position for the annotation
        coords: Coordinate system ('data', 'axes', 'figure')
                - 'data': Use data coordinates
                - 'axes': Use axes coordinates (0-1 range)
                - 'figure': Use figure coordinates (0-1 range)
        fontsize: Font size (uses default if None)
        color: Text color
        bbox_style: Dictionary with bbox styling parameters
                   Example: {'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.5}
        arrow_props: Dictionary with arrow properties for pointing annotation
                    Example: {'arrowstyle': '->', 'connectionstyle': 'arc3,rad=0.3'}
        **kwargs: Additional arguments passed to ax.annotate()
        
    Returns:
        Matplotlib Annotation object
        
    Raises:
        ValidationError: If coords is not recognized
        
    Examples:
        >>> fig, ax = plt.subplots()
        >>> ax.plot([1, 2, 3], [1, 4, 9])
        >>> add_text_annotation(ax, 'Peak', (2, 4), arrow_props={'arrowstyle': '->'})
        
        >>> add_text_annotation(ax, 'Note', (0.5, 0.95), coords='axes',
        ...                    bbox_style={'boxstyle': 'round', 'facecolor': 'yellow'})
    """
    # Validate coordinate system
    coord_map = {
        'data': 'data',
        'axes': 'axes fraction',
        'figure': 'figure fraction'
    }
    
    if coords not in coord_map:
        raise ValidationError(f"coords must be one of: {', '.join(coord_map.keys())}")
    
    # Set default fontsize
    if fontsize is None:
        fontsize = DEFAULT_STYLE['font_size']
    
    # Set default bbox style if provided
    if bbox_style is None and arrow_props is None:
        bbox_style = {'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.5}
    
    # Create annotation
    annotation = ax.annotate(
        text,
        xy=position,
        xycoords=coord_map[coords],
        fontsize=fontsize,
        color=color,
        bbox=bbox_style,
        arrowprops=arrow_props,
        **kwargs
    )
    
    return annotation


def add_parameter_box(ax: plt.Axes,
                     params: Dict[str, Any],
                     position: str = 'upper right',
                     fontsize: Optional[int] = None,
                     **kwargs) -> plt.Annotation:
    """
    Add a parameter information box to a plot.
    
    Creates a formatted text box displaying analysis parameters in a
    consistent style. Useful for documenting experimental conditions
    directly on plots.
    
    Args:
        ax: Matplotlib Axes object
        params: Dictionary of parameters to display
               Keys are parameter names, values are parameter values
        position: Position of the box ('upper right', 'upper left', 
                 'lower right', 'lower left', 'center')
        fontsize: Font size (uses default if None)
        **kwargs: Additional styling arguments
        
    Returns:
        Matplotlib Annotation object
        
    Examples:
        >>> fig, ax = plt.subplots()
        >>> ax.plot([1, 2, 3], [1, 4, 9])
        >>> params = {'τ': '2.5 ns', 'T': '25 ns', 'bins': 512}
        >>> add_parameter_box(ax, params, position='upper left')
    """
    # Format parameter text
    param_lines = []
    for key, value in params.items():
        if isinstance(value, float):
            param_lines.append(f"{key}: {value:.3g}")
        else:
            param_lines.append(f"{key}: {value}")
    
    param_text = '\n'.join(param_lines)
    
    # Map position to coordinates
    position_map = {
        'upper right': (0.98, 0.98),
        'upper left': (0.02, 0.98),
        'lower right': (0.98, 0.02),
        'lower left': (0.02, 0.02),
        'center': (0.5, 0.5)
    }
    
    if position not in position_map:
        raise ValidationError(f"position must be one of: {', '.join(position_map.keys())}")
    
    pos = position_map[position]
    
    # Determine alignment
    ha = 'right' if 'right' in position else 'left'
    va = 'top' if 'upper' in position else 'bottom'
    if position == 'center':
        ha, va = 'center', 'center'
    
    # Set default fontsize
    if fontsize is None:
        fontsize = DEFAULT_STYLE['font_size'] - 2
    
    # Default bbox style
    bbox_style = kwargs.pop('bbox_style', {
        'boxstyle': 'round',
        'facecolor': 'white',
        'alpha': 0.8,
        'edgecolor': 'gray'
    })
    
    # Add annotation
    annotation = ax.text(
        pos[0], pos[1],
        param_text,
        transform=ax.transAxes,
        fontsize=fontsize,
        verticalalignment=va,
        horizontalalignment=ha,
        bbox=bbox_style,
        **kwargs
    )
    
    return annotation


def add_scale_bar(ax: plt.Axes,
                 length: float,
                 label: Optional[str] = None,
                 position: Tuple[float, float] = (0.8, 0.1),
                 coords: str = 'axes',
                 linewidth: int = 3,
                 color: str = 'black',
                 **kwargs) -> List:
    """
    Add a scale bar to a plot.
    
    Useful for images and spatial data where a scale reference is needed.
    
    Args:
        ax: Matplotlib Axes object
        length: Length of the scale bar in data units
        label: Optional text label for the scale bar
        position: (x, y) position for the scale bar
        coords: Coordinate system ('data' or 'axes')
        linewidth: Width of the scale bar line
        color: Color of the scale bar
        **kwargs: Additional styling arguments
        
    Returns:
        List of matplotlib artists (line and text)
        
    Examples:
        >>> fig, ax = plt.subplots()
        >>> ax.imshow(image_data)
        >>> add_scale_bar(ax, length=10, label='10 μm')
    """
    artists = []
    
    # Determine coordinate transform
    if coords == 'axes':
        transform = ax.transAxes
    elif coords == 'data':
        transform = ax.transData
    else:
        raise ValidationError("coords must be 'data' or 'axes'")
    
    # Draw scale bar line
    x_start, y_pos = position
    x_end = x_start + (length if coords == 'data' else 0.1)
    
    line = ax.plot([x_start, x_end], [y_pos, y_pos],
                   color=color, linewidth=linewidth,
                   transform=transform, **kwargs)[0]
    artists.append(line)
    
    # Add label if provided
    if label:
        text = ax.text(x_start + (x_end - x_start) / 2, y_pos - 0.02,
                      label, transform=transform,
                      ha='center', va='top',
                      fontsize=DEFAULT_STYLE['font_size'],
                      color=color)
        artists.append(text)
    
    return artists


def create_custom_colormap(colors: List[str], 
                          name: str = 'custom',
                          n_bins: int = 256) -> mcolors.LinearSegmentedColormap:
    """
    Create a custom colormap from a list of colors.
    
    Generates a smooth colormap interpolating between the provided colors.
    Useful for creating publication-specific or brand-specific colormaps.
    
    Args:
        colors: List of color specifications (hex, RGB, or named colors)
        name: Name for the colormap
        n_bins: Number of discrete color bins (default: 256)
        
    Returns:
        LinearSegmentedColormap object
        
    Examples:
        >>> cmap = create_custom_colormap(['#000000', '#ff0000', '#ffff00'])
        >>> plt.imshow(data, cmap=cmap)
        
        >>> cmap = create_custom_colormap(['blue', 'white', 'red'], name='bwr_custom')
    """
    if len(colors) < 2:
        raise ValidationError("Need at least 2 colors to create a colormap")
    
    # Create colormap
    cmap = mcolors.LinearSegmentedColormap.from_list(name, colors, N=n_bins)
    
    return cmap


def save_plot_with_metadata(fig: plt.Figure,
                           filepath: str,
                           metadata: Optional[Dict[str, Any]] = None,
                           formats: Optional[Union[str, List[str]]] = None,
                           **kwargs) -> List[str]:
    """
    Save a plot with embedded metadata.
    
    Exports figures with metadata embedded in the file (for formats that support it).
    Useful for maintaining provenance and analysis parameters with figures.
    
    For PDF format, metadata is mapped to standard PDF fields:
    - Title, Author, Subject, Keywords, Creator, Producer
    
    Args:
        fig: Matplotlib Figure object
        filepath: Base filepath without extension
        metadata: Dictionary of metadata to embed
                 Standard keys: 'Title', 'Author', 'Subject', 'Keywords'
                 Custom keys will be converted to 'Keywords' field
                 Example: {'Title': 'Fisher Analysis', 'Author': 'Lab', 
                          'Subject': 'FLIM Analysis', 'tau': 2.5}
        formats: Export format(s)
        **kwargs: Additional arguments for export_figure()
        
    Returns:
        List of saved file paths
        
    Examples:
        >>> metadata = {
        ...     'Title': 'Fisher Information Analysis',
        ...     'Author': 'Research Lab',
        ...     'Subject': 'FLIM IRF Analysis',
        ...     'tau': 2.5,
        ...     'T': 25
        ... }
        >>> save_plot_with_metadata(fig, 'analysis_plot', metadata, formats='pdf')
    """
    if metadata:
        # Map metadata to PDF-compatible format
        if formats is None or 'pdf' in (formats if isinstance(formats, list) else [formats]):
            # Standard PDF metadata fields
            pdf_fields = {'Title', 'Author', 'Subject', 'Keywords', 'Creator', 'Producer'}
            
            pdf_metadata = {}
            custom_fields = []
            
            # Separate standard and custom fields
            for key, value in metadata.items():
                if key in pdf_fields:
                    pdf_metadata[key] = str(value)
                else:
                    # Add custom fields to Keywords
                    custom_fields.append(f"{key}={value}")
            
            # Append custom fields to Keywords
            if custom_fields:
                existing_keywords = pdf_metadata.get('Keywords', '')
                if existing_keywords:
                    pdf_metadata['Keywords'] = existing_keywords + '; ' + '; '.join(custom_fields)
                else:
                    pdf_metadata['Keywords'] = '; '.join(custom_fields)
            
            if pdf_metadata:
                kwargs['metadata'] = pdf_metadata
    
    # Use standard export function
    return export_figure(fig, filepath, formats=formats, **kwargs)


def plot_fisher_analysis(results: AnalysisResults, 
                        tau_index: Optional[int] = None,
                        sigma_index: Optional[int] = None,
                        log_scale: bool = True,
                        show_colorbar: bool = True,
                        title: Optional[str] = None,
                        save_path: Optional[str] = None,
                        **kwargs) -> plt.Figure:
    """
    Create standardized Fisher information analysis plots.
    
    Generates publication-ready plots for Fisher information analysis results,
    supporting both 2D heatmaps and 1D line plots depending on the data
    dimensions and selected parameters.
    
    Args:
        results: AnalysisResults object containing Fisher information data
        tau_index: Index of lifetime to plot (for 3D data, creates 2D slice)
        sigma_index: Index of IRF sigma to plot (for 3D data, creates 2D slice)
        log_scale: Whether to use logarithmic scale for axes
        show_colorbar: Whether to show colorbar for 2D plots
        title: Custom title for the plot
        save_path: Path to save the figure
        **kwargs: Additional matplotlib parameters
        
    Returns:
        matplotlib Figure object
        
    Raises:
        ValidationError: If results data is invalid
    """
    if not isinstance(results, AnalysisResults):
        raise ValidationError("Input must be an AnalysisResults object")
    
    if results.f_values.size == 0:
        raise ValidationError("Results contain no data")
    
    # Set up plot style
    setup_plot_style()
    
    # Determine plot type based on data dimensions
    f_values = results.f_values
    params = results.parameters
    
    if f_values.ndim == 2:
        # 2D data: tau vs time_bins
        fig, ax = plt.subplots(figsize=DEFAULT_STYLE['figure_size'])
        
        # Create meshgrid for plotting
        tau_grid, bins_grid = np.meshgrid(params.lifetimes, params.time_bins, indexing='ij')
        
        # Create contour plot
        if log_scale:
            im = ax.contourf(tau_grid, bins_grid, f_values, 
                           levels=20, cmap='viridis', **kwargs)
            ax.set_yscale('log')
        else:
            im = ax.contourf(tau_grid, bins_grid, f_values, 
                           levels=20, cmap='viridis', **kwargs)
        
        # Add colorbar
        if show_colorbar:
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('F-value', fontsize=DEFAULT_STYLE['label_size'])
        
        # Set labels and title
        ax.set_xlabel('Fluorescence Lifetime τ (ns)', fontsize=DEFAULT_STYLE['label_size'])
        ax.set_ylabel('Number of Time Bins', fontsize=DEFAULT_STYLE['label_size'])
        
        if title is None:
            title = f'{results.analysis_type.replace("_", " ").title()} - Fisher Information Analysis'
        ax.set_title(title, fontsize=DEFAULT_STYLE['title_size'])
        
    elif f_values.ndim == 3:
        # 3D data: tau vs time_bins vs sigma
        if tau_index is not None:
            # Plot slice at specific lifetime
            fig, ax = plt.subplots(figsize=DEFAULT_STYLE['figure_size'])
            
            data_slice = f_values[tau_index, :, :]
            bins_grid, sigma_grid = np.meshgrid(params.time_bins, params.irf_sigmas, indexing='ij')
            
            if log_scale:
                im = ax.contourf(bins_grid, sigma_grid, data_slice, 
                               levels=20, cmap='viridis', **kwargs)
                ax.set_xscale('log')
            else:
                im = ax.contourf(bins_grid, sigma_grid, data_slice, 
                               levels=20, cmap='viridis', **kwargs)
            
            if show_colorbar:
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('F-value', fontsize=DEFAULT_STYLE['label_size'])
            
            ax.set_xlabel('Number of Time Bins', fontsize=DEFAULT_STYLE['label_size'])
            ax.set_ylabel('IRF Sigma (ns)', fontsize=DEFAULT_STYLE['label_size'])
            
            if title is None:
                title = f'Fisher Analysis - τ = {params.lifetimes[tau_index]:.1f} ns'
            ax.set_title(title, fontsize=DEFAULT_STYLE['title_size'])
            
        elif sigma_index is not None:
            # Plot slice at specific IRF sigma
            fig, ax = plt.subplots(figsize=DEFAULT_STYLE['figure_size'])
            
            data_slice = f_values[:, :, sigma_index]
            tau_grid, bins_grid = np.meshgrid(params.lifetimes, params.time_bins, indexing='ij')
            
            if log_scale:
                im = ax.contourf(tau_grid, bins_grid, data_slice, 
                               levels=20, cmap='viridis', **kwargs)
                ax.set_yscale('log')
            else:
                im = ax.contourf(tau_grid, bins_grid, data_slice, 
                               levels=20, cmap='viridis', **kwargs)
            
            if show_colorbar:
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('F-value', fontsize=DEFAULT_STYLE['label_size'])
            
            ax.set_xlabel('Fluorescence Lifetime τ (ns)', fontsize=DEFAULT_STYLE['label_size'])
            ax.set_ylabel('Number of Time Bins', fontsize=DEFAULT_STYLE['label_size'])
            
            if title is None:
                title = f'Fisher Analysis - σ = {params.irf_sigmas[sigma_index]:.2f} ns'
            ax.set_title(title, fontsize=DEFAULT_STYLE['title_size'])
            
        else:
            # Create multiple subplots for different sigma values
            n_sigmas = min(4, len(params.irf_sigmas))  # Show up to 4 subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.flatten()
            
            for i in range(n_sigmas):
                ax = axes[i]
                data_slice = f_values[:, :, i]
                tau_grid, bins_grid = np.meshgrid(params.lifetimes, params.time_bins, indexing='ij')
                
                if log_scale:
                    im = ax.contourf(tau_grid, bins_grid, data_slice, 
                                   levels=15, cmap='viridis', **kwargs)
                    ax.set_yscale('log')
                else:
                    im = ax.contourf(tau_grid, bins_grid, data_slice, 
                                   levels=15, cmap='viridis', **kwargs)
                
                ax.set_xlabel('Fluorescence Lifetime τ (ns)')
                ax.set_ylabel('Number of Time Bins')
                ax.set_title(f'σ = {params.irf_sigmas[i]:.2f} ns')
                
                if show_colorbar:
                    plt.colorbar(im, ax=ax)
            
            # Hide unused subplots
            for i in range(n_sigmas, 4):
                axes[i].set_visible(False)
            
            if title is None:
                title = 'Gaussian IRF Fisher Information Analysis'
            fig.suptitle(title, fontsize=DEFAULT_STYLE['title_size'])
    
    else:
        raise ValidationError(f"Unsupported data dimensions: {f_values.ndim}")
    
    # Add grid
    if hasattr(ax, 'grid'):
        ax.grid(True, alpha=DEFAULT_STYLE['grid_alpha'])
    
    # Tight layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_monte_carlo_results(results: Dict[str, Any],
                           plot_type: str = 'precision',
                           comparison_data: Optional[Dict] = None,
                           log_scale: bool = True,
                           show_error_bars: bool = True,
                           title: Optional[str] = None,
                           save_path: Optional[str] = None,
                           **kwargs) -> plt.Figure:
    """
    Create visualization for Monte Carlo simulation results.
    
    Generates plots for Monte Carlo simulation results including precision
    analysis, convergence plots, and comparison with Fisher information
    theoretical predictions.
    
    Args:
        results: Dictionary containing Monte Carlo results with keys:
                - 'tau_values': Array of true lifetime values
                - 'tau_fitted': Array of fitted lifetime values  
                - 'tau_errors': Array of fitting errors
                - 'f_values': Array of F-values (precision measures)
                - 'success_rate': Fraction of successful fits
        plot_type: Type of plot ('precision', 'convergence', 'comparison')
        comparison_data: Optional Fisher information data for comparison
        log_scale: Whether to use logarithmic scale
        show_error_bars: Whether to show error bars
        title: Custom title for the plot
        save_path: Path to save the figure
        **kwargs: Additional matplotlib parameters
        
    Returns:
        matplotlib Figure object
        
    Raises:
        ValidationError: If results data is invalid
    """
    if not isinstance(results, dict):
        raise ValidationError("Results must be a dictionary")
    
    required_keys = ['tau_values', 'tau_fitted', 'tau_errors', 'f_values']
    missing_keys = [key for key in required_keys if key not in results]
    if missing_keys:
        raise ValidationError(f"Missing required keys in results: {missing_keys}")
    
    # Set up plot style
    setup_plot_style()
    
    if plot_type == 'precision':
        # Plot F-values vs lifetime
        fig, ax = plt.subplots(figsize=DEFAULT_STYLE['figure_size'])
        
        tau_values = results['tau_values']
        f_values = results['f_values']
        
        # Plot Monte Carlo results
        if show_error_bars and 'f_value_errors' in results:
            ax.errorbar(tau_values, f_values, yerr=results['f_value_errors'],
                       fmt='o-', color=DEFAULT_COLORS['primary'], 
                       label='Monte Carlo', linewidth=DEFAULT_STYLE['line_width'],
                       markersize=DEFAULT_STYLE['marker_size'], **kwargs)
        else:
            ax.plot(tau_values, f_values, 'o-', color=DEFAULT_COLORS['primary'],
                   label='Monte Carlo', linewidth=DEFAULT_STYLE['line_width'],
                   markersize=DEFAULT_STYLE['marker_size'], **kwargs)
        
        # Add comparison data if provided
        if comparison_data:
            if 'tau_values' in comparison_data and 'f_values' in comparison_data:
                ax.plot(comparison_data['tau_values'], comparison_data['f_values'],
                       's--', color=DEFAULT_COLORS['secondary'], 
                       label='Fisher Information', linewidth=DEFAULT_STYLE['line_width'],
                       markersize=DEFAULT_STYLE['marker_size'])
        
        ax.set_xlabel('Fluorescence Lifetime τ (ns)', fontsize=DEFAULT_STYLE['label_size'])
        ax.set_ylabel('F-value (Precision Measure)', fontsize=DEFAULT_STYLE['label_size'])
        
        if log_scale:
            ax.set_yscale('log')
        
        ax.legend()
        ax.grid(True, alpha=DEFAULT_STYLE['grid_alpha'])
        
        if title is None:
            title = 'Monte Carlo Precision Analysis'
        ax.set_title(title, fontsize=DEFAULT_STYLE['title_size'])
    
    elif plot_type == 'convergence':
        # Plot convergence of fitted parameters
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(DEFAULT_STYLE['figure_size'][0], 
                                                      DEFAULT_STYLE['figure_size'][1] * 1.2))
        
        if 'iteration_data' in results:
            iterations = results['iteration_data']['iterations']
            mean_tau = results['iteration_data']['mean_tau']
            std_tau = results['iteration_data']['std_tau']
            
            # Plot mean convergence
            ax1.plot(iterations, mean_tau, color=DEFAULT_COLORS['primary'],
                    linewidth=DEFAULT_STYLE['line_width'])
            ax1.axhline(y=results.get('true_tau', np.mean(mean_tau)), 
                       color=DEFAULT_COLORS['warning'], linestyle='--', 
                       label='True Value')
            ax1.set_ylabel('Mean Fitted τ (ns)', fontsize=DEFAULT_STYLE['label_size'])
            ax1.legend()
            ax1.grid(True, alpha=DEFAULT_STYLE['grid_alpha'])
            
            # Plot standard deviation convergence
            ax2.plot(iterations, std_tau, color=DEFAULT_COLORS['accent'],
                    linewidth=DEFAULT_STYLE['line_width'])
            ax2.set_xlabel('Iteration Number', fontsize=DEFAULT_STYLE['label_size'])
            ax2.set_ylabel('Std Dev of τ (ns)', fontsize=DEFAULT_STYLE['label_size'])
            ax2.grid(True, alpha=DEFAULT_STYLE['grid_alpha'])
            
            if log_scale:
                ax2.set_yscale('log')
        
        if title is None:
            title = 'Monte Carlo Convergence Analysis'
        fig.suptitle(title, fontsize=DEFAULT_STYLE['title_size'])
    
    elif plot_type == 'comparison':
        # Comparison plot between MC and Fisher information
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        tau_values = results['tau_values']
        f_values_mc = results['f_values']
        
        # Left plot: Direct comparison
        if comparison_data and 'f_values' in comparison_data:
            f_values_fisher = comparison_data['f_values']
            
            ax1.plot(tau_values, f_values_mc, 'o-', color=DEFAULT_COLORS['primary'],
                    label='Monte Carlo', linewidth=DEFAULT_STYLE['line_width'],
                    markersize=DEFAULT_STYLE['marker_size'])
            ax1.plot(tau_values, f_values_fisher, 's--', color=DEFAULT_COLORS['secondary'],
                    label='Fisher Information', linewidth=DEFAULT_STYLE['line_width'],
                    markersize=DEFAULT_STYLE['marker_size'])
            
            ax1.set_xlabel('Fluorescence Lifetime τ (ns)')
            ax1.set_ylabel('F-value')
            ax1.legend()
            ax1.grid(True, alpha=DEFAULT_STYLE['grid_alpha'])
            ax1.set_title('Method Comparison')
            
            if log_scale:
                ax1.set_yscale('log')
            
            # Right plot: Relative difference
            rel_diff = (f_values_mc - f_values_fisher) / f_values_fisher * 100
            ax2.plot(tau_values, rel_diff, 'o-', color=DEFAULT_COLORS['warning'],
                    linewidth=DEFAULT_STYLE['line_width'],
                    markersize=DEFAULT_STYLE['marker_size'])
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax2.set_xlabel('Fluorescence Lifetime τ (ns)')
            ax2.set_ylabel('Relative Difference (%)')
            ax2.grid(True, alpha=DEFAULT_STYLE['grid_alpha'])
            ax2.set_title('MC vs Fisher Info Difference')
        
        if title is None:
            title = 'Monte Carlo vs Fisher Information Comparison'
        fig.suptitle(title, fontsize=DEFAULT_STYLE['title_size'])
    
    else:
        raise ValidationError(f"Unknown plot type: {plot_type}")
    
    # Tight layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_irf_comparison(irf_data: Dict[str, np.ndarray],
                       time_array: np.ndarray,
                       normalize: bool = True,
                       log_scale: bool = False,
                       show_legend: bool = True,
                       title: Optional[str] = None,
                       save_path: Optional[str] = None,
                       **kwargs) -> plt.Figure:
    """
    Create comparison plots for different IRF shapes and parameters.
    
    Generates plots comparing different instrument response functions,
    showing their shapes, widths, and effects on the measurement process.
    
    Args:
        irf_data: Dictionary with IRF names as keys and IRF arrays as values
                 Example: {'Dirac': dirac_irf, 'Gaussian σ=0.1': gauss_irf1, ...}
        time_array: Time array corresponding to the IRF data
        normalize: Whether to normalize IRFs to unit area
        log_scale: Whether to use logarithmic scale for y-axis
        show_legend: Whether to show legend
        title: Custom title for the plot
        save_path: Path to save the figure
        **kwargs: Additional matplotlib parameters
        
    Returns:
        matplotlib Figure object
        
    Raises:
        ValidationError: If input data is invalid
    """
    if not isinstance(irf_data, dict):
        raise ValidationError("IRF data must be a dictionary")
    
    if len(irf_data) == 0:
        raise ValidationError("IRF data dictionary cannot be empty")
    
    if not isinstance(time_array, np.ndarray):
        time_array = np.array(time_array)
    
    if len(time_array) == 0:
        raise ValidationError("Time array cannot be empty")
    
    # Set up plot style
    setup_plot_style()
    
    # Create figure
    fig, ax = plt.subplots(figsize=DEFAULT_STYLE['figure_size'])
    
    # Color cycle for different IRFs
    colors = [DEFAULT_COLORS['primary'], DEFAULT_COLORS['secondary'], 
              DEFAULT_COLORS['accent'], DEFAULT_COLORS['warning'],
              DEFAULT_COLORS['info'], DEFAULT_COLORS['success'],
              DEFAULT_COLORS['neutral']]
    
    # Line styles for variety
    line_styles = ['-', '--', '-.', ':', '-', '--', '-.']
    
    # Plot each IRF
    for i, (irf_name, irf_values) in enumerate(irf_data.items()):
        if not isinstance(irf_values, np.ndarray):
            irf_values = np.array(irf_values)
        
        if len(irf_values) != len(time_array):
            warnings.warn(f"IRF '{irf_name}' length doesn't match time array, skipping")
            continue
        
        # Normalize if requested
        if normalize:
            irf_norm = irf_values / np.trapz(irf_values, time_array)
        else:
            irf_norm = irf_values
        
        # Select color and line style
        color = colors[i % len(colors)]
        line_style = line_styles[i % len(line_styles)]
        
        # Plot IRF
        ax.plot(time_array, irf_norm, color=color, linestyle=line_style,
               linewidth=DEFAULT_STYLE['line_width'], label=irf_name, **kwargs)
    
    # Set labels and formatting
    ax.set_xlabel('Time (ns)', fontsize=DEFAULT_STYLE['label_size'])
    
    if normalize:
        ax.set_ylabel('Normalized IRF Amplitude', fontsize=DEFAULT_STYLE['label_size'])
    else:
        ax.set_ylabel('IRF Amplitude', fontsize=DEFAULT_STYLE['label_size'])
    
    if log_scale:
        ax.set_yscale('log')
    
    if show_legend:
        ax.legend(fontsize=DEFAULT_STYLE['font_size'])
    
    ax.grid(True, alpha=DEFAULT_STYLE['grid_alpha'])
    
    if title is None:
        title = 'Instrument Response Function Comparison'
    ax.set_title(title, fontsize=DEFAULT_STYLE['title_size'])
    
    # Tight layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig



def plot_separability_analysis(tau1: float,
                               tau2_range: np.ndarray,
                               f_values: np.ndarray,
                               threshold: float = 2.0,
                               params: Optional[Dict] = None,
                               show_threshold: bool = True,
                               title: Optional[str] = None,
                               save_path: Optional[str] = None,
                               **kwargs) -> plt.Figure:
    """
    Create separability analysis plot for biochemical resolution.
    
    Analyzes the ability to distinguish between two fluorescence lifetimes
    (e.g., different biochemical species) based on Fisher information.
    The separability is quantified by the difference in F-values normalized
    by their uncertainties.
    
    Biochemical Context:
    In FLIM, different molecular species or environmental conditions produce
    different fluorescence lifetimes. The ability to distinguish between
    two lifetimes (separability) is crucial for identifying distinct
    biochemical populations or states.
    
    Args:
        tau1: Reference fluorescence lifetime (ns)
        tau2_range: Array of comparison lifetimes (ns)
        f_values: Array of F-values or separability measures
        threshold: Threshold value for "resolvable" lifetimes (default: 2.0)
        params: Optional dictionary with analysis parameters
        show_threshold: Whether to show threshold line
        title: Custom title for the plot
        save_path: Path to save the figure
        **kwargs: Additional matplotlib parameters
        
    Returns:
        matplotlib Figure object
        
    Raises:
        ValidationError: If input data is invalid
        
    Notes:
        The separability criterion is typically based on the Cramér-Rao bound:
        Two lifetimes are considered resolvable if their difference exceeds
        a threshold number of standard deviations (typically 2-3σ).
    """
    if not isinstance(tau2_range, np.ndarray):
        tau2_range = np.array(tau2_range)
    
    if not isinstance(f_values, np.ndarray):
        f_values = np.array(f_values)
    
    if len(tau2_range) != len(f_values):
        raise ValidationError("tau2_range and f_values must have the same length")
    
    if tau1 <= 0:
        raise ValidationError("tau1 must be positive")
    
    # Set up plot style
    setup_plot_style()
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(DEFAULT_STYLE['figure_size'][0], 
                                                   DEFAULT_STYLE['figure_size'][1] * 1.2))
    
    # Top plot: Separability vs lifetime ratio
    tau_ratio = tau2_range / tau1
    
    ax1.plot(tau_ratio, f_values, color=DEFAULT_COLORS['primary'],
            linewidth=DEFAULT_STYLE['line_width'], **kwargs)
    
    # Add threshold line if requested
    if show_threshold:
        ax1.axhline(y=threshold, color=DEFAULT_COLORS['warning'], 
                   linestyle='--', linewidth=2, label=f'Threshold ({threshold}σ)')
        
        # Shade resolvable region
        ax1.fill_between(tau_ratio, threshold, f_values.max() * 1.1,
                        where=(f_values >= threshold), alpha=0.2,
                        color=DEFAULT_COLORS['success'], label='Resolvable')
        
        # Shade non-resolvable region
        ax1.fill_between(tau_ratio, 0, threshold,
                        alpha=0.2, color=DEFAULT_COLORS['warning'],
                        label='Not Resolvable')
    
    ax1.set_xlabel('Lifetime Ratio (τ₂/τ₁)', fontsize=DEFAULT_STYLE['label_size'])
    ax1.set_ylabel('Separability (σ units)', fontsize=DEFAULT_STYLE['label_size'])
    ax1.set_title(f'Separability Analysis - Reference τ₁ = {tau1:.2f} ns',
                 fontsize=DEFAULT_STYLE['title_size'])
    ax1.grid(True, alpha=DEFAULT_STYLE['grid_alpha'])
    ax1.legend()
    
    # Add vertical line at tau_ratio = 1
    ax1.axvline(x=1.0, color='black', linestyle=':', alpha=0.5)
    
    # Bottom plot: Absolute lifetime difference
    tau_diff = np.abs(tau2_range - tau1)
    
    ax2.plot(tau_diff, f_values, color=DEFAULT_COLORS['accent'],
            linewidth=DEFAULT_STYLE['line_width'], **kwargs)
    
    if show_threshold:
        ax2.axhline(y=threshold, color=DEFAULT_COLORS['warning'],
                   linestyle='--', linewidth=2)
        ax2.fill_between(tau_diff, threshold, f_values.max() * 1.1,
                        where=(f_values >= threshold), alpha=0.2,
                        color=DEFAULT_COLORS['success'])
    
    ax2.set_xlabel('Lifetime Difference |τ₂ - τ₁| (ns)', fontsize=DEFAULT_STYLE['label_size'])
    ax2.set_ylabel('Separability (σ units)', fontsize=DEFAULT_STYLE['label_size'])
    ax2.grid(True, alpha=DEFAULT_STYLE['grid_alpha'])
    
    # Add parameter information if provided
    if params:
        info_text = []
        if 'time_bins' in params:
            info_text.append(f"Time bins: {params['time_bins']}")
        if 'repetition_period' in params:
            info_text.append(f"Period: {params['repetition_period']} ns")
        if 'irf_sigma' in params:
            info_text.append(f"IRF σ: {params['irf_sigma']} ns")
        
        if info_text:
            ax2.text(0.98, 0.98, '\n'.join(info_text),
                    transform=ax2.transAxes, fontsize=10,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    if title:
        fig.suptitle(title, fontsize=DEFAULT_STYLE['title_size'] + 2)
    
    # Tight layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_resolving_power(tau_range: np.ndarray,
                        min_resolvable_diff: np.ndarray,
                        params: Optional[Dict] = None,
                        comparison_data: Optional[Dict] = None,
                        log_scale: bool = False,
                        title: Optional[str] = None,
                        save_path: Optional[str] = None,
                        **kwargs) -> plt.Figure:
    """
    Plot resolving power analysis showing minimum resolvable lifetime differences.
    
    The resolving power quantifies the minimum lifetime difference that can
    be reliably distinguished at each reference lifetime value. This is
    critical for experimental design and understanding the limitations of
    the measurement system.
    
    Biochemical Context:
    In biological systems, fluorophores often have similar lifetimes
    (e.g., NADH free vs bound: ~0.4 ns vs ~2.5 ns). Understanding the
    resolving power helps determine if the measurement system can
    distinguish between different molecular states or species.
    
    Args:
        tau_range: Array of reference lifetime values (ns)
        min_resolvable_diff: Minimum resolvable lifetime difference at each tau (ns)
        params: Optional dictionary with measurement parameters
        comparison_data: Optional dict with additional datasets for comparison
                        Format: {'label': {'tau': array, 'diff': array}, ...}
        log_scale: Whether to use logarithmic scale
        title: Custom title for the plot
        save_path: Path to save the figure
        **kwargs: Additional matplotlib parameters
        
    Returns:
        matplotlib Figure object
        
    Raises:
        ValidationError: If input data is invalid
        
    Notes:
        The resolving power is typically defined as the minimum Δτ such that
        |τ₁ - τ₂| / σ(τ) ≥ threshold (usually 2-3), where σ(τ) is derived
        from the Cramér-Rao bound (Fisher information).
    """
    if not isinstance(tau_range, np.ndarray):
        tau_range = np.array(tau_range)
    
    if not isinstance(min_resolvable_diff, np.ndarray):
        min_resolvable_diff = np.array(min_resolvable_diff)
    
    if len(tau_range) != len(min_resolvable_diff):
        raise ValidationError("tau_range and min_resolvable_diff must have the same length")
    
    # Set up plot style
    setup_plot_style()
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left plot: Absolute resolving power
    ax1.plot(tau_range, min_resolvable_diff, color=DEFAULT_COLORS['primary'],
            linewidth=DEFAULT_STYLE['line_width'], marker='o',
            markersize=DEFAULT_STYLE['marker_size'], label='Primary', **kwargs)
    
    # Add comparison data if provided
    if comparison_data:
        colors = [DEFAULT_COLORS['secondary'], DEFAULT_COLORS['accent'], 
                 DEFAULT_COLORS['info'], DEFAULT_COLORS['success']]
        markers = ['s', '^', 'D', 'v']
        
        for i, (label, data) in enumerate(comparison_data.items()):
            if 'tau' in data and 'diff' in data:
                color = colors[i % len(colors)]
                marker = markers[i % len(markers)]
                ax1.plot(data['tau'], data['diff'], color=color,
                        linewidth=DEFAULT_STYLE['line_width'], marker=marker,
                        markersize=DEFAULT_STYLE['marker_size'], label=label)
    
    ax1.set_xlabel('Reference Lifetime τ (ns)', fontsize=DEFAULT_STYLE['label_size'])
    ax1.set_ylabel('Min Resolvable Δτ (ns)', fontsize=DEFAULT_STYLE['label_size'])
    ax1.set_title('Absolute Resolving Power', fontsize=DEFAULT_STYLE['title_size'])
    ax1.grid(True, alpha=DEFAULT_STYLE['grid_alpha'])
    
    if log_scale:
        ax1.set_xscale('log')
        ax1.set_yscale('log')
    
    if comparison_data or params:
        ax1.legend()
    
    # Right plot: Relative resolving power (Δτ/τ)
    relative_resolution = min_resolvable_diff / tau_range
    
    ax2.plot(tau_range, relative_resolution * 100, color=DEFAULT_COLORS['accent'],
            linewidth=DEFAULT_STYLE['line_width'], marker='o',
            markersize=DEFAULT_STYLE['marker_size'], label='Primary', **kwargs)
    
    # Add comparison data
    if comparison_data:
        colors = [DEFAULT_COLORS['secondary'], DEFAULT_COLORS['accent'],
                 DEFAULT_COLORS['info'], DEFAULT_COLORS['success']]
        markers = ['s', '^', 'D', 'v']
        
        for i, (label, data) in enumerate(comparison_data.items()):
            if 'tau' in data and 'diff' in data:
                color = colors[i % len(colors)]
                marker = markers[i % len(markers)]
                rel_res = data['diff'] / data['tau'] * 100
                ax2.plot(data['tau'], rel_res, color=color,
                        linewidth=DEFAULT_STYLE['line_width'], marker=marker,
                        markersize=DEFAULT_STYLE['marker_size'], label=label)
    
    ax2.set_xlabel('Reference Lifetime τ (ns)', fontsize=DEFAULT_STYLE['label_size'])
    ax2.set_ylabel('Relative Resolution Δτ/τ (%)', fontsize=DEFAULT_STYLE['label_size'])
    ax2.set_title('Relative Resolving Power', fontsize=DEFAULT_STYLE['title_size'])
    ax2.grid(True, alpha=DEFAULT_STYLE['grid_alpha'])
    
    if log_scale:
        ax2.set_xscale('log')
    
    if comparison_data:
        ax2.legend()
    
    # Add parameter information if provided
    if params:
        info_text = []
        if 'time_bins' in params:
            info_text.append(f"Time bins: {params['time_bins']}")
        if 'repetition_period' in params:
            info_text.append(f"Period: {params['repetition_period']} ns")
        if 'irf_sigma' in params:
            info_text.append(f"IRF σ: {params['irf_sigma']} ns")
        if 'threshold' in params:
            info_text.append(f"Threshold: {params['threshold']}σ")
        
        if info_text:
            ax2.text(0.98, 0.02, '\n'.join(info_text),
                    transform=ax2.transAxes, fontsize=10,
                    verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    if title is None:
        title = 'Resolving Power Analysis'
    fig.suptitle(title, fontsize=DEFAULT_STYLE['title_size'] + 2)
    
    # Tight layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_loss_analysis(tau_range: np.ndarray,
                      f_values_ideal: np.ndarray,
                      f_values_actual: np.ndarray,
                      irf_label: str = 'Actual IRF',
                      params: Optional[Dict] = None,
                      show_percentage: bool = True,
                      title: Optional[str] = None,
                      save_path: Optional[str] = None,
                      **kwargs) -> plt.Figure:
    """
    Create loss analysis plot comparing ideal vs actual IRF performance.
    
    Analyzes the information loss due to non-ideal IRF characteristics
    (e.g., finite width, shape effects). This helps quantify the degradation
    in measurement precision compared to an ideal (Dirac delta) IRF.
    
    Biochemical Context:
    Real measurement systems have finite IRF widths due to detector response,
    electronics, and optical effects. Understanding the information loss helps
    optimize system design and set realistic expectations for measurement
    precision in biological applications.
    
    Args:
        tau_range: Array of lifetime values (ns)
        f_values_ideal: F-values for ideal (Dirac) IRF
        f_values_actual: F-values for actual IRF
        irf_label: Label for the actual IRF condition
        params: Optional dictionary with IRF parameters
        show_percentage: Whether to show percentage loss
        title: Custom title for the plot
        save_path: Path to save the figure
        **kwargs: Additional matplotlib parameters
        
    Returns:
        matplotlib Figure object
        
    Raises:
        ValidationError: If input data is invalid
        
    Notes:
        Information loss is quantified as the ratio of F-values:
        Loss = 1 - (F_actual / F_ideal)
        This represents the fractional increase in measurement uncertainty
        due to non-ideal IRF characteristics.
    """
    if not isinstance(tau_range, np.ndarray):
        tau_range = np.array(tau_range)
    
    if not isinstance(f_values_ideal, np.ndarray):
        f_values_ideal = np.array(f_values_ideal)
    
    if not isinstance(f_values_actual, np.ndarray):
        f_values_actual = np.array(f_values_actual)
    
    if not (len(tau_range) == len(f_values_ideal) == len(f_values_actual)):
        raise ValidationError("All input arrays must have the same length")
    
    # Set up plot style
    setup_plot_style()
    
    # Create figure with three subplots
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, :])
    ax3 = fig.add_subplot(gs[2, 0])
    ax4 = fig.add_subplot(gs[2, 1])
    
    # Top plot: Direct comparison of F-values
    ax1.plot(tau_range, f_values_ideal, color=DEFAULT_COLORS['primary'],
            linewidth=DEFAULT_STYLE['line_width'], marker='o',
            markersize=DEFAULT_STYLE['marker_size'], label='Ideal (Dirac IRF)', **kwargs)
    ax1.plot(tau_range, f_values_actual, color=DEFAULT_COLORS['secondary'],
            linewidth=DEFAULT_STYLE['line_width'], marker='s',
            markersize=DEFAULT_STYLE['marker_size'], label=irf_label, **kwargs)
    
    ax1.set_xlabel('Fluorescence Lifetime τ (ns)', fontsize=DEFAULT_STYLE['label_size'])
    ax1.set_ylabel('F-value (Precision)', fontsize=DEFAULT_STYLE['label_size'])
    ax1.set_title('F-value Comparison', fontsize=DEFAULT_STYLE['title_size'])
    ax1.set_yscale('log')
    ax1.grid(True, alpha=DEFAULT_STYLE['grid_alpha'])
    ax1.legend()
    
    # Middle plot: Information loss
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        loss_ratio = f_values_actual / f_values_ideal
        loss_ratio = np.where(f_values_ideal > 0, loss_ratio, np.nan)
    
    if show_percentage:
        loss_percentage = (1 - loss_ratio) * 100
        ax2.plot(tau_range, loss_percentage, color=DEFAULT_COLORS['warning'],
                linewidth=DEFAULT_STYLE['line_width'], marker='o',
                markersize=DEFAULT_STYLE['marker_size'], **kwargs)
        ax2.set_ylabel('Information Loss (%)', fontsize=DEFAULT_STYLE['label_size'])
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Shade loss region
        ax2.fill_between(tau_range, 0, loss_percentage, where=(loss_percentage > 0),
                        alpha=0.3, color=DEFAULT_COLORS['warning'])
    else:
        ax2.plot(tau_range, loss_ratio, color=DEFAULT_COLORS['accent'],
                linewidth=DEFAULT_STYLE['line_width'], marker='o',
                markersize=DEFAULT_STYLE['marker_size'], **kwargs)
        ax2.set_ylabel('F-value Ratio (Actual/Ideal)', fontsize=DEFAULT_STYLE['label_size'])
        ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='No Loss')
        ax2.legend()
    
    ax2.set_xlabel('Fluorescence Lifetime τ (ns)', fontsize=DEFAULT_STYLE['label_size'])
    ax2.set_title('Information Loss Analysis', fontsize=DEFAULT_STYLE['title_size'])
    ax2.grid(True, alpha=DEFAULT_STYLE['grid_alpha'])
    
    # Bottom left: Precision degradation factor
    with np.errstate(divide='ignore', invalid='ignore'):
        precision_factor = np.sqrt(1 / loss_ratio)
        precision_factor = np.where(np.isfinite(precision_factor), precision_factor, np.nan)
    
    ax3.plot(tau_range, precision_factor, color=DEFAULT_COLORS['info'],
            linewidth=DEFAULT_STYLE['line_width'], marker='o',
            markersize=DEFAULT_STYLE['marker_size'], **kwargs)
    ax3.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Fluorescence Lifetime τ (ns)', fontsize=DEFAULT_STYLE['label_size'])
    ax3.set_ylabel('Precision Degradation Factor', fontsize=DEFAULT_STYLE['label_size'])
    ax3.set_title('Uncertainty Increase', fontsize=DEFAULT_STYLE['title_size'])
    ax3.grid(True, alpha=DEFAULT_STYLE['grid_alpha'])
    
    # Add annotation
    ax3.text(0.98, 0.98, 'Factor = √(F_ideal/F_actual)\n= σ_actual/σ_ideal',
            transform=ax3.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # Bottom right: Summary statistics
    ax4.axis('off')
    
    # Calculate summary statistics
    mean_loss = np.nanmean((1 - loss_ratio) * 100)
    max_loss = np.nanmax((1 - loss_ratio) * 100)
    min_loss = np.nanmin((1 - loss_ratio) * 100)
    mean_precision_factor = np.nanmean(precision_factor)
    
    summary_text = f"""
    Summary Statistics:
    
    Mean Information Loss: {mean_loss:.2f}%
    Max Information Loss: {max_loss:.2f}%
    Min Information Loss: {min_loss:.2f}%
    
    Mean Precision Factor: {mean_precision_factor:.3f}×
    
    Interpretation:
    • Loss > 0: Actual IRF reduces precision
    • Factor > 1: Increased uncertainty
    • Factor = 1: No degradation (ideal)
    """
    
    if params:
        summary_text += f"\n    Measurement Parameters:\n"
        if 'time_bins' in params:
            summary_text += f"    • Time bins: {params['time_bins']}\n"
        if 'repetition_period' in params:
            summary_text += f"    • Period: {params['repetition_period']} ns\n"
        if 'irf_sigma' in params:
            summary_text += f"    • IRF σ: {params['irf_sigma']} ns\n"
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
            family='monospace')
    
    if title is None:
        title = f'Information Loss Analysis: {irf_label} vs Ideal IRF'
    fig.suptitle(title, fontsize=DEFAULT_STYLE['title_size'] + 2)
    
    # Save if requested
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_comparison_grid(data_dict: Dict[str, AnalysisResults],
                        plot_type: str = 'heatmap',
                        tau_index: Optional[int] = None,
                        log_scale: bool = True,
                        title: Optional[str] = None,
                        save_path: Optional[str] = None,
                        **kwargs) -> plt.Figure:
    """
    Create grid comparison plot for multiple analysis conditions.
    
    Generates a grid of subplots comparing different IRF conditions,
    parameter settings, or analysis methods side-by-side for easy
    visual comparison.
    
    Args:
        data_dict: Dictionary with condition labels as keys and AnalysisResults as values
                  Example: {'Dirac': results1, 'Gaussian σ=0.1': results2, ...}
        plot_type: Type of plot ('heatmap', 'line', 'contour')
        tau_index: Index of lifetime for line plots
        log_scale: Whether to use logarithmic scale
        title: Custom title for the plot
        save_path: Path to save the figure
        **kwargs: Additional matplotlib parameters
        
    Returns:
        matplotlib Figure object
        
    Raises:
        ValidationError: If input data is invalid
    """
    if not isinstance(data_dict, dict):
        raise ValidationError("data_dict must be a dictionary")
    
    if len(data_dict) == 0:
        raise ValidationError("data_dict cannot be empty")
    
    # Set up plot style
    setup_plot_style()
    
    # Determine grid layout
    n_plots = len(data_dict)
    n_cols = min(3, n_plots)
    n_rows = int(np.ceil(n_plots / n_cols))
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Plot each condition
    for i, (label, results) in enumerate(data_dict.items()):
        ax = axes[i]
        
        if not isinstance(results, AnalysisResults):
            warnings.warn(f"Skipping '{label}': not an AnalysisResults object")
            continue
        
        f_values = results.f_values
        params = results.parameters
        
        if plot_type == 'heatmap' and f_values.ndim == 2:
            tau_grid, bins_grid = np.meshgrid(params.lifetimes, params.time_bins, indexing='ij')
            
            if log_scale:
                im = ax.contourf(tau_grid, bins_grid, f_values, levels=15, cmap='viridis', **kwargs)
                ax.set_yscale('log')
            else:
                im = ax.contourf(tau_grid, bins_grid, f_values, levels=15, cmap='viridis', **kwargs)
            
            plt.colorbar(im, ax=ax, label='F-value')
            ax.set_xlabel('Lifetime τ (ns)')
            ax.set_ylabel('Time Bins')
            
        elif plot_type == 'line' and tau_index is not None:
            if f_values.ndim >= 2:
                line_data = f_values[tau_index, :]
                ax.plot(params.time_bins, line_data, 'o-',
                       linewidth=DEFAULT_STYLE['line_width'],
                       markersize=DEFAULT_STYLE['marker_size'])
                
                if log_scale:
                    ax.set_xscale('log')
                    ax.set_yscale('log')
                
                ax.set_xlabel('Time Bins')
                ax.set_ylabel('F-value')
                ax.grid(True, alpha=DEFAULT_STYLE['grid_alpha'])
        
        ax.set_title(label, fontsize=DEFAULT_STYLE['title_size'])
    
    # Hide unused subplots
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)
    
    if title is None:
        title = 'Comparison Grid Analysis'
    fig.suptitle(title, fontsize=DEFAULT_STYLE['title_size'] + 2)
    
    # Tight layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
