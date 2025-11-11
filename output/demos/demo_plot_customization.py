"""
Demonstration of Plot Customization and Export Capabilities

This script demonstrates the new features added in task 5.3:
1. Consistent color schemes and styling options
2. Figure export in multiple formats (PDF, PNG, SVG)
3. Plot annotation and labeling functions

Usage:
    python demo_plot_customization.py
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

# Add src to path
sys.path.insert(0, 'src')

from visualization import (
    get_color_scheme,
    apply_color_scheme,
    export_figure,
    add_text_annotation,
    add_parameter_box,
    add_scale_bar,
    create_custom_colormap,
    save_plot_with_metadata,
    setup_plot_style
)


def demo_color_schemes():
    """Demonstrate different color schemes."""
    print("Demonstrating color schemes...")
    
    schemes = ['default', 'publication', 'colorblind', 'viridis']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    x = np.linspace(0, 10, 100)
    
    for i, scheme in enumerate(schemes):
        apply_color_scheme(scheme)
        from visualization import DEFAULT_COLORS
        
        ax = axes[i]
        ax.plot(x, np.sin(x), color=DEFAULT_COLORS['primary'], 
               linewidth=2, label='Primary')
        ax.plot(x, np.cos(x), color=DEFAULT_COLORS['secondary'], 
               linewidth=2, label='Secondary')
        ax.plot(x, np.sin(x+1), color=DEFAULT_COLORS['accent'], 
               linewidth=2, label='Accent')
        
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('Signal')
        ax.set_title(f'{scheme.title()} Color Scheme')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Export in multiple formats
    saved = export_figure(fig, 'demo_color_schemes', 
                         formats=['png', 'pdf'], dpi=150)
    print(f"  Saved: {saved}\n")
    
    plt.close(fig)
    
    # Reset to default
    apply_color_scheme('default')


def demo_annotations():
    """Demonstrate annotation capabilities."""
    print("Demonstrating annotations...")
    
    # Create a realistic FLIM decay plot
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Simulate fluorescence decay
    t = np.linspace(0, 25, 500)
    tau1, tau2 = 2.5, 8.0
    A1, A2 = 0.6, 0.4
    
    # Two-component exponential decay
    signal = A1 * np.exp(-t/tau1) + A2 * np.exp(-t/tau2)
    # Add some noise
    signal += np.random.normal(0, 0.02, len(signal))
    
    # Plot
    apply_color_scheme('publication')
    from visualization import DEFAULT_COLORS
    
    ax.plot(t, signal, color=DEFAULT_COLORS['primary'], 
           linewidth=2, label='Measured Signal')
    ax.plot(t, A1 * np.exp(-t/tau1), '--', 
           color=DEFAULT_COLORS['secondary'], 
           linewidth=1.5, label=f'Component 1 (τ={tau1} ns)')
    ax.plot(t, A2 * np.exp(-t/tau2), '--', 
           color=DEFAULT_COLORS['accent'], 
           linewidth=1.5, label=f'Component 2 (τ={tau2} ns)')
    
    ax.set_xlabel('Time (ns)', fontsize=12)
    ax.set_ylabel('Normalized Intensity', fontsize=12)
    ax.set_title('Two-Component Fluorescence Decay Analysis', fontsize=14)
    ax.set_yscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add annotations
    add_text_annotation(
        ax, 'Fast component\ndominates early', 
        (3, 0.5), coords='data',
        arrow_props={'arrowstyle': '->', 'color': 'red', 'lw': 2},
        fontsize=10
    )
    
    add_text_annotation(
        ax, 'Slow component\nvisible at late times', 
        (15, 0.1), coords='data',
        arrow_props={'arrowstyle': '->', 'color': 'blue', 'lw': 2},
        fontsize=10
    )
    
    # Add parameter box
    params = {
        'τ₁': f'{tau1} ns',
        'τ₂': f'{tau2} ns',
        'A₁': f'{A1:.1f}',
        'A₂': f'{A2:.1f}',
        'T': '25 ns',
        'Bins': 512
    }
    add_parameter_box(ax, params, position='upper right', fontsize=10)
    
    plt.tight_layout()
    
    # Save with metadata
    metadata = {
        'Title': 'Two-Component Fluorescence Decay',
        'Subject': 'FLIM Analysis',
        'Author': 'IRF Analysis Package',
        'tau1': tau1,
        'tau2': tau2,
        'amplitudes': f'{A1},{A2}'
    }
    saved = save_plot_with_metadata(
        fig, 'demo_annotations', 
        metadata=metadata,
        formats=['png', 'pdf']
    )
    print(f"  Saved: {saved}\n")
    
    plt.close(fig)
    apply_color_scheme('default')


def demo_custom_colormap():
    """Demonstrate custom colormap creation."""
    print("Demonstrating custom colormaps...")
    
    # Create Fisher information heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Generate sample data
    tau = np.linspace(0.5, 10, 50)
    bins = 2 ** np.arange(6, 12)
    TAU, BINS = np.meshgrid(tau, bins, indexing='ij')
    
    # Simulate F-values (higher for more bins and intermediate lifetimes)
    F_values = BINS * np.exp(-(TAU - 3)**2 / 10) / 100
    
    # Plot with default colormap
    im1 = ax1.contourf(TAU, BINS, F_values, levels=20, cmap='viridis')
    ax1.set_xlabel('Fluorescence Lifetime τ (ns)')
    ax1.set_ylabel('Number of Time Bins')
    ax1.set_yscale('log')
    ax1.set_title('Default Colormap (Viridis)')
    plt.colorbar(im1, ax=ax1, label='F-value')
    
    # Plot with custom colormap
    custom_cmap = create_custom_colormap(
        ['#000033', '#0066cc', '#00ccff', '#ffff00', '#ff6600'],
        name='custom_fisher'
    )
    im2 = ax2.contourf(TAU, BINS, F_values, levels=20, cmap=custom_cmap)
    ax2.set_xlabel('Fluorescence Lifetime τ (ns)')
    ax2.set_ylabel('Number of Time Bins')
    ax2.set_yscale('log')
    ax2.set_title('Custom Colormap')
    plt.colorbar(im2, ax=ax2, label='F-value')
    
    # Add parameter box
    params = {'T': '25 ns', 'IRF': 'Dirac'}
    add_parameter_box(ax2, params, position='lower right')
    
    plt.tight_layout()
    
    saved = export_figure(fig, 'demo_custom_colormap', 
                         formats=['png', 'svg'], dpi=200)
    print(f"  Saved: {saved}\n")
    
    plt.close(fig)


def demo_export_formats():
    """Demonstrate exporting in multiple formats."""
    print("Demonstrating export formats...")
    
    # Create a publication-ready plot
    setup_plot_style({'figure_size': (8, 6), 'dpi': 150})
    apply_color_scheme('publication')
    from visualization import DEFAULT_COLORS
    
    fig, ax = plt.subplots()
    
    # Plot IRF comparison
    t = np.linspace(-2, 10, 500)
    
    # Dirac (approximated as very narrow Gaussian)
    dirac = np.exp(-(t-0)**2 / (2*0.01**2))
    dirac = dirac / np.trapz(dirac, t)
    
    # Gaussian IRFs with different widths
    sigmas = [0.1, 0.5, 1.0]
    
    ax.plot(t, dirac, color=DEFAULT_COLORS['primary'], 
           linewidth=2.5, label='Dirac (ideal)')
    
    colors = [DEFAULT_COLORS['secondary'], DEFAULT_COLORS['accent'], 
             DEFAULT_COLORS['warning']]
    
    for sigma, color in zip(sigmas, colors):
        gauss = np.exp(-(t-0)**2 / (2*sigma**2))
        gauss = gauss / np.trapz(gauss, t)
        ax.plot(t, gauss, color=color, linewidth=2, 
               label=f'Gaussian σ={sigma} ns')
    
    ax.set_xlabel('Time (ns)', fontsize=12)
    ax.set_ylabel('Normalized IRF', fontsize=12)
    ax.set_title('Instrument Response Function Comparison', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1, 5)
    
    plt.tight_layout()
    
    # Export in all supported formats
    formats = ['png', 'pdf', 'svg', 'eps']
    saved = export_figure(fig, 'demo_irf_comparison', 
                         formats=formats, dpi=300)
    print(f"  Saved in {len(formats)} formats: {saved}\n")
    
    plt.close(fig)
    apply_color_scheme('default')


def main():
    """Run all demonstrations."""
    print("=" * 70)
    print("Plot Customization and Export Capabilities - Demonstration")
    print("=" * 70)
    print()
    
    demo_color_schemes()
    demo_annotations()
    demo_custom_colormap()
    demo_export_formats()
    
    print("=" * 70)
    print("All demonstrations completed!")
    print("Check the generated files:")
    print("  - demo_color_schemes.png/pdf")
    print("  - demo_annotations.png/pdf")
    print("  - demo_custom_colormap.png/svg")
    print("  - demo_irf_comparison.png/pdf/svg/eps")
    print("=" * 70)


if __name__ == '__main__':
    main()
