"""
Test script for plot customization and export capabilities.

This script tests the new functionality added in task 5.3:
- Color scheme management
- Figure export in multiple formats
- Plot annotation and labeling functions
"""

import numpy as np
import matplotlib.pyplot as plt
import os
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


def test_color_schemes():
    """Test color scheme functionality."""
    print("Testing color schemes...")
    
    # Test getting different color schemes
    schemes = ['default', 'viridis', 'publication', 'colorblind', 'grayscale']
    
    for scheme in schemes:
        colors = get_color_scheme(scheme)
        print(f"  {scheme}: {len(colors)} colors")
        assert len(colors) == 7, f"Expected 7 colors in {scheme}"
    
    # Test applying color scheme
    apply_color_scheme('publication')
    print("  Applied 'publication' color scheme")
    
    # Reset to default
    apply_color_scheme('default')
    
    print("✓ Color scheme tests passed\n")


def test_export_formats():
    """Test figure export in multiple formats."""
    print("Testing figure export...")
    
    # Create a simple test figure
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    ax.plot(x, y, linewidth=2)
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Signal')
    ax.set_title('Test Plot')
    ax.grid(True, alpha=0.3)
    
    # Test single format export
    saved = export_figure(fig, 'test_export_single', formats='png')
    print(f"  Single format: {saved}")
    assert len(saved) == 1
    assert os.path.exists(saved[0])
    
    # Test multiple format export
    saved = export_figure(fig, 'test_export_multi', formats=['pdf', 'png', 'svg'])
    print(f"  Multiple formats: {saved}")
    assert len(saved) == 3
    for filepath in saved:
        assert os.path.exists(filepath)
    
    # Test with custom parameters
    saved = export_figure(fig, 'test_export_custom', formats='png', 
                         dpi=150, transparent=True)
    print(f"  Custom parameters: {saved}")
    assert os.path.exists(saved[0])
    
    plt.close(fig)
    
    # Clean up test files
    for ext in ['png', 'pdf', 'svg']:
        for prefix in ['test_export_single', 'test_export_multi', 'test_export_custom']:
            filepath = f'{prefix}.{ext}'
            if os.path.exists(filepath):
                os.remove(filepath)
    
    print("✓ Export format tests passed\n")


def test_annotations():
    """Test annotation functions."""
    print("Testing annotations...")
    
    # Create test figure
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.linspace(0, 10, 100)
    y = np.exp(-x/3) * np.sin(x)
    ax.plot(x, y, linewidth=2)
    
    # Test text annotation
    annotation = add_text_annotation(
        ax, 'Peak', (1.5, 0.3),
        arrow_props={'arrowstyle': '->', 'color': 'red'}
    )
    print("  Added text annotation with arrow")
    assert annotation is not None
    
    # Test annotation in axes coordinates
    annotation2 = add_text_annotation(
        ax, 'Note', (0.5, 0.95), coords='axes',
        bbox_style={'boxstyle': 'round', 'facecolor': 'yellow', 'alpha': 0.5}
    )
    print("  Added annotation in axes coordinates")
    assert annotation2 is not None
    
    # Test parameter box
    params = {
        'τ': '2.5 ns',
        'T': '25 ns',
        'bins': 512,
        'IRF': 'Gaussian'
    }
    param_box = add_parameter_box(ax, params, position='upper right')
    print("  Added parameter box")
    assert param_box is not None
    
    # Test scale bar
    scale_artists = add_scale_bar(ax, length=2, label='2 ns', 
                                  position=(0.7, 0.1), coords='axes')
    print("  Added scale bar")
    assert len(scale_artists) == 2  # line and text
    
    plt.close(fig)
    
    print("✓ Annotation tests passed\n")


def test_custom_colormap():
    """Test custom colormap creation."""
    print("Testing custom colormap...")
    
    # Create custom colormap
    colors = ['#000000', '#ff0000', '#ffff00', '#ffffff']
    cmap = create_custom_colormap(colors, name='test_cmap')
    print(f"  Created colormap with {len(colors)} colors")
    assert cmap is not None
    assert cmap.N == 256  # default n_bins
    
    # Test with named colors
    cmap2 = create_custom_colormap(['blue', 'white', 'red'], name='bwr_custom', n_bins=128)
    print("  Created colormap from named colors")
    assert cmap2.N == 128
    
    # Create a test plot using the colormap
    fig, ax = plt.subplots()
    data = np.random.rand(10, 10)
    im = ax.imshow(data, cmap=cmap)
    plt.colorbar(im, ax=ax)
    plt.close(fig)
    
    print("✓ Custom colormap tests passed\n")


def test_metadata_export():
    """Test saving plots with metadata."""
    print("Testing metadata export...")
    
    # Create test figure
    fig, ax = plt.subplots()
    x = np.linspace(0, 5, 50)
    y = x**2
    ax.plot(x, y)
    ax.set_title('Test Plot with Metadata')
    
    # Test metadata export
    metadata = {
        'analysis_type': 'Fisher Information',
        'date': '2024-01-15',
        'tau': 2.5,
        'repetition_period': 25,
        'time_bins': 512
    }
    
    saved = save_plot_with_metadata(
        fig, 'test_metadata', 
        metadata=metadata,
        formats=['pdf', 'png']
    )
    print(f"  Saved with metadata: {saved}")
    assert len(saved) == 2
    
    for filepath in saved:
        assert os.path.exists(filepath)
        os.remove(filepath)
    
    plt.close(fig)
    
    print("✓ Metadata export tests passed\n")


def test_color_scheme_application():
    """Test applying different color schemes to plots."""
    print("Testing color scheme application...")
    
    schemes = ['default', 'publication', 'colorblind']
    
    for scheme in schemes:
        apply_color_scheme(scheme)
        
        fig, ax = plt.subplots()
        x = np.linspace(0, 10, 100)
        
        # Get current colors
        from visualization import DEFAULT_COLORS
        
        # Plot with different colors from scheme
        ax.plot(x, np.sin(x), color=DEFAULT_COLORS['primary'], label='Primary')
        ax.plot(x, np.cos(x), color=DEFAULT_COLORS['secondary'], label='Secondary')
        ax.plot(x, np.sin(x+1), color=DEFAULT_COLORS['accent'], label='Accent')
        
        ax.legend()
        ax.set_title(f'Color Scheme: {scheme}')
        
        plt.close(fig)
        print(f"  Applied and tested '{scheme}' scheme")
    
    # Reset to default
    apply_color_scheme('default')
    
    print("✓ Color scheme application tests passed\n")


def test_comprehensive_plot():
    """Create a comprehensive test plot using all features."""
    print("Creating comprehensive test plot...")
    
    # Set up custom style
    setup_plot_style({'figure_size': (12, 8), 'dpi': 100})
    
    # Apply colorblind-friendly scheme
    apply_color_scheme('colorblind')
    from visualization import DEFAULT_COLORS
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left plot: Line plot with annotations
    x = np.linspace(0, 10, 100)
    y1 = np.exp(-x/3) * np.sin(2*x)
    y2 = np.exp(-x/4) * np.cos(2*x)
    
    ax1.plot(x, y1, color=DEFAULT_COLORS['primary'], linewidth=2, label='Signal 1')
    ax1.plot(x, y2, color=DEFAULT_COLORS['secondary'], linewidth=2, label='Signal 2')
    ax1.set_xlabel('Time (ns)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Fluorescence Decay Signals')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add annotation
    add_text_annotation(ax1, 'Decay region', (5, 0), 
                       arrow_props={'arrowstyle': '->', 'color': 'red'})
    
    # Add parameter box
    params = {'τ₁': '2.5 ns', 'τ₂': '3.5 ns', 'T': '25 ns'}
    add_parameter_box(ax1, params, position='upper right')
    
    # Right plot: 2D heatmap with custom colormap
    data = np.random.rand(20, 20)
    cmap = create_custom_colormap(['blue', 'white', 'red'], name='custom')
    im = ax2.imshow(data, cmap=cmap, aspect='auto')
    ax2.set_xlabel('Time Bin')
    ax2.set_ylabel('Lifetime (ns)')
    ax2.set_title('Fisher Information Heatmap')
    plt.colorbar(im, ax=ax2, label='F-value')
    
    # Add scale bar
    add_scale_bar(ax2, length=5, label='5 bins', 
                 position=(0.7, 0.1), coords='axes')
    
    plt.tight_layout()
    
    # Export in multiple formats
    saved = export_figure(fig, 'test_comprehensive', 
                         formats=['png', 'pdf'], dpi=150)
    print(f"  Saved comprehensive plot: {saved}")
    
    # Also save with metadata
    metadata = {
        'test_type': 'comprehensive',
        'features': 'annotations, color schemes, export',
        'date': '2024-01-15'
    }
    saved_meta = save_plot_with_metadata(fig, 'test_comprehensive_meta',
                                        metadata=metadata, formats='pdf')
    print(f"  Saved with metadata: {saved_meta}")
    
    plt.close(fig)
    
    # Clean up
    for filepath in saved + saved_meta:
        if os.path.exists(filepath):
            os.remove(filepath)
    
    # Reset to default scheme
    apply_color_scheme('default')
    
    print("✓ Comprehensive plot test passed\n")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Plot Customization and Export Capabilities")
    print("=" * 60)
    print()
    
    try:
        test_color_schemes()
        test_export_formats()
        test_annotations()
        test_custom_colormap()
        test_metadata_export()
        test_color_scheme_application()
        test_comprehensive_plot()
        
        print("=" * 60)
        print("All tests passed successfully! ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
