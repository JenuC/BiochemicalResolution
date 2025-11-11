# Plot Customization and Export Guide

This guide documents the plot customization and export capabilities added to the visualization module (Task 5.3).

## Table of Contents

1. [Color Schemes](#color-schemes)
2. [Figure Export](#figure-export)
3. [Annotations and Labels](#annotations-and-labels)
4. [Custom Colormaps](#custom-colormaps)
5. [Examples](#examples)

---

## Color Schemes

The visualization module provides multiple predefined color schemes optimized for different use cases.

### Available Color Schemes

- **default**: Standard matplotlib colors, good for general use
- **viridis**: Perceptually uniform, colorblind-friendly
- **publication**: Black and white friendly, high contrast
- **colorblind**: Optimized for colorblind accessibility
- **grayscale**: For black and white printing

### Usage

```python
from visualization import get_color_scheme, apply_color_scheme

# Get a specific color scheme
colors = get_color_scheme('publication')
plt.plot(x, y, color=colors['primary'])

# Apply a color scheme globally
apply_color_scheme('colorblind')
# All subsequent plots will use colorblind-friendly colors
```

### Color Roles

Each scheme provides these color roles:
- `primary`: Main data series
- `secondary`: Secondary data series
- `accent`: Highlighting important features
- `warning`: Warnings or critical values
- `info`: Informational elements
- `success`: Positive results
- `neutral`: Neutral or background elements

---

## Figure Export

Export figures in multiple formats with consistent quality settings.

### Supported Formats

- **PDF**: Vector format, ideal for publications
- **PNG**: Raster format, good for presentations
- **SVG**: Vector format, web-friendly
- **EPS**: Vector format, legacy publication format
- **JPG**: Compressed raster format
- **TIFF**: High-quality raster format

### Basic Export

```python
from visualization import export_figure

fig, ax = plt.subplots()
ax.plot(x, y)

# Export single format
export_figure(fig, 'myplot', formats='png')
# Creates: myplot.png

# Export multiple formats
export_figure(fig, 'myplot', formats=['pdf', 'png', 'svg'])
# Creates: myplot.pdf, myplot.png, myplot.svg
```

### Advanced Export Options

```python
# High-resolution export
export_figure(fig, 'myplot', formats='png', dpi=600)

# Transparent background
export_figure(fig, 'myplot', formats='png', transparent=True)

# Custom bounding box
export_figure(fig, 'myplot', formats='pdf', bbox_inches='standard')
```

### Export with Metadata

```python
from visualization import save_plot_with_metadata

metadata = {
    'Title': 'Fisher Information Analysis',
    'Author': 'Research Lab',
    'Subject': 'FLIM IRF Analysis',
    'tau': 2.5,
    'repetition_period': 25,
    'time_bins': 512
}

save_plot_with_metadata(fig, 'analysis', metadata=metadata, formats='pdf')
```

**Note**: For PDF format, standard fields (Title, Author, Subject, Keywords) are embedded directly. Custom fields are added to the Keywords field.

---

## Annotations and Labels

Add informative annotations to plots with consistent styling.

### Text Annotations

```python
from visualization import add_text_annotation

# Simple annotation with arrow
add_text_annotation(
    ax, 'Peak value', 
    position=(5, 10),
    arrow_props={'arrowstyle': '->', 'color': 'red'}
)

# Annotation in axes coordinates (0-1 range)
add_text_annotation(
    ax, 'Note: Data normalized', 
    position=(0.5, 0.95),
    coords='axes',
    bbox_style={'boxstyle': 'round', 'facecolor': 'yellow', 'alpha': 0.5}
)
```

### Parameter Boxes

Display analysis parameters directly on plots:

```python
from visualization import add_parameter_box

params = {
    'τ': '2.5 ns',
    'T': '25 ns',
    'bins': 512,
    'IRF': 'Gaussian σ=0.5 ns'
}

add_parameter_box(ax, params, position='upper right')
```

Available positions: `'upper right'`, `'upper left'`, `'lower right'`, `'lower left'`, `'center'`

### Scale Bars

Add scale bars for spatial data:

```python
from visualization import add_scale_bar

# Add scale bar with label
add_scale_bar(ax, length=10, label='10 μm', 
             position=(0.7, 0.1), coords='axes')
```

---

## Custom Colormaps

Create custom colormaps for specialized visualizations.

### Basic Usage

```python
from visualization import create_custom_colormap

# Create colormap from hex colors
cmap = create_custom_colormap(
    ['#000000', '#ff0000', '#ffff00', '#ffffff'],
    name='custom_heat'
)

# Use in plots
plt.imshow(data, cmap=cmap)
```

### Named Colors

```python
# Create colormap from named colors
cmap = create_custom_colormap(
    ['blue', 'white', 'red'],
    name='blue_white_red',
    n_bins=256
)
```

---

## Examples

### Example 1: Publication-Ready Plot

```python
from visualization import (
    apply_color_scheme, export_figure, 
    add_parameter_box, setup_plot_style
)

# Set up publication style
setup_plot_style({'figure_size': (8, 6), 'dpi': 150})
apply_color_scheme('publication')

# Create plot
fig, ax = plt.subplots()
ax.plot(tau_values, f_values, 'o-', linewidth=2)
ax.set_xlabel('Fluorescence Lifetime τ (ns)')
ax.set_ylabel('F-value')
ax.set_title('Fisher Information Analysis')
ax.grid(True, alpha=0.3)

# Add parameters
params = {'T': '25 ns', 'Bins': 512, 'IRF': 'Dirac'}
add_parameter_box(ax, params, position='upper right')

# Export for publication
export_figure(fig, 'figure1', formats=['pdf', 'eps'], dpi=600)
```

### Example 2: Colorblind-Friendly Comparison

```python
from visualization import apply_color_scheme, DEFAULT_COLORS

# Use colorblind-friendly colors
apply_color_scheme('colorblind')

fig, ax = plt.subplots()
ax.plot(x, y1, color=DEFAULT_COLORS['primary'], label='Method A')
ax.plot(x, y2, color=DEFAULT_COLORS['secondary'], label='Method B')
ax.plot(x, y3, color=DEFAULT_COLORS['accent'], label='Method C')
ax.legend()

export_figure(fig, 'comparison', formats='png')
```

### Example 3: Annotated Decay Analysis

```python
from visualization import add_text_annotation, add_parameter_box

fig, ax = plt.subplots()
ax.plot(time, signal, linewidth=2)
ax.set_yscale('log')

# Annotate features
add_text_annotation(
    ax, 'Fast component', 
    position=(2, 0.5),
    arrow_props={'arrowstyle': '->', 'color': 'red'}
)

add_text_annotation(
    ax, 'Slow component', 
    position=(10, 0.1),
    arrow_props={'arrowstyle': '->', 'color': 'blue'}
)

# Add analysis parameters
params = {'τ₁': '2.5 ns', 'τ₂': '8.0 ns', 'A₁': '0.6', 'A₂': '0.4'}
add_parameter_box(ax, params, position='upper right')

export_figure(fig, 'decay_analysis', formats=['png', 'pdf'])
```

### Example 4: Custom Colormap Heatmap

```python
from visualization import create_custom_colormap

# Create custom colormap
cmap = create_custom_colormap(
    ['#000033', '#0066cc', '#00ccff', '#ffff00', '#ff6600'],
    name='custom_fisher'
)

# Create heatmap
fig, ax = plt.subplots()
im = ax.contourf(TAU, BINS, F_values, levels=20, cmap=cmap)
ax.set_xlabel('Lifetime τ (ns)')
ax.set_ylabel('Time Bins')
ax.set_yscale('log')
plt.colorbar(im, ax=ax, label='F-value')

export_figure(fig, 'heatmap', formats=['png', 'svg'])
```

---

## Best Practices

1. **Color Schemes**: Use `colorblind` or `viridis` schemes for accessibility
2. **Export Formats**: 
   - Use PDF/SVG for publications (vector graphics)
   - Use PNG for presentations (raster, widely compatible)
   - Use EPS for legacy journal submissions
3. **Resolution**: Use `dpi=300` minimum for publications, `dpi=600` for high-quality prints
4. **Annotations**: Keep annotations concise and use arrows to point to specific features
5. **Parameter Boxes**: Include key analysis parameters for reproducibility
6. **Metadata**: Always include metadata when exporting for publications

---

## API Reference

### Color Scheme Functions

- `get_color_scheme(scheme_name)`: Get a color scheme dictionary
- `apply_color_scheme(scheme_name)`: Apply a color scheme globally
- `setup_plot_style(style_params)`: Configure plot styling

### Export Functions

- `export_figure(fig, filepath, formats, dpi, transparent, **kwargs)`: Export figure
- `save_plot_with_metadata(fig, filepath, metadata, formats, **kwargs)`: Export with metadata

### Annotation Functions

- `add_text_annotation(ax, text, position, coords, **kwargs)`: Add text annotation
- `add_parameter_box(ax, params, position, **kwargs)`: Add parameter box
- `add_scale_bar(ax, length, label, position, **kwargs)`: Add scale bar

### Colormap Functions

- `create_custom_colormap(colors, name, n_bins)`: Create custom colormap

---

## Testing

Run the test suite to verify functionality:

```bash
python test_plot_customization.py
```

Run the demonstration to see examples:

```bash
python demo_plot_customization.py
```

---

## Requirements

This functionality requires:
- matplotlib >= 3.0
- numpy >= 1.18

See `requirements.txt` for complete dependencies.
