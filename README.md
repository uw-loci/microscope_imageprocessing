# Microscope ImageProcessing

General-purpose microscopy imaging utilities -- debayering, background correction, OME-TIFF I/O, Z-stack projections, and focus metrics.

> **Part of the [QPSC (QuPath Scope Control)](https://github.com/uw-loci/qupath-extension-qpsc) system.**
> For complete installation and setup instructions, see the [QPSC Installation Guide](https://github.com/uw-loci/qupath-extension-qpsc/blob/main/documentation/INSTALLATION.md).
>
> This package was extracted from `ppm_library` to provide modality-independent imaging
> utilities that can be shared across microscopy packages without pulling in PPM-specific
> dependencies.

## Features

- **Debayering**: CPU-based Bayer pattern demosaicing (RGGB, GRBG, GBRG, BGGR), with optional GPU acceleration via CuPy
- **Background Correction**: Flat-field correction and background color estimation using histogram mode analysis
- **OME-TIFF I/O**: Standards-compliant OME-TIFF writing with resolution metadata
- **Z-stack Projections**: Max, min, mean, sum, and standard deviation intensity projections for Z-stack reduction
- **Focus Metrics**: Autofocus quality metrics and tissue detection (placeholder for future expansion)

## Installation

**Requirements:**
- Python 3.10 or later
- pip (Python package installer)
- Git (for `pip install git+https://...` commands)

### Quick Install (from GitHub)

```bash
pip install git+https://github.com/uw-loci/microscope_imageprocessing.git
```

### With GPU Support (optional)

```bash
pip install "microscope-imageprocessing[gpu] @ git+https://github.com/uw-loci/microscope_imageprocessing.git"
```

This installs CuPy for GPU-accelerated debayering.

### Development Install (editable mode)

```bash
git clone https://github.com/uw-loci/microscope_imageprocessing.git
cd microscope_imageprocessing
pip install -e ".[dev]"
```

## Quick Start

### Debayering

```python
from microscope_imageprocessing import CPUDebayer

# Create debayerer for your camera's Bayer pattern
debayer = CPUDebayer(pattern='RGGB')

# Convert raw Bayer image to RGB
rgb_image = debayer.debayer(bayer_image)
```

### Background Correction

```python
from microscope_imageprocessing import BackgroundCorrectionUtils

# Estimate background color from histogram mode
bg_color, confidence = BackgroundCorrectionUtils.calculate_background_color_from_mode(image)

# Apply flat-field correction
corrected = BackgroundCorrectionUtils.apply_flatfield(raw_image, background_image)
```

### OME-TIFF Writing

```python
from microscope_imageprocessing import ome_tiff_writer

# Write image with pixel size metadata
ome_tiff_writer(
    filename="output.ome.tif",
    pixel_size_um=0.325,
    data=image_array,
    compression="lzw",
)
```

### Z-stack Projections

```python
from microscope_imageprocessing.zstack import (
    max_intensity_projection,
    mean_projection,
    get_projection,
    generate_z_offsets,
)

# Maximum intensity projection (fluorescence/SHG)
mip = max_intensity_projection(z_stack_list)

# Mean projection (noise reduction)
avg = mean_projection(z_stack_list)

# Get projection by name (useful for config-driven pipelines)
proj = get_projection("max")
result = proj(z_stack_list)

# Generate Z offsets for acquisition
offsets = generate_z_offsets(n_planes=5, step_um=2.0)
# -> [-4.0, -2.0, 0.0, 2.0, 4.0]
```

## Module Reference

### `microscope_imageprocessing.debayering` - Bayer Demosaicing
- `CPUDebayer` - CPU-based bilinear interpolation for Bayer patterns (RGGB, GRBG, GBRG, BGGR)
- `GPUDebayer` - GPU-accelerated debayering via CuPy (requires `[gpu]` extra)

### `microscope_imageprocessing.correction` - Background Correction
- `BackgroundCorrectionUtils` - Flat-field correction and background color estimation

### `microscope_imageprocessing.io` - Image I/O
- `ome_tiff_writer()` - Write OME-TIFF files with pixel size and resolution metadata

### `microscope_imageprocessing.zstack` - Z-stack Projections
- `max_intensity_projection()` - Maximum intensity (fluorescence, SHG)
- `min_intensity_projection()` - Minimum intensity (absorption/transmitted light)
- `mean_projection()` - Mean across Z planes (noise reduction)
- `sum_projection()` - Sum with overflow protection
- `std_projection()` - Standard deviation
- `get_projection(name)` - Look up projection function by name string
- `generate_z_offsets(n_planes, step_um)` - Compute symmetric Z offsets for acquisition

### `microscope_imageprocessing.focus` - Focus Metrics
- Placeholder for autofocus quality metrics and tissue detection utilities

## Dependency Chain

This package sits at the base of the QPSC Python dependency chain:

```
microscope_imageprocessing   (standalone - no hardware or modality deps)
        |
        +-- ppm_library            (adds PPM-specific analysis)
        +-- microscope_control     (adds hardware abstraction via Pycromanager)
        |
        +-- microscope_command_server  (orchestration server)
                depends on: microscope_imageprocessing (required)
                            microscope_control (required)
                            ppm_library (optional, for PPM modality)
```

## Testing

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with coverage
pytest --cov=microscope_imageprocessing --cov-report=html
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Authors

- Mike Nelson (msnelson8@wisc.edu)

## AI-Assisted Development

This project was developed with assistance from [Claude](https://claude.ai) (Anthropic). Claude was used as a development tool for code generation, architecture design, debugging, and documentation throughout the project.
