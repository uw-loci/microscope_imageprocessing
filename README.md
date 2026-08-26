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
- **Z-stack Projections**: Max, min, mean, sum, standard deviation, and extended depth of field (EDF) projections for Z-stack reduction
- **Per-Pixel Sharpness Maps**: Local focus measurement at each pixel (Tenengrad, modified Laplacian, variance) for EDF fusion and tilt diagnosis
- **Focus Metrics**: Autofocus quality metrics and tissue detection

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

# Attach arbitrary key/value metadata (emitted as an OME MapAnnotation).
# Use this for handling rules a reader cannot infer from the pixels -- the
# motivating case is an axial orientation map, which must be resampled through
# the doubled angle rather than averaged, and whose value negates under a
# single mirror. Get either wrong and the result looks plausible but is not.
from microscope_imageprocessing.io.ome_writer import StackWriter
import numpy as np

writer = StackWriter(
    "orientation.ome.tif",
    size_t=1, size_z=1, size_c=1, size_y=512, size_x=512,
    dtype=np.dtype("float32"),
    pixel_size_um=0.325,
    channel_names=["Slow Axis Orientation (rad, axial)"],
    map_annotations={
        "polscope.units": "radians",
        "polscope.axial": "true",
        "polscope.resample": "doubled-angle: sin(2t)/cos(2t); NEVER average the angle",
    },
    granularity="single",
)
try:
    writer.write_frame(orientation_rad, t=0, z=0, c=0)
finally:
    writer.close()
```

### Z-stack Projections

```python
from microscope_imageprocessing.zstack import (
    max_intensity_projection,
    mean_projection,
    extended_depth_of_field,
    focus_height_map,
    get_projection,
    make_edf_projection,
    generate_z_offsets,
)

# Maximum intensity projection (fluorescence/SHG)
mip = max_intensity_projection(z_stack_list)

# Mean projection (noise reduction)
avg = mean_projection(z_stack_list)

# Extended depth of field: fuse Z-stack by selecting sharpest plane per pixel
# Use for brightfield and tilted samples where no single plane is in focus everywhere
edf_fused = extended_depth_of_field(z_stack_list)

# Get the per-pixel focal plane index (useful for diagnosing sample tilt)
height_map = focus_height_map(z_stack_list)

# Get projection by name (useful for config-driven pipelines)
proj = get_projection("edf")  # or "max", "min", "mean", etc.
result = proj(z_stack_list)

# Build an EDF projection with custom settings (for registry-driven config).
# The defaults (tenengrad, window=9, index_smooth=5) are reasoned starting
# points, NOT measured optima -- window scales with pixel size and camera
# noise, so tune it: raise it if the fused output looks blocky or speckled,
# lower it if in-focus boundaries look smeared. Raise index_smooth for a
# tilted but flat sample; lower it (or 0) where focus genuinely steps, since
# a large median bridges the step and picks a plane sharp on neither side.
custom_edf = make_edf_projection(metric="variance", window=5, index_smooth=0)
result = custom_edf(z_stack_list)

# Generate Z offsets for acquisition
offsets = generate_z_offsets(z_range_um=8.0, z_step_um=2.0)
# -> [-4.0, -2.0, 0.0, 2.0, 4.0]  (total range, not plane count)
```

### Per-Pixel Sharpness Maps

```python
from microscope_imageprocessing.focus.sharpness_maps import (
    tenengrad_map,
    modified_laplacian_map,
    variance_map,
    resolve_sharpness_map,
)

# Compute per-pixel sharpness using different metrics
# All three return a 2D float array (H, W) of the same shape as the input
tenengrad = tenengrad_map(image)     # Best for edge contrast (stained tissue)
laplacian = modified_laplacian_map(image)  # Sharper peaks in Z, more noise-sensitive
variance = variance_map(image)       # Cheapest, forgiving of noise

# Look up sharpness map by name (used internally by extended_depth_of_field)
metric = resolve_sharpness_map("tenengrad")
```

## Module Reference

### `microscope_imageprocessing.debayering` - Bayer Demosaicing
- `CPUDebayer` - CPU-based bilinear interpolation for Bayer patterns (RGGB, GRBG, GBRG, BGGR)
- `GPUDebayer` - GPU-accelerated debayering via CuPy (requires `[gpu]` extra)

### `microscope_imageprocessing.correction` - Background Correction
- `BackgroundCorrectionUtils` - Flat-field correction and background color estimation

### `microscope_imageprocessing.io` - Image I/O
- `ome_tiff_writer()` - Write OME-TIFF files with pixel size and resolution metadata
- `StackWriter` - Lower-level OME-TIFF writer for frame-by-frame output. Supports optional `map_annotations` dict for embedding arbitrary key/value metadata (emitted as OME MapAnnotations)

### `microscope_imageprocessing.zstack` - Z-stack Projections
- `max_intensity_projection()` - Maximum intensity (fluorescence, SHG)
- `min_intensity_projection()` - Minimum intensity (absorption/transmitted light)
- `mean_projection()` - Mean across Z planes (noise reduction)
- `sum_projection()` - Sum with overflow protection
- `std_projection()` - Standard deviation
- `extended_depth_of_field(stack, metric, ...)` - Focus-aware fusion: select sharpest plane per pixel
- `focus_height_map(stack, metric, ...)` - Per-pixel index of sharpest plane (diagnose tilt)
- `get_projection(name)` - Look up projection function by name string (supports "max", "min", "mean", "sum", "std", "edf")
- `make_edf_projection(metric, window, index_smooth)` - Build a registry-shaped EDF projection with custom settings (for config-driven pipelines with non-default parameters)
- `generate_z_offsets(z_range_um, z_step_um)` - Compute symmetric Z offsets spanning the TOTAL range (so 8.0 gives +/-4.0)

### `microscope_imageprocessing.focus` - Focus Metrics and Autofocus Strategies
- `worst_channel_saturation_fraction()` - Measure pixel saturation (0-1) in the worst channel; used by AF auto-exposure control to prevent metric inversion from clipping
- `to_gray()` - Reduce multi-channel image to grayscale with equal-weighted mean
- **Autofocus Strategies** (`DenseTextureStrategy`, `SparseSignalStrategy`, `DarkFieldStrategy`, `DenseFluorescenceStrategy`, `ManualOnlyStrategy`) - Per-sample-regime focus quality metrics with configurable saturation tolerances:
  - `saturation_acceptable(image)` - Check if image brightness is within strategy's saturation tolerance before trusting focus metrics
  - `saturation_threshold` - Configurable per-strategy: dense tissue/fluorescence ≈10%, sparse signal/dark-field ≈3% (tighter because signal clips in fewer pixels)
  - Other validation methods: `brightness_acceptable()`, `is_valid()`

### `microscope_imageprocessing.focus.sharpness_maps` - Per-Pixel Sharpness Maps
- `tenengrad_map(image, window)` - Squared gradient magnitude; best for edge contrast
- `modified_laplacian_map(image, window)` - Modified Laplacian (Nayar and Nakagawa); sharper Z peaks, noise-sensitive
- `variance_map(image, window)` - Local variance; cheapest, most forgiving
- `resolve_sharpness_map(name)` - Look up map function by canonical name
- `list_sharpness_map_names()` - Available sharpness map names

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
