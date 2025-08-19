# FastMRI Knee Dataset Implementation

This directory contains the implementation of a PyTorch dataset for FastMRI knee single-coil .h5 files, specifically designed for use with the CoSIGN framework for inverse problems in medical imaging.

## Overview

The FastMRI dataset implementation provides:

- **2D Gaussian k-space undersampling** for realistic acceleration patterns
- **Flexible H5 file loading** with support for different keys and shapes
- **Automatic preprocessing** including FFT/IFFT operations and normalization
- **Integration with existing CoSIGN infrastructure** through standardized interfaces
- **Configurable data loading** with support for different image sizes and batch configurations

## Files

- `MRI_datasets_knee_kspace.py` - Main dataset implementation
- `../config/mri_reconstruction_config.yaml` - Example configuration file
- `../examples/fastmri_usage.py` - Usage examples and demos

## Key Features

### 1. Gaussian K-space Undersampling

The `mask_2d_gauss_accel()` function implements 2D Gaussian random sampling that:
- Provides configurable acceleration factors (default 8x)
- Weights k-space center more heavily to reduce aliasing artifacts
- Returns probabilistic sampling masks for realistic undersampling

### 2. FastMRI Dataset Class

The `FastMRIH5SliceDataset` class:
- Automatically discovers all .h5 files in a directory tree
- Loads individual 2D slices from 3D volumes
- Handles cropping/padding to fixed target shapes
- Provides robust error handling for corrupted files

### 3. Data Processing Pipeline

Each sample undergoes:
1. **Loading**: Read reconstruction_rss data from H5 files
2. **Normalization**: Initial normalization to [0, 1] range
3. **FFT**: Forward FFT to k-space domain
4. **Masking**: Apply 2D Gaussian undersampling mask
5. **IFFT**: Inverse FFT back to image domain
6. **Resizing**: Resize to target dimensions using PyTorch transforms
7. **Final Normalization**: Normalize to [-1, 1] range for model compatibility

### 4. Output Format

Each sample returns a dictionary with:
- `lr_img`: Low-resolution (undersampled) image
- `hr_img`: High-resolution (fully-sampled) target image
- `hr_inte`: Intermediate reconstruction (copy of lr_img)
- `gt`: Ground truth (copy of hr_img)
- `mask`: Undersampling mask resized to target dimensions
- `x_init`: Initial estimate (copy of lr_img)

All tensors have shape `[1, H, W]` (single channel) and dtype `float32`.

## Usage

### Basic Usage

```python
from datasets.MRI_datasets_knee_kspace import FastMRIH5SliceDataset

# Create dataset
dataset = FastMRIH5SliceDataset(
    root='/path/to/fastmri/data',
    key='reconstruction_rss',
    target_shape=(256, 256)
)

# Use with DataLoader
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=8, shuffle=True)
```

### Config-based Usage

```python
from datasets.MRI_datasets_knee_kspace import create_dataloader

# With config object
train_loader, val_loader = create_dataloader(
    configs=my_config,
    data_dir='/path/to/data',
    evaluation=False
)
```

### Framework Integration

```python
# Integrated with cc.image_datasets
from cc.image_datasets import load_fastmri

loader_gen = load_fastmri(
    data_dir='/path/to/data',
    batch_size=16,
    image_size=256,
    deterministic=False
)
```

## Configuration

Example configuration (see `config/mri_reconstruction_config.yaml`):

```yaml
data:
  name: fastmri_knee
  root: datasets/fastmri_knee
  image_size: 256
  h5_key: reconstruction_rss
  is_complex: true

measurement:
  operator:
    name: mri
    acceleration: 8
    sigma_frac: 0.3
```

## Data Format Requirements

Your FastMRI .h5 files should contain:
- `reconstruction_rss`: Root sum of squares reconstructions (float32, shape: [N, H, W])
- `kspace`: K-space data (complex64 or separate real/imaginary, shape: [N, H, W] or [N, H, W, 2])

Where N is the number of slices, H and W are spatial dimensions.

## Testing

The implementation includes comprehensive tests:

```bash
# Run basic functionality tests
python /tmp/test_mri_dataset.py

# Run standalone tests (no MPI required)
python /tmp/test_standalone.py
```

## Performance Considerations

- **Memory Usage**: Images are loaded on-demand to minimize memory footprint
- **Preprocessing**: FFT operations are performed in NumPy for efficiency
- **Caching**: No caching is implemented to save memory; consider adding if I/O becomes a bottleneck
- **Multiprocessing**: Supports DataLoader num_workers for parallel loading

## Customization

The dataset can be easily customized by:
- **Changing undersampling patterns**: Modify `mask_2d_gauss_accel()` or implement new masking functions
- **Adjusting preprocessing**: Override the `__getitem__` method for different preprocessing pipelines
- **Supporting new file formats**: Extend the file discovery logic in `__init__`
- **Adding data augmentation**: Integrate with torchvision transforms

## Integration with CoSIGN

This dataset integrates seamlessly with the CoSIGN framework:
- **Measurement operators**: Works with MRI forward operators
- **Consistency models**: Provides properly normalized inputs/outputs
- **Configuration system**: Uses existing YAML configuration format
- **Distributed training**: Supports MPI-based distributed data loading

## License

This implementation follows the same license as the parent CoSIGN project. The original FastMRI data should be used according to Facebook's license terms.