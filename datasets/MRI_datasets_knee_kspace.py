# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
MRI_datasets_knee_kspace.py

PyTorch dataloaders for FastMRI knee single-coil .h5 files, returning uniform 2D slices.
"""

from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import zoom
from torchvision import transforms
#from numpy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.sparse.linalg import LinearOperator, lsqr

def mask_2d_gauss_accel(H, W, accel=8, sigma_frac=0.5):
    """
    2D Gaussian random sampling for ×8 acceleration.
    Produces low aliasing artifacts by weighting k-space center more heavily.
    """
    ky = np.linspace(-1, 1, H)[:, None]
    kx = np.linspace(-1, 1, W)[None, :]
    pdf = np.exp(-0.5 * ((ky / sigma_frac)**2 + (kx / sigma_frac)**2))
    pdf /= pdf.max()
    # scale so expected samples = H*W/accel
    scale = (H * W / accel) / pdf.sum()
    pdf_scaled = np.clip(pdf * scale, 0, 1)
    return (np.random.rand(H, W) < pdf_scaled).astype(np.float32)

class FastMRIH5SliceDataset(Dataset):
    """Dataset of individual slices from FastMRI single-coil .h5 files, cropping/padding to a fixed shape."""
    def __init__(self, root, is_complex=False, key='reconstruction_rss', target_shape=None, sort=True):
        root = Path(root)
        self.key = key
        self.is_complex = is_complex

        # discover all .h5 files
        files = sorted(root.rglob('*.h5')) if sort else list(root.rglob('*.h5'))
        if not files:
            raise RuntimeError(f"No .h5 files found in {root}")

        # build index: list of (file_path, slice_index)
        self.index = []
        for fpath in files:
            try:
                with h5py.File(fpath, 'r') as h:
                    num_slices = h[self.key].shape[0]
            except Exception as e:
                print(f"Warning: skipping corrupted file {fpath}: {e}")
                continue
            for i in range(num_slices):
                self.index.append((fpath, i))
        if not self.index:
            raise RuntimeError(f"No valid slices found in {root}")

        # determine target spatial shape
        if target_shape is None:
            f0, idx0 = self.index[0]
            with h5py.File(f0, 'r') as h:
                sample = h[self.key][idx0]
            self.target_shape = sample.shape[-2:]
        else:
            self.target_shape = target_shape

    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, idx):
        fpath, slice_idx = self.index[idx]

        with h5py.File(fpath, 'r') as h:
            if 'reconstruction_rss' not in h or 'kspace' not in h:
                raise ValueError(f"Missing keys in file: {fpath}")
            hr_img_orig = h['reconstruction_rss'][slice_idx]  # (H, W), float32
            hr_img_orig = hr_img_orig.astype(np.float32)
            hr_img_orig = (hr_img_orig - np.min(hr_img_orig)) / (np.max(hr_img_orig) - np.min(hr_img_orig) + 1e-8)

        th, tw = self.target_shape

        # 1. Resize HR image for model input/target
        def resize(image, target_h, target_w):
            tensor_img = torch.from_numpy(image).unsqueeze(0)  # [1, H, W]
            resized = transforms.functional.resize(tensor_img, [target_h, target_w])
            return resized.squeeze(0).numpy()

        hr_img = resize(hr_img_orig, th, tw)

        # 2. Perform FFT on original (non-resized) image
        kspace = np.fft.fftshift(np.fft.fft2(hr_img))
    
        # 4. Apply Gaussian mask
        mask = mask_2d_gauss_accel(hr_img.shape[0], hr_img.shape[1], sigma_frac=0.3, accel=8)
        masked_kspace = kspace * mask

        # 5. IFFT to get LR image
        lr_img = np.abs(np.fft.ifft2(np.fft.ifftshift(masked_kspace))).astype(np.float32)

        #lr_img = (lr_img - lr_img.min()) / (lr_img.max() - lr_img.min() + 1e-8)

        # import os
        # import matplotlib.pyplot as plt

        # os.makedirs("debug_outputs", exist_ok=True)
        # plt.imshow(lr_img, cmap='gray')
        # plt.savefig("debug_outputs/debug_lr.png")
        # print("Saved mask to debug_outputs/debug_lr.png")

        lr_img = resize(lr_img, th, tw)
        hr_inte = lr_img.copy()
        x_init = lr_img.copy()  # Can be same as lr_img unless defined otherwise

        # 6. Add channel dim
        hr_img = hr_img[None, ...].astype(np.float32)
        lr_img = lr_img[None, ...].astype(np.float32)
        hr_inte = hr_inte[None, ...].astype(np.float32)
        x_init = x_init[None, ...].astype(np.float32)

        # 7. Normalize safely
        def normalize(x):
            min_val = x.min()
            max_val = x.max()
            if (max_val - min_val) < 1e-5:
                return np.zeros_like(x)
            return 2 * (x - min_val) / (max_val - min_val) - 1
        
        mask_resized = transforms.functional.resize(
            torch.from_numpy(mask.astype(np.float32)).unsqueeze(0),
            [th, tw]
        ).squeeze(0).numpy()[None, ...].astype(np.float32)


        return {
            'lr_img': normalize(lr_img),
            'hr_img': normalize(hr_img),
            'hr_inte': normalize(hr_inte),
            'gt': normalize(hr_img),
            'mask': mask_resized,
            'x_init': normalize(x_init)
        }


def create_dataloader(configs, data_dir=None, evaluation=False, sort=True):
    """
    Create PyTorch DataLoader(s) for FastMRI H5 slices.

    Args:
      configs: config object with:
        - configs.data.root       (fallback path)
        - configs.data.is_complex (bool)
        - configs.data.h5_key     (string)
        - configs.data.image_size (spatial target)
        - configs.training.batch_size
        - configs.training.num_workers
      data_dir: overrides configs.data.root if provided
      evaluation: if True, disables shuffle
      sort: whether to sort files

    Returns:
      (train_loader, val_loader)
    """
    root = data_dir or configs.data.root

    ds = FastMRIH5SliceDataset(
        root,
        is_complex=getattr(configs.data, 'is_complex', True),
        key=getattr(configs.data, 'h5_key', 'reconstruction_rss'),
        target_shape=(configs.data.image_size, configs.data.image_size),
        sort=sort
    )
    loader = DataLoader(
        ds,
        batch_size=configs.training.batch_size,
        shuffle=not evaluation,
        num_workers=configs.training.num_workers,
        drop_last=not evaluation
    )
    return loader, loader