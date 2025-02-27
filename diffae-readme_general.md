# DiffAE - Diffusion Autoencoder

This repository contains the implementation of DiffAE (Diffusion Autoencoder), a framework that combines the strengths of diffusion models with autoencoding principles for high-quality image generation and manipulation.

Paper: https://arxiv.org/pdf/2405.17111
Project Page: https://diff-ae.github.io/
Video: https://www.youtube.com/watch?v=i3rjEsiHoUU

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Checkpoints](#checkpoints)
- [Models](#models)
- [Basic Usage](#basic-usage)
- [Training](#training)
- [Advanced Usage](#advanced-usage)

## Overview

DiffAE combines the strengths of diffusion models with autoencoding, enabling high-quality image generation, manipulation, and reconstruction. The key components are:

- **Semantic Encoder**: Extracts structural and semantic information from input images
- **Diffusion Process**: Applies controlled noise to image representations
- **Decoder**: Reconstructs images from semantic representations and noise

The framework supports multiple operations including:
- Image reconstruction
- Image sampling/generation
- Semantic manipulation
- Feature interpolation

## Installation

### Environment Setup

```bash
# Create conda environment
conda create -p /opt/envs/diffae python=3.11 conda-forge::mamba -y
conda activate /opt/envs/diffae

# Install PyTorch with CUDA support
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install dependencies
mamba install conda-forge::pytorch-lightning==1.5.10 -y
mamba install -c conda-forge ipywidgets ipykernel ipython matplotlib transformers einops omegaconf wandb git -y

# Setup Jupyter kernel
python -m ipykernel install --name=diffae --display-name "diffae"
```

### Alternative: Using Original Dependencies
If you encounter compatibility issues, you can use the original versions:
```bash
pip install torch==1.8.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu111
pip install pytorch-lightning==1.4.5 torchmetrics==0.5.0
```

### Clone Repository
```bash
git clone https://github.com/kk-digital/kcg-ml-diffae.git
cd kcg-ml-diffae
```

## Checkpoints

We provide pretrained checkpoints for various models and datasets. The checkpoints should be placed in a `checkpoints` directory with the following structure:

```
checkpoints/
- bedroom128_autoenc/
    - last.ckpt       # diffae checkpoint
    - latent.ckpt     # predicted z_sem on the dataset
- bedroom128_autoenc_latent/
    - last.ckpt       # diffae + latent DPM checkpoint
- bedroom128_ddpm/
    - last.ckpt       # DDPM checkpoint
- ...
```

### Downloading Checkpoints

#### Option 1: Using the Download Script
```bash
sh download-models.sh "https://drive.google.com/drive/folders/1l75eX_tYCersmnAyJYQ84qi9hZyyv9vL?usp=drive_link"
```

#### Option 2: Using gdown (for single folders)
```python
!gdown --folder https://drive.google.com/drive/folders/FOLDER_ID -O /path/to/diffae/checkpoints
```

### Available Checkpoint Links

1. **DDIM Models**:
   - FFHQ128: [72M](https://drive.google.com/drive/folders/1-fa46UPSgy9ximKngBflgSj3u87-DLrw), [130M](https://drive.google.com/drive/folders/1-Sqes07fs1y9sAYXuYWSoDE_xxTtH4yx)
   - [Bedroom128](https://drive.google.com/drive/folders/1-_8LZd5inoAOBT-hO5f7RYivt95FbYT1)
   - [Horse128](https://drive.google.com/drive/folders/10Hq3zIlJs9ZSiXDQVYuVJVf0cX4a_nDB)

2. **DiffAE (autoencoding only)**:
   - [FFHQ256](https://drive.google.com/drive/folders/1-5zfxT6Gl-GjxM7z9ZO2AHlB70tfmF6V)
   - FFHQ128: [72M](https://drive.google.com/drive/folders/10bmB6WhLkgxybkhso5g3JmIFPAnmZMQO), [130M](https://drive.google.com/drive/folders/10UNtFNfxbHBPkoIh003JkSPto5s-VbeN)
   - [Bedroom128](https://drive.google.com/drive/folders/12EdjbIKnvP5RngKsR0UU-4kgpPAaYtlp)
   - [Horse128](https://drive.google.com/drive/folders/12EtTRXzQc5uPHscpjIcci-Rg-OGa_N30)

3. **DiffAE (with latent DPM, can sample)**:
   - [FFHQ256](https://drive.google.com/drive/folders/1-H8WzKc65dEONN-DQ87TnXc23nTXDTYb)
   - [FFHQ128](https://drive.google.com/drive/folders/11pdjMQ6NS8GFFiGOq3fziNJxzXU1Mw3l)
   - [Bedroom128](https://drive.google.com/drive/folders/11mdxv2lVX5Em8TuhNJt-Wt2XKt25y8zU)
   - [Horse128](https://drive.google.com/drive/folders/11k8XNDK3ENxiRnPSUdJ4rnagJYo4uKEo)

4. **DiffAE's classifiers (for manipulation)**:
   - [FFHQ256's latent on CelebAHQ](https://drive.google.com/drive/folders/117Wv7RZs_gumgrCOIhDEWgsNy6BRJorg)
   - [FFHQ128's latent on CelebAHQ](https://drive.google.com/drive/folders/11EYIyuK6IX44C8MqreUyMgPCNiEnwhmI)

## Models

DiffAE implements several model variants with different capabilities:

### 1. DDIM Models

**Model Variants**: FFHQ128 (72M, 130M), Bedroom128, Horse128

**Input/Output**:
- **Input**: Latent code (batch, latent_dim), Noise level (batch, 1)
- **Output**: Generated image (batch, channels, height, width)

**Inference Script**: `interpolate.ipynb`

**Training Script**: 
```bash
python run_ffhq128_ddim.py
python run_bedroom128_ddim.py
python run_horse128_ddim.py
```

### 2. DiffAE (Autoencoding Only)

**Model Variants**: FFHQ256, FFHQ128, Bedroom128, Horse128, Celeba64

**Input/Output**:
- **Input**: Input image (batch, channels, height, width)
- **Output**: 
  - Encoded representation (batch, latent_dim)
  - Reconstructed image (batch, channels, height, width)

**Inference Script**: `autoencoding.ipynb`

**Training Script**: 
```bash
python run_ffhq256.py
python run_bedroom128.py
python run_horse128.py
python run_celeba64.py
```

### 3. DiffAE (With Latent DPM, Can Sample)

**Model Variants**: FFHQ256, FFHQ128, Bedroom128, Horse128

**Input/Output**:
- **Input**: Latent code (batch, latent_dim)
- **Output**: Generated image (batch, channels, height, width)

**Inference Script**: `sample.ipynb`

**Training Script**: 
```bash
python run_ffhq256_latent.py
# Similar scripts for other datasets
```

### 4. DiffAE Classifiers (For Manipulation)

**Model Variants**: FFHQ128's latent on CelebAHQ, FFHQ256's latent on CelebAHQ

**Input/Output**:
- **Input**: Latent code (batch, latent_dim)
- **Output**: Classification (batch, 1)

**Inference Script**: `manipulate.ipynb`

**Training Script**: 
```bash
python run_ffhq128_cls.py
```

## Basic Usage

### Loading a Model

```python
from templates import ffhq256_autoenc, LitModel

# Load configuration for FFHQ 256x256
conf = ffhq256_autoenc()

# Create model instance 
model = LitModel(conf)

# Load checkpoint
model = model.load_from_checkpoint('checkpoints/ffhq256_autoenc/last.ckpt', strict=False)

# Move model to device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
```

### Extracting XSEM (Semantic Encoding) from an Image

```python
from PIL import Image
import torch
from torchvision.transforms import functional as VF

# Load and preprocess image
img = Image.open('example.jpg').resize((256, 256)).convert('RGB')
x = VF.to_tensor(img).unsqueeze(0).to(device)

# Extract semantic encoding
xsem = model.encode(x)
```

### Computing XT (Noisy Representation)

```python
# Compute noisy representation at timestep t
xt = model.encode_stochastic(x, cond=xsem, T=250)
```

### Generating Images

```python
# Reconstruct image from semantic encoding and noisy representation
reconstructed_img = model.render(xt, xsem, T=20)

# Generate variation using semantic encoding and random noise
random_xt = torch.randn_like(model.encode_t(x, t=50))
variation_img = model.decode(xsem, random_xt)
```

## Training

### Dataset Requirements

Before training, you need:
1. Prepare your dataset in LMDB format
2. Configure paths in the configuration files

### Starting Training

```bash
# For FFHQ 256x256 autoencoder model
python run_ffhq256.py

# For FFHQ 128x128 DDIM model
python run_ffhq128_ddim.py
```

### Configuration

Training parameters can be modified in the respective run scripts or by creating custom configurations in `config.py`. Key parameters:

- Batch size (default: 16)
- Learning rate (default: 0.0001)
- Diffusion steps (default: 1000)
- Model size (72M or 130M parameters for some variants)

## Advanced Usage

### Model Size Calculation

Calculate model size for different precision types:

```python
python xx-load-model.py --model_name ffhq128_autoenc_130M
python xx-load-model.py --model_name bedroom128_ddpm
python xx-load-model.py --model_name ffhq256_autoenc
```

### Image Manipulation

For attribute manipulation (e.g., smiling, age), use the classifier models:

```python
# See manipulate.ipynb for detailed examples
```

### Visualization and Debugging

Several Jupyter notebooks are provided for visualization and debugging:
- `interpolate.ipynb`: Interpolation between latent codes
- `autoencoding.ipynb`: Visualization of autoencoding capabilities
- `sample.ipynb`: Sampling new images
- `manipulate.ipynb`: Attribute manipulation

## Citation

If you use this code in your research, please cite:
```
@article{preechakul2022diffusion,
  title={Diffusion Autoencoders: Toward a Meaningful and Decodable Representation},
  author={Preechakul, Konpat and Chatthee, Nattanat and Wizadwongsa, Suttisak and Suwajanakorn, Supasorn},
  journal={arXiv preprint arXiv:2405.17111},
  year={2022}
}
```
