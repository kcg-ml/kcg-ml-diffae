# DiffAE - Diffusion Autoencoder

This repository contains the implementation of DiffAE (Diffusion Autoencoder), a framework that combines the strengths of diffusion models with autoencoding principles for high-quality image generation and manipulation.

Paper: https://arxiv.org/pdf/2405.17111
Project Page: https://diff-ae.github.io/
Video: https://www.youtube.com/watch?v=i3rjEsiHoUU

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Checkpoints](#checkpoints)
- [Datasets](#Datasets)
- [cond/xsem](#cond/xsem)
- [Models](#models) 
- [Advanced](#advanced)

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
mamba install -c conda-forge matplotlib transformers einops omegaconf wandb torchmetrics scipy=1.15.2 numpy=1.24 tqdm=4.64.0 pandas=1.5.0 lmdb=1.3.0 ftfy=6.1.1 regex=2022.10.31 -y

pip install pytorch-fid lpips
```

### Alternative: Using Original Dependencies
If you encounter compatibility issues, you can use the original versions:
```bash
pip install torch==1.8.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu111
pip install pytorch-lightning==1.4.5 torchmetrics==0.5.0
```

## Checkpoints
You need to download the checkpoints for the models and then upload them to mega , to download&upload them please follow the instructions in this md file  :
```
01.02-download-models-checkpoints.md
```



### Datasets

# LMDB Datasets

We do not own any of the following datasets. We provide the LMDB ready-to-use dataset for the sake of convenience.

- [FFHQ](https://1drv.ms/f/s!Ar2O0vx8sW70uLV1Ivk2pTjam1A8VA)
- [CelebAHQ](https://1drv.ms/f/s!Ar2O0vx8sW70uL4GMeWEciHkHdH6vQ) 

**Broken links**

Note: I'm trying to recover the following links. 

- [CelebA](https://drive.google.com/drive/folders/1HJAhK2hLYcT_n0gWlCu5XxdZj-bPekZ0?usp=sharing) 
- [LSUN Bedroom](https://drive.google.com/drive/folders/1O_3aT3LtY1YDE2pOQCp6MFpCk7Pcpkhb?usp=sharing)
- [LSUN Horse](https://drive.google.com/drive/folders/1ooHW7VivZUs4i5CarPaWxakCwfeqAK8l?usp=sharing)

The directory tree should be:

```
datasets/
- bedroom256.lmdb
- celebahq256.lmdb
- celeba.lmdb
- ffhq256.lmdb
- horse256.lmdb
```

You can also download from the original sources, and use our provided codes to package them as LMDB files.
Original sources for each dataset is as follows:

- FFHQ (https://github.com/NVlabs/ffhq-dataset)
- CelebAHQ (https://github.com/switchablenorms/CelebAMask-HQ)
- CelebA (https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- LSUN (https://github.com/fyu/lsun)

The conversion codes are provided as:

```
data_resize_bedroom.py
data_resize_celebhq.py
data_resize_celeba.py
data_resize_ffhq.py
data_resize_horse.py
```




## cond/xsem
What is cond/xsem, and How to Compute xt?
you can find the answer of this questions in this md file :
```
02.07-cond_xsem.md

XSEM from Image ```02.00-xsem-from-image.md ```

XT from Image ```02.01-xt-from-image.md ```

XT and XSEM from Image ```02.03-image-from-xt-and-xsem.md ```

Image from XSEM and random XT ``` 02.04-image-from-xsem-and-random-xt.md```

Image from XT and random XSEM ```02.05-image-from-xt-and-random-xsem.md ```

## Models

To see each model variants , input ,output and how to train & run the infreences of the models please follow the instructions in this md file :
```
03.01-how-to-use-models-training-and-inference-scripts.md
```


## Advanced

### Model Size Calculation

To Calculate Model Size for Different Precision Types you can follow the instructions in this md file:
```
03.00-load-and-calculate-model-size-script.md
```

### Visualization and Debugging

Several Jupyter notebooks are provided for visualization and debugging:
- `interpolate.ipynb`: Interpolation between latent codes
- `autoencoding.ipynb`: Visualization of autoencoding capabilities
- `sample.ipynb`: Sampling new images
- `manipulate.ipynb`: Attribute manipulation


