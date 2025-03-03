# DiffAE Installation Guide

This document provides guidance on how to install and run the DiffAE repository with current Python and package versions.

## System Requirements

- **Python Version**: 3.8 - 3.11
- **CUDA**: Recommended for better performance (but optional)
- **OS**: Windows, Linux, or macOS

## Installation Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kk-digital/diffae.git
   cd diffae
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   # Using venv
   python -m venv diffae_env
   
   # On Windows:
   .\diffae_env\Scripts\activate
   
   # On Linux/macOS:
   source diffae_env/bin/activate
   ```

3. **Install updated dependencies**:
   
   The original requirements.txt file contains outdated package versions that are no longer compatible with current Python distributions. Use the updated requirements file provided below:
   
   ```bash
   # Create the updated requirements file
   # Copy the content from requirements_updated.txt in this repo
   
   # Install dependencies
   pip install -r requirements_updated.txt
   ```

4. **Note on PyTorch versions**:
   
   The original code was written for PyTorch 1.8.1, but modern versions (2.0.0+) should be compatible. If you encounter issues with specific model behaviors, you might need to modify the code slightly to accommodate API changes.

## Troubleshooting

- **CUDA errors**: If you encounter CUDA-related errors, make sure your PyTorch installation matches your CUDA version. You can install a specific CUDA version of PyTorch using:
  ```bash
  pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118
  ```
  (Replace cu118 with your CUDA version, e.g., cu117, cu116)

- **Import errors**: Some module imports might fail due to API changes in newer package versions. If this happens, check the error message and adjust the imports accordingly.

## Running the Models

The repository contains several scripts for running different models:

- `run_ffhq128.py`: For FFHQ dataset at 128x128 resolution
- `run_celeba64.py`: For CelebA dataset at 64x64 resolution
- `run_bedroom128.py`: For LSUN Bedroom dataset at 128x128 resolution
- `run_horse128.py`: For LSUN Horse dataset at 128x128 resolution

Example command:
```bash
python run_ffhq128.py
```

