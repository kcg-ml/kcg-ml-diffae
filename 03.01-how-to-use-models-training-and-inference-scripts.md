
## 1. DDIM Models

### Model Variants
- **FFHQ128 (72M, 130M)**
- **Bedroom128**
- **Horse128**

### Inputs
| Name        | Type   | Tensor Size                      |
|-------------|--------|----------------------------------|
| Latent Code | Tensor | (Batch, Latent Dim)             |
| Noise Level | Tensor | (Batch, 1)                      |

### Outputs
| Name          | Type   | Tensor Size                      |
|--------------|--------|----------------------------------|
| Generated Image | Tensor | (Batch, Channels, Height, Width) |

### Inference Script
```python
interpolate.ipynb
```

### Training Script
```
python run_ffhq128_ddim.py

python run_bedroom128_ddim.py

python run_horse128_ddim.py
```

---

## 2. DiffAE (Autoencoding Only)

### Model Variants
- **FFHQ256**
- **Bedroom128**
- **Horse128**
- **Celeba64**


### Inputs
| Name        | Type   | Tensor Size                      |
|-------------|--------|----------------------------------|
| Input Image | Tensor | (Batch, Channels, Height, Width) |

### Outputs
| Name                   | Type   | Tensor Size                      |
|------------------------|--------|----------------------------------|
| Encoded Representation | Tensor | (Batch, Latent Dim)              |
| Reconstructed Image    | Tensor | (Batch, Channels, Height, Width) |

### Inference Script
```
autoencoding.ipynb
```

### Training Script
```
sbatch run_ffhq256.py

python run_bedroom128.py

python run_horse128.py

python run_celeba64.py
```

---

## 3. DiffAE (With Latent DPM, Can Sample)

### Model Variants
- **FFHQ256**


### Inputs
| Name        | Type   | Tensor Size         |
|-------------|--------|---------------------|
| Latent Code | Tensor | (Batch, Latent Dim) |

### Outputs
| Name            | Type   | Tensor Size                      |
|-----------------|--------|----------------------------------|
| Generated Image | Tensor | (Batch, Channels, Height, Width) |

### Inference Script
```python
sample.ipynb
```

### Training Script
```
python run_ffhq256_latent.py
```

---

## 4. DiffAE Classifiers (For Manipulation)

### Model Variants
- **FFHQ128's Latent on CelebAHQ**

### Inputs
| Name        | Type   | Tensor Size         |
|-------------|--------|---------------------|
| Latent Code | Tensor | (Batch, Latent Dim) |

### Outputs
| Name            | Type   | Tensor Size |
|-----------------|--------|-------------|
| Classification  | Tensor | (Batch, 1)  |

### Inference Script
```python
manipulate.ipynb
```

### Training Script
```
python run_ffhq128_cls.py
```

---




