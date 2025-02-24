##### **1. Title & Purpose**

```md

# XT from Image

This document explains how to Extracts `xt` from an image using the DiffAE model.

```


```md

## Extracting xt from an Image

  


  

```python

from PIL import Image

import torch

from torchvision.transforms import functional as VF

from templates import ffhq256_autoenc, LitModel

  

device = 'cuda'

conf = ffhq256_autoenc()

model = LitModel(conf)


  

# Load Image

img = Image.open('example.jpg').resize((256, 256)).convert('RGB')

  

# Convert to Tensor

x = VF.to_tensor(img).unsqueeze(0).to(device)

  

# Encode

xt = model.encode_stochastic(x, cond, T=250)

```

  

##### **4. Expected Output**

```md

## Expected Output

- The variable `xt`  different timesteps contains progressively noisier representations of the image.


```

  

---

  

