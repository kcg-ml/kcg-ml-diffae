What is cond/xsem, and How to Compute xt?

1. Understanding cond and xsem

• cond (conditioning variable) represents a latent feature vector extracted from an image. It is used to guide the diffusion process. • xsem (semantic latent variable) is a specific form of cond used for semantic guidance (e.g., modifying attributes of an image).

Where is cond Computed?

Defined in model/unet\_autoenc.py:

```python
def encode(self, x):
    cond = self.encoder.forward(x)
    return {'cond': cond}
```

• encode(x) extracts the latent representation (cond) from an input image. • This latent vector is later used in the diffusion model to reconstruct or modify images.

Where is xsem Computed?

xsem is not explicitly found in the main training scripts, but it is likely used as a semantic feature vector extracted during training.

2. How to Compute xt

xt represents the noisy version of the original image x0 at timestep t. It follows the standard forward diffusion equation:

Code Implementation (experiment.py)

The timestep t is sampled, and noise is added to x0:

```python
t, weight = self.T_sampler.sample(len(x_start), x_start.device)
losses = self.sampler.training_losses(model=self.model, x_start=x_start, t=t)
```

• T\_sampler.sample() picks a random timestep t. • training\_losses() calculates noisy latents (xt).

Explicit Computation of xt (diffusion/diffusion.py)

The function q\_sample() implements the diffusion process:

```python
def q_sample(x0, t, noise):
    sqrt_alpha_cumprod_t = torch.sqrt(alphas_cumprod[t])[:, None, None, None]
    sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1 - alphas_cumprod[t])[:, None, None, None]
    return sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * noise
```

• q\_sample() applies the diffusion formula to compute xt. • It scales x0 and adds noise according to the schedule.

3. Summary

• cond = Encoded latent feature extracted from x0 using encode(x). • xsem = Special latent representation used for semantic control. • xt is computed using: • Code Implementation: o q\_sample() applies the diffusion process to x0. o experiment.py samples timesteps and injects noise.

