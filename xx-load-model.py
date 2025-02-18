from templates import *
from templates_latent import *
from experiment import *
if __name__ == '__main__':
    conf = ffhq128_autoenc_130M()
    model = LitModel(conf)
    num_weights = sum(p.numel() for p in model.parameters())



    # Calculate sizes for FP8 (assuming 1 byte per weight), FP16 (2 bytes per weight), FP32 (4 bytes per weight)
    size_fp8 = num_weights * 1 / (1024**2)  # in MB
    size_fp16 = num_weights * 2 / (1024**2)  # in MB
    size_fp32 = num_weights * 4 / (1024**2)  # in MB

    # Calculate total model size in MB
    model_size_mb = sum(p.element_size() * p.numel() for p in model.parameters()) / (1024 * 1024)

    print(f"Total Weights: {num_weights}")
    print(f"FP8 Weights: {size_fp8}")
    print(f"FP16 Weights: {size_fp16}")
    print(f"FP32 Weights: {size_fp32}")
    print(f"Model Size on Disk: {model_size_mb:.2f} MB")
