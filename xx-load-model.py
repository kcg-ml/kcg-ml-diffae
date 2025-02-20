from templates import *
from templates_latent import *
from experiment import *
import argparse

def calc_weights(name,model):
    num_weights = sum(p.numel() for p in model.parameters())
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
    
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="calc_weights.")
    parser.add_argument("model_name", type=str, help="Model name")
    args = parser.parse_args()
    model_name = args.model_name

    print(f"Hello, {args.model_name}!")
    
    if model_name=='ffhq128_autoenc_130M':
        conf = ffhq128_autoenc_130M()
        model = LitModel(conf)
        calc_weights(model_name,model)
        
    if model_name=='bedroom128_ddpm':
        conf = bedroom128_ddpm()
        model = LitModel(conf)
        calc_weights(model_name,model)

    if model_name=='celeba64d2c_autoenc':
        conf = celeba64d2c_autoenc()
        model = LitModel(conf)
        calc_weights(model_name,model)

    if model_name=='ffhq128_autoenc_cls':
        conf = ffhq128_autoenc_cls()
        model = LitModel(conf)
        calc_weights(model_name,model)
        
    if model_name=='ffhq256_autoenc':
        conf = ffhq256_autoenc()
        model = LitModel(conf)
        calc_weights(model_name,model)
        
    if model_name=='horse128_autoenc':
        conf = horse128_autoenc()
        model = LitModel(conf)
        calc_weights(model_name,model)

    
    if model_name=='horse128_ddpm':
        conf = horse128_ddpm()
        model = LitModel(conf)
        calc_weights(model_name,model)    

