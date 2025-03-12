import argparse
import os
import sys
from safetensors.torch import load as safetensors_load
import torch
from torchvision.transforms import functional as VF
from PIL import Image
from io import BytesIO

base_dir = "./"
sys.path.insert(0, base_dir)
sys.path.insert(0, os.getcwd())
from diffae.experiment import LitModel
from diffae.templates import ffhq256_autoenc
from utility.path import separate_bucket_and_file_path
from utility.utils import get_device
from utility.minio import minio_manager

class DiffAEInferencePipeline:
    def __init__(self,
                 minio_client, 
                 model_id, 
                 num_checkpoint, 
                 device):
        
        self.minio_client = minio_client
        self.model_id = model_id
        self.num_checkpoint = num_checkpoint
        self.device = get_device(device)

        # models
        self.diffae = None
        self.ema_model = None

    def load_base_model(self):
        # initialization base model configuration
        self.conf = ffhq256_autoenc()

        # Model Initialization
        self.diffae = LitModel(self.conf).to(self.device)
        # load the checkpoint
        self.load_diffae_checkpoint(self.num_checkpoint)
        # move model to the device
        self.diffae.model= self.diffae.model.to(self.device)
        self.ema_model = self.diffae.ema_model.to(self.device)
    
    def load_diffae_checkpoint(self, num_checkpoint):
        # Extract bucket name and file path from minio_path
        checkpoint_path = f"models/diffae/trained_models/{str(self.model_id).zfill(4)}/checkpoints/checkpoint_{num_checkpoint}.safetensors"
        bucket, checkpoint_path = separate_bucket_and_file_path(checkpoint_path) 

        # Download the checkpoint from MinIO
        print(f"Downloading checkpoint from MinIO: {checkpoint_path} ...")
        response = self.minio_client.get_object(bucket, checkpoint_path)
        checkpoint_data = response.read()  # Read into memory
        
        # Load safetensors checkpoint into a dictionary
        print("Loading checkpoint using safetensors...")
        checkpoint_dict = safetensors_load(checkpoint_data)

        # Load weights into model (diffae model + EMA model)
        print("Updating model state...")
        self.diffae.model.load_state_dict(checkpoint_dict, strict=False)
        self.diffae.ema_model.load_state_dict(checkpoint_dict, strict=False)

        print(f"Model loaded successfully from MinIO {checkpoint_path}")

    def pre_process_image(self, image_path: str, image_size: int):
        # open the input image
        image = Image.open(image_path)
        # downscale the image to target size
        downscaled_image = self.downscale_image(image , target_size= image_size)
        # Convert to tensor
        image_tensor = VF.to_tensor(downscaled_image)
        # Normalize to [-1, 1]
        image_tensor = image_tensor * 2. - 1.

        return image_tensor

    @staticmethod
    def downscale_image(image: Image.Image, target_size: int, scale_factor: float = 2.0):
        current_size = image.size[0]  # Assume the image is square
        resized_image = image

        # Generate intermediate sizes based on the scale factor
        while current_size / scale_factor > target_size:
            current_size = int(current_size / scale_factor)
            resized_image = resized_image.resize((current_size, current_size), Image.LANCZOS)

        # Final resize to the exact target size
        resized_image = resized_image.resize((target_size, target_size), Image.LANCZOS)

        return resized_image
    
    def run_inference(self, image_path: str, image_size: int):
        # load the checkpoint
        if self.diffae is None:
            self.load_base_model()
    
        # preprocess the input image
        x = self.pre_process_image(image_path, image_size)
        x = x.unsqueeze(0).to(self.device)

        with torch.no_grad():
            
            cond = self.diffae.encode(x)
            cond_R = torch.randn_like(cond)
            print(f"Z-sem shape: {cond.shape}")
            
            # xT = torch.randn_like(x)
            xT = self.diffae.encode_stochastic(x, cond, T=500)
            
            pred = self.diffae.render(xT, cond_R, T=100)

        result_image= VF.to_pil_image(torch.cat([
            torch.clamp(x[0] * .5 + .5, 0., 1.),
            torch.clamp(xT[0] * .125 + .5, 0., 1.),
            torch.clamp(pred[0], 0., 1.)
        ], dim=2))

        return result_image
    
def parse_args():
    parser = argparse.ArgumentParser()

    # MinIO credentials
    parser.add_argument('--minio-access-key', type=str, required=True, help='Access key for model MinIO storage.')
    parser.add_argument('--minio-secret-key', type=str, required=True, help='Secret key for model MinIO storage.')

    # parameters
    parser.add_argument('--model-id', type=int, help='Model id', required=True)
    parser.add_argument('--num-checkpoint', type=int, help='Checkpoint number', required=True)
    parser.add_argument('--image-path', type=str, default="local path to the input image", required=True)
    parser.add_argument('--image-size', type=int, help='image size', required=True, default=256)
    parser.add_argument('--device', type=str, required=True, default="cuda")

    return parser.parse_args()

def main():
    args= parse_args()

    # get minio client
    minio_client = minio_manager.get_minio_client(minio_ip_addr= "192.168.3.6:9000",
                                        minio_access_key=args.minio_access_key,
                                        minio_secret_key=args.minio_secret_key)
    
    inference_pipeline = DiffAEInferencePipeline(minio_client=minio_client,
                                                 model_id= args.model_id,
                                                 num_checkpoint= args.num_checkpoint,
                                                 device= args.device)
    # load the checkpoint
    inference_pipeline.load_base_model()
    # run inference
    result_image = inference_pipeline.run_inference(image_path = args.image_path, image_size= args.image_size)
    # upload to minio
    image_data = BytesIO()
    result_image.save(image_data, format="PNG")
    image_data.seek(0)
    minio_manager.upload_data(minio_client, "models", f"diffae/experiments/inference_test/{args.model_id}/result_checkpoint_{args.num_checkpoint}.png", image_data)

if __name__=="__main__":
    main()    