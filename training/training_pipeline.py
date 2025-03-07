import argparse
from datetime import datetime
from datetime import timezone as tz
from hashlib import blake2b
import json
import random
import shutil
import threading
import time
import uuid
import concurrent.futures
import msgpack
from pytz import timezone
from io import BytesIO
import os
import sys
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import functional as VF
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from safetensors.torch import save as safetensors_save
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

base_dir = "./"
sys.path.insert(0, base_dir)
sys.path.insert(0, os.getcwd())
from dataloaders.image_dataset_loader import HttpRequestHandler
from diffae.experiment import LitModel, WarmupLR, ema
from diffae.templates import ffhq256_autoenc
from utility.http import worker_request
from utility.minio import minio_manager
from utility.path import separate_bucket_and_file_path
from diffae.choices import TrainMode
from diffae.model.nn import mean_flat

class ImageDataset(Dataset):
    def __init__(self, image_hashes, image_uuids, image_paths):
        self.image_hashes = image_hashes
        self.image_uuids = image_uuids
        self.image_paths = image_paths
    
    def __len__(self):
        return len(self.image_hashes)
    
    def __getitem__(self, idx):
        return {
            'image_hash': self.image_hashes[idx],
            'image_path': self.image_paths[idx],
            'image_uuid': self.image_uuids[idx]
        }

class UnclipDataset(Dataset):
    def __init__(self, image_hashes, image_uuids, image_tensors):
        self.image_hashes = image_hashes
        self.image_uuids = image_uuids
        self.image_tensors = image_tensors
    
    def __len__(self):
        return len(self.image_hashes)
    
    def __getitem__(self, idx):
        return {
            'image_hash': self.image_hashes[idx],
            'image_uuid': self.image_uuids[idx],
            'image_tensor': self.image_tensors[idx]
        }

def collate_fn(batch):
    image_batch = torch.stack([item['image_tensor'] for item in batch])
    image_uuids = [item['image_uuid'] for item in batch]
    image_hashes = [item['image_hash'] for item in batch]

    return {'image_hashes': image_hashes, 'image_uuids': image_uuids, 'image_batch': image_batch}

def get_rank():
    rank= dist.get_rank()
    return rank

def print_in_rank(msg: str):
    rank= get_rank()
    print(f"gpu {rank}: {msg}")

def create_model_id(dataset:str, hyperparameters:dict):

    model_dict={
        "uuid": str(uuid.uuid4()),
        "model_name": "diffae",
        "training_dataset": dataset,
        "training_hyperparameters_dict": hyperparameters
    }

    model_json= json.dumps(model_dict, indent=4)
    model= worker_request.http_add_model(model_json)

    return model["uuid"], model['sequence_number']

def set_sampling_seed(device):
    # setting seed for sampling
    if dist.get_rank() == 0:
        seed = random.randint(0, 2 ** 24 - 1)
        seed = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        seed = torch.tensor(0, dtype=torch.int32, device=device)
    
    dist.barrier()
    # Broadcast the seed from rank 0 to all other GPUs
    dist.broadcast(seed, src=0)
    sampling_seed = seed.item()

    return sampling_seed

def set_model_seed(device, model_seed):
    # set seed for model generation
    if model_seed is None:
        if dist.get_rank() == 0:
            seed = random.randint(0, 2 ** 24 - 1)
            seed = torch.tensor(seed, dtype=torch.int32, device=device)
        else:
            seed = torch.tensor(0, dtype=torch.int32, device=device)
        
        dist.barrier()
        # Broadcast the seed from rank 0 to all other GPUs
        dist.broadcast(seed, src=0)
        model_seed = seed.item()
    
    # set torch seed
    torch.manual_seed(model_seed)

    return model_seed

class DiffaeTrainingPipeline:
    def __init__(self,
                 minio_client,
                 local_rank,
                 world_size,
                 dataset,
                 epoch_size,
                 finetune=False,
                 model_seed= None,
                 model_id= None,
                 num_checkpoint= None,
                 weight_dtype="float32",
                 learning_rate=5e-5,
                 beta1 = 0.9,
                 beta2 = 0.999,
                 weight_decay = 0.0,
                 ema_decay = 0.9999,
                 epsilon = 1e-4,
                 lr_warmup_steps = 500,
                 max_train_steps = 10000,
                 gradient_accumulation_steps = 1,
                 checkpointing_steps = 100,
                 micro_batch_size= 1,
                 save_results= True,
                 loading_workers= 10,
                 image_resolution = 512):

        # get minio client and local rank
        self.minio_client= minio_client
        self.local_rank = local_rank
        self.world_size = world_size
 
        # hyperparameters
        self.dataset = dataset
        self.epoch_size = epoch_size
        self.finetune= finetune
        self.model_seed = model_seed
        self.model_id= model_id
        self.num_checkpoint= num_checkpoint
        self.weight_dtype= torch.float32 if weight_dtype=="float32" else torch.float16
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.ema_decay = ema_decay
        self.epsilon = epsilon
        self.lr_warmup_steps = lr_warmup_steps
        self.max_train_steps = max_train_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.checkpointing_steps = checkpointing_steps
        self.micro_batch_size = micro_batch_size
        self.loading_workers = loading_workers
        self.save_results = save_results
        self.image_resolution = image_resolution
        self.date = datetime.now(tz=timezone("Asia/Hong_Kong")).strftime('%Y-%m-%d')

        # models
        self.diffae= None
        self.ema_model= None
        
        # loading thread
        self.data_loading_thread = None
        self.next_epoch_data = None
    
    def initialize_model(self, device):
        # initialization base model configuration
        self.conf = ffhq256_autoenc()
        # set necessary parameters
        self.conf.seed = self.model_seed
        self.conf.img_size = self.image_resolution
        self.conf.model_conf.image_size = self.image_resolution
        self.conf.batch_size = self.micro_batch_size
        self.conf.lr= self.learning_rate
        self.conf.weight_decay = self.weight_decay
        self.conf.ema_decay = self.ema_decay
        self.conf.warmup = self.lr_warmup_steps
        self.conf.fp16 = False if self.weight_dtype=="float32" else True

        # Model Initialization
        self.diffae = LitModel(self.conf).to(device, dtype=self.weight_dtype)
        self.diffae.model= self.diffae.model.to(device, dtype=self.weight_dtype)
        self.ema_model = self.diffae.ema_model.to(device, dtype=self.weight_dtype)
        # wrap the diffae model with ddp
        self.diffae_model = DDP(self.diffae.model, device_ids=[device], output_device=device, find_unused_parameters=True)

        # initialize optimize and scheduler
        self.optimizer = torch.optim.AdamW(self.diffae.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        if self.lr_warmup_steps > 0:
            self.schedueler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=WarmupLR(self.lr_warmup_steps))

    def get_image_dataset(self, image_metadata, sampling_seed):
        image_hashes, image_uuids, image_paths = [], [], []
        for image_data in image_metadata:
            image_hash, image_uuid, image_path = image_data
            image_hashes.append(image_hash)
            image_uuids.append(image_uuid)
            image_paths.append(image_path)

        train_dataset = ImageDataset(image_hashes, image_uuids, image_paths)

        image_dataloader = DataLoader(
            train_dataset,
            batch_size= 1,
            sampler= DistributedSampler(train_dataset, shuffle=True, drop_last=True, seed=sampling_seed),
            num_workers= 5 
        )

        return image_dataloader
    
    def load_images(self, image_hash, image_uuid, image_path):
        # load vae and clip vectors
        bucket_name, image_path = separate_bucket_and_file_path(image_path)
        response = self.minio_client.get_object(bucket_name, image_path)
        image_data = response.read()
        image = Image.open(BytesIO(image_data))

        # Apply random horizontal flip
        if np.random.random() < 0.5:
            image = VF.hflip(image)

        # Convert to tensor
        image = VF.to_tensor(image)

        # Normalize to [-1, 1]
        image_tensor = image * 2. - 1.

        return image_hash, image_uuid, image_tensor
    
    def load_dataset(self, dataloader):
        """
        Load latents for a batch of image hashes and file paths using multiple threads.
        """
        results = []
        futures = []
        # Use a thread pool to load latents in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.loading_workers) as executor:
            for batch in dataloader:
                image_hash = batch["image_hash"][0]
                image_uuid = batch["image_uuid"][0]
                image_path = batch["image_path"][0]

                future= executor.submit(self.load_images, image_hash, image_uuid, image_path)
                futures.append(future)

            # Collect results as they complete
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"Failed to load image tensor for an image: {e}")

        # Extract data from results
        results.sort(key=lambda x: x[0])

        sorted_image_hashes = [r[0] for r in results]
        sorted_uuids = [r[1] for r in results]
        image_tensors = [r[2] for r in results]

        return sorted_image_hashes, sorted_uuids, image_tensors

    def load_epoch_data(self, dataloader):
        """Background data loading thread"""
        # Load the data
        image_hashes, image_uuids, image_tensors=self.load_dataset(dataloader)
        self.next_epoch_data= image_hashes, image_uuids, image_tensors

    def start_data_loading_thread(self, dataloader):
        """Start the background data loading thread"""
        self.data_loading_thread = threading.Thread(
            target=self.load_epoch_data,
            args=(dataloader,),
            daemon=True
        )
        self.data_loading_thread.start()
    
    def train_step(self, image_batch, device):
        """Perform one training step"""
        t, weight = self.diffae.T_sampler.sample(image_batch.shape[0], device=device).to(dtype=self.weight_dtype)
        noise = torch.randn_like(image_batch, dtype=self.weight_dtype)
        x_t = self.diffae.sampler.q_sample(image_batch, t, noise=noise).to(dtype=self.weight_dtype)

        model_output = self.diffae_model.forward(
            x=x_t.detach(),
            t=self.diffae.sampler._scale_timesteps(t).to(dtype=self.weight_dtype),
            x_start=image_batch.detach()
        ).pred

        target = noise
        loss = mean_flat((target - model_output) ** 2).mean()

        return loss

    def train_diffae_model(self):
        # set device for each process
        torch.cuda.set_device(self.local_rank)
        device = torch.device("cuda", self.local_rank)
        
        # set seed for model generation
        self.model_seed = set_model_seed(device, self.model_seed)

        # load the diffae base model in each gpu and the optimizer and scheduler
        self.initialize_model(device)
        
        # create new model instance
        if dist.get_rank() == 0:
            if self.finetune:
                # load a euler checkpoint to fine tune
                assert self.model_id and self.num_checkpoint, f"You need to provide the model id and checkpoint number to fine tune a checkpoint."
                model_details = worker_request.http_get_model_details(self.model_id)
                model_uuid = model_details["uuid"]
                sequence_num = self.model_id
            else:
                # Store model in mongoDB
                model_uuid, sequence_num = create_model_id(dataset=self.dataset, hyperparameters=self.conf.serialize())

            # Convert sequence_num to a tensor for broadcasting
            sequence_num_tensor = torch.tensor(sequence_num, dtype=torch.int32, device=device)
        else:
            # Create a placeholder tensor for receiving sequence_num
            sequence_num_tensor = torch.tensor(0, dtype=torch.int32, device=device)

        # Synchronize all GPUs here before proceeding
        dist.barrier()

        # Broadcast the sequence_num from rank 0 to all other ranks
        dist.broadcast(sequence_num_tensor, src=0)
        self.model_id = sequence_num_tensor.item()

        # get the output minio directory where training results are stored 
        self.output_directory = f"models/diffae/trained_models/{str(self.model_id).zfill(4)}"

        # get image metadata (file paths and hashes)
        request_handler= HttpRequestHandler("extracts", self.dataset, 0)
        dataset_metadata = request_handler.get_all_image_data()
        current_metadata = dataset_metadata.copy()
        
        # set sampling seed for the current epoch
        sampling_seed= set_sampling_seed(device)
        random.seed(sampling_seed)
        random.shuffle(current_metadata)

        # get epoch data
        epoch_metadata = current_metadata[:self.epoch_size]
        current_metadata = current_metadata[self.epoch_size:]
        image_dataloader= self.get_image_dataset(epoch_metadata, sampling_seed)
        next_epoch_data= self.load_dataset(iter(image_dataloader))

        dist.barrier()
        print("finished loading images")

        self.diffae.train()
        step, epoch = 0, 1
        losses = []

        while step < self.max_train_steps:
            # get next epoch data
            epoch_metadata = current_metadata[:self.epoch_size]
            current_metadata = current_metadata[self.epoch_size:]
            image_dataloader= self.get_image_dataset(epoch_metadata, sampling_seed)
            
            # Start background data loading
            self.start_data_loading_thread(iter(image_dataloader))

            # Retrieve the current epoch's data
            image_hashes, image_uuids, image_tensors = next_epoch_data
            print_in_rank(f"number of images loaded: {len(image_hashes)}")

            train_dataset = UnclipDataset(image_hashes, image_uuids, image_tensors)
            train_dataloader = DataLoader(
                train_dataset,
                batch_size= self.micro_batch_size,
                collate_fn=collate_fn,
                num_workers= 5
            )

            data_iter = iter(train_dataloader)
            total_batches = len(train_dataloader)
            
            for i, batch in enumerate(data_iter, start=1):
                print_in_rank(f"Processing step {step} of {self.max_train_steps}")

                self.optimizer.zero_grad()
                image_hashes_batch = batch["image_hashes"]
                image_batch = batch["image_batch"].to(dtype=self.weight_dtype, device=device)

                loss = self.train_step(image_batch, device)
                print_in_rank(f"Computed loss: {loss.item()} for image hashes: {image_hashes_batch}")
                
                loss.backward()
                losses.append(loss.item())
                dist.barrier()

                if step % self.gradient_accumulation_steps == 0:
                    if hasattr(self.diffae, 'on_before_optimizer_step'):
                        self.diffae.on_before_optimizer_step(self.optimizer, 0)

                    self.optimizer.step()

                    if self.lr_warmup_steps > 0:
                        self.schedueler.step()
                    
                    # only apply ema on the last gradient accumulation step,
                    # if it is the iteration that has optimizer.step()
                    if self.conf.train_mode == TrainMode.latent_diffusion:
                        # it trains only the latent hence change only the latent
                        ema(self.diffae.model.latent_net, self.ema_model.latent_net, self.ema_decay)
                    else:
                        ema(self.diffae.model, self.ema_model, self.ema_decay)

                step += 1
                if step >= self.max_train_steps:
                    break
            
            epoch += 1
            
            # At the end of the epoch, fetch the next epoch's data
            while(self.next_epoch_data is None):
                print("data loading has not finished yet")
                time.sleep(5)
                
            next_epoch_data = self.next_epoch_data

            # If the loaded epoch data is empty, reset the dataset loader
            if len(next_epoch_data[0])==0:  # Check if there is no data left
                print("starting a new epoch")
                current_metadata = dataset_metadata.copy()
                # set sampling seed for the current epoch
                sampling_seed= set_sampling_seed(device)
                random.seed(sampling_seed)
                random.shuffle(current_metadata)

                # load the first psuedo epoch
                epoch_metadata = current_metadata[:self.epoch_size]
                current_metadata = current_metadata[self.epoch_size:]
                image_dataloader= self.get_image_dataset(epoch_metadata, sampling_seed)
                next_epoch_data= self.load_dataset(iter(image_dataloader))     

        print("Training complete.")
    
def parse_args():
    parser = argparse.ArgumentParser()

    # MinIO credentials
    parser.add_argument('--minio-access-key', type=str, required=True, help='Access key for model MinIO storage.')
    parser.add_argument('--minio-secret-key', type=str, required=True, help='Secret key for model MinIO storage.')

    # Training configuration
    parser.add_argument('--dataset', type=str, help='Name of the dataset used during training', required=True)
    parser.add_argument('--epoch-size', type=int, help='size of each epoch', default=1000)
    parser.add_argument('--model-seed', type=int, help='seed for model initialization', default=None)
    parser.add_argument('--weight-dtype', type=str, default='float32', help='Data type for weights, e.g., "float32".')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Initial learning rate.')
    parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 hyperparameter for optimizer.')
    parser.add_argument('--beta2', type=float, default=0.999, help='Beta2 hyperparameter for optimizer.')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='Weight decay for the optimizer.')
    parser.add_argument('--ema-decay', type=float, default=0.9999, help='Ema decay')
    parser.add_argument('--epsilon', type=float, default=1e-8, help='Epsilon value for the optimizer.')
    parser.add_argument('--lr-warmup-steps', type=int, default=500, help='Number of warmup steps for learning rate scheduler.')
    parser.add_argument('--max-train-steps', type=int, default=10000, help='Maximum number of training steps.')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1, help='Number of steps to accumulate gradients before updating model parameters.')
    parser.add_argument('--checkpointing-steps', type=int, default=1000, help='Frequency of checkpointing model weights.')
    parser.add_argument('--train-batch-size', type=int, default=1, help='Batch size for training.')
    parser.add_argument('--micro-batch-size', type=int, default=1, help='Micro batch size per gpu.')
    parser.add_argument('--save-results', action="store_true", default=False)
    parser.add_argument('--save-metrics', action="store_true", default=False)
    parser.add_argument('--finetune', action="store_true", default=False)
    parser.add_argument('--model-id', type=int, help='id of the model you want to finetune', default=None)
    parser.add_argument('--num-checkpoint', type=int, help='number of the checkpoint to load', default=None)
    parser.add_argument('--loading-workers', type=int, default=10, help='Number of workers used for downloading data from minio per gpu.')
    parser.add_argument('--image-resolution', type=int, default=512, help='Resolution of images used for training.')

    return parser.parse_args()

def setup_distributed(world_size: int):
    """ Initialize torch distributed training (NCCL) """
    # Set up distributed environment variables
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500" 
    # initialize NCCL backend
    dist.init_process_group(backend="nccl", world_size=world_size)
    # get local rank
    local_rank = int(os.environ["LOCAL_RANK"])
    return local_rank

def main():
    args= parse_args()

    # get minio client
    minio_client = minio_manager.get_minio_client(minio_ip_addr= "192.168.3.6:9000",
                                        minio_access_key=args.minio_access_key,
                                        minio_secret_key=args.minio_secret_key)
    
    # initialize distributed training with nccl
    world_size = torch.cuda.device_count()
    local_rank = setup_distributed(world_size)

    # run diffae training
    training_pipeline = DiffaeTrainingPipeline(minio_client=minio_client,
                                            local_rank = local_rank,
                                            world_size = world_size,
                                            dataset= args.dataset,
                                            finetune=args.finetune,
                                            model_seed = args.model_seed,
                                            model_id= args.model_id,
                                            num_checkpoint= args.num_checkpoint,
                                            weight_dtype= args.weight_dtype,
                                            learning_rate= args.learning_rate,
                                            beta1 = args.beta1,
                                            beta2 = args.beta2,
                                            weight_decay = args.weight_decay,
                                            ema_decay= args.ema_decay,
                                            epsilon = args.epsilon,
                                            lr_warmup_steps= args.lr_warmup_steps,
                                            epoch_size = args.epoch_size,
                                            max_train_steps = args.max_train_steps,
                                            gradient_accumulation_steps = args.gradient_accumulation_steps,
                                            checkpointing_steps = args.checkpointing_steps,
                                            micro_batch_size= args.micro_batch_size,
                                            save_results= args.save_results,
                                            loading_workers= args.loading_workers,
                                            image_resolution = args.image_resolution)

    # run training
    training_pipeline.train_diffae_model()
    # clean process group
    dist.destroy_process_group()

if __name__=="__main__":
    main()