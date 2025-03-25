import argparse
from datetime import datetime
from datetime import timezone as tz
import enum
from hashlib import blake2b
import hashlib
import importlib.util
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
from safetensors.torch import load as safetensors_load
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
if importlib.util.find_spec("'distributed_shampoo'") is not None:
    from distributed_shampoo import AdamGraftingConfig, DistributedShampoo

base_dir = "./"
sys.path.insert(0, base_dir)
sys.path.insert(0, os.getcwd())
from dataloaders.proportional_sampling_loader import ProportionalSamplingLoader
from diffae.experiment import LitModel, WarmupLR, ema
from diffae.templates import ffhq256_autoenc
from utility.http import worker_request
from utility.minio import minio_manager
from utility.model_cards.model_card import ModelCard
from utility.path import separate_bucket_and_file_path
from training.utils import get_all_image_hashes, save_loss_per_image
from diffae.choices import LossType, ModelMeanType, OptimizerType, TrainMode
from diffae.model.nn import mean_flat
from utility.uuid64 import Uuid64

class ImageDataset(Dataset):
    def __init__(self, image_hashes, file_paths, tags):
        self.image_hashes = image_hashes
        self.tags = tags
        self.file_paths = file_paths
    
    def __len__(self):
        return len(self.image_hashes)
    
    def __getitem__(self, idx):
        return {
            'image_hash': self.image_hashes[idx],
            'file_path': self.file_paths[idx],
            'tag': self.tags[idx]
        }

class UnclipDataset(Dataset):
    def __init__(self, image_hashes, file_paths, tags, image_tensors):
        self.image_hashes = image_hashes
        self.file_paths = file_paths
        self.tags = tags
        self.image_tensors = image_tensors
    
    def __len__(self):
        return len(self.image_hashes)
    
    def __getitem__(self, idx):
        return {
            'image_hash': self.image_hashes[idx],
            'file_path': self.file_paths[idx],
            'tag': self.tags[idx],
            'image_tensor': self.image_tensors[idx]
        }

def collate_fn(batch):
    image_batch = torch.stack([item['image_tensor'] for item in batch])
    tags = [item['tag'] for item in batch]
    image_hashes = [item['image_hash'] for item in batch]
    file_paths = [item['file_path'] for item in batch]

    return {'image_hashes': image_hashes, 'file_paths': file_paths, 'tags': tags, 'image_batch': image_batch}

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
                 epoch_size,
                 tag_categories= None,
                 finetune=False,
                 model_seed= None,
                 model_id= None,
                 num_checkpoint= None,
                 optimizer_type = "adam",
                 max_preconditioner_dim = 8192,
                 precondition_frequency = 100,
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
                 image_resolution = 512,
                 tensorboard_log_dir= "output/tensorboard_logs/"):

        # get minio client and local rank
        self.minio_client= minio_client
        self.local_rank = local_rank
        self.world_size = world_size
 
        # hyperparameters
        self.epoch_size = epoch_size
        self.finetune= finetune
        self.model_seed = model_seed
        self.model_id= model_id
        self.num_checkpoint= num_checkpoint
        self.tag_categories = tag_categories
        self.optimizer_type = optimizer_type
        self.max_preconditioner_dim = max_preconditioner_dim
        self.precondition_frequency = precondition_frequency
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
        self.tensorboard_log_dir = tensorboard_log_dir

        # models
        self.diffae= None
        self.ema_model= None
        
        # loading thread
        self.data_loading_thread = None
        self.next_epoch_data = None

    def set_default_config(self):
        # initialization base model configuration
        self.conf = ffhq256_autoenc()
        # set necessary parameters
        self.conf.seed = self.model_seed
        self.conf.img_size = self.image_resolution
        self.conf.model_conf.image_size = self.image_resolution
    
    def initialize_model(self, device):
        # Model Initialization
        self.diffae = LitModel(self.conf).to(device)
        # load a checkpoint if finetuning an existing model
        if self.finetune:
            self.load_diffae_checkpoint(self.num_checkpoint)

        self.diffae.model= self.diffae.model.to(device)
        self.ema_model = self.diffae.ema_model.to(device)
        # wrap the diffae model with ddp
        self.diffae_model = DDP(self.diffae.model, device_ids=[device], output_device=device)

        # initialize optimize and scheduler
        if self.optimizer_type == "adam":
            self.optimizer = torch.optim.Adam(self.diffae_model.parameters(),
                                    lr=self.conf.lr,
                                    weight_decay=self.conf.weight_decay,
                                    fused=True,
                                    # foreach=False
            )
        elif self.optimizer_type == "adamw":
            self.optimizer = torch.optim.AdamW(self.diffae_model.parameters(),
                                    lr=self.conf.lr,
                                    weight_decay=self.conf.weight_decay,
                                    fused=True,
                                    # foreach=False
            )
        elif self.optimizer_type == "shampoo":
            self.optimizer = DistributedShampoo(
                            self.diffae_model.parameters(),
                            lr=self.conf.lr,
                            betas=(self.beta1, self.beta2),
                            epsilon= self.epsilon, 
                            weight_decay=self.conf.weight_decay,
                            max_preconditioner_dim= self.max_preconditioner_dim,
                            precondition_frequency= self.precondition_frequency,
                            use_decoupled_weight_decay=True,
                            grafting_config=AdamGraftingConfig(
                                beta2=self.beta2,
                                epsilon=self.epsilon,
                            ),
                        )
        else:
            raise NotImplementedError()

        if self.conf.warmup > 0:
            self.schedueler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=WarmupLR(self.conf.warmup)) 
    
    def load_diffae_checkpoint(self, num_checkpoint):
        # Extract bucket name and file path from minio_path
        checkpoint_path = f"{self.output_directory}/checkpoints/checkpoint_{num_checkpoint}.safetensors"
        bucket, checkpoint_path = separate_bucket_and_file_path(checkpoint_path) 

        # Download the checkpoint from MinIO
        print_in_rank(f"Downloading checkpoint from MinIO: {checkpoint_path} ...")
        response = self.minio_client.get_object(bucket, checkpoint_path)
        checkpoint_data = response.read()  # Read into memory
        
        # Load safetensors checkpoint into a dictionary
        print_in_rank("Loading checkpoint using safetensors...")
        checkpoint_dict = safetensors_load(checkpoint_data)

        # Load weights into model (diffae model + EMA model)
        print_in_rank("Updating model state...")
        self.diffae.model.load_state_dict(checkpoint_dict, strict=False)
        self.diffae.ema_model.load_state_dict(checkpoint_dict, strict=False)

        print_in_rank(f"Model loaded successfully from MinIO {checkpoint_path}")
    
    def get_image_dataset(self, image_metadata, sampling_seed):
        image_hashes, file_paths, tags = image_metadata
        train_dataset = ImageDataset(image_hashes, file_paths, tags)

        image_dataloader = DataLoader(
            train_dataset,
            batch_size= 1,
            # sampler= DistributedSampler(train_dataset, shuffle=True, drop_last=True, seed= sampling_seed),
            num_workers= 5 
        )

        return image_dataloader
    
    def load_images(self, image_hash, file_path, tag):
        # load vae and clip vectors
        bucket_name, image_path = separate_bucket_and_file_path(file_path)
        response = self.minio_client.get_object(bucket_name, image_path)
        image_data = response.read()
        image = Image.open(BytesIO(image_data))

        image= self.downscale_image(image, self.image_resolution)

        # Apply random horizontal flip
        if np.random.random() < 0.5:
            image = VF.hflip(image)

        # Convert to tensor
        image = VF.to_tensor(image)

        # Normalize to [-1, 1]
        image_tensor = image * 2. - 1.

        return image_hash, file_path, tag, image_tensor
    
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
                file_path = batch["file_path"][0]
                tag = batch["tag"][0]

                future= executor.submit(self.load_images, image_hash, file_path, tag)
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
        sorted_file_paths = [r[1] for r in results]
        sorted_tags = [r[2] for r in results]
        image_tensors = [r[3] for r in results]

        return sorted_image_hashes, sorted_file_paths, sorted_tags, image_tensors

    def load_epoch_data(self, dataloader):
        """Background data loading thread"""
        # Load the data
        image_hashes, file_paths, tags, image_tensors=self.load_dataset(dataloader)
        self.next_epoch_data= image_hashes, file_paths, tags, image_tensors

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
        t, weight = self.diffae.T_sampler.sample(image_batch.shape[0], device=device)
        noise = torch.randn_like(image_batch, device=device)
        x_t = self.diffae.sampler.q_sample(image_batch, t, noise=noise).to(device=device)
        
        terms = {'x_t': x_t}
        model_output = self.diffae_model.forward(
            x=x_t.detach(),
            t=self.diffae.sampler._scale_timesteps(t),
            x_start=image_batch.detach()
        ).pred
        
        target_types = {
            ModelMeanType.eps: noise,
        }
        target = target_types[self.diffae.sampler.model_mean_type]
        assert model_output.shape == target.shape == image_batch.shape

        if self.diffae.sampler.loss_type == LossType.mse:
            if self.diffae.sampler.model_mean_type == ModelMeanType.eps:
                # (n, c, h, w) => (n, )
                terms["mse"] = mean_flat((target - model_output)**2)
            else:
                raise NotImplementedError()
        elif self.diffae.sampler.loss_type == LossType.l1:
                # (n, c, h, w) => (n, )
                terms["mse"] = mean_flat((target - model_output).abs())
        else:
            raise NotImplementedError()
            
        if "vb" in terms:
            # if learning the variance also use the vlb loss
            terms["loss"] = terms["mse"] + terms["vb"]
        else:
            terms["loss"] = terms["mse"]

        return terms

    def train_diffae_model(self):
        # set device for each process
        torch.cuda.set_device(self.local_rank)
        device = torch.device("cuda", self.local_rank)
        
        # set seed for model generation
        self.model_seed = set_model_seed(device, self.model_seed)
        # set default configuration for the model
        self.set_default_config()
        
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
                model_uuid, sequence_num = create_model_id(dataset="proportional_sampling", hyperparameters=self.conf.serialize())

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

        # load the diffae base model in each gpu and the optimizer and scheduler
        self.initialize_model(device)
        dist.barrier()

        # set up the proportional sampling dataset loader
        dataset_loader= ProportionalSamplingLoader(minio_client=self.minio_client, image_resolution=self.image_resolution)
        
        # load tag scores
        num_tags, total_images = dataset_loader.load_tag_scores(prefixes= self.tag_categories)
        print(f"{total_images} total images found across {num_tags} tags.")

        # check of epoch size is at least larger then the number of tags 
        if self.epoch_size < num_tags:
            raise Exception(f"The epoch size isn't large enough for proportional sampling, should be at least {num_tags}")
        
        # set sampling seed for the current epoch
        sampling_seed= set_sampling_seed(device)
        random.seed(sampling_seed)

        next_epoch_metadata = dataset_loader.get_pseudo_epoch(self.epoch_size, random_seed= sampling_seed)
        image_dataloader= self.get_image_dataset(next_epoch_metadata, sampling_seed)
        next_epoch_data=self.load_dataset(iter(image_dataloader))

        dist.barrier()
        print("finished loading images")
         
        self.diffae.train()
        
        epoch=1
        losses = []
        used_images= set()

        # initialize tensorobard summary writer
        if dist.get_rank() == 0:
            if self.finetune:
                print("Downloading the monitor log files from the previous checkpoint....")
                # download the tensorboard log files
                self.download_checkpoint_tensorboard_logs()

            tensorboard_writer = SummaryWriter(log_dir=f"{self.tensorboard_log_dir}{self.model_id}")
        dist.barrier()

        if self.finetune:
            num_checkpoint = self.num_checkpoint
            step, k_images= self.get_last_checkpoint_step(num_checkpoint)
            initial_step= step
        else:
            step, initial_step, k_images, num_checkpoint = 0,0,0,0
        
        while step < self.max_train_steps:
            # get next epoch data
            next_epoch_metadata = dataset_loader.get_pseudo_epoch(self.epoch_size, random_seed=sampling_seed)
            image_dataloader= self.get_image_dataset(next_epoch_metadata, sampling_seed)
            # Start background data loading
            self.start_data_loading_thread(iter(image_dataloader))

            # Retrieve the current epoch's data
            image_hashes, file_paths, tags, image_tensors = next_epoch_data
            print_in_rank(f"number of images loaded: {len(image_hashes)}")

            train_dataset = UnclipDataset(image_hashes, file_paths, tags, image_tensors)
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
                image_path_batch = batch["file_paths"]
                k_images += len(image_hashes_batch) * self.world_size
                image_batch = batch["image_batch"].to(device=device)
                
                tensor_hashes = []
                for image_tensor in image_batch:
                    tensor_hash= hashlib.md5(image_tensor.cpu().numpy().tobytes()).hexdigest()
                    tensor_hashes.append(tensor_hash)
                    used_images.add(tensor_hash) 

                terms = self.train_step(image_batch, device)
                loss = terms['loss'].mean()
                print_in_rank(f"Computed loss: {loss.item()} for images: {tensor_hashes}")

                # Log loss to TensorBoard (Only on Rank 0)
                if dist.get_rank() == 0:
                    tensorboard_writer.add_scalar("Loss/Step", loss.item(), step)
                    tensorboard_writer.add_scalar("Loss/k_images", loss.item(), k_images)

                # for image_path in image_path_batch:
                #     used_images.add(image_path) 
                
                # Save model periodically
                if ((step - initial_step) % self.checkpointing_steps == 0 or step == self.max_train_steps) and self.save_results:
                    if dist.get_rank() == 0:    
                        # save the checkpoint and update the path in mongo
                        state_dict, _= self.to_safetensors(self.diffae_model.module)
                        model_info = self.save_model_to_safetensor(state_dict, num_checkpoint)

                        # generate a model card and a loss curve graph
                        if step > initial_step or (not self.finetune):
                            self.save_model_card(model_info, sequence_num, num_checkpoint, step, k_images, self.checkpointing_steps)
                            # save the monitoring files to minio
                            self.save_monitoring_files(num_checkpoint)

                    num_checkpoint += 1
                    dist.barrier()
                
                loss.backward()
                losses.append(loss.item())

                if step % self.gradient_accumulation_steps == 0:
                    if hasattr(self.diffae, 'on_before_optimizer_step'):
                        self.diffae.on_before_optimizer_step(self.optimizer)

                    self.optimizer.step()

                    if self.lr_warmup_steps > 0:
                        self.schedueler.step()
                    
                    # only apply ema on the last gradient accumulation step,
                    # if it is the iteration that has optimizer.step()
                    if self.conf.train_mode == TrainMode.latent_diffusion:
                        # it trains only the latent hence change only the latent
                        ema(self.diffae_model.module.latent_net, self.ema_model.latent_net, self.ema_decay)
                    else:
                        ema(self.diffae_model.module, self.ema_model, self.ema_decay)

                step += 1
                if step >= self.max_train_steps:
                    break
            
            total_unique_images = self.get_total_used_images(used_images)
            if dist.get_rank() == 0:
                print(f"Unique images trained on (epoch {epoch}): {total_unique_images}/{total_images}")
            
            epoch += 1

            # At the end of the epoch, fetch the next epoch's data
            while(self.next_epoch_data is None):
                print("data loading has not finished yet")
                time.sleep(5)
                
            next_epoch_data = self.next_epoch_data

            # If the loaded epoch data is empty, reset the dataset loader
            if len(next_epoch_data[0])==0:  # Check if there is no data left
                print("starting a new epoch")
                used_images= set()
                dataset_loader.used_image_hashes = set()
                # set sampling seed for the current epoch
                sampling_seed= set_sampling_seed(device)
                random.seed(sampling_seed)

                # load the first psuedo epoch
                next_epoch_metadata = dataset_loader.get_pseudo_epoch(self.epoch_size, random_seed= sampling_seed)
                image_dataloader= self.get_image_dataset(next_epoch_metadata, sampling_seed)
                next_epoch_data=self.load_dataset(iter(image_dataloader))

                dist.barrier()    

        # Close TensorBoard Writer
        if dist.get_rank() == 0:
            tensorboard_writer.close()

        print("Training complete.")
    
    def get_total_used_images(self, used_images):
        # Convert set to list
        used_images_list = list(used_images)

        # Use all_gather_object to collect sets from all ranks
        gathered_lists = [None] * self.world_size
        dist.all_gather_object(gathered_lists, used_images_list)
        
        global_used_images = set()
        if dist.get_rank() == 0:
            # Merge sets from all ranks and count unique images
            for images in gathered_lists:
                global_used_images.update(images)  # Merge into one set

        dist.barrier()

        return len(global_used_images)

    def save_monitoring_files(self, num_checkpoint: int):
        # get output directory
        bucket, output_directory= separate_bucket_and_file_path(self.output_directory)

        # Upload TensorBoard logs
        tensorboard_logs_path = os.path.join(self.tensorboard_log_dir, str(self.model_id))
        for root, _, files in os.walk(tensorboard_logs_path):
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, tensorboard_logs_path)
                minio_path = os.path.join(output_directory, f"tensorboard_logs/checkpoint_{num_checkpoint}", relative_path)
                
                # Read file content into BytesIO
                with open(file_path, "rb") as f:
                    file_data = BytesIO(f.read())
                    minio_manager.upload_data(self.minio_client, bucket, minio_path, file_data)

    def get_last_checkpoint_step(self, checkpoint:int):
        print("loading checkpoint steps")
        # load checkpoint info
        model_card= ModelCard(self.minio_client)
        model_info= model_card.load_checkpoint_model_card("diffae", self.model_id, checkpoint)
        # get batch size and checkpointing steps
        current_step= model_info["step"]
        current_k_images = model_info["k_images"]
        
        return current_step, current_k_images

    def save_model_card(self, model_info, model_id, num_checkpoint, current_step, k_images, checkpointing_steps):
        """Generate a model card for a checkpoint."""

        model_size = model_info['size']
        model_uuid= Uuid64._create_random_value_with_date(datetime.now(tz.utc))

        model_card = {
            "model_id": model_uuid,
            "model_name": "diffae",
            "model_size": model_size,
            "model_seed": self.model_seed,
            "num_checkpoint": num_checkpoint,
            "step": current_step,
            "k_images": k_images,
            "checkpointing_steps":checkpointing_steps,
            "model_file_list":[
                {
                "minio_file_path": model_info["minio_file_path"],
                "file_path": model_info["file_path"],
                "hash": model_info["hash"],
                "size": model_size
                }
            ]
        }
        
        # add hyperparameters to the model card
        model_card["hyperparameters"]= self.conf.serialize()
        # initialize the model card class
        card= ModelCard(self.minio_client)
        # save the checkpoint model card
        card.save_checkpoint_model_card(model_card, model_id, num_checkpoint)

    def to_safetensors(self, model):

        metadata = {
            "weight_dtype": "float32" if self.weight_dtype==torch.float32 else "float16",
            "learning_rate": "{}".format(self.learning_rate),
            "beta1": "{}".format(self.beta1),
            "beta2": "{}".format(self.beta2),
            "weight_decay": "{}".format(self.weight_decay),
            "epsilon": "{}".format(self.epsilon),
            "max_train_steps": "{}".format(self.max_train_steps),
            "gradient_accumulation_steps": "{}".format(self.gradient_accumulation_steps),
            "checkpointing_steps": "{}".format(self.checkpointing_steps),
            "lr_warmup_steps": "{}".format(self.lr_warmup_steps),
            "micro_batch_size": "{}".format(self.micro_batch_size),
            "image_resolution": "{}".format(self.image_resolution),
        }

        model_dict = {name: param.clone().detach().cpu() for name, param in model.state_dict().items()}

        return model_dict, metadata

    def save_model_to_safetensor(self,
                                state_dict,
                                num_checkpoint):
        # Step 1: Convert the state_dict to SafeTensors
        file_name = f"checkpoint_{num_checkpoint}.safetensors"
        bucket, output_dir = separate_bucket_and_file_path(self.output_directory)
        safetensor_path = f"{output_dir}/checkpoints/{file_name}"

        print(f"Saving to SafeTensors format...")
        buffer = BytesIO()
        safetensors_buffer = safetensors_save(tensors=state_dict)
        buffer.write(safetensors_buffer)
        buffer.seek(0)

        # Step 2: Compute metadata (size and hash)
        model_size_mb = len(safetensors_buffer) / (1024 * 1024)
        hasher = blake2b(digest_size=20)
        hasher.update(safetensors_buffer)
        model_hash = hasher.hexdigest()

        # Step 3: Upload the SafeTensors file to MinIO
        print(f"Uploading SafeTensors file to MinIO at {safetensor_path}...")
        minio_manager.upload_data(self.minio_client, bucket, safetensor_path, buffer)

        # Step 4: Return model information
        model_info = {
            "minio_file_path": f"{bucket}/{safetensor_path}",
            "file_path": f"models/diffae/diffusion_pytorch_model.safetensors",
            "hash": model_hash,
            "size": f"{model_size_mb:.2f} MB"
        }

        print(f"Model saved and uploaded successfully.")
        return model_info
    
    def download_checkpoint_tensorboard_logs(self):
        tensorboard_model_dir= os.path.join(self.tensorboard_log_dir, str(self.model_id))
        # empty the tensorboard log from any previous files
        if os.listdir(tensorboard_model_dir):
            shutil.rmtree(tensorboard_model_dir)
            os.makedirs(tensorboard_model_dir, exist_ok=True)

        # get tensorobard log minio path
        bucket , model_directory= separate_bucket_and_file_path(self.output_directory)
        tensorboard_log_directory= f"{model_directory}/tensorboard_logs/checkpoint_{self.num_checkpoint}/events.out.tfevents."
        tensorboard_log_files= minio_manager.get_list_of_objects_with_prefix(self.minio_client, bucket, tensorboard_log_directory)

        # download the tensorboard files locally
        for file in tensorboard_log_files:
            file_name= file.split("/")[-1]
            local_tensorboard_log_path= os.path.join(tensorboard_model_dir, file_name)
            self.minio_client.fget_object(bucket, file, local_tensorboard_log_path)
            print(f"Downloaded tensorboard file to {local_tensorboard_log_path}")

    
    @staticmethod
    def downscale_image(image: Image.Image, target_size: int, scale_factor: float = 2.0):
        """
        Downscales a square image incrementally by a scale factor until it reaches the lowest 
        resolution greater than the target size, then scales directly to the target size.
        
        Args:
            image (PIL.Image.Image): The input image (assumed to be square).
            target_size (int): The final target size (e.g., 128).
            scale_factor (float): The factor by which to downscale at each step (default is 2.0).
        
        Returns:
            PIL.Image.Image: The downscaled image.
        """
        current_size = image.size[0]  # Assume the image is square
        resized_image = image

        # Generate intermediate sizes based on the scale factor
        while current_size / scale_factor > target_size:
            current_size = int(current_size / scale_factor)
            resized_image = resized_image.resize((current_size, current_size), Image.LANCZOS)

        # Final resize to the exact target size
        resized_image = resized_image.resize((target_size, target_size), Image.LANCZOS)

        return resized_image

def parse_args():
    parser = argparse.ArgumentParser()

    # MinIO credentials
    parser.add_argument('--minio-access-key', type=str, required=True, help='Access key for model MinIO storage.')
    parser.add_argument('--minio-secret-key', type=str, required=True, help='Secret key for model MinIO storage.')

    # Training configuration
    parser.add_argument('--tag-categories', type=str, help="list of tag categories separated by comma, like (perspective,style,game)", default=None)
    parser.add_argument('--epoch-size', type=int, help='size of each epoch', default=1000)
    parser.add_argument('--model-seed', type=int, help='seed for model initialization', default=None)
    parser.add_argument('--optimizer-type', type=str, default='adam', help='Name of the optimizer to use.')
    parser.add_argument('--max-preconditioner-dim', type=int, default=8192, help='Max preconditioning dimension used for shampoo.')
    parser.add_argument('--precondition-frequency', type=int, default=100, help='Number of steps to apply preconditioner.')
    parser.add_argument('--weight-dtype', type=str, default='float32', help='Data type for weights, e.g., "float32".')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Initial learning rate.')
    parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 hyperparameter for optimizer.')
    parser.add_argument('--beta2', type=float, default=0.999, help='Beta2 hyperparameter for optimizer.')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='Weight decay for the optimizer.')
    parser.add_argument('--ema-decay', type=float, default=0.9999, help='Ema decay')
    parser.add_argument('--epsilon', type=float, default=1e-8, help='Epsilon value for the optimizer.')
    parser.add_argument('--lr-warmup-steps', type=int, default=1, help='Number of warmup steps for learning rate scheduler.')
    parser.add_argument('--max-train-steps', type=int, default=10000, help='Maximum number of training steps.')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1, help='Number of steps to accumulate gradients before updating model parameters.')
    parser.add_argument('--checkpointing-steps', type=int, default=1000, help='Frequency of checkpointing model weights.')
    parser.add_argument('--micro-batch-size', type=int, default=1, help='Micro batch size per gpu.')
    parser.add_argument('--save-results', action="store_true", default=False)
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
    
    # get tag categories
    tag_categories= None
    if args.tag_categories:
        tag_categories= args.tag_categories.split(",")
    
    # initialize distributed training with nccl
    world_size = torch.cuda.device_count()
    local_rank = setup_distributed(world_size)

    # run diffae training
    training_pipeline = DiffaeTrainingPipeline(minio_client=minio_client,
                                            local_rank = local_rank,
                                            world_size = world_size,
                                            tag_categories= tag_categories,
                                            finetune=args.finetune,
                                            model_seed = args.model_seed,
                                            model_id= args.model_id,
                                            num_checkpoint= args.num_checkpoint,
                                            optimizer_type = args.optimizer_type,
                                            max_preconditioner_dim = args.max_preconditioner_dim,
                                            precondition_frequency= args.precondition_frequency,
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