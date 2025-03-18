import argparse
import copy
import json
import os, sys
import random
import re
import numpy as np
import pandas as pd
from minio import Minio
import pytorch_lightning as pl
import torch
from numpy.lib.function_base import flip
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import *
from torch import nn
from torch.cuda import amp
from torch.distributions import Categorical
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataset import ConcatDataset, TensorDataset
from torchvision.utils import make_grid, save_image
from torchvision.transforms import functional as VF

base_dir = "./"
sys.path.insert(0, base_dir)
sys.path.insert(0, os.getcwd())
from dataloaders.image_dataset_loader import HttpRequestHandler
from diffae.templates import ffhq256_autoenc
from utility.minio import minio_manager
from diffae.config import *
from diffae.dataset import *
from diffae.dist_utils import *
from diffae.lmdb_writer import *
from diffae.metrics import *
from diffae.renderer import *
from utility.path import separate_bucket_and_file_path

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

class ImageDataset(Dataset):
    def __init__(self, minio_client, image_paths, image_resolution=256, do_augment=True):
        """
        Args:
            minio_client: A MinIO client instance.
            image_paths (list of str): List of image paths (MinIO paths).
            image_resolution (int): Target resolution for images.
            do_augment (bool): Whether to apply random horizontal flip.
        """
        super().__init__()
        self.minio_client = minio_client
        self.image_paths = image_paths
        self.image_resolution = image_resolution
        self.do_augment = do_augment

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # Get the MinIO path for this image.
        image_path = self.image_paths[index]
        # Split the image_path into bucket name and object key.
        bucket_name, key = separate_bucket_and_file_path(image_path)
        response = self.minio_client.get_object(bucket_name, key)
        image_data = response.read()
        image = Image.open(BytesIO(image_data)).convert("RGB")
        
        # Downscale the image to the desired resolution.
        image = downscale_image(image, self.image_resolution)
        
        # Apply random horizontal flip with 50% probability.
        if self.do_augment and np.random.random() < 0.5:
            image = VF.hflip(image)
        
        # Convert image to tensor (range [0, 1]).
        image_tensor = VF.to_tensor(image)
        # Normalize to [-1, 1]:
        image_tensor = image_tensor * 2.0 - 1.0
        
        return {"img": image_tensor, "path": image_path, "index": index}

class SizedIterableWrapper:
    # The constructor accepts a dataloader and a length.
    # 'dataloader' can be any iterable (like a list, generator, etc.),
    # and 'length' represents the total number of items it is supposed to yield.
    def __init__(self, dataloader, length):
        self.dataloader = dataloader  # Store the provided dataloader
        self._length = length  # Store the provided length

    # The __iter__ method makes the object iterable.
    # It returns an iterator for the wrapped dataloader.
    def __iter__(self):
        return iter(self.dataloader)

    # The __len__ method returns the stored length.
    # This is useful when you need to know how many items the dataloader should yield.
    def __len__(self):
        return self._length

class LitModel(pl.LightningModule):
    def __init__(self, conf: TrainConfig, minio_client: Minio, dataset: str):
        super().__init__()

        self.minio_client = minio_client
        self.dataset = dataset
        assert conf.train_mode != TrainMode.manipulate
        if conf.seed is not None:
            pl.seed_everything(conf.seed)

        self.save_hyperparameters(conf.as_dict_jsonable())

        self.conf = conf

        # Initialize the model based on configuration
        self.model = conf.make_model_conf().make_model()

        # Create an Exponential Moving Average (EMA) copy of the model
        # EMA models typically produce higher quality samples
        self.ema_model = copy.deepcopy(self.model)
        self.ema_model.requires_grad_(False) # No gradients for EMA model
        self.ema_model.eval() # Always in evaluation mode

        # Print model size (number of parameters)
        model_size = 0
        for param in self.model.parameters():
            model_size += param.data.nelement()
        print('Model params: %.2f M' % (model_size / 1024 / 1024))

        # Set up diffusion samplers for training and evaluation
        self.sampler = conf.make_diffusion_conf().make_sampler()
        self.eval_sampler = conf.make_eval_diffusion_conf().make_sampler()

        # Timestep sampler (shared between model and latent diffusion)
        self.T_sampler = conf.make_T_sampler()

        # Set up latent diffusion samplers if using a latent network
        if conf.train_mode.use_latent_net():
            self.latent_sampler = conf.make_latent_diffusion_conf(
            ).make_sampler()
            self.eval_latent_sampler = conf.make_latent_eval_diffusion_conf(
            ).make_sampler()
        else:
            self.latent_sampler = None
            self.eval_latent_sampler = None

        # Register a fixed noise buffer for consistent sampling during evaluation
        # This ensures samples are comparable across training iterations
        self.register_buffer(
            'x_T',
            torch.randn(conf.sample_size, 3, conf.img_size, conf.img_size))

        #if conf.pretrain is not None:
        #    print(f'loading pretrain ... {conf.pretrain.name}')
        #    state = torch.load(conf.pretrain.path, map_location='cpu')
        #    print('step:', state['global_step'])
        #    self.load_state_dict(state['state_dict'], strict=False)

        if conf.pretrain is not None:  # Check if a pretrain configuration is provided
            print(
                f'loading pretrain ... {conf.pretrain.name}')  # Print the name of the pretrain configuration being loaded
            # Load the saved model state from the provided path.
            # 'map_location' is set to 'cpu' to move the loaded tensors to CPU.
            # 'weights_only=False' ensures the full state (not just the model weights) is loaded.
            state = torch.load(conf.pretrain.path, map_location='cpu', weights_only=False)
            print('step:', state['global_step'])  # Print the current global step from the loaded checkpoint state
            # Load the state dictionary into the model.
            # 'strict=False' allows for some keys in the model's state dict to be missing or extra.
            self.load_state_dict(state['state_dict'], strict=False)

        if conf.latent_infer_path is not None:
            print('loading latent stats ...')
            # same here, loading stuff
            state = torch.load(conf.latent_infer_path)
            self.conds = state['conds']
            self.register_buffer('conds_mean', state['conds_mean'][None, :])
            self.register_buffer('conds_std', state['conds_std'][None, :])
        else:
            self.conds_mean = None
            self.conds_std = None

    def normalize(self, cond):
        """
        Normalize conditional latent vectors using precomputed statistics.

        This standardizes latent vectors to zero mean and unit variance.
        """
        cond = (cond - self.conds_mean.to(self.device)) / self.conds_std.to(
            self.device)
        return cond

    def denormalize(self, cond):
        """
        Denormalize conditional latent vectors back to their original scale.

        This reverses the normalization process for generation or visualization.
        """
        cond = (cond * self.conds_std.to(self.device)) + self.conds_mean.to(
            self.device)
        return cond

    def sample(self, N, device, T=None, T_latent=None):
        """
        Generate new images using the diffusion model.

        Args:
            N: Number of samples to generate
            device: Device to generate on
            T: Optional override for number of diffusion steps
            T_latent: Optional override for number of latent diffusion steps

        Returns:
            Tensor of generated images in [0,1] range
        """

        # Choose samplers based on whether custom timestep counts are provided
        if T is None:
            sampler = self.eval_sampler
            latent_sampler = self.latent_sampler
        else:
            # Create new samplers with the specified number of timesteps
            sampler = self.conf._make_diffusion_conf(T).make_sampler()
            latent_sampler = self.conf._make_latent_diffusion_conf(T_latent).make_sampler()

        # Generate random noise as starting point for sampling
        noise = torch.randn(N,
                            3,
                            self.conf.img_size,
                            self.conf.img_size,
                            device=device)
        # Generate images through the unconditioned diffusion process
        pred_img = render_uncondition(
            self.conf,
            self.ema_model, # Use the EMA model for better quality
            noise, # Starting noise
            sampler=sampler,  # Pixel-space diffusion sampler
            latent_sampler=latent_sampler, # Latent-space diffusion sampler
            conds_mean=self.conds_mean,  # Statistics for latent normalization
            conds_std=self.conds_std,
        )
        pred_img = (pred_img + 1) / 2 # Convert from [-1,1] to [0,1] range for visualization
        return pred_img

    def render(self, noise, cond=None, T=None):
        if T is None:
            sampler = self.eval_sampler
        else:
            sampler = self.conf._make_diffusion_conf(T).make_sampler()

        if cond is not None:
            pred_img = render_condition(self.conf,
                                        self.ema_model,
                                        noise,
                                        sampler=sampler,
                                        cond=cond)
        else:
            pred_img = render_uncondition(self.conf,
                                          self.ema_model,
                                          noise,
                                          sampler=sampler,
                                          latent_sampler=None)
        pred_img = (pred_img + 1) / 2
        return pred_img

    def encode(self, x):
        # TODO:
        assert self.conf.model_type.has_autoenc()
        cond = self.ema_model.encoder.forward(x)
        return cond

    def encode_stochastic(self, x, cond, T=None):
        if T is None:
            sampler = self.eval_sampler
        else:
            sampler = self.conf._make_diffusion_conf(T).make_sampler()
        out = sampler.ddim_reverse_sample_loop(self.ema_model,
                                               x,
                                               model_kwargs={'cond': cond})
        return out['sample']

    def forward(self, noise=None, x_start=None, ema_model: bool = False):
        with amp.autocast(False):
            if ema_model:
                model = self.ema_model
            else:
                model = self.model
            gen = self.eval_sampler.sample(model=model,
                                           noise=noise,
                                           x_start=x_start)
            return gen

    def setup(self, stage=None) -> None:
        """
        make datasets & seeding each worker separately
        """
        ##############################################
        # NEED TO SET THE SEED SEPARATELY HERE
        if self.conf.seed is not None:
            seed = self.conf.seed * get_world_size() + self.global_rank
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            print('local seed:', seed)
        ##############################################

        self.train_data = self.conf.make_dataset()
        print('train data:', len(self.train_data))
        self.val_data = self.train_data
        print('val data:', len(self.val_data))

    def _train_dataloader(self, drop_last=True):
        """
           Really make the dataloader.
           """
        if not hasattr(self, "train_data"):
            self.setup('fit')
        if not hasattr(self, "train_data"):
            raise ValueError(
                "train_data is not initialized even after setup() call. Please ensure setup() properly initializes train_data."
            )

        # Clone the configuration and set the correct batch size.
        conf = self.conf.clone()
        conf.batch_size = self.batch_size

        # Create a DataLoader directly instead of make loader, picke issues and multiprocessing
        dataloader = torch.utils.data.DataLoader(
            self.train_data,
            batch_size=conf.batch_size,
            shuffle=True,
            drop_last=drop_last,
            num_workers=0,  # Use 0 on Windows to avoid pickling issues.
            persistent_workers=False
        )
        return dataloader

    def train_dataloader(self, drop_last=True):
        """
        Return the dataloader:
              - If in diffusion mode, return an image dataset.
              - If in latent mode, return the inferred latent dataset.
        """
        print('on train dataloader start ...')

        # get image metadata (file paths and hashes)
        request_handler= HttpRequestHandler("extracts", self.dataset, 0)
        dataset_metadata = request_handler.get_all_image_data()

        # get data
        image_paths = []
        for image_data in dataset_metadata:
            _, _, image_path = image_data
            image_paths.append(image_path)
        
        # get image dataset
        train_data = ImageDataset(self.minio_client, image_paths, image_resolution= self.conf.img_size)
        # Create a DataLoader directly instead of make loader, picke issues and multiprocessing
        dataloader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.conf.batch_size,
            shuffle=True,
            drop_last=drop_last,
            num_workers=12,
            persistent_workers=False
        )

        return dataloader

    @property
    def batch_size(self):
        """
        local batch size for each worker
        """
        ws = get_world_size()
        assert self.conf.batch_size % ws == 0
        return self.conf.batch_size // ws

    @property
    def num_samples(self):
        """
        (global) batch size * iterations
        """
        # batch size here is global!
        # global_step already takes into account the accum batches
        return self.global_step * self.conf.batch_size_effective

    def is_last_accum(self, batch_idx):
        """
        is it the last gradient accumulation loop? 
        used with gradient_accum > 1 and to see if the optimizer will
        perform "step" in this iteration or not
        """
        return (batch_idx + 1) % self.conf.accum_batches == 0

    def infer_whole_dataset(self,
                            with_render=False,
                            T_render=None,
                            render_save_path=None):
        """
        Encode the entire dataset into latent representations using the model's encoder.

        This is a crucial step for the second phase of training (latent diffusion),
        where we need the latent codes for all training images.

        Args:
            with_render: Whether to also generate images from the latent codes
            T_render: Number of diffusion timesteps for rendering
            render_save_path: Path to save the rendered images in LMDB format
        """
        # Set up the dataset with appropriate transforms
        data = self.conf.make_dataset()
        if isinstance(data, CelebAlmdb) and data.crop_d2c:
            # special case where we need the d2c crop
            data.transform = make_transform(self.conf.img_size,
                                            flip_prob=0,
                                            crop_d2c=True)
        else:
            data.transform = make_transform(self.conf.img_size, flip_prob=0)

        # data = SubsetDataset(data, 21)
        # Create a data loader for efficient batch processing
        loader = self.conf.make_loader(
            data,
            shuffle=False,
            drop_last=False,
            batch_size=self.conf.batch_size_eval,
            parallel=True,
        )

        # Use the EMA model for more stable results
        model = self.ema_model
        model.eval()
        conds = []

        # Set up rendering if requested
        if with_render:
            # Create a sampler with specified or default evaluation timesteps
            sampler = self.conf._make_diffusion_conf(
                T=T_render or self.conf.T_eval).make_sampler()
            # Only the main process (rank 0) writes images to disk
            if self.global_rank == 0:
                writer = LMDBImageWriter(render_save_path,
                                         format='webp',
                                         quality=100)
            else:
                writer = nullcontext() # No-op for other processes
        else:
            writer = nullcontext()

        # Process the entire dataset
        with writer:
            for batch in tqdm(loader, total=len(loader), desc='infer'):
                with torch.no_grad():
                    # (n, c)
                    # print('idx:', batch['index'])
                    # Encode images to latent space
                    cond = model.encoder(batch['img'].to(self.device))

                    # used for reordering to match the original dataset
                    # Get indices for proper ordering
                    idx = batch['index']
                    idx = self.all_gather(idx)  # Gather indices from all processes
                    if idx.dim() == 2:
                        idx = idx.flatten(0, 1) # Flatten distributed batch dimension
                    argsort = idx.argsort() # Get ordering to match original dataset

                    # Optionally render images from latent codes
                    if with_render:
                        # Generate random noise as starting point
                        noise = torch.randn(len(cond),
                                            3,
                                            self.conf.img_size,
                                            self.conf.img_size,
                                            device=self.device)

                        # Sample images using the diffusion model
                        render = sampler.sample(model, noise=noise, cond=cond)
                        render = (render + 1) / 2 # Convert from [-1,1] to [0,1] range
                        # print('render:', render.shape)
                        # (k, n, c, h, w)

                        # Gather rendered images from all processes
                        render = self.all_gather(render)
                        if render.dim() == 5:
                            # (k*n, c)
                            render = render.flatten(0, 1) # Flatten distributed batch dimension

                        # print('global_rank:', self.global_rank)
                        # Only the main process saves images
                        if self.global_rank == 0:
                            writer.put_images(render[argsort])

                    # (k, n, c)
                    # Gather latent codes from all processes
                    cond = self.all_gather(cond)

                    if cond.dim() == 3:
                        # (k*n, c)
                        cond = cond.flatten(0, 1) # Flatten distributed batch dimension

                    # Add to collection, ensuring correct ordering
                    conds.append(cond[argsort].cpu())
                # break
        model.train() # Reset model to training mode
        # (N, c) cpu

        # Concatenate all latent codes into a single tensor
        conds = torch.cat(conds).float()
        return conds

    def training_step(self, batch, batch_idx):
        """
        given an input, calculate the loss function
        no optimization at this stage.
        """
        with amp.autocast(False):
            # batch size here is local!
            # forward
            if self.conf.train_mode.require_dataset_infer():
                # this mode as pre-calculated cond
                cond = batch[0]
                if self.conf.latent_znormalize:
                    cond = (cond - self.conds_mean.to(
                        self.device)) / self.conds_std.to(self.device)
            else:
                imgs, idxs = batch['img'], batch['index']
                # print(f'(rank {self.global_rank}) batch size:', len(imgs))
                x_start = imgs

            if self.conf.train_mode == TrainMode.diffusion:
                """
                main training mode!!!
                """
                # with numpy seed we have the problem that the sample t's are related!
                t, weight = self.T_sampler.sample(len(x_start), x_start.device)
                losses = self.sampler.training_losses(model=self.model,
                                                      x_start=x_start,
                                                      t=t)
            elif self.conf.train_mode.is_latent_diffusion():
                """
                training the latent variables!
                """
                # diffusion on the latent
                t, weight = self.T_sampler.sample(len(cond), cond.device)
                latent_losses = self.latent_sampler.training_losses(
                    model=self.model.latent_net, x_start=cond, t=t)
                # train only do the latent diffusion
                losses = {
                    'latent': latent_losses['loss'],
                    'loss': latent_losses['loss']
                }
            else:
                raise NotImplementedError()

            loss = losses['loss'].mean()
            # divide by accum batches to make the accumulated gradient exact!
            for key in ['loss', 'vae', 'latent', 'mmd', 'chamfer', 'arg_cnt']:
                if key in losses:
                    losses[key] = self.all_gather(losses[key]).mean()

            if self.global_rank == 0:
                self.logger.experiment.add_scalar('loss', losses['loss'],
                                                  self.num_samples)
                for key in ['vae', 'latent', 'mmd', 'chamfer', 'arg_cnt']:
                    if key in losses:
                        self.logger.experiment.add_scalar(
                            f'loss/{key}', losses[key], self.num_samples)

        return {'loss': loss}

    # Change in PyLightning framework
    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        """
        after each training step ...
        """
        if self.is_last_accum(batch_idx):
            # only apply ema on the last gradient accumulation step,
            # if it is the iteration that has optimizer.step()
            if self.conf.train_mode == TrainMode.latent_diffusion:
                # it trains only the latent hence change only the latent
                ema(self.model.latent_net, self.ema_model.latent_net,
                    self.conf.ema_decay)
            else:
                ema(self.model, self.ema_model, self.conf.ema_decay)

            # logging
            if self.conf.train_mode.require_dataset_infer():
                imgs = None
            else:
                imgs = batch['img']
            self.log_sample(x_start=imgs)
            self.evaluate_scores()

    # Change in PyLightning framework
    def on_before_optimizer_step(self, optimizer: Optimizer,**kwargs) -> None:
        # fix the fp16 + clip grad norm problem with pytorch lightning
        # this is the currently correct way to do it
        if self.conf.grad_clip > 0:
            params = [p for group in optimizer.param_groups for p in group['params']]
            torch.nn.utils.clip_grad_norm_(params, max_norm=self.conf.grad_clip)

    def log_sample(self, x_start):
        """
        put images to the tensorboard
        """
        def do(model,
               postfix,
               use_xstart,
               save_real=False,
               no_latent_diff=False,
               interpolate=False):
            model.eval()
            with torch.no_grad():
                all_x_T = self.split_tensor(self.x_T)
                batch_size = min(len(all_x_T), self.conf.batch_size_eval)
                # allow for superlarge models
                loader = DataLoader(all_x_T, batch_size=batch_size)

                Gen = []
                for x_T in loader:
                    if use_xstart:
                        _xstart = x_start[:len(x_T)]
                    else:
                        _xstart = None

                    if self.conf.train_mode.is_latent_diffusion(
                    ) and not use_xstart:
                        # diffusion of the latent first
                        gen = render_uncondition(
                            conf=self.conf,
                            model=model,
                            x_T=x_T,
                            sampler=self.eval_sampler,
                            latent_sampler=self.eval_latent_sampler,
                            conds_mean=self.conds_mean,
                            conds_std=self.conds_std)
                    else:
                        if not use_xstart and self.conf.model_type.has_noise_to_cond(
                        ):
                            model: BeatGANsAutoencModel
                            # special case, it may not be stochastic, yet can sample
                            cond = torch.randn(len(x_T),
                                               self.conf.style_ch,
                                               device=self.device)
                            cond = model.noise_to_cond(cond)
                        else:
                            if interpolate:
                                with amp.autocast(self.conf.fp16):
                                    cond = model.encoder(_xstart)
                                    i = torch.randperm(len(cond))
                                    cond = (cond + cond[i]) / 2
                            else:
                                cond = None
                        gen = self.eval_sampler.sample(model=model,
                                                       noise=x_T,
                                                       cond=cond,
                                                       x_start=_xstart)
                    Gen.append(gen)

                gen = torch.cat(Gen)
                gen = self.all_gather(gen)
                if gen.dim() == 5:
                    # (n, c, h, w)
                    gen = gen.flatten(0, 1)

                if save_real and use_xstart:
                    # save the original images to the tensorboard
                    real = self.all_gather(_xstart)
                    if real.dim() == 5:
                        real = real.flatten(0, 1)

                    if self.global_rank == 0:
                        grid_real = (make_grid(real) + 1) / 2
                        self.logger.experiment.add_image(
                            f'sample{postfix}/real', grid_real,
                            self.num_samples)

                if self.global_rank == 0:
                    # save samples to the tensorboard
                    grid = (make_grid(gen) + 1) / 2
                    sample_dir = os.path.join(self.conf.logdir,
                                              f'sample{postfix}')
                    if not os.path.exists(sample_dir):
                        os.makedirs(sample_dir)
                    path = os.path.join(sample_dir,
                                        '%d.png' % self.num_samples)
                    save_image(grid, path)
                    self.logger.experiment.add_image(f'sample{postfix}', grid,
                                                     self.num_samples)
            model.train()

        if self.conf.sample_every_samples > 0 and is_time(
                self.num_samples, self.conf.sample_every_samples,
                self.conf.batch_size_effective):

            if self.conf.train_mode.require_dataset_infer():
                do(self.model, '', use_xstart=False)
                do(self.ema_model, '_ema', use_xstart=False)
            else:
                if self.conf.model_type.has_autoenc(
                ) and self.conf.model_type.can_sample():
                    do(self.model, '', use_xstart=False)
                    do(self.ema_model, '_ema', use_xstart=False)
                    # autoencoding mode
                    do(self.model, '_enc', use_xstart=True, save_real=True)
                    do(self.ema_model,
                       '_enc_ema',
                       use_xstart=True,
                       save_real=True)
                elif self.conf.train_mode.use_latent_net():
                    do(self.model, '', use_xstart=False)
                    do(self.ema_model, '_ema', use_xstart=False)
                    # autoencoding mode
                    do(self.model, '_enc', use_xstart=True, save_real=True)
                    do(self.model,
                       '_enc_nodiff',
                       use_xstart=True,
                       save_real=True,
                       no_latent_diff=True)
                    do(self.ema_model,
                       '_enc_ema',
                       use_xstart=True,
                       save_real=True)
                else:
                    do(self.model, '', use_xstart=True, save_real=True)
                    do(self.ema_model, '_ema', use_xstart=True, save_real=True)

    def evaluate_scores(self):
        """
        evaluate FID and other scores during training (put to the tensorboard)
        For, FID. It is a fast version with 5k images (gold standard is 50k).
        Don't use its results in the paper!
        """
        def fid(model, postfix):
            score = evaluate_fid(self.eval_sampler,
                                 model,
                                 self.conf,
                                 device=self.device,
                                 train_data=self.train_data,
                                 val_data=self.val_data,
                                 latent_sampler=self.eval_latent_sampler,
                                 conds_mean=self.conds_mean,
                                 conds_std=self.conds_std)
            if self.global_rank == 0:
                self.logger.experiment.add_scalar(f'FID{postfix}', score,
                                                  self.num_samples)
                if not os.path.exists(self.conf.logdir):
                    os.makedirs(self.conf.logdir)
                with open(os.path.join(self.conf.logdir, 'eval.txt'),
                          'a') as f:
                    metrics = {
                        f'FID{postfix}': score,
                        'num_samples': self.num_samples,
                    }
                    f.write(json.dumps(metrics) + "\n")

        def lpips(model, postfix):
            if self.conf.model_type.has_autoenc(
            ) and self.conf.train_mode.is_autoenc():
                # {'lpips', 'ssim', 'mse'}
                score = evaluate_lpips(self.eval_sampler,
                                       model,
                                       self.conf,
                                       device=self.device,
                                       val_data=self.val_data,
                                       latent_sampler=self.eval_latent_sampler)

                if self.global_rank == 0:
                    for key, val in score.items():
                        self.logger.experiment.add_scalar(
                            f'{key}{postfix}', val, self.num_samples)

        if self.conf.eval_every_samples > 0 and self.num_samples > 0 and is_time(
                self.num_samples, self.conf.eval_every_samples,
                self.conf.batch_size_effective):
            print(f'eval fid @ {self.num_samples}')
            lpips(self.model, '')
            fid(self.model, '')

        if self.conf.eval_ema_every_samples > 0 and self.num_samples > 0 and is_time(
                self.num_samples, self.conf.eval_ema_every_samples,
                self.conf.batch_size_effective):
            print(f'eval fid ema @ {self.num_samples}')
            fid(self.ema_model, '_ema')
            # it's too slow
            # lpips(self.ema_model, '_ema')

    def configure_optimizers(self):
        out = {}
        if self.conf.optimizer == OptimizerType.adam:
            optim = torch.optim.Adam(self.model.parameters(),
                                     lr=self.conf.lr,
                                     weight_decay=self.conf.weight_decay)
        elif self.conf.optimizer == OptimizerType.adamw:
            optim = torch.optim.AdamW(self.model.parameters(),
                                      lr=self.conf.lr,
                                      weight_decay=self.conf.weight_decay)
        else:
            raise NotImplementedError()
        out['optimizer'] = optim
        if self.conf.warmup > 0:
            sched = torch.optim.lr_scheduler.LambdaLR(optim,
                                                      lr_lambda=WarmupLR(
                                                          self.conf.warmup))
            out['lr_scheduler'] = {
                'scheduler': sched,
                'interval': 'step',
            }
        return out

    def split_tensor(self, x):
        """
        extract the tensor for a corresponding "worker" in the batch dimension

        Args:
            x: (n, c)

        Returns: x: (n_local, c)
        """
        n = len(x)
        rank = self.global_rank
        world_size = get_world_size()
        # print(f'rank: {rank}/{world_size}')
        per_rank = n // world_size
        return x[rank * per_rank:(rank + 1) * per_rank]

    def test_step(self, batch, *args, **kwargs):
        """
        Evaluation method that supports multiple evaluation protocols.

        The method parses the 'eval_programs' configuration list to determine
        which evaluations to run. Each program in the list specifies a different
        evaluation task.
        
        We just want the multi-gpu support. 
        """
        # Ensure proper seeding for reproducibility
        # make sure you seed each worker differently!
        self.setup()

        # ============ LATENT INFERENCE EVALUATION ============
        # it will run only one step!
        print('global step:', self.global_step)
        """
        "infer" = predict the latent variables using the encoder on the whole dataset
        """
        if 'infer' in self.conf.eval_programs:
            if 'infer' in self.conf.eval_programs:
                print('infer ...')
                # Run inference on the entire dataset
                conds = self.infer_whole_dataset().float()
                # NOTE: always use this path for the latent.pkl files
                # Save latents to standardized path
                save_path = f'checkpoints/{self.conf.name}/latent.pkl'
            else:
                raise NotImplementedError()

            # Only the main process (rank 0) saves the results
            if self.global_rank == 0:
                # Calculate statistics of the latent space
                conds_mean = conds.mean(dim=0)
                conds_std = conds.std(dim=0)

                # Create directory if needed
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))

                # Save latents and their statistics
                torch.save(
                    {
                        'conds': conds,
                        'conds_mean': conds_mean,
                        'conds_std': conds_std,
                    }, save_path)

        # ============ LATENT INFERENCE WITH RENDERING ============
        """
        "infer+render" = predict the latent variables using the encoder on the whole dataset
        THIS ALSO GENERATE CORRESPONDING IMAGES
        """
        # infer + reconstruction quality of the input
        # Program: "infer+render<T>" - Encode and also render the latents
        for each in self.conf.eval_programs:
            if each.startswith('infer+render'):
                # Parse timestep parameter using regex
                m = re.match(r'infer\+render([0-9]+)', each)
                if m is not None:
                    T = int(m[1])
                    self.setup()
                    print(f'infer + reconstruction T{T} ...')

                    # Run inference with rendering enabled
                    conds = self.infer_whole_dataset(
                        with_render=True,
                        T_render=T,
                        render_save_path=
                        f'latent_infer_render{T}/{self.conf.name}.lmdb',
                    )

                    # Save results
                    save_path = f'latent_infer_render{T}/{self.conf.name}.pkl'
                    conds_mean = conds.mean(dim=0)
                    conds_std = conds.std(dim=0)
                    if not os.path.exists(os.path.dirname(save_path)):
                        os.makedirs(os.path.dirname(save_path))
                    torch.save(
                        {
                            'conds': conds,
                            'conds_mean': conds_mean,
                            'conds_std': conds_std,
                        }, save_path)

        # ============ FID EVALUATION ============
        # evals those "fidXX"
        """
        "fid<T>" = unconditional generation (conf.train_mode = diffusion).
            Note:   Diff. autoenc will still receive real images in this mode.
        "fid<T>,<T_latent>" = unconditional generation for latent models (conf.train_mode = latent_diffusion).
            Note:   Diff. autoenc will still NOT receive real images in this made.
                    but you need to make sure that the train_mode is latent_diffusion.
        """
        for each in self.conf.eval_programs:
            if each.startswith('fid'):

                # Check for pattern: fid(T1,T2) - for latent diffusion evaluation
                m = re.match(r'fid\(([0-9]+),([0-9]+)\)', each)
                clip_latent_noise = False
                if m is not None:

                    # Two timestep parameters: pixel diffusion and latent diffusion
                    # eval(T1,T2)
                    T = int(m[1])
                    T_latent = int(m[2])
                    print(f'evaluating FID T = {T}... latent T = {T_latent}')
                else:

                    # Check for pattern: fidclip(T1,T2) - with latent noise clipping
                    m = re.match(r'fidclip\(([0-9]+),([0-9]+)\)', each)
                    if m is not None:
                        # fidclip(T1,T2)
                        T = int(m[1])
                        T_latent = int(m[2])
                        clip_latent_noise = True
                        print(
                            f'evaluating FID (clip latent noise) T = {T}... latent T = {T_latent}'
                        )
                    else:
                        # Simple pattern: fidT - just pixel diffusion
                        # evalT
                        _, T = each.split('fid')
                        T = int(T)
                        T_latent = None
                        print(f'evaluating FID T = {T}...')

                # Prepare data
                self.train_dataloader()

                # Create samplers with specified timesteps
                sampler = self.conf._make_diffusion_conf(T=T).make_sampler()
                if T_latent is not None:
                    latent_sampler = self.conf._make_latent_diffusion_conf(
                        T=T_latent).make_sampler()
                else:
                    latent_sampler = None

                # Set up evaluation configuration
                conf = self.conf.clone()
                conf.eval_num_images = 50_000 # Standard FID sample size

                # Calculate FID score
                score = evaluate_fid(
                    sampler,
                    self.ema_model,
                    conf,
                    device=self.device,
                    train_data=self.train_data,
                    val_data=self.val_data,
                    latent_sampler=latent_sampler,
                    conds_mean=self.conds_mean,
                    conds_std=self.conds_std,
                    remove_cache=False,
                    clip_latent_noise=clip_latent_noise,
                )

                # Log results with appropriate naming
                if T_latent is None:
                    self.log(f'fid_ema_T{T}', score)
                else:
                    name = 'fid'
                    if clip_latent_noise:
                        name += '_clip'
                    name += f'_ema_T{T}_Tlatent{T_latent}'
                    self.log(name, score)
        """
        "recon<T>" = reconstruction & autoencoding (without noise inversion)
        """

        # ============ RECONSTRUCTION EVALUATION ============

        # Program: "recon<T>" - Evaluate autoencoding reconstruction quality
        for each in self.conf.eval_programs:
            if each.startswith('recon'):
                self.model: BeatGANsAutoencModel
                _, T = each.split('recon')
                T = int(T)
                print(f'evaluating reconstruction T = {T}...')

                sampler = self.conf._make_diffusion_conf(T=T).make_sampler()

                conf = self.conf.clone()
                # eval whole val dataset
                # Evaluate on entire validation dataset
                conf.eval_num_images = len(self.val_data)
                # {'lpips', 'mse', 'ssim'}

                # Calculate multiple metrics (LPIPS, MSE, SSIM)
                score = evaluate_lpips(sampler,
                                       self.ema_model,
                                       conf,
                                       device=self.device,
                                       val_data=self.val_data,
                                       latent_sampler=None)

                # Log each metric separately
                for k, v in score.items():
                    self.log(f'{k}_ema_T{T}', v)

        # ============ NOISE INVERSION EVALUATION ============

        # Program: "inv<T>" - Evaluate reconstruction with noise inversion
        """
        "inv<T>" = reconstruction with noise inversion
        """
        for each in self.conf.eval_programs:
            if each.startswith('inv'):
                self.model: BeatGANsAutoencModel
                _, T = each.split('inv')
                T = int(T)
                print(
                    f'evaluating reconstruction with noise inversion T = {T}...'
                )

                sampler = self.conf._make_diffusion_conf(T=T).make_sampler()

                conf = self.conf.clone()
                # eval whole val dataset
                # Evaluate on entire validation dataset
                conf.eval_num_images = len(self.val_data)
                # {'lpips', 'mse', 'ssim'}
                # Calculate metrics with noise inversion enabled
                score = evaluate_lpips(sampler,
                                       self.ema_model,
                                       conf,
                                       device=self.device,
                                       val_data=self.val_data,
                                       latent_sampler=None,
                                       use_inverted_noise=True) # This enables noise inversion
                # Log results with distinct prefix
                for k, v in score.items():
                    self.log(f'{k}_inv_ema_T{T}', v)


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(target_dict[key].data * decay +
                                    source_dict[key].data * (1 - decay))


class WarmupLR:
    def __init__(self, warmup) -> None:
        self.warmup = warmup

    def __call__(self, step):
        return min(step, self.warmup) / self.warmup


def is_time(num_samples, every, step_size):
    closest = (num_samples // every) * every
    return num_samples - closest < step_size


def train(conf: TrainConfig, minio_client: Minio, dataset: str, gpus, nodes=1, mode: str = 'train'):
    print('conf:', conf.name)
    # assert not (conf.fp16 and conf.grad_clip > 0
    #             ), 'pytorch lightning has bug with amp + gradient clipping'
    model = LitModel(conf, minio_client, dataset)

    if not os.path.exists(conf.logdir):
        os.makedirs(conf.logdir)
    checkpoint = ModelCheckpoint(dirpath=f'{conf.logdir}',
                                 save_last=True,
                                 save_top_k=1,
                                 every_n_train_steps=conf.save_every_samples //
                                 conf.batch_size_effective)
    checkpoint_path = f'{conf.logdir}/last.ckpt'
    print('ckpt path:', checkpoint_path)
    if os.path.exists(checkpoint_path):
        resume = checkpoint_path
        print('resume!')
    else:
        if conf.continue_from is not None:
            # continue from a checkpoint
            resume = conf.continue_from.path
        else:
            resume = None

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=conf.logdir,
                                             name=None,
                                             version='')

    if len(gpus) == 1 and nodes == 1:
        accelerator = 'cuda'
        trainer_kwargs = {}
        plugins = None

    else:
        accelerator = 'ddp'
        # For PyTorch Lightning 2.x
        from pytorch_lightning.strategies import DDPStrategy

        # important for working with gradient checkpoint
        plugins = []  # Keep your existing plugins list initialization
        trainer_kwargs = {
            'strategy': DDPStrategy(find_unused_parameters=False)
        }

    use_dist_sampler = True if (len(gpus) >= 2 or nodes >= 2) else False

    trainer = pl.Trainer(
        max_steps=conf.total_samples // conf.batch_size_effective,
        devices=gpus,
        num_nodes=nodes,
        accelerator=accelerator,
        precision=16 if conf.fp16 else 32,
        callbacks=[
            checkpoint,
            LearningRateMonitor(),
        ],
        use_distributed_sampler=use_dist_sampler,
        logger=tb_logger,
        accumulate_grad_batches=conf.accum_batches,
        plugins=plugins if len(gpus) > 1 else None,
        **trainer_kwargs if 'trainer_kwargs' in locals() else {},  # Add this line
    )

    if mode == 'train':

        # Get the train dataloader from your model
        train_loader = model.train_dataloader()
        # If multiple loaders are returned, manually create a CombinedLoader and prime it.
        if isinstance(train_loader, (list, dict)):
            from pytorch_lightning.utilities.data import CombinedLoader

            combined = CombinedLoader(train_loader, mode="max_size_cycle")
            _ = iter(combined)  # This ensures internal state is set

        trainer.fit(model, ckpt_path=resume)

        #trainer.fit(model)

    elif mode == 'eval':
        # load the latest checkpoint
        # perform lpips
        # dummy loader to allow calling "test_step"
        dummy = DataLoader(TensorDataset(torch.tensor([0.] * conf.batch_size)),
                           batch_size=conf.batch_size)
        eval_path = conf.eval_path or checkpoint_path
        # conf.eval_num_images = 50
        print('loading from:', eval_path)
        state = torch.load(eval_path, map_location='cpu')
        print('step:', state['global_step'])
        model.load_state_dict(state['state_dict'])
        # trainer.fit(model)
        out = trainer.test(model, dataloaders=dummy)
        # first (and only) loader
        out = out[0]
        print(out)

        if get_rank() == 0:
            # save to tensorboard
            for k, v in out.items():
                tb_logger.experiment.add_scalar(
                    k, v, state['global_step'] * conf.batch_size_effective)

            # # save to file
            # # make it a dict of list
            # for k, v in out.items():
            #     out[k] = [v]
            tgt = f'evals/{conf.name}.txt'
            dirname = os.path.dirname(tgt)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            with open(tgt, 'a') as f:
                f.write(json.dumps(out) + "\n")
            # pd.DataFrame(out).to_csv(tgt)
    else:
        raise NotImplementedError()

def parse_args():
    parser = argparse.ArgumentParser()

    # MinIO credentials
    parser.add_argument('--minio-access-key', type=str, required=True, help='Access key for model MinIO storage.')
    parser.add_argument('--minio-secret-key', type=str, required=True, help='Secret key for model MinIO storage.')

    # Training configuration
    parser.add_argument('--dataset', type=str, help='Name of the dataset used during training', required=True)
    parser.add_argument('--micro-batch-size', type=int, default=1, help='Micro batch size per gpu.')
    parser.add_argument('--model-seed', type=int, help='seed for model initialization', default=None)
    parser.add_argument('--image-resolution', type=int, default=512, help='Resolution of images used for training.')

    return parser.parse_args()

def set_default_config(train_batch_size, model_seed, image_resolution):
    # initialization base model configuration
    conf = ffhq256_autoenc()

    # set model seed
    if model_seed:
        conf.seed = model_seed
    else:
        conf.seed = random.randint(0, 2 ** 24 - 1)
    
    # set batch size and input image resolution
    conf.batch_size = train_batch_size
    conf.img_size = image_resolution
    conf.model_conf.image_size = image_resolution

    return conf

def main():
    args= parse_args()

    # get minio client
    minio_client = minio_manager.get_minio_client(minio_ip_addr= "192.168.3.6:9000",
                                        minio_access_key=args.minio_access_key,
                                        minio_secret_key=args.minio_secret_key)
    
    # get device count
    world_size = torch.cuda.device_count()
    gpus = [i for i in range(world_size)]

    # set default configuration
    conf = set_default_config(args.micro_batch_size, args.model_seed, args.image_resolution)

    # run the training pipeline
    train(conf, minio_client, args.dataset, gpus=gpus)

if __name__=="__main__":
    main()