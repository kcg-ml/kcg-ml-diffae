
## Variables
 - **x**: Input tensor, size **3 × 256 × 256**
 - **xT**: Transformed input tensor, size **3 × 256 × 256** 
 -  **Zsem**: Semantic representation vector, size **512**
## Setup Instructions
 ### Create and Activate Environment
  ```bash
conda create -p /opt/envs/diffae python=3.11 conda-forge::mamba -y

conda activate /opt/envs/diffae

conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y

mamba install conda-forge::pytorch-lightning==1.5.10 -y

mamba install -c conda-forge ipywidgets ipykernel ipython -y

python -m ipykernel install --name=diffae --display-name "diffae"

mamba install -c conda-forge matplotlib transformers einops omegaconf wandb -y

mamba install -c conda-forge git -y
```