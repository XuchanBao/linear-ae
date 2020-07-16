import os
import torch
import torchvision
import numpy as np
import wandb

from models.data_generators import DataGeneratorPCA
from utils.train import train_models
from configs.utils import create_model_from_config, create_metric_config, update_config
from configs.mnist import optimal_lrs

# if you don't wish to create a wandb sweep, you can directly edit the following parameters
# - model_name: one of ('uniform_sum', 'non_uniform_sum', 'rotation', 'nd', 'nd_exp', 'vae')
# - optimizer: one of ('SGD', 'Adam')
default_hparams = dict(
    hdim=20,
    model_name=None,
    optimizer=None,
    train_itr=30000,
    seed=1234
)

wandb.init(project='linear-ae-neurips', config=default_hparams)

config = update_config(optimal_lrs)

# set random seed
np.random.seed(config.seed)
torch.manual_seed(config.seed)

# uncomment the following to save checkpoints
ckpt_dir = os.path.join('results', 'mnist',
                        'hdim{}'.format(config.hdim), 'lr{}'.format(config.lr), config.optimizer)
os.makedirs(ckpt_dir, exist_ok=True)

# Get MNIST data
input_dim = 28 * 28

mnist_data = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                  transform=torchvision.transforms.Compose([
                                      torchvision.transforms.ToTensor()
                                  ]))
# full batch
batch_size = len(mnist_data)

mnist_loader = torch.utils.data.DataLoader(mnist_data, batch_size=batch_size, shuffle=True)

_, (raw_data, __) = next(enumerate(mnist_loader))
raw_data = torch.squeeze(raw_data.view(-1, input_dim))

# Center the data, and find ground truth principle directions
data_mean = torch.mean(raw_data, dim=0)
centered_data = raw_data - data_mean

data = DataGeneratorPCA(input_dim, config.hdim, load_data=centered_data.numpy())
data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False)

# non-uniform L2 regularization parameters
reg_min = 0.1
reg_max = 0.9

init_scale = 0.0001
train_itr = config.train_itr

# create model config
model_config = create_model_from_config(config, input_dim,
                                        init_scale=init_scale, reg_min=reg_min, reg_max=reg_max)

# define metrics
metric_config, eval_metrics_list = create_metric_config(data, data_loader)

train_stats_hdim, _ = train_models(
    data_loader, train_itr, metric_config,
    model_configs=[model_config], eval_metrics_list=eval_metrics_list,
    ckpt_dir=ckpt_dir
)

train_stats_hdim.close()
