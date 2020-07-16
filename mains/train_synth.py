import os
import torch
import numpy as np
import wandb

from models.data_generators import DataGeneratorPCA

from utils.train import train_models
from configs.utils import create_model_from_config, create_metric_config, update_config
from configs.synth import optimal_lrs

# if you don't wish to create a wandb sweep, you can directly edit the following parameters
# - hdim: hidden dimension, one of (2, 5, 10, 20, 50, 100, 200, 300, 400, 500)
# - model_name: one of ('non_uniform_sum', 'rotation', 'nd', 'nd_exp', 'vae')
# - optimizer: one of ('SGD', 'Adam')
default_hparams = dict(
    hdim=None,
    model_name=None,
    optimizer=None,
    train_itr=50000,
    seed=1234
)

wandb.init(project='linear-ae-neurips', config=default_hparams)

config = update_config(optimal_lrs)

# set random seed
np.random.seed(config.seed)
torch.manual_seed(config.seed)

ckpt_dir = os.path.join('paper_exp', 'results', 'fig4', 'synth_1k',
                        'hdim{}'.format(config.hdim), 'lr{}'.format(config.lr), config.optimizer)
os.makedirs(ckpt_dir, exist_ok=True)


input_dim = 1000
hidden_dim = config.hdim

n_data = 5000
batch_size = n_data

max_sv = float(input_dim) * 0.1
min_sv = 1.0
sigma = 0.5

gt_data = DataGeneratorPCA(input_dim, hidden_dim, min_sv=min_sv, max_sv=max_sv, total=n_data)
data = DataGeneratorPCA(input_dim, hidden_dim, load_data=gt_data.x_sample)

loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False)

# non-uniform L2 regularized LAE
reg_min = 0.1
reg_max = min(float(input_dim - hidden_dim - 1) * 0.1, input_dim * 0.1 * 0.1)

reg_list = list(np.linspace(reg_min, reg_max, num=hidden_dim))

init_scale = 0.0001
train_itr = config.train_itr

# create model config
model_config = create_model_from_config(config, input_dim,
                                        init_scale=init_scale, reg_min=reg_min, reg_max=reg_max)

# define metrics
metric_config, eval_metrics_list = create_metric_config(data, loader)

train_stats, _ = train_models(loader, train_itr, metric_config,
                              model_configs=[model_config],
                              eval_metrics_list=eval_metrics_list,
                              ckpt_dir=ckpt_dir)
train_stats.close()
