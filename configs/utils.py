import torch
import numpy as np
import wandb
import ipdb
from models.linear_autoencoder import LinearAE
from models.linear_ae_nested_dropout import LinearAENestedDropout
from models.linear_vae import LinearVAE
from models.model_config import ModelTypes, ModelConfig
from utils.metrics import metric_alignment, metric_transpose_theorem, metric_subspace, metric_loss, metric_recon_loss
from utils.optimizers import CpRMSprop, MyRMSprop


def create_model_from_config(config, input_dim, init_scale=0.0001, reg_min=0.1, reg_max=0.9):
    if config.optimizer == 'Adam':
        optim_class = torch.optim.Adam
        extra_optim_args = {}
    elif config.optimizer == 'SGD':
        optim_class = torch.optim.SGD
        extra_optim_args = {'momentum': 0.9, 'nesterov': True}
    elif config.optimizer == 'RMSprop':
        optim_class = torch.optim.RMSprop
        extra_optim_args = {}
    elif config.optimizer == 'Cp_RMSprop':
        optim_class = CpRMSprop
        extra_optim_args = {}
    elif config.optimizer == 'RMSprop_grad_acc' or config.optimizer == 'RMSprop_rotation_acc':
        optim_class = MyRMSprop
        extra_optim_args = {'grad_type': config.optimizer}
    else:
        raise ValueError('config parameter "optimizer" needs to be one of ("Adam", "SGD")')

    model_name = config.model_type

    # non-uniform L2
    if config.model_type == ModelTypes.UNIFORM_SUM:
        model_class = LinearAE
        reg_list = list(np.linspace(reg_min, reg_max, num=config.hdim))
        extra_model_args = {"weight_reg_type": "uniform_sum", "l2_reg_list": [np.mean(reg_list)] * config.hdim}

    elif config.model_type == ModelTypes.NON_UNIFORM_SUM:
        model_class = LinearAE
        reg_list = list(np.linspace(reg_min, reg_max, num=config.hdim))
        extra_model_args = {"weight_reg_type": "non_uniform_sum", "l2_reg_list": reg_list}

    # rotation
    elif config.model_type == ModelTypes.ROTATION:
        model_class = LinearAE
        extra_model_args = {"weight_reg_type": None}

    # nested_dropout
    elif config.model_type == ModelTypes.NESTED_DROPOUT:
        model_class = LinearAENestedDropout
        extra_model_args = {'use_expectation': config.nd_expectation == "true"}
        if config.nd_expectation == 'true':
            model_name = 'nd_expectation'
        else:
            model_name = 'nd'

    # VAE
    elif config.model_type == ModelTypes.VAE:
        model_class = LinearVAE
        extra_model_args = {"use_analytic_elbo": True}
    else:
        raise ValueError('invalid config parameter "model_type"')

    model_config = ModelConfig(
        model_name, model_type=config.model_type,
        model_class=model_class, input_dim=input_dim, hidden_dim=config.hdim, init_scale=init_scale,
        extra_model_args=extra_model_args,
        optim_class=optim_class, lr=config.lr,
        extra_optim_args=extra_optim_args
    )
    return model_config


def create_metric_config(data, data_loader):
    # define metrics
    metric_config = {
        "transpose": {
            "func": metric_transpose_theorem,
            "ylabel": "$||W_1 - W_2||_F^2$",
            "yscale": "log",
            "title": "Transpose"
        },
        "axis-alignment": {
            "func": lambda m: metric_alignment(m, data.eigvectors),
            "ylabel": "Distance to axes alignment",
            "yscale": "linear",
            "title": "Axis-alignment"
        },
        "subspace-distance": {
            "func": lambda m: metric_subspace(m, data.eigvectors, data.eigs),
            "ylabel": "Distance to correct subspace",
            "yscale": "log",
            "title": "Subspace Distance"
        },
        "loss": {
            "func": lambda m: metric_loss(m, data_loader),
            "ylabel": "Loss with non-uniform $L_2$ regularization",
            "yscale": "log",
            "title": "Loss"
        },
        "reconstruction_loss": {
            "func": lambda m: metric_recon_loss(m, data_loader),
            "ylabel": "Reconstruction loss",
            "yscale": "log",
            "title": "Reconstruction Loss"
        }
    }
    eval_metrics_list = ["axis-alignment", "subspace-distance", "transpose"]
    return metric_config, eval_metrics_list


def update_config(optimal_lrs):
    # update wandb config, and return updated config dictionary
    if wandb.config.model_name == 'nd_exp':
        model_type = ModelTypes.NESTED_DROPOUT
        nd_expectation = 'true'
    elif wandb.config.model_name == 'nd':
        model_type = ModelTypes.NESTED_DROPOUT
        nd_expectation = 'false'
    else:
        model_type = wandb.config.model_name
        nd_expectation = None
    lr = optimal_lrs[wandb.config.model_name][wandb.config.optimizer][wandb.config.hdim]

    wandb.config.update({'model_type': model_type, 'nd_expectation': nd_expectation, 'lr': lr})
    return wandb.config
