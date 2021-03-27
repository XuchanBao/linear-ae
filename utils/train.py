import os
import torch
import wandb

from models.model_config import ModelTypes
from utils.logger import TrainStats, WeightHistory

def train_models(data_loader, train_itr, metrics_dict, model_configs,
                 eval_metrics_list=None, tie_weights=False, logger=None, ckpt_dir=None):
    first_model = model_configs[0].get_model()
    if tie_weights:
        first_model.decoder.weight.data.copy_(first_model.encoder.weight.data.T)

    for model_i in range(1, len(model_configs)):
        cur_model = model_configs[model_i].get_model()
        cur_model.encoder.weight.data.copy_(first_model.encoder.weight.data)

        if tie_weights:
            cur_model.decoder.weight.data.copy_(cur_model.encoder.weight.data.T)
        else:
            cur_model.decoder.weight.data.copy_(first_model.decoder.weight.data)

    # ---- Config train stats and weight history ----
    train_stats = TrainStats()
    weight_history = WeightHistory()

    for model_config in model_configs:
        train_stats.add_model(model_config.name, model_config.get_model())
        weight_history.add_model(model_config.name, model_config.get_model())

    if eval_metrics_list is None:
        eval_metrics_list = list(metrics_dict.keys())

    for m_name, metric_info in metrics_dict.items():
        train_stats.add_metric(m_name, metric_info["func"])

    # ---- Start training ----
    train_stats.evaluate_metrics(epoch=0, eval_metrics_list=eval_metrics_list)

    for train_i in range(train_itr):
        for x in data_loader:
            x_cuda = x.cuda()

            # ---- Log weights ----
            if (train_i == 0) or (train_i + 1) % 100 == 0:
                weight_history.log_weights(epoch=train_i)

            # ---- Optimize ----
            losses = {}
            for model_config in model_configs:
                model = model_config.get_model()
                optimizer = model_config.get_optimizer()

                optimizer.zero_grad()

                if model_config.type == ModelTypes.VAE:
                    # VAE model outputs elbo
                    loss = - model(x_cuda)
                else:
                    loss = model(x_cuda)

                loss.backward()

#                if model_config.type == ModelTypes.ROTATION and (model_config.optimizer.grad_type == "RMSprop_grad_acc" or model_config.optimizer.grad_type == "RMSprop_rotation_acc"):
#                    y = model.encoder.weight @ x_cuda.T
#                    yy_t_norm = y @ y.T 
#                    yy_t_upper = yy_t_norm - yy_t_norm.tril()
#                    gamma = 0.5 * (yy_t_upper - yy_t_upper.T)
#                elif model_config.type == ModelTypes.ROTATION:
                if model_config.type == ModelTypes.ROTATION:
                    y = model.encoder.weight @ x_cuda.T
                    yy_t_norm = y @ y.T / float(len(x))
                    yy_t_upper = yy_t_norm - yy_t_norm.tril()
                    gamma = 0.5 * (yy_t_upper - yy_t_upper.T)
                    model.encoder.weight.grad -= gamma @ model.encoder.weight
                    model.decoder.weight.grad -= model.decoder.weight @ gamma.T

#                if model_config.optimizer.grad_type == "RMSprop_grad_acc" or model_config.optimizer.grad_type == "RMSprop_rotation_acc":
#                    optimizer.step(gamma=gamma, batch_size=len(x))
#                else:
#                    optimizer.step()
                optimizer.step()

                losses[model_config.name] = loss.item()

        # ---- Log statistics ----
        if train_i == 0 or (train_i + 1) % 10 == 0:
            if logger:
                logger.info("".join(["Iteration = {}, Losses: ".format(train_i + 1)]
                                    + ["{} = {} ".format(key, val) for key, val in losses.items()]))
            else:
                print("".join(["Iteration = {}, Losses: ".format(train_i + 1)]
                              + ["{} = {} ".format(key, val) for key, val in losses.items()]))
            for key, val in losses.items():
                wandb.log({'loss/{}'.format(key): val}, step=train_i + 1)

        # ---- Evaluate metric ----
        if (train_i + 1) % 10 == 0:
            train_stats.evaluate_metrics(epoch=train_i + 1, eval_metrics_list=eval_metrics_list)

        if (train_i + 1) % 1000 == 0:
            # save model checkpoints
            for model_config in model_configs:
                model_name = model_config.name
                model = model_config.get_model()
                optimizer = model_config.get_optimizer()
                os.makedirs(os.path.join(ckpt_dir, model_name), exist_ok=True)
                torch.save(
                    {"epoch": train_i + 1,
                     "model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     },
                    os.path.join(ckpt_dir, model_name, "ckpt_epoch_{}.pt".format(train_i + 1))
                )

    train_stats.convert_to_numpy()
    weight_history.convert_to_numpy()

    return train_stats, weight_history
