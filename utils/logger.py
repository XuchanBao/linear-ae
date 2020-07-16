import os
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from models.utils import get_weight_tensor_from_seq
import wandb


class ModelStats:
    def __init__(self, model_name):
        self.name = model_name
        self.model_stats = {
        }

    def add_metric(self, metric_name):
        self.model_stats[metric_name] = {
            "epoch": [],
            "value": []
        }

    def log_metric_value(self, epoch, metric_name, metric_value):
        self.model_stats[metric_name]['epoch'].append(epoch)
        self.model_stats[metric_name]['value'].append(metric_value)

    def convert_to_numpy(self):
        for metric_data in self.model_stats.values():
            metric_data['epoch'] = np.array(metric_data['epoch'])
            metric_data['value'] = np.array(metric_data['value'])

    def convert_to_list(self):
        for metric_data in self.model_stats.values():
            metric_data['epoch'] = list(np.array(metric_data['epoch']))
            metric_data['value'] = list(np.array(metric_data['value']))

    def get_data_for_metric(self, metric_name):
        return self.model_stats[metric_name]


class TrainStats:
    def __init__(self, summary_dir=None):
        self.model_dict = {}
        self.metrics_dict = {}
        self.all_stats = {}

        self.summary_dir = summary_dir
        self.summary_writers = {}
        self.write_to_tb = self.summary_dir is not None

    def add_model(self, model_name, model):
        self.model_dict[model_name] = model
        self.all_stats[model_name] = ModelStats(model_name)
        for metric_name in self.metrics_dict.keys():
            self.all_stats[model_name].add_metric(metric_name)

        if self.write_to_tb:
            summary_dir = os.path.join(self.summary_dir, model_name)
            os.makedirs(summary_dir, exist_ok=True)
            self.summary_writers[model_name] = SummaryWriter(log_dir=summary_dir)

    def add_metric(self, metric_name, metric_func):
        self.metrics_dict[metric_name] = metric_func
        for model_stat in self.all_stats.values():
            model_stat.add_metric(metric_name)

    def _log_metric_model(self, metric_name, model_name, metric_value, epoch):
        # log to wandb
        wandb.log({'{}/{}'.format(metric_name, model_name): metric_value}, step=epoch)

        # log to summary writer
        if self.write_to_tb:
            self.summary_writers[model_name].add_scalar('{}'.format(metric_name), metric_value, global_step=epoch)

    def _evaluate_metrics_for_model(self, epoch, model_name, eval_metrics_list):
        model = self.model_dict[model_name]
        # for metric_name, metric in self.metrics_dict.items():
        metric_values_for_model = {}
        for metric_name in eval_metrics_list:
            metric_value = self.metrics_dict[metric_name](model)
            metric_values_for_model[metric_name] = metric_value
            self.all_stats[model_name].log_metric_value(epoch, metric_name, metric_value)

            # log metric
            self._log_metric_model(metric_name, model_name, metric_value, epoch)

        return metric_values_for_model

    def _evaluate_metric_for_model(self, epoch, model_name, metric_name):
        model = self.model_dict[model_name]
        metric = self.metrics_dict[metric_name]
        metric_value = metric(model)

        # log metric
        self._log_metric_model(metric_name, model_name, metric_value, epoch)

        self.all_stats[model_name].log_metric_value(epoch, metric_name, metric_value)
        return {metric_name: metric_value}

    def evaluate_metrics(self, epoch, eval_metrics_list, model_name=None):
        metric_values = {}
        if model_name is None:
            for model_name in self.model_dict.keys():
                metric_values_for_model = self._evaluate_metrics_for_model(epoch, model_name, eval_metrics_list)
                metric_values[model_name] = metric_values_for_model
        else:
            metric_values_for_model = self._evaluate_metrics_for_model(epoch, model_name, eval_metrics_list)
            metric_values[model_name] = metric_values_for_model
        return metric_values

    def evaluate_metric(self, epoch, metric_name, model_name=None):
        metric_values = {}
        if model_name is None:
            for model_name in self.model_dict.keys():
                metric_values_for_model = self._evaluate_metric_for_model(epoch, model_name, metric_name)
                metric_values[model_name] = metric_values_for_model
        else:
            metric_values_for_model = self._evaluate_metric_for_model(epoch, model_name, metric_name)
            metric_values[model_name] = metric_values_for_model
        return metric_values

    def convert_to_numpy(self):
        for model_stats in self.all_stats.values():
            model_stats.convert_to_numpy()

    def convert_to_list(self):
        for model_stats in self.all_stats.values():
            model_stats.convert_to_list()

    def get_metric_names(self):
        return list(self.metrics_dict.keys())

    def get_data_for_metric(self, metric_name):
        metric_data = {}
        for model_name, model_stats in self.all_stats.items():
            metric_data[model_name] = model_stats.get_data_for_metric(metric_name)

        return metric_data

    def get_metric_datapoint(self, epoch, model_name, metric_name):
        return self.all_stats[model_name]

    def close(self):
        for writer in self.summary_writers.values():
            writer.close()


class ModelWeightHistory:
    def __init__(self, model_name):
        self.name = model_name
        self.epoch = []
        self.weights = {}

    def log_weights(self, epoch, **weights_np):
        self.epoch.append(epoch)
        if len(self.weights) == 0:
            for weight_name, weight_value in weights_np.items():
                self.weights[weight_name] = [weight_value]
        else:
            for weights_name, weight_value in weights_np.items():
                self.weights[weights_name].append(weight_value)

    def convert_to_numpy(self):
        self.epoch = np.array(self.epoch)

        for weight_name, weight_value_list in self.weights.items():
            self.weights[weight_name] = np.array(weight_value_list)

    def convert_to_list(self):
        self.epoch = list(np.array(self.epoch))
        for weight_name, weight_value in self.weights.items():
            self.weights[weight_name] = list(np.array(weight_value))

    def __getitem__(self, item: str):
        if item == "epoch":
            return self.epoch
        return self.weights.get(item)


class WeightHistory:
    def __init__(self):
        self.model_dict = {}
        self.weight_history = {}

    def add_model(self, model_name, model):
        self.model_dict[model_name] = model
        self.weight_history[model_name] = ModelWeightHistory(model_name)

    def _log_weights_for_model(self, epoch, model_name):
        model = self.model_dict[model_name]

        weight_np_dict = {"encoder": model.encoder.weight.detach().cpu().numpy()}
        if isinstance(model.decoder, nn.Linear):
            weight_np_dict["decoder"] = model.decoder.weight.detach().cpu().numpy()
        elif isinstance(model.decoder, nn.Sequential):
            full_decoder_weight = get_weight_tensor_from_seq(model.decoder)
            weight_np_dict["decoder"] = full_decoder_weight.detach().cpu().numpy()
            for layer_i, decoder_layer in enumerate(model.decoder):
                if isinstance(decoder_layer, nn.Linear):
                    weight_np_dict["decoder_linear{}".format(layer_i)] = decoder_layer.weight.detach().cpu().numpy()
                elif isinstance(decoder_layer, nn.BatchNorm1d):
                    weight_np_dict["decoder_bn{}_weight".format(layer_i)] = decoder_layer.weight.detach().cpu().numpy()
                    weight_np_dict["decoder_bn{}_bias".format(layer_i)] = decoder_layer.bias.detach().cpu().numpy()

        self.weight_history[model_name].log_weights(epoch, **weight_np_dict)

    def log_weights(self, epoch, model_name=None):
        if model_name is None:
            for model_name in self.model_dict.keys():
                self._log_weights_for_model(epoch, model_name)
        else:
            self._log_weights_for_model(epoch, model_name)

    def convert_to_numpy(self):
        for model_weight_history in self.weight_history.values():
            model_weight_history.convert_to_numpy()

    def convert_to_list(self):
        for model_weight_history in self.weight_history.values():
            model_weight_history.convert_to_list()

    def get_weights(self):
        return self.weight_history
