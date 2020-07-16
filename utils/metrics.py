import torch
import numpy as np
from models.utils import get_weight_tensor_from_seq


def metric_transpose_theorem(model):
    """
    Metric for how close encoder and decoder.T are
    :param model: LinearAE model
    :return: ||W1 - W2^T||_F^2 / hidden_dim
    """
    encoder_weight = get_weight_tensor_from_seq(model.encoder)
    decoder_weight = get_weight_tensor_from_seq(model.decoder)

    transpose_metric = torch.norm(encoder_weight - decoder_weight.T) ** 2
    return transpose_metric.item() / float(model.hidden_dim)


def metric_alignment(model, gt_eigvectors):
    """
    Metric for alignment of decoder columns to ground truth eigenvectors
    :param model: Linear AE model
    :param gt_eigvectors: ground truth eigenvectors (input_dims,hidden_dims)
    :return: sum_i (1 - max_j (cos(eigvector_i, normalized_decoder column_j)))
    """
    decoder_weight = get_weight_tensor_from_seq(model.decoder)
    decoder_np = decoder_weight.detach().cpu().numpy()

    # normalize columns of gt_eigvectors
    norm_gt_eigvectors = gt_eigvectors / np.linalg.norm(gt_eigvectors, axis=0)
    # normalize columns of decoder
    norm_decoder = decoder_np / (np.linalg.norm(decoder_np, axis=0) + 1e-8)

    total_angles = 0.0
    for eig_i in range(gt_eigvectors.shape[1]):
        eigvector = norm_gt_eigvectors[:, eig_i]
        total_angles += 1. - np.max(np.abs(norm_decoder.T @ eigvector)) ** 2

    return total_angles / float(model.hidden_dim)


def metric_subspace(model, gt_eigvectors, gt_eigs):
    decoder_weight = get_weight_tensor_from_seq(model.decoder)
    decoder_np = decoder_weight.detach().cpu().numpy()

    # k - tr(UU^T WW^T), where W is left singular vector matrix of decoder
    u, s, vh = np.linalg.svd(decoder_np, full_matrices=False)
    return 1 - np.trace(gt_eigvectors @ gt_eigvectors.T @ u @ u.T) / float(model.hidden_dim)


def metric_loss(model, data_loader):
    """
    Measures the full batch loss
    :param model: a linear (variational) AE model
    :param data_loader: full batch data loader. Should be different from the training data loader, if in minibatch mode
    """
    loss = None
    for x in data_loader:
        loss = model(x.cuda()).item()
    return loss


def metric_recon_loss(model, data_loader):
    recon_loss = None
    for x in data_loader:
        recon_loss = model.get_reconstruction_loss(x.cuda()).item()
    return recon_loss
