import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import multivariate_normal
from scipy.stats import ortho_group


def min_eigvec_angles(eigvectors, decoder_weight):
    all_angles = cosine_similarity(eigvectors, decoder_weight)

    max_angles = -1.0 * np.ones((len(all_angles), ))
    max_inds = -1.0 * np.ones((len(all_angles), ))

    all_abs_angles = np.abs(all_angles)
    for col_i in range(len(all_abs_angles)):
        max_ind = all_abs_angles.argmax()
        max_row = max_ind // len(all_angles)
        max_col = max_ind % len(all_angles)
        max_angles[max_row] = all_abs_angles[max_row, max_col]
        all_abs_angles[:, max_col] = -1.0
        all_abs_angles[max_row, :] = -1.0
        max_inds[max_row] = max_col

    return max_angles, max_inds


def preordered_min_eigvec_angles(eigvectors, decoder_weight):
    all_angles = cosine_similarity(eigvectors, decoder_weight)
    return np.diagonal(np.abs(all_angles))


def ppca(samples, hdims, data_cov=None, dequantized=False):
    if data_cov is None:
        cov = np.cov(samples, rowvar=False)
    else:
        cov = data_cov
    if dequantized:
        cov += np.eye(cov.shape[0]) / 12.0
    w, v = np.linalg.eigh(cov)
    eigorder = w.argsort()
    w = w[eigorder[::-1]]
    v = v[:, eigorder[::-1]]

    using_eigvals = w[:hdims]
    using_eigvecs = v[:, :hdims]
    sigma_mle = w[hdims:].mean()
    weight_mle = np.matmul(using_eigvecs, np.diag(np.sqrt(using_eigvals - sigma_mle)))
    return weight_mle, sigma_mle


def avg_marginal_loglik(samples, W, sigma):
    C = np.matmul(W, W.T) + sigma * np.eye(W.shape[0])
    mean = samples.mean(0)
    mv = multivariate_normal(mean, C)
    loglik = 0.0
    for x in samples:
        loglik += mv.logpdf(x)
    return loglik / samples.shape[0]


def get_weight_tensor_from_seq(weight_seq):
    if isinstance(weight_seq, nn.Linear):
        return weight_seq.weight.detach()
    elif isinstance(weight_seq, nn.Sequential):
        weight_tensor = None
        for layer in weight_seq:
            if isinstance(layer, nn.Linear):
                layer_weight = layer.weight.detach()
                if weight_tensor is None:
                    weight_tensor = layer_weight
                else:
                    weight_tensor = layer_weight @ weight_tensor
            elif isinstance(layer, nn.BatchNorm1d):
                bn_weight = layer.weight.detach()

                # ignore bias

                if weight_tensor is None:
                    weight_tensor = torch.diag(bn_weight)
                else:
                    weight_tensor = torch.diag(bn_weight) @ weight_tensor
            else:
                raise ValueError("Layer type {} not supported!".format(type(layer)))
        return weight_tensor


def subspace_init(vae_model, w_gt, sigma_gt):
    w_gt_tensor = torch.from_numpy(w_gt).float().cuda()
    orth_matrix = torch.from_numpy(ortho_group.rvs(vae_model.hidden_dim)).float().cuda()

    if isinstance(vae_model.decoder, nn.Linear):
        # assign rotated decoder
        decoder_tensor = w_gt_tensor @ orth_matrix
        vae_model.decoder.weight.data.copy_(decoder_tensor)

    elif isinstance(vae_model.decoder, nn.Sequential):
        prev_layer = None
        for layer_i in range(0, len(vae_model.decoder) - 1):
            layer = vae_model.decoder[layer_i]
            if isinstance(layer, nn.Linear):
                # assign current layer as the orthogonal matrix (not including the last linear layer)
                layer.weight.data.copy_(orth_matrix)
                # rotate previous layer
                if prev_layer is not None:
                    prev_layer.weight.data.copy_(orth_matrix.T @ prev_layer.weight.detach())

                # generate new orthogonal matrix
                orth_matrix = torch.from_numpy(ortho_group.rvs(vae_model.hidden_dim)).float().cuda()

                # update previous layer
                prev_layer = layer

        # assign the last linear layer
        vae_model.decoder[-1].weight.data.copy_(w_gt_tensor @ orth_matrix)
        # rotate previous layer
        if prev_layer is not None:
            prev_layer.weight.data.copy_(orth_matrix.T @ prev_layer.weight.detach())

        decoder_tensor = get_weight_tensor_from_seq(vae_model.decoder)

    else:
        raise ValueError("Decoder type {} not supported!".format(type(vae_model.decoder)))

    # given the decoder, assign optimal encoder
    m = w_gt_tensor.T @ w_gt_tensor + sigma_gt ** 2 * torch.eye(vae_model.hidden_dim).cuda()
    vae_model.encoder.weight.data.copy_(torch.inverse(m) @ decoder_tensor.T)
