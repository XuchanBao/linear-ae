import torch
import torch.nn as nn
import numpy as np


class LinearAE(nn.Module):
    def __init__(self,
                 input_dim, hidden_dim, init_scale=0.001,
                 weight_reg_type=None, l2_reg_list=None):
        super(LinearAE, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.encoder = nn.Linear(input_dim, hidden_dim, bias=False)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)

        self.weight_reg_type = weight_reg_type
        self.l2_reg_scalar = None
        self.l2_reg_list = l2_reg_list

        self.encoder.weight.data.normal_(0.0, init_scale)
        self.decoder.weight.data.normal_(0.0, init_scale)

        # configure regularization parameters

        assert self.weight_reg_type is None or isinstance(self.l2_reg_list, list), \
            "l2_reg_list must be a list if weight_reg_type is not None"

        assert self.l2_reg_list is None or len(self.l2_reg_list) == hidden_dim, \
            "Length of l2_reg_list must match latent dimension"

        if weight_reg_type in ("uniform_product", "uniform_sum"):
            self.l2_reg_scalar = l2_reg_list[0] ** 2    # more efficient to use scalar than diag_weights

        elif weight_reg_type == "non_uniform_sum":
            self.reg_weights = torch.tensor(
                np.array(self.l2_reg_list).astype(np.float32)
            )
            self.diag_weights = nn.Parameter(torch.diag(self.reg_weights), requires_grad=False)

    def forward(self, x):
        return self.get_reconstruction_loss(x) + self._get_reg_loss()

    def compute_trace_norm(self):
        """
        Computes the trace norm of the autoencoder, as well as decoder and encoder individually
        :return: trace_norm(W2W1), trace_norm(W1), trace_norm(W2)
        """
        return torch.matmul(self.decoder.weight, self.encoder.weight).norm(p='nuc'), \
               self.encoder.weight.norm(p='nuc'), \
               self.decoder.weight.norm(p='nuc'),

    def get_reconstruction_loss(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)

        recon_loss = torch.sum((x - recon) ** 2) / len(x)
        return recon_loss

    def get_reg_weights_np(self):
        if self.weight_reg_type is None:
            return np.zeros(self.hidden_dim)
        return np.array(self.l2_reg_list)

    def _get_reg_loss(self):
        # Standard L2 regularization, applied to W2W1 (product loss)
        if self.weight_reg_type == 'uniform_product':
            return self.l2_reg_scalar * (torch.norm(torch.matmul(self.decoder.weight, self.encoder.weight)) ** 2)

        # Standard L2 regularization for encoder and decoder separately (sum loss)
        elif self.weight_reg_type == 'uniform_sum':
            # regularize both encoder and decoder
            return self.l2_reg_scalar * (torch.norm(self.encoder.weight) ** 2 + torch.norm(self.decoder.weight) ** 2)

        # non-uniform sum
        elif self.weight_reg_type == 'non_uniform_sum':
            return torch.norm(self.diag_weights @ self.encoder.weight) ** 2 \
                   + torch.norm(self.decoder.weight @ self.diag_weights) ** 2

        # Do not apply regularization
        elif self.weight_reg_type is None:
            return 0.0

        else:
            raise ValueError("weight_reg_type should be one of (uniform_product, uniform_sum, non_uniform_sum, None)")
