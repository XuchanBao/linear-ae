import torch
import torch.nn as nn
import numpy as np


class LinearAENestedDropout(nn.Module):
    def __init__(self,
                 input_dim, hidden_dim, init_scale=0.001, prior_probs=None, use_expectation=False):
        super(LinearAENestedDropout, self).__init__()

        self.use_expectation = use_expectation

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.encoder = nn.Linear(input_dim, hidden_dim, bias=False)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)

        self.encoder.weight.data.normal_(0.0, init_scale)
        self.decoder.weight.data.normal_(0.0, init_scale)

        if prior_probs is None:
            # use geometric distribution
            # p(b) = rho^b (1 - rho) (b = 0 ... k - 2)
            # p(b = k-1) = 1 - sum(p(b), b < k-1)

            self.geom_p = 0.9
            prior_probs = [self.geom_p ** b * (1 - self.geom_p) for b in range(self.hidden_dim - 1)]
            prior_probs.append(1.0 - sum(prior_probs))

        self.prior_probs = torch.tensor(prior_probs)

        cum_probs = [1. - sum(prior_probs[:i]) for i in range(self.hidden_dim)]
        self.cum_probs = torch.tensor(cum_probs)
        self.diag_expected_mask = nn.Parameter(torch.diag(self.cum_probs), requires_grad=False)
        l_expected_mask = np.zeros((self.hidden_dim, self.hidden_dim))
        for i in range(self.hidden_dim):
            l_expected_mask[i, i] = cum_probs[i]
            l_expected_mask[:i, i] = cum_probs[i]
            l_expected_mask[i, :i] = cum_probs[i]
        self.l_expected_mask = nn.Parameter(torch.from_numpy(l_expected_mask).float(), requires_grad=False)

    def forward(self, x):
        if self.use_expectation:
            tr_xtx = torch.norm(x) ** 2
            w1_x = self.encoder(x).T        # (k, n)
            tr_xt_w2_y = torch.trace(w1_x @ x @ self.decoder.weight @ self.diag_expected_mask)
            w2t_w2_masked = (self.decoder.weight.T @ self.decoder.weight) * self.l_expected_mask
            tr_yt_w2t_w2_y = torch.trace(w1_x @ w1_x.T @ w2t_w2_masked)

            recon_loss = (tr_xtx - 2 * tr_xt_w2_y + tr_yt_w2t_w2_y) / len(x)
        else:
            hidden_units = self.encoder(x)
            hidden_units = self._nested_dropout(hidden_units)
            recon = self.decoder(hidden_units)

            recon_loss = torch.sum((x - recon) ** 2) / len(x)
        return recon_loss

    def _nested_dropout(self, hidden_units):
        prior_inds = torch.multinomial(self.prior_probs, len(hidden_units), replacement=True)
        mask = torch.ones_like(hidden_units)
        for hdim_i in range(1, self.hidden_dim):
            drop_row_inds = (prior_inds < hdim_i).float()     # 1 if row is dropped, 0 if kept
            mask[:, hdim_i] = 1 - drop_row_inds     # 1 if kept, 0 if dropped

        masked_hidden_units = hidden_units * mask
        return masked_hidden_units

    def get_reconstruction_loss(self, x):
        hidden_units = self.encoder(x)
        recon = self.decoder(hidden_units)

        recon_loss = torch.sum((x - recon) ** 2) / len(x)
        return recon_loss
