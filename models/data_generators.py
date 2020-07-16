from torch.utils.data import Dataset

import numpy as np
from scipy.stats import ortho_group


class DataGeneratorPPCA(Dataset):

    def __init__(self, dims, hdims, min_sv=0.11, max_sv=5.0, sigma_sq=0.1, deterministic=True, total=10000):
        self.dims = dims
        self.hdims = hdims

        self.eigs = min_sv + (max_sv - min_sv) * np.linspace(0, 1, hdims)
        self.eigvectors = ortho_group.rvs(dims)[:, :hdims]
        self.w = np.matmul(self.eigvectors, np.diag(np.sqrt(self.eigs - sigma_sq)))

        self.sigma_sq = sigma_sq
        self.sigma = np.sqrt(sigma_sq)

        self.total = total
        self.deterministic = deterministic
        if self.deterministic:
            self.z_sample = np.random.normal(size=(total, self.hdims))
            self.x_sample = np.random.normal(np.matmul(self.z_sample, self.w.T), self.sigma).astype(np.float32)

    def __getitem__(self, i):
        if self.deterministic:
            return self.x_sample[i]
        else:
            z_sample = np.random.normal(size=self.hdims)
            return np.random.normal(self.w.dot(z_sample), self.sigma).astype(np.float32)

    def __len__(self):
        # Return a large number for an epoch
        return self.total


class DataGeneratorPCA(Dataset):
    def __init__(self, dims, hdims, min_sv=0.11, max_sv=5.0, total=10000, sv_list=None,
                 load_data=None):
        self.dims = dims
        self.hdims = hdims

        if load_data is None:
            if isinstance(sv_list, list):
                assert len(sv_list) == dims
                self.full_eigs = np.array(sorted(sv_list, reverse=True))
            else:
                self.full_eigs = min_sv + (max_sv - min_sv) * np.linspace(1, 0, dims)
            self.eigs = self.full_eigs[:hdims]

            self.full_svs = np.sqrt(self.full_eigs)

            self.full_eigvectors = ortho_group.rvs(dims)
            self.eigvectors = self.full_eigvectors[:, :hdims]

            self.total = total

            self.full_z_sample = np.random.normal(size=(total, self.dims))
            self.x_sample = (self.full_eigvectors @ np.diag(self.full_svs) @ self.full_z_sample.T).T.astype(np.float32)

        else:
            self.x_sample = load_data
            u, s, vh = np.linalg.svd(self.x_sample.T, full_matrices=False)
            self.eigs = s[:self.hdims]
            self.eigvectors = u[:, :self.hdims]
            self.total = len(self.x_sample)

    def __getitem__(self, i):
        return self.x_sample[i]

    def __len__(self):
        return self.total

    @property
    def shape(self):
        return self.x_sample.shape
