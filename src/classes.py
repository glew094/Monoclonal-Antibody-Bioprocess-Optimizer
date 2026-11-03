import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class StandardScalerVAR(BaseEstimator, TransformerMixin):
    def __init__(self, p=1):
        self.p = p

    def fit(self, x, y=None):
        # Determine how many process variables are present
        n_var_features = x.shape[1] // self.p
        assert n_var_features * self.p == x.shape[1]

        x_sf = []
        for i in range(self.p):
            idx_start = i * n_var_features
            idx_end = idx_start + n_var_features
            x_sf.append(x[:, idx_start:idx_end])

        # Concatenate the lag blocks vertically
        x_sf = np.concatenate(x_sf, axis=0)

        # Calculate scaling statistics independently for each process variable
        x_sf_mean = np.mean(x_sf, axis=0)
        x_sf_sd = np.std(x_sf, axis=0)

        # Tile the statistics so that every lagged copy uses identical scaling parameters
        self.mean_ = np.tile(x_sf_mean, (1, self.p))
        self.scale_ = np.tile(x_sf_sd, (1, self.p))

        self.n_samples_seen_ = x.shape[0]
        self.n_features_in_ = x.shape[1]
        return self

    def transform(self, x):
        return (x - self.mean_) / self.scale_