import numpy as np

from causality.data.datasets.linear import Linear


def linear_dataset(ate=10, num_units=10**4, num_covariates=5, seed=None):
    return Linear.random(
        seed=seed, ate=ate, num_units=num_units, num_covariates=num_covariates
    )
