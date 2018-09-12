""" Diagnostics that can only be computed on synthetic data
    that comes with ground truth for treatment outcomes with treatment
    `mu1` and without treatment `mu0`.
"""
import numpy as np
from sklearn.metrics import mean_squared_error


def abs_ate(mu1, mu0, predicted_ate):
    true_ate = np.mean(mu1 - mu0)
    return abs(true_ate - predicted_ate)

def rmse_ite(mu1, mu0, predicted_ite):
    true_ite = mu1 - mu0
    return np.sqrt(mean_squared_error(y_pred=predicted_ite, y_true=true_ite))
