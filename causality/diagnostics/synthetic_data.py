import numpy as np
from sklearn.metrics import mean_squared_error


def abs_ate(test_data, predicted_ate):
    true_ate = np.mean(test_data.mu1 - test_data.mu0)
    return abs(true_ate - predicted_ate)

def rmse_ite(test_data, predicted_ite):
    true_ite = test_data.mu1 - test_data.mu0
    return np.sqrt(mean_squared_error(y_pred=predicted_ite, y_true=true_ite))
