""" Diagnostics that can only be computed on synthetic data
    that comes with ground truth for treatment outcomes with treatment
    `mu1` and without treatment `mu0`.
"""
import numpy as np
from sklearn.metrics import mean_squared_error


def abs_ate(mu1, mu0, predicted_ate):
    """ Absolute error of average treatment prediction.
        `mu1` is an array of treatment effects for each unit when treated
        and `mu0` is an array of corresponding treatment effects for the same units
        when in the control group.
        These quantities are unknown in general and can only be obtained for
        synthetic data.

    Parameters
    ----------
    mu1 : np.ndarray
        Array of treatment effects for all units when treated.
    mu0 : np.ndarray
        Array of treatment effects for all units when not treated.
    predicted_ate : float
        Predicted average treatment effect over all units.

    Returns
    ----------
    abs_ate_error : float
        Absolute error of predicted average treatment effect wrt. the actual one.

    Examples
    ----------
    This function is useful to measure a models capability to accurately predict
    average treatment effects on synthetic data, such as a replicate of
    the Infant Health and Development Program (IHDP) dataset:

    >>> from causality.data.datasets.ihdp import IHDP
    >>> from causality.estimation.propensity_score_matching import NearestNeighborMatching
    >>> train_data, test_data = IHDP.from_npz(replicate_number=1)
    >>> estimator = NearestNeighborMatching(n_neighbors=50).fit(**train_data.asdict())
    >>> predicted_ate = estimator.predict_ate(test_data.covariates)
    >>> abs_ate(mu1=test_data.mu1, mu0=test_data.mu0, predicted_ate=predicted_ate)
    0.15562686654655167

    """
    true_ate = np.mean(mu1 - mu0)
    return abs(true_ate - predicted_ate)

def rmse_ite(mu1, mu0, predicted_ite):
    """ Root mean squared error of individualized treatment effect prediction.
        `mu1` is an array of treatment effects for each unit when treated
        and `mu0` is an array of corresponding treatment effects for the same units
        when in the control group.
        These quantities are unknown in general and can only be obtained for
        synthetic data.

    Parameters
    ----------
    mu1 : np.ndarray
        Array of treatment effects for all units when treated.
    mu0 : np.ndarray
        Array of treatment effects for all units when not treated.
    predicted_ate : float
        Predicted individualized treatment effect for each unit.

    Returns
    ----------
    rmse_ite : float
        Root mean squared error of predicted individualized treatment effect.

    Examples
    ----------
    This function is useful to measure a models capability to accurately predict
    individualized treatment effects on synthetic data, such as a replicate of
    the Infant Health and Development Program (IHDP) dataset:

    >>> from causality.data.datasets.ihdp import IHDP
    >>> from causality.estimation.propensity_score_matching import NearestNeighborMatching
    >>> train_data, test_data = IHDP.from_npz(replicate_number=1)
    >>> estimator = NearestNeighborMatching(n_neighbors=50).fit(**train_data.asdict())
    >>> predicted_ite = estimator.predict_ite(test_data.covariates)
    >>> rmse_ite(mu1=test_data.mu1, mu0=test_data.mu0, predicted_ite=predicted_ite)
    0.8328293762635052

    """
    true_ite = mu1 - mu0
    return np.sqrt(mean_squared_error(y_pred=predicted_ite, y_true=true_ite))
