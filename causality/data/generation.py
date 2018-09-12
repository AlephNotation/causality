import numpy as np


def linear_dataset(ate=10, num_units=10**4, num_covariates=5, binary_treatment=True):
    assert num_units > 0
    assert num_covariates > 0

    c2 = np.random.uniform(0, ate * 0.5, num_covariates)

    covariates = np.random.multivariate_normal(
        mean=np.random.uniform(-1, 1, num_covariates),
        cov=np.diag(np.ones(num_covariates)),
        size=num_units
    )

    num_covariates, *_ = covariates.shape
    treatment_assignment = np.random.choice([0., 1.], size=num_covariates)

    observed_outcomes = covariates @ c2 + ate * treatment_assignment

    data = {
        "covariates": covariates + np.random.uniform(0, 10 ** -1, size=covariates.shape[0]).reshape((-1, 1)),
        "treatment_assignment": treatment_assignment,
        "observed_outcomes": observed_outcomes
    }

    ground_truth = {
        "ate": ate,
    }

    return data, ground_truth
