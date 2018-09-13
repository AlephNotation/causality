import numpy as np
from sklearn.ensemble import RandomForestRegressor

from causality.estimation.estimator import Estimator


# TODO: There is an extension that includes additional information -> support that too
# see: http://web.ccs.miami.edu/~hishwaran/papers/LSFI.JCGS2017.pdf
class VirtualTwins(RandomForestRegressor, Estimator):
    def fit(self, covariates, observed_outcomes, treatment_assignment, *args, **kwargs):
        features = np.concatenate((covariates, np.expand_dims(treatment_assignment, axis=1)), axis=1)
        print(features.shape)
        super().fit(features, observed_outcomes)
        return self

    def predict(self, covariates):
        num_covariates, *_ = covariates.shape
        covariates_treated = np.concatenate(
            (covariates, np.ones((num_covariates, 1))), axis=1
        )

        covariates_control = np.concatenate(
            (covariates, np.zeros((num_covariates, 1))), axis=1
        )

        return super().predict(covariates_treated) - super().predict(covariates_control)
