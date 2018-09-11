import numpy as np
from sklearn.linear_model import LinearRegression as sklearn_LinearRegression
from sklearn.exceptions import NotFittedError

from causality.estimation.estimator import Estimator
from causality.exceptions import CannotPredictITEError


class LinearRegression(sklearn_LinearRegression, Estimator):
    def __init__(self, *args, **kwargs):
        self.treatment_coefficient = None
        super().__init__(*args, **kwargs)

    def fit(self, covariates, observed_outcomes, treatment_assignment, *args, **kwargs):
        features = np.concatenate((np.expand_dims(treatment_assignment, axis=1), covariates), axis=1)
        super().fit(features, observed_outcomes)
        self.treatment_coefficient, *_ = self.coef_
        return self

    def predict(self, *args, **kwargs):
        raise CannotPredictITEError()

    def predict_ate(self, covariates):
        if self.treatment_coefficient is None:
            raise NotFittedError('This LinearRegression instance is not fitted yet',)
        return self.treatment_coefficient