from sklearn.linear_model import LinearRegression as sklearn_LinearRegression
from sklearn.exceptions import NotFittedError

from causality.data.transformations import treatment_is_covariate
from causality.estimation.estimator import Estimator
from causality.exceptions import CannotPredictITEError


class LinearRegression(sklearn_LinearRegression, Estimator):
    def __init__(self, *args, **kwargs):
        self.treatment_coefficient = None
        super().__init__(*args, **kwargs)

    @treatment_is_covariate
    def fit(self, covariates, observed_outcomes, *args, **kwargs):
        super().fit(covariates, observed_outcomes)
        *_, self.treatment_coefficient = self.coef_
        return self

    def predict(self, *args, **kwargs):
        raise CannotPredictITEError()

    def predict_ate(self, covariates):
        if self.treatment_coefficient is None:
            raise NotFittedError('This LinearRegression instance is not fitted yet',)
        return self.treatment_coefficient
