from sklearn.ensemble import RandomForestRegressor

from causality.data.transformations import (
    treatment_is_covariate, treatment_control_clone
)
from causality.estimation.estimator import Estimator


# TODO: There is an extension that includes additional information -> support that too
# see: http://web.ccs.miami.edu/~hishwaran/papers/LSFI.JCGS2017.pdf
class VirtualTwins(RandomForestRegressor, Estimator):
    @treatment_is_covariate
    def fit(self, covariates, observed_outcomes, *args, **kwargs):
        super().fit(covariates, observed_outcomes)
        return self

    @treatment_control_clone()
    def predict(self, covariates_treated, covariates_control):
        return super().predict(covariates_treated) - super().predict(covariates_control)
