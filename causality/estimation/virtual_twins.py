""" Virtual Twins.
Simply learn a regression model to predict outcomes given covariates
and treatment assignment.  Then, to predict on a new unit, construct
two virtual twins: one that has received treatment and one that has not.
Use the model to predict outcomes for both twins and subtract predicted
control outcome from predicted treated outcome.
"""
from sklearn.ensemble import RandomForestRegressor

from causality.data.transformations import (
    treatment_is_covariate, virtual_twins
)
from causality.estimation.estimator import Estimator


# TODO: There is an extension that includes additional information.
# -> support that too, see: http://web.ccs.miami.edu/~hishwaran/papers/LSFI.JCGS2017.pdf
class VirtualTwins(Estimator):
    def __init__(self, regressor=RandomForestRegressor(), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.regressor = regressor

    @treatment_is_covariate
    def fit(self, covariates, observed_outcomes, *args, **kwargs):
        self.regressor.fit(covariates, observed_outcomes)
        return self

    @virtual_twins()
    def predict(self, covariates_treated, covariates_control):
        return (
            self.regressor.predict(covariates_treated) -
            self.regressor.predict(covariates_control)
        )
