import numpy as np
from rpy2.robjects import r as R
from sklearn.exceptions import NotFittedError

from causality.data.transformations import r_compatibility
from causality.estimation.estimator import Estimator


class CausalForest(Estimator):
    def __init__(self, *args, **kwargs):
        R("library(grf)")  # load grf R library
        super().__init__(*args, **kwargs)
        self.rforest = None

    @r_compatibility
    def fit(self, covariates, observed_outcomes, treatment_assignment, *args, **kwargs):
        self.rforest = R("causal_forest")(
            X=covariates, Y=observed_outcomes, W=treatment_assignment
        )
        return self

    @r_compatibility
    def predict(self, covariates):
        if self.rforest is None:
            raise NotFittedError('This CausalForest instance is not fitted yet',)
        predictions = np.squeeze(
            R("predict")(self.rforest, covariates).rx2("predictions")
        )
        return predictions
