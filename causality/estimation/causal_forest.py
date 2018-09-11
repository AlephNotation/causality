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
    def fit(self, covariates, observed_outcomes, treatment_assignment,
            num_trees=5000, tune_parameters=True, sample_fraction=0.5,
            min_node_size=5, ci_group_size=2, alpha=0.05,
            imbalance_penalty=0., stabilize_splits=True,
            seed=1, num_fit_trees=200, num_fit_reps=10,
            num_optimize_reps=1000, **kwargs):

        self.rforest = R("causal_forest")(
            X=covariates, Y=observed_outcomes, W=treatment_assignment,
            sample_fraction=sample_fraction, num_trees=num_trees,
            tune_parameters=tune_parameters, min_node_size=min_node_size,
            alpha=alpha, ci_group_size=ci_group_size,
            imbalance_penalty=imbalance_penalty,
            stabilize_splits=stabilize_splits,
            seed=seed,
            num_fit_trees=num_fit_trees, num_fit_reps=num_fit_reps,
            num_optimize_reps=num_optimize_reps,
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
