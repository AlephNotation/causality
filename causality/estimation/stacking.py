import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from causality.data.transformations import treatment_is_covariate, treatment_control_clone
from causality.estimation.estimator import Estimator
# from causality.estimation.propensity_score_matching import NearestNeighborMatching


class Stacking(Estimator):
    def __init__(self, meta_learner=RandomForestRegressor(),
                 base_learners=(
                     RandomForestRegressor(),
                     LinearRegression(),)):
        self.meta_learner = meta_learner
        self.base_learners = base_learners

    @treatment_is_covariate
    def fit(self, covariates, observed_outcomes, seed=None, *args, **kwargs):

        num_units, *_ = covariates.shape
        (covariates_train, covariates_validation, outcomes_train, outcomes_validation) = train_test_split(
            covariates, observed_outcomes, random_state=seed
        )

        for base_learner in self.base_learners:
            base_learner.fit(covariates_train, outcomes_train)

        base_learner_predictions = np.squeeze([[
            base_learner.predict(np.expand_dims(validation_covariate, 0))
            for base_learner in self.base_learners
        ] for validation_covariate in covariates_validation])

        self.meta_learner.fit(base_learner_predictions, outcomes_validation)

        return self

    @treatment_control_clone()
    def predict(self, covariates_treated, covariates_control):
        predictions_treated = self.meta_learner.predict(np.squeeze([[
            base_learner.predict(np.expand_dims(unit, 0))
            for base_learner in self.base_learners
        ] for unit in covariates_treated]))

        predictions_control = self.meta_learner.predict(np.squeeze([[
            base_learner.predict(np.expand_dims(unit, 0))
            for base_learner in self.base_learners
        ] for unit in covariates_control]))
        return predictions_treated - predictions_control
