import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

from causality.estimation.estimator import Estimator


class PropensityScoreMatching(LogisticRegression, Estimator):
    def __init__(self,
                 random_state=None,
                 matching_method=lambda: NearestNeighbors(
                     n_neighbors=100, algorithm="ball_tree"
                 ), *args, **kwargs):

        self.matching_method = matching_method
        super().__init__(*args, random_state=random_state, **kwargs)

    def fit(self, covariates, observed_outcomes, treatment_assignment, *args, **kwargs):
        # Learn propensity scores = probability of receiving treatment for covariates using logistic regression
        super().fit(covariates, treatment_assignment)

        propensity_predictions = {
            "treated": super().predict(covariates[treatment_assignment == 1]).reshape(-1, 1),
            "control": super().predict(covariates[treatment_assignment == 0]).reshape(-1, 1),
        }

        self.nearest_neighbors = {
            group: self.matching_method().fit(propensity_predictions[group])
            for group in ("treated", "control")
        }

        self.outcomes = {
            "treated": observed_outcomes[treatment_assignment == 1],
            "control": observed_outcomes[treatment_assignment == 0],
        }
        return self

    def _get_propensity_neighbors(self, covariates, group="treated"):
        distances, indices = self.nearest_neighbors[group].kneighbors(super().predict(covariates).reshape(-1, 1))
        return self.outcomes[group][indices]

    def predict(self, covariates, matching_method=NearestNeighbors(n_neighbors=1, algorithm="ball_tree")):
        treated_neighbor_outcomes = self._get_propensity_neighbors(covariates, group="treated")
        control_neighbor_outcomes = self._get_propensity_neighbors(covariates, group="control")

        return np.asarray([
            np.mean(treated_outcomes) - np.mean(control_outcomes)
            for treated_outcomes, control_outcomes
            in zip(treated_neighbor_outcomes, control_neighbor_outcomes)
        ])


class NearestNeighborMatching(PropensityScoreMatching):
    def __init__(self, n_neighbors=100, algorithm="ball_tree", random_state=None):
        super().__init__(
            matching_method=lambda: NearestNeighbors(
                n_neighbors=n_neighbors, algorithm=algorithm,
            ), random_state=random_state,
        )


class PropensityScoreStratification(PropensityScoreMatching):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError()


class PropensityScoreWeighting(PropensityScoreMatching):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError()
