import numpy as np
from sklearn.linear_model import LogisticRegression

from causality.estimation.estimator import Estimator


class PropensityScoreWeighting(LogisticRegression, Estimator):
    def __init__(self, *args, **kwargs):
        pass

    def fit(self, covariates, observed_outcomes, treatment_assignment,
            min_propensity_score=0.05, max_propensity_score=0.95,
            *args, **kwargs):
        assert min_propensity_score >= 0.
        assert max_propensity_score <= 1.

        # learn propensity scores
        propensity_scores = np.clip(
            super().fit(covariates, treatment_assignment).predict(covariates),
            a_min=min_propensity_score, a_max=max_propensity_score
        )

        # trim propensity score weights
        ipst_sum = sum(treatment_assignment / propensity_scores)
        ipsc_sum = sum((1. - treatment_assignment) / (1. - propensity_scores))

        ips_weights = (
            (treatment_assignment / propensity_scores / ipst_sum) +
            ((1. - treatment_assignment) / (1. - propensity_scores) / ipsc_sum)
        )

        # nips_weights = ips_weights / ips_weights.sum()

        # ips2 = propensity_scores / (1. - propensity_scores)
        # treated_ips_sum = (ips2 * treatment_assignment).sum()
        # control_ips_sum = (ips2 * (1. - treatment_assignment)).sum()
        # itps_weight = ips2 / treated_ips_sum
        # icps_weight = ips2 / control_ips_sum

        # XXX: Make this more flexible, see dowhy reference code
        d_y = ips_weights * treatment_assignment * observed_outcomes
        dbar_y = ips_weights * (1. - treatment_assignment) * observed_outcomes

        # XXX: Figure out how to compute the weight for a single unit
        # then we can use that to compute treatment effects

