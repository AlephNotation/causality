from collections import namedtuple

import numpy as np
import pandas as pd
import rpy2
from rpy2.robjects import r as R


ZhaoTreatmentDifference = namedtuple(
    "ZhaoTreatmentDifference", ["quantiles", "effect_thresholds", "subgroup_statistics"]
)


def zhao_treatment_difference(treatment_effect_predictions,
                              validation_responses,
                              validation_treatment_assignment,
                              quantiles=np.arange(0., 1.0, 0.01),
                              subgroup_function=np.mean,
                              as_dataframe=False):
    """ For each quantile, compute subgroup of treatment_predictions with magnitude
        below this quantile and compute a given function on that subgroup.
        By default: compute average treatment effect of the subgroup.
    """
    assert all(0 <= quantile <= 1 for quantile in quantiles), "Quantiles must be `>= 0` and `<= 1`. "
    assert len(treatment_effect_predictions) == len(validation_responses) == len(validation_treatment_assignment)

    effect_thresholds = np.quantile(treatment_effect_predictions, q=quantiles)

    treatment_assignment = validation_treatment_assignment.astype(bool)

    subgroup_statistics = []

    for threshold in effect_thresholds:
        treated_subgroup = np.logical_and(treatment_assignment == 1, treatment_effect_predictions >= threshold)
        control_subgroup = np.logical_and(treatment_assignment == 0, treatment_effect_predictions >= threshold)

        treated_ate = subgroup_function(validation_responses[treated_subgroup])
        control_ate = subgroup_function(validation_responses[control_subgroup])
        subgroup_statistics.append(treated_ate - control_ate)

    results = ZhaoTreatmentDifference(
        quantiles=quantiles,
        effect_thresholds=effect_thresholds,
        subgroup_statistics=subgroup_statistics
    )

    if as_dataframe:
        return pd.DataFrame(data=dict(
            q=results.quantiles,
            tau_hat=results.effect_thresholds,
            delta=results.subgroup_statistics
        ))

    return results


def zhao_auc(zhao_results):
    """ Area under the zhao treatment effect curve.
        Higher values correspond to better treatment effect predictions.

    Parameters
    ----------
    zhao_results : ZhaoTreatmentDifference or pandas.DataFrame
        TODO

    """
    if isinstance(zhao_results, tuple):
        zhao_results = pd.DataFrame(data=dict(
            q=zhao_results.quantiles,
            tau_hat=zhao_results.effect_thresholds,
            delta=zhao_results.subgroup_statistics

        ))

    try:
        R("library(flux)")  # load grf R library
    except rpy2.rinterface.RRuntimeError:
        try:
            R('install.packages("flux")')
        except rpy2.rinterface.RRuntimeError:
            raise ValueError(
                "Attempted to install necessary R dependency 'flux' "
                "in R, but installation failed.\nPlease first install 'flux' "
                "from R-package repository CRAN using "
                'install.packages("flux") inside an R-shell.'
            )

    x = zhao_results.quantiles
    y = zhao_results.subgroup_statistics

    threshold = zhao_results.subgroup_statistics[0]
    assert threshold >= 0
    y_thresholded = y - threshold

    return R("flux::auc")(x=x, y=y_thresholded)
