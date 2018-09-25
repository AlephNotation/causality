#!/usr/bin/python3
# -*- coding: iso-8859-15 -*-
from collections import namedtuple
import numpy as np
import pandas as pd


ZhaoTreatmentDifference = namedtuple(
    "ZhaoTreatmentDifference", ["quantiles", "effect_thresholds", "subgroup_statistics"]
)


def zhao_treatment_difference(treatment_effect_predictions,
                              quantiles=np.arange(0., 1.0, 0.01),
                              subgroup_function=np.mean,
                              as_dataframe=False):
    """ For each quantile, compute subgroup of treatment_predictions with magnitude
        below this quantile and compute a given function on that subgroup.
        By default: compute average treatment effect of the subgroup.
    """
    assert all(0 <= quantile <= 1 for quantile in quantiles), "Quantiles must be `>= 0` and `<= 1`. "

    effect_thresholds = np.quantile(treatment_effect_predictions, q=quantiles)

    subgroups = [
        treatment_effect_predictions[treatment_effect_predictions >= threshold]
        for threshold in effect_thresholds
    ]

    subgroup_statistics = [subgroup_function(subgroup) for subgroup in subgroups]

    results = ZhaoTreatmentDifference(
        quantiles=quantiles,
        effect_thresholds=effect_thresholds,
        subgroup_statistics=subgroup_statistics
    )

    if as_dataframe:
        return pd.DataFrame.from_dict(dict(
            q=results.quantiles,
            tau_hat=results.effect_thresholds,
            delta=results.subgroup_statistics
        ))

    return results
