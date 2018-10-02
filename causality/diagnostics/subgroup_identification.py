from collections import namedtuple
import numpy as np
import pandas as pd


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

    effect_thresholds = np.quantile(treatment_effect_predictions, q=quantiles)

    treatment_assignment = validation_treatment_assignment.astype(bool)

    subgroups = []
    for threshold in effect_thresholds:
        subgroups.append({
            "treated": validation_responses[
                (treatment_effect_predictions > threshold) & treatment_assignment
            ],

            "control": validation_responses[
                (treatment_effect_predictions > threshold) & (~treatment_assignment)
            ],
        })

    subgroup_statistics = [
        subgroup_function(subgroup["treated"]) - subgroup_function(subgroup["control"])
        for subgroup in subgroups
    ]

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
