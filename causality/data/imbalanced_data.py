import numpy as np
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import RandomUnderSampler

from causality.data.datasets.dataset import Dataset


def imbalanced_treatment_assignment(treatment_assignment, imbalance_percentage=75):
    # if more than imbalance_percentage % of datapoints were either treated or
    # not treated, than we consider the dataset imbalanced.
    assert 50 < imbalance_percentage < 100
    imbalance_factor = imbalance_percentage / 100.
    total_units, *_ = treatment_assignment.shape
    treated_units = treatment_assignment.sum()

    return (
        (treated_units > imbalance_factor * total_units) or
        (treated_units < (1. - imbalance_factor) * total_units)
    )


def balance_treatment(dataset, seed=None, sampler=RandomOverSampler):
    num_units, *_ = dataset.covariates.shape
    unit_indices = np.expand_dims(range(num_units), 1)

    indices, _ = sampler(random_state=seed).fit_sample(
        unit_indices, dataset.treatment_assignment
    )

    return Dataset(
        covariates=dataset.covariates[np.squeeze(indices), ...],
        observed_outcomes=dataset.observed_outcomes[np.squeeze(indices), ...],
        treatment_assignment=dataset.treatment_assignment[np.squeeze(indices), ...],
    )


def oversample_treatment(dataset, seed=None):
    return balance_treatment(dataset=dataset, seed=seed, sampler=RandomOverSampler)


def undersample_treatment(dataset, seed=None):
    return balance_treatment(dataset=dataset, seed=seed, sampler=RandomUnderSampler)
