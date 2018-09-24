import numpy as np
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from causality.data.datasets.dataset import Dataset


def imbalanced_treatment_assignment(treatment_assignment, imbalance_percentage=75) -> bool:

    """ Determine if a given `treatment_assignment` is imbalanced.
        Imbalance is given if more than `imbalance_percentage` percent of the units
        were assigned to the same condition.

    Parameters
    ----------
    treatment_assignment : numpy.ndarray
        Binary treatment assignment, represented as array of shape (num_units,).
        A `1` indicates that the corresponding unit was given the treatment, while
        a `0` indicates that this unit was part of the control group.
    imbalance_percentage : int, optional
        Integer percentage that serves as threshold to determine imbalance
        between classes.
        Default: `75`, in which case imbalance is given if `75` percent or more
        of the units were in the same group.

    Returns
    ----------
    is_imbalanced : bool
        `True` iff control and treatment group have imbalanced size, `False` otherwise.

    """
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


def balance_treatment(dataset: Dataset, seed: int=None, sampler=RandomOverSampler) -> Dataset:
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


def oversample_treatment(dataset: Dataset, seed: int=None) -> Dataset:
    return balance_treatment(dataset=dataset, seed=seed, sampler=RandomOverSampler)


def undersample_treatment(dataset: Dataset, seed: int=None) -> Dataset:
    return balance_treatment(dataset=dataset, seed=seed, sampler=RandomUnderSampler)
