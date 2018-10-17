from collections import namedtuple
import warnings

import numpy as np
from sklearn.model_selection import StratifiedKFold

TrainValidationSplit = namedtuple("TrainValidationSplit", ["train", "validation"])
TrainTestValidationSplit = namedtuple(
    "TrainTestValidationSplit", ["train", "validation", "test"]
)
TrainTestSplit = namedtuple("TrainTestSplit", ["train", "test"])


class Dataset(object):
    def __init__(self, covariates, observed_outcomes=None, treatment_assignment=None,
                 **kwargs):

        self.num_units, *_ = covariates.shape
        self.covariates = covariates
        self.observed_outcomes = observed_outcomes
        self.treatment_assignment = treatment_assignment

        for argument_name, argument_value in kwargs.items():
            self.__setattr__(argument_name, argument_value)

    def keep_units(self, unit_indices):
        assert max(unit_indices) <= self.num_units
        return self.__class__(**self.asdict(units=unit_indices))

    def balanced_folds(self, num_splits, include_test_indices=False, seed=None):

        folds = StratifiedKFold(n_splits=10, random_state=seed)
        fold_indices = folds.split(
            np.zeros_like(self.treatment_assignment), self.treatment_assignment
        )

        for train_indices, test_indices in fold_indices:
            if include_test_indices:
                yield (self.keep_units(unit_indices=train_indices),
                       self.keep_units(unit_indices=test_indices),
                       test_indices)
            else:
                yield (self.keep_units(unit_indices=train_indices),
                       self.keep_units(unit_indices=test_indices))

    def asdict(self, units=None):
        if units is not None:
            return {
                "covariates": self.covariates[units, ...],
                "observed_outcomes": self.observed_outcomes[units, ...],
                "treatment_assignment": self.treatment_assignment[units, ...],
            }
        return {
            "covariates": self.covariates,
            "observed_outcomes": self.observed_outcomes,
            "treatment_assignment": self.treatment_assignment
        }

    def split(self, validation_fraction=0.3, test_fraction=None, seed=None):
        assert validation_fraction is not None or test_fraction is not None
        assert (validation_fraction or 0.) + (test_fraction or 0.) < 1.0

        validation_data, test_data = None, None

        indices = np.asarray(list(range(self.num_units)))

        rng = np.random.RandomState(seed=seed)
        rng.shuffle(indices)

        if validation_fraction:
            num_validation_units = int(validation_fraction * self.num_units)
            validation_units = indices[:num_validation_units]
            validation_data = self.__class__(**self.asdict(units=validation_units))

            indices = np.delete(indices, list(range(num_validation_units)))

        if test_fraction:
            num_test_units = int(test_fraction * self.num_units)
            test_units = indices[:num_test_units]
            test_data = self.__class__(**self.asdict(units=test_units))

            indices = np.delete(indices, list(range(num_test_units)))

        train_data = self.__class__(**self.asdict(units=indices))

        if validation_data and test_data:
            return TrainTestValidationSplit(
                train=train_data, validation=validation_data, test=test_data
            )

        elif validation_data:
            return TrainValidationSplit(train=train_data, validation=validation_data)
        elif test_data:
            return TrainTestSplit(train=train_data, test=test_data)

        raise ValueError()

    def generate_batches(self, batch_size=20, seed=None):
        assert 0 < batch_size
        if batch_size > self.num_units:
            warnings.warn(
                "Batch size {batch_size} larger than dataset size {data_size}. "
                "Generating batches of size {data_size} "
                "containing all datapoints. ".format(
                    batch_size=batch_size, data_size=self.num_units
                )
            )
            batch_size = self.num_units

        rng = np.random.RandomState(seed=seed)

        indices = list(range(self.num_units))
        while True:
            batch_units = rng.choice(indices, size=batch_size, replace=False)
            yield self.__class__(**self.asdict(units=batch_units))
