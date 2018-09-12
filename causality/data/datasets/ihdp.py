from os.path import dirname, join as path_join

import numpy as np
from causality.data.datasets.dataset import Dataset, TrainTestSplit


DATA_PATH = path_join(
    dirname(__file__), "..", "..", "..", "datasets", "public", "ihdp"
)


class IHDP(Dataset):
    @classmethod
    def from_npz(cls,
                 replicate_number,
                 npz_train_filename=path_join(DATA_PATH, "ihdp_npci_1-100.train.npz"),
                 npz_test_filename=path_join(DATA_PATH, "ihdp_npci_1-100.test.npz")):
        assert 0 <= replicate_number <= 99

        train_data = dict(np.load(npz_train_filename))
        test_data = dict(np.load(npz_test_filename))

        train_dataset = IHDP(
            replicate_number=replicate_number,
            covariates=train_data["x"][..., replicate_number],
            observed_outcomes=train_data["yf"][..., replicate_number],
            treatment_assignment=train_data["t"][..., replicate_number],
            mu1=train_data["mu1"][..., replicate_number],
            mu0=train_data["mu0"][..., replicate_number],
        )

        test_dataset = IHDP(
            replicate_number=replicate_number,
            covariates=test_data["x"][..., replicate_number],
            observed_outcomes=test_data["yf"][..., replicate_number],
            treatment_assignment=test_data["t"][..., replicate_number],
            mu1=test_data["mu1"][..., replicate_number],
            mu0=test_data["mu0"][..., replicate_number],
        )

        return TrainTestSplit(train=train_dataset, test=test_dataset)

    def asdict(self, units=None):
        dict_representation = super().asdict(units=units)
        dict_representation.update({"mu1": self.mu1[units, ...], "mu0": self.mu0[units, ...]})
        return dict_representation
