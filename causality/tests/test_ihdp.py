import numpy as np

from causality.data.datasets.ihdp import IHDP
from causality.estimation.causal_forest import CausalForest
from causality.estimation.linear_regression import LinearRegression
from causality.estimation.propensity_score_matching import NearestNeighborMatching
from causality.diagnostics.synthetic_data import abs_ate


def test_causal_forest(replicate_number=0):
    train_data, test_data = IHDP.from_npz(replicate_number=replicate_number)
    error = abs_ate(
        test_data=test_data,
        predicted_ate=CausalForest().fit(**train_data.asdict()).predict_ate(
            test_data.covariates
        )
    )

    assert np.allclose(error, 0., atol=0.1)


def test_linear_regression(replicate_number=0):
    train_data, test_data = IHDP.from_npz(replicate_number=replicate_number)
    error = abs_ate(
        test_data=test_data,
        predicted_ate=LinearRegression().fit(**train_data.asdict()).predict_ate(
            test_data.covariates
        )
    )

    assert np.allclose(error, 0., atol=0.5)


def test_nearest_neighbor_matching(replicate_number=0, n_neighbors=100):
    train_data, test_data = IHDP.from_npz(replicate_number=replicate_number)
    error = abs_ate(
        test_data=test_data,
        predicted_ate=NearestNeighborMatching(n_neighbors=n_neighbors).fit(
            **train_data.asdict()
        ).predict_ate(test_data.covariates)
    )

    assert np.allclose(error, 0., atol=0.1)
