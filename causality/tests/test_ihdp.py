""" Test estimation of ATE (and ITE) on a single replicate of the IHDP dataset. """
from argparse import Namespace
from os.path import dirname, join as path_join
import sys
sys.path.insert(0, path_join(dirname(__file__), "..", "..", "cfrnet"))
import json
import numpy as np
import tensorflow as tf
import pytest

from causality.data.datasets.ihdp import IHDP
from causality.estimation.bart import BART
from causality.estimation.causal_forest import CausalForest
from causality.estimation.linear_regression import LinearRegression
from causality.estimation.propensity_score_matching import NearestNeighborMatching
from causality.estimation.virtual_twins import VirtualTwins
from causality.estimation.neural_networks.cfrnet import CFRNet
from causality.diagnostics.synthetic_data import abs_ate

try:
    import cfrnet  # noqa
except ImportError:
    CFRNET_AVAILABLE = False
else:
    CFRNET_AVAILABLE = True


def test_causal_forest(replicate_number=0):
    train_data, test_data = IHDP.from_npz(replicate_number=replicate_number)
    error = abs_ate(
        mu1=test_data.mu1,
        mu0=test_data.mu0,
        predicted_ate=CausalForest().fit(**train_data.asdict()).predict_ate(
            test_data.covariates
        )
    )

    assert np.allclose(error, 0., atol=0.1)


def test_linear_regression(replicate_number=0):
    train_data, test_data = IHDP.from_npz(replicate_number=replicate_number)
    error = abs_ate(
        mu1=test_data.mu1,
        mu0=test_data.mu0,
        predicted_ate=LinearRegression().fit(**train_data.asdict()).predict_ate(
            test_data.covariates
        )
    )

    assert np.allclose(error, 0., atol=0.5)


def test_nearest_neighbor_matching(replicate_number=0, n_neighbors=100):
    train_data, test_data = IHDP.from_npz(replicate_number=replicate_number)
    error = abs_ate(
        mu1=test_data.mu1,
        mu0=test_data.mu0,
        predicted_ate=NearestNeighborMatching(n_neighbors=n_neighbors).fit(
            **train_data.asdict()
        ).predict_ate(test_data.covariates)
    )

    assert np.allclose(error, 0., atol=0.1)


def test_virtual_twins(replicate_number=0):
    train_data, test_data = IHDP.from_npz(replicate_number=replicate_number)
    error = abs_ate(
        mu1=test_data.mu1,
        mu0=test_data.mu0,
        predicted_ate=VirtualTwins().fit(
            **train_data.asdict()
        ).predict_ate(test_data.covariates)
    )

    assert np.allclose(error, 0., atol=0.1)


@pytest.mark.skipif(not CFRNET_AVAILABLE, reason="CFRNet not installed.")
def test_cfrnet(replicate_number=0):
    configfile = path_join(dirname(__file__), "..", "..", "cfrnet", "configs", "default.json")
    with open(configfile) as f:
        configuration = json.load(f)
        configuration = Namespace(**configuration)

        configuration.seed = 1

    train_data, test_data = IHDP.from_npz(replicate_number=replicate_number)
    with tf.Session() as session:
        error = abs_ate(
            mu1=test_data.mu1,
            mu0=test_data.mu0,
            predicted_ate=CFRNet().fit(
                tensorflow_session=session,
                configuration=configuration,
                num_iterations=300,
                **train_data.asdict()
            ).predict_ate(tensorflow_session=session, covariates=test_data.covariates)
        )

    assert np.allclose(error, 0., atol=0.5)


def test_bart(replicate_number=0):
    train_data, test_data = IHDP.from_npz(replicate_number=replicate_number)
    error = abs_ate(
        mu1=test_data.mu1,
        mu0=test_data.mu0,
        predicted_ate=BART().fit_predict_ate(
            train_covariates=train_data.covariates,
            train_treatment_assignment=train_data.treatment_assignment,
            train_observed_outcomes=train_data.observed_outcomes,
            test_covariates=test_data.covariates
        )
    )

    assert np.allclose(error, 0., atol=0.1)
