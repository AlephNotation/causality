#!/usr/bin/python3
# -*- coding: iso-8859-15 -*-
from causality.data.datasets.ihdp import IHDP
from causality.estimation.causal_forest import CausalForest
from causality.diagnostics.subgroup_identification import zhao_treatment_difference, zhao_auc
from causality.diagnostics.synthetic_data import rmse_ite

from scipy.stats.stats import pearsonr
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging

replicate_number = 100

aucs = []
avg_rmseites = []

logging.basicConfig(level=logging.INFO)

from tqdm import tqdm
for replicate_number in tqdm(range(0, 100), total=100):
    curves = []
    errors = []
    folds = StratifiedKFold(n_splits=10)

    train_data, test_data = IHDP.from_npz(replicate_number=replicate_number)

    for train_indices, test_indices in folds.split(np.zeros_like(train_data.treatment_assignment), train_data.treatment_assignment):
        train_covariates = train_data.covariates[train_indices, ...]
        train_responses = train_data.observed_outcomes[train_indices]
        train_treatment = train_data.treatment_assignment[train_indices]

        test_covariates = train_data.covariates[test_indices, ...]
        test_responses = train_data.observed_outcomes[test_indices]
        test_treatment = train_data.treatment_assignment[test_indices]
        ite = CausalForest().fit(
            covariates=train_covariates,
            observed_outcomes=train_responses,
            treatment_assignment=train_treatment
        ).predict_ite(test_covariates)
        curves.append(zhao_treatment_difference(ite, validation_responses=test_responses, validation_treatment_assignment=test_treatment))
        errors.append(rmse_ite(predicted_ite=ite, mu1=train_data.mu1[test_indices], mu0=train_data.mu0[test_indices]))

    curve = curves[0]

    avg_curve = pd.DataFrame(data=dict(
        q=curve.quantiles,
        delta=np.mean([curve.subgroup_statistics for curve in curves], axis=0)
    ))

    aucs.append(zhao_auc(avg_curve))
    avg_rmseites.append(np.mean(errors))

    if replicate_number and replicate_number % 2 == 0:
        logging.info(aucs)
        logging.info(avg_rmseites)
        logging.info(pearsonr(aucs, avg_rmseites))


from scipy.stats.stats import pearsonr
print(pearsonr(aucs, avg_rmseites))
