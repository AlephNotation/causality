import abc
from abc import ABCMeta

import numpy as np


class Estimator(metaclass=ABCMeta):
    @abc.abstractmethod
    def fit(self, covariates, observed_outcomes, treatment_assignment, *args, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def predict(self, covariates):
        raise NotImplementedError()

    def predict_ate(self, covariates):
        return np.mean(self.predict(covariates))

    def predict_ite(self, covariates):
        return self.predict(covariates)
