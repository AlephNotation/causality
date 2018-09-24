import abc
from abc import ABCMeta

import numpy as np


class Estimator(metaclass=ABCMeta):
    """ Base class for causal effect estimators. """
    @abc.abstractmethod
    def fit(self, covariates, observed_outcomes, treatment_assignment, *args, **kwargs):
        """ Train an estimator on training data obtained in a study.

        Parameters
        ----------
        covariates : TODO
        observed_outcomes : TODO
        treatment_assignment : TODO

        Returns
        ----------
        self : causality.estimation.estimator.Estimator
            Return the estimator object to allow convenient chaining of calls, as in:
            `estimator.fit(...).predict_ate(...)`.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def predict(self, covariates):
        """ Predict individualized treatment effects (ITE) for new, unseen units described by `covariates`.
            A shortcut alias for `estimator.predict_ite(covariates)`.

        Parameters
        ----------
        covariates : np.ndarray
            Array of covariates describing new units.
            Shape: `(num_units, num_covariates)`.

        Returns
        ----------
        unit_ites : np.ndarray
            Predictions for individualized treatment effects (ITE) of units
            described by `covariates`. Shape: `(num_units,)`.

        """
        raise NotImplementedError()

    def predict_ate(self, covariates):
        """ Predict average treatment effect (ATE) for new, unseen units described by `covariates`.

        Parameters
        ----------
        covariates : np.ndarray
            Array of covariates describing new units.
            Shape: `(num_units, num_covariates)`.

        Returns
        ----------
        unit_ate : float
            Predicted average treatment effect (ATE) for units described by `covariates`.

        """
        return np.mean(self.predict(covariates))

    def predict_ite(self, covariates):
        """ Predict individualized treatment effects (ITE) for new, unseen units described by `covariates`.
            A more readable alternative for `estimator.predict(covariates)`.

        Parameters
        ----------
        covariates : np.ndarray
            Array of covariates describing new units.
            Shape: `(num_units, num_covariates)`.

        Returns
        ----------
        unit_ites : np.ndarray
            Predictions for individualized treatment effects (ITE) of units
            described by `covariates`. Shape: `(num_units,)`.

        """
        return self.predict(covariates)
