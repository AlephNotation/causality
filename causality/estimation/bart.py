import numpy as np
import rpy2
from rpy2.robjects import r as R
from rpy2.robjects import FloatVector

from causality.data.transformations import to_Rmatrix
from causality.estimation.estimator import Estimator

# for continuous outcomes, check: https://cran.r-project.org/web/packages/BART/BART.pdf
# for a complete list of available types.
BART_TYPE = "wbart"


class BART(Estimator):
    def __init__(self, bart_type=BART_TYPE) -> None:
        """ Initialize BART. Loads `R` library `BART`.  """
        super().__init__()

        try:
            R("library(BART)")  # load BART R library
        except rpy2.rinterface.RRuntimeError:
            try:
                R('install.packages("BART")')
            except rpy2.rinterface.RRuntimeError:
                raise ValueError(
                    "Attempted to install necessary R dependency 'BART' "
                    "in R, but installation failed.\nPlease first install 'BART' "
                    "from R-package repository CRAN using "
                    'install.packages("BART") inside an R-shell.'
                )
        self.bart_type = bart_type
        self.rbart = None

    def fit_predict(self, train_covariates, train_treatment_assignment,
                    train_observed_outcomes, test_covariates):
        treated_responses = train_observed_outcomes[train_treatment_assignment == 1]
        treated_covariates = train_covariates[train_treatment_assignment == 1, ...]

        treated_predictions = R(self.bart_type)(
            x_train=to_Rmatrix(treated_covariates),
            y_train=FloatVector(treated_responses),
            x_test=to_Rmatrix(test_covariates)
        ).rx2("yhat.test.mean")

        control_responses = train_observed_outcomes[train_treatment_assignment == 0]
        control_covariates = train_covariates[train_treatment_assignment == 0, ...]

        control_predictions = R(self.bart_type)(
            x_train=to_Rmatrix(control_covariates),
            y_train=FloatVector(control_responses),
            x_test=to_Rmatrix(test_covariates)
        ).rx2("yhat.test.mean")

        return np.asarray(treated_predictions) - np.asarray(control_predictions)

    def fit_predict_ite(self, train_covariates, train_treatment_assignment,
                        train_observed_outcomes, test_covariates):
        return self.fit_predict(
            train_covariates=train_covariates,
            train_treatment_assignment=train_treatment_assignment,
            train_observed_outcomes=train_observed_outcomes,
            test_covariates=test_covariates,
        )

    def fit_predict_ate(self, train_covariates, train_treatment_assignment,
                        train_observed_outcomes, test_covariates):
        return np.mean(self.fit_predict_ite(
            train_covariates=train_covariates,
            train_treatment_assignment=train_treatment_assignment,
            train_observed_outcomes=train_observed_outcomes,
            test_covariates=test_covariates
        ))

    def fit(self, treated_covariates, treated_outcomes,
            control_covariates, control_outcomes, *args, **kwargs):
        raise NotImplementedError(
            "In BART, it is currently not possible to "
            "properly split 'fit', and 'predict' in two steps."
            "To estimate ITE, please call `BART.fit_predict` instead."
        )

    def predict(self, covariates, *args, **kwargs):
        raise NotImplementedError(
            "In BART, it is currently not possible to "
            "properly split 'fit', and 'predict' in two steps."
            "To estimate ITE, please call `BART.fit_predict` instead."
        )
