import numpy as np
import rpy2
from rpy2.robjects import r as R

from causality.data.transformations import (
    split_covariates, virtual_twins, to_Rmatrix
)
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

    @split_covariates
    def fit(self, treated_covariates, treated_outcomes,
            control_covariates, control_outcomes, *args, **kwargs):
        self.treated_bart = R(self.bart_type)(
            x_train=treated_covariates,
            y_train=treated_outcomes
        )

        self.control_bart = R(self.bart_type)(
            x_train=control_covariates,
            y_train=control_outcomes
        )
        return self

    def predict(self, covariates, *args, **kwargs):
        treated_predictions = np.mean(
            R("predict.{}".format(self.bart_type))(
                self.treated_bart, newdata=to_Rmatrix(covariates)
            ), axis=0
        )

        control_predictions = np.mean(
            R("predict.{}".format(self.bart_type))(
                self.control_bart,
                to_Rmatrix(covariates)
            ), axis=0
        )

        return treated_predictions - control_predictions
