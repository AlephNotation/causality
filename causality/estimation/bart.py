import numpy as np
import rpy2
from rpy2.robjects import r as R

from causality.data.transformations import (
    r_compatibility, treatment_is_covariate, virtual_twins
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

    @treatment_is_covariate
    @r_compatibility
    def fit(self, covariates, observed_outcomes, *args, **kwargs):
        self.rbart = R(self.bart_type)(
            x_train=covariates, y_train=observed_outcomes
        )
        return self

    @virtual_twins(convert_to_robjects=True)
    def predict(self, covariates_treated, covariates_control, *args, **kwargs):
        # XXX: Why does this not work to predict ate properly?
        treated_predictions = np.mean(
            R("predict.{}".format(self.bart_type))(
                self.rbart, covariates_treated
            ), axis=0
        )

        control_predictions = np.mean(
            R("predict.{}".format(self.bart_type))(
                self.rbart, covariates_control
            ), axis=0
        )

        return treated_predictions - control_predictions
