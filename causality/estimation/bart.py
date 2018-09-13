import rpy2
from rpy2.robjects import r as R

from causality.data.transformations import (
    r_compatibility, treatment_is_covariate, treatment_control_clone
)
from causality.estimation.estimator import Estimator


class BART(Estimator):
    def __init__(self) -> None:
        """ Initialize a causal forest. Loads `R` library `BART`.  """
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
        self.rbart = None

    @treatment_is_covariate
    @r_compatibility
    def fit(self, covariates, observed_outcomes, *args, **kwargs):
        self.rbart = R("wbart")(
            x_train=covariates, y_train=observed_outcomes
        )
        return self

    @treatment_control_clone(convert_to_robjects=True)
    def predict(self, covariates_treated, covariates_control, *args, **kwargs):
        # XXX: Figure out how to use the bart predictions to predict ite.
        treated_predictions = R("predict.wbart")(
            self.rbart, covariates_treated
        )

        control_predictions = R("predict.wbart")(
            self.rbart, covariates_control
        )
        return treated_predictions
