# vim:foldmethod=marker
from rpy2.robjects import r as R
from rpy2.robjects import FloatVector
from functools import wraps


#  R Compatibility {{{ #

def to_robjects(covariates, observed_outcomes=None, treatment_assignment=None):
    num_units, *_ = covariates.shape

    robjects = {
        "covariates": R("matrix")(
            FloatVector(covariates.flatten()),
            nrow=num_units
        ),
    }

    if observed_outcomes is not None:
        robjects.update({
            "observed_outcomes": FloatVector(observed_outcomes.tolist())
        })
    if treatment_assignment is not None:
        robjects.update({
            "treatment_assignment": FloatVector(treatment_assignment.tolist())
        })

    return robjects


def r_compatibility(function):
    @wraps(function)
    def wrapped(self, covariates, observed_outcomes=None, treatment_assignment=None, **kwargs):
        return function(
            self,
            **to_robjects(
                covariates=covariates,
                observed_outcomes=observed_outcomes,
                treatment_assignment=treatment_assignment
            ),
            **kwargs
        )

    return wrapped
#  }}} R Compatibility #
