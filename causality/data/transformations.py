# vim:foldmethod=marker
from rpy2.robjects import r as R
from rpy2.robjects import FloatVector
from functools import wraps


#  R Compatibility {{{ #

def to_robjects(covariates, observed_outcomes=None, treatment_assignment=None):
    """ Transform the given data into `rpy2.robjects` objects that can be fed to `R`.
        Returns transformed data as a dictionary.

    Parameters
    ----------
    covariates : np.ndarray
        Covariate data as a 2-d array of shape `(num_units, num_covariates)`.
    observed_outcomes : np.ndarray, optional
        Observed outcome data as a 1-d array of shape `(num_units,)`
    treatment_assignment : np.ndarray, optional
        (Binary) treatment assignment as a 1-d array of shape `(num_units,)`

    Returns
    ----------
    robject_dictionary : dict
        Dictionary mapping names of given data (e.g. `covariates`) to
        corresponding `rpy2.robjects` objects.

    Examples
    ----------
    TODO

    """
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


def r_compatibility(method):
    """ Decorates `method` allowing it to handle `numpy.ndarray` input as r-objects.
        Used to provide an easy user interface to code that actually interfaces
        to `R` code using `rpy2.robjects` objects inside the method body.

    Parameters
    ----------
    method : TODO
        Method to decorate. This method can be written as if all inputs are
        `rpy2.robjects` objects, but once decorated all inputs can be passed
        as `numpy.ndarray` objects instead.

    Returns
    ----------
    decorated_method : TODO
        Method that takes `numpy.ndarray` objects for its inputs and
        uses `rpy2.robjects` inside its body to compute results.

    Examples
    ----------
    TODO

    """
    @wraps(method)
    def wrapped(self, covariates, observed_outcomes=None, treatment_assignment=None, **kwargs):
        return method(
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
