# vim:foldmethod=marker
from functools import wraps

import numpy as np
import rpy2
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
from rpy2.robjects import FloatVector

rpy2.robjects.numpy2ri.activate()


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

    """
    num_units, *_ = covariates.shape

    robjects = {"covariates": to_Rmatrix(covariates)}

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


def to_Rmatrix(array):
    num_rows, num_columns = array.shape
    return ro.r.matrix(array, nrow=num_rows, ncol=num_columns)

#  }}} R Compatibility #


def split_covariates(method):
    @wraps(method)
    def wrapped(self, covariates, observed_outcomes, treatment_assignment, **kwargs):
        treatment = treatment_assignment.astype(bool)
        treated_covariates = to_Rmatrix(covariates[treatment == 1, ...])
        treated_outcomes = FloatVector(observed_outcomes[treatment == 1, ...])

        control_covariates = to_Rmatrix(covariates[treatment == 0, ...])
        control_outcomes = FloatVector(observed_outcomes[treatment == 0, ...])

        return method(
            self,
            treated_covariates=treated_covariates,
            treated_outcomes=treated_outcomes,
            control_covariates=control_covariates,
            control_outcomes=control_outcomes,
            **kwargs
        )

    return wrapped


def treatment_to_covariate(covariates, treatment_assignment):
    """ Append a treatment assignment columns to a given `covariate` matrix.

    Parameters
    ----------
    covariates : np.ndarray
        Covariates describing units. Array of shape `(num_units, num_covariates_per_unit)`.
    treatment_assignment : np.ndarray
        Binary indicator array describing which units were treated.

    Returns
    ----------
    new_covariates : np.ndarray
        Covariates containing appended column for treatment assignment.
        Array of shape `(num_units, num_covariates_per_unit + 1)`.

    """
    return np.concatenate(
        (covariates, np.expand_dims(treatment_assignment, 1)), axis=1
    )


def virtual_twins(convert_to_robjects=False):
    """ Decorator to clone covariates, resulting in one treated and one control twin of each unit.
        Hereby, the treatment assignment is appended as a new covariate.
        Optionally may convert the two resulting covariate arrays into `rpy2.robjects`.

    Parameters
    ----------
    convert_to_robjects : bool, optional
        Whether resulting covariate arrays should be converted to `rpy2.robjects`.
        Default: `False`

    Returns
    ----------
    clone_dict : dict
        Dictionary with keys `"covariates_treated"`, `"covariates_control"`.
        Each maps to a `numpy.ndarray` of shape `(num_units, num_covariates_per_unit + 1)`.

    """
    def virtual_twins_inner(method):
        """ Decorator to wrap a method that requires access to treated and control version of covariates. """
        @wraps(method)
        def wrapped(self, covariates, **kwargs):
            num_units, *_ = covariates.shape

            all_treated = np.ones(num_units)
            none_treated = np.zeros(num_units)

            covariates_treated = treatment_to_covariate(
                covariates=covariates, treatment_assignment=all_treated
            )

            covariates_control = treatment_to_covariate(
                covariates=covariates, treatment_assignment=none_treated
            )

            if convert_to_robjects:

                covariates_treated = to_Rmatrix(covariates_treated)
                covariates_control = to_Rmatrix(covariates_control)

            covariate_clones = {
                "covariates_treated": covariates_treated,
                "covariates_control": covariates_control
            }

            return method(
                self,
                **covariate_clones, **kwargs
            )

        return wrapped
    return virtual_twins_inner


def treatment_is_covariate(method):
    """ Decorate `method` that expects treatment assignment as covariate.
        After decoration, the method can be called with separate covariates
        and treatment assignment arrays and conversion between those two
        interfaces is handled by this decorator.

    Parameters
    ----------
    method : TODO

    Returns
    ----------
    TODO

    Examples
    ----------
    TODO

    """
    @wraps(method)
    def wrapped(self, covariates, observed_outcomes, treatment_assignment, **kwargs):
        new_covariates = treatment_to_covariate(
            covariates=covariates, treatment_assignment=treatment_assignment
        )
        return method(
            self,
            covariates=new_covariates, observed_outcomes=observed_outcomes,
            **kwargs
        )
    return wrapped
