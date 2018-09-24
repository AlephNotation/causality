import sklearn.exceptions as sklearn_exceptions


class NotFittedError(sklearn_exceptions.NotFittedError):
    """ Exception class to raise if estimator is used before fitting.
        Inherits from `sklearn.exceptions.NotFittedError`.
    """


class CannotPredictITEError(ValueError):
    """ Raised if a given model does not allow predicting individualized treatment effects. """
