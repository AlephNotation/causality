import numpy as np

from causality.data.datasets.dataset import Dataset


class Linear(Dataset):
    """ Dataset with a constant treatment effect of `ate` on each unit.
        Outcomes are a linear function of the covariates with added treatment effect.
        This dataset is synthetic, so ground truth treatment effect is available
        and can be retrieved using member variables `mu1` and `mu0` respectively.
    """
    def __init__(self, seed=None, ate=10, num_units=10**4, num_covariates=5):
        """ Generate a dataset with a constant treatment effect of `ate` on each treated unit.
            Covariates are drawn uniformly at random.
            This dataset is synthetic, so ground truth treatment effect is available.

        Parameters
        ----------
        ate : float, optional
            Desired average (and individualized) treatment effect on the generated data.
            Default: `10.`
        num_units : int, optional
            Number of units to generate. Default: `10**4`
        num_covariates : TODO, optional
            Number of covariates per unit to generate. Default: `5`
        seed : int, optional
            Seed used during generation. Default: `None`.

        Returns
        ----------
        dataset : causality.data.datasets.linear.Linear
            Dataset class that wraps generated data.

        Examples
        ----------
        This class is useful to quickly generate a simple dataset that can be
        used to sanity check new algorithms:

        >>> desired_ate = 10.
        >>> dataset = Linear(seed=1, ate=desired_ate, num_units=100, num_covariates=5)

        Ground truth for individualized treatment effects for each unit can
        be retrieved using the fields `mu1` (unit responses when receiving treatment)
        and `mu0` (unit responses when not receiving treatment):

        >>> 100 == len(dataset.mu0) and 100 == len(dataset.mu1)
        True

        From these individualized treatment effects, it is easy to compute
        the true average treatment effect (ATE) as well:

        >>> from math import isclose
        >>> ate = np.mean(dataset.mu1 - dataset.mu0)
        >>> isclose(ate, desired_ate)
        True

        Fitting one of our causal effect estimators is done as for all our datasets:

        >>> from causality.estimation.linear_regression import LinearRegression
        >>> predicted_ate = LinearRegression().fit(**dataset.asdict()).predict_ate(dataset.covariates)
        >>> isclose(ate, predicted_ate)
        True

        """
        assert num_units > 0
        assert num_covariates > 0

        c2 = np.random.uniform(0, ate * 0.5, num_covariates)

        rng = np.random.RandomState(seed=seed)

        covariates = rng.multivariate_normal(
            mean=np.random.uniform(-1, 1, num_covariates),
            cov=np.diag(np.ones(num_covariates)), size=num_units
        )

        num_covariates, *_ = covariates.shape
        treatment_assignment = np.random.choice([0., 1.], size=num_covariates)
        observed_outcomes = covariates @ c2 + ate * treatment_assignment

        mu1 = np.asarray([ate for _ in range(num_units)])
        mu0 = np.asarray([0. for _ in range(num_units)])

        super().__init__(
            covariates=covariates,
            treatment_assignment=treatment_assignment,
            observed_outcomes=observed_outcomes,
            mu1=mu1, mu0=mu0,
        )
