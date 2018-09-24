from causality.data.datasets.linear import Linear


def linear_dataset(ate: float=10., num_units: int=10**4, num_covariates: int=5,
                   seed: int=None) -> Linear:
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

    """
    return Linear(
        seed=seed, ate=ate, num_units=num_units, num_covariates=num_covariates
    )
