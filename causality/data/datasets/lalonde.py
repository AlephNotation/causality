import rpy2
from rpy2.robjects import r as R, pandas2ri

from causality.data.datasets.dataset import Dataset


class LaLonde(Dataset):
    @classmethod
    def as_dataframe(cls):
        try:
            R("library(Matching)")
        except rpy2.rinterface.RRuntimeError:
            try:
                R('install.packages("Matching")')
            except rpy2.rinterface.RRuntimeError:
                raise ValueError(
                    "Attempted to install necessary R dependency 'Matching' "
                    "in R, but installation failed.\nPlease first install 'Matching' "
                    "from R-package repository CRAN using "
                    'install.packages("Matching") inside an R-shell.'
                )
        R("library(Matching)")
        R("data(lalonde)")
        pandas2ri.activate()
        data = (R("lalonde"))
        pandas2ri.deactivate()
        return data

    @classmethod
    def from_r(cls, covariate_columns=None):
        data = cls.as_dataframe()
        covariates = data.drop(["treat", "re78"], axis=1)
        if covariate_columns is not None:
            unnecessary_columns = set(covariates.columns).difference(covariate_columns)
            covariates = covariates.drop(list(unnecessary_columns), axis=1)

        return LaLonde(
            treatment_assignment=data["treat"].as_matrix(),
            observed_outcomes=data["re78"].as_matrix(),
            covariates=covariates.as_matrix()
        )
