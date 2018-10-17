from os.path import realpath, dirname, join as path_join
from setuptools import setup, find_packages

NAME = "causality"
DESCRIPTION = "Causal Inference at the drop of a hat."
LONG_DESCRIPTION = "Causality is a python package for robust estimation of individualized and average treatment effect with a focus on randomized controlled trials. We also provide facilities to visualize data, experiment setup and model predictions for causal inference problems."
MAINTAINER = "Moritz Freidank"
MAINTAINER_EMAIL = "freidankm@gmail.com"
URL = "https://github.com/MFreidank/causality"
license = "MIT"
VERSION = "0.0.1"

PROJECT_ROOT = dirname(realpath(__file__))
REQUIREMENTS_FILE = path_join(PROJECT_ROOT, "requirements.txt")

with open(REQUIREMENTS_FILE, "r") as f:
    INSTALL_REQUIREMENTS = [
        requirement for requirement in f.read().splitlines()
        if not requirement.startswith("-e git")
    ]

SETUP_REQUIREMENTS = ["pytest-runner"]
TEST_REQUIREMENTS = ["pytest", "pytest-cov", "hypothesis"]

if __name__ == "__main__":
    setup(
        name=NAME,
        version=VERSION,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        url=URL,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        package_data={"docs": ["*"]},
        include_package_data=True,
        install_requires=INSTALL_REQUIREMENTS + ["pip @ https://github.com/MFreidank/cfrnet.git#egg=cfrnet"],
        setup_requires=SETUP_REQUIREMENTS,
        tests_require=TEST_REQUIREMENTS,
    )
