"""AutoML Ops: automated ML lifecycle management platform."""

import os
from importlib.metadata import version

# Conservative native thread pools for this small local machine (BUILD_PLAN Step 9: algorithms run
# one at a time). The OpenMP pool used by scikit-learn's gradient boosting and the BLAS pools read
# these variables when they are first loaded, so they are set here, before anything imports
# scikit-learn. A value already set in the environment is respected.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_name, "2")

__version__ = version("automl-ops")
