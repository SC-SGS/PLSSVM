# import the plssvm module explicitly
from . import plssvm
# export everything
from .plssvm import *  # noqa: F405

# explicitly set the module level attributes
__doc__ = plssvm.__doc__
__version__ = plssvm.__version__
__version_info__ = plssvm.__version_info__
__has_mpi_support__ = plssvm.__has_mpi_support__

# explicitly register the submodules as importable Python module
import sys

possible_submodules = \
    ["svm", "performance_tracking",
     "adaptivecpp", "cuda", "dpcpp", "hip", "hpx", "kokkos", "opencl", "openmp", "stdpar", "sycl"]
for submodule in possible_submodules:
    if hasattr(plssvm, submodule):
        sys.modules[f"plssvm.{submodule}"] = getattr(plssvm, submodule)
