# import the plssvm module explicitly
from . import plssvm
# export everything
from .plssvm import *  # noqa: F405

# explicitly set the module level attributes
__doc__ = plssvm.__doc__
__version__ = plssvm.__version__
