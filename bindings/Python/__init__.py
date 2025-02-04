# import all bindings from the compiled PLSSVM module
from .plssvm import *

# explicitly set the module level attributes
__doc__ = plssvm.__doc__
__version__ = plssvm.__version__