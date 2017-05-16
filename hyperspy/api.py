from hyperspy.api_nogui import *

__doc__ = hyperspy.api_nogui.__doc__

try:
    # Register ipywidgets by importing the module
    import hyperspy_gui_ipywidgets
except ImportError:
    _logger.warning(
        "ipywidgets GUI elements not available because the "
        "hyperspy_gui_ipywidgets package does not seem to be installed.")
try:
    # Register traitui UI elements by importing the module
    import hyperspy_gui_traitsui
except ImportError:
    _logger.warning(
        "traitsui GUI elements not available because the "
        "traitsui package is not installed.")
