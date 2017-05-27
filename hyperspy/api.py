from hyperspy.api_nogui import *
import logging
_logger = logging.getLogger(__name__)

__doc__ = hyperspy.api_nogui.__doc__

try:
    # Register ipywidgets by importing the module
    import hyperspy_gui_ipywidgets
except ImportError:
    _logger.warning(
        "The ipywidgets GUI elements are not available, probably because the "
        "hyperspy_gui_ipywidgets package is not installed.")
try:
    # Register traitui UI elements by importing the module
    import hyperspy_gui_traitsui
except ImportError:
    _logger.warning(
        "The traitsui GUI elements are not available, probably because the "
        "hyperspy_gui_traitui package is not installed.")
