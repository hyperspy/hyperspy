from hyperspy.api_nogui import *
import logging
_logger = logging.getLogger(__name__)

__doc__ = hyperspy.api_nogui.__doc__

try:
    # Register ipywidgets by importing the module
    import hyperspy_gui_ipywidgets
except ImportError:
    from hyperspy.defaults_parser import preferences
    if preferences.GUIs.warn_if_guis_are_missing:
        _logger.warning(
            "The ipywidgets GUI elements are not available, probably because the "
            "hyperspy_gui_ipywidgets package is not installed.")
try:
    # Register traitui UI elements by importing the module
    import hyperspy_gui_traitsui
except ImportError:
    from hyperspy.defaults_parser import preferences
    if preferences.GUIs.warn_if_guis_are_missing:
        _logger.warning(
            "The traitsui GUI elements are not available, probably because the "
            "hyperspy_gui_traitsui package is not installed.")
