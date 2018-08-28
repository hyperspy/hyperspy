'''Registry of user interface widgets.

Format {"tool_key" : {"toolkit" : <function(obj, display, **kwargs)>}}

The ``tool_key` is defined by the "model function" to which the widget provides
and user interface. That function gets the widget function from this registry
and executes it passing the ``obj``, ``display`` and any extra keyword
arguments. When ``display`` is true, ``function`` displays the widget. If
``False`` it returns a dictionary with whatever is needed to display the
widgets externally (usually for testing or customisation purposes).

'''

import functools
import types

from hyperspy.misc.utils import isiterable


UI_REGISTRY = {}

TOOLKIT_REGISTRY = set()
KNOWN_TOOLKITS = set(("ipywidgets", "traitsui"))


def register_widget(toolkit, toolkey):
    """Decorator to register a UI widget.

    Parameters
    ----------
    f: function
        Function that returns or display the UI widget. The signature must
        include ``obj``, ``display`` and ``**kwargs``.
    toolkit: string
        The name of the widget toolkit e.g. ipywidgets
    toolkey: string
        The "key" of the tool for which the widget provides an interface. If
        the toolkey is not in the ``UI_REGISTRY`` dictionary a ``NameError``
        is raised.

    Returns
    -------
    widgets: dictionary or None
        Dictionary containing the widget objects if display is False, else None.

    """
    if not toolkey in UI_REGISTRY:
        raise NameError("%s is not a registered toolkey" % toolkey)
    TOOLKIT_REGISTRY.add(toolkit)

    def decorator(f):
        UI_REGISTRY[toolkey][toolkit] = f
        return f
    return decorator


def register_toolkey(toolkey):
    """Register a toolkey.

    Parameters
    ----------
    toolkey: string

    """
    if toolkey in UI_REGISTRY:
        raise NameError(
            "Another tool has been registered with the same name.")
    UI_REGISTRY[toolkey] = {}


def _toolkits_to_string(toolkits):
    if isinstance(toolkits, str):
        return "{} toolkit".format(toolkits)
    else:
        toolkits = tuple(toolkits)
        if len(toolkits) == 1:
            return "{} toolkit".format(toolkits[0])

        elif len(toolkits) == 2:
            return " and ".join(toolkits) + " toolkits"
        else:  # > 2
            txt = ", ".join(toolkits[:-1])
            return txt + " and {}".format(toolkits[-1]) + " toolkits"


def get_gui(self, toolkey, display=True, toolkit=None, **kwargs):
    if not TOOLKIT_REGISTRY:
        raise ImportError(
            "No toolkit registered. Install hyperspy_gui_ipywidgets or "
            "hyperspy_gui_traitsui GUI elements. If hyperspy_gui_traits"
            "is installed, initialize a toolkit supported by traitsui "
            "before importing HyperSpy."
        )
    from hyperspy.defaults_parser import preferences
    if isinstance(toolkit, str):
        toolkit = (toolkit,)
    if isiterable(toolkit):
        toolkits = set()
        for tk in toolkit:
            if tk in TOOLKIT_REGISTRY:
                toolkits.add(tk)
            else:
                raise ValueError(
                    "{} is not a registered toolkit.".format(tk)
                )
    elif toolkit is None:
        toolkits = set()
        available_disabled_toolkits = set()
        if "ipywidgets" in TOOLKIT_REGISTRY:
            if preferences.GUIs.enable_ipywidgets_gui:
                toolkits.add("ipywidgets")
            else:
                available_disabled_toolkits.add("ipywidgets")
        if "traitsui" in TOOLKIT_REGISTRY:
            if preferences.GUIs.enable_traitsui_gui:
                toolkits.add("traitsui")
            else:
                available_disabled_toolkits.add("traitsui")
        if not toolkits and available_disabled_toolkits:
            is_or_are = "is" if len(
                available_disabled_toolkits) == 1 else "are"
            them_or_it = ("it" if len(available_disabled_toolkits) == 1
                          else "them")
            raise ValueError(
                "No toolkit available. The {} {} installed but "
                "disabled in `preferences`. Enable {} in `preferences` or "
                "manually select a toolkit with the `toolkit` argument.".format(
                    _toolkits_to_string(available_disabled_toolkits),
                    is_or_are, them_or_it)
            )

    else:
        raise ValueError(
            "`toolkit` must be a string, an iterable of strings or None.")
    if toolkey not in UI_REGISTRY or not UI_REGISTRY[toolkey]:
        propose = KNOWN_TOOLKITS - TOOLKIT_REGISTRY
        if propose:
            propose = ["hyperspy_gui_{}".format(tk) for tk in propose]
            if len(propose) > 1:
                propose_ = ", ".join(propose[:-1])
                propose = propose_ + " and/or {}".format(propose[-1])
            else:
                propose = propose.pop()
        raise NotImplementedError(
            "There is no user interface registered for this feature."
            "Try installing {}.".format(propose))
    if not display:
        widgets = {}
    available_toolkits = set()
    used_toolkits = set()
    for toolkit, f in UI_REGISTRY[toolkey].items():
        if toolkit in toolkits:
            used_toolkits.add(toolkit)
            thisw = f(obj=self, display=display, **kwargs)
            if not display:
                widgets[toolkit] = thisw
        else:
            available_toolkits.add(toolkit)
    if not used_toolkits and available_toolkits:
        is_or_are = "is" if len(toolkits) == 1 else "are"
        raise NotImplementedError(
            "The {} {} not available for this functionality,try with "
            "the {}.".format(
                _toolkits_to_string(toolkits),
                is_or_are,
                _toolkits_to_string(available_toolkits)))
    if not display:
        return widgets


def get_partial_gui(toolkey):
    def pg(self, display=True, toolkit=None, **kwargs):
        return get_gui(self, toolkey=toolkey, display=display,
                       toolkit=toolkit, **kwargs)
    return pg


DISPLAY_DT = """display: bool
    If True, display the user interface widgets. If False, return the widgets
    container in a dictionary, usually for customisation or testing."""

TOOLKIT_DT = """toolkit: str, iterable of strings or None
    If None (default), all available widgets are displayed or returned. If
    string, only the widgets of the selected toolkit are displayed if available.
    If an interable of toolkit strings, the widgets of all listed toolkits are
    displayed or returned."""
GUI_DT = """Display or return interactive GUI element if available.

Parameters
----------
%s
%s

""" % (DISPLAY_DT, TOOLKIT_DT)


def add_gui_method(toolkey):
    def decorator(cls):
        register_toolkey(toolkey)
        # Not using functools.partialmethod because it is not possible to set
        # the docstring that way.
        setattr(cls, "gui", get_partial_gui(toolkey))
        setattr(cls.gui, "__doc__", GUI_DT)
        return cls
    return decorator


register_toolkey("interactive_range_selector")
register_toolkey("navigation_sliders")
register_toolkey("load")
