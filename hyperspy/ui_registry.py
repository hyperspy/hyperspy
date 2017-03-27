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

ui_registry = {}

toolkit_registry = set()


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
        the toolkey is not in the ``ui_registry`` dictionary a ``NameError``
        is raised.

    """
    if not toolkey in ui_registry:
        raise NameError("%s is not a registered toolkey" % toolkey)
    toolkit_registry.add(toolkit)
    def decorator(f):
        ui_registry[toolkey][toolkit] = f
        return f
    return decorator

def register_toolkey(toolkey):
    """Register a toolkey.

    Parameters
    ----------
    toolkey: string

    """
    if toolkey in ui_registry:
        raise NameError(
            "Another tool has been registered with the same name.")
    ui_registry[toolkey] = {}

def _gui(self, toolkey, display=True, toolkit=None, **kwargs):
    error = "There is not user interface registered for this feature."
    from hyperspy.ui_registry import ui_registry
    toolkits = None
    if isinstance(toolkit, str):
        toolkits = (toolkit,)
    elif isiterable(toolkit):
        toolkits = toolkit
    elif toolkit is not None:
        raise ValueError(
            "`toolkit` must be a string, an iterable of strings or None.")
    if toolkey not in ui_registry or not ui_registry[toolkey]:
        raise NotImplementedError(error)
    for toolkit, f in ui_registry[toolkey].items():
        if toolkits is None or toolkit in toolkits:
            return f(obj=self, display=display, **kwargs)

def get_partial_gui(toolkey):
    def pg(self, display=True, toolkit=None, **kwargs):
        """testing
        """
        return _gui(self, toolkey=toolkey, display=True, toolkit=None, **kwargs)
    return pg



def gui(toolkey):
    def decorator(cls):
        register_toolkey(toolkey)
        # Not using functools.partialmethod because it is not possible to set
        # the docstring that way.
        setattr(cls, "gui", get_partial_gui(toolkey))
        # setattr(cls.gui, "__doc__", "testing")
        return cls
    return decorator
