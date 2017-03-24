'''Registry of user interface widgets.

Format {"tool_key" : {"widget_name" : <function(obj, display, **kwargs)>}}

The ``tool_key` is defined by the "model function" to which the widget provides
and user interface. That function gets the widget function from this registry
and executes it passing the ``obj``, ``display`` and any extra keyword
arguments. When ``display`` is true, ``function`` displays the widget. If
``False`` it returns a dictionary with whatever is needed to display the
widgets externally (usually for testing or customisation purposes).

'''

ui_registry = {}
