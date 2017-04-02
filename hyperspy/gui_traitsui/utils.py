import functools

from hyperspy.ui_registry import register_widget


register_traitsui_widget = functools.partial(
    register_widget, toolkit="traitsui")


def add_display_arg(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        display = kwargs.pop("display", True)
        obj, kwargs = f(*args, **kwargs)
        if display:
            obj.edit_traits(**kwargs)
        else:
            return obj
    return wrapper
