import traits.api as t
import traitsui.api as tu
from traitsui.menu import OKButton, CancelButton

from hyperspy.gui_traitsui.utils import (
    register_traitsui_widget, add_display_arg)


class Message(t.HasTraits):
    text = t.Str
    is_ok = t.Bool(False)

    def __init__(self, text):
        self.text = text


class MessageHandler(tu.Handler):

    def close(self, info, is_ok):
        # Removes the span selector from the plot
        if is_ok is True:
            info.object.is_ok = True
        else:
            info.object.is_ok = False
        return True


def information(text):
    message = Message(text)
    message.text = text
    view = tu.View(tu.Group(
        tu.Item('text',
                show_label=False,
                style='readonly',
                springy=True,
                width=300,
                padding=15),),
        kind='modal',
        buttons=[OKButton, CancelButton],
        handler=MessageHandler,
        title='Message')
    message.edit_traits(view=view)
    return message.is_ok

@register_traitsui_widget(toolkey="SimpleMessage")
@add_display_arg
def simple_message(obj, **kwargs):
    view = tu.View(tu.Group(
        tu.Item('text',
                show_label=False,
                style='readonly',
                springy=True,
                width=300,
                padding=15),),
        kind='modal',
        buttons=[OKButton],
        title='Message')
    return obj, {"view": view}
