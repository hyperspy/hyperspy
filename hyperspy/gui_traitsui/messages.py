import traitsui.api as tu
from traitsui.menu import OKButton, CancelButton

class MessageHandler(tu.Handler):

    def close(self, info, is_ok):
        # Removes the span selector from the plot
        if is_ok is True:
            info.object.is_ok = True
        else:
            info.object.is_ok = False
        return True

information_view = tu.View(tu.Group(
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
