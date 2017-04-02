import traitsui.api as tu

from hyperspy.gui_traitsui.utils import (
    register_traitsui_widget, add_display_arg)
from hyperspy.gui_traitsui.buttons import OurFitButton, OurCloseButton
from hyperspy.gui_traitsui.tools import SpanSelectorInSignal1DHandler


class ComponentFitHandler(SpanSelectorInSignal1DHandler):

    def fit(self, info):
        """Handles the **Apply** button being clicked.

        """
        obj = info.object
        obj._fit_fired()
        return


@register_traitsui_widget(toolkey="Model1D.fit_component")
@add_display_arg
def fit_component_tratisui(obj, **kwargs):
    fit_component_view = tu.View(
        tu.Item('only_current', show_label=True,),
        buttons=[OurFitButton, OurCloseButton],
        title='Fit single component',
        handler=ComponentFitHandler,
    )
    return obj, {"view": fit_component_view}
