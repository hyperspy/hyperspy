from traitsui.menu import (OKButton, CancelButton, OKCancelButtons)
import traitsui.api as tu

from hyperspy.gui_traitsui.buttons import *
from hyperspy.gui_traitsui.utils import (
    register_traitsui_widget, add_display_arg)


class SmoothingHandler(tu.Handler):

    def close(self, info, is_ok):
        # Removes the span selector from the plot
        if is_ok is True:
            info.object.apply()
        else:
            info.object.close()
        return True


class SpanSelectorInSignal1DHandler(tu.Handler):

    def close(self, info, is_ok):
        # Removes the span selector from the plot
        info.object.span_selector_switch(False)
        if is_ok is True:
            self.apply(info)

        return True

    def close_directly(self, info):
        if (info.ui.owner is not None) and self.close(info, False):
            info.ui.owner.close()

    def apply(self, info, *args, **kwargs):
        """Handles the **Apply** button being clicked.

        """
        obj = info.object
        obj.is_ok = True
        if hasattr(obj, 'apply'):
            obj.apply()

        return

    def next(self, info, *args, **kwargs):
        """Handles the **Next** button being clicked.

        """
        obj = info.object
        obj.is_ok = True
        if hasattr(obj, 'next'):
            next(obj)
        return


class Signal1DRangeSelectorHandler(tu.Handler):

    def close(self, info, is_ok):
        # Removes the span selector from the plot
        info.object.span_selector_switch(False)
        if is_ok is True:
            self.apply(info)
        return True

    def apply(self, info, *args, **kwargs):
        """Handles the **Apply** button being clicked.

        """
        obj = info.object
        if obj.ss_left_value != obj.ss_right_value:
            info.object.span_selector_switch(False)
            for method, cls in obj.on_close:
                method(cls, obj.ss_left_value, obj.ss_right_value)
            info.object.span_selector_switch(True)

        obj.is_ok = True

        return


class CalibrationHandler(SpanSelectorInSignal1DHandler):

    def apply(self, info, *args, **kwargs):
        """Handles the **Apply** button being clicked.
        """
        if info.object.signal is None:
            return
        axis = info.object.axis
        axis.scale = info.object.scale
        axis.offset = info.object.offset
        axis.units = info.object.units
        info.object.span_selector_switch(on=False)
        info.object.signal._replot()
        info.object.span_selector_switch(on=True)
        info.object.last_calibration_stored = True
        return


class ImageContrastHandler(tu.Handler):

    def close(self, info, is_ok):
        # Removes the span selector from the plot
        #        info.object.span_selector_switch(False)
        #        if is_ok is True:
        #            self.apply(info)
        if is_ok is False:
            info.object.image.update()
        info.object.close()
        return True

    def apply(self, info):
        """Handles the **Apply** button being clicked.

        """
        obj = info.object
        obj.apply()

        return

    @staticmethod
    def reset(info):
        """Handles the **Apply** button being clicked.

        """
        obj = info.object
        obj.reset()
        return

    @staticmethod
    def our_help(info):
        """Handles the **Apply** button being clicked.

        """
        info.object._help()


@register_traitsui_widget(toolkey="Signal1D.calibrate")
@add_display_arg
def calibration_traitsui(obj, **kwargs):
    view = tu.View(
        tu.Group(
            'left_value',
            'right_value',
            tu.Item('ss_left_value',
                    label='Left',
                    style='readonly'),
            tu.Item('ss_right_value',
                    label='Right',
                    style='readonly'),
            tu.Item(name='offset',
                    style='readonly'),
            tu.Item(name='scale',
                    style='readonly'),
            'units',),
        handler=CalibrationHandler,
        buttons=[OKButton, OurApplyButton, CancelButton],
        kind='live',
        title='Calibration parameters')
    return obj, {"view": view}


@register_traitsui_widget(toolkey="interactive_range_selector")
@add_display_arg
def interactive_range_selector(obj, **kwargs):
    view = tu.View(
        tu.Item('ss_left_value', label='Left', style='readonly'),
        tu.Item('ss_right_value', label='Right', style='readonly'),
        handler=Signal1DRangeSelectorHandler,
        buttons=[OKButton, OurApplyButton, CancelButton],)
    return obj, {"view": view}


@register_traitsui_widget(toolkey="Signal1D.smooth_savitzky_golay")
@add_display_arg
def smooth_savitzky_golay_traitsui(obj, **kwargs):
    view = tu.View(
        tu.Group(
            tu.Group(
                'window_length',
                tu.Item(
                    'decrease_window_length',
                    show_label=False),
                tu.Item(
                    'increase_window_length',
                    show_label=False),
                orientation="horizontal"),
            'polynomial_order',
            tu.Item(
                name='differential_order',
                tooltip='The order of the derivative to compute. This must '
                'be a nonnegative integer. The default is 0, which '
                'means to filter the data without differentiating.',
            ),
            # 'line_color',
        ),
        kind='live',
        handler=SmoothingHandler,
        buttons=OKCancelButtons,
        title='Savitzky-Golay Smoothing',
    )
    return obj, {"view": view}


@register_traitsui_widget(toolkey="Signal1D.smooth_lowess")
@add_display_arg
def smooth_lowess_traitsui(obj, **kwargs):
    view = tu.View(
        tu.Group(
            'smoothing_parameter',
            'number_of_iterations',
            # 'line_color',
        ),
        kind='live',
        handler=SmoothingHandler,
        buttons=OKCancelButtons,
        title='Lowess Smoothing',)
    return obj, {"view": view}


@register_traitsui_widget(toolkey="Signal1D.smooth_total_variation")
@add_display_arg
def smooth_tv_traitsui(obj, **kwargs):
    view = tu.View(
        tu.Group(
            'smoothing_parameter',
            # 'line_color',
        ),
        kind='live',
        handler=SmoothingHandler,
        buttons=OKCancelButtons,
        title='Total Variation Smoothing',)
    return obj, {"view": view}


@register_traitsui_widget(toolkey="Signal1D.smooth_butterworth")
@add_display_arg
def smooth_butterworth(obj, **kwargs):
    view = tu.View(
        tu.Group(
            'cutoff_frequency_ratio',
            'order',
            'type'),
        kind='live',
        handler=SmoothingHandler,
        buttons=OKCancelButtons,
        title='Butterworth filter',)
    return obj, {"view": view}


@register_traitsui_widget(toolkey="load")
@add_display_arg
def load(obj, **kwargs):
    view = tu.View(
        tu.Group('filename', "lazy"),
        kind='livemodal',
        buttons=[OKButton, CancelButton],
        title='Load file')
    return obj, {"view": view}


@register_traitsui_widget(toolkey="Signal1D.contrast_editor")
@add_display_arg
def image_constast_editor_traitsui(obj, **kwargs):
    view = tu.View(tu.Item('ss_left_value',
                           label='vmin',
                           show_label=True,
                           style='readonly',),
                   tu.Item('ss_right_value',
                           label='vmax',
                           show_label=True,
                           style='readonly'),
                   handler=ImageContrastHandler,
                   buttons=[OKButton,
                            OurApplyButton,
                            OurResetButton,
                            CancelButton, ],
                   title='Constrast adjustment tool',
                   )
    return obj, {"view": view}


@register_traitsui_widget(toolkey="Signal1D.integrate_in_range")
@add_display_arg
def integrate_in_range_traitsui(obj, **kwargs):
    view = tu.View(
        buttons=[OKButton, CancelButton],
        title='Integrate in range',
        handler=SpanSelectorInSignal1DHandler,
    )
    return obj, {"view": view}


@register_traitsui_widget(toolkey="Signal1D.remove_background")
@add_display_arg
def remove_background_traitsui(obj, **kwargs):
    view = tu.View(
        tu.Group(
            'background_type',
            'fast',
            tu.Group(
                'polynomial_order',
                visible_when='background_type == \'Polynomial\''), ),
        buttons=[OKButton, CancelButton],
        handler=SpanSelectorInSignal1DHandler,
        title='Background removal tool',
        resizable=True,
        width=300,
    )
    return obj, {"view": view}


class SpikesRemovalHandler(tu.Handler):

    def close(self, info, is_ok):
        # Removes the span selector from the plot
        info.object.span_selector_switch(False)
        return True

    def apply(self, info, *args, **kwargs):
        """Handles the **Apply** button being clicked.

        """
        obj = info.object
        obj.is_ok = True
        if hasattr(obj, 'apply'):
            obj.apply()

        return

    def find(self, info, *args, **kwargs):
        """Handles the **Next** button being clicked.

        """
        obj = info.object
        obj.is_ok = True
        if hasattr(obj, 'find'):
            obj.find()
        return

    def back(self, info, *args, **kwargs):
        """Handles the **Next** button being clicked.

        """
        obj = info.object
        obj.is_ok = True
        if hasattr(obj, 'find'):
            obj.find(back=True)
        return


@register_traitsui_widget(toolkey="Signal1D.spikes_removal_tool")
@add_display_arg
def spikes_removal_traitsui(obj, **kwargs):

    thisOKButton = tu.Action(name="OK",
                             action="OK",
                             tooltip="Close the spikes removal tool")

    thisApplyButton = tu.Action(name="Remove spike",
                                action="apply",
                                tooltip="Remove the current spike by "
                                "interpolating\n"
                                       "with the specified settings (and find\n"
                                       "the next spike automatically)")
    thisFindButton = tu.Action(name="Find next",
                               action="find",
                               tooltip="Find the next (in terms of navigation\n"
                               "dimensions) spike in the data.")

    thisPreviousButton = tu.Action(name="Find previous",
                                   action="back",
                                   tooltip="Find the previous (in terms of "
                                   "navigation\n"
                                          "dimensions) spike in the data.")
    view = tu.View(tu.Group(
        tu.Group(
            tu.Item('click_to_show_instructions',
                    show_label=False, ),
            tu.Item('show_derivative_histogram',
                    show_label=False,
                    tooltip="To determine the appropriate threshold,\n"
                    "plot the derivative magnitude histogram, \n"
                    "and look for outliers at high magnitudes \n"
                    "(which represent sudden spikes in the data)"),
            'threshold',
            show_border=True,
        ),
        tu.Group(
            'add_noise',
            'interpolator_kind',
            'default_spike_width',
            tu.Group(
                'spline_order',
                enabled_when='interpolator_kind == \'Spline\''),
            show_border=True,
            label='Advanced settings'),
    ),
        buttons=[thisOKButton,
                 thisPreviousButton,
                 thisFindButton,
                 thisApplyButton, ],
        handler=SpikesRemovalHandler,
        title='Spikes removal tool',
        resizable=False,
    )
    return obj, {"view": view}
