import types
from unittest import mock

from hyperspy.defaults_parser import preferences
from hyperspy.drawing._widgets.rectangles import RectangleWidget
from hyperspy.drawing.image import ImagePlot
from hyperspy.drawing.mpl_hse import MPL_HyperSignal1D_Explorer
from hyperspy.drawing.widget import ResizableDraggableWidgetBase


def _make_handler(klass, method_name, target):
    """Bind a real instance method to a mock object."""
    method = getattr(klass, method_name)
    setattr(target, method_name, types.MethodType(method, target))


class TestKeyHandlerPreferences:
    def test_key2switch_right_pointer_reads_preference(self):
        explorer = mock.MagicMock(right_pointer_on=False)
        _make_handler(MPL_HyperSignal1D_Explorer, "key2switch_right_pointer", explorer)

        original = preferences.Plot.key_toggle_pointer
        try:
            preferences.Plot.key_toggle_pointer = "q"

            event = mock.MagicMock(key="e")
            explorer.key2switch_right_pointer(event)
            assert explorer.right_pointer_on is False, (
                '"e" should not toggle when key_toggle_pointer is "q"'
            )

            event = mock.MagicMock(key="q")
            explorer.key2switch_right_pointer(event)
            assert explorer.right_pointer_on is True
        finally:
            preferences.Plot.key_toggle_pointer = original

    def test_image_on_key_press_contrast_reads_preference(self):
        plot = mock.MagicMock()
        plot.gui_adjust_contrast = mock.MagicMock()
        plot.toggle_norm = mock.MagicMock()
        _make_handler(ImagePlot, "on_key_press", plot)

        original = preferences.Plot.key_adjust_contrast
        try:
            preferences.Plot.key_adjust_contrast = "z"

            event = mock.MagicMock(key="h")
            plot.on_key_press(event)
            plot.gui_adjust_contrast.assert_not_called()

            event = mock.MagicMock(key="z")
            plot.on_key_press(event)
            plot.gui_adjust_contrast.assert_called_once()
        finally:
            preferences.Plot.key_adjust_contrast = original

    def test_image_on_key_press_log_reads_preference(self):
        plot = mock.MagicMock()
        plot.gui_adjust_contrast = mock.MagicMock()
        plot.toggle_norm = mock.MagicMock()
        _make_handler(ImagePlot, "on_key_press", plot)

        original = preferences.Plot.key_toggle_log
        try:
            preferences.Plot.key_toggle_log = "g"

            event = mock.MagicMock(key="l")
            plot.on_key_press(event)
            plot.toggle_norm.assert_not_called()

            event = mock.MagicMock(key="g")
            plot.on_key_press(event)
            plot.toggle_norm.assert_called_once()
        finally:
            preferences.Plot.key_toggle_log = original

    def test_widget_on_key_press_increase_reads_preference(self):
        widget = mock.MagicMock()
        widget.increase_size = mock.MagicMock()
        widget.decrease_size = mock.MagicMock()
        _make_handler(ResizableDraggableWidgetBase, "on_key_press", widget)

        original = preferences.Plot.key_widget_increase
        try:
            preferences.Plot.key_widget_increase = "="

            event = mock.MagicMock(key="+")
            widget.on_key_press(event)
            widget.increase_size.assert_not_called()

            event = mock.MagicMock(key="=")
            widget.on_key_press(event)
            widget.increase_size.assert_called_once()
        finally:
            preferences.Plot.key_widget_increase = original

    def test_widget_on_key_press_decrease_reads_preference(self):
        widget = mock.MagicMock()
        widget.increase_size = mock.MagicMock()
        widget.decrease_size = mock.MagicMock()
        _make_handler(ResizableDraggableWidgetBase, "on_key_press", widget)

        original = preferences.Plot.key_widget_decrease
        try:
            preferences.Plot.key_widget_decrease = "_"

            event = mock.MagicMock(key="-")
            widget.on_key_press(event)
            widget.decrease_size.assert_not_called()

            event = mock.MagicMock(key="_")
            widget.on_key_press(event)
            widget.decrease_size.assert_called_once()
        finally:
            preferences.Plot.key_widget_decrease = original

    def test_rectangle_on_key_press_reads_preference(self):
        rect = mock.MagicMock(selected=True)
        rect._increase_xsize = mock.MagicMock()
        rect._decrease_xsize = mock.MagicMock()
        rect._increase_ysize = mock.MagicMock()
        rect._decrease_ysize = mock.MagicMock()
        _make_handler(RectangleWidget, "on_key_press", rect)

        original = preferences.Plot.key_rectangle_x_increase
        try:
            preferences.Plot.key_rectangle_x_increase = "j"

            event = mock.MagicMock(key="j")
            rect.on_key_press(event)
            rect._increase_xsize.assert_called_once()
        finally:
            preferences.Plot.key_rectangle_x_increase = original

    def test_rectangle_on_key_press_ignores_when_not_selected(self):
        rect = mock.MagicMock(selected=False)
        rect._increase_xsize = mock.MagicMock()
        _make_handler(RectangleWidget, "on_key_press", rect)

        event = mock.MagicMock(key=preferences.Plot.key_rectangle_x_increase)
        rect.on_key_press(event)
        rect._increase_xsize.assert_not_called()
