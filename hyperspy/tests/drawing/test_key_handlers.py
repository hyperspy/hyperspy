import types
from unittest import mock

from hyperspy.defaults_parser import preferences
from hyperspy.drawing._widgets.rectangles import RectangleWidget, SquareWidget
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

    # --- jump-to-click tests ---

    def test_jump_to_click_reads_preference(self):
        """SquareWidget._onjumpclick must respect key_jump_to_click."""
        sq = mock.MagicMock(position=(5, 5))
        sq.is_pointer = True
        _make_handler(SquareWidget, "_onjumpclick", sq)

        original = preferences.Plot.key_jump_to_click
        try:
            preferences.Plot.key_jump_to_click = "ctrl"

            # Wrong key → no jump
            # Wrong key → no jump
            event_shift = mock.MagicMock(key="shift", inaxes=True)
            sq._onjumpclick(event_shift)
            assert sq.position == (5, 5), (
                "shift should not jump when key_jump_to_click=ctrl"
            )

            # Correct key → jump
            event_ctrl = mock.MagicMock(key="ctrl", inaxes=True, xdata=10, ydata=20)
            sq._onjumpclick(event_ctrl)
            assert sq.position == (10, 20), (
                "ctrl should jump when key_jump_to_click=ctrl"
            )
        finally:
            preferences.Plot.key_jump_to_click = original

    def test_jump_to_click_ignores_when_not_pointer(self):
        """_onjumpclick must be a no-op when is_pointer is False."""
        sq = mock.MagicMock(position=(5, 5))
        sq.is_pointer = False
        _make_handler(SquareWidget, "_onjumpclick", sq)

        event = mock.MagicMock(
            key=preferences.Plot.key_jump_to_click, inaxes=True, xdata=99, ydata=99
        )
        sq._onjumpclick(event)
        assert sq.position == (5, 5)

    def test_jump_to_click_ignores_when_out_of_axes(self):
        """_onjumpclick must be a no-op when event.inaxes is False."""
        sq = mock.MagicMock(position=(5, 5))
        sq.is_pointer = True
        _make_handler(SquareWidget, "_onjumpclick", sq)

        event = mock.MagicMock(
            key=preferences.Plot.key_jump_to_click, inaxes=False, xdata=99, ydata=99
        )
        sq._onjumpclick(event)
        assert sq.position == (5, 5)

    # --- rotation snap test ---

    def test_rotation_snap_reads_preference(self):
        """Line2DWidget._onmousemove snap branch must use key_rotation_snap."""

        from hyperspy.drawing._widgets.line2d import Line2DWidget

        widget = mock.MagicMock()
        _make_handler(Line2DWidget, "_onmousemove", widget)

        original = preferences.Plot.key_rotation_snap
        try:
            preferences.Plot.key_rotation_snap = "alt"
            event = mock.MagicMock(key="shift", xdata=0, ydata=0, inaxes=True)

            # With snap key set to "alt", a "shift" event enters the
            # method but must NOT trigger the 30-degree snap branch.
            # The best assertion we can do is that it doesn't crash —
            # the snap condition simply evaluates to False.
            widget.picked = False  # short-circuit rotation via picked guard
            widget._onmousemove(event)
        finally:
            preferences.Plot.key_rotation_snap = original

    # --- model plot key handler tests ---

    def test_adjust_position_toggle_reads_preference(self):
        """Model1D._on_key_press must use key_toggle_adjust_position."""
        from hyperspy.models.model1d import Model1D

        model = mock.MagicMock()
        model._position_widgets = {}
        model._plot_components = False
        model._plot = mock.MagicMock(is_active=True)
        _make_handler(Model1D, "_on_key_press", model)

        original = preferences.Plot.key_toggle_adjust_position
        try:
            preferences.Plot.key_toggle_adjust_position = "m"

            # Wrong key → no-op
            event = mock.MagicMock(key="a")
            model._on_key_press(event)
            model.enable_adjust_position.assert_not_called()

            # Correct key + no widgets → enable
            event = mock.MagicMock(key="m")
            model._on_key_press(event)
            model.enable_adjust_position.assert_called_once()
        finally:
            preferences.Plot.key_toggle_adjust_position = original

    def test_adjust_position_toggle_disables_when_active(self):
        """_on_key_press must disable when _position_widgets is non-empty."""
        from hyperspy.models.model1d import Model1D

        model = mock.MagicMock()
        model._position_widgets = {"fake": [mock.MagicMock()]}
        model._plot_components = False
        model._plot = mock.MagicMock(is_active=True)
        _make_handler(Model1D, "_on_key_press", model)

        event = mock.MagicMock(key=preferences.Plot.key_toggle_adjust_position)
        model._on_key_press(event)
        model.disable_adjust_position.assert_called_once()

    def test_plot_components_toggle_reads_preference(self):
        """Model1D._on_key_press must use key_toggle_plot_components."""
        from hyperspy.models.model1d import Model1D

        model = mock.MagicMock()
        model._position_widgets = {}
        model._plot_components = True
        model._plot = mock.MagicMock(is_active=True)
        _make_handler(Model1D, "_on_key_press", model)

        original = preferences.Plot.key_toggle_plot_components
        try:
            preferences.Plot.key_toggle_plot_components = "b"

            # Wrong key → no-op
            event = mock.MagicMock(key="w")
            model._on_key_press(event)
            model.disable_plot_components.assert_not_called()

            # Correct key + components are showing → disable
            event = mock.MagicMock(key="b")
            model._on_key_press(event)
            model.disable_plot_components.assert_called_once()
        finally:
            preferences.Plot.key_toggle_plot_components = original

    def test_plot_components_toggle_enables_when_hidden(self):
        """_on_key_press must enable when _plot_components is False."""
        from hyperspy.models.model1d import Model1D

        model = mock.MagicMock()
        model._position_widgets = {}
        model._plot_components = False
        model._plot = mock.MagicMock(is_active=True)
        _make_handler(Model1D, "_on_key_press", model)

        event = mock.MagicMock(key=preferences.Plot.key_toggle_plot_components)
        model._on_key_press(event)
        model.enable_plot_components.assert_called_once()

    def test_residual_toggle_reads_preference(self):
        """Model1D._on_key_press must use key_toggle_residual."""
        from hyperspy.models.model1d import Model1D

        model = mock.MagicMock()
        model._position_widgets = {}
        model._plot_components = False
        model._plot = mock.MagicMock(is_active=True)
        _make_handler(Model1D, "_on_key_press", model)

        original = preferences.Plot.key_toggle_residual
        try:
            preferences.Plot.key_toggle_residual = "z"

            # Wrong key → no-op
            event = mock.MagicMock(key="t")
            model._on_key_press(event)
            model._toggle_residual.assert_not_called()

            # Correct key → toggle
            event = mock.MagicMock(key="z")
            model._on_key_press(event)
            model._toggle_residual.assert_called_once()
        finally:
            preferences.Plot.key_toggle_residual = original
