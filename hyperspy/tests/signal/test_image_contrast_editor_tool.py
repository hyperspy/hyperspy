import numpy as np

import hyperspy.api as hs
from hyperspy.signal_tools import ImageContrastEditor


class TestContrastEditorTool:

    def setup_method(self, method):
        s = hs.signals.Signal2D(np.arange(2*3*10*10).reshape(2, 3, 10, 10))
        self.s = s

    def test_reset_vmin_vmax(self):
        s = self.s
        s.plot(vmin='10th', vmax='99th')
        ceditor = ImageContrastEditor(s._plot.signal_plot)

        np.testing.assert_allclose(ceditor._vmin, 9.9)
        np.testing.assert_allclose(ceditor._vmax, 98.01)

        ceditor._vmin = 20
        ceditor._vmax = 90
        ceditor._reset_original_settings()

        np.testing.assert_allclose(ceditor._vmin, 9.9)
        np.testing.assert_allclose(ceditor._vmax, 98.01)

    def test_reset_span_selector(self):
        s = self.s
        s.plot(vmin='10th', vmax='99th')
        ceditor = ImageContrastEditor(s._plot.signal_plot)

        ceditor.span_selector.set_initial((20, 90))
        ceditor.update_span_selector()
        ax_image = s._plot.signal_plot.ax.images[0]

        np.testing.assert_allclose(ax_image.norm.vmin, 20)
        np.testing.assert_allclose(ax_image.norm.vmax, 90)

        ceditor._reset_span_selector()

        np.testing.assert_allclose(ax_image.norm.vmin, 9.9)
        np.testing.assert_allclose(ax_image.norm.vmax, 98.01)

    def test_change_navigation_coordinate(self):
        s = self.s
        s.plot(vmin='10th', vmax='99th')
        ceditor = ImageContrastEditor(s._plot.signal_plot)

        np.testing.assert_allclose(ceditor._vmin, 9.9)
        np.testing.assert_allclose(ceditor._vmax, 98.01)
        try:
            # Convenience to be able to run test on systems using backends
            # supporting blit
            s.axes_manager.indices = (1, 1)
        except TypeError:
            pass

        np.testing.assert_allclose(ceditor._vmin, 409.9)
        np.testing.assert_allclose(ceditor._vmax, 498.01)

    def test_vmin_vmax_changed(self):
        s = self.s
        s.plot(vmin='0th', vmax='100th')
        ceditor = ImageContrastEditor(s._plot.signal_plot)

        np.testing.assert_allclose(ceditor._vmin, 0.0)
        np.testing.assert_allclose(ceditor._vmax, 99.0)
        try:
            # Convenience to be able to run test on systems using backends
            # supporting blit
            ceditor._vmin_percentile_changed(0, 10)
        except TypeError:
            pass
        try:
            # Convenience to be able to run test on systems using backends
            # supporting blit
            ceditor._vmax_percentile_changed(100, 99)
        except TypeError:
            pass

        np.testing.assert_allclose(ceditor._vmin, 9.9)
        np.testing.assert_allclose(ceditor._vmax, 98.01)
