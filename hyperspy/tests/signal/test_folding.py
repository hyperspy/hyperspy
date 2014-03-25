import nose.tools
import numpy as np

from hyperspy.signal import Signal


class TestSignalFolding:

    def setUp(self):
        self.s = Signal(np.empty((2, 3, 4, 5)))
        self.s.axes_manager.set_signal_dimension(2)

    def test_unfold_navigation(self):
        s = self.s.deepcopy()
        s.unfold_navigation_space()
        nose.tools.assert_equal(s.axes_manager.navigation_shape,
                                (self.s.axes_manager.navigation_size,))

    def test_unfold_signal(self):
        s = self.s.deepcopy()
        s.unfold_signal_space()
        nose.tools.assert_equal(s.axes_manager.signal_shape,
                                (self.s.axes_manager.signal_size,))

    def test_unfolded_repr(self):
        self.s.unfold()
        nose.tools.assert_true("unfolded" in repr(self.s))


class TestSignalVarianceFolding:

    def setUp(self):
        self.s = Signal(np.empty((2, 3, 4, 5)))
        self.s.axes_manager.set_signal_dimension(2)
        self.s.estimate_poissonian_noise_variance()

    def test_unfold_navigation(self):
        s = self.s.deepcopy()
        s.unfold_navigation_space()
        nose.tools.assert_equal(s.metadata.Signal.Noise_properties.variance.axes_manager.navigation_shape,
                                (self.s.axes_manager.navigation_size,))

    def test_unfold_signal(self):
        s = self.s.deepcopy()
        s.unfold_signal_space()
        nose.tools.assert_equal(s.metadata.Signal.Noise_properties.variance.axes_manager.signal_shape,
                                (self.s.axes_manager.signal_size,))
