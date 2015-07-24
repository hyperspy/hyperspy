import nose.tools as nt
import numpy as np

from hyperspy.signal import Signal


class TestSignalFolding:

    def setUp(self):
        self.s = Signal(np.empty((2, 3, 4, 5)))
        self.s.axes_manager.set_signal_dimension(2)

    def test_unfold_navigation(self):
        s = self.s.deepcopy()
        s.unfold_navigation_space()
        nt.assert_equal(s.axes_manager.navigation_shape,
                        (self.s.axes_manager.navigation_size,))

    def test_unfold_signal(self):
        s = self.s.deepcopy()
        s.unfold_signal_space()
        nt.assert_equal(s.axes_manager.signal_shape,
                        (self.s.axes_manager.signal_size,))

    def test_unfolded_repr(self):
        self.s.unfold()
        nt.assert_true("unfolded" in repr(self.s))

    def test_unfolded_context_manager(self):
        s = self.s.deepcopy()
        with s.unfolded():
            # Check that both spaces unfold as expected
            nt.assert_equal(s.axes_manager.navigation_shape,
                            (self.s.axes_manager.navigation_size,))
            nt.assert_equal(s.axes_manager.signal_shape,
                            (self.s.axes_manager.signal_size,))
        # Check that it folds back as expected
        nt.assert_equal(s.axes_manager.navigation_shape,
                        self.s.axes_manager.navigation_shape)
        nt.assert_equal(s.axes_manager.signal_shape,
                        self.s.axes_manager.signal_shape)


class TestSignalVarianceFolding:

    def setUp(self):
        self.s = Signal(np.empty((2, 3, 4, 5)))
        self.s.axes_manager.set_signal_dimension(2)
        self.s.estimate_poissonian_noise_variance()

    def test_unfold_navigation(self):
        s = self.s.deepcopy()
        s.unfold_navigation_space()
        meta_am = s.metadata.Signal.Noise_properties.variance.axes_manager
        nt.assert_equal(
            meta_am.navigation_shape, (self.s.axes_manager.navigation_size,))

    def test_unfold_signal(self):
        s = self.s.deepcopy()
        s.unfold_signal_space()
        meta_am = s.metadata.Signal.Noise_properties.variance.axes_manager
        nt.assert_equal(
            meta_am.signal_shape, (self.s.axes_manager.signal_size,))
