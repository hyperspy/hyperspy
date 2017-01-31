
import numpy as np

from hyperspy.signal import BaseSignal


class TestSignalFolding:

    def setup_method(self, method):
        self.s = BaseSignal(np.zeros((2, 3, 4, 5)))
        self.s = self.s.transpose(signal_axes=2)

    def test_unfold_navigation(self):
        s = self.s.deepcopy()
        s.unfold_navigation_space()
        assert (s.axes_manager.navigation_shape ==
                (self.s.axes_manager.navigation_size,))

    def test_unfold_signal(self):
        s = self.s.deepcopy()
        s.unfold_signal_space()
        assert (s.axes_manager.signal_shape ==
                (self.s.axes_manager.signal_size,))

    def test_unfolded_repr(self):
        self.s.unfold()
        assert "unfolded" in repr(self.s)

    def test_unfold_navigation_by_keyword(self):
        s = self.s.deepcopy()
        s.unfold(unfold_navigation=True, unfold_signal=False)
        assert (s.axes_manager.navigation_shape ==
                (self.s.axes_manager.navigation_size,))

    def test_unfold_signal_by_keyword(self):
        s = self.s.deepcopy()
        s.unfold(unfold_navigation=False, unfold_signal=True)
        assert (s.axes_manager.signal_shape ==
                (self.s.axes_manager.signal_size,))

    def test_unfold_nothing_by_keyword(self):
        s = self.s.deepcopy()
        s.unfold(unfold_navigation=False, unfold_signal=False)
        assert s.data.shape == self.s.data.shape

    def test_unfold_full_by_keyword(self):
        s = self.s.deepcopy()
        s.unfold(unfold_navigation=True, unfold_signal=True)
        assert (s.axes_manager.signal_shape ==
                (self.s.axes_manager.signal_size,))
        assert (s.axes_manager.navigation_shape ==
                (self.s.axes_manager.navigation_size,))

    def test_unfolded_context_manager(self):
        s = self.s.deepcopy()
        with s.unfolded():
            # Check that both spaces unfold as expected
            assert (s.axes_manager.navigation_shape ==
                    (self.s.axes_manager.navigation_size,))
            assert (s.axes_manager.signal_shape ==
                    (self.s.axes_manager.signal_size,))
        # Check that it folds back as expected
        assert (s.axes_manager.navigation_shape ==
                self.s.axes_manager.navigation_shape)
        assert (s.axes_manager.signal_shape ==
                self.s.axes_manager.signal_shape)

    def test_unfolded_full_by_keywords(self):
        s = self.s.deepcopy()
        with s.unfolded(unfold_navigation=True, unfold_signal=True) as folded:
            assert folded
            # Check that both spaces unfold as expected
            assert (s.axes_manager.navigation_shape ==
                    (self.s.axes_manager.navigation_size,))
            assert (s.axes_manager.signal_shape ==
                    (self.s.axes_manager.signal_size,))
        # Check that it folds back as expected
        assert (s.axes_manager.navigation_shape ==
                self.s.axes_manager.navigation_shape)
        assert (s.axes_manager.signal_shape ==
                self.s.axes_manager.signal_shape)

    def test_unfolded_navigation_by_keyword(self):
        s = self.s.deepcopy()
        with s.unfolded(unfold_navigation=True, unfold_signal=False) as folded:
            assert folded
            # Check that only navigation space unfolded
            assert (s.axes_manager.navigation_shape ==
                    (self.s.axes_manager.navigation_size,))
            assert (s.axes_manager.signal_shape ==
                    self.s.axes_manager.signal_shape)
        # Check that it folds back as expected
        assert (s.axes_manager.navigation_shape ==
                self.s.axes_manager.navigation_shape)
        assert (s.axes_manager.signal_shape ==
                self.s.axes_manager.signal_shape)

    def test_unfolded_signal_by_keyword(self):
        s = self.s.deepcopy()
        with s.unfolded(unfold_navigation=False, unfold_signal=True) as folded:
            assert folded
            # Check that only signal space unfolded
            assert (s.axes_manager.navigation_shape ==
                    self.s.axes_manager.navigation_shape)
            assert (s.axes_manager.signal_shape ==
                    (self.s.axes_manager.signal_size,))
        # Check that it folds back as expected
        assert (s.axes_manager.navigation_shape ==
                self.s.axes_manager.navigation_shape)
        assert (s.axes_manager.signal_shape ==
                self.s.axes_manager.signal_shape)

    def test_unfolded_nothin_by_keyword(self):
        s = self.s.deepcopy()
        with s.unfolded(False, False) as folded:
            assert not folded
            # Check that nothing folded
            assert (s.axes_manager.navigation_shape ==
                    self.s.axes_manager.navigation_shape)
            assert (s.axes_manager.signal_shape ==
                    self.s.axes_manager.signal_shape)
        # Check that it "folds back" as expected
        assert (s.axes_manager.navigation_shape ==
                self.s.axes_manager.navigation_shape)
        assert (s.axes_manager.signal_shape ==
                self.s.axes_manager.signal_shape)


class TestSignalVarianceFolding:

    def setup_method(self, method):
        self.s = BaseSignal(np.zeros((2, 3, 4, 5)))
        self.s = self.s.transpose(signal_axes=2)
        self.s.estimate_poissonian_noise_variance()

    def test_unfold_navigation(self):
        s = self.s.deepcopy()
        s.unfold_navigation_space()
        meta_am = s.metadata.Signal.Noise_properties.variance.axes_manager
        assert (
            meta_am.navigation_shape == (self.s.axes_manager.navigation_size,))

    def test_unfold_signal(self):
        s = self.s.deepcopy()
        s.unfold_signal_space()
        meta_am = s.metadata.Signal.Noise_properties.variance.axes_manager
        assert (
            meta_am.signal_shape == (self.s.axes_manager.signal_size,))
