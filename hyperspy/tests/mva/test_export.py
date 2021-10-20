import numpy as np

from hyperspy import signals


class TestMVAExport:

    def setup_method(self, method):
        s = signals.Signal1D(np.random.random((2, 3, 4, 5)))
        sa = s.axes_manager[-1]
        na = s.axes_manager[0]
        sa.offset = 100
        sa.scale = 0.1
        s.learning_results.factors = np.arange(5 * 5).reshape((5, 5))
        s.learning_results.loadings = np.arange(24 * 5).reshape((24, 5))
        s.learning_results.bss_factors = np.arange(5 * 2).reshape((5, 2))
        s.learning_results.bss_loadings = np.arange(24 * 2).reshape((24, 2))
        self.s = s
        self.na = na
        self.sa = sa

    def test_get_bss_factor(self):
        bss_factors = self.s.get_bss_factors()
        assert bss_factors.axes_manager[-1].scale == self.sa.scale
        assert bss_factors.axes_manager[-1].offset == self.sa.offset
        assert (bss_factors.axes_manager.signal_shape ==
                self.s.axes_manager.signal_shape)

    def test_get_bss_loadings(self):
        bss_loadings = self.s.get_bss_loadings()
        assert bss_loadings.axes_manager.navigation_dimension == 1
        assert (bss_loadings.axes_manager.signal_shape ==
                self.s.axes_manager.navigation_shape)
