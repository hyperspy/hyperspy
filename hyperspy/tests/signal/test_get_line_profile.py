import pytest
import numpy as np
import matplotlib.pyplot as plt
from hyperspy import signals
from hyperspy.decorators import lazifyTestClass


@lazifyTestClass
class TestGetLineProfileValues:

    def teardown_method(self):
        plt.close('all')

    def test_all_ones_2d(self):
        s = signals.BaseSignal(np.ones((10, 12)))
        s_line = s.get_line_profile()
        assert len(s_line.axes_manager.shape) == 1
        assert (s_line.data == 1.).all()

    def test_all_ones_3d(self):
        s = signals.BaseSignal(np.ones((8, 10, 12)))
        s_line = s.get_line_profile(interactive=False)
        assert len(s_line.axes_manager.shape) == 2
        assert (s_line.data == 1.).all()

    def test_all_ones_4d(self):
        s = signals.BaseSignal(np.ones((8, 10, 12, 14)))
        s_line = s.get_line_profile(interactive=False)
        assert len(s_line.axes_manager.shape) == 3
        assert (s_line.data == 1.).all()

    def test_line_profile_values(self):
        data = np.ones((10, 20))
        data[:, 6:9] = 2.0
        s = signals.BaseSignal(data)
        s_line0 = s.get_line_profile(
                x1=7, y1=0, x2=7, y2=9, linewidth=1)
        assert s_line0.axes_manager[0].size == 10
        assert (s_line0.data == 2).all()

        s_line1 = s.get_line_profile(
                x1=0.0001, y1=5, x2=19, y2=5, linewidth=1)
        assert s_line1.axes_manager[0].size == 20
        assert (s_line1.data[0:6] == 1).all()
        assert (s_line1.data[6:9] == 2).all()
        assert (s_line1.data[9:] == 1).all()

    def test_specify_axes(self):
        data = np.ones((10, 5, 20, 30), dtype=np.int8)
        s = signals.Signal1D(data)
        s.axes_manager.navigation_axes[0].name = 'n0'
        s.axes_manager.navigation_axes[1].name = 'n1'
        s.axes_manager.navigation_axes[2].name = 'n2'
        s.axes_manager.signal_axes[0].name = 's0'
        s_line0_0 = s.get_line_profile(axes=(0, 1))
        s_line0_1 = s.get_line_profile(axes=('n0', 'n1'))
        s_line0_2 = s.get_line_profile(
                axes=(s.axes_manager[0], s.axes_manager[1]))
        assert s_line0_0.data.shape == s_line0_1.data.shape
        assert s_line0_1.data.shape == s_line0_2.data.shape

        s_line1_0 = s.get_line_profile(axes=(1, 2))
        s_line1_1 = s.get_line_profile(axes=('n1', 'n2'))
        s_line1_2 = s.get_line_profile(
                axes=(s.axes_manager[1], s.axes_manager[2]))
        assert s_line1_0.data.shape == s_line1_1.data.shape
        assert s_line1_1.data.shape == s_line1_2.data.shape

        assert not s_line0_0.data.shape == s_line1_0.data.shape


class TestGetLineProfileAxes:

    def setup_method(self):
        data = np.ones((10, 5, 20, 4, 3), dtype=np.int8)
        s = signals.Signal2D(data)
        s.axes_manager.navigation_axes[0].name = 'n0'
        s.axes_manager.navigation_axes[1].name = 'n1'
        s.axes_manager.navigation_axes[2].name = 'n2'
        s.axes_manager.signal_axes[0].name = 's0'
        s.axes_manager.signal_axes[1].name = 's1'
        self.s = s

    def teardown_method(self):
        plt.close('all')

    def test_specify_axes(self):
        s = self.s
        s_line0_0 = s.get_line_profile(axes=(0, 1))
        s_line0_1 = s.get_line_profile(axes=('n0', 'n1'))
        s_line0_2 = s.get_line_profile(
                axes=(s.axes_manager[0], s.axes_manager[1]))
        assert s_line0_0.data.shape == s_line0_1.data.shape
        assert s_line0_1.data.shape == s_line0_2.data.shape

    def test_compare_output(self):
        s = self.s
        s_line0 = s.get_line_profile(axes=(0, 1))
        s_line1 = s.get_line_profile(axes=(1, 2))
        assert not s_line0.data.shape == s_line1.data.shape

    def test_wrong_input_nav_and_sig(self):
        s = self.s
        with pytest.raises(ValueError):
            s.get_line_profile(axes=('n0', 's0'))

    def test_wrong_input_number_of_axes(self):
        s = self.s
        with pytest.raises(ValueError):
            s.get_line_profile(axes=('n0'))
        with pytest.raises(ValueError):
            s.get_line_profile(axes=('n0', 'n1', 'n2'))

    def test_axes_type(self):
        s = self.s
        s_line = s.get_line_profile(axes_type='nav')
        line_signal_shape = s_line.axes_manager.signal_shape
        orig_signal_shape = s.axes_manager.signal_shape
        assert orig_signal_shape == line_signal_shape
        with pytest.raises(ValueError):
            s.get_line_profile(axes_type='sig')
        with pytest.raises(ValueError):
            s.get_line_profile(axes_type='wrong_input')


@lazifyTestClass
class TestGetLineProfileMisc:

    def teardown_method(self):
        plt.close('all')

    def test_interactive(self):
        s = signals.BaseSignal(np.ones((10, 12)))
        s_line0 = s.get_line_profile(interactive=False)
        s_line1 = s.get_line_profile()
        assert (s_line0.data == s_line1.data).all()
