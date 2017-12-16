import numpy as np
from hyperspy import signals
from hyperspy.decorators import lazifyTestClass


@lazifyTestClass
class TestGetLineProfile:

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
