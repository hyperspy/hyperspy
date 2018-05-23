from hyperspy.misc.utils import signal_range_from_roi
from hyperspy import roi


def test_signal_range_from_roi():
    sr = roi.SpanROI(20, 50)
    left, right = signal_range_from_roi(sr)
    assert left == 20
    assert right == 50
    left, right = signal_range_from_roi((20, 50))
    assert left == 20
    assert right == 50
