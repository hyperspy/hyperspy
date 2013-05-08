import numpy as np
from nose.tools import (
    assert_true,
    assert_equal,
    assert_not_equal,
    raises)

from hyperspy.signal import Signal

class Test2D:
    def setUp(self):
        self.signal = Signal({'data' : np.arange(5*10).reshape(5,10)})
        self.signal.axes_manager[0].name = "x"
        self.signal.axes_manager[1].name = "E"
        self.signal.axes_manager[0].scale = 0.5
        self.data = self.signal.data.copy()
        
    def test_axis_by_str(self):
        s1 = self.signal.deepcopy()
        s2 = self.signal.deepcopy()
        s1.crop(0, 2,4)
        s2.crop("x", 2, 4)
        assert_true((s1.data==s2.data).all())
        
    def test_crop_int(self):
        s = self.signal
        d = self.data
        s.crop(0, 2,4)
        assert_true((s.data==d[2:4,:]).all())
        
    def test_crop_float(self):
        s = self.signal
        d = self.data
        s.crop(0, 2,2.)
        assert_true((s.data==d[2:4,:]).all())

