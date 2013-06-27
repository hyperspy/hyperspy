import numpy as np
from nose.tools import (
    assert_true,
    assert_equal,
    assert_not_equal,
    raises)

from hyperspy.signal import Signal
from hyperspy import signals

class Test2D:
    def setUp(self):
        self.signal = Signal(np.arange(5*10).reshape(5,10))
        self.signal.axes_manager[0].name = "x"
        self.signal.axes_manager[1].name = "E"
        self.signal.axes_manager[0].scale = 0.5
        self.signal.mapped_parameters.set_item('splitting.axis', 0)
        self.signal.mapped_parameters.set_item(
                                        'splitting.step_sizes',[2,2])
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
        s.crop(0, 2, 2.)
        assert_true((s.data==d[2:4,:]).all())
        
    def test_split_axis0(self):
        result = self.signal.split(0,2)
        assert_true(len(result) == 2)
        assert_true((result[0].data == self.data[:2,:]).all())
        assert_true((result[1].data == self.data[2:4,:]).all())
    
    def test_split_axis1(self):
        result = self.signal.split(1,2)
        assert_true(len(result) == 2)
        assert_true((result[0].data == self.data[:,:5]).all())
        assert_true((result[1].data == self.data[:,5:]).all())
        
    def test_split_axisE(self):
        result = self.signal.split("E",2)
        assert_true(len(result) == 2)
        assert_true((result[0].data == self.data[:,:5]).all())
        assert_true((result[1].data == self.data[:,5:]).all())
        
    def test_split_default(self):
        result = self.signal.split()
        assert_true(len(result) == 2)
        assert_true((result[0].data == self.data[:2,:]).all())
        assert_true((result[1].data == self.data[2:4,:]).all())
        
class Test3D:
    def setUp(self):
        self.signal = Signal(np.arange(2*4*6).reshape(2,4,6))
        self.signal.axes_manager[0].name = "x"
        self.signal.axes_manager[1].name = "y"
        self.signal.axes_manager[2].name = "E"
        self.signal.axes_manager[0].scale = 0.5
        self.signal.mapped_parameters.set_item('splitting.axis', 0)
        self.signal.mapped_parameters.set_item(
                                        'splitting.step_sizes',[2,2])
        self.data = self.signal.data.copy()
    def test_rebin(self):
        s = self.signal
        s.rebin((2,1,6))
        assert_true(s.data.shape == (1,2,6))
        
    def test_swap_axes(self):
        s = self.signal
        assert_equal(s.swap_axes(0,1).data.shape, (4,2,6))
        assert_true(s.swap_axes(0,2).data.flags['C_CONTIGUOUS'])
        
class TestRollAxis:
    def setUp(self):
        s = signals.Spectrum(np.ones((5,4,3,6)))
        for axis, name in zip(
            s.axes_manager._get_axes_in_natural_order(),
            ['x', 'y', 'z', 'E']):
            axis.name = name
        self.s = s
    def test_int(self):
        assert_equal(self.s.rollaxis(2,0).data.shape, (4, 3, 5, 6))
    def test_str(self):
        assert_equal(self.s.rollaxis("z", "x").data.shape, (4, 3, 5, 6))        
        
        

