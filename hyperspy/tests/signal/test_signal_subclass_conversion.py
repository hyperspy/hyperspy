from nose.tools import assert_true, assert_equal, raises
import numpy as np

from hyperspy.signal import Signal
from hyperspy.exceptions import DataDimensionError

class Test1d():
    def setUp(self):    
        self.s = Signal(np.arange(2))
        
    @raises(DataDimensionError)
    def test_as_image(self):
        assert_true((self.s.data == self.s.as_image((0,1)).data).all())
        
    def test_as_spectrum(self):
        assert_true((self.s.data == self.s.as_spectrum(0).data).all())

class Test2d():
    def setUp(self):    
        self.s = Signal(np.random.random((2,3)))
        
    def test_as_image_T(self):
        assert_true(
            self.s.data.T.shape == self.s.as_image((0,1)).data.shape)
    def test_as_image(self):
        assert_true(
            self.s.data.shape == self.s.as_image((1,0)).data.shape)       
    def test_as_spectrum_T(self):
        assert_true(
            self.s.data.T.shape == self.s.as_spectrum(0).data.shape)

    def test_as_spectrum(self):
        assert_true(
            self.s.data.shape == self.s.as_spectrum(1).data.shape)    


class Test3d():
    def setUp(self):    
        self.s = Signal(np.random.random((2,3,4)))
        
    def test_as_image_contigous(self):
        assert_true(self.s.as_image((0,1)).data.flags['C_CONTIGUOUS'])
        
    def test_as_image_1(self):
        assert_equal(
            self.s.as_image((0,1)).data.shape, (4, 2, 3))
            
    def test_as_image_2(self):
        assert_equal(
            self.s.as_image((1,0)).data.shape, (4, 3, 2))
            
    def test_as_image_3(self):
        assert_equal(
            self.s.as_image((1,2)).data.shape, (3, 4, 2))
        
    def test_as_spectrum_contigous(self):
        assert_true(self.s.as_spectrum(0).data.flags['C_CONTIGUOUS'])
            
    def test_as_spectrum_0(self):
        assert_equal(
            self.s.as_spectrum(0).data.shape, (2, 4, 3))  
    
    def test_as_spectrum_1(self):
        assert_equal(
            self.s.as_spectrum(1).data.shape, (3, 4, 2))
            
    def test_as_spectrum_2(self):
        assert_equal(
            self.s.as_spectrum(1).data.shape, (3, 4, 2))
            
    def test_as_spectrum_3(self):
        assert_equal(
            self.s.as_spectrum(2).data.shape, (2, 3, 4))