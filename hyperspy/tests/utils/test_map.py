import numpy as np

from hyperspy.signal import BaseSignal as hssig
from hyperspy.signals import Signal1D as hsspc
from hyperspy.signals import Signal2D as hsimg
from scipy.ndimage import gaussian_filter as gf
from itertools import cycle

class TestMapScalar:
    """
    Test map onto signal with scalar parameter(s). Proceed by applying a
    Gaussian filter to all the images in an image dataset using variable sigma
    parameter(s)
    """
    def setup_method(self, method):
        self.sigma  = 2.5
        self.sigmas = hssig(np.linspace(2,5,10)).T
        self.image  = hsimg(np.random.random((10, 64, 64)))
        self.image.metadata.General.title = 'test'

    def test_A(self):
        """
        Test A: The sigma parameter is a single scalar variable
        """
        sigma = self.sigma
        img   = self.image
        # smart way
        res_smart = img.map(gf, sigma=sigma, inplace=False)
        # non-smart way
        res_silly = np.stack( [gf(imi.data, sigma) for imi in img] )
        # Done.
        np.testing.assert_array_almost_equal( res_smart.data, res_silly )

    def test_B(self):
        """
        Test B: The sigma parameters are several scalar variables across the
        signal navigation space.
        """
        sigmas = self.sigmas
        img    = self.image
        # smart way
        res_smart = img.map(gf, sigma=sigmas, inplace=False)
        # non-smart way
        res_silly = np.stack( [gf(imi.data, sig()[0])
                             for imi, sig in zip(img, sigmas)])
        # Done
        np.testing.assert_array_almost_equal( res_smart.data, res_silly )

    def test_C(self):
        """
        Test C: The sigma parameter is a single scalar variable but with
        navigation space
        """
        sigma = hssig(self.sigma).T
        img   = self.image
        # smart way
        res_smart = img.map(gf, sigma=sigma, inplace=False)
        # non-smart way
        res_silly = np.stack( [gf(imi.data, sig()[0])
                             for imi, sig in zip(img, cycle(sigma))])

        # Done
        np.testing.assert_array_almost_equal( res_smart.data, res_silly )

    def test_D(self):
        """
        Test D: Compare C with D (yes, this was in fact a bit convoluted)
        """
        img = self.image
        # simple way
        sigma = self.sigma
        res_simple = img.map(gf, sigma=sigma, inplace=False)
        # convoluted way
        sigma = hssig(self.sigma).T
        res_convol = img.map(gf, sigma=sigma, inplace=False)
        # Done
        np.testing.assert_array_almost_equal( res_simple.data, res_convol.data )

class TestMapSignal:
    """
    Test map onto signal with signal parameter(s). Proceed by applying a
    very simple function and parameters that have to produce zero as result.
    """
    def setup_method(self, method):
        def foo(bar, val=0.):
            result = bar - val
            return result
        self.foo = foo

    def test_A(self):
        """
        Test A: The signal parameter has the same navigation space as the input
        signal
        """
        spc = hsspc(np.random.rand(10, 1024))
        res = spc.map(self.foo, val=spc, inplace=False)
        np.testing.assert_array_almost_equal( res.data,np.zeros_like(spc.data) )

    def test_B(self):
        """
        Test B: The signal parameter has no navigation space
        """
        spc = hsspc(np.repeat(np.random.rand(1, 1024), 10, 0))
        val = hsspc(spc.data[1])
        res = spc.map(self.foo, val=val, inplace=False)
        np.testing.assert_array_almost_equal( res.data,np.zeros_like(spc.data) )
