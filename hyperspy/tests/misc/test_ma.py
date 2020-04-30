import pytest
import numpy as np
import dask.array as da
from hyperspy.signals import Signal2D
from hyperspy._lazy_signals import LazySignal2D
from hyperspy.roi import CircleROI,RectangularROI
from hyperspy.api import load
import tempfile
import os
import gc


@pytest.fixture()
def tmpfilepath():
    with tempfile.TemporaryDirectory() as tmp:
        yield os.path.join(tmp, "test")
        gc.collect()

class TestMa():
    @pytest.fixture
    def signal(self):
        signal = Signal2D(np.ones((3,5,5)))
        return signal

    @pytest.fixture
    def lazy_signal(self):
        signal = LazySignal2D(da.ones((3,5,5)))
        return signal

    def test_save(self, signal, lazy_signal, tmpfilepath):
        signal.as_masked()
        signal.isig[0:2, :] = np.ma.masked
        assert isinstance(signal.data, np.ma.masked_array)
        signal.save(tmpfilepath)
        l = load(tmpfilepath + ".hspy")
        assert isinstance(l.data, np.ma.masked_array)
        lazy_signal.as_masked()
        lazy_signal.data = da.ma.masked_greater(lazy_signal.data,5)
        lazy_signal.save(tmpfilepath, overwrite=True)
        l = load(tmpfilepath + ".hspy")
        assert isinstance(l.data, np.ma.masked_array)

    def test_as_mask(self, signal,lazy_signal):
        signal.as_masked()
        assert isinstance(signal.data, np.ma.MaskedArray)
        lazy_signal.as_masked()
        print(lazy_signal.data)


    @pytest.mark.parametrize("outside", [True, False])
    @pytest.mark.parametrize("axes", ["signal", [0,1],[0,2]])
    @pytest.mark.parametrize("inner_r",[None, 1])
    def test_masked_circle_roi(self,signal,lazy_signal,outside,axes,inner_r):
        c = CircleROI(2,2,2, inner_r)
        c.mask(signal, axes=axes, outside=outside)
        assert np.ma.is_masked(signal.data)
        c.mask(lazy_signal, axes=axes, outside=outside)
        # Can't check lazy signal for is_masked...

    @pytest.mark.parametrize("outside", [True,False])
    def test_masked_rectangle_roi(self,signal,lazy_signal, outside):
        r = RectangularROI(0,2,4,4)
        r.mask(signal, outside=outside)
        assert np.ma.is_masked(signal.data)
        r.mask(lazy_signal,outside=outside)
