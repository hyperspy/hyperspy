import pytest
import numpy as np
import dask.array as da
from hyperspy.signals import Signal2D
from hyperspy._lazy_signals import LazySignal2D
from hyperspy.misc import ma
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

    def test_asarray(self,signal, lazy_signal):
        ma.asarray(signal)
        assert isinstance(signal.data, np.ma.masked_array)
        ma.asarray(lazy_signal)
        # should test that chuck is numpy masked array
        assert isinstance(lazy_signal.data, da.core.Array)

    def test_masked_equal(self,signal, lazy_signal):
        ma.masked_equal(signal, 1)
        assert isinstance(signal.data, np.ma.masked_array)
        ma.masked_equal(lazy_signal, 1)
        # should test that chuck is numpy masked array
        assert isinstance(lazy_signal.data, da.core.Array)

    def test_masked_greater_equal(self,signal, lazy_signal):
        ma.masked_greater_equal(signal, 1)
        assert isinstance(signal.data, np.ma.masked_array)
        ma.masked_greater_equal(lazy_signal, 1)
        # should test that chuck is numpy masked array
        assert isinstance(lazy_signal.data, da.core.Array)

    def test_masked_inside(self,signal, lazy_signal):
        ma.masked_inside(signal, 1, 2)
        assert isinstance(signal.data, np.ma.masked_array)
        ma.masked_inside(lazy_signal, 1, 2)
        # should test that chuck is numpy masked array
        assert isinstance(lazy_signal.data, da.core.Array)

    def test_masked_invalid(self,signal, lazy_signal):
        ma.masked_invalid(signal)
        assert isinstance(signal.data, np.ma.masked_array)
        ma.masked_invalid(lazy_signal)
        # should test that chuck is numpy masked array
        assert isinstance(lazy_signal.data, da.core.Array)

    def test_masked_less(self,signal, lazy_signal):
        ma.masked_less(signal,1)
        assert isinstance(signal.data, np.ma.masked_array)
        ma.masked_less(lazy_signal,1)
        # should test that chuck is numpy masked array
        assert isinstance(lazy_signal.data, da.core.Array)

    def test_masked_less_equal(self,signal, lazy_signal):
        ma.masked_less_equal(signal,1)
        assert isinstance(signal.data, np.ma.masked_array)
        ma.masked_less_equal(lazy_signal,1)
        # should test that chuck is numpy masked array
        assert isinstance(lazy_signal.data, da.core.Array)

    def test_masked_not_equal(self,signal, lazy_signal):
        ma.masked_not_equal(signal, 1)
        assert isinstance(signal.data, np.ma.masked_array)
        ma.masked_not_equal(lazy_signal,1)
        # should test that chuck is numpy masked array
        assert isinstance(lazy_signal.data,  da.core.Array)

    def test_masked_outside(self,signal, lazy_signal):
        ma.masked_outside(signal, 1, 2)
        assert isinstance(signal.data, np.ma.masked_array)
        ma.masked_outside(lazy_signal, 1, 2)
        # should test that chuck is numpy masked array
        assert isinstance(lazy_signal.data, da.core.Array)

    def test_masked_values(self,signal, lazy_signal):
        ma.masked_values(signal, 1.0)
        assert isinstance(signal.data, np.ma.masked_array)
        ma.masked_values(lazy_signal, 1.0)
        # should test that chuck is numpy masked array
        assert isinstance(lazy_signal.data, da.core.Array)

    def test_masked_where(self,signal, lazy_signal):
        ma.masked_where(signal.data==1, signal)
        assert isinstance(signal.data, np.ma.masked_array)
        ma.masked_where(lazy_signal.data==1, lazy_signal)
        # should test that chuck is numpy masked array
        assert isinstance(lazy_signal.data, da.core.Array)

    def test_mask_slicing(self,signal, lazy_signal):
        ma.asarray(signal)
        ma.asarray(lazy_signal)
        signal.isig[0:2, :] = ma.masked
        assert (np.ma.is_masked(signal.isig[0:2,:].data))
        # maks slicing doesn't work on lazy signals

    def test_save(self, signal, lazy_signal, tmpfilepath):
        ma.asarray(signal)
        assert isinstance(signal.data, np.ma.masked_array)
        signal.save(tmpfilepath)
        l = load(tmpfilepath + ".hspy")
        print(type(l.data))
        #assert isinstance(l.data, np.ma.masked_array)

    def test_masked_circle_roi(self,signal,lazy_signal):
        c = CircleROI(2,2,2)
        ma.masked_roi(signal,c, axes=[1,2])
        assert np.ma.is_masked(signal.isig[2,2].data)
        ma.masked_roi(lazy_signal, c)

    def test_masked_rectangle_roi(self,signal,lazy_signal):
        r = RectangularROI(0,2,4,4)
        ma.masked_roi(signal, r, axes=[1, 2])
        assert np.ma.is_masked(signal.isig[2,3].data)
        ma.masked_roi(lazy_signal, r)




