# -*- coding: utf-8 -*-
# Copyright 2007-2016 The HyperSpy developers
#
# This file is part of  HyperSpy.
#
#  HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  HyperSpy.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import numpy.testing as npt
import os
import tempfile
import pytest
from time import perf_counter, sleep

from hyperspy.io import load, save
from hyperspy.signals import Signal2D

#==============================================================================
# MRCZ Test
#
# Internal python-only test. Build a random image and save and re-load it.
#==============================================================================
tmpDir = tempfile.gettempdir()
mrcName = os.path.join(tmpDir, "testMage.mrcz")
MAX_ASYNC_TIME = 2.0
dtype_list = ['float32', 'int8', 'int16', 'uint16', 'complex64']


def _generate_parameters():
    parameters = []
    for dtype in ['float32', 'int8', 'int16', 'uint16', 'complex64']:
        for compressor in [None, 'zstd', 'lz4']:
            for clevel in [1, 9]:
                parameters.append([dtype, compressor, clevel])
    return parameters


class TestPythonMrcz:

    def setup_method(self, method):
        pass

    def compareSaveLoad(self, testShape, dtype='int8', compressor=None,
                        clevel=1, do_async=False, **kwargs):
        # This is the main function which reads and writes from disk.
        dtype = np.dtype(dtype)
        if dtype == 'float32' or dtype == 'float64':
            testData = np.random.normal(size=testShape).astype(dtype)
        elif dtype == 'complex64' or dtype == 'complex128':
            testData = np.random.normal(size=testShape).astype(
                dtype) + 1.0j * np.random.normal(size=testShape).astype(dtype)
        else:  # integers
            testData = np.random.randint(10, size=testShape).astype(dtype)

        testSignal = Signal2D(testData)
        # Unfortunately one cannot iterate over axes_manager in a Pythonic way
        # for axis in testSignal.axes_manager:
        testSignal.axes_manager[0].name = 'z'
        testSignal.axes_manager[0].scale = np.random.uniform(low=0.0, high=1.0)
        testSignal.axes_manager[0].units = '\AA'
        testSignal.axes_manager[1].name = 'x'
        testSignal.axes_manager[1].scale = np.random.uniform(low=0.0, high=1.0)
        testSignal.axes_manager[1].units = '\AA'
        testSignal.axes_manager[2].name = 'y'
        testSignal.axes_manager[2].scale = np.random.uniform(low=0.0, high=1.0)
        testSignal.axes_manager[2].units = '\AA'

        # Meta-data that goes into MRC fixed header
        testSignal.metadata.set_item(
            'Acquisition_instrument.TEM.beam_energy', 300.0)
        # Meta-data that goes into JSON extended header
        testSignal.metadata.set_item(
            'Acquisition_instrument.TEM.magnification', 25000)

        save(mrcName, testSignal, compressor=compressor,
             clevel=clevel, do_async=do_async, **kwargs)
        if do_async:
            # Poll file on disk since we don't return the
            # concurrent.futures.Future
            t_stop = perf_counter() + MAX_ASYNC_TIME
            sleep(0.005)
            while(perf_counter() < t_stop):
                try:
                    fh = open(mrcName, 'a')
                    fh.close()
                    break
                except IOError:
                    sleep(0.001)

        reSignal = load(mrcName)
        try:
            os.remove(mrcName)
        except IOError:
            print("Warning: file {} left on disk".format(mrcName))

        npt.assert_array_almost_equal(
            testSignal.data.shape,
            reSignal.data.shape)
        npt.assert_array_almost_equal(testSignal.data, reSignal.data)
        npt.assert_almost_equal(
            testSignal.metadata.Acquisition_instrument.TEM.beam_energy,
            reSignal.metadata.Acquisition_instrument.TEM.beam_energy)
        npt.assert_almost_equal(
            testSignal.metadata.Acquisition_instrument.TEM.magnification,
            reSignal.metadata.Acquisition_instrument.TEM.magnification)
        # Comparing axes is a bit difficult, due to the mixed format ordering
        # print( testSignal.axes_manager )
        # print( reSignal.axes_manager )
        # for I in range(3):
        #     print( "Test #{}  name: {}, scale: {:.3f}".format(I, testSignal.axes_manager[I].name, testSignal.axes_manager[I].scale ) )
        # print( "Re   #{}  name: {}, scale: {:.3f}".format(I,
        # reSignal.axes_manager[I].name, reSignal.axes_manager[I].scale ) )
        for aName in ['x', 'y', 'z']:
            npt.assert_equal(
                testSignal.axes_manager[aName].size,
                reSignal.axes_manager[aName].size)
            npt.assert_almost_equal(
                testSignal.axes_manager[aName].scale,
                reSignal.axes_manager[aName].scale)

        # These assertions need to be True
#        assert testSignal.axes_manager.as_dictionary() == reSignal.axes_manager.as_dictionary()

        return reSignal

    @pytest.mark.parametrize(("dtype", "compressor", "clevel"),
                             _generate_parameters())
    def test_MRC_uncompressed(self, dtype, compressor, clevel):
        t_start = perf_counter()
        t_start = perf_counter()
        self.compareSaveLoad([2, 64, 32], dtype=dtype, compressor=compressor,
                             clevel=clevel)
        print("MRCZ test ({}, {} {}) finished in {} s".format(
            dtype, compressor, clevel, perf_counter() - t_start))

    @pytest.mark.parametrize("dtype", dtype_list)
    def test_Async(self, dtype):
        t_start = perf_counter()
        self.compareSaveLoad([2, 64, 32], dtype=dtype, compressor='zstd',
                             clevel=1, do_async=True)
        print("MRCZ Asychronous test finished in {} s".format(
            perf_counter() - t_start))


if __name__ == '__main__':
    theSuite = TestPythonMrcz()
    parameters = _generate_parameters()
    for parameter in parameters:
        theSuite.test_MRC_uncompressed(*parameter)
    theSuite.test_Async(parameter[0])
