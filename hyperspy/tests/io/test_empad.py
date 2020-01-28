
import os
import traits.api as t
import numpy as np
import numpy.testing as nt

import hyperspy.api as hs


DATA_DIR = os.path.join(os.path.dirname(__file__), 'empad_data')
FILENAME_STACK_RAW = os.path.join(DATA_DIR, 'series_x1000.raw')
FILENAME_MAP_RAW = os.path.join(DATA_DIR, 'scan_x64_y64.raw')


def _create_raw_data(filename, shape):
    size = np.prod(shape)
    data = np.arange(size).reshape(shape).astype('float32')
    data.tofile(filename)


def setup_module():
    _create_raw_data(FILENAME_STACK_RAW, (16640000,))
    _create_raw_data(FILENAME_MAP_RAW, (64*64*130*128))


def teardown_module():
    if os.path.exists(FILENAME_STACK_RAW):
        os.remove(FILENAME_STACK_RAW)
    if os.path.exists(FILENAME_MAP_RAW):
        os.remove(FILENAME_MAP_RAW)


class TestEMPAD:

    def test_read_stack(self):
        s = hs.load(os.path.join(DATA_DIR, 'stack_images.xml'))
        # nt.assert_allclose(s.data, np.arange(16640000).astype('float32'))
        assert s.data.dtype == 'float32'
        signal_axes = s.axes_manager.signal_axes
        assert signal_axes[0].name == 'width'
        assert signal_axes[1].name == 'height'
        for axis in signal_axes:
            assert axis.units == t.Undefined
            assert axis.scale == 1.0
        navigation_axes = s.axes_manager.navigation_axes
        assert navigation_axes[0].name == 'series_count'
        assert navigation_axes[0].units == 'ms'
        assert navigation_axes[0].scale == 1.0

        assert s.metadata.General.date == '2019-06-07'
        assert s.metadata.General.time == '13:17:22.590279'

    def test_read_map(self):
        s = hs.load(os.path.join(DATA_DIR, 'map64x64.xml'))
        assert s.data.dtype == 'float32'
        signal_axes = s.axes_manager.signal_axes
        assert signal_axes[0].name == 'width'
        assert signal_axes[1].name == 'height'
        for axis in signal_axes:
            assert axis.units == '1/nm'
            nt.assert_allclose(axis.scale, 0.1826537)
        navigation_axes = s.axes_manager.navigation_axes
        assert navigation_axes[0].name == 'scan_y'
        assert navigation_axes[1].name == 'scan_x'
        for axis in navigation_axes:
            assert axis.units == 'µm'
            nt.assert_allclose(axis.scale, 0.071349103)

        assert s.metadata.General.date == '2019-06-06'
        assert s.metadata.General.time == '13:30:00.164675'
