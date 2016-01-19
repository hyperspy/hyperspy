import numpy as np
import nose.tools as nt

from os import remove
from hyperspy.signal import Signal
import h5py


class PlotMetadataAxesDeepcopy:

    def test_axes_deepcopy(self):
        dc = self.s.deepcopy()
        s = self.s
        np.testing.assert_equal(
            s.axes_manager._get_axes_dicts(),
            dc.axes_manager._get_axes_dicts())
        nt.assert_is_not(s.axes_manager, dc.axes_manager)

    def test_metadata_deepcopy(self):
        dc = self.s.deepcopy()
        s = self.s
        nt.assert_dict_equal(
            s.metadata.as_dictionary(),
            dc.metadata.as_dictionary())
        nt.assert_is_not(s.metadata, dc.metadata)

    def test_plot_deepcopy(self):
        dc = self.s.deepcopy()
        s = self.s
        nt.assert_is_none(dc._plot)
        nt.assert_is(s._plot, self.plot_obj)


class TestInmemoryDeepcopy(PlotMetadataAxesDeepcopy):

    def setUp(self):
        s = Signal(np.arange(5 * 10).reshape(5, 10))
        s.axes_manager[0].name = "x"
        s.axes_manager[1].name = "E"
        s.axes_manager[0].scale = 0.5
        s.metadata.General.title = 'Some signal title'
        s.metadata.test_tuple = ('one', 'two', 'three')
        self.plot_obj = [42, 1. / 137]
        s._plot = self.plot_obj
        self.s = s

    def test_data_deepcopy(self):
        dc = self.s.deepcopy()
        s = self.s
        np.testing.assert_equal(s.data, dc.data)
        nt.assert_is_not(s.data, dc.data)

    def test_deepcopy_with_new_oom_data(self):
        s = self.s
        from hyperspy.io_plugins.hdf5 import get_temp_hdf5_file, write_empty_signal
        import h5py
        tempf = get_temp_hdf5_file()
        s.data = write_empty_signal(tempf.file,
                                    s.data.shape,
                                    s.data.dtype,
                                    metadata=s.metadata)['data']
        dc = s.deepcopy()
        nt.assert_is_instance(dc.data, h5py.Dataset)


class TestOOMDeepcopy(PlotMetadataAxesDeepcopy):

    def setUp(self):
        f = h5py.File('tmp.hdf5')
        data = f.create_dataset(
            'data',
            data=np.arange(
                5 *
                10).reshape(
                5,
                10),
            maxshape=(
                None,
                None))
        s = Signal(data)
        s.axes_manager[0].name = "x"
        s.axes_manager[1].name = "E"
        s.axes_manager[0].scale = 0.5
        s.metadata.General.title = 'Some signal title'
        s.metadata.test_tuple = ('one', 'two', 'three')
        self.plot_obj = [42, 1. / 137]
        s._plot = self.plot_obj
        self.s = s
        self.file = f

    def test_tempfile_assignment(self):
        dc = self.s.deepcopy()
        nt.assert_true(hasattr(dc, '_tempfile'))
        nt.assert_is_not_none(dc._tempfile)
        nt.assert_equal(dc._tempfile.file, dc.data.file)

    def test_deepcopy_data(self):
        dc = self.s.deepcopy()
        s = self.s
        nt.assert_not_equal(s.data, dc.data)
        nt.assert_not_equal(s.data.file, dc.data.file)
        np.testing.assert_almost_equal(s.data[:], dc.data[:])

    def test_deepcopy_data_properties(self):
        dc = self.s.deepcopy()
        s = self.s
        nt.assert_equal(dc.data.fillvalue, s.data.fillvalue)
        nt.assert_equal(dc.data.maxshape, s.data.maxshape)
        nt.assert_equal(dc.data.chunks, s.data.chunks)

    def test_deepcopy_with_new_data_none(self):
        s = self.s
        dc = s._deepcopy_with_new_data()
        np.testing.assert_equal(np.zeros(s.data.shape)[:], dc.data[:])

    def tearDown(self):
        remove('tmp.hdf5')
