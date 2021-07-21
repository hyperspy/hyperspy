import tempfile

import dask.array as da
import h5py
import numpy as np
import pytest

from hyperspy import api as hs

usid = pytest.importorskip("pyUSID", reason="pyUSID not installed")
sidpy = pytest.importorskip("sidpy", reason="sidpy not installed")


# ##################### HELPER FUNCTIONS ######################################

def _array_translator_basic_checks(h5_f):
    assert len(h5_f.items()) == 1
    assert 'Measurement_000' in h5_f.keys()
    h5_meas_grp = h5_f['Measurement_000']
    assert isinstance(h5_meas_grp, h5py.Group)

    # Again, this group should only have one group - Channel_000
    assert len(h5_meas_grp.items()) == 1
    assert 'Channel_000' in h5_meas_grp.keys()
    h5_chan_grp = h5_meas_grp['Channel_000']
    assert isinstance(h5_chan_grp, h5py.Group)

    # This channel group will contain the main dataset
    assert len(h5_chan_grp.items()) == 5
    for dset_name in ['Raw_Data', 'Position_Indices', 'Position_Values',
                      'Spectroscopic_Indices', 'Spectroscopic_Values']:
        assert dset_name in h5_chan_grp.keys()
        h5_dset = h5_chan_grp[dset_name]
        assert isinstance(h5_dset, h5py.Dataset)

    assert len(usid.hdf_utils.get_all_main(h5_f)) == 1

    assert usid.hdf_utils.check_if_main(h5_chan_grp['Raw_Data'])


def _compare_axes(hs_axes, dim_descriptors, usid_val_func, axes_defined=True,
                  invalid_axes=False):
    usid_descriptors = [_split_descriptor(_) for _ in dim_descriptors]
    for dim_ind, axis in enumerate(hs_axes):
        # Remember that the signal axes are actually reversed!
        real_dim_ind = len(hs_axes) - dim_ind - 1
        if axes_defined:
            assert axis.name == usid_descriptors[real_dim_ind][0]
            assert axis.units == usid_descriptors[real_dim_ind][1]

        if not invalid_axes:
            axis_vals = np.arange(axis.offset,
                                  axis.offset + axis.size * axis.scale,
                                  axis.scale)
            usid_vals = usid_val_func(usid_descriptors[real_dim_ind][0])
            np.testing.assert_allclose(axis_vals, usid_vals)


def _assert_empty_dims(hs_axes, usid_labels, usid_val_func):
    assert len(hs_axes) == 0
    assert len(usid_labels) == 1
    dim_vals = usid_val_func(usid_labels[0])
    assert len(dim_vals) == 1


def _split_descriptor(desc):
    desc = desc.strip()
    ind = desc.rfind('(')
    if ind < 0:
        ind = desc.rfind('[')
        if ind < 0:
            return desc, ''

    quant = desc[:ind].strip()
    units = desc[ind:]
    for item in '()[]':
        units = units.replace(item, '')
    return quant, units


def _validate_metadata_from_h5dset(sig, h5_dset, compound_comp_name=None):
    if compound_comp_name is None:
        quant = usid.hdf_utils.get_attr(h5_dset, 'quantity')
        units = usid.hdf_utils.get_attr(h5_dset, 'units')
    else:
        quant, units = _split_descriptor(compound_comp_name)
    assert sig.original_metadata.quantity == quant
    assert sig.original_metadata.units == units
    assert sig.metadata.General.original_filename == h5_dset.file.filename
    assert sig.metadata.General.title == h5_dset.name.split('/')[-1]
    assert sig.original_metadata.dataset_path == h5_dset.name
    assert sig.original_metadata.original_file_type == 'USID HDF5'
    assert sig.original_metadata.pyUSID_version == usid.__version__

    h5_chan_grp = h5_dset.parent
    usid_grp_parms = dict()
    if 'Channel' in h5_chan_grp.name.split('/')[-1]:
        usid_grp_parms = sidpy.hdf.hdf_utils.get_attributes(h5_chan_grp)
        h5_meas_grp = h5_chan_grp.parent
        if 'Measurement' in h5_meas_grp.name.split('/')[-1]:
            temp = sidpy.hdf.hdf_utils.get_attributes(h5_meas_grp)
            usid_grp_parms.update(temp)
    # Remove timestamp key since there is 1s difference occasionally
    usid_grp_parms.pop('timestamp', None)
    om_dict = sig.original_metadata.parameters.as_dictionary()
    om_dict.pop('timestamp', None)
    assert om_dict == usid_grp_parms


def compare_usid_from_signal(sig, h5_path, empty_pos=False, empty_spec=False,
                             dataset_path=None, **kwargs):
    with h5py.File(h5_path, mode='r') as h5_f:
        # 1. Validate that what has been written is a USID Main dataset
        if dataset_path is None:
            _array_translator_basic_checks(h5_f)
            h5_main = usid.hdf_utils.get_all_main(h5_f)[0]
        else:
            h5_main = usid.USIDataset(h5_f[dataset_path])

        usid_data = h5_main.get_n_dim_form().squeeze()
        # 2. Validate that raw data has been written correctly:
        np.testing.assert_allclose(sig.data, usid_data)
        # 3. Validate that axes / dimensions have been translated correctly:
        if empty_pos:
            _assert_empty_dims(sig.axes_manager.navigation_axes,
                               h5_main.pos_dim_labels, h5_main.get_pos_values,
                               **kwargs)
        else:
            _compare_axes(sig.axes_manager.navigation_axes,
                          h5_main.pos_dim_descriptors, h5_main.get_pos_values,
                          **kwargs)
        # 4. Check to make sure that there is only one spectroscopic dimension
        # of size 1
        if empty_spec:
            _assert_empty_dims(sig.axes_manager.signal_axes,
                               h5_main.spec_dim_labels,
                               h5_main.get_spec_values, **kwargs)
        else:
            _compare_axes(sig.axes_manager.signal_axes,
                          h5_main.spec_dim_descriptors,
                          h5_main.get_spec_values, **kwargs)


def compare_signal_from_usid(file_path, ndata, new_sig, axes_to_spec=[],
                             sig_type=hs.signals.BaseSignal, dataset_path=None,
                             compound_comp_name=None, **kwargs):
    # 1. Validate object type
    assert isinstance(new_sig, sig_type)
    if len(axes_to_spec) > 0:
        new_sig = new_sig.as_signal2D(axes_to_spec)

    # 2. Validate that data has been read in correctly:
    np.testing.assert_allclose(new_sig.data, ndata)
    with h5py.File(file_path, mode='r') as h5_f:
        if dataset_path is None:
            h5_main = usid.hdf_utils.get_all_main(h5_f)[0]
        else:
            h5_main = usid.USIDataset(h5_f[dataset_path])
        # 3. Validate that all axes / dimensions have been translated correctly
        if len(axes_to_spec) > 0:
            _compare_axes(new_sig.axes_manager.navigation_axes,
                          h5_main.pos_dim_descriptors, h5_main.get_pos_values,
                          **kwargs)
        else:
            assert new_sig.axes_manager.navigation_dimension == 0
        # 3. Validate that all spectroscopic axes / dimensions have been
        # translated correctly:
        if h5_main.shape[1] == 1:
            _compare_axes(new_sig.axes_manager.signal_axes,
                          h5_main.pos_dim_descriptors, h5_main.get_pos_values,
                          **kwargs)
        else:
            _compare_axes(new_sig.axes_manager.signal_axes,
                          h5_main.spec_dim_descriptors,
                          h5_main.get_spec_values, **kwargs)

        # 5. Validate that metadata has been read in correctly:
        _validate_metadata_from_h5dset(new_sig, h5_main,
                                       compound_comp_name=compound_comp_name)


def gen_2pos_2spec(s2f_aux=True, mode=None):
    pos_dims = [usid.Dimension('X', 'nm', [-250, 750]),
                usid.Dimension('Y', 'um', np.linspace(0, 60, num=7))]
    spec_dims = [usid.Dimension('Frequency', 'kHz', [300, 350, 400]),
                 usid.Dimension('Bias', 'V', np.linspace(-4, 4, num=5))]
    if s2f_aux:
        pos_dims = pos_dims[::-1]
        spec_dims = spec_dims[::-1]

    ndim_shape = (7, 2, 5, 3)
    if mode is None:
        # Typcial floating point dataset
        ndata = np.random.rand(*ndim_shape)
    elif mode == 'complex':
        ndata = np.random.rand(*ndim_shape) + 1j * np.random.rand(*ndim_shape)
    elif mode == 'compound':
        struc_dtype = np.dtype({'names': ['amp', 'phas'],
                                'formats': [np.float16, np.float32]})
        ndata = np.zeros(shape=ndim_shape, dtype=struc_dtype)
        ndata['amp'] = np.random.random(size=ndim_shape)
        ndata['phas'] = np.random.random(size=ndim_shape)

    data_2d = ndata.reshape(np.prod(ndata.shape[:2]),
                              np.prod(ndata.shape[2:]))
    return pos_dims, spec_dims, ndata, data_2d


def gen_2dim(all_pos=False, s2f_aux=True):
    ndata = np.random.rand(3, 2)
    if all_pos:
        pos_dims = [usid.Dimension('X', 'nm', [-250, 750]),
                    usid.Dimension('Y', 'um', [-2, 0, 2])]
        spec_dims = [usid.Dimension('arb', 'a.u.', 1)]
        data_2d = ndata.reshape(-1, 1)
    else:
        pos_dims = [usid.Dimension('arb', 'a.u.', 1)]
        spec_dims = [usid.Dimension('Frequency', 'kHz', [0, 100]),
                     usid.Dimension('Bias', 'V', [-5, 0, 5])]
        data_2d = ndata.reshape(1, -1)
    if s2f_aux:
        pos_dims = pos_dims[::-1]
        spec_dims = spec_dims[::-1]
    return pos_dims, spec_dims, ndata, data_2d

# ################ HyperSpy Signal to h5USID ##################################


@pytest.mark.filterwarnings("ignore:This dataset does not have an N-dimensional form:UserWarning")
class TestHS2USIDallKnown:

    def test_n_pos_0_spec(self):
        sig = hs.signals.BaseSignal(np.random.randint(0, high=100,
                                                      size=(2, 3)),
                                    axes=[{'size': 2, 'name': 'X',
                                           'units': 'nm', 'scale': 0.2,
                                           'offset': 1},
                                          {'size': 3, 'name': 'Y',
                                           'units': 'um', 'scale': 0.1,
                                           'offset': 2}])
        sig = sig.transpose()

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = tmp_dir + 'usid_n_pos_0_spec_real.h5'
        sig.save(file_path)

        compare_usid_from_signal(sig, file_path, empty_pos=False,
                                 empty_spec=True)

    def test_0_pos_n_spec(self):
        sig = hs.signals.BaseSignal(np.random.randint(0, high=100,
                                                      size=(2, 3)),
                                    axes=[{'size': 2, 'name': 'Freq',
                                           'units': 'Hz', 'scale': 30,
                                           'offset': 300},
                                          {'size': 3, 'name': 'Bias',
                                           'units': 'V', 'scale': 0.25,
                                           'offset': -0.25}])

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = tmp_dir + 'usid_0_pos_n_spec_real.h5'
        sig.save(file_path)

        compare_usid_from_signal(sig, file_path, empty_pos=True,
                                 empty_spec=False)

    def test_n_pos_m_spec(self):
        sig = hs.signals.Signal2D(np.random.randint(0, high=100,
                                                    size=(2, 3, 5, 7)),
                                  axes=[{'size': 2, 'name': 'X', 'units': 'nm',
                                         'scale': 0.2, 'offset': 1},
                                        {'size': 3, 'name': 'Y', 'units': 'um',
                                         'scale': 0.1, 'offset': 2},
                                        {'size': 5, 'name': 'Freq',
                                         'units': 'Hz', 'scale': 30,
                                         'offset': 300},
                                        {'size': 7, 'name': 'Bias',
                                         'units': 'V', 'scale': 0.25,
                                         'offset': -0.25}])

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = tmp_dir + 'usid_n_pos_n_spec_real.h5'
        sig.save(file_path)

        compare_usid_from_signal(sig, file_path, empty_pos=False,
                                 empty_spec=False)

    def test_append_to_existing_file(self):
        sig = hs.signals.BaseSignal(np.random.randint(0, high=100,
                                                      size=(2, 3)),
                                    axes=[{'size': 2, 'name': 'X',
                                           'units': 'nm', 'scale': 0.2,
                                           'offset': 1},
                                          {'size': 3, 'name': 'Y',
                                           'units': 'um', 'scale': 0.1,
                                           'offset': 2}])
        sig = sig.transpose()

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = tmp_dir + 'usid_n_pos_0_spec_real.h5'
        sig.save(file_path)

        compare_usid_from_signal(sig, file_path, empty_pos=False,
                                 empty_spec=True)

        sig = hs.signals.BaseSignal(np.random.randint(0, high=100,
                                                      size=(2, 3)),
                                    axes=[{'size': 2, 'name': 'Freq',
                                           'units': 'Hz', 'scale': 30,
                                           'offset': 300},
                                          {'size': 3, 'name': 'Bias',
                                           'units': 'V', 'scale': 0.25,
                                           'offset': -0.25}])

        sig.save(file_path, overwrite=True)

        new_dataset_path = '/Measurement_001/Channel_000/Raw_Data'

        compare_usid_from_signal(sig, file_path, empty_pos=True,
                                 empty_spec=False, dataset_path=new_dataset_path)


@pytest.mark.filterwarnings("ignore:This dataset does not have an N-dimensional form:UserWarning")
class TestHS2USIDlazy:

    def test_base_nd(self):
        sig = hs.signals.Signal2D(da.random.randint(0, high=100,
                                                    size=(2, 3, 5, 7),
                                                    chunks='auto')).as_lazy()
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = tmp_dir + 'usid_n_pos_n_spec_dask.h5'
        sig.save(file_path)

        compare_usid_from_signal(sig, file_path, empty_pos=False,
                                 empty_spec=False, axes_defined=False)


@pytest.mark.filterwarnings("ignore:This dataset does not have an N-dimensional form:UserWarning")
class TestHS2USIDedgeAxes:

    def test_no_axes(self):
        sig = hs.signals.Signal2D(np.random.randint(0, high=100,
                                                    size=(2, 3, 5, 7)))

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = tmp_dir + 'usid_n_pos_n_spec_no_axes.h5'
        sig.save(file_path)

        compare_usid_from_signal(sig, file_path, empty_pos=False,
                                 empty_spec=False, axes_defined=False)

    def test_incorrect_axes(self):
        sig = hs.signals.Signal2D(np.random.randint(0, high=100,
                                                    size=(2, 3, 5, 7)),
                                  axes=[{'size': 1259, 'name': 'X',
                                         'units': 'nm', 'scale': 0.2,
                                         'offset': 1},
                                        {'size': 245, 'name': 'Y',
                                         'units': 'um', 'scale': 0.1,
                                         'offset': 2},
                                        {'size': 205, 'name': 'Freq',
                                         'units': 'Hz', 'scale': 30,
                                         'offset': 300},
                                        {'size': 1005, 'name': 'Bias',
                                         'units': 'V', 'scale': 0.25,
                                         'offset': -0.25}])

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = tmp_dir + 'usid_n_pos_n_spec_incorrect_axes.h5'
        sig.save(file_path)

        compare_usid_from_signal(sig, file_path, empty_pos=False,
                                 empty_spec=False, invalid_axes=True)

    def test_too_many_axes(self):

        sig = hs.signals.Signal2D(np.random.randint(0, high=100,
                                                    size=(2, 3, 5, 7)),
                                  axes=[{'size': 2, 'name': 'X', 'units': 'nm',
                                         'scale': 0.2, 'offset': 1},
                                        {'size': 3, 'name': 'Y', 'units': 'um',
                                         'scale': 0.1, 'offset': 2},
                                        {'size': 5, 'name': 'Freq',
                                         'units': 'Hz', 'scale': 30,
                                         'offset': 300},
                                        {'size': 7, 'name': 'Bias',
                                         'units': 'V', 'scale': 0.25,
                                         'offset': -0.25},
                                        {'size': 27, 'name': 'Does not exist',
                                         'units': 'V', 'scale': 0.25,
                                         'offset': -0.25}])

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = tmp_dir + 'usid_n_pos_n_spec_too_many_axes.h5'
        with pytest.raises(ValueError):
            sig.save(file_path)

# ################## h5USID to HyperSpy Signal(s)  ############################

@pytest.mark.filterwarnings("ignore:This dataset does not have an N-dimensional form:UserWarning")
class TestUSID2HSbase:

    def test_n_pos_0_spec(self):
        phy_quant = 'Current'
        phy_unit = 'nA'
        slow_to_fast = True
        pos_dims, spec_dims, ndata, data_2d = gen_2dim(all_pos=True,
                                                       s2f_aux=slow_to_fast)
        tran = usid.ArrayTranslator()
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = tmp_dir + 'usid_n_pos_0_spec.h5'
        _ = tran.translate(file_path, 'Blah', data_2d, phy_quant, phy_unit,
                           pos_dims, spec_dims, slow_to_fast=slow_to_fast)

        new_sig = hs.load(file_path)
        compare_signal_from_usid(file_path, ndata, new_sig,
                                 sig_type=hs.signals.BaseSignal,
                                 axes_to_spec=[])

    def test_0_pos_n_spec(self):
        phy_quant = 'Current'
        phy_unit = 'nA'
        slow_to_fast = True
        pos_dims, spec_dims, ndata, data_2d = gen_2dim(all_pos=False,
                                                       s2f_aux=slow_to_fast)
        tran = usid.ArrayTranslator()
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = tmp_dir + 'usid_0_pos_n_spec.h5'
        _ = tran.translate(file_path, 'Blah', data_2d, phy_quant, phy_unit,
                           pos_dims, spec_dims, slow_to_fast=slow_to_fast)

        new_sig = hs.load(file_path)
        compare_signal_from_usid(file_path, ndata, new_sig,
                                 sig_type=hs.signals.BaseSignal,
                                 axes_to_spec=[])

    @pytest.mark.parametrize('lazy', [True, False])
    def test_base_n_pos_m_spec(self, lazy, slow_to_fast=True):
        phy_quant = 'Current'
        phy_unit = 'nA'
        ret_vals = gen_2pos_2spec(s2f_aux=slow_to_fast)
        pos_dims, spec_dims, ndata, data_2d = ret_vals
        tran = usid.ArrayTranslator()
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = tmp_dir + 'usid_n_pos_n_spec.h5'
        _ = tran.translate(file_path, 'Blah', data_2d, phy_quant, phy_unit,
                           pos_dims, spec_dims, slow_to_fast=slow_to_fast)

        new_sig = hs.load(file_path, lazy=lazy)
        compare_signal_from_usid(file_path, ndata, new_sig,
                                 sig_type=hs.signals.BaseSignal,
                                 axes_to_spec=['Frequency', 'Bias'])


@pytest.mark.filterwarnings("ignore:This dataset does not have an N-dimensional form:UserWarning")
class TestUSID2HSdtype:

    def test_complex(self):
        phy_quant = 'Current'
        phy_unit = 'nA'
        slow_to_fast = True
        ret_vals = gen_2pos_2spec(s2f_aux=slow_to_fast, mode='complex')
        pos_dims, spec_dims, ndata, data_2d = ret_vals
        tran = usid.ArrayTranslator()
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = tmp_dir + 'usid_n_pos_n_spec_complex.h5'
        _ = tran.translate(file_path, 'Blah', data_2d, phy_quant, phy_unit,
                           pos_dims, spec_dims, slow_to_fast=slow_to_fast)

        new_sig = hs.load(file_path)
        compare_signal_from_usid(file_path, ndata, new_sig,
                                 sig_type=hs.signals.ComplexSignal,
                                 axes_to_spec=['Frequency', 'Bias'])

    def test_compound(self):
        phy_quant = 'Current'
        phy_unit = 'nA'
        slow_to_fast = True
        ret_vals = gen_2pos_2spec(s2f_aux=slow_to_fast, mode='compound')
        pos_dims, spec_dims, ndata, data_2d = ret_vals
        tran = usid.ArrayTranslator()
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = tmp_dir + 'usid_n_pos_n_spec_compound.h5'
        _ = tran.translate(file_path, 'Blah', data_2d, phy_quant, phy_unit,
                           pos_dims, spec_dims, slow_to_fast=slow_to_fast)

        objects = hs.load(file_path)
        assert isinstance(objects, list)
        assert len(objects) == 2

        # 1. Validate object type
        for new_sig, comp_name in zip(objects, ['amp', 'phas']):
            compare_signal_from_usid(file_path, ndata[comp_name], new_sig,
                                     sig_type=hs.signals.BaseSignal,
                                     axes_to_spec=['Frequency', 'Bias'],
                                     compound_comp_name=comp_name)

    def test_non_uniform_dimension(self):
        pos_dims = [usid.Dimension('Y', 'um', np.linspace(0, 60, num=5)),
                    usid.Dimension('X', 'nm', [-250, 750])]
        spec_dims = [usid.Dimension('Bias', 'V',
                                    np.sin(np.linspace(0, 2 * np.pi, num=7))),
                     usid.Dimension('Frequency', 'kHz', [300, 350, 400])]
        ndata = np.random.rand(5, 2, 7, 3)
        phy_quant = 'Current'
        phy_unit = 'nA'
        data_2d = ndata.reshape(np.prod(ndata.shape[:2]),
                                np.prod(ndata.shape[2:]))
        tran = usid.ArrayTranslator()
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = tmp_dir + 'usid_n_pos_n_spec_non_lin_dim.h5'
        _ = tran.translate(file_path, 'Blah', data_2d, phy_quant, phy_unit,
                           pos_dims, spec_dims,
                           slow_to_fast=True)

        with pytest.raises(ValueError):
            _ = hs.load(file_path, ignore_non_uniform_dims=False)

        with pytest.warns(UserWarning) as _:
            new_sig = hs.load(file_path)
        compare_signal_from_usid(file_path, ndata, new_sig,
                                 axes_to_spec=['Frequency', 'Bias'],
                                 invalid_axes=True)


@pytest.mark.filterwarnings("ignore:This dataset does not have an N-dimensional form:UserWarning")
class TestUSID2HSmultiDsets:

    def test_pick_specific(self):
        slow_to_fast = True
        ret_vals = gen_2pos_2spec(s2f_aux=slow_to_fast)
        pos_dims, spec_dims, ndata, data_2d = ret_vals
        phy_quant = 'Current'
        phy_unit = 'nA'
        tran = usid.ArrayTranslator()
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = tmp_dir + 'usid_n_pos_n_spec.h5'
        _ = tran.translate(file_path, 'Blah', data_2d, phy_quant, phy_unit,
                           pos_dims, spec_dims, slow_to_fast=slow_to_fast)

        ret_vals = gen_2dim(all_pos=True, s2f_aux=slow_to_fast)
        pos_dims, spec_dims, ndata_2, data_2d_2 = ret_vals
        phy_quant = 'Current'
        phy_unit = 'nA'

        with h5py.File(file_path, mode='r+') as h5_f:
            h5_meas_grp = h5_f.create_group('Measurement_001')
            h5_chan_grp = h5_meas_grp.create_group('Channel_000')
            _ = usid.hdf_utils.write_main_dataset(h5_chan_grp, data_2d_2,
                                                  'Raw_Data', phy_quant,
                                                  phy_unit, pos_dims,
                                                  spec_dims,
                                                  slow_to_fast=slow_to_fast)

        dataset_path = '/Measurement_001/Channel_000/Raw_Data'
        new_sig = hs.load(file_path, dataset_path=dataset_path)
        compare_signal_from_usid(file_path, ndata_2, new_sig,
                                 dataset_path=dataset_path)

    def test_read_all_by_default(self):
        slow_to_fast = True
        pos_dims, spec_dims, ndata, data_2d = gen_2dim(all_pos=False,
                                                       s2f_aux=slow_to_fast)
        phy_quant = 'Current'
        phy_unit = 'nA'
        tran = usid.ArrayTranslator()
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = tmp_dir + 'usid_0_pos_n_spec.h5'
        _ = tran.translate(file_path, 'Blah', data_2d, phy_quant, phy_unit,
                           pos_dims, spec_dims, slow_to_fast=slow_to_fast)

        ret_vals = gen_2dim(all_pos=True, s2f_aux=slow_to_fast)
        pos_dims, spec_dims, ndata_2, data_2d_2 = ret_vals
        phy_quant = 'Current'
        phy_unit = 'nA'

        with h5py.File(file_path, mode='r+') as h5_f:
            h5_meas_grp = h5_f.create_group('Measurement_001')
            _ = usid.hdf_utils.write_main_dataset(h5_meas_grp, data_2d_2,
                                                  'Spat_Map', phy_quant,
                                                  phy_unit, pos_dims,
                                                  spec_dims,
                                                  slow_to_fast=slow_to_fast)

        objects = hs.load(file_path)
        assert isinstance(objects, list)
        assert len(objects) == 2

        dset_names = ['/Measurement_000/Channel_000/Raw_Data',
                      'Measurement_001/Spat_Map']

        # 1. Validate object type
        for new_sig, ndim_data, dataset_path in zip(objects,
                                                    [ndata, ndata_2],
                                                    dset_names):
            compare_signal_from_usid(file_path, ndim_data, new_sig,
                                     dataset_path=dataset_path)
