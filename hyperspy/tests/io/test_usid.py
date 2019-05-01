import tempfile
import pytest
import numpy as np
import h5py
from hyperspy import api as hs
try:
    import pyUSID as usid
    pyusid_installed = True
except ModuleNotFoundError:
    pyusid_installed = False

pytestmark = pytest.mark.skipif(not pyusid_installed, reason="pyUSID not installed")


# ################################ HELPER FUNCTIONS ####################################################################

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
    for dset_name in ['Raw_Data', 'Position_Indices', 'Position_Values', 'Spectroscopic_Indices',
                      'Spectroscopic_Values']:
        assert dset_name in h5_chan_grp.keys()
        h5_dset = h5_chan_grp[dset_name]
        assert isinstance(h5_dset, h5py.Dataset)

    assert len(usid.hdf_utils.get_all_main(h5_f)) == 1

    assert usid.hdf_utils.check_if_main(h5_chan_grp['Raw_Data'])


def __compare_axes(hs_axes, dim_descriptors, usid_val_func, axes_defined=True, invalid_axes=False):
    usid_descriptors = [__split_descriptor(_) for _ in dim_descriptors]
    for dim_ind, axis in enumerate(hs_axes):
        # Remember that the signal axes are actually reversed!
        real_dim_ind = len(hs_axes) - dim_ind - 1
        if axes_defined:
            assert axis.name == usid_descriptors[real_dim_ind][0]
            assert axis.units == usid_descriptors[real_dim_ind][1]

        if not invalid_axes:
            axis_vals = np.arange(axis.offset, axis.offset + axis.size * axis.scale, axis.scale)
            usid_vals = usid_val_func(usid_descriptors[real_dim_ind][0])
            assert np.allclose(axis_vals, usid_vals)


def __assert_empty_dims(hs_axes, usid_labels, usid_val_func):
    assert len(hs_axes) == 0
    assert len(usid_labels) == 1
    dim_vals = usid_val_func(usid_labels[0])
    assert len(dim_vals) == 1


def __split_descriptor(desc):
    desc = desc.strip()
    ind = desc.rfind('(')
    if ind < 0:
        ind = desc.rfind('[')
        if ind < 0:
            return desc, ''

    quant = desc[:ind].strip()
    units = desc[ind:]
    units = units.replace('(', '')
    units = units.replace(')', '')
    units = units.replace('[', '')
    units = units.replace(']', '')
    return quant, units


def __validate_metadata_from_h5dset(sig, h5_dset, compound_comp_name=None):
    if compound_comp_name is None:
        assert sig.metadata.Signal.quantity == usid.hdf_utils.get_attr(h5_dset, 'quantity')
        assert sig.metadata.Signal.units == usid.hdf_utils.get_attr(h5_dset, 'units')
    else:
        quant, units = __split_descriptor(compound_comp_name)
        assert sig.metadata.Signal.quantity == quant
        assert sig.metadata.Signal.units == units
    assert sig.metadata.General.original_filename == h5_dset.file.filename
    assert sig.metadata.General.dataset_path == h5_dset.name
    assert sig.metadata.General.original_file_type == 'USID HDF5'
    assert sig.metadata.General.pyUSID_version == usid.__version__


def compare_usid_from_signal(sig, h5_path, empty_pos=False, empty_spec=False, **kwargs):
    with h5py.File(h5_path, mode='r') as h5_f:
        # 1. Validate that what has been written is a USID Main dataset
        _array_translator_basic_checks(h5_f)
        h5_main = usid.hdf_utils.get_all_main(h5_f)[0]
        usid_data = h5_main.get_n_dim_form().squeeze()
        # 2. Validate that raw data has been written correctly:
        assert np.allclose(sig.data, usid_data)
        # 3. Validate that axes / dimensions have been translated correctly:
        if empty_pos:
            __assert_empty_dims(sig.axes_manager.navigation_axes, h5_main.pos_dim_labels, h5_main.get_pos_values,
                                **kwargs)
        else:
            __compare_axes(sig.axes_manager.navigation_axes, h5_main.pos_dim_descriptors, h5_main.get_pos_values,
                           **kwargs)
        # 4. Check to make sure that there is only one spectroscopic dimension of size 1
        if empty_spec:
            __assert_empty_dims(sig.axes_manager.signal_axes, h5_main.spec_dim_labels, h5_main.get_spec_values,
                                **kwargs)
        else:
            __compare_axes(sig.axes_manager.signal_axes, h5_main.spec_dim_descriptors, h5_main.get_spec_values,
                           **kwargs)


def compare_signal_from_usid(file_path, ndata, new_sig, sig_type=hs.signals.BaseSignal,
                             axes_to_spec=[], dset_path=None, compound_comp_name=None, **kwargs):
    # 1. Validate object type
    assert isinstance(new_sig, sig_type)
    if len(axes_to_spec) > 0:
        new_sig = new_sig.as_signal2D(axes_to_spec)

    # 2. Validate that data has been read in correctly:
    assert np.allclose(new_sig.data, ndata)
    with h5py.File(file_path, mode='r') as h5_f:
        if dset_path is None:
            h5_main = usid.hdf_utils.get_all_main(h5_f)[0]
        else:
            h5_main = usid.USIDataset(h5_f[dset_path])
        # 3. Validate that all axes / dimensions have been translated correctly:
        if len(axes_to_spec) > 0:
            __compare_axes(new_sig.axes_manager.navigation_axes, h5_main.pos_dim_descriptors, h5_main.get_pos_values,
                           **kwargs)
        else:
            assert new_sig.axes_manager.navigation_dimension == 0
        # 3. Validate that all spectroscopic axes / dimensions have been translated correctly:
        if h5_main.shape[1] == 1:
            __compare_axes(new_sig.axes_manager.signal_axes, h5_main.pos_dim_descriptors, h5_main.get_pos_values,
                           **kwargs)
        else:
            __compare_axes(new_sig.axes_manager.signal_axes, h5_main.spec_dim_descriptors, h5_main.get_spec_values,
                           **kwargs)
        # 5. Validate that metadata has been read in correctly:
        __validate_metadata_from_h5dset(new_sig, h5_main, compound_comp_name=compound_comp_name)


# ################################ HyperSpy Signal to h5USID ###########################################################

class TestHS2USIDallKnown:

    def test_n_pos_0_spec(self):
        sig = hs.signals.BaseSignal(np.random.randint(0, high=100, size=(2, 3)),
                                  axes=[{'size': 2, 'name': 'X', 'units': 'nm', 'scale': 0.2, 'offset': 1},
                                        {'size': 3, 'name': 'Y', 'units': 'um', 'scale': 0.1, 'offset': 2}])
        sig = sig.transpose()

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = tmp_dir + 'usid_n_pos_0_spec_real.h5'
        sig.save(file_path)

        compare_usid_from_signal(sig, file_path, empty_pos=False, empty_spec=True)

    def test_0_pos_n_spec(self):
        sig = hs.signals.BaseSignal(np.random.randint(0, high=100, size=(2, 3)),
                                    axes=[{'size': 2, 'name': 'Freq', 'units': 'Hz', 'scale': 30, 'offset': 300},
                                          {'size': 3, 'name': 'Bias', 'units': 'V', 'scale': 0.25, 'offset': -0.25}])

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = tmp_dir + 'usid_0_pos_n_spec_real.h5'
        sig.save(file_path)

        compare_usid_from_signal(sig, file_path, empty_pos=True, empty_spec=False)

    def test_n_pos_m_spec(self):
        sig = hs.signals.Signal2D(np.random.randint(0, high=100, size=(2, 3, 5, 7)),
                                 axes=[{'size': 2, 'name': 'X', 'units': 'nm', 'scale': 0.2, 'offset': 1},
                                       {'size': 3, 'name': 'Y', 'units': 'um', 'scale': 0.1, 'offset': 2},
                                       {'size': 5, 'name': 'Freq', 'units': 'Hz', 'scale': 30, 'offset': 300},
                                       {'size': 7, 'name': 'Bias', 'units': 'V', 'scale': 0.25, 'offset': -0.25}])

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = tmp_dir + 'usid_n_pos_n_spec_real.h5'
        sig.save(file_path)

        compare_usid_from_signal(sig, file_path, empty_pos=False, empty_spec=False)


class TestHS2USIDedgeAxes:

    def test_no_axes(self):
        sig = hs.signals.Signal2D(np.random.randint(0, high=100, size=(2, 3, 5, 7)))

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = tmp_dir + 'usid_n_pos_n_spec_no_axes.h5'
        sig.save(file_path)

        compare_usid_from_signal(sig, file_path, empty_pos=False, empty_spec=False, axes_defined=False)

    def test_incorrect_axes(self):
        sig = hs.signals.Signal2D(np.random.randint(0, high=100, size=(2, 3, 5, 7)),
                                 axes=[{'size': 1259, 'name': 'X', 'units': 'nm', 'scale': 0.2, 'offset': 1},
                                       {'size': 245, 'name': 'Y', 'units': 'um', 'scale': 0.1, 'offset': 2},
                                       {'size': 205, 'name': 'Freq', 'units': 'Hz', 'scale': 30, 'offset': 300},
                                       {'size': 1005, 'name': 'Bias', 'units': 'V', 'scale': 0.25, 'offset': -0.25}])

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = tmp_dir + 'usid_n_pos_n_spec_incorrect_axes.h5'
        sig.save(file_path)

        compare_usid_from_signal(sig, file_path, empty_pos=False, empty_spec=False, invalid_axes=True)

    def test_too_many_axes(self):

        sig = hs.signals.Signal2D(np.random.randint(0, high=100, size=(2, 3, 5, 7)),
                                  axes=[{'size': 2, 'name': 'X', 'units': 'nm', 'scale': 0.2, 'offset': 1},
                                        {'size': 3, 'name': 'Y', 'units': 'um', 'scale': 0.1, 'offset': 2},
                                        {'size': 5, 'name': 'Freq', 'units': 'Hz', 'scale': 30, 'offset': 300},
                                        {'size': 7, 'name': 'Bias', 'units': 'V', 'scale': 0.25, 'offset': -0.25},
                                        {'size': 27, 'name': 'Does not exist', 'units': 'V', 'scale': 0.25,
                                         'offset': -0.25}])

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = tmp_dir + 'usid_n_pos_n_spec_too_many_axes.h5'
        with pytest.raises(ValueError):
            sig.save(file_path)

# ################################ h5USID to HyperSpy Signal(s)  #######################################################


class TestUSID2HSbase:

    def test_n_pos_0_spec(self):
        pos_dims = [usid.Dimension('X', 'nm', [-250, 750]), usid.Dimension('Y', 'um', [-2, 0, 2])]
        spec_dims = [usid.Dimension('arb', 'a.u.', 1)]
        ndata = np.random.rand(2, 3)
        phy_quant = 'Current'
        phy_unit = 'nA'
        # Rearrange to slow to fast which is what python likes
        data_2d = ndata.transpose([1, 0])
        data_2d = data_2d.reshape(-1, 1)
        tran = usid.NumpyTranslator()
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = tmp_dir + 'usid_n_pos_0_spec.h5'
        _ = tran.translate(file_path, 'Blah', data_2d, phy_quant, phy_unit, pos_dims, spec_dims)

        new_sig = hs.load(file_path)
        compare_signal_from_usid(file_path, ndata, new_sig, sig_type=hs.signals.BaseSignal, axes_to_spec=[])

    def test_0_pos_n_spec(self):
        pos_dims = [usid.Dimension('arb', 'a.u.', 1)]
        spec_dims = [usid.Dimension('Frequency', 'kHz', [0, 100]), usid.Dimension('Bias', 'V', [-5, 0, 5])]
        ndata = np.random.rand(2, 3)
        phy_quant = 'Current'
        phy_unit = 'nA'
        # Rearrange to slow to fast which is what python likes
        data_2d = ndata.transpose([1, 0])
        data_2d = data_2d.reshape(1, -1)
        tran = usid.NumpyTranslator()
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = tmp_dir + 'usid_0_pos_n_spec.h5'
        _ = tran.translate(file_path, 'Blah', data_2d, phy_quant, phy_unit, pos_dims, spec_dims)
        
        new_sig = hs.load(file_path)
        compare_signal_from_usid(file_path, ndata, new_sig, sig_type=hs.signals.BaseSignal, axes_to_spec=[])
        
    def test_n_pos_m_spec(self):
        pos_dims = [usid.Dimension('X', 'nm', [-250, 750]), usid.Dimension('Y', 'um', np.linspace(0, 60, num=7))]
        spec_dims = [usid.Dimension('Frequency', 'kHz', [300, 350, 400]),
                     usid.Dimension('Bias', 'V', np.linspace(-4, 4, num=5))]
        ndata = np.random.rand(2, 7, 3, 5)
        phy_quant = 'Current'
        phy_unit = 'nA'
        # Rearrange to slow to fast which is what python likes
        data_2d = ndata.transpose([1, 0, 3, 2])
        data_2d = data_2d.reshape(np.prod(data_2d.shape[:2]), np.prod(data_2d.shape[2:]))
        tran = usid.NumpyTranslator()
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = tmp_dir + 'usid_n_pos_n_spec.h5'
        _ = tran.translate(file_path, 'Blah', data_2d, phy_quant, phy_unit, pos_dims, spec_dims)
        
        new_sig = hs.load(file_path)
        compare_signal_from_usid(file_path, ndata, new_sig,
                                 sig_type=hs.signals.BaseSignal, axes_to_spec=['Bias', 'Frequency'])


class TestUSID2HSdtype:
    
    def test_complex(self):
        pos_dims = [usid.Dimension('X', 'nm', [-250, 750]), usid.Dimension('Y', 'um', np.linspace(0, 60, num=7))]
        spec_dims = [usid.Dimension('Frequency', 'kHz', [300, 350, 400]),
                     usid.Dimension('Bias', 'V', np.linspace(-4, 4, num=5))]
        ndata = np.random.rand(2, 7, 3, 5) + 1j * np.random.rand(2, 7, 3, 5)
        phy_quant = 'Current'
        phy_unit = 'nA'
        # Rearrange to slow to fast which is what python likes
        data_2d = ndata.transpose([1, 0, 3, 2])
        data_2d = data_2d.reshape(np.prod(data_2d.shape[:2]), np.prod(data_2d.shape[2:]))
        tran = usid.NumpyTranslator()
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = tmp_dir + 'usid_n_pos_n_spec_complex.h5'
        _ = tran.translate(file_path, 'Blah', data_2d, phy_quant, phy_unit, pos_dims, spec_dims)
        
        new_sig = hs.load(file_path)
        compare_signal_from_usid(file_path, ndata, new_sig,
                                 sig_type=hs.signals.ComplexSignal, axes_to_spec=['Bias', 'Frequency'])
        
    def test_compound(self):  
        pos_dims = [usid.Dimension('X', 'nm', [-250, 750]), usid.Dimension('Y', 'um', np.linspace(0, 60, num=7))]
        spec_dims = [usid.Dimension('Frequency', 'kHz', [300, 350, 400]),
                     usid.Dimension('Bias', 'V', np.linspace(-4, 4, num=5))]
        
        struc_dtype = np.dtype({'names': ['amp', 'phas'],
                                'formats': [np.float32, np.float32]})
        
        ndim_shape = (2, 7, 3, 5)
        
        ndata = np.zeros(shape=ndim_shape, dtype=struc_dtype)
        ndata['amp'] = amp_vals = np.random.random(size=ndim_shape)
        ndata['phas'] = phase_vals = np.random.random(size=ndim_shape)
        
        phy_quant = 'Current'
        phy_unit = 'nA'
        # Rearrange to slow to fast which is what python likes
        data_2d = ndata.transpose([1, 0, 3, 2])
        data_2d = data_2d.reshape(np.prod(data_2d.shape[:2]), np.prod(data_2d.shape[2:]))
        tran = usid.NumpyTranslator()
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = tmp_dir + 'usid_n_pos_n_spec_complex.h5'
        _ = tran.translate(file_path, 'Blah', data_2d, phy_quant, phy_unit, pos_dims, spec_dims)
        
        objects = hs.load(file_path)
        assert isinstance(objects, list)
        assert len(objects) == 2

        # 1. Validate object type
        for new_sig, ndim_data, comp_name in zip(objects, [amp_vals, phase_vals], ['amp', 'phas']):
            compare_signal_from_usid(file_path, ndim_data, new_sig,
                                     sig_type=hs.signals.BaseSignal,
                                     axes_to_spec=['Bias', 'Frequency'],
                                     compound_comp_name=comp_name)
            
    def test_non_linear_dimension(self):
        pos_dims = [usid.Dimension('X', 'nm', [-250, 750]), usid.Dimension('Y', 'um', np.linspace(0, 60, num=5))]
        spec_dims = [usid.Dimension('Frequency', 'kHz', [300, 350, 400]),
                     usid.Dimension('Bias', 'V', np.sin(np.linspace(0, 2 * np.pi, num=7)))]
        ndata = np.random.rand(2, 5, 3, 7)
        phy_quant = 'Current'
        phy_unit = 'nA'
        # Rearrange to slow to fast which is what python likes
        data_2d = ndata.transpose([1, 0, 3, 2])
        data_2d = data_2d.reshape(np.prod(data_2d.shape[:2]), np.prod(data_2d.shape[2:]))
        tran = usid.NumpyTranslator()
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = tmp_dir + 'usid_n_pos_n_spec_non_lin_dim.h5'
        _ = tran.translate(file_path, 'Blah', data_2d, phy_quant, phy_unit, pos_dims, spec_dims)

        with pytest.raises(ValueError):
            _ = hs.load(file_path)

        with pytest.warns(UserWarning) as _:
            new_sig = hs.load(file_path, ignore_non_linear_dims=True)
        compare_signal_from_usid(file_path, ndata, new_sig, axes_to_spec=['Bias', 'Frequency'], invalid_axes=True)


class TestUSID2HSmultiDsets:

    def test_auto_pick_first_if_unspecified(self):
        pos_dims = [usid.Dimension('X', 'nm', [-250, 750]), usid.Dimension('Y', 'um', np.linspace(0, 60, num=7))]
        spec_dims = [usid.Dimension('Frequency', 'kHz', [300, 350, 400]),
                     usid.Dimension('Bias', 'V', np.linspace(-4, 4, num=5))]
        ndata = np.random.rand(2, 7, 3, 5)
        phy_quant = 'Current'
        phy_unit = 'nA'
        # Rearrange to slow to fast which is what python likes
        data_2d = ndata.transpose([1, 0, 3, 2])
        data_2d = data_2d.reshape(np.prod(data_2d.shape[:2]), np.prod(data_2d.shape[2:]))
        tran = usid.NumpyTranslator()
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = tmp_dir + 'usid_n_pos_n_spec.h5'
        _ = tran.translate(file_path, 'Blah', data_2d, phy_quant, phy_unit, pos_dims, spec_dims)
        
        pos_dims = [usid.Dimension('X', 'nm', [-250, 750]), usid.Dimension('Y', 'um', [-2, 0, 2])]
        spec_dims = [usid.Dimension('arb', 'a.u.', 1)]
        ndata_2 = np.random.rand(2, 3)
        phy_quant = 'Current'
        phy_unit = 'nA'
        # Rearrange to slow to fast which is what python likes
        data_2d_2 = ndata_2.transpose([1, 0])
        data_2d_2 = data_2d_2.reshape(-1, 1)
        
        with h5py.File(file_path, mode='r+') as h5_f:
            h5_meas_grp = h5_f.create_group('Measurement_001')
            _ = usid.hdf_utils.write_main_dataset(h5_meas_grp, data_2d_2, 'Raw_Data',
                                                  phy_quant, phy_unit, pos_dims, spec_dims)
        
        with pytest.warns(UserWarning) as _:
            new_sig = hs.load(file_path)
        compare_signal_from_usid(file_path, ndata, new_sig, axes_to_spec=['Bias', 'Frequency'])

    def test_pick_specific(self):
        pos_dims = [usid.Dimension('X', 'nm', [-250, 750]), usid.Dimension('Y', 'um', np.linspace(0, 60, num=7))]
        spec_dims = [usid.Dimension('Frequency', 'kHz', [300, 350, 400]),
                     usid.Dimension('Bias', 'V', np.linspace(-4, 4, num=5))]
        ndata = np.random.rand(2, 7, 3, 5)
        phy_quant = 'Current'
        phy_unit = 'nA'
        # Rearrange to slow to fast which is what python likes
        data_2d = ndata.transpose([1, 0, 3, 2])
        data_2d = data_2d.reshape(np.prod(data_2d.shape[:2]), np.prod(data_2d.shape[2:]))
        tran = usid.NumpyTranslator()
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = tmp_dir + 'usid_n_pos_n_spec.h5'
        _ = tran.translate(file_path, 'Blah', data_2d, phy_quant, phy_unit, pos_dims, spec_dims)
        
        pos_dims = [usid.Dimension('X', 'nm', [-250, 750]), usid.Dimension('Y', 'um', [-2, 0, 2])]
        spec_dims = [usid.Dimension('arb', 'a.u.', 1)]
        ndata_2 = np.random.rand(2, 3)
        phy_quant = 'Current'
        phy_unit = 'nA'
        # Rearrange to slow to fast which is what python likes
        data_2d_2 = ndata_2.transpose([1, 0])
        data_2d_2 = data_2d_2.reshape(-1, 1)
        
        with h5py.File(file_path, mode='r+') as h5_f:
            h5_meas_grp = h5_f.create_group('Measurement_001')
            _ = usid.hdf_utils.write_main_dataset(h5_meas_grp, data_2d_2, 'Raw_Data',
                                                  phy_quant, phy_unit, pos_dims, spec_dims)
        
        dset_path = '/Measurement_001/Raw_Data'
        new_sig = hs.load(file_path, path_to_main_dataset=dset_path)
        compare_signal_from_usid(file_path, ndata_2, new_sig, dset_path=dset_path)
        
    def test_read_all(self):
        pos_dims = [usid.Dimension('arb', 'a.u.', 1)]
        spec_dims = [usid.Dimension('Frequency', 'kHz', [0, 100]), usid.Dimension('Bias', 'V', [-5, 0, 5])]
        ndata = np.random.rand(2, 3)
        phy_quant = 'Current'
        phy_unit = 'nA'
        # Rearrange to slow to fast which is what python likes
        data_2d = ndata.transpose([1, 0])
        data_2d = data_2d.reshape(1, -1)
        tran = usid.NumpyTranslator()
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = tmp_dir + 'usid_0_pos_n_spec.h5'
        _ = tran.translate(file_path, 'Blah', data_2d, phy_quant, phy_unit, pos_dims, spec_dims)
        
        pos_dims = [usid.Dimension('X', 'nm', [-250, 750]), usid.Dimension('Y', 'um', [-2, 0, 2])]
        spec_dims = [usid.Dimension('arb', 'a.u.', 1)]
        ndata_2 = np.random.rand(2, 3)
        phy_quant = 'Current'
        phy_unit = 'nA'
        # Rearrange to slow to fast which is what python likes
        data_2d_2 = ndata_2.transpose([1, 0])
        data_2d_2 = data_2d_2.reshape(-1, 1)
        
        with h5py.File(file_path, mode='r+') as h5_f:
            h5_meas_grp = h5_f.create_group('Measurement_001')
            _ = usid.hdf_utils.write_main_dataset(h5_meas_grp, data_2d_2, 'Raw_Data',
                                                  phy_quant, phy_unit, pos_dims, spec_dims)
        
        objects = hs.load(file_path, path_to_main_dataset=None)
        assert isinstance(objects, list)
        assert len(objects) == 2

        # 1. Validate object type
        for new_sig, ndim_data, dset_path in zip(objects, [ndata, ndata_2],
                                                 ['/Measurement_000/Channel_000/Raw_Data',
                                                  'Measurement_001/Raw_Data']):
            compare_signal_from_usid(file_path, ndim_data, new_sig,
                                     dset_path=dset_path)
