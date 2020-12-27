import os
import logging
from warnings import warn
from functools import partial
from collections.abc import MutableMapping
import h5py
import numpy as np
import pyUSID as usid
import sidpy

_logger = logging.getLogger(__name__)


# Plugin characteristics
# ----------------------
format_name = 'USID'
description = \
    'Data structured according to the Universal Spectroscopic and Imaging ' \
    'Data (USID) model written into Hierarchical Data Format (HDF5) files'
full_support = False
# Recognised file extension
file_extensions = ['h5', 'hdf5']
default_extension = 0
# Reading capabilities
reads_images = True
reads_spectrum = True
reads_spectrum_image = True
# Writing capabilities
writes_images = True
writes_spectrum = True
writes_spectrum_image = True
# Writing capabilities
writes = True
version = usid.__version__

# ######### UTILITIES THAT SIMPLIFY READING FROM H5USID FILES #################


def _get_dim_dict(labels, units, val_func, ignore_non_linear_dims=True):
    """
    Gets a list of dictionaries that correspond to axes for HyperSpy Signal
    objects

    Parameters
    ----------
    labels : list
        List of strings denoting the names of the dimension
    units : list
        List of strings denoting the units for the dimensions
    val_func : callable
        Function that will return the values over which a dimension was varied
    ignore_non_linear_dims : bool, Optional. Default = True
        If set to True, a warning will be raised instead of a ValueError when a
        dimension is encountered which was non-linearly.

    Returns
    -------
    dict
        Dictionary of dictionaries that correspond to axes for HyperSpy Signal
        objects

    Notes
    -----
    For a future release of HyperSpy:
    If a dimension was varied non-linearly, one would need to set the
    appropriate quantity in the quantity equal to dim_vals. At that point,
    the typical offset and scale parameters would be (hopefully) ignored.
    """
    dim_dict = dict()
    for dim_name, units in zip(labels, units):
        # dim_vals below contains the full 1D tensor that shows how a dimension
        # was varied. If the parameter was varied linearly, the offset, size,
        # and scale can be extracted easily.
        dim_vals = val_func(dim_name)
        if len(dim_vals) == 1:
            # Empty dimension!
            continue
        else:
            try:
                step_size = sidpy.base.num_utils.get_slope(dim_vals)
            except ValueError:
                # Non-linear dimension! - see notes above
                if ignore_non_linear_dims:
                    warn('Ignoring non-linearity of dimension: '
                         '{}'.format(dim_name))
                    step_size = 1
                    dim_vals[0] = 0
                else:
                    raise ValueError('Cannot load provided dataset. '
                                     'Parameter: {} was varied '
                                     'non-linearly. Supply keyword '
                                     'argument "ignore_non_linear_dims='
                                     'True" to ignore this '
                                     'error'.format(dim_name))

        dim_dict[dim_name] = {'size': len(dim_vals),
                              'name': dim_name,
                              'units': units,
                              'scale': step_size,
                              'offset': dim_vals[0]}
    return dim_dict


def _assemble_dim_list(dim_dict, dim_names):
    """
    Assembles a list of dictionary objects (axes) in the same order as
    specified in dim_names

    Parameters
    ----------
    dim_dict : dict
        Dictionary of dictionaries that correspond to axes for HyperSpy Signal
        objects
    dim_names : list
        List of strings denoting the names of the dimension

    Returns
    -------
    list
        List of dictionaries that correspond to axes for HyperSpy Signal
        objects
    """
    dim_list = []
    for dim_name in dim_names:
        try:
            dim_list.append(dim_dict[dim_name])
        except KeyError:
            pass
    return dim_list


def _split_descriptor(desc):
    """
    Splits a string such as "Quantity [units]" or "Quantity (units)" into the
    quantity and unit strings

    Parameters
    ----------
    desc : str
        Descriptor of a dimension or the main dataset itself

    Returns
    -------
    quant : str
        Name of the physical quantity
    units : str
        Units corresponding to the physical quantity
    """
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


def _convert_to_signal_dict(ndim_form, quantity, units, dim_dict_list,
                            h5_path, h5_dset_path, name, sig_type='',
                            group_attrs={}):
    """
    Packages required components that make up a Signal object

    Parameters
    ----------
    ndim_form : numpy.ndarray
        N-dimensional form of the main dataset
    quantity : str
        Physical quantity of the measurement
    units : str
        Corresponding units
    dim_dict_list : list
        List of dictionaries that instruct the axes corresponding to the main
        dataset
    h5_path : str
        Absolute path of the original USID HDF5 file
    h5_dset_path : str
        Absolute path of the USIDataset within the HDF5 file
    name : str
        Name of the HDF5 dataset
    sig_type : str, Optional
        Type of measurement
    group_attrs : dict, Optional. Default = {}
        Any attributes at the channel and group levels

    Returns
    -------

    """

    sig = {'data': ndim_form,
           'axes': dim_dict_list,
           'metadata': {
               'Signal': {'signal_type': sig_type},
               'General': {'original_filename': h5_path,
                           'title': name}
                        },
           'original_metadata': {'quantity': quantity,
                                 'units': units,
                                 'dataset_path': h5_dset_path,
                                 'original_file_type': 'USID HDF5',
                                 'pyUSID_version': usid.__version__,
                                 'parameters': group_attrs
                                 },
           }
    return sig


def _usidataset_to_signal(h5_main, ignore_non_linear_dims=True, lazy=True,
                          *kwds):
    """
    Converts a single specified USIDataset object to one or more Signal objects

    Parameters
    ----------
    h5_main : pyUSID.USIDataset object
        USID Main dataset
    ignore_non_linear_dims : bool, Optional
        If True, parameters that were varied non-linearly in the desired
        dataset will result in Exceptions.
        Else, all such non-linearly varied parameters will be treated as
        linearly varied parameters and
        a Signal object will be generated.
    lazy : bool, Optional
        If set to True, data will be read as a Dask array.
        Else, data will be read in as a numpy array

    Returns
    -------
    list of hyperspy.signals.BaseSignal objects
        USIDatasets with compound datatypes are broken down to multiple Signal
        objects.
    """
    h5_main = usid.USIDataset(h5_main)
    # TODO: Cannot handle data without N-dimensional form yet
    # First get dictionary of axes that HyperSpy likes to see. Ignore singular
    # dimensions
    pos_dict = _get_dim_dict(h5_main.pos_dim_labels,
                             usid.hdf_utils.get_attr(h5_main.h5_pos_inds,
                                                     'units'),
                             h5_main.get_pos_values,
                             ignore_non_linear_dims=ignore_non_linear_dims)
    spec_dict = _get_dim_dict(h5_main.spec_dim_labels,
                              usid.hdf_utils.get_attr(h5_main.h5_spec_inds,
                                                      'units'),
                              h5_main.get_spec_values,
                              ignore_non_linear_dims=ignore_non_linear_dims)

    num_spec_dims = len(spec_dict)
    num_pos_dims = len(pos_dict)
    _logger.info('Dimensions: Positions: {}, Spectroscopic: {}'
                 '.'.format(num_pos_dims, num_spec_dims))

    ret_vals = usid.hdf_utils.reshape_to_n_dims(h5_main, get_labels=True,
                                                lazy=lazy)
    ds_nd, success, dim_labs = ret_vals

    if success is not True:
        raise ValueError('Dataset could not be reshaped!')
    ds_nd = ds_nd.squeeze()
    _logger.info('N-dimensional shape: {}'.format(ds_nd.shape))
    _logger.info('N-dimensional labels: {}'.format(dim_labs))

    # Capturing metadata present in conventional h5USID files:
    group_attrs = dict()
    h5_chan_grp = h5_main.parent
    if isinstance(h5_chan_grp, h5py.Group):
        if 'Channel' in h5_chan_grp.name.split('/')[-1]:
            group_attrs = sidpy.hdf_utils.get_attributes(h5_chan_grp)
            h5_meas_grp = h5_main.parent
            if isinstance(h5_meas_grp, h5py.Group):
                if 'Measurement' in h5_meas_grp.name.split('/')[-1]:
                    temp = sidpy.hdf_utils.get_attributes(h5_meas_grp)
                    group_attrs.update(temp)

    """
    Normally, we might have been done but the order of the dimensions may be
    different in N-dim form and
    attributes in ancillary dataset
    """
    num_pos_dims = len(h5_main.pos_dim_labels)
    pos_dim_list = _assemble_dim_list(pos_dict, dim_labs[:num_pos_dims])
    spec_dim_list = _assemble_dim_list(spec_dict, dim_labs[num_pos_dims:])
    dim_list = pos_dim_list + spec_dim_list

    _, is_complex, is_compound, _, _ = sidpy.hdf.dtype_utils.check_dtype(h5_main)

    trunc_func = partial(_convert_to_signal_dict,
                         dim_dict_list=dim_list,
                         h5_path=h5_main.file.filename,
                         h5_dset_path=h5_main.name,
                         name=h5_main.name.split('/')[-1],
                         group_attrs=group_attrs)

    # Extracting the quantity and units of the main dataset
    quant, units = _split_descriptor(h5_main.data_descriptor)

    if is_compound:
        sig = []
        # Iterate over each dimension name:
        for name in ds_nd.dtype.names:
            q_sub, u_sub = _split_descriptor(name)
            sig.append(trunc_func(ds_nd[name], q_sub, u_sub, sig_type=quant))
    else:
        sig = [trunc_func(ds_nd, quant, units)]

    return sig


# ######## UTILITIES THAT SIMPLIFY WRITING TO H5USID FILES ####################


def _flatten_dict(nested_dict, parent_key='', sep='-'):
    """
    Flattens a nested dictionary

    Parameters
    ----------
    nested_dict : dict
        Nested dictionary
    parent_key : str, Optional
        Name of current parent
    sep : str, Optional. Default='-'
        Separator between the keys of different levels

    Returns
    -------
    dict
        Dictionary whose keys are flattened to a single level
    Notes
    -----
    Taken from https://stackoverflow.com/questions/6027558/flatten-nested-
    dictionaries-compressing-keys
    """
    items = []
    for k, v in nested_dict.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(_flatten_dict(v, new_key,
                                       sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _axes_list_to_dimensions(axes_list, data_shape, is_spec):
    dim_list = []
    dim_type = 'Pos'
    if is_spec:
        dim_type = 'Spec'
    # for dim_ind, (dim_size, dim) in enumerate(zip(data_shape, axes_list)):
    # we are going by data_shape for order (slowest to fastest)
    # so the order in axes_list does not matter
    for dim_ind, dim in enumerate(axes_list):
        dim = axes_list[dim_ind]
        dim_name = dim_type + '_Dim_' + str(dim_ind)
        if isinstance(dim.name, str):
            temp = dim.name.strip()
            if len(temp) > 0:
                dim_name = temp
        dim_units = 'a. u.'
        if isinstance(dim.units, str):
            temp = dim.units.strip()
            if len(temp) > 0:
                dim_units = temp
                # use REAL dimension size rather than what is presented in the
                # axes manager
        dim_size = data_shape[len(data_shape) - 1 - dim_ind]
        ar = np.arange(dim_size) * dim.scale + dim.offset
        dim_list.append(usid.Dimension(dim_name, dim_units, ar))
    if len(dim_list) == 0:
        return usid.Dimension('Arb', 'a. u.', 1)
    return dim_list[::-1]

# ####### REQUIRED FUNCTIONS FOR AN IO PLUGIN #################################


def file_reader(filename, dataset_path=None, ignore_non_linear_dims=True,
                lazy=False, **kwds):
    """
    Reads a USID Main dataset present in an HDF5 file into a HyperSpy Signal

    Parameters
    ----------
    filename : str
        path to HDF5 file
    dataset_path : str, Optional
        Absolute path of USID Main HDF5 dataset.
        Default - None - all Main Datasets will be read. Given that HDF5 files
        can accommodate very large datasets, lazy reading is strongly
        recommended.
        If a string like ``'/Measurement_000/Channel_000/My_Dataset'`` is
        provided, the specific dataset will be loaded.
    ignore_non_linear_dims : bool, Optional
        If True, parameters that were varied non-linearly in the desired
        dataset will result in Exceptions.
        Else, all such non-linearly varied parameters will be treated as
        linearly varied parameters and a Signal object will be generated.

    Returns
    -------
    list of hyperspy.signals.Signal object
    """
    if not isinstance(filename, str):
        raise TypeError('filename should be a string')
    if not os.path.isfile(filename):
        raise FileNotFoundError(f'No file found at: {filename}')

    # Need to keep h5 file handle open indefinitely if lazy
    # Using "with" will cause the file to be closed
    h5_f = h5py.File(filename, 'r')
    if dataset_path is None:
        all_main_dsets = usid.hdf_utils.get_all_main(h5_f)
        signals = []
        for h5_dset in all_main_dsets:
            # Note that the function returns a list already.
            # Should not append
            signals += _usidataset_to_signal(h5_dset,
                                             ignore_non_linear_dims=
                                             ignore_non_linear_dims,
                                             lazy=lazy, **kwds)
        return signals
    else:
        if not isinstance(dataset_path, str):
            raise TypeError("'dataset_path' should be a string")
        h5_dset = h5_f[dataset_path]
        return _usidataset_to_signal(h5_dset,
                                     ignore_non_linear_dims=
                                     ignore_non_linear_dims,
                                     lazy=lazy, **kwds)

    # At least close the file handle if not lazy load
    if not lazy:
        h5_f.close()


def file_writer(filename, object2save, **kwds):
    """
    Writes a HyperSpy Signal object to a HDF5 file formatted according to USID

    Parameters
    ----------
    filename : str
        Path to target HDF5 file
    object2save : hyperspy.signals.Signal
        A HyperSpy signal
    """
    append = False
    if os.path.exists(filename):
        append = True

    hs_shape = object2save.data.shape

    # Not sure how to safely ignore spurious / additional dimensions
    if len(object2save.axes_manager.shape) != len(hs_shape):
        raise ValueError('Number of dimensions in data (shape: {}) does not '
                         'match number of axes: ({})'
                         '.'.format(hs_shape,
                                    len(object2save.axes_manager.shape)))

    parm_dict = _flatten_dict(object2save.metadata.as_dictionary())
    temp = object2save.original_metadata.as_dictionary()
    parm_dict.update(_flatten_dict(temp, parent_key='Original'))

    num_pos_dims = object2save.axes_manager.navigation_dimension
    nav_axes = object2save.axes_manager.navigation_axes
    sig_axes = object2save.axes_manager.signal_axes

    data_2d = object2save.data
    # data_2d is assumed to have dimensions arranged from slowest to fastest
    # varying dimensions
    if num_pos_dims > 0 and object2save.axes_manager.signal_dimension > 0:
        # now flatten to 2D:
        data_2d = data_2d.reshape(np.prod(hs_shape[:num_pos_dims]),
                                  np.prod(hs_shape[num_pos_dims:]))
        pos_dims = _axes_list_to_dimensions(nav_axes,
                                            hs_shape[:num_pos_dims], False)
        spec_dims = _axes_list_to_dimensions(sig_axes,
                                             hs_shape[num_pos_dims:], True)
    elif num_pos_dims == 0:
        # only spectroscopic:
        # now flatten to 2D:
        data_2d = data_2d.reshape(1, -1)
        pos_dims = _axes_list_to_dimensions(nav_axes, [], False)
        spec_dims = _axes_list_to_dimensions(sig_axes, hs_shape, True)
    else:
        # now flatten to 2D:
        data_2d = data_2d.reshape(-1, 1)
        pos_dims = _axes_list_to_dimensions(nav_axes, hs_shape, False)
        spec_dims = _axes_list_to_dimensions(sig_axes, [], True)

    #  Does HyperSpy store the physical quantity and units somewhere?
    phy_quant = 'Unknown Quantity'
    phy_units = 'Unknown Units'
    dset_name = 'Raw_Data'



    if not append:
        tran = usid.NumpyTranslator()
        _ = tran.translate(filename, dset_name, data_2d, phy_quant, phy_units,
                           pos_dims, spec_dims, parm_dict=parm_dict,
                           slow_to_fast=True, **kwds)
    else:
        with h5py.File(filename, mode='r+') as h5_f:
            h5_grp = usid.hdf_utils.create_indexed_group(h5_f, 'Measurement')
            usid.hdf_utils.write_simple_attrs(h5_grp, parm_dict)
            h5_grp = usid.hdf_utils.create_indexed_group(h5_grp, 'Channel')
            _ = usid.hdf_utils.write_main_dataset(h5_grp, data_2d, dset_name,
                                                  phy_quant, phy_units,
                                                  pos_dims,  spec_dims,
                                                  slow_to_fast=True, **kwds)
