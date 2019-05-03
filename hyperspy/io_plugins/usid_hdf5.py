import os
import logging
from warnings import warn
from functools import partial
import collections
import h5py
import numpy as np
import dask.array as da
import pyUSID as usid

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
version = "0.0.5.1"

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
    """
    dim_dict = dict()
    for dim_name, units in zip(labels, units):
        dim_vals = val_func(dim_name)
        if len(dim_vals) == 1:
            # Empty dimension!
            continue
        else:
            step_size = np.unique(np.diff(dim_vals))
            if len(step_size) > 1:
                # often we end up here. In most cases,
                step_avg = step_size.max()
                step_size -= step_avg
                var = np.mean(np.abs(step_size))
                if var / step_avg < 1E-3:
                    step_size = [step_avg]
                else:
                    # TODO: return such dimensions as Signals
                    if ignore_non_linear_dims:
                        warn('Ignoring non-linearity of dimension: '
                             '{}'.format(dim_name))
                        step_size = [1]
                        dim_vals[0] = 0
                    else:
                        raise ValueError('Cannot load provided dataset. '
                                         'Parameter: {} was varied '
                                         'non-linearly. Supply keyword '
                                         'argument "ignore_non_linear_dims='
                                         'True" to ignore this '
                                         'error'.format(dim_name))

        step_size = step_size[0]
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
    units = units.replace('(', '')
    units = units.replace(')', '')
    units = units.replace('[', '')
    units = units.replace(']', '')
    return quant, units


def _convert_to_signal_dict(ndim_form, quantity, units, dim_dict_list,
                            h5_path, h5_dset_path, sig_type=''):
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
    sig_type : str
        Type of measurement

    Returns
    -------

    """

    sig = {'data': ndim_form,
           'axes': dim_dict_list,
           'metadata': {
               'Signal': {'signal_type': sig_type},
               'General': {'original_filename': h5_path}
                        },
           'original_metadata': {'quantity': quantity,
                                 'units': units,
                                 'dataset_path': h5_dset_path,
                                 'original_file_type': 'USID HDF5',
                                 'pyUSID_version': usid.__version__
                                 },
           }
    return sig


def _usidataset_to_signal(h5_main, ignore_non_linear_dims=True):
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

    Returns
    -------
    list of hyperspy.signals.BaseSignal objects. USIDatasets with compound
    datatypes are broken down to multiple
    Signal objects.
    """
    h5_main = usid.USIDataset(h5_main)
    # TODO: Cannot handle data without N-dimensional form
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

    ret_vals = usid.hdf_utils.reshape_to_n_dims(h5_main, get_labels=True)
    ds_nd, success, dim_labs = ret_vals

    if success != True:
        raise ValueError('Dataset could not be reshaped!')
    ds_nd = ds_nd.squeeze()
    _logger.info('N-dimensional shape: {}'.format(ds_nd.shape))
    _logger.info('N-dimensional labels: {}'.format(dim_labs))

    """
    Normally, we might have been done but the order of the dimensions may be 
    different in N-dim form and 
    attributes in ancillary dataset
    """
    num_pos_dims = len(h5_main.pos_dim_labels)
    pos_dim_list = _assemble_dim_list(pos_dict, dim_labs[:num_pos_dims])
    spec_dim_list = _assemble_dim_list(spec_dict, dim_labs[num_pos_dims:])
    dim_list = pos_dim_list + spec_dim_list

    _, is_complex, is_compound, _, _ = usid.dtype_utils.check_dtype(h5_main)

    trunc_func = partial(_convert_to_signal_dict,
                         dim_dict_list=dim_list,
                         h5_path=h5_main.file.filename,
                         h5_dset_path=h5_main.name)

    # Extracting the quantity and units of the main dataset
    quant, units = _split_descriptor(h5_main.data_descriptor)

    if is_compound:
        sig = []
        # Iterate over each dimension name:
        for name in ds_nd.dtype.names:
            q_sub, u_sub = _split_descriptor(name)
            # TODO: Check to make sure that this will work with Dask.array
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
        if isinstance(v, collections.MutableMapping):
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
    for dim_ind in range(len(data_shape)):
        dim_size = data_shape[len(data_shape) - 1 - dim_ind]
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
        dim_list.append(usid.Dimension(dim_name, dim_units,
                                       np.arange(dim.offset,
                                                 dim.offset + dim_size *
                                                 dim.scale,
                                                 dim.scale)))
    if len(dim_list) == 0:
        return usid.Dimension('Arb', 'a. u.', 1)
    return dim_list[::-1]

# ####### REQUIRED FUNCTIONS FOR AN IO PLUGIN #################################


def file_reader(filename, dset_path='', ignore_non_linear_dims=True, **kwds):
    """
    Reads a USID Main dataset present in an HDF5 file into a HyperSpy Signal

    Parameters
    ----------
    filename : str
        path to HDF5 file
    dset_path : str, Optional.
        Absolute path of USID Main HDF5 dataset.
        Default - '' - the very first Main Dataset will be used
        If None - all Main Datasets will be read
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
        raise FileNotFoundError('No file found at: {}'.format(filename))

    with h5py.File(filename, mode='r') as h5_f:
        if dset_path is None:
            all_main_dsets = usid.hdf_utils.get_all_main(h5_f)
            signals = []
            for h5_dset in all_main_dsets:
                # Note that the function returns a list already.
                # Should not append
                signals += _usidataset_to_signal(h5_dset,
                                                 ignore_non_linear_dims=
                                                 ignore_non_linear_dims)
            return signals
        else:
            if not isinstance(dset_path, str):
                raise TypeError('dset_path should be a string')
            if len(dset_path) > 0:
                h5_dset = h5_f[dset_path]
                # All other checks will be handled by helper function
            else:
                all_main_dsets = usid.hdf_utils.get_all_main(h5_f)
                if len(all_main_dsets) > 1:
                    warn('{} contains multiple USID Main datasets.\n{}\nhas '
                         'been selected as the desired dataset. If this is not'
                         ' the desired dataset, please supply the path to the'
                         ' main USID dataset via the "dset_path" keyword '
                         'argument.\nTo read all datasets, '
                         'use "dset_path=None"'
                         '.'.format(h5_f, all_main_dsets[0].name))
                h5_dset = all_main_dsets[0]
            return _usidataset_to_signal(h5_dset,
                                         ignore_non_linear_dims=
                                         ignore_non_linear_dims)


def file_writer(filename, object2save):
    """
    Writes a HyperSpy Signal object to a HDF5 file formatted according to USID

    Parameters
    ----------
    filename : str
        Path to target HDF5 file
    object2save : hyperspy.signals.Signal
        A HyperSpy signal
    """
    if not isinstance(filename, str):
        raise TypeError('filename should be a string')
    if os.path.exists(filename):
        raise FileExistsError('Appending is not yet supported. A file already '
                              'exists at:\n{}.\n Please delete the file at the'
                              ' location or specify a different path for the '
                              'file'.format(filename))

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
    if num_pos_dims > 0 and object2save.axes_manager.signal_dimension > 0:
        # Reverse order of dimensions:
        rev_pos_order = list(range(num_pos_dims))[::-1]
        rev_spec_order = list(range(num_pos_dims, len(hs_shape)))[::-1]
        data_2d = data_2d.transpose(rev_pos_order + rev_spec_order)
        # now flatten to 2D:
        data_2d = data_2d.reshape(np.prod(hs_shape[:num_pos_dims]),
                                  np.prod(hs_shape[num_pos_dims:]))
        pos_dims = _axes_list_to_dimensions(nav_axes,
                                            hs_shape[:num_pos_dims], False)
        spec_dims = _axes_list_to_dimensions(sig_axes,
                                             hs_shape[num_pos_dims:], True)
    elif num_pos_dims == 0:
        # only spectroscopic:
        # Reverse order of dimensions:
        data_2d = data_2d.transpose(list(range(len(hs_shape)))[::-1])
        # now flatten to 2D:
        data_2d = data_2d.reshape(1, -1)
        pos_dims = _axes_list_to_dimensions(nav_axes, [], False)
        spec_dims = _axes_list_to_dimensions(sig_axes, hs_shape, True)
    else:
        # Reverse order of dimensions:
        data_2d = data_2d.transpose(list(range(len(hs_shape)))[::-1])
        # now flatten to 2D:
        data_2d = data_2d.reshape(-1, 1)
        pos_dims = _axes_list_to_dimensions(nav_axes, hs_shape, False)
        spec_dims = _axes_list_to_dimensions(sig_axes, [], True)

    #  Does HyperSpy store the physical quantity and units somewhere?
    tran = usid.NumpyTranslator()
    _ = tran.translate(filename, 'Raw_Data', data_2d, 'Unknown Quantity',
                       'Unknown Units', pos_dims, spec_dims,
                       parm_dict=parm_dict)
