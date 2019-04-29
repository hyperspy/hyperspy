import os
from warnings import warn
from functools import partial
import collections
import h5py
import numpy as np
import dask.array as da
import pyUSID as usid
# from hyperspy.signals import BaseSignal, ComplexSignal


# Plugin characteristics
# ----------------------
format_name = 'USID'
description = \
    'Data structured according to the Universal Spectroscopic and Imaging Data (USID) model written into ' \
    'Hierarchical Data Format (HDF5) files'
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

# ######################## UTILITIES THAT SIMPLIFY READING FROM H5USID FILES ###########################################


def __get_dim_dict(labels, units, val_func, ignore_non_linear_dims=True):
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
                    if ignore_non_linear_dims:
                        warn('Ignoring non-linearity of dimension: {}'.format(dim_name))
                        step_size = [1]
                        dim_vals[0] = 0
                    else:
                        raise ValueError('Cannot load provided dataset. '
                                         'Parameter: {} was varied non-linearly'.format(dim_name))

        step_size = step_size[0]
        dim_dict[dim_name] = {'size': len(dim_vals),
                              'name': dim_name,
                              'units': units,
                              'scale': step_size,
                              'offset': dim_vals[0]}
    return dim_dict


def __assemble_dim_list(dim_dict, dim_names):
    dim_list = []
    for dim_name in dim_names:
        try:
            dim_list.append(dim_dict[dim_name])
        except KeyError:
            pass
    return dim_list


def __split_descriptor(desc):
    ind = desc.rfind('(')
    if ind < 0:
        ind = desc.rfind('[')
        if ind < 0:
            return '', ''

    quant = desc[:ind].strip()
    units = desc[ind:]
    units = units.replace('(', '')
    units = units.replace(')', '')
    units = units.replace('[', '')
    units = units.replace(']', '')
    return quant, units


def __convert_to_hs_signal(ndim_form, quantity, units, converter, dim_dict_list, spec_dim_names,
                           h5_path, h5_dset_path, sig_type='', verbose=False):

    sig = {'data': ndim_form,
           'axes': dim_dict_list,
           'metadata': {
               'Signal':{'quantity': quantity, 'units': units, 'signal_type': sig_type},
               'General': {'original_filename': h5_path, 'dataset_path': h5_dset_path,
                           'original_file_type': 'USID HDF5', 'pyUSID_version': usid.__version__}
           }}

    """
    # NOT sure how to apply transposes etc. This information is lost!
    if len(spec_dim_names) == 0:
        if verbose:
            print('No Spectroscopic dimensions - so transposing')
        sig = sig.transpose()
    else:
        if verbose:
            print('Explicitely stating spec dims')
        sig = sig.as_signal2D(spec_dim_names)
    if verbose:
        print('Signal after separation of dimensions:')
        print(sig)
        print(sig.axes_manager)
    """
    return sig


def usidataset_to_signal(h5_main, verbose=False, ignore_non_linear_dims=True):
    """
    Converts a single specified USIDataset object to one or more Signal objects
    Parameters
    ----------
    h5_main : pyUSID.USIDataset object
        USID Main dataset
    verbose : bool, Optional. Default = False
        Whether or not to print debugging statements
    ignore_non_linear_dims : bool, Optional
        If True, parameters that were varied non-linearly in the desired dataset will result in Exceptions.
        Else, all such non-linearly varied parameters will be treated as linearly varied parameters and
        a Signal object will be generated.

    Returns
    -------
    list of hyperspy.signals.BaseSignal objects. USIDatasets with compound datatypes are broken down to multiple
    Signal objects.
    """
    h5_main = usid.USIDataset(h5_main)
    # TODO: Cannot handle data without N-dimensional form
    # First get dictionary of axes that HyperSpy likes to see. Ignore singular dimensions
    pos_dict = __get_dim_dict(h5_main.pos_dim_labels,
                              usid.hdf_utils.get_attr(h5_main.h5_pos_inds, 'units'),
                              h5_main.get_pos_values,
                              ignore_non_linear_dims=ignore_non_linear_dims)
    spec_dict = __get_dim_dict(h5_main.spec_dim_labels,
                               usid.hdf_utils.get_attr(h5_main.h5_spec_inds, 'units'),
                               h5_main.get_spec_values,
                               ignore_non_linear_dims=ignore_non_linear_dims)

    num_spec_dims = len(spec_dict)
    num_pos_dims = len(pos_dict)
    if verbose:
        print(num_pos_dims, num_spec_dims)

    ds_nd, success, dim_labs = usid.hdf_utils.reshape_to_n_dims(h5_main, get_labels=True)
    if success != True:
        raise ValueError('Dataset could not be reshaped!')
    ds_nd = ds_nd.squeeze()
    if verbose:
        print(ds_nd.shape, dim_labs)

    """
    Normally, we might have been done but the order of the dimensions may be different in N-dim form and 
    attributes in ancillary dataset
    """
    num_pos_dims = len(h5_main.pos_dim_labels)
    pos_dim_list = __assemble_dim_list(pos_dict, dim_labs[:num_pos_dims])
    spec_dim_list = __assemble_dim_list(spec_dict, dim_labs[num_pos_dims:])
    dim_list = pos_dim_list + spec_dim_list

    _, is_complex, is_compound, _, _ = usid.dtype_utils.check_dtype(h5_main)

    """
    converter = BaseSignal
    if is_complex:
        converter = ComplexSignal
    """
    converter = None

    trunc_func = partial(__convert_to_hs_signal,
                         converter=converter,
                         dim_dict_list=dim_list,
                         spec_dim_names=list(spec_dict.keys()),
                         h5_path=h5_main.file.filename,
                         h5_dset_path=h5_main.name)

    # Extracting the quantity and units of the main dataset
    quant, units = __split_descriptor(h5_main.data_descriptor)

    if is_compound:
        sig = []
        # Iterate over each dimension name:
        for name in ds_nd.dtype.names:
            q_sub, u_sub = __split_descriptor(name)
            # TODO: Check to make sure that this will work with Dask.array
            sig.append(trunc_func(ds_nd[name], q_sub, u_sub, sig_type=quant, verbose=verbose))
    else:
        sig = [trunc_func(ds_nd, quant, units, verbose=verbose)]

    return sig


# ######################## UTILITIES THAT SIMPLIFY WRITING TO H5USID FILES #############################################


def __flatten_nested_dictionary(d, parent_key='', sep='-'):
    # https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(__flatten_nested_dictionary(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def __axes_list_to_dimensions(axes_list, data_shape):
    dim_list = []
    # for dim_ind, (dim_size, dim) in enumerate(zip(data_shape, axes_list)):
    for dim_ind in range(len(data_shape)):
        dim_size = data_shape[len(data_shape) - 1 - dim_ind]
        dim = axes_list[dim_ind]
        dim_name = 'Unknown_Dimension_' + str(dim_ind)
        if isinstance(dim.name, str):
            temp = dim.name.strip()
            if len(temp) > 0:
                dim_name = temp
        dim_units = 'a. u.'
        if isinstance(dim.units, str):
            temp = dim.units.strip()
            if len(temp) > 0:
                dim_units = temp
                # use REAL dimension size rather than what is presented in the axes manager
        print(dim_name, dim.size, dim_size) # dim_units, dim.offset, dim.scale,
        dim_list.append(usid.Dimension(dim_name, dim_units,
                                       np.arange(dim.offset,
                                                 dim.offset + dim_size * dim.scale,
                                                 dim.scale)))
    if len(dim_list) == 0:
        return usid.Dimension('Arb', 'a. u.', 1)
    return dim_list[::-1]

# ######################################################################################################################


def read_all_main_datasets(filename, verbose=False, ignore_non_linear_dims=False):
    """
    Reads all USID Main datasets present in the provided HDF5 file into HyperSpy Signal objects

    Parameters
    ----------
    filename : str
        path to HDF5 file
    verbose : bool, Optional. Default = False
        Whether or not to print debugging statements
    ignore_non_linear_dims : bool, Optional
        If True, parameters that were varied non-linearly in the desired dataset will result in Exceptions.
        Else, all such non-linearly varied parameters will be treated as linearly varied parameters and
        a Signal object will be generated.

    Returns
    -------
    list of hyperspy.signals.Signal object
    """
    if not isinstance(filename, str):
        raise TypeError('filename should be a string')
    if not os.path.isfile(filename):
        raise FileNotFoundError('No file found at: {}'.format(filename))

    with h5py.File(filename, mode='r') as h5_f:
        all_main_dsets = usid.hdf_utils.get_all_main(h5_f)
        signals = []
        for dset in all_main_dsets:
            signals.append(usidataset_to_signal(dset, verbose=verbose, ignore_non_linear_dims=ignore_non_linear_dims))
        return signals


def file_reader(filename, path_to_main_dataset=None, verbose=False, ignore_non_linear_dims=False, **kwds):
    """
    Reads a USID Main dataset present in an HDF5 file into a HyperSpy Signal

    Parameters
    ----------
    filename : str
        path to HDF5 file
    path_to_main_dataset : str, Optional. Default = None
        Absolute path of USID Main HDF5 dataset.
        If None, the very first Main Dataset will be used
    verbose : bool, Optional. Default = False
        Whether or not to print debugging statements
    ignore_non_linear_dims : bool, Optional
        If True, parameters that were varied non-linearly in the desired dataset will result in Exceptions.
        Else, all such non-linearly varied parameters will be treated as linearly varied parameters and
        a Signal object will be generated.

    Returns
    -------
    list of hyperspy.signals.Signal object
    """
    if not isinstance(filename, str):
        raise TypeError('filename should be a string')
    if not os.path.isfile(filename):
        raise FileNotFoundError('No file found at: {}'.format(filename))

    with h5py.File(filename, mode='r') as h5_f:
        if path_to_main_dataset is not None:
            if not isinstance(path_to_main_dataset, str):
                raise TypeError('path_to_main_dataset should be a string')
            h5_dset = h5_f[path_to_main_dataset]
            # All other checks will be handled by helper function
        else:
            all_main_dsets = usid.hdf_utils.get_all_main(h5_f)
            if len(all_main_dsets) > 0:
                warn('{} contains multiple USID Main datasets. {} has been selected as the desired dataset.'
                     'If this is not the desired dataset, please supply the path to the main dataset via'
                     'the "path_to_main_dataset" keyword argument'.format(h5_f, all_main_dsets[0]))
            h5_dset = all_main_dsets[0]
        return usidataset_to_signal(h5_dset, verbose=verbose, ignore_non_linear_dims=ignore_non_linear_dims)


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
        raise FileExistsError('A file already exists at: {}. Please delete the file at the location or specify a '
                              'different path for the file'.format(filename))
    """
    # CANNOT import BaseSignal! Probably circular import
    if not isinstance(object2save, BaseSignal):
        raise TypeError('object2save should be a valid hyperspy.signals.BaseSignal object')
    """

    # Not sure how to safely ignore spurious / additional dimensions
    if len(object2save.axes_manager.shape) != len(object2save.data.shape):
        raise ValueError('Number of dimensions in data (shape: {}) does not match number of axes: ({})'
                         '.'.format(object2save.data.shape, len(object2save.axes_manager.shape)))

    parm_dict = __flatten_nested_dictionary(object2save.metadata.as_dictionary())
    parm_dict.update(__flatten_nested_dictionary(object2save.original_metadata.as_dictionary(), parent_key='Original'))
    
    num_pos_dims = object2save.axes_manager.navigation_dimension

    data_2d = object2save.data
    if num_pos_dims > 0 and object2save.axes_manager.signal_dimension > 0:
        # Reverse order of dimensions:
        rev_pos_order = list(range(num_pos_dims))[::-1]
        rev_spec_order = list(range(num_pos_dims, len(object2save.data.shape)))[::-1]
        data_2d = data_2d.transpose(rev_pos_order + rev_spec_order)
        # now flatten to 2D:
        data_2d = data_2d.reshape(np.prod(object2save.data.shape[:num_pos_dims]),
                                  np.prod(object2save.data.shape[num_pos_dims:]))
        pos_dims = __axes_list_to_dimensions(object2save.axes_manager.navigation_axes,
                                             object2save.data.shape[:num_pos_dims])
        spec_dims = __axes_list_to_dimensions(object2save.axes_manager.signal_axes,
                                              object2save.data.shape[num_pos_dims:])
    elif num_pos_dims == 0:
        # only spectroscopic:
        # Reverse order of dimensions:
        data_2d = data_2d.transpose(list(range(len(object2save.data.shape)))[::-1])
        # now flatten to 2D:
        data_2d = data_2d.reshape(1, -1)
        pos_dims = __axes_list_to_dimensions(object2save.axes_manager.navigation_axes, [])
        spec_dims = __axes_list_to_dimensions(object2save.axes_manager.signal_axes, object2save.data.shape)
    else:
        # Reverse order of dimensions:
        data_2d = data_2d.transpose(list(range(len(object2save.data.shape)))[::-1])
        # now flatten to 2D:
        data_2d = data_2d.reshape(-1, 1)
        pos_dims = __axes_list_to_dimensions(object2save.axes_manager.navigation_axes, object2save.data.shape)
        spec_dims = __axes_list_to_dimensions(object2save.axes_manager.signal_axes, [])

    # TODO: Does HyperSpy store the physical quantity and units somewhere?
    tran = usid.NumpyTranslator()
    _ = tran.translate(filename, 'Raw_Data', data_2d, 'Unknown Quantity', 'Unknown Units', pos_dims, spec_dims,
                       parm_dict=parm_dict)
