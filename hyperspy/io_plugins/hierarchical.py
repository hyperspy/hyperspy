from packaging.version import Version
import warnings
import logging
import datetime
import ast

import h5py

import numpy as np
import dask.array as da
from traits.api import Undefined
from hyperspy.misc.utils import ensure_unicode, multiply, get_object_package_info
from hyperspy.axes import AxesManager


version = "3.1"

default_version = Version(version)

not_valid_format = 'The file is not a valid HyperSpy hdf5 file'

_logger = logging.getLogger(__name__)


def _overwrite_dataset(group,
                       data,
                       key,
                       signal_axes=None,
                       chunks=None,
                       get_signal_chunks=None,
                       get_object_dset=None,
                       store_data=None,
                       **kwds):
    """Overwrites some dataset in a hierarchical dataset.

    Parameters
    ----------
    group: Zarr.Group or h5py.Group
        The group to write the data to
    data: Array-like
        The data to be written
    key: str
        The key for the data
    signal_axes: tuple
        The indexes of the signal axes
    chunks: tuple
        The chunks for the dataset. If None then get_signal_chunks will be called
    get_signal_chunks: func
        A function to get the signal chunks for the dataset
    get_object_dset: func
        A function to get the object dset for saving ragged arrays.
    store_data: func
        A function to store the data in some hierarchical data format
    kwds:
        Any additional keywords for to be passed to the store data function
        The store data function is passed for each hierarchical data format
    Returns
    -------

    """
    if chunks is None:
        if isinstance(data, da.Array):
            # For lazy dataset, by default, we use the current dask chunking
            chunks = tuple([c[0] for c in data.chunks])
        else:
            # If signal_axes=None, use automatic h5py chunking, otherwise
            # optimise the chunking to contain at least one signal per chunk
            chunks = get_signal_chunks(data.shape, data.dtype, signal_axes)
    if np.issubdtype(data.dtype, np.dtype('U')):
        # Saving numpy unicode type is not supported in h5py
        data = data.astype(np.dtype('S'))
    if data.dtype == np.dtype('O'):
        dset = get_object_dset(group, data, key, chunks, **kwds)

    else:
        got_data = False
        maxshape = tuple(None for _ in data.shape)
        while not got_data:
            try:
                these_kwds = kwds.copy()
                these_kwds.update(dict(shape=data.shape,
                                       dtype=data.dtype,
                                       exact=True,
                                       maxshape=maxshape,
                                       chunks=chunks,
                                       shuffle=True, ))

                # If chunks is True, the `chunks` attribute of `dset` below
                # contains the chunk shape guessed by h5py
                dset = group.require_dataset(key, **these_kwds)
                got_data = True
            except TypeError:
                # if the shape or dtype/etc do not match,
                # we delete the old one and create new in the next loop run
                del group[key]
    if dset == data:
        # just a reference to already created thing
        pass
    else:
        _logger.info(f"Chunks used for saving: {chunks}")
        store_data(data, dset, group, key, chunks, **kwds)


class HierarchicalReader:
    """A generic Reader class for reading data from hierarchical file types."""
    def __init__(self,
                 file):
        """ Initalizes a general reader for hierarchical signals.

        Parmeters
        ---------
        file: str
            A file to be read.
        """
        self.file = file
        self.version = self.get_format_version()
        self.Dataset = None
        self.Group = None
        self.unicode_kwds = None
        self.ragged_kwds = None

    def get_format_version(self):
        if "file_format_version" in self.file.attrs:
            version = self.file.attrs["file_format_version"]
            if isinstance(version, bytes):
                version = version.decode()
            if isinstance(version, float):
                version = str(round(version, 2))
        elif "Experiments" in self.file:
            # Chances are that this is a HSpy hdf5 file version 1.0
            version = "1.0"
        elif "Analysis" in self.f:
            # Starting version 2.0 we have "Analysis" field as well
            version = "2.0"
        else:
            raise IOError(not_valid_format)
        return Version(version)

    def read(self, lazy):
        models_with_signals = []
        standalone_models = []
        if 'Analysis/models' in self.file:
            try:
                m_gr = self.file['Analysis/models']
                for model_name in m_gr:
                    if '_signal' in m_gr[model_name].attrs:
                        key = m_gr[model_name].attrs['_signal']
                        # del m_gr[model_name].attrs['_signal']
                        res = self.group2dict(
                            m_gr[model_name],
                            lazy=lazy)
                        del res['_signal']
                        models_with_signals.append((key, {model_name: res}))
                    else:
                        standalone_models.append(
                            {model_name: self.group2dict(
                                m_gr[model_name], lazy=lazy)})
            except TypeError:
                raise IOError(not_valid_format)
        experiments = []
        exp_dict_list = []
        if 'Experiments' in self.file:
            for ds in self.file['Experiments']:
                if isinstance(self.file['Experiments'][ds], self.Group):
                    if 'data' in self.file['Experiments'][ds]:
                        experiments.append(ds)
            # Parse the file
            for experiment in experiments:

                exg = self.file['Experiments'][experiment]
                exp = self.group2signaldict(exg, lazy)
                # assign correct models, if found:
                _tmp = {}
                for (key, _dict) in reversed(models_with_signals):
                    if key == exg.name:
                        _tmp.update(_dict)
                        models_with_signals.remove((key, _dict))
                exp['models'] = _tmp

                exp_dict_list.append(exp)

        for _, m in models_with_signals:
            standalone_models.append(m)

        exp_dict_list.extend(standalone_models)
        if not len(exp_dict_list):
            raise IOError('This is not a valid HyperSpy HDF5 file. '
                          'You can still load the data using a hdf5 reader, '
                          'e.g. h5py, and manually create a Signal. '
                          'Please, refer to the User Guide for details')
        return exp_dict_list

    def group2signaldict(self,
                         group,
                         lazy=False):
        if self.version < Version("1.2"):
            metadata = "mapped_parameters"
            original_metadata = "original_parameters"
        else:
            metadata = "metadata"
            original_metadata = "original_metadata"

        exp = {'metadata': self.group2dict(
            group[metadata], lazy=lazy),
            'original_metadata': self.group2dict(
                group[original_metadata], lazy=lazy),
            'attributes': {}
        }
        if "package" in group.attrs:
            # HyperSpy version is >= 1.5
            exp["package"] = group.attrs["package"]
            exp["package_version"] = group.attrs["package_version"]
        else:
            # Prior to v1.4 we didn't store the package information. Since there
            # were already external package we cannot assume any package provider so
            # we leave this empty.
            exp["package"] = ""
            exp["package_version"] = ""

        data = group['data']
        if lazy:
            data = da.from_array(data, chunks=data.chunks)
            exp['attributes']['_lazy'] = True
        else:
            data = np.asanyarray(data)
        exp['data'] = data
        axes = []
        for i in range(len(exp['data'].shape)):
            try:
                axes.append(self.group2dict(group['axis-%i' % i]))
                axis = axes[-1]
                for key, item in axis.items():
                    if isinstance(item, np.bool_):
                        axis[key] = bool(item)
                    else:
                        axis[key] = ensure_unicode(item)
            except KeyError:
                break
        if len(axes) != len(exp['data'].shape):  # broke from the previous loop
            try:
                axes = [i for k, i in sorted(iter(self.group2dict(
                    group['_list_' + str(len(exp['data'].shape)) + '_axes'],
                    lazy=lazy).items()))]
            except KeyError:
                raise IOError(not_valid_format)
        exp['axes'] = axes
        if 'learning_results' in group.keys():
            exp['attributes']['learning_results'] = \
                self.group2dict(
                    group['learning_results'],
                    lazy=lazy)
        if 'peak_learning_results' in group.keys():
            exp['attributes']['peak_learning_results'] = \
                self.group2dict(
                    group['peak_learning_results'],
                    lazy=lazy)

        # If the title was not defined on writing the Experiment is
        # then called __unnamed__. The next "if" simply sets the title
        # back to the empty string
        if "General" in exp["metadata"] and "title" in exp["metadata"]["General"]:
            if '__unnamed__' == exp['metadata']['General']['title']:
                exp['metadata']["General"]['title'] = ''

        if self.version < Version("1.1"):
            # Load the decomposition results written with the old name,
            # mva_results
            if 'mva_results' in group.keys():
                exp['attributes']['learning_results'] = self.group2dict(
                    group['mva_results'], lazy=lazy)
            if 'peak_mva_results' in group.keys():
                exp['attributes']['peak_learning_results'] = self.group2dict(
                    group['peak_mva_results'], lazy=lazy)
            # Replace the old signal and name keys with their current names
            if 'signal' in exp['metadata']:
                if "Signal" not in exp["metadata"]:
                    exp["metadata"]["Signal"] = {}
                exp['metadata']["Signal"]['signal_type'] = \
                    exp['metadata']['signal']
                del exp['metadata']['signal']

            if 'name' in exp['metadata']:
                if "General" not in exp["metadata"]:
                    exp["metadata"]["General"] = {}
                exp['metadata']['General']['title'] = \
                    exp['metadata']['name']
                del exp['metadata']['name']

        if self.version < Version("1.2"):
            if '_internal_parameters' in exp['metadata']:
                exp['metadata']['_HyperSpy'] = \
                    exp['metadata']['_internal_parameters']
                del exp['metadata']['_internal_parameters']
                if 'stacking_history' in exp['metadata']['_HyperSpy']:
                    exp['metadata']['_HyperSpy']["Stacking_history"] = \
                        exp['metadata']['_HyperSpy']['stacking_history']
                    del exp['metadata']['_HyperSpy']["stacking_history"]
                if 'folding' in exp['metadata']['_HyperSpy']:
                    exp['metadata']['_HyperSpy']["Folding"] = \
                        exp['metadata']['_HyperSpy']['folding']
                    del exp['metadata']['_HyperSpy']["folding"]
            if 'Variance_estimation' in exp['metadata']:
                if "Noise_properties" not in exp["metadata"]:
                    exp["metadata"]["Noise_properties"] = {}
                exp['metadata']['Noise_properties']["Variance_linear_model"] = \
                    exp['metadata']['Variance_estimation']
                del exp['metadata']['Variance_estimation']
            if "TEM" in exp["metadata"]:
                if "Acquisition_instrument" not in exp["metadata"]:
                    exp["metadata"]["Acquisition_instrument"] = {}
                exp["metadata"]["Acquisition_instrument"]["TEM"] = \
                    exp["metadata"]["TEM"]
                del exp["metadata"]["TEM"]
                tem = exp["metadata"]["Acquisition_instrument"]["TEM"]
                if "EELS" in tem:
                    if "dwell_time" in tem:
                        tem["EELS"]["dwell_time"] = tem["dwell_time"]
                        del tem["dwell_time"]
                    if "dwell_time_units" in tem:
                        tem["EELS"]["dwell_time_units"] = tem["dwell_time_units"]
                        del tem["dwell_time_units"]
                    if "exposure" in tem:
                        tem["EELS"]["exposure"] = tem["exposure"]
                        del tem["exposure"]
                    if "exposure_units" in tem:
                        tem["EELS"]["exposure_units"] = tem["exposure_units"]
                        del tem["exposure_units"]
                    if "Detector" not in tem:
                        tem["Detector"] = {}
                    tem["Detector"] = tem["EELS"]
                    del tem["EELS"]
                if "EDS" in tem:
                    if "Detector" not in tem:
                        tem["Detector"] = {}
                    if "EDS" not in tem["Detector"]:
                        tem["Detector"]["EDS"] = {}
                    tem["Detector"]["EDS"] = tem["EDS"]
                    del tem["EDS"]
                del tem
            if "SEM" in exp["metadata"]:
                if "Acquisition_instrument" not in exp["metadata"]:
                    exp["metadata"]["Acquisition_instrument"] = {}
                exp["metadata"]["Acquisition_instrument"]["SEM"] = \
                    exp["metadata"]["SEM"]
                del exp["metadata"]["SEM"]
                sem = exp["metadata"]["Acquisition_instrument"]["SEM"]
                if "EDS" in sem:
                    if "Detector" not in sem:
                        sem["Detector"] = {}
                    if "EDS" not in sem["Detector"]:
                        sem["Detector"]["EDS"] = {}
                    sem["Detector"]["EDS"] = sem["EDS"]
                    del sem["EDS"]
                del sem

            if "Sample" in exp["metadata"] and "Xray_lines" in exp[
                    "metadata"]["Sample"]:
                exp["metadata"]["Sample"]["xray_lines"] = exp[
                    "metadata"]["Sample"]["Xray_lines"]
                del exp["metadata"]["Sample"]["Xray_lines"]

            for key in ["title", "date", "time", "original_filename"]:
                if key in exp["metadata"]:
                    if "General" not in exp["metadata"]:
                        exp["metadata"]["General"] = {}
                    exp["metadata"]["General"][key] = exp["metadata"][key]
                    del exp["metadata"][key]
            for key in ["record_by", "signal_origin", "signal_type"]:
                if key in exp["metadata"]:
                    if "Signal" not in exp["metadata"]:
                        exp["metadata"]["Signal"] = {}
                    exp["metadata"]["Signal"][key] = exp["metadata"][key]
                    del exp["metadata"][key]

        if self.version < Version("3.0"):
            if "Acquisition_instrument" in exp["metadata"]:
                # Move tilt_stage to Stage.tilt_alpha
                # Move exposure time to Detector.Camera.exposure_time
                if "TEM" in exp["metadata"]["Acquisition_instrument"]:
                    tem = exp["metadata"]["Acquisition_instrument"]["TEM"]
                    exposure = None
                    if "tilt_stage" in tem:
                        tem["Stage"] = {"tilt_alpha": tem["tilt_stage"]}
                        del tem["tilt_stage"]
                    if "exposure" in tem:
                        exposure = "exposure"
                    # Digital_micrograph plugin was parsing to 'exposure_time'
                    # instead of 'exposure': need this to be compatible with
                    # previous behaviour
                    if "exposure_time" in tem:
                        exposure = "exposure_time"
                    if exposure is not None:
                        if "Detector" not in tem:
                            tem["Detector"] = {"Camera": {
                                "exposure": tem[exposure]}}
                        tem["Detector"]["Camera"] = {"exposure": tem[exposure]}
                        del tem[exposure]
                # Move tilt_stage to Stage.tilt_alpha
                if "SEM" in exp["metadata"]["Acquisition_instrument"]:
                    sem = exp["metadata"]["Acquisition_instrument"]["SEM"]
                    if "tilt_stage" in sem:
                        sem["Stage"] = {"tilt_alpha": sem["tilt_stage"]}
                        del sem["tilt_stage"]

        return exp

    def group2dict(self,
                  group,
                  dictionary=None,
                  lazy=False):
        if dictionary is None:
            dictionary = {}
        for key, value in group.attrs.items():
            if isinstance(value, bytes):
                value = value.decode()
            if isinstance(value, (np.string_, str)):
                if value == '_None_':
                    value = None
            elif isinstance(value, np.bool_):
                value = bool(value)
            elif isinstance(value, np.ndarray) and value.dtype.char == "S":
                # Convert strings to unicode
                value = value.astype("U")
                if value.dtype.str.endswith("U1"):
                    value = value.tolist()
            # skip signals - these are handled below.
            if key.startswith('_sig_'):
                pass
            elif key.startswith('_list_empty_'):
                dictionary[key[len('_list_empty_'):]] = []
            elif key.startswith('_tuple_empty_'):
                dictionary[key[len('_tuple_empty_'):]] = ()
            elif key.startswith('_bs_'):
                dictionary[key[len('_bs_'):]] = value.tobytes()
            # The following two elif stataments enable reading date and time from
            # v < 2 of HyperSpy's metadata specifications
            elif key.startswith('_datetime_date'):
                date_iso = datetime.date(
                    *ast.literal_eval(value[value.index("("):])).isoformat()
                dictionary[key.replace("_datetime_", "")] = date_iso
            elif key.startswith('_datetime_time'):
                date_iso = datetime.time(
                    *ast.literal_eval(value[value.index("("):])).isoformat()
                dictionary[key.replace("_datetime_", "")] = date_iso
            else:
                dictionary[key] = value
        if not isinstance(group, self.Dataset):
            for key in group.keys():
                if key.startswith('_sig_'):
                    from hyperspy.io import dict2signal
                    dictionary[key[len('_sig_'):]] = (
                        dict2signal(self.group2signaldict(
                            group[key], lazy=lazy)))
                elif isinstance(group[key], self.Dataset):
                    dat = group[key]
                    kn = key
                    if key.startswith("_list_"):
                        if (h5py.check_string_dtype(dat.dtype) and
                                hasattr(dat, 'asstr')):
                            # h5py 3.0 and newer
                            # https://docs.h5py.org/en/3.0.0/strings.html
                            dat = dat.asstr()[:]
                        ans = np.array(dat)
                        ans = ans.tolist()
                        kn = key[6:]
                    elif key.startswith("_tuple_"):
                        ans = np.array(dat)
                        ans = tuple(ans.tolist())
                        kn = key[7:]
                    elif dat.dtype.char == "S":
                        ans = np.array(dat)
                        try:
                            ans = ans.astype("U")
                        except UnicodeDecodeError:
                            # There are some strings that must stay in binary,
                            # for example dill pickles. This will obviously also
                            # let "wrong" binary string fail somewhere else...
                            pass
                    elif lazy:
                        ans = da.from_array(dat, chunks=dat.chunks)
                    else:
                        ans = np.array(dat)
                    dictionary[kn] = ans
                elif key.startswith('_hspy_AxesManager_'):
                    dictionary[key[len('_hspy_AxesManager_'):]] = AxesManager(
                        [i for k, i in sorted(iter(
                            self.group2dict(
                                group[key], lazy=lazy).items()
                        ))])
                elif key.startswith('_list_'):
                    dictionary[key[7 + key[6:].find('_'):]] = \
                        [i for k, i in sorted(iter(
                            self.group2dict(
                                group[key], lazy=lazy).items()
                        ))]
                elif key.startswith('_tuple_'):
                    dictionary[key[8 + key[7:].find('_'):]] = tuple(
                        [i for k, i in sorted(iter(
                            self.group2dict(
                                group[key], lazy=lazy).items()
                        ))])
                else:
                    dictionary[key] = {}
                    self.group2dict(
                        group[key],
                        dictionary[key],
                        lazy=lazy)

        return dictionary

class HierarchicalWriter:
    """An object used to simplify and orgainize the process for
    writing a Hierachical signal.  (.hspy format)
    """
    def __init__(self,
                 file,
                 signal,
                 group,
                 **kwds):
        """Initialize a generic file writer for hierachical data storage types.

        Parameters
        ----------
        file: str
            The file where the signal is to be saved
        signal: BaseSignal
            A BaseSignal to be saved
        group: Group
            A group to where the experimental data will be saved.
        kwds:
            Any additional keywords used for saving the data.
        """
        self.file = file
        self.signal = signal
        self.group = group
        self.Dataset = None
        self.Group = None
        self.unicode_kwds = None
        self.ragged_kwds = None
        self.kwds = kwds
        self.overwrite_dataset=None

    def write(self):
        self.write_signal(self.signal,
                          self.group,
                          **self.kwds)

    def write_signal(self, signal, group, **kwds):
        "Writes a hyperspy signal to a hdf5 group"

        group.attrs.update(get_object_package_info(signal))
        if Version(version) < Version("1.2"):
            metadata = "mapped_parameters"
            original_metadata = "original_parameters"
        else:
            metadata = "metadata"
            original_metadata = "original_metadata"



        for axis in signal.axes_manager._axes:
            axis_dict = axis.get_axis_dictionary()
            coord_group = group.create_group(
                'axis-%s' % axis.index_in_array)
            self.dict2group(axis_dict, coord_group, **kwds)
        mapped_par = group.create_group(metadata)
        metadata_dict = signal.metadata.as_dictionary()
        self.overwrite_dataset(group,
                               signal.data,
                               'data',
                               signal_axes=signal.axes_manager.signal_indices_in_array,
                               **kwds)
        if default_version < Version("1.2"):
            metadata_dict["_internal_parameters"] = \
                metadata_dict.pop("_HyperSpy")
        # Remove chunks from the kwds since it wouldn't have the same rank as the
        # dataset and can't be used
        kwds.pop('chunks', None)
        self.dict2group(metadata_dict, mapped_par, **kwds)
        original_par = group.create_group(original_metadata)
        self.dict2group(signal.original_metadata.as_dictionary(), original_par,
                      **kwds)
        learning_results = group.create_group('learning_results')
        self.dict2group(signal.learning_results.__dict__,
                      learning_results, **kwds)
        if hasattr(signal, 'peak_learning_results'):
            peak_learning_results = group.create_group(
                'peak_learning_results')
            self.dict2group(signal.peak_learning_results.__dict__,
                          peak_learning_results, **kwds)

        if len(signal.models):
            model_group = self.file.require_group('Analysis/models')
            self.dict2group(signal.models._models.as_dictionary(),
                          model_group, **kwds)
            for model in model_group.values():
                model.attrs['_signal'] = group.name

    def dict2group(self, dictionary, group, **kwds):
        "Recursive writer of dicts and signals"

        from hyperspy.misc.utils import DictionaryTreeBrowser
        from hyperspy.signal import BaseSignal

        for key, value in dictionary.items():
            if isinstance(value, dict):
                self.dict2group(value, group.create_group(key),
                              **kwds)
            elif isinstance(value, DictionaryTreeBrowser):
                self.dict2group(value.as_dictionary(),
                              group.create_group(key),
                              **kwds)
            elif isinstance(value, BaseSignal):
                kn = key if key.startswith('_sig_') else '_sig_' + key
                self.write_signal(value, group.require_group(kn))
            elif isinstance(value, (np.ndarray, self.Dataset, da.Array)):
                self.overwrite_dataset(group, value, key, **kwds)
            elif value is None:
                group.attrs[key] = '_None_'
            elif isinstance(value, bytes):
                try:
                    # binary string if has any null characters (otherwise not
                    # supported by hdf5)
                    value.index(b'\x00')
                    group.attrs['_bs_' + key] = np.void(value)
                except ValueError:
                    group.attrs[key] = value.decode()
            elif isinstance(value, str):
                group.attrs[key] = value
            elif isinstance(value, AxesManager):
                self.dict2group(value.as_dictionary(),
                              group.create_group('_hspy_AxesManager_' + key),
                              **kwds)
            elif isinstance(value, list):
                if len(value):
                    self.parse_structure(key, group, value, '_list_', **kwds)
                else:
                    group.attrs['_list_empty_' + key] = '_None_'
            elif isinstance(value, tuple):
                if len(value):
                    self.parse_structure(key, group, value, '_tuple_', **kwds)
                else:
                    group.attrs['_tuple_empty_' + key] = '_None_'

            elif value is Undefined:
                continue
            else:
                try:
                    group.attrs[key] = value
                except BaseException:
                    _logger.exception(
                        "The hdf5 writer could not write the following "
                        "information in the file: %s : %s", key, value)

    def parse_structure(self, key, group, value, _type, **kwds):
        from hyperspy.signal import BaseSignal
        try:
            # Here we check if there are any signals in the container, as
            # casting a long list of signals to a numpy array takes a very long
            # time. So we check if there are any, and save numpy the trouble
            if np.any([isinstance(t, BaseSignal) for t in value]):
                tmp = np.array([[0]])
            else:
                tmp = np.array(value)
        except ValueError:
            tmp = np.array([[0]])
        if tmp.dtype == np.dtype('O') or tmp.ndim != 1:
            self.dict2group(dict(zip(
                [str(i) for i in range(len(value))], value)),
                group.create_group(_type + str(len(value)) + '_' + key),
                **kwds)
        elif tmp.dtype.type is np.unicode_:
            if _type + key in group:
                del group[_type + key]
            group.create_dataset(_type + key,
                                 shape=tmp.shape,
                                 **self.unicode_kwds,
                                 **kwds)
            group[_type + key][:] = tmp[:]
        else:
            if _type + key in group:
                del group[_type + key]
            group.create_dataset(_type + key,
                                 data=tmp,
                                 **kwds)


