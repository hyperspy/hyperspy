# -*- coding: utf-8 -*-
# Copyright 2007-2011 The Hyperspy developers
#
# This file is part of  Hyperspy.
#
#  Hyperspy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  Hyperspy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  Hyperspy.  If not, see <http://www.gnu.org/licenses/>.

import copy
import os.path

import numpy as np
import traits.api as t
import traitsui.api as tui

from hyperspy import messages
from hyperspy.axes import AxesManager
from hyperspy import io
from hyperspy.drawing import mpl_hie, mpl_hse
from hyperspy.misc import utils
from hyperspy.learn.mva import MVA, LearningResults
from hyperspy.misc.utils import DictionaryBrowser
from hyperspy.drawing import signal as sigdraw
from hyperspy.decorators import auto_replot
from hyperspy.defaults_parser import preferences
from hyperspy.misc.utils import ensure_directory

from matplotlib import pyplot as plt


class Signal(t.HasTraits, MVA):
    data = t.Any()
    axes_manager = t.Instance(AxesManager)
    original_parameters = t.Instance(DictionaryBrowser)
    mapped_parameters = t.Instance(DictionaryBrowser)
    tmp_parameters = t.Instance(DictionaryBrowser)
    _default_record_by = 'image'

    def __init__(self, file_data_dict=None, *args, **kw):
        """All data interaction is made through this class or its subclasses


        Parameters:
        -----------
        dictionary : dictionary
           see load_dictionary for the format
        """
        super(Signal, self).__init__()
        self.mapped_parameters = DictionaryBrowser()
        self.original_parameters = DictionaryBrowser()
        self.tmp_parameters = DictionaryBrowser()
        self.learning_results=LearningResults()
        self.peak_learning_results=LearningResults()
        if file_data_dict is not None:
            self.load_dictionary(file_data_dict)
        self._plot = None
        self._shape_before_unfolding = None
        self._axes_manager_before_unfolding = None
        self.auto_replot = True
        self.variance = None

    def __repr__(self):
        string = '<'
        string += self.__class__.__name__
        string+=", title: %s" % self.mapped_parameters.title
        string += ", dimensions: %s" % (str(self.data.shape))
        string += '>'

        return string
        
    def print_summary(self):
        string = "\n\tTitle: "
        string += self.mapped_parameters.title.decode('utf8')
        if hasattr(self.mapped_parameters,'signal_type'):
            string += "\n\tSignal type: "
            string += self.mapped_parameters.signal_type
        string += "\n\tData dimensions: "
        string += str(self.data.shape)
        if hasattr(self.mapped_parameters, 'record_by'):
            string += "\n\tData representation: "
            string += self.mapped_parameters.record_by
            string += "\n\tData type: "
            string += str(self.data.dtype)
        print string

    def load_dictionary(self, file_data_dict):
        """Parameters:
        -----------
        file_data_dict : dictionary
            A dictionary containing at least a 'data' keyword with an array of
            arbitrary dimensions. Additionally the dictionary can contain the
            following keys:
                axes: a dictionary that defines the axes (see the
                    AxesManager class)
                attributes: a dictionary which keywords are stored as
                    attributes of the signal class
                mapped_parameters: a dictionary containing a set of parameters
                    that will be stored as attributes of a Parameters class.
                    For some subclasses some particular parameters might be
                    mandatory.
                original_parameters: a dictionary that will be accesible in the
                    original_parameters attribute of the signal class and that
                    typically contains all the parameters that has been
                    imported from the original data file.

        """
        self.data = file_data_dict['data']
        if 'axes' not in file_data_dict:
            file_data_dict['axes'] = self._get_undefined_axes_list()
        self.axes_manager = AxesManager(
            file_data_dict['axes'])
        if not 'mapped_parameters' in file_data_dict:
            file_data_dict['mapped_parameters'] = {}
        if not 'original_parameters' in file_data_dict:
            file_data_dict['original_parameters'] = {}
        if 'attributes' in file_data_dict:
            for key, value in file_data_dict['attributes'].iteritems():
                if hasattr(self,key):
                    if isinstance(value,dict):
                        for k,v in value.iteritems():
                            eval('self.%s.__setattr__(k,v)'%key)
                    else:
                        self.__setattr__(key, value)
        self.original_parameters._load_dictionary(
            file_data_dict['original_parameters'])
        self.mapped_parameters._load_dictionary(
            file_data_dict['mapped_parameters'])
        if not hasattr(self.mapped_parameters,'title'):
            self.mapped_parameters.title = ''
        if not hasattr(self.mapped_parameters,'record_by'):
            self.mapped_parameters.record_by = self._default_record_by
        self.squeeze()
                
    def squeeze(self):
        """Remove single-dimensional entries from the shape of an array and the 
        axes.
        """
        self.data = self.data.squeeze()
        for axis in self.axes_manager.axes:
            if axis.size == 1:
                self.axes_manager.axes.remove(axis)
                for ax in self.axes_manager.axes:
                    ax.index_in_array -= 1
        self.data = self.data.squeeze()

    def _get_signal_dict(self):
        dic = {}
        dic['data'] = self.data.copy()
        dic['axes'] = self.axes_manager._get_axes_dicts()
        dic['mapped_parameters'] = \
        self.mapped_parameters.as_dictionary()
        dic['original_parameters'] = \
        self.original_parameters.as_dictionary()
        if hasattr(self,'learning_results'):
            dic['learning_results'] = self.learning_results.__dict__
        return dic

    def _get_undefined_axes_list(self):
        axes = []
        for i in xrange(len(self.data.shape)):
            axes.append({
                        'name': 'undefined',
                        'scale': 1.,
                        'offset': 0.,
                        'size': int(self.data.shape[i]),
                        'units': 'undefined',
                        'index_in_array': i, })
        return axes

    def __call__(self, axes_manager=None):
        if axes_manager is None:
            axes_manager = self.axes_manager
        return self.data.__getitem__(axes_manager._getitem_tuple)

    def _get_hse_1D_explorer(self, *args, **kwargs):
        islice = self.axes_manager.signal_axes[0].index_in_array
        inslice = self.axes_manager.navigation_axes[0].index_in_array
        if islice > inslice:
            return self.data.squeeze()
        else:
            return self.data.squeeze().T

    def _get_hse_2D_explorer(self, *args, **kwargs):
        islice = self.axes_manager.signal_axes[0].index_in_array
        data = np.nan_to_num(self.data).sum(islice)
        return data

    def _get_hie_explorer(self, *args, **kwargs):
        isslice = [self.axes_manager.signal_axes[0].index_in_array,
                   self.axes_manager.signal_axes[1].index_in_array]
        isslice.sort()
        data = self.data.sum(isslice[1]).sum(isslice[0])
        return data

    def _get_explorer(self, *args, **kwargs):
        nav_dim = self.axes_manager.navigation_dimension
        if self.axes_manager.signal_dimension == 1:
            if nav_dim == 1:
                return self._get_hse_1D_explorer(*args, **kwargs)
            elif nav_dim == 2:
                return self._get_hse_2D_explorer(*args, **kwargs)
            else:
                return None
        if self.axes_manager.signal_dimension == 2:
            if nav_dim == 1 or nav_dim == 2:
                return self._get_hie_explorer(*args, **kwargs)
            else:
                return None
        else:
            return None

    def plot(self, axes_manager=None):
        if self._plot is not None:
                try:
                    self._plot.close()
                except:
                    # If it was already closed it will raise an exception,
                    # but we want to carry on...
                    pass

        if axes_manager is None:
            axes_manager = self.axes_manager

        if axes_manager.signal_dimension == 1:
            # Hyperspectrum
            self._plot = mpl_hse.MPL_HyperSpectrum_Explorer()
            
        elif axes_manager.signal_dimension == 2:
            self._plot = mpl_hie.MPL_HyperImage_Explorer()
        else:
            raise ValueError('Plotting is not supported for this view')
        
        self._plot.axes_manager = axes_manager
        self._plot.signal_data_function = self.__call__
        if self.mapped_parameters.title:
            self._plot.signal_title = self.mapped_parameters.title
        elif self.tmp_parameters.has_item('filename'):
            self._plot.signal_title = self.tmp_parameters.filename
            

        # Navigator properties
        if self.axes_manager.navigation_axes:
            self._plot.navigator_data_function = self._get_explorer
        self._plot.plot()
            

    def plot_residual(self, axes_manager=None):
        """Plot the residual between original data and reconstructed data

        Requires you to have already run PCA or ICA, and to reconstruct data
        using either the get_decomposition_model or get_bss_model methods.
        """

        if hasattr(self, 'residual'):
            self.residual.plot(axes_manager)
        else:
            print "Object does not have any residual information.  Is it a \
reconstruction created using either get_decomposition_model or get_bss_model methods?"

    def save(self, filename=None, overwrite=None, extension=None,
             **kwds):
        """Saves the signal in the specified format.

        The function gets the format from the extension. You can use:
            - hdf5 for HDF5
            - rpl for Ripple (usefult to export to Digital Micrograph)
            - msa for EMSA/MSA single spectrum saving.
            - Many image formats such as png, tiff, jpeg...

        If no extension is provided the default file format as defined in the
        `preferences` is used.
        Please note that not all the formats supports saving datasets of
        arbitrary dimensions, e.g. msa only suports 1D data.
        
        Each format accepts a different set of parameters. For details 
        see the specific format documentation.

        Parameters
        ----------
        filename : str or None
            If None and tmp_parameters.filename and 
            tmp_paramters.folder are defined, the
            filename and extension will be taken from them.
        overwrite : None, bool
            If None, if the file exists it will query the user. If 
            True(False) it (does not) overwrites the file if it exists.
        extension : str
            The extension of the file that defines the file format, 
            e.g. 'rpl'. It overwrite the extension given in filename
            if any.
            
        """
        if filename is None:
            if (self.tmp_parameters.has_item('filename') and 
                self.tmp_parameters.has_item('folder')):
                filename = os.path.join(
                    self.tmp_parameters.folder,
                    self.tmp_parameters.filename + '.' +
                    self.tmp_parameters.extension)
            elif self.mapped_parameters.has_item('original_filename'):
                filename = self.mapped_parameters.original_filename
            else:
                raise ValueError('File name not defined')
        if extension is not None:
            basename, ext = os.path.splitext(filename)
            filename = basename + '.' + extension
        io.save(filename, self, overwrite=overwrite, **kwds)

    def _replot(self):
        if self._plot is not None:
            if self._plot.is_active() is True:
                self.plot()

    @auto_replot
    def get_dimensions_from_data(self):
        """Get the dimension parameters from the data_cube. Useful when the
        data_cube was externally modified, or when the SI was not loaded from
        a file
        """
        dc = self.data
        for axis in self.axes_manager.axes:
            axis.size = int(dc.shape[axis.index_in_array])
            print("%s size: %i" %
            (axis.name, dc.shape[axis.index_in_array]))

    def crop_in_pixels(self, axis, i1=None, i2=None, copy=True):
        """Crops the data in a given axis. The range is given in pixels
        axis : int
        i1 : int
            Start index
        i2 : int
            End index
        copy : bool
            If True makes a copy of the data, otherwise the cropping
            performs just a view.

        See also:
        ---------
        crop_in_units
        
        """
        axis = self._get_positive_axis_index_index(axis)
        if i1 is not None:
            new_offset = self.axes_manager.axes[axis].axis[i1]
        # We take a copy to guarantee the continuity of the data
        self.data = self.data[
        (slice(None),)*axis + (slice(i1, i2), Ellipsis)].copy()

        if i1 is not None:
            self.axes_manager.axes[axis].offset = new_offset
        self.get_dimensions_from_data()
        self.squeeze()

    def crop_in_units(self, axis, x1=None, x2=None, copy=True):
        """Crops the data in a given axis. The range is given in the units of
        the axis

        axis : int
        i1 : float
            Start value
        i2 : float
            End value
        copy : bool
            If True makes a copy of the data, otherwise the cropping
            performs just a view.

        See also:
        ---------
        crop_in_pixels

        """
        i1 = self.axes_manager.axes[axis].value2index(x1)
        i2 = self.axes_manager.axes[axis].value2index(x2)
        self.crop_in_pixels(axis, i1, i2,copy=copy)

    @auto_replot
    def roll_xy(self, n_x, n_y = 1):
        """Roll over the x axis n_x positions and n_y positions the former rows

        This method has the purpose of "fixing" a bug in the acquisition of the
        Orsay's microscopes and probably it does not have general interest

        Parameters
        ----------
        n_x : int
        n_y : int

        Note: Useful to correct the SI column storing bug in Marcel's
        acquisition routines.
        """
        self.data = np.roll(self.data, n_x, 0)
        self.data[:n_x, ...] = np.roll(self.data[:n_x, ...], n_y, 1)

    # TODO: After using this function the plotting does not work
    @auto_replot
    def swap_axis(self, axis1, axis2):
        """Swaps the axes

        Parameters
        ----------
        axis1 : positive int
        axis2 : positive int
        """
        self.data = self.data.swapaxes(axis1, axis2)
        c1 = self.axes_manager.axes[axis1]
        c2 = self.axes_manager.axes[axis2]
        c1.index_in_array, c2.index_in_array =  \
            c2.index_in_array, c1.index_in_array
        self.axes_manager.axes[axis1] = c2
        self.axes_manager.axes[axis2] = c1
        self.axes_manager.set_signal_dimension()

    def rebin(self, new_shape):
        """
        Rebins the data to the new shape

        Parameters
        ----------
        new_shape: tuple of ints
            The new shape must be a divisor of the original shape
        """
        factors = np.array(self.data.shape) / np.array(new_shape)
        self.data = utils.rebin(self.data, new_shape)
        for axis in self.axes_manager.axes:
            axis.scale *= factors[axis.index_in_array]
        self.get_dimensions_from_data()

    def split_in(self, axis, number_of_parts = None, steps = None):
        """Splits the data

        The split can be defined either by the `number_of_parts` or by the
        `steps` size.

        Parameters
        ----------
        number_of_parts : int or None
            Number of parts in which the SI will be splitted
        steps : int or None
            Size of the splitted parts
        axis : int
            The splitting axis

        Return
        ------
        tuple with the splitted signals
        """
        axis = self._get_positive_axis_index_index(axis)
        if number_of_parts is None and steps is None:
            if not self._splitting_steps:
                messages.warning_exit(
                "Please provide either number_of_parts or a steps list")
            else:
                steps = self._splitting_steps
                print "Splitting in ", steps
        elif number_of_parts is not None and steps is not None:
            print "Using the given steps list. number_of_parts dimissed"
        splitted = []
        shape = self.data.shape

        if steps is None:
            rounded = (shape[axis] - (shape[axis] % number_of_parts))
            step = rounded / number_of_parts
            cut_node = range(0, rounded+step, step)
        else:
            cut_node = np.array([0] + steps).cumsum()
        signal_dict = self._get_signal_dict()
        axes_dict = signal_dict['axes']
        for i in xrange(len(cut_node)-1):
            axes_dict[axis]['offset'] = \
                self.axes_manager.axes[axis].index2value(cut_node[i])
            data = self.data[
            (slice(None), ) * axis + (slice(cut_node[i], cut_node[i + 1]),
            Ellipsis)]
            signal_dict['data'] = data
            s = Signal(signal_dict)
            # TODO: When copying plotting does not work
#            s.axes = copy.deepcopy(self.axes_manager)
            s.get_dimensions_from_data()
            splitted.append(s)
        return splitted

    def unfold_if_multidim(self):
        """Unfold the datacube if it is >2D

        Returns
        -------

        Boolean. True if the data was unfolded by the function.
        """
        if len(self.axes_manager.axes)>2:
            print "Automatically unfolding the data"
            self.unfold()
            return True
        else:
            return False

    @auto_replot
    def _unfold(self, steady_axes, unfolded_axis):
        """Modify the shape of the data by specifying the axes the axes which
        dimension do not change and the axis over which the remaining axes will
        be unfolded

        Parameters
        ----------
        steady_axes : list
            The indexes of the axes which dimensions do not change
        unfolded_axis : int
            The index of the axis over which all the rest of the axes (except
            the steady axes) will be unfolded

        See also
        --------
        fold
        """

        # It doesn't make sense unfolding when dim < 3
        if len(self.data.squeeze().shape) < 3:
            return False

        # We need to store the original shape and coordinates to be used by
        # the fold function only if it has not been already stored by a
        # previous unfold
        if self._shape_before_unfolding is None:
            self._shape_before_unfolding = self.data.shape
            self._axes_manager_before_unfolding = self.axes_manager

        new_shape = [1] * len(self.data.shape)
        for index in steady_axes:
            new_shape[index] = self.data.shape[index]
        new_shape[unfolded_axis] = -1
        self.data = self.data.reshape(new_shape)
        self.axes_manager = self.axes_manager.deepcopy()
        i = 0
        uname = ''
        uunits = ''
        to_remove = []
        for axis, dim in zip(self.axes_manager.axes, new_shape):
            if dim == 1:
                uname += ',' + axis.name
                uunits = ',' + axis.units
                to_remove.append(axis)
            else:
                axis.index_in_array = i
                i += 1
        self.axes_manager.axes[unfolded_axis].name += uname
        self.axes_manager.axes[unfolded_axis].units += uunits
        self.axes_manager.axes[unfolded_axis].size = \
                                                self.data.shape[unfolded_axis]
        for axis in to_remove:
            self.axes_manager.axes.remove(axis)

        self.data = self.data.squeeze()

    def unfold(self):
        """Modifies the shape of the data by unfolding the signal and
        navigation dimensions separaterly

        """
        self.unfold_navigation_space()
        self.unfold_signal_space()

    def unfold_navigation_space(self):
        """Modify the shape of the data to obtain a navigation space of
        dimension 1
        """

        if self.axes_manager.navigation_dimension < 2:
            return False
        steady_axes = [
                        axis.index_in_array for axis in
                        self.axes_manager.signal_axes]
        unfolded_axis = self.axes_manager.navigation_axes[-1].index_in_array
        self._unfold(steady_axes, unfolded_axis)

    def unfold_signal_space(self):
        """Modify the shape of the data to obtain a signal space of
        dimension 1
        """
        if self.axes_manager.signal_dimension < 2:
            return False
        steady_axes = [
                        axis.index_in_array for axis in
                        self.axes_manager.navigation_axes]
        unfolded_axis = self.axes_manager.signal_axes[-1].index_in_array
        self._unfold(steady_axes, unfolded_axis)

    @auto_replot
    def fold(self):
        """If the signal was previously unfolded, folds it back"""
        if self._shape_before_unfolding is not None:
            self.data = self.data.reshape(self._shape_before_unfolding)
            self.axes_manager = self._axes_manager_before_unfolding
            self._shape_before_unfolding = None
            self._axes_manager_before_unfolding = None

    def _get_positive_axis_index_index(self, axis):
        if axis < 0:
            axis = len(self.data.shape) + axis
        return axis

    def iterate_axis(self, axis = -1, copy=True):
        # We make a copy to guarantee that the data in contiguous,
        # otherwise it will not return a view of the data
        if copy is True:
            self.data = self.data.copy()
        axis = self._get_positive_axis_index_index(axis)
        unfolded_axis = axis - 1
        new_shape = [1] * len(self.data.shape)
        new_shape[axis] = self.data.shape[axis]
        new_shape[unfolded_axis] = -1
        # Warning! if the data is not contigous it will make a copy!!
        data = self.data.reshape(new_shape)
        for i in xrange(data.shape[unfolded_axis]):
            getitem = [0] * len(data.shape)
            getitem[axis] = slice(None)
            getitem[unfolded_axis] = i
            yield(data[getitem])
            
    def _iterate_signal(self, copy=True):
        # We make a copy to guarantee that the data in contiguous,
        # otherwise it will not return a view of the data
        if copy is True:
            self.data = self.data.copy()
        axes = [axis.index_in_array for 
                axis in self.axes_manager.signal_axes]
        unfolded_axis = self.axes_manager.navigation_axes[-1].index_in_array
        new_shape = [1] * len(self.data.shape)
        for axis in axes:
            new_shape[axis] = self.data.shape[axis]
        new_shape[unfolded_axis] = -1
        # Warning! if the data is not contigous it will make a copy!!
        data = self.data.reshape(new_shape)
        for i in xrange(data.shape[unfolded_axis]):
            getitem = [0] * len(data.shape)
            for axis in axes:
                getitem[axis] = slice(None)
            getitem[unfolded_axis] = i
            yield(data[getitem])

    @auto_replot
    def sum(self, axis, return_signal=False):
        """Sum the data over the specify axis

        Parameters
        ----------
        axis : int
            The axis over which the operation will be performed
        return_signal : bool
            If False the operation will be performed on the current object. If
            True, the current object will not be modified and the operation
             will be performed in a new signal object that will be returned.

        Returns
        -------
        Depending on the value of the return_signal keyword, nothing or a
        signal instance

        See also
        --------
        sum_in_mask, mean

        Usage
        -----
        >>> import numpy as np
        >>> s = Signal({'data' : np.random.random((64,64,1024))})
        >>> s.data.shape
        (64,64,1024)
        >>> s.sum(-1)
        >>> s.data.shape
        (64,64)
        # If we just want to plot the result of the operation
        s.sum(-1, True).plot()
        """
        if return_signal is True:
            s = self.deepcopy()
        else:
            s = self
        s.data = s.data.sum(axis)
        s.axes_manager.axes.remove(s.axes_manager.axes[axis])
        for _axis in s.axes_manager.axes:
            if _axis.index_in_array > axis:
                _axis.index_in_array -= 1
        s.axes_manager.set_signal_dimension()
        if return_signal is True:
            return s
            
    def diff(self, axis, order=1, return_signal=False):
        """Differentiate the data over the specify axis

        Parameters
        ----------
        axis: int
            The axis over which the operation will be performed
        order: the order of the derivative
        return_signal : bool
            If False the operation will be performed on the current object. If
            True, the current object will not be modified and the operation
            will be performed in a new signal object that will be returned.

        Returns
        -------
        Depending on the value of the return_signal keyword, nothing or a
        signal instance

        See also
        --------
        mean, sum

        Usage
        -----
        >>> import numpy as np
        >>> s = Signal({'data' : np.random.random((64,64,1024))})
        >>> s.data.shape
        (64,64,1024)
        >>> s.diff(-1)
        >>> s.data.shape
        (64,64,1023)
        # If we just want to plot the result of the operation
        s.diff(-1, True).plot()
        """
        if return_signal is True:
            s = self.deepcopy()
        else:
            s = self
        s.data = np.diff(s.data,order,axis)
        axis = s.axes_manager.axes[axis]
        axis.offset = axis.axis[:2].mean()
        s.get_dimensions_from_data()
        if return_signal is True:
            return s

    def mean(self, axis, return_signal = False):
        """Average the data over the specify axis

        Parameters
        ----------
        axis : int
            The axis over which the operation will be performed
        return_signal : bool
            If False the operation will be performed on the current object. If
            True, the current object will not be modified and the operation
            will be performed in a new signal object that will be returned.

        Returns
        -------
        Depending on the value of the return_signal keyword, nothing or a
        signal instance

        See also
        --------
        sum_in_mask, mean

        Usage
        -----
        >>> import numpy as np
        >>> s = Signal({'data' : np.random.random((64,64,1024))})
        >>> s.data.shape
        (64,64,1024)
        >>> s.mean(-1)
        >>> s.data.shape
        (64,64)
        # If we just want to plot the result of the operation
        s.mean(-1, True).plot()
        """
        if return_signal is True:
            s = self.deepcopy()
        else:
            s = self
        s.data = s.data.mean(axis)
        s.axes_manager.axes.remove(s.axes_manager.axes[axis])
        for _axis in s.axes_manager.axes:
            if _axis.index_in_array > axis:
                _axis.index_in_array -= 1
        s.axes_manager.set_signal_dimension()
        if return_signal is True:
            return s

    def copy(self):
        return copy.copy(self)

    def deepcopy(self):
        s = copy.deepcopy(self)
        if self.data is not None:
            s.data = s.data.copy()
        return s
        
    def change_dtype(self, dtype):
        """Change the data type
        
        Parameters
        ----------

        dtype : str or dtype
            Typecode or data-type to which the array is cast.
            
        Example
        -------
        >>> import numpy as np
        >>> from hyperspy.signals.spectrum import Spectrum        ns = 
        ns.data = self.data.copy()
        >>> s = signals.Spectrum({'data' : np.array([1,2,3,4,5])})
        >>> s.data
        array([1, 2, 3, 4, 5])
        >>> s.change_dtype('float')
        >>> s.data
        array([ 1.,  2.,  3.,  4.,  5.])
        
        """
        
        self.data = self.data.astype(dtype)
        
    def _plot_factors_or_pchars(self, factors, comp_ids=None, 
                                calibrate=True, avg_char=False,
                                same_window=None, comp_label='PC', 
                                img_data=None,
                                plot_shifts=True, plot_char=4, 
                                cmap=plt.cm.gray, quiver_color='white',
                                vector_scale=1,
                                per_row=3,ax=None):
        """Plot components from PCA or ICA, or peak characteristics

        Parameters
        ----------

        comp_ids : None, int, or list of ints
            if None, returns maps of all components.
            if int, returns maps of components with ids from 0 to given int.
            if list of ints, returns maps of components with ids in given list.
        calibrate : bool
            if True, plots are calibrated according to the data in the axes
            manager.
        same_window : bool
            if True, plots each factor to the same window.  They are not scaled.
        comp_label : string, the label that is either the plot title (if plotting in
            separate windows) or the label in the legend (if plotting in the 
            same window)
        cmap : a matplotlib colormap
            The colormap used for factor images or
            any peak characteristic scatter map
            overlay.

        Parameters only valid for peak characteristics (or pk char factors):
        --------------------------------------------------------------------        

        img_data - 2D numpy array, 
            The array to overlay peak characteristics onto.  If None,
            defaults to the average image of your stack.

        plot_shifts - bool, default is True
            If true, plots a quiver (arrow) plot showing the shifts for each
            peak present in the component being plotted.

        plot_char - None or int
            If int, the id of the characteristic to plot as the colored 
            scatter plot.
            Possible components are:
               4: peak height
               5: peak orientation
               6: peak eccentricity

       quiver_color : any color recognized by matplotlib
           Determines the color of vectors drawn for 
           plotting peak shifts.

       vector_scale : integer or None
           Scales the quiver plot arrows.  The vector 
           is defined as one data unit along the X axis.  
           If shifts are small, set vector_scale so 
           that when they are multiplied by vector_scale, 
           they are on the scale of the image plot.
           If None, uses matplotlib's autoscaling.
               
        """
        if same_window is None:
            same_window = preferences.MachineLearning.same_window
        if comp_ids is None:
            comp_ids=xrange(factors.shape[1])

        elif not hasattr(comp_ids,'__iter__'):
            comp_ids=xrange(comp_ids)

        n=len(comp_ids)
        if same_window:
            rows=int(np.ceil(n/float(per_row)))

        if plot_char==4:
            cbar_label='Peak Height'
        elif plot_char==5:
            cbar_label='Peak Orientation'
        elif plot_char==6:
            cbar_label='Peak Eccentricity'
        else:
            cbar_label=None

        fig_list=[]

        if n<per_row: per_row=n

        if same_window and self.axes_manager.signal_dimension==2:
            f=plt.figure(figsize=(4*per_row,3*rows))
        else:
            f=plt.figure()
        for i in xrange(len(comp_ids)):
            if self.axes_manager.signal_dimension==1:
                if same_window:
                    ax=plt.gca()
                else:
                    if i>0:
                        f=plt.figure()
                    ax=f.add_subplot(111)
                ax=sigdraw._plot_1D_component(factors=factors,
                        idx=comp_ids[i],axes_manager=self.axes_manager,
                        ax=ax, calibrate=calibrate,
                        comp_label=comp_label,
                        same_window=same_window)
                if same_window:
                    plt.legend(ncol=factors.shape[1]//2, loc='best')
            elif self.axes_manager.signal_dimension==2:
                if same_window:
                    ax=f.add_subplot(rows,per_row,i+1)
                else:
                    if i>0:
                        f=plt.figure()
                    ax=f.add_subplot(111)

                sigdraw._plot_2D_component(factors=factors, 
                    idx=comp_ids[i], 
                    axes_manager=self.axes_manager,
                    calibrate=calibrate,ax=ax, 
                    cmap=cmap,comp_label=comp_label)
            if not same_window:
                fig_list.append(f)
        try:
            plt.tight_layout()
        except:
            pass
        if not same_window:
            return fig_list
        else:
            return f

    def _plot_loadings(self, loadings, comp_ids=None, calibrate=True,
                     same_window=None, comp_label=None, 
                     with_factors=False, factors=None,
                     cmap=plt.cm.gray, no_nans=False, per_row=3):
        if same_window is None:
            same_window = preferences.MachineLearning.same_window
        if comp_ids is None:
            comp_ids=xrange(loadings.shape[0])

        elif not hasattr(comp_ids,'__iter__'):
            comp_ids=xrange(comp_ids)

        n=len(comp_ids)
        if same_window:
            rows=int(np.ceil(n/float(per_row)))

        fig_list=[]

        if n<per_row: per_row=n

        if same_window and self.axes_manager.signal_dimension==2:
            f=plt.figure(figsize=(4*per_row,3*rows))
        else:
            f=plt.figure()

        for i in xrange(n):
            if self.axes_manager.navigation_dimension==1:
                if same_window:
                    ax=plt.gca()
                else:
                    if i>0:
                        f=plt.figure()
                    ax=f.add_subplot(111)
            elif self.axes_manager.navigation_dimension==2:
                if same_window:
                    ax=f.add_subplot(rows,per_row,i+1)
                else:
                    if i>0:
                        f=plt.figure()
                    ax=f.add_subplot(111)
            sigdraw._plot_loading(loadings,idx=comp_ids[i],
                                axes_manager=self.axes_manager,
                                no_nans=no_nans, calibrate=calibrate,
                                cmap=cmap,comp_label=comp_label,ax=ax,
                                same_window=same_window)
            if not same_window:
                fig_list.append(f)
        try:
            plt.tight_layout()
        except:
            pass
        if not same_window:
            if with_factors:
                return fig_list, self._plot_factors_or_pchars(factors, 
                                            comp_ids=comp_ids, 
                                            calibrate=calibrate,
                                            same_window=same_window, 
                                            comp_label=comp_label, 
                                            per_row=per_row)
            else:
                return fig_list
        else:
            if self.axes_manager.navigation_dimension==1:
                plt.legend(ncol=loadings.shape[0]//2, loc='best')
            if with_factors:
                return f, self._plot_factors_or_pchars(factors, 
                                            comp_ids=comp_ids, 
                                            calibrate=calibrate,
                                            same_window=same_window, 
                                            comp_label=comp_label, 
                                            per_row=per_row)
            else:
                return f

    def _export_factors(self,
                        factors,
                        folder=None,
                        comp_ids=None,
                        multiple_files=None,
                        save_figures=False,
                        save_figures_format='png',
                        factor_prefix=None,
                        factor_format=None,
                        comp_label=None,
                        cmap=plt.cm.gray,
                        plot_shifts=True,
                        plot_char=4,
                        img_data=None,
                        same_window=False,
                        calibrate=True,
                        quiver_color='white',
                        vector_scale=1,
                        no_nans=True, per_row=3):

        from hyperspy.signals.image import Image
        from hyperspy.signals.spectrum import Spectrum
        
        if multiple_files is None:
            multiple_files = preferences.MachineLearning.multiple_files
        
        if factor_format is None:
            factor_format = preferences.MachineLearning.\
                export_factors_default_file_format

        # Select the desired factors
        if comp_ids is None:
            comp_ids=xrange(factors.shape[1])
        elif not hasattr(comp_ids,'__iter__'):
            comp_ids=range(comp_ids)
        mask=np.zeros(factors.shape[1],dtype=np.bool)
        for idx in comp_ids:
            mask[idx]=1
        factors=factors[:,mask]

        if save_figures is True:
            plt.ioff()
            fac_plots=self._plot_factors_or_pchars(factors,
                                           comp_ids=comp_ids, 
                                           same_window=same_window,
                                           comp_label=comp_label, 
                                           img_data=img_data,
                                           plot_shifts=plot_shifts,
                                           plot_char=plot_char, 
                                           cmap=cmap,
                                           per_row=per_row,
                                           quiver_color=quiver_color,
                                           vector_scale=vector_scale)
            for idx in xrange(len(comp_ids)):
                filename = '%s_%02i.%s' % (factor_prefix, comp_ids[idx],
                save_figures_format)
                if folder is not None:
                    filename = os.path.join(folder, filename)
                ensure_directory(filename)
                fac_plots[idx].savefig(filename, save_figures_format,
                    dpi=600)
            plt.ion()
            
        elif multiple_files is False:
            if self.axes_manager.signal_dimension==2:
                # factor images
                axes_dicts=[]
                axes=self.axes_manager.signal_axes
                shape=(axes[1].size,axes[0].size)
                factor_data=np.rollaxis(
                        factors.reshape((shape[0],shape[1],-1)),2)
                axes_dicts.append(axes[0].get_axis_dictionary())
                axes_dicts.append(axes[1].get_axis_dictionary())
                axes_dicts.append({'name': 'factor_index',
                        'scale': 1.,
                        'offset': 0.,
                        'size': int(factors.shape[1]),
                        'units': 'factor',
                        'index_in_array': 0, })
                s=Image({'data':factor_data,
                         'axes':axes_dicts,
                         'mapped_parameters':{
                            'title':'%s from %s'%(factor_prefix,
                                self.mapped_parameters.title),
                            }})
            elif self.axes_manager.signal_dimension==1:
                axes=[]
                axes.append(
                self.axes_manager.signal_axes[0].get_axis_dictionary())
                axes[0]['index_in_array']=1
                  

                axes.append({
                    'name': 'factor_index',
                    'scale': 1.,
                    'offset': 0.,
                    'size': int(factors.shape[1]),
                    'units': 'factor',
                    'index_in_array': 0,
                        })
                s=Spectrum({'data' : factors.T,
                            'axes' : axes,
                            'mapped_parameters' : {
                            'title':'%s from %s'%(factor_prefix, 
                                self.mapped_parameters.title),}})
            filename = '%ss.%s' % (factor_prefix, factor_format)
            if folder is not None:
                filename = os.path.join(folder, filename)
            s.save(filename)
        else: # Separate files
            if self.axes_manager.signal_dimension == 1:
            
                axis_dict = self.axes_manager.signal_axes[0].\
                    get_axis_dictionary()
                axis_dict['index_in_array']=0
                for dim,index in zip(comp_ids,range(len(comp_ids))):
                    s=Spectrum({'data':factors[:,index],
                                'axes': [axis_dict,],
                                'mapped_parameters' : {
                            'title':'%s from %s'%(factor_prefix, 
                                self.mapped_parameters.title),}})
                    filename = '%s-%i.%s' % (factor_prefix, dim, factor_format)
                    if folder is not None:
                        filename = os.path.join(folder, filename)
                    s.save(filename)
                    
            if self.axes_manager.signal_dimension == 2:
                axes = self.axes_manager.signal_axes
                axes_dicts=[]
                axes_dicts.append(axes[0].get_axis_dictionary())
                axes_dicts.append(axes[1].get_axis_dictionary())
                axes_dicts[0]['index_in_array'] = 0
                axes_dicts[1]['index_in_array'] = 1
                
                factor_data = factors.reshape(
                    self.axes_manager.signal_shape + [-1,])
                
                for dim,index in zip(comp_ids,range(len(comp_ids))):
                    im = Image({
                                'data' : factor_data[...,index],
                                'axes' : axes_dicts,
                                'mapped_parameters' : {
                                'title' : '%s from %s' % (factor_prefix,
                                    self.mapped_parameters.title),
                                }})
                    filename = '%s-%i.%s' % (factor_prefix, dim, factor_format)
                    if folder is not None:
                        filename = os.path.join(folder, filename)
                    im.save(filename)

    def _export_loadings(self,
                         loadings,
                         folder=None,
                         comp_ids=None,
                         multiple_files=None,
                         loading_prefix=None,
                         loading_format=None,
                         save_figures_format = 'png',
                         comp_label=None,
                         cmap=plt.cm.gray,
                         save_figures = False,
                         same_window=False,
                         calibrate=True,
                         no_nans=True,
                         per_row=3):

        from hyperspy.signals.image import Image
        from hyperspy.signals.spectrum import Spectrum

        if multiple_files is None:
            multiple_files = preferences.MachineLearning.multiple_files
        
        if loading_format is None:
            loading_format = preferences.MachineLearning.\
                export_loadings_default_file_format

        if comp_ids is None:
            comp_ids=range(loadings.shape[0])
        elif not hasattr(comp_ids,'__iter__'):
            comp_ids=range(comp_ids)
        mask=np.zeros(loadings.shape[0],dtype=np.bool)
        for idx in comp_ids:
            mask[idx]=1
        loadings=loadings[mask]

        if save_figures is True:
            plt.ioff()
            sc_plots=self._plot_loadings(loadings, comp_ids=comp_ids, 
                                       calibrate=calibrate,
                                       same_window=same_window, 
                                       comp_label=comp_label,
                                       cmap=cmap, no_nans=no_nans,
                                       per_row=per_row)
            for idx in xrange(len(comp_ids)):
                filename = '%s_%02i.%s'%(loading_prefix, comp_ids[idx],
                                                  save_figures_format)
                if folder is not None:
                    filename = os.path.join(folder, filename)
                ensure_directory(filename)
                sc_plots[idx].savefig(filename, dpi=600)
            plt.ion()
        elif multiple_files is False:
            if self.axes_manager.navigation_dimension==2:
                axes_dicts=[]
                axes=self.axes_manager.navigation_axes
                shape=(axes[1].size,axes[0].size)
                loading_data=loadings.reshape((-1,shape[0],shape[1]))
                axes_dicts.append(axes[0].get_axis_dictionary())
                axes_dicts[0]['index_in_array']=1
                axes_dicts.append(axes[1].get_axis_dictionary())
                axes_dicts[1]['index_in_array']=2
                axes_dicts.append({'name': 'loading_index',
                        'scale': 1.,
                        'offset': 0.,
                        'size': int(loadings.shape[0]),
                        'units': 'factor',
                        'index_in_array': 0, })
                s=Image({'data':loading_data,
                         'axes':axes_dicts,
                         'mapped_parameters':{
                            'title':'%s from %s'%(loading_prefix, 
                                self.mapped_parameters.title),
                            }})
            elif self.axes_manager.navigation_dimension==1:
                cal_axis=self.axes_manager.navigation_axes[0].\
                    get_axis_dictionary()
                cal_axis['index_in_array']=1
                axes=[]
                axes.append({'name': 'loading_index',
                        'scale': 1.,
                        'offset': 0.,
                        'size': int(loadings.shape[0]),
                        'units': 'comp_id',
                        'index_in_array': 0, })
                axes.append(cal_axis)
                s=Image({'data':loadings,
                            'axes':axes,
                            'mapped_parameters':{
                            'title':'%s from %s'%(loading_prefix,
                                self.mapped_parameters.title),}})
            filename = '%ss.%s' % (loading_prefix, loading_format)
            if folder is not None:
                filename = os.path.join(folder, filename)
            s.save(filename)
        else: # Separate files
            if self.axes_manager.navigation_dimension == 1:
                axis_dict = self.axes_manager.navigation_axes[0].\
                    get_axis_dictionary()
                axis_dict['index_in_array']=0
                for dim,index in zip(comp_ids,range(len(comp_ids))):
                    s=Spectrum({'data':loadings[index],
                                'axes': [axis_dict,]})
                    filename = '%s-%i.%s' % (loading_prefix, dim,loading_format)
                    if folder is not None:
                        filename = os.path.join(folder, filename)
                    s.save(filename)
            elif self.axes_manager.navigation_dimension == 2:
                axes_dicts=[]
                axes=self.axes_manager.navigation_axes
                shape=(axes[0].size, axes[1].size)
                loading_data=loadings.reshape((-1,shape[0],shape[1]))
                axes_dicts.append(axes[0].get_axis_dictionary())
                axes_dicts[0]['index_in_array']=0
                axes_dicts.append(axes[1].get_axis_dictionary())
                axes_dicts[1]['index_in_array']=1
                for dim,index in zip(comp_ids,range(len(comp_ids))):
                    s=Image({'data':loading_data[index,...],
                             'axes':axes_dicts,
                             'mapped_parameters':{
                                'title':'%s from %s'%(
                                    loading_prefix, 
                                    self.mapped_parameters.title),
                                }})
                    filename = '%s-%i.%s' % (loading_prefix, dim, loading_format)
                    if folder is not None:
                        filename = os.path.join(folder, filename)
                    s.save(filename)

    def plot_decomposition_factors(self,comp_ids=None, calibrate=True,
                        same_window=None, comp_label='Decomposition factor', 
                        per_row=3):
        """Plot factors from a decomposition

        Parameters
        ----------

        comp_ids : None, int, or list of ints
            if None, returns maps of all components.
            if int, returns maps of components with ids from 0 to given int.
            if list of ints, returns maps of components with ids in given list.

        calibrate : bool
            if True, calibrates plots where calibration is available from
            the axes_manager.  If False, plots are in pixels/channels.

        same_window : bool
            if True, plots each factor to the same window.  They are not scaled.
        
        comp_label : string, the label that is either the plot title (if plotting in
            separate windows) or the label in the legend (if plotting in the 
            same window)

        cmap : The colormap used for the factor image, or for peak 
            characteristics, the colormap used for the scatter plot of
            some peak characteristic.
        
        per_row : int, the number of plots in each row, when the same_window
            parameter is True.
        """
        if same_window is None:
            same_window = preferences.MachineLearning.same_window
        factors=self.learning_results.factors
        if comp_ids is None:
            comp_ids = self.learning_results.output_dimension
            
        return self._plot_factors_or_pchars(factors, 
                                            comp_ids=comp_ids, 
                                            calibrate=calibrate,
                                            same_window=same_window, 
                                            comp_label=comp_label, 
                                            per_row=per_row)

    def plot_bss_factors(self,comp_ids=None, calibrate=True,
                        same_window=None, comp_label='BSS factor',
                        per_row=3):
        """Plot factors from blind source separation results.

        Parameters
        ----------

        comp_ids : None, int, or list of ints
            if None, returns maps of all components.
            if int, returns maps of components with ids from 0 to given int.
            if list of ints, returns maps of components with ids in given list.

        calibrate : bool
            if True, calibrates plots where calibration is available from
            the axes_manager.  If False, plots are in pixels/channels.

        same_window : bool
            if True, plots each factor to the same window.  They are not scaled.
        
        comp_label : string, the label that is either the plot title (if plotting in
            separate windows) or the label in the legend (if plotting in the 
            same window)

        cmap : The colormap used for the factor image, or for peak 
            characteristics, the colormap used for the scatter plot of
            some peak characteristic.
        
        per_row : int, the number of plots in each row, when the same_window
            parameter is True.
        """
        if same_window is None:
            same_window = preferences.MachineLearning.same_window
        factors=self.learning_results.bss_factors
        return self._plot_factors_or_pchars(factors, 
                                            comp_ids=comp_ids, 
                                            calibrate=calibrate,
                                            same_window=same_window, 
                                            comp_label=comp_label, 
                                            per_row=per_row)

    def plot_decomposition_loadings(self, comp_ids=None, calibrate=True,
                       same_window=None, comp_label='Decomposition loading', 
                       with_factors=False, cmap=plt.cm.gray, 
                       no_nans=False,per_row=3):
        """Plot loadings from PCA

        Parameters
        ----------

        comp_ids : None, int, or list of ints
            if None, returns maps of all components.
            if int, returns maps of components with ids from 0 to given int.
            if list of ints, returns maps of components with ids in given list.

        calibrate : bool
            if True, calibrates plots where calibration is available from
            the axes_manager.  If False, plots are in pixels/channels.

        same_window : bool
            if True, plots each factor to the same window.  They are not scaled.
        
        comp_label : string, 
            The label that is either the plot title (if plotting in
            separate windows) or the label in the legend (if plotting in the 
            same window)

        with_factors : bool
            If True, also returns figure(s) with the factors for the
            given comp_ids.

        cmap : matplotlib colormap
            The colormap used for the factor image, or for peak 
            characteristics, the colormap used for the scatter plot of
            some peak characteristic.
        
        no_nans : bool
            If True, removes NaN's from the loading plots.

        per_row : int 
            the number of plots in each row, when the same_window
            parameter is True.
        """
        if same_window is None:
            same_window = preferences.MachineLearning.same_window
        loadings=self.learning_results.loadings.T
        if with_factors:
            factors=self.learning_results.factors
        else:
            factors=None
        
        if comp_ids is None:
            comp_ids = self.learning_results.output_dimension
        return self._plot_loadings(loadings, comp_ids=comp_ids, 
                                 with_factors=with_factors, factors=factors,
                                 same_window=same_window, comp_label=comp_label,
                                 cmap=cmap, no_nans=no_nans,per_row=per_row)

    def plot_bss_loadings(self, comp_ids=None, calibrate=True,
                       same_window=None, comp_label='BSS loading', 
                       with_factors=False, cmap=plt.cm.gray, 
                       no_nans=False,per_row=3):
        """Plot loadings from ICA

        Parameters
        ----------

        comp_ids : None, int, or list of ints
            if None, returns maps of all components.
            if int, returns maps of components with ids from 0 to given int.
            if list of ints, returns maps of components with ids in given list.

        calibrate : bool
            if True, calibrates plots where calibration is available from
            the axes_manager.  If False, plots are in pixels/channels.

        same_window : bool
            if True, plots each factor to the same window.  They are not scaled.
        
        comp_label : string, 
            The label that is either the plot title (if plotting in
            separate windows) or the label in the legend (if plotting in the 
            same window)

        with_factors : bool
            If True, also returns figure(s) with the factors for the
            given comp_ids.

        cmap : matplotlib colormap
            The colormap used for the factor image, or for peak 
            characteristics, the colormap used for the scatter plot of
            some peak characteristic.
        
        no_nans : bool
            If True, removes NaN's from the loading plots.

        per_row : int 
            the number of plots in each row, when the same_window
            parameter is True.
        """
        if same_window is None:
            same_window = preferences.MachineLearning.same_window
        loadings=self.learning_results.bss_loadings.T
        if with_factors:
            factors=self.learning_results.bss_factors
        else: factors=None
        return self._plot_loadings(loadings, comp_ids=comp_ids, 
                                 with_factors=with_factors, factors=factors,
                                 same_window=same_window, comp_label=comp_label,
                                 cmap=cmap, no_nans=no_nans,per_row=per_row)

    def export_decomposition_results(self, comp_ids=None,
                                     folder=None,
                                     calibrate=True,
                                     factor_prefix='factor',
                                     factor_format=None,
                                     loading_prefix='loading',
                                     loading_format=None, 
                                     comp_label=None,
                                     cmap=plt.cm.gray,
                                     same_window=False,
                                     multiple_files=None,
                                     no_nans=True,
                                     per_row=3,
                                     save_figures=False,
                                     save_figures_format ='png'):
        """Export results from a decomposition to any of the supported formats.

        Parameters
        ----------
        comp_ids : None, int, or list of ints
            if None, returns all components/loadings.
            if int, returns components/loadings with ids from 0 to given int.
            if list of ints, returns components/loadings with ids in given list.
        folder : str or None
            The path to the folder where the file will be saved. If `None` the
            current folder is used by default.
        factor_prefix : string
            The prefix that any exported filenames for factors/components 
            begin with
        factor_format : string
            The extension of the format that you wish to save to.
        loading_prefix : string
            The prefix that any exported filenames for factors/components 
            begin with
        loading_format : string
            The extension of the format that you wish to save to.  Determines
            the kind of output.
                - For image formats (tif, png, jpg, etc.), plots are created 
                  using the plotting flags as below, and saved at 600 dpi.
                  One plot per loading is saved.
                - For multidimensional formats (rpl, hdf5), arrays are saved
                  in single files.  All loadings are contained in the one
                  file.
                - For spectral formats (msa), each loading is saved to a
                  separate file.
        multiple_files : Bool
            If True, on exporting a file per factor and per loading will be 
            created. Otherwise only two files will be created, one for the
            factors and another for the loadings. The default value can be
            chosen in the preferences.
        save_figures : Bool
            If True the same figures that are obtained when using the plot 
            methods will be saved with 600 dpi resolution

        Plotting options (for save_figures = True ONLY)
        ----------------------------------------------

        calibrate : bool
            if True, calibrates plots where calibration is available from
            the axes_manager.  If False, plots are in pixels/channels.
        same_window : bool
            if True, plots each factor to the same window.
        comp_label : string, the label that is either the plot title 
            (if plotting in separate windows) or the label in the legend 
            (if plotting in the same window)
        cmap : The colormap used for the factor image, or for peak 
            characteristics, the colormap used for the scatter plot of
            some peak characteristic.
        per_row : int, the number of plots in each row, when the same_window
            parameter is True.
        save_figures_format : str
            The image format extension.
            
        """
        
        factors=self.learning_results.factors
        loadings=self.learning_results.loadings.T
        self._export_factors(factors, folder=folder,comp_ids=comp_ids,
                             calibrate=calibrate, multiple_files=multiple_files,
                             factor_prefix=factor_prefix,
                             factor_format=factor_format,
                             comp_label=comp_label, save_figures = save_figures,
                             cmap=cmap,
                             no_nans=no_nans,
                             same_window=same_window,
                             per_row=per_row,
                             save_figures_format=save_figures_format)
        self._export_loadings(loadings,comp_ids=comp_ids,folder=folder,
                            calibrate=calibrate, multiple_files=multiple_files,
                            loading_prefix=loading_prefix,
                            loading_format=loading_format,
                            comp_label=comp_label,
                            cmap=cmap, save_figures = save_figures,
                            same_window=same_window,
                            no_nans=no_nans,
                            per_row=per_row)

    def export_bss_results(self,
                           comp_ids=None,
                           folder=None,
                           calibrate=True,
                           multiple_files=None,
                           save_figures=False,
                           factor_prefix='bss_factor',
                           factor_format=None,
                           loading_prefix='bss_loading',
                           loading_format=None, 
                           comp_label=None, cmap=plt.cm.gray,
                           same_window=False,
                           no_nans=True,
                           per_row=3,
                           save_figures_format='png'):
        """Export results from ICA to any of the supported formats.

        Parameters
        ----------
        comp_ids : None, int, or list of ints
            if None, returns all components/loadings.
            if int, returns components/loadings with ids from 0 to given int.
            if list of ints, returns components/loadings with ids in given list.
        folder : str or None
            The path to the folder where the file will be saved. If `None` the
            current folder is used by default.
        factor_prefix : string
            The prefix that any exported filenames for factors/components 
            begin with
        factor_format : string
            The extension of the format that you wish to save to.  Determines
            the kind of output.
                - For image formats (tif, png, jpg, etc.), plots are created 
                  using the plotting flags as below, and saved at 600 dpi.
                  One plot per factor is saved.
                - For multidimensional formats (rpl, hdf5), arrays are saved
                  in single files.  All factors are contained in the one
                  file.
                - For spectral formats (msa), each factor is saved to a
                  separate file.
                
        loading_prefix : string
            The prefix that any exported filenames for factors/components 
            begin with
        loading_format : string
            The extension of the format that you wish to save to.
        multiple_files : Bool
            If True, on exporting a file per factor and per loading will be 
            created. Otherwise only two files will be created, one for the
            factors and another for the loadings. The default value can be
            chosen in the preferences.
        save_figures : Bool
            If True the same figures that are obtained when using the plot 
            methods will be saved with 600 dpi resolution

        Plotting options (for save_figures = True ONLY)
        ----------------------------------------------
        calibrate : bool
            if True, calibrates plots where calibration is available from
            the axes_manager.  If False, plots are in pixels/channels.
        same_window : bool
            if True, plots each factor to the same window.
        comp_label : string
            the label that is either the plot title (if plotting in
            separate windows) or the label in the legend (if plotting in the 
            same window)
        cmap : The colormap used for the factor image, or for peak 
            characteristics, the colormap used for the scatter plot of
            some peak characteristic.
        per_row : int, the number of plots in each row, when the same_window
            parameter is True.
        save_figures_format : str
            The image format extension.
            
        """
        
        factors=self.learning_results.bss_factors
        loadings=self.learning_results.bss_loadings.T
        self._export_factors(factors,
                             folder=folder,
                             comp_ids=comp_ids,
                             calibrate=calibrate,
                             multiple_files=multiple_files,
                             factor_prefix=factor_prefix,
                             factor_format=factor_format,
                             comp_label=comp_label,
                             save_figures=save_figures,
                             cmap=cmap,
                             no_nans=no_nans,
                             same_window=same_window,
                             per_row=per_row,
                             save_figures_format=save_figures_format)
                             
        self._export_loadings(loadings,
                              comp_ids=comp_ids,
                              folder=folder,
                              calibrate=calibrate, 
                              multiple_files=multiple_files,
                              loading_prefix=loading_prefix,
                              loading_format=loading_format,
                              comp_label=comp_label,
                              cmap=cmap,
                              save_figures=save_figures,
                              same_window=same_window, 
                              no_nans=no_nans,
                              per_row=per_row,
                              save_figures_format=save_figures_format)

   
#    def sum_in_mask(self, mask):
#        """Returns the result of summing all the spectra in the mask.
#
#        Parameters
#        ----------
#        mask : boolean numpy array
#
#        Returns
#        -------
#        Spectrum
#        """
#        dc = self.data_cube.copy()
#        mask3D = mask.reshape([1,] + list(mask.shape)) * np.ones(dc.shape)
#        dc = (mask3D*dc).sum(1).sum(1) / mask.sum()
#        s = Spectrum()
#        s.data_cube = dc.reshape((-1,1,1))
#        s.get_dimensions_from_cube()
#        utils.copy_energy_calibration(self,s)
#        return s
#
#    def mean(self, axis):
#        """Average the SI over the given axis
#
#        Parameters
#        ----------
#        axis : int
#        """
#        dc = self.data_cube
#        dc = dc.mean(axis)
#        dc = dc.reshape(list(dc.shape) + [1,])
#        self.data_cube = dc
#        self.get_dimensions_from_cube()
#
#    def roll(self, axis = 2, shift = 1):
#        """Roll the SI. see numpy.roll
#
#        Parameters
#        ----------
#        axis : int
#        shift : int
#        """
#        self.data_cube = np.roll(self.data_cube, shift, axis)
#        self._replot()
#

#
#    def get_calibration_from(self, s):
#        """Copy the calibration from another Spectrum instance
#        Parameters
#        ----------
#        s : spectrum instance
#        """
#        utils.copy_energy_calibration(s, self)
#
    def estimate_variance(self, dc = None, gaussian_noise_var = None):
        """Variance estimation supposing Poissonian noise

        Parameters
        ----------
        dc : None or numpy array
            If None the SI is used to estimate its variance. Otherwise, the
            provided array will be used.
        Note
        ----
        The gain_factor and gain_offset from the aquisition parameters are used
        """
        gain_factor = 1
        gain_offset = 0
        correlation_factor = 1
        if not self.mapped_parameters.has_item("Variance_estimation"):
            print("No Variance estimation parameters found in mapped"
                  " parameters. The variance will be estimated supposing "
                  "perfect poissonian noise")
        if self.mapped_parameters.has_item('Variance_estimation.gain_factor'):
            gain_factor = self.mapped_parameters.Variance_estimation.gain_factor
        if self.mapped_parameters.has_item('Variance_estimation.gain_offset'):
            gain_offset = self.mapped_parameters.Variance_estimation.gain_offset
        if self.mapped_parameters.has_item(
            'Variance_estimation.correlation_factor'):
            correlation_factor = \
                self.mapped_parameters.Variance_estimation.correlation_factor
        print "Gain factor = ", gain_factor
        print "Gain offset = ", gain_offset
        print "Correlation factor = ", correlation_factor
        if dc is None:
            dc = self.data
        self.variance = dc * gain_factor + gain_offset
        if self.variance.min() < 0:
            if gain_offset == 0 and gaussian_noise_var is None:
                print "The variance estimation results in negative values"
                print "Maybe the gain_offset is wrong?"
                self.variance = None
                return
            elif gaussian_noise_var is None:
                print "Clipping the variance to the gain_offset value"
                minimum = 0 if gain_offset < 0 else gain_offset
                self.variance = np.clip(self.variance, minimum,
                np.Inf)
            else:
                print "Clipping the variance to the gaussian_noise_var"
                self.variance = np.clip(self.variance, gaussian_noise_var,
                np.Inf)
                
    def get_current_signal(self):
        data = self.data
        self.data = None
        cs = self.deepcopy()
        self.data = data
        del data
        cs.data = self()
        for axis in cs.axes_manager.navigation_axes:
            cs.axes_manager.axes.remove(axis)
            cs.axes_manager.set_signal_dimension()
        if cs.tmp_parameters.has_item('filename'):
            basename = cs.tmp_parameters.filename
            ext = cs.tmp_parameters.extension
            cs.tmp_parameters.filename = (basename + '_' +
                    str(self.axes_manager.coordinates) + '.' + ext)
        cs.mapped_parameters.title = (cs.mapped_parameters.title +
                    ' ' + str(self.axes_manager.coordinates))
        return cs
        
    def _get_navigation_signal(self):
        if self.axes_manager.navigation_dimension == 0:
            return None
        elif self.axes_manager.navigation_dimension == 1:
            from hyperspy.signals.spectrum import Spectrum
            s = Spectrum({
                'data' : np.zeros(
                    self.axes_manager.navigation_shape),
                'axes' : self.axes_manager._get_navigation_axes_dicts()})
        elif self.axes_manager.navigation_dimension == 2:
            from hyperspy.signals.image import Image
            s = Image({
                'data' : np.zeros(
                    self.axes_manager.navigation_shape),
                'axes' : self.axes_manager._get_navigation_axes_dicts()})
        else:
            s = Signal({
                'data' : np.zeros(
                    self.axes_manager.navigation_shape),
                'axes' : self.axes_manager._get_navigation_axes_dicts()})
        return s
                
        
    def __iter__(self):
        return self
        
    def next(self):
        self.axes_manager.next()
        return self.get_current_signal()
        
    def __len__(self):
        if self.axes_manager.navigation_size == 0:
            return 1
        else:
            return self.axes_manager.navigation_size
        

#
#    def sum_every_n(self,n):
#        """Bin a line spectrum
#
#        Parameters
#        ----------
#        step : float
#            binning size
#
#        Returns
#        -------
#        Binned line spectrum
#
#        See also
#        --------
#        sum_every
#        """
#        dc = self.data_cube
#        if dc.shape[1] % n != 0:
#            messages.warning_exit(
#            "n is not a divisor of the size of the line spectrum\n"
#            "Try giving a different n or using sum_every instead")
#        size_list = np.zeros((dc.shape[1] / n))
#        size_list[:] = n
#        return self.sum_every(size_list)
#
#    def sum_every(self,size_list):
#        """Sum a line spectrum intervals given in a list and return the
#        resulting SI
#
#        Parameters
#        ----------
#        size_list : list of floats
#            A list of the size of each interval to sum.
#
#        Returns
#        -------
#        SI
#
#        See also
#        --------
#        sum_every_n
#        """
#        dc = self.data_cube
#        dc_shape = self.data_cube.shape
#        if np.sum(size_list) != dc.shape[1]:
#            messages.warning_exit(
#            "The sum of the elements of the size list is not equal to the size"
#            " of the line spectrum")
#        new_dc = np.zeros((dc_shape[0], len(size_list), 1))
#        ch = 0
#        for i in xrange(len(size_list)):
#            new_dc[:,i,0] = dc[:,ch:ch + size_list[i], 0].sum(1)
#            ch += size_list[i]
#        sp = Spectrum()
#        sp.data_cube = new_dc
#        sp.get_dimensions_from_cube()
#        return sp



