# -*- coding: utf-8 -*-
"""
Created on Wed Oct 06 09:48:42 2010

"""
import types
import copy

import numpy as np
import enthought.traits.api as t
import enthought.traits.ui.api as tui 

from eelslab import messages
from eelslab.axes import AxesManager
from eelslab import file_io
from eelslab.drawing import mpl_hie, mpl_hse
from eelslab import utils
from eelslab.mva.mva import MVA, MVA_Results

class Parameters(t.HasTraits,object):
    """A class to comfortably access some parameters as attributes"""
    name = t.Str("UnnamedFile")

    def __init__(self, dictionary={}):
        super(Parameters, self).__init__()
        self.load_dictionary(dictionary)
        
    def load_dictionary(self, dictionary):
        for key, value in dictionary.iteritems():
            self.__setattr__(key, value)
 
    def print_items(self):
        """Prints only the attributes that are not methods"""
        for item in self.__dict__.items():
            if type(item) != types.MethodType:
                print("%s = %s") % item
                
    def _get_parameters_dictionary(self):
        par_dict = {}
        for item in self.__dict__.items():
            if type(item) != types.MethodType:
                par_dict.__setitem__(*item)
        return par_dict

class Signal(t.HasTraits, MVA):
    data = t.Any()
    axes_manager = t.Instance(AxesManager)
    original_parameters = t.Dict
    mapped_parameters = t.Instance(Parameters)
    physical_property = t.Str()
    
    def __init__(self, file_data_dict=None,*args,**kw):
        """All data interaction is made through this class or its subclasses
            
        
        Parameters:
        -----------
        dictionary : dictionary
           see load_dictionary for the format
        """    
        super(Signal, self).__init__()
        self.mapped_parameters=Parameters()
        if type(file_data_dict).__name__ == "dict":
            self.load_dictionary(file_data_dict)
        self._plot = None
        self.mva_results=MVA_Results()
        self._shape_before_unfolding = None
        
    def load_dictionary(self, file_data_dict):
        """Parameters:
        -----------
        file_data_dict : dictionary
            A dictionary containing at least a 'data' keyword with an array of 
            arbitrary dimensions. Additionally the dictionary can contain the 
            following keys:
                axes: a dictionary that defines the axes (see the 
                    AxesManager class)
                attributes: a dictionary which keywords are stored as attributes 
                of the signal class
                mapped_parameters: a dictionary containing a set of parameters that 
                    will be stored as attributes of a Parameters class. 
                    For some subclasses some particular parameters might be 
                    mandatory.
                original_parameters: a dictionary that will be accesible in the 
                    original_parameters attribute of the signal class and that 
                    typically contains all the parameters that has been imported
                    from the original data file.
        """
        self.data = file_data_dict['data']
        if not file_data_dict.has_key('axes'):
            file_data_dict['axes'] = self._get_undefined_axes_list()
        self.axes_manager = AxesManager(
            file_data_dict['axes'])
        if not file_data_dict.has_key('mapped_parameters'):
            file_data_dict['mapped_parameters'] = {}
        if not file_data_dict.has_key('original_parameters'):
            file_data_dict['original_parameters'] = {}
        if file_data_dict.has_key('attributes'):
            for key, value in file_data_dict['attributes'].iteritems():
                self.__setattr__(key, value)
        self.original_parameters = file_data_dict['original_parameters']
        self.mapped_parameters.load_dictionary(
            file_data_dict['mapped_parameters'])
            
    def _get_signal_dict(self):
        dic = {}
        dic['data'] = self.data.copy()
        dic['axes'] = self.axes_manager._get_axes_dicts()
        dic['mapped_parameters'] = \
        self.mapped_parameters._get_parameters_dictionary()
        dic['original_parameters'] = \
        self.original_parameters
        return dic
        
    def _get_undefined_axes_list(self):
        axes = []
        for i in xrange(len(self.data.shape)):
            axes.append({   'name' : 'undefined',
                            'scale' : 1.,
                            'offset' : 0.,
                            'size' : int(self.data.shape[i]),
                            'units' : 'undefined',
                            'index_in_array' : i,})
        return axes
                
    def __call__(self, axes_manager = None):
        if axes_manager is None:
            axes_manager = self.axes_manager
        return self.data.__getitem__(axes_manager._getitem_tuple)
        
    def _get_hse_1D_explorer(self, *args, **kwargs):
        islice = self.axes_manager._slicing_axes[0].index_in_array
        inslice = self.axes_manager._non_slicing_axes[0].index_in_array
        if islice > inslice:
            return self.data.squeeze()
        else:
            return self.data.squeeze().T
            
    def _get_hse_2D_explorer(self, *args, **kwargs):
        islice = self.axes_manager._slicing_axes[0].index_in_array
        data = self.data.sum(islice)
        return data
    
    def _get_hie_explorer(self, *args, **kwargs):
        isslice = [self.axes_manager._slicing_axes[0].index_in_array, 
                   self.axes_manager._slicing_axes[1].index_in_array]
        isslice.sort()
        data = self.data.sum(isslice[1]).sum(isslice[0])
        return data
         
    def _get_explorer(self, *args, **kwargs):
        nav_dim = self.axes_manager.navigation_dim
        if self.axes_manager.output_dim == 1:
            if nav_dim == 1:
                return self._get_hse_1D_explorer(*args, **kwargs)
            elif nav_dim == 2:
                return self._get_hse_2D_explorer(*args, **kwargs)
            else:
                return None
        if self.axes_manager.output_dim == 2:
            if nav_dim == 1 or nav_dim == 2:
                return self._get_hie_explorer(*args, **kwargs)
            else:
                return None
        else:
            return None
            
    def plot(self, axes_manager = None):
        if self._plot is not None:
                try:
                    self._plot.close()
                except:
                    # If it was already closed it will raise an exception,
                    # but we want to carry on...
                    pass
                
        if axes_manager is None:
            axes_manager = self.axes_manager
            
        if axes_manager.output_dim == 1:
            # Hyperspectrum
                            
            self._plot = mpl_hse.MPL_HyperSpectrum_Explorer()
            self._plot.spectrum_data_function = self.__call__
            self._plot.spectrum_title = self.mapped_parameters.name
            self._plot.xlabel = '%s (%s)' % (
                self.axes_manager._slicing_axes[0].name, 
                self.axes_manager._slicing_axes[0].units)
            self._plot.ylabel = 'Intensity'
            self._plot.axes_manager = axes_manager
            self._plot.axis = self.axes_manager._slicing_axes[0].axis
            
            # Image properties
            if self.axes_manager._non_slicing_axes:
                self._plot.image_data_function = self._get_explorer
                self._plot.image_title = ''
                self._plot.pixel_size = \
                self.axes_manager._non_slicing_axes[0].scale
                self._plot.pixel_units = \
                self.axes_manager._non_slicing_axes[0].units
            self._plot.plot()
            
        elif axes_manager.output_dim == 2:
            
            # Mike's playground with new plotting toolkits - needs to be a branch.
            """
            if len(self.data.shape)==2:
                from drawing.guiqwt_hie import image_plot_2D
                image_plot_2D(self)
            
            import drawing.chaco_hie
            self._plot = drawing.chaco_hie.Chaco_HyperImage_Explorer(self)
            self._plot.configure_traits()
            """
            self._plot = mpl_hie.MPL_HyperImage_Explorer()
            self._plot.image_data_function = self.__call__
            self._plot.navigator_data_function = self._get_explorer
            self._plot.axes_manager = axes_manager
            self._plot.plot()
            
        else:
            messages.warning_exit('Plotting is not supported for this view')
        
    traits_view = tui.View(
        tui.Item('name'),
        tui.Item('physical_property'),
        tui.Item('units'),
        tui.Item('offset'),
        tui.Item('scale'),)
    
    def save(self, filename, only_view = False, **kwds):
        """Saves the signal in the specified format.
        
        The function gets the format from the extension. You can use:
            - hdf5 for HDF5
            - nc for NetCDF
            - msa for EMSA/MSA single spectrum saving.
            - bin to produce a raw binary file
            - Many image formats such as png, tiff, jpeg...
        
        Please note that not all the formats supports saving datasets of 
        arbitrary dimensions, e.g. msa only suports 1D data.
        
        Parameters
        ----------
        filename : str
        msa_format : {'Y', 'XY'}
            'Y' will produce a file without the energy axis. 'XY' will also 
            save another column with the energy axis. For compatibility with 
            Gatan Digital Micrograph 'Y' is the default.
        only_view : bool
            If True, only the current view will be saved. Otherwise the full 
            dataset is saved. Please note that not all the formats support this 
            option at the moment.
        """
        file_io.save(filename, self, **kwds)
        
    def _replot(self):
        if self._plot is not None:
            if self._plot.is_active() is True:
                self.plot()
                
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
        self._replot()
        
    # Transform ________________________________________________________________
        
    def crop_in_pixels(self, axis, i1 = None, i2 = None):
        """Crops the data in a given axis. The range is given in pixels
        axis : int
        i1 : int
            Start index
        i2 : int
            End index
            
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
        
    def crop_in_units(self, axis, x1 = None, x2 = None):
        """Crops the data in a given axis. The range is given in the units of 
        the axis
         
        axis : int
        i1 : int
            Start index
        i2 : int
            End index
            
        See also:
        ---------
        crop_in_pixels
        
        """
        i1 = self.axes_manager.axes[axis].value2index(x1)
        i2 = self.axes_manager.axes[axis].value2index(x2)
        self.crop_in_pixels(axis, i1, i2)
        
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
        self.data[:n_x,...] = np.roll(self.data[:n_x,...],n_y,1)
        self._replot()

    # TODO: After using this function the plotting does not work  
    def swap_axis(self, axis1, axis2):
        """Swaps the axes
        
        Parameters
        ----------
        axis1 : positive int
        axis2 : positive int        
        """
        self.data = self.data.swapaxes(axis1,axis2)
        c1 = self.axes_manager.axes[axis1]
        c2 = self.axes_manager.axes[axis2]
        c1.index_in_array, c2.index_in_array =  \
            c2.index_in_array, c1.index_in_array
        self.axes_manager.axes[axis1] = c2
        self.axes_manager.axes[axis2] = c1
        self.axes_manager.set_output_dim()
        self._replot()
        
    def rebin(self, new_shape):
        """
        Rebins the data to the new shape
        
        Parameters
        ----------
        new_shape: tuple of ints
            The new shape must be a divisor of the original shape        
        """
        factors = np.array(self.data.shape) / np.array(new_shape)
        self.data = utils.rebin(self.data,new_shape)
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
            step =  rounded / number_of_parts
            cut_node = range(0,rounded+step,step)
        else:
            cut_node = np.array([0] + steps).cumsum()
        for i in xrange(len(cut_node)-1):
            data = self.data[
            (slice(None),)*axis + (slice(cut_node[i],cut_node[i+1]), Ellipsis)]
            s = Signal({'data' : data})
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

    def unfold(self, steady_axis = -1, unfolded_axis = -2):
        """Modify the shape of the data to obtain a 2D object
        
        Parameters
        ----------
        steady_axis : int
            The index of the axis which dimension does not change
        unfolded_axis : int
            The index of the axis over which all the rest of the axes (except 
            the steady axis) will be projected
            
        See also
        --------
        fold
        """
        
        # It doesn't make sense unfolding when dim < 3
        if len(self.data.squeeze().shape) < 3: return
        
        # We need to store the original shape and coordinates to be used by
        # the fold function
        self._shape_before_unfolding = self.data.shape
        self._axes_manager_before_unfolding = self.axes_manager
        
        new_shape = [1] * len(self.data.shape)
        new_shape[steady_axis] = self.data.shape[steady_axis]
        new_shape[unfolded_axis] = -1
        self.data = self.data.reshape(new_shape).squeeze()
        self.axes_manager = AxesManager(
            self._get_undefined_axes_list())
        if steady_axis > unfolded_axis:
            index = 1
        else:
            index = 0
        nc = self._axes_manager_before_unfolding.axes[
            steady_axis].get_axis_dictionary()
        nc['index_in_array'] = index
        self.axes_manager.axes[index].__init__(**nc)
        self.axes_manager.axes[index].slice = slice(None)
        self.axes_manager.axes[index - 1].slice = None
        self._replot()
            
    def fold(self):
        """If the SI was previously unfolded, folds it back"""
        if self._shape_before_unfolding is not None:
            self.data = self.data.reshape(self._shape_before_unfolding)
            self.axes_manager = self._axes_manager_before_unfolding
            self._shape_before_unfolding = None
            self._axes_manager_before_unfolding = None
            self._unfolded4pca=False
            self._replot()

    def _get_positive_axis_index_index(self, axis):
        if axis < 0:
            axis = len(self.data.shape) + axis
        return axis



    def iterate_axis(self, axis = -1):
        # We make a copy to guarantee that the data in contiguous, otherwise
        # it will not return a view of the data
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


        
    def sum(self, axis, return_signal = False):
        """Sum the data over the specify axis
        
        Parameters
        ----------
        axis : int
            The axis over which the operation will be performed
        return_signal : bool
            If False the operation will be performed on the current object. If
            True, the current object will not be modified and the operation will
             be performed in a new signal object that will be returned.
             
        Returns
        -------
        Depending on the value of the return_signal keyword, nothing or a signal
         instance
         
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
        s.axes_manager.set_output_dim()
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
            True, the current object will not be modified and the operation will
             be performed in a new signal object that will be returned.
             
        Returns
        -------
        Depending on the value of the return_signal keyword, nothing or a signal
         instance
         
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
        s.axes_manager.set_output_dim()
        if return_signal is True:
            return s 
            
    def copy(self):
        return(copy.copy(self))
        
    def deepcopy(self):
        return(copy.deepcopy(self))

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
#    def estimate_variance(self, dc = None, gaussian_noise_var = None):
#        """Variance estimation supposing Poissonian noise
#        
#        Parameters
#        ----------
#        dc : None or numpy array
#            If None the SI is used to estimate its variance. Otherwise, the 
#            provided array will be used.   
#        Note
#        ----
#        The gain_factor and gain_offset from the aquisition parameters are used
#        """
#        print "Variace estimation using the following values:"
#        print "Gain factor = ", self.acquisition_parameters.gain_factor
#        print "Gain offset = ", self.acquisition_parameters.gain_offset
#        if dc is None:
#            dc = self.data_cube
#        gain_factor = self.acquisition_parameters.gain_factor
#        gain_offset = self.acquisition_parameters.gain_offset
#        self.variance = dc*gain_factor + gain_offset
#        if self.variance.min() < 0:
#            if gain_offset == 0 and gaussian_noise_var is None:
#                print "The variance estimation results in negative values"
#                print "Maybe the gain_offset is wrong?"
#                self.variance = None
#                return
#            elif gaussian_noise_var is None:
#                print "Clipping the variance to the gain_offset value"
#                self.variance = np.clip(self.variance, np.abs(gain_offset), 
#                np.Inf)
#            else:
#                print "Clipping the variance to the gaussian_noise_var"
#                self.variance = np.clip(self.variance, gaussian_noise_var, 
#                np.Inf) 
#   
#    def calibrate(self, lcE = 642.6, rcE = 849.7, lc = 161.9, rc = 1137.6, 
#    modify_calibration = True):
#        dispersion = (rcE - lcE) / (rc - lc)
#        origin = lcE - dispersion * lc
#        print "Energy step = ", dispersion
#        print "Energy origin = ", origin
#        if modify_calibration is True:
#            self.set_new_calibration(origin, dispersion)
#        return origin, dispersion
#    
    def _correct_spatial_mask_when_unfolded(self, spatial_mask = None,):
        #if 'unfolded' in self.history:
        if spatial_mask is not None:
           spatial_mask = \
               spatial_mask.reshape((-1,))
        return spatial_mask
#        
#    def get_single_spectrum(self):
#        s = Spectrum({'calibration' : {'data_cube' : self()}})
#        s.get_calibration_from(self)
#        return s
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
        
        
#class SignalwHistory(Signal):
#    def _get_cube(self):
#        return self.__cubes[self.current_cube]['data']
#    
#    def _set_cube(self,arg):
#        self.__cubes[self.current_cube]['data'] = arg
#    data_cube = property(_get_cube,_set_cube)
#    
#    def __new_cube(self, cube, treatment):
#        history = copy.copy(self.history)
#        history.append(treatment)
#        if self.backup_cubes:
#            self.__cubes.append({'data' : cube, 'history': history})
#        else:
#            self.__cubes[-1]['data'] = cube
#            self.__cubes[-1]['history'] = history
#        self.current_cube = -1
#        
#    def _get_history(self):
#        return self.__cubes[self.current_cube]['history']
#    def _set_treatment(self,arg):
#        self.__cubes[self.current_cube]['history'].append(arg)
#        
#    history = property(_get_history,_set_treatment)
#    
#    def print_history(self):
#        """Prints the history of the SI to the stdout"""
#        i = 0
#        print
#        print "Cube\tHistory"
#        print "----\t----------"
#        print
#        for cube in self.__cubes:
#            print i,'\t', cube['history']
#            i+=1
#            
        
        
        
    
    
        
        
