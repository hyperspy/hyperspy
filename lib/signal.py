# -*- coding: utf-8 -*-
"""
Created on Wed Oct 06 09:48:42 2010

"""
import numpy as np
import enthought.traits.api as t
import enthought.traits.ui.api as tui 

import messages
import new_coordinates
import file_io
import drawing
import utils

class Signal(t.HasTraits):
    data = t.Array()
    coordinates = t.Instance(new_coordinates.CoordinatesManager)
    extra_parameters = t.Dict()
    parameters = t.Dict()
    name = t.Str('')
    units = t.Str()
    scale = t.Float()
    offset = t.Float()
    
    def __init__(self, dictionary):
        super(Signal, self).__init__()
        self.data = dictionary['data']
        self.coordinates = new_coordinates.CoordinatesManager(
        dictionary['coordinates'])
        self.extra_parameters = dictionary['extra_parameters']
        self.parameters = dictionary['parameters']
        self._plot = None
        
    def __call__(self, coordinates = None):
        if coordinates is None:
            coordinates = self.coordinates
        return self.data.__getitem__(coordinates._getitem_tuple)
        
    def is_spectrum_line(self):
        if len(self.data.squeeze().shape) == 2:
            return True
        else:
            return False
        
    def is_spectrum_image(self):
        if len(self.data.squeeze().shape) == 3:
            return True
        else:
            return False
        
    def is_single_spectrum(self):
        if len(self.data.squeeze().shape) == 1:
            return True
        else:
            return False
        
    def get_image(self, spectral_range = slice(None), background_range = None):
        data = self.data
        if self.is_spectrum_line() is True:
            return self.data.squeeze()
        elif self.is_single_spectrum() is True:
            return None
        if background_range is not None:
            bg_est = utils.two_area_powerlaw_estimation(self, 
                                                        background_range.start, 
                                                        background_range.stop, )
            A = bg_est['A'][np.newaxis,:,:]
            r = bg_est['r'][np.newaxis,:,:]
            E = self.energy_axis[spectral_range,np.newaxis,np.newaxis]
            bg = A*E**-r
            return (data[spectral_range,:,:] - bg).sum(0)
        else:
            return data[..., spectral_range].sum(-1)
    
    def plot(self, coordinates = None):
        if coordinates is None:
            coordinates = self.coordinates
        if coordinates.output_dim == 1:
            # Hyperspectrum
            if self._plot is not None:
#            if self.coordinates is not self.hse.coordinates:
                try:
                    self._plot.close()
                except:
                    # If it was already closed it will raise an exception,
                    # but we want to carry on...
                    pass
                
                self.hse = None
            self._plot = drawing.mpl_hse.MPL_HyperSpectrum_Explorer()
            self._plot.spectrum_data_function = self.__call__
            self._plot.spectrum_title = self.name
            self._plot.xlabel = '%s (%s)' % (
                self.coordinates.coordinates[-1].name, 
                self.coordinates.coordinates[-1].units)
            self._plot.ylabel = 'Intensity'
            self._plot.coordinates = coordinates
            self._plot.axis = self.coordinates.coordinates[-1].axis
            
            # Image properties
            self._plot.image_data_function = self.get_image
            self._plot.image_title = ''
            self._plot.pixel_size = self.coordinates.coordinates[0].scale
            self._plot.pixel_units = self.coordinates.coordinates[0].units
            
        elif coordinates.output_dim == 2:
            self._plot = drawing.mpl_ise.MPL_HyperImage_Explorer()
        else:
            messages.warning_exit('Plotting is not supported for this view')
        
        self._plot.plot()
        
    traits_view = tui.View(
        tui.Item('name'),
        tui.Item('units'),
        )
    
    def save(self, filename, **kwds):
        '''Saves the SI in the specified format.
        
        Supported formats: netCDF, msa and bin. netCDF is the default. msa does 
        not support SI, only the current spectrum will be saved. bin produce a 
        binary file that can be imported easily in Gatan's Digital Micrograph. 
        Because the calibration will be lost when saving in bin format, a MSA 
        file will be created to easy the transfer to DM.
        
        Parameters
        ----------
        filename : str
        format : {'netcdf', 'msa', 'bin'}
            'msa' only saves the current spectrum.
        msa_format : {'Y', 'XY'}
            'Y' will produce a file without the energy axis. 'XY' will also 
            save another column with the energy axis. For compatibility with 
            Gatan Digital Micrograph 'Y' is the default.
        '''
        file_io.save(filename, self, **kwds)
        
    def _replot(self):
        if self.hse is not None:
            if self.hse.is_active() is True:
                self.plot()
        
        
        
    
    
        
        
