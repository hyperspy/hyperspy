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
import new_plot


class Signal(t.HasTraits):
    data = t.Array()
    coordinates = t.Instance(new_coordinates.NewCoordinates)
    extra_parameters = t.Dict()
    parameters = t.Dict()
    name = t.Str('')
    units = t.Str()
    
    def __init__(self, dictionary):
        super(Signal, self).__init__()
        self.data = dictionary['data']
        self.coordinates = new_coordinates.NewCoordinates(dictionary['coordinates'])
        self.extra_parameters = dictionary['extra_parameters']
        self.parameters = dictionary['parameters']
        
    def __call__(self):
        return self.data.__getitem__(self.coordinates.getitem_tuple)
    
    def plot(self):
        if self.coordinates.output_dim == 1:
            self._plot = new_plot.Plot1D(self, self.coordinates)
        elif self.coordinates.output_dim == 2:
            self._plot = new_plot.Plot2D(self, self.coordinates)
        self._plot.plot()
        
    traits_view = tui.View(
        tui.Item('name'),
        tui.Item('units'),
        )
    
    def save(self, filename, format = 'hdf5', **kwds):
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
        
        
        
    
    
        
        
