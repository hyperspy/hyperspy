# -*- coding: utf-8 -*-
"""
Created on Wed Oct 06 09:48:42 2010

@author: fd227872
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
    imported_parameters = t.Dict()
    parameters = t.Dict()
    name = t.Str()
    units = t.Str()
    
    def __init__(self, dictionary):
        super(Signal, self).__init__()
        self.data = dictionary['data']
        self.coordinates = new_coordinates.NewCoordinates(dictionary['coordinates'])
        self.imported_parameters = dictionary['imported_parameters']
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
        
        
    
    
        
        
