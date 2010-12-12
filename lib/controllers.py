# -*- coding: utf-8 -*-
# Copyright © 2007 Francisco Javier de la Peña
#
# This file is part of EELSLab.
#
# EELSLab is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# EELSLab is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with EELSLab; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301
# USA
import coordinates
import drawing

class Coordinates_Controller():
    def __init__(self):
        self.registered_coordinates = []
    
    def assign_coordinates(self, signal):
        '''Add coordenates to a signal.
        
        Checks if the shape is equal to any other coordinate that has been 
        registered before, if any; if yes it assigns the matching registered 
        coordinates to the signal and return False, otherwise it creates a new 
        coordinates instance and returns True
        
        Parameters:
        -----------
        signal : Signal instance
        
        '''
        shape = self.get_coordinates_shape(signal)
        # Check if coordinates with this shape have been registered
        for coord in self.registered_coordinates:
            if shape == coord.shape:
                signal.coordinates = coord
                return False
        # There are no coordinates with that shape so we register a new one
        coord = coordinates.Coordinates(shape)
        signal.coordinates = coord
        self.registered_coordinates.append(coord) 
        return True
        
    def get_coordinates_shape(self, signal):
        shape = signal.data_cube.squeeze().shape
        if len(shape) == 2:
            shape = (signal.data_cube.shape[1],1)
        elif len(shape) == 3:
            shape = signal.data_cube.shape[1:]
        return shape

coordinates_controller = Coordinates_Controller()