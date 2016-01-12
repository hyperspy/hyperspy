# -*- coding: utf-8 -*-
# Copyright 2007-2016 The HyperSpy developers
#
# This file is part of  HyperSpy.
#
#  HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  HyperSpy.  If not, see <http://www.gnu.org/licenses/>.


from hyperspy._signals.eels import EELSSpectrum
from hyperspy._signals.spectrum_simulation import SpectrumSimulation


class EELSSpectrumSimulation(SpectrumSimulation, EELSSpectrum):
    pass


#    @auto_replot
#    def add_energy_instability(self, std):
#        """Introduce random energy instability
#
#        Parameters
#        ----------
#        std : float
#            std in energy units of the energy instability.
#        See also
#        --------
#        Spectrum.simulate
#        """
#        if abs(std) > 0:
#            delta_map = np.random.normal(
#            size = (self.xdimension, self.ydimension),
#            scale = abs(std))
#        else:
#            delta_map = np.zeros((self.xdimension,
#                    self.ydimension))
#        for edge in self.edges:
#            edge.delta.map = delta_map
#            edge.delta.already_set_map = np.ones((self.xdimension,
#            self.ydimension), dtype = 'Bool')
#        return delta_map


#    def simulate(self, maps = None, energy_instability = 0,
#    min_intensity = 0., max_intensity = 1.):
#        """Create a simulated SI.
#
#        If an image is provided, it will use each RGB color channel as the
#        intensity map of each three elements that must be previously defined as
#        a set in self.elements. Otherwise it will create a random map for each
#        element defined.
#
#        Parameters
#        ----------
#        maps : list/tuple of arrays
#            A list with as many arrays as elements are defined.
#        energy_instability : float
#            standard deviation in energy units of the energy instability.
#        min_intensity : float
#            minimum edge intensity
#        max_intensity : float
#            maximum edge intensity
#
#        Returns
#        -------
#
#        If energy_instability != 0 it returns the energy shift map
#        """
#        if maps is not None:
#            self.xdimension = maps[0].shape[0]
#            self.ydimension = maps[0].shape[1]
#            self.xscale = 1.
#            self.yscale = 1.
#            i = 0
#            if energy_instability > 0:
#                delta_map = np.random.normal(np.zeros((self.xdimension,
#                self.ydimension)), energy_instability)
#            for edge in self.edges:
#                edge.fine_structure_active = False
#                if not edge.intensity.twin:
#                    edge.intensity.map = maps[i]
#                    edge.intensity.already_set_map = np.ones((
#                    self.xdimension, self.ydimension), dtype = 'Bool')
#                    i += 1
#            if energy_instability != 0:
#                instability_map = self.add_energy_instability(energy_instability)
#            for edge in self.edges:
#                edge.fetch_stored_values(0,0)
#            self.create_data_cube()
#            self.model = Model(self, auto_background=False)
#            self.model.charge()
#            self.model.generate_data_from_model()
#            self.data_cube = self.model.model_cube
#            self.type = 'simulation'
#        else:
#            print "No image defined. Producing a gaussian mixture image of the \
#            elements"
#            i = 0
#            if energy_instability:
#                delta_map = np.random.normal(np.zeros((self.xdimension,
#                self.ydimension)), energy_instability)
#                print delta_map.shape
#            size = self.xdimension * self.ydimension
#            for edge in self.edges:
#                edge.fine_structure_active = False
#                if not edge.intensity.twin:
#                    edge.intensity.map = np.random.uniform(0, max_intensity,
#                    size).reshape(self.xdimension, self.ydimension)
#                    edge.intensity.already_set_map = np.ones((self.xdimension,
#                    self.ydimension), dtype = 'Bool')
#                    if energy_instability:
#                        edge.delta.map = delta_map
#                        edge.delta.already_set_map = np.ones((self.xdimension,
#                        self.ydimension), dtype = 'Bool')
#                    i += 1
#            self.create_data_cube()
#            self.model = Model(self, auto_background=False)
#            self.model.generate_data_from_model()
#            self.data_cube = self.model.model_cube
#            self.type = 'simulation'
#        if energy_instability != 0:
#            return instability_map
