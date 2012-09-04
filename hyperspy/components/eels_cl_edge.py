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

from __future__ import division
import math, copy

import numpy as np
import scipy as sp
import scipy.interpolate
from scipy.interpolate import splev,splrep,splint

from hyperspy.defaults_parser import preferences
from hyperspy.component import Component
from hyperspy import messages
from hyperspy.misc.eels.hartree_slater_gos import HartreeSlaterGOS
from hyperspy.misc.eels.hydrogenic_gos import HydrogenicGOS

from hyperspy.misc.eels.effective_angle import EffectiveAngle

class EELSCLEdge(Component):
    """EELS core loss ionisation edge from hydrogenic or tabulated 
    Hartree-Slater GOS with splines for fine structure fitting.
    
    Currently it only supports Peter Rez's Hartree Slater cross sections
    parametrised as distributed by Gatan in their 
    Digital Micrograph (DM) software. If Digital Micrograph is intalled
    in the system Hyperespy in the standard location HyperSpy should 
    find the path to the HS GOS folder. Otherwise, the location of the 
    folder can be defined in Hyperspy preferences.
    
    Hydrogenic GOS are limited to K and L shells.
    
    Parameters
    ----------
    element_subshell : str
            For example, 'Ti_L3' for the GOS of the titanium L3 subshell
            
    GOS : {'hydrogenic', 'Hartree-Slater', None}
        The GOS to use. If None it will use the Hartree-Slater GOS if 
        they are available, otherwise it will use the hydrogenic GOS.
    
    Returns
    -------
    array: cross section in barns (1E-28 m)
    
    """

    def __init__(self, element_subshell, GOS=None):
        # Declare the parameters
        Component.__init__(self,
            ['delta', 'intensity', 'fslist', 'effective_angle'])
        self.name = element_subshell
        self.element, self.subshell = element_subshell.split('_')
        self.energy_scale = None
        self.effective_angle.free = False
        self.fs_state = preferences.EELS.fs_state
        self.fs_emax = preferences.EELS.fs_emax
        self.fslist.ext_force_positive = False
        
        self.delta.value = 0
        self.delta.free = False
        self.delta.ext_force_positive = False
        self.freedelta = False
        
        self.intensity.grad = self.grad_intensity
        self.intensity.value = 1
        self.intensity.bmin = 0.
        self.intensity.bmax = None

        self.knots_factor = preferences.EELS.knots_factor

        # Set initial actions
        if GOS is None:
            try:
                self.GOS = HartreeSlaterGOS(element_subshell)
            except IOError:
                GOS = 'hydrogenic'
                messages.information(
                    'Hartree-Slater GOS not available'
                    'Using hydrogenic GOS')
        elif GOS == 'Hartree-Slater':
            self.GOS = HartreeSlaterGOS(element_subshell)
        elif GOS == 'hydrogenic':
            self.GOS = HydrogenicGOS(element_subshell)
        else:
            raise ValueError('gos must be one of: None, \'hydrogenic\''
                              '\'Hartree-Slater\'')
                    
    # Automatically fix the fine structure when the fine structure is 
    # disable.
    # In this way we avoid a common source of problems when fitting
    # However the fine structure must be *manually* freed when we 
    # reactivate the fine structure.
    def _get_fs_state(self):
            return self.__fs_state
    def _set_fs_state(self,arg):
        if arg is False:
            self.fslist.free = False
        self.__fs_state = arg
    fs_state = property(_get_fs_state,_set_fs_state)
    
    def _get_fs_emax(self):
        return self.__fs_emax
    def _set_fs_emax(self,arg):
        self.__fs_emax = arg
        self.setfslist()
    fs_emax = property(_get_fs_emax,_set_fs_emax)
    
    
    # E0
    def _get_E0(self):
        return self.__E0
    def _set_E0(self,arg):
        self.__E0 = arg
        self.calculate_effective_angle()
    E0 = property(_get_E0,_set_E0)
    
    # Collection angles
    def _get_collection_angle(self):
        return self.__collection_angle
    def _set_collection_angle(self,arg):
        self.__collection_angle = arg
        self.calculate_effective_angle()
    collection_angle = property(_get_collection_angle, 
                                _set_collection_angle)
    # Convergence angle
    def _get_convergence_angle(self):
        return self.__convergence_angle
    def _set_convergence_angle(self,arg):
        self.__convergence_angle = arg
        self.calculate_effective_angle()
    convergence_angle = property(_get_convergence_angle, 
                                 _set_convergence_angle)
                            
    
    
    def calculate_effective_angle(self):
        try:
            self.effective_angle.value = EffectiveAngle(
                                                self.E0,
                                                self.GOS.onset_energy, 
                                                self.convergence_angle,
                                                self.collection_angle)
        except:
            #All the parameters may not be defined yet...
            pass

    def edge_position(self):
        return self.GOS.onset_energy + self.delta.value
        
    def setfslist(self):
        if self.energy_scale is None:
            return
        self.fslist._number_of_elements = \
        int(round(self.knots_factor * self.fs_emax / self.energy_scale)) + 4        
        self.fslist.bmin, self.fslist.bmax = None, None
        self.fslist.value = np.zeros(self.fslist._number_of_elements).tolist()
        self.calculate_knots()
        if self.fslist.map is not None:
            self.fslist.create_array(self.fslist.map.shape)
            
    def set_microscope_parameters(self, E0, alpha, beta, energy_scale):
        """
        Parameters
        ----------
        E0 : float
            Electron beam energy in keV.
        alpha: float
            Convergence angle in mrad.
        beta: float
            Collection angle in mrad.
        energy_scale : float
            The energy step in eV.
        """
        # Relativistic correction factors

        self.convergence_angle = alpha
        self.collection_angle = beta
        self.energy_scale = energy_scale
        self.E0 = E0
        self.integrate_GOS()
                
    def integrate_GOS(self):
        # Integration over q using splines                                        
        angle = self.effective_angle.value * 1e-3 # in rad
        self.tab_xsection = self.GOS.integrateq(
                self.delta.value, angle, self.E0)                
        # Calculate extrapolation powerlaw extrapolation parameters
        E1 = self.GOS.energy_axis[-2] + self.delta.value
        E2 = self.GOS.energy_axis[-1] + self.delta.value
        y1 = self.GOS.qint[-2] # in m**2/bin */
        y2 = self.GOS.qint[-1] # in m**2/bin */
        self.r = math.log(y2 / y1) / math.log(E1 / E2)
        self.A = y1 / E1**-self.r
        
        # Connect them at this point where it is certain that all the 
        # parameters are well defined
        self.effective_angle.connect(self.integrate_GOS)
        self.effective_angle.connection_active = True
        self.delta.connect(self.integrate_GOS)
        self.delta.connect(self.calculate_knots)
        self.delta.connection_active = True
        
    def calculate_knots(self):    
        # Recompute the knots
        start = self.GOS.onset_energy + self.delta.value
        stop = start + self.fs_emax
        self.__knots = np.r_[[start]*4,
        np.linspace(start, stop, self.fslist._number_of_elements)[2:-2], 
        [stop]*4]
        
    def function(self,E) :
        """Returns the number of counts in barns
        
        """
        Emax = self.GOS.energy_axis[-1] + self.delta.value 
        cts = np.zeros((len(E)))
        bsignal = (E >= self.edge_position())
        if self.fs_state is True:
            bfs = bsignal * (E < (self.edge_position() + self.fs_emax))
            cts[bfs] = splev(E[bfs],
                        (self.__knots, self.fslist.value + [0,]*4, 3))
            bsignal[bfs] = False
        itab = bsignal * (E <= Emax)
        cts[itab] = self.tab_xsection(E[itab])
        bsignal[itab] = False
        cts[bsignal] = self.A * E[bsignal]**-self.r
        return cts * self.intensity.value * self.energy_scale
    
    def grad_intensity(self,E) :
        return self.function(E) / self.intensity.value    

    def fslist_to_txt(self,filename) :
        np.savetxt(filename + '.dat', self.fslist.value, fmt="%12.6G")
 
    def txt_to_fslist(self,filename) :
        fs = np.loadtxt(filename)
        self.calculate_knots()
        if len(fs) == len(self.__knots) :
            self.fslist.value = fs
        else :
            messages.warning_exit("The provided fine structure file "  
            "doesn't match the size of the current fine structure")
