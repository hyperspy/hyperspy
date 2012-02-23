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


import math, copy, os, csv

import numpy as np
import scipy as sp
from scipy.interpolate import splev,splrep,splint
from numpy import log, exp
from scipy.signal import cspline1d_eval

from hyperspy.defaults_parser import preferences
from hyperspy.component import Component
from hyperspy import messages
from hyperspy.misc.config_dir import config_path

# Global constants
# Fundamental constants
R = 13.6056923 #Rydberg of energy in eV
e = 1.602176487e-19 #electron charge in C
m0 = 9.10938215e-31 #electron rest mass in kg
a0 = 5.2917720859e-11 #Bohr radius in m
c = 2997.92458e8 #speed of light in m/s

file_path = os.path.join(config_path, 'edges_db.csv') 
f = open(file_path, 'r')
reader = csv.reader(f)
edges_dict = {}

for row in reader:
    twin_subshell = None
    element, subshell = row[0].split('.')
    Z = row[1]
    if edges_dict.has_key(element) is not True :
        edges_dict[element]={}
        edges_dict[element]['subshells'] = {}
        edges_dict[element]['Z'] = Z
    if row[3] is not '':
        if subshell == "L3":
            twin_subshell = "L2"
            factor = 0.5
        if subshell == "M3":
            twin_subshell = "M2"
            factor = 0.5
        if subshell == "M5":
            twin_subshell = "M4"
            factor = 4/6.
        if subshell == "N3":
            twin_subshell = "N2"
            factor = 2/4.
        if subshell == "N5":
            twin_subshell = "N4"
            factor = 4/6.
        if subshell == "N7":
            twin_subshell = "N6"
            factor = 6/8.
        if subshell == "O5":
            twin_subshell = "O4"
            factor = 4/6.
            
    edges_dict[element]['subshells'][subshell] = {}
    edges_dict[element]['subshells'][subshell]['onset_energy'] = float(row[2])
    edges_dict[element]['subshells'][subshell]['filename'] = row[0]
    edges_dict[element]['subshells'][subshell]['relevance'] = row[4]
    edges_dict[element]['subshells'][subshell]['factor'] = 1
    
    if twin_subshell is not None :
        edges_dict[element]['subshells'][twin_subshell] = {}
        edges_dict[element]['subshells'][twin_subshell]['onset_energy'] = \
        float(row[3])
        edges_dict[element]['subshells'][twin_subshell]['filename'] = row[0]
        edges_dict[element]['subshells'][twin_subshell]['relevance'] = row[4]
        edges_dict[element]['subshells'][twin_subshell]['factor'] = factor



def EffectiveAngle(E0,E,alpha,beta):
    """Calculates the effective collection angle
    
    Parameters
    ----------
    E0 : float
        incident energy in eV
    E : float
        energy loss in eV
    alpha : float
        convergence angle in mrad
    beta : float
        collection angle in mrad
        
    Returns
    -------
    float : effective collection angle
    
    Notes
    -----
    Code translated to Python from Egerton (second edition) page 420
    """	   
    if alpha == 0:
        return beta * 10**-3
    E0=10.**-3*E0 # In KeV
    E=float(E)
    alpha=float(alpha)
    beta=float(beta)
    TGT=E0*(1. + E0/1022.)/(1.+E0/511.)
    thetaE=E/TGT
    A2=alpha*alpha*1e-6
    B2=beta*beta*1e-6
    T2=thetaE*thetaE*1e-6
    eta1=math.sqrt((A2+B2+T2)**2-4.*A2*B2)-A2-B2-T2
    eta2=2.*B2*math.log(0.5/T2*(math.sqrt((A2+T2-B2)**2+4.*B2*T2)+A2+T2-B2))
    eta3=2.*A2*math.log(0.5/T2*(math.sqrt((B2+T2-A2)**2+4.*A2*T2)+B2+T2-A2))
#    ETA=(eta1+eta2+eta3)/A2/math.log(4./T2)
    F1=(eta1+eta2+eta3)/2/A2/math.log(1.+B2/T2)
    F2=F1
    if (alpha/beta)> 1 :
        F2=F1*A2/B2
    BSTAR=thetaE*math.sqrt(math.exp(F2*math.log(1.+B2/T2))-1.)
    return BSTAR*10**-3 # In rad

class EELSCLEdge(Component):
    """EELS core loss ionisation edge.
    
    This component reads the cross section from a folder specified in the 
    GOS_directory parameter of the Preferences. It supports fitting a spline
    to the fine structure area of the ionisation edge.
    
    Currently it only supports P. Rez Hartree Slater cross sections parametrised
    as distributed by Gatan in their Digital Micrograph software.
    
    """

    def __init__(self, element_subshell, intensity=1.,delta=0.):
        # Check if the Peter Rez's Hartree Slater GOS distributed by Gatan 
        # are available. Otherwise exit
        if not os.path.isdir(preferences.EELS.eels_gos_files_path):
            message = (
            "The path to the GOS files could not be found. " 
            "Please define a valid location for the EELS GOS files in the "
            "folder location in the configuration file.")
            raise IOError(message)
        # Declare which are the "real" parameters
        Component.__init__(self, ['delta', 'intensity', 'fslist', 
        'effective_angle'])
        self.name = element_subshell
        # Set initial values
        self.__element, self.__subshell = element_subshell.split('_')
        self.energy_scale = None
        self.T = None
        self.gamma = None
        self.convergence_angle = None
        self.collection_angle = None
        self.E0 = None
        self.effective_angle.value = 0
        self.effective_angle.free = False
        self.fs_state = preferences.EELS.fs_state
        self.fs_emax = preferences.EELS.fs_emax
        self.fs_mode = "new_spline"
        self.fslist.ext_force_positive = False
        
        self.delta.value = delta
        self.delta.free = False
        self.delta.ext_force_positive = False
        self.delta.grad = self.grad_delta
        self.freedelta = False
        self._previous_delta = delta
                                
        self.intensity.grad = self.grad_intensity
        self.intensity.value = intensity
        self.intensity.bmin = 0.
        self.intensity.bmax = None

        self.knots_factor = preferences.EELS.knots_factor

        # Set initial actions
        self.readgosfile()
        
    # Automatically fix the fine structure when the fine structure is disable.
    # This way we avoid a common source of problems when fitting
    # However the fine structure must be *manually* freed when we reactivate
    # the fine structure.
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

    def edge_position(self):
        return self.edgeenergy + self.delta.value
        
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
        # Relativistic correction factors
        self.gamma = 1.0 + (e * E0) / (m0 * pow(c,2.0)) #dimensionless
        self.T = E0 * (1.0 + self.gamma) / (2.0 * pow(self.gamma, 2.0)) #in eV
        self.convergence_angle = alpha
        self.collection_angle = beta
        self.energy_scale = energy_scale
        self.E0 = E0
        self.integrategos(self.delta.value)
        
    def readgosfile(self): 
        element = self.__element
        # Convert to the "GATAN" nomenclature
        if self.__subshell == "K" :
            subshell = "K1"
        else:
            subshell = self.__subshell
        if edges_dict.has_key(element) is not True:
            message = "The given element " + element + \
            " is not in the database."
            messages.warning_exit(message)
        elif edges_dict[element]['subshells'].has_key(subshell) is not True :
            message =  "The given subshell " + subshell + \
            " is not in the database." + "\nThe available subshells are:\n" + \
            str(edges_dict[element]['subshells'].keys())
            messages.warning_exit(message)
            
        self.edgeenergy = \
        edges_dict[element]['subshells'][subshell]['onset_energy']
        self.__subshell_factor = \
        edges_dict[element]['subshells'][subshell]['factor']
        print "\nLoading Hartree-Slater cross section from the Gatan tables"
        print "Element: ", element
        print "Subshell: ", subshell
        print "Onset Energy = ", self.edgeenergy
        #Read file
        file = os.path.join(preferences.EELS.eels_gos_files_path, 
        edges_dict[element]['subshells'][subshell]['filename'])
        f = open(file)
 
        #Tranfer the content of the file to a list
        GosList = f.read().replace('\r','').split()

        #Extract the parameters

        self.material = GosList[0]
        self.__info1_1 = float(GosList[2])
        self.__info1_2 = float(GosList[3])
        self.__info1_3 = float(GosList[4])
        self.__ncol    = int(GosList[5])
        self.__info2_1 = float(GosList[6])
        self.__info2_2 = float(GosList[7])
        self.__nrow    = int(GosList[8])
        self.__gos_array = np.array(GosList[9:]).reshape(self.__nrow, 
        self.__ncol).astype(np.float64)
        
        # Calculate the scale of the matrix
        self.energyaxis = self.__info2_1 * (exp(np.linspace(0, 
        self.__nrow-1,self.__nrow) * self.__info2_2 / self.__info2_1) - 1.0)
        
        self.__qaxis=(self.__info1_1 * (exp(np.linspace(1, self.__ncol, 
        self.__ncol) * self.__info1_2) - 1.0)) * 1.0e10
        self.__sqa0qaxis = (a0 * self.__qaxis)**2
        self.__logsqa0qaxis = log((a0 * self.__qaxis)**2)
        
    def integrategos(self, delta = 0):
        """
        Calculates the knots of the spline interpolation of the cross section 
        after integrating q. It calculates it for Ek in the range 
        (Ek-Ekrange,Ek+Ekrange) for optimizing the time of the fitting. 
        For a value outside of the range it returns the closer limit, 
        however this is not likely to happen in real data
        """	
        
        def getgosenergy(i):
            """
            Given the row number i (starting from 0) returns the corresponding 
            energy
            """	
            return self.__info2_1 * (math.exp(i * self.__info2_2 / \
            self.__info2_1) - 1.0)
        
        def emax(edgeenergy,i): return self.energyaxis[i] + edgeenergy
        qint = sp.zeros((self.__nrow))
        
        # Integration over q using splines
        self.effective_angle.value = EffectiveAngle(self.E0, self.edgeenergy, 
            self.convergence_angle, self.collection_angle)
        self._previous_effective_angle = self.effective_angle.value
        effective_angle = self.effective_angle.value
        for i in xrange(0,self.__nrow):
            qtck = splrep(self.__logsqa0qaxis, self.__gos_array[i, :], s=0)
            qa0sqmin = (emax(self.edgeenergy + self.delta.value, i)**2) / (
            4.0 * R * self.T) + (emax(self.edgeenergy + self.delta.value, 
            i)**3) / (8.0 * self.gamma ** 3.0 * R * self.T**2)
            qa0sqmax = qa0sqmin + 4.0 * self.gamma**2 * (self.T/R) * math.sin(
            effective_angle / 2.0)**2.0
            qmin = math.sqrt(qa0sqmin) / a0
            qmax=math.sqrt(qa0sqmax) / a0
            
            # Error messages for out of tabulated data
            if qmax > self.__qaxis[-1] :
                print "i=",i
                print "Maximum tabulated q reached!!"
                print "qa0sqmax=",qa0sqmax
                qa0sqmax = self.__sqa0qaxis[self.__ncol-1]
                print "qa0sqmax tabulated maximum", 
                self.__sqa0qaxis[self.__ncol-1]
                
            if qmin < self.__qaxis[0] :
                print "i=",i
                print "Minimum tabulated q reached!! Accuracy not garanteed"
                print "qa0sqmin",qa0sqmin
                qa0sqmin = self.__sqa0qaxis[0]
                print "qa0sqmin tabulated minimum", qa0sqmin
            
            # Writes the integrated values to the qint array.
            qint[i] = splint(math.log(qa0sqmin), math.log(qa0sqmax), qtck)
        self.__qint = qint        
        self.__goscoeff = splrep(self.energyaxis,qint,s=0)
        
        # Calculate extrapolation powerlaw extrapolation parameters
        E1 = self.energyaxis[-2] + self.edgeenergy + self.delta.value
        E2 = self.energyaxis[-1] + self.edgeenergy + self.delta.value
        factor = 4.0 * np.pi * a0 ** 2.0 * R**2.0 / E1 / self.T
        y1 = factor * splev((E1 - self.edgeenergy - self.delta.value), 
        self.__goscoeff) # in m**2/bin */
        factor = 4.0 * np.pi * a0 ** 2.0 * R ** 2.0 / E2 / self.T
        y2 = factor * splev((E2 - self.edgeenergy - self.delta.value), 
        self.__goscoeff) # in m**2/bin */
        self.r = math.log(y2 / y1) / math.log(E1 / E2)
        self.A = y1 / E1**-self.r
        
    def calculate_knots(self):    
        # Recompute the knots
        start = self.edgeenergy + self.delta.value
        stop = start + self.fs_emax
        self.__knots = np.r_[[start]*4,
        np.linspace(start, stop, self.fslist._number_of_elements)[2:-2], 
        [stop]*4]
        
    def function(self,E) :
        """ Calculates the number of counts in barns"""
        
        if self.delta.value != self._previous_delta:
            self._previous_delta = copy.copy(self.delta.value)
            self.integrategos(self.delta.value)
            self.calculate_knots()

        if self._previous_effective_angle != self.effective_angle.value:
            self.integrategos()
            
        factor = 4.0 * np.pi * a0 ** 2.0 * R**2 / E / self.T #to convert to m**2/bin
        Emax = self.energyaxis[-1] + self.edgeenergy + \
        self.delta.value #maximum tabulated energy
        cts = np.zeros((len(E)))
        
        if self.fs_state is True:
            if self.__knots[-1] > Emax : Emax = self.__knots[-1]
            fine_structure_indices=np.logical_and(np.greater_equal(E, 
            self.edgeenergy+self.delta.value), 
            np.less(E, self.edgeenergy + self.delta.value + self.fs_emax))
            tabulated_indices = np.logical_and(np.greater_equal(E, 
            self.edgeenergy + self.delta.value + self.fs_emax), 
            np.less(E, Emax))
            if self.fs_mode == "new_spline" :
                cts = np.where(fine_structure_indices, 
                1E-25*splev(E,(self.__knots,self.fslist.value,3),0), cts)
            elif self.fs_mode == "spline" :
                cts = np.where(fine_structure_indices, 
                cspline1d_eval(self.fslist.value, 
                E, 
                dx = self.energy_scale / self.knots_factor, 
                x0 = self.edgeenergy+self.delta.value), 
                cts)
            elif self.fs_mode == "spline_times_edge" :
                cts = np.where(fine_structure_indices, 
                factor*splev((E-self.edgeenergy-self.delta.value), 
                self.__goscoeff)*cspline1d_eval(self.fslist.value, 
                E,dx = self.energy_scale / self.knots_factor, 
                x0 = self.edgeenergy+self.delta.value), 
                cts )
        else:
            tabulated_indices = np.logical_and(np.greater_equal(E, 
            self.edgeenergy + self.delta.value), np.less(E, Emax))            
        powerlaw_indices = np.greater_equal(E,Emax)  
        cts = np.where(tabulated_indices, 
        factor * splev((E-self.edgeenergy-self.delta.value), 
        self.__goscoeff),
         cts)
        
        # Convert to barns/dispersion.
        #Note: The R factor is introduced in order to give the same value
        # as DM, although it is not in the equations.
        cts = np.where(powerlaw_indices, self.A * E**-self.r, cts) 
        return (self.__subshell_factor * self.intensity.value * self.energy_scale 
        * 1.0e28 / R) * cts       
    
    def grad_intensity(self,E) :
        
        if self.delta.value != self._previous_delta :
            self._previous_delta = copy.copy(self.delta.value)
            self.integrategos(self.delta.value)
            self.calculate_knots()
            
        factor = 4.0 * np.pi * a0 ** 2.0 * \
        (R ** 2.0) / (E * self.T) #to convert to m**2/bin
        Emax = self.energyaxis[-1] + self.edgeenergy + \
        self.delta.value #maximum tabulated energy
        cts = np.zeros((len(E)))
        
        if self.fs_state is True:
            if self.__knots[-1] > Emax : Emax = self.__knots[-1]
            fine_structure_indices=np.logical_and(np.greater_equal(E, 
            self.edgeenergy+self.delta.value), 
            np.less(E, self.edgeenergy + self.delta.value + self.fs_emax))
            tabulated_indices = np.logical_and(np.greater_equal(E, 
            self.edgeenergy + self.delta.value + self.fs_emax), 
            np.less(E, Emax))
            if self.fs_mode == "new_spline" :
                cts = np.where(fine_structure_indices, 
                1E-25*splev(E,(self.__knots,self.fslist.value,3),0), cts)
            elif self.fs_mode == "spline" :
                cts = np.where(fine_structure_indices, 
                cspline1d_eval(self.fslist.value, 
                E, 
                dx = self.energy_scale / self.knots_factor, 
                x0 = self.edgeenergy+self.delta.value), 
                cts)
            elif self.fs_mode == "spline_times_edge" :
                cts = np.where(fine_structure_indices, 
                factor*splev((E-self.edgeenergy-self.delta.value), 
                self.__goscoeff)*cspline1d_eval(self.fslist.value, 
                E,dx = self.energy_scale / self.knots_factor, 
                x0 = self.edgeenergy+self.delta.value), 
                cts )
        else:
            tabulated_indices = np.logical_and(np.greater_equal(E, 
            self.edgeenergy + self.delta.value), np.less(E, Emax))
        powerlaw_indices = np.greater_equal(E,Emax)  
        cts = np.where(tabulated_indices, 
        factor * splev((E-self.edgeenergy-self.delta.value), 
        self.__goscoeff),
         cts)
        
        # Convert to barns/dispersion.
        #Note: The R factor is introduced in order to give the same value
        # as DM, although it is not in the equations.
        cts = np.where(powerlaw_indices, self.A * pow(E,-self.r), cts)
        return ((1.0e28 *self.__subshell_factor * self.energy_scale)/R)*cts        

    
    def grad_delta(self,E) :
        """ Calculates the number of counts in barns"""
        
        if self.delta.value != self._previous_delta :
            self._previous_delta = copy.copy(self.delta.value)
            self.integrategos(self.delta.value)
            self.calculate_knots()
        factor = 4.0 * np.pi * (a0**2.0) * (
        R**2.0) / (E * self.T) #to convert to m**2/bin
        Emax = self.energyaxis[-1] + self.edgeenergy + \
        self.delta.value #maximum tabulated energy
        cts = np.zeros((len(E)))
        
        if self.fs_state is True:
            if self.__knots[-1] > Emax : Emax = self.__knots[-1]
            fine_structure_indices=np.logical_and(np.greater_equal(E, 
            self.edgeenergy+self.delta.value), 
            np.less(E, self.edgeenergy + self.delta.value + self.fs_emax))
            tabulated_indices = np.logical_and(np.greater_equal(E, 
            self.edgeenergy + self.delta.value + self.fs_emax), 
            np.less(E, Emax))
            cts = 1E-25 * np.where(fine_structure_indices, 
            splev(E,(self.__knots,self.fslist.value,3),1), cts)
        else:
            tabulated_indices = np.logical_and(np.greater_equal(E, 
            self.edgeenergy + self.delta.value), np.less(E, Emax))
        
        powerlaw_indices = np.greater_equal(E,Emax)  
        cts = np.where(tabulated_indices, 
        factor * splev((E-self.edgeenergy-self.delta.value), 
        self.__goscoeff, 1),
         cts)
        
        # Convert to barns/dispersion.
        #Note: The R factor is introduced in order to give the same value
        # as DM, although it is not in the equations.
        cts = np.where(powerlaw_indices, -self.r * self.A *\
         (E**-self.r-1), cts)
        return - ((1.0e28 *self.__subshell_factor * self.intensity.value 
    * self.energy_scale)/R) * cts         

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
