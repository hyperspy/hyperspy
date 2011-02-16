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

import copy

import numpy as np
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.ndimage import  gaussian_filter1d
try:
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
# Set the matplotlib cmap to gray (the default is jet)
plt.rcParams['image.cmap'] = 'gray'
import file_io
import messages
import utils
from microscope import microscope
from image import Image
from components.power_law import PowerLaw
from edges_db import edges_dict
from model import Model
from components.edge import Edge
from defaults_parser import defaults
from interactive_ns import interactive_ns
from utils import generate_axis
from utils import rebin
from mva import MVA, MVA_Results
import drawing.mpl_hse
import controllers


#TODO Acquisition_Parameters and Treatments must be merged into a more general
#class to store somewhe else, and to use with the Image class too.
class Acquisition_Parameters(object):
    '''
    Container for the acquisition parameters.
    
    Calling the class prints all the acquisition parameters.
    '''
    binning = None
    exposure = None
    readout_frequency = None
    ccd_height = None
    blanking = None
    gain_factor = None
    gain_offset = None
    
    def __call__(self):
        for item in self.__dict__.items():
            print "%s = %s" % item
    def define(self):
        '''Prints the already defined acquisiotions parameters and ask you to 
        supply the missing ones. 
        '''
        for item in self.__class__.__dict__.items():
            if item[0][:2] != '__' and item[0] != 'define':
                if item[0] in self.__dict__:
                    print "%s = %s" % (item[0], eval('self.%s' % item[0])) 
                else:
                    value = raw_input("%s = " % item[0])
                    exec('self.%s = float(value)' % item[0])
            
class Treatments(object):
    dark_current = None
    readout = None
    gain = None
    pppc = None
    def __call__(self):
        for item in self.__dict__.items():
            print "%s = %s" % item
    
class Spectrum(object, MVA):
    '''Base class for SI
    
    This class contains the SI (Spectrum Image) basic methods. It can be used to
    load an SI or to create a monte carlo. It also heritage all the multivariate
    analysis methods from the MVA class.
     
    The addition is defined for members of the Spectrum class that has the same 
    number of channel, the same energy step and the same number of pixels in the 
    x direction. It return a vertically stacked SI.
    
    Parameters:
    -----------
    filename : str
    gain : str
        filename of the gain correction spectrum or SI
    readout: str
        filename of the readout correction spectrum or SI
    dark_current: str
        filename of the dark current correction spectrum or SI
    image: str
        filename of the image associated to the SI
    apply_treatments : Bool
        If True, the readout, dark current and gain corrections are applied
        if the corresponding files are provided
    '''    
    
    def __init__(self, dictionary = None,  gain = None, readout = None, 
    dark_current = None, image = None, apply_treatments = True):
        
        # Attributes defaults
        self.subshells = set()
        self.elements = set()
        self.edges = list()
        self._get_dimensions_from_cube = False
        self.__cubes = []
        self.__cubes.append({'history': ['None'], 'data': None})
        self.current_cube = -1
        self.zero_loss = None
        self.variance = None
        self.readout = None
        self.dark_current = None
        self.gain_correction = None
        self.backup_cubes = False
        self.hse = None
        self.coordinates = None
                
        # Attributes declaration (for simulation)
        self.xdimension = 1
        self.ydimension = 1
        self.energydimension = 1024
        self.xscale = 1.
        self.yscale = 1.
        self.energyscale = 1.
        self.xorigin = 0.
        self.yorigin = 0.
        self.energyorigin = 0.
        self.xunits = ''
        self.yunits = ''
        self.energyunits = ''
        self.title = ''
        self.mva_results = MVA_Results()
        
        self.acquisition_parameters = Acquisition_Parameters()
        self.treatments = Treatments()
        
        # Load the spectrum and image if they exist
        if dictionary is not None:            
            self.load_data_from_dictionary(dictionary)
        else:
            apply_treatments = False
        if image is not None:
            self.load_image(image)
        else:
            self.image=None

        # Perform treatments if pretreatments is True
        if apply_treatments:
            # Corrects the readout if the readout file is provided
            if dark_current is not None:
                self.dark_current = Spectrum(dark_current, 
                apply_treatments = False)
                self._process_dark_current()
                self.dark_current_correction()
            
            if readout is not None:
                self.readout = Spectrum(readout, 
                dark_current = dark_current)
                self._process_readout()
                self.readout_correction()

            if gain is not None:
                self.gain_correction = Spectrum(gain, apply_treatments = False)
                self.correct_gain()
            # Corrects the gain of the acquisition system

    def __add__(self, spectrum):
        if hasattr(self, '_splitting_steps') is False:
            self._splitting_steps = [self.data_cube.shape[2]]
        if 'unfolded' in self.history:
            self.fold()
        if 'unfolded' in spectrum.history:
            self.fold()
        if self.data_cube.shape[0] == spectrum.data_cube.shape[0]:
            if (self.energyscale == spectrum.energyscale):
                if self.data_cube.shape[2] > 1 or \
                spectrum.data_cube.shape[2] > 1:
                    if self.data_cube.shape[1] == spectrum.data_cube.shape[1]:
                        new_dc = np.concatenate((self.data_cube, 
                        spectrum.data_cube),2)
                        new_spectrum = Spectrum(
                        {'calibration' : {'data_cube' : new_dc}})
                        new_spectrum.get_calibration_from(self)
                        new_spectrum._splitting_steps = self._splitting_steps
                        new_spectrum._splitting_steps.append(
                        spectrum.data_cube.shape[2])
                        return new_spectrum
                    else:
                        print "Cannot sum spectra with different x dimensions"
                else:
                    new_dc = np.concatenate((self.data_cube, 
                    spectrum.data_cube),1)
                    new_spectrum = Spectrum(
                        {'calibration' : {'data_cube' : new_dc}})
                    new_spectrum.get_calibration_from(self)
                    
                    return new_spectrum
            else:
                messages.warning_exit(
                "Cannot sum spectra with different energy scales")
        else:
            messages.warning_exit(
            "Cannot sum spectra with different number of channels")

            
    def updateenergy_axis(self):
        '''Creates a new energy axis using the defined energyorigin, energyscale
         and energydimension'''
        self.energy_axis = generate_axis(self.energyorigin, self.energyscale, 
        self.energydimension)
        self._replot()

    def set_new_calibration(self,energy_origin, energy_scale):
        '''Updates the energy origin and scale and the energy axis with the 
        given values
        Parameters:
        -----------
        energy_origin : float
        energy_scale : float 
        '''
        self.energyorigin = energy_origin
        self.energyscale = energy_scale
        self.updateenergy_axis()
        
    def energy2index(self, energy):
        '''
        Return the index for the given energy if between the limits,
        otherwise it will return either the upper or lower limits
        
        Parameters
        ----------
        energy : float
        
        Returns
        -------
        int
        '''
        if energy is None:
            return None
        else :
            index = int(round((energy-self.energyorigin) / \
            self.energyscale))
            if self.energydimension > index >= 0: 
                return index
            elif index < 0:
                print "Warning: the given energy is bellow the axis limits"
                return 0
            else:
                print "Warning: the given energy is above the axis limits"
                return int(self.energydimension - 1)
                
    def _get_cube(self):
        return self.__cubes[self.current_cube]['data']
    
    def _set_cube(self,arg):
        self.__cubes[self.current_cube]['data'] = arg
    data_cube = property(_get_cube,_set_cube)
    
    def __new_cube(self, cube, treatment):
        history = copy.copy(self.history)
        history.append(treatment)
        if self.backup_cubes:
            self.__cubes.append({'data' : cube, 'history': history})
        else:
            self.__cubes[-1]['data'] = cube
            self.__cubes[-1]['history'] = history
        self.current_cube = -1
    
    def __call__(self, coordinates = None):
        '''Returns the spectrum at the coordinates given by the choosen cursor
        Parameters
        ----------
        cursor : 1 or 2
        '''
        if coordinates is None:
            coordinates = self.coordinates

        dc = self.data_cube[:, coordinates.ix, coordinates.iy]
        return dc
    
    def get_dimensions_from_cube(self):
        '''Get the dimension parameters from the data_cube. Useful when the 
        data_cube was externally modified, or when the SI was not loaded from
        a file
        '''
        dc = self.data_cube
        # Make the data_cube 3D if it is not
        if len(dc.shape) == 1:
            self.data_cube = dc.reshape((-1,1,1))
        elif len(dc.shape) == 2:
            self.data_cube = dc.reshape((dc.shape + (1,)))
            
        self.energydimension = self.data_cube.shape[0]
        self.xdimension = self.data_cube.shape[1]
        self.ydimension = self.data_cube.shape[2]
        self.updateenergy_axis()
        controllers.coordinates_controller.assign_coordinates(self)
        self._replot()
        
    # Transform ________________________________________________________________
    def delete_spectrum(self, index):
        '''Remove a spectrum from a Line Spectrum
        Parameters
        ----------
        index: int
            The index of the spectrum to remove
        '''
        self.delete_column(index)
        self._replot()
        
    def delete_column(self, index, until = None):
        '''
        Removes a column or a range of rows.
        Parameters
        ----------
        index : int
            Index of the first (or only) column to delete
        until: int
            Index of the last column of the range to delete
        '''
        width = 1
        if until:
            if until < index:
                until, index = index, until
            width = until - index
        for cube in self.__cubes:
            cube['data'] = np.hstack((cube['data'][:, None:index, :], 
            cube['data'][:, index + width:None, :]))
            self.get_dimensions_from_cube()
        self._replot()
        
    def delete_row(self, index, until = None):
        '''
        Removes a row or a range of rows.
        Parameters
        ----------
        index : int
            Index of the first (or only) row to delete
        until: int
            Index of the last column of the row to delete
        '''
        width = 1
        if until:
            if until < index:
                until, index = index, until
            width = until - index
        for cube in self.__cubes:
            _cube = np.swapaxes(cube['data'].copy(), 1, 2)
            cube['data'] = np.swapaxes(np.hstack((_cube[:, None:index, :], 
            _cube[:, index + width:None, :])),1,2)
        self.get_dimensions_from_cube()

        

    def spatial_crop(self, ix1 = None, iy1 = None, ix2 = None, iy2 = None):
        '''Crops the SI with the given indexes.
        ix1 : int
        ix2 : int
        iy1 : int
        iy2 : int
        '''
        print "Cropping the SI from (%s, %s) to (%s, %s)" % (
        ix1, iy1, ix2, iy2)
        for cube in self.__cubes:
            cube['data'] = cube['data'][:,ix1:ix2, iy1:iy2]
        if self.image:
            self.image.crop(ix1, iy1, ix2, iy2)
        self.get_dimensions_from_cube()
        self._replot()
        
    def energy_crop(self, _from = None, to = None, in_energy_units = False):
        '''Crop the spectrum on energy
        Parameters
        ----------
        _from : int or float
            Starting channel index or energy
        to :  int or float
            End channel index or energy
        in_energy_units : bool
            By default the `from` are `to` are treated as energy index. If 
            `in_energy_units` the values are treated as energy.
        '''
        if in_energy_units:
            _from = self.energy2index(_from)
            to = self.energy2index(to)
        for cube in self.__cubes:
            cube['data'] = cube['data'][_from: to]
        self.energy_axis = self.energy_axis[_from: to]
        self.energyorigin = self.energy_axis[0]
        self.get_dimensions_from_cube()

           
    def roll_xy(self, n_x, n_y = 1):
        '''Roll over the x axis n_x positions and n_y positions the former rows 
        
        Parameters
        ----------
        n_x : int
        n_y : int
        
        Note: Useful to correct the SI column storing bug in Marcel's 
        acquisition routines.
        '''
        self.data_cube = np.roll(self.data_cube, n_x, 1)
        self.data_cube[:,:n_x,:] = np.roll(self.data_cube[:,:n_x,:],n_y,2)
        self._replot()
    
    def swap_x_y(self):
        '''Swaps the x and y axes'''
        
        print "Swapping x and y"
        data = self.data_cube.swapaxes(1,2)
        self.yorigin, self.xorigin = self.xorigin, self.yorigin
        self.yscale, self.xscale = self.xscale, self.yscale
        self.yunits, self.xunits = self.xunits, self.yunits
        self.__new_cube(data, 'x and y swapped')
        self.get_dimensions_from_cube()

        
    def rebin(self, new_shape):
        '''
        Rebins the SI to the new shape
        
        Parameters
        ----------
        new_shape: tuple of int of dimension 3
        '''
        for cube in self.__cubes:
            cube['data'] = rebin(cube['data'],new_shape)
        if self.image:
            self.image.data_cube = \
            rebin(self.image.data_cube, new_shape[1:])
        self.get_dimensions_from_cube()

    
    # Process __________________________________________________________________

    def extract_zero_loss(self, zl = None,right = 0.2,around = 0.05):
        '''
        Zero loss extraction by the reflected-tail or fingerprinting methods.
        
        Creates a new spectrum instance self.zero_loss with the zero loss 
        extracted by the reflected-tail method if no zero loss in the vacuum is 
        provided. Otherwise it use the zero loss fingerprinting method.

        Parameters
        ----------
        zl : str
            name of the zero loss in vacuum file for the fingerprinting method
        right : float
            maximum channel in energy units to use to fit the zero loss in the 
            vacuum. Only has effect for the fingerprinting method.
        around : float
            interval around the origin to remove from the fit for the of the 
            zero loss in the vacuum. Only has effect for the fingerprinting 
            method.
        
        Notes
        -----
        It is convenient to align the SI and correct the baseline before using
        this method.
        '''

        print "Extracting the zero loss"
        if zl is None: # reflected-tail
            # Zero loss maximum
            i0 = self.data_cube[:,0,0].argmax(0)
            # FWQM from the first spectrum in channels
            # Search only 2eV around the max to avoid counting the plasmons
            # in thick samples
            i_range = int(round(2. / self.energyscale))
            fwqm_bool = self.data_cube[i0-i_range:i0+i_range,0,0] > \
            0.25 * self.data_cube[i0,0,0]
            ch_fwqm = len(fwqm_bool[fwqm_bool])
            self.zero_loss = copy.deepcopy(self)
            data = self.zero_loss.data_cube
            canvas = np.zeros(data.shape)
            # Reflect the tail
            width = int(round(1.5 * ch_fwqm))
            canvas[i0 + width : 2 * i0 + 1,:,:] = \
            data[i0 - width::-1,:,:]
            # Remove the "background" = mean of first 4 channels and reflects the
            # tail
            bkg = np.mean(data[0:4])
            canvas -= bkg
            # Scale the extended tail with the ratio obtained from
            # 2 overlapping channels
            ch = i0 + width
            ratio = np.mean(data[ch: ch + 2] / canvas[ch: ch + 2], 0)
            for ix in range(data.shape[1]):
                for iy in range(data.shape[2]):
                    canvas[:,ix,iy] *= ratio[ix,iy]
            # Copy the extension
            data[i0 + width:] = canvas[i0 + width:]
        else:
            import components
            fp = components.ZL_Fingerprinting(zl)
            m = Model(self,False)
            m.append(fp)
            m.set_energy_region(None,right)
            m.remove_data_range(-around,around)
            m.multifit()
            self.zero_loss = copy.deepcopy(self)
            self.zero_loss.data_cube = m.model_cube
            self.zl_substracted = copy.deepcopy(self)
            self.zl_substracted.data_cube -= self.zero_loss.data_cube
        self._replot()
        
    def _process_gain_correction(self):
        gain = self.gain_correction
        # Check if the spectrum has the same number of channels:
        if self.data_cube.shape[0] != gain.data_cube.shape[0]:
            print 
            messages.warning_exit(
            "The gain and spectrum don't have the same number of channels")
        dc = gain.data_cube.copy()
        dc = dc.sum(1).mean(1)
        dc /= dc.mean()
        gain.normalized_gain = dc

    def _process_readout(self):
        '''Readout conditioning
        
        Checks if the readout file provided contains more than one spectrum.
        If that is the case, it makes the average and produce a single spectrum
        Spectrum object to feed the correct spectrum function'''
        channels = self.readout.data_cube.shape[0]
        if self.readout.data_cube.shape[1:]  > (1, 1):
            self.readout.data_cube = np.average(
            np.average(self.readout.data_cube,1),1).reshape(channels, 1, 1)
            self.readout.get_dimensions_from_cube()
            self.readout.set_new_calibration(0,1)
            if self.readout.dark_current:
                self.readout._process_dark_current()
                self.readout.dark_current_correction()

    def _process_dark_current(self):
        '''Dark current conditioning.
        
        Checks if the readout file provided contains more than one spectrum.
        If that is the case, it makes the average and produce a single spectrum
        Spectrum object to feed the correct spectrum function. If 
        a readout correction is provided, it corrects the readout in the dark
        current spim.'''
        if self.dark_current.data_cube.shape[1:]  > (1, 1):
            self.dark_current.data_cube = np.average(
            np.average(self.dark_current.data_cube,1),1).reshape((-1, 1, 1))
            self.dark_current.get_dimensions_from_cube()
            self.dark_current.set_new_calibration(0,1)


    # Elements _________________________________________________________________
    def add_elements(self, elements, include_pre_edges = False):
        '''Declare the elements present in the SI.
        
        Instances of components.edge.Edge for the current energy range will be 
        created automatically and add to self.subshell.
        
        Parameters
        ----------
        elements : tuple of strings
            The strings must represent a chemical element.
        include_pre_edges : bool
            If True, the ionization edges with an onset below the lower energy 
            limit of the SI will be incluided
        '''
        for element in elements:
            self.elements.add(element)
        self.generate_subshells(include_pre_edges)
        
    def generate_subshells(self, include_pre_edges = False):
        '''Calculate the subshells for the current energy range for the elements
         present in self.elements
         
        Parameters
        ----------
        include_pre_edges : bool
            If True, the ionization edges with an onset below the lower energy 
            limit of the SI will be incluided
        '''
        if not include_pre_edges:
            start_energy = self.energy_axis[0]
        else:
            start_energy = 0.
        end_energy = self.energy_axis[-1]
        for element in self.elements:
            e_shells = list()
            for shell in edges_dict[element]['subshells']:
                if shell[-1] != 'a':
                    if start_energy <= \
                    edges_dict[element]['subshells'][shell]['onset_energy'] \
                    <= end_energy :
                        subshell = '%s_%s' % (element, shell)
                        if subshell not in self.subshells:
                            print "Adding %s subshell" % (subshell)
                            self.subshells.add('%s_%s' % (element, shell))
                            e_shells.append(subshell)
            if len(e_shells) > 0: 
                self.generate_edges(e_shells)
    
    def generate_edges(self, e_shells, copy2interactive_ns = True):
        '''Create the Edge instances and configure them appropiately
        Parameters
        ----------
        e_shells : list of strings
        copy2interactive_ns : bool
            If True, variables with the format Element_Shell will be created in
            IPython's interactive shell
        '''
        e_shells.sort()
        master_edge = Edge(e_shells.pop())
        self.edges.append(master_edge)
        interactive_ns[self.edges[-1].__repr__()] = self.edges[-1]
        element = self.edges[-1].__repr__().split('_')[0]
        interactive_ns[element] = []
        interactive_ns[element].append(self.edges[-1])
        while len(e_shells) > 0:
            self.edges.append(Edge(e_shells.pop()))
            self.edges[-1].intensity.twin = master_edge.intensity
            self.edges[-1].delta.twin = master_edge.delta
            self.edges[-1].freedelta = False
            interactive_ns[self.edges[-1].__repr__()] = self.edges[-1]
            interactive_ns[element].append(self.edges[-1])

    # History _______________________________________________________________
    def _get_history(self):
        return self.__cubes[self.current_cube]['history']
    def _set_treatment(self,arg):
        self.__cubes[self.current_cube]['history'].append(arg)
        
    history = property(_get_history,_set_treatment)
    
    def print_history(self):
        '''Prints the history of the SI to the stdout'''
        i = 0
        print
        print "Cube\tHistory"
        print "----\t----------"
        print
        for cube in self.__cubes:
            print i,'\t', cube['history']
            i+=1
            
    # Transform
    # TODO check!!
    def split_in(self, number_of_parts = None, steps = None, 
    direction = 'rows'):
        '''Splits the SI
        
        The split can be defined either by the `number_of_parts` or by the 
        `steps` size.
        
        Parameters
        ----------
        number_of_parts : int
            Number of parts in which the SI will be splitted
        steps : int
            Size of the splitted parts
        direction : {'rows', 'columns'}
            The direction of splitting.
            
        Return
        ------
        tuple with the splitted SIs
        '''
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
        shape = self.data_cube.shape
        if direction == 'rows':
            if steps is None:
                rounded = (shape[2] - (shape[2] % number_of_parts))
                step =  rounded / number_of_parts
                cut_node = range(0,rounded+step,step)
            else:
                cut_node = np.array([0] + steps).cumsum()
            for i in range(len(cut_node)-1):
                s = copy.deepcopy(self)
                for cube in s.__cubes:
                    cube['data'] = cube['data'][:,:,cut_node[i]:cut_node[i+1]]
                s.get_dimensions_from_cube()
                splitted.append(s)
            return splitted
        if direction == 'columns':
            if steps is None:
                rounded = (shape[1] - (shape[1] % number_of_parts))
                step =  rounded / number_of_parts
                cut_node = range(0,rounded+step,step)
            else:
                cut_node = np.array([0] + steps).cumsum()
            for i in range(len(cut_node)-1):
                s = copy.deepcopy(self)
                s.data_cube = self.data_cube[:,cut_node[i]:cut_node[i+1], :]
                s.get_dimensions_from_cube()
                splitted.append(s)
            return splitted
    # TODO: maybe move to MVA?
            
    def unfold(self):
        '''If the SI dimension is > 2, it folds it to dimension 2'''
        if self.xdimension > 1 and self.ydimension > 1:
            self.shape_before_folding = list(self.data_cube.shape)
            for cube in self.__cubes:
                cube['data'] = cube['data'].reshape((self.energydimension,-1,1), 
                order = 'F')
                cube['history'].append('unfolded')
            self.get_dimensions_from_cube()
            self._replot()
            print "\nSI unfolded"
            
        else:
            print "Nothing done, cannot unfold an 1D SI"
            
    def fold(self):
        '''If the SI was previously unfolded, folds it'''
        if 'unfolded' in self.history:
            # Just in case the number of channels have changed...
            self.shape_before_folding[0] = self.energydimension
            for cube in self.__cubes:
                cube['data'] = cube['data'].reshape(self.shape_before_folding, 
                order = 'F')
                cube['history'].remove('unfolded')
            self.get_dimensions_from_cube()
            print "\nSI folded back"
        else:
            print "Nothing done, the SI was not unfolded"
    def energy_center(self):
        '''Substract the mean energy pixel by pixel'''
        print "\nCentering the energy axis"
        self._energy_mean = np.mean(self.data_cube, 0)
        data = (self.data_cube - self._energy_mean)
        self.__new_cube(data, 'energy centering')
        self._replot()
        
    def undo_energy_center(self):
        if hasattr(self,'_energy_mean'):
            data = (self.data_cube + self._energy_mean)
            self.__new_cube(data, 'undo energy centering')
            self._replot()
        
    def variance2one(self):
        # Whitening
        data = copy.deepcopy(self.data_cube)
        self._std = np.std(data, 0)
        data /= self._std
        self.__new_cube(data, 'variance2one')
        self._replot()
        
    def undo_variance2one(self):
        if hasattr(self,'_std'):
            data = (self.data_cube * self._std)
            self.__new_cube(data, 'undo variance2one')
            self._replot()
                    
    def correct_bad_pixels(self, indexes):
        '''Substitutes the energy channels by the average of the 
        adjencent channels
        Parameters
        ----------
        indexes : tuple of int
        '''
        data = copy.copy(self.data_cube)
        print "Correcting bad pixels of the spectrometer"
        for channel in indexes:
            data[channel,:,:] = (data[channel-1,:,:] + \
            data[channel+1,:,:]) / 2
        self.__new_cube(data, 'bad pixels correction')
        self._replot()
    
    def remove_background(self, start_energy = None, mask = None):
        '''Removes the power law background of the EELS SI if the present 
        elements were defined.
        
        It stores the background in self.background.
        
        Parameters
        ----------
        start_energy : float
            The starting energy for the fitting routine
        mask : boolean numpy array
        '''
        from spectrum import Spectrum
        if mask is None:
            mask = np.ones(self.data_cube.shape[1:], dtype = 'bool')
        m = Model(self)
        m.fit_background(startenergy = start_energy, type = 'multi', 
        mask = mask)
        m.model_cube[:, mask == False] *= 0
        self.background = Spectrum()
        self.background.data_cube = m.model_cube
        self.background.get_dimensions_from_cube()
        utils.copy_energy_calibration(self, self.background)
#        self.background.get_calibration_from()
        print "Background stored in self.background"
        self.__new_cube(self.data_cube[:] - m.model_cube[:], 
        'background removal')
        self._replot()
        
    def normalize(self, value = 1):
        '''Make the sum of each spectrum equal to a given value
        Parameters:
        -----------
        value : float
        '''
        data = copy.copy(self.data_cube)
        print "Normalizing the spectrum/a"
        for ix in range(0,self.xdimension):
            for iy in range(0,self.ydimension):
                sum_ = np.sum(data[:,ix,iy])
                data[:,ix,iy] *= (value / sum_)
        self.__new_cube(data, 'normalization')
        self._replot()
        
    def calculate_I0(self, threshold = None):
        '''Estimates the integral of the ZLP from a LL SI
        
        The value is stored in self.I0 as an Image.
        
        Parameters
        ----------
        thresh : float or None
            If float, it estimates the intensity of the ZLP as the sum 
            of all the counts of the SI until the threshold. If None, it 
            calculates the sum of the ZLP previously stored in 
            self.zero_loss
        '''
        if threshold is None:
            if self.zero_loss is None:
                messages.warning_exit(
                "Please, provide a threshold value of define the " 
                "self.zero_loss attribute by, for example, using the "
                "extract_zero_loss method")
            else:
                self.I0 = Image(dc = self.zero_loss.sum(0))
        else:
            threshold = self.energy2index(threshold)
            self.I0 = Image(dc = self.data_cube[:threshold,:,:].sum(0)) 
        
    def correct_gain(self):
        '''Apply the gain correction stored in self.gain_correction
        '''
        if not self.treatments.gain:
            self._process_gain_correction()
            gain = self.gain_correction
            print "Applying gain correction"
            # Gain correction
            data = np.zeros(self.data_cube.shape)
            for ix in range(0, self.xdimension):
                for iy in range(0, self.ydimension):
                    np.divide(self.data_cube[:,ix,iy], 
                    gain.normalized_gain, 
                    data[:,ix,iy])
            self.__new_cube(data, 'gain correction')
            self.treatments.gain = 1
            self._replot()
        else:
            print "Nothing done, the SI was already gain corrected"

    def correct_baseline(self, kind = 'pixel', positive2zero = True, 
    averaged = 10, fix_negative = True):
        '''Set the minimum value to zero
        
        It can calculate the correction globally or pixel by pixel.
        
        Parameters
        ----------
        kind : {'pixel', 'global'}
            if 'pixel' it calculates the correction pixel by pixel.
            If 'global' the correction is calculated globally.
        positive2zero : bool
            If False it will only set the baseline to zero if the 
            minimum is negative
        averaged : int
            If > 0, it will only find the minimum in the first and last 
            given channels
        fix_negative : bool
            When averaged, it will take the abs of the data_cube to assure
            that no value is negative.
        
        '''
        data = copy.copy(self.data_cube)
        print "Correcting the baseline of the low loss spectrum/a"
        if kind == 'global':
            if averaged == 0:
                minimum = data.min()
            else:
                minimum = np.vstack(
                (data[:averaged,:,:], data[-averaged:,:,:])).min()
            if minimum < 0. or positive2zero is True:
                data -= minimum
        elif kind == 'pixel':
            if averaged == 0:
                minimum = data.min(0).reshape(
            (1,data.shape[1], data.shape[2]))
            else:
                minimum = np.vstack((data[:averaged,:,:], data[-averaged:,:,:])
                ).min(0).reshape((1,data.shape[1], data.shape[2]))
            mask = np.ones(data.shape[1:], dtype = 'bool')
            if positive2zero is False:
                mask[minimum.squeeze() > 0] = False
            data[:,mask] -= minimum[0,mask]
        else:
            messages.warning_exit(
            "Wrong kind keyword. Possible values are pixel or global")
        
        if fix_negative:
            data = np.abs(data)
        self.__new_cube(data, 'baseline correction')
        self._replot()

    def readout_correction(self):
        if not self.treatments.readout:
            if hasattr(self, 'readout'):
                data = copy.copy(self.data_cube)
                print "Correcting the readout"
                for ix in range(0,self.xdimension):
                    for iy in range(0,self.ydimension):
                        data[:, ix, iy] -= self.readout.data_cube[:,0,0]
                self.__new_cube(data, 'readout correction')
                self.treatments.readout = 1
                self._replot()
            else:
                print "To correct the readout, please define the readout attribute"
        else:
            print "Nothing done, the SI was already readout corrected"

    def dark_current_correction(self):
        '''Apply the dark_current_correction stored in self.dark_current'''
        if self.treatments.dark_current:
            print "Nothing done, the dark current was already corrected"
        else:
            ap = self.acquisition_parameters
            if hasattr(self, 'dark_current'):
                if (ap.exposure is not None) and \
                (self.dark_current.acquisition_parameters.exposure):
                    if (ap.readout_frequency is not None) and \
                    (ap.blanking is not None):
                        if not self.acquisition_parameters.blanking:
                            exposure = ap.exposure + self.data_cube.shape[0] * \
                            ap.ccd_height / (ap.binning * ap.readout_frequency)
                            ap.effective_exposure = exposure
                        else:
                            exposure = ap.exposure
                    else:
                        print \
    '''Warning: no information about binning and readout frequency found. Please 
    define the following attributes for a correct dark current correction:
    exposure, binning, readout_frequency, ccd_height, blanking
    The correction proceeds anyway
    '''
                            
                        exposure = self.acquisition_parameters.exposure
                    data = copy.copy(self.data_cube)
                    print "Correcting the dark current"
                    self.dark_current.data_cube[:,0,0] *= \
                    (exposure / self.dark_current.acquisition_parameters.exposure)
                    data -= self.dark_current.data_cube
                    self.__new_cube(data, 'dark current correction')
                    self.treatments.dark_current = 1
                    self._replot()
                else:
                    
                    messages.warning_exit(
                    "Please define the exposure attribute of the spectrum"
                    "and its dark_current")
            else:
                messages.warning_exit(
               "To correct the readout, please define the dark_current " \
                "attribute")
    
    def add_poissonian_noise(self):
        '''Add Poissonian noise to the SI'''
        self.__new_cube(np.random.poisson(self.data_cube).astype('float64'), 
        'poissonian noise')
        self._replot()

    def add_gaussian_noise(self, std):
        '''Add Gaussian noise to the SI
        Parameters
        ----------
        std : float
        
        See also
        --------
        Spectrum.simulate
        '''
        self.__new_cube(np.random.normal(self.data_cube,std), 'gaussian_noise')
        self._replot()
        
    def align_with_map(self, shift_map, cut = 'left', 
    interpolation_kind = 'linear'):
        '''Shift each spectrum by the energy increment indicated in an array.
        
        The shifts are relative. The direction of the shift will be determined 
        by wether we prefer to crop the SI on the left or on the right
        
        Parameters
        ----------
        shift_map : numpy array
        cut : {'left', 'right'}
        interpolation_kind : str or int
            Specifies the kind of interpolation as a string ('linear',
            'nearest', 'zero', 'slinear', 'quadratic, 'cubic') or as an integer
            specifying the order of the spline interpolator to use.
        '''
        
        dc = self.data_cube
        ea = np.empty(dc.shape)
        if cut == 'left':
            shift_map -= shift_map.max()
        elif cut == 'right':
            shift_map -= shift_map.min()
        else:
            messages.warning_exit(
            "Parameter cut only accepts \'left\' or '\right\'")
        ea[:] = self.energy_axis.reshape((-1,1,1)) + shift_map.reshape(
        (1, dc.shape[1], dc.shape[2]))
        new_dc = np.empty(dc.shape)
        for j in range(dc.shape[2]):
            for  i in range(dc.shape[1]):
                print "(%s, %s)" % (i, j)
                sp = interp1d(self.energy_axis ,dc[:,i,j], bounds_error = False, 
                fill_value = 0, kind = interpolation_kind)
                new_dc[:,i,j] = sp(ea[:,i,j])
        s = Spectrum()
        s.data_cube = new_dc
        utils.copy_energy_calibration(self, s)
        if cut == 'left':
            iE_min = 1 + np.floor(-1*shift_map.min()/self.energyscale)
            print iE_min
            s.energy_crop(int(iE_min),None,False)
        elif cut == 'right':
            iE_max = 1 + np.floor(shift_map.max()/self.energyscale)
            s.energy_crop(None,int(iE_max),False)
        return s
    
    def energy_interpolation(self, E1, E2, xch = 20, kind = 3):
        dc = self.data_cube
        ix1 = self.energy2index(E1)
        ix2 = self.energy2index(E2)
        ix0 = np.clip(ix1 - xch, 0, np.inf)
        ix3 = np.clip(ix2 + xch, 0, len(self.energy_axis)+1)
        for iy in range(dc.shape[2]):
            for ix in range(dc.shape[1]):
                sp = interp1d(range(ix0,ix1) + range(ix2,ix3),
                dc[ix0:ix1,ix,iy].tolist() + dc[ix2:ix3,ix,iy].tolist(), 
                kind = kind)
                dc[ix1:ix2, ix, iy] = sp(range(ix1,ix2))
        
    def _interpolate_spectrum(self,ip, (ix,iy)):
        data = self.data_cube
        ch = self.data_cube.shape[0]
        old_ax = np.linspace(0, 100,ch)
        new_ax = np.linspace(0, 100,ch*ip - (ip-1))
        sp = interp1d(old_ax,data[:,ix,iy])
        return sp(new_ax)
    
    def align(self, energy_range = (None,None), 
    reference_spectrum_coordinates = (0,0), max_energy_shift = None, 
    sync_SI = None, interpolate = True, interp_points = 5, progress_bar = True):
        ''' Align the SI by cross-correlation.
                
        Parameters
        ----------
        energy_range : tuple of floats (E1, E2)
            Restricts to the given range the area of the spectrum used for the 
            aligment.
        reference_spectrum_coordinates : tuple of int (x_coordinate, y_coordinate)
            The coordianates of the spectrum that will be taken as a reference
            to align them all
        max_energy_shift : float
            The maximum energy shift permitted
        sync_SI: Spectrum instance
            Another spectrum instance to align with the same calculated energy 
            shift
        interpolate : bool
        interp_points : int
            Number of interpolation points. Warning: making this number too big 
            can saturate the memory   
        '''
        
        print "Aligning the SI"
        ip = interp_points + 1
        data = self.data_cube
        channel_1 = self.energy2index(energy_range[0])
        channel_2 = self.energy2index(energy_range[1])
        ch, size_x, size_y = data.shape
        channels , size_x, size_y = data.shape
        channels = channel_2 - channel_1
        shift_map = np.zeros((size_x, size_y))
        ref_ix, ref_iy = reference_spectrum_coordinates
        if channel_1 is not None:
            channel_1 *= ip
        if channel_2 is not None:
            channel_2 = np.clip(np.array(channel_2 * ip),a_min = 0, 
            a_max = ch*ip-2)
        if interpolate:
            ref = self._interpolate_spectrum(ip, 
            (ref_ix, ref_iy))[channel_1:channel_2]
        else:
            ref = data[channel_1:channel_2, ref_ix, ref_iy]
        print "Calculating the shift"
        
        if progress_bar is True:
            from progressbar import progressbar
            maxval = max(1,size_x) * max(1,size_y)
            pbar = progressbar(maxval = maxval)
        for iy in range(size_y):
            for ix in range(size_x):
                if progress_bar is True:
                    i = (ix + 1)*(iy+1)
                    pbar.update(i)
                if interpolate:
                    dc = self._interpolate_spectrum(ip, (ix, iy))
                shift_map[ix,iy] = np.argmax(np.correlate(ref, 
                dc[channel_1:channel_2],'full')) - channels + 1
        if progress_bar is True:
            pbar.finish()
        if np.min(shift_map) < 0:
            shift_map -= np.min(shift_map)
        if max_energy_shift:
            max_index = self.energy2index(max_energy_shift)
            if interpolate:
                max_index *= ip
            shift_map.clip(a_max = max_index)
            
        def apply_correction(spectrum):
            data = spectrum.data_cube
            print "Applying the correction"
            if progress_bar is True:
                maxval = max(1,size_x) * max(1,size_y)
                pbar = progressbar(maxval = maxval)
            for iy in range(size_y):
                for ix in range(size_x):
                    if progress_bar is True:
                        i = (ix + 1)*(iy+1)
                        pbar.update(i)

                    if interpolate:
                        sp = spectrum._interpolate_spectrum(ip, (ix, iy))
                        data[:,ix,iy] = np.roll(sp, 
                        int(shift_map[ix,iy]), axis = 0)[::ip]
                        spectrum.updateenergy_axis()
                    else:
                        data[:,ix,iy] = np.roll(data[:,ix,iy], 
                        int(shift_map[ix,iy]), axis = 0)
            if progress_bar is True:
                pbar.finish()
            spectrum.__new_cube(data, 'alignment by cross-correlation')
            if interpolate is True:
                spectrum.energy_crop(shift_map.max()/ip)
            else:
                spectrum.energy_crop(shift_map.max())
        apply_correction(self)

        if sync_SI is not None:
            apply_correction(sync_SI)

        return shift_map

    def find_low_loss_origin(self, sync_SI = None):
        '''Calculate the position of the zero loss origin as the average of the 
        postion of the maximum of all the spectra'''
        old_origin = self.energyorigin
        imax = np.mean(np.argmax(self.data_cube,0))
        self.energyorigin = generate_axis(0, self.energyscale, 
            self.energydimension, imax)[0]
        self.updateenergy_axis()
        if sync_SI:
            sync_SI.energyorigin += self.energyorigin - old_origin
            sync_SI.updateenergy_axis()

    def fourier_log_deconvolution(self):
        '''Performs fourier-log deconvolution of the full SI.
        
        The zero-loss can be specified by defining the parameter 
        self.zero_loss that must be an instance of Spectrum. Otherwise the 
        zero loss will be extracted by the reflected tail method
        '''
        if self.zero_loss is None:
            self.extract_zero_loss()
        z = np.fft.fft(self.zero_loss.data_cube, axis=0)
        j = np.fft.fft(self.data_cube, axis=0)
        j1 = z*np.log(j/z)
        self.__new_cube(np.fft.ifft(j1, axis = 0).real, 
        'fourier-log deconvolution')
        self._replot()
        
    # IO _______________________________________________________________________
    
    def save(self, filename, format = defaults.file_format, msa_format = 'Y', 
    **kwds):
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

    def load_data_from_dictionary(self, dictionary):
        for key in dictionary['calibration']:
            exec('self.%s = dictionary[\'calibration\'][\'%s\']' % (key, key))
        if 'acquisition' in dictionary:
            for key in dictionary['acquisition']:
                exec('self.acquisition_parameters.%s = ' 
                'dictionary[\'acquisition\'][\'%s\']' % (key, key))
        if 'treatments' in dictionary:
            for key in dictionary['treatments']:
                exec('self.treatments.%s = dictionary[\'treatments\'][\'%s\']' \
                % (key, key))
        if 'imported_parameters' in dictionary:
            self.imported_parameters = dictionary['imported_parameters']
        self.get_dimensions_from_cube()
        print "Shape: ", self.data_cube.shape
        print "History:"
        for treatment in self.history:
            print treatment
        controllers.coordinates_controller.assign_coordinates(self)

    def load_image(self,filename):
        print "Loading the image..."
        self.image = file_io.load(filename)
            
    # Info _____________________________________________________________________
    def calculate_thickness(self, method = 'threshold', threshold = 3, 
    factor = 1):
        '''Calculates the thickness from a LL SI.
        
        The resulting thickness map is stored in self.thickness as an image 
        instance. To visualize it: self.thickness.plot()
        
        Parameters
        ----------
        method : {'threshold', 'zl'}
            If 'threshold', it will extract the zero loss by just splittin the 
            spectrum at the threshold value. If 'zl', it will use the 
            self.zero_loss SI (if defined) to perform the calculation.
        threshold : float
            threshold value.
        factor : float
            factor by which to multiple the ZLP
        '''
        print "Calculating the thickness"
        # Create the thickness array
        dc = self.data_cube
        integral = dc.sum(0)
        if method == 'zl':
            if self.zero_loss is None:
                self.extract_zero_loss()
            zl = self.zero_loss.data_cube
            zl_int = zl.sum(0)
            
        elif method == 'threshold':
            ti =self.energy2index(threshold)
            zl_int = dc[:ti,...].sum(0) * factor 
        self.thickness = \
        Image({'calibration' : {'data_cube' : np.log( integral / zl_int)}})
                
    def calculate_FWHM(self, factor = 0.5, channels = 7, der_roots = False):
        '''Use a third order spline interpolation to estimate the FWHM of 
        the zero loss peak.
        
        Parameters:
        -----------
        factor : float < 1
            By default is 0.5 to give FWHM. Choose any other float to give
            find the position of a different fraction of the peak.
        channels : int
            radius of the interval around the origin were the algorithm will 
            perform the estimation.
        der_roots: bool
            If True, compute the roots of the first derivative
            (2 times slower).  
        
        Returns:
        --------
        dictionary. Keys:
            'FWHM' : float
                 width, at half maximum or other fraction as choosen by
            `factor`. 
            'FWHM_E' : tuple of floats
                Coordinates in energy units of the FWHM points.
            'der_roots' : tuple
                Position in energy units of the roots of the first
            derivative if der_roots is True (False by default)
        '''
        ix = self.coordinates.ix
        iy = self.coordinates.iy
        i0 = np.argmax(self.data_cube[:,ix, iy])
        data = self.data_cube[i0 - channels:i0 + channels + 1, ix, iy]
        x = self.energy_axis[i0 - channels:i0 + channels + 1]
        height = np.max(data)
        spline_fwhm = UnivariateSpline(x, data - factor * height)
        pair_fwhm = spline_fwhm.roots()[0:2]
        print spline_fwhm.roots()
        fwhm = pair_fwhm[1] - pair_fwhm[0]
        if der_roots:
            der_x = np.arange(x[0], x[-1] + 1, (x[1] - x[0]) * 0.2)
            derivative = spline_fwhm(der_x, 1)
            spline_der = UnivariateSpline(der_x, derivative)
            return {'FWHM' : fwhm, 'pair' : pair_fwhm, 
            'der_roots': spline_der.roots()}
        else:
            return {'FWHM' : fwhm, 'FWHM_E' : pair_fwhm}
        
    def gaussian_filter(self, FWHM):
        '''Applies a Gaussian filter in the energy dimension.
        
        Parameters
        ----------
        FWHM : float

        See also
        --------
        Spectrum.simulate
        '''
        if FWHM > 0:
            self.data_cube = gaussian_filter1d(self.data_cube, axis = 0, 
            sigma = FWHM/2.35482)

            
    def add_energy_instability(self, std):
        '''Introduce random energy instability
        
        Parameters
        ----------
        std : float
            std in energy units of the energy instability.
        See also
        --------
        Spectrum.simulate
        '''
        if abs(std) > 0:
            delta_map = np.random.normal(
            size = (self.xdimension, self.ydimension), 
            scale = abs(std))
        else:
            delta_map = np.zeros((self.xdimension, 
                    self.ydimension))
        for edge in self.edges:
            edge.delta.map = delta_map
            edge.delta.already_set_map = np.ones((self.xdimension, 
            self.ydimension), dtype = 'Bool')
        return delta_map
    
    def create_data_cube(self):
        '''Generate an empty data_cube from the dimension parameters
        
        The parameters self.energydimension, self.xdimension and 
        self.ydimension will be used to generate an empty data_cube.
        
        See also
        --------
        Spectrum.simulate
        '''
        self.data_cube = np.zeros((self.energydimension, self.xdimension, 
        self.ydimension))
        self.get_dimensions_from_cube()
        self.updateenergy_axis()
        
        
    def simulate(self, maps = None, energy_instability = 0, 
    min_intensity = 0., max_intensity = 1.):
        '''Create a simulated SI.
        
        If an image is provided, it will use each RGB color channel as the 
        intensity map of each three elements that must be previously defined as 
        a set in self.elements. Otherwise it will create a random map for each 
        element defined.
        
        Parameters:
        -----------
        maps : list/tuple of arrays
            A list with as many arrays as elements are defined.
        energy_instability : float
            standard deviation in energy units of the energy instability.
        min_intensity : float
            minimum edge intensity
        max_intensity : float
            maximum edge intensity
            
        Returns:
        --------
        
        If energy_instability != 0 it returns the energy shift map
        '''
        if maps is not None:
            self.xdimension = maps[0].shape[0]
            self.ydimension = maps[0].shape[1]
            self.xscale = 1.
            self.yscale = 1.
            i = 0
            if energy_instability > 0:
                delta_map = np.random.normal(np.zeros((self.xdimension, 
                self.ydimension)), energy_instability)
            for edge in self.edges:
                edge.fs_state = False
                if not edge.intensity.twin:
                    edge.intensity.map = maps[i]
                    edge.intensity.already_set_map = np.ones((
                    self.xdimension, self.ydimension), dtype = 'Bool')
                    i += 1
            if energy_instability != 0:
                instability_map = self.add_energy_instability(energy_instability)
            for edge in self.edges:
                edge.charge_value_from_map(0,0)
            self.create_data_cube()
            self.model = Model(self, auto_background=False)
            self.model.charge()
            self.model.generate_cube()
            self.data_cube = self.model.model_cube
            self.type = 'simulation'
        else:
            print "No image defined. Producing a gaussian mixture image of the \
            elements"
            i = 0
            if energy_instability:
                delta_map = np.random.normal(np.zeros((self.xdimension, 
                self.ydimension)), energy_instability)
                print delta_map.shape
            size = self.xdimension * self.ydimension
            for edge in self.edges:
                edge.fs_state = False
                if not edge.intensity.twin:
                    edge.intensity.map = np.random.uniform(0, max_intensity, 
                    size).reshape(self.xdimension, self.ydimension)
                    edge.intensity.already_set_map = np.ones((self.xdimension, 
                    self.ydimension), dtype = 'Bool')
                    if energy_instability:
                        edge.delta.map = delta_map
                        edge.delta.already_set_map = np.ones((self.xdimension, 
                        self.ydimension), dtype = 'Bool')
                    i += 1
            self.create_data_cube()
            self.model = Model(self, auto_background=False)
            self.model.generate_cube()
            self.data_cube = self.model.model_cube
            self.type = 'simulation'
        if energy_instability != 0:
            return instability_map
            
    def power_law_extension(self, interval, new_size = 1024, 
    to_the = 'right'):
        '''Extend the SI with a power law.
        
        Fit the SI with a power law in the given interval and use the result 
        to extend it to the left of to the right.
        
        Parameters
        ----------
        interval : tuple
            Interval to perform the fit in energy units        
        new_size : int
            number of channel after the extension.
        to_the : {'right', 'left'}
            extend the SI to the left or to the right
        '''
        left, right = interval
        s = self.data_cube.shape
        original_size = s[0]
        if original_size >= new_size:
            print "The new size (in channels) must be bigger than %s" % \
            original_size
        new_cube = np.zeros((new_size, s[1], s[2]))
        iright = self.energy2index(right)
        new_cube[:iright,:,:] = self.data_cube[:iright,:,:]
        self.data_cube = new_cube
        self.get_dimensions_from_cube()
        m = Model(self, False, auto_add_edges = False)
        pl = PowerLaw()
        m.append(pl)
        m.set_energy_region(left,right)
        m.multifit(grad = True)
        self.data_cube[iright:,:,:] = m.model_cube[iright:,:,:]
        
    def hanning_taper(self, side = 'left', channels = 20,offset = 0):
        '''Hanning taper
        
        Parameters
        ----------
        side : {'left', 'right'}
        channels : int
        offset : int
        '''        
        dc = self.data_cube
        if side == 'left':
            dc[offset:channels+offset,:,:] *= \
            (np.hanning(2*channels)[:channels]).reshape((-1,1,1))
            dc[:offset,:,:] *= 0. 
        if side == 'right':
            dc[-channels-offset:-offset,:,:] *= \
            (np.hanning(2*channels)[-channels:]).reshape((-1,1,1))
            dc[-offset:,:,:] *= 0. 
        
    def remove_spikes(self, threshold = 2200, subst_width = 5, 
    coordinates = None):
        '''Remove the spikes in the SI.
        
        Detect the spikes above a given threshold and fix them by interpolating 
        in the give interval. If coordinates is given, it will only remove the 
        spikes for the specified spectra.
        
        Paramerters:
        ------------
        threshold : float
            A suitable threshold can be determined with 
            Spectrum.spikes_diagnosis
        subst_width : tuple of int or int
            radius of the interval around the spike to substitute with the 
            interpolation. If a tuple, the dimension must be equal to the 
            number of spikes in the threshold. If int the same value will be 
            applied to all the spikes.
        
        See also
        --------
        Spectrum.spikes_diagnosis, Spectrum.plot_spikes
        '''
        int_window = 20
        dc = self.data_cube
        der = np.diff(dc,1,0)
        E_ax = self.energy_axis
        n_ch = len(E_ax)
        index = 0
        if coordinates is None:
            for i in range(dc.shape[1]):
                for j in range(dc.shape[2]):
                    if der[:,i,j].max() >= threshold:
                        print "Spike detected in (%s, %s)" % (i, j)
                        argmax = der[:,i,j].argmax()
                        if hasattr(subst_width, '__iter__'):
                            subst__width = subst_width[index]
                        else:
                            subst__width = subst_width
                        lp1 = np.clip(argmax - int_window, 0, n_ch)
                        lp2 = np.clip(argmax - subst__width, 0, n_ch)
                        rp2 = np.clip(argmax + int_window, 0, n_ch)
                        rp1 = np.clip(argmax + subst__width, 0, n_ch)
                        x = np.hstack((E_ax[lp1:lp2], E_ax[rp1:rp2]))
                        y = np.hstack((dc[lp1:lp2,i,j], dc[rp1:rp2,i,j])) 
                        # The weights were commented because the can produce nans
                        # Maybe it should be an option?
                        intp =UnivariateSpline(x,y) #,w = 1/np.sqrt(y))
                        x_int = E_ax[lp2:rp1+1]
                        dc[lp2:rp1+1,i,j] = intp(x_int)
                        index += 1
        else:
            for spike_spectrum in coordinates:
                i, j = spike_spectrum
                print "Spike detected in (%s, %s)" % (i, j)
                argmax = der[:,i,j].argmax()
                if hasattr(subst_width, '__iter__'):
                    subst__width = subst_width[index]
                else:
                    subst__width = subst_width
                lp1 = np.clip(argmax - int_window, 0, n_ch)
                lp2 = np.clip(argmax - subst__width, 0, n_ch)
                rp2 = np.clip(argmax + int_window, 0, n_ch)
                rp1 = np.clip(argmax + subst__width, 0, n_ch)
                x = np.hstack((E_ax[lp1:lp2], E_ax[rp1:rp2]))
                y = np.hstack((dc[lp1:lp2,i,j], dc[rp1:rp2,i,j])) 
                # The weights were commented because the can produce nans
                # Maybe it should be an option?
                intp =UnivariateSpline(x,y) # ,w = 1/np.sqrt(y))
                x_int = E_ax[lp2:rp1+1]
                dc[lp2:rp1+1,i,j] = intp(x_int)
                index += 1
                
    def spikes_diagnosis(self):
        '''Plots a histogram to help in choosing the threshold for spikes
        removal.
        See also
        --------
        Spectrum.remove_spikes, Spectrum.plot_spikes
        '''
        dc = self.data_cube
        der = np.diff(dc,1,0)
        plt.figure()
        plt.hist(np.ravel(der.max(0)),100)
        plt.xlabel('Threshold')
        plt.ylabel('Counts')
        plt.draw()
        
    def plot_spikes(self, threshold = 2200):
        '''Plot the spikes in the given threshold
        
        Parameters
        ----------
        threshold : float
        
        Returns
        -------
        list of spikes coordinates
        
        See also
        --------
        Spectrum.remove_spikes, Spectrum.spikes_diagnosis
        '''
        dc = self.data_cube
        der = np.diff(dc,1,0)
        index = 0
        spikes =[]
        for i in range(dc.shape[1]):
            for j in range(dc.shape[2]):
                if der[:,i,j].max() >= threshold:
                    print "Spike detected in (%s, %s)" % (i, j)
                    spikes.append((i,j))
                    argmax = der[:,i,j].argmax()
                    toplot = dc[np.clip(argmax-100,0,dc.shape[0]-1): 
                    np.clip(argmax+100,0,dc.shape[0]-1), i, j]
                    plt.figure()
                    plt.step(range(len(toplot)), toplot)
                    plt.title(str(index))
                    index += 1
        return spikes
                        
    def build_SI_from_substracted_zl(self,ch, taper_nch = 20):
        '''Modify the SI to have fit with a smoothly decaying ZL
        
        Parameters
        ----------
        ch : int
            channel index to start the ZL decay to 0
        taper_nch : int
            number of channels in which the ZL will decay to 0 from `ch`
        '''
        sp = copy.deepcopy(self)
        dc = self.zl_substracted.data_cube.copy()
        dc[0:ch,:,:] *= 0
        for i in range(dc.shape[1]):
            for j in range(dc.shape[2]):
                dc[ch:ch+taper_nch,i,j] *= np.hanning(2 * taper_nch)[:taper_nch]
        sp.zl_substracted.data_cube = dc.copy()
        dc += self.zero_loss.data_cube
        sp.data_cube = dc.copy()
        return sp
                
    def sum_every_n(self,n):
        '''Bin a line spectrum
        
        Parameters
        ----------
        step : float
            binning size
        
        Returns
        -------
        Binned line spectrum
        
        See also
        --------
        sum_every
        '''
        dc = self.data_cube
        if dc.shape[1] % n != 0:
            messages.warning_exit(
            "n is not a divisor of the size of the line spectrum\n"
            "Try giving a different n or using sum_every instead")
        size_list = np.zeros((dc.shape[1] / n))
        size_list[:] = n
        return self.sum_every(size_list)
    
    def sum_every(self,size_list):
        '''Sum a line spectrum intervals given in a list and return the 
        resulting SI
        
        Parameters
        ----------
        size_list : list of floats
            A list of the size of each interval to sum.
        
        Returns
        -------
        SI
        
        See also
        --------
        sum_every_n
        '''
        dc = self.data_cube
        dc_shape = self.data_cube.shape
        if np.sum(size_list) != dc.shape[1]:
            messages.warning_exit(
            "The sum of the elements of the size list is not equal to the size" 
            " of the line spectrum")
        new_dc = np.zeros((dc_shape[0], len(size_list), 1))
        ch = 0
        for i in range(len(size_list)):
            new_dc[:,i,0] = dc[:,ch:ch + size_list[i], 0].sum(1)
            ch += size_list[i]
        sp = Spectrum()
        sp.data_cube = new_dc
        sp.get_dimensions_from_cube()
        return sp
    
    def jump_ratio(self, left_interval, right_interval):
        '''Returns the jump ratio in the given intervals
        
        Parameters
        ----------
        left_interval : tuple of floats
            left interval in energy units
        right_interval : tuple of floats
            right interval in energy units
            
        Returns
        -------
        float
        '''
        ilt1 = self.energy2index(left_interval[0])
        ilt2 = self.energy2index(left_interval[1])
        irt1 = self.energy2index(right_interval[0])
        irt2 = self.energy2index(right_interval[1])
        jump_ratio = (self.data_cube[irt1:irt2,:,:].sum(0) \
        / self.data_cube[ilt1:ilt2,:,:].sum(0))
        return jump_ratio
    
    def sum(self, axis):
        '''Sum the SI over the given axis
        
        Parameters
        ----------
        axis : int
        '''
        dc = self.data_cube
        dc = dc.sum(axis)
        dc = dc.reshape(list(dc.shape) + [1,])
        self.data_cube = dc
        self.get_dimensions_from_cube()

    def mean(self, axis):
        '''Average the SI over the given axis
        
        Parameters
        ----------
        axis : int
        '''
        dc = self.data_cube
        dc = dc.mean(axis)
        dc = dc.reshape(list(dc.shape) + [1,])
        self.data_cube = dc
        self.get_dimensions_from_cube()
        
    def roll(self, axis = 2, shift = 1):
        '''Roll the SI. see numpy.roll
        
        Parameters
        ----------
        axis : int
        shift : int
        '''
        self.data_cube = np.roll(self.data_cube, shift, axis)
        self._replot()
        
    def remove_Shirley_background(self, max_iter = 10, eps = 1e-6):
        '''Remove the inelastic background of photoemission SI by the shirley 
        iterative method.
        
        Parameters
        ----------
        max_iter : int
            maximum number of iterations
        eps : float
            convergence limit
        '''
        bg_list = []
        iter = 0
        s = self.data_cube.copy()
        a = s[:3,:,:].mean()
        b = s[-3:,:,:].mean()
        B = b * 0.9 * np.ones(s.shape)
        bg_list.append(B)
        mean_epsilon = 10*eps
        integral = None
        old_integral = None
        while  (mean_epsilon > eps) and (iter < max_iter):
            if integral is not None:
                old_integral = integral
            sb = s - B
            integral = np.cumsum(
            sb[::-1,:,:], axis = 0)[::-1, :, :] * self.energyscale
            B = (a-b)*integral/integral[0,:,:] + b
            bg_list.append(B)
            if old_integral is not None:
                epsilon = np.abs(integral[0,:,:] - old_integral[0,:,:])
                mean_epsilon = epsilon.mean()
            print "iter: %s\t mean epsilon: %s" % (iter, mean_epsilon)
            iter += 1
        self.data_cube = sb
        return epsilon, bg_list
    
    def sum_in_mask(self, mask):
        '''Returns the result of summing all the spectra in the mask.
        
        Parameters
        ----------
        mask : boolean numpy array
        
        Returns
        -------
        Spectrum
        '''
        dc = self.data_cube.copy()
        mask3D = mask.reshape([1,] + list(mask.shape)) * np.ones(dc.shape)
        dc = (mask3D*dc).sum(1).sum(1) / mask.sum()
        s = Spectrum()
        s.data_cube = dc.reshape((-1,1,1))
        s.get_dimensions_from_cube()
        utils.copy_energy_calibration(self,s)
        return s
    
    def correct_dual_camera_step(self, show_lev = False, mean_interval = 3, 
    pca_interval = 20, pcs = 2, normalize_poissonian_noise = False):
        '''Correct the gain difference in a dual camera using PCA.
        
        Parameters
        ----------
        show_lev : boolen
            Plot PCA lev
        mean_interval : int
        pca_interval : int
        pcs : int
            number of principal components
        normalize_poissonian_noise : bool
        ''' 
        # The step is between pixels 1023 and 1024
        pw = pca_interval
        mw = mean_interval
        s = copy.deepcopy(self)
        s.energy_crop(1023-pw, 1023 + pw)
        s.principal_components_analysis(normalize_poissonian_noise)
        if show_lev:
            s.plot_lev()
            pcs = int(raw_input('Number of principal components? '))
        sc = s.pca_build_SI(pcs)
        step = sc.data_cube[(pw-mw):(pw+1),:,:].mean(0) - \
        sc.data_cube[(pw+1):(pw+1+mw),:,:].mean(0)
        self.data_cube[1024:,:,:] += step.reshape((1, step.shape[0], 
        step.shape[1]))
        self._replot()
        return step
    
    def get_calibration_from(self, s):
        '''Copy the calibration from another Spectrum instance
        Parameters
        ----------
        s : spectrum instance
        '''
        utils.copy_energy_calibration(s, self)
    
    def estimate_variance(self, dc = None, gaussian_noise_var = None):
        '''Variance estimation supposing Poissonian noise
        
        Parameters
        ----------
        dc : None or numpy array
            If None the SI is used to estimate its variance. Otherwise, the 
            provided array will be used.   
        Note
        ----
        The gain_factor and gain_offset from the aquisition parameters are used
        '''
        print "Variace estimation using the following values:"
        print "Gain factor = ", self.acquisition_parameters.gain_factor
        print "Gain offset = ", self.acquisition_parameters.gain_offset
        if dc is None:
            dc = self.data_cube
        gain_factor = self.acquisition_parameters.gain_factor
        gain_offset = self.acquisition_parameters.gain_offset
        self.variance = dc*gain_factor + gain_offset
        if self.variance.min() < 0:
            if gain_offset == 0 and gaussian_noise_var is None:
                print "The variance estimation results in negative values"
                print "Maybe the gain_offset is wrong?"
                self.variance = None
                return
            elif gaussian_noise_var is None:
                print "Clipping the variance to the gain_offset value"
                self.variance = np.clip(self.variance, np.abs(gain_offset), 
                np.Inf)
            else:
                print "Clipping the variance to the gaussian_noise_var"
                self.variance = np.clip(self.variance, gaussian_noise_var, 
                np.Inf) 
                
    def is_spectrum_line(self):
        if len(self.data_cube.squeeze().shape) == 2:
            return True
        else:
            return False
        
    def is_spectrum_image(self):
        if len(self.data_cube.squeeze().shape) == 3:
            return True
        else:
            return False
        
    def is_single_spectrum(self):
        if len(self.data_cube.squeeze().shape) == 1:
            return True
        else:
            return False
   
    def calibrate(self, lcE = 642.6, rcE = 849.7, lc = 161.9, rc = 1137.6, 
    modify_calibration = True):
        dispersion = (rcE - lcE) / (rc - lc)
        origin = lcE - dispersion * lc
        print "Energy step = ", dispersion
        print "Energy origin = ", origin
        if modify_calibration is True:
            self.set_new_calibration(origin, dispersion)
        return origin, dispersion
    
    def _correct_spatial_mask_when_unfolded(self, spatial_mask = None,):
        if 'unfolded' in self.history:
            if spatial_mask is not None:
                spatial_mask = \
                spatial_mask.reshape((-1,), order = 'F')
        return spatial_mask

    def get_image(self, spectral_range = slice(None), background_range = None):
        data = self.data_cube
        if self.is_spectrum_line() is True:
            return self.data_cube.squeeze()
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
            return data[spectral_range,:,:].sum(0)
    
    def plot(self):
        '''Plots the current spectrum to the screen and a map with a cursor to 
        explore the SI.
        '''
        
        # If new coordinates are assigned
        controllers.coordinates_controller.assign_coordinates(self)
        if self.hse is not None:
#            if self.coordinates is not self.hse.coordinates:
            try:
                self.hse.close()
            except:
                pass
            del(self.hse)
            self.hse = None
            
                    
        # Spectrum properties
        self.hse = drawing.mpl_hse.MPL_HyperSpectrum_Explorer()
        self.hse.spectrum_data_function = self.__call__
        self.hse.spectrum_title = 'EELS Spectrum'
        self.hse.xlabel = 'Energy Loss (%s)' % self.energyunits
        self.hse.ylabel = 'Counts (arbitraty)'
        self.hse.coordinates = self.coordinates
        self.hse.axis = self.energy_axis
        
        # Image properties
        self.hse.image_data_function = self.get_image
        self.hse.image_title = ''
        self.hse.pixel_size = self.xscale
        self.hse.pixel_units = self.xunits
        
        self.hse.plot()
        
    def _replot(self):
        if self.hse is not None:
            if self.hse.is_active() is True:
                self.plot()
                
    def get_single_spectrum(self):
        s = Spectrum({'calibration' : {'data_cube' : self()}})
        s.get_calibration_from(self)
        return s