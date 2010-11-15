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

from __future__ import division
import copy
import os
import sys
import tempfile


import numpy as np
import scipy as sp

try:
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

try:
    import mdp
except:
    print \
    '''
    Warning! In order to enjoy the PCA features you must install python-mdp
    '''

import utils
from svd_pca import pca    
from widgets import cursors
from utils import unfold_if_2D
from mlpca import mlpca
from image import Image
from utils import center_and_scale
from defaults_parser import defaults
import messages
import config_dir

def compile_kica():
    kica_path = os.path.join(config_dir.data_path, 'kica')
    kica_compile = os.path.join(config_dir.data_path, 'kica_compile.m')
    print('octave %s %s' % (kica_compile, kica_path))
    os.system('octave %s %s' % (kica_compile, kica_path))

def perform_kica(pc):
    '''Wrapper for kernel independent components analysis that runs in octave 
    (octave is required).
    '''
    import subprocess
    # Until I get pytave working, a temporary file is necessary to interact 
    # with octave
    fd, temp_file = tempfile.mkstemp(suffix = '.mat')
    subprocess.call('command --output %s' % temp_file)
    kica_file = os.path.join(config_dir.data_path, 'kica.m')
    kica_path = os.path.join(config_dir.data_path, 'kica')
    print('octave %s %s %s' % (kica_file, temp_file, kica_path))
    # The recommend parameters for kica depend on the number of energy channels
    sp.io.savemat(temp_file, {'x' : pc.T})
    def call_kica():
        os.system('octave %s %s %s' % (kica_file, temp_file, kica_path)) 
        w = sp.io.loadmat(temp_file)['w']
        os.close(fd)
        os.remove(temp_file)
        # TODO: although it works, there is an annoying warning that someone 
        # should investigate
        return w
    try:
        return call_kica()
    except:
        try:
            # Try to compile
            compile_kica()
            return call_kica()
        except:
            messages.warning('It is not possible to run the KICA algorithm.\n'
            'Verify that:'
            '- Octave is istalled\n'
            '- If you are running Windows, define the octave path\n'
            '- kica has been compiled for your platform:\n'
            '   call the eelslab function mva.compile_kica().'
            '   (Note that root privilages may be required)'
            '   In Linux you can simple run \'sudo eelslab_compile_kica\' in a'
            '    terminal')
    try:

        # Delete the temporary file if it still exists.
        if os.path.isfile(temp_file):
            os.remove(temp_file)
    except:
        pass
    
        
class MVA():
    '''
    Multivariate analysis capabilities for the Spectrum class.
    '''
    def __init__(self):
        self.mva_results = MVA_Results()
    
    def principal_components_analysis(self, normalize_poissonian_noise = False, 
    algorithm = 'svd', output_dim = None, spatial_mask = None, 
    energy_mask = None, center = False, variance2one = False, var_array = None, 
    var_func = None, polyfit = None):
        '''Principal components analysis.
        
        The results are stored in self.mva_results
        
        Parameters
        ----------
        normalize_poissonian_noise : bool
            If True, scale the SI to normalize Poissonian noise
        algorithm : {'svd', 'mlpca', 'mdp', 'NIPALS'}
        output_dim : None or int
            number of PCA to keep
        spatial_mask : boolean numpy array
        energy_mask : boolean numpy array
        center : bool
            Perform energy centering before PCA
        variance2one : bool
            Perform whitening before PCA
        var_array : numpy array
            Array of variance for the maximum likelihood PCA algorithm
        var_func : function or numpy array
            If function, it will apply it to the dataset to obtain the 
            var_array. Alternatively, it can a an array with the coefficients 
            of a polynomy.
        polyfit : 
            
            
        See also
        --------
        plot_principal_components, plot_principal_components_maps, plot_lev
        '''
        
        # Check for conflicting options and correct them when possible   
        if (algorithm == 'mdp' or algorithm == 'NIPALS') and center is False:
            print \
            '''
            The PCA algorithms from the MDP toolking (mdp and NIPALS) 
            do not permit desactivating data centering.
            Therefore, the algorithm will proceed to center the data.
            '''
            center = True
        if algorithm == 'mlpca':
            if normalize_poissonian_noise is True:
                messages.warning(
                "It makes no sense to do normalize_poissonian_noise with "
                "the MLPCA algorithm. Therefore, "
                "normalize_poissonian_noise is set to False")
                normalize_poissonian_noise = False
            if output_dim is None:
                messages.warning_exit(
                "With the mlpca algorithm the output_dim must be expecified")
        
        if center is True and normalize_poissonian_noise is True:
            messages.warning(
            "Centering is not compatible with poissonian noise normalization\n"
            "Disabling centering")
            center = False
        
        if variance2one is True and normalize_poissonian_noise is True:
            messages.warning(
            "Variance normalization is not compatible with poissonian noise" 
            "normalization.\n"
            "Disabling variance2one")
            variance2one = False
        
        # Apply pre-treatments
        
        # Centering
        if center is True:
            self.energy_center()
        # Variance normalization
        if variance2one is True:
            self.variance2one()
        # Transform the data in a line spectrum    
        self._unfolded4pca = unfold_if_2D(self)
        dc = self.data_cube.squeeze()
        # Normalize the poissonian noise
        # Note that this function can change the masks
        if normalize_poissonian_noise is True:
            spatial_mask, energy_mask = \
            self.normalize_poissonian_noise(spatial_mask = spatial_mask, 
            energy_mask = energy_mask, return_masks = True)

        spatial_mask = self._correct_spatial_mask_when_unfolded(spatial_mask)            

        messages.information('Performing principal components analysis')
        # Transform the None masks in slices to get the right behaviour
        if spatial_mask is None:
            spatial_mask = slice(None)
        if energy_mask is None:
            energy_mask = slice(None)
        if algorithm == 'mdp' or algorithm == 'NIPALS':
            if algorithm == 'mdp':
                self.mva_results.pca_node = mdp.nodes.PCANode(
                output_dim=output_dim, svd = True)
            elif algorithm == 'NIPALS':
                self.mva_results.pca_node = mdp.nodes.NIPALSNode(
                output_dim=output_dim)
            # Train the node
            print "\nPerforming the PCA node training"
            print "This include variance normalizing"
            self.mva_results.pca_node.train(
            dc[energy_mask,:][:,spatial_mask])
            print "Performing PCA projection"
            pc = self.mva_results.pca_node.execute(dc[:,spatial_mask])
            pca_v = self.mva_results.pca_node.v
            pca_V = self.mva_results.pca_node.d
            self.mva_results.output_dim = output_dim
            
        elif algorithm == 'svd':
            pca_v, pca_V = pca(dc[energy_mask,:][:,spatial_mask])
            pc = np.dot(dc[:,spatial_mask], pca_v)
            
        elif algorithm == 'mlpca':
            print "Performing the MLPCA training"
            if output_dim is None:
                messages.warning_exit(
                "For MLPCA it is mandatory to define the output_dim")
            if var_array is None and polyfit is None:
                messages.warning_exit(
                "For MLPCA it is mandatory to define either the variance array "
                "or the polyfit functions")
            if var_array is not None and var_func is not None:
                messages.warning_exit(
                "You have defined both the var_func and var_array keywords"
                "Please, define just one of them")
            if var_func is not None:
                if hasattr(var_func, '__call__'):
                    var_array = var_func(dc[energy_mask,...][:,spatial_mask])
                else:
                    try:
                        var_array = np.polyval(polyfit,dc[energy_mask, 
                        spatial_mask])
                    except:
                        messages.warning_exit(
                        'var_func must be either a function or an array'
                        'defining the coefficients of a polynom')             
                
            self.mva_results.mlpca_output = mlpca(
                dc.squeeze()[energy_mask,:][:,spatial_mask], 
                var_array.squeeze(), 
                output_dim)
            U,S,V,Sobj, ErrFlag  = self.mva_results.mlpca_output
            print "Performing PCA projection"
            pc = np.dot(dc[:,spatial_mask], V)
            pca_v = V
            pca_V = S ** 2
            
        self.mva_results.pc = pc
        self.mva_results.v = pca_v
        self.mva_results.V = pca_V
        self.mva_results.pca_algorithm = algorithm
        self.mva_results.centered = center
        self.mva_results.poissonian_noise_normalized = \
        normalize_poissonian_noise
        self.mva_results.output_dim = output_dim
        self.mva_results.unfolded = self._unfolded4pca
        self.mva_results.variance2one = variance2one
        
        if self._unfolded4pca is True:
            self.mva_results.original_shape = self.shape_before_folding
            
        if variance2one is True:
            self.undo_variance2one()
        
        if center is True:
            if self._unfolded4pca is True:
                self.fold()
            self.undo_energy_center()
            if self._unfolded4pca is True:
                self.unfold()
        
        # Rescale the results if the noise was normalized
        if normalize_poissonian_noise is True:
            print "I'm here"
            self.mva_results.pc[energy_mask,:] *= self._root_bH
            self.mva_results.v *= self._root_aG.T
            if isinstance(spatial_mask, slice):
                spatial_mask = None
            if isinstance(energy_mask, slice):
                energy_mask = None
            self.undo_normalize_poissonian_noise(spatial_mask = spatial_mask,
            energy_mask = energy_mask)
        
        # Set the pixels that were not processed to nan
        if spatial_mask is not None or not isinstance(spatial_mask, slice):
            v = np.zeros((dc.shape[1], self.mva_results.v.shape[1]), 
                    dtype = self.mva_results.v.dtype)
            v[spatial_mask == False,:] = np.nan
            v[spatial_mask,:] = self.mva_results.v
            self.mva_results.v = v
                
        if self._unfolded4pca is True:
            self.fold()
            self._unfolded4pca is False
            
    def independent_components_analysis(self, number_of_components = None, 
    algorithm = 'CuBICA', diff_order = 1, pc = None, 
    comp_list = None, mask = None, **kwds):
        '''Independent components analysis.
        
        Available algorithms: FastICA, JADE, CuBICA, TDSEP, kica, MILCA
        
        Parameters
        ----------
        number_of_components : int
            number of principal components to pass to the ICA algorithm
        algorithm : {FastICA, JADE, CuBICA, TDSEP, kica, milca}
        diff : bool
        diff_order : int
        pc : numpy array
            externally provided components
        comp_list : boolen numpy array
            choose the components to use by the boolen list. It permits to 
            choose non contiguous components.
        mask : numpy boolean array with the same dimension as the PC
            If not None, only the selected channels will be used by the 
            algorithm.
        '''
        if hasattr(self.mva_results, 'pc'):
            if pc is None:
                pc = self.mva_results.pc
            bool_index = np.zeros((pc.shape[0]), dtype = 'bool')
            if number_of_components is not None:
                bool_index[:number_of_components] = True
            else:
                if self.output_dim is not None:
                    number_of_components = self.output_dim
                    bool_index[:number_of_components] = True
                    
            if comp_list is not None:
                for ipc in comp_list:
                    bool_index[ipc] = True
                number_of_components = len(comp_list)
            pc = pc[:,bool_index]
            if diff_order > 0:
                pc = np.diff(pc, diff_order, axis = 0)
            if mask is not None:
                pc = pc[mask, :]
            if algorithm == 'kica':
                self.mva_results.w = perform_kica(pc)
            elif algorithm == 'milca':
                try:
                    import milca
                except:
                    messages.warning_exit('MILCA is not installed')
                # first centers and scales data
                invsqcovmat, pc = center_and_scale(pc).itervalues()
                self.mva_results.w = np.dot(milca.milca(pc, **kwds), invsqcovmat)
            else:
                # first centers and scales data
                invsqcovmat, pc = center_and_scale(pc).itervalues()
                exec('self.ica_node = mdp.nodes.%sNode(white_parm = \
                {\'svd\' : True})' % algorithm)
                self.ica_node.variance2oneed = True
                self.ica_node.train(pc)
                self.mva_results.w = np.dot(self.ica_node.get_recmatrix(), invsqcovmat)
            self._ic_from_w()
            self.mva_results.ica_algorithm = algorithm
            self.output_dim = number_of_components
        else:
            "You have to perform Principal Components Analysis before"
            sys.exit(0)
    
    def reverse_ic(self, i):
        '''Reverse the independent component
        
        Parameters
        ----------
        i : int
            component index
        '''
        self.ic[:,i] *= -1
        self.mva_results.w[i,:] *= -1
    
    def _ic_from_w(self):
        w = self.mva_results.w
        n = len(w)
        self.ic = np.dot(self.mva_results.pc[:,:n], w.T)
        for i in range(n):
            if np.all(self.ic[:,i] <= 0):
                self.reverse_ic(i)
        
    def pca_build_SI(self,number_of_components=None, comp_list = None):
        '''Return the spectrum generated with the selected number of principal 
        components
        
        Parameters
        ------------
        number_of_components : int
        comp_list : boolen numpy array
            choose the components to use by the boolen list. It permits to 
            choose non contiguous components.
        
        Returns
        -------
        Spectrum instance
        '''
        bool_index = np.zeros((self.mva_results.pc.shape[0]), dtype = 'bool')
        if number_of_components is not None:
            bool_index[:number_of_components] = True
        if comp_list is not None:
            for ipc in comp_list:
                bool_index[ipc] = True
            number_of_components = len(comp_list)
        self._unfolded4pca = unfold_if_2D(self)
        a = np.atleast_3d(np.dot(self.mva_results.pc[:,bool_index], 
        self.mva_results.v.T[bool_index, :]))
#        rebuilded_spectrum = copy.deepcopy(self)
#        rebuilded_spectrum._Spectrum__new_cube(a, 
#        'rebuilded from PCA with %s components' % number_of_components)
#            rebuilded_spectrum.fold()
#        if self.mva_results.variance2one is True:
#            rebuilded_spectrum.undo_variance2one()        
#        if self.mva_results.centered is True:
#            rebuilded_spectrum.undo_energy_center()
        from spectrum import Spectrum
        sc = Spectrum({'calibration': {'data_cube' : a}})
        if self._unfolded4pca is True:
            self.fold()
            sc.shape_before_folding = copy.copy(self.shape_before_folding)
            sc.history.append('unfolded')
            sc.fold()
        sc.get_calibration_from(self)
        
        return sc
    
    def _calculate_recmatrix(self, n = None):
        if not n:
            n = self.ic.shape[1]
        W = self.mva_results.v.T[:n, :]
        Q = np.linalg.inv(self.mva_results.w.T)
        recmatrix = np.dot(Q,W)
        return recmatrix
        
    def ica_build_SI(self,number_of_components = None, ic2zero = None):
        '''Return the spectrum generated with the selected number of 
        independent components
        
        Parameters
        ------------
        number_of_components : int
        ic2zero : tuple of ints
            tuple of index of independent components that must be excluded 
            although they are in the range.  It permits to choose non 
            contiguous components.
        
        Returns
        -------
        Spectrum instance
        '''
        recmatrix = self._calculate_recmatrix()
        n = number_of_components
        if not n:
            n = self.ic.shape[1]
        ic = copy.copy(self.ic)
        if ic2zero is not None:
            for comp in ic2zero:
                ic[:,comp] *= 0
        self._unfolded4pca = unfold_if_2D(self)
        a = np.dot(ic[:,:n], 
        recmatrix[:n, :])
        rebuilded_spectrum = copy.deepcopy(self)
        rebuilded_spectrum._Spectrum__new_cube(np.atleast_3d(a), 
        'rebuilded from PCA with %s components' % n)
        if self._unfolded4pca:
            self.fold()
            rebuilded_spectrum.fold()
        return rebuilded_spectrum
        
    def plot_principal_components(self, n = None):
        '''Plot the principal components up to the given number
        
        Parameters
        ----------
        n : int
            number of principal components to plot.
        '''
        if n is None:
            n = self.mva_results.pc.shape[1]
        for i in range(n):
            plt.figure()
            plt.plot(self.energy_axis, self.mva_results.pc[:,i])
            plt.title('Principal component %s' % i)
            plt.xlabel('Energy (eV)')
            
    def plot_principal_components_maps(self, n, cmap=plt.cm.gray, plot = True):
        '''Plot the map associated to the principal components up to the given 
        number
        
        Parameters
        ----------
        n : int
            number of principal component maps to plot
        cmap : plt.cm object
        plot : Bool
            If True it actually plots the maps, otherwise it only returns them
        
        Returns
        -------
        List with the maps as Image instances
        '''
        from spectrum import Spectrum
        recmatrix = self.mva_results.v.T[:n, :]
#        if 'unfolded' in self.history:
#            self.fold()
        shape = (self.data_cube.shape[1], self.data_cube.shape[2])
        im_list = []
        for i in range(n):
            print i
            if plot is True:
                figure = plt.figure()
                ax = figure.add_subplot(111)
            if shape[1] != 1:
                toplot = recmatrix[i,:].reshape(shape, 
                order = 'F').T
                im_list.append(Image({'calibration' : {'data_cube': toplot.T}}))
#                if np.all(toplot <= 0):
#                    toplot *= -1
                if plot is True:
                    mapa = ax.matshow(toplot, cmap = cmap)
                    figure.colorbar(mapa)
                    figure.canvas.draw()
                    cursors.add_axes(ax)
            else:
                im_list.append(Spectrum())
                toplot = recmatrix[i,:]
                im_list.append(Spectrum())
                im_list[-1].data_cube = toplot
                im_list[-1].get_dimensions_from_cube()
                if plot is True:
                    plt.step(range(len(toplot)), toplot)
            if plot is True:
                plt.title('Principal component number %s map' % i)
        return im_list
            
    def plot_independent_components(self, ic = None):
        '''Plot the independent components.
        
        Parameters
        ----------
        ic : numpy array
            externally provided independent components array
        '''
        if ic is None:
            ic = self.ic
        n = ic.shape[1]
        for i in range(n):
            plt.figure()
            plt.plot(self.energy_axis, ic[:,i])
            plt.title('Independent component %s' % i)
            plt.xlabel('Energy (eV)')
        
    def plot_lev(self, n=50):
        '''Plot the principal components LEV up to the given number
        
        Parameters
        ----------
        n : int
        '''        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(range(n), self.mva_results.V[:n], 'o')
        ax.semilogy()
        ax.set_title('Log(eigenvalues)')
        ax.set_xlabel('Principal component')
        plt.draw()
        plt.show()
        return ax
        
    def plot_explained_variance(self,n=50):
        '''Plot the principal components explained variance up to the given 
        number
        
        Parameters
        ----------
        n : int
        ''' 
        cumu = np.cumsum(self.mva_results.V) / np.sum(self.mva_results.V)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(range(n), cumu[:n])
        ax.set_xlabel('Principal component')
        ax.set_ylabel('Explained variance')
        plt.draw()
        plt.show()
        return ax
        
    def plot_independent_components_maps(self,cmap=plt.cm.gray, recmatrix = None, 
    comp_list = None, with_ic = True, plot = True, ic = None):
        '''Plot the map associated to each independent component
        
        Parameters
        ----------
        cmap : plt.cm object
        recmatrix : numpy array
            externally suplied recmatrix
        comp_list : boolen numpy array
            choose the components to use by the boolen list. It permits to 
            choose non contiguous components.
        with_ic : bool
            If True, plots also the corresponding independent component in the 
            same figure
        plot : bool
            If True it will plot the figures. Otherwise it will only return the 
            images.
        ic : numpy array
            externally supplied independent components
             
        Returns
        -------
        List with the maps as Image instances
        '''
        from spectrum import Spectrum
        if ic is None:
            ic = self.ic
        n = ic.shape[1]
        bool_index = np.zeros((self.mva_results.pc.shape[0]), dtype = 'bool')
        if comp_list is None:
            bool_index[:n] = True
        else:
            for ipc in comp_list:
                bool_index[ipc] = True
            n = len(comp_list)
        if recmatrix is None:
            W = self.mva_results.v.T[bool_index, :]
            Q = np.linalg.inv(self.mva_results.w.T)
            recmatrix = np.dot(Q,W)
        shape = self.data_cube.shape[1], self.data_cube.shape[2]
        im_list = []
        for i in range(n):
            if plot is True:
                figure = plt.figure()
                if with_ic:
                    ax = figure.add_subplot(121)
                    ax2 = figure.add_subplot(122)
                else:
                    ax = figure.add_subplot(111)
            if shape[1] != 1:
                toplot = recmatrix[i,:].reshape(shape, 
                order = 'F').T
                im_list.append(Image({'calibration' : {'data_cube': toplot.T}}))
                if plot is True:
                    mapa = ax.matshow(toplot, cmap = cmap)
                    if with_ic:
                        ax2.plot(self.energy_axis, ic[:,i])
                        ax2.set_title('Independent component %s' % i)
                        ax2.set_xlabel('Energy (eV)')
                    figure.colorbar(mapa)
                    figure.canvas.draw()
                    cursors.add_axes(ax)
            else:
                toplot = recmatrix[i,:]
                im_list.append(Spectrum())
                im_list[-1].data_cube = toplot
                im_list[-1].get_dimensions_from_cube()
                if plot is True:
                    ax.step(range(len(toplot)), toplot)
                    
                    if with_ic:
                        ax2.plot(self.energy_axis, ic[:,i])
                        ax2.set_title('Independent component %s' % i)
                        ax2.set_xlabel('Energy (eV)')
            if plot is True:
                ax.set_title('Independent component number %s map' % i)
                figure.canvas.draw()
        return im_list
    
    def save_principal_components(self, n, spectrum_prefix = 'pc', 
    image_prefix = 'im', spectrum_format = 'msa', image_format = 'tif'):
        '''Save the `n` first principal components  and score maps 
        in the specified format
        
        Parameters
        ----------
        n : int
            Number of principal components to save_als_ica_results
        image_prefix : string
            Prefix for the image file names
        spectrum_prefix : string
            Prefix for the spectrum file names
        spectrum_format : string
        image_format : string
                 
        '''
        from spectrum import Spectrum
        im_list = self.plot_principal_components_maps(n, plot = False)
        s = Spectrum({'calibration' : {'data_cube' : self.mva_results.pc[:,0]}})
        s.get_calibration_from(self)
        for i in range(n):
            s.data_cube = self.mva_results.pc[:,i]
            s.get_dimensions_from_cube()
            s.save('%s-%i.%s' % (spectrum_prefix, i, spectrum_format))
            im_list[i].save('%s-%i.%s' % (image_prefix, i, image_format))
        
    def save_independent_components(self, elements = None, 
    spectrum_format = 'msa', image_format = 'tif', recmatrix = None, ic = None):
        '''Saves the result of the ICA in image and spectrum format
        
        Parameters
        ----------
        elements : None or tuple of strings
            a list of names (normally an element) to be assigned to IC. If not 
            the will be name ic-0, ic-1 ...
        image_format : string
        spectrum_format : string
        rectmatrix : None or numpy array
            externally supplied recmatrix
        ic : None or numpy array 
            externally supplied IC
        '''
        from spectrum import Spectrum
        pl = self.plot_independent_components_maps(plot = False, 
        recmatrix = recmatrix, ic = ic)
        if ic is None:
            ic = self.ic
        if self.data_cube.shape[2] > 1:
            maps = True
        else:
            maps = False
        for i in range(ic.shape[1]):
            sp = Spectrum()
            sp.data_cube = ic[:,i].reshape((-1,1,1))
            sp.get_dimensions_from_cube()
            utils.copy_energy_calibration(self,sp)
            if elements is None:
                sp.save('ic-%s.%s' % (i, spectrum_format))
                if maps is True:
                    pl[i].save('map_ic-%s.%s' % (i, image_format))
                else:
                    pl[i].save('profile_ic-%s' % (i, spectrum_format))
            else:
                element = elements[i]
                sp.save('ic-%s.%s' % (element, spectrum_format))
                if maps:
                    pl[i].save('map_ic-%s.%s' % (element, image_format))
                else:
                    pl[i].save('profile_ic-%s.%s' % (element, spectrum_format))
                    
    def snica(self, coordinates = None):
        '''Stochastic Non-negative Independent Component Analysis.
        
        Is intended to be used with a denoised SI, i.e. by PCA. The method will 
        store the demixed components to self.ic
        
        Parameters
        -----------
        coordinates: tuple of 2 integral tuples or None
            If None, all the SI will be analyzed. Otherwise only the coordinates 
            defined by the tuples.
        
        Returns
        -------
        A 2 rows array. First row are the demixed components in the format 
        (channels, components) and the second row is the mixing matrix.
        '''
        if coordinates is None:
            fold_back = utils.unfold_if_2D(self)
            ic = self.data_cube.squeeze()
        else:
            clist = []
            for coord in coordinates:
                clist.append(self.data_cube[:,coord[0], coord[1]])
            ic = np.array(clist).T
        snica = utils.snica(ic)
        self.ic = snica[0]
        if fold_back is True: self.fold()
        return snica
    
    def als(self, **kwargs):
        '''Alternate Least Squares imposing positivity constraints 
        to the result of a previous ICA
        
        Stores in result in self.als_out
        
        Parameters
        ----------
        thresh : float
            default=.001
        nonnegS : bool
            Impose non-negative constraint of the components. Default: True
        nonnegC : bool
            Impose non-negative constraint of the maps. Default: True
        
        See also
        -------
        plot_als_ic_maps, plot_als_ic
        '''
        shape = (self.data_cube.shape[2], self.data_cube.shape[1],-1)
        if hasattr(self, 'ic') and (self.ic is not None):
            also = utils.ALS(self, **kwargs)
            self.als_ic = also['S']
            self.als_maps = also['CList'].reshape(shape, order = 'C')
            self.als_output = also
            
    def plot_als_ic_maps(self):
        '''Same as plot_ic_maps for the ALS results'''
        return self.plot_independent_components_maps(recmatrix = 
        self.als_output['CList'].T, ic = self.als_ic)
    
    def plot_als_ic(self):
        '''Same as plot_independent_componets for the ALS results'''
        self.plot_independent_components(ic = self.als_ic)
        
    def save_als_ica_results(self, elements = None, 
    format = defaults.file_format, image_format = 'tif'):
        '''Same as save_ica_results for the ALS results'''
        self.save_ica_results(elements = elements, image_format = image_format, 
        recmatrix = self.als_output['CList'].T, ic = self.als_ic)
                
    def normalize_poissonian_noise(self, spatial_mask = None, 
    energy_mask = None, return_masks = False):
        '''
        Scales the SI following Surf. Interface Anal. 2004; 36: 203–212 to
        "normalize" the poissonian data for PCA analysis
        
        Parameters
        ----------
        spatial_mask : boolen numpy array
        energy_mask  : boolen numpy array
        '''
        refold = unfold_if_2D(self)
        dc = self.data_cube.squeeze()
        spatial_mask = \
        self._correct_spatial_mask_when_unfolded(spatial_mask)
        if spatial_mask is None:
            spatial_mask = slice(None)
        if energy_mask is None:
            energy_mask = slice(None)
        # Rescale the data to gaussianize the poissonian noise
        aG = dc[energy_mask,:][:,spatial_mask].sum(0).squeeze()
        bH = dc[energy_mask,:][:,spatial_mask].sum(1).squeeze()
        # Checks if any is negative
        if (aG < 0).any() or (bH < 0).any():
            messages.warning_exit(
            "Data error: negative values\n"
            "Are you sure that the data follow a poissonian distribution?")
        # Update the spatial and energy masks so it does not include rows
        # or colums that sum zero.
        aG0 = (aG == 0)
        bH0 = (bH == 0)
        if aG0.any():
            if isinstance(spatial_mask, slice):
                # Convert the slice into a mask before setting its values
                spatial_mask = np.ones((self.data_cube.shape[1]),dtype = 'bool')
            # Set colums summing zero as masked
            spatial_mask[aG0] = False
            aG = aG[aG0 == False]
        if bH0.any():
            if isinstance(energy_mask, slice):
                # Convert the slice into a mask before setting its values
                energy_mask = np.ones((self.data_cube.shape[0]), dtype = 'bool')
            # Set rows summing zero as masked
            energy_mask[bH0] = False
            bH = bH[bH0 == False]
        messages.information(
            "Scaling the data to normalize the (presumably) Poissonian noise")
        self._root_aG = np.sqrt(aG)[np.newaxis,:]
        self._root_bH = np.sqrt(bH)[:, np.newaxis]
        temp = (dc[energy_mask,:][:,spatial_mask] / 
        (self._root_aG * self._root_bH))
        if  isinstance(energy_mask,slice) or isinstance(spatial_mask,slice):
            dc[energy_mask,spatial_mask] = temp
        else:
            mask3D = energy_mask[:, np.newaxis] * \
            spatial_mask[np.newaxis, :]
            dc[mask3D] = temp.ravel()
        if refold is True:
            print "Automatically refolding the SI after scaling"
            self.fold()
        if return_masks is True:
            if isinstance(spatial_mask, slice):
                spatial_mask = None
            if isinstance(energy_mask, slice):
                energy_mask = None
            return spatial_mask, energy_mask
        
    def undo_normalize_poissonian_noise(self, spatial_mask = None, 
    energy_mask = None):
        '''Undo normalize_poissonian_noise'''
        print "Undoing the noise normalization"
        refold = unfold_if_2D(self)
        dc = self.data_cube.squeeze()
        spatial_mask = \
        self._correct_spatial_mask_when_unfolded(spatial_mask)
        if spatial_mask is None:
            spatial_mask = slice(None)
        if energy_mask is None:
            energy_mask = slice(None)
        temp = (dc[energy_mask,:][:,spatial_mask] * 
        (self._root_aG * self._root_bH))
        if  isinstance(energy_mask,slice) or isinstance(spatial_mask,slice):
            dc[energy_mask,spatial_mask] = temp
        else:
            mask3D = energy_mask[:, np.newaxis] * \
            spatial_mask[np.newaxis, :]
            dc[mask3D] = temp.ravel()
        if refold is True:
            print "Automatically refolding the SI after scaling"
            self.fold()
        
class MVA_Results():
    def __init__(self):
        self.pc = None
        self.v = None
        self.V = None
        self.pca_algorithm = None
        self.ica_algorithm = None
        self.centered = None
        self.poissonian_noise_normalized = None
        self.output_dim = None
        self.unfolded = None
        self.original_shape = None
        # Demixing matrix
        self.w = None
        
    
    def save(self, filename):
        '''Save the result of the PCA analysis
        
        Parameters
        ----------
        filename : string
        '''
        np.savez(filename, pc = self.pc, v = self.v, V = self.V, 
        pca_algorithm = self.pca_algorithm, centered = self.centered, 
        output_dim = self.output_dim, variance2one = self.variance2one, 
        poissonian_noise_normalized = self.poissonian_noise_normalized, 
        w = self.w, ica_algorithm = self.ica_algorithm)

    def load(self, filename):
        '''Load the result of the PCA analysis
        
        Parameters
        ----------
        filename : string
        '''
        pca = np.load(filename)
        for key in pca.files:
            exec('self.%s = pca[\'%s\']' % (key, key))
        print "\n%s loaded correctly" %  filename
        
        # For compatibility with old version
        if hasattr(self, 'algorithm'):
            self.pca_algorithm = self.algorithm
            del self.algorithm
        defaults = {
        'centered' : False,
        'variance2one' : False,
        'poissonian_noise_normalized' : False,
        'output_dim' : None,
        'last_used_pca_algorithm' : None
        }
        for attrib in defaults.keys():
            if not hasattr(self, attrib):
                exec('self.%s = %s' % (attrib, defaults[attrib]))
        self.summary()
        
    def summary(self):
        '''Prints a summary of the PCA parameters to the stdout
        '''
        print
        print "MVA Summary:"
        print "------------"
        print
        print "PCA algorithm : ", self.pca_algorithm
        print "Scaled to normalize poissonina noise : %s" % \
        self.poissonian_noise_normalized
        print "Energy centered : %s" % self.centered
        print "Variance normalized : %s" % self.variance2one
        print "Output dimension : %s" % self.output_dim
        print "ICA algorithm : %s" % self.ica_algorithm
        
    def crop_v(self, n):
        '''
        Crop the score matrix up to the given number.
        
        It is mainly useful to save memory and redude the storage size
        '''
        self.v = self.v[:,:n].copy()
