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
import scipy.io

import matplotlib.pyplot as plt

import mdp
from eelslab import utils
from svd_pca import pca    
from mlpca import mlpca
from eelslab.utils import center_and_scale
from eelslab.defaults_parser import defaults
from eelslab import messages
from eelslab import config_dir
from exceptions import *

def compile_kica():
    kica_path = os.path.join(config_dir.data_path, 'kica')
    kica_compile = os.path.join(config_dir.data_path, 'kica_compile.m')
    # print('Running "octave -q %s %s"' % (kica_compile, kica_path))
    return os.system('octave -q %s %s' % (kica_compile, kica_path))

def perform_kica(pc):
    """Wrapper for kernel independent components analysis that runs in octave 
    (octave is required).
    """
#    import subprocess
    # Until I get pytave working, a temporary file is necessary to interact 
    # with octave
    fd, temp_file = tempfile.mkstemp(suffix = '.mat')
#    subprocess.call('command --output %s' % temp_file)
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
    """
    Multivariate analysis capabilities for the Spectrum class.
    """
    def __init__(self):
        self.mva_results = MVA_Results()
    
    def principal_components_analysis(self, normalize_poissonian_noise = False, 
    algorithm = 'svd', output_dim = None, spatial_mask = None, 
    energy_mask = None, center = False, variance2one = False, var_array = None, 
    var_func = None, polyfit = None):
        """Principal components analysis.
        
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
        """
        # backup the original data
        self._data_before_treatments = self.data.copy()
        # Check for conflicting options and correct them when possible   
        if (algorithm == 'mdp' or algorithm == 'NIPALS') and center is False:
            print \
            """
            The PCA algorithms from the MDP toolking (mdp and NIPALS) 
            do not permit desactivating data centering.
            Therefore, the algorithm will proceed to center the data.
            """
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
        self._unfolded4pca = self.unfold_if_multidim()
        # Normalize the poissonian noise
        # Note that this function can change the masks
        if normalize_poissonian_noise is True:
            spatial_mask, energy_mask = \
                self.normalize_poissonian_noise(spatial_mask = spatial_mask, 
                                                energy_mask = energy_mask, 
                                                return_masks = True)

        spatial_mask = self._correct_spatial_mask_when_unfolded(spatial_mask)

        messages.information('Performing principal components analysis')
        dc_transposed=False
        last_axis_units=self.axes_manager.axes[-1].units
        if last_axis_units=='eV' or last_axis_units=='keV':
            print "Transposing data so that energy axis makes up rows."
            dc = self.data.T.squeeze()
            dc_transposed=True
        else:
            dc = self.data.squeeze()
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
            self.mva_results.dc_transposed=dc_transposed
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
            self.mva_results.original_shape = self._shape_before_unfolding

        # Rescale the results if the noise was normalized
        if normalize_poissonian_noise is True:
            self.mva_results.pc[energy_mask,:] *= self._root_bH
            self.mva_results.v *= self._root_aG.T
            if isinstance(spatial_mask, slice):
                spatial_mask = None
            if isinstance(energy_mask, slice):
                energy_mask = None

        #undo any pre-treatments
        self.undo_treatments()

        # Set the pixels that were not processed to nan
        if spatial_mask is not None or not isinstance(spatial_mask, slice):
            v = np.zeros((dc.shape[1], self.mva_results.v.shape[1]), 
                    dtype = self.mva_results.v.dtype)
            v[spatial_mask == False,:] = np.nan
            v[spatial_mask,:] = self.mva_results.v
            self.mva_results.v = v
                
        if self._unfolded4pca is True:
            self.fold()
            assert(self._unfolded4pca is False)
            
    def independent_components_analysis(self, number_of_components = None, 
    algorithm = 'CuBICA', diff_order = 1, pc = None, 
    comp_list = None, mask = None, **kwds):
        """Independent components analysis.
        
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
        """
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
    
    def reverse_ic(self, *ic_n):
        """Reverse the independent component
        
        Parameters
        ----------
        ic_n : int
            component indexes

        Examples
        -------
        >>> s = load('some_file')
        >>> s.principal_components_analysis(True) # perform PCA
        >>> s.independent_components_analysis(3)  # perform ICA on 3 PCs
        >>> s.reverse_ic(1) # reverse IC 1
        >>> s.reverse_ic(0, 2) # reverse ICs 0 and 2
        """
        for i in ic_n:
            self.ic[:,i] *= -1
            self.mva_results.w[i,:] *= -1
    
    def _ic_from_w(self):
        w = self.mva_results.w
        n = len(w)
        self.ic = np.dot(self.mva_results.pc[:,:n], w.T)
        for i in xrange(n):
            if np.all(self.ic[:,i] <= 0):
                self.reverse_ic(i)

    def _get_ica_scores(self):
        """
        Returns the ICA score matrix (formerly known as the recmatrix)
        """
        W = self.mva_results.v.T[:self.ic.shape[1],:]
        Q = np.linalg.inv(self.mva_results.w.T)
        return np.dot(Q,W)
          
    def _calculate_recmatrix(self, components = None, mva_type=None):
        """
        Rebuilds SIs from selected components

        Parameters
        ------------
        components : None, int, or list of ints
             if None, rebuilds SI from all components
             if int, rebuilds SI from components in range 0-given int
             if list of ints, rebuilds SI from only components in given list
        mva_type : string, currently either 'pca' or 'ica'
             (not case sensitive)

        Returns
        -------
        Signal instance
        """

        if mva_type.lower() == 'pca':
            factors = self.mva_results.pc
            scores = self.mva_results.v.T
        elif mva_type.lower() == 'ica':
            factors = self.ic
            scores = self._get_ica_scores()
        if components is None:
            a = np.atleast_3d(np.dot(factors,scores))
            signal_name = 'rebuilt from %s with %i components' % (
            mva_type,factors.shape[1])
        elif hasattr(components, '__iter__'):
            tfactors = np.zeros((factors.shape[0],len(components)))
            tscores = np.zeros((len(components),scores.shape[1]))
            for i in xrange(len(components)):
                tfactors[:,i] = factors[:,components[i]]
                tscores[i,:] = scores[components[i],:]
            a = np.atleast_3d(np.dot(tfactors, tscores))
            signal_name = 'rebuilt from %s with components %s' % (
            mva_type,components)
        else:
            a = np.atleast_3d(np.dot(factors[:,:components], 
                                     scores[:components,:]))
            signal_name = 'rebuilt from %s with %i components' % (
            mva_type,components)

        self._unfolded4pca = self.unfold_if_multidim()

        sc = self.deepcopy()
        dc_transposed = False
        import eelslab.signals.spectrum
        if isinstance(self, eelslab.signals.spectrum.Spectrum):
            print "Transposing data so that energy axis makes up rows."
            sc.data = a.T.squeeze()
        else:
            sc.data = a.squeeze()
        sc.name = signal_name
        if self._unfolded4pca is True:
            self.fold()
            sc.history = ['unfolded']
            sc.fold()
        return sc

    def pca_build_SI(self,components=None):
        """Return the spectrum generated with the selected number of principal 
        components
        
        Parameters
        ------------
        components : None, int, or list of ints
             if None, rebuilds SI from all components
             if int, rebuilds SI from components in range 0-given int
             if list of ints, rebuilds SI from only components in given list
        
        Returns
        -------
        Signal instance
        """
        return self._calculate_recmatrix(components=components, mva_type='pca')
        
    def ica_build_SI(self,components = None):
        """Return the spectrum generated with the selected number of 
        independent components
        
        Parameters
        ------------
        components : None, int, or list of ints
             if None, rebuilds SI from all components
             if int, rebuilds SI from components in range 0-given int
             if list of ints, rebuilds SI from only components in given list        
        
        Returns
        -------
        Signal instance
        """
        return self._calculate_recmatrix(components=components, mva_type='ica')
        
    def energy_center(self):
        """Subtract the mean energy pixel by pixel"""
        print "\nCentering the energy axis"
        self._energy_mean = np.mean(self.data, 0)
        self.data = (self.data - self._energy_mean)
        self._replot()
        
    def undo_energy_center(self):
        if hasattr(self,'_energy_mean'):
            self.data = (self.data + self._energy_mean)
            self._replot()
        
    def variance2one(self):
        # Whitening
        self._std = np.std(self.data, 0)
        self.data /= self._std
        self._replot()
        
    def undo_variance2one(self):
        if hasattr(self,'_std'):
            self.data *= self._std
            self._replot()

    def plot_lev(self, n=50):
        """Plot the principal components LEV up to the given number
        
        Parameters
        ----------
        n : int
        """       
	if n>self.mva_results.V.shape[0]:
            n=self.mva_results.V.shape[0]
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
        """Plot the principal components explained variance up to the given 
        number
        
        Parameters
        ----------
        n : int
        """ 
	if n>self.mva_results.V.shape[0]:
            n=self.mva_results.V.shape[0]
        cumu = np.cumsum(self.mva_results.V) / np.sum(self.mva_results.V)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(range(n), cumu[:n])
        ax.set_xlabel('Principal component')
        ax.set_ylabel('Explained variance')
        plt.draw()
        plt.show()
        return ax

    def plot_principal_components(self, n = None):
        """Plot the principal components up to the given number
        
        Parameters
        ----------
        n : int
            number of principal components to plot.
        """
        if n is None:
            n = self.mva_results.pc.shape[1]
        for i in xrange(n):
            plt.figure()
            plt.plot(self.axes_manager.axes[-1].axis, self.mva_results.pc[:,i])
            plt.title('Principal component %s' % i)
            plt.xlabel('Energy (eV)')
            
    def plot_independent_components(self, ic=None, same_window=False):
        """Plot the independent components.
        
        Parameters
        ----------
        ic : numpy array (optional)
             externally provided independent components array
             The shape of 'ic' must be (channels, n_components),
             so that e.g. ic[:, 0] is the first independent component.

        same_window : bool (optional)
                    if 'True', the components will be plotted in the
                    same window. Default is 'False'.
        """
        if ic is None:
            ic = self.ic
            x = self.axes_manager.axes[-1].axis
        else:
            if len(ic.shape) != 2:
                raise ShapeError(ic)
            x = ic.shape[1]     # no way that we know the calibration
            
        n = ic.shape[1]

        if not same_window:
            for i in xrange(n):
                plt.figure()
                plt.plot(x, ic[:, i])
                plt.title('Independent component %s' % i)
                plt.xlabel('Energy (eV)')
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            for i in xrange(n):
                # ic = ic / ic.sum(axis=0) # normalize
                lbl = 'IC %i' % i
                # print 'plotting %s' % lbl
                ax.plot(x, ic[:, i], label=lbl)
            col = (ic.shape[1]) // 2
            ax.legend(ncol=col, loc='best')
            ax.set_xlabel('Energy (eV)')
            ax.set_title('Independent components')
            plt.draw()
            plt.show()

    def plot_maps(self, components, mva_type=None, scores=None, factors=None, cmap=plt.cm.gray,
                  no_nans=False, with_components=True, plot=True):
        """
        Plot component maps for the different MSA types

        Parameters
        ----------
        components : None, int, or list of ints
            if None, returns maps of all components.
            if int, returns maps of components with ids from 0 to given int.
            if list of ints, returns maps of components with ids in given list.
        mva_type: string, currently either 'pca' or 'ica'
        scores: numpy array, the array of score maps
        factors: numpy array, the array of components, with each column as a component.
        cmap: matplotlib colormap instance
        no_nans: bool, 
        with_components: bool,
        plot: bool, 
        """
        from eelslab.signals.image import Image
        from eelslab.signals.spectrum import Spectrum

        if scores is None or (factors is None and with_components is True):
            print "Either recmatrix or components were not provided."
            print "Loading existing values from object."
            if mva_type is None:
                print "No scores nor analysis type specified.  Cannot proceed."
                return
            
            elif mva_type.lower() == 'pca':
                scores=self.mva_results.v.T
                factors=self.mva_results.pc
            elif mva_type.lower() == 'ica':
                scores = self._get_ica_scores()
                factors=self.ic
                if no_nans:
                    print 'Removing NaNs for a visually prettier plot.'
                    scores = np.nan_to_num(scores) # remove ugly NaN pixels
            else:
                print "No scores provided and analysis type '%s' unrecognized. Cannot proceed."%mva_type
                return

#        if len(self.axes_manager.axes)==2:
#            shape=self.data.shape[0],1
#        else:
#            shape=self.data.shape[0],self.data.shape[1]
        im_list = []
        
        if components is None:
            components=xrange(factors.shape[1])

        elif type(components).__name__!='list':
            components=xrange(components)

        for i in components:
            if plot is True:
                figure = plt.figure()
                if with_components:
                    ax = figure.add_subplot(121)
                    ax2 = figure.add_subplot(122)
                else:
                    ax = figure.add_subplot(111)
            if self.axes_manager.navigation_dim == 2:
                toplot = scores[i,:].reshape(self.axes_manager.navigation_shape)
                im_list.append(Image({'data' : toplot, 
                    'axes' : self.axes_manager._get_non_slicing_axes_dicts()}))
                if plot is True:
                    mapa = ax.matshow(toplot, cmap = cmap)
                    if with_components:
                        ax2.plot(self.axes_manager.axes[-1].axis, factors[:,i])
                        ax2.set_title('%s component %i' % (mva_type.upper(),i))
                        ax2.set_xlabel('Energy (eV)')
                    figure.colorbar(mapa)
                    figure.canvas.draw()
                    #pointer = widgets.DraggableSquare(self.coordinates)
                    #pointer.add_axes(ax)
            elif self.axes_manager.navigation_dim == 1:
                toplot = scores[i,:]
                im_list.append(Spectrum({"data" : toplot, 
                    'axes' : self.axes_manager._get_non_slicing_axes_dicts()}))
                im_list[-1].get_dimensions_from_data()
                if plot is True:
                    ax.step(range(len(toplot)), toplot)
                    
                    if with_components:
                        ax2.plot(self.axes_manager.axes[-1].axis, factors[:,i])
                        ax2.set_title('%s component %s' % (mva_type.upper(),i))
                        ax2.set_xlabel('Energy (eV)')
            else:
                messages.warning_exit('View not supported')
            if plot is True:
                ax.set_title('%s component number %s map' % (mva_type.upper(),i))
                figure.canvas.draw()
        return im_list

    def plot_principal_components_maps(self, comp_ids=None, cmap=plt.cm.gray, recmatrix=None,
                                         with_pc=True, plot=True, pc=None):
        """Plot the map associated to each independent component
        
        Parameters
        ----------
        comp_ids : None, int, or list of ints
            if None, returns maps of all components.
            if int, returns maps of components with ids from 0 to given int.
            if list of ints, returns maps of components with ids in given list.
        cmap : plt.cm object
        recmatrix : numpy array
            externally suplied recmatrix
        with_ic : bool
            If True, plots also the corresponding independent component in the 
            same figure
        plot : bool
            If True it will plot the figures. Otherwise it will only return the 
            images.
        ic : numpy array
            externally supplied independent components
        no_nans : bool (optional)
             whether substituting NaNs with zeros for a visually prettier plot
             (default is False)

        Returns
        -------
        List with the maps as MVA instances
        """
        return self.plot_maps(components=comp_ids,mva_type='pca',cmap=cmap,scores=recmatrix,
                        with_components=with_pc,plot=plot, factors=pc)
        
    def plot_independent_components_maps(self, comp_ids=None, cmap=plt.cm.gray, recmatrix=None,
                                         with_ic=True, plot=True, ic=None, no_nans=False):
        """Plot the map associated to each independent component
        
        Parameters
        ----------
        cmap : plt.cm object
        recmatrix : numpy array
            externally suplied recmatrix
        comp_ids : int or list of ints
            if None, returns maps of all components.
            if int, returns maps of components with ids from 0 to given int.
            if list of ints, returns maps of components with ids in given list.
        with_ic : bool
            If True, plots also the corresponding independent component in the 
            same figure
        plot : bool
            If True it will plot the figures. Otherwise it will only return the 
            images.
        ic : numpy array
            externally supplied independent components
        no_nans : bool (optional)
             whether substituting NaNs with zeros for a visually prettier plot
             (default is False)
        Returns
        -------
        List with the maps as MVA instances
        """
        return self.plot_maps(components=comp_ids,mva_type='ica',cmap=cmap,scores=recmatrix,
                        with_components=with_ic,plot=plot, factors=ic, no_nans=no_nans)

    
    def save_principal_components(self, n, spectrum_prefix = 'pc', 
    image_prefix = 'im', spectrum_format = 'msa', image_format = 'tif'):
        """Save the `n` first principal components  and score maps 
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
                 
        """
        from spectrum import Spectrum
        im_list = self.plot_principal_components_maps(n, plot = False)
        s = Spectrum({'calibration' : {'data_cube' : self.mva_results.pc[:,0]}})
        s.get_calibration_from(self)
        for i in xrange(n):
            s.data_cube = self.mva_results.pc[:,i]
            s.get_dimensions_from_cube()
            s.save('%s-%i.%s' % (spectrum_prefix, i, spectrum_format))
            im_list[i].save('%s-%i.%s' % (image_prefix, i, image_format))
        
    def save_independent_components(self, elements=None, 
                                    spectrum_format='msa',
                                    image_format='tif',
                                    recmatrix=None, ic=None):
        """Saves the result of the ICA in image and spectrum format.
        Note that to save the image, the NaNs in the map will be converted
        to zeros.
        
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
        """
        from eelslab.signals.spectrum import Spectrum
        pl = self.plot_independent_components_maps(plot=False, 
                                                   recmatrix=recmatrix,
                                                   ic=ic,
                                                   no_nans=True)
        if ic is None:
            ic = self.ic
        if self.data.shape[2] > 1:
            maps = True
        else:
            maps = False
        for i in xrange(ic.shape[1]):
            axes = (self.axes_manager._slicing_axes[0].get_axis_dictionary(),)
            axes[0]['index_in_array'] = 0
            sp = Spectrum({'data' : ic[:,i], 'axes' : axes})
            sp.data_cube = ic[:,i].reshape((-1,1,1))

            if elements is None:
                sp.save('ic-%s.%s' % (i, spectrum_format))
                if maps is True:
                    pl[i].save('map_ic-%s.%s' % (i, image_format))
                else:
                    pl[i].save('profile_ic-%s.%s' % (i, spectrum_format))
            else:
                element = elements[i]
                sp.save('ic-%s.%s' % (element, spectrum_format))
                if maps:
                    pl[i].save('map_ic-%s.%s' % (element, image_format))
                else:
                    pl[i].save('profile_ic-%s.%s' % (element, spectrum_format))
                    
    def snica(self, coordinates = None):
        """Stochastic Non-negative Independent Component Analysis.
        
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
        """
        if coordinates is None:
            fold_back = utils.self.unfold_if_multidim()
            ic = self.data.squeeze()
        else:
            clist = []
            for coord in coordinates:
                clist.append(self.data[:,coord[0], coord[1]])
            ic = np.array(clist).T
        snica = utils.snica(ic)
        self.ic = snica[0]
        if fold_back is True: self.fold()
        return snica
    
    def als(self, **kwargs):
        """Alternate Least Squares imposing positivity constraints 
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
        """
        shape = (self.data.shape[2], self.data.shape[1],-1)
        if hasattr(self, 'ic') and (self.ic is not None):
            also = utils.ALS(self, **kwargs)
            self.als_ic = also['S']
            self.als_maps = also['CList'].reshape(shape, order = 'C')
            self.als_output = also
            
    def plot_als_ic_maps(self):
        """Same as plot_ic_maps for the ALS results"""
        return self.plot_independent_components_maps(recmatrix = 
        self.als_output['CList'].T, ic = self.als_ic)
    
    def plot_als_ic(self):
        """Same as plot_independent_componets for the ALS results"""
        self.plot_independent_components(ic = self.als_ic)
        
    def save_als_ica_results(self, elements = None, 
    format = defaults.file_format, image_format = 'tif'):
        """Same as save_ica_results for the ALS results"""
        self.save_ica_results(elements = elements, image_format = image_format, 
        recmatrix = self.als_output['CList'].T, ic = self.als_ic)
                
    def normalize_poissonian_noise(self, spatial_mask = None, 
                                   energy_mask = None, return_masks = False):
        """
        Scales the SI following Surf. Interface Anal. 2004; 36: 203–212 to
        "normalize" the poissonian data for PCA analysis
        
        Parameters
        ----------
        spatial_mask : boolen numpy array
        energy_mask  : boolen numpy array
        """
        messages.information(
            "Scaling the data to normalize the (presumably) Poissonian noise")
        # If energy axis is not first, it needs to be for MVA.
        refold = self.unfold_if_multidim()
        dc_transposed=False
        last_axis_units=self.axes_manager.axes[-1].units
        if last_axis_units=='eV' or last_axis_units=='keV':
            # don't print this here, since PCA will have already printed it.
            # print "Transposing data so that energy axis makes up rows."
            dc = self.data.T.squeeze()
            dc_transposed=True
        else:
            dc = self.data.squeeze()
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
                spatial_mask = np.ones((self.data.shape[1]),dtype = 'bool')
            # Set colums summing zero as masked
            spatial_mask[aG0] = False
            aG = aG[aG0 == False]
        if bH0.any():
            if isinstance(energy_mask, slice):
                # Convert the slice into a mask before setting its values
                energy_mask = np.ones((self.data.shape[0]), dtype = 'bool')
            # Set rows summing zero as masked
            energy_mask[bH0] = False
            bH = bH[bH0 == False]
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
        # TODO - dc was never modifying self.data - was normalization ever
        # really getting applied?  Comment next lines as necessary.
        if dc_transposed:
            # don't print this here, since PCA will print it.
            # print "Undoing data transpose."
            self.data=dc.T
        else:
            self.data=dc
        # end normalization write to self.data.
        if refold is True:
            print "Automatically refolding the SI after scaling"
            self.fold()
        if return_masks is True:
            if isinstance(spatial_mask, slice):
                spatial_mask = None
            if isinstance(energy_mask, slice):
                energy_mask = None
            return spatial_mask, energy_mask
        
    def undo_treatments(self):
        """Undo normalize_poissonian_noise"""
        print "Undoing data pre-treatments"
        self.data=self._data_before_treatments
        del self._data_before_treatments
        
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
        """Save the result of the PCA analysis
        
        Parameters
        ----------
        filename : string
        """
        np.savez(filename, pc = self.pc, v = self.v, V = self.V, 
        pca_algorithm = self.pca_algorithm, centered = self.centered, 
        output_dim = self.output_dim, variance2one = self.variance2one, 
        poissonian_noise_normalized = self.poissonian_noise_normalized, 
        w = self.w, ica_algorithm = self.ica_algorithm)

    def load(self, filename):
        """Load the result of the PCA analysis
        
        Parameters
        ----------
        filename : string
        """
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

    def peak_pca(self):
        self.principal_component_analysis(self.peak_chars)
        pass

    def peak_ica(self, number_of_components):
        pass
        
    def summary(self):
        """Prints a summary of the PCA parameters to the stdout
        """
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
        """
        Crop the score matrix up to the given number.
        
        It is mainly useful to save memory and redude the storage size
        """
        self.v = self.v[:,:n].copy()
