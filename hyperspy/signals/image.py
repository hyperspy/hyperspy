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


from hyperspy.signal import Signal
import hyperspy.peak_char as pc
from hyperspy.misc import utils_varia
from hyperspy.learn.mva import MVA_Results
from hyperspy import messages

import numpy as np
import matplotlib.pyplot as plt
import os

from mpl_toolkits.axes_grid1 import make_axes_locatable

class Image(Signal):
    """
    """    
    def __init__(self, *args, **kw):
        super(Image,self).__init__(*args, **kw)
        self.axes_manager.set_view('image')
        self.target_locations=None
        self.peak_width=None
        self.peak_mva_results=MVA_Results()

    def peak_char_stack(self, peak_width, subpixel=False, target_locations=None,
                        peak_locations=None, imcoords=None, target_neighborhood=20,
                        medfilt_radius=5):
        """
        Characterizes the peaks in the stack of images.  Creates a class member
        "peak_chars" that is a 2D array of the following form:
        - One column per image
        - 7 rows per peak located.  These rows are, in order:
            0-1: x,y coordinate of peak
            2-3: aberration of this peak from its target value
            4: height of the peak
            5: orientation of the peak
            6: eccentricity of the peak
        - optionally, 2 additional rows at the end containing the coordinates
           from which the image was cropped (should be passed as the imcoords 
           parameter)  These should be excluded from any MVA.

        Parameters:
        ----------

        peak_width : int (required)
                expected peak width.  Affects subpixel precision fitting window,
		which takes the center of gravity of a box that has sides equal
		to this parameter.  Too big, and you'll include other peaks.
        
        subpixel : bool (optional)
                default is set to False

        target_locations : numpy array (n x 2) (optional)
                array of n target locations.  If left as None, will create 
                target locations by locating peaks on the average image of the stack.
                default is None (peaks detected from average image)

        peak_locations : numpy array (n x m x 2) (optional)
                array of n peak locations for m images.  If left as None,
                will find all peaks on all images, and keep only the ones closest to
                the peaks specified in target_locations.
                default is None (peaks detected from average image)

        imcoords : numpy array (n x 2) (optional)
                array of n coordinates, to keep track of locations from which
                sub-images were cropped.  Critical for plotting results.

        target_neighborhood : int (optional)
                pixel neighborhood to limit peak search to.  Peaks outside the
                square defined by 2x this value around the peak will be excluded
                from any fitting.
        
        medfilt_radius : int (optional)
                median filter window to apply to smooth the data
                (see scipy.signal.medfilt)
                if 0, no filter will be applied.
                default is set to 5
        
        """
        self.target_locations=target_locations
        self.peak_width=peak_width
        self.peak_chars=pc.peak_attribs_stack(self.data, 
                                              peak_width,
                                              subpixel=subpixel, 
                                              target_locations=target_locations,
                                              peak_locations=peak_locations, 
                                              imcoords=imcoords, 
                                              target_neighborhood=target_neighborhood,
                                              medfilt_radius=medfilt_radius
                                              )

#==============================================================================
# Plotting methods
#==============================================================================

    def plot_image_peaks(self, index=0, peak_width=10, subpixel=False,
                       medfilt_radius=5):
        # TODO: replace with hyperimage explorer
        plt.imshow(self.data[index,:,:],cmap=plt.gray())
        peaks=pc.two_dim_peakfind(self.data[index,:,:], subpixel=subpixel,
                                  peak_width=peak_width, 
                                  medfilt_radius=medfilt_radius)
        plt.scatter(peaks[:,0],peaks[:,1])

    def plot_peak_ids(self):
        """Overlays id numbers for identified peaks on an average image of the
        stack.  Identified peaks are either those you specified as 
        target_locations, or if not specified, those automatically 
        identified from the average image.

        Use this function to identify a peak to overlay a characteristic of
        onto the original experimental image using the plot_image_overlay
        function.
        """
        f=plt.figure()
        imgavg=np.average(self.data,axis=0)
        plt.imshow(imgavg)
        plt.gray()
        if self.target_locations is None:
            # identify the peaks on the average image
            if self.peak_width is None:
                self.peak_width=10
            self.target_locations=pc.peak_attribs_image(imgavg, self.peak_width)[:,:2]
        # plot the peak labels
        for pk_id in xrange(self.target_locations.shape[0]):
            plt.text(self.target_locations[pk_id,0], self.target_locations[pk_id,1], 
                     "%s"%pk_id, size=10, rotation=0.,
                     ha="center", va="center",
                     bbox = dict(boxstyle="round",
                                 ec=(1., 0.5, 0.5),
                                 fc=(1., 0.8, 0.8),
                                 )
                     )
        return f

    def plot_image_overlay(self, plot_component=None, mva_type='PCA', 
                           peak_mva=True, peak_id=None, plot_char=None, 
                           plot_shift=False):
        """Overlays scores, or some peak characteristic on top of an image
        plot of the original experimental image.  Useful for obtaining a 
        bird's-eye view of some image characteristic.

        plot_component - None or int 
        (optional, but required to plot score overlays)
            The integer index of the component to plot scores for.
            Creates a scatter plot that is colormapped according to 
            score values.

        mva_type - string, either 'PCA' or 'ICA' (case insensitive)
        (optional, but required to plot score overlays)
            Choose between the components that will be used for plotting
            component maps.  Note that whichever analysis you choose
            here has to actually have already been performed.

        peak_mva - bool (default is False)
        (optional, if True, the peak characteristics, shifts, and components
            are drawn from the mva_results done on the peak characteristics.
            Namely, these are the self.peak_mva_results attribute.

        peak_id - None or int
        (optional, but required to plot peak characteristic and shift overlays)
            If int, the peak id for plotting characteristics of.
            To identify peak id's, use the plot_peak_ids function, which will
            overlay the average image with the identified peaks used
            throughout the image series.

        plot_char - None or int
        (optional, but required to plot peak characteristic overlays)
            If int, the id of the characteristic to plot as the colored 
            scatter plot.
            Possible components are:
               4: peak height
               5: peak orientation
               6: peak eccentricity

        plot_shift - bool, optional
            If True, plots shift overlays for given peak_id onto the parent image(s)

        """
        if not hasattr(self.mapped_parameters, "original_files"):
            messages.warning("""No original files available.  Can't map anything to nothing.
If you use the cell_cropper function to crop your cells, the cell locations and original files 
will be tracked for you.""")
            return None
        if peak_id is not None and (plot_shift is False and plot_char is None):
            messages.warning("""Peak ID provided, but no plot_char given , and plot_shift disabled.
Nothing to plot.  Try again.""")
            return None
        if peak_mva and not (plot_char is not None or plot_shift or plot_component):
            messages.warning("""peak_mva specified, but no peak characteristic, peak \
shift, or component score selected for plotting.  Nothing to plot.""")
            return None
        if plot_char is not None and plot_component is not None:
            messages.warning("""Both plot_char and plot_component provided.  Can only plot one
of these at a time.  Try again.

Note that you can actually plot shifts and component scores simultaneously.""")
            return None
        figs=[]
        for key in self.mapped_parameters.original_files.keys():
            f=plt.figure()
            plt.title(key)
            plt.imshow(self.mapped_parameters.original_files[key].data)
            plt.gray()
            # get a shorter handle on the peak locations on THIS image
            locs=self.mapped_parameters.locations
            # binary mask to exclude peaks from other images
            mask=locs['filename']==key
            mask=mask.squeeze()
            # grab the array of peak locations, only from THIS image
            locs=locs[mask]['position'].squeeze()
            char=[]                
            if peak_id is not None and plot_char is not None :
                # list comprehension to obtain the peak characteristic
                # peak_id selects the peak
                # multiply by 7 because each peak has 7 characteristics
                # add the index of the characteristic of interest
                # mask.nonzero() identifies the indices corresponding to peaks 
                #     from this image (if many origins are present for this 
                #     stack).  This selects the column, only plotting peak 
                #     characteristics that are from this image.
                char=np.array([char.append(self.peak_chars[peak_id*7+plot_char,
                               mask.nonzero()[x]]) for x in xrange(locs.shape[0])])
                plt.scatter(locs[:,0],locs[:,1],c=char)
            if peak_id is not None and plot_shift is not None:
                # list comprehension to obtain the peak shifts
                # peak_id selects the peak
                # multiply by 7 because each peak has 7 characteristics
                # add the indices of the peak shift [2:4]
                # mask.nonzero() identifies the indices corresponding to peaks 
                #    from this image (if many origins are present for this 
                #    stack).  This selects the column, only plotting shifts 
                #    for peaks that are from this image.
                shifts=np.array([char.append(self.peak_chars[peak_id*7+2:peak_id*7+4,
                               mask.nonzero()[x]]) for x in xrange(locs.shape[0])])
                plt.quiver(locs[:,0],locs[:,1],
                           shifts[:,0], shifts[:,1],
                           units='xy', color='white'
                           )
            if plot_component is not None:
                if peak_mva: target=self.peak_mva_results
                else: target=self.mva_results
                if mva_type.upper() == 'PCA':
                    scores=target.v[plot_component][mask]
                elif mva_type.upper() == 'ICA':
                    scores=target.ica_scores[plot_component][mask]
                else:
                    messages.warning("Unrecognized MVA type.  Currently supported MVA types are \
PCA and ICA (case insensitive)")
                    return None
                print mask
                print locs
                print scores
                plt.scatter(locs[:,0],locs[:,1],c=scores)
                plt.jet()
                plt.colorbar()
            figs.append(f)
        return figs
        
    def plot_cell_overlays(self, plot_component=None, mva_type='PCA', peak_mva=True,
                                plot_shifts=True, plot_char=None):
        """Overlays peak characteristics on an image plot of the average image.

        Only appropriate for Image objects that consist of 3D stacks of cropped
        data.

        Parameters:

        plot_component - None or int
            The integer index of the component to plot scores for.
            If specified, the values plotted for the shifts (if enabled by the plot_shifts flag)
            and the values plotted for the plot characteristics (if enabled by the plot_char flag)
            will be drawn from the given component resulting from MVA on the peak characteristics.
            NOTE: only makes sense right now for examining results of MVA on peak characteristics,
                NOT MVA results on the images themselves (factor images).

        mva_type - str, 'PCA' or 'ICA', case insensitive. default is 'PCA'
            Choose between the components that will be used for plotting
            component maps.  Note that whichever analysis you choose
            here has to actually have already been performed.            

        peak_mva - bool, default is True
            If True, draws the information to be plotted from the mva results derived
            from peak characteristics.  If False, does the following with Factor images:
            - Reconstructs the data using all available components
            - locates peaks on all images in reconstructed data
            - reconstructs the data using all components EXCEPT the component specified
                by the plot_component parameter
            - locates peaks on all images in reconstructed data
            - subtracts the peak characteristics of the first (complete) data from the
                data without the component included.  This difference data is what gets
                plotted.

        plot_shifts - bool, default is True
            If true, plots a quiver (arrow) plot showing the shifts for each
            peak present in the component being plotted.

        plot_char - None or int
            If int, the id of the characteristic to plot as the colored 
            scatter plot.
            Possible components are:
               4: peak height
               5: peak orientation
               6: peak eccentricity

        """
        f=plt.figure()

        imgavg=np.average(self.data,axis=0)

        if self.target_locations is None:
            # identify the peaks on the average image
            if self.peak_width is None:
                self.peak_width=10
            self.target_locations=pc.peak_attribs_image(imgavg, self.peak_width)[:,:2]

        stl=self.target_locations

        shifts=np.zeros((stl.shape[0],2))
        char=np.zeros(stl.shape[0])

        if plot_component is not None:
            # get the mva_results (components) for the peaks
            if mva_type.upper()=='PCA':
                component=self.peak_mva_results.pc[:,plot_component]
            elif mva_type.upper()=='ICA':
                component=self.peak_mva_results.ic[:,plot_component]          

        for pos in xrange(stl.shape[0]):
            shifts[pos]=component[pos*7+2:pos*7+4]
            if plot_char:
                char[pos]=component[pos*7+plot_char]

        plt.imshow(imgavg)
        plt.gray()

        if plot_shifts:
            plt.quiver(stl[:,0],stl[:,1],
                       shifts[:,0], shifts[:,1],
                       units='xy', color='white'
                       )
        if plot_char is not None :
            plt.scatter(stl[:,0],stl[:,1],c=char)
            plt.jet()
            plt.colorbar()
        return f

    def _plot_pc(self, idx, on_peaks=False):
        target=self._get_target(on_peaks)
        ax=plt.gca()
        im=ax.imshow(target.pc[:,idx].reshape(self.axes_manager.axes[1].size,self.axes_manager.axes[2].size))
        plt.title('PC %s' % idx)
        div=make_axes_locatable(ax)
        cax=div.append_axes("right",size="5%",pad=0.05)
        plt.colorbar(im,cax=cax)

        

    def plot_principal_components(self, n = None, same_window=True, per_row=3, 
                                  on_peaks=False):
        """Plot the principal components up to the given number

        Parameters
        ----------
        n : int
            number of principal components to plot.

        same_window : bool (optional)
                    if 'True', the components will be plotted in the
                    same window. Default is 'False'.

        per_row : int (optional)
                    When same_window is True, this is the number of plots
                    per row in the single window.

        on_peaks : bool (optional)
        """
        target=self._get_target(on_peaks)
        if n is None:
            n = target.pc.shape[1]
        if not same_window:
            for i in xrange(n):
                plt.figure()
                self._plot_pc(i,on_peaks)

        else:
            fig = plt.figure()
            rows=int(np.ceil(n/float(per_row)))
            idx=0
            for i in xrange(rows):
                for j in xrange(per_row):
                    if idx<n:
                        fig.add_subplot(rows,per_row,idx+1)
                        self._plot_pc(idx,on_peaks)
                        idx+=1
            plt.suptitle('Principal components')
            plt.draw()

    def _plot_ic(self, idx, on_peaks=False):
        target=self._get_target(on_peaks)
        ax=plt.gca()
        im=ax.imshow(target.ic[:,idx].reshape(self.axes_manager.axes[1].size,self.axes_manager.axes[2].size))
        plt.title('IC %s' % idx)
        div=make_axes_locatable(ax)
        cax=div.append_axes("right",size="5%",pad=0.05)
        plt.colorbar(im,cax=cax)


    def plot_independent_components(self, ic=None, same_window=True,
                                    per_row=3, on_peaks=False):
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

        per_row : int (optional)
                    When same_window is True, this is the number of plots
                    per row in the single window.

        on_peaks : bool (optional)
        """
        target=self._get_target(on_peaks)
        if ic is None:
            ic = target.ic
            x = self.axes_manager.axes[-1].axis
            x = ic.shape[1]     # no way that we know the calibration

        n = ic.shape[1]

        if not same_window:
            for i in xrange(n):
                plt.figure()
                self._plot_ic(i, on_peaks)

        else:
            fig = plt.figure()
            rows=int(np.ceil(n/float(per_row)))
            idx=0
            for i in xrange(rows):
                for j in xrange(per_row):
                    if idx<n:
                        fig.add_subplot(rows,per_row,idx+1)
                        self._plot_ic(idx, on_peaks)
                        idx+=1
            plt.suptitle('Independent components')

    def plot_maps(self, components, mva_type=None, scores=None, factors=None,
                  cmap=plt.cm.gray, no_nans=False, per_row=3, on_peaks=False, 
                  save_figs=False, directory = None):
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
        
        per_row: int (optional)

        on_peaks: bool (optional)

        save_figs: bool (optional)
            If true, saves figures at 600 dpi to directory.  If directory is None,
            saves to current working directory.
        """
        from hyperspy.signals.image import Image
        from hyperspy.signals.spectrum import Spectrum

        target=self._get_target(on_peaks)

        if scores is None or (factors is None and with_components is True):
            if mva_type is None:
                messages.warning(
                "Neither scores nor analysis type specified.  Cannot proceed.")
                return

            elif mva_type.lower() == 'pca':
                scores=target.v.T
                factors=target.pc
            elif mva_type.lower() == 'ica':
                scores = self._get_ica_scores(target)
                factors=target.ic
                if no_nans:
                    messages.information(
                        'Removing NaNs for a visually prettier plot.')
                    scores = np.nan_to_num(scores) # remove ugly NaN pixels
            else:
                messages.warning(
                    "No scores provided and analysis type '%s' unrecognized"  
                    % mva_type)
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
            figure = plt.figure()
            if self.axes_manager.navigation_dimension == 2:
                # 4D data - 2D arrays of diffraction patterns?
                messages.warning('View not supported')
            elif self.axes_manager.navigation_dimension == 1:
                locs=self.mapped_parameters.locations
                if hasattr(self.mapped_parameters,"original_files"):
                    parents=self.mapped_parameters.original_files
                else:
                    if not hasattr(self.mapped_parameters,'parent'):
                        messages.warning('No parent image - mapping not possible')
                        return None
                    else:
                        parents={self.mapped_parameters.parent.mapped_parameters.name:self.mapped_parameters.parent}
                idx=0
                keys=parents.keys()
                rows=int(np.ceil((len(keys)+1)/float(per_row)))
                if (len(keys)+1)<per_row:
                    per_row=len(keys)+1
                # plot factor image first
                figure.add_subplot(rows,per_row,1)
                plt.gray()
                if mva_type.upper()=='PCA':
                    self._plot_pc(i,on_peaks)
                elif mva_type.upper()=='ICA':
                    self._plot_ic(i,on_peaks)
                for j in xrange(rows):
                    for k in xrange(per_row):
                        # plot score maps overlaid on experimental images
                        if idx<len(keys):
                            ax=figure.add_subplot(rows,per_row,idx+2)
                            # p is the parent image that we're working with
                            p=keys[idx]
                            # the locations of peaks on that parent
                            # binary mask to exclude peaks from other images
                            mask=locs['filename']==p
                            mask=mask.squeeze()
                            # grab the array of peak locations, only from THIS image
                            loc=locs[mask]['position'].squeeze()
                            plt.imshow(parents[keys[idx]].data)
                            plt.gray()
                            sc=ax.scatter(loc[:,0], loc[:,1],
                                        c=scores[i].squeeze()[mask],
                                        cmap='jet')
                            shp=parents[keys[idx]].data.shape
                            plt.xlim(0,shp[1])
                            plt.ylim(shp[0],0)
                            div=make_axes_locatable(ax)
                            cax=div.append_axes("right",size="5%",pad=0.05)
                            plt.colorbar(sc,cax=cax)
                            idx+=1
            else:
                messages.warning('View not supported')
            if save_figs:
                #ax.set_title('%s component number %s map' % (mva_type.upper(),i))
                #figure.canvas.draw()
                if directory is not None:
                    if not os.path.isdir(directory):
                        os.makedirs(directory)
                    figure.savefig(os.path.join(directory, '%s-map-%i.png' % (mva_type.upper(),i)),
                                      dpi = 600)
                else:
                    figure.savefig( '%s-map-%i.png' % (mva_type.upper(),i),
                              dpi = 600)


    def (self, comp_ids=None, cmap=plt.cm.gray,
                                       recmatrix=None, plot=True, pc=None, 
                                       on_peaks=False, save_figs=False):
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
        return self.plot_maps(components=comp_ids,mva_type='pca',cmap=cmap,
                              scores=recmatrix, factors=pc, on_peaks=on_peaks,
                              save_figs=save_figs)

    def plot_independent_components_maps(self, comp_ids=None, cmap=plt.cm.gray,
                                         recmatrix=None, ic=None, no_nans=False,
                                         on_peaks=False, save_figs=False, 
                                         directory = None):
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
        return self.plot_maps(components=comp_ids,mva_type='ica',cmap=cmap,
                              scores=recmatrix, factors=ic, no_nans=no_nans,
                              on_peaks=on_peaks, save_figs=save_figs, 
                              directory = directory)


    def save_principal_components(self, n, pc_prefix = 'pc',
    score_prefix = 'score', spectrum_format = 'msa', hs_format = 'tif',
                                  on_peaks=False):
        """Save the `n` first principal components  and score maps
        in the specified format

        Parameters
        ----------
        n : int
            Number of principal components to save
        score_prefix : string
            Prefix for the score file names
        pc_prefix : string
            Prefix for the principal component file names
        spectrum_format : string
            Any of Hyperspy's supported file formats for spectral data
        hs_format : string
            Any of Hyperspy's supported file formats for hyperspectral data

        """
        from hyperspy.signals.spectrum import Spectrum
        target=self._get_target(on_peaks)
        im_list = self.(n, plot = False,
                                                      on_peaks=on_peaks)
        axis_dict = self.axes_manager._non_slicing_axes[0].get_axis_dictionary()
        axis_dict['index_in_array'] = 0
        s = Spectrum({'data' : target.pc[:,0],
                      'axes' : [axis_dict,]})
        for i in xrange(n):
            s.data = target.pc[:,i]
            s.save('%s-%i.%s' % (pc_prefix, i, spectrum_format))
            im_list[i].save('%s-%i.%s' % (score_prefix, i, hs_format))

    def save_independent_components(self, elements=None,
                                    spectrum_format='msa',
                                    hs_format='tif',
                                    recmatrix=None, ic=None,
                                    on_peaks=False):
        """Saves the result of the ICA in image and spectrum format.
        Note that to save the image, the NaNs in the map will be converted
        to zeros.

        Parameters
        ----------
        elements : None or tuple of strings
            a list of names (normally an element) to be assigned to IC. If not
            the will be name ic-0, ic-1 ...
        hs_format : string
        spectrum_format : string
        recmatrix : None or numpy array
            externally supplied recmatrix
        ic : None or numpy array
            externally supplied IC
        """
        from hyperspy.signals.spectrum import Spectrum
        target=self._get_target(on_peaks)
        pl = self.plot_independent_components_maps(plot=False,
                                                   recmatrix=recmatrix,
                                                   ic=ic,
                                                   no_nans=True,
                                                   on_peaks=on_peaks)
        if ic is None:
            ic = target.ic
        if self.data.shape[2] > 1:
            maps = True
        else:
            maps = False
        for i in xrange(ic.shape[1]):
            axes = (self.axes_manager._slicing_axes[0].get_axis_dictionary(),)
            axes[0]['index_in_array'] = 0
            spectrum = Spectrum({'data' : ic[:,i], 'axes' : axes})
            spectrum.data_cube = ic[:,i].reshape((-1,1,1))

            if elements is None:
                spectrum.save('ic-%s.%s' % (i, spectrum_format))
                if maps is True:
                    pl[i].save('map_ic-%s.%s' % (i, hs_format))
                else:
                    pl[i].save('profile_ic-%s.%s' % (i, spectrum_format))
            else:
                element = elements[i]
                spectrum.save('ic-%s.%s' % (element, spectrum_format))
                if maps:
                    pl[i].save('map_ic-%s.%s' % (element, hs_format))
                else:
                    pl[i].save('profile_ic-%s.%s' % (element, spectrum_format))

#=============================================================================
        
    def cell_cropper(self):
        if not hasattr(self.mapped_parameters,"picker"):
            import hyperspy.drawing.ucc as ucc
            self.mapped_parameters.picker=ucc.TemplatePicker(self)
        self.mapped_parameters.picker.configure_traits()
        self.data=self.data.squeeze()
        return self.mapped_parameters.picker.crop_sig

    def kmeans_cluster_stack(self, clusters=None):
        import mdp
        if self._unfolded:
            self.fold()
        # if clusters not given, try to determine what it should be.
        if clusters is None:
            pass
        d=self.data
        kmeans=mdp.nodes.KMeansClassifier(clusters)
        cluster_arrays=[]

        avg_stack=np.zeros((clusters,d.shape[1],d.shape[2]))
        kmeans.train(d.reshape((-1,d.shape[0])).T)
        kmeans.stop_training()
        groups=kmeans.label(d.reshape((-1,d.shape[0])).T)
        try:
            # test if location data is available
            self.mapped_parameters.locations[0]
        except:
            messages.warning("No cell location information was available.")
        for i in xrange(clusters):
            # get number of members of this cluster
            members=groups.count(i)
            cluster_array=np.zeros((members,d.shape[1],d.shape[2]))
            cluster_idx=0
            positions=np.zeros((members,3))
            for j in xrange(len(groups)):
                if groups[j]==i:
                    cluster_array[cluster_idx,:,:]=d[j,:,:]
                    try:
                        positions[cluster_idx]=self.mapped_parameters.locations[j]
                    except:
                        pass
                    cluster_idx+=1
            cluster_array_Image=Image({'data':avg_stack,
                    'mapped_parameters':{
                        'name':'Cluster %s from %s'%(i,
                                         self.mapped_parameters.name),
                        'locations':positions,
                        'members':members,
                        }
                    })
            cluster_arrays.append(cluster_array_Image)
            avg_stack[i,:,:]=np.sum(cluster_array,axis=0)
        members_list=[groups.count(i) for i in xrange(clusters)]
        avg_stack_Image=Image({'data':avg_stack,
                    'mapped_parameters':{
                        'name':'Cluster averages from %s'%self.mapped_parameters.name,
                        'member_counts':members_list,
                        }
                    })
        return avg_stack_Image, cluster_arrays

    def peakfind_2D(self, subpixel=False, peak_width=10, medfilt_radius=5,
                        maxpeakn=30000):
            """Find peaks in a 2D array (peaks in an image).

            Function to locate the positive peaks in a noisy x-y data set.
    
            Returns an array containing pixel position of each peak.
            
            Parameters
            ---------
            subpixel : bool (optional)
                    default is set to True

            peak_width : int (optional)
                    expected peak width.  Affects subpixel precision fitting window,
                    which takes the center of gravity of a box that has sides equal
                    to this parameter.  Too big, and you'll include other peaks.
                    default is set to 10

            medfilt_radius : int (optional)
                     median filter window to apply to smooth the data
                     (see scipy.signal.medfilt)
                     if 0, no filter will be applied.
                     default is set to 5

            maxpeakn : int (optional)
                    number of maximum detectable peaks
                    default is set to 30000             
            """
            from peak_char import two_dim_findpeaks
            if len(self.data.shape)==2:
                self.peaks=two_dim_findpeaks(self.data, subpixel=subpixel,
                                             peak_width=peak_width, 
                                             medfilt_radius=medfilt_radius)
                
            elif len(self.data.shape)==3:
                # preallocate a large array for the results
                self.peaks=np.zeros((maxpeakn,2,self.data.shape[2]))
                for i in xrange(self.data.shape[2]):
                    tmp=two_dim_findpeaks(self.data[:,:,i], 
                                             subpixel=subpixel,
                                             peak_width=peak_width, 
                                             medfilt_radius=medfilt_radius)
                    self.peaks[:tmp.shape[0],:,i]=tmp
                trim_id=np.min(np.nonzero(np.sum(np.sum(self.peaks,axis=2),axis=1)==0))
                self.peaks=self.peaks[:trim_id,:,:]
            elif len(self.data.shape)==4:
                # preallocate a large array for the results
                self.peaks=np.zeros((maxpeakn,2,self.data.shape[0],self.data.shape[1]))
                for i in xrange(self.data.shape[0]):
                    for j in xrange(self.data.shape[1]):
                        tmp=two_dim_findpeaks(self.data[i,j,:,:], 
                                             subpixel=subpixel,
                                             peak_width=peak_width, 
                                             medfilt_radius=medfilt_radius)
                        self.peaks[:tmp.shape[0],:,i,j]=tmp
                trim_id=np.min(np.nonzero(np.sum(np.sum(np.sum(self.peaks,axis=3),axis=2),axis=1)==0))
                self.peaks=self.peaks[:trim_id,:,:,:]
                
    def to_spectrum(self):
        from hyperspy.signals.spectrum import Spectrum
        dic = self._get_signal_dict()
        dic['mapped_parameters']['record_by'] = 'spectrum'
        dic['data'] = np.swapaxes(dic['data'], 0, -1)
        utils_varia.swapelem(dic['axes'],0,-1)
        dic['axes'][0]['index_in_array'] = 0
        dic['axes'][-1]['index_in_array'] = len(dic['axes']) - 1
        return Spectrum(dic)
