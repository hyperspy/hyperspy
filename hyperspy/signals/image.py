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

from hyperspy.drawing import image as imgdraw
from hyperspy.drawing import signal as sigdraw

from matplotlib import pyplot as plt
import numpy as np
import os

class Image(Signal):
    """
    """    
    def __init__(self, *args, **kw):
        super(Image,self).__init__(*args, **kw)
        self.axes_manager.set_view('image')
        self.peak_mva_results=MVA_Results()

    def peak_char_stack(self, peak_width, subpixel=False, target_locations=None,
                        peak_locations=None, imcoords=None, target_neighborhood=20,
                        medfilt_radius=5):
        """
        Characterizes the peaks in the stack of images.  Creates a class member
        "mapped_parameters.peak_chars" that is a 2D array of the following form:
        - One column per image
        - 7 rows per peak located.  These rows are, in order:
            0-1: x,y coordinate of peak
            2-3: aberration of this peak from its target value
            4: height of the peak
            5: orientation of the peak
            6: eccentricity of the peak

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
                target locations by locating peaks on the average image of the 
                stack.
                default is None (peaks detected from average image)

        peak_locations : numpy array (n x m x 2) (optional)
                array of n peak locations for m images.  If left as None,
                will find all peaks on all images, and keep only the ones 
                closest to the peaks specified in target_locations.
                default is None (peaks detected from stack)

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
        smp=self.mapped_parameters
        if target_locations is None:
            target_locations=pc.two_dim_findpeaks(np.average(self.data,axis=0), 
                                               subpixel=subpixel,
                                               peak_width=peak_width)
        smp.target_locations=target_locations
        smp.peak_width=peak_width
        smp.peak_chars=pc.peak_attribs_stack(self.data, 
                                peak_width,
                                subpixel=subpixel, 
                                target_locations=target_locations,
                                peak_locations=peak_locations, 
                                imcoords=imcoords, 
                                target_neighborhood=target_neighborhood,
                                medfilt_radius=medfilt_radius
                                )

    def recon_peak_chars(factors,scores,comp_ids=None,target_locations=None):
        """INCOMPLETE!!

        Does the following with Factor images:
            - Reconstructs the data using all available components
            - locates peaks on all images in reconstructed data
            - reconstructs the data using all components EXCEPT the component specified
                by the plot_component parameter
            - locates peaks on all images in reconstructed data
            - subtracts the peak characteristics of the first (complete) data from the
                data without the component included.  This difference data is what gets
                plotted.
        """
        pass

    def _get_pk_shifts_and_char(self,f_pc,plot_shifts=False,plot_char=4):
        locations=self.mapped_parameters.target_locations

        if plot_shifts:
            shifts=np.zeros((locations.shape[0]),dtype=[('location','f4',(1,2)),
                                                        ('shift','f4',(1,2))])
        else: shifts=None
        if plot_char:
            char=np.zeros(locations.shape[0],dtype=[('location','f4',(1,2)),
                                                          ('char','f4')])
        else: char=None

        for pos in xrange(locations.shape[0]):
            if plot_shifts:
                shifts[pos]=(locations[pos,:2],
                             f_pc[pos*7+2:pos*7+4])
            if plot_char:
                char[pos]=(locations[pos,:2],
                           f_pc[pos*7+plot_char])

        return shifts, char

    #==============================
    # MVA plotting routines
    # These grab the data to be plotted, and pass it to functions
    # in drawing/signal.py and drawing/image.py
    #=============================

    def plotPca_factors(self, comp_ids=None, calibrate=True,
                        same_window=True, comp_label='PC', 
                        on_peaks=False, img_data=None,
                        plot_shifts=True, plot_char=None, 
                        cmap=plt.cm.jet, per_row=3):
        """Plot components from PCA, either factor images or
           peak characteristics.

        Parameters
        ----------

        comp_ids : None, int, or list of ints
            if None, returns maps of all components.
            if int, returns maps of components with ids from 0 to given int.
            if list of ints, returns maps of components with ids in given list.

        calibrate : bool
            if True, calibrates plots where calibration is available from
            the axes_manager.  If False, plots are in pixels/channels.

        same_window : bool
            if True, plots each factor to the same window.  They are not scaled.
        
        comp_label : string, the label that is either the plot title (if plotting in
            separate windows) or the label in the legend (if plotting in the 
            same window)

        on_peaks : bool
            Plot peak characteristics (True), or factor images (False)

        cmap : The colormap used for the factor image, or for peak 
            characteristics, the colormap used for the scatter plot of
            some peak characteristic.
        
        per_row : int, the number of plots in each row, when the same_window
            parameter is True.

        Peak characteristic-specific functions:
        ---------------------------------------

        img_data : 2D numpy array
            If on_peaks is specified, this is the image that gets overlaid
            with the peak information. If not specified, defaults to the
            average image of your image stack.

        plot_char - None or int
        (optional, but required to plot peak characteristic overlays)
            If int, the id of the characteristic to plot as the colored 
            scatter plot.
            Possible components are:
               0 or 1: peak coordinates
               2 or 3: position difference relative to nearest target location
               4: peak height
               5: peak orientation
               6: peak eccentricity

        plot_shift - bool, optional
            If True, plots shift overlays from the factor onto the image given in
            the cell_data parameter
        """
        factors=self._get_target(on_peaks).pc
        return self._plot_factors_or_pchars(factors, comp_ids=comp_ids, 
                                same_window=same_window, comp_label=comp_label, 
                                on_peaks=on_peaks, img_data=img_data,
                                plot_shifts=plot_shifts, plot_char=plot_char, 
                                cmap=cmap, per_row=per_row)

    def plotIca_factors(self,comp_ids=None, calibrate=True,
                        same_window=True, comp_label='IC', 
                        on_peaks=False, img_data=None,
                        plot_shifts=True, plot_char=None, 
                        cmap=plt.cm.jet, per_row=3):
        """Plot components from ICA, either factor images or
           peak characteristics.

        Parameters
        ----------

        comp_ids : None, int, or list of ints
            if None, returns maps of all components.
            if int, returns maps of components with ids from 0 to given int.
            if list of ints, returns maps of components with ids in given list.

        calibrate : bool
            if True, calibrates plots where calibration is available from
            the axes_manager.  If False, plots are in pixels/channels.

        same_window : bool
            if True, plots each factor to the same window.  They are not scaled.
        
        comp_label : string, the label that is either the plot title (if plotting in
            separate windows) or the label in the legend (if plotting in the 
            same window)

        on_peaks : bool
            Plot peak characteristics (True), or factor images (False)

        cmap : The colormap used for the factor image, or for peak 
            characteristics, the colormap used for the scatter plot of
            some peak characteristic.
        
        per_row : int, the number of plots in each row, when the same_window
            parameter is True.

        Peak characteristic-specific functions:
        ---------------------------------------

        img_data : 2D numpy array
            If on_peaks is specified, this is the image that gets overlaid
            with the peak information. If not specified, defaults to the
            average image of your image stack.

        plot_char - None or int
        (optional, but required to plot peak characteristic overlays)
            If int, the id of the characteristic to plot as the colored 
            scatter plot.
            Possible components are:
               0 or 1: peak coordinates
               2 or 3: position difference relative to nearest target location
               4: peak height
               5: peak orientation
               6: peak eccentricity

        plot_shift - bool, optional
            If True, plots shift overlays from the factor onto the image given in
            the cell_data parameter
        """
        factors=self._get_target(on_peaks).ic
        return self._plot_factors_or_pchars(factors, comp_ids=comp_ids, 
                                same_window=same_window, comp_label=comp_label, 
                                on_peaks=on_peaks, img_data=img_data,
                                plot_shifts=plot_shifts, plot_char=plot_char, 
                                cmap=cmap, per_row=per_row)

    def plotPca_scores(self, comp_ids=None, calibrate=True,
                       same_window=True, comp_label='PC', 
                       with_factors=False,
                       on_peaks=False, cmap=plt.cm.jet, 
                       no_nans=True,per_row=3):
        """Plot scores from PCA, either factor images or
           peak characteristics.

        Parameters
        ----------

        comp_ids : None, int, or list of ints
            if None, returns maps of all components.
            if int, returns maps of components with ids from 0 to given int.
            if list of ints, returns maps of components with ids in given list.

        calibrate : bool
            if True, calibrates plots where calibration is available from
            the axes_manager.  If False, plots are in pixels/channels.

        same_window : bool
            if True, plots each factor to the same window.  They are not scaled.
        
        comp_label : string, 
            The label that is either the plot title (if plotting in
            separate windows) or the label in the legend (if plotting in the 
            same window)

        with_factors : bool
            If True, also returns figure(s) with the factors for the
            given comp_ids.

        on_peaks : bool
            Plot scores from peak characteristics (True), 
            or factor images (False)

        cmap : matplotlib colormap
            The colormap used for the factor image, or for peak 
            characteristics, the colormap used for the scatter plot of
            some peak characteristic.
        
        no_nans : bool
            If True, removes NaN's from the score plots.

        per_row : int 
            the number of plots in each row, when the same_window
            parameter is True.
        """
        scores=self._get_target(on_peaks).v.T
        if with_factors:
            factors=self._get_target(on_peaks).pc
        else: factors=None
        return self._plot_scores(scores, comp_ids=comp_ids,
                                 with_factors=with_factors, factors=factors,
                                 same_window=same_window, comp_label=comp_label,
                                 on_peaks=on_peaks, cmap=cmap,
                                 no_nans=no_nans,per_row=per_row)

    def plotIca_scores(self, comp_ids=None, calibrate=True,
                       same_window=True, comp_label='IC', 
                       with_factors=False,
                       on_peaks=False, cmap=plt.cm.jet, 
                       no_nans=True,per_row=3):
        """Plot scores from ICA, either factor images or
           peak characteristics.

        Parameters
        ----------

        comp_ids : None, int, or list of ints
            if None, returns maps of all components.
            if int, returns maps of components with ids from 0 to given int.
            if list of ints, returns maps of components with ids in given list.

        calibrate : bool
            if True, calibrates plots where calibration is available from
            the axes_manager.  If False, plots are in pixels/channels.

        same_window : bool
            if True, plots each factor to the same window.  They are not scaled.
        
        comp_label : string, 
            The label that is either the plot title (if plotting in
            separate windows) or the label in the legend (if plotting in the 
            same window)

        with_factors : bool
            If True, also returns figure(s) with the factors for the
            given comp_ids.

        on_peaks : bool
            Plot scores from peak characteristics (True), 
            or factor images (False)

        cmap : matplotlib colormap
            The colormap used for the factor image, or for peak 
            characteristics, the colormap used for the scatter plot of
            some peak characteristic.
        
        no_nans : bool
            If True, removes NaN's from the score plots.

        per_row : int 
            the number of plots in each row, when the same_window
            parameter is True.
        """
        scores=self._get_ica_scores(self._get_target(on_peaks))
        if with_factors:
            factors=self.get_target(on_peaks).ic
        else: factors=None
        return self._plot_scores(scores, comp_ids=comp_ids,
                                 with_factors=with_factors, factors=factors,
                                 same_window=same_window, comp_label=comp_label,
                                 on_peaks=on_peaks, cmap=cmap,
                                 no_nans=no_nans,per_row=per_row)

    def exportPca_results(self, comp_ids=None, 
                          factor_prefix='pc', factor_format='rpl',
                          score_prefix='PC_score', score_format='rpl', 
                          on_peaks=False,
                          calibrate=True, same_window=False,
                          comp_label='PC',cmap=plt.cm.jet,
                          no_nans=True,per_row=3,
                          plot_shifts=True, 
                          plot_char=None, img_data=None):
        """Export results from PCA to any of the supported formats.

        Parameters
        ----------

        comp_ids : None, int, or list of ints
            if None, returns all components/scores.
            if int, returns components/scores with ids from 0 to given int.
            if list of ints, returns components/scores with ids in given list.

        factor_prefix : string
            The prefix that any exported filenames for factors/components 
            begin with

        factor_format : string
            The extension of the format that you wish to save to.  Determines
            the kind of output.
                - For image formats (tif, png, jpg, etc.), plots are created 
                  using the plotting flags as below, and saved at 600 dpi.
                  One plot per factor is saved.
                - For multidimensional formats (rpl, hdf5), arrays are saved
                  in single files.  All factors are contained in the one
                  file.
                - For spectral formats (msa), each factor is saved to a
                  separate file.
                
        score_prefix : string
            The prefix that any exported filenames for factors/components 
            begin with

        score_format : string
            The extension of the format that you wish to save to.  Determines
            the kind of output.
                - For image formats (tif, png, jpg, etc.), plots are created 
                  using the plotting flags as below, and saved at 600 dpi.
                  One plot per score is saved.
                - For multidimensional formats (rpl, hdf5), arrays are saved
                  in single files.  All scores are contained in the one
                  file.
                - For spectral formats (msa), each score is saved to a
                  separate file.

        on_peaks : bool
            Export peak characteristics (True), or image data (False)

        Plotting options (for image file formats ONLY)
        ----------------------------------------------

        calibrate : bool
            if True, calibrates plots where calibration is available from
            the axes_manager.  If False, plots are in pixels/channels.

        same_window : bool
            if True, plots each factor to the same window.
        
        comp_label : string, the label that is either the plot title (if plotting in
            separate windows) or the label in the legend (if plotting in the 
            same window)

        cmap : The colormap used for the factor image, or for peak 
            characteristics, the colormap used for the scatter plot of
            some peak characteristic.
        
        per_row : int, the number of plots in each row, when the same_window
            parameter is True.

        Peak characteristic-specific plotting functions:
        ---------------------------------------

        img_data : 2D numpy array
            If on_peaks is specified, this is the image that gets overlaid
            with the peak information. If not specified, defaults to the
            average image of your image stack.

        plot_char - None or int
        (optional, but required to plot peak characteristic overlays)
            If int, the id of the characteristic to plot as the colored 
            scatter plot.
            Possible components are:
               0 or 1: peak coordinates
               2 or 3: position difference relative to nearest target location
               4: peak height
               5: peak orientation
               6: peak eccentricity

        plot_shift - bool, optional
            If True, plots shift overlays from the factor onto the image given in
            the cell_data parameter
        """
        factors=self._get_target(on_peaks).pc
        scores=self._get_target(on_peaks).v.T
        self._export_factors(factors, comp_ids=comp_ids,
                             calibrate=calibrate,
                             plot_shifts=plot_shifts,
                             plot_char=plot_char,
                             img_data=img_data,
                             factor_prefix=factor_prefix,
                             factor_format=factor_format,
                             comp_label=comp_label,
                             on_peaks=on_peaks,
                             cmap=cmap,
                             no_nans=no_nans,
                             same_window=same_window,
                             per_row=per_row)
        self._export_scores(scores,comp_ids=comp_ids,
                            calibrate=calibrate,
                            score_prefix=score_prefix,
                            score_format=score_format,
                            comp_label=comp_label,
                            cmap=cmap,
                            same_window=same_window,
                            no_nans=no_nans,
                            per_row=per_row)

    def exportIca_results(self, comp_ids=None, 
                          factor_prefix='ic', factor_format='rpl',
                          score_prefix='IC_score', score_format='rpl', 
                          on_peaks=False,
                          calibrate=True, same_window=False,
                          comp_label='IC',cmap=plt.cm.jet,
                          no_nans=True,per_row=3,
                          plot_shifts=True, 
                          plot_char=None, img_data=None):
        """Export results from ICA to any of the supported formats.

        Parameters
        ----------

        comp_ids : None, int, or list of ints
            if None, returns all components/scores.
            if int, returns components/scores with ids from 0 to given int.
            if list of ints, returns components/scores with ids in given list.

        factor_prefix : string
            The prefix that any exported filenames for factors/components 
            begin with

        factor_format : string
            The extension of the format that you wish to save to.  Determines
            the kind of output.
                - For image formats (tif, png, jpg, etc.), plots are created 
                  using the plotting flags as below, and saved at 600 dpi.
                  One plot per factor is saved.
                - For multidimensional formats (rpl, hdf5), arrays are saved
                  in single files.  All factors are contained in the one
                  file.
                - For spectral formats (msa), each factor is saved to a
                  separate file.
                
        score_prefix : string
            The prefix that any exported filenames for factors/components 
            begin with

        score_format : string
            The extension of the format that you wish to save to.  Determines
            the kind of output.
                - For image formats (tif, png, jpg, etc.), plots are created 
                  using the plotting flags as below, and saved at 600 dpi.
                  One plot per score is saved.
                - For multidimensional formats (rpl, hdf5), arrays are saved
                  in single files.  All scores are contained in the one
                  file.
                - For spectral formats (msa), each score is saved to a
                  separate file.

        on_peaks : bool
            Export peak characteristics (True), or image data (False)

        Plotting options (for image file formats ONLY)
        ----------------------------------------------

        calibrate : bool
            if True, calibrates plots where calibration is available from
            the axes_manager.  If False, plots are in pixels/channels.

        same_window : bool
            if True, plots each factor to the same window.
        
        comp_label : string, the label that is either the plot title (if plotting in
            separate windows) or the label in the legend (if plotting in the 
            same window)

        cmap : The colormap used for the factor image, or for peak 
            characteristics, the colormap used for the scatter plot of
            some peak characteristic.
        
        per_row : int, the number of plots in each row, when the same_window
            parameter is True.

        Peak characteristic-specific plotting functions:
        ---------------------------------------

        img_data : 2D numpy array
            If on_peaks is specified, this is the image that gets overlaid
            with the peak information. If not specified, defaults to the
            average image of your image stack.

        plot_char - None or int
        (optional, but required to plot peak characteristic overlays)
            If int, the id of the characteristic to plot as the colored 
            scatter plot.
            Possible components are:
               0 or 1: peak coordinates
               2 or 3: position difference relative to nearest target location
               4: peak height
               5: peak orientation
               6: peak eccentricity

        plot_shift - bool, optional
            If True, plots shift overlays from the factor onto the image given in
            the cell_data parameter
        """
        factors=self._get_target(on_peaks).ic
        scores=self._get_ica_scores(self._get_target(on_peaks))
        self._export_factors(factors, comp_ids=comp_ids,
                             calibrate=calibrate,
                             plot_shifts=plot_shifts,
                             plot_char=plot_char,
                             img_data=img_data,
                             factor_prefix=factor_prefix,
                             factor_format=factor_format,
                             comp_label=comp_label,
                             on_peaks=on_peaks,
                             cmap=cmap,
                             no_nans=no_nans,
                             same_window=same_window,
                             per_row=per_row)
        self._export_scores(scores,comp_ids=comp_ids,
                            calibrate=calibrate,
                            score_prefix=score_prefix,
                            score_format=score_format,
                            comp_label=comp_label,
                            cmap=cmap,
                            same_window=same_window,
                            no_nans=no_nans,
                            per_row=per_row)

    def plot_image_peaks(self,img_data=None, locations=None, peak_width=10, 
                         subpixel=False, medfilt_radius=5):
        """Overlay an image with a scatter map of peak locations.

        Parameters:
        -----------
        img_data : None or 2D numpy array
            If None, uses the average image (for stacks), or just
            the image (for 2D images)
        locations : None or nx2 numpy array
            If None, function locates peaks on img_data using parameters
            below.
        plot_ids : bool
            If True, plots the peak id instead of a simple scatter map.
            Use this function to identify a peak to overlay a characteristic of
            onto the original experimental image using the plot_image_overlay
            function.

        Peak locator parameters:
        ------------------------
        peak_width : int
            The width of peaks to be found
         
        subpixel : bool
            If True, finds the subpixel center of mass of peaks
            If False, finds the index at which the peak max occurs.

        medfilt_radius : int or None
            If int, the radius of a median filter applied to the image
            prior to peak finding.  NOTE: must be an odd number.
            If None, no median filter is applied.
        """
        if img_data==None:
            if len(self.data.shape)>2:
                img_data=np.average(self.data,axis=0)
            else:
                img_data=self.data
            if locations==None and hasattr(self.mapped_parameters,
                                           'target_locations'):
                locations=self.mapped_parameters.target_locations
        if locations==None:
            locations=pc.two_dim_peakfind(img_data, subpixel=subpixel,
                                  peak_width=peak_width, 
                                  medfilt_radius=medfilt_radius)
        return imgdraw.plot_image_peaks(img_data, locations, plot_ids)

    def plot_image_overlay(plot_component=None, mva_type='PCA', 
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
        smp=self.mapped_parameters
        if not hasattr(smp, "original_files"):
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
        for key in smp.original_files.keys():
            f=plt.figure()
            plt.title(key)
            plt.imshow(smp.original_files[key].data, 
                interpolation = 'nearest')
            plt.gray()
            # get a shorter handle on the peak locations on THIS image
            locs=smp.locations
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
                char=np.array([char.append(smp.peak_chars[peak_id*7+plot_char,
                               mask.nonzero()[x]]) for x in xrange(
                            locs.shape[0])])
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
                shifts=np.array([char.append(smp.peak_chars[peak_id*7+2:peak_id*7+4,
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
                plt.scatter(locs[:,0],locs[:,1],c=scores)
                plt.jet()
                plt.colorbar()
            figs.append(f)
        return figs

    def plot_peak_cell_overlay(self, img_ids=None, calibrate=True,
                        same_window=True, comp_label='Peak Char - img', 
                        plot_shifts=True, plot_char=None, quiver_color='white',
                        vector_scale=100,
                        cmap=plt.cm.jet, per_row=3):
        """Overlays peak characteristics on an image plot of the average image.

        Parameters
        ----------

        img_ids : None, int, or list of ints
            if None, returns maps of average image.
            if int, returns maps of characteristics with ids from 0 to given int.
            if list of ints, returns maps of characteristics with ids in given list.

        calibrate : bool
            if True, calibrates plots where calibration is available from
            the axes_manager.  If False, plots are in pixels/channels.

        same_window : bool
            if True, plots each factor to the same window.  They are not scaled.
        
        comp_label : string, the label that is either the plot title (if plotting in
            separate windows) or the label in the legend (if plotting in the 
            same window)

        cmap : The colormap used for the factor image, or for peak 
            characteristics, the colormap used for the scatter plot of
            some peak characteristic.
        
        per_row : int, the number of plots in each row, when the same_window
            parameter is True.

        plot_char - None or int
        (optional, but required to plot peak characteristic overlays)
            If int, the id of the characteristic to plot as the colored 
            scatter plot.
            Possible components are:
               0 or 1: peak coordinates
               2 or 3: position difference relative to nearest target location
               4: peak height
               5: peak orientation
               6: peak eccentricity

        plot_shift - bool, optional
            If True, plots shift overlays from the factor onto the image given in
            the cell_data parameter
        """
        smp=self.mapped_parameters
        factors=smp.peak_chars
        #slice the images based on the img_ids
        if img_ids is not None:
            if not hasattr(img_ids,'__iter__'):
                img_ids=range(img_ids)
            mask=np.zeros(self.data.shape[0],dtype=np.bool)
            for idx in img_ids:
                mask[idx]=1
            img_data=self.data[mask]
            avg_char=False
        else:
            img_data=None
            avg_char=True
        return self._plot_factors_or_pchars(factors=factors, comp_ids=img_ids,
                                     calibrate=calibrate, comp_label=comp_label,
                                     img_data=img_data,avg_char=avg_char,
                                     on_peaks=True, plot_char=plot_char,
                                     plot_shifts=plot_shifts, cmap=cmap,
                                     quiver_color=quiver_color,
                                     vector_scale=vector_scale,
                                     per_row=per_row)

    def plotIca_peak_factors(self, comp_ids=None, calibrate=True,
                        same_window=True, comp_label='IC', 
                        img_data=None,
                        plot_shifts=True, plot_char=None, 
                        cmap=plt.cm.jet, per_row=3):
        """Overlays peak characteristics on an image plot of the average image.

        Parameters
        ----------

        comp_ids : None, int, or list of ints
            if None, returns maps of all components.
            if int, returns maps of components with ids from 0 to given int.
            if list of ints, returns maps of components with ids in given list.

        calibrate : bool
            if True, calibrates plots where calibration is available from
            the axes_manager.  If False, plots are in pixels/channels.

        same_window : bool
            if True, plots each factor to the same window.  They are not scaled.
        
        comp_label : string, the label that is either the plot title (if plotting in
            separate windows) or the label in the legend (if plotting in the 
            same window)

        on_peaks : bool
            Plot peak characteristics (True), or factor images (False)

        cmap : The colormap used for the factor image, or for peak 
            characteristics, the colormap used for the scatter plot of
            some peak characteristic.
        
        per_row : int, the number of plots in each row, when the same_window
            parameter is True.

        img_data : 2D numpy array
            This is the image that gets overlaid with the peak information. If 
            not specified, defaults to the average image of your image stack.

        plot_char - None or int
        (optional, but required to plot peak characteristic overlays)
            If int, the id of the characteristic to plot as the colored 
            scatter plot.
            Possible components are:
               0 or 1: peak coordinates
               2 or 3: position difference relative to nearest target location
               4: peak height
               5: peak orientation
               6: peak eccentricity

        plot_shift - bool, optional
            If True, plots shift overlays from the factor onto the image given in
            the cell_data parameter
        """
        factors=self.peak_mva_results.ic
        return self._plot_factors_or_pchars(factors=factors, comp_ids=comp_ids,
                                     calibrate=calibrate, comp_label=comp_label,
                                     img_data=img_data,
                                     on_peaks=True, plot_char=plot_char,
                                     plot_shifts=plot_shifts, cmap=cmap,
                                     quiver_color=quiver_color,
                                     vector_scale=vector_scale,
                                     per_row=per_row)

    def plotPca_peak_factors(self, comp_ids=None, calibrate=True,
                        same_window=True, comp_label='PC', 
                        img_data=None,
                        plot_shifts=True, plot_char=None, 
                        quiver_color='white',vector_scale=100,
                        cmap=plt.cm.jet, per_row=3):
        """Overlays peak characteristics on an image plot of the average image.

        Parameters
        ----------

        comp_ids : None, int, or list of ints
            if None, returns maps of all components.
            if int, returns maps of components with ids from 0 to given int.
            if list of ints, returns maps of components with ids in given list.

        calibrate : bool
            if True, calibrates plots where calibration is available from
            the axes_manager.  If False, plots are in pixels/channels.

        same_window : bool
            if True, plots each factor to the same window.  They are not scaled.
        
        comp_label : string, the label that is either the plot title (if plotting in
            separate windows) or the label in the legend (if plotting in the 
            same window)

        on_peaks : bool
            Plot peak characteristics (True), or factor images (False)

        cmap : The colormap used for the factor image, or for peak 
            characteristics, the colormap used for the scatter plot of
            some peak characteristic.
        
        per_row : int, the number of plots in each row, when the same_window
            parameter is True.

        img_data : 2D numpy array
            This is the image that gets overlaid with the peak information. If 
            not specified, defaults to the average image of your image stack.

        plot_char - None or int
        (optional, but required to plot peak characteristic overlays)
            If int, the id of the characteristic to plot as the colored 
            scatter plot.
            Possible components are:
               0 or 1: peak coordinates
               2 or 3: position difference relative to nearest target location
               4: peak height
               5: peak orientation
               6: peak eccentricity

        plot_shift - bool, optional
            If True, plots shift overlays from the factor onto the image given in
            the cell_data parameter
        """
        factors=self.peak_mva_results.pc
        return self._plot_factors_or_pchars(factors=factors, comp_ids=comp_ids,
                                     calibrate=calibrate, comp_label=comp_label,
                                     img_data=img_data,
                                     on_peaks=True, plot_char=plot_char,
                                     plot_shifts=plot_shifts, cmap=cmap,
                                     quiver_color=quiver_color,
                                     vector_scale=vector_scale,
                                     per_row=per_row)

    def plot_maps(self, components, mva_type=None, scores=None, factors=None,
                  cmap=plt.cm.gray, no_nans=False, per_row=3, on_peaks=False, 
                  scoremap=True, save_figs=False, directory = None):
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
        
        per_row : int (optional)
            The number of plots per row in the multi-pane window.

        on_peaks : bool (optional)
            If True, plots factors/score maps based on peak characteristics.
               You must have first run peak_char_stack to obtain peak characteristics,
               then run your MVA technique(s) with the on_peaks flag set to True in
               order to obtain this information.

        scoremap : bool (optional)
            If True, plots scores of subimages overlaid as a scatter plot
            on the original images.  Not possible unless the cell cropper
            has been used to obtain your stack of subimages.

        save_figs : bool (optional)
            If true, saves figures at 600 dpi to directory.  If directory is None,
            saves to current working directory.

        directory : string or None
            The folder to save images to, if save_figs is True.  If None, saves
            to current working directory.
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
                if hasattr(self.mapped_parameters,'locations'):
                    locs=self.mapped_parameters.locations
                    if hasattr(self.mapped_parameters,"original_files"):
                        parents=self.mapped_parameters.original_files
                    elif hasattr(self.mapped_parameters,'parent'):
                        parents={self.mapped_parameters.parent.mapped_parameters.title:self.mapped_parameters.parent}
                else:
                    scoremap=False
                    parents=None
                    locs=None
                # plot factor image first
                if scoremap:
                    idx=0
                    keys=parents.keys()
                    rows=int(np.ceil((len(keys)+1)/float(per_row)))
                    if (len(keys)+1)<per_row:
                        per_row=len(keys)+1
                    figure.add_subplot(rows,per_row,1)
                else:
                    figure.add_subplot(121)
                plt.gray()
                if mva_type.upper()=='PCA':
                    _plot_pc(i,on_peaks,cmap=cmap)
                elif mva_type.upper()=='ICA':
                    _plot_ic(i,on_peaks,cmap=cmap)
                if scoremap:
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
                                plt.imshow(parents[keys[idx]].data, 
                                    interpolation = 'nearest')
                                plt.gray()
                                sc=ax.scatter(loc[:,0], loc[:,1],
                                        c=scores[i].squeeze()[mask],
                                        cmap=cmap)
                                shp=parents[keys[idx]].data.shape
                                plt.xlim(0,shp[1])
                                plt.ylim(shp[0],0)
                                div=make_axes_locatable(ax)
                                cax=div.append_axes("right",size="5%",pad=0.05)
                                plt.colorbar(sc,cax=cax)
                            idx+=1
                else:
                    ax=figure.add_subplot(122)
                    plt.plot(np.arange(scores[i].shape[0]),scores[i],'bo')
                    plt.xlabel('Image index')
                    plt.ylabel('Score, component %i'%i)
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


    def plot_principal_components_maps(comp_ids=None, cmap=plt.cm.gray,
                                       recmatrix=None, plot=True, pc=None, on_peaks=False,
                                       per_row=3, scoremap=True, save_figs=False,
                                       directory=None):
        """Plot the map associated to each independent component

        Parameters
        ----------
        comp_ids : None, int, or list of ints
            if None, returns maps of all components.
            if int, returns maps of components with ids from 0 to given int.
            if list of ints, returns maps of components with ids in given list.

        cmap : plt.cm object, the colormap of the factor image

        recmatrix : numpy array
            externally suplied recmatrix

        plot : bool
            If True it will plot the figures. Otherwise it will only return the
            images.

        pc : numpy array
            externally supplied principal components

        no_nans : bool (optional)
             whether substituting NaNs with zeros for a visually prettier plot
             (default is False)

        per_row : int (optional)
            The number of plots per row in the multi-pane window.

        on_peaks : bool (optional)
            If True, plots factors/score maps based on peak characteristics.
               You must have first run peak_char_stack to obtain peak characteristics,
               then run your MVA technique(s) with the on_peaks flag set to True in
               order to obtain this information.

        scoremap : bool (optional)
            If True, plots scores of subimages overlaid as a scatter plot
            on the original images.  Not possible unless the cell cropper
            has been used to obtain your stack of subimages.

        save_figs : bool (optional)
            If true, saves figures at 600 dpi to directory.  If directory is None,
            saves to current working directory.

        directory : string or None
            The folder to save images to, if save_figs is True.  If None, saves
            to current working directory.

        Returns
        -------
        List with the maps as MVA instances
        """
        return plot_maps(components=comp_ids,mva_type='pca',cmap=cmap,
                              scores=recmatrix, factors=pc, on_peaks=on_peaks,
                              scoremap=scoremap,save_figs=save_figs,
                              per_row=per_row, directory=directory)

    def plot_independent_components_maps(comp_ids=None, cmap=plt.cm.gray,
                                         recmatrix=None, ic=None, no_nans=False,
                                         on_peaks=False, scoremap=True, per_row=3,
                                         save_figs=False, directory = None):
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

        per_row : int (optional)
            The number of plots per row in the multi-pane window.

        on_peaks : bool (optional)
            If True, plots factors/score maps based on peak characteristics.
               You must have first run peak_char_stack to obtain peak characteristics,
               then run your MVA technique(s) with the on_peaks flag set to True in
               order to obtain this information.

        scoremap : bool (optional)
            If True, plots scores of subimages overlaid as a scatter plot
            on the original images.  Not possible unless the cell cropper
            has been used to obtain your stack of subimages.

        Returns
        -------
        List with the maps as MVA instances
        """
        return imgdraw.plot_maps(comp_ids=comp_ids,mva_type='ica',cmap=cmap,
                              scores=recmatrix, factors=ic, no_nans=no_nans,
                              on_peaks=on_peaks, scoremap=scoremap, per_row=per_row,
                              save_figs=save_figs, directory = directory)


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
            cluster_array_Image=Image({
                'data':avg_stack,
                'mapped_parameters':{
                    'title' : 'Cluster %s from %s'%(i,
                        self.mapped_parameters.title),
                    'locations':positions,
                    'members':members,}
            })
            cluster_arrays.append(cluster_array_Image)
            avg_stack[i,:,:]=np.sum(cluster_array,axis=0)
        members_list=[groups.count(i) for i in xrange(clusters)]
        avg_stack_Image=Image({'data':avg_stack,
                    'mapped_parameters':{
                        'title':'Cluster averages from %s'%self.mapped_parameters.title,
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
            smp=self.mapped_parameters
            if len(self.data.shape)==2:
                smp.peaks=two_dim_findpeaks(self.data, subpixel=subpixel,
                                             peak_width=peak_width, 
                                             medfilt_radius=medfilt_radius)
                
            elif len(self.data.shape)==3:
                # preallocate a large array for the results
                smp.peaks=np.zeros((maxpeakn,2,self.data.shape[2]))
                for i in xrange(self.data.shape[2]):
                    tmp=two_dim_findpeaks(self.data[:,:,i], 
                                             subpixel=subpixel,
                                             peak_width=peak_width, 
                                             medfilt_radius=medfilt_radius)
                    smp.peaks[:tmp.shape[0],:,i]=tmp
                trim_id=np.min(np.nonzero(np.sum(np.sum(self.peaks,axis=2),axis=1)==0))
                smp.peaks=smp.peaks[:trim_id,:,:]
            elif len(self.data.shape)==4:
                # preallocate a large array for the results
                smp.peaks=np.zeros((maxpeakn,2,self.data.shape[0],self.data.shape[1]))
                for i in xrange(self.data.shape[0]):
                    for j in xrange(self.data.shape[1]):
                        tmp=two_dim_findpeaks(self.data[i,j,:,:], 
                                             subpixel=subpixel,
                                             peak_width=peak_width, 
                                             medfilt_radius=medfilt_radius)
                        smp.peaks[:tmp.shape[0],:,i,j]=tmp
                trim_id=np.min(np.nonzero(np.sum(np.sum(np.sum(self.peaks,axis=3),axis=2),axis=1)==0))
                smp.peaks=smp.peaks[:trim_id,:,:,:]
                
    def to_spectrum(self):
        from hyperspy.signals.spectrum import Spectrum
        dic = self._get_signal_dict()
        dic['mapped_parameters']['record_by'] = 'spectrum'
        dic['data'] = np.swapaxes(dic['data'], 0, -1)
        utils_varia.swapelem(dic['axes'],0,-1)
        dic['axes'][0]['index_in_array'] = 0
        dic['axes'][-1]['index_in_array'] = len(dic['axes']) - 1
        return Spectrum(dic)
