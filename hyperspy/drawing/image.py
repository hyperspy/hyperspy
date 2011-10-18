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

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets
from mpl_toolkits.axes_grid1 import make_axes_locatable

import widgets
import utils

class ImagePlot:
    def __init__(self):
        self.data_function = None
        self.pixel_size = None
        self.pixel_units = None
        self.plot_scale_bar = True
        self.figure = None
        self.ax = None
        self.title = ''
        self.window_title = ''
        self.vmin = None
        self.vmax = None
        self.auto_contrast = True
        
    def optimize_contrast(self, data, perc = 0.01):
        dc = data[np.isnan(data) == False]
        try:
            # check if it's an RGB structured array
            dc = dc['R']
        except ValueError, msg:
            if 'field named R not found.' in msg:
                pass
            else:
                raise
        if 'complex' in dc.dtype.name:
            dc = np.log(np.abs(dc))
        i = int(round(len(dc)*perc/100.))
        i = i if i > 0 else 1
        vmin = np.min(dc)
        vmax = np.max(dc)
        self.vmin = vmin
        self.vmax = vmax
        
    def create_figure(self):
        self.figure = utils.create_figure()
        
    def create_axis(self):
        self.ax = self.figure.add_subplot(111)
        self.ax.set_axis_off()
        self.figure.subplots_adjust(0,0,1,1)
        
    def plot(self):
        if not utils.does_figure_object_exists(self.figure):
            self.create_figure()
            self.create_axis()     
        data = self.data_function()
        if self.auto_contrast is True:
            self.optimize_contrast(data)
        self.update_image()
        if self.plot_scale_bar is True:
            if self.pixel_size is not None:
                self.ax.scale_bar = widgets.Scale_Bar(
                 ax = self.ax, units = self.pixel_units, 
                 pixel_size = self.pixel_size)
        
        # Adjust the size of the window
        #size = [ 6,  6.* data.shape[0] / data.shape[1]]
        #self.figure.set_size_inches(size, forward = True)        
        self.figure.canvas.draw()
        self.connect()
        
    def update_image(self):
        ims = self.ax.images
        if ims:
            ims.remove(ims[0])
        data = self.data_function()
        if self.auto_contrast is True:
            self.optimize_contrast(data)
        if 'complex' in data.dtype.name:
            data = np.log(np.abs(data))
        try:
            # check if it's an RGB structured array
            data_r = data['R']
            data_g = data['G']
            data_b = data['B']
            # modify the data so that it can be read by matplotlib
            data = np.rollaxis(np.array((data_r, data_g, data_b)), 0, 3)
        except ValueError, msg:
            if 'field named R not found.' in msg:
                pass
            else:
                raise
        self.ax.imshow(data, interpolation='nearest', vmin = self.vmin, 
                       vmax = self.vmax)
        self.figure.canvas.draw()
        
    def connect(self):
        self.figure.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.figure.canvas.draw()

    def on_key_press(self, event):
        if event.key == 'h':
            self.plot_histogram()
        
            
    def histogram_key_press(self, event):
        if event.key == 'a':
            self.optimize_contrast(self.data_function())
            self.set_contrast(self.vmin, self.vmax)
            
    def set_contrast(self, vmin, vmax):
        self.vmin, self.vmax =  vmin, vmax
        del(self.histogram_span_selector)
        plt.close(self.histogram_figure)
        del(self.histogram_figure)
        self.plot_histogram()
        self.update_image()
        
    def plot_histogram(self):
        f = plt.figure()
        ax = f.add_subplot(111)
        data = self.data_function().ravel()
        ax.hist(data,100, range = (self.vmin, self.vmax))
        self.histogram_span_selector = matplotlib.widgets.SpanSelector(
        ax, self.set_contrast, direction = 'horizontal')
        self.histogram_figure = f
        self.histogram_figure.canvas.mpl_connect(
        'key_press_event', self.histogram_key_press)
    
    # TODO The next function must be improved
    
    def optimize_colorbar(self, number_of_ticks = 5, tolerance = 5, step_prec_max = 1):
        vmin, vmax = self.vmin, self.vmax
        _range = vmax - vmin
        step = _range / (number_of_ticks - 1)
        step_oom = utils.order_of_magnitude(step)
        def optimize_for_oom(oom):
            self.colorbar_step = math.floor(step / 10**oom)*10**oom
            self.colorbar_vmin = math.floor(vmin / 10**oom)*10**oom
            self.colorbar_vmax = self.colorbar_vmin + \
            self.colorbar_step * (number_of_ticks - 1)
            self.colorbar_locs = np.arange(0,number_of_ticks)* self.colorbar_step \
            + self.colorbar_vmin
        def check_tolerance():
            if abs(self.colorbar_vmax - vmax) / vmax > (tolerance / 100.) or \
            abs(self.colorbar_vmin - vmin) >  (tolerance / 100.):
                return True
            else:
                return False
                    
        optimize_for_oom(step_oom)
        i = 1
        while check_tolerance() and i <= step_prec_max:
            optimize_for_oom(step_oom - i)
            i += 1
            
    def close(self):
        if utils.does_figure_object_exists(self.figure) is True:
            plt.close(self.figure)

#==============================================================================
# Plotting methods from MVA
#==============================================================================

def _plot_component(f_pc, idx, cell_data=None, locations=None, 
                    ax=None, shape=None, cal_axis=None, 
                    on_peaks=True, plot_shifts=True, 
                    plot_char=None, comp_label='PC',
                    cmap=plt.cm.jet):
    if ax==None:
        ax=plt.gca()
    plt.title('%s %s' % (comp_label,idx))
    if on_peaks:
        return _plot_cell_overlay(cell_data=cell_data, f_pc=f_pc, 
                                  locations=locations, ax=ax, 
                                  comp_id=comp_id, on_peaks=on_peaks,
                                  plot_shifts=plot_shifts, plot_char=plot_char, 
                                  cmap=plt.cm.jet)
    else:
        im=ax.imshow(f_pc[:,idx].reshape(shape[0], 
                                       shape[1]),
                     cmap=cmap, interpolation = 'nearest')
        div=make_axes_locatable(ax)
        cax=div.append_axes("right",size="5%",pad=0.05)
        plt.colorbar(im,cax=cax)
        return ax

def plot_image_peaks(cell_data, peaks=None, index=0, peak_width=10, subpixel=False,
                     medfilt_radius=5):
    # TODO: replace with hyperimage explorer
    plt.imshow(cell_data[index,:,:],cmap=plt.gray(), 
               interpolation = 'nearest')
    peaks=pc.two_dim_peakfind(cell_data[index,:,:], subpixel=subpixel,
                              peak_width=peak_width, 
                              medfilt_radius=medfilt_radius)
    plt.scatter(peaks[:,0],peaks[:,1])

def plot_peak_ids(self, cell_data, target_locations=None, peak_width=10):
    """Overlays id numbers for identified peaks on an average image of the
    stack.  Identified peaks are either those you specified as 
    target_locations, or if not specified, those automatically 
    identified from the average image.

    Use this function to identify a peak to overlay a characteristic of
    onto the original experimental image using the plot_image_overlay
    function.

    
    """
    f=plt.figure()
    imgavg=np.average(cell_data,axis=0)
    plt.imshow(imgavg, interpolation = 'nearest')
    if target_locations is None:
        # identify the peaks on the average image
        target_locations=pc.peak_attribs_image(imgavg, peak_width)[:,:2]
    # plot the peak labels
    for pk_id in xrange(target_locations.shape[0]):
        plt.text(target_locations[pk_id,0], target_locations[pk_id,1], 
                 "%s"%pk_id, size=10, rotation=0.,
                 ha="center", va="center",
                 bbox = dict(boxstyle="round",
                             ec=(1., 0.5, 0.5),
                             fc=(1., 0.8, 0.8),
                             )
                 )
    return f

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
    if not hasattr(self.mapped_parameters, "original_files"):
        messages.warning(
            "No original files available.  Can't map anything to nothing."
            'If you use the cell_cropper function to crop your cells, the '
            'cell locations and original files will be tracked for you.')
        return None
    if peak_id is not None and (plot_shift is False and plot_char is None):
        messages.warning(
            'Peak ID provided, but no plot_char given , and plot_shift '
            'disabled. Nothing to plot.  Try again.')
        return None
    if peak_mva and not (plot_char is not None or plot_shift or 
                         plot_component):
        messages.warning(
        'peak_mva specified, but no peak characteristic, peak shift, or '
        'component score selected for plotting.  Nothing to plot.')
        return None
    if plot_char is not None and plot_component is not None:
        messages.warning(
        'Both plot_char and plot_component provided.  Can only plot one '
        'of these at a time.  Try again.\nNote that you can actually plot '
        'shifts and component scores simultaneously.')
        return None
    figs=[]
    for key in self.mapped_parameters.original_files.keys():
        f=plt.figure()
        plt.title(key)
        plt.imshow(self.mapped_parameters.original_files[key].data, 
            interpolation = 'nearest')
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
                messages.warning(
                    "Unrecognized MVA type.  Currently supported MVA types "
                    "are PCA and ICA (case insensitive)")
                return None
            print mask
            print locs
            print scores
            plt.scatter(locs[:,0],locs[:,1],c=scores)
            plt.jet()
            plt.colorbar()
        figs.append(f)
    return figs
    
def plot_cell_overlay(cell_data, f_pc, locations, ax=None, plot_shifts=True, 
                      plot_char=None, cmap=plt.cm.jet):
    """Overlays peak characteristics on an image plot of the average image.

    Only appropriate for Image objects that consist of 3D stacks of cropped
    data.

    Parameters:

    cell_data - numpy array
        The data array containing a cell image on which to overlay peak characteristics.
        Generally the average cell from a stack of images.

    f_pc - numpy array (short for factors/peak characteristics)
        The data array containing either peak characteristics (positions, height, etc.)
        or the factors derived from such peak characteristics using PCA or ICA.
        Supplied by the peak_char.peak_char_stack function.

    locations - numpy array
        The data array containing the locations of peaks that have been characterized.         

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
           
    cmap : a matplotlib colormap
        The colormap used for any peak characteristic scatter map
        overlay.
    """

    shifts=np.zeros((locations.shape[0],2))
    char=np.zeros(locations.shape[0])   

    for pos in xrange(locations.shape[0]):
        shifts[pos]=f_pc[pos*7+2:pos*7+4]
        if plot_char:
            char[pos]=f_pc[pos*7+plot_char]

    if ax==None:
        ax=plt.gca()
    ax.imshow(cell_data, interpolation = 'nearest',cmap=plt.cm.gray)

    if plot_shifts:
        ax.quiver(locations[:,0],locations[:,1],
                   shifts[:,0], shifts[:,1],
                   units='xy', color='white'
                   )
    if plot_char is not None :
        ax.scatter(locations[:,0],locations[:,1],c=char,
                    cmap=cmap)
        div=make_axes_locatable(ax)
        cax=div.append_axes("right",size="5%",pad=0.05)
        plt.colorbar(im,cax=cax)
    return f

def _plot_pc(idx, on_peaks=False,cmap=plt.cm.gray):
    target=self._get_target(on_peaks)
    ax=plt.gca()
    im=ax.imshow(target.pc[:,idx].reshape(self.axes_manager.axes[1].size,
                self.axes_manager.axes[2].size), cmap=cmap, 
                interpolation = 'nearest')
    plt.title('PC %s' % idx)
    div=make_axes_locatable(ax)
    cax=div.append_axes("right",size="5%",pad=0.05)
    plt.colorbar(im,cax=cax)

    

def plot_principal_components(n = None, same_window=True, per_row=3, 
                              on_peaks=False, cmap=plt.cm.gray):
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
            _plot_pc(i,on_peaks,cmap=cmap)

    else:
        fig = plt.figure()
        rows=int(np.ceil(n/float(per_row)))
        idx=0
        for i in xrange(rows):
            for j in xrange(per_row):
                if idx<n:
                    fig.add_subplot(rows,per_row,idx+1)
                    _plot_pc(idx,on_peaks,cmap=cmap)
                    idx+=1
        plt.suptitle('Principal components')
        plt.draw()

def plot_independent_components(ic, same_window=True, per_row=3, 
                                cell_data=None, on_peaks=False, 
                                cmap=plt.cm.gray):
    """Plot the independent components.

    Parameters
    ----------
    ic : numpy array
         externally provided independent components array
         The shape of 'ic' must be (channels, n_components),
         so that e.g. ic[:, 0] is the first independent component.

    same_window : bool (optional)
                if 'True', the components will be plotted in the
                same window. Default is 'False'.

    per_row : int (optional)
                When same_window is True, this is the number of plots
                per row in the single window.

    Image-specific parameters
    -------------------------
    cell_data : 2D numpy array (required if on_peaks is True)
        A cell image from which peak characteristics have been
        derived.  Most often you would use an average image here.
        The characteristics are overlaid on this image.

    on_peaks : bool (optional)
        If True, plots factors/score maps based on peak characteristics.
        Requires ic to be based on peak characteristic data, and requires
        cell_data to be the stack from which ic was derived.

    cmap : the matplotlib colormap to apply to the factor image.
    """

    n = ic.shape[1]

    if not same_window:
        for i in xrange(n):
            plt.figure()
            _plot_ic(i, on_peaks, cmap=cmap)

    else:
        fig = plt.figure()
        rows=int(np.ceil(n/float(per_row)))
        idx=0
        for i in xrange(rows):
            for j in xrange(per_row):
                if idx<n:
                    fig.add_subplot(rows,per_row,idx+1)
                    _plot_ic(idx, on_peaks,cmap=cmap)
                    idx+=1
        plt.suptitle('Independent components')

def plot_maps(scores, factors=None, comp_ids=None, locations=None,
              original_files=None, mva_type=None,
              cmap=plt.cm.gray, no_nans=False, per_row=3,
              scoremap=True, save_figs=False, directory = None):
    """
    Plot component maps for the different MSA types

    Parameters
    ----------

    scores : numpy array, the array of scores

    factors : numpy array, the array of components, with each column as a component.

    mva_type : string, currently either 'pca' or 'ica'

    comp_ids : None, int, or list of ints
        if None, returns maps of all components.
        if int, returns maps of components with ids from 0 to given int.
        if list of ints, returns maps of components with ids in given list.

    locations : numpy recarray
        'filename' : string, the filename from which a cropped cell came from.
            Corresponds with the keys in original_files parameter.
        'id' : integer, the integer id of the cropped cell
        'position' : 1x2 numpy array, the location of the cropped cell on its
            original image.  For cells cropped using the cell_cropper function,
            this is the upper-left corner of the cropped cell.

    original_files : dictionary
        A dictionary of the files that data is mapped to.  Keys are
        the file names, and the values are the Signal (or subclass) 
        instances.

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
                    parents={self.mapped_parameters.parent.mapped_parameters.name:self.mapped_parameters.parent}
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


def plot_principal_components_maps(scores, pc, comp_ids=None, locations=None,
                                   original_files=None,
                                   cmap=plt.cm.gray, per_row=3, 
                                   scoremap=True, save_figs=False, 
                                   directory=None):
    """Plot the map associated to each independent component

    Parameters
    ----------
    scores : numpy array
        externally supplied scores matrix

    pc : numpy array
        externally supplied principal components

    locations : numpy recarray
        'filename' : string, the filename from which a cropped cell came from.
            Corresponds with the keys in original_files parameter.
        'id' : integer, the integer id of the cropped cell
        'position' : 1x2 numpy array, the location of the cropped cell on its
            original image.  For cells cropped using the cell_cropper function,
            this is the upper-left corner of the cropped cell.

    original_files : dictionary
        A dictionary of the files that data is mapped to.  Keys are
        the file names, and the values are the Signal (or subclass) 
        instances.

    comp_ids : None, int, or list of ints
        if None, returns maps of all components.
        if int, returns maps of components with ids from 0 to given int.
        if list of ints, returns maps of components with ids in given list.

    cmap : plt.cm object, the colormap of the factor image

    no_nans : bool (optional)
         whether substituting NaNs with zeros for a visually prettier plot
         (default is False)

    per_row : int (optional)
        The number of plots per row in the multi-pane window.

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
    return plot_maps(scores=scores, factors=pc, locations=locations,
                     original_files=original_files, comp_ids=comp_ids, 
                     mva_type='pca',cmap=cmap,
                     scoremap=scoremap,save_figs=save_figs,
                     per_row=per_row, directory=directory)

def plot_independent_components_maps(scores, ic, locations=None,
                                     original_files=None, comp_ids=None, 
                                     cmap=plt.cm.gray, no_nans=False,
                                     scoremap=True, per_row=3,
                                     save_figs=False, directory = None):
    """Plot the map associated to each independent component

    Parameters
    ----------

    scores : numpy array
        externally suplied scores matrix

    ic : numpy array
        externally supplied independent components

    locations : numpy recarray
        'filename' : string, the filename from which a cropped cell came from.
            Corresponds with the keys in original_files parameter.
        'id' : integer, the integer id of the cropped cell
        'position' : 1x2 numpy array, the location of the cropped cell on its
            original image.  For cells cropped using the cell_cropper function,
            this is the upper-left corner of the cropped cell.

    original_files : dictionary
        A dictionary of the files that data is mapped to.  Keys are
        the file names, and the values are the Signal (or subclass) 
        instances.

    comp_ids : int or list of ints
        if None, returns maps of all components.
        if int, returns maps of components with ids from 0 to given int.
        if list of ints, returns maps of components with ids in given list.

    cmap : plt.cm object

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
    return plot_maps(scores=scores, factors=ic, components=comp_ids,
                     locations=locations, original_files=original_files,
                     mva_type='ica',cmap=cmap, no_nans=no_nans,
                     scoremap=scoremap, per_row=per_row,
                     save_figs=save_figs, directory = directory)
