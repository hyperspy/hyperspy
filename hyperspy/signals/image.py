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


import matplotlib.pyplot as plt

from hyperspy.signal import Signal
import hyperspy.peak_char as pc
from hyperspy.misc import utils_varia
from hyperspy.learn.mva import MVA_Results


import numpy as np

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

    def plot_image_peaks(self, index=0, peak_width=10, subpixel=False,
                       medfilt_radius=5):
        # TODO: replace with hyperimage explorer
        plt.imshow(self.data[:,:,index],cmap=plt.gray())
        peaks=pc.two_dim_peakfind(self.data[:,:,index], subpixel=subpixel,
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
        imgavg=np.average(self.data,axis=2)
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
            print "No original files available.  Can't map anything to nothing."
            print "If you use the cell_cropper function to crop your cells, \n\
the cell locations and original files will be tracked for you."
            return None
        if peak_id is not None and (plot_shift is False and plot_char is None):
            print "Peak ID provided, but no plot_char given , and plot_shift disabled."
            print "  Nothing to plot.  Try again."
            return None
        if peak_mva and not (plot_char is not None or plot_shift or plot_component):
            print " peak_mva specified, but no peak characteristic, peak \
shift, or component score selected for plotting.  Nothing to plot."
        if plot_char is not None and plot_component is not None:
            print "Both plot_char and plot_component provided.  Can only plot one\n\
of these at a time.  Try again.\n\n\
Note that you can actually plot shifts and component scores simultaneously."
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
                    print "Unrecognized MVA type.  Currently supported MVA types are \
PCA and ICA (case insensitive)"
                    return None
                print mask
                print locs
                print scores
                plt.scatter(locs[:,0],locs[:,1],c=scores)
                plt.jet()
                plt.colorbar()
            figs.append(f)
        return figs
        
    def plot_cell_peak_overlays(self, plot_component=None, mva_type='PCA', peak_mva=True,
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

        imgavg=np.average(self.data,axis=2)

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

        avg_stack=np.zeros((d.shape[0],d.shape[1],clusters))
        kmeans.train(d.reshape((-1,d.shape[2])).T)
        kmeans.stop_training()
        groups=kmeans.label(d.reshape((-1,d.shape[2])).T)
        try:
            # test if location data is available
            self.mapped_parameters.locations[0]
        except:
            print "Warning: No cell location information was available."
        for i in xrange(clusters):
            # get number of members of this cluster
            members=groups.count(i)
            cluster_array=np.zeros((d.shape[0],d.shape[1],members))
            cluster_idx=0
            positions=np.zeros((members,3))
            for j in xrange(len(groups)):
                if groups[j]==i:
                    cluster_array[:,:,cluster_idx]=d[:,:,j]
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
            avg_stack[:,:,i]=np.sum(cluster_array,axis=2)
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
