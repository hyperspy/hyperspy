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

from eelslab.signal import Signal
from eelslab.peak_char import *
import matplotlib.pyplot as plt

class Image(Signal):
    """
    """    
    def __init__(self, *args, **kwargs):
        Signal.__init__(self, *args, **kwargs)
        self.axes_manager.set_view('image')



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
        self.peak_chars=peak_attribs_stack(self.data, 
                                              peak_width,
                                              subpixel=subpixel, 
                                              target_locations=target_locations,
                                              peak_locations=peak_locations, 
                                              imcoords=imcoords, 
                                              target_neighborhood=target_neighborhood,
                                              medfilt_radius=medfilt_radius
                                              )

    def plot_img_peaks(self, index=0, peak_width=10, subpixel=False,
                       medfilt_radius=5):
        # TODO: replace with hyperimage explorer
        plt.imshow(self.data[:,:,index],cmap=plt.gray())
        peaks=two_dim_peakfind(self.data[:,:,index], subpixel=subpixel,
                                  peak_width=peak_width, 
                                  medfilt_radius=medfilt_radius)
        plt.scatter(peaks[:,0],peaks[:,1])
        
    def cell_cropper(self):
        import eelslab.drawing.ucc as ucc
        picker=ucc.TemplatePicker(self)
        picker.configure_traits()
        return picker.crop_sig

    def kmeans_cluster_stack(self, clusters=None):
        if len(self.data.shape)<>3:
            print "Sorry, this function only works on image stacks (3 dimensions)."
            print " Your data appears to be ", len(self.data.shape), "dimensions."
            return None
        import mdp
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
        for i in xrange(clusters):
            # get number of members of this cluster
            nmembers=groups.count(i)
            cluster_array=np.zeros((d.shape[0],d.shape[1],members))
            cluster_idx=0
            positions=np.zeros((nmembers,3))
            for j in xrange(len(groups)):
                if groups[j]==i:
                    cluster_array[:,:,cluster_idx]=d[:,:,j]
                    positions[cluster_idx]=self.locations[j]
                    cluster_idx+=1
            cluster_array_Image=Image({'data':avg_stack,
                    'mapped_parameters':{
                        'name':'Cluster %s from crops of %s'%(i,self.name),
                        'locations':positions
                        }
                    })
            cluster_arrays.append(cluster_array_Image)
            avg_stack[:,:,i]=np.sum(cluster_array,axis=2)
        avg_stack_Image=Image({'data':avg_stack,
                    'mapped_parameters':{
                        'name':'Cluster averages from crops of %s'%self.name,
                        }
                    })
        return avg_stack_Image, cluster_arrays
