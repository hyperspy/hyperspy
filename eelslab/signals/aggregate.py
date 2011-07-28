# -*- coding: utf-8 -*-
# Copyright © 2011 Michael Sarahan
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

from eelslab.signal import Signal, Parameters
from eelslab.signals.image import Image
from eelslab.signals.spectrum import Spectrum
import enthought.traits.api as t
from eelslab.mva.mva import MVA_Results
from eelslab.axes import AxesManager

import numpy as np
import mdp

from collections import OrderedDict

class Aggregate(Signal):
    def __init__(self, *args, **kw):
        # this axes_manager isn't really ideal for Aggregates.
        self.axes_manager=AxesManager([{   'name' : 'undefined',
                            'scale' : 1.,
                            'offset' : 0.,
                            'size' : 1,
                            'units' : 'undefined',
                            'index_in_array' : 0,}])
        super(Aggregate, self).__init__(*args, **kw)
        self.data=None
        self.mapped_parameters.original_files=OrderedDict()

    """
    def plot(self):
        print "Plotting not yet supported for aggregate objects"
        return None

    def unfold(self):
        print "Aggregate objects are already unfolded, and cannot be folded. \
Perhaps you'd like to instead access its component members?"
        return None

    def fold(self):
        print "Folding not supported for Aggregate objects."
        return None
    """
    def print_keys(self):
        smp=self.mapped_parameters
        print smp.locations.keys()

    def append(self):
        print "sorry, append method not implemented generally yet."


class AggregateSpectrum(Aggregate,Spectrum):
    def __init__(self, *args, **kw):
        super(AggregateSpectrum,self).__init__(*args,**kw)
        self.mapped_parameters.aggregate_address=OrderedDict()
        self.mapped_parameters.aggregate_end_pointer=0
        if len(args)>0:
            self.append(*args)

    def unfold(self):
        print "AggregateSpectrum objects are already unfolded, and cannot be folded. \
Perhaps you'd like to instead access its component members, or try the split_results method?"
        return None

    def fold(self):
        print "Folding not supported for AggregateSpectrum objects."
        return None

    def append(self,*args):
        if len(args)<1:
            pass
        else:
            smp=self.mapped_parameters
            for arg in args:
                #object parameters
                mp=arg.mapped_parameters
                if mp.name not in smp.locations.keys():
                    smp.original_files[mp.name]=arg
                    arg.unfold()
                    smp.aggregate_address[mp.name]=(
                        smp.aggregate_end_pointer,smp.aggregate_end_pointer+arg.data.shape[-1]-1)
                    # add the data to the aggregate array
                    if self.data==None:
                        self.data=np.atleast_3d(arg.data)
                    else:
                        self.data=np.append(self.data,arg.data,axis=2)
                    smp.aggregate_end_pointer=self.data.shape[2]
                else:
                    print "Data from file %s already in this aggregate. \n \
    Delete it first if you want to update it."%mp.parent
            # refresh the axes for the new sized data
            self.axes_manager=AxesManager(self._get_undefined_axes_list())
            smp.name="Aggregate Cells: %s"%smp.locations.keys()

    def remove(self,*keys):
        smp=self.mapped_parameters
        for key in keys:
            del smp.locations[key]
            del smp.original_files[key]
            del smp.image_stacks[key]
            address=smp.aggregate_address[key]
            self.data=np.delete(self.data,np.s_[address[0]:address[1]:1],2)
        self.axes_manager=AxesManager(self._get_undefined_axes_list())
        smp.aggregate_end_pointer=self.data.shape[2]
        smp.name="Aggregate Cells: %s"%smp.locations.keys()

    def split_results(self):
        """Method to take any multivariate analysis results from the aggregate and
        split them into the constituent SI's.  Required before the individual mva_results
        can be made sense of.

        Note: this method is called automatically at the end of PCA or ICA on AggregateSpectrum
        objects.

        """
        for f in self.mapped_parameters.original_files.keys():
            pass
            

class AggregateImage(Aggregate,Image):
    def __init__(self, *args, **kw):
        super(AggregateImage, self).__init__(*args,**kw)
        if len(args)>0:
            self.append(*args)

    def append(self, *args):
        if len(args)<1:
            pass
        else:
            smp=self.mapped_parameters
            print args
            for arg in args:
                #object parameters
                mp=arg.mapped_parameters
                
                if mp.name not in smp.original_files.keys():
                    smp.original_files[mp.name]=arg
                    # add the data to the aggregate array
                    if self.data==None:
                        self.data=np.atleast_3d(arg.data)
                    else:
                        self.data=np.append(self.data,np.atleast_3d(arg.data),axis=2)
                else:
                    print "Data from file %s already in this aggregate. \n \
    Delete it first if you want to update it."%mp.parent
            # refresh the axes for the new sized data
            self.axes_manager=AxesManager(self._get_undefined_axes_list())
            smp.name="Aggregate Image: %s"%smp.original_files.keys()

    def remove(self,*keys):
        smp=self.mapped_parameters
        for key in keys:
            idx=smp.original_files.keys().index(key)
            self.data=np.delete(self.data,np.s_[idx:idx+1:1],2)
            del smp.original_files[key]
        self.axes_manager=AxesManager(self._get_undefined_axes_list())
        smp.name="Aggregate Image: %s"%smp.original_files.keys()

    def cell_cropper(self):
        if not hasattr(self.mapped_parameters,"picker"):
            import eelslab.drawing.ucc as ucc
            self.mapped_parameters.picker=ucc.TemplatePicker(self)
        self.mapped_parameters.picker.configure_traits()
        return self.mapped_parameters.picker.crop_sig

class AggregateCells(Aggregate,Image):
    """ A class to deal with several image stacks, each consisting of cropped
    sub-images from a template-matched experimental image.
    """

    def __init__(self, *args, **kw):
        super(AggregateCells,self).__init__(*args, **kw)
        self._shape_before_unfolding = None
        self.mapped_parameters.locations=OrderedDict()
        self.mapped_parameters.original_files=OrderedDict()
        self.mapped_parameters.image_stacks=OrderedDict()
        self.mapped_parameters.aggregate_address=OrderedDict()
        self.mapped_parameters.aggregate_end_pointer=0
        self.mapped_parameters.name="Aggregate: no data"
        if len(args)>0:
            self.append(*args)

    def append(self,*args):
        if len(args)<1:
            pass
        else:
            smp=self.mapped_parameters
            for arg in args:
                #object parameters
                mp=arg.mapped_parameters
                pmp=mp.parent.mapped_parameters
                
                if pmp.name not in smp.locations.keys():
                    smp.locations[pmp.name]=mp.locations
                    smp.original_files[pmp.name]=mp.parent
                    smp.image_stacks[pmp.name]=arg
                    smp.aggregate_address[pmp.name]=(
                        smp.aggregate_end_pointer,smp.aggregate_end_pointer+arg.data.shape[-1]-1)
                    # add the data to the aggregate array
                    if self.data==None:
                        self.data=np.atleast_3d(arg.data)
                    else:
                        self.data=np.append(self.data,arg.data,axis=2)
                    smp.aggregate_end_pointer=self.data.shape[2]
                else:
                    print "Data from file %s already in this aggregate. \n \
    Delete it first if you want to update it."%mp.parent
            # refresh the axes for the new sized data
            self.axes_manager=AxesManager(self._get_undefined_axes_list())
            smp.name="Aggregate Cells: %s"%smp.locations.keys()

    def remove(self,*keys):
        smp=self.mapped_parameters
        for key in keys:
            del smp.locations[key]
            del smp.original_files[key]
            del smp.image_stacks[key]
            address=smp.aggregate_address[key]
            self.data=np.delete(self.data,np.s_[address[0]:address[1]:1],2)
        self.axes_manager=AxesManager(self._get_undefined_axes_list())
        smp.aggregate_end_pointer=self.data.shape[2]
        smp.name="Aggregate Cells: %s"%smp.locations.keys()

    def kmeans_cluster_stack(self, clusters=None):
        smp=self.mapped_parameters
        d=self.data
        # if clusters not given, try to determine what it should be.
        if clusters is None:
            pass
        kmeans=mdp.nodes.KMeansClassifier(clusters)
        avg_stack=np.zeros((d.shape[0],d.shape[1],clusters))
        kmeans.train(d.reshape((-1,d.shape[2])).T)
        kmeans.stop_training()
        groups=kmeans.label(d.reshape((-1,d.shape[2])).T)
        cluster_arrays=[]

        try:
            # test if location data is available
            smp.locations.values()[0]
        except:
            print "Warning: No cell location information was available."
        for i in xrange(clusters):
            # which file are we pulling from?
            file_index=0
            address=smp.aggregate_address.values()[file_index]
            fname=smp.locations.keys()[file_index]
            # get number of members of this cluster
            members=groups.count(i)
            cluster_array=np.zeros((d.shape[0],d.shape[1],members))
            cluster_idx=0
            # positions is a recarray, with each row consisting of a filename and the position from
            # which the crop was taken.
            positions=np.zeros((members,1),dtype=[('filename','a256'),('position','i4',(1,2))])
            for j in xrange(len(groups)):
                if j>(address[1]) and fname<>smp.locations.keys()[-1]:
                    file_index+=1
                    fname=smp.locations.keys()[file_index]
                    address=self.mapped_parameters.aggregate_address.values()[file_index]
                file_j=j-address[0]
                if groups[j]==i:
                    cluster_array[:,:,cluster_idx]=d[:,:,j]
                    try:
                        positions[cluster_idx]=(fname,smp.locations[fname][file_j,:2])
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
        self.mapped_parameters.avgs=avg_stack_Image
        self.mapped_parameters.clusters=cluster_arrays
        print "Averages and classes stored in mapped_parameters.  Access them as: \n\n\
your_object.mapped_parameters.avgs \n\n\
or \n\n\
your_object.mapped_parameters.clusters\n"

    def plot_cell_overlays(self):
        if hasattr(self,mapped_parameters.clusters):
            figs={}
            # come up with the color map for the scatter plot

            for key in self.mapped_parameters.original_files.keys():
                figs[key]=plt.figure()
                # plot the initial images
                
            for cluster in xrange(len(self.mapped_parameters.clusters)):
                for loc_id in xrange(cluster.mapped_parameters.locations.shape[0]):
                    # get the right figure
                    fig=figs[cluster.mapped_paramters.locations[loc_id][0][0]]
                    # add scatter point
                    

    def save_avgs(self):
        avgs=self.mapped_parameters.avgs
        f=plt.figure()
        for i in xrange(avgs.data.shape[2]):
            
            img=plt.imshow(avgs.data[:,:,i],title="Class avg %02i, %02i members"%(i,avgs.mapped_parameters.member_counts[i]))
            f.savefig('class_avg_%02i_[%02i].png'%(i,avg.mapped_parameters.member_counts[i]))
        plt.close(f)

                
