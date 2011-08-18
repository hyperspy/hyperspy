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

from copy import deepcopy
from collections import OrderedDict

from matplotlib import pyplot as plt

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

    def summary(self):
        smp=self.mapped_parameters
        print "Aggregate Contents: "
        for f in smp.original_files.keys():
            print f
        print "Total size: %s"%(str(self.data.shape))
    """
    def plot(self):
        print "Plotting not yet supported for generic aggregate objects"
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
        print "sorry, append method not implemented generally yet.  Try either \
the AggregateImage, AggregateCells, or AggregateSpectrum classes"

    def remove(self):
        print "sorry, remove method not implemented generally yet.  Try either \
the AggregateImage, AggregateCells, or AggregateSpectrum classes"

class AggregateSpectrum(Aggregate,Spectrum):
    def __init__(self, *args, **kw):
        super(AggregateSpectrum,self).__init__(*args,**kw)
        self.mapped_parameters.aggregate_address=OrderedDict()
        self.mapped_parameters.aggregate_end_pointer=0
        if len(args)>0:
            self.append(*args)

    def unfold(self):
        print "AggregateSpectrum objects are already unfolded, and cannot be folded. \
Perhaps you'd like to instead access its component members, as \n\n\
f=this_agg_obj.mapped_parameters.original_files['file_name.ext']"
        return None

    def fold(self):
        print "Folding not supported for AggregateSpectrum objects. \n\
Perhaps you'd like to instead access its component members, as \n\n\
f=this_agg_obj.mapped_parameters.original_files['file_name.ext']"
        return None
        
    def append(self,*args):
        if len(args)<1:
            pass
        else:
            smp=self.mapped_parameters
            for arg in args:
                #object parameters
                mp=arg.mapped_parameters
                if mp.name not in smp.original_files.keys():
                    smp.original_files[mp.name]=arg
                    # save the original data shape to the mva_results for later use
                    smp.original_files[mp.name].mva_results.original_shape = arg.data.shape[:-1]
                    arg.unfold()
                    smp.aggregate_address[mp.name]=(
                        smp.aggregate_end_pointer,smp.aggregate_end_pointer+arg.data.shape[0]-1)
                    # add the data to the aggregate array
                    if self.data==None:
                        self.data=np.atleast_2d(arg.data)
                        # copy the axes for the sake of calibration
                        self.axes_manager=deepcopy(arg.axes_manager)
                    else:
                        self.data=np.append(self.data,arg.data,axis=0)
                    smp.aggregate_end_pointer=self.data.shape[0]
                    print "File %s added to aggregate."%mp.name
                else:
                    print "Data from file %s already in this aggregate. \n \
    Delete it first if you want to update it."%mp.parent

            # refresh the axes for the new sized data
            self.axes_manager.axes[0].size=self.data.shape[0]
            smp.name="Aggregate Spectra: %s"%smp.original_files.keys()
            self.summary()

    def remove(self,*keys):
        smp=self.mapped_parameters
        for key in keys:
            del smp.original_files[key]
            address=smp.aggregate_address[key]
            self.data=np.delete(self.data,np.s_[address[0]:(address[1]+1):1],0)
            print "File %s removed from aggregate."%key
        self.axes_manager.axes[0].size=self.data.shape[0]
        smp.aggregate_end_pointer=self.data.shape[0]
        smp.name="Aggregate Spectra: %s"%smp.original_files.keys()
        self.summary()

    def principal_components_analysis(self, normalize_poissonian_noise = False, 
                                     algorithm = 'svd', output_dim = None, spatial_mask = None, 
                                     energy_mask = None, center = False, variance2one = False, var_array = None, 
                                     var_func = None, polyfit = None):
        """Principal components analysis for Aggregate Spectra.
        Different from normal PCA only in that it operates on your
        aggregate data as a whole, and splits the results into
        your constituent files afterwards.
        
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
        super(AggregateSpectrum,self).principal_components_analysis(normalize_poissonian_noise, 
                                     algorithm, output_dim, spatial_mask, 
                                     energy_mask, center, variance2one, var_array, 
                                     var_func, polyfit)
        self._split_mva_results()

    def independent_components_analysis(self, number_of_components = None, 
                                       algorithm = 'CuBICA', diff_order = 1, pc = None, 
                                       comp_list = None, mask = None, **kwds):
        """Independent components analysis.
        
        Available algorithms: FastICA, JADE, CuBICA, TDSEP
        
        Parameters
        ----------
        number_of_components : int
            number of principal components to pass to the ICA algorithm
        algorithm : {FastICA, JADE, CuBICA, TDSEP}
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
        super(AggregateSpectrum,self).independent_components_analysis(number_of_components, 
                                       algorithm, diff_order, pc, 
                                       comp_list, mask, **kwds)
        self._split_mva_results()


    def _split_mva_results(self):
        """Method to take any multivariate analysis results from the aggregate and
        split them into the constituent SI's.  Required before the individual mva_results
        can be made sense of.

        Note: this method is called automatically at the end of PCA or ICA on AggregateSpectrum
        objects.

        """
        smp=self.mapped_parameters
        # shorter handle on the origin
        smvar=self.mva_results
        for key in smp.original_files.keys():
            # get a shorter handle on the destination
            mvar=smp.original_files[key].mva_results
            # copy the principal components
            mvar.pc = smvar.pc
            # copy the appropriate section of the aggregate scores
            agg_address = smp.aggregate_address[key]
            mvar.v  = smvar.v[agg_address[0]:(agg_address[1]+1)]
            # copy eigenvalues (though really, these don't make much 
            # sense on this object, now separate from the whole.)
            mvar.V  = smvar.V
            mvar.pca_algorithm = smvar.pca_algorithm
            mvar.ica_algorithm = smvar.ica_algorithm
            mvar.centered = smvar.centered
            mvar.poissonian_noise_normalized = smvar.poissonian_noise_normalized
            # number of independent components derived
            mvar.output_dim = smvar.output_dim
            mvar.unfolded = True
            # Demixing matrix
            mvar.w = smvar.w
            smp.original_files[key].fold()
            
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
                    print "File %s added to aggregate."%mp.name
                else:
                    print "Data from file %s already in this aggregate. \n \
    Delete it first if you want to update it."%mp.parent
            # refresh the axes for the new sized data
            self.axes_manager=AxesManager(self._get_undefined_axes_list())
            smp.name="Aggregate Image: %s"%smp.original_files.keys()
            self.summary()

    def remove(self,*keys):
        smp=self.mapped_parameters
        for key in keys:
            idx=smp.original_files.keys().index(key)
            self.data=np.delete(self.data,np.s_[idx:idx+1:1],2)
            del smp.original_files[key]
            print "File %s removed from aggregate."%key
        self.axes_manager=AxesManager(self._get_undefined_axes_list())
        smp.name="Aggregate Image: %s"%smp.original_files.keys()
        self.summary()

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
                    print "File %s added to aggregate."%mp.name
                    smp.aggregate_end_pointer=self.data.shape[2]
                else:
                    print "Data from file %s already in this aggregate. \n \
    Delete it first if you want to update it."%mp.parent
            # refresh the axes for the new sized data
            self.axes_manager=AxesManager(self._get_undefined_axes_list())
            smp.name="Aggregate Cells: %s"%smp.locations.keys()
            self.summary()

    def remove(self,*keys):
        smp=self.mapped_parameters
        for key in keys:
            del smp.locations[key]
            del smp.original_files[key]
            del smp.image_stacks[key]
            address=smp.aggregate_address[key]
            self.data=np.delete(self.data,np.s_[address[0]:address[1]:1],2)
            print "File %s removed from aggregate."%key
        self.axes_manager=AxesManager(self._get_undefined_axes_list())
        smp.aggregate_end_pointer=self.data.shape[2]
        smp.name="Aggregate Cells: %s"%smp.locations.keys()
        self.summary()

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
            # create a trimmed dict of the original files for this particular
            # cluster.  Only include files that thie cluster includes members
            # from.
            constrained_orig_files={}
            for key in self.mapped_parameters.original_files.keys():
                if key in positions['filename']:
                    constrained_orig_files[key]=self.mapped_parameters.original_files[key]
            cluster_array_Image=Image({'data':cluster_array,
                    'mapped_parameters':{
                        'name':'Cluster %s from %s'%(i,
                                         self.mapped_parameters.name),
                        'locations':positions,
                        'members':members,
                        'original_files':self.mapped_parameters.original_files,
                        }
                    })
            cluster_arrays.append(cluster_array_Image)
            avg_stack[:,:,i]=np.sum(cluster_array,axis=2)
        members_list=[groups.count(i) for i in xrange(clusters)]
        avg_stack_Image=Image({'data':avg_stack,
                    'mapped_parameters':{
                        'name':'Cluster averages from %s'%self.mapped_parameters.name,
                        'member_counts':members_list,
                        'original_files':self.mapped_parameters.original_files,
                        }
                    })
        self.mapped_parameters.avgs=avg_stack_Image
        self.mapped_parameters.clusters=cluster_arrays
        print "Averages and classes stored in mapped_parameters.  Access them as: \n\n\
your_object.mapped_parameters.avgs \n\n\
or \n\n\
your_object.mapped_parameters.clusters\n"

    def plot_cell_overlays(self, plot_shifts=True, plot_other=None):
        """
        Overlays peak characteristics on an image plot of the average image.
        This the AggregateCells version of this function, which creates plots 
        for all of the classes obtained from kmeans clustering.

        Parameters:
        plot_shifts - bool
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
        if hasattr(self.mapped_parameters,"clusters"):
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
            img=plt.imshow(avgs.data[:,:,i])
            #plt.title(title="Class avg %02i, %02i members"%(i,avgs.mapped_parameters.member_counts[i]))
            f.savefig('class_avg_%02i_[%02i].png'%(i,avgs.mapped_parameters.member_counts[i]))
        plt.close(f)

                
