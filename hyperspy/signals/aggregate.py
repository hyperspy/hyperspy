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
from hyperspy.signals.image import Image
from hyperspy.signals.spectrum import Spectrum
import enthought.traits.api as t
from hyperspy.learn.mva import MVA_Results
from hyperspy.axes import AxesManager, DataAxis
from hyperspy.misc.utils import DictionaryBrowser
from hyperspy import messages

from hyperspy.io import load

from copy import deepcopy
import numpy as np
import mdp

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
        self.data=None
        self.mapped_parameters=DictionaryBrowser()
        self.mapped_parameters.original_files=OrderedDict()
        super(Aggregate, self).__init__(*args, **kw)


    def summary(self):
        smp=self.mapped_parameters
        print "\nAggregate Contents: "
        for f in smp.original_files.keys():
            print f
        print "\nTotal size: %s"%(str(self.data.shape))
        print "Data representation: %s"%self.mapped_parameters.record_by
        if hasattr(self.mapped_parameters,'signal'):
            print "Signal type: %s"%self.mapped_parameters.signal
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
        self.mapped_parameters.aggregate_address=OrderedDict()
        self.mapped_parameters.aggregate_end_pointer=0
        super(AggregateSpectrum,self).__init__(*args,**kw)
        if len(args)>0:
            self.append(*args)
            self.summary()

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
            for arg in args:
                if arg.__class__.__name__=='str':
                    if '*' in arg:
                        from glob import glob
                        flist=glob(arg)
                        for f in flist:
                            d=load(f)
                            if d.mapped_parameters.record_by=="spectrum":
                                self._add_object(d)
                    else:
                        arg=load(arg)
                        self._add_object(arg)
                elif isinstance(arg,Signal):
                    self._add_object(arg)
                else:
                    # skip over appending if something like a dict is passed as arg
                    return
            self.axes_manager.navigation_dimension=1

            # refresh the axes for the new sized data
            smp.name="Aggregate Spectra: %s"%smp.original_files.keys()

    def _crop_bounds(self,arg,points_to_interpolate=3):
        argbounds=np.array([arg.axes_manager.axes[-1].low_value,
                          arg.axes_manager.axes[-1].high_value])
        selfbounds=np.array([self.axes_manager.axes[-1].low_value,
                          self.axes_manager.axes[-1].high_value])
        newlims=selfbounds-argbounds
        # file to be added is below current spectral range.
        if argbounds[1]<selfbounds[0]:
            messages.warning('file "%s" is below current spectral range\
.  Omitting it.'%arg.mapped_parameters.name)
            return None
        # file to be added is above current spectral range.
        if argbounds[0]>selfbounds[1]:
            messages.warning('file "%s" is above current spectral range\
.  Omitting it.'%arg.mapped_parameters.name)
            return None

        selflims=slice(None,None,1)
        datalims=slice(None,None,1)
        if (newlims[0]<0):
            # trim left of existing bounds.
            self_idx=np.argmin(np.abs(self.axes_manager.axes[-1].axis-argbounds[0]))
            selflims=slice(self_idx,None,1)
            self.axes_manager.axes[-1].offset=argbounds[0]
        elif newlims[0]>0:
            # trim left of file to be added.
            data_idx=np.argmin(np.abs(arg.axes_manager.axes[-1].axis-selfbounds[0]))
            datalims=slice(data_idx,None,1)
        if newlims[1]>0:
            # trim right of existing bounds
            self_idx=np.argmin(np.abs(self.axes_manager.axes[-1].axis-argbounds[1]))+1
            selflims=slice(selflims.start,self_idx,1)
        elif newlims[1]<0:
            # trim right of file to be added
            data_idx=np.argmin(np.abs(arg.axes_manager.axes[-1].axis-selfbounds[1]))+1
            datalims=slice(datalims.start,data_idx,1)
          
        # trim to the shorter of the two data sets. (max 2 points)
        datashape=arg.data[:,datalims].shape
        selfshape=self.data[:,selflims].shape
        if (datashape[-1]<selfshape[-1]):
            if (datashape[-1]-selfshape[-1])<-2:
                messages.warning("large array size difference (%i) in file %s - are you using similar binning and dispersion?  File omitted."%(datashape[-1]-selfshape[-1],arg.mapped_parameters.name))
                return None
            else:
                if selflims.start<>None:
                    selflims=slice(selflims.start-(datashape[-1]-selfshape[-1]),selflims.stop,1)
                elif selflims.stop<>None:
                    selflims=slice(selflims.start,selflims.stop+(datashape[-1]-selfshape[-1]),1)
                else:
                    selflims=slice(selfshape[-1]-datashape[-1],None,1)
        if datashape[-1]-selfshape[-1]:
            if (datashape[-1]-selfshape[-1])>2:
                messages.warning("large array size difference (%i) in file %s- are you using similar binning and dispersion? File omitted."%(datashape[-1]-selfshape[-1],arg.mapped_parameters.name))
                return None
            else:
                if datalims.start<>None:
                    datalims=slice(datalims.start+(datashape[-1]-selfshape[-1]),datalims.stop,1)
                elif datalims.stop<>None:
                    datalims=slice(datalims.start,datalims.stop-(datashape[-1]-selfshape[-1]),1)
                else:
                    datalims=slice(datashape[-1]-selfshape[-1],None,1)
        # recalculate the axis size (and bounds) for any future rounds.
        self.axes_manager.axes[-1].size=self.data.shape[-1]
        return datalims,selflims
        

    def _add_object(self,arg):
        #object parameters
        mp=arg.mapped_parameters
        smp=self.mapped_parameters
        if mp.original_filename not in smp.original_files.keys():
            smp.original_files[mp.name]=arg
            # save the original data shape to the mva_results for later use
            smp.original_files[mp.name].mva_results.original_shape = arg.data.shape[:-1]
            if self.data==None:
                self.axes_manager=arg.axes_manager.copy()
                if len(arg.data.shape)==1:
                    new_axis=DataAxis(**{
                            'name': 'Depth',
                            'scale': 1.,
                            'offset': 0.,
                            'size': int(arg.data.shape[0]),
                            'units': 'undefined',
                            'index_in_array': 0, })
                    for ax_idx in xrange(len(arg.axes_manager.axes)):
                        self.axes_manager.axes[ax_idx].index_in_array+=1
                    self.axes_manager.axes.insert(0,new_axis)
            if len(arg.data.shape)==3:
                arg.unfold()
                smp.aggregate_address[mp.name]=(
                    smp.aggregate_end_pointer,smp.aggregate_end_pointer+arg.data.shape[0]-1)
            if len(arg.data.shape)==2:
                smp.aggregate_address[mp.name]=(
                    smp.aggregate_end_pointer,smp.aggregate_end_pointer+arg.data.shape[0]-1)
            if len(arg.data.shape)==1:
                arg.data=arg.data[np.newaxis,:]
                smp.aggregate_address[mp.name]=(
                    smp.aggregate_end_pointer,smp.aggregate_end_pointer+1)

            # add the data to the aggregate array
            if self.data==None:
                # copy the axes for the sake of calibration
                self.data=arg.data
                self.mapped_parameters.record_by=arg.mapped_parameters.record_by
                if hasattr(arg.mapped_parameters,'signal'):
                    self.mapped_parameters.signal=arg.mapped_parameters.signal
            else:
                bounds=self._crop_bounds(arg)
                if bounds:
                    arglims,selflims=bounds
                    self.data=self.data[:,selflims]
                    arg.data=arg.data[:,arglims]
                    try:
                        self.data=np.append(self.data,arg.data,axis=0)
                        smp.aggregate_end_pointer=self.data.shape[0]
                        self.axes_manager.axes[0].size=self.data.shape[0]
                        self.axes_manager.axes[-1].size=self.data.shape[-1]
                    except:
                        messages.warning('Adding file %s to aggregate failed.  \
Are you sure its dimensions agree with all the other files you\'re trying \
to add?'%arg.mapped_parameters.name)
                        return None
        else:
            print "Data from file %s already in this aggregate. \n \
    Delete it first if you want to update it."%mp.original_filename

    def remove(self,*keys):
        smp=self.mapped_parameters
        for key in keys:
            del smp.original_files[key]
            address=smp.aggregate_address[key]
            self.data=np.delete(self.data,np.s_[address[0]:(address[1]+1):1],0)
        self.axes_manager.axes[0].size=int(self.data.shape[0])
        smp.aggregate_end_pointer=self.data.shape[0]
        smp.name="Aggregate Spectra: %s"%smp.original_files.keys()

    def principal_components_analysis(self, normalize_poissonian_noise = False, 
                                     algorithm = 'svd', output_dimension = None, navigation_mask = None, 
                                     signal_mask = None, center = False, variance2one = False, var_array = None, 
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
        output_dimension : None or int
            number of PCA to keep
        navigation_mask : boolean numpy array
        signal_mask : boolean numpy array
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
                                     algorithm, output_dimension, navigation_mask, 
                                     signal_mask, center, variance2one, var_array, 
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
            mvar.output_dimension = smvar.output_dimension
            mvar.unfolded = True
            # Demixing matrix
            mvar.w = smvar.w
            smp.original_files[key].fold()
            
class AggregateImage(Aggregate,Image):
    def __init__(self, *args, **kw):
        self.mapped_parameters=DictionaryBrowser()
        super(AggregateImage, self).__init__(*args,**kw)
        if not hasattr(self.mapped_parameters,'original_files'):
            self.mapped_parameters.original_files=OrderedDict()
        if len(args)>0:
            self.append(*args)
            self.summary()

    def append(self, *args):
        if len(args)<1:
            pass
        else:
            for arg in args:
                if arg.__class__.__name__=='str':
                    if '*' in arg:
                        from glob import glob
                        flist=glob(arg)
                        for f in flist:
                            d=load(f)
                            if d.mapped_parameters.record_by=="image":
                                self._add_object(d)
                    else:
                        arg=load(arg)
                        self._add_object(arg)
                elif isinstance(arg,Signal):
                    self._add_object(arg)
                else:
                    # skip over appending if something like a dict is passed as arg
                    return

    def _add_object(self, arg):
        smp=self.mapped_parameters
        mp=arg.mapped_parameters
        if mp.original_filename not in smp.original_files.keys():
            smp.original_files[mp.name]=arg
            # add the data to the aggregate array
            if self.data==None:
                self.data=arg.data[np.newaxis,:,:]
                smp.record_by=mp.record_by
                if hasattr(mp,'signal'):
                    smp.signal=mp.signal
                self.axes_manager=AxesManager(self._get_undefined_axes_list())
                self.axes_manager.axes[1]=deepcopy(arg.axes_manager.axes[0])
                self.axes_manager.axes[1].index_in_array+=1
                self.axes_manager.axes[2]=deepcopy(arg.axes_manager.axes[1])
                self.axes_manager.axes[2].index_in_array+=1
            else:
                self.data=np.append(self.data,arg.data[np.newaxis,:,:],axis=0)
                self.axes_manager.axes[0].size+=1
        else:
            print "Data from file %s already in this aggregate. \n \
    Delete it first if you want to update it."%mp.original_filename
            # refresh the axes for the new sized data
            #self.axes_manager=AxesManager(self._get_undefined_axes_list())
        smp.name="Aggregate Image: %s"%smp.original_files.keys()


    def remove(self,*keys):
        smp=self.mapped_parameters
        for key in keys:
            idx=smp.original_files.keys().index(key)
            self.data=np.delete(self.data,np.s_[idx:idx+1:1],0)
            self.axes_manager.axes[0].size-=1
            del smp.original_files[key]
        smp.name="Aggregate Image: %s"%smp.original_files.keys()

class AggregateCells(Aggregate,Image):
    """ A class to deal with several image stacks, each consisting of cropped
    sub-images from a template-matched experimental image.
    """

    def __init__(self, *args, **kw):
        super(AggregateCells,self).__init__(*args, **kw)
        smp=self.mapped_parameters
        if not hasattr(self,'_shape_before_unfolding'):
            self._shape_before_unfolding = None
        if not hasattr(smp,'locations'):
            smp.locations=np.zeros((2,1),dtype=[('filename','a256'),('id','i4'),('position','i4',(1,2))])
        if not hasattr(smp,'original_files'):
            smp.original_files=OrderedDict()
        if not hasattr(smp,'image_stacks'):
            smp.image_stacks=OrderedDict()
        if not hasattr(smp,'aggregate_end_pointer'):
            smp.aggregate_end_pointer=0
        if not hasattr(smp,'name'):
            smp.name="Aggregate: no data"
        if len(args)>0:
            self.append(*args)
            self.summary()
            
    def append(self,*args):
        if len(args)<1:
            pass
        else:
            for arg in args:
                if arg.__class__.__name__=='str':
                    if '*' in arg:
                        from glob import glob
                        flist=glob(arg)
                        for f in flist:
                            d=load(f)
                            if d.mapped_parameters.record_by=="image":
                                self._add_object(d)
                    else:
                        arg=load(arg)
                        self._add_object(arg)
                elif isinstance(arg,Signal):
                    self._add_object(arg)
                else:
                    # skip over appending if something like a dict is passed as arg
                    return

    def _add_object(self,arg):
        smp=self.mapped_parameters
        #object parameters
        mp=arg.mapped_parameters
        pmp=mp.parent.mapped_parameters
                
        if pmp.original_filename not in list(set(smp.locations['filename'].squeeze())):
            smp.original_files[pmp.name]=mp.parent
            smp.image_stacks[pmp.name]=arg
            # add the data to the aggregate array
            if self.data==None:
                smp.record_by=mp.record_by
                if hasattr(mp,'locations'):
                    smp.locations=mp.locations
                if hasattr(mp,'signal'):
                    smp.signal=mp.signal
                if len(arg.data.shape)<3:
                    self.data=arg.data[np.newaxis,:,:]
                    self.axes_manager=AxesManager(self._get_undefined_axes_list())
                    self.axes_manager.axes[1]=deepcopy(arg.axes_manager.axes[0])
                    self.axes_manager.axes[1].index_in_array+=1
                    self.axes_manager.axes[2]=deepcopy(arg.axes_manager.axes[1])
                    self.axes_manager.axes[2].index_in_array+=1
                else:
                    self.axes_manager=arg.axes_manager.copy()
                    self.data=arg.data
            else:
                mp.locations['id']=mp.locations['id']+smp.aggregate_end_pointer
                smp.locations=np.append(smp.locations,mp.locations)
                self.data=np.append(self.data,arg.data,axis=0)
                self.axes_manager.axes[0].size+=int(arg.data.shape[0])
                smp.aggregate_end_pointer=self.data.shape[0]
        else:
            print "Data from file %s already in this aggregate. \n \
    Delete it first if you want to update it."%pmp.original_filename
            # refresh the axes for the new sized data
        smp.name="Aggregate Cells: %s"%list(set(smp.locations['filename'].squeeze()))

    def remove(self,*keys):
        smp=self.mapped_parameters
        for key in keys:
            del smp.original_files[key]
            del smp.image_stacks[key]
            mask=locs['filename']==key
            self.data=np.delete(self.data,mask,0)
            smp.locations=np.delete(smp.locations,mask,0)
            self.axes_manager.axes[0].size=int(self.data.shape[0])
        smp.aggregate_end_pointer=self.data.shape[0]
        smp.name="Aggregate Cells: %s"%list(set(smp.locations['filename'].squeeze()))

    def kmeans_cluster_stack(self, clusters=None):
        smp=self.mapped_parameters
        d=self.data
        # if clusters not given, try to determine what it should be.
        if clusters is None:
            pass
        kmeans=mdp.nodes.KMeansClassifier(clusters)
        avg_stack=np.zeros((clusters,d.shape[1],d.shape[2]))
        kmeans.train(d.reshape((-1,d.shape[0])).T)
        kmeans.stop_training()
        groups=np.array(kmeans.label(d.reshape((-1,d.shape[0])).T))
        cluster_arrays=[]

        try:
            # test if location data is available
            smp.locations[0]
        except:
            print "Warning: No cell location information was available."
        for i in xrange(clusters):
            # get number of members of this cluster
            members=groups[groups==i].shape[0]
            cluster_array=np.zeros((members,d.shape[1],d.shape[2]))
            # positions is a recarray, with each row consisting of a filename and the position from
            # which the crop was taken.
            positions=smp.locations[groups==i]
            for j in xrange(positions.shape[0]):
                cluster_array[j,:,:]=d[positions[j]['id'],:,:]
            # create a trimmed dict of the original files for this particular
            # cluster.  Only include files that thie cluster includes members
            # from.
            constrained_orig_files=OrderedDict()
            for key in self.mapped_parameters.original_files.keys():
                if key in positions['filename']:
                    constrained_orig_files[key]=smp.original_files[key]
            cluster_array_Image=Image({'data':cluster_array,
                    'mapped_parameters':{
                        'name':'Cluster %s from %s'%(i,
                                         smp.name),
                        'locations':positions,
                        'members':members,
                        'original_files':constrained_orig_files,
                        }
                    })
            cluster_arrays.append(cluster_array_Image)
            avg_stack[i,:,:]=np.sum(cluster_array,axis=0)
        members_list=[groups[groups==i].shape[0] for i in xrange(clusters)]
        avg_stack_Image=Image({'data':avg_stack,
                    'mapped_parameters':{
                        'name':'Cluster averages from %s'%smp.name,
                        'member_counts':members_list,
                        'original_files':smp.original_files,
                        }
                    })
        smp.avgs=avg_stack_Image
        smp.clusters=cluster_arrays
        print "Averages and classes stored in mapped_parameters.  Access them as: \n\n\
your_object.mapped_parameters.avgs \n\n\
or \n\n\
your_object.mapped_parameters.clusters\n"

    def plot_cell_overlays(self, plot_component=None, mva_type='PCA', peak_mva=True, 
                           plot_shifts=True, plot_char=None):
        """
        Overlays peak characteristics on an image plot of the average image.
        This the AggregateCells version of this function, which creates plots 
        for all of the classes obtained from kmeans clustering.

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
            for idx in xrange(len(self.mapped_parameters.clusters)):
                clusters[idx].plot_cell_overlays(plot_component=plot_component, mva_type=mva_type, 
                                                 peak_mva=peak_mva, plot_shifts=plot_shifts, 
                                                 plot_char=plot_char)
        else:
            print "No clusters found on aggregate object.  Have you run K-means clustering?"
            return
                  
    def save_avgs(self):
        avgs=self.mapped_parameters.avgs
        f=plt.figure()
        for i in xrange(avgs.data.shape[0]):
            img=plt.imshow(avgs.data[i,:,:])
            #plt.title(title="Class avg %02i, %02i members"%(i,avgs.mapped_parameters.member_counts[i]))
            f.savefig('class_avg_%02i_[%02i].png'%(i,avgs.mapped_parameters.member_counts[i]))
        plt.close(f)

                
