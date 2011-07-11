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
import enthought.traits.api as t
from eelslab.mva.mva import MVA_Results
from eelslab.axes import AxesManager

import numpy as np

class Aggregate(Signal):
    pass
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

class AggregateImage(Aggregate,Image):
    """ A class to deal with several image stacks, each consisting of cropped
    sub-images from a template-matched experimental image.
    """
    mapped_parameters=t.Instance(Parameters)

    def __init__(self, *args, **kw):
        #super(AggregateImage,self).__init__(*args, **kw)
        super(t.HasTraits,self).__init__()
        self._plot = None
        self.mva_results=MVA_Results()
        self._shape_before_unfolding = None
        self.mapped_parameters=Parameters()
        self.mapped_parameters.locations={}
        self.mapped_parameters.original_files={}
        self.mapped_parameters.image_stacks={}
        self.mapped_parameters.aggregate_address={}
        self.mapped_parameters.aggregate_end_pointer=0
        self.data=None
        if len(args)>0:
            self.append_stacks(*args)

    def append_stacks(self,*args):
        if len(args)<1:
            pass
        else:
            smp=self.mapped_parameters
            for arg in args:
                #object parameters
                mp=arg.mapped_parameters
                
                if mp.parent not in smp.locations.keys():
                    smp.locations[mp.parent]=mp.locations
                    smp.original_files[mp.parent]=mp.parent
                    smp.image_stacks[mp.parent]=arg
                    smp.aggregate_address[mp.parent]=(
                        smp.aggregate_end_pointer,smp.aggregate_end_pointer+arg.data.shape[-1])
                    # add the data to the aggregate array
                    if self.data==None:
                        self.data=arg.data
                    else:
                        self.data=np.append(self.data,arg.data,axis=2)
                else:
                    print "Data from file %s already in this aggregate. \n \
    Delete it first if you want to update it."%mp.parent
            # refresh the axes for the new sized data
            self.axes_manager=AxesManager(self._get_undefined_axes_list())

    def remove_stacks(self,*keys):
        smp=self.mapped_parameters
        for key in keys:
            del smp.locations[key]
            del smp.original_files[key]
            del smp.image_stacks[key]
            address=smp.aggregate_address[key]
            smp.data=np.delete(smp.data,np.s_[address[0]:address[1]:1],2)
        smp.axes_manager=AxesManager(smp._get_undefined_axes_list())

    def print_keys(self):
        smp=self.mapped_parameters
        print smp.locations.keys()
