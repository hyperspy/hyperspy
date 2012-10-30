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

import numpy as np

class Slice:
    """A consistent class for slices"""
    def __init__(self, sli, axe):
        self._orig_sli = sli
        self._axe = axe
        self._orig_offset = axe.offset
        #Init params
        self.start = 0
        self.stop = axe.size
        self.step = 1
        self.offset = axe.offset
        #Init flags
        self.is_array = False
        #Update params
        self._update()

    def gen(self, to_values=False):
        #Generate actual slices
        if to_values:
            i2v = self._axe.index2value
            if self.is_array:
                return [i2v(el) if not isinstance(el, float) else el
                        for el in self._orig_sli]
            else:
                return slice(i2v(self.start) if not 
                                        isinstance(self.start, float) 
                             else self.start,
                             i2v(self.stop) if not 
                                        isinstance(self.stop, float) 
                             else self.stop,
                             i2v(self.step) if not 
                                        isinstance(self.step, float) 
                             else self.step)
        else:
            v2i = self._axe.value2index
            if self.is_array:
                return [v2i(el) if isinstance(el, float) else el
                        for el in self._orig_sli]
            else:
                return slice(v2i(self.start) if 
                                        isinstance(self.start, float) 
                             else self.start,
                             v2i(self.stop) if 
                                            isinstance(self.stop, float) 
                             else self.stop,
                             v2i(self.step) if 
                                            isinstance(self.step, float) 
                             else self.step)

    def _update(self):
        """Don't do too much (no value2index or index2value):
        Original values will be needed once interpolation is 
        implemented.
        
        """
        sli = self._orig_sli
        try:
            #Suppose "sli" is a slice instance...
            if sli.start is not None:
                self.start = sli.start
            if sli.stop is not None:
                self.stop = sli.stop
            if sli.step is not None and abs(sli.step) >=1 :
                self.step = int(round(sli.step)) #No non-int values for now 
            else:
                self.step = None #idem
        except (AttributeError, TypeError):
            try:
                #Suppose "sli" is a tuple-like instance
                self.start = sli[0]
                self.stop = None
                self.step = None
                self.is_array = True
            except TypeError:
                #Suppose "sli" is an int/float
                self.start = self._orig_sli
                if isinstance(self.start, float):
                    self.stop = self._axe.index2value(self.start) + 1
                else:
                    self.stop = self.start + 1
                self.step = None
        if isinstance(self.start, float):
            self.offset = self.start
        else:
            self.offset = self._axe.index2value(self.start)

class SliceSignal:
    def __init__(self, slices, isNavigation=None, XYZ_ordering=True):
        try:
            len(slices)
        except TypeError:
            slices = (slices,)
        self._orig_slices = slices

        self.nav_indexes = None
        self.signal_indexes = None 

        self._signal = None
        
        self.idx = None
        self._Slices = None
        self.slices = None
        self.offset = None
        
        self.XYZ_ordering = XYZ_ordering

        if isNavigation is not None:
            if isNavigation:
                self.has_nav = True
                self.has_signal = False
            else:
                self.has_nav = False
                self.has_signal = True
        else:
            self.has_nav = True
        self.has_signal = True
    
    def __call__(self, signal):
        self._signal = signal.deepcopy()
        self._process_slices()
        return self.apply()

    def _process_slices(self):
        """Change axes order and get processed slices from the Slice
         class.
        
        """
        
        self.nav_indexes =  np.array([el.index_in_array for el in
                    self._signal.axes_manager.navigation_axes])
        self.signal_indexes =  np.array([el.index_in_array for el in
                    self._signal.axes_manager.signal_axes])

        if self.XYZ_ordering:
            cut = slice(None, None, -1)
        else:
            cut = slice(None, None, 1)

        nav_idx = self.nav_indexes[cut]
        signal_idx = self.signal_indexes[cut]
        
        if list(nav_idx):
            self.index = nav_idx
        elif list(signal_idx):
            self.index = signal_idx
        else:
            self.index = np.append(nav_idx, signal_idx)

        if self.has_nav and not self.has_signal:
            self.idx =  nav_idx
        elif not self.has_nav and self.has_signal:
            self.idx =  signal_idx
        else:
            self.idx =  self.index

        slices = np.array([slice(None,)]*len(self._signal.axes_manager.axes))
        slices[self.idx] = self._fill(self._orig_slices, len(self.idx))
        axes = [self._signal.axes_manager.axes[i] for i in self.index]

        self._Slices = [Slice(sli, axe) for sli, axe in zip(slices,axes)]
        self.slices = [sli.gen() for sli in self._Slices]
        self.offset = [sli.offset for sli in self._Slices]

    def _fill(self, slices, num):
        return np.append(slices, [slice(None,)]*max(0,num-len(slices)))

    def _clean_axes(self):
        #Update axe sizes and offsets
        for axe, slice_len, j in zip(self._signal.axes_manager.axes,
                                     self._signal.data.shape, self.index):
            axe.size = slice_len
            axe.offset = self.offset[j]
        self._signal.squeeze()
                     

    def update_slices(self, slices):
        self.__init__(slices)

    def apply(self):
        self._signal.data = self._signal.data[self.slices]
        self._clean_axes()
        return self._signal
