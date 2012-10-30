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
    """A consistent class for slices.
    
    """
    def __init__(self, sli, axis):
        self._orig_sli = sli
        self._axis = axis
        self._orig_offset = axis.offset
        #Init params
        self.start = 0
        self.stop = axis.size
        self.step = 1
        self.offset = axis.offset
        #Init flags
        self.is_array = False
        #Update params
        self._update()

    def gen(self, to_values=False):
        #Generate actual slices
        if to_values:
            i2v = self._axis.index2value
            if self.is_array:
                return [i2v(el)
                            if not isinstance(el, float)
                            else el
                        for el in self._orig_sli]
            else:
                return slice(i2v(self.start) 
                                if not isinstance(self.start, float) 
                                else self.start,
                             i2v(self.stop) 
                                if not isinstance(self.stop, float) 
                                else self.stop,
                             i2v(self.step) 
                                if not isinstance(self.step, float) 
                                else self.step)
        else:
            v2i = self._axis.value2index
            if self.is_array:
                return [v2i(el)
                            if isinstance(el, float)
                            else el
                        for el in self._orig_sli]
            else:
                return slice(v2i(self.start) 
                                if isinstance(self.start, float) 
                                else self.start,
                             v2i(self.stop) 
                                if isinstance(self.stop, float)  
                                else self.stop,
                             v2i(self.step) 
                                if isinstance(self.step, float) 
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
                    self.stop = self._axis.index2value(self.start) + 1
                else:
                    self.stop = self.start + 1
                self.step = None
        if isinstance(self.start, float):
            self.offset = self.start
        else:
            self.offset = self._axis.index2value(self.start)
