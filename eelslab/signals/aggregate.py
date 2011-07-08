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

from eelslab.signal import Signal

class Aggregate(Signal):
    def __init__(self):
        super(Aggregate,self).__init__()


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
