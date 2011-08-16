# -*- coding: utf-8 -*-
# Copyright © 2007 Francisco Javier de la Peña
#
# This file is part of Hyperspy.
#
# Hyperspy is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# Hyperspy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Hyperspy; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  
# USA


from  scipy.interpolate import splev

from hyperspy.component import Component

class Spline(Component):

    def __init__(self, tck):
        Component.__init__(self, ('c', 'dump'))
        self.name = 'Spline'
        self.t,self.c.value,self.k = tck
        self.dump.free = False
        
    def function(self, x):
        return splev(x, (self.t, self.c.value, 3))
