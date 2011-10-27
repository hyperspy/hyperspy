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


name = 'hyperspy'

version = '0.3.preview-1'

description = "Hyperspectral data analysis toolbox"

license = 'GPL v3'

authors = {
    'F_DLP' : (u'Francisco de la Peña', 'hyperspy-devel@googlegroups.com'),
    'S_M'    : ('Stefano Mazzucco', 'hyperspy-devel@googlegroups.com'),
    'M_S'    : ('Michael Sarahan', 'hyperspy-devel@googlegroups.com'),
    'all'    : ('The Hyperspy developers', 'hyperspy-devel@googlegroups.com'),}

url = 'http://www.hyperspy.org'

download_url = 'http://www.hyperspy.org'

platforms = ['Linux','Mac OSX','Windows XP/2000/NT','Windows 95/98/ME']

keywords = [
    'hyperspy', 'hyperspectral', 'data analysis', 'electron', 'spectroscopy', 
    'python', 'numpy', 'scipy', 'microscopy', 'TEM', 'STEM', 'quantification',
    'EDX', 'EELS', 'EFTEM', 'PES', 'PCA', 'ICA', 'curve fitting', 'model']

info = u"""
    H y p e r s p y
    Version %s
    
    Copyright (C) 2007-2010 Francisco de la Peña
    Copyright (C) 2010-2011 F. de la Peña, S. Mazzucco, M. Sarahan
    
    http://www.hyperspy.org
    
    """ % version.replace('_', ' ')
