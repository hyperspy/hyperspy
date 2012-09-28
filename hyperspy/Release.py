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

import os

name = 'hyperspy'

# The commit following to a release must update the version number
# to the version number of the release followed by "+dev", e.g.
# if the version of the last release is 0.4.1 the version of the
# next development version afterwards must be 0.4.1+dev.
# When running setup.py the "+dev" string will be replaced (if possible)
# by the output of "git describe" if git is available or the git 
# hash if .git is present.
version = "0.5.1"
description = "Hyperspectral data analysis toolbox"
license = 'GPL v3'

authors = {
    'F_DLP' : (u'Francisco de la Peña',
               'hyperspy-devel@googlegroups.com'),
    'S_M'    : ('Stefano Mazzucco',
                'hyperspy-devel@googlegroups.com'),
    'M_S'    : ('Michael Sarahan',
                'hyperspy-devel@googlegroups.com'),
    'all'    : ('The Hyperspy developers',
                'hyperspy-devel@googlegroups.com'),}

url = 'http://hyperspy.org'

download_url = 'http://www.hyperspy.org'
documentation_url = 'http://hyperspy.org/hyperspy-doc/current/index.html'

platforms = ['Linux','Mac OSX','Windows XP/2000/NT','Windows 95/98/ME']

keywords = ['EDX',
            'EELS',
            'EFTEM',
            'EMSA',
            'FEI',
            'ICA',
            'PCA',
            'PES',
            'STEM',
            'TEM',
            'curve fitting',
            'data analysis',
            'dm3',
            'electron energy loss spectroscopy',
            'electron microscopy',
            'emi',
            'energy dispersive x-rays',
            'hyperspectral',
            'hyperspectrum',
            'hyperspy',
            'machine learning',
            'microscopy',
            'model',
            'msa',
            'numpy',
            'python',
            'quantification',
            'scipy',
            'ser',
            'spectroscopy',
            'spectrum image']

info = u"""
    H y p e r s p y
    Version %s
    
    http://www.hyperspy.org
    
    """ % version.replace('_', ' ')
