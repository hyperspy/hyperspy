# -*- coding: utf-8 -*-
# Copyright 2007-2022 The HyperSpy developers
#
# This file is part of HyperSpy.
#
# HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HyperSpy. If not, see <https://www.gnu.org/licenses/#GPL>.

name = 'hyperspy'

# The commit following to a release must update the version number
# to the version number of the release followed by ".dev", e.g.
# if the version of the last release is 0.4.1 the version of the
# next development version afterwards must be 0.4.1.dev.
# When running setup.py the ".dev" string will be replaced (if possible)
# by the output of "git describe" if git is available or the git
# hash if .git is present.
version = "1.7.6"
__version__ = version
description = "Multidimensional data analysis toolbox"
license = 'GPL v3'

authors = {
    'all': ('The HyperSpy developers',), }

url = 'https://hyperspy.org'

download_url = 'https://www.hyperspy.org'
documentation_url = 'https://hyperspy.org/hyperspy-doc/current/index.html'

PROJECT_URLS = {
    'Bug Tracker': 'https://github.com/hyperspy/hyperspy/issues',
    'Documentation': 'https://hyperspy.org/hyperspy-doc/current/index.html',
    'Source Code': 'https://github.com/hyperspy/hyperspy',
    'Support' : 'https://gitter.im/hyperspy/hyperspy'
}

platforms = ['Linux', 'Mac OSX', 'Windows XP/2000/NT', 'Windows 95/98/ME']

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
            'multidimensional',
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

info = """
    H y p e r S p y
    Version %s

    https://www.hyperspy.org

    """ % version.replace('_', ' ')
