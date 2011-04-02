# -*- coding: utf-8 -*-
# Copyright © 2007 Francisco Javier de la Peña
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

# The following code was commented because it does not work with
# ETS 3.6.0 , to investigate...
from enthought.etsconfig.api import ETSConfig
import matplotlib
if matplotlib.get_backend() == 'Qt4Agg':
    ETSConfig.toolkit ='qt4'
elif matplotlib.get_backend() == 'WXAgg':
    ETSConfig.toolkit ='wx'
else:
    ETSConfig.toolkit ='null'

import silib.Release
import silib.components as components

from silib.spectrum import Spectrum
from silib.image import Image
from silib.experiments import Experiments
from silib.model import Model
from silib.file_io import load
from silib.edges_db import edges_dict
from silib.microscope import microscope
from silib.defaults_parser import defaults
from silib import utils

__version__ = silib.Release.version
__revision__ = silib.Release.revision

print(silib.Release.info)

def get_configuration_directory_path():
    print(silib.config_dir.config_path)

# The gui can produce a crash for certain toolkits and certain versions of
# python-traits. There, until we find out which is the right configuration for
# each platform, its availability will depend on the user luck:
try:   
    import silib.gui.main_window
    def gui():
        silib.gui.main_window.MainWindow().configure_traits()
except:
    pass
