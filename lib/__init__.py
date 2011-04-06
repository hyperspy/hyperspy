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


import Release
import components

#from spectrum import Spectrum
#from image import Image
from experiments import Experiments
from signal import Signal
from model import Model
from file_io import load
from edges_db import edges_dict
from microscope import microscope
from defaults_parser import defaults
import utils

__version__ = Release.version
__revision__ = Release.revision

__all__=["Experiments", "Signal", "Model", "load", "edges_dict", "microscope", 
         "defaults", "utils"]

print(Release.info)

def get_configuration_directory_path():
    print(eelslab.config_dir.config_path)

# The gui can produce a crash for certain toolkits and certain versions of
# python-traits. There, until we find out which is the right configuration for
# each platform, its availability will depend on the user luck:
try:   
    import eelslab.gui.main_window
    def gui():
        eelslab.gui.main_window.MainWindow().configure_traits()
except:
    pass