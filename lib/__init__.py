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

# There is an incompatibility between matplolib 1.0.1 and enthought 0.3.6 
# because matplotlib uses QString 1 api and ETS QString 2. Therefore, the 
# traits toolkit is set to 'null' for QT4. 
from enthought.etsconfig.api import ETSConfig
import matplotlib
if matplotlib.get_backend() == 'Qt4Agg':
    ETSConfig.toolkit ='null'
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
#from microscope import microscope
from defaults_parser import defaults
import utils

__version__ = Release.version
__revision__ = Release.revision

__all__=["Experiments", "Signal", "Model", "load", "edges_dict", "microscope", 
         "defaults", "utils"]

print(Release.info)

def get_configuration_directory_path():
    print(eelslab.config_dir.config_path)

if ETSConfig.toolkit != 'null':
    import eelslab.gui.main_window
    def gui():
        eelslab.gui.main_window.MainWindow().configure_traits()
