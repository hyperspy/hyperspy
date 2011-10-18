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

from traits.etsconfig.api import ETSConfig
import matplotlib
#if matplotlib.get_backend() != 'WXAgg':
#    ETSConfig.toolkit ='null'
#else:
#    ETSConfig.toolkit ='wx'

matplotlib.rcParams['image.cmap'] = 'gray'
from hyperspy import Release
from hyperspy import components

from hyperspy.io import load
from hyperspy.defaults_parser import defaults
from hyperspy.misc import utils
from hyperspy import tests

import os
from time import strftime

__version__ = Release.version

# start up the log file
try:
    _ip = get_ipython()
    filename = os.path.join(os.getcwd(), 'hyperspy_log.py')
    new = not os.path.exists(filename)
    _ip.logger.logstart(logfname=filename,logmode='append')
    if new:
        _ip.logger.log_write("""#!/usr/bin/env python \n""")
    _ip.logger.log_write("# ============================\n")
    _ip.logger.log_write("# %s \n" % strftime('%Y-%m-%d'))
    _ip.logger.log_write("# %s \n"% strftime('%H:%M'))
    _ip.logger.log_write("# ============================\n" )

except RuntimeError:
    print " Already logging to "+_ip.logger.logfname
    
except NameError:
    # It is not running in the ipython console or the ipython version does not 
    # provide the get_ipython function.
    pass

def get_configuration_directory_path():
    import hyperspy.misc.config_dir
    print(hyperspy.misc.config_dir.config_path)

def start_gui():
    if ETSConfig.toolkit != 'null':
        import gui.main_window
        gui.main_window.MainWindow().configure_traits()
        
def create_model(signal, *args, **kwargs):
    from hyperspy.signals.eels import EELSSpectrum
    from hyperspy.models.eelsmodel import EELSModel
    from hyperspy.model import Model
    if isinstance(signal, EELSSpectrum):
        return EELSModel(signal, *args, **kwargs)
    else:
        return Model(signal, *args, **kwargs)
        
# Install the tutorial in the home folder if the file is available
#tutorial_file = os.path.join(data_path, 'tutorial.tar.gz')
#tutorial_directory = os.path.expanduser('~/hyperspy_tutorial')
#if os.path.isfile(tutorial_file) is True:
#    if os.path.isdir(tutorial_directory) is False:
#        messages.alert(
#        "Installing the tutorial in: %s" % tutorial_directory) 
#        tar = tarfile.open(tutorial_file)
#        os.mkdir(tutorial_directory)
#        tar.extractall(tutorial_directory)
        
#if os.path.isdir(gos_path) is False and os.path.isfile(eels_gos_files) is True:
#    messages.information(
#    "Installing the EELS GOS files in: %s" % gos_path) 
#    tar = tarfile.open(eels_gos_files)
#    os.mkdir(gos_path)
#    tar.extractall(gos_path)
#if os.path.isdir(gos_path):
#    defaults_dict['eels_eels_gos_filess_path'] = gos_path
