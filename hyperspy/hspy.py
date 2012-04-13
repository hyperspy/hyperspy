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
from hyperspy.defaults_parser import preferences
from hyperspy.misc import utils
from hyperspy import tests
from hyperspy.signals.spectrum import Spectrum
from hyperspy.signals.image import Image



__version__ = Release.version

# start up the log file

def get_configuration_directory_path():
    import hyperspy.misc.config_dir
    print(hyperspy.misc.config_dir.config_path)

def start_gui():
    if ETSConfig.toolkit != 'null':
        import gui.main_window
        gui.main_window.MainWindow().configure_traits()
        
def create_model(signal, *args, **kwargs):
    """Create a model object
    
    Any extra argument is passes to the Model constructor.
    
    Parameters
    ----------
    
    signal: A signal class
    
    Returns
    -------
    
    A Model class
    
    """
    
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
