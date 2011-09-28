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


import os.path
import tarfile
import ConfigParser

import enthought.traits.api as t
import enthought.traits.ui.api as tui

from hyperspy.misc.config_dir import config_path, os_name, data_path
from hyperspy import messages
from hyperspy.misc.utils import DictionaryBrowser

defaults_file = os.path.join(config_path, 'hyperspyrc')
eels_gos_files = os.path.join(data_path, 'EELS_GOS.tar.gz')

def guess_gos_path():
    if os_name == 'windows':
        # If DM is installed, use the GOS tables from the default installation
        # location in windows
        program_files = os.environ['PROGRAMFILES']
        gos = 'Gatan\DigitalMicrograph\EELS Reference Data\H-S GOS Tables'
        gos_path = os.path.join(program_files, gos)
        
        # Else, use the default location in the .hyperspy forlder
        if os.path.isdir(gos_path) is False and \
            'PROGRAMFILES(X86)' in os.environ:
            program_files = os.environ['PROGRAMFILES(X86)']
            gos_path = os.path.join(program_files, gos)
            if os.path.isdir(gos_path) is False:
                    gos_path = os.path.join(config_path, 'EELS_GOS')
    else:
        gos_path = os.path.join(config_path, 'EELS_GOS')
    return gos_path
    

if os.path.isfile(defaults_file):
    # Remove config file if obsolated
    f = open(defaults_file)
    if 'Not really' in f.readline():
        # It is the old config file
        f.close()
        messages.information('Removing obsoleted config file')
        os.remove(defaults_file)
        defaults_file_exists = False
    else:
       defaults_file_exists = True 
else:
    defaults_file_exists = False

# Defaults template definition starts###########################################
# This "section" is all that has to be modified to add or remove sections and
# options from the defaults   
class GeneralConfig(t.HasTraits):
    default_file_format = t.CStr('hdf5')
    plot_on_load = t.CBool(False)
    interactive = t.CBool(False)
    
class ModelConfig(t.HasTraits):
    default_fitter = t.CStr('leastsq')
    
class EELSConfig(t.HasTraits):
    eels_gos_files_path = t.CStr(guess_gos_path())
    fs_emax = t.CFloat(30)
    synchronize_cl_with_ll = t.CBool(False)
    preedge_safe_window_width = t.CFloat(2)
    min_distance_between_edges_for_fine_structure = t.CFloat(0)

template = {
    'General' : GeneralConfig(),
    'Model' : ModelConfig(),
    'EELS' : EELSConfig(),}

# Defaults template definition ends ############################################       



def template2config(template, config):
    for section, traited_class in template.iteritems():
        config.add_section(section)
        for key, item in traited_class.get().iteritems():
            config.set(section, key, str(item))
            
def config2template(template, config):
    defaults_dictionary = {}
    for section, traited_class in template.iteritems():
        config_dict = {}
        for name, value in config.items(section):
            config_dict[name] = value
        traited_class.set(True, **config_dict)
        
def dictionary_from_template(template):
    dictionary = {}
    for section, traited_class in template.iteritems():
        dictionary[section] = traited_class.get()
    return dictionary

config = ConfigParser.SafeConfigParser(allow_no_value = True)
template2config(template, config)
rewrite = False
if defaults_file_exists:
    # Parse the config file. It only copy to config the options that are 
    # already defined. If the file contains any option that was not already 
    # define the config file is rewritten because it is obsolate
    
    config2 = ConfigParser.SafeConfigParser(allow_no_value = True)
    config2.read(defaults_file)
    for section in config2.sections():
        if config.has_section(section):
            for option in config2.options(section):
                if config.has_option(section, option):
                    config.set(section, option, config2.get(section, option))
                else:
                    rewrite = True
        else:
            rewrite = True
                
if not defaults_file_exists or rewrite is True:
    messages.information('Writing the config file')
    config.write(open(defaults_file, 'w'))
        
# Use the traited classes to cast the content of the ConfigParser
config2template(template, config)
    
defaults = DictionaryBrowser(dictionary_from_template(template))



