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

import traits.api as t
import traitsui.api as tui

from hyperspy.misc.config_dir import config_path, os_name, data_path
from hyperspy import messages
from hyperspy.misc.utils import DictionaryBrowser
from hyperspy.misc.interactive_ns import turn_logging_on, turn_logging_off
from hyperspy.io import default_write_ext

defaults_file = os.path.join(config_path, 'hyperspyrc')
eels_gos_files = os.path.join(data_path, 'EELS_GOS.tar.gz')

def guess_gos_path():
    if os_name == 'windows':
        # If DM is installed, use the GOS tables from the default 
        # installation
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
    default_file_format = t.Enum('hdf5', 'rpl',
        desc = 'Using the hdf5 format is highly reccomended because is the '
               'only one fully supported. The Ripple (rpl) format it is useful '
               'to export data to other software that do not support hdf5')
    default_export_format = t.Enum(*default_write_ext,
        desc = 'Using the hdf5 format is highly reccomended because is the '
               'only one fully supported. The Ripple (rpl) format it is useful '
               'to export data to other software that do not support hdf5')
    plot_on_load = t.CBool(False,
        desc = 'If enabled, the object will be plot automatically on loading')
    interactive = t.CBool(True,
        desc = 'If enabled, Hyperspy will prompt the user when optios are '
               'available, otherwise it will use the default values if possible')
    logger_on = t.CBool(False,
        label = 'Automatic logging',
        desc = 'If enabled, Hyperspy will store a log in the current directory '
               'of all the commands typed')
    
    def _logger_on_changed(self, old, new):
        if new is True:
            turn_logging_on()
        else:
            turn_logging_off()

    
class ModelConfig(t.HasTraits):
    default_fitter = t.Enum('leastsq', 'mpfit',
        desc = 'Choose leastsq if no bounding is required. '
               'Otherwise choose mpfit')

class MachineLearningConfig(t.HasTraits):
    export_factors_default_file_format = t.Enum(*default_write_ext)
    export_loadings_default_file_format = t.Enum(*default_write_ext)
    multiple_files = t.Bool(True,
        label = 'Export to multiple files',
        desc = 'If enabled, on exporting the PCA or ICA results one file'
               'per factor and loading will be created. Otherwise only two files'
               'will contain the factors and loadings')
    same_window = t.Bool(True,
        label = 'Plot components in the same window',
        desc = 'If enabled the principal and independent components will all'
               ' be plotted in the same window')
    
class EELSConfig(t.HasTraits):
    eels_gos_files_path = t.Directory(guess_gos_path(),
        label = 'GOS directory',
        desc = 'The GOS files are required to create the EELS edge components')
    fine_structure_width = t.CFloat(30,
        label = 'Fine structure lenght',
        desc = 'The default lenght of the fine structure from the edge onset')
    fine_structure_active = t.CBool(False,
        label = 'Enable fine structure',
        desc = "If enabled, the regions of the EELS spectrum defined as fine "
               "structure will be fitted with a spline. Please note that it "
               "enabling this feature only makes sense when the model is "
               "convolved to account for multiple scattering")
    knots_factor = t.Range(0., 1., value = 0.3,
        label = 'Fine structure smoothing factor',
        desc = 'The lower the value the smoother the fine structure spline fit')
    synchronize_cl_with_ll = t.CBool(False)
    preedge_safe_window_width = t.CFloat(2,
        label = 'Pre-onset region (in eV)',
        desc = 'Some functions needs to define the regions between two '
               'ionisation edges. Due to limited energy resolution or chemical '
               'shift, the region is limited on its higher energy side by '
               'the next ionisation edge onset minus an offset defined by this '
               'parameters')
    min_distance_between_edges_for_fine_structure = t.CFloat(0,
        label = 'Minimum distance between edges',
        desc = 'When automatically setting the fine structure energy regions, '
               'the fine structure of an EELS edge component is automatically '
               'disable if the next ionisation edge onset distance to the '
               'higher energy side of the fine structure region is lower that '
               'the value of this parameter')
    view = tui.View(
        tui.Group(
        'synchronize_cl_with_ll',
        label = 'General'),
        tui.Group(
            'eels_gos_files_path',
            'preedge_safe_window_width',
            tui.Group(
                'fine_structure_width', 'fine_structure_active', 'knots_factor', 
                'min_distance_between_edges_for_fine_structure',
                label = 'Fine structure'),
            label = 'Model')
            )            
            

template = {
    'General' : GeneralConfig(),
    'Model' : ModelConfig(),
    'EELS' : EELSConfig(),
    'MachineLearning' : MachineLearningConfig(),}

# Set the enums defaults
template['MachineLearning'].export_factors_default_file_format = 'rpl'
template['MachineLearning'].export_loadings_default_file_format = 'rpl'
template['General'].default_export_format = 'rpl'

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
            if value == 'True':
                value = True
            elif value == 'False':
                value = False
            if name == 'knots_factor':
                value = float(value)
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
    
#preferences = DictionaryBrowser(dictionary_from_template(template))

class PreferencesHandler(tui.Handler):
    def close(self, info, is_ok):
        # Removes the span selector from the plot
        info.object.save()
        return True


class Preferences(t.HasTraits):
    EELS = t.Instance(EELSConfig)
    Model = t.Instance(ModelConfig)
    General = t.Instance(GeneralConfig)
    MachineLearning = t.Instance(MachineLearningConfig)
    view = tui.View(
        tui.Group(tui.Item('General', style='custom', show_label=False, ),
            label = 'General'),
        tui.Group(tui.Item('Model', style='custom', show_label=False, ),
            label = 'Model'),
        tui.Group(tui.Item('EELS', style='custom', show_label=False, ),
            label = 'EELS'),
        tui.Group(tui.Item('MachineLearning', style='custom',
            show_label=False,),
            label = 'Machine Learning'),
        title = 'Preferences',
        handler = PreferencesHandler,)
    
    def gui(self):
        self.edit_traits()
        
    def save(self):
        config = ConfigParser.SafeConfigParser(allow_no_value = True)
        template2config(template, config)
        config.write(open(defaults_file, 'w'))

    
preferences = Preferences(
            EELS = template['EELS'],
            General = template['General'],
            Model = template['Model'],
            MachineLearning = template['MachineLearning'])
            
if preferences.General.logger_on:
    turn_logging_on(verbose = 0)
