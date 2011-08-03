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

# Proposal for when we start using configparser
#[Plotting]
#plot_on_load 0
#
#[EELSModel]
#GOS_dir None 
#fs_emax 30
#fs_state 1
#knots_factor 0.3
#min_distance_between_edges_for_fine_structure 0.
#preedge_safe_window_width 2

import os.path
import tarfile

from eelslab.misc.config_dir import config_path, os_name, data_path
from eelslab import messages

defaults_file = os.path.join(config_path, 'eelslabrc')
bool_keys = ['fs_state', 'synchronize_cl_with_ll', 'plot_on_load']
str_keys = ['GOS_dir', 'fitter', 'file_format']
float_keys = ['fs_emax', 'preedge_safe_window_width', 
'knots_factor', 'min_distance_between_edges_for_fine_structure' ]

Gos_file = os.path.join(data_path, 'GOS.tar.gz')
f = open(defaults_file, 'r')
defaults_dict = {}
for line in f:
    if line[0] != "#":
        line_list = line.split()
        if line_list != []:
            if line_list[0] in float_keys:
                key, value = line_list
                defaults_dict[key] = float(value)
            elif line_list[0] in str_keys:
                key, value = line_list
                defaults_dict[key] = value
            elif line_list[0] in bool_keys:
                key, value = line_list
                defaults_dict[key] = bool(int(value))
f.close()

if defaults_dict['GOS_dir'] == 'None':
    if os_name == 'windows':
        # If DM is installed, use the GOS tables from the default installation
        # location in windows
        program_files = os.environ['PROGRAMFILES']
        gos = 'Gatan\DigitalMicrograph\EELS Reference Data\H-S GOS Tables'
        gos_path = os.path.join(program_files, gos)
        
        # Else, use the default location in the .eelslab forlder
        if os.path.isdir(gos_path) is False and 'PROGRAMFILES(X86)' in os.environ:
            program_files = os.environ['PROGRAMFILES(X86)']
            gos_path = os.path.join(program_files, gos)
            if os.path.isdir(gos_path) is False:
                    gos_path = os.path.join(config_path, 'GOS')
    else:
        gos_path = os.path.join(config_path, 'GOS')
    
    if os.path.isdir(gos_path) is False and os.path.isfile(Gos_file) is True:
        messages.alert(
        "Installing the GOS files in: %s" % gos_path) 
        tar = tarfile.open(Gos_file)
        os.mkdir(gos_path)
        tar.extractall(gos_path)
    if os.path.isdir(gos_path):
        defaults_dict['GOS_dir'] = gos_path
        
class Defaults:
    pass

defaults = Defaults()
defaults.__dict__ = defaults_dict


# Install the tutorial in the home folder if the file is available
tutorial_file = os.path.join(data_path, 'tutorial.tar.gz')
tutorial_directory = os.path.expanduser('~/eelslab_tutorial')
if os.path.isfile(tutorial_file) is True:
    if os.path.isdir(tutorial_directory) is False:
        messages.alert(
        "Installing the tutorial in: %s" % tutorial_directory) 
        tar = tarfile.open(tutorial_file)
        os.mkdir(tutorial_directory)
        tar.extractall(tutorial_directory)
