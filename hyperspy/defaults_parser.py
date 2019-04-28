# -*- coding: utf-8 -*-
# Copyright 2007-2016 The HyperSpy developers
#
# This file is part of  HyperSpy.
#
#  HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  HyperSpy.  If not, see <http://www.gnu.org/licenses/>.


import os.path
import configparser
import logging

import traits.api as t

from hyperspy.misc.config_dir import config_path, os_name, data_path
from hyperspy.misc.ipython_tools import turn_logging_on, turn_logging_off
from hyperspy.ui_registry import add_gui_method

defaults_file = os.path.join(config_path, 'hyperspyrc')
eels_gos_files = os.path.join(data_path, 'EELS_GOS.tar.gz')

_logger = logging.getLogger(__name__)


def guess_gos_path():
    if os_name == 'windows':
        # If DM is installed, use the GOS tables from the default
        # installation
        # location in windows
        program_files = os.environ['PROGRAMFILES']
        gos = 'Gatan\\DigitalMicrograph\\EELS Reference Data\\H-S GOS Tables'
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
    with open(defaults_file) as f:
        if 'Not really' in f.readline():
                # It is the old config file
            defaults_file_exists = False
        else:
            defaults_file_exists = True
    if not defaults_file_exists:
        # It actually exists, but is an obsoleted unsupported version of it
        # so we delete it.
        _logger.info('Removing obsoleted config file')
        os.remove(defaults_file)
else:
    defaults_file_exists = False

# Defaults template definition starts#####################################
# This "section" is all that has to be modified to add or remove sections and
# options from the defaults

# Due to https://github.com/enthought/traitsui/issues/23 the desc text as
# displayed in the tooltip get "Specifies" prepended.


class GeneralConfig(t.HasTraits):
    logger_on = t.CBool(
        False,
        label='Automatic logging (requires IPython)',
        desc='If enabled, HyperSpy will store a log in the current directory '
        'of all the commands typed')

    show_progressbar = t.CBool(
        True,
        label='Show progress bar',
        desc='If enabled, show a progress bar when available')

    dtb_expand_structures = t.CBool(
        True,
        label='Expand structures in DictionaryTreeBrowser',
        desc='If enabled, when printing DictionaryTreeBrowser (e.g. '
             'metadata), long lists and tuples will be expanded and any '
             'dictionaries in them will be printed similar to '
             'DictionaryTreeBrowser, but with double lines')
    logging_level = t.Enum(['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', ],
                           desc='the log level of all hyperspy modules.')
    parallel = t.CBool(
        True,
        desc='Use parallel threads for computations by default.'
    )

    nb_progressbar = t.CBool(
        True,
        desc='Attempt to use ipywidgets progressbar'
    )

    def _logger_on_changed(self, old, new):
        if new is True:
            turn_logging_on()
        else:
            turn_logging_off()


class EELSConfig(t.HasTraits):
    eels_gos_files_path = t.Directory(
        guess_gos_path(),
        label='GOS directory',
        desc='The GOS files are required to create the EELS edge components')


class GUIs(t.HasTraits):
    enable_ipywidgets_gui = t.CBool(
        True,
        desc="Display ipywidgets in the Jupyter Notebook. "
        "Requires installing hyperspy_gui_ipywidgets.")
    enable_traitsui_gui = t.CBool(
        True,
        desc="Display traitsui user interface elements. "
        "Requires installing hyperspy_gui_traitsui.")
    warn_if_guis_are_missing = t.CBool(
        True,
        desc="Display warnings, if hyperspy_gui_ipywidgets or hyperspy_gui_traitsui are missing.")


class PlotConfig(t.HasTraits):
    dims_024_increase = t.Str('right',
                              label='Navigate right'
                              )
    dims_024_decrease = t.Str('left',
                              label='Navigate left',
                              )
    dims_135_increase = t.Str('down',
                              label='Navigate down',
                              )
    dims_135_decrease = t.Str('up',
                              label='Navigate up',
                              )
    modifier_dims_01 = t.Enum(['ctrl', 'alt', 'shift', 'ctrl+alt', 'ctrl+shift', 'alt+shift',
                               'ctrl+alt+shift'], label='Modifier key for 1st and 2nd dimensions')  # 0 elem is default
    modifier_dims_23 = t.Enum(['shift', 'alt', 'ctrl', 'ctrl+alt', 'ctrl+shift', 'alt+shift',
                               'ctrl+alt+shift'], label='Modifier key for 3rd and 4th dimensions')  # 0 elem is default
    modifier_dims_45 = t.Enum(['alt', 'ctrl', 'shift', 'ctrl+alt', 'ctrl+shift', 'alt+shift',
                               'ctrl+alt+shift'], label='Modifier key for 5th and 6th dimensions')  # 0 elem is default


class EDSConfig(t.HasTraits):
    eds_mn_ka = t.CFloat(130.,
                         label='Energy resolution at Mn Ka (eV)',
                         desc='default value for FWHM of the Mn Ka peak in eV,'
                         'This value is used as a first approximation'
                         'of the energy resolution of the detector.')
    eds_tilt_stage = t.CFloat(
        0.,
        label='Stage tilt',
        desc='default value for the stage tilt in degree.')
    eds_detector_azimuth = t.CFloat(
        0.,
        label='Azimuth angle',
        desc='default value for the azimuth angle in degree. If the azimuth'
        ' is zero, the detector is perpendicular to the tilt axis.')
    eds_detector_elevation = t.CFloat(
        35.,
        label='Elevation angle',
        desc='default value for the elevation angle in degree.')


template = {
    'General': GeneralConfig(),
    'GUIs': GUIs(),
    'EELS': EELSConfig(),
    'EDS': EDSConfig(),
    'Plot': PlotConfig(),
}

# Set the enums defaults
template['General'].logging_level = 'WARNING'

# Defaults template definition ends ######################################


def template2config(template, config):
    for section, traited_class in template.items():
        config.add_section(section)
        for key, item in traited_class.trait_get().items():
            config.set(section, key, str(item))


def config2template(template, config):
    for section, traited_class in template.items():
        config_dict = {}
        for name, value in config.items(section):
            if value == 'True':
                value = True
            elif value == 'False':
                value = False
            if name == 'fine_structure_smoothing':
                value = float(value)
            config_dict[name] = value
        traited_class.trait_set(True, **config_dict)


def dictionary_from_template(template):
    dictionary = {}
    for section, traited_class in template.items():
        dictionary[section] = traited_class.get()
    return dictionary


config = configparser.ConfigParser(allow_no_value=True)
template2config(template, config)
rewrite = False
if defaults_file_exists:
    # Parse the config file. It only copy to config the options that are
    # already defined. If the file contains any option that was not already
    # define the config file is rewritten because it is obsolate

    config2 = configparser.ConfigParser(allow_no_value=True)
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
    _logger.info('Writing the config file')
    with open(defaults_file, "w") as df:
        config.write(df)

# Use the traited classes to cast the content of the ConfigParser
config2template(template, config)


@add_gui_method(toolkey="Preferences")
class Preferences(t.HasTraits):
    EELS = t.Instance(EELSConfig)
    EDS = t.Instance(EDSConfig)
    General = t.Instance(GeneralConfig)
    GUIs = t.Instance(GUIs)
    Plot = t.Instance(PlotConfig)

    def save(self):
        config = configparser.ConfigParser(allow_no_value=True)
        template2config(template, config)
        config.write(open(defaults_file, 'w'))


preferences = Preferences(
    EELS=template['EELS'],
    EDS=template['EDS'],
    General=template['General'],
    GUIs=template['GUIs'],
    Plot=template['Plot'],
)

if preferences.General.logger_on:
    turn_logging_on(verbose=0)


def file_version(fname):
    with open(fname, 'r') as f:
        for l in f.readlines():
            if '__version__' in l:
                return l[l.find('=') + 1:].strip()
    return '0'
