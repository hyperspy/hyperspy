# -*- coding: utf-8 -*-
# Copyright 2007-2023 The exSpy developers
#
# This file is part of exSpy.
#
# exSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# exSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with exSpy. If not, see <https://www.gnu.org/licenses/#GPL>.

import configparser
import logging
import os
from pathlib import Path

import traits.api as t
from hyperspy.ui_registry import add_gui_method


config_path = Path("~/.exspy").expanduser()
config_path.mkdir(parents=True, exist_ok=True)
defaults_file = Path(config_path, 'exspyrc')

_logger = logging.getLogger(__name__)


def guess_gos_path():
    if os.name in ["nt", "dos"]:
        # If DM is installed, use the GOS tables from the default
        # installation
        # location in windows
        program_files = os.environ['PROGRAMFILES']
        gos = 'Gatan\\DigitalMicrograph\\EELS Reference Data\\H-S GOS Tables'
        gos_path = Path(program_files, gos)

        # Else, use the default location in the .hyperspy forlder
        if not gos_path.is_dir() and 'PROGRAMFILES(X86)' in os.environ:
            program_files = os.environ['PROGRAMFILES(X86)']
            gos_path = Path(program_files, gos)
            if not gos_path.is_dir():
                gos_path = Path(config_path, 'EELS_GOS')
    else:
        gos_path = Path(config_path, 'EELS_GOS')
    return gos_path


class EELSConfig(t.HasTraits):
    eels_gos_files_path = t.Directory(
        guess_gos_path(),
        label='Hartree-Slater GOS directory',
        desc='The GOS files are used to create the EELS edge components')


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
    'EELS': EELSConfig(),
    'EDS': EDSConfig(),
}


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
            config_dict[name] = value
        traited_class.trait_set(True, **config_dict)


defaults_file_exists = defaults_file.is_file()

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


@add_gui_method(toolkey="exspy.Preferences")
class Preferences(t.HasTraits):
    EELS = t.Instance(EELSConfig)
    EDS = t.Instance(EDSConfig)

    def save(self):
        config = configparser.ConfigParser(allow_no_value=True)
        template2config(template, config)
        config.write(open(defaults_file, 'w'))


preferences = Preferences(
    EELS=template['EELS'],
    EDS=template['EDS'],
)
