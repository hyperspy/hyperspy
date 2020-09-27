# -*- coding: utf-8 -*-
# Copyright 2007-2020 The HyperSpy developers
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

#  for more information on the RPL/RAW format, see
#  http://www.nist.gov/lispix/
#  and
#  http://www.nist.gov/lispix/doc/image-file-formats/raw-file-format.htm

import glob
import os
import logging
import numpy as np

_logger = logging.getLogger(__name__)

# Plugin characteristics
# ----------------------
format_name = 'HYPCard_AttolightSEM'
description = 'Reads Cathodoluminescence data from the Attolight SEM system'
full_support = False
# Recognised file extension
file_extensions = ['bin', 'BIN']
default_extension = 0
# Writing capabilities
writes = False


# ----------------------

def _get_original_metadata(folder, md_file_name, channels):
    """
    :param folder: experiment folder path
    :param md_file_name: name of the metadata txt file
    :param channels: number of channels
    :return: dictionary

    """
    path = os.path.join(folder, md_file_name)

    original_metadata = {}
    with open(path, encoding='windows-1252') as f:
        for line in f:
            try:
                key, value = line.split(":")
                original_metadata[key] = value
            except ValueError:
                # not enough values to unpack
                pass

    # Correct channels to the actual value, accounting for binning. Get
    # channels on the detector used (if channels not defined, then assume
    # its 1024)

    try:
        original_metadata['Total channels']
    except KeyError:
        original_metadata['Total channels'] = channels

    original_metadata['Channels'] = original_metadata['Total channels'] // int(original_metadata['Horizontal Binning'])

    return original_metadata


def _parse_relevant_metadata_values(filename, md_file_name, md_subcategory, channels):
    original_metadata = _get_original_metadata(filename, md_file_name, channels)

    from pint import UnitRegistry, UndefinedUnitError

    ureg = UnitRegistry()
    q_reg = ureg.Quantity

    keys_of_interest = {
        'Spectrometer': [
            'Grating - Groove Density',
            'Central wavelength',
        ],
        'CCD': [
            'Horizontal Binning',
            'Channels',
            'Signal Amplification',
            'Readout Rate (horizontal pixel shift)',
            'Exposure Time',
        ],
        'SEM': [
            'Resolution_X',
            'Resolution_Y',
            'Real Magnification',
            'HYP Dwelltime',
            'Beam Energy',
            'Gun Lens',
            'Objective Lens',
            'Aperture',
            'Aperture Chamber Pressure',
        ]}

    metadata = {}
    for key, value in original_metadata.items():
        if key in keys_of_interest[md_subcategory]:
            try:
                value = q_reg(value).magnitude
            except UndefinedUnitError:
                value = value

            key = key.replace(" ", "_")
            metadata[key] = value

    return metadata


def _store_metadata(meta_dict, hypcard_folder, md_file_name,
                    attolight_acquisition_system, channels):
    """
    Store metadata in the DictionaryTreeBrowser metadata. Stores
    binning, nx, ny, FOV, grating and central_wavelength and others.
    """

    # Get metadata
    metadata = {}
    metadata['Spectrometer'] = _parse_relevant_metadata_values(hypcard_folder, md_file_name,
                                                               'Spectrometer', channels)
    metadata['CCD'] = _parse_relevant_metadata_values(hypcard_folder, md_file_name,
                                                      'CCD', channels)
    metadata['SEM'] = _parse_relevant_metadata_values(hypcard_folder, md_file_name,
                                                      'SEM', channels)

    # Store metadata
    for group in metadata:
        g = {}
        for key, value in metadata[group].items():
            d = {key: value}
            g.update(d)
        meta_dict["Acquisition_instrument"].update(g)

    meta_dict["General"]["folder_path"] = hypcard_folder
    meta_dict["Acquisition_instrument"]["acquisition_system"] = attolight_acquisition_system

    return


def _create_signal_axis(data):
    """
    Create the signal axis in pixel units

    Returns
    ----------
    dictionary_axis: []
        Dictionary with the parameters of a linear axis.
    """
    # Get relevant parameters from metadata

    size = data.shape[0]
    name = 'Signal axis'
    scale = 1
    offset = 0
    units = 'px'

    axis_dict = {'name': name,
                 'units': units,
                 'navigate': False,
                 'size': size,
                 'scale': scale,
                 'offset': offset}

    return axis_dict


def _create_navigation_axis(data, metadata, cal_factor_x_axis, system_name):
    # Edit the navigation axes
    if cal_factor_x_axis is not None:
        # Get relevant parameters from metadata and acquisition_systems
        # parameters
        fov = metadata["Acquisition_instrument"]["SEM"]["Real_Magnification"]
        nx = metadata["Acquisition_instrument"]["SEM"]["Resolution_X"]

        # Get the calibrated scanning axis scale from the acquisition_systems
        # dictionary
        calax = cal_factor_x_axis / (fov * nx)
        scale = calax * 1000
        # changes micrometer to nm, value for the size of 1 pixel
        units = 'nm'

        # Add in metadata calibration tag
        metadata["Signal"]["calibration_file"] = system_name

    else:
        scale = 1
        units = ''

    size = data.shape[1]
    axis_dict = {'units': units,
                 'navigate': True,
                 'size': size,
                 'scale': scale, }

    return axis_dict


def _save_background_metadata(metadata, hypcard_folder, background_file_name):
    """
    Based on the Attolight background savefunction.
    If background is found in the folder, it saves background as in the metadata.
    """
    # Get the absolute path
    background_file_name = os.path.basename(background_file_name)
    path = os.path.join(hypcard_folder, background_file_name)

    # Try to load the file, if it exists.
    try:
        # Find the exact filename, using the * wildcard
        path = glob.glob(path)[0]
        # Load the file as a numpy array
        bkg = np.loadtxt(path)
        # The bkg file contains [wavelength, background]
        metadata["Signal"]["background"] = bkg
        return
    except FileNotFoundError:
        return


def _get_calibration_dictionary(calibration_path):
    from pint import UnitRegistry, UndefinedUnitError
    ureg = UnitRegistry()
    q_reg = ureg.Quantity

    calibration_dict = {}
    with open(calibration_path) as f:
        for line in f:
            try:
                key, value = line.split(":")
                try:
                    value = q_reg(value).magnitude
                except UndefinedUnitError:
                    pass
                calibration_dict[key] = value
            except ValueError:
                # not enough values to unpack
                pass
    return calibration_dict


def file_reader(filename, attolight_calibration_file=None, background_file=None,):
    """
    Loads data into CLSEMSpectrum lumispy object.
    Reads the HYPCard.bin file, containing the hyperspectral CL data.
    Reads the metadata files contained in the same folder as HYPCard.bin.
        - MicroscopeStatus.txt to load all the metadata values
        - Background file contained in the same folder (or imported with `background_file` param)
        - SEM image (taken simultaneously) with the hyperspectral map.
    You can also import a microscope-specific calibration .txt file as input to calibrate the
    navigation axis.

    Parameters
    ----------
    :param filename: str
        The HYPCard.bin filepath for the file to be loaded, created by the AttoLight software.
    :param attolight_calibration_file: str
        The calibration.txt filepath for the file to be calibrated.
        If not given, assume 1024 channels and no navigation axis calibration.
    :param background_file: str
        An external background .txt file,
        in case none is found in the same folder as the HYPCard.bin file.
    Returns
    -------
    s: dictionary
        A dictionary which will be loaded by the hyperspy loader.
    """

    # Get folder name
    hypcard_folder = os.path.dirname(filename)

    # Get folder name (which is the experiment name)
    name = os.path.basename(hypcard_folder)

    if len(name) > 37:
        # CAUTION: Specifically delimited by Attolight default naming system
        name = name[:-37]

    # Load calibration dictionary if possible
    if attolight_calibration_file is not None:
        calibration_dict = _get_calibration_dictionary(attolight_calibration_file)
    else:
        calibration_dict = {
            'system_name': None,
            'channels': 1024,
            'metadata_file_name': 'MicroscopeStatus.txt',
            'cal_factor_x_axis': None, }

    # Create metadata dictionary
    meta = {
        "General": {"title": name},
        "Signal": {"signal_type": "CL_SEM", "background": None, "calibration_file": None},
        "Acquisition_instrument": {},
    }

    # Add all parameters as metadata
    _store_metadata(meta, hypcard_folder,
                    calibration_dict['metadata_file_name'],
                    calibration_dict['system_name'],
                    calibration_dict['channels'])

    # Load data
    channels = meta["Acquisition_instrument"]["CCD"]["Channels"]
    nx = meta["Acquisition_instrument"]["SEM"]["Resolution_X"]
    ny = meta["Acquisition_instrument"]["SEM"]["Resolution_Y"]

    with open(filename, 'rb') as f:
        data = np.fromfile(f, dtype=[('bar', '<i4')], count=channels * nx * ny)
        array = np.reshape(data, [channels, nx, ny], order='F')

    # Swap x-y axes to get the right xy orientation
    data = np.swapaxes(array, 1, 2)

    # Save background in metadata if it exists or if it is provided
    if background_file is None:
        background_file = 'Background*.txt'
    _save_background_metadata(meta, hypcard_folder, background_file)

    # Create axes
    signal_axis = _create_signal_axis(data)
    navigation_axis = _create_navigation_axis(data, meta,
                                              calibration_dict['cal_factor_x_axis'],
                                              calibration_dict['system_name'])

    axes = [navigation_axis, navigation_axis, signal_axis]

    # Create signal dictionary
    dictionary = {
        'data': data.squeeze(),
        'axes': axes,
        'metadata': meta,
        "package": 'lumispy',
    }
    return [dictionary, ]
