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
from hyperspy.misc.utils import DictionaryTreeBrowser

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

# Attolight system specific parameters
# ------------------------------------
# Store the system specific parameters in a dictionary for
# different Attolight SEMs around the world, with their identifier.

attolight_systems = {
    'cambridge_uk_attolight': {
        'channels': 1024,
        'cal_factor_x_axis': 131072,
        'metadata_file_name': 'MicroscopeStatus.txt',
        'grating_corrfactors': {
            150: 2.73E-04,
            600: 6.693659836087227e-05,
        }
    }
}


def _get_metadata(filename, md_file_name, attolight_acquisition_system):
    """Import the metadata from the MicroscopeStatus.txt file.
    Returns binning, nx, ny, FOV, grating and central_wavelength.
    Parameters
    ----------
    filename : str
        The absolute folder path where the md_file_name exists.
    """
    path = os.path.join(filename, md_file_name)
    with open(path, encoding='windows-1252') as status:
        for line in status:
            if 'Horizontal Binning:' in line:
                binning = int(line[line.find(':') + 1:-1])  # binning = binning status
            if 'Resolution_X' in line:
                nx = int(line[line.find(':') + 1:-7])
                # nx = pixel in x-direction
            if 'Resolution_Y' in line:
                ny = int(line[line.find(':') + 1:-7])
                # ny = pixel in y-direction
            if 'Real Magnification' in line:
                FOV = float(line[line.find(':') + 1:-1])
            if 'Grating - Groove Density:' in line:
                grating = float(line[line.find(':') + 1:-6])
            if 'Central wavelength:' in line:
                central_wavelength_nm = float(line[line.find(':') + 1:-4])
            if 'Channels:' in line:
                total_channels = int(line[line.find(':') + 1:-1])
            if 'Signal Amplification:' in line:
                amplification = int(line[line.find(':x') + 2:-1])
            if 'Readout Rate (horizontal pixel shift):' in line:
                readout_rate = int(line[line.find(':') + 1:-4])

            if 'Exposure Time:' in line:
                exposure_time_ccd_s = float(line[line.find(':') + 1:-3])
            if 'HYP Dwelltime:' in line:
                dwell_time_scan_s = float(line[line.find(':') + 1:-4]) / 1000
            if 'Beam Energy:' in line:
                beam_acc_voltage_kv = float(line[line.find(':') + 1:-3]) / 1000
            if 'Gun Lens:' in line:
                gun_lens_amps = float(line[line.find(':') + 1:-3])
            if 'Objective Lens:' in line:
                obj_lens_amps = float(line[line.find(':') + 1:-3])
            if 'Aperture:' in line:
                aperture_um = float(line[line.find(':') + 1:-4])
            if 'Aperture Chamber Pressure:' in line:
                chamber_pressure_torr = float(line[line.find(':') + 1:-6])
            if 'Real Magnification:' in line:
                real_magnification = float(line[line.find(':') + 1:-3])

    # Correct channels to the actual value, accounting for binning. Get
    # channels on the detector used (if channels not defined, then assume
    # its 1024)
    try:
        total_channels
    except:
        total_channels = attolight_systems[attolight_acquisition_system]['channels']
    channels = total_channels // binning

    # Return metadata
    return binning, nx, ny, FOV, grating, central_wavelength_nm, channels, amplification, readout_rate, \
           exposure_time_ccd_s, dwell_time_scan_s, beam_acc_voltage_kv, gun_lens_amps, obj_lens_amps, aperture_um, \
           chamber_pressure_torr, real_magnification


def _store_metadata(dict_tree, hypcard_folder, md_file_name,
                    attolight_acquisition_system):
    """
    Store metadata in the DictionaryTreeBrowser metadata. Stores
    binning, nx, ny, FOV, grating and central_wavelength and others.
    """

    # Get metadata
    binning, nx, ny, FOV, grating, central_wavelength_nm, channels, amplification, readout_rate, \
    exposure_time_ccd_s, dwell_time_scan_s, beam_acc_voltage_kv, gun_lens_amps, obj_lens_amps, aperture_um, \
    chamber_pressure_torr, real_magnification = _get_metadata(
        hypcard_folder, md_file_name, attolight_acquisition_system)

    # Store metadata
    dict_tree.set_item("Acquisition_instrument.Spectrometer.grating",
                                grating)
    dict_tree.set_item("Acquisition_instrument.Spectrometer.central_wavelength_nm",
                                central_wavelength_nm)
    dict_tree.set_item("Acquisition_instrument.SEM.resolution_x",
                                nx)
    dict_tree.set_item("Acquisition_instrument.SEM.resolution_y",
                                ny)
    dict_tree.set_item("Acquisition_instrument.SEM.FOV", FOV)
    dict_tree.set_item("Acquisition_instrument.CCD.binning",
                                binning)
    dict_tree.set_item("Acquisition_instrument.CCD.channels",
                                channels)
    dict_tree.set_item("Acquisition_instrument.acquisition_system",
                                attolight_acquisition_system)
    dict_tree.set_item("Acquisition_instrument.CCD.amplification", amplification)
    dict_tree.set_item("Acquisition_instrument.CCD.readout_rate", readout_rate)
    dict_tree.set_item("Acquisition_instrument.CCD.exposure_time_s", exposure_time_ccd_s)
    dict_tree.set_item("Acquisition_instrument.SEM.dwell_time_scan_s", dwell_time_scan_s)
    dict_tree.set_item("Acquisition_instrument.SEM.beam_acc_voltage_kv", beam_acc_voltage_kv)
    dict_tree.set_item("Acquisition_instrument.SEM.gun_lens_amps", gun_lens_amps)
    dict_tree.set_item("Acquisition_instrument.SEM.obj_lens_amps", obj_lens_amps)
    dict_tree.set_item("Acquisition_instrument.SEM.aperture_um", aperture_um)
    dict_tree.set_item("Acquisition_instrument.SEM.chamber_pressure_torr", chamber_pressure_torr)
    dict_tree.set_item("Acquisition_instrument.SEM.real_magnification", real_magnification)
    dict_tree.set_item("General.folder_path", hypcard_folder)

    return


def _create_signal_axis_in_wavelength(data, metadata):
    """
    Based on the Attolight software export function. Need to be automatised.
    Two calibrated sets show the trend:
    #Centre at 650 nm:
        spec_start= 377.436, spec_end = 925.122
    #Centre at 750:
        spec_start= 478.2, spec_end = 1024.2472

    TO DO: ALLOW FOR NON-LINEAR-AXIS

    Returns
    ----------
    dictionary_axis: []
        Dictionary with the parameters of a linear axis.
    """
    # Get relevant parameters from metadata
    central_wavelength = metadata.Acquisition_instrument.Spectrometer.central_wavelength_nm

    # Estimate start and end wavelengths
    spectra_offset_array = [central_wavelength - 273, central_wavelength + 273]

    # Apply calibration
    size = data.shape[0]
    name = 'Wavelength'
    scale = (spectra_offset_array[1] - spectra_offset_array[0]) \
            / size
    offset = spectra_offset_array[0]
    units = '$nm$'

    axis_dict = {'name': name,
                 'units': units,
                 'navigate': False,
                 'size': size,
                 'scale': scale,
                 'offset': offset}

    return axis_dict


def _create_navigation_axis(data, metadata):
    # Edit the navigation axes

    # Get relevant parameters from metadata and acquisition_systems
    # parameters
    acquisition_system = metadata.Acquisition_instrument.acquisition_system
    cal_factor_x_axis \
        = attolight_systems[acquisition_system]['cal_factor_x_axis']
    FOV = metadata.Acquisition_instrument.SEM.FOV
    nx = metadata.Acquisition_instrument.SEM.resolution_x

    # Get the calibrated scanning axis scale from the acquisition_systems
    # dictionary
    calax = cal_factor_x_axis / (FOV * nx)

    size = data.shape[1]
    name = ''
    scale = calax * 1000
    # changes micrometer to nm, value for the size of 1 pixel
    units = '$nm$'

    axis_dict = {'units': units,
                 'navigate': True,
                 'size': size,
                 'scale': scale, }

    return axis_dict


def _save_background_metadata(metadata, hypcard_folder, background_file_name='Background*.txt'):
    """
    Based on the Attolight background savefunction.
    If background is found in the folder, it saves background as in the metadata.
    """
    # Get the absolute path
    path = os.path.join(hypcard_folder, background_file_name)

    # Try to load the file, if it exists.
    try:
        # Find the exact filename, using the * wildcard
        path = glob.glob(path)[0]
        # Load the file as a numpy array
        bkg = np.loadtxt(path)
        # The bkg file contains [wavelength, background]
        metadata.set_item("Signal.background", bkg)
        return
    except:
        return


def file_reader(filename, attolight_acquisition_system='cambridge_uk_attolight',
                *args, **kwds):
    """Loads data into CLSEMSpectrum lumispy object.
    Reads the HYPCard.bin file, containing the hyperspectral CL data.
    Reads the metadata files contained in the same folder as HYPCard.bin.
        - MicroscopeStatus.txt to load all the metadata values
        - Background file contained in the same folder
        - SEM image (taken simultaneously) with the hyperspectral map.

    Parameters
    ----------
    filename : str, None
        The HYPCard.bin filepath for the file to be loaded, created by
        the AttoLight software.
        If None, a pop-up window will be loaded.
    attolight_acquisition_system : str
        Specify which acquisition system the HYPCard was taken with, from the
        attolight_systems dictionary file. By default, it assumes it is
        the Cambridge Attolight SEM system.
    lazy : bool
        If True the file will be opened lazily, i.e. without actually reading
        the data from the disk until required. Allows datasets much larger than
        available memory to be loaded.

    Returns
    -------
    s : Signal
        A CLSEMSpectrum lumispy object containing the loaded data with the metadata,
        the background (if available) and the respective SEM image.

    NOTE
    ----

    The lumispy package must be installed for these files to run.
    https://github.com/LumiSpy/lumispy

    """

    # Get folder name
    hypcard_folder = os.path.split(os.path.abspath(filename))[0]

    # Get folder name (which is the experiment name)
    name = os.path.basename(hypcard_folder)

    if attolight_acquisition_system == 'cambridge_attolight' and len(name) > 37:
        # CAUTION: Specifically delimeted by Attolight default naming system
            name = name[:-37]

    # Create metadata dictionary
    meta = DictionaryTreeBrowser({
        'General': {'title': name},
        "Signal": {'signal_type': 'CL_SEM_Spectrum',
                   'background': None},
    })

    # Import metadata
    metadata_file_name \
        = attolight_systems[attolight_acquisition_system]['metadata_file_name']

    binning, nx, ny, FOV, grating, central_wavelength_nm, channels, amplification, readout_rate, \
    exposure_time_ccd_s, dwell_time_scan_s, beam_acc_voltage_kv, gun_lens_amps, obj_lens_amps, aperture_um, \
    chamber_pressure_torr, real_magnification = \
        _get_metadata(hypcard_folder, metadata_file_name, attolight_acquisition_system)

    # Add all parameters as metadata
    _store_metadata(meta, hypcard_folder, metadata_file_name, attolight_acquisition_system)

    # Load data
    with open(filename, 'rb') as f:
        data = np.fromfile(f, dtype=[('bar', '<i4')], count=channels * nx * ny)
        array = np.reshape(data, [channels, nx, ny], order='F')

    # Swap x-y axes to get the right xy orientation
    data = np.swapaxes(array, 1, 2)

    # Save background in metadata
    _save_background_metadata(meta, hypcard_folder,)

    # Create axes
    signal_axis = _create_signal_axis_in_wavelength(data, meta)
    navigation_axis = _create_navigation_axis(data, meta)

    axes = [navigation_axis, navigation_axis, signal_axis]

    # Create signal dictionary
    dictionary = {
        'data': data.squeeze(),
        'axes': axes,
        'metadata': meta.as_dictionary(),
        "package": 'lumispy',
    }
    return [dictionary, ]
