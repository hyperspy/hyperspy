# -*- coding: utf-8 -*-
# Copyright 2007-2021 The HyperSpy developers
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

from pathlib import Path


def _resolve_dir():
    """Returns the absolute path to this file's directory."""
    return Path(__file__).resolve().parent


def load_1D_EDS_SEM_spectrum():
    """
    Load an EDS-SEM spectrum

    Notes
    -----
    - Sample: EDS-TM002 provided by BAM (www.webshop.bam.de)
    - SEM Microscope: Nvision40 Carl Zeiss
    - EDS Detector: X-max 80 from Oxford Instrument
    - Signal is loaded "read-only" to ensure data access regardless of 
      install location
    """
    from hyperspy.io import load

    file_path = _resolve_dir().joinpath(
        "eds", "example_signals", "1D_EDS_SEM_Spectrum.hspy"
    )

    return load(file_path, mode="r")


def load_1D_EDS_TEM_spectrum():
    """
    Load an EDS-TEM spectrum

    Notes
    -----
    - Sample: FePt bimetallic nanoparticles
    - SEM Microscope: Tecnai Osiris 200 kV D658 AnalyticalTwin
    - EDS Detector: Super-X 4 detectors Brucker
    - Signal is loaded "read-only" to ensure data access regardless of 
      install location
    """
    from hyperspy.io import load

    file_path = _resolve_dir().joinpath(
        "eds", "example_signals", "1D_EDS_TEM_Spectrum.hspy"
    )

    return load(file_path, mode="r")


def load_object_hologram():
    """
    Load an object hologram image

    Notes
    -----
    Sample: Fe needle with YOx nanoparticle inclusions. See reference for more 
    details

        Migunov, V. et al. Model-independent measurement of the charge density
        distribution along an Fe atom probe needle using off-axis electron 
        holography without mean inner potential effects. J. Appl. Phys. 117, 
        134301 (2015). https://doi.org/10.1063/1.4916609

    TEM: FEI Titan G2 60-300 HOLO

        Boothroyd, C. et al. FEI Titan G2 60-300 HOLO. Journal of large-scale 
        research facilities JLSRF 2, 44 (2016).
        https://doi.org/10.17815/jlsrf-2-70

    Signal is loaded "read-only" to ensure data access regardless of 
    install location
    """
    from hyperspy.io import load

    file_path = _resolve_dir().joinpath(
        "holography", "example_signals", "01_holo_Vbp_130V_0V_bin2_crop.hdf5"
    )

    return load(file_path, signal_type="hologram", mode="r")


def load_reference_hologram():
    """
    Load a reference hologram image

    Notes
    -----
    Sample: Fe needle with YOx nanoparticle inclusions. See reference for more 
    details

        Migunov, V. et al. Model-independent measurement of the charge density
        distribution along an Fe atom probe needle using off-axis electron 
        holography without mean inner potential effects. J. Appl. Phys. 117, 
        134301 (2015). https://doi.org/10.1063/1.4916609

    TEM: FEI Titan G2 60-300 HOLO

        Boothroyd, C. et al. FEI Titan G2 60-300 HOLO. Journal of large-scale 
        research facilities JLSRF 2, 44 (2016).
        https://doi.org/10.17815/jlsrf-2-70

    Signal is loaded "read-only" to ensure data access regardless of 
    install location
    """
    from hyperspy.io import load

    file_path = _resolve_dir().joinpath(
        "holography", "example_signals", "00_ref_Vbp_130V_0V_bin2_crop.hdf5"
    )

    return load(file_path, signal_type="hologram", mode="r")

