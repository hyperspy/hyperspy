# -*- coding: utf-8 -*-
# Copyright 2007-2015 The HyperSpy developers
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

# The details of the format were taken from
# http://www.biochem.mpg.de/doc_tom/TOM_Release_2008/IOfun/tom_mrcread.html
# and http://ami.scripps.edu/software/mrctools/mrc_specification.php

import os
import logging
import numpy as np
from hyperspy.misc.array_tools import sarray2dict
import traits.api as t
from hyperspy.misc.elements import atomic_number2name

_logger = logging.getLogger(__name__)

# Plugin characteristics
# ----------------------
format_name = 'EDAX TEAM'
description = 'Reader for EDS maps and spectra saved by the EDAX TEAM' \
              'software: \n' \
              'An SPD file contains map data. The spectral information is ' \
              '\nheld in an SPC file with the same name, while the spatial ' \
              '\ncalibration is held in a related IPR file. If an SPD file ' \
              '\nis loaded, the result will be a Hyperspy EDSSpectrum map, ' \
              '\nand the calibration will be loaded from appropriate SPC ' \
              '\nand IPR files (if available). \n' \
              'If an SPC file is loaded, the result will be a single \n' \
              'EDSSpectrum with no other files needed for calibration.'
full_support = False
# Recognised file extension
file_extensions = ['spd', 'SPD', 'spc', 'SPC']
default_extension = 0
# Writing capabilities
writes = False

spd_extensions = ('spd', 'SPD', 'Spd')
spc_extensions = ('spc', 'SPC', 'Spc')


def get_spd_dtype_list(endianess='<'):
    """
    Get the data type list for an SPD map.
    Further information about the file format is available `here
    <https://github.com/hyperspy/hyperspy/
    files/29505/SpcMap-spd.file.format.pdf>`__.

    Table of header tags:
     - tag: 16 byte char array;    *File ID tag ("MAPSPECTRA_DATA")*
     - version: 4 byte long;       *File version*
     - nSpectra: 4 byte long;      *Number of spectra in file*
     - nPoints: 4 byte long;       *Number of map pixels in X direction*
     - nLines: 4 byte long;        *Number of map pixels in Y direction*
     - nChannels: 4 byte long;     *Number of channels per spectrum*
     - countBytes: 4 byte long;    *Number of count bytes per channel*
     - dataOffset: 4 byte long;    *File offset in bytes for data start*
     - nFrames: 4 byte long;       *Number of frames in live spectrum mapping*
     - fName: 120 byte char array; *File name of electron image acquired during mapping*


    Parameters
    ----------
    endianess : byte-order used to read the data

    Returns
    -------
    dtype_list : list
        List of the data tags and data types that will be used by numpy to
        read an SPD file header.
    """
    end = endianess
    dtype_list = \
        [
            ('tag', '16i1'),
            ('version', end + 'i4'),
            ('nSpectra', end + 'i4'),
            ('nPoints', end + 'i4'),
            ('nLines', end + 'u4'),
            ('nChannels', end + 'u4'),
            ('countBytes', end + 'u4'),
            ('dataOffset', end + 'u4'),
            ('nFrames', end + 'u4'),
            ('fName', end + '120i1'),
            ('filler', end + 'V900'),
        ]
    return dtype_list


def __get_spc_header(f, endianess, load_all_spc):
    """
    Get the header of an spc file, checking for the file version as necessary

    Parameters
    ----------
    f : file
        A file object for the .spc file to be read (i.e. file should be
        already opened with ``open()``)
    endianess : char
        Byte-order of data to read
    load_all_spc : bool
        Switch to control if all of the .spc header is read, or just the parts
        relevant to HyperSpy

    Returns
    -------
    spc_header : np.ndarray
        Array containing the binary information read from the .spc file
    """
    version = np.fromfile(f,
                          dtype=[('version', '{}f4'.format(endianess))],
                          count=1)
    version = round(np.asscalar(version)[0], 2)  # convert to scalar
    f.seek(0)

    spc_header = np.fromfile(f,
                             dtype=get_spc_dtype_list(
                                 load_all=load_all_spc,
                                 endianess=endianess,
                                 version=version),
                             count=1)

    _logger.debug(' .spc version is {}'.format(version))

    return spc_header


def get_spc_dtype_list(load_all=False, endianess='<', version=0.61):
    """
    Get the data type list for an SPC spectrum.
    Further information about the file format is available `here
    <https://github.com/hyperspy/hyperspy/files/29506/SPECTRUM-V70.pdf>`__.

    Parameters
    ----------
    load_all : bool
        Switch to control if all the data is loaded, or if just the
        important pieces of the signal will be read (speeds up loading time)
    endianess : char
        byte-order used to read the data
    version : float
        version of spc file to read (only 0.61 and 0.70 have been tested)
        Default is 0.61 to be as backwards-compatible as possible, but the
        file version can be read from the file anyway, so this parameter
        should always be set programmatically

    Table of header tags:
        - fVersion: 4 byte float; *File format Version*
        - aVersion: 4 byte float; *Application Version*
        - fileName: 8 array of 1 byte char; *File name w/o '.spc' extension (OLD)*
        - collectDateYear: 2 byte short; *Year the spectrum was collected*
        - collectDateDay: 1 byte char; *Day the spectrum was collected*
        - collectDateMon: 1 byte char; *Month the spectrum was collected*
        - collectTimeMin: 1 byte char; *Minute the spectrum was collected*
        - collectTimeHour: 1 byte char; *Hour the spectrum was collected*
        - collectTimeHund: 1 byte char; *Hundredth second the spectrum was collected*
        - collectTimeSec: 1 byte char; *Second the spectrum was collected*
        - fileSize: 4 byte long; *Size of spectrum file in bytes*
        - dataStart: 4 byte long; *Start of spectrum data in bytes offset from 0 of file*
        - numPts: 2 byte short; *Number of spectrum pts*
        - intersectingDist: 2 byte short; *Intersecting distance * 100 (mm)*
        - workingDist: 2 byte short; *Working distance * 100*
        - scaleSetting: 2 byte short; *Scale setting distance * 100*

        - filler1: 24 byte;

        - spectrumLabel: 256 array of 1 byte char; *Type label for spectrum, 0-39=material type, 40-255=sample*
        - imageFilename: 8 array of 1 byte char; *Parent Image filename*
        - spotX: 2 byte short; *Spot X  in parent image file*
        - spotY: 2 byte short; *Spot Y in parent image file*
        - imageADC: 2 byte short; *Image ADC value 0-4095*
        - discrValues: 5 array of 4 byte long; *Analyzer Discriminator Values*
        - discrEnabled: 5 array of 1 byte unsigned char; *Discriminator Flags (0=Disabled,1=Enabled)*
        - pileupProcessed: 1 byte char; *Pileup Processed Flag (0=No PU,1=Static PU, 2=Dynamic PU,...)*
        - fpgaVersion: 4 byte long; *Firmware Version.*
        - pileupProcVersion: 4 byte long; *Pileup Processing Software Version*
        - NB5000CFG: 4 byte long; *Defines Hitachi NB5000 Dual Stage Cfg 0=None, 10=Eucentric Crossx,11= Eucentric Surface 12= Side Entry - Side 13 = Side Entry - Top*

        - filler2: 12 byte;

        - evPerChan: 4 byte long; *EV/channel*
        - ADCTimeConstant: 2 byte short; *ADC Time constant*
        - analysisType: 2 byte short; *Preset mode 1=clock, 2=count, 3=none, 4=live, 5=resume*
        - preset: 4 byte float; *Analysis Time Preset value*
        - maxp: 4 byte long; *Maximum counts of the spectrum*
        - maxPeakCh: 4 byte long; *Max peak channel number*
        - xRayTubeZ: 2 byte short; *XRF*
        - filterZ: 2 byte short; *XRF*
        - current: 4 byte float; *XRF*
        - sampleCond: 2 byte short; *XRF Air= 0, Vacuum= 1, Helium= 2*
        - sampleType: 2 byte short; *Bulk or thin*
        - xrayCollimator: 2 byte unsigned short; *0=None, 1=Installed*
        - xrayCapilaryType: 2 byte unsigned short; *0=Mono, 1=Poly*
        - xrayCapilarySize: 2 byte unsigned short; *Range : 20 – 5000 Microns*
        - xrayFilterThickness: 2 byte unsigned short; *Range : 0 – 10000 Microns*
        - spectrumSmoothed: 2 byte unsigned short; *1= Spectrum Smoothed, Else 0*
        - detector_Size_SiLi: 2 byte unsigned short; *Eagle Detector 0=30mm, 1=80mm*
        - spectrumReCalib: 2 byte unsigned short; *1= Peaks Recalibrated, Else 0*
        - eagleSystem: 2 byte unsigned short; *0=None, 2=Eagle2, 3=Eagle3, 4-Xscope*
        - sumPeakRemoved: 2 byte unsigned short; *1= Sum Peaks Removed, Else 0*
        - edaxSoftwareType: 2 byte unsigned short; *1= Team Spectrum, Else 0*

        - filler3: 6 byte;

        - escapePeakRemoved: 2 byte unsigned short; *1=Escape Peak Was Removed, Else 0*
        - analyzerType: 4 byte unsigned long; *Hardware type 1=EDI1, 2=EDI2, 3=DPP2, 31=DPP-FR, 32=DPP-FR2, 4=DPP3, 5= APOLLO XLT/XLS/DPP-4 (EDPP)*
        - startEnergy: 4 byte float; *Starting energy of spectrum*
        - endEnergy: 4 byte float; *Ending energy of spectrum*
        - liveTime: 4 byte float; *LiveTime*
        - tilt: 4 byte float; *Tilt angle*
        - takeoff: 4 byte float; *Take off angle*
        - beamCurFact: 4 byte float; *Beam current factor*
        - detReso: 4 byte float; *Detector resolution*
        - detectType: 4 byte unsigned long; *Detector Type: 1=Std-BE, 2=UTW, 3=Super UTW, 4=ECON 3/4 Open, 5=ECON 3/4 Closed, 6=ECON 5/6 Open, 7=ECON 5/6 Closed, 8=TEMECON; Add + 10 For Sapphire SiLi Detectors, (11-18), which started shipping in 1996. 30 = APOLLO 10 SDD, 31=APOLLO XV, 32 = APOLLO 10+, 40 = APOLLO 40 SDD ,50 = APOLLO-X, 51=APOLLO-XP, 52 = APOLLO-XL, 53 = APOLLO XL-XRF, 60 =APOLLO-XLT-LS, 61 =APOLLO-XLT-NW, 62 =APOLLO-XLT-SUTW*
        - parThick: 4 byte float; *Parlodion light shield thickness*
        - alThick: 4 byte float; *Aluminum light shield thickness*
        - beWinThick: 4 byte float; *Be window thickness*
        - auThick: 4 byte float; *Gold light shield thickness*
        - siDead: 4 byte float; *Si dead layer thickness*
        - siLive: 4 byte float; *Si live layer thickness*
        - xrayInc: 4 byte float; *X-ray incidence angle*
        - azimuth: 4 byte float; *Azimuth angle of detector*
        - elevation: 4 byte float; *Elevation angle of detector*
        - bCoeff: 4 byte float; *K-line B coefficient*
        - cCoeff: 4 byte float; *K-line C coefficient*
        - tailMax: 4 byte float; *Tail function maximum channel*
        - tailHeight: 4 byte float; *Tail height adjustment percentage*
        - kV: 4 byte float; *Acc voltage*
        - apThick: 4 byte float; *Ap window thickness*
        - xTilt: 4 byte float; *x tilt angle for mDX*
        - yTilt: 4 byte float; *y tilt angle for mDX*
        - yagStatus: 4 byte unsigned long; *0 = N/A, 1 = YAG OUT, 2 = YAG IN*

        - filler4: 24 byte;

        - rawDataType: 2 byte unsigned short; *TEM or SEM data*
        - totalBkgdCount: 4 byte float; *Accumulated background counts*
        - totalSpectralCount: 4 byte unsigned long; *Accumulated spectrum counts*
        - avginputCount: 4 byte float; *Average spectral counts*
        - stdDevInputCount: 4 byte float; *Standard deviation of spectral counts*
        - peakToBack: 2 byte unsigned short; *Peak to background setting. 0 = off, 1 = low, 2 = medium, 3 = high, 4 = user selected*
        - peakToBackValue: 4 byte float; *Peak to back value*

        - filler5: 38 byte;

        - numElem: 2 byte short; *Number of peak id elements 0-48*
        - at: 48 array of 2 byte unsigned short; *atomic numbers for peak id elems*
        - line: 48 array of 2 byte unsigned short; *line numbers for peak id elems*
        - energy: 48 array of 4 byte float; *float energy of identified peaks*
        - height: 48 array of 4 byte unsigned long; *height in counts of id' ed peaks*
        - spkht: 48 array of 2 byte short; *sorted peak height of id' ed peaks*

        - filler5_1: 30 byte;

        - numRois: 2 byte short; *Number of ROI's defined 0-48*
        - st: 48 array of 2 byte short; *Start channel # for each ROI*
        - end: 48 array of 2 byte short; *End channel # for each ROI*
        - roiEnable: 48 array of 2 byte short; *ROI enable/disable flags*
        - roiNames: (24 x 8) array of 1 byte char; *8 char name for eah ROI*

        - filler5_2: 1 byte;

        - userID: 80 array of 1 byte char; *User ID (Vision S/W) - Overlapping*

        - filler5_3: 111 byte;

        - sRoi: 48 array of 2 byte short; *sorted ROI heights*
        - scaNum: 48 array of 2 byte short; *SCA number assigned for each ROI*

        - filler6: 12 byte;

        - backgrdWidth: 2 byte short; *Background width*
        - manBkgrdPerc: 4 byte float; *Percentage to move manual background down*
        - numBkgrdPts: 2 byte short; *Number of background points (2-64)*
        - backMethod: 4 byte unsigned long; *Background method 1=auto, 2=manual*
        - backStEng: 4 byte float; *Starting energy of background*
        - backEndEng: 4 byte float; *Ending energy of background*
        - bg: 64 array of 2 byte short; *Channel # of background point*
        - bgType: 4 byte unsigned long; *Background type. 1 = curve, 2 = linear.*
        - concenKev1: 4 byte float; *First concentration background point*
        - concenKev2: 4 byte float; *Second concentration background point*
        - concenMethod: 2 byte short; *0 = Off, 1 = On*
        - jobFilename: 32 array of 1 byte char; *Vision Job Filename*

        - filler7: 16 byte;

        - numLabels: 2 byte short; *Number of displayed labels*
        - label: (10 x 32) array 1 byte char; *32 character labels on the spectrum*
        - labelx: 10 array of 2 byte short; *x position of label in terms of channel #*
        - labely: 10 array of 4 byte long; *y position of label in terms of counts*
        - zListFlag: 4 byte long; *Flag to indicate if Z List was written*
        - bgPercents: 64 array of 4 byte float; *Percentage to move background point up and down.*
        - IswGBg: 2 byte short; *= 1 if new backgrd pts exist*
        - BgPoints: 5 array of 4 byte float; *Background points*
        - IswGConc: 2 byte short; *= 1 if given concentrations exist*
        - numConcen: 2 byte short; *Number of elements (up to 24)*
        - ZList: 24 array of 2 byte short; *Element list for which given concentrations exist*
        - GivenConc: 24 array of 4 byte float; *Given concentrations for each element in Zlist*

        - filler8: 598 byte;

        - s: 4096 array of 4 byte long; *counts for each channel*
        - longFileName: 256 array of 1 byte char; *Long filename for 32 bit version*
        - longImageFileName: 256 array of 1 byte char; *Associated long image file name*
        - ADCTimeConstantNew: 4 byte float; *Time constant: 2.5… 100 OR 1.6… 102.4 us*

        # the following datatypes are only included for version 0.70:

        - filler9: 60 byte;

        - numZElements: 2 byte short; *number of Z List elements for quant*
        - zAtoms: 48 array of 2 byte short; *Z List Atomic numbers*
        - zShells: 48 array of 2 byte short; *Z List Shell numbers*

    Returns
    -------
    dtype_list : list
        List of the data tags and data types that will be used by numpy to
        read an SPC file header.
    """
    end = endianess
    # important parameters are marked by "**" in comment
    if load_all:
        dtype_list = \
            [  # data offset (bytes)
                ('fVersion', end + 'f4'),  # 0
                ('aVersion', end + 'f4'),  # 4
                ('fileName', '8i1'),  # 8
                ('collectDateYear', end + 'i2'),  # 16
                ('collectDateDay', end + 'i1'),  # 17
                ('collectDateMon', end + 'i1'),
                ('collectTimeMin', end + 'i1'),
                ('collectTimeHour', end + 'i1'),
                ('collectTimeHund', end + 'i1'),
                ('collectTimeSec', end + 'i1'),
                ('fileSize', end + 'i4'),  # 24
                ('dataStart', end + 'i4'),  # 28
                ('numPts', end + 'i2'),  # 32
                ('intersectingDist', end + 'i2'),  # 34
                ('workingDist', end + 'i2'),  # 36
                ('scaleSetting', end + 'i2'),  # 38

                ('filler1', 'V24'),  # 40

                ('spectrumLabel', '256i1'),  # 64
                ('imageFilename', '8i1'),  # 320
                ('spotX', end + 'i2'),  # 328
                ('spotY', end + 'i2'),  # 330
                ('imageADC', end + 'i2'),  # 332
                ('discrValues', end + '5i4'),  # 334
                ('discrEnabled', end + '5i1'),  # 354
                ('pileupProcessed', end + 'i1'),  # 359
                ('fpgaVersion', end + 'i4'),  # 360
                ('pileupProcVersion', end + 'i4'),  # 364
                ('NB5000CFG', end + 'i4'),  # 368

                ('filler2', 'V12'),  # 380

                ('evPerChan', end + 'i4'),  # 384 **
                ('ADCTimeConstant', end + 'i2'),  # 388
                ('analysisType', end + 'i2'),  # 390
                ('preset', end + 'f4'),  # 392
                ('maxp', end + 'i4'),  # 396
                ('maxPeakCh', end + 'i4'),  # 400
                ('xRayTubeZ', end + 'i2'),  # 404
                ('filterZ', end + 'i2'),  # 406
                ('current', end + 'f4'),  # 408
                ('sampleCond', end + 'i2'),  # 412
                ('sampleType', end + 'i2'),  # 414
                ('xrayCollimator', end + 'u2'),  # 416
                ('xrayCapilaryType', end + 'u2'),  # 418
                ('xrayCapilarySize', end + 'u2'),  # 420
                ('xrayFilterThickness', end + 'u2'),  # 422
                ('spectrumSmoothed', end + 'u2'),  # 424
                ('detector_Size_SiLi', end + 'u2'),  # 426
                ('spectrumReCalib', end + 'u2'),  # 428
                ('eagleSystem', end + 'u2'),  # 430
                ('sumPeakRemoved', end + 'u2'),  # 432
                ('edaxSoftwareType', end + 'u2'),  # 434

                ('filler3', 'V6'),  # 436

                ('escapePeakRemoved', end + 'u2'),  # 442
                ('analyzerType', end + 'u4'),  # 444
                ('startEnergy', end + 'f4'),  # 448 **
                ('endEnergy', end + 'f4'),  # 452
                ('liveTime', end + 'f4'),  # 456 **
                ('tilt', end + 'f4'),  # 460 **
                ('takeoff', end + 'f4'),  # 464
                ('beamCurFact', end + 'f4'),  # 468
                ('detReso', end + 'f4'),  # 472 **
                ('detectType', end + 'u4'),  # 476
                ('parThick', end + 'f4'),  # 480
                ('alThick', end + 'f4'),  # 484
                ('beWinThick', end + 'f4'),  # 488
                ('auThick', end + 'f4'),  # 492
                ('siDead', end + 'f4'),  # 496
                ('siLive', end + 'f4'),  # 500
                ('xrayInc', end + 'f4'),  # 504
                ('azimuth', end + 'f4'),  # 508 **
                ('elevation', end + 'f4'),  # 512 **
                ('bCoeff', end + 'f4'),  # 516
                ('cCoeff', end + 'f4'),  # 520
                ('tailMax', end + 'f4'),  # 524
                ('tailHeight', end + 'f4'),  # 528
                ('kV', end + 'f4'),  # 532 **
                ('apThick', end + 'f4'),  # 536
                ('xTilt', end + 'f4'),  # 540
                ('yTilt', end + 'f4'),  # 544
                ('yagStatus', end + 'u4'),  # 548

                ('filler4', 'V24'),  # 552

                ('rawDataType', end + 'u2'),  # 576
                ('totalBkgdCount', end + 'f4'),  # 578
                ('totalSpectralCount', end + 'u4'),  # 582
                ('avginputCount', end + 'f4'),  # 586
                ('stdDevInputCount', end + 'f4'),  # 590
                ('peakToBack', end + 'u2'),  # 594
                ('peakToBackValue', end + 'f4'),  # 596

                ('filler5', 'V38'),  # 600

                ('numElem', end + 'i2'),  # 638 **
                ('at', end + '48u2'),  # 640 **
                ('line', end + '48u2'),  # 736
                ('energy', end + '48f4'),  # 832
                ('height', end + '48u4'),  # 1024
                ('spkht', end + '48i2'),  # 1216

                ('filler5_1', 'V30'),  # 1312

                ('numRois', end + 'i2'),  # 1342
                ('st', end + '48i2'),  # 1344
                ('end', end + '48i2'),  # 1440
                ('roiEnable', end + '48i2'),  # 1536
                ('roiNames', '(24,8)i1'),  # 1632

                ('filler5_2', 'V1'),  # 1824

                ('userID', '80i1'),  # 1825

                ('filler5_3', 'V111'),  # 1905

                ('sRoi', end + '48i2'),  # 2016
                ('scaNum', end + '48i2'),  # 2112

                ('filler6', 'V12'),  # 2208

                ('backgrdWidth', end + 'i2'),  # 2220
                ('manBkgrdPerc', end + 'f4'),  # 2222
                ('numBkgrdPts', end + 'i2'),  # 2226
                ('backMethod', end + 'u4'),  # 2228
                ('backStEng', end + 'f4'),  # 2232
                ('backEndEng', end + 'f4'),  # 2236
                ('bg', end + '64i2'),  # 2240
                ('bgType', end + 'u4'),  # 2368
                ('concenKev1', end + 'f4'),  # 2372
                ('concenKev2', end + 'f4'),  # 2376
                ('concenMethod', end + 'i2'),  # 2380
                ('jobFilename', end + '32i1'),  # 2382

                ('filler7', 'V16'),  # 2414

                ('numLabels', end + 'i2'),  # 2430
                ('label', end + '(10,32)i1'),  # 2432
                ('labelx', end + '10i2'),  # 2752
                ('labely', end + '10i4'),  # 2772
                ('zListFlag', end + 'i4'),  # 2812
                ('bgPercents', end + '64f4'),  # 2816
                ('IswGBg', end + 'i2'),  # 3072
                ('BgPoints', end + '5f4'),  # 3074
                ('IswGConc', end + 'i2'),  # 3094
                ('numConcen', end + 'i2'),  # 3096
                ('ZList', end + '24i2'),  # 3098
                ('GivenConc', end + '24f4'),  # 3146

                ('filler8', 'V598'),  # 3242

                ('s', end + '4096i4'),  # 3840
                ('longFileName', end + '256i1'),  # 20224
                ('longImageFileName', end + '256i1'),  # 20480
            ]

        if version >= 0.7:
            dtype_list.extend([
                ('ADCTimeConstantNew', end + 'f4'),  # 20736

                ('filler9', 'V60'),  # 20740

                ('numZElements', end + 'i2'),  # 20800
                ('zAtoms', end + '48i2'),  # 20802
                ('zShells', end + '48i2'),  # 20898
            ])

    else:
        dtype_list = \
            [
                ('filler1', 'V28'),  # 0

                ('dataStart', end + 'i4'),  # 28
                ('numPts', end + 'i2'),  # 32 **

                ('filler1_1', 'V350'),  # 34

                ('evPerChan', end + 'i4'),  # 384 **

                ('filler2', 'V60'),  # 388

                ('startEnergy', end + 'f4'),  # 448 **
                ('endEnergy', end + 'f4'),  # 452
                ('liveTime', end + 'f4'),  # 456 **
                ('tilt', end + 'f4'),  # 460 **

                ('filler3', 'V8'),  # 464

                ('detReso', end + 'f4'),  # 472 **

                ('filler4', 'V32'),  # 476

                ('azimuth', end + 'f4'),  # 508 **
                ('elevation', end + 'f4'),  # 512 **

                ('filler5', 'V16'),  # 516

                ('kV', end + 'f4'),  # 532 **

                ('filler6', 'V102'),  # 536

                ('numElem', end + 'i2'),  # 638 **
                ('at', end + '48u2'),  # 640 **

                ('filler7', 'V20004'),  # 736

            ]
    return dtype_list


def __get_ipr_header(f, endianess):
    """
    Get the header of an spc file, checking for the file version as necessary

    Parameters
    ----------
    f : file
        A file object for the .spc file to be read (i.e. file should be
        already opened with ``open()``)
    endianess : char
        Byte-order of data to read

    Returns
    -------
    ipr_header : np.ndarray
        Array containing the binary information read from the .ipr file
    """
    version = np.fromfile(f,
                          dtype=[('version', '{}i2'.format(endianess))],
                          count=1)
    version = np.asscalar(version)[0]  # convert to scalar
    f.seek(0)
    _logger.debug(' .ipr version is {}'.format(version))

    ipr_header = np.fromfile(f,
                             dtype=get_ipr_dtype_list(
                                 endianess=endianess,
                                 version=version),
                             count=1)

    return ipr_header


def get_ipr_dtype_list(endianess='<', version=333):
    """
    Get the data type list for an IPR image description file.
    Further information about the file format is available `here
    <https://github.com/hyperspy/hyperspy/files/29507/ImageIPR.pdf>`__.

    Table of header tags:
        -  version: 2 byte unsigned short; *Current version number: 334*
        -  imageType: 2 byte unsigned short; *0=empty; 1=electron; 2=xmap; 3=disk; 4=overlay*
        -  label: 8 byte char array; *Image label*
        -  sMin: 2 byte unsigned short; *Min collected signal*
        -  sMax: 2 byte unsigned short; *Max collected signal*
        -  color: 2 byte unsigned short; *color: 0=gray; 1=R; 2=G; 3=B; 4=Y; 5=M; 6=C; 8=overlay*
        -  presetMode: 2 byte unsigned short; *0=clock; 1=live*
        -  presetTime: 4 byte unsigned long; *Dwell time for x-ray (millisec)*
        -  dataType: 2 byte unsigned short; *0=ROI;  1=Net intensity; 2=K ratio; 3=Wt%;  4=Mthin2*
        -  timeConstantOld: 2 byte unsigned short; *Amplifier pulse processing time [usec]*
        -  reserved1: 2 byte short; *Not used*
        -  roiStartChan: 2 byte unsigned short; *ROI starting channel*
        -  roiEndChan: 2 byte unsigned short; *ROI ending channel*
        -  userMin: 2 byte short; *User Defined Min signal range*
        -  userMax: 2 byte short; *User Defined Max signal range*
        -  iADC: 2 byte unsigned short; *Electron detector number: 1; 2; 3; 4*
        -  reserved2: 2 byte short; *Not used*
        -  iBits: 2 byte unsigned short; *conversion type: 8; 12 (not used)*
        -  nReads: 2 byte unsigned short; *No. of reads per point*
        -  nFrames: 2 byte unsigned short; *No. of frames averaged (not used)*
        -  fDwell: 4 byte float; *Dwell time (not used)*
        -  accV: 2 byte unsigned short; *V_acc in units of 100V*
        -  tilt: 2 byte short; *Sample tilt [deg]*
        -  takeoff: 2 byte short; *Takeoff angle [deg]*
        -  mag: 4 byte unsigned long; *Magnification*
        -  wd: 2 byte unsigned short; *Working distance [mm]*
        -  mppX: 4 byte float; *Microns per pixel in X direction*
        -  mppY: 4 byte float; *Microns per pixel in Y direction*
        -  nTextLines: 2 byte unsigned short; *No. of comment lines *
        -  charText: (4 x 32) byte character array; *Comment text*
        -  reserved3: 4 byte float; *Not used*
        -  nOverlayElements: 2 byte unsigned short; *No. of overlay elements*
        -  overlayColors: 16 array of 2 byte unsigned short; *Overlay colors*

        # These two are specific to V334 of the file format, and are omitted
        # for compatibility with V333 of the IPR format
        -  timeConstantNew: 4 byte float; *Amplifier time constant [usec]*
        -  reserved4: 2 array of 4 byte float; *Not used*


    Parameters
    ----------
    endianess : char
        byte-order used to read the data
    version : float
        version of .ipr file to read (only 333 and 334 have been tested)
        Default is 333 to be as backwards-compatible as possible, but the
        file version can be read from the file anyway, so this parameter
        should always be set programmatically

    Returns
    -------
    dtype_list : list
        List of the data tags and data types that will be used by numpy to
        read an IPR file.
    """
    end = endianess
    dtype_list = \
        [
            ('version', end + 'u2'),
            ('imageType', end + 'u2'),
            ('label', end + 'a8'),
            ('sMin', end + 'u2'),
            ('sMax', end + 'u2'),
            ('color', end + 'u2'),
            ('presetMode', end + 'u2'),
            ('presetTime', end + 'u4'),
            ('dataType', end + 'u2'),
            ('timeConstantOld', end + 'u2'),
            ('reserved1', end + 'i2'),
            ('roiStartChan', end + 'u2'),
            ('roiEndChan', end + 'u2'),
            ('userMin', end + 'i2'),
            ('userMax', end + 'i2'),
            ('iADC', end + 'u2'),
            ('reserved2', end + 'i2'),
            ('iBits', end + 'u2'),
            ('nReads', end + 'u2'),
            ('nFrames', end + 'u2'),
            ('fDwell', end + 'f4'),
            ('accV', end + 'u2'),
            ('tilt', end + 'i2'),
            ('takeoff', end + 'i2'),
            ('mag', end + 'u4'),
            ('wd', end + 'u2'),
            ('mppX', end + 'f4'),
            ('mppY', end + 'f4'),
            ('nTextLines', end + 'u2'),
            ('charText', end + '4a32'),
            ('reserved3', end + '4f4'),
            ('nOverlayElements', end + 'u2'),
            ('overlayColors', end + '16u2')]

    if version >= 334:
        dtype_list.extend([
            ('timeConstantNew', end + 'f4'),
            ('reserved4', end + '2f4'),
        ])

    return dtype_list


def _add_spc_metadata(metadata, spc_header):
    """
    Return metadata with information from the .spc header added

    Parameters
    ----------
    metadata : dict
        current metadata of signal without spectral calibration information
        added
    spc_header : dict
        header of .spc file that contains spectral information such as
        azimuth and elevation angles, energy resolution, etc.

    Returns
    -------
    metadata : dict
        copy of original dictionary with spectral calibration added
    """
    metadata['Acquisition_instrument'] = {
        'SEM':
            {'Detector':
             {'EDS': {'azimuth_angle': spc_header['azimuth'],
                      'elevation_angle': spc_header['elevation'],
                      'energy_resolution_MnKa': spc_header['detReso'],
                      'live_time': spc_header['liveTime']}},
             'beam_energy': spc_header['kV'],
             'Stage': {'tilt_alpha': spc_header['tilt']}}
    }

    # Get elements stored in spectrum:
    num_elem = spc_header['numElem']
    if num_elem > 0:
        element_list = sorted([atomic_number2name[i] for
                               i in spc_header['at'][:num_elem]])
        metadata['Sample'] = {'elements': element_list}
        _logger.info(" Elemental information found in the spectral metadata "
                     "was added to the signal.\n"
                     "Elements found were: {}\n".format(element_list))

    return metadata


def spc_reader(filename,
               endianess='<',
               load_all_spc=False,
               **kwargs):
    """
    Read data from an SPC spectrum specified by filename.

    Parameters
    ----------
    filename : str
        Name of SPC file to read
    endianess : char
        Byte-order of data to read
    load_all_spc : bool
        Switch to control if all of the .spc header is read, or just the
        important parts for import into HyperSpy
    **kwargs
        Remaining arguments are passed to the Numpy ``memmap`` function

    Returns
    -------
    list
        list with dictionary of signal information to be passed back to
        hyperspy.io.load_with_reader
    """
    with open(filename, 'rb') as f:
        _logger.debug(' Reading {}'.format(filename))
        spc_header = __get_spc_header(f, endianess, load_all_spc)

        spc_dict = sarray2dict(spc_header)
        original_metadata = {'spc_header': spc_dict}

        nz = original_metadata['spc_header']['numPts']
        data_offset = original_metadata['spc_header']['dataStart']

        mode = kwargs.pop('mode', 'c')
        lazy = kwargs.pop('lazy', False)
        if lazy:
            mode = 'r'

        # Read data from file into a numpy memmap object
        data = np.memmap(f, mode=mode, offset=data_offset,
                         dtype='u4', shape=(1, nz), **kwargs).squeeze()

    # create the energy axis dictionary:
    energy_axis = {
        'size': data.shape[0],
        'index_in_array': 0,
        'name': 'Energy',
        'scale': original_metadata['spc_header']['evPerChan'] / 1000.0,
        'offset': original_metadata['spc_header']['startEnergy'],
        'units': 'keV'
    }

    # Assign metadata for spectrum:
    metadata = {'General': {'original_filename': os.path.split(filename)[1],
                            'title': 'EDS Spectrum'},
                "Signal": {'signal_type': "EDS_SEM",
                           'record_by': 'spectrum', }, }
    metadata = _add_spc_metadata(metadata, spc_dict)

    dictionary = {'data': data,
                  'axes': [energy_axis],
                  'metadata': metadata,
                  'original_metadata': original_metadata}

    return [dictionary, ]


def spd_reader(filename,
               endianess='<',
               spc_fname=None,
               ipr_fname=None,
               load_all_spc=False,
               **kwargs):
    """
    Read data from an SPD spectral map specified by filename.

    Parameters
    ----------
    filename : str
        Name of SPD file to read
    endianess : char
        Byte-order of data to read
    spc_fname : None or str
        Name of file from which to read the spectral calibration. If data
        was exported fully from EDAX TEAM software, an .spc file with the
        same name as the .spd should be present.
        If `None`, the default filename will be searched for.
        Otherwise, the name of the .spc file to use for calibration can
        be explicitly given as a string.
    ipr_fname : None or str
        Name of file from which to read the spatial calibration. If data
        was exported fully from EDAX TEAM software, an .ipr file with the
        same name as the .spd (plus a "_Img" suffix) should be present.
        If `None`, the default filename will be searched for.
        Otherwise, the name of the .ipr file to use for spatial calibration
        can be explicitly given as a string.
    load_all_spc : bool
        Switch to control if all of the .spc header is read, or just the
        important parts for import into HyperSpy
    **kwargs
        Remaining arguments are passed to the Numpy ``memmap`` function

    Returns
    -------
    list
        list with dictionary of signal information to be passed back to
        hyperspy.io.load_with_reader
    """
    with open(filename, 'rb') as f:
        spd_header = np.fromfile(f,
                                 dtype=get_spd_dtype_list(endianess),
                                 count=1)

        original_metadata = {'spd_header': sarray2dict(spd_header)}

        # dimensions of map data:
        nx = original_metadata['spd_header']['nPoints']
        ny = original_metadata['spd_header']['nLines']
        nz = original_metadata['spd_header']['nChannels']
        data_offset = original_metadata['spd_header']['dataOffset']
        data_type = {'1': 'u1',
                     '2': 'u2',
                     '4': 'u4'}[str(original_metadata['spd_header'][
                         'countBytes'])]
        lazy = kwargs.pop('lazy', False)
        mode = kwargs.pop('mode', 'c')
        if lazy:
            mode = 'r'

        # Read data from file into a numpy memmap object
        data = np.memmap(f, mode=mode, offset=data_offset, dtype=data_type,
                         **kwargs).squeeze().reshape((nz, nx, ny), order='F').T

    # Convert char arrays to strings:
    original_metadata['spd_header']['tag'] = \
        spd_header['tag'][0].view('S16')[0]
    # fName is the name of the .bmp (and .ipr) file of the map
    original_metadata['spd_header']['fName'] = \
        spd_header['fName'][0].view('S120')[0]

    # Get name of .spc file from the .spd map (if not explicitly given):
    if spc_fname is None:
        spc_path = os.path.dirname(filename)
        spc_basename = os.path.splitext(os.path.basename(filename))[
            0] + '.spc'
        spc_fname = os.path.join(spc_path, spc_basename)

    # Get name of .ipr file from bitmap image (if not explicitly given):
    if ipr_fname is None:
        ipr_basename = os.path.splitext(
            os.path.basename(
                original_metadata['spd_header'][
                    'fName']))[0].decode() + '.ipr'
        ipr_path = os.path.dirname(filename)
        ipr_fname = os.path.join(ipr_path, ipr_basename)

    # Flags to control reading of files
    read_spc = os.path.isfile(spc_fname)
    read_ipr = os.path.isfile(ipr_fname)

    # Read the .ipr header (if possible)
    if read_ipr:
        with open(ipr_fname, 'rb') as f:
            _logger.debug(' From .spd reader - '
                          'reading .ipr {}'.format(ipr_fname))
            ipr_header = __get_ipr_header(f, endianess)
            original_metadata['ipr_header'] = sarray2dict(ipr_header)

            # Workaround for type error when saving hdf5:
            # save as list of strings instead of numpy unicode array
            # see https://github.com/hyperspy/hyperspy/pull/2007 and
            #     https://github.com/h5py/h5py/issues/289 for context
            original_metadata['ipr_header']['charText'] = \
                [np.string_(i) for i in
                 original_metadata['ipr_header']['charText']]
    else:
        _logger.warning('Could not find .ipr file named {}.\n'
                        'No spatial calibration will be loaded.'
                        '\n'.format(ipr_fname))

    # Read the .spc header (if possible)
    if read_spc:
        with open(spc_fname, 'rb') as f:
            _logger.debug(' From .spd reader - '
                          'reading .spc {}'.format(spc_fname))
            spc_header = __get_spc_header(f, endianess, load_all_spc)
            spc_dict = sarray2dict(spc_header)
            original_metadata['spc_header'] = spc_dict
    else:
        _logger.warning('Could not find .spc file named {}.\n'
                        'No spectral metadata will be loaded.'
                        '\n'.format(spc_fname))

    # create the energy axis dictionary:
    energy_axis = {
        'size': data.shape[2],
        'index_in_array': 2,
        'name': 'Energy',
        'scale': original_metadata['spc_header']['evPerChan'] / 1000.0 if
        read_spc else 1,
        'offset': original_metadata['spc_header']['startEnergy'] if
        read_spc else 1,
        'units': 'keV' if read_spc else t.Undefined,
    }

    nav_units = 'µm'
    # Create navigation axes dictionaries:
    x_axis = {
        'size': data.shape[1],
        'index_in_array': 1,
        'name': 'x',
        'scale': original_metadata['ipr_header']['mppX'] if read_ipr
        else 1,
        'offset': 0,
        'units': nav_units if read_ipr else t.Undefined,
    }

    y_axis = {
        'size': data.shape[0],
        'index_in_array': 0,
        'name': 'y',
        'scale': original_metadata['ipr_header']['mppY'] if read_ipr
        else 1,
        'offset': 0,
        'units': nav_units if read_ipr else t.Undefined,
    }

    # Assign metadata for spectrum image:
    metadata = {'General': {'original_filename': os.path.split(filename)[1],
                            'title': 'EDS Spectrum Image'},
                "Signal": {'signal_type': "EDS_SEM",
                           'record_by': 'spectrum', }, }

    # Add spectral calibration and elements (if present):
    if read_spc:
        metadata = _add_spc_metadata(metadata, spc_dict)

    # Define navigation and signal axes:
    axes = [y_axis, x_axis, energy_axis]

    dictionary = {'data': data,
                  'axes': axes,
                  'metadata': metadata,
                  'original_metadata': original_metadata}

    return [dictionary, ]


def file_reader(filename,
                record_by='spectrum',
                endianess='<',
                **kwargs):
    """

    Parameters
    ----------
    filename : str
        Name of file to read
    record_by : str
        EDAX EDS data is always recorded by 'spectrum', so this parameter
        is not used
    endianess : char
        Byte-order of data to read
    **kwargs
        Additional keyword arguments supplied to the readers

    Returns
    -------

    """
    ext = os.path.splitext(filename)[1][1:]
    if ext in spd_extensions:
        return spd_reader(filename,
                          endianess,
                          **kwargs)
    elif ext in spc_extensions:
        return spc_reader(filename,
                          endianess,
                          **kwargs)
    else:
        raise IOError("Did not understand input file format.")
