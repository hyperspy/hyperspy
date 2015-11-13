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
import numpy as np
from hyperspy.misc.array_tools import sarray2dict

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

spd_extensions = ('spd', 'SPD')
spc_extensions = ('spc', 'SPC')


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


def get_spc_dtype_list(endianess='<'):
    """
    Get the data type list for an SPC spectrum.
    Further information about the file format is available `here
    <https://github.com/hyperspy/hyperspy/files/29506/SPECTRUM-V70.pdf>`__.

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

        - filler9: 60 byte;

        - numZElements: 2 byte short; *number of Z List elements for quant*
        - zAtoms: 48 array of 2 byte short; *Z List Atomic numbers*
        - zShells: 48 array of 2 byte short; *Z List Shell numbers*


    Parameters
    ----------
    endianess : byte-order used to read the data

    Returns
    -------
    dtype_list : list
        List of the data tags and data types that will be used by numpy to
        read an SPC file header.
    """
    end = endianess
    dtype_list = \
        [
            ('fVersion', end + 'f4'),
            ('aVersion', end + 'f4'),
            ('fileName', '8i1'),
            ('collectDateYear', end + 'i2'),
            ('collectDateDay', end + 'i1'),
            ('collectDateMon', end + 'i1'),
            ('collectTimeMin', end + 'i1'),
            ('collectTimeHour', end + 'i1'),
            ('collectTimeHund', end + 'i1'),
            ('collectTimeSec', end + 'i1'),
            ('fileSize', end + 'i4'),
            ('dataStart', end + 'i4'),
            ('numPts', end + 'i2'),
            ('intersectingDist', end + 'i2'),
            ('workingDist', end + 'i2'),
            ('scaleSetting', end + 'i2'),

            ('filler1', 'V24'),

            ('spectrumLabel', '256i1'),
            ('imageFilename', '8i1'),
            ('spotX', end + 'i2'),
            ('spotY', end + 'i2'),
            ('imageADC', end + 'i2'),
            ('discrValues', end + '5i4'),
            ('discrEnabled', end + '5i1'),
            ('pileupProcessed', end + 'i1'),
            ('fpgaVersion', end + 'i4'),
            ('pileupProcVersion', end + 'i4'),
            ('NB5000CFG', end + 'i4'),

            ('filler2', 'V12'),

            ('evPerChan', end + 'i4'),
            ('ADCTimeConstant', end + 'i2'),
            ('analysisType', end + 'i2'),
            ('preset', end + 'f4'),
            ('maxp', end + 'i4'),
            ('maxPeakCh', end + 'i4'),
            ('xRayTubeZ', end + 'i2'),
            ('filterZ', end + 'i2'),
            ('current', end + 'f4'),
            ('sampleCond', end + 'i2'),
            ('sampleType', end + 'i2'),
            ('xrayCollimator', end + 'u2'),
            ('xrayCapilaryType', end + 'u2'),
            ('xrayCapilarySize', end + 'u2'),
            ('xrayFilterThickness', end + 'u2'),
            ('spectrumSmoothed', end + 'u2'),
            ('detector_Size_SiLi', end + 'u2'),
            ('spectrumReCalib', end + 'u2'),
            ('eagleSystem', end + 'u2'),
            ('sumPeakRemoved', end + 'u2'),
            ('edaxSoftwareType', end + 'u2'),

            ('filler3', 'V6'),

            ('escapePeakRemoved', end + 'u2'),
            ('analyzerType', end + 'u4'),
            ('startEnergy', end + 'f4'),
            ('endEnergy', end + 'f4'),
            ('liveTime', end + 'f4'),
            ('tilt', end + 'f4'),
            ('takeoff', end + 'f4'),
            ('beamCurFact', end + 'f4'),
            ('detReso', end + 'f4'),
            ('detectType', end + 'u4'),
            ('parThick', end + 'f4'),
            ('alThick', end + 'f4'),
            ('beWinThick', end + 'f4'),
            ('auThick', end + 'f4'),
            ('siDead', end + 'f4'),
            ('siLive', end + 'f4'),
            ('xrayInc', end + 'f4'),
            ('azimuth', end + 'f4'),
            ('elevation', end + 'f4'),
            ('bCoeff', end + 'f4'),
            ('cCoeff', end + 'f4'),
            ('tailMax', end + 'f4'),
            ('tailHeight', end + 'f4'),
            ('kV', end + 'f4'),
            ('apThick', end + 'f4'),
            ('xTilt', end + 'f4'),
            ('yTilt', end + 'f4'),
            ('yagStatus', end + 'u4'),

            ('filler4', 'V24'),

            ('rawDataType', end + 'u2'),
            ('totalBkgdCount', end + 'f4'),
            ('totalSpectralCount', end + 'u4'),
            ('avginputCount', end + 'f4'),
            ('stdDevInputCount', end + 'f4'),
            ('peakToBack', end + 'u2'),
            ('peakToBackValue', end + 'f4'),

            ('filler5', 'V38'),

            ('numElem', end + 'i2'),
            ('at', end + '48u2'),
            ('line', end + '48u2'),
            ('energy', end + '48f4'),
            ('height', end + '48u4'),
            ('spkht', end + '48i2'),

            ('filler5_1', 'V30'),

            ('numRois', end + 'i2'),
            ('st', end + '48i2'),
            ('end', end + '48i2'),
            ('roiEnable', end + '48i2'),
            ('roiNames', '(24,8)i1'),

            ('filler5_2', 'V1'),

            ('userID', '80i1'),

            ('filler5_3', 'V111'),

            ('sRoi', end + '48i2'),
            ('scaNum', end + '48i2'),

            ('filler6', 'V12'),

            ('backgrdWidth', end + 'i2'),
            ('manBkgrdPerc', end + 'f4'),
            ('numBkgrdPts', end + 'i2'),
            ('backMethod', end + 'u4'),
            ('backStEng', end + 'f4'),
            ('backEndEng', end + 'f4'),
            ('bg', end + '64i2'),
            ('bgType', end + 'u4'),
            ('concenKev1', end + 'f4'),
            ('concenKev2', end + 'f4'),
            ('concenMethod', end + 'i2'),
            ('jobFilename', end + '32i1'),

            ('filler7', 'V16'),

            ('numLabels', end + 'i2'),
            ('label', end + '(10,32)i1'),
            ('labelx', end + '10i2'),
            ('labely', end + '10i4'),
            ('zListFlag', end + 'i4'),
            ('bgPercents', end + '64f4'),
            ('IswGBg', end + 'i2'),
            ('BgPoints', end + '5f4'),
            ('IswGConc', end + 'i2'),
            ('numConcen', end + 'i2'),
            ('ZList', end + '24i2'),
            ('GivenConc', end + '24f4'),

            ('filler8', 'V598'),

            ('s', end + '4096i4'),
            ('longFileName', end + '256i1'),
            ('longImageFileName', end + '256i1'),
            ('ADCTimeConstantNew', end + 'f4'),

            ('filler9', 'V60'),

            ('numZElements', end + 'i2'),
            ('zAtoms', end + '48i2'),
            ('zShells', end + '48i2'),
        ]
    return dtype_list


def get_ipr_dtype_list(endianess='<'):
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
        -  timeConstantNew: 4 byte float; *Amplifier time constant [usec]*
        -  reserved4: 2 array of 4 byte float; *Not used*


    Parameters
    ----------
    endianess : byte-order used to read the data

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
            ('overlayColors', end + '16u2'),
            ('timeConstantNew', end + 'f4'),
            ('reserved4', end + '2f4'),
        ]
    return dtype_list


def spc_reader(filename, endianess='<', *args):
    """
    Read data from an SPC spectrum specified by filename.

    Parameters
    ----------
    filename : str
        Name of SPC file to read
    endianess : char
        Byte-order of data to read
    args

    Returns
    -------

    """


def spd_reader(filename, endianess='<', *args):
    """
    Read data from an SPD spectral map specified by filename.

    Parameters
    ----------
    filename : str
        Name of SPD file to read
    endianess : char
        Byte-order of data to read
    args

    Returns
    -------

    """
    f = open(filename, 'rb')
    spd_header = np.fromfile(f,
                             dtype=get_spd_dtype_list(endianess),
                             count=1)



def file_reader(filename, endianess='<', *args):
    """

    Parameters
    ----------
    filename : str
        Name of file to read
    endianess : char
        Byte-order of data to read
    args

    Returns
    -------

    """
    ext = os.path.splitext(filename)[1][1:]
    if ext in spd_extensions:
        return spd_reader(filename, endianess, *args)
    elif ext in spc_extensions:
        return spc_reader(filename, endianess, *args)
