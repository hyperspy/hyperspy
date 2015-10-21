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

# INFORMATION ABOUT THE SEMPER FORMAT LABEL:
# Picture labels consist of a sequence of bytes
# ---------------------------------------------
# v61:256B | v7:at least 256B, rounded up to multiple of block size 64,
#               with max set by LNLAB in params.f
# The versions have the same contents for the first 256B, as set out below,
# and referred to as the label 'base'; beyond this, in the label 'extension',
# the structure is deliberately undefined, allowing you to make your own use
# of the additional storage.
#
# From Aug14:
#  B1-6    S e m p e r (ic chars)
#   7,8    ncol msb,lsb (BIG-ended)
#   9,10   nrow msb,lsb
#  11,12   nlay msb,lsb
#  13,14   ccol msb,lsb
#  15,16   crow msb,lsb
#  17,18   clay msb,lsb
#    19    class: 1-20
#            for image,macro,fou,spec,correln,undef,walsh,histog,plist,lut
#    20    form: 0,1,2,3,4 = byte,i*2,fp,com,i*4 from Aug08
#    21    wp: non-zero if prot
#  22-27   creation year(-1900?),month,day,hour,min,sec
#    28    v61|v7 # chars in range record | 255
#  29-55   v61: min,max values present (ic chars for decimal repn)
#           v7: min,max vals as two Fp values in B29-36 (LE ordered)
#               followed by 19 unused bytes B37-55
#    56    plist type: 1,2,3 = list,opencurve,closedcurve
#            acc to EXAMPD - code appears to use explicit numbers
# 57,58,59 ncol,nrow,nlay hsb
# 60,61,62 ccol,crow,clay hsb
#    63    RealCoords flag (non-zero -> DX,DY,DZ,X0,Y0,Z0,units held as below)
#    64    v61:0 | v7: # blocks in (this) picture label (incl extn)
#  65-67   0 (free at present)
#  68-71   DATA cmd V7   4-byte Fp values, order LITTLE-ended
#  72-75   DATA cmd V6
#  76-79   RealCoord DZ / V5  RealCoord pars as 4-byte Fp values, LITTLE-ended
#  80-83   ...       Z0 / V4
#  84-87   ...       DY / V3
#  88-91   ...       Y0 / V2
#  92-95   ...       DX / V1
#  96-99   ...       X0 / V0
#   100    # chars in title
# 101-244  title (ic chars)
# 245-248  RealCoord X unit (4 ic chars)
# 249-252  RealCoord Y unit (4 ic chars)
# 253-256  RealCoord Z unit (4 ic chars)
#
# Aug08-Aug14 labels held no RealCoord information, so:
#   B63    'free' and zero - flagging absence of RealCoord information
# 101-256  title (12 chars longer than now)
#
# Before Aug08 labels held no hsb for pic sizes, so:
#  57-99   all free/zero except for use by DATA cmd
# 101-256  title (ic chars)


from time import strftime
import numpy as np
import struct

import logging


# Plugin characteristics
# ----------------------
format_name = 'Semper UNF (unformatted)'
description = 'Read data from Sempers UNF files.'
full_support = True  # Hopefully?
# Recognised file extension
file_extensions = ('unf', 'UNF')
default_extension = 0
# Writing features
writes = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1)]  # All up to 3D
# ----------------------


class SemperFormat(object):

    """Class for importing and exporting Semper `.unf`-files.

    The :class:`~.SemperFormat` class represents a Semper binary file format with a header, which
    holds additional information. `.unf`-files can be saved and read from files.

    Attributes
    ----------
    data : :class:`~numpy.ndarray` (N=3)
        The phase map or magnetization information in a 3D array (with one slice).
    title : string
        Title of the file (not to be confused with the filename).
    offsets : tuple (N=3) of floats
        Offset shifts (in nm) of the grid origin (does not have to start at 0) in x, y, z.
    scales : tuple (N=3) of floats
        Grid spacing (nm per pixel) in x, y, z.
    units : tuple (N=3) of strings
        Units of the grid in x, y, z.
    date : string
        The date of file construction.
    iclass : int
        Defines the image class defined in `ICLASS_DICT`. Normally `image` (1) is chosen.
    iform : int
        Defines the data format defined in 'IFORM_DICT'.
    iversn :
        Current `.unf`-format version. Current: 2.
    ilabel : int
        Defines if a label is present (1) or not (0).
    iformat : int
        Defines if the file is formatted (1) or not (0).
    iwp : int
        Write protect flag, determining if picture is (1) or is not (0) write-projtected.
    ipltyp : int
        Position type list. Standard seems to be 0 (picture not a position list).
    iccoln : int
        Column number of picture origin.
    icrown : int
        Row number of picture origin.
    iclayn : int
        Layer number of picture origin.

    """

    _log = logging.getLogger(__name__)

    ICLASS_DICT = {1: 'image', 2: 'macro', 3: 'fourier', 4: 'spectrum',
                   5: 'correlation', 6: 'undefined', 7: 'walsh', 8: 'position list',
                   9: 'histogram', 10: 'display look-up table'}

    ICLASS_DICT_INV = {v: k for k, v in ICLASS_DICT.iteritems()}

    IFORM_DICT = {0: np.byte, 1: np.int32, 2: np.float32, 3: np.complex64}

    IFORM_DICT_INV = {v: k for k, v in IFORM_DICT.iteritems()}

    def __init__(self, arg_dict):
        self._log.debug('Calling __init__')
        self.data = arg_dict['data']
        self.title = arg_dict['title']
        self.offsets = arg_dict['offsets']
        self.scales = arg_dict['scales']
        self.units = arg_dict['units']
        self.date = arg_dict['date']
        self.iclass = arg_dict['ICLASS']
        self.iform = arg_dict['IFORM']
        self.iversn = arg_dict['IVERSN']
        self.ilabel = arg_dict['ILABEL']
        self.iformat = arg_dict['IFORMAT']
        self.iwp = arg_dict['IWP']
        self.ipltyp = arg_dict['IPLTYP']
        self.iccoln = arg_dict['ICCOLN']
        self.icrown = arg_dict['ICROWN']
        self.iclayn = arg_dict['ICLAYN']
        self._log.debug('Created '+str(self))

    @classmethod
    def load_from_unf(self, filename):
        """Load a `.unf`-file into a :class:`~.SemperFormat` object.

        Parameters
        ----------
        filename : string
            The name of the unf-file from which to load the data. Standard format is '\*.unf'.

        Returns
        -------
        semper : :class:`~.SemperFormat` (N=1)
            Semper file format object containing the loaded information.

        """
        self._log.debug('Calling from_file')
        with open(filename, 'rb') as f:
            # Read header:
            rec_length = np.frombuffer(f.read(4), dtype=np.int32)[0]  # length of header
            header = np.frombuffer(f.read(rec_length), dtype=np.int16)
            ncol, nrow, nlay = header[:3]
            iclass = header[3]
            iform = header[4]
            data_format = self.IFORM_DICT[iform]
            iflag = header[5]
            iversn, remain = divmod(iflag, 10000)
            ilabel, ntitle = divmod(remain, 1000)
            iformat = header[6] if len(header) == 7 else None
            assert np.frombuffer(f.read(4), dtype=np.int32)[0] == rec_length
            # Read title:
            title = ''
            assert np.frombuffer(f.read(4), dtype=np.int32)[0] == ntitle  # length of title
            if ntitle > 0:
                title_bytes = np.frombuffer(f.read(ntitle), dtype=np.byte)
                title = ''.join(map(chr, title_bytes))
            assert np.frombuffer(f.read(4), dtype=np.int32)[0] == ntitle
            # Read label:
            iwp, date, range_string, ipltype, a = [None] * 5  # Initialization!
            iccoln, icrown, iclayn = [None] * 3
            rec_length = np.frombuffer(f.read(4), dtype=np.int32)[0]  # length of label
            if ilabel and rec_length > 0:
                label = np.frombuffer(f.read(512), dtype=np.int16)
                assert ''.join([chr(l) for l in label[:6]]) == 'Semper'
                assert struct.unpack('>h', ''.join([chr(x) for x in label[6:8]]))[0] == ncol
                assert struct.unpack('>h', ''.join([chr(x) for x in label[8:10]]))[0] == nrow
                assert struct.unpack('>h', ''.join([chr(x) for x in label[10:12]]))[0] == nlay
                iccoln = struct.unpack('>h', ''.join([chr(x) for x in label[12:14]]))[0]
                icrown = struct.unpack('>h', ''.join([chr(x) for x in label[14:16]]))[0]
                iclayn = struct.unpack('>h', ''.join([chr(x) for x in label[16:18]]))[0]
                assert label[18] == iclass
                assert label[19] == iform
                iwp = label[20]
                date = '{}-{}-{} {}:{}:{}'.format(label[21]+1900, *label[22:27])
                # No test for ncrang, range is extracted from data itself (also prone to errors)!
                ipltyp = label[55]  # position list type
                real_coords = label[62]
                dz, dy, dx, z0, y0, x0 = [1., 1., 1., 0., 0., 0.]
                if real_coords:
                    dz = struct.unpack('<f', ''.join([chr(x) for x in label[75:79]]))[0]
                    z0 = struct.unpack('<f', ''.join([chr(x) for x in label[79:83]]))[0]
                    dy = struct.unpack('<f', ''.join([chr(x) for x in label[83:87]]))[0]
                    y0 = struct.unpack('<f', ''.join([chr(x) for x in label[87:91]]))[0]
                    dx = struct.unpack('<f', ''.join([chr(x) for x in label[91:95]]))[0]
                    x0 = struct.unpack('<f', ''.join([chr(x) for x in label[95:99]]))[0]
                assert ''.join([str(unichr(l)) for l in label[100:100+ntitle]]) == title
                ux = ''.join([chr(l) for l in label[244:248]]).replace('\x00', '')
                uy = ''.join([chr(l) for l in label[248:252]]).replace('\x00', '')
                uz = ''.join([chr(l) for l in label[252:256]]).replace('\x00', '')
            assert np.frombuffer(f.read(4), dtype=np.int32)[0] == rec_length
            # Read picture data:
            data = np.empty((nlay, nrow, ncol), dtype=data_format)
            for k in range(nlay):
                for j in range(nrow):
                    rec_length = np.frombuffer(f.read(4), dtype=np.int32)[0]  # length of row
                    row = np.frombuffer(f.read(rec_length), dtype=data_format)
                    data[k, j, :] = row
                    assert np.frombuffer(f.read(4), dtype=np.int32)[0] == rec_length
        arg_dict = {'data': data,
                    'title': title,
                    'offsets': (x0, y0, z0),
                    'scales': (dx, dy, dz),
                    'units': (ux, uy, uz),
                    'date': date,
                    'ICLASS': iclass,
                    'IFORM': iform,
                    'IVERSN': iversn,
                    'ILABEL': ilabel,
                    'IFORMAT': iformat,
                    'IWP': iwp,
                    'IPLTYP': ipltyp,
                    'ICCOLN': iccoln,
                    'ICROWN': icrown,
                    'ICLAYN': iclayn}
        return SemperFormat(arg_dict)

    def save_to_unf(self, filename='semper.unf', skip_header=False):
        """Save a :class:`~.SemperFormat` to a file.

        Parameters
        ----------
        filename : string, optional
            The name of the unf-file to which the data should be written.
        skip_header : boolean, optional
            Determines if the header, title and label should be skipped (useful for some other
            programs). Default is False.

        Returns
        -------
        None

        """
        self._log.debug('Calling to_file')
        nlay, nrow, ncol = self.data.shape
        with open(filename, 'wb') as f:
            if not skip_header:
                # Create header:
                header = []
                header.extend(reversed(list(self.data.shape)))  # inverse order!
                header.append(self.iclass)
                header.append(self.iform)
                header.append(self.iversn*10000 + self.ilabel*1000 + len(self.title))
                if self.iformat is not None:
                    header.append(self.iformat)
                # Write header:
                f.write(struct.pack('I', 2*len(header)))  # record length, 4 byte format!
                for element in header:
                    f.write(struct.pack('h', element))  # 2 byte format!
                f.write(struct.pack('I', 2*len(header)))  # record length!
                # Write title:
                f.write(struct.pack('I', len(self.title)))  # record length, 4 byte format!
                f.write(self.title)
                f.write(struct.pack('I', len(self.title)))  # record length, 4 byte format!
                # Create label:
                if self.ilabel:
                    label = np.zeros(256, dtype=np.int32)
                    label[:6] = [ord(c) for c in 'Semper']
                    label[6:8] = divmod(ncol, 256)
                    label[8:10] = divmod(nrow, 256)
                    label[10:12] = divmod(nlay, 256)
                    label[12:14] = divmod(self.iccoln, 256)
                    label[14:16] = divmod(self.icrown, 256)
                    label[16:18] = divmod(self.iclayn, 256)
                    label[18] = self.iclass
                    label[19] = self.iform
                    label[20] = self.iwp
                    year, time = self.date.split(' ')
                    label[21:24] = map(int, year.split('-'))
                    label[21] -= 1900
                    label[24:27] = map(int, time.split(':'))
                    range_string = '{:.4g},{:.4g}'.format(self.data.min(), self.data.max())
                    ncrang = len(range_string)
                    label[27] = ncrang
                    label[28:28+ncrang] = [ord(c) for c in range_string]
                    label[55] = self.ipltyp
                    label[62] = 1  # Use real coords!
                    label[75:79] = [ord(c) for c in struct.pack('<f', self.scales[2])]  # DZ
                    label[79:83] = [ord(c) for c in struct.pack('<f', self.offsets[2])]  # Z0
                    label[83:87] = [ord(c) for c in struct.pack('<f', self.scales[1])]  # DY
                    label[87:91] = [ord(c) for c in struct.pack('<f', self.offsets[1])]  # Y0
                    label[91:95] = [ord(c) for c in struct.pack('<f', self.scales[0])]  # DX
                    label[95:99] = [ord(c) for c in struct.pack('<f', self.offsets[0])]  # X0
                    label[100:100+len(self.title)] = [ord(s) for s in self.title]
                    label[244:248] = [ord(c) for c in self.units[0]] + [0]*(4-len(self.units[0]))
                    label[248:252] = [ord(c) for c in self.units[1]] + [0]*(4-len(self.units[1]))
                    label[252:256] = [ord(c) for c in self.units[2]] + [0]*(4-len(self.units[2]))
                # Write label:
                if self.ilabel:
                    f.write(struct.pack('I', 2*256))  # record length, 4 byte format!
                    for element in label:
                        f.write(struct.pack('h', element))  # 2 byte format!
                    f.write(struct.pack('I', 2*256))  # record length!
            # Write picture data:
            for k in range(nlay):
                for j in range(nrow):
                    row = self.data[k, j, :]
                    factor = 8 if self.iform == 3 else 4  # complex numbers need more space!
                    f.write(struct.pack('I', factor*ncol))  # record length, 4 byte format!
                    if self.iform == 0:  # bytes:
                        raise Exception('Byte data is not supported! Use int, float or complex!')
                    elif self.iform == 1:  # int:
                        for element in row:
                            f.write(struct.pack('i', element))  # 4 bytes per data entry!
                    elif self.iform == 2:  # float:
                        for element in row:
                            f.write(struct.pack('f', element))  # 4 bytes per data entry!
                    elif self.iform == 3:  # complex:
                        for element in row:
                            f.write(struct.pack('f', element.real))  # 4 bytes per data entry!
                            f.write(struct.pack('f', element.imag))  # 4 bytes per data entry!
                    f.write(struct.pack('I', factor*ncol))  # record length, 4 byte format!

    @classmethod
    def from_signal(cls, signal):
        """Import a :class:`~.SemperFormat` object from a :class:`~hyperspy.signals.Signal` object.

        Parameters
        ----------
        signal: :class:`~hyperspy.signals.Signal`
            The signal which should be imported.

        Returns
        -------
        None

        """
        data = signal.data
        assert len(data.shape) <= 3, 'Only up to 3-dimensional datasets can be handled!'
        scales, offsets, units = [1., 1., 1.], [0., 0., 0.], ['', '', '']
        for i in range(len(data.shape)):
            scales[i] = signal.axes_manager[i].scale
            offsets[i] = signal.axes_manager[i].offset
            units[i] = signal.axes_manager[i].units
            if str(units[i]) == '<undefined>':
                units[i] = ''   # Traits can not be written!
        for i in range(3 - len(data.shape)):  # Make sure data is 3D!
            data = np.expand_dims(data, axis=0)
        iclass = cls.ICLASS_DICT_INV.get(signal.metadata.Signal.record_by, 6)  # 6: undefined
        if data.dtype.name == 'int8':
            iform = 0  # byte
        elif data.dtype.name in ['int16', 'int32']:
            iform = 1  # int
        elif data.dtype.name in ['float16', 'float32']:
            iform = 2  # float
        elif data.dtype.name == 'complex64':
            iform = 3  #
        else:
            raise TypeError('Data type not understood!')
        arg_dict = {'data': data,
                    'title': signal.metadata.General.as_dictionary().get('title', 'HyperSpy'),
                    'offsets': offsets,
                    'scales': scales,
                    'units': units,
                    'date': strftime('%y-%m-%d %H:%M:%S'),
                    'ICLASS': iclass,
                    'IFORM': iform,
                    'IVERSN': 2,  # current standard
                    'ILABEL': 1,  # True
                    'IFORMAT': None,  # not needed
                    'IWP': 0,  # seems standard
                    'IPLTYP': 248,  # seems standard
                    'ICCOLN': data.shape[2]//2 + 1,
                    'ICROWN': data.shape[1]//2 + 1,
                    'ICLAYN': data.shape[0]//2 + 1}
        return SemperFormat(arg_dict)

    def to_signal(self):
        """Export a :class:`~.SemperFormat` object to a :class:`~hyperspy.signals.Signal` object.

        Parameters
        ----------
        None

        Returns
        -------
        signal: :class:`~hyperspy.signals.Signal`
            The exported signal.

        """
        import hyperspy.api as hp
        data = np.squeeze(self.data)  # Reduce unneeded dimensions!
        if self.ICLASS_DICT[self.iclass] == 'spectrum':
            signal = hp.signals.Spectrum(data)
        elif self.ICLASS_DICT[self.iclass] == 'image':
            signal = hp.signals.Image(data)
        else:
            signal = hp.signals.Signal(data)
        for i in range(len(data.shape)):
            signal.axes_manager[i].name = {0: 'x', 1: 'y', 2: 'z'}[i]
            signal.axes_manager[i].scale = self.scales[i]
            signal.axes_manager[i].offset = self.offsets[i]
            signal.axes_manager[i].units = self.units[i]
        signal.metadata.General.title = self.title
        signal.original_metadata.add_dictionary({'date': self.date,
                                                 'ICLASS': self.iclass,
                                                 'IFORM': self.iform,
                                                 'IVERSN': self.iversn,
                                                 'ILABEL': self.ilabel,
                                                 'IFORMAT': self.iformat,
                                                 'IWP': self.iwp,
                                                 'IPLTYP': self.ipltyp,
                                                 'ICCOLN': self.iccoln,
                                                 'ICROWN': self.icrown,
                                                 'ICLAYN': self.iclayn})
        return signal

    def print_info(self):
        """Print important flag information of the :class:`.~SemperFormat` object.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        self._log.debug('Calling print_info')
        print '\n------------------------------------------------------'
        print self.title
        print self.date, '\n'
        print 'dimensions: x: {}, y: {}, z: {}'.format(*reversed(self.data.shape))
        print 'scaling:    x: {:.3g}, y: {:.3g}, z: {:.3g}'.format(*self.scales)
        print 'offsets:    x: {:.3g}, y: {:.3g}, z: {:.3g}'.format(*self.offsets)
        print 'units:      x: {}, y: {}, z: {}'.format(*self.units)
        print 'data range:', (self.data.min(), self.data.max()), '\n'
        print 'ICLASS: ', self.ICLASS_DICT[self.iclass]
        print 'IFORM : ', self.IFORM_DICT[self.iform]
        print 'IVERSN: ', self.iversn
        print 'ILABEL: ', self.ilabel == 1
        if self.iformat is not None:
            print 'IFORMAT:', self.iformat
        if self.ilabel:
            print 'IWP   : ', self.iwp
            print 'ICCOLN: ', self.iccoln
            print 'ICROWN: ', self.icrown
            print 'ICLAYN: ', self.iclayn
        print '------------------------------------------------------\n'


def file_reader(filename, print_info=False, **kwds):
    semper = SemperFormat.load_from_unf(filename)
    if print_info:
        semper.print_info()
    signal = semper.to_signal()
    axes = []
    for i in range(len(signal.data.shape)):
        axes.append({'size': signal.axes_manager[i].size,
                     'index_in_array': signal.axes_manager[i].index_in_array,
                     'name': signal.axes_manager[i].name,
                     'scale': signal.axes_manager[i].scale,
                     'offset': signal.axes_manager[i].offset,
                     'units': signal.axes_manager[i].units})
    dictionary = {'data': signal.data,
                  'axes': axes,
                  'metadata': signal.metadata.as_dictionary(),
                  'original_metadata': signal.original_metadata.as_dictionary()}
    return [dictionary]


def file_writer(filename, signal, **kwds):
    semper = SemperFormat.from_signal(signal)
    semper.save_to_unf(filename)
