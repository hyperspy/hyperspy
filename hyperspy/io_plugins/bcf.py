# -*- coding: utf-8 -*-
#
# Copyright 2015 Petras Jokubauskas
# Copyright 2015 The HyperSpy developers
#
# This library is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with any project and source this library is coupled.
# If not, see <http://www.gnu.org/licenses/>.
#
# This python library also provides basic reading capabilities of
# proprietary AidAim Software(tm) SFS (Single File System).


# Plugin characteristics
# ----------------------
format_name = 'bruker composite file bcf'
description = """the proprietary format used by Bruker's
Esprit(R) software to save hyppermaps together with 16bit SEM imagery,
EDS spectra and metadata describing the dimentions of the data and
SEM/TEM (limited) parameters"""
full_support = False
# Recognised file extension
file_extensions = ('bcf',)
default_extension = 0
# Reading capabilities
reads_images = True
reads_spectrum = True
reads_spectrum_image = True
# Writing capabilities
writes_images = False
writes_spectrum = False
writes_spectrum_image = False

import io

from lxml import objectify
import codecs
from datetime import datetime, timedelta
import numpy as np
from struct import unpack as strct_unp
import json
from skimage.measure import block_reduce

import sys
byte_order = sys.byteorder

# temporary statically assigned value, should be tied to debug if present...:
verbose = True

try:
    import unbcf_fast
    fast_unbcf = True
    if verbose:
        print("The fast cython based bcf unpacking library were found")
except ImportError:
    fast_unbcf = False
    if verbose:
        print("No fast bcf library present... ",
              "Falling back to python only backend")


class Container(object):
    pass


class SFSTreeItem(object):
    def __init__(self, item_raw_string, parent):
        self.sfs = parent
        self._pointer_to_pointer_table, self.size, create_time, \
        mod_time, some_time, self.permissions, \
        self.parent, _, self.is_dir, _, name, _ = strct_unp(
                       '<iQQQQIi176s?3s256s32s', item_raw_string)
        self.create_time = self._filetime_to_unix(create_time)
        self.mod_time = self._filetime_to_unix(mod_time)
        self.some_time = self._filetime_to_unix(some_time)
        self.name = name.strip(b'\x00').decode('utf-8')
        self.size_in_chunks = self._calc_pointer_table_size()
        if self.is_dir == 0:
            self._fill_pointer_table()

    def _calc_pointer_table_size(self):
        n_chunks = -(-self.size // self.sfs.usable_chunk)
        return n_chunks

    def _filetime_to_unix(self, time):
        """return recalculated windows filetime to unix time"""
        return datetime(1601, 1, 1) + timedelta(microseconds=time / 10)

    def __repr__(self):
        return '<SFS internal file {0:.2f} MB>'.format(self.size / 1048576)

    def _fill_pointer_table(self):
        #table size in number of chunks
        n_of_chunks = -(-self.size_in_chunks //
                       (self.sfs.usable_chunk // 4))
        with open(self.sfs.filename, 'rb') as fn:
            if n_of_chunks > 1:
                next_chunk = self._pointer_to_pointer_table
                temp_string = io.BytesIO()
                for j in range(0, n_of_chunks, 1):
                    fn.seek(self.sfs.chunksize * next_chunk + 0x118)
                    next_chunk = strct_unp('<I', fn.read(4))[0]
                    fn.seek(28, 1)
                    temp_string.write(fn.read(self.sfs.usable_chunk))
                temp_string.seek(0)
                temp_table = temp_string.read()
                temp_string.close()
            else:
                fn.seek(self.sfs.chunksize *
                        self._pointer_to_pointer_table + 0x138)
                temp_table = fn.read(self.sfs.usable_chunk)
            self.pointers = np.fromstring(temp_table[:self.size_in_chunks * 4],
                                          dtype='uint32').astype(np.int64) *\
                                                   self.sfs.chunksize + 0x138

    def read_piece(self, offset, length):
        """reads and returns raw byte string of the file. It do not do
        any decompression if stream is compressed and have to be done
        with other functions.

        requires two arguments:
        offset -- seek value
        lenght -- lenght of the data counting from the offset

        It may have some significant overhead compared to method
        (get_as_BytesIO_string) of loading whole file.
        """
        data = io.BytesIO()
        #first block index:
        fb_idx = offset // self.sfs.usable_chunk
        #first block offset:
        fbo = offset % self.sfs.usable_chunk
        #last block index:
        lb_idx = (offset + length) // self.sfs.usable_chunk
        #last block cut off:
        lbco = (offset + length) % self.sfs.usable_chunk
        with open(self.sfs.filename, 'rb') as fn:
            if fb_idx != lb_idx:
                fn.seek(self.pointers[fb_idx] + fbo)
                data.write(fn.read(self.sfs.usable_chunk - fbo))
                for i in self.pointers[fb_idx + 1:lb_idx]:
                    fn.seek(i)
                    data.write(fn.read(self.sfs.usable_chunk))
                if lbco > 0:
                    fn.seek(self.pointers[lb_idx])
                    data.write(fn.read(lbco))
            else:
                fn.seek(self.pointers[fb_idx] + fbo)
                data.write(fn.read(length))
        data.seek(0)
        return data.read()

    def iter_read_chunks(self, first=0, chunks=False):
        """Generate and return iterator reading and returning chunks
        of file. By default it creates iterator for all chunks, however
        with kwargs 'first' and 'chunks' the range of chunks
        for iterator can be set
        """
        if not chunks:
            last = self.size_in_chunks
        else:
            last = chunks + first
        with open(self.sfs.filename, 'rb') as fn:
            for idx in range(first, last - 1):
                fn.seek(self.pointers[idx])
                yield fn.read(self.sfs.usable_chunk)
            fn.seek(self.pointers[last - 1])
            if last == self.size_in_chunks:
                last_stuff = self.size % self.sfs.usable_chunk
                if last_stuff != 0:
                    yield fn.read(last_stuff)
                else:
                    yield fn.read(self.sfs.usable_chunk)
            else:
                yield fn.read(self.sfs.usable_chunk)

    def setup_compression_metadata(self):
        """setup the number of chunks and uncompressed size as class
        atributes"""
        with open(self.sfs.filename, 'rb') as fn:
            fn.seek(self.pointers[0])
            #AACS signature, uncompressed size, undef var, number of blocks
            aacs, uc_size, _, n_of_blocks = strct_unp('<IIII', fn.read(16))
        if aacs == 0x53434141:
            self.uncompressed_blk_size = uc_size
            self.no_of_compr_blk = n_of_blocks
        else:
            raise ValueError("""The file is marked to be compressed,
but compression signature is missing in the header. Aborting....""")

    def iter_read_compr_chunks(self):
        """generate and return iterative reader for compressed file with
        zlib or bzip2 compression, where iterator returns uncompressed
        data in chunks as iterator.
        """
        if self.sfs.compression == 'zlib':
            from zlib import decompress as unzip_block
        else:
            from bzip2 import decompress as unzip_block  # lint:ok
        offset = 0x80  # the 1st compression block header
        for dymmy1 in range(self.no_of_compr_blk):
            cpr_size, _uncpr_size, _unknwn, _dummy_size = strct_unp('<IIII',
                                                  self.read_piece(offset, 16))
            #_unknwn is probably some kind of checksum but non
            # known (crc16, crc32, adler32) algorithm could match.
            # _dummy_size == cpr_size + 0x10 which have no use...
            offset += 16
            raw_string = self.read_piece(offset, cpr_size)
            offset += cpr_size
            yield unzip_block(raw_string)

    def get_as_BytesIO_string(self):
        if self.sfs.compression == 'None':
            data = io.BytesIO()
            data.write(b''.join(self.iter_read_chunks()))
            return data
        elif self.sfs.compression in ('zlib', 'bzip2'):
            data = io.BytesIO()
            data.write(b''.join(self.iter_read_compr_chunks()))
            return data
        else:
            raise RuntimeError('file',
                               str(self.sfs.filename),
                               ' is compressed by not known and not',
                               'implemented algorithm.',
                               'Aborting...')


class SFS_reader(object):
    def __init__(self, filename):
        self.filename = filename
        self.hypermap = {}
        with open(filename, 'rb') as fn:
            a = fn.read(8)
            if a != b'AAMVHFSS':
                raise TypeError(
                    "file '{0}' is not SFS container".format(filename))
            fn.seek(0x124)  # this looks to be version, as float value is always
            # nicely rounded and at older bcf versions (<1.9) it was 2.40,
            # at new (v2) - 2.60
            version, self.chunksize = strct_unp('<fI', fn.read(8))
            self.sfs_version = '{0:4.2f}'.format(version)
            self.usable_chunk = self.chunksize - 32
            fn.seek(0x140)

            #the sfs tree and number of the items / files + directories in it,
            #and the number in chunks of whole sfs:
            tree_address, n_file_tree_items, self.sfs_n_of_chunks = strct_unp(
                                                           '<III', fn.read(12))
            #check if file tree do not exceed one chunk:
            n_file_tree_chunks = -((-n_file_tree_items * 0x200) //
                                             (self.chunksize - 512))
            if n_file_tree_chunks is 1:
                fn.seek(self.chunksize * tree_address + 0x138)
                raw_tree = fn.read(0x200 * n_file_tree_items)
            else:
                temp_str = io.BytesIO()
                for i in range(n_file_tree_chunks):
                    # jump to tree/list address:
                    fn.seek(self.chunksize * tree_address + 0x118)
                    # next tree/list address:
                    tree_address = strct_unp('<I', fn.read(4))[0]
                    fn.seek(28, 1)
                    temp_str.write(fn.read(self.chunksize - 512))
                temp_str.seek(0)
                raw_tree = temp_str.read(n_file_tree_items * 0x200)
                temp_str.close()
            # setting up virtual file system in python dictionary
            temp_item_list = [SFSTreeItem(raw_tree[i * 0x200:(i + 1) * 0x200],
                                       self) for i in range(n_file_tree_items)]
            paths = [[h.parent] for h in temp_item_list]
            #Find if there is compression:
            for c in temp_item_list:
                if not c.is_dir:
                    fn.seek(c.pointers[0])
                    if fn.read(4) == b'\x41\x41\x43\x53':  # string AACS
                        fn.seek(0x8C, 1)
                        compression_head = fn.read(2)
                        byte_one = strct_unp('BB', compression_head)[0]
                        if byte_one == 0x78:
                            self.compression = 'zlib'
                        elif compression_head == b'\x42\x5A':
                            self.compression = 'bzip2'
                        else:
                            self.compression = 'unknown'
                    else:
                        self.compression = 'None'
                    break  # compression is global, can't be per file in sfs
            #checking the compression header which can be different per file:
            if self.compression in ('zlib', 'bzip2'):
                for c in temp_item_list:
                    if not c.is_dir:
                        c.setup_compression_metadata()

            while not all(g[-1] == -1 for g in paths):
                for f in range(len(paths)):
                    if paths[f][-1] != -1:
                        paths[f].extend(paths[paths[f][-1]])
            names = [j.name for j in temp_item_list]
            names.append('root')
            for p in paths:
                for r in range(len(p)):
                    p[r] = names[p[r]]
            for p in paths:
                p.reverse()
            root = {}
            for i in range(len(temp_item_list)):
                dir_pointer = root
                for j in paths[i]:
                    if j in dir_pointer:
                        dir_pointer = dir_pointer[j]
                    else:
                        dir_pointer[j] = {}
                        dir_pointer = dir_pointer[j]
                if temp_item_list[i].is_dir:
                    dir_pointer[temp_item_list[i].name] = {}
                else:
                    dir_pointer[temp_item_list[i].name] = temp_item_list[i]
            # and finaly Virtual file system:
            self.vfs = root['root']

    def print_file_tree(self):
        tree = json.dumps(self.vfs, sort_keys=True, indent=4, default=str)
        print(tree)

    def get_file(self, path):
        item = self.vfs
        try:
            for i in path.split('/'):
                item = item[i]
            return item
        except KeyError:
            print("""Check the requested path!
There is no such file or folder in this single file system.""")


class EDXSpectrum(object):
    def __init__(self, spectrum):
        """Wrap the objectified bruker EDS spectrum xml part
        to the python object, leaving all the xml and bruker clutter behind
        """
        if str(spectrum.attrib['Type']) != 'TRTSpectrum':
            raise IOError
        try:
            self.realTime = int(spectrum.TRTHeaderedClass.ClassInstance.RealTime)
            self.lifeTime = int(spectrum.TRTHeaderedClass.ClassInstance.LifeTime)
            self.deadTime = int(spectrum.TRTHeaderedClass.ClassInstance.DeadTime)
        except AttributeError:
            if verbose:
                print('spectrum have no dead time records...')
            else:
                pass
        self.zeroPeakPosition = int(spectrum.TRTHeaderedClass.ClassInstance.ZeroPeakPosition)
        self.amplification = int(spectrum.TRTHeaderedClass.ClassInstance.Amplification)
        self.shapingTime = int(spectrum.TRTHeaderedClass.ClassInstance.ShapingTime)
        self.detectorType = str(spectrum.TRTHeaderedClass.ClassInstance[1].Type)
        self.hv = float(spectrum.TRTHeaderedClass.ClassInstance[2].PrimaryEnergy)
        self.elevationAngle = float(spectrum.TRTHeaderedClass.ClassInstance[2].ElevationAngle)
        self.azimutAngle = float(spectrum.TRTHeaderedClass.ClassInstance[2].AzimutAngle)
        self.calibAbs = float(spectrum.ClassInstance[0].CalibAbs)
        self.calibLin = float(spectrum.ClassInstance[0].CalibLin)
        self.chnlCnt = int(spectrum.ClassInstance[0].ChannelCount)
        self.date = str(spectrum.ClassInstance[0].Date)
        self.time = str(spectrum.ClassInstance[0].Time)
        self.data = np.fromstring(str(spectrum.Channels), dtype='Q', sep=",")
        self.energy = np.arange(self.calibAbs,
                           self.calibLin * self.chnlCnt + self.calibAbs,
                           self.calibLin)  # the x axis for ploting spectra

    def energy_to_channel(self, energy, kV=True):
        """ convert energy to channel index,
        optional kwarg 'kV' (default: True) should be set to False
        if given energy units is in V"""
        if not kV:
            en_temp = energy / 1000.
        else:
            en_temp = energy
        return int(round((en_temp - self.calibAbs) / self.calibLin))

    def channel_to_energy(self, channel, kV=True):
        """convert given channel index to energy,
        optional kwarg 'kV' (default: True) decides if returned value
        is in kV or V"""
        if not kV:
            kV = 1000
        else:
            kV = 1
        return (channel * self.calibLin + self.calibAbs) * kV


class HyperHeader(object):
    """Wrap Bruker HyperMaping xml header into python object.
    For instantionion have to be provided with extracted Header xml from bcf.
    If Bcf is version 2, the bcf can contain stacks
    of hypermaps - thus header part contains sum eds spectras and it's
    metadata per hypermap slice.
    Bcf can record number of imagery from different
    imagining detectors (BSE, SEI, ARGUS, etc...): access to imagery is throught
    image index.
    """
    def __init__(self, xml_str):
        # Due to Delphi(TM) xml implementation literaly shits into xml,
        # we need lxml parser to be more forgiving (recover=True):
        oparser = objectify.makeparser(recover=True)
        root = objectify.fromstring(xml_str, parser=oparser).ClassInstance
        try:
            self.name = str(root.attrib['Name'])
        except AttributeError:
            self.name = 'Undefinded'
        self.datetime = datetime.strptime(' '.join([str(root.Header.Date),
                                                    str(root.Header.Time)]),
                                          "%d.%m.%Y %H:%M:%S")
        self.version = int(root.Header.FileVersion)
        semData = root.xpath("ClassInstance[@Type='TRTSEMData']")[0]
        #create containers:
        self.sem = Container()
        self.stage = Container()
        self.image = Container()
        # sem acceleration voltage, working distance, magnification:
        self.sem.hv = float(semData.HV)  # in kV
        self.sem.wd = float(semData.WD)  # in mm
        self.sem.mag = float(semData.Mag)  # in times
        # image/hypermap resolution in um/pixel:
        self.image.x_res = float(semData.DX)
        self.image.y_res = float(semData.DY)
        semStageData = root.xpath("ClassInstance[@Type='TRTSEMStageData']")[0]
        # stage position data in um cast to m (that data anyway is not used
        # by hyperspy):
        try:
            self.stage.x = float(semStageData.X) / 1.0e6
            self.stage.y = float(semStageData.Y) / 1.0e6
        except AttributeError:
            self.stage.x = self.stage.y = None
        try:
            self.stage.z = float(semStageData.Z) / 1.0e6
        except AttributeError:
            self.stage.z = None
        try:
            self.stage.rotation = float(semStageData.Rotation)  # in degrees
        except AttributeError:
            self.stage.rotation = None
        DSPConf = root.xpath("ClassInstance[@Type='TRTDSPConfiguration']")[0]
        self.stage.tilt_angle = float(DSPConf.TiltAngle)
        imageData = root.xpath("ClassInstance[@Type='TRTImageData']")[0]
        self.image.width = int(imageData.Width)  # in pixels
        self.image.height = int(imageData.Height)  # # in pixels
        self.image.plane_count = int(imageData.PlaneCount)
        self.multi_image = int(imageData.MultiImage)
        self.image.images = []
        for i in range(self.image.plane_count):
            img = imageData.xpath("Plane" + str(i))[0]
            raw = codecs.decode((img.Data.text).encode('ascii'), 'base64')
            array1 = np.fromstring(raw, dtype=np.uint16)
            if any(array1):
                temp_img = Container()
                temp_img.data = array1.reshape((self.image.height,
                                                self.image.width))
                temp_img.detector_name = str(img.Description.text)
                self.image.images.append(temp_img)
        self.elements = []
        try:
            elements = root.xpath(
               "ClassInstance[@Type='TRTContainerClass']/ChildClassInstances" +
               "/ClassInstance[@Type='TRTElementInformationList']" +
               "/ClassInstance[@Type='TRTSpectrumRegionList']" +
               "/ChildClassInstances")[0]
            for j in elements.xpath("ClassInstance[@Type='TRTSpectrumRegion']"):
                self.elements.append(int(j.Element))
        except IndexError:
            if verbose:
                print('no element selection present..')
            else:
                pass
        self.line_counter = np.fromstring(str(root.LineCounter),
                                          dtype=np.uint16, sep=',')
        self.channel_count = int(root.ChCount)
        self.mapping_count = int(root.DetectorCount)
        self.channel_factors = {}
        self.spectra_data = {}
        for i in range(self.mapping_count):
            self.channel_factors[i] = int(root.xpath("ChannelFactor" +
                                                                    str(i))[0])
            self.spectra_data[i] = EDXSpectrum(root.xpath("SpectrumData" +
                                                       str(i))[0].ClassInstance)

    def estimate_map_channels(self, index=0):
        """estimate minimal size of array that any pixel of spectra would
        not be truncated.
        args:
        index -- index of the map if multiply hypermaps are present
        in the same bcf.
        returns:
        maximum non empty channel +1 what equals to the needed size of
        final array depth (energy).
        """

        sum_eds = self.spectra_data[index].data
        return sum_eds.nonzero()[0][-1] + 1  # +1: the number not the index

    def estimate_map_depth(self, index=0):
        """estimate minimal dtype of array from the cumulative spectra
        of the all pixels so that none would be truncated.
        args:
        index -- index of the map channels if multiply hypermaps are
        present in the same bcf.
        returns:
        numpy dtype large enought to use in final hypermap numpy array.

        The method estimates the value from sum eds spectra, dividing
        the maximum value from raster width and hight and to be on the
        safe side multiplying by 2.
        """
        sum_eds = self.spectra_data[index].data
        #the most intensive peak is Bruker reference peak at 0kV:
        roof = np.max(sum_eds) // self.image.width // self.image.height * 2
        if roof > 0xFF:
            if roof > 0xFFFF:
                depth = np.uint32
            else:
                depth = np.uint16
        else:
            depth = np.uint8
        return depth

    def get_spectra_metadata(self, index=0):
        """return objectified xml with spectra metadata"""
        return self.spectra_data[index]


st = {1: 'B', 2: 'B', 4: 'H', 8: 'I', 16: 'Q'}



def bin_to_numpy(data, pointers, max_channels, depth):
    """unpack the delphi/bruker binary hypermap and returns
    nupy array.

    -----------
    Arguments:
    data -- object of io.ByteIO with binary string of bruker
     SpectrumData0 (or 1,2,...)
    pointers -- list with offsets pointing to SpectrumData0
     for every pixel, obtainable by spect_pos_from_file, or
     bin_to_spect_pos functions for ver.2 and ver. 1 bcf
    max_channels -- number of channels in the mapping:
        typically 1024 or 2048....
    depth -- the dtype. Should be chosen so that max peaks
     would not be truncated.
       (for low intensity 'uit8' is enought, for longer
        mappings even 'uint16' and more)
    """
    data.seek(0)
    height, width = strct_unp('<ii', data.read(8))  # +8
    total_pixels = height * width
    total_channels = total_pixels * max_channels
    #hyper map as very flat array:
    vfa = np.zeros(total_channels, dtype=depth)
    # some tricks for decoding 12bit data:
    if byte_order == 'little':
        trick_dtype = '>u2'  # it is not error, it is intended to be oposite
    else:
        trick_dtype = '<u2'
    for pix in range(0, total_pixels, 1):
        if pointers[pix] > 0:
            data.seek(pointers[pix])
            #_1 and _2 dummy - throwaway
            #_data_size1 - sometimes is equal to data_size2, sometimes 0
            chan1, chan2, _1, flag, _data_size1, n_of_pulses, data_size2, _2 =\
                                        strct_unp('<HHIHHHHH', data.read(18))
            if flag == 1:  # and (chan1 != chan2)
                #Unpack packed 12-bit data to 16-bit uints:
                data1 = data.read(data_size2)
                switched_i2 = np.fromstring(data1, dtype=trick_dtype)
                data2 = np.fromstring(switched_i2.tostring(),
                                      dtype=np.uint8
                                     ).repeat(2)
                mask = np.ones_like(data2, dtype=bool)
                mask[0::6] = mask[5::6] = False
                # Reinterpret expanded as 16-bit
                # as the string, array will have always big-endian represent'n
                exp16 = np.fromstring(data2[mask].tostring(),
                                      dtype='>u2', count=n_of_pulses)
                exp16[0::2] >>= 4             # Shift every second short by 4
                exp16 &= np.uint16(0x0FFF)    # Mask upper 4-bits on all shorts
                pixel = np.bincount(exp16, minlength=chan1 - 1)
            else:
                #Unpack instructively packed data to pixel channels:
                offset = 0
                pixel = []
                while offset < data_size2 - 4:
                    #this would work on py3
                    #size, channels = data.read(2)
                    # this is needed on py2:
                    size, channels = strct_unp('<BB', data.read(2))
                    if size == 0:
                        pixel += channels * [0]
                        offset += 2
                    else:
                        addition = strct_unp('<' + st[size * 2],
                                             data.read(size))[0]
                        if size == 1:
                            # special case with nibble switching
                            lenght = -(-channels // 2)
                            #a = list(data.read(lenght))  # valid py3 code
                            #this have to be used on py2
                            a = strct_unp('<' + 'B' * lenght, data.read(lenght))
                            g = []
                            for i in a:
                                g += (i & 0x0F) + addition, (i >> 4) + addition
                            pixel += g[:channels]
                        else:
                            lenght = int(channels * size / 2)
                            temp = strct_unp('<' + channels * st[size],
                                             data.read(lenght))
                            pixel += [l + addition for l in temp]
                        offset += 2 + size + lenght
                if chan2 < chan1:
                    rest = chan1 - chan2
                    #pixel += [0 for i in range(0,rest,1)]
                    pixel += rest * [0]
                # additional data size:
                if n_of_pulses > 0:
                    add_s = strct_unp('<I', data.read(4))[0]
                    # the additional pulses:
                    thingy = strct_unp('<' + 'H' * n_of_pulses,
                                       data.read(add_s))
                    for i in thingy:
                        pixel[i] += 1
            vfa[0 + max_channels * pix:chan1 + max_channels * pix] = pixel
    vfa.resize((height, width, max_channels))
    return vfa.swapaxes(2, 0)


def spect_pos_from_file(sp_data):
    """read and return the pointer table of the pixels
    as iterable. (intended to BCF v2!!!)
    args:
        BytesIO object of the pixel pointer table file
         in the bcf.
    The function is supposed to be applied upon
    version2 bcf, where such table of pixels is
    already precalculated.
    """
    sp_data.seek(0)
    height, width = strct_unp('<ii', sp_data.read(8))
    n_of_pix = height * width
    mapingpointers2 = strct_unp('<' + 'q' * n_of_pix,
                                    sp_data.read())
    return mapingpointers2


def bin_to_spect_pos(data):
    """parses whole data stream and creates iterable
    with pixel offsets/pointers pointing to SpectrumData
    file inside bruker bcf container. (intended BCF v1)
    Such table is presented in bcf version 2, but is not
    in version 1.

    Arguments:
    data -- io.BytesIO string with data of SpectrumData*
    Returns iterable
    """
    data.seek(0)
    height, width = strct_unp('<ii', data.read(8))
    n_of_pix = height * width
    data.seek(0x1A0)
    # create the list with all values -1
    mapping_pointers = [-1] * n_of_pix
    #mapping_pointers = np.full(n_of_pix, -1, dtype=np.int64)
    for line_cnt in range(height):
        #line_head contains number of non-empty pixels in line
        line_head = strct_unp('<i', data.read(4))[0]
        for dummy1 in range(line_head):
            #x_index of the pixel:
            x_pix = strct_unp('<i', data.read(4))[0] + width * line_cnt
            offset = data.tell()
            mapping_pointers[x_pix] = offset
            # skip channel number and some placeholder:
            data.seek(8, 1)
            flag, _data_size1, n_of_pulses, data_size2 = strct_unp(
                                                 '<HHHH', data.read(8))
            data.seek(2, 1)  # always 0x0000
            # depending to packing type (flag) do:
            if flag == 1:  # and (chan1 != chan2)
                data.seek(data_size2, 1)  # skip to next pixel/line
            else:
                if n_of_pulses > 0:
                    data.seek(data_size2 - 4, 1)  # skip to pulses size
                    #additional pulses for data with flag 2 and 3:
                    add_s = strct_unp('<i', data.read(4))[0]
                    data.seek(add_s, 1)
                else:  # if there is no addition pulses, jump to
                    data.seek(data_size2, 1)  # next pixel or line
    return mapping_pointers


class BCF_reader(SFS_reader):
    def __init__(self, filename):
        SFS_reader.__init__(self, filename)
        header_file = self.get_file('EDSDatabase/HeaderData')
        header_byte_str = header_file.get_as_BytesIO_string().getvalue()
        self.header = HyperHeader(header_byte_str)

    def parse_hyper_map(self, index=0):
        """ return the numpy array from given bcf file for given slice

        Arguments:
        filename -- bcf file name/path
        version -- version of bcf (1 or 2)
        max_channels -- number of channels to fit last non zero channel
          from sum of all spectras.
        depth -- numpy dtype sufficient to hold the data.
        index -- the slice of hyppermap in the bcf (casualy there is just
          1, so default value of index is 0)

        returns numpy array
        """
        ind = index
        data = self.get_file('EDSDatabase/SpectrumData' +
                                              str(ind)).get_as_BytesIO_string()
        max_channels = self.header.estimate_map_channels(index=ind)
        depth = self.header.estimate_map_depth(index=ind)
        if self.header.version == 1:
            pointers = bin_to_spect_pos(data)
        else:
            sp_data = self.get_file('EDSDatabase/SpectrumPositions' +
                                              str(ind)).get_as_BytesIO_string()
            pointers = spect_pos_from_file(sp_data)

        return bin_to_numpy(data, pointers, max_channels, depth)

    def _parse_line_positions(self, index=0):
        ind = index
        data = self.get_file('EDSDatabase/SpectrumData' +
                                              str(ind)).get_as_BytesIO_string()
        pointers = bin_to_spect_pos(data)
        return pointers

    def persistent_parse_hypermap(self, index=0, downsample=None):
        """parse and assign the hypermap to the Hypermap python object"""
        ind = index
        dwn = downsample
        self.hypermap[ind] = HyperMap(self.parse_hyper_map(index=ind),
                                      self, downsample=dwn, index=ind)


class HyperMap(object):
    def __init__(self, nparray, parent, index=0, downsample=None):
        ind = index
        sp_meta = parent.header.get_spectra_metadata(index=ind)
        self.calib_abs = sp_meta.calibAbs
        self.calib_lin = sp_meta.calibLin
        self.xcalib = parent.header.image.x_res
        self.ycalib = parent.header.image.y_res
        if downsample and type(downsample) == int:
            self.hypermap = block_reduce(nparray,
                (1, downsample, downsample), func=np.sum)
        else:
            self.hypermap = nparray