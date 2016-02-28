# -*- coding: utf-8 -*-
#
# Copyright 2016 Petras Jokubauskas
# Copyright 2016 The HyperSpy developers
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
# This python library subset provides read functionality of
#  Bruker bcf files.
# The basic reading capabilities of proprietary AidAim Software(tm)
#  SFS (Single File System) (used in bcf technology) is present in
#  the same library.


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

# temporary statically assigned value, should be tied to debug if present...:
verbose = True

try:
    from . import unbcf_fast
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
        """Return recalculated windows filetime to unix time."""
        return datetime(1601, 1, 1) + timedelta(microseconds=time / 10)

    def __repr__(self):
        return '<SFS internal file {0:.2f} MB>'.format(self.size / 1048576)

    def _fill_pointer_table(self):
        #table size in number of chunks:
        n_of_chunks = -(-self.size_in_chunks //
                       (self.sfs.usable_chunk // 4))
        with open(self.sfs.filename, 'rb') as fn:
            if n_of_chunks > 1:
                next_chunk = self._pointer_to_pointer_table
                temp_string = io.BytesIO()
                for dummy1 in range(n_of_chunks):
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
        """Read and returns raw byte string of the file without
        any decompression.

        Arguments:
        ----------
        offset: seek value
        length: length of the data counting from the offset

        Returns:
        ----------
        io.ByteIO object
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

    def _iter_read_chunks(self, first=0, chunks=False):
        """Generate and return iterator for reading and returning
        sfs internal file in chunks.
        By default it creates iterator for whole file, however
        with kwargs 'first' and 'chunks' the range of chunks
        for iterator can be set.
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
        """
        Setup the number of chunks and uncompressed size as class
        atributes.
        """
        with open(self.sfs.filename, 'rb') as fn:
            fn.seek(self.pointers[0])
            #AACS signature, uncompressed size, undef var, number of blocks
            aacs, uc_size, _, n_of_blocks = strct_unp('<IIII', fn.read(16))
        if aacs == 0x53434141:  # AACS as string
            self.uncompressed_blk_size = uc_size
            self.no_of_compr_blk = n_of_blocks
        else:
            raise ValueError("""The file is marked to be compressed,
but compression signature is missing in the header. Aborting....""")

    def _iter_read_larger_chunks(self, chunk_size=524288):
        """
        Generate and return iterator for reading
        the raw data in sensible sized chunks.
        default chunk size = 524288 bytes (0.5MB)
        """
        chunks = -(-self.size // chunk_size)
        last_chunk = self.size % chunk_size
        offset = 0
        for dummy1 in range(chunks - 1):
            raw_string = self.read_piece(offset, chunk_size)
            offset += chunk_size
            yield raw_string
        if last_chunk != 0:
            raw_string = self.read_piece(offset, last_chunk)
        else:
            raw_string = self.read_piece(offset, chunk_size)
        yield raw_string

    def _iter_read_compr_chunks(self):
        """Generate and return iterator for compressed file with
        zlib or bzip2 compression, where iterator returns uncompressed
        data in chunks as iterator.
        """
        if self.sfs.compression == 'zlib':
            from zlib import decompress as unzip_block
        else:
            from bzip2 import decompress as unzip_block  # lint:ok
        offset = 0x80  # the 1st compression block header
        for dummy1 in range(self.no_of_compr_blk):
            cpr_size, dummy_size, dummy_unkn, dummy_size2 = strct_unp('<IIII',
                                                  self.read_piece(offset, 16))
            # dummy_unkn is probably some kind of checksum but non
            # known (crc16, crc32, adler32) algorithm could match.
            # dummy_size2 == cpr_size + 0x10 which have no use...
            # dummy_size, which is decompressed size, also have no use...
            # as it is the same in file compression_header
            offset += 16
            raw_string = self.read_piece(offset, cpr_size)
            offset += cpr_size
            yield unzip_block(raw_string)

    def get_iter_and_properties(self, larger_chunks=False):
        """Get the the iterator and properties of its chunked size and
        number of chunks for compressed or not compressed data
        accordingly.
        ----------
        Returns:
            (iterator, chunk_size, number_of_chunks)
        """
        if self.sfs.compression == 'None':
            if not larger_chunks:
                return self._iter_read_chunks(), self.sfs.usable_chunk,\
                   self.size_in_chunks
            else:
                return self._iter_read_larger_chunks(chunk_size=larger_chunks),\
                    larger_chunks, -(-self.size // larger_chunks)
        elif self.sfs.compression in ('zlib', 'bzip2'):
            return self._iter_read_compr_chunks(), self.uncompressed_blk_size,\
                   self.no_of_compr_blk
        else:
            raise RuntimeError('file', str(self.sfs.filename),
                               ' is compressed by not known and not',
                               'implemented algorithm.\n Aborting...')

    def get_as_BytesIO_string(self):
        """Get the whole file as io.BytesIO object (in memory!)."""
        data = io.BytesIO()
        data.write(b''.join(self.get_iter_and_properties()[0]))
        return data


class SFS_reader(object):
    def __init__(self, filename):
        self.filename = filename
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
            self.tree_address, self.n_tree_items, self.sfs_n_of_chunks =\
                                                 strct_unp('<III', fn.read(12))
        self._setup_vfs()

    def _setup_vfs(self):
        with open(self.filename, 'rb') as fn:
            #check if file tree do not exceed one chunk:
            n_file_tree_chunks = -((-self.n_tree_items * 0x200) //
                                             (self.chunksize - 512))
            if n_file_tree_chunks is 1:
                fn.seek(self.chunksize * self.tree_address + 0x138)
                raw_tree = fn.read(0x200 * self.n_tree_items)
            else:
                temp_str = io.BytesIO()
                for i in range(n_file_tree_chunks):
                    # jump to tree/list address:
                    fn.seek(self.chunksize * self.tree_address + 0x118)
                    # next tree/list address:
                    self.tree_address = strct_unp('<I', fn.read(4))[0]
                    fn.seek(28, 1)
                    temp_str.write(fn.read(self.chunksize - 512))
                temp_str.seek(0)
                raw_tree = temp_str.read(self.n_tree_items * 0x200)
                temp_str.close()
            # temp flat list of items:
            temp_item_list = [SFSTreeItem(raw_tree[i * 0x200:(i + 1) * 0x200],
                                       self) for i in range(self.n_tree_items)]
            # temp list with parents of items
            paths = [[h.parent] for h in temp_item_list]
        #checking the compression header which can be different per file:
        self._check_the_compresion(temp_item_list)
        if self.compression in ('zlib', 'bzip2'):
            for c in temp_item_list:
                if not c.is_dir:
                    c.setup_compression_metadata()
        # Shufling items from flat list into dictionary tree:
        while not all(g[-1] == -1 for g in paths):
            for f in range(len(paths)):
                if paths[f][-1] != -1:
                    paths[f].extend(paths[paths[f][-1]])
        names = [j.name for j in temp_item_list]
        names.append('root')  # temp root item in dictionary
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

    def _check_the_compresion(self, temp_item_list):
        """check the compression and set the compression
        attrib accordingly"""
        with open(self.filename, 'rb') as fn:
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
                    # compression is global, can't be diferent per file in sfs
                    break

    def print_file_tree(self):
        """print the internal file/dir tree of sfs container
        as json string
        """
        tree = json.dumps(self.vfs, sort_keys=True, indent=4, default=str)
        print(tree)

    def get_file(self, path):
        """Return the SFSTreeItem (aka internal file) object from
        sfs container.

        Arguments:
        ---------
        path: internal file path in sfs file tree. Path accepts only
            standard - forward slash for directories.

        Returns:
        ---------
        object (SFSTreeItem), which can be read into byte stream, in
        chunks or whole using objects methods.

        example:
        ---------
        to get "file" object 'kitten.png' in folder 'catz' which
        resides in root directory of sfs, you would use:

        >>> instance_of_SFSReader.get_file('catz/kitten.png')
        """
        item = self.vfs
        try:
            for i in path.split('/'):
                item = item[i]
            return item
        except KeyError:
            print("""Check the requested path!
There is no such file or folder in this single file system.
Try printing out the file tree with print_file_tree method""")


class EDXSpectrum(object):
    def __init__(self, spectrum):
        """
        Wrap the objectified bruker EDS spectrum xml part
        to the python object, leaving all the xml and bruker clutter behind
        """
        if str(spectrum.attrib['Type']) != 'TRTSpectrum':
            raise IOError('Not valid objectified xml passed',
                          ' to Bruker EDXSpectrum class')
        try:
            self.realTime = int(
                            spectrum.TRTHeaderedClass.ClassInstance.RealTime)
            self.lifeTime = int(
                            spectrum.TRTHeaderedClass.ClassInstance.LifeTime)
            self.deadTime = int(
                            spectrum.TRTHeaderedClass.ClassInstance.DeadTime)
        except AttributeError:
            if verbose:
                print('spectrum have no dead time records...')
            else:
                pass
        self.zeroPeakPosition = int(
                      spectrum.TRTHeaderedClass.ClassInstance.ZeroPeakPosition)
        self.amplification = int(
                      spectrum.TRTHeaderedClass.ClassInstance.Amplification)
        self.shapingTime = int(
                      spectrum.TRTHeaderedClass.ClassInstance.ShapingTime)
        self.detectorType = str(spectrum.TRTHeaderedClass.ClassInstance[1].Type)
        self.hv = float(
                      spectrum.TRTHeaderedClass.ClassInstance[2].PrimaryEnergy)
        self.elevationAngle = float(
                      spectrum.TRTHeaderedClass.ClassInstance[2].ElevationAngle)
        self.azimutAngle = float(
                      spectrum.TRTHeaderedClass.ClassInstance[2].AzimutAngle)
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
    For instantionion have to be provided with extracted Header xml
    from bcf.
    If Bcf is version 2, the bcf can contain stacks
    of hypermaps - thus header part contains sum eds spectras and it's
    metadata per hypermap slice.
    Bcf can record number of imagery from different
    imagining detectors (BSE, SEI, ARGUS, etc...): access to imagery
    is throught image index.
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
        #create containers:
        self.sem = Container()
        self.stage = Container()
        self.image = Container()
        #fill the sem and stage attributes:
        self._set_sem(root)
        self._set_image(root)
        self.elements = []
        self._set_elements(root)
        self.line_counter = np.fromstring(str(root.LineCounter),
                                          dtype=np.uint16, sep=',')
        self.channel_count = int(root.ChCount)
        self.mapping_count = int(root.DetectorCount)
        self.channel_factors = {}
        self.spectra_data = {}
        self._set_sum_edx(root)

    def _set_sem(self, root):
        semData = root.xpath("ClassInstance[@Type='TRTSEMData']")[0]
        # sem acceleration voltage, working distance, magnification:
        self.sem.hv = float(semData.HV)  # in kV
        self.sem.wd = float(semData.WD)  # in mm
        self.sem.mag = float(semData.Mag)  # in times
        # image/hypermap resolution in um/pixel:
        self.image.x_res = float(semData.DX) / 1.0e6  # in meters
        self.image.y_res = float(semData.DY) / 1.0e6  # in meters
        semStageData = root.xpath("ClassInstance[@Type='TRTSEMStageData']")[0]
        # stage position data in um cast to m (that data anyway is not used
        # by hyperspy):
        try:
            self.stage.x = float(semStageData.X) / 1.0e6  # in meters
            self.stage.y = float(semStageData.Y) / 1.0e6  # in meters
        except AttributeError:
            self.stage.x = self.stage.y = None
        try:
            self.stage.z = float(semStageData.Z) / 1.0e6  # in meters
        except AttributeError:
            self.stage.z = None
        try:
            self.stage.rotation = float(semStageData.Rotation)  # in degrees
        except AttributeError:
            self.stage.rotation = None
        DSPConf = root.xpath("ClassInstance[@Type='TRTDSPConfiguration']")[0]
        self.stage.tilt_angle = float(DSPConf.TiltAngle)

    def _set_image(self, root):
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

    def _set_elements(self, root):
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

    def _set_sum_edx(self, root):
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
        try:
            return sum_eds.nonzero()[0][-1] + 1  # +1: the number not the index
        except IndexError:
            print(
                'The spectrum of mapping with selected index have no counts!!!')
            return len(sum_eds)

    def estimate_map_depth(self, index=0, downsample=1, for_numpy=False):
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
        roof = np.max(sum_eds) // self.image.width // self.image.height * 2 *\
                                          downsample * downsample
        #this complicated nonsence bellow is due to numpy regression in adding
        # integer inplace to unsigned integer array. (python integers is signed)
        if roof > 0xFF:
            if roof > 0xFFFF:
                if for_numpy and (downsample > 1):
                    if roof > 0xEFFFFFFF:
                        depth = np.int64
                    else:
                        depth = np.int32
                else:
                    depth = np.uint32
            else:
                if for_numpy and (downsample > 1):
                    if roof > 0xEFFF:
                        depth = np.int32
                    else:
                        depth = np.int16
                else:
                    depth = np.uint16
        else:
            if for_numpy and (downsample > 1):
                if roof > 0xEF:
                    depth = np.int16
                else:
                    depth = np.int8
            else:
                depth = np.uint8
        return depth

    def get_spectra_metadata(self, index=0):
        """return objectified xml with spectra metadata"""
        return self.spectra_data[index]


class BCF_reader(SFS_reader):
    def __init__(self, filename):
        SFS_reader.__init__(self, filename)
        header_file = self.get_file('EDSDatabase/HeaderData')
        header_byte_str = header_file.get_as_BytesIO_string().getvalue()
        self.header = HyperHeader(header_byte_str)
        self.hypermap = {}

    def print_the_metadata(self):
        print('selected bcf contains:\n * imagery from detectors:')
        for i in self.header.image.images:
            print("\t*", i.detector_name)
        ed = self.header.get_spectra_metadata()
        print(' *', len(self.header.spectra_data), ' spectral cube(s)',
            'with', ed.chnlCnt,
            'channels recorded, coresponding to {0:.2f}kV'.format(
            ed.channel_to_energy(ed.chnlCnt)))

    def persistent_parse_hypermap(self, index=0, downsample=None,
                                  cutoff_at_kV=None):
        """parse and assign the hypermap to the Hypermap python object"""
        dwn = downsample
        hypermap = self.parse_hypermap(index=index,
                                       downsample=dwn,
                                       cutoff_at_kV=cutoff_at_kV)
        self.hypermap[index] = HyperMap(hypermap,
                                        self,
                                        index=index,
                                        downsample=dwn)

    def parse_hypermap(self, index=0, downsample=1, cutoff_at_kV=None):
        """Unpack the Delphi/Bruker binary spectral map and return
        numpy array in memory efficient way.
        Pure python/numpy implimentation -- slow, or
        cython/memoryview/numpy implimentaion if complied (fast)
        is used.

        Arguments:
        ---------
        index: the index of hypermap in bcf if there is more than one
            hyper map in file.
        downsample: downsampling factor (integer). Diferently than
            block_reduce from skimage.measure, the parser populates
            reduced array by suming results of pixels, thus saving memory.
            Downsampled (differently than not) hypermaps are returned
            wiht signed integer dtypes larger if required.

        Returns:
        ---------
        numpy array of hypermap, where spectral channels are on
            the first axis.
        """

        if type(cutoff_at_kV) in (int, float):
            eds = self.header.get_spectra_metadata()
            cutoff_chan = eds.energy_to_channel(cutoff_at_kV)
        else:
            cutoff_chan = None

        if fast_unbcf:
            spectrum_file = self.get_file('EDSDatabase/SpectrumData' +
                                                                    str(index))
            return unbcf_fast.parse_to_numpy(spectrum_file,
                                             downsample=downsample,
                                             cutoff=cutoff_chan)
        else:
            if verbose:
                print('this is going to take a while... please wait')
            return self.py_parse_hypermap(index=0,
                                     downsample=downsample,
                                     cutoff_at_channel=cutoff_chan)

    def py_parse_hypermap(self, index=0, downsample=1, cutoff_at_channel=None):

        st = {1: 'B', 2: 'B', 4: 'H', 8: 'I', 16: 'Q'}
        spectrum_file = self.get_file('EDSDatabase/SpectrumData' + str(index))
        iter_data, size_chnk, chunks = spectrum_file.get_iter_and_properties()
        if type(cutoff_at_channel) == int:
            max_chan = cutoff_at_channel
        else:
            max_chan = self.header.estimate_map_channels(index=index)
        depth = self.header.estimate_map_depth(index=index,
                                               downsample=downsample,
                                               for_numpy=True)
        buffer1 = next(iter_data)
        height, width = strct_unp('<ii', buffer1[:8])
        dwn_factor = downsample
        total_pixels = -(-height // dwn_factor) * -(-width // dwn_factor)
        total_channels = total_pixels * max_chan
        #hyper map as very flat array:
        vfa = np.zeros(total_channels, dtype=depth)
        offset = 0x1A0
        size = size_chnk
        for line_cnt in range(height):
            if (offset + 4) >= size:
                size = size_chnk + size - offset
                buffer1 = buffer1[offset:] + next(iter_data)
                offset = 0
            line_head = strct_unp('<i', buffer1[offset:offset + 4])[0]
            offset += 4
            for dummy1 in range(line_head):
                if (offset + 22) >= size:
                    size = size_chnk + size - offset
                    buffer1 = buffer1[offset:] + next(iter_data)
                    offset = 0
                #the pixel header contains such information:
                # x index of pixel,
                # number of channels for whole mapping,
                # number of channels for pixel,
                # some dummy placehollder (same value in every known bcf),
                # flag distinguishing 12bit packing (1) or instructed packing,
                # value which sometimes shows the size of packed data,
                # number of pulses if data is 12bit packed, or contains 16bit
                #   packed additional to instructed data,
                # packed data size - next header is after that size,
                # dummy -- empty 2bytes
                x_pix, chan1, chan2, dummy1, flag, dummy_size1, n_of_pulses,\
                     data_size2, dummy2 = strct_unp('<IHHIHHHHH',
                                                    buffer1[offset:offset + 22])
                pix_idx = (x_pix // dwn_factor) + ((width // dwn_factor) *
                                                      (line_cnt // dwn_factor))
                offset += 22
                if (offset + data_size2) >= size:
                    buffer1 = buffer1[offset:] + next(iter_data)
                    size = size_chnk + size - offset
                    offset = 0
                if flag == 1:  # and (chan1 != chan2)
                    #Unpack packed 12-bit data to 16-bit uints:
                    data1 = buffer1[offset:offset + data_size2]
                    switched_i2 = np.fromstring(data1,
                                                dtype='<u2'
                                                ).byteswap(True)
                    data2 = np.fromstring(switched_i2.tostring(),
                                          dtype=np.uint8
                                         ).repeat(2)
                    mask = np.ones_like(data2, dtype=bool)
                    mask[0::6] = mask[5::6] = False
                    # Reinterpret expanded as 16-bit:
                    # string representation of array after swith will have
                    # always BE independently from endianess of machine
                    exp16 = np.fromstring(data2[mask].tostring(),
                                          dtype='>u2', count=n_of_pulses)
                    exp16[0::2] >>= 4           # Shift every second short by 4
                    exp16 &= np.uint16(0x0FFF)  # Mask all shorts to 12bit
                    pixel = np.bincount(exp16, minlength=chan1 - 1)
                    offset += data_size2
                else:
                    #Unpack instructively packed data to pixel channels:
                    pixel = []
                    the_end = offset + data_size2 - 4
                    while offset < the_end:
                        #this would work on py3
                        #size_p, channels = buffer1[offset:offset + 2]
                        # this is needed on py2:
                        size_p, channels = strct_unp('<BB',
                                                   buffer1[offset:offset + 2])
                        offset += 2
                        if size_p == 0:
                            pixel += channels * [0]
                        else:
                            gain = strct_unp('<' + st[size_p * 2],
                                            buffer1[offset:offset + size_p])[0]
                            offset += size_p
                            if size_p == 1:
                                # special case with nibble switching
                                length = -(-channels // 2)  # integer roof
                                # valid py3 code
                                #a = list(buffer1[offset:offset + length])
                                #this have to be used on py2:
                                a = strct_unp('<' + 'B' * length,
                                              buffer1[offset:offset + length])
                                g = []
                                for i in a:
                                    g += (i & 0x0F) + gain, (i >> 4) + gain
                                pixel += g[:channels]
                            else:
                                length = int(channels * size_p / 2)
                                temp = strct_unp('<' + channels * st[size_p],
                                                buffer1[offset:offset + length])
                                pixel += [l + gain for l in temp]
                            offset += length
                    if chan2 < chan1:
                        rest = chan1 - chan2
                        pixel += rest * [0]
                    # additional data size:
                    if n_of_pulses > 0:
                        add_s = strct_unp('<I', buffer1[offset:offset + 4])[0]
                        offset += 4
                        if (offset + add_s) >= size:
                            buffer1 = buffer1[offset:] + next(iter_data)
                            size = size_chnk + size - offset
                            offset = 0
                        # the additional pulses:
                        thingy = strct_unp('<' + 'H' * n_of_pulses,
                                           buffer1[offset:offset + add_s])
                        offset += add_s
                        for i in thingy:
                            pixel[i] += 1
                    else:
                        offset += 4
                # if no downsampling is needed, or if it is first
                # pixel encountered with downsampling on, then
                # use assigment, which is ~4 times faster, than inplace add
                if max_chan < chan1:  # if pixel have more channels than we need
                    chan1 = max_chan
                if (dwn_factor == 1) or\
                          ((line_cnt % dwn_factor) and (x_pix % dwn_factor)):
                    vfa[max_chan * pix_idx:chan1 + max_chan * pix_idx] =\
                                                                 pixel[:chan1]
                else:
                    vfa[max_chan * pix_idx:chan1 + max_chan * pix_idx] +=\
                                                                 pixel[:chan1]
        vfa.resize((-(-height // dwn_factor),
                    -(-width // dwn_factor),
                    max_chan))
        return vfa.swapaxes(2, 0)


class HyperMap(object):
    def __init__(self, nparray, parent, index=0, downsample=1):
        sp_meta = parent.header.get_spectra_metadata(index=index)
        self.calib_abs = sp_meta.calibAbs
        self.calib_lin = sp_meta.calibLin
        self.xcalib = parent.header.image.x_res * downsample
        self.ycalib = parent.header.image.y_res * downsample
        self.hypermap = nparray


#wrapper functions for hyperspy:
def file_reader(filename,
                record_by=None,
                index=0,
                downsample=1,
                cutoff_at_kV=None,
                **kwds):
    #objectified bcf file:
    obj_bcf = BCF_reader(filename)
    if record_by == 'image':
        return bcf_imagery(obj_bcf)
    elif record_by == 'spectrum':
        pass  # return bcf_hyperspectra(obj_bcf)
    else:
        return bcf_imagery(obj_bcf)  # + bcf_hyperspectra(obj_bcf)


def bcf_imagery(obj_bcf):
    """
    return hyperspy required list of dict with sem
    imagery and metadata
    """
    imagery_list = []
    for img in obj_bcf.header.image.images:
        imagery_list.append(
          {'data': img.data,
           'axes': [{'index_in_array': 1,
                     'name': 'width',
                     'size': obj_bcf.header.image.width,
                     'offset': 0,
                     'scale': obj_bcf.header.image.y_res,
                     'units': 'm'
                    },
                    {'index_in_array': 0,
                     'name': 'height',
                     'size': obj_bcf.header.image.height,
                     'offset': 0,
                     'scale': obj_bcf.header.image.x_res,
                     'units': 'm'}],
           'metadata':
             # where is no way to determine what kind of instrument was used:
             # TEM or SEM
             {'Acquisition_instrument': {
                          'SEM': {
                             'beam_current': 0.0,  # I have no technical
                             #possibilities to get such parameter by bruker
                             'beam_energy': obj_bcf.header.sem.hv,
                             'tilt_stage': obj_bcf.header.stage.tilt_angle,
                             'stage_x': obj_bcf.header.stage.x,
                             'stage_y': obj_bcf.header.stage.y
                                 }
                                        },
              'General': {'original_filename': obj_bcf.filename.split('/')[-1]},
              'Signal': {'signal_type': img.detector_name,
                           'record_by': 'image', },
             }
           })
    return imagery_list