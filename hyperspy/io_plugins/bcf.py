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
writes = False

import io

try:
    from lxml import objectify
except ImportError:
    raise ImportError("""The lxml or/and python-lxml bindings are missing
required to read Bruker bcf files.
Try to install python-lxml package with pip or other python packaging system""")

import json
import codecs
from ast import literal_eval
from datetime import datetime, timedelta
import numpy as np
import dask.array as da
import dask.delayed as dd
from struct import unpack as strct_unp
from zlib import decompress as unzip_block
import logging

_logger = logging.getLogger(__name__)

warn_once = True

try:
    from hyperspy.io_plugins import unbcf_fast
    fast_unbcf = True
    _logger.info("The fast cython based bcf unpacking library were found")
except ImportError:  # pragma: no cover
    fast_unbcf = False
    _logger.info("""unbcf_fast library is not present...
Falling back to slow python only backend.""")


class Container(object):
    pass


class SFSTreeItem(object):
    """Class to manage one internal sfs file.

    Reading, reading in chunks, reading and extracting, reading without
    extracting even if compression is pressent.

    Attributes:
    item_raw_string -- the bytes from sfs file table describing the file
    parent -- the item higher hierarchicaly in the sfs file tree

    Methods:
    read_piece, setup_compression_metadata, get_iter_and_properties,
    get_as_BytesIO_string
    """

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

    def _fill_pointer_table(self):
        """Parse the sfs and populate self.pointers table.

        self.pointer is the sfs pointer table containing addresses of
        every chunk of the file.

        The pointer table if the file is big can extend throught many
        sfs chunks. Differently than files, the pointer table of file have no
        table of pointers to the chunks. Instead if pointer table is larger
        than sfs chunk, the chunk header contains next chunk number (address
        can be calculated using known chunk size and global offset) with
        continuation of file pointer table, thus it have to be read and filled
        consecutive.
        """
        # table size in number of chunks:
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
        """ Read and returns raw byte string of the file without applying
        any decompression.

        Arguments:
        offset: seek value
        length: length of the data counting from the offset

        Returns:
        io.ByteIO object
        """
        data = io.BytesIO()
        # first block index:
        fb_idx = offset // self.sfs.usable_chunk
        # first block offset:
        fbo = offset % self.sfs.usable_chunk
        # last block index:
        lb_idx = (offset + length) // self.sfs.usable_chunk
        # last block cut off:
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

    def _iter_read_chunks(self, first=0):
        """Generate and return iterator for reading and returning
        sfs internal file in chunks.

        By default it creates iterator for whole file, however
        with kwargs 'first' and 'chunks' the range of chunks
        for iterator can be set.

        Keyword arguments:
        first -- the index of first chunk from which to read. (default 0)
        chunks -- the number of chunks to read. (default False)
        """
        last = self.size_in_chunks
        with open(self.sfs.filename, 'rb') as fn:
            for idx in range(first, last - 1):
                fn.seek(self.pointers[idx])
                yield fn.read(self.sfs.usable_chunk)
            fn.seek(self.pointers[last - 1])
            last_stuff = self.size % self.sfs.usable_chunk
            if last_stuff != 0:
                yield fn.read(last_stuff)
            else:
                yield fn.read(self.sfs.usable_chunk)

    def setup_compression_metadata(self):
        """ parse and setup the number of compression chunks

        and uncompressed chunk size as class attributes.

        Sets up attributes:
        self.uncompressed_blk_size, self.no_of_compr_blk

        """
        with open(self.sfs.filename, 'rb') as fn:
            fn.seek(self.pointers[0])
            # AACS signature, uncompressed size, undef var, number of blocks
            aacs, uc_size, _, n_of_blocks = strct_unp('<IIII', fn.read(16))
        if aacs == 0x53434141:  # AACS as string
            self.uncompressed_blk_size = uc_size
            self.no_of_compr_blk = n_of_blocks
        else:
            raise ValueError("""The file is marked to be compressed,
but compression signature is missing in the header. Aborting....""")

    def _iter_read_compr_chunks(self):
        """Generate and return reader and decompressor iterator
        for compressed with zlib compression sfs internal file.

        Returns:
        iterator of decompressed data chunks.
        """

        offset = 0x80  # the 1st compression block header
        for dummy1 in range(self.no_of_compr_blk):
            cpr_size, dummy_size, dummy_unkn, dummy_size2 = strct_unp('<IIII',
                                                                      self.read_piece(offset, 16))
            # dummy_unkn is probably some kind of checksum but
            # none of known (crc16, crc32, adler32) algorithm could match.
            # dummy_size2 == cpr_size + 0x10 which have no use...
            # dummy_size, which is decompressed size, also have no use...
            # as it is the same in file compression_header
            offset += 16
            raw_string = self.read_piece(offset, cpr_size)
            offset += cpr_size
            yield unzip_block(raw_string)

    def get_iter_and_properties(self):
        """Generate and return the iterator of data chunks and
        properties of such chunks such as size and count.

        Method detects if data is compressed and uses iterator with
        decompression involved, else uses simple iterator of chunks.

        Returns:
            (iterator, chunk_size, number_of_chunks)
        """
        if self.sfs.compression == 'None':
            return self._iter_read_chunks(), self.sfs.usable_chunk,\
                self.size_in_chunks
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
    """Class to read sfs file.
    SFS is AidAim software's(tm) single file system.
    The class provides basic reading capabilities of such container.
    It is capable to read compressed data in zlib, but
    SFS can contain other compression which is not implemented here.
    It is also not able to read encrypted sfs containers.

    This class can be used stand alone or inherited in construction of
    file readers using sfs technolgy.

    Attributes:
    filename

    Methods:
    get_file
    """

    def __init__(self, filename):
        self.filename = filename
        # read the file header
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
            # the sfs tree and number of the items / files + directories in it,
            # and the number in chunks of whole sfs:
            self.tree_address, self.n_tree_items, self.sfs_n_of_chunks =\
                strct_unp('<III', fn.read(12))
        self._setup_vfs()

    def _setup_vfs(self):
        """Setup the virtual file system tree represented as python dictionary
        with values populated with SFSTreeItem instances

        See also:
        SFSTreeItem
        """
        with open(self.filename, 'rb') as fn:
            # file tree do not exceed one chunk in bcf:
            fn.seek(self.chunksize * self.tree_address + 0x138)
            raw_tree = fn.read(0x200 * self.n_tree_items)
            temp_item_list = [SFSTreeItem(raw_tree[i * 0x200:(i + 1) * 0x200],
                                          self) for i in range(self.n_tree_items)]
            # temp list with parents of items
            paths = [[h.parent] for h in temp_item_list]
        # checking the compression header which can be different per file:
        self._check_the_compresion(temp_item_list)
        if self.compression == 'zlib':
            for c in temp_item_list:
                if not c.is_dir:
                    c.setup_compression_metadata()
        # convert the items to virtual file system tree
        dict_tree = self._flat_items_to_dict(paths, temp_item_list)
        # and finaly set the Virtual file system:
        self.vfs = dict_tree['root']

    def _flat_items_to_dict(self, paths, temp_item_list):
        """place items from flat list into dictionary tree
        of virtual file system
        """
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
        # return dict tree:
        return root

    def _check_the_compresion(self, temp_item_list):
        """parse, check and setup the self.compression"""
        with open(self.filename, 'rb') as fn:
            # Find if there is compression:
            for c in temp_item_list:
                if not c.is_dir:
                    fn.seek(c.pointers[0])
                    if fn.read(4) == b'\x41\x41\x43\x53':  # string AACS
                        self.compression = 'zlib'
                    else:
                        self.compression = 'None'
                    # compression is global, can't be diferent per file in sfs
                    break

    def get_file(self, path):
        """Return the SFSTreeItem (aka internal file) object from
        sfs container.

        Arguments:
        path -- internal file path in sfs file tree. Path accepts only
            standard - forward slash for directories.

        Returns:
        object (SFSTreeItem), which can be read into byte stream, in
        chunks or whole using objects methods.

        Example:
        to get "file" object 'kitten.png' in folder 'catz' which
        resides in root directory of sfs, you would use:

        >>> instance_of_SFSReader.get_file('catz/kitten.png')

        See also:
        SFSTreeItem
        """
        item = self.vfs

        for i in path.split('/'):
            item = item[i]
        return item


def interpret(string):
    """interpret any string and return casted to appropriate
    dtype python object
    """
    try:
        return literal_eval(string)
    except (ValueError, SyntaxError):
        # SyntaxError due to:
        # literal_eval have problems with strings like this '8842_80'
        return string


class ObjectifyJSONEncoder(json.JSONEncoder):
    """ JSON encoder that can handle simple lxml objectify types,
        Handles xml attributes, also returns all data types"""

    def default(self, o):
        dictionary = {}
        if hasattr(o, '__dict__') and len(o.__dict__) > 0:
            d1 = o.__dict__.copy()
            for k in d1.keys():
                if len(d1[k]) > 1:
                    d1[k] = [interpret(i.text) for i in d1[k]]
            dictionary.update(d1)
        if len(o.attrib) > 0:
            d2 = dict(o.attrib)
            for j in d2.keys():
                if j in dictionary.keys() or j == 'Type':
                    d2['XmlClass' + j] = interpret(d2[j])
                    del d2[j]
                else:
                    d2[j] = interpret(d2[j])
            dictionary.update(d2)
        if o.text is not None:
            if len(dictionary) > 0:
                dictionary.update({'value': o.pyval})
            else:
                return interpret(o.text)
        if len(dictionary) > 0:
            return dictionary


class EDXSpectrum(object):

    def __init__(self, spectrum):
        """
        Wrap the objectified bruker EDS spectrum xml part
        to the python object, leaving all the xml and bruker clutter behind

        Arguments:
        spectrum -- lxml objectified xml where spectrum.attrib['Type'] should
            be 'TRTSpectrum'
        """
        TRTHeader = spectrum.TRTHeaderedClass
        #<ClassInstance Type="TRTSpectrumHardwareHeader>:
        hardware_header = TRTHeader.ClassInstance
        #<ClassInstance Type="TRTDetectorHeader>:
        detector_header = TRTHeader.ClassInstance[1]
        #<ClassInstance Type="TRTESMAHeader">:
        esma_header = TRTHeader.ClassInstance[2]
        # what TRT means?
        # ESMA could stand for Electron Scanning Microscope Analysis
        spectrum_header = spectrum.ClassInstance[0]

        # map stuff from harware xml branch:
        self.hardware_metadata = json.loads(json.dumps(hardware_header,
                                                       cls=ObjectifyJSONEncoder))
        self.amplification = self.hardware_metadata['Amplification']  # USED

        # map stuff from detector xml branch
        self.detector_metadata = json.loads(json.dumps(detector_header,
                                                       cls=ObjectifyJSONEncoder))
        self.detector_type = self.detector_metadata['Type']  # USED

        # decode silly hidden detector layer info:
        det_l_str = self.detector_metadata['DetLayers']
        dec_det_l_str = codecs.decode(det_l_str.encode('ascii'), 'base64')
        mini_xml = objectify.fromstring(unzip_block(dec_det_l_str))
        self.detector_metadata['DetLayers'] = {}  # Overwrite with dict
        for i in mini_xml.getchildren():
            self.detector_metadata['DetLayers'][i.tag] = dict(i.attrib)

        # map stuff from esma xml branch:
        self.esma_metadata = json.loads(json.dumps(esma_header,
                                                   cls=ObjectifyJSONEncoder))
        # USED:
        self.hv = self.esma_metadata['PrimaryEnergy']
        self.elevationAngle = self.esma_metadata['ElevationAngle']
        #self.azimutAngle = self.esma_metadata['AzimutAngle']

        # map stuff from spectra xml branch:
        self.spectrum_metadata = json.loads(json.dumps(spectrum_header,
                                                       cls=ObjectifyJSONEncoder))
        self.calibAbs = self.spectrum_metadata['CalibAbs']
        self.calibLin = self.spectrum_metadata['CalibLin']
        self.chnlCnt = self.spectrum_metadata['ChannelCount']
        self.date = self.spectrum_metadata['Date']  # Not Used?
        self.time = self.spectrum_metadata['Time']  # Not Used?

        # main data:
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


class HyperHeader(object):
    """Wrap Bruker HyperMaping xml header into python object.

    Arguments:
    xml_str -- the uncompressed to be provided with extracted Header xml
    from bcf.

    Methods:
    estimate_map_channels, estimate_map_depth

    If Bcf is version 2, the bcf can contain stacks
    of hypermaps - thus header part  can contain multiply sum eds spectras
    and it's metadata per hypermap slice which can be selected using index.
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
        except KeyError:
            self.name = 'Undefinded'
            _logger.info("hypermap have no name. Giving it 'Undefined' name")
        dt = datetime.strptime(' '.join([str(root.Header.Date),
                                         str(root.Header.Time)]),
                               "%d.%m.%Y %H:%M:%S")
        self.date = dt.date().isoformat()
        self.time = dt.time().isoformat()
        self.version = int(root.Header.FileVersion)
        # create containers:
        self.sem = Container()
        self.stage = Container()
        self.image = Container()
        # fill the sem and stage attributes:
        self._set_sem(root)
        self._set_image(root)
        self.elements = {}
        self._set_elements(root)
        self.line_counter = interpret(root.LineCounter.text)
        self.channel_count = int(root.ChCount)
        self.mapping_count = int(root.DetectorCount)
        #self.channel_factors = {}
        self.spectra_data = {}
        self._set_sum_edx(root)

    def _set_sem(self, root):
        """wrap objectified xml part to class attributes for self.sem,
        self.stage and self.image.*_res
        """
        semData = root.xpath("ClassInstance[@Type='TRTSEMData']")[0]
        # sem acceleration voltage, working distance, magnification:
        self.sem.hv = semData.HV.pyval  # in kV
        self.sem.wd = semData.WD.pyval  # in mm
        self.sem.mag = semData.Mag.pyval  # in times
        # image/hypermap resolution in um/pixel:
        try:
            self.image.x_res = semData.DX.pyval  # in micrometers
            self.image.y_res = semData.DY.pyval  # in micrometers
            self.units = 'µm'
        except AttributeError:
            self.image.x_res = 1.0  # in pixels
            self.image.y_res = 1.0  # in pixels
            self.units = 'pix'
        semStageData = root.xpath("ClassInstance[@Type='TRTSEMStageData']")[0]
        # stage position:
        self.stage_metadata = json.loads(json.dumps(semStageData,
                                                    cls=ObjectifyJSONEncoder))
        DSPConf = root.xpath("ClassInstance[@Type='TRTDSPConfiguration']")[0]
        self.image.dsp_metadata = json.loads(json.dumps(DSPConf,
                                                        cls=ObjectifyJSONEncoder))

    def get_acq_instrument_dict(self, detector=False, **kwargs):
        """return python dictionary with aquisition instrument
        mandatory data
        """
        acq_inst = {
            'beam_energy': self.sem.hv,
            'magnification': self.sem.mag,
        }
        if 'Tilt' in self.stage_metadata:
            acq_inst['tilt_stage'] = self.stage_metadata['Tilt']
        if detector:
            eds_metadata = self.get_spectra_metadata(**kwargs)
            acq_inst['Detector'] = {'EDS': {
                #'azimuth_angle': eds_metadata.azimutAngle,
                'elevation_angle': eds_metadata.elevationAngle,
                'detector_type': eds_metadata.detector_type,
                'real_time': self.calc_real_time()
            }
            }
            if 'AzimutAngle' in eds_metadata.esma_metadata:
                acq_inst['Detector']['EDS'][
                    'azimuth_angle'] = eds_metadata.esma_metadata['AzimutAngle']
        return acq_inst

    def _set_image(self, root):
        """Wrap objectified xml part with image to class attributes
        for self.image.
        """
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
        """wrap objectified xml part with selection of elements to
        self.elements list
        """
        try:
            elements = root.xpath("".join([
                "ClassInstance[@Type='TRTContainerClass']/ChildClassInstances",
                "/ClassInstance[@Type='TRTElementInformationList']",
                "/ClassInstance[@Type='TRTSpectrumRegionList']",
                "/ChildClassInstances"]))[0]
            for j in elements.xpath(
                    "ClassInstance[@Type='TRTSpectrumRegion']"):
                self.elements[j.attrib['Name']] = {'line': j.Line.pyval,
                                                   'energy': j.Energy.pyval,
                                                   'width': j.Width.pyval}
        except IndexError:
            _logger.info('no element selection present in the spectra..')

    def _set_sum_edx(self, root):
        for i in range(self.mapping_count):
            # self.channel_factors[i] = int(root.xpath("ChannelFactor" +
            #                                         str(i))[0])
            self.spectra_data[i] = EDXSpectrum(root.xpath("SpectrumData" +
                                                          str(i))[0].ClassInstance)

    def estimate_map_channels(self, index=0):
        """estimate minimal size of energy axis so any spectra from any pixel
        would not be truncated.

        Arguments:
        index -- index of the map if multiply hypermaps are present
        in the same bcf.

        Returns:
        optimal channel number
        """
        bruker_hv_range = self.spectra_data[index].amplification / 1000
        if self.sem.hv >= bruker_hv_range:
            return self.spectra_data[index].data.shape[0]
        else:
            return self.spectra_data[index].energy_to_channel(self.sem.hv)

    def estimate_map_depth(self, index=0, downsample=1, for_numpy=False):
        """estimate minimal dtype of array using cumulative spectra
        of the all pixels so that no data would be truncated.

        Arguments:
        index -- index of the hypermap if multiply hypermaps are
        present in the same bcf. (default 0)
        downsample -- downsample factor (should be integer; default 1)
        for_numpy -- if estimation will be used in parsing using oure python
            and numpy inplace integer addition will be used, so the dtype
            should be signed; if cython implementation will be used (default),
            then any returned dtypes can be safely unsigned. (default False)

        Returns:
        numpy dtype large enought to use in final hypermap numpy array.

        The method estimates the value from sum eds spectra, dividing
        the maximum  energy pulse value from raster x and y and to be on the
        safe side multiplying by 2.
        """
        sum_eds = self.spectra_data[index].data
        # the most intensive peak is Bruker reference peak at 0kV:
        roof = np.max(sum_eds) // self.image.width // self.image.height * 2 *\
            downsample * downsample
        # this complicated nonsence bellow is due to numpy regression in adding
        # integer inplace to unsigned integer array. (python integers is
        # signed)
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
        """return objectified xml with spectra metadata
        Arguments:
        index -- index of hypermap/spectra (default 0)
        """
        return self.spectra_data[index]

    def calc_real_time(self):
        """calculate and return real time for whole hypermap
        in seconds
        """
        line_cnt_sum = np.sum(self.line_counter)
        line_avg = self.image.dsp_metadata['LineAverage']
        pix_avg = self.image.dsp_metadata['PixelAverage']
        pix_time = self.image.dsp_metadata['PixelTime']
        width = self.image.width
        real_time = line_cnt_sum * line_avg * pix_avg * pix_time * width / 1000000.0
        return float(real_time)


class BCF_reader(SFS_reader):

    """Class to read bcf (Bruker hypermapping) file.

    Inherits SFS_reader and all its attributes and methods.

    Attributes:
    filename

    Methods:
    print_the_metadata, persistent_parse_hypermap, parse_hypermap,
    py_parse_hypermap
    (Inherited from SFS_reader: print_file_tree, get_file)

    The class instantiates HyperHeader class as self.header attribute
    where all metadata, sum eds spectras, (SEM) imagery are stored.
    if persistent_parse_hypermap is called, the hypermap is stored
    as instance of HyperMap inside the self.hypermap dictionary,
    where index of the hypermap (default 0) is the key to the instance.
    """

    def __init__(self, filename):
        SFS_reader.__init__(self, filename)
        header_file = self.get_file('EDSDatabase/HeaderData')
        header_byte_str = header_file.get_as_BytesIO_string().getvalue()
        self.header = HyperHeader(header_byte_str)
        self.hypermap = {}

    def persistent_parse_hypermap(self, index=0, downsample=None,
                                  cutoff_at_kV=None,
                                  lazy=False):
        """Parse and assign the hypermap to the HyperMap instance.

        Arguments:
        index -- index of hypermap in bcf if v2 (default 0)
        downsample -- downsampling factor of hypermap (default None)
        cutoff_at_kV -- low pass cutoff value at keV (default None)

        Method does not return anything, it adds the HyperMap instance to
        self.hypermap dictionary.

        See also:
        HyperMap, parse_hypermap
        """
        dwn = downsample
        hypermap = self.parse_hypermap(index=index,
                                       downsample=dwn,
                                       cutoff_at_kV=cutoff_at_kV,
                                       lazy=lazy)
        self.hypermap[index] = HyperMap(hypermap,
                                        self,
                                        index=index,
                                        downsample=dwn)

    def parse_hypermap(self, index=0, downsample=1, cutoff_at_kV=None,
                       lazy=False):
        """Unpack the Delphi/Bruker binary spectral map and return
        numpy array in memory efficient way.

        Pure python/numpy implimentation -- slow, or
        cython/memoryview/numpy implimentation if compilied and present
        (fast) is used.

        Arguments:
        index -- the index of hypermap in bcf if there is more than one
            hyper map in file.
        downsample -- downsampling factor (integer). Diferently than
            block_reduce from skimage.measure, the parser populates
            reduced array by suming results of pixels, thus having lower
            memory requiriments. (default 1)
        cutoff_at_kV -- value in keV to truncate the array at. Helps reducing
          size of array. (default None)
        lazy -- return dask.array (True) or numpy.array (False) (default False)

        Returns:
        numpy or dask array of bruker hypermap, with (y,x,E) shape.
        """

        if type(cutoff_at_kV) in (int, float):
            eds = self.header.get_spectra_metadata()
            cutoff_chan = eds.energy_to_channel(cutoff_at_kV)
        else:
            cutoff_chan = None

        if fast_unbcf:
            fh = dd(self.get_file)('EDSDatabase/SpectrumData'+str(index))
            value = dd(unbcf_fast.parse_to_numpy)(fh,
                                                  downsample=downsample,
                                                  cutoff=cutoff_chan,
                                                  description=False)
            if lazy:
                shape, dtype = unbcf_fast.parse_to_numpy(fh.compute(),
                                                         downsample=downsample,
                                                         cutoff=cutoff_chan,
                                                         description=True)
                res = da.from_delayed(value, shape=shape, dtype=dtype)
            else:
                res = value.compute()
            return res
        else:
            _logger.warning("""using slow python parser,
this is going to take a while... please wait""")
            value = dd(self.py_parse_hypermap)(index=0,
                                               downsample=downsample,
                                               cutoff_at_channel=cutoff_chan,
                                               description=False)
            if lazy:
                shape, dtype = self.py_parse_hypermap(
                    index=0, downsample=downsample,
                    cutoff_at_channel=cutoff_chan, description=True)
                res = da.from_delayed(value, shape=shape, dtype=dtype)
            else:
                res = value.compute()
            return res

    def py_parse_hypermap(self, index=0, downsample=1, cutoff_at_channel=None,
                          description=False):
        """Unpack the Delphi/Bruker binary spectral map and return
        numpy array in memory efficient way using pure python implementation.
        (Slow!)

        The function is long and complicated because Delphi/Bruker array
        packing is complicated. Whole parsing is done in one function/method
        to reduce overhead from python function calls. For cleaner parsing
        logic check out fast cython implementation at
        hyperspy/io_plugins/unbcf_fast.pyx

        The method is only meant to be used if for some
        reason c (generated with cython) version of the parser is not compiled.

        Arguments:
        ---------
        index -- the index of hypermap in bcf if there is more than one
            hyper map in file.
        downsample -- downsampling factor (integer). Diferently than
            block_reduce from skimage.measure, the parser populates
            reduced array by suming results of pixels, thus having lower
            memory requiriments. (default 1)
        cutoff_at_kV -- value in keV to truncate the array at. Helps reducing
          size of array. (default None)

        Returns:
        ---------
        numpy array of bruker hypermap, with (y,x,E) shape.
        """
        # dict of nibbles to struct notation for reading:
        st = {1: 'B', 2: 'B', 4: 'H', 8: 'I', 16: 'Q'}
        spectrum_file = self.get_file('EDSDatabase/SpectrumData' + str(index))
        iter_data, size_chnk = spectrum_file.get_iter_and_properties()[:2]
        if isinstance(cutoff_at_channel, int):
            max_chan = cutoff_at_channel
        else:
            max_chan = self.header.estimate_map_channels(index=index)
        depth = self.header.estimate_map_depth(index=index,
                                               downsample=downsample,
                                               for_numpy=True)
        buffer1 = next(iter_data)
        height, width = strct_unp('<ii', buffer1[:8])
        dwn_factor = downsample
        shape = (-(-height // dwn_factor), -(-width // dwn_factor), max_chan)
        if description:
            return shape, depth
        # hyper map as very flat array:
        vfa = np.zeros(shape[0]*shape[1]*shape[2], dtype=depth)
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
                # the pixel header contains such information:
                # x index of pixel,
                # number of channels for whole mapping,
                # number of channels for pixel,
                # some dummy placehollder (same value in every known bcf),
                # flag distinguishing 12bit packing (1) or instructed packing,
                # value which sometimes shows the size of packed data,
                # number of pulses if data is 12bit packed, or contains 16bit
                #  packed additional to instructed data,
                # packed data size - next header is after that size,
                # dummy -- empty 2bytes
                x_pix, chan1, chan2, dummy1, flag, dummy_size1, n_of_pulses,\
                    data_size2, dummy2 = strct_unp('<IHHIHHHHH',
                                                   buffer1[offset:offset + 22])
                pix_idx = (x_pix // dwn_factor) + ((-(-width // dwn_factor)) *
                                                   (line_cnt // dwn_factor))
                offset += 22
                if (offset + data_size2) >= size:
                    buffer1 = buffer1[offset:] + next(iter_data)
                    size = size_chnk + size - offset
                    offset = 0
                if flag == 1:  # and (chan1 != chan2)
                    # Unpack packed 12-bit data to 16-bit uints:
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
                    # string representation of array after switch will have
                    # always BE independently from endianess of machine
                    exp16 = np.fromstring(data2[mask].tostring(),
                                          dtype='>u2', count=n_of_pulses)
                    exp16[0::2] >>= 4           # Shift every second short by 4
                    exp16 &= np.uint16(0x0FFF)  # Mask all shorts to 12bit
                    pixel = np.bincount(exp16, minlength=chan1 - 1)
                    offset += data_size2
                else:
                    # Unpack instructively packed data to pixel channels:
                    pixel = []
                    the_end = offset + data_size2 - 4
                    while offset < the_end:
                        # this would work on py3
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
                                # this have to be used on py2:
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
                        add_pulses = strct_unp('<' + 'H' * n_of_pulses,
                                               buffer1[offset:offset + add_s])
                        offset += add_s
                        for i in add_pulses:
                            pixel[i] += 1
                    else:
                        offset += 4
                # if no downsampling is needed, or if it is first
                # pixel encountered with downsampling on, then
                # use assigment, which is ~4 times faster, than inplace add
                if max_chan < chan1:  # if pixel have more channels than we need
                    chan1 = max_chan
                if (dwn_factor == 1):
                    vfa[max_chan * pix_idx:chan1 + max_chan * pix_idx] =\
                        pixel[:chan1]
                else:
                    vfa[max_chan * pix_idx:chan1 + max_chan * pix_idx] +=\
                        pixel[:chan1]
        vfa.resize((-(-height // dwn_factor),
                    -(-width // dwn_factor),
                    max_chan))
        # check if array is signed, and convert to unsigned
        if str(vfa.dtype)[0] == 'i':
            new_dtype = ''.join(['u', str(vfa.dtype)])
            vfa.dtype = new_dtype
        return vfa


class HyperMap(object):

    """Container class to hold the parsed bruker hypermap
    and its scale calibrations"""

    def __init__(self, nparray, parent, index=0, downsample=1):
        sp_meta = parent.header.get_spectra_metadata(index=index)
        self.calib_abs = sp_meta.calibAbs  # in keV
        self.calib_lin = sp_meta.calibLin
        self.xcalib = parent.header.image.x_res * downsample
        self.ycalib = parent.header.image.y_res * downsample
        self.hypermap = nparray


# wrapper functions for hyperspy:
def file_reader(filename, select_type=None, index=0, downsample=1,
                cutoff_at_kV=None, instrument=None, lazy=False):
    """Reads a bruker bcf file and loads the data into the appropriate class,
    then wraps it into appropriate hyperspy required list of dictionaries
    used by hyperspy.api.load() method.

    Keyword arguments:
    select_type -- One of: spectrum, image. If none specified, then function
      loads everything, else if specified, loads either just sem imagery,
      or just hyper spectral mapping data. (default None)
    index -- index of dataset in bcf v2 (delaut 0)
    downsample -- the downsample ratio of hyperspectral array (downsampling
      height and width only), can be integer from 1 to inf, where '1' means
      no downsampling will be applied (default 1).
    cutoff_at_kV -- if set (can be int of float >= 0) can be used either, to
       crop or enlarge energy range at max values. (default None)
    instrument -- str, either 'TEM' or 'SEM'. Default is None.
      """

    # objectified bcf file:
    obj_bcf = BCF_reader(filename)
    if select_type == 'image':
        return bcf_imagery(obj_bcf)
    elif select_type == 'spectrum':
        return bcf_hyperspectra(obj_bcf, index=index,
                                downsample=downsample,
                                cutoff_at_kV=cutoff_at_kV,
                                instrument=instrument,
                                lazy=lazy)
    else:
        return bcf_imagery(obj_bcf, instrument=instrument) + bcf_hyperspectra(
            obj_bcf,
            index=index,
            downsample=downsample,
            cutoff_at_kV=cutoff_at_kV,
            instrument=instrument,
            lazy=lazy)


def bcf_imagery(obj_bcf, instrument=None):
    """ return hyperspy required list of dict with sem
    imagery and metadata.
    """
    imagery_list = []
    mode = _get_mode(obj_bcf, instrument=instrument)
    for img in obj_bcf.header.image.images:
        imagery_list.append(
            {'data': img.data,
             'axes': [{'name': 'height',
                       'size': obj_bcf.header.image.height,
                       'offset': 0,
                       'scale': obj_bcf.header.image.y_res,
                       'units': obj_bcf.header.units},
                      {'name': 'width',
                       'size': obj_bcf.header.image.width,
                       'offset': 0,
                       'scale': obj_bcf.header.image.x_res,
                       'units': obj_bcf.header.units}],
             'metadata':
             # where is no way to determine what kind of instrument was used:
             # TEM or SEM (mode variable)
             {'Acquisition_instrument': {
                 mode: obj_bcf.header.get_acq_instrument_dict()
             },
                 'General': {'original_filename': obj_bcf.filename.split('/')[-1],
                             'title': img.detector_name},
                 'Sample': {'name': obj_bcf.header.name},
                 'Signal': {'signal_type': img.detector_name,
                            'record_by': 'image'},
             },
             'original_metadata': {
                 'DSP Configuration': obj_bcf.header.image.dsp_metadata,
                 'Stage': obj_bcf.header.stage_metadata
             }
             })
    return imagery_list


def bcf_hyperspectra(obj_bcf, index=0, downsample=None, cutoff_at_kV=None,
                     instrument=None, lazy=False):
    """ Return hyperspy required list of dict with eds
    hyperspectra and metadata.
    """
    global warn_once
    if (fast_unbcf == False) and warn_once:
        _logger.warning("""unbcf_fast library is not present...
Parsing BCF with Python-only backend.
If parsing is uncomfortably slow, first install cython, then reinstall hyperspy.
For more information, check the 'Installing HyperSpy' section in the documentation.""")
        warn_once = False
    obj_bcf.persistent_parse_hypermap(index=index, downsample=downsample,
                                      cutoff_at_kV=cutoff_at_kV, lazy=lazy)
    eds_metadata = obj_bcf.header.get_spectra_metadata(index=index)
    mode = _get_mode(obj_bcf, instrument=instrument)
    hyperspectra = [{'data': obj_bcf.hypermap[index].hypermap,
                     'axes': [{'name': 'height',
                               'size': obj_bcf.hypermap[index].hypermap.shape[0],
                               'offset': 0,
                               'scale': obj_bcf.hypermap[index].ycalib,
                               'units': obj_bcf.header.units},
                              {'name': 'width',
                               'size': obj_bcf.hypermap[index].hypermap.shape[1],
                               'offset': 0,
                               'scale': obj_bcf.hypermap[index].xcalib,
                               'units': obj_bcf.header.units},
                              {'name': 'Energy',
                               'size': obj_bcf.hypermap[index].hypermap.shape[2],
                               'offset': obj_bcf.hypermap[index].calib_abs,
                               'scale': obj_bcf.hypermap[index].calib_lin,
                               'units': 'keV'}],
                     'metadata':
                     # where is no way to determine what kind of instrument was used:
                     # TEM or SEM
                     {'Acquisition_instrument': {
                         mode: obj_bcf.header.get_acq_instrument_dict(detector=True,
                                                                      index=index)
                     },
        'General': {'original_filename': obj_bcf.filename.split('/')[-1],
                         'title': 'EDX',
                         'date': obj_bcf.header.date,
                                  'time': obj_bcf.header.time},
        'Sample': {'name': obj_bcf.header.name,
                         'elements': sorted(list(obj_bcf.header.elements)),
                         'xray_lines': sorted(gen_elem_list(obj_bcf.header.elements))},
        'Signal': {'signal_type': 'EDS_%s' % mode,
                         'record_by': 'spectrum',
                         'quantity': 'X-rays (Counts)'}
    },
        'original_metadata': {'Hardware': eds_metadata.hardware_metadata,
                              'Detector': eds_metadata.detector_metadata,
                              'Analysis': eds_metadata.esma_metadata,
                              'Spectrum': eds_metadata.spectrum_metadata,
                              'DSP Configuration': obj_bcf.header.image.dsp_metadata,
                              'Line counter': obj_bcf.header.line_counter,
                              'Stage': obj_bcf.header.stage_metadata}
    }]
    return hyperspectra


def gen_elem_list(the_dict):
    return ['_'.join([i, parse_line(the_dict[i]['line'])]) for i in the_dict]
#    return [z_to_element(i) for i in the_list]


def parse_line(line_string):
    """standardize line describtion.

    Bruker saves line describtion in all caps
    and omits the type if only one exists instead of
    using alfa"""
    if len(line_string) == 1:
        line_string = line_string + 'a'
    return line_string.capitalize()


def _get_mode(obj_bcf, instrument=None):
    if instrument is not None:
        return instrument
    hv = obj_bcf.header.sem.hv
    if hv > 30.0:  # workaround to know if TEM or SEM
        mode = 'TEM'
    else:
        mode = 'SEM'
    _logger.info("Guessing that the acquisition instrument is %s " % mode +
                 "because the beam energy is %i keV. If this is wrong, " % hv +
                 "please provide the right instrument using the 'instrument' " +
                 "keyword.")
    return mode
