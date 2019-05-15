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
Esprit(R) software to save hypermaps together with 16bit SEM imagery,
EDS spectra and metadata describing the dimentions of the data and
SEM/TEM (limited) parameters"""
full_support = False
# Recognised file extension
file_extensions = ('bcf', 'spx')
default_extension = 0
# Reading capabilities
reads_images = True
reads_spectrum = True
reads_spectrum_image = True
# Writing capabilities
writes = False

import io

from collections import defaultdict
import xml.etree.ElementTree as ET
import codecs
from ast import literal_eval
from datetime import datetime, timedelta
import numpy as np
import dask.array as da
import dask.delayed as dd
from struct import unpack as strct_unp
from zlib import decompress as unzip_block
import logging
import re
from math import ceil
from os.path import splitext, basename

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

# define re with two capturing groups with comma in between
# firstgroup looks for numeric value after <tag> (the '>' char) with or
# without minus sign, second group looks for numeric value with following
# closing <\tag> (the '<' char); '([Ee]-?\d*)' part (optionally a third group)
# checks for scientific notation (e.g. 8,843E-7 -> 'E-7');
# compiled pattern is binary, as raw xml string is binary.:
fix_dec_patterns = re.compile(b'(>-?\\d+),(\\d*([Ee]-?\\d*)?<)')


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
        n_chunks = ceil(self.size / self.sfs.usable_chunk)
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
        n_of_chunks = ceil(self.size_in_chunks /
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
            self.pointers = np.frombuffer(temp_table[:self.size_in_chunks * 4],
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
            cpr_size = strct_unp('<I12x', self.read_piece(offset, 16))[0]
            # cpr_size, dum_size, dum_unkn, dum_size2 = strct_unp('<IIII',...
            # dum_unkn is probably some kind of checksum but
            # none of known (crc16, crc32, adler32) algorithm could match.
            # dum_size2 == cpr_size + 0x10 which have no use...
            # dum_size, which is decompressed size, also have no use...
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
        elif self.sfs.compression == 'zlib':
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
            #check if file tree do not exceed one chunk:
            n_file_tree_chunks = ceil((self.n_tree_items * 0x200) /
                                      (self.chunksize - 0x20))
            if n_file_tree_chunks == 1:
                # file tree do not exceed one chunk in bcf:
                fn.seek(self.chunksize * self.tree_address + 0x138)
                raw_tree = fn.read(0x200 * self.n_tree_items)
            else:
                temp_str = io.BytesIO()
                tree_address = self.tree_address
                tree_items_in_chunk = (self.chunksize - 0x20) // 0x200
                for i in range(n_file_tree_chunks):
                    # jump to tree/list address:
                    fn.seek(self.chunksize * tree_address + 0x118)
                    # next tree/list address:
                    tree_address = strct_unp('<I', fn.read(4))[0]
                    fn.seek(28, 1)
                    temp_str.write(fn.read(tree_items_in_chunk * 0x200))
                temp_str.seek(0)
                raw_tree = temp_str.read(self.n_tree_items * 0x200)
                temp_str.close()
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


def dictionarize(t):
    d = {t.tag: {} if t.attrib else None}
    children = list(t)
    if children:
        dd = defaultdict(list)
        for dc in map(dictionarize, children):
            for k, v in dc.items():
                dd[k].append(v)
        d = {t.tag: {k: interpret(v[0]) if len(
            v) == 1 else v for k, v in dd.items()}}
    if t.attrib:
        d[t.tag].update(('XmlClass' + k if list(t) else k, interpret(v))
                        for k, v in t.attrib.items())
    if t.text:
        text = t.text.strip()
        if children or t.attrib:
            if text:
                d[t.tag]['#text'] = interpret(text)
        else:
            d[t.tag] = interpret(text)
    if 'ClassInstance' in d:
        return d['ClassInstance']
    else:
        return d


class EDXSpectrum(object):

    def __init__(self, spectrum):
        """
        Wrap the objectified bruker EDS spectrum xml part
        to the python object, leaving all the xml and bruker clutter behind.

        Arguments:
        spectrum -- etree xml object, where spectrum.attrib['Type'] should
            be 'TRTSpectrum'
        """
        TRTHeader = spectrum.find('./TRTHeaderedClass')
        hardware_header = TRTHeader.find(
            "./ClassInstance[@Type='TRTSpectrumHardwareHeader']")
        detector_header = TRTHeader.find(
            "./ClassInstance[@Type='TRTDetectorHeader']")
        esma_header = TRTHeader.find(
            "./ClassInstance[@Type='TRTESMAHeader']")
        # what TRT means?
        # ESMA could stand for Electron Scanning Microscope Analysis
        spectrum_header = spectrum.find(
            "./ClassInstance[@Type='TRTSpectrumHeader']")

        # map stuff from harware xml branch:
        self.hardware_metadata = dictionarize(hardware_header)
        self.amplification = self.hardware_metadata['Amplification']  # USED

        # map stuff from detector xml branch
        self.detector_metadata = dictionarize(detector_header)
        self.detector_type = self.detector_metadata['Type']  # USED

        # decode silly hidden detector layer info:
        det_l_str = self.detector_metadata['DetLayers']
        dec_det_l_str = codecs.decode(det_l_str.encode('ascii'), 'base64')
        mini_xml = ET.fromstring(unzip_block(dec_det_l_str))
        self.detector_metadata['DetLayers'] = {}  # Overwrite with dict
        for i in list(mini_xml):
            self.detector_metadata['DetLayers'][i.tag] = dict(i.attrib)

        # map stuff from esma xml branch:
        self.esma_metadata = dictionarize(esma_header)
        # USED:
        self.hv = self.esma_metadata['PrimaryEnergy']
        self.elev_angle = self.esma_metadata['ElevationAngle']
        date_time = gen_iso_date_time(spectrum_header)
        if date_time is not None:
            self.date, self.time = date_time
        
        # map stuff from spectra xml branch:
        self.spectrum_metadata = dictionarize(spectrum_header)
        self.offset = self.spectrum_metadata['CalibAbs']
        self.scale = self.spectrum_metadata['CalibLin']

        # main data:
        self.data = np.fromstring(spectrum.find('./Channels').text,
                                  dtype='Q', sep=",")

    def energy_to_channel(self, energy, kV=True):
        """ convert energy to channel index,
        optional kwarg 'kV' (default: True) should be set to False
        if given energy units is in V"""
        if not kV:
            en_temp = energy / 1000.
        else:
            en_temp = energy
        return int(round((en_temp - self.offset) / self.scale))


class HyperHeader(object):
    """Wrap Bruker HyperMaping xml header into python object.

    Arguments:
    xml_str -- the uncompressed to be provided with extracted Header xml
    from bcf.
    indexes -- list of indexes of available datasets

    Methods:
    estimate_map_channels, estimate_map_depth

    If Bcf is version 2, the bcf can contain stacks
    of hypermaps - thus header part  can contain multiply sum eds spectras
    and it's metadata per hypermap slice which can be selected using index.
    Bcf can record number of imagery from different
    imagining detectors (BSE, SEI, ARGUS, etc...): access to imagery
    is throught image index.
    """

    def __init__(self, xml_str, indexes, instrument=None):
        root = ET.fromstring(xml_str)
        root = root.find("./ClassInstance[@Type='TRTSpectrumDatabase']")
        try:
            self.name = str(root.attrib['Name'])
        except KeyError:
            self.name = 'Undefinded'
            _logger.info("hypermap have no name. Giving it 'Undefined' name")
        hd = root.find("./Header")
        self.date, self.time = gen_iso_date_time(hd)
        self.version = int(hd.find('./FileVersion').text)
        # fill the sem and stage attributes:
        self._set_microscope(root)
        self._set_mode(instrument)
        self._set_images(root)
        self.elements = {}
        self._set_elements(root)
        self.line_counter = interpret(root.find('./LineCounter').text)
        self.channel_count = int(root.find('./ChCount').text)
        self.mapping_count = int(root.find('./DetectorCount').text)
        #self.channel_factors = {}
        self.spectra_data = {}
        self._set_sum_edx(root, indexes)

    def _set_microscope(self, root):
        """set microscope metadata from objectified xml part (TRTSEMData,
        TRTSEMStageData, TRTDSPConfiguration).

        BCF can contain basic parameters of SEM column, and optionaly
        the stage. This metadata can be not fully or at all availbale to
        Esprit and saved into bcf file as it depends from license and
        the link and implementation state between the microscope's
        software and Bruker system.
        """

        semData = root.find("./ClassInstance[@Type='TRTSEMData']")
        self.sem_metadata = dictionarize(semData)
        # parse values for use in hspy metadata:
        self.hv = self.sem_metadata.get('HV', 0.0)  # in kV
        # image/hypermap resolution in um/pixel:
        if 'DX' in self.sem_metadata:
            self.units = 'µm'
        else:
            self.units = 'pix'
        self.x_res = self.sem_metadata.get('DX', 1.0)
        self.y_res = self.sem_metadata.get('DY', 1.0)
        # stage position:
        semStageData = root.find("./ClassInstance[@Type='TRTSEMStageData']")
        self.stage_metadata = dictionarize(semStageData)
        # DSP configuration (always present, part of Bruker system):
        DSPConf = root.find("./ClassInstance[@Type='TRTDSPConfiguration']")
        self.dsp_metadata = dictionarize(DSPConf)

    def _set_mode(self, instrument=None):
        if instrument is not None:
            self.mode = instrument
        else:
            self.mode = guess_mode(self.hv)

    def get_acq_instrument_dict(self, detector=False, **kwargs):
        """return python dictionary with aquisition instrument
        mandatory data
        """
        acq_inst = {'beam_energy': self.hv}
        if 'Mag' in self.sem_metadata:
            acq_inst['magnification'] = self.sem_metadata['Mag']
        if detector:
            eds_metadata = self.get_spectra_metadata(**kwargs)
            det = gen_detector_node(eds_metadata)
            det['EDS']['real_time'] = self.calc_real_time()
            acq_inst['Detector'] = det
        return acq_inst

    def _parse_image(self, xml_node, overview=False):
        """parse image from bruker xml image node."""
        if overview:
            rect_node = xml_node.find("./ChildClassInstances"
                "/ClassInstance["
                #"@Type='TRTRectangleOverlayElement' and "
                "@Name='Map']/TRTSolidOverlayElement/"
                "TRTBasicLineOverlayElement/TRTOverlayElement")
            if rect_node is not None:
                over_rect = dictionarize(rect_node)['TRTOverlayElement']['Rect']
                rect = {'y1': over_rect['Top'] * self.y_res,
                        'x1': over_rect['Left'] * self.x_res,
                        'y2': over_rect['Bottom'] * self.y_res,
                        'x2': over_rect['Right'] * self.x_res}
                over_dict = {'marker_type': 'Rectangle',
                            'plot_on_signal': True,
                            'data': rect,
                            'marker_properties': {'color': 'yellow',
                                                'linewidth': 2}}
        image = Container()
        image.width = int(xml_node.find('./Width').text)  # in pixels
        image.height = int(xml_node.find('./Height').text)  # in pixels
        image.dtype = 'u' + xml_node.find('./ItemSize').text  # in bytes ('u1','u2','u4') 
        image.plane_count = int(xml_node.find('./PlaneCount').text)
        image.images = []
        for i in range(image.plane_count):
            img = xml_node.find("./Plane" + str(i))
            raw = codecs.decode((img.find('./Data').text).encode('ascii'),'base64')
            array1 = np.frombuffer(raw, dtype=image.dtype)
            if any(array1):
                item = self.gen_hspy_item_dict_basic()
                data = array1.reshape((image.height, image.width))
                desc = img.find('./Description')
                item['data'] = data
                item['axes'][0]['size'] = image.height
                item['axes'][1]['size'] = image.width
                item['metadata']['Signal'] = {'record_by': 'image'}
                item['metadata']['General'] = {}
                if desc is not None:
                    item['metadata']['General']['title'] = str(desc.text)
                if overview and (rect_node is not None):
                    item['metadata']['Markers'] = {'overview': over_dict}
                image.images.append(item)
        return image

    def _set_images(self, root):
        """Wrap objectified xml part with image to class attributes
        for self.image.
        """
        image_nodes = root.findall("./ClassInstance[@Type='TRTImageData']")
        for n in image_nodes:
            if not(n.get('Name')):
                image_node = n
        self.image = self._parse_image(image_node)
        if self.version == 2:
            overview_node = root.findall(
                "./ClassInstance[@Type='TRTContainerClass']"
                "/ChildClassInstances"
                "/ClassInstance["
                #"@Type='TRTContainerClass' and "
                "@Name='OverviewImages']"
                "/ChildClassInstances"
                "/ClassInstance[@Type='TRTImageData']")
            if len(overview_node) > 0:  # in case there is no image
                self.overview = self._parse_image(
                    overview_node[0], overview=True)

    def _set_elements(self, root):
        """wrap objectified xml part with selection of elements to
        self.elements list
        """
        try:
            elements = root.find(
                "./ClassInstance[@Type='TRTContainerClass']"
                "/ChildClassInstances"
                "/ClassInstance[@Type='TRTElementInformationList']"
                "/ClassInstance[@Type='TRTSpectrumRegionList']"
                "/ChildClassInstances")
            for j in elements.findall(
                    "./ClassInstance[@Type='TRTSpectrumRegion']"):
                tmp_d = dictionarize(j)
                self.elements[tmp_d['XmlClassName']] = {'line': tmp_d['Line'],
                                                        'energy': tmp_d['Energy'],
                                                        'width': tmp_d['Width']}
        except AttributeError:
            _logger.info('no element selection present in the spectra..')

    def _set_sum_edx(self, root, indexes):
        for i in indexes:
            spec_node = root.find(
                "./SpectrumData{0}/ClassInstance".format(str(i)))
            self.spectra_data[i] = EDXSpectrum(spec_node)

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
        if self.hv >= bruker_hv_range:
            return self.spectra_data[index].data.shape[0]
        else:
            return self.spectra_data[index].energy_to_channel(self.hv)

    def estimate_map_depth(self, index=0, downsample=1, for_numpy=False):
        """estimate minimal dtype of array using cumulative spectra
        of the all pixels so that no data would be truncated.

        Arguments:
        index -- index of the hypermap if multiply hypermaps are
          present in the same bcf. (default 0)
        downsample -- downsample factor (should be integer; default 1)
        for_numpy -- False produce unsigned, True signed (or unsigned) types:
          if hypermap will be loaded using the pure python
          function where numpy's inplace integer addition will be used --
          the dtype should be signed; if cython implementation will
          be used (default), then any returned dtypes can be safely
          unsigned. (default False)

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
        line_avg = self.dsp_metadata['LineAverage']
        pix_avg = self.dsp_metadata['PixelAverage']
        pix_time = self.dsp_metadata['PixelTime']
        width = self.image.width
        real_time = line_cnt_sum * line_avg * pix_avg * pix_time * width / 1000000.0
        return float(real_time)

    def gen_hspy_item_dict_basic(self):
        i = {'axes': [{'name': 'height',
                       'offset': 0,
                       'scale': self.y_res,
                       'units': self.units},
                      {'name': 'width',
                       'offset': 0,
                       'scale': self.x_res,
                       'units': self.units}],
             'metadata': {
            'Acquisition_instrument':
            {self.mode: self.get_acq_instrument_dict()},
                'Sample': {'name': self.name},
        },
            'original_metadata': {
                 'Microscope': self.sem_metadata,
                 'DSP Configuration': self.dsp_metadata,
                 'Stage': self.stage_metadata
        }
        }
        return i


class BCF_reader(SFS_reader):

    """Class to read bcf (Bruker hypermapping) file.

    Inherits SFS_reader and all its attributes and methods.

    Attributes:
    filename

    Methods:
    check_index_valid, parse_hypermap

    The class instantiates HyperHeader class as self.header attribute
    where all metadata, sum eds spectras, (SEM) images are stored.
    """

    def __init__(self, filename, instrument=None):
        SFS_reader.__init__(self, filename)
        header_file = self.get_file('EDSDatabase/HeaderData')
        self.available_indexes = []
        # get list of presented indexes from file tree of binary sfs container
        # while looking for file names containg the hypercube data:
        for i in self.vfs['EDSDatabase'].keys():
            if 'SpectrumData' in i:
                self.available_indexes.append(int(i[-1]))
        self.def_index = min(self.available_indexes)
        header_byte_str = header_file.get_as_BytesIO_string().getvalue()
        hd_bt_str = fix_dec_patterns.sub(b'\\1.\\2', header_byte_str)
        self.header = HyperHeader(
            hd_bt_str, self.available_indexes, instrument=instrument)
        self.hypermap = {}

    def check_index_valid(self, index):
        """check and return if index is valid"""
        if type(index) != int:
            raise TypeError("provided index should be integer")
        if index not in self.available_indexes:
            raise IndexError("requisted index is not in the list of available indexes. "
                             "Available maps are under indexes: {0}".format(str(self.available_indexes)))
        return index

    def parse_hypermap(self, index=None,
                       downsample=1, cutoff_at_kV=None,
                       lazy=False):
        """Unpack the Delphi/Bruker binary spectral map and return
        numpy array in memory efficient way.

        Pure python/numpy implementation -- slow, or
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
        if index is None:
            index = self.def_index
        if type(cutoff_at_kV) in (int, float):
            eds = self.header.spectra_data[index]
            max_chan = eds.energy_to_channel(cutoff_at_kV)
        else:
            max_chan = self.header.estimate_map_channels(index=index)
        shape = (ceil(self.header.image.height / downsample),
                 ceil(self.header.image.width / downsample),
                 max_chan)
        sfs_file = SFS_reader(self.filename)
        vrt_file_hand = sfs_file.get_file(
            'EDSDatabase/SpectrumData' + str(index))
        if fast_unbcf:
            parse_func = unbcf_fast.parse_to_numpy
            dtype = self.header.estimate_map_depth(index=index,
                                                   downsample=downsample,
                                                   for_numpy=False)
        else:
            parse_func = py_parse_hypermap
            dtype = self.header.estimate_map_depth(index=index,
                                                   downsample=downsample,
                                                   for_numpy=True)
        if lazy:
            value = dd(parse_func)(vrt_file_hand, shape,
                                   dtype, downsample=downsample)
            result = da.from_delayed(value, shape=shape, dtype=dtype)
        else:
            result = parse_func(vrt_file_hand, shape,
                                dtype, downsample=downsample)
        return result

    def add_filename_to_general(self, item):
        """hypy helper method"""
        item['metadata']['General']['original_filename'] = \
            basename(self.filename)

def spx_reader(filename, lazy=False):
    with open(filename, 'br') as fn:
        xml_str = fn.read()
    root = ET.fromstring(xml_str)
    sp_node = root.find("./ClassInstance[@Type='TRTSpectrum']")
    try:
        name = str(sp_node.attrib['Name'])
    except KeyError:
        name = 'Undefinded'
        _logger.info("spectra have no name. Giving it 'Undefined' name")
    spectrum = EDXSpectrum(sp_node)
    mode = guess_mode(spectrum.hv)
    results_xml = sp_node.find("./ClassInstance[@Type='TRTResult']")
    elements_xml = sp_node.find("./ClassInstance[@Type='TRTPSEElementList']")
    hy_spec = {'data': spectrum.data,
               'axes': [{'name': 'Energy',
                         'size': len(spectrum.data),
                         'offset': spectrum.offset,
                         'scale': spectrum.scale,
                         'units': 'keV'}],
               'metadata':
               # where is no way to determine what kind of instrument was used:
               # TEM or SEM
               {'Acquisition_instrument': {
                 mode: {'Detector':
                            gen_detector_node(spectrum),
                         'beam_energy': spectrum.hv}
               },
                'General': {'original_filename': basename(filename),
                            'title': 'EDX',
                            'date': spectrum.date,
                             'time': spectrum.time},
                 'Sample': {'name': name},
                 'Signal': {'signal_type': 'EDS_%s' % mode,
                            'record_by': 'spectrum',
                            'quantity': 'X-rays (Counts)'}
               },
               'original_metadata': {'Hardware': spectrum.hardware_metadata,
                                     'Detector': spectrum.detector_metadata,
                                     'Analysis': spectrum.esma_metadata,
                                     'Spectrum': spectrum.spectrum_metadata,}
              }
    if results_xml is not None:
        hy_spec['original_metadata']['Results'] = dictionarize(results_xml)
    if elements_xml is not None:
        elem = dictionarize(elements_xml)['ChildClassInstances']
        hy_spec['original_metadata']['Selected_elements'] = elem
        hy_spec['metadata']['Sample']['elements'] = elem['XmlClassName']
    return [hy_spec]


# dict of nibbles to struct notation for reading:
st = {1: 'B', 2: 'B', 4: 'H', 8: 'I', 16: 'Q'}


def py_parse_hypermap(virtual_file, shape, dtype, downsample=1):
    """Unpack the Delphi/Bruker binary spectral map and return
    numpy array in memory efficient way using pure python implementation.
    (Slow!)

    The function is long and complicated due to complexity of Delphi packed
    array.
    Whole parsing is placed in one function to reduce overhead of
    python function calls. For cleaner parsing logic, please, see
    fast cython implementation at hyperspy/io_plugins/unbcf_fast.pyx

    The method is only meant to be used if for some
    reason c (generated with cython) version of the parser is not compiled.

    Arguments:
    ---------
    virtual_file -- virtual file handle returned by SFS_reader instance
        or by object inheriting it (e.g. BCF_reader instance)
    shape -- numpy shape
    dtype -- numpy dtype
    downsample -- downsample factor

    note!: downsample, shape and dtype are interconnected and needs
    to be properly calculated otherwise wrong output or segfault
    is expected

    Returns:
    ---------
    numpy array of bruker hypermap, with (y, x, E) shape.
    """
    iter_data, size_chnk = virtual_file.get_iter_and_properties()[:2]
    dwn_factor = downsample
    max_chan = shape[2]
    buffer1 = next(iter_data)
    height, width = strct_unp('<ii', buffer1[:8])
    # hyper map as very flat array:
    vfa = np.zeros(shape[0] * shape[1] * shape[2], dtype=dtype)
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
            # x index of pixel (uint32);
            # number of channels for whole mapping (unit16);
            # number of channels for pixel (uint16);
            # dummy placehollder (same value in every known bcf) (32bit);
            # flag distinguishing packing data type (16bit):
            #    0 - 16bit packed pulses, 1 - 12bit packed pulses,
            #    >1 - instructively packed spectra;
            # value which sometimes shows the size of packed data (uint16);
            # number of pulses if pulse data are present (uint16) or
            #      additional pulses to the instructively packed data;
            # packed data size (32bit) (without additional pulses) \
            #       next header is after that amount of bytes;
            x_pix, chan1, chan2, dummy1, flag, dummy_size1, n_of_pulses,\
                data_size2 = strct_unp('<IHHIHHHI',
                                       buffer1[offset:offset + 22])
            pix_idx = (x_pix // dwn_factor) + (ceil(width / dwn_factor) *
                                               (line_cnt // dwn_factor))
            offset += 22
            if (offset + data_size2) >= size:
                buffer1 = buffer1[offset:] + next(iter_data)
                size = size_chnk + size - offset
                offset = 0
            if flag == 0:
                data1 = buffer1[offset:offset + data_size2]
                arr16 = np.frombuffer(data1, dtype=np.uint16)
                pixel = np.bincount(arr16, minlength=chan1 - 1)
                offset += data_size2
            elif flag == 1:  # and (chan1 != chan2)
                # Unpack packed 12-bit data to 16-bit uints:
                data1 = buffer1[offset:offset + data_size2]
                switched_i2 = np.frombuffer(data1,
                                            dtype='<u2'
                                            ).copy().byteswap(True)
                data2 = np.frombuffer(switched_i2.tostring(),
                                      dtype=np.uint8
                                      ).copy().repeat(2)
                mask = np.ones_like(data2, dtype=bool)
                mask[0::6] = mask[5::6] = False
                # Reinterpret expanded as 16-bit:
                # string representation of array after switch will have
                # always BE independently from endianess of machine
                exp16 = np.frombuffer(data2[mask].tostring(),
                                      dtype='>u2', count=n_of_pulses).copy()
                exp16[0::2] >>= 4           # Shift every second short by 4
                exp16 &= np.uint16(0x0FFF)  # Mask all shorts to 12bit
                pixel = np.bincount(exp16, minlength=chan1 - 1)
                offset += data_size2
            else:  # flag > 1
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
                            length = ceil(channels / 2)
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
    vfa.resize((ceil(height / dwn_factor),
                ceil(width / dwn_factor),
                max_chan))
    # check if array is signed, and convert to unsigned
    if str(vfa.dtype)[0] == 'i':
        new_dtype = ''.join(['u', str(vfa.dtype)])
        vfa.dtype = new_dtype
    return vfa


def file_reader(filename, *args, **kwds):
    ext = splitext(filename)[1][1:]
    if ext == 'bcf':
        return bcf_reader(filename, *args, **kwds)
    elif ext == 'spx':
        return spx_reader(filename, *args, **kwds)


def bcf_reader(filename, select_type=None, index=None,  # noqa
               downsample=1, cutoff_at_kV=None, instrument=None, lazy=False):
    """Reads a bruker bcf file and loads the data into the appropriate class,
    then wraps it into appropriate hyperspy required list of dictionaries
    used by hyperspy.api.load() method.

    Keyword arguments:
    select_type -- One of: spectrum_image, image. If none specified, then function
      loads everything, else if specified, loads either just sem imagery,
      or just hyper spectral mapping data (default None).
    index -- index of dataset in bcf v2 can be None integer and 'all'
      (default None); None will select first available mapping if more than one.
      'all' will return all maps if more than one present;
      integer will return only selected map.
    downsample -- the downsample ratio of hyperspectral array (downsampling
      height and width only), can be integer from 1 to inf, where '1' means
      no downsampling will be applied (default 1).
    cutoff_at_kV -- if set (can be int of float >= 0) can be used either, to
       crop or enlarge energy range at max values. (default None)
    instrument -- str, either 'TEM' or 'SEM'. Default is None.
      """

    # objectified bcf file:
    obj_bcf = BCF_reader(filename, instrument=instrument)
    if select_type == 'spectrum':
        select_type = 'spectrum_image'
        from hyperspy.misc.utils import deprecation_warning
        msg = (
            "The 'spectrum' option for the `select_type` parameter is "
            "deprecated and will be removed in v2.0. Use 'spectrum_image' "
            "instead.")
        deprecation_warning(msg)
    if select_type == 'image':
        return bcf_images(obj_bcf)
    elif select_type == 'spectrum_image':
        return bcf_hyperspectra(obj_bcf, index=index,
                                downsample=downsample,
                                cutoff_at_kV=cutoff_at_kV,
                                lazy=lazy)
    else:
        return bcf_images(obj_bcf) + bcf_hyperspectra(
            obj_bcf,
            index=index,
            downsample=downsample,
            cutoff_at_kV=cutoff_at_kV,
            lazy=lazy)


def bcf_images(obj_bcf):
    """ return hyperspy required list of dict with sem
    images and metadata.
    """
    images_list = []
    for img in obj_bcf.header.image.images:
        obj_bcf.add_filename_to_general(img)
        images_list.append(img)
    if hasattr(obj_bcf.header, 'overview'):
        for img2 in obj_bcf.header.overview.images:
            obj_bcf.add_filename_to_general(img2)
            images_list.append(img2)
    return images_list


def bcf_hyperspectra(obj_bcf, index=None, downsample=None, cutoff_at_kV=None,  # noqa
                     lazy=False):
    """ Return hyperspy required list of dict with eds
    hyperspectra and metadata.
    """
    global warn_once
    if (fast_unbcf == False) and warn_once:
        _logger.warning("""unbcf_fast library is not present...
Parsing BCF with Python-only backend, which is slow... please wait.
If parsing is uncomfortably slow, first install cython, then reinstall hyperspy.
For more information, check the 'Installing HyperSpy' section in the documentation.""")
        warn_once = False
    if index is None:
        indexes = [obj_bcf.def_index]
    elif index == 'all':
        indexes = obj_bcf.available_indexes
    else:
        indexes = [obj_bcf.check_index_valid(index)]
    hyperspectra = []
    mode = obj_bcf.header.mode
    mapping = get_mapping(mode)
    for index in indexes:
        hypermap = obj_bcf.parse_hypermap(index=index,
                                          downsample=downsample,
                                          cutoff_at_kV=cutoff_at_kV,
                                          lazy=lazy)
        eds_metadata = obj_bcf.header.get_spectra_metadata(index=index)
        hyperspectra.append(
            {'data': hypermap,
             'axes': [{'name': 'height',
                       'size': hypermap.shape[0],
                       'offset': 0,
                       'scale': obj_bcf.header.y_res * downsample,
                       'units': obj_bcf.header.units},
                      {'name': 'width',
                       'size': hypermap.shape[1],
                       'offset': 0,
                       'scale': obj_bcf.header.y_res * downsample,
                       'units': obj_bcf.header.units},
                      {'name': 'Energy',
                       'size': hypermap.shape[2],
                       'offset': eds_metadata.offset,
                       'scale': eds_metadata.scale,
                       'units': 'keV'}],
             'metadata':
             # where is no way to determine what kind of instrument was used:
             # TEM or SEM
             {'Acquisition_instrument': {
                 mode: obj_bcf.header.get_acq_instrument_dict(
                     detector=True,
                     index=index)
             },
                 'General': {'original_filename': basename(obj_bcf.filename),
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
                                      'DSP Configuration': obj_bcf.header.dsp_metadata,
                                      'Line counter': obj_bcf.header.line_counter,
                                      'Stage': obj_bcf.header.stage_metadata,
                                      'Microscope': obj_bcf.header.sem_metadata},
                'mapping': mapping,
             })
    return hyperspectra


def gen_elem_list(the_dict):
    return ['_'.join([i, parse_line(the_dict[i]['line'])]) for i in the_dict]


def parse_line(line_string):
    """standardize line describtion.

    Bruker saves line description in all caps
    and omits the type if only one exists instead of
    using alfa"""
    if len(line_string) == 1:
        line_string = line_string + 'a'
    elif len(line_string) > 2:
        line_string = line_string[:2]
    return line_string.capitalize()


def get_mapping(mode):
    return {
        'Stage.Rotation':
        ("Acquisition_instrument.%s.Stage.rotation" % mode, None),
        'Stage.Tilt':
        ("Acquisition_instrument.%s.Stage.tilt_alpha" % mode, None),
        'Stage.X':
        ("Acquisition_instrument.%s.Stage.x" % mode, None),
        'Stage.Y':
        ("Acquisition_instrument.%s.Stage.y" % mode, None),
        'Stage.Z':
        ("Acquisition_instrument.%s.Stage.z" % mode, None),
    }

def guess_mode(hv):
    """there is no way to determine what kind of instrument
    was used from metadata: TEM or SEM.
    However simple guess can be made using the acceleration
    voltage, assuming that SEM is <= 30kV or TEM is >30kV"""
    if hv > 30.0:
        mode = 'TEM'
    else:
        mode = 'SEM'
    _logger.info(
        "Guessing that the acquisition instrument is %s " % mode +
        "because the beam energy is %i keV. If this is wrong, " % hv +
        "please provide the right instrument using the 'instrument' " +
        "keyword.")
    return mode

def gen_detector_node(spectrum):
    eds_dict = {'EDS': {'elevation_angle': spectrum.elev_angle,
                        'detector_type': spectrum.detector_type,}}
    if 'AzimutAngle' in spectrum.esma_metadata:
        eds_dict['EDS']['azimuth_angle'] = spectrum.esma_metadata['AzimutAngle']
    if 'RealTime' in spectrum.hardware_metadata:
        eds_dict['EDS']['real_time'] = spectrum.hardware_metadata['RealTime'] / 1000
        eds_dict['EDS']['live_time'] = spectrum.hardware_metadata['LifeTime'] / 1000
    return eds_dict

def gen_iso_date_time(node):
    date_xml = node.find('./Date')
    time_xml = node.find('./Time')
    if date_xml is not None:
        dt = datetime.strptime(' '.join([date_xml.text, time_xml.text]),
                               "%d.%m.%Y %H:%M:%S")
        date = dt.date().isoformat()
        time = dt.time().isoformat()
        return date, time
