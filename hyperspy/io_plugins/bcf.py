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

import os
import io

from lxml import objectify, etree
import codecs
from datetime import datetime, timedelta
import numpy as np
import struct
from struct import unpack as strct_unp

import json

from skimage.measure import block_reduce


try:
    import unbcf_fast
    fast_unbcf = True
    print("the fast cython library of parsing were found")
except:
    fast_unbcf = False
    print("Falling back to python only backend")


class Container(object):
    pass

class SFSTreeItem(object):
    def __init__(self, item_raw_string, parent):
        self.sfs = parent
        self._pointer_to_pointer_table, self.size, create_time, \
        mod_time, some_time, self.permissions, \
        self.parent, _, self.is_dir, _, name, _ = struct.unpack(
                       '<iQQQQIi176s?3s256s32s', item_raw_string)
        self.create_time = self._filetime_to_unix(create_time)
        self.mod_time = self._filetime_to_unix(mod_time)
        self.some_time = self._filetime_to_unix(some_time)
        self.name = name.strip(b'\x00').decode('utf-8')
        self.size_in_chunks = self._calc_pointer_table_size()
        if self.is_dir == 0:
            self._fill_pointer_table()
        self.compr_chunk_pointers = []
        self.cache = Container()
        self.cache.status = None
            
    def _calc_pointer_table_size(self):
        n_chunks = -(-self.size // self.sfs.usable_chunk)
        return n_chunks
    
    def _filetime_to_unix(self, time):
        return datetime(1601,1,1) + timedelta(microseconds=time/10)
    
    def __repr__(self):
        return '<SFS internal file {0:.2f} MB>'.format(self.size/1024/1024)
    
    def _fill_pointer_table(self):
        #table size in number of chunks
        table_size = -(-self.size_in_chunks //
                       (self.sfs.usable_chunk // 4))
        with open(self.sfs.filename, 'rb') as fn:
            if table_size > 1:
                next_chunk = self._pointer_to_pointer_table
                temp_string = io.BytesIO()
                for j in range(0, table_size, 1):
                    fn.seek(self.sfs.chunksize *
                            next_chunk +
                            0x118)
                    next_chunk = struct.unpack('<I', fn.read(4))[0]
                    fn.seek(28, 1)
                    temp_string.write(fn.read(self.sfs.usable_chunk))
                temp_string.seek(0)
                temp_table = temp_string.read()
                temp_string.close()
            else:
                fn.seek(self.sfs.chunksize *
                        self._pointer_to_pointer_table
                        + 0x138)
                temp_table = fn.read(self.sfs.usable_chunk)
            self.pointers = np.fromstring(temp_table[:self.size_in_chunks*4],
              dtype='uint32').astype(self.sfs.pointers_table_type) *\
                                               self.sfs.chunksize + 0x138
                
    def _read_piece(self, offset,length):
        """reads and returns piece of file. Intentionaly
        differs from builtin read method.
        
        requires two arguments:
        offset -- seek value
        lenght -- lenght of the data counting from the offset
        
        function is slower than reading whole thing, but is intended
        to be used with more memory efficient implimentations of very
        large file reading operations."""
        
        if self.sfs.compression == 'None':
            data = io.BytesIO()
            #first block index:
            fb_idx = offset // self.sfs.usable_chunk
            #first block offset:
            fbo = offset % self.sfs.usable_chunk
            #last block index:
            lb_idx = (offset+length) // self.sfs.usable_chunk
            #last block cut off:
            lbco = (offset+length) % self.sfs.usable_chunk
            with open(self.sfs.filename, 'rb') as fn:
                if fb_idx != lb_idx:
                    fn.seek(self.pointers[fb_idx]+fbo)
                    data.write(fn.read(self.sfs.usable_chunk - fbo))
                    for i in self.pointers[fb_idx+1:lb_idx]:
                        fn.seek(i)
                        data.write(fn.read(self.sfs.usable_chunk))
                    if lbco > 0:
                        fn.seek(self.pointers[lb_idx])
                        data.write(fn.read(lbco))
                else:
                    fn.seek(self.pointers[fb_idx]+fbo)
                    data.write(fn.read(length))                    
            return data
        elif self.sfs.compression in ('zlib', 'bzip2'):
            raise RuntimeError('Reading piece of compressed data is not implemented')

        else:
            raise RuntimeError('file', str(self.sfs.filename),
                               ' is compressed by not known and not implemented algorithm.',
                               'Aborting...')
    
    def get_as_BytesIO_string(self):
        if self.sfs.compression == 'None':
            data = io.BytesIO()
            with open(self.sfs.filename, 'rb') as fn:
                for i in self.pointers[:-1]:
                    fn.seek(i)
                    data.write(fn.read(self.sfs.usable_chunk))
                fn.seek(self.pointers[-1])
                data.write(fn.read(self.size % self.sfs.usable_chunk))
            return data
        elif self.sfs.compression in ('zlib', 'bzip2'):
            # import required library just once:
            if self.sfs.compression == 'zlib':
                from zlib import decompress as unzip_block
            else:
                from bzip2 import decompress as unzip_block
            data = io.BytesIO()
            block = []
            with open(self.sfs.filename, 'rb') as fn:
                fn.seek(self.pointers[0] + 12)  # number of compression blocks
                number_of_blocks = struct.unpack('<I', fn.read(4))[0]
                fn.seek(self.pointers[0] + 0x80)  # the 1st compression block header
                block_remainder = struct.unpack('<I', fn.read(4))[0]
                fn.seek(12, 1)  # go to compressed data begining
                if len(self.pointers) > 1:
                    chunk_remainder = self.sfs.usable_chunk - 0x90
                    chunk_point_iter = iter(self.pointers[1:])
                else:
                    chunk_remainder = self.size % self.sfs.usable_chunk - 0x90
                for j in range(0, number_of_blocks, 1):
                    #read the chunks, until block is filled
                    # then read next block header, decompress the block
                    # append the result to main data output, reset block to empty
                    while block_remainder > chunk_remainder:
                        block.append(fn.read(chunk_remainder))
                        block_remainder -= chunk_remainder
                        chunk_remainder = self.sfs.usable_chunk
                        fn.seek(next(chunk_point_iter))
                    else:
                        block.append(fn.read(block_remainder))
                        if j != number_of_blocks:
                            chunk_remainder = chunk_remainder - block_remainder - 0x10
                            # setting the new block_remainder from header:
                            block_remainder = struct.unpack('<I', fn.read(4))[0]
                            
                            if chunk_remainder > 0:
                                fn.seek(12, 1)  # jump to next block
                            else:
                                fn.seek(next(chunk_point_iter)-chunk_remainder)
                                chunk_remainder = self.sfs.usable_chunk + chunk_remainder
                                
                    data.write(unzip_block(b''.join(block)))
                    block = []

            return data
        else:
            raise RuntimeError('file', str(self.sfs.filename),
                               ' is compressed by not known and not implemented algorithm.',
                               'Aborting...')



class SFS_reader(object):
    def __init__(self,filename):
        self.filename = filename
        self.hypermap = {}
        with open(filename, 'rb') as fn:
            a = fn.read(8)
            if a != b'AAMVHFSS':
                raise TypeError("file '{0}' is not SFS container".format(filename))
            fn.seek(0x124)  # this looks to be version, as float value is always nicely rounded
            # and at older bcf versions it was 2.40, at newer 2.60
            version, self.chunksize = struct.unpack('<fI', fn.read(8))
            self.sfs_version = '{0:4.2f}'.format(version)
            self.usable_chunk = self.chunksize - 32
            fn.seek(0x140)
            
            self.sfs_size = os.path.getsize(filename)
            # if self.sfs_size < 0xFFFF:
            #     self.pointers_table_type = np.uint16
            #elif self.sfs_size < 0xFFFFFFFF:
            #    self.pointers_table_type = np.uint32
            #else:
            #    self.pointers_table_type = np.uint64
            self.pointers_table_type = np.int64
            ''' I practicaly saw some sfs containers of >60GB...
            So the pointer tables compared to data is not so big... 
            so more practically it would be to use uint64, as it 
            will make easier using the table with cython faster code
            UPDATE: at the moment uint64 is broken, thus rather
            int64 will be used, as it have enought space to contain
            all posibble pointers for file sizes
            for coming next ~10 years.... I guess...
            '''
            #the sfs tree and number of the items / files + directories in it,
            #and the number in chunks of whole sfs:
            tree_address, n_file_tree_items, self.sfs_n_of_chunks = struct.unpack(
                                                                '<III', fn.read(12))
            n_file_tree_chunks = -((-n_file_tree_items*0x200) //
                                      (self.chunksize-512))
            if n_file_tree_chunks is 1:
                fn.seek(self.chunksize * tree_address + 0x138)  # skip with header
                raw_tree = fn.read(0x200*n_file_tree_items)
            else:
                temp_str = io.BytesIO()
                for i in range(n_file_tree_chunks):
                    # jump to tree/list address:
                    fn.seek(self.chunksize * tree_address + 0x118)
                    # next tree/list address:
                    tree_address = struct.unpack('<I', fn.read(4))[0]
                    fn.seek(28, 1)
                    temp_str.write(fn.read(self.chunksize - 512))
                temp_str.seek(0)
                raw_tree = temp_str.read(n_file_tree_items*0x200)
                temp_str.close()
            # setting up virtual file system in python dictionary
            temp_item_list = [SFSTreeItem(raw_tree[i*0x200 : (i+1)*0x200], self)
                              for i in range(n_file_tree_items)]
            paths = [[h.parent] for h in temp_item_list]
            #Find if there is compression:
            for c in temp_item_list:
                if c.is_dir == False:
                    fn.seek(c.pointers[0])
                    if fn.read(4) == b'\x41\x41\x43\x53':  # string AACS
                        fn.seek(0x8C, 1)
                        compression_head = fn.read(2)
                        byte_one = struct.unpack('BB', compression_head)[0]
                        if byte_one == 0x78:
                            self.compression = 'zlib'
                        elif compression_head == b'\x42\x5A':
                            self.compression = 'bzip2'
                        else:
                            self.compression = 'unknown'
                    else:
                        self.compression = 'None'
                    break
                        
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
                    temp_nondir_item = temp_item_list[i]
            # and finaly Virtual file system:
            self.vfs = root['root']
                   
    def print_file_tree(self):
        print(json.dumps(self.vfs, sort_keys=True, indent=4, default=str))
        
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
        """The class which should be fed with part of objectified bruker xml:
        exactly -- ClassInstance with Attributes (Type='TRTSpectrum')"""
        if str(spectrum.attrib['Type']) != 'TRTSpectrum':
            raise IOError
        try:
            self.realTime = int(spectrum.TRTHeaderedClass.ClassInstance.RealTime)
            self.lifeTime = int(spectrum.TRTHeaderedClass.ClassInstance.LifeTime)
            self.deadTime = int(spectrum.TRTHeaderedClass.ClassInstance.DeadTime)
        except:
            print('spectrum have no dead time records. skipping...')
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
        self.x = np.arange(self.calibAbs, self.calibLin * self.chnlCnt + self.calibAbs, self.calibLin)
        
    def e2ch(self, energy):
        """converts energy to channel"""
        en_temp = energy/1000.
        return (en_temp - self.calibAbs) / self.calibLin
    
    def ch2e(self, channel, kV=True):
        """converts given channel to energy,
        optional attribute kV (default: True) decides if returned value
        is in kV or V"""
        if kV == False:
            kV = 1000
        else:
            kV = 1
        return (channel * self.calibLin + self.calibAbs) * kV


class HyperHeader(object):
    """ objectifies the xml header of bcf file and provides
    the shortcuts to the deeple nested parts of the objectified data.
    When instantiated needs the string with xml 'HeaderData' from bcf
    The object provides two types of shortcuts resambling structure
    of the bcf header. If Bcf is version 2, the bcf can contain stacks
    of hypermaps - thus header part contains sum eds spectras and it's
    metadata per hypermap slice.
    The object initialise shortcuts or provides conversion to apropriate
    data type for single records which apears in header just once or
    the object provides methods to get or estimate data/shortcuts for
    records which are per slice of hypermap, with default index=0.
    Exception is the imagery which is stored in xml just once for all
    the slices. However bcf can record number of imagery from different
    imagining detectors (BSE, SEI, ARGUS, etc...) thus access to imagery
    is provided by method which retrieves image according to image index.
    
    Methods of object starts with 'get' or 'estimate' prefixes.
    Both the methods and initialized values have mixed case when it returns
    shortcut to objectified xml part, else it returns value.
    """
    def __init__(self, xml_str):
        oparser = objectify.makeparser(recover=True)
        root = objectify.fromstring(xml_str, parser=oparser).ClassInstance
        try:
            self.name = str(root.attrib['Name'])
        except:
            self.name = 'Undefinded'
        self.datetime = datetime.strptime(' '.join([str(root.Header.Date),
                                                    str(root.Header.Time)]),
                                          "%d.%m.%Y %H:%M:%S")
        self.version = int(root.Header.FileVersion)
        semData = root.xpath("ClassInstance[@Type='TRTSEMData']")[0]
        #create containers
        self.sem = Container()
        self.stage = Container()
        self.image = Container()
        # sem acceleration voltage, working distance, magnification:
        self.sem.hv = float(semData.HV) # in kV
        self.sem.wd = float(semData.WD) # in mm
        self.sem.mag = float(semData.Mag) # in times
        # image/hypermap resolution in um/pixel:
        self.image.x_res = float(semData.DX)
        self.image.y_res = float(semData.DY)
        semStageData = root.xpath("ClassInstance[@Type='TRTSEMStageData']")[0]
        # stage position data in um cast to m (that data anyway is not used by hyperspy):
        try:
            self.stage.x = float(semStageData.X) / 1.0e6
            self.stage.y = float(semStageData.Y) / 1.0e6
        except:
            self.stage.x = self.stage.y = None
        try:
            self.stage.z = float(semStageData.Z) / 1.0e6
        except:
            self.stage.z = None
        try:
            self.stage.rotation = float(semStageData.Rotation) # in degrees
        except:
            self.stage.rotation = None
        DSPConf = root.xpath("ClassInstance[@Type='TRTDSPConfiguration']")[0]
        self.stage.tilt_angle = float(DSPConf.TiltAngle) # manualy described sample tilt angle?
        imageData = root.xpath("ClassInstance[@Type='TRTImageData']")[0]
        self.image.width = int(imageData.Width)  # in pixels
        self.image.height = int(imageData.Height)  # # in pixels
        self.image.plane_count = int(imageData.PlaneCount)
        self.multi_image = int(imageData.MultiImage)
        self.image.images = []
        for i in range(self.image.plane_count):
            img = imageData.xpath("Plane"+str(i))[0]
            raw = codecs.decode((img.Data.text).encode('ascii'), 'base64')
            array1 = np.fromstring(raw, dtype=np.uint16)
            if any(array1):
                temp_img = Container()
                temp_img.data = array1.reshape((self.image.height,
                                                self.image.width))
                temp_img.detector_name = str(img.Description.text)
                self.image.images.append(temp_img)
        self.selection = []
        try:
            selection = root.xpath("ClassInstance[@Type='TRTContainerClass']/ChildClassInstances/ClassInstance[@Type='TRTElementInformationList']/ClassInstance[@Type='TRTSpectrumRegionList']/ChildClassInstances")[0] 
            for j in selection.xpath("ClassInstance[@Type='TRTSpectrumRegion']"):
                self.selection.append(int(j.Element))
        except:
            pass #what elese to do, no element selection....
        
        self.line_counter = np.fromstring(str(root.LineCounter), dtype=np.uint16, sep=',')
        self.channel_count = int(root.ChCount)
        self.mapping_count = int(root.DetectorCount)
        self.channel_factors = {}
        self.spectra_data = {}
        for i in range(self.mapping_count):
            self.channel_factors[i] = int(root.xpath("ChannelFactor"+str(i))[0])
            self.spectra_data[i] = EDXSpectrum(root.xpath("SpectrumData"+str(i))[0].ClassInstance)
        
        
    def estimate_map_channels(self, index=0):
        """ estimates minimal size of array that any pixel of spectra would
        not be truncated.
        args:
        index -- index of the map channels if multiply hypermaps are present
        in the same bcf.
        returns:
        index of maximum non empty channel of all hypermap spectras.
        """
        #the lazy method 1:
        #return int(self.root.ChCount) // int(self.root.ChannelFactor0)
        #the method 2:
        sum_eds = self.spectra_data[index].data
        return sum_eds.nonzero()[0][-1] + 1  # +1 to get the number not the index
    
    def estimate_map_depth(self, index=0):
        """ estimates minimal dtype of array that any spectra of all pixels would be
        not truncated.
        args:
        index -- index of the map channels if multiply hypermaps are present
        in the same bcf.
        returns:
        numpy dtype large enought to use in final hypermap numpy array.
        
        The method estimates the value from sum eds spectra, dividing the maximal
        value from raster width and hight and to be on the safe side multiplying by 2.
        """
        sum_eds = self.spectra_data[index].data
        roof = np.max(sum_eds) // self.image.width // self.image.height * 2 #this is allways 0kV peak
        if roof > 0xFF:
            if roof > 0xFFFF:
                depth = np.uint32
            else:
                depth = np.uint16
        else:
            depth = np.uint8
        return depth
    
    def get_spectra_metadata(self, index=0):
        return self.spectra_data[index]


st = {1: 'B', 2: 'B', 4: 'H', 8: 'I',16: 'Q'}


def bin_to_numpy(data, pointers, max_channels, depth):
    """unpacks the delphi/bruker binary hypermap and returns
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
    height, width = strct_unp('<ii', data.read(8))  #+8
    total_pixels = height * width
    total_channels = total_pixels * max_channels
    #hyper map as very flat array:
    vfa = np.zeros(total_channels, dtype=depth)
    for pix in range(0,total_pixels,1):
        if pointers[pix] > 0:
            ##############func###########################################
            data.seek(pointers[pix])
            #d_ dummy - throwaway
            chan1, chan2, d_, flag, data_size1, n_of_pulses, data_size2, d_ =\
                                            strct_unp('<HHIHHHHH', data.read(18)) # pointer +18
            if flag == 1:  # and (chan1 != chan2)
                data1 = data.read(data_size2)                    # pointer + data_size2
                """Unpack packed 12-bit data to 16-bit uints"""
                switched_i2 = np.fromstring(data1,
                                            dtype=np.uint16
                                            ).byteswap(True)
                data2 = np.fromstring(switched_i2.tostring(),
                                        dtype=np.uint8
                                        ).repeat(2)
                mask = np.ones_like(data2, dtype=bool)
                mask[0::6] = mask[5::6] = False
                # Reinterpret expanded as 16-bit
                exp16 = np.fromstring(data2[mask].tostring(),
                                        np.uint16,
                                        count=n_of_pulses
                                        ).byteswap(True)
                exp16[0::2] >>= 4             # Shift every second short by 4
                exp16 &= np.uint16(0x0FFF)    # Mask upper 4-bits on all shorts
                pixel = np.bincount(exp16, minlength=chan1-1)
            else:
                #######function instructed to pixel##########################
                offset = 0
                pixel = []
                while offset < data_size2-4:
                    #size, channels = data.read(2) #this would work on py3
                    size, channels = strct_unp('<BB',data.read(2)) # this is needed on py2
                    if size == 0:
                        #pixel += [0 for l in range(channels)]
                        pixel += channels * [0]
                        offset += 2
                    else:
                        addition = strct_unp('<'+st[size*2],
                                                data.read(size)
                                                )[0]
                        if size == 1:
                            # special case with nibble switching
                            lenght = -(-channels // 2)
                            #a = list(data.read(lenght))  # valid py3 code
                            a = strct_unp('<'+'B'*lenght, data.read(lenght)) #this have to be used on py2
                            g = []
                            for i in a:
                                g += (i & 0x0F) + addition, (i >> 4) + addition #
                            pixel += g[:channels]
                        else:
                            lenght = int(channels * size / 2)
                            temp = strct_unp('<' + channels*st[size],
                                            data.read(lenght))
                            pixel += [l + addition for l in temp]
                        offset += 2 + size + lenght
                if chan2 < chan1:
                    rest =  chan1 - chan2
                    #pixel += [0 for i in range(0,rest,1)]
                    pixel += rest * [0]
                # additional data size:
                if n_of_pulses > 0:
                    add_s = strct_unp('<I', data.read(4))[0]
                    # the additional pulses:
                    thingy = strct_unp('<'+'H'*n_of_pulses, data.read(add_s))
                    for i in thingy:
                        pixel[i] += 1
            vfa[0+max_channels*pix : chan1+max_channels*pix] = pixel
    vfa.resize((height, width, max_channels))
    return vfa.swapaxes(2,0)


def spect_pos_from_file(sp_data):
    """(intended to BCF v2) Reads and returns
    the offset/pointer table of the pixels as numpy
    array (identical to function bin_to_spect_pos).
    The function is supposed to be applied upon
    version2 bcf, where such table of pixels is
    already precalculated; however in version1
    there is no table, and instead
    the function bin_to_spect_pos() should be used.
    """
    sp_data.seek(0)
    height, width = strct_unp('<ii', sp_data.read(8))
    n_of_pix = height * width
    mapingpointers2 = strct_unp('<'+'q'*n_of_pix,
                                    sp_data.read())
    return mapingpointers2


def bin_to_spect_pos(data):
    """parses whole data stream and creates numpy array
    with pixel offsets/pointers pointing to SpectrumData
    file inside bruker bcf container. (intended BCF v1)
    Such table is presented in bcf version 2, but is not
    in version 1.
    
    Arguments:
    data -- io.BytesIO string with data of SpectrumData*
    Returns flat numpy array.
    """
    data.seek(0)
    height, width = strct_unp('<ii', data.read(8))
    n_of_pix = height * width
    data.seek(0x1A0)
    # create the numpy array with all values -1
    mapping_pointers = [-1 for h in range(0, n_of_pix, 1)]
    #mapping_pointers = np.full(n_of_pix, -1, dtype=np.int64)
    for line_cnt in range(0, height, 1):
        #line_head contains number of non-empty pixels in line
        line_head = strct_unp('<i', data.read(4))[0]
        for pix_cnt in range(0, line_head, 1):
            #x_index of the pixel:
            x_pix = strct_unp('<i', data.read(4))[0] + width * line_cnt
            offset = data.tell()
            mapping_pointers[x_pix] = offset
            # skip channel number and some placeholder:
            data.seek(8, 1)
            flag, data_size1, n_of_pulses, data_size2 = strct_unp(
                                                 '<HHHH', data.read(8))
            data.seek(2, 1) # always 0x0000
            # depending to packing type (flag) do:
            if flag == 1:  # and (chan1 != chan2)
                data.seek(data_size2, 1) #skip to next pixel/line
            else:
                if n_of_pulses > 0:
                    data.seek(data_size2-4, 1) #skip to pulses size
                    #additional pulses for data with flag 2 and 3:
                    add_s = strct_unp('<i', data.read(4))[0]
                    data.seek(add_s, 1)
                else:  # if there is no addition pulses, jump to
                    data.seek(data_size2, 1)  # next pixel or line
    return mapping_pointers

def bin_to_line_pos(data):
    """parses whole data stream and creates list of tuples
    with data offset and size per line
    
    Arguments:
    data -- io.BytesIO string with data of SpectrumData*
    Returns flat numpy array.
    """
    data.seek(0)
    height, width = strct_unp('<ii', data.read(8))
    n_of_pix = height * width
    data.seek(0x1A0)
    #mapping_pointers = np.full(n_of_pix, -1, dtype=np.int64)
    for line_cnt in range(0, height, 1):
        #line_head contains number of non-empty pixels in line
        line_head = strct_unp('<i', data.read(4))[0]
        for pix_cnt in range(0, line_head, 1):
            #x_index of the pixel:
            x_pix = strct_unp('<i', data.read(4))[0] + width * line_cnt
            offset = data.tell()
            mapping_pointers[x_pix] = offset
            # skip channel number and some placeholder:
            data.seek(8, 1)
            flag, data_size1, n_of_pulses, data_size2 = strct_unp(
                                                 '<HHHH', data.read(8))
            data.seek(2, 1) # always 0x0000
            # depending to packing type (flag) do:
            if flag == 1:  # and (chan1 != chan2)
                data.seek(data_size2, 1) #skip to next pixel/line
            else:
                if n_of_pulses > 0:
                    data.seek(data_size2-4, 1) #skip to pulses size
                    #additional pulses for data with flag 2 and 3:
                    add_s = strct_unp('<i', data.read(4))[0]
                    data.seek(add_s, 1)
                else:  # if there is no addition pulses, jump to
                    data.seek(data_size2, 1)  # next pixel or line
    return mapping_pointers


class BCF_reader(SFS_reader):
    def __init__(self, filename):
        SFS_reader.__init__(self, filename)
        self.header = HyperHeader(self.get_file('EDSDatabase/HeaderData').get_as_BytesIO_string().getvalue())
    
    def parse_hyper_map(self, index=0):
        """ returns the numpy array from given bcf file for given slice

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
        data = self.get_file('EDSDatabase/SpectrumData' + str(ind)).get_as_BytesIO_string()
        max_channels = self.header.estimate_map_channels(index=ind)
        depth = self.header.estimate_map_depth(index=ind)
        if self.header.version == 1:
            pointers = bin_to_spect_pos(data)
        else:
            sp_data = self.get_file('EDSDatabase/SpectrumPositions' + str(ind)).get_as_BytesIO_string()
            pointers = spect_pos_from_file(sp_data)

        return bin_to_numpy(data, pointers, max_channels, depth)
        
    def _parse_line_positions(self, index=0):
        ind = index
        data = self.get_file('EDSDatabase/SpectrumData' + str(ind)).get_as_BytesIO_string()
        pointers = bin_to_spect_pos(data)
        return pointers
    
    def persistent_parse_hypermap(self, index=0, downsample=None):
        ind = index
        dwn = downsample
        self.hypermap[ind] = HyperMap(self.parse_hyper_map(index=ind), self, downsample=dwn, index=ind)



class HyperMap(object):
    def __init__(self, nparray, parent, index=0, downsample=None):
        ind = index
        sp_meta = parent.header.get_spectra_metadata(index=ind)
        self.calib_abs = sp_meta.calibAbs
        self.calib_lin = sp_meta.calibLin
        self.xcalib = parent.header.image.x_res
        self.ycalib = parent.header.image.y_res
        if downsample and type(downsample) == int:
            self.hypermap = block_reduce(nparray,(1,downsample,downsample), func=np.sum)
        else:
            self.hypermap = nparray