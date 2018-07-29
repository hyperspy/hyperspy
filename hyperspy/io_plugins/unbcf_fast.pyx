import cython
import numpy as np
import sys

cdef int byte_order
if sys.byteorder == 'little':
    byte_order = 0
else:
    byte_order = 1

from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t

# fused unsigned integer type for generalised programing:

ctypedef fused channel_t:
    uint8_t
    uint16_t
    uint32_t


# instructivelly packed array structs:

cdef packed struct Bunch_head:  # size 2bytes
    uint8_t size
    uint8_t channels

# endianess agnostic reading functions... probably very slow:

@cython.boundscheck(False)
cdef uint16_t read_16(unsigned char *pointer):

    return ((<uint16_t>pointer[1]<<8) & 0xff00) | <uint16_t>pointer[0]

@cython.boundscheck(False)
cdef uint32_t read_32(unsigned char *pointer):

    return ((<uint32_t>pointer[3]<<24) & <uint32_t>0xff000000) |\
           ((<uint32_t>pointer[2]<<16) & <uint32_t>0xff0000) |\
           ((<uint32_t>pointer[1]<<8) & <uint32_t>0xff00) |\
             <uint32_t>pointer[0]

@cython.boundscheck(False)
cdef uint64_t read_64(unsigned char *pointer):
    # skiping the most high bits, as such a huge values is impossible
    # for present bruker technology. If it would change - uncomment bellow and recompile.
    #return ((<uint64_t>pointer[7]<<56) & <uint64_t>0xff00000000000000) |\
    #       ((<uint64_t>pointer[6]<<48) & <uint64_t>0xff000000000000) |\
    #       ((<uint64_t>pointer[5]<<40) & <uint64_t>0xff0000000000) |\
    return ((<uint64_t>pointer[4]<<32) & <uint64_t>0xff00000000) |\
           ((<uint64_t>pointer[3]<<24) & <uint64_t>0xff000000) |\
           ((<uint64_t>pointer[2]<<16) & <uint64_t>0xff0000) |\
           ((<uint64_t>pointer[1]<<8) & <uint64_t>0xff00) |\
             <uint64_t>pointer[0]


# datastream class:

@cython.boundscheck(False)
cdef class DataStream:

    cdef unsigned char *buffer2
    cdef int size, size_chnk
    cdef int offset
    cdef bytes raw_bytes
    cdef public object blocks  # public - because it is python object

    def __cinit__(self, blocks, int size_chnk):
        self.size_chnk = size_chnk
        self.size = size_chnk
        self.offset = 0

    def __init__(self, blocks, int size_chnk):
        self.blocks = blocks
        self.raw_bytes = next(self.blocks)  # python bytes buffer
        self.buffer2 = <bytes>self.raw_bytes  # C unsigned char buffer

    cdef void seek(self, int value):
        """move offset to given value.
        NOTE: it do not check if value is in bounds of buffer!"""
        self.offset = value

    cdef void skip(self, int length):
        """increase offset by given value,
        check if new offset is in bounds of buffer length
        else load up next block"""
        if (self.offset + length) > self.size:
            self.load_next_block()
        self.offset = self.offset + length

    cdef uint8_t read_8(self):
        if (self.offset + 1) > self.size:
            self.load_next_block()
        self.offset += 1
        return <uint8_t>self.buffer2[self.offset-1]

    cdef uint16_t read_16(self):
        if (self.offset + 2) > self.size:
            self.load_next_block()
        self.offset += 2
        # endianess agnostic way... probably very slow:
        return read_16(&self.buffer2[self.offset-2])

    cdef uint32_t read_32(self):
        if (self.offset + 4) > self.size:
            self.load_next_block()
        self.offset += 4
        # endianess agnostic way... probably very slow:
        return read_32(&self.buffer2[self.offset-4])

    cdef uint64_t read_64(self):
        if (self.offset + 8) > self.size:
            self.load_next_block()
        self.offset += 8
        return read_64(&self.buffer2[self.offset-8])

    cdef unsigned char *ptr_to(self, int length):
        """get the pointer to the raw buffer,
        making sure the array have the required length
        counting from the offset, increase the internal offset
        by given length"""
        if (self.offset + length) > self.size:
            self.load_next_block()
        self.offset += length
        return &self.buffer2[self.offset-length]

    cdef void load_next_block(self):
        """take the reminder of buffer (offset:end) and
        append new block of raw data, and overwrite old buffer
        handle with new, set offset to 0"""
        self.size = self.size_chnk + self.size - self.offset
        self.buffer2 = b''
        self.raw_bytes = self.raw_bytes[self.offset:] + next(self.blocks)
        self.offset = 0
        self.buffer2 = <bytes>self.raw_bytes


# function for looping throught the bcf pixels:

@cython.cdivision(True)
@cython.boundscheck(False)
cdef bin_to_numpy(DataStream data_stream,
                  channel_t[:, :, :] hypermap,
                  int max_chan,
                  int downsample):

    cdef uint32_t height, width, pix_in_line, pixel_x, add_pulse_size
    cdef uint32_t dummy1, line_cnt, data_size2
    cdef uint16_t chan1, chan2, flag, data_size1, n_of_pulses
    cdef uint16_t add_val, j

    height = data_stream.read_32()
    width = data_stream.read_32()
    data_stream.seek(<int>0x1A0) #the begining of the array
    for line_cnt in range(height):
        pix_in_line = data_stream.read_32()
        for dummy1 in range(pix_in_line):
            pixel_x = data_stream.read_32()
            chan1 = data_stream.read_16()
            chan2 = data_stream.read_16()
            data_stream.skip(4)  # unknown static value
            flag = data_stream.read_16()
            data_size1 = data_stream.read_16()
            n_of_pulses = data_stream.read_16()
            data_size2 = data_stream.read_32()
            if flag == 0:
                unpack16bit(hypermap,
                            pixel_x // downsample,
                            line_cnt // downsample,
                            data_stream.ptr_to(data_size2),
                            n_of_pulses,
                            max_chan)
            elif flag == 1:
                unpack12bit(hypermap,
                            pixel_x // downsample,
                            line_cnt // downsample,
                            data_stream.ptr_to(data_size2),
                            n_of_pulses,
                            max_chan)
            else:
                unpack_instructed(hypermap,
                                  pixel_x // downsample,
                                  line_cnt // downsample,
                                  data_stream.ptr_to(data_size2 - 4),
                                  data_size2 - 4,
                                  max_chan)
                if n_of_pulses > 0:
                    add_pulse_size = data_stream.read_32()
                    for j in range(n_of_pulses):
                        add_val = data_stream.read_16()
                        if add_val < max_chan:
                            hypermap[line_cnt // downsample,
                                     pixel_x // downsample,
                                     add_val] += 1
                else:
                    data_stream.skip(4)


#functions to extract pixel spectrum:

@cython.cdivision(True)
@cython.boundscheck(False)
cdef void unpack_instructed(channel_t[:, :, :] dest, int x, int y,
                            unsigned char * src, uint16_t data_size,
                            int cutoff):
    """
    unpack instructivelly packed delphi array into selection
    of memoryview
    """
    cdef int offset = 0
    cdef int channel = 0
    cdef int i, j, length
    cdef int gain = 0
    cdef Bunch_head* head
    cdef uint16_t val16
    cdef uint32_t val32
    while (offset < data_size):
        head =<Bunch_head*>&src[offset]
        offset +=2
        if head.size == 0:  # empty channels (zero counts)
            channel += head.channels
        else:
            if head.size == 1:
                gain = <int>(src[offset])
            elif head.size == 2:
                gain = <int>read_16(&src[offset])
            elif head.size == 4:
                gain = <int>read_32(&src[offset])
            else:
                gain = <int>read_64(&src[offset])
            offset += head.size
            if head.size == 1:  # special nibble switching case
                for i in range(head.channels):
                    if (i+channel) < cutoff:
                        #reverse the nibbles:
                        if i % 2 == 0:
                            dest[y, x, i+channel] += <channel_t>((src[offset +(i//2)] & 15) + gain)
                        else:
                            dest[y, x, i+channel] += <channel_t>((src[offset +(i//2)] >> 4) + gain)
                if head.channels % 2 == 0:
                    length = <int>(head.channels // 2)
                else:
                    length = <int>((head.channels // 2) +1)
            elif head.size == 2:
                for i in range(head.channels):
                    if (i+channel) < cutoff:
                        dest[y, x, i+channel] += <channel_t>(src[offset + i] + gain)
                length = <int>(head.channels * head.size // 2)
            elif head.size == 4:
                for i in range(head.channels):
                    if (i+channel) < cutoff:
                        val16 = read_16(&src[offset + i*2])
                        dest[y, x, i+channel] += <channel_t>(val16 + gain)
                length = <int>(head.channels * head.size // 2)
            else:
                for i in range(head.channels):
                    if (i+channel) < cutoff:
                        val32 = read_32(&src[offset + i*2])
                        dest[y, x, i+channel] += <channel_t>(val32 + gain)
                length = <int>(head.channels * head.size // 2)
            offset += length
            channel += head.channels


@cython.cdivision(True)
@cython.boundscheck(False)
cdef void unpack12bit(channel_t[:, :, :] dest, int x, int y,
                      unsigned char * src,
                      uint16_t no_of_pulses,
                      int cutoff):
    """unpack 12bit packed array into selection of memoryview"""
    cdef int i, channel
    for i in range(no_of_pulses):
        if i % 4 == 0:
            channel = <int>((src[6*(i//4)] >> 4)+(src[6*(i//4)+1] << 4))
        elif i % 4 == 1:
            channel = <int>(((src[6*(i//4)] << 8 ) + (src[6*(i//4)+3])) & 0xFFF)
        elif i % 4 == 2:
            channel = <int>((src[6*(i//4)+2] << 4) + (src[6*(i//4)+5] >> 4))
        else:
            channel = <int>(((src[6*(i//4)+5] << 8) + src[6*(i//4)+4]) & 0xFFF)
        if channel < cutoff:
            dest[y, x, channel] += 1


@cython.cdivision(True)
@cython.boundscheck(False)
cdef void unpack16bit(channel_t[:, :, :] dest, int x, int y,
                      unsigned char * src,
                      uint16_t no_of_pulses,
                      int cutoff):
    """unpack 16bit packed array into selection of memoryview"""
    cdef int i, channel
    for i in range(no_of_pulses):
        channel = <int>(src[2*i] + ((src[2*i+1] << 8) & 0xff00))
        if channel < cutoff:
            dest[y, x, channel] += 1


#the main function:

def parse_to_numpy(virtual_file, shape, dtype, downsample=1):
    """parse the hyperspectral cube from brukers bcf binary file
    and return it as numpy array"""
    blocks, block_size = virtual_file.get_iter_and_properties()[:2]
    map_depth = shape[2]
    hypermap = np.zeros(shape, dtype=dtype)
    cdef DataStream data_stream = DataStream(blocks, block_size)
    if dtype == np.uint8:
        bin_to_numpy[uint8_t](data_stream, hypermap, map_depth, downsample)
        return hypermap
    elif dtype == np.uint16:
        bin_to_numpy[uint16_t](data_stream, hypermap, map_depth, downsample)
        return hypermap
    elif dtype == np.uint32:
        bin_to_numpy[uint32_t](data_stream, hypermap, map_depth, downsample)
        return hypermap
    else:
        raise NotImplementedError('64bit array not implemented!')
