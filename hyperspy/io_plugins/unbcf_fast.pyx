import cython
import numpy as np
import sys

cdef int byte_order
if sys.byteorder == 'little':
    byte_order = 0
else:
    byte_order = 1

from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t, int64_t
from libc.string cimport memcpy, memmove

# structs for headers:

cdef struct Map_Header: #size 8
    uint32_t height
    uint32_t width

cdef packed struct Line_Header: #size 4
    uint32_t pixels

cdef packed struct Additional_pulses: #size 4
    uint32_t size

cdef packed struct Pixel_Header: #size 22
    uint32_t pixel_x
    uint16_t chan1
    uint16_t chan2
    uint32_t dummy         # unknown purpoise
    uint16_t flag
    uint16_t data_size1
    uint16_t n_of_pulses
    uint16_t data_size2
    uint16_t dummy2        # unknown purpoise

# instructivelly packed array structs:

cdef packed struct Bunch_head: #size 2bytes
    uint8_t size
    uint8_t channels

cdef struct Gain8:
    uint8_t val

cdef struct Gain16:
    uint16_t val

cdef struct Gain32:
    uint32_t val

cdef struct Gain64:
    uint64_t val

cdef struct Val16:
    uint16_t val

cdef struct Val32:
    uint32_t val


@cython.cdivision(True)
@cython.boundscheck(False)
def bin_to_8bit_numpy(blocks,
                      int size_chnk,
                      int total_blocks,
                      uint8_t[:, :, :] hypermap8,
                      int max_chan,
                      int downsample):
    cdef int dummy1, line_cnt, i, j
    cdef int size
    cdef bytes raw_bytes
    cdef int block_no = 0
    cdef Map_Header* header
    cdef char map_header_data[8]
    cdef Pixel_Header* pixel
    cdef char pixel_header_data[22]
    cdef Line_Header* pix_in_line
    cdef char line_header_data[4]
    cdef Additional_pulses* add_s
    cdef char addition_pulses_data[4]
    cdef unsigned char *buffer1
    cdef Val16* add_val
    raw_bytes = next(blocks)
    buffer1 = <bytes>raw_bytes
    memcpy(map_header_data, &buffer1[0], 8)
    header = <Map_Header*>&map_header_data[0]
    if byte_order == 1:
        header.height = swap_32(header.height)
        header.width = swap_32(header.width)
    cdef int offset
    offset = 0x1A0 #the begining of the array
    size = size_chnk
    for line_cnt in range(header.height):
        if (offset + 4) > size:
            size = size_chnk + size - offset
            buffer1 = b''
            raw_bytes = raw_bytes[offset:] + next(blocks)
            offset = 0
            buffer1 = <bytes>raw_bytes
        memcpy(line_header_data, &buffer1[offset], 4)
        pix_in_line = <Line_Header*>&line_header_data[0]
        if byte_order == 1:
            pix_in_line.pixels = swap_32(pix_in_line.pixels)
        offset += 4
        for dummy1 in range(pix_in_line.pixels):
            if (offset + 22) > size:
                size = size_chnk + size - offset
                buffer1 = b''
                raw_bytes = raw_bytes[offset:] + next(blocks)
                buffer1 = <bytes>raw_bytes
                offset = 0
            memcpy(pixel_header_data, &buffer1[offset], 22)
            pixel = <Pixel_Header*>&pixel_header_data[0]
            if byte_order == 1:
                pixel.pixel_x = swap_32(pixel.pixel_x)
                pixel.chan1 = swap_16(pixel.chan1)
                pixel.chan2 = swap_16(pixel.chan2)
                pixel.flag = swap_16(pixel.flag)
                pixel.data_size1 = swap_16(pixel.data_size1)
                pixel.n_of_pulses = swap_16(pixel.n_of_pulses)
                pixel.data_size2 = swap_16(pixel.data_size2)
            offset += 22
            if (offset + pixel.data_size2) > size:
                size = size_chnk + size - offset
                buffer1 = b''
                raw_bytes = raw_bytes[offset:] + next(blocks)
                buffer1 = <bytes>raw_bytes
                offset = 0
            if pixel.flag == 1:
                unpack12_to_8bit(hypermap8,
                                 pixel.pixel_x // downsample,
                                 line_cnt // downsample,
                                 &buffer1[offset],
                                 pixel.n_of_pulses,
                                 max_chan)
                offset += pixel.data_size2
            else:
                instructed_to_8bit(hypermap8,
                                   pixel.pixel_x // downsample,
                                   line_cnt // downsample,
                                   &buffer1[offset],
                                   pixel.data_size2 -4,
                                   max_chan)
                offset += pixel.data_size2 - 4
                if pixel.n_of_pulses > 0:
                    add_s = <Additional_pulses*>&buffer1[offset]
                    if byte_order == 1:
                        add_s.size = swap_32(add_s.size)
                    offset += 4
                    if (offset + add_s.size) > size:
                        size = size_chnk + size - offset
                        buffer1 = b''
                        raw_bytes = raw_bytes[offset:] + next(blocks)
                        buffer1 = <bytes>raw_bytes
                        offset = 0
                    for j in range(pixel.n_of_pulses):
                        add_val = <Val16*>&buffer1[offset + j*2]
                        if byte_order == 1:
                            add_val.val = swap_16(add_val.val)
                        if add_val.val < max_chan:
                            hypermap8[add_val.val,
                                      pixel.pixel_x // downsample,
                                      line_cnt // downsample] += 1
                    offset += add_s.size
                else:
                    offset +=4


@cython.cdivision(True)
@cython.boundscheck(False)
def bin_to_16bit_numpy(blocks,
                      int size_chnk,
                      int total_blocks,
                      uint16_t[:, :, :] hypermap16,
                      int max_chan,
                      int downsample):
    cdef int dummy1, line_cnt, i, j
    cdef int size
    cdef bytes raw_bytes
    cdef int block_no = 0
    cdef Map_Header* header
    cdef char map_header_data[8]
    cdef Pixel_Header* pixel
    cdef char pixel_header_data[22]
    cdef Line_Header* pix_in_line
    cdef char line_header_data[4]
    cdef Additional_pulses* add_s
    cdef char addition_pulses_data[4]
    cdef unsigned char *buffer1
    cdef Val16* add_val
    raw_bytes = next(blocks)
    buffer1 = <bytes>raw_bytes
    memcpy(map_header_data, &buffer1[0], 8)
    header = <Map_Header*>&map_header_data[0]
    if byte_order == 1:
        header.height = swap_32(header.height)
        header.width = swap_32(header.width)
    cdef int offset
    offset = 0x1A0 #initiating block offset
    size = size_chnk
    for line_cnt in range(header.height):
        if (offset + 4) > size:
            size = size_chnk + size - offset
            buffer1 = b''
            raw_bytes = raw_bytes[offset:] + next(blocks)
            offset = 0
            buffer1 = <bytes>raw_bytes
        memcpy(line_header_data, &buffer1[offset], 4)
        pix_in_line = <Line_Header*>&line_header_data[0]
        if byte_order == 1:
            pix_in_line.pixels = swap_32(pix_in_line.pixels)
        offset += 4
        for dummy1 in range(pix_in_line.pixels):
            if (offset + 22) > size:
                size = size_chnk + size - offset
                buffer1 = b''
                raw_bytes = raw_bytes[offset:] + next(blocks)
                buffer1 = <bytes>raw_bytes
                offset = 0
            memcpy(pixel_header_data, &buffer1[offset], 22)
            pixel = <Pixel_Header*>&pixel_header_data[0]
            if byte_order == 1:
                pixel.pixel_x = swap_32(pixel.pixel_x)
                pixel.chan1 = swap_16(pixel.chan1)
                pixel.chan2 = swap_16(pixel.chan2)
                pixel.flag = swap_16(pixel.flag)
                pixel.data_size1 = swap_16(pixel.data_size1)
                pixel.n_of_pulses = swap_16(pixel.n_of_pulses)
                pixel.data_size2 = swap_16(pixel.data_size2)
            offset += 22
            if (offset + pixel.data_size2) > size:
                size = size_chnk + size - offset
                buffer1 = b''
                raw_bytes = raw_bytes[offset:] + next(blocks)
                buffer1 = <bytes>raw_bytes
                offset = 0
            if pixel.flag == 1:
                unpack12_to_16bit(hypermap16,
                                 pixel.pixel_x // downsample,
                                 line_cnt // downsample,
                                 &buffer1[offset],
                                 pixel.n_of_pulses,
                                 max_chan)
                offset += pixel.data_size2
            else:
                instructed_to_16bit(hypermap16,
                                   pixel.pixel_x // downsample,
                                   line_cnt // downsample,
                                   &buffer1[offset],
                                   pixel.data_size2 -4,
                                   max_chan)
                offset += pixel.data_size2 - 4
                if pixel.n_of_pulses > 0:
                    add_s = <Additional_pulses*>&buffer1[offset]
                    if byte_order == 1:
                        add_s.size = swap_32(add_s.size)
                    offset += 4
                    if (offset + add_s.size) > size:
                        size = size_chnk + size - offset
                        buffer1 = b''
                        raw_bytes = raw_bytes[offset:] + next(blocks)
                        buffer1 = <bytes>raw_bytes
                        offset = 0
                    for j in range(pixel.n_of_pulses):
                        add_val = <Val16*>&buffer1[offset + j*2]
                        if byte_order == 1:
                            add_val.val = swap_16(add_val.val)
                        if add_val.val < max_chan:
                            hypermap16[add_val.val,
                                      pixel.pixel_x // downsample,
                                      line_cnt // downsample] += 1
                    offset += add_s.size
                else:
                    offset +=4

@cython.cdivision(True)
@cython.boundscheck(False)
def bin_to_32bit_numpy(blocks,
                      int size_chnk,
                      int total_blocks,
                      uint32_t[:, :, :] hypermap32,
                      int max_chan,
                      int downsample):
    cdef int dummy1, line_cnt, i, j
    cdef int size
    cdef bytes raw_bytes
    cdef int block_no = 0
    cdef Map_Header* header
    cdef char map_header_data[8]
    cdef Pixel_Header* pixel
    cdef char pixel_header_data[22]
    cdef Line_Header* pix_in_line
    cdef char line_header_data[4]
    cdef Additional_pulses* add_s
    cdef char addition_pulses_data[4]
    cdef unsigned char *buffer1
    cdef Val16* add_val
    raw_bytes = next(blocks)
    buffer1 = <bytes>raw_bytes
    memcpy(map_header_data, &buffer1[0], 8)
    header = <Map_Header*>&map_header_data[0]
    if byte_order == 1:
        header.height = swap_32(header.height)
        header.width = swap_32(header.width)
    cdef int offset
    offset = 0x1A0 #initiating block offset
    size = size_chnk
    for line_cnt in range(header.height):
        if (offset + 4) > size:
            size = size_chnk + size - offset
            buffer1 = b''
            raw_bytes = raw_bytes[offset:] + next(blocks)
            offset = 0
            buffer1 = <bytes>raw_bytes
        memcpy(line_header_data, &buffer1[offset], 4)
        pix_in_line = <Line_Header*>&line_header_data[0]
        if byte_order == 1:
            pix_in_line.pixels = swap_32(pix_in_line.pixels)
        offset += 4
        for dummy1 in range(pix_in_line.pixels):
            if (offset + 22) > size:
                size = size_chnk + size - offset
                buffer1 = b''
                raw_bytes = raw_bytes[offset:] + next(blocks)
                buffer1 = <bytes>raw_bytes
                offset = 0
            memcpy(pixel_header_data, &buffer1[offset], 22)
            pixel = <Pixel_Header*>&pixel_header_data[0]
            if byte_order == 1:
                pixel.pixel_x = swap_32(pixel.pixel_x)
                pixel.chan1 = swap_16(pixel.chan1)
                pixel.chan2 = swap_16(pixel.chan2)
                pixel.flag = swap_16(pixel.flag)
                pixel.data_size1 = swap_16(pixel.data_size1)
                pixel.n_of_pulses = swap_16(pixel.n_of_pulses)
                pixel.data_size2 = swap_16(pixel.data_size2)
            offset += 22
            if (offset + pixel.data_size2) > size:
                size = size_chnk + size - offset
                buffer1 = b''
                raw_bytes = raw_bytes[offset:] + next(blocks)
                buffer1 = <bytes>raw_bytes
                offset = 0
            if pixel.flag == 1:
                unpack12_to_32bit(hypermap32,
                                 pixel.pixel_x // downsample,
                                 line_cnt // downsample,
                                 &buffer1[offset],
                                 pixel.n_of_pulses,
                                 max_chan)
                offset += pixel.data_size2
            else:
                instructed_to_32bit(hypermap32,
                                   pixel.pixel_x // downsample,
                                   line_cnt // downsample,
                                   &buffer1[offset],
                                   pixel.data_size2 -4,
                                   max_chan)
                offset += pixel.data_size2 - 4
                if pixel.n_of_pulses > 0:
                    add_s = <Additional_pulses*>&buffer1[offset]
                    if byte_order == 1:
                        add_s.size = swap_32(add_s.size)
                    offset += 4
                    if (offset + add_s.size) > size:
                        size = size_chnk + size - offset
                        buffer1 = b''
                        raw_bytes = raw_bytes[offset:] + next(blocks)
                        buffer1 = <bytes>raw_bytes
                        offset = 0
                    for j in range(pixel.n_of_pulses):
                        add_val = <Val16*>&buffer1[offset + j*2]
                        if byte_order == 1:
                            add_val.val = swap_16(add_val.val)
                        if add_val.val < max_chan:
                            hypermap32[add_val.val,
                                      pixel.pixel_x // downsample,
                                      line_cnt // downsample] += 1
                    offset += add_s.size
                else:
                    offset +=4


@cython.cdivision(True)
@cython.boundscheck(False)
cdef void instructed_to_8bit(uint8_t[:, :, :] dest, int x, int y,
                             unsigned char * src, uint16_t data_size,
                             int cutoff):
    """
    unpack instructivelly packed delphi array into selection
    of memoryview of 8bit array
    """
    cdef int offset = 0
    cdef int channel = 0
    cdef int i, j, length
    cdef int gain = 0
    cdef Bunch_head* head
    cdef Gain8* gain8
    cdef Gain16* gain16
    cdef Gain32* gain32
    cdef Val16* val16
    while (offset < data_size):
        head =<Bunch_head*>&src[offset]
        offset +=2
        if head.size == 0:  # empty channels (zero counts)
            channel += head.channels
        else:
            if head.size == 1:
                gain8 = <Gain8*>&src[offset]
                gain = <int>gain8.val
            elif head.size == 2:
                gain16 = <Gain16*>&src[offset]
                if byte_order == 1:
                    gain16.val = swap_16(gain16.val)
                gain = <int>gain16.val
            elif head.size == 4:
                gain32 = <Gain32*>&src[offset]
                if byte_order == 1:
                    gain32.val = swap_32(gain32.val)
                gain = <int>gain32.val
            # this is theoretically impossible:
            else:
                gain = 0
            offset += head.size
            if head.size == 1:  # special nibble switching case on LE and BE machines
                for i in range(head.channels):
                    if (i+channel) < cutoff:
                        #reverse the nibbles:
                        if i % 2 == 0:
                            dest[i+channel, x, y] += <uint8_t>((src[offset +(i//2)] & 0x0F) + gain)
                        else:
                            dest[i+channel, x, y] += <uint8_t>((src[offset +(i//2)] >> 4) + gain)
                if head.channels % 2 == 0:
                    length = <int>(head.channels // 2)
                else:
                    length = <int>((head.channels // 2) +1)
            elif head.size == 2:
                for i in range(head.channels):
                    if (i+channel) < cutoff:
                        dest[i+channel, x, y] += <uint8_t>(src[offset + i] + gain)
                length = <int>(head.channels * head.size // 2)
            # further is not actual for 8 bit arrays:
            else:
                length = <int>(head.channels * head.size // 2)
            offset += length
            channel += head.channels


@cython.cdivision(True)
@cython.boundscheck(False)
cdef void instructed_to_16bit(uint16_t[:, :, :] dest, int x, int y,
                             unsigned char * src, uint16_t data_size,
                             int cutoff):
    """
    unpack instructivelly packed delphi array into selection
    of memoryview of 16bit array
    """
    cdef int offset = 0
    cdef int channel = 0
    cdef int i, j, length
    cdef int gain = 0
    cdef Bunch_head* head
    cdef Gain8* gain8
    cdef Gain16* gain16
    cdef Gain32* gain32
    cdef Val16* val16
    while (offset < data_size):
        head =<Bunch_head*>&src[offset]
        offset +=2
        if head.size == 0:  # empty channels (zero counts)
            channel += head.channels
        else:
            if head.size == 1:
                gain8 = <Gain8*>&src[offset]
                gain = <int>gain8.val
            elif head.size == 2:
                gain16 = <Gain16*>&src[offset]
                if byte_order == 1:
                    gain16.val = swap_16(gain16.val)
                gain = <int>gain16.val
            elif head.size == 4:
                gain32 = <Gain32*>&src[offset]
                if byte_order == 1:
                    gain32.val = swap_32(gain32.val)
                gain = <int>gain32.val
            else:
                #gain from far future:
                gain = 0
            offset += head.size
            if head.size == 1:  # special nibble switching case
                for i in range(head.channels):
                    if (i+channel) < cutoff:
                        #reverse the nibbles:
                        if i % 2 == 0:
                            dest[i+channel, x, y] += <uint16_t>((src[offset +(i//2)] & 0x0F) + gain)
                        else:
                            dest[i+channel, x, y] += <uint16_t>((src[offset +(i//2)] >> 4) + gain)
                if head.channels % 2 == 0:
                    length = <int>(head.channels // 2)
                else:
                    length = <int>((head.channels // 2) +1)
            elif head.size == 2:
                for i in range(head.channels):
                    if (i+channel) < cutoff:
                        dest[i+channel, x, y] += <uint16_t>(src[offset + i] + gain)
                length = <int>(head.channels * head.size // 2)
            else:
                for i in range(head.channels):
                    if (i+channel) < cutoff:
                        val16 = <Val16*>&src[offset + i*2]
                        if byte_order == 1:
                            val16.val = swap_16(val16.val)
                        dest[i+channel, x, y] += <uint16_t>(val16.val + gain)
                length = <int>(head.channels * head.size // 2)
            offset += length
            channel += head.channels

@cython.cdivision(True)
@cython.boundscheck(False)
cdef void instructed_to_32bit(uint32_t[:, :, :] dest, int x, int y,
                             unsigned char * src, uint16_t data_size,
                             int cutoff):
    """
    unpack instructivelly packed delphi array into selection
    of memoryview of 32bit array
    """
    cdef int offset = 0
    cdef int channel = 0
    cdef int i, j, length
    cdef int gain = 0
    cdef Bunch_head* head
    cdef Gain8* gain8
    cdef Gain16* gain16
    cdef Gain32* gain32
    cdef Gain64* gain64
    cdef Val16* val16
    cdef Val32* val32
    while (offset < data_size):
        head =<Bunch_head*>&src[offset]
        offset +=2
        if head.size == 0:  # empty channels (zero counts)
            channel += head.channels
        else:
            if head.size == 1:
                gain8 = <Gain8*>&src[offset]
                gain = <int>gain8.val
            elif head.size == 2:
                gain16 = <Gain16*>&src[offset]
                if byte_order == 1:
                    gain16.val = swap_16(gain16.val)
                gain = <int>gain16.val
            elif head.size == 4:
                gain32 = <Gain32*>&src[offset]
                if byte_order == 1:
                    gain32.val = swap_32(gain32.val)
                gain = <int>gain32.val
            else:
                gain64 = <Gain64*>&src[offset]
                if byte_order == 1:
                    gain64.val = swap_32(<uint32_t>(gain64.val >> 32)) | (<uint64_t>(swap_32(<uint32_t>(gain64.val & <uint64_t>0xFFFFFFFF))) << 32)
                gain = <int> gain64.val
            offset += head.size
            if head.size == 1:  # special nibble switching case
                for i in range(head.channels):
                    if (i+channel) < cutoff:
                        #reverse the nibbles:
                        if i % 2 == 0:
                            dest[i+channel, x, y] += <uint32_t>((src[offset +(i//2)] & 15) + gain)
                        else:
                            dest[i+channel, x, y] += <uint32_t>((src[offset +(i//2)] >> 4) + gain)
                if head.channels % 2 == 0:
                    length = <int>(head.channels // 2)
                else:
                    length = <int>((head.channels // 2) +1)
            elif head.size == 2:
                for i in range(head.channels):
                    if (i+channel) < cutoff:
                        dest[i+channel, x, y] += <uint32_t>(src[offset + i] + gain)
                length = <int>(head.channels * head.size // 2)
            elif head.size == 4:
                for i in range(head.channels):
                    if (i+channel) < cutoff:
                        val16 = <Val16*>&src[offset + i*2]
                        if byte_order == 1:
                            val16.val = swap_16(val16.val)
                        dest[i+channel, x, y] += <uint32_t>(val16.val + gain)
                length = <int>(head.channels * head.size // 2)
            else:
                for i in range(head.channels):
                    if (i+channel) < cutoff:
                        val32 = <Val32*>&src[offset + i*2]
                        if byte_order == 1:
                            val32.val = swap_32(val32.val)
                        dest[i+channel, x, y] += <uint32_t>(val32.val + gain)
                length = <int>(head.channels * head.size // 2)
            offset += length
            channel += head.channels



@cython.cdivision(True)
@cython.boundscheck(False)
cdef void unpack12_to_8bit(uint8_t[:, :, :] dest, int x, int y,
                           unsigned char * src,
                           uint16_t no_of_pulses,
                           int cutoff):
    """unpack 12bit packed array into selection of memoryview of 8bit array """
    cdef int i, channel
    for i in range(no_of_pulses):
        if i % 4 == 0:
            channel = <int>((src[6*(i//4)] >> 4) + (src[6*(i//4)+1] << 4))
        elif i % 4 == 1:
            channel = <int>(((src[6*(i//4)] << 8 ) + (src[6*(i//4)+3])) & 4095)  #0xFFF
        elif i % 4 == 2:
            channel = <int>((src[6*(i//4)+2] << 4) + (src[6*(i//4)+5] >> 4))
        else:
            channel = <int>(((src[6*(i//4)+5] << 8) + src[6*(i//4)+4]) & 4095)
        if channel < cutoff:
            dest[channel, x, y] += 1

@cython.cdivision(True)
@cython.boundscheck(False)
cdef void unpack12_to_16bit(uint16_t[:, :, :] dest, int x, int y,
                           unsigned char * src,
                           uint16_t no_of_pulses,
                           int cutoff):
    """unpack 12bit packed array into selection of memoryview of 16bit array """
    cdef int i, channel
    for i in range(no_of_pulses):
        if i % 4 == 0:
            channel = <int>((src[6*(i//4)] >> 4)+(src[6*(i//4)+1] << 4))
        elif i % 4 == 1:
            channel = <int>(((src[6*(i//4)] << 8 ) + (src[6*(i//4)+3])) & 4095)
        elif i % 4 == 2:
            channel = <int>((src[6*(i//4)+2] << 4) + (src[6*(i//4)+5] >> 4))
        else:
            channel = <int>(((src[6*(i//4)+5] << 8) + src[6*(i//4)+4]) & 4095)
        if channel < cutoff:
            dest[channel, x, y] += 1

@cython.cdivision(True)
@cython.boundscheck(False)
cdef void unpack12_to_32bit(uint32_t[:, :, :] dest, int x, int y,
                           unsigned char * src,
                           uint16_t no_of_pulses,
                           int cutoff):
    """unpack 12bit packed array into selection of memoryview of 32bit array """
    cdef int i, channel
    for i in range(no_of_pulses):
        if i % 4 == 0:
            channel = <int>((src[6*(i//4)] >> 4)+(src[6*(i//4)+1] << 4))
        elif i % 4 == 1:
            channel = <int>(((src[6*(i//4)] << 8 ) + (src[6*(i//4)+3])) & 4095)
        elif i % 4 == 2:
            channel = <int>((src[6*(i//4)+2] << 4) + (src[6*(i//4)+5] >> 4))
        else:
            channel = <int>(((src[6*(i//4)+5] << 8) + src[6*(i//4)+4]) & 4095)
        if channel < cutoff:
            dest[channel, x, y] += 1

#the main function:

def parse_to_numpy(bcf, downsample=1, cutoff=None):
    blocks, block_size, total_blocks = bcf.get_iter_and_properties()
    map_depth = bcf.sfs.header.estimate_map_channels()
    if type(cutoff) == int:
        map_depth = cutoff
    dtype = bcf.sfs.header.estimate_map_depth(downsample=downsample)
    hypermap = np.zeros((map_depth,
                         bcf.sfs.header.image.width // downsample,
                         bcf.sfs.header.image.height // downsample),
                         dtype=dtype)
    if dtype == np.uint8:
        bin_to_8bit_numpy(blocks, block_size, total_blocks,
                          hypermap, map_depth, downsample)
        return hypermap
    elif dtype == np.uint16:
        bin_to_16bit_numpy(blocks, block_size, total_blocks,
                          hypermap, map_depth, downsample)
        return hypermap
    elif dtype == np.uint32:
        bin_to_32bit_numpy(blocks, block_size, total_blocks,
                          hypermap, map_depth, downsample)
        return hypermap
    else:
        print('64bit array not implemented!')


# helper functions for byteswaps for big-endian machines

cdef uint32_t swap_32(uint32_t value):
    return ((value>>24) & 255) |\
           ((value<<8) & <uint32_t>0xff0000) |\
           ((value>>8) & <uint32_t>0xff00) |\
           ((value<<24) & <uint32_t>0xff000000)

cdef uint16_t swap_16(uint16_t value):
    return ((value<<8) & 65280) | ((value>>8) & 255)
