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
# This python library provides basic reading capability of sfs proprietary
# AidAim Software(tm) files.
#
#####################################################
#Reverse engineering of this file format should be considered as FAIR USE:
#  The main reason behind RE is interoperability of Bruker composite files (*.bcf,
#   *.pan saved with Esprit software) with open source packages/software allowing
#   much more advanced data manipulation and analysis.
#
#####################################################
### SFS File format ###
#SFS is proprietary single file system developed by AidAim Software(tm)
# natively used with Delphi(tm) draconically over-expensive and useless languages.
# This function tries to implement minimal reading capabilities based on
#  reverse engineering (RE), and can become useless with future format if mentioned
#  developers would introduce critical changes.
#
# This library is very basic and can read just un-encrypted type of sfs.
# At least two compression methods were observed beeing in use
# for compression: zlib and bzip2.
# However sfs containers could use different compression methods which
#  is unknown for developer of this library, thus just zlib and bzip2
#  decompressions are implemented.
#
#STRUCTURE:
# The all non string data are little-endian.
# The data in sfs container are placed in chunks, and can be highly fragmented,
#  every chunk begins with 32-byte header.
# The SFS tries to mimic the file system, thus it contains tree/list of files,
#  file table (order and index of chunks with the file) and data.
# The header of the first (0) chunk is special as it contains:
#   * 8-bytes filled with string 'AASFSSGN' (probably means
#    AidAimSingleFileSystemSiGNature).
#   * 8-bytes some kind of signature which differs for files generated on the different
#     machines. (4 first bytes, looks random, 4 last bytes are stable on
#     same machine.)
#   * 4-bytes -- size of the chunks.
#   * the meaninng of the rest of values in this header is still unknown.
#   After this header there are some unknown values up to last 3:
#    * value at 0x140 (4-bytes (uint32)) -- the number of chunk where
#       file_tree starts
#    * value at 0x144 (4-bytes (uint32)) -- the number of the items in the
#       file_tree
#    * value at 0x148 (4-bytes (uint32)) means the total number of chunks
#       in sfs container.
#
# If file tree is larger than a chunk, then it is continued in another chunk,
#  in such case the chunk header begins with 4-byte integer chunk_index where the list
#  is continued, otherwise the beginning of the header are filled with 00's, ff's
#  or some number higher than max chunk index.
#
# The tree/list is constructed so:
#  * 0x200 total address space/item:
#    * first 0xe0 bytes contains non string information about item:
#      * 4 bytes - the chunk index with file chunk table (or begining of it)
#          (if it is 0xffffffff -- it is dir (and have no data) or file is empty)
#      * 8 bytes - the lenght of the file (if preceeding 4 bytes ff's: 0)
#      * 8 bytes + 8 bytes + 8 bytes -- 3 identical values in freshly made files,
#          most possibly "times" (creation, modification, something else?)
#      * 4 bytes of unknown; observed valules 0, 10... (guess it is file permissions,
#          most of known file systems have such feature)
#      * 4 bytes - the child of item (index, where first item 0);
#            if value == 0xffffffff, it is child of root
#      * 0xb0 bytes filled with 00's (what a waste of space)
#      * last 4 bytes (uint32) can be 0 or 1:
#        * 0 - item is a file
#        * 1 - item is a directory
#    * second part (next 0x120 bytes) -- string -- the file/dir name
#        (this probably means that file/dir name should be <= 288 chars)
#     # THE NOTE: the items in chunks is allways aligned with the beginning of chunk
#     #  (+ 32 bytes of header). The chunk size after removal of 32 bytes header and
#     #  divided by 512 (0x200) will have reminder, which have to be simply skipped.
#     #  Most commonly the reminder is of 0x1e0 (512 - 32) size, but can be different
#     #  if chunk size is not dividable by 512. The simplest practice would be to read
#     # (chunksize - 512) bytes from chunk after the header.
#
#     The table of file is constructed so:
#     * the 32byte header, where 4 first bytes contains index of chunk if table exceeds
#          space of 1 chunk, else ff's 00's or something higher than max index.
#     * ordered chunk indexes (4bytes/ index)
#
#  Compression are used (if used) just on the data part, file tree, and
#  file tables are never compressed.

import os
import io

import numpy as np
import struct

# helper functions:

def read_pointer_table(filename, item_tree, chunk_usable):
    """ The function which seeks to first pointer table retrieves
    and updates item_tree with full pointer table of file
    Args:
    filename -- sfs file
    item_tree -- the dict from get_the_file_tree function
    chunk_usable -- the the data size in chunk after substracting
      the header (32bit)
    """

    #1st determine required numpy dtype for file_chunk_tables
    sfs_size = os.path.getsize(filename)
    chunksize = chunk_usable + 32
    first_pointer = item_tree['pointers']
    size = item_tree['size']
    if sfs_size < 0xFFFF:
        table_type = np.uint16
    elif sfs_size < 0xFFFFFFFF:
        table_type = np.uint32
    else:
        table_type = np.uint64
    size_in_last_chunk = size % (chunk_usable)
    if size_in_last_chunk == 0:
        last_size = chunk_usable
    else:
        last_size = size_in_last_chunk
    item_tree['last_size'] = last_size
    table_entries = -(-size // (chunk_usable))  # getting ceiling
    table_size = -(-table_entries // (chunk_usable // 4))  # in chunks
    with open(filename, 'rb') as fn:
        fn.seek(chunksize * first_pointer + 0x118)
        if table_size > 1:
            j = 0  # chunk counter
            next_chunk = first_pointer
            temp_string = io.BytesIO()
            for j in range(0, table_size, 1):
                fn.seek(chunksize * next_chunk + 0x118)
                next_chunk = struct.unpack('<I', fn.read(4))[0]
                fn.seek(28, 1)
                temp_string.write(fn.read(chunk_usable))
            temp_string.seek(0)
            temp_table = temp_string.read()
            temp_string.close()  # optional
        else:
            fn.seek(chunksize * first_pointer + 0x138)
            temp_table = fn.read(chunk_usable)
        table = np.trim_zeros(
          np.fromstring(temp_table, dtype='uint32').astype(table_type),
          'b') * chunksize + 0x138
    item_tree['pointers'] = table

    return item_tree


def bin_tree_to_lists(data, n_items):
    """unpacks sfs list/tree of content into python lists.

    args:
    data -- the cocatenated binary/string of sfs content tree/list
    n_items -- number of items in the data

    returns: 5 lists (file table, size, parent (it's index in the sfs list),
       is_dir (boolen), name)
    """

    # due to possibility of child items appearing before parent in sfs list,
    # before we construct tree like structure, we need to sort out
    # child/parent relation, thus different kind of data should be fetched
    # into separate lists. Function have lots of space for optimization....
    # some of the values from common knowlidge about file systems are supposed
    # to be the time and file permissions, but however couldn't be sucessfully
    # RE. They are left and comented in code, maybe somebody will figure out
    # how to recalculate them

    # initializing dictionaries:
    tab_pointer = []  # index of the chunk with file table (or beginning of it)
    size = []  # in bytes
    #create_time = []
    #mod_time = []
    #wtf_time = []
    #permissions = []
    parent = []  # if 0xffffffff, then file/dir is directly in root
    is_dir = []  # boolen
    name = []

    #get data into lists:
    for item in range(0, n_items, 1):
        #tab_pointers += table_pointer
        tab_pointer += struct.unpack('<I', data[item*0x200 : 4 + item*0x200])
        size += struct.unpack('<Q', data[4 + item*0x200 : 12 + item*0x200])
        #create_time += struct.unpack('<d', data[12+item*0x200:20+item*0x200])
        #mod_time += struct.unpack('<d', data[20+item*0x200 : 28+item*0x200])
        #wtf_time += struct.unpack('<d', data[28+item*0x200 : 36+item*0x200])
        #permissions += struct.unpack('<I', data[36+item*0x200 : 40+item*0x200])
        parent += struct.unpack('<I', data[40 + item*0x200 : 44 + item*0x200])
        is_dir += struct.unpack('?', data[0xDC + item*0x200 : 0xDD + item*0x200])
        name.append(data[0xE0 + item*0x200 : 0x200 + item*0x200].strip(b'\x00').\
                                                                 decode('utf-8'))

    return tab_pointer, size, parent, is_dir, name


def items_to_dict(file_tables, size, parent, is_dir, name):
    """ returns sfs tree structure as dict""" 
    tree0 = {}
    ## check if there is any tree structure or list is flat:
    n_parents = np.sort(np.unique(np.fromiter(parent, dtype='uint8')))
    ## if above will cause trouble, change it to uint16 or uint32
    if len(n_parents) > 1:               # there is tree
        for i in range(0, len(is_dir), 1):
            if parent[i] == 0xFFFFFFFF:
                if is_dir[i]:
                    tree0[name[i]] = {}
                else:
                    tree0[name[i]] = {'size': size[i],
                                      'pointers': file_tables[i]}
        if len(n_parents) == 2:  # *.bcf's known till 2015.09
            for i in range(0, len(is_dir), 1):
                if parent[i] != 0xFFFFFFFF:
                    if is_dir[i]:
                        tree0[name[parent[i]]][name[i]] = {}
                    else:
                        tree0[name[parent[i]]][name[i]] = {'size': size[i],
                                            'pointers': file_tables[i]}
        else:  # unknow, but possible stuff
            # if you need it, add the code there
            print('Deep trees are Not Implemented')
    else:                               # list is flat
        for i in range(0, len(is_dir), 1):
            if is_dir[i]:
                tree0[name[i]] = {}
            else:
                tree0[name[i]] = {'size': size[i],
                                  'pointers': file_tables[i]}
    return tree0


# main functions:

def get_sfs_file_tree(filename):
    """ Function scans sfs type file and returns dictionary with file/dir
    structure.

    args: filename

    returns:
        dict with 3 items:
            file_tree,
            max_data_in_chunk
            compression
    """
    with open(filename, 'rb') as fn:
        a = fn.read(8)
        if a == b'AAMVHFSS':  # check: is file SFS container?
            # get the chunk size:
            fn.seek(0x128)
            chunksize = struct.unpack('<I', fn.read(4))[0]
            chunk_usable = chunksize - 32
            fn.seek(0x140)
            tree_address = struct.unpack('<I', fn.read(4))[0]
            fn.seek(0x144)
            #the number of the items / files + directories
            n_of_file_tree_items = struct.unpack('<I', fn.read(4))[0]
            n_of_file_tree_chunks = -((-n_of_file_tree_items*0x200) //
                                      (chunksize-512))
            # get value, how many chunks are in sfs file?
            fn.seek(0x148)
            n_of_chunks = struct.unpack('<I', fn.read(4))[0]
            if n_of_file_tree_chunks == 1:
                fn.seek(chunksize * tree_address + 0x138)  # skip with header
                list_data = fn.read(chunksize - 512)
            else:
                temp_str = io.BytesIO()
                for i in range(0, n_of_file_tree_chunks, 1):
                    # jump to tree/list address:
                    fn.seek(chunksize * tree_address + 0x118)
                    # next tree/list address:
                    tree_address = struct.unpack('<I', fn.read(4))[0]
                    fn.seek(28, 1)
                    temp_str.write(fn.read(chunksize - 512))
                temp_str.seek(0)
                list_data = temp_str.read(n_of_file_tree_items*0x200)
                temp_str.close()
            tab_pointers, size, parent, is_dir, name = bin_tree_to_lists(
                                                 list_data, n_of_file_tree_items)
            #check if data is compressed on the first bits of data:
            fn.seek(chunksize * min(tab_pointers) + 0x138)
            first_data_pointer = struct.unpack('<I', fn.read(4))[0]
            fn.seek(chunksize * first_data_pointer +0x138)
            first_bytes = fn.read(4)
            if first_bytes == b'\x41\x41\x43\x53':  # string AACS
                compressed = True
            else:
                compressed = False
            if compressed:
                fn.seek(0x8C, 1)
                compression_head = fn.read(2)
                #on python3 it could simply be 'if compression_head == 120:'
                byte_one = struct.unpack('BB', compression_head)[0]
                if byte_one == 0x78:
                    compression = 'zlib'
                elif compression_head == b'\x42\x5A':  # string BZ
                    compression = 'bzip2'
                else:
                    compression = 'unknown'
            else:
                compression = 'None'

            tree = items_to_dict(tab_pointers, size, parent, is_dir, name)
            meta_data = {'file_tree': tree,
                         'readable_chunk': chunk_usable,
                         'compression': compression}
            return meta_data
        else:
            raise IOError('file is not SFS container')


def get_the_item(filename, item_dict, chunk_data_size, compression):
    """Extracts requested file/item from the SFS file.

    Arguments:
    filename -- the path to the .sfs file.
    item_dict -- the dictionary of one item/file inside the sfs
    chunk_data_size -- (whole chunk size - size of chunk header)
    compression -- used compression in the sfs

    Returns binary string, decompressed if needed or possible.

    add info:
    SFS compresses files in blocks, the compressed data are not
    continoues. To decompress compressed data, the header before
    every block (this should not be mixed with chunk, and chunk size,
    they are different beasts) have to be read, which holds the size
    of the block, and the offset to the next block. The most
    important and the only needed information is the block size, other
    are irelevant, as the block header allways is of 0x10 size, and the
    general structure is as this: head0, block0, head1, block1. However
    block can start and finish in different chunks, thus at least two
    counters have to be used to track the pointer position in the block,
    and the position of the pointer in the chunks.
    """
    # read the file table and fill the pointer table:
    item_dict = read_pointer_table(filename, item_dict, chunk_data_size)
    #proceed further:
    if compression == 'None':
        data = io.BytesIO()
        with open(filename, 'rb') as fn:
            for i in item_dict['pointers'][:-1]:
                fn.seek(i)
                data.write(fn.read(chunk_data_size))
            fn.seek(item_dict['pointers'][-1])
            data.write(fn.read(item_dict['last_size']))
        return data
    elif compression in ('zlib', 'bzip2'):
        # import required library just once:
        if compression == 'zlib':
            from zlib import decompress as unzip_block
        else:
            from bzip2 import decompress as unzip_block
        data = io.BytesIO()
        block = []
        with open(filename, 'rb') as fn:
            fn.seek(item_dict['pointers'][0] + 12)  # number of compression blocks
            number_of_blocks = struct.unpack('<I', fn.read(4))[0]
            fn.seek(item_dict['pointers'][0] + 0x80)  # the 1st compression block header
            block_remainder = struct.unpack('<I', fn.read(4))[0]
            fn.seek(12, 1)  # go to compressed data begining
            if len(item_dict['pointers']) > 1:
                chunk_remainder = chunk_data_size - 0x90
                chunk_point_iter = iter(item_dict['pointers'][1:])
            else:
                chunk_remainder = item_dict['last_size'] - 0x90
            for j in range(0, number_of_blocks, 1):
                #read the chunks, until until block is filled
                # then read next block header, decompress the block
                # append the result to main data output, reset block to empty
                while block_remainder > chunk_remainder:
                    block.append(fn.read(chunk_remainder))
                    block_remainder -= chunk_remainder
                    chunk_remainder = chunk_data_size
                    fn.seek(next(chunk_point_iter))
                else:
                    block.append(fn.read(block_remainder))
                    if j != number_of_blocks:
                        chunk_remainder = chunk_data_size - block_remainder - 0x10
                        #               =               -         - block header
                        # setting the new block_remainder from header:
                        block_remainder = struct.unpack('<I', fn.read(4))[0]
                        fn.seek(12, 1)  # jump to next block
                data.write(unzip_block(b''.join(block)))
                block = []

        return data
    else:
        raise RuntimeError('file', str(filename),' is compressed by not known method.',
                           'Aborting...')


# the combined function which uses all above functions

def getFileFromTheSFS(sfs_filename, files_internal_sfs_path):
    """Extracts one file with known path in the SFS file.

    Arguments:
    ------------
    sfs_filename -- the file name (can be with path) of .sfs file
      (or any SFS container with different file extentions i.e.*.bcf, *.pan)
    files_internal_sfs_path -- the name of the file inside sfs with full path 
    
    Returns:  io.BytesIO string with binary data.
    ------------
    examples:
    >>> xml_data = getFileFromTheSFS('Sample.bcf', 'EDSDatabase/HeaderData')
    >>> xml_data.seek(0)
    0
    >>> xml_data.read(62)
    b'<?xml version="1.0" encoding="WINDOWS-1252" standalone="yes"?>'
    
    >>> data = getFileFromTheSFS('Sample.bcf', 'EDSDatabase/SpectrumData0')
    .. .. .. .. ..
    >>> particle_list = getFileFromTheSFS('Features_sample.pan', 'Analysis.pab')
    """

    tree = get_sfs_file_tree(sfs_filename)
    chunk_data_size = tree['readable_chunk']
    compression = tree['compression']
    file_tree = tree['file_tree']
    file_path = files_internal_sfs_path.split('/')
    try:
        for i in file_path:
            file_tree = file_tree[i]

        return get_the_item(sfs_filename,
                            file_tree,
                            chunk_data_size,
                            compression)
    except(KeyError):
       print(files_internal_sfs_path, 'could not be found.',
             'Check the internal sfs path')
