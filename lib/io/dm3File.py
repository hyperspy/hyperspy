#!/usr/bin/env python
# -*- coding: latin-1 -*-

# Plugin to read the Gatan Digital Micrograph(TM) file format
#
# Copyright (c) 2010 Stefano Mazzucco.
# All rights reserved.
#
# This program is still at an early stage to be released, so the use of this
# code must be explicitly authorized by its author and cannot be shared for any reason.
#
# Once the program will be mature, it will be released under a GNU GPL license

from __future__ import with_statement #for Python versions < 2.6

from temUtils import overwrite
from temFile import *
from temExceptions import *

readChar = readByte # dm3 uses chars for 1-Byte signed integers

def readInfoArray(f):
    """Read the infoArray from file f.
    Returns the tuple infoArray.
    """
    infoArraySize = readLong(f, 'big')
    infoArray = [readLong(f, 'big') for element in range(infoArraySize)]
    infoArray = tuple(infoArray)
    return infoArray

def _infoArrayDataBytes(iarray):
    """Read the infoArray iarray and return the number of bytes
    of the corresponding TagData.
    """
    if iarray[0] in _complexType:
        if iarray[0] == 18: # it's a string
            nbytes = iarray[1]
        elif iarray[0] == 15:   # it's a struct
            fieldType =  [iarray[i] for i in range(4, len(iarray), 2)]
            fieldBytes = [_dataType[i][1] for i in fieldType]
            nbytes = reduce(lambda x, y: x +y, fieldBytes)
        elif iarray[0] == 20:   # it's an array            
            if iarray[1] != 15:
                nbytes = iarray[-1] * _dataType[iarray[1]][1]
            else:  # it's an array of structs
                subiarray = iarray[1:-1]
                nbytes = _infoArrayDataBytes(subiarray) * iarray[-1]
    elif iarray[0] in _simpleType:
        nbytes = _dataType[iarray[0]][1]
    else:
        raise TypeError, "DataType not recognized"
    return nbytes
 
def readString(f, iarray, endian):
    """Read a string defined by the infoArray iarray from
     file f with a given endianness (byte order).
    endian can be either 'big' or 'little'.

    If it's a tag name, each char is 1-Byte;
    if it's a tag data, each char is 2-Bytes Unicode,
    UTF-16-BE or UTF-16-LE.
    E.g 43 00 for little-endian files (UTF-16-LE)
    i.e. a char followed by an empty Byte.
    E.g 'hello' becomes 'h.e.l.l.o.' in UTF-16-LE
    """    
    if (endian != 'little') and (endian != 'big'):
        print 'File address:', f.tell()
        raise ByteOrderError, endian
    else:
        if iarray[0] != 18:
            print 'File address:', f.tell()
            raise TypeError, "not a DM3 string DataType"
        data = ''
        if endian == 'little':
            s = L_char
        elif endian == 'big':
            s = B_char
        for char in range(iarray[1]):
            data += s.unpack(f.read(1))[0]
        if '\x00' in data:      # it's a Unicode string (TagData)
            uenc = 'utf_16_'+endian[0]+'e'
            data = unicode(data, uenc, 'replace')
        return data

def readStruct(f, iarray, endian):
    """Read a struct, defined by iarray, from file f
    with a given endianness (byte order).
    Returns a list of 2-tuples in the form
    (fieldAddress, fieldValue).
    endian can be either 'big' or 'little'.
    """
    if (endian != 'little') and (endian != 'big'):
        print 'File address:', f.tell()
        raise ByteOrderError, endian
    else:    
        if iarray[0] != 15:
            print 'File address:', f.tell()
            raise TypeError, "not a DM3 struct DataType"
        # nameLength = iarray[1]      # always 0?
        nFields = iarray[2]
        # fieldNameLength = [iarray[i] for i in range(3, len(iarray), 2)] # always 0?
        fieldType =  [iarray[i] for i in range(4, len(iarray), 2)]
        # fieldCType = [_dataType[iarray[i]][2] for i in range(4, len(iarray), 2)]
        fieldAddress = []
        # fieldBytes = [_dataType[i][1] for i in fieldType]
        fieldValue = []
        for dtype in fieldType:
            if dtype in _simpleType:
                fieldAddress.append(f.tell())
                readData = _dataType[dtype][0]
                data = readData(f, endian)
                fieldValue.append(data)
            else:
                raise TypeError, "can't read field type"    
        return zip(fieldAddress, fieldValue)
    
def readArray(f, iarray, endian):
    """Read an array, defined by iarray, from file f
    with a given endianness (byte order).
    endian can be either 'big' or 'little'.
    """
    if (endian != 'little') and (endian != 'big'):
        print 'File address:', f.tell()
        raise ByteOrderError, endian
    else:        
        if iarray[0] != 20:
            print 'File address:', f.tell()
            raise TypeError, "not a DM3 array DataType"
        arraysize = iarray[-1]
        if arraysize == 0:
            return None
        eltype = _dataType[iarray[1]][0] # same for all elements
        if len(iarray) > 3:  # complex type
            subiarray = iarray[1:-1]
            data = [eltype(f, subiarray, endian) for element in range(arraysize)]
        else: # simple type
            data = [eltype(f, endian) for element in range(arraysize)]
            # if _dataType[iarray[1]][2] == 'H': # it's actually a string
            if iarray[1] == 4: # it's actually a string
                data = [chr(i) for i in data]
                data = reduce(lambda x, y: x + y, data)
        return data
    
# _dataType dictionary.
# The first element of the InfoArray in the TagType
# will always be one of _dataType keys.
# the tuple reads: ('read bytes function', 'number of bytes', 'type')
_dataType = {
    2 : (readShort, 2, 'h'),
    3 : (readLong, 4, 'l'),
    4 : (readUShort, 2, 'H'), # dm3 uses ushorts for unicode chars
    5 : (readULong, 4, 'L'),
    6 : (readFloat, 4, 'f'),
    7 : (readDouble, 8, 'd'),
    8 : (readBoolean, 1, 'B'),
    9 : (readChar, 1, 'b'), # dm3 uses chars for 1-Byte signed integers
    10 : (readByte, 1, 'b'),   # 0x0a
    15 : (readStruct, None, 'struct',), # 0x0f
    18 : (readString, None, 'c'), # 0x12
    20 : (readArray, None, 'array'),  # 0x14
    }
                          
_complexType = (10, 15, 18, 20)
_simpleType =  (2, 3, 4, 5, 6, 7, 8, 9, 10)

def parseTagGroup(f, endian='big'):
    """Parse the root TagGroup of the given DM3 file f.
    Returns the tuple (isSorted, isOpen, nTags).
    endian can be either 'big' or 'little'.
    """
    isSorted = readByte(f, endian)
    isOpen = readByte(f, endian)
    nTags = readLong(f, endian)
    return bool(isSorted), bool(isOpen), nTags

def parseTagEntry(f, endian='big'):
    """Parse a tag entry of the given DM3 file f.
    Returns the tuple (tagID, tagNameLength, tagName).
    endian can be either 'big' or 'little'.
    """
    tagID = readByte(f, endian)
    tagNameLength = readShort(f, endian)
    strInfoArray = (18, tagNameLength)
    tagNameArr = readString(f, strInfoArray, endian)
    if len(tagNameArr):
        tagName = reduce(lambda x, y: x + y, tagNameArr)
    else:
        tagName = ''
    return tagID, tagNameLength, tagName

def parseTagType(f):
    """Parse a tag type of the given DM3 file f.
    Returns the tuple infoArray.
    """
    strInfoArray = (18, 4)
    delim = readString(f, strInfoArray, 'big')
    delimiter = reduce(lambda x, y: x + y, delim)
    if delimiter != '%%%%':
        print 'Wrong delimiter: "', delimiter, '".'
        print 'File address:', f.tell()
        raise TypeError, 'not a DM3 TagType'
    else:        
        return readInfoArray(f)
       
def parseTagData(f, iarray, endian, skip=0):
    """Parse the data of the given DM3 file f
    with the given endianness (byte order).
    The infoArray iarray specifies how to read the data.
    Returns the tuple (file address, data).
    The tag data is stored in the platform's byte order:
    'little' endian for Intel, PC; 'big' endian for Mac, Motorola.
    If skip != 0 the data is actually skipped.
    """
    faddress = f.tell()
    # nbytes = _infoArrayDataBytes(iarray)
    if not skip:
        readData = _dataType[iarray[0]][0]
        if iarray[0] in _complexType:
            data = readData(f, iarray, endian)
        elif iarray[0] in _simpleType:
            data = readData(f, endian)
        else:
            raise TypeError, "DataType not recognized"
    else:
        data = '__skipped__'        
        # print 'Skipping', nbytes, 'Bytes.'
        nbytes = _infoArrayDataBytes(iarray)
        f.seek(nbytes, 1)
    # return faddress, nbytes, data
    return faddress, data

def parseImageData(f, iarray):
    """Returns a tuple with the file offset and the number
    of bytes corresponding to the image data:
    (offset, bytes)
    """
    faddress = f.tell()
    nbytes = _infoArrayDataBytes(iarray)
    f.seek(nbytes, 1)        
    return faddress, nbytes

def parseHeader(f, dataDict, endian='big', debug=0):
    """Parse the header (first 12 Bytes) of the given DM3 file f.
    The relevant information is saved in the dictionary dataDict.
    Optionally, a debug level !=0 may be specified.
    endian can be either 'little' or 'big'.
    Returns the boolean isLittleEndian (byte order) of the DM3 file.
    """
    iarray = (3, ) # long
    dmVersion = parseTagData(f, iarray, endian)
    if dmVersion[1] != 3:
        print 'File address:', dmVersion[1]
        raise SyntaxError, "not a DM3 file"
    dataDict['DM3.Version'] = dmVersion

    filesizeB = parseTagData(f, iarray, endian)
    filesizeB = list(filesizeB)
    filesizeB[1] = filesizeB[1] + 16
    filesizeB = tuple(filesizeB)
    dataDict['DM3.FileSize'] = filesizeB

    isLittleEndian = parseTagData(f, iarray, endian)
    dataDict['DM3.isLittleEndian'] = isLittleEndian

    if debug > 0:
        filesizeKB = filesizeB[1] / 2.**10
        # filesizeMB = filesizeB[3] / 2.**20
        print 'DM version:', dmVersion[1]
        print 'size %i B (%.2f KB)' % (filesizeB[1], filesizeKB)
        # print 'size: {0} B ({1:.2f} KB)'.format(filesizeB[1] , filesizeKB)
        print 'Is file Little endian?', bool(isLittleEndian[1])
    return bool(isLittleEndian[1])

def crawlDM3(f, dataDict, endian, ntags, grpname='root', skip=0, debug=0):
    """Recursively scan the ntags TagEntrys in DM3 file f
    with a given endianness (byte order) looking for
    TagTypes (data) or TagGroups (groups).
    endian can be either 'little' or 'big'.
    The dictionary dataDict is filled with tags and data.
    If skip != 0 the data reading is actually skipped.
    """
    for tag in range(ntags):
        if debug > 3 and debug < 10:
            print 'Crawling at address:', f.tell()            

        tagID, tagNameLength, tagName = parseTagEntry(f)

        if debug > 5 and debug < 10:
            print 'Tag name:', tagName
            print 'Tag ID:', tagID

        if tagID == 21: # it's a TagType (DATA)

            if len(tagName) == 0:
                tagName = '__data__'

            if debug > 3 and debug < 10:
                print 'Crawling at address:', f.tell()

            infoarray = parseTagType(f)

            if debug > 5 and debug < 10:
                print 'Infoarray:', infoarray
            
            dataKey = grpname + '.' + tagName

            # Don't overwrite duplicate keys, instead, rename them!
            i = 0
            if dataDict.has_key(dataKey):
                if debug > 5 and debug < 10:
                    print 'key exists... renaming'
                oldval = dataDict.pop(dataKey)
                newKey = dataKey + '.' + str(i)
                dataDict[newKey] = oldval
                dataKey = dataKey + '.' + str(i+1)
            i += 1
            
            if re.search('\.Data$', dataKey):
                # don't read the data now
                # dataDict[dataKey] = parseImageData(f, infoarray, endian, skip=1)
                dataDict[dataKey] = parseImageData(f, infoarray)
            else:
                dataDict[dataKey] = parseTagData(f, infoarray, endian, skip)

            if debug > 10:  # start an interactive session
                try:
                    if not raw_input('(debug) Press "Enter" to continue\n'):
                        print '######################################'
                        print 'TAG:\n', dataKey, '\n'
                        print 'ADDRESS:\n', dataDict[dataKey][0], '\n'
                        try:
                            if len(dataDict[dataKey][1]) > 10:
                                print 'VALUE:\n', dataDict[dataKey][1][:10], '[...]\n'
                            else:
                                print 'VALUE:\n', dataDict[dataKey][1], '\n'
                        except:
                            print 'VALUE:\n', dataDict[dataKey][1], '\n'
                        print '######################################\n'
                except KeyboardInterrupt:
                    print '\n\n###################'
                    print 'Operation canceled.'
                    print '###################\n\n'
                    raise       # exit violently

        elif tagID == 20: # it's a TagGroup (GROUP)
            if len(tagName) == 0:
                tagName = '__group__'
            if debug > 3 and debug < 10:
                print 'Crawling at address:', f.tell()
            grpname += '.' + tagName
            ntags = parseTagGroup(f)[2]
            crawlDM3(f, dataDict, endian, ntags, grpname, skip, debug) # recursion
        else:
            print 'File address:', f.tell()
            raise TypeError, "DM3 TagID not recognized"

def openDM3(fname, skip=0, debug=0, log=''):
    """Open a DM3 file given its name and return the dictionary dataDict
    containint the parsed information.
    If skip != 0 the data is actually skipped.
    Optionally, a debug value debug > 0 may be given.
    If log='filename' is specified, the keys, file address and
    (part of) the data parsed in dataDict are written in the log file.

    NOTE:
    All fields, except the TagData are stored using the Big-endian
    byte order. The TagData are stored in the platform's
    byte order (e.g. 'big' for Mac, 'little' for PC).
    """
    with open(fname, 'r+b') as dm3file:
        fmap = mmap.mmap(dm3file.fileno(), 0, access=mmap.ACCESS_READ)
        dataDict = {}
        if parseHeader(fmap, dataDict, debug=debug):
            fendian = 'little'
        else:
            fendian = 'big'
        rntags = parseTagGroup(fmap)[2]
        if debug > 3:
            print 'Total tags in root group:', rntags
        rname = 'DM3'
        crawlDM3(fmap, dataDict, fendian, rntags, rname, skip, debug)
#         if platform.system() in ('Linux', 'Unix'):
#             try:
#                 fmap.flush()
#             except:
#                 print "Error. Could not write to file", fname
#         if platform.system() in ('Windows', 'Microsoft'):
#             if fmap.flush() == 0:
#                 print "Error. Could not write to file", fname
        fmap.close()

        if log:
            exists = overwrite(log)
            if exists:
                with open(log, 'w') as logfile:
                    for key in dataDict:
                        try:
                            line = '%s    %s    %s' % (key, dataDict[key][0], dataDict[key][1][:10])
                            # line = '{0}    {1}    {2}'.format(key, dataDict[key][0], dataDict[key][1][:10])
                        except:
                            try:
                                line = '%s    %s    %s' % (key, dataDict[key][0], dataDict[key][1])
                                # line = '{0}    {1}    {2}'.format(key, dataDict[key][0], dataDict[key][1])
                            except:
                                line = '%s    %s    %s' % (key, dataDict[key][0], dataDict[key][1])
                        print >> logfile, line, '\n'
                print log, 'saved in current directory'

        return dataDict
