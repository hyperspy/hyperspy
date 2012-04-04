#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-

# Filename: dm3reader.py

################################################################################
## Python script for parsing GATAN DM3 (DigitalMicrograph) files
## and extracting various metadata
## --
## warning: *tested on single-image files only*
## --
## based on the DM3_Reader plug-in (v 1.3.4) for ImageJ by Greg Jefferis <jefferis@stanford.edu>
## http://rsb.info.nih.gov/ij/plugins/DM3_Reader.html
## --
## Python adaptation: Pierre-Ivan Raynal <raynal@med.univ-tours.fr>
## http://microscopies.med.univ-tours.fr/
################################################################################

import sys, struct
from hyperspy.misc.utils import ensure_unicode

version='0.72'

## constants for encoded data types ##

SHORT = 2
LONG = 3
USHORT = 4
ULONG = 5
FLOAT = 6
DOUBLE = 7
BOOLEAN = 8
CHAR = 9
OCTET = 10
STRUCT = 15
STRING = 18
ARRAY = 20

## END constants ##


debugLevel = 0   # 0=none, 1-3=basic, 4-5=simple, 6-10 verbose

#chosenImage = 1
#
#IMGLIST = "root.ImageList."
#OBJLIST = "root.DocumentObjectList."


### initialize variables ###
f=''
# track currently read group
MAXDEPTH = 64
curGroupLevel = -1
curGroupAtLevelX = [ 0 for x in xrange(MAXDEPTH) ]
curGroupNameAtLevelX = [ "" for x in xrange(MAXDEPTH) ]
# track current tag
curTagAtLevelX = [ "" for x in xrange(MAXDEPTH) ]
curTagName = ""
storedTags = []
tagHash = {}

### END init. variables ###


### sub-routines ###

## reading n bytes functions
def readLong( file ):
	"""Read 4 bytes as integer in file"""
	read_bytes = file.read(4)
	return struct.unpack('>l', read_bytes)[0]

def readShort( file ):
	"""Read 2 bytes as integer in file"""
	read_bytes = file.read(2)
	return struct.unpack('>h', read_bytes)[0]

def readByte( file ):
	"""Read 1 byte as integer in file"""
	read_bytes = file.read(1)
	return struct.unpack('>b', read_bytes)[0]

def readChar( file ):
	"""Read 1 byte as char in file"""
	read_bytes = file.read(1)
	return struct.unpack('c', read_bytes)[0]

def readString( file, len=1 ):
	"""Read len bytes as a string in file"""
	read_bytes = file.read(len)
	str_fmt = '>'+str(len)+'s'
	return struct.unpack( str_fmt, read_bytes )[0]

def readLEShort( file ):
	"""Read 2 bytes as *little endian* integer in file"""
	read_bytes = file.read(2)
	return struct.unpack('<h', read_bytes)[0]

def readLELong( file ):
	"""Read 4 bytes as *little endian* integer in file"""
	read_bytes = file.read(4)
	return struct.unpack('<l', read_bytes)[0]

def readLEUShort( file ):
	"""Read 2 bytes as *little endian* unsigned integer in file"""
	read_bytes = file.read(2)
	return struct.unpack('<H', read_bytes)[0]

def readLEULong( file ):
	"""Read 4 bytes as *little endian* unsigned integer in file"""
	read_bytes = file.read(4)
	return struct.unpack('<L', read_bytes)[0]

def readLEFloat( file ):
	"""Read 4 bytes as *little endian* float in file"""
	read_bytes = file.read(4)
	return struct.unpack('<f', read_bytes)[0]

def readLEDouble( file ):
	"""Read 8 bytes as *little endian* double in file"""
	read_bytes = file.read(8)
	return struct.unpack('<d', read_bytes)[0]


## utility functions
def makeGroupString():
	global curGroupLevel, curGroupAtLevelX

	tString = curGroupAtLevelX[0]
	for i in xrange( 1, curGroupLevel+1 ):
		tString += "."+curGroupAtLevelX[i]
	return tString

def makeGroupNameString():
	global curGroupLevel, curGroupNameAtLevelX

	tString = curGroupNameAtLevelX[0]
	for i in xrange( 1, curGroupLevel+1 ):
		tString += "." + str( curGroupNameAtLevelX[i] )
		
	return tString


def readTagGroup():
	global curGroupLevel, curGroupAtLevelX, curTagAtLevelX
	
	# go down a level
	curGroupLevel += 1
	# increment group counter
	curGroupAtLevelX[curGroupLevel] += 1
	# set number of current tag to -1 --- readTagEntry() pre-increments => first gets 0
	curTagAtLevelX[curGroupLevel] = -1

	if ( debugLevel > 5):
		print "rTG: Current Group Level:", curGroupLevel

	# is the group sorted?
	sorted = readByte(f)
	if ( sorted == 1 ):
		isSorted = True
	else:
		isSorted = False

	# is the group open?
	open = readByte(f)
	if ( open == 1 ):
		isOpen = True
	else:
		isOpen = False

	# number of Tags
	nTags = readLong(f)
	
	if ( debugLevel > 5):
		print "rTG: Iterating over the", nTags, "tag entries in this group"

	# read Tags
	for i in xrange( nTags ):
		readTagEntry()
	
	# go back up one level as reading group is finished
	curGroupLevel += -1
	
	return 1


def	readTagEntry():
	global curGroupLevel, curGroupAtLevelX, curTagAtLevelX, curTagName

	# is data or a new group?
	data = readByte(f)
	if ( data == 21 ):
		isData = True
	else:
		isData = False

	curTagAtLevelX[curGroupLevel] += 1

	# get tag label if exists
	lenTagLabel = readShort(f)
	
	if ( lenTagLabel != 0 ):
		tagLabel = readString(f, lenTagLabel)
	else:
		tagLabel = str( curTagAtLevelX[curGroupLevel] )
	
	if ( debugLevel > 5):
		print str(curGroupLevel)+"|"+makeGroupString()+": Tag label = "+tagLabel
	elif ( debugLevel > 0 ):
		print str(curGroupLevel)+": Tag label = "+tagLabel

	if isData:
		# give it a name
		curTagName = makeGroupNameString()+"."+tagLabel
		# read it
		readTagType()
	else:
		# it is a tag group
		curGroupNameAtLevelX[curGroupLevel+1] = tagLabel
		readTagGroup()  # increments curGroupLevel
	
	return 1


def readTagType():
	delim = readString(f, 4)
	if ( delim != "%%%%" ):
		print hex( f.tell() )+": Tag Type delimiter not %%%%"
		sys.exit()

	nInTag = readLong(f)

	readAnyData()
	
	return 1


def encodedTypeSize(eT):
	# returns the size in bytes of the data type

	width=-1; 	# returns -1 for unrecognised types
	
	if eT == 0:
		width = 0
	elif ( (eT == BOOLEAN) or (eT == CHAR) or (eT == OCTET) ):
		width = 1
	elif ( (eT == SHORT) or (eT == USHORT) ):
		width = 2
	elif ( (eT == LONG) or (eT == ULONG) or (eT == FLOAT) ):
		width = 4
	elif (eT == DOUBLE):			
		width = 8
	
	return width


def readAnyData():
	## higher level function dispatching to handling data types to other functions
		
	# get Type category (short, long, array...)
	encodedType = readLong(f)
	# calc size of encodedType
	etSize = encodedTypeSize(encodedType)

	if ( debugLevel > 5):
		print "rAnD, " + hex( f.tell() ) + ": Tag Type = " + str(encodedType) +  ", Tag Size = " + str(etSize)
	
	if ( etSize > 0 ):
		storeTag( curTagName, readNativeData(encodedType, etSize) )
	elif ( encodedType == STRING ):
		stringSize = readLong(f)
		readStringData(stringSize)
	elif ( encodedType == STRUCT ):
		# does not store tags yet
		structTypes = readStructTypes()
		readStructData(structTypes)
	elif ( encodedType == ARRAY ):
		# does not store tags yet
		# indicates size of skipped data blocks
		arrayTypes = readArrayTypes()
		readArrayData(arrayTypes)
	else:
		print "rAnD, " + hex( f.tell() ) + ": Can't understand encoded type"
		sys.exit()
		
	return 1
	

def readNativeData( encodedType, etSize ):
	# reads ordinary data types

	if ( encodedType == SHORT ):
		val = readLEShort(f)
	elif ( encodedType == LONG ):
		val = readLELong(f)
	elif ( encodedType == USHORT ):
		val = readLEUShort(f)
	elif ( encodedType == ULONG ):
		val = readLEULong(f)
	elif ( encodedType == FLOAT ):
		val = readLEFloat(f)
	elif ( encodedType == DOUBLE ):
		val = readLEDouble(f)
	elif ( encodedType == BOOLEAN ):
		bool = readByte(f)
		if bool == 0:
			val = False
		else:
			val = True 
	elif ( encodedType == CHAR ):
		val = readChar(f)
	elif ( encodedType == OCTET):
		val = readChar(f)   # difference with char???
	else:
		print "rND, " + hex( f.tell() ) + ": Unknown data type " + str(encodedType)
		sys.exit()
	
	if ( debugLevel > 3 ):
		print "rND, " + hex( f.tell() ) + ": " + str(val)
	elif ( debugLevel > 0 ):
		print val
	
	return val


def readStringData(stringSize):
	# reads string data
 	if ( stringSize <= 0 ):
		rString = ""
	else:
		if ( debugLevel > 3 ):
			print "rSD @ " + str( f.tell() ) + "/" + hex( f.tell() ) +" :",
			
		## !!! *Unicode* string... convert to latin-1 string
		rString = readString(f, stringSize)

		if ( debugLevel > 3 ):
			print rString + "   <"  + repr( rString ) + ">"

	if ( debugLevel > 0 ):
		print "StringVal:", rString
	
	storeTag( curTagName, rString )

	return rString
	
	
def readArrayTypes():
	# determines the data types in an array data type
	arrayType = readLong(f)
	
	itemTypes=[]
	if ( arrayType == STRUCT ):
		itemTypes = readStructTypes()
	elif ( arrayType == ARRAY ):
		itemTypes = readArrayTypes()
	else:
		itemTypes.append( arrayType )
	
	return itemTypes


def readArrayData(arrayTypes):
	# reads array data
	
	arraySize = readLong(f)
	
	if ( debugLevel > 3 ):
		print "rArD, " + hex( f.tell() ) + ": Reading array of size = " + str(arraySize)
	
	itemSize = 0
	encodedType = 0
	
	for i in xrange( len(arrayTypes) ):
		encodedType = int( arrayTypes[i] )
		etSize = encodedTypeSize(encodedType)
		itemSize += etSize
		if ( debugLevel > 5 ):
			print "rArD: Tag Type = " + str(encodedType) + ", Tag Size = " + str(etSize)
		##! readNativeData( encodedType, etSize ) !##
	
	if ( debugLevel > 5 ):
		print "rArD: Array Item Size = " + str(itemSize)

	bufSize = arraySize * itemSize
	
	if ( (not curTagName.endswith("ImageData.Data"))
			and  ( len(arrayTypes) == 1 )
			and  ( encodedType == USHORT )
			and  ( arraySize < 256 ) ):
		# treat as string
		val = readStringData( bufSize )
	else:
		# treat as binary data
		# - store data size and offset as tags 
		storeTag( curTagName + ".Size", bufSize )
		storeTag( curTagName + ".Offset", f.tell() )
		# - skip data w/o reading
		f.seek( f.tell() + bufSize )
	
	return 1


def readStructTypes():
	# analyses data types in a struct
	
	if ( debugLevel > 3 ):
		print "Reading Struct Types at Pos = " + hex( f.tell() )

	structNameLength = readLong(f)
	nFields = readLong(f)

	if ( debugLevel > 5 ):
		print "nFields = ", nFields

	if ( nFields > 100 ):
		print hex( f.tell() ), "Too many fields"
		sys.exit()
		
	fieldTypes = []	
	nameLength = 0
	for i in xrange( nFields ):
		nameLength = readLong(f)
		if ( debugLevel > 9 ):
			print i + "th namelength = " + nameLength
		fieldType = readLong(f)
		fieldTypes.append( fieldType )

	return fieldTypes

	
def readStructData( structTypes ):
	# reads struct data based on type info in structType
	for i in xrange( len(structTypes) ):
		encodedType = structTypes[i]
		etSize = encodedTypeSize(encodedType)

		if ( debugLevel > 5 ):
			print "Tag Type = " + str(encodedType) + ", Tag Size = " + str(etSize)

		# get data
		readNativeData(encodedType, etSize)
	
	return 1
	
	
def storeTag( tagName, tagValue ):
	global storedTags, tagHash
	storedTags.append( str(tagName) + " = " + str(tagValue) )
	tagHash[ ensure_unicode(tagName, 'utf8', 'latin-1')] = ensure_unicode(tagValue, 'utf16')
	
	
### END sub-routines ###



### parse DM3 file ###
def parseDM3( filename, dump=False ):
	"""Function parses DM3 file and returns dict with extracted Tags.
	Dumps Tags in a txt file if 'dump' set to 'True'."""

	global f


	f = open( filename, 'rb' )
	isDM3 = True
	## read header (first 3 4-byte int)
	# get version
	fileVersion = readLong(f)
	if ( fileVersion != 3 ):
		isDM3 = False
	# get indicated file size
	FileSize = readLong(f)
	# get byte-ordering
	lE = readLong(f)
	if ( lE == 1 ):
		littleEndian = True
	else :
		littleEndian = False
		isDM3 = False
	# check file header, raise Exception if not DM3
	if (not (isDM3 and littleEndian) ):
		raise NameError("Is_Not_a_DM3_File")
	
	if ( debugLevel > 5):
		print "Header info.:"
		print "File version:", version
		print "lE:", lE
		print "File size:", FileSize

	# print '%s appears to be a DM3 file' % filename,

	# set name of root group (contains all data)...
	curGroupNameAtLevelX[0] = "root"
	# ... then read it
	global storedTags, tagHash
	storedTags = []
	tagHash = {}
	readTagGroup()
	
	f.close()
	
	# print "--", len(storedTags), "Tags read"
	
	# dump Tags in txt file if requested
	if dump:
		dump_file = filename + ".dump.txt"
		try:
			log = open( dump_file, 'w' )
		except:
			print "Error -- could not access output file."
			sys.exit()
		for tag in storedTags:
			log.write( tag + "\n" )
		log.close

	# return Tag list
	return tagHash
	

def getDM3FileInfo( dm3_file, makePGMtn=False, tn_file='dm3tn_temp.pgm' ):
	"""Extracts useful experiment info from DM3 file and 
	exports thumbnail to a PGM file if 'makePGMtn' set to 'True'."""
	
	# define useful information
	info_keys = {
		'descrip': 'root.ImageList.1.Description',
		'acq_date': 'root.ImageList.1.ImageTags.DataBar.Acquisition Date',
		'acq_time': 'root.ImageList.1.ImageTags.DataBar.Acquisition Time',
		'name': 'root.ImageList.1.ImageTags.Microscope Info.Name',
		'micro': 'root.ImageList.1.ImageTags.Microscope Info.Microscope',
		'hv': 'root.ImageList.1.ImageTags.Microscope Info.Voltage',
		'mag': 'root.ImageList.1.ImageTags.Microscope Info.Indicated Magnification',
		'mode': 'root.ImageList.1.ImageTags.Microscope Info.Operation Mode',
		'operator': 'root.ImageList.1.ImageTags.Microscope Info.Operator',
		'specimen': 'root.ImageList.1.ImageTags.Microscope Info.Specimen',
#		'image_notes': 'root.DocumentObjectList.10.Text' # = Image Notes 		
		}
		
	# parse DM3 file
	tags = parseDM3( dm3_file, dump=False )

	# if OK, extract Tags [and thumbnail]
	if tags:
		if makePGMtn:
			# get thumbnail
			tn_size = int( tags[ 'root.ImageList.0.ImageData.Data.Size' ] )
			tn_offset = int( tags[ 'root.ImageList.0.ImageData.Data.Offset' ] )
			tn_width = int( tags[ 'root.ImageList.0.ImageData.Dimensions.0' ] )
			tn_height = int( tags[ 'root.ImageList.0.ImageData.Dimensions.1' ] )
			
			if ( debugLevel > 0 ):
				print "tn data in", dm3_file, "starts at", hex( tn_offset )
				print "tn dimension:", tn_width, "x", tn_height
			
			if ( (tn_width*tn_height*4) != tn_size ):
				print "Error: cannot extract thumbnail from", dm3_file
				sys.exit()
				
			# access DM3 file
			try:
				dm3 = open( dm3_file, 'rb' )
			except:
				print "Error accessing DM3 file"
				sys.exit()
		
			# open PGM file
			try:
				pgm_file = open( tn_file, 'w' )
			except:
				print 'Error creating PGM output file!'
				sys.exit()
			
			# build plain PGM file header
			pgm_file.write( "P2 "+str(tn_width)+" "+str(tn_height)+" 255\n" )
			
			# read tn image data
			dm3.seek( tn_offset )
			for i in xrange( tn_height ):
				for ii in xrange( tn_width ):
					data_bytes = dm3.read(4)
					pgm_data = struct.unpack('<L', data_bytes)[0]
					pgm_data = int( pgm_data/65536 )
					pgm_file.write( str(pgm_data) + ' ' )
				pgm_file.write("\n")
			
			pgm_file.close()
			dm3.close()

		# store experiment information
		infoHash = {}
		for key, tag in info_keys.items():
			if tags.has_key( tag ):
				infoHash[ key ] = tags[ tag ]
	
		return infoHash
	# else, return false value
	else:
		return 0

### END dm3reader.py
