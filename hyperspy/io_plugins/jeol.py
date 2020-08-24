# -*- coding: utf-8 -*-
# Copyright 2007-2016 The HyperSpy developers
#
# This file is part of  HyperSpy.
#
#  HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  HyperSpy.  If not, see <http://www.gnu.org/licenses/>.

import os
from datetime import datetime, timedelta
import numpy as np
import numba

# Plugin characteristics
# ----------------------
format_name = 'JEOL'
description = 'Read JEOL files output by Analysis Station software'
full_support = False	# Whether all the Hyperspy features are supported
# Recognised file extension
file_extensions = ('ASW', 'asw', 'img', 'map', 'pts', 'eds')
default_extension = 0	# Index of the extension that will be used by default
# Reading capabilities
reads_images = True
reads_spectrum = True
reads_spectrum_image = True
# Writing capabilities
writes = False



jTYPE = { 1: 'B',  2: 'H',  3: 'i',  4: 'f',  5: 'd',
					6: 'B',  7: 'H',  8: 'i',  9: 'f', 10: 'd',
				 11: '?', 12: 'c', 13: 'c', 14: 'H', 20: 'c',
			65553: '?', 65552: '?'}



def file_reader(filename, *args, **kwds):
	dictionary = []
	file_ext = os.path.splitext(filename)[-1][1:].lower()
	if file_ext == 'asw':
		fd = open(filename, 'br')
		file_magic = np.fromfile(fd, '<I', 1)[0]
		if (file_magic == 0):
			fd.seek(12)
			filetree = parsejeol(fd)
			filepath = filename[:filename.rfind('/')+1]
			if 'SampleInfo' in filetree.keys():
				for i in sorted(filetree['SampleInfo'].keys(), key=float):
					if 'ViewInfo' in filetree['SampleInfo'][i].keys():
						for j in sorted(filetree['SampleInfo'][i]['ViewInfo'].keys(), key=float):
							if 'ViewData' in filetree['SampleInfo'][i]['ViewInfo'][j].keys():
								scale = filetree['SampleInfo'][i]['ViewInfo'][j]['PositionMM']*1000
								for k in sorted(filetree['SampleInfo'][i]['ViewInfo'][j]['ViewData'].keys(), key=float): 
									subfile = filetree['SampleInfo'][i]['ViewInfo'][j]['ViewData'][k]['Filename'].replace('\\','/')
									sub_ext = os.path.splitext(subfile)[-1][1:]
									if sub_ext == 'img' or sub_ext == 'map':
										dictionary.append(read_img(filepath+subfile, scale))
									elif sub_ext == 'pts':
										dictionary.append(read_pts(filepath+subfile, scale))
									elif sub_ext == 'eds':
										dictionary.append(read_eds(filepath+subfile))
									else:
										print('Unknow extension')
		else:
			print('Not a valid JEOL asw format')
	elif file_ext == 'img' or file_ext == 'map':
		dictionary.append(read_img(filename))
	elif file_ext == 'pts':
		dictionary.append(read_pts(filename))
	elif file_ext == 'eds':
		dictionary.append(read_eds(filename))

	return dictionary 



def read_img(filename, scale=None):
	fd = open(filename, 'br')
	file_magic = np.fromfile(fd, '<I', 1)[0]
	if (file_magic == 52):
		fileformat = fd.read(32).rstrip(b'\x00').decode("utf-8")
		head_pos, head_len, data_pos = np.fromfile(fd, '<I', 3)
		fd.seek(head_pos+12)
		header_short = parsejeol(fd)
		fd.seek(data_pos+12)
		header_long = parsejeol(fd)
		width, height = header_long['Image']['Size']
		header_long['Image']['Bits'].resize((height, width))
		data = header_long['Image']['Bits']
		if scale is not None: 
			xscale = -scale[2]/width
			yscale = scale[3]/height
			units = 'µm'
		else:
			xscale = 1
			yscale = 1
			units = 'px'

		axes = [{	'name': 'height',
							'size': height,
							'offset': 0,
 							'scale': yscale,
							'units': units},
						{	'name': 'width',
							'size': width,
							'offset': 0,
							'scale': xscale,
							'units': units},]

		datefile = datetime(1899,12,30)+timedelta(days= header_long['Image']['Created'])
		hv = header_long['Instrument']['AccV']
		if hv <= 30.0:
			mode = 'SEM'
		else:
			mode = 'TEM'

		metadata = {	'Acquisition_instrument': {	mode: {	'beam_energy': hv,
																											'magnification': header_long['Instrument']['Mag'],},},
									'General': {	'original_filename': os.path.basename(filename),
																'date': datefile.date().isoformat(),
																'time': datefile.time().isoformat(),
																'title': header_long['Image']['Title'],},
									"Signal": {	'record_by': 'image',},
								}

		dictionary = {'data': data,
									'axes': axes,
									'metadata': metadata,
									'original_metadata': header_long, }
	else:
		print('Not a valid JEOL img format')

	fd.close()

	return dictionary



def read_pts(filename, scale=None):
	fd = open(filename, 'br')
	file_magic = np.fromfile(fd, '<I', 1)[0]
	if (file_magic == 304):
		fileformat = fd.read(8).rstrip(b'\x00').decode("utf-8")
		a, b, head_pos, head_len, data_pos, data_len = np.fromfile(fd, '<I', 6)
		groupename = fd.read(128).rstrip(b'\x00').decode("utf-8")
		memo = fd.read(132).rstrip(b'\x00').decode("utf-8")
		datefile = datetime(1899,12,30) + timedelta(days= np.fromfile(fd, 'd', 1)[0])
		fd.seek(head_pos+12)
		header = parsejeol(fd)
		width, height = header['PTTD Data']['AnalyzableMap MeasData']['Meas Cond']['Pixels'].split('x')
		width = int(width)
		height = int(height)
		energy = 1536 #2146, 4096 total
		fd.seek(data_pos)
		data = readcube(fd, width, height, energy)
		if scale is not None: 
			xscale = -scale[2]/width
			yscale = scale[3]/height
			units = 'µm'
		else:
			xscale = 1
			yscale = 1
			units = 'px'

		ch_mod = header['PTTD Data']['AnalyzableMap MeasData']['Meas Cond']['Tpl']
		ch_res = header['PTTD Data']['AnalyzableMap MeasData']['Doc']['CoefA']
		ch_ini = header['PTTD Data']['AnalyzableMap MeasData']['Doc']['CoefB']
		ch_pos = header['PTTD Param']['Params']['PARAMPAGE1_EDXRF']['Tpl'][ch_mod]['DigZ']

		axes = [{	'name': 'height',
							'size': height,
							'offset': 0,
 							'scale': yscale,
							'units': units},
						{	'name': 'width',
							'size': width,
							'offset': 0,
							'scale': xscale,
							'units': units},
						{	'name': 'Energy',
							'size': energy,
							'offset': ch_ini-ch_res*ch_pos,
							'scale': ch_res,
							'units': 'keV'}]

		hv = header['PTTD Data']['AnalyzableMap MeasData']['MeasCond']['AccKV']
		if hv <= 30.0:
			mode = 'SEM'
		else:
			mode = 'TEM'

		metadata = {	'Acquisition_instrument': {	mode: {	'beam_energy': hv,
																											'magnification':  header['PTTD Data']['AnalyzableMap MeasData']['MeasCond']['Mag'],
																											'Detector': {	'EDS': {	'azimuth_angle': header['PTTD Param']['Params']['PARAMPAGE0_SEM']['DirAng'],
																																							'detector_type': header['PTTD Param']['Params']['PARAMPAGE0_SEM']['DetT'],
																																							'elevation_angle': header['PTTD Param']['Params']['PARAMPAGE0_SEM']['ElevAng'],
																																							'energy_resolution_MnKa': header['PTTD Param']['Params']['PARAMPAGE0_SEM']['MnKaRES'],
																																							'real_time': header['PTTD Data']['AnalyzableMap MeasData']['Doc']['RealTime']},},},},
									'General': {	'original_filename': os.path.basename(filename),
																'date': datefile.date().isoformat(),
																'time': datefile.time().isoformat(),
																'title': 'EDX',},
									"Signal": {	'record_by': 'spectrum',
															'quantity': 'X-rays (Counts)',
															'signal_type': 'EDS_'+mode},
								}

		dictionary = {'data': data,
									'axes': axes,
									'metadata': metadata,
									'original_metadata': header, }
	else:
		print('Not a valid JEOL pts format')

	fd.close()

	return dictionary



def parsejeol(fd):
	final_dict = {}
	tmp_list = []
	tmp_dict = final_dict

	mark = 1
	while (abs(mark)==1):
		mark = np.fromfile(fd, 'b', 1)[0]
		if (mark==1):
			str_len = np.fromfile(fd, '<i', 1)[0]
			kwrd = fd.read(str_len).rstrip(b'\x00')

			if (kwrd == b'\xce\xdf\xb0\xc4'): # correct variable name which might be 'Port'
				kwrd = 'Port'
			elif (kwrd[-1] == 222): # remove undecodable byte at the end of first ScanSize variable
				kwrd = kwrd[:-1].decode("utf-8")
			else:
				kwrd = kwrd.decode("utf-8")
			val_type, val_len = np.fromfile(fd, '<i', 2)
			tmp_list.append(kwrd)
			if (val_type==0):
				tmp_dict[kwrd] = {}
			else:
				c_type = jTYPE[val_type]
				arr_len = val_len // np.dtype(c_type).itemsize
				if (c_type == 'c'):
					value = fd.read(val_len).rstrip(b'\x00')
					value = value.decode("utf-8")
					value = os.path.normpath(value.replace('\\','/')).split('\x00')
				else:
					value = np.fromfile(fd, c_type, arr_len)
				if (len(value)==1):
					value = value[0]
				if (kwrd[-5:-1]=='PAGE'):
					kwrd = kwrd+'_'+value
					tmp_dict[kwrd] = {}
					tmp_list[-1] = kwrd
				elif (kwrd == 'CountRate' or kwrd == 'DeadTime'):
					tmp_dict[kwrd] = {}
					tmp_dict[kwrd]['value'] = value
				elif (kwrd == 'Limits'):
					print('skip')
				elif (val_type == 14):
					tmp_dict[kwrd] = {}
					tmp_dict[kwrd]['index'] = value
				else:
					tmp_dict[kwrd] = value
			if (kwrd=='Limits'):
				print('skip2')
			else:
				tmp_dict = tmp_dict[kwrd]
		else:
			if (len(tmp_list) != 0):
				del tmp_list[-1]
				tmp_dict = final_dict
				for k in tmp_list:
					tmp_dict = tmp_dict[k]
			else:
				mark = 0

	return final_dict


@numba.jit
def readcube(fd, w, h, e):
	rawdata = np.fromfile(fd, dtype='u2')
	hypermap = np.zeros([h,w,e])
	for value in rawdata:
		if (value>=32768) and (value<36864):
			y = int((value-32768)/8-1)
		elif (value>=36864) and (value<40960):
			x = int((value-36864)/8-1)
		elif (value>=45056) and (value<45056+e): #49152 max but memory prob
			z = int(value-45056)
			hypermap[x,y,z] = hypermap[x,y,z]+1
	return hypermap



def read_eds(filename):
	fd = open(filename, 'br')
	file_magic = np.fromfile(fd, '<I', 1)[0]
	fd.seek(0xA)
	filedate = np.fromfile(fd, '<d', 1)
	fd.seek(0x62)
#	filedate = np.fromfile(fd, '<d', 1)
	fd.seek(0x102)
	live_time, real_time = np.fromfile(fd, '<d', 2)
	fd.seek(0x16A)
	x_res, x_offset = np.fromfile(fd, '<d', 2)
	fd.seek(454)
	array_size = np.fromfile(fd, '<i', 1)[0]
	data = np.fromfile(fd, '<i', array_size)
	return data, x_res, x_offset
