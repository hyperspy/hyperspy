# -*- coding: utf-8 -*-
# Copyright 2007-2011 The Hyperspy developers
#
# This file is part of  Hyperspy.
#
#  Hyperspy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  Hyperspy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  Hyperspy.  If not, see <http://www.gnu.org/licenses/>.


'''Creates Digital Micrograph scripts to generate the dm3 testing files
'''

data_types = {
    1 :  '<i2', # 2 byte integer signed ("short")
    2 :  '<f4', # 4 byte real (IEEE 754)
    3 :  '<c8', # 8 byte complex
    5 :  '<c8', # 8 byte complex (packed)
    6 :  '<u1', # 1 byte integer unsigned ("byte")
    7 :  '<i4', # 4 byte integer signed ("long")
#    8 : (np.float32, 
#    {'R':('<u1',0), 'G':('<u1',1), 'B':('<u1',2), 'A':('<u1',3)}),
    9 :  '<i1', # byte integer signed
    10 : '<u2', # 2 byte integer unsigned
    11 : '<u4', # 4 byte integer unsigned
    12 : '<f8', # 8 byte real
    13 : '<c16', # byte complex
    14 : 'bool', # 1 byte binary (ie 0 or 1)
#    23 :  (np.float32, 
#    {'R':('<u1',0), 'G':('<u1',1), 'B':('<u1',2), 'A':('<u1',3)}),
     }

def generate_1D_files():
    f = open('generate_dm_1D_testing_files.s', 'w')
    f.write('image im\nstring filename\n')
    for key in data_types.iterkeys():
        f.write(
            'filename = "D:\\\\Python\\\\hyperspy\\\\lib\\\\test\\\\io\\\\'
            'dm3_1D_data\\\\test-%s.dm3"\n' 
            'im := NewImage("test", %i, 2)\n'
            'im[0,1] = 1\n'
            'im[1,2] = 2\n'
            'im.SaveImage(filename)\n' % (key, key))

    f.close()

def generate_2D_files():   
    f = open('generate_dm_2D_testing_files.s', 'w')
    f.write('image im\nstring filename\n''')
    for key in data_types.iterkeys():
        f.write(
            'filename = "D:\\\\Python\\\\hyperspy\\\\lib\\\\test\\\\io\\\\'
            'dm3_2D_data\\\\test-%s.dm3"\n'
            'im := NewImage("test", %i, 2, 2)\n'
            'im[0,0,1,1] = 1\n'
            'im[0,1,1,2] = 2\n'
            'im[1,0,2,1] = 3\n'
            'im[1,1,2,2] = 4\n'
            'im.SaveImage(filename)\n' % (key, key))

    f.close()

def generate_3D_files():
    f = open('generate_dm_3D_testing_files.s', 'w')
    f.write('image im\nstring filename\n')
    for key in data_types.iterkeys():
        f.write(
            'filename = "D:\\\\Python\\\\hyperspy\\\\lib\\\\test\\\\io\\\\'
            'dm3_3D_data\\\\test-%s.dm3"\n'
            'im := NewImage("test", %i, 2, 2,2)\n'
            'im[0,0,0,1,1,1] = 1\n'
            'im[1,0,0,2,1,1] = 2\n'
            'im[0,1,0,1,2,1] = 3\n'
            'im[1,1,0,2,2,1] = 4\n'
            'im[0,0,1,1,1,2] = 5\n'
            'im[1,0,1,2,1,2] = 6\n'
            'im[0,1,1,1,2,2] = 7\n'
            'im[1,1,1,2,2,2] = 8\n'
            'im.SaveImage(filename)\n' % (key, key))

    f.close()

def generate_4D_files():
    f = open('generate_dm_4D_testing_files.s', 'w')
    f.write('image im\nstring filename\n')
    for key in data_types.iterkeys():
        f.write(
            'filename = "D:\\\\Python\\\\hyperspy\\\\lib\\\\test\\\\io\\\\'
            'dm3_3D_data\\\\test-%s.dm3"\n'
            'im := NewImage("test", %i, 2,2,2,2)\n'
            'im[0,0,0,0,1,1,1,1] = 1\n'
            'im[1,0,0,0,2,1,1,1] = 2\n'
            'im[0,1,0,0,1,2,1,1] = 3\n'
            'im[1,1,0,0,2,2,1,1] = 4\n'
            'im[0,0,1,0,1,1,2,1] = 5\n'
            'im[1,0,1,0,2,1,2,1] = 6\n'
            'im[0,1,1,0,1,2,2,1] = 7\n'
            'im[1,1,1,0,2,2,2,1] = 8\n'
            'im[0,0,0,1,1,1,1,2] = 9\n'
            'im[1,0,0,1,2,1,1,2] = 10\n'
            'im[0,1,0,1,1,2,1,2] = 11\n'
            'im[1,1,0,1,2,2,1,2] = 12\n'
            'im[0,0,1,1,1,1,2,2] = 13\n'
            'im[1,0,1,1,2,1,2,2] = 14\n'
            'im[0,1,1,1,1,2,2,2] = 15\n'
            'im[1,1,1,1,2,2,2,2] = 16\n'
            'im.SaveImage(filename)\n' % (key, key))

    f.close()
    
if __name__ == '__main__':
    generate_1D_files()
    generate_2D_files()
    generate_3D_files()
    generate_4D_files()
    


