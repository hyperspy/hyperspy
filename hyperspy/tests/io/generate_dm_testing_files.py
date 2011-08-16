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
# TODO: write correctely the path
# 1D
f = open('generate_dm_1D_testing_files.s', 'w')
f.write('''image im
string filename''')
for key in data_types.iterkeys():
    f.write('''
filename = "D:\\\\Python\\\\hyperspy\\\\lib\\\\test\\\\io\\\\dm3_1D_data\\\\test-%s.dm3"
im := NewImage("test", %i, 2)
im[0,1] = 1
im[1,2] = 2
im.SaveImage(filename)
''' % (key, key))

f.close()

# 2D   
f = open('generate_dm_2D_testing_files.s', 'w')
f.write('''image im
string filename''')
for key in data_types.iterkeys():
    f.write('''
filename = "D:\\\\Python\\\\hyperspy\\\\lib\\\\test\\\\io\\\\dm3_2D_data\\\\test-%s.dm3"
im := NewImage("test", %i, 2, 2)
im[0,0,1,1] = 1
im[0,1,1,2] = 2
im[1,0,2,1] = 3
im[1,1,2,2] = 4
im.SaveImage(filename)
''' % (key, key))

f.close()

# 3D

f = open('generate_dm_3D_testing_files.s', 'w')
f.write('''image im
string filename''')
for key in data_types.iterkeys():
    f.write('''
filename = "D:\\\\Python\\\\hyperspy\\\\lib\\\\test\\\\io\\\\dm3_3D_data\\\\test-%s.dm3"
im := NewImage("test", %i, 2, 2,2)
im[0,0,0,1,1,1] = 1
im[1,0,0,2,1,1] = 2
im[0,1,0,1,2,1] = 3
im[1,1,0,2,2,1] = 4
im[0,0,1,1,1,2] = 5
im[1,0,1,2,1,2] = 6
im[0,1,1,1,2,2] = 7
im[1,1,1,2,2,2] = 8
im.SaveImage(filename)
''' % (key, key))

f.close()

# 4D
f = open('generate_dm_4D_testing_files.s', 'w')
f.write('''image im
string filename''')
for key in data_types.iterkeys():
    f.write('''
filename = "D:\\\\Python\\\\hyperspy\\\\lib\\\\test\\\\io\\\\dm3_3D_data\\\\test-%s.dm3"
im := NewImage("test", %i, 2,2,2,2)
im[0,0,0,0,1,1,1,1] = 1
im[1,0,0,0,2,1,1,1] = 2
im[0,1,0,0,1,2,1,1] = 3
im[1,1,0,0,2,2,1,1] = 4
im[0,0,1,0,1,1,2,1] = 5
im[1,0,1,0,2,1,2,1] = 6
im[0,1,1,0,1,2,2,1] = 7
im[1,1,1,0,2,2,2,1] = 8
im[0,0,0,1,1,1,1,2] = 9
im[1,0,0,1,2,1,1,2] = 10
im[0,1,0,1,1,2,1,2] = 11
im[1,1,0,1,2,2,1,2] = 12
im[0,0,1,1,1,1,2,2] = 13
im[1,0,1,1,2,1,2,2] = 14
im[0,1,1,1,1,2,2,2] = 15
im[1,1,1,1,2,2,2,2] = 16
im.SaveImage(filename)
''' % (key, key))

f.close()


