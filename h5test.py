import h5py
import numpy as np

'''
# 打开文件
f = open('D:\pyfile\deepxde-master\examples\\function\\test.dat', 'r')

# 读取文件内容
content = f.read()

# 关闭文件
f.close()

# 输出文件内容
print(content)
'''


#file_path = 'D:\data\pansharpening\\training_data\\train_gf2.h5'
file_path = 'D:\data\pansharpening\\test_data\WV2\\test_wv2_multiExm1.h5'
f = h5py.File(file_path, 'r')
# fre = h5py.File("D:\data\pansharpening\\test_data\GF2\\new.h5", "w")
# d = np.load("D:\data\deeponet_antiderivative_aligned\\antiderivative_aligned_train.npz",allow_pickle=True)
# fre1 = h5py.File("D:\data\pansharpening\\test_data\GF2\\new1.h5", "r")
# fre2 = h5py.File("D:\data\pansharpening\\test_data\GF2\\new2.h5", "r")
# fre3 = h5py.File("D:\data\pansharpening\\test_data\GF2\\new3.h5", "r")
print("Objects in the HDF5 file:")
for name in f:
    print(name)

# readgt = np.zeros((19809,4))
# readms = np.zeros((19809,4))
# readlms = np.zeros((20,4,128,128))
# readpan = np.zeros((19809,1))
read0 = np.zeros((16384,5))
#readpair = np.zeros((19809,))

print(f['gt'][:].shape)
print(f['ms'][:].shape)
print(f['lms'][:].shape)
print(f['pan'][:].shape)


'''

#for i in range(20):
    #readpan[i, 0] = f['gt'][i, 0, 0, 0]
for j in range(4):
    for k in range(128):
        for l in range(128):
            #readgt[i,j]=f['gt'][i,j,0,0]
            #readms[i, j] = f['ms'][i, j, 0, 0]
            read0[128*k+l,j] = f['gt'][1,j,k,l]
            read0[128*k+l, 4] = f['pan'][1, 0, k, l]
        #readpair[i,0] = readms[i,:]
        #readpair[i,1] = readpan[i,:]
        #print(readpair[i,:])

#np.savetxt('data_train0.txt', readms, delimiter=' ')
np.savetxt('data_test1fig_gf2.txt', read0, delimiter=' ')


fre.create_dataset('gt', data=readgt)
fre.create_dataset('ms', data=readms)
fre.create_dataset('pan', data=readpan)
print(fre['gt'][:].shape)
print(fre['ms'][:].shape)
print(fre['pan'][:].shape)




print("Objects in the HDF5 file:")
for name in f:
    print(name)
    # print('first, we get values of gt:', f['lms'][:])
    # print('then, we get values of lms:', f['ms'][:])
    # print('then, we get values of lms:', f['pan'][:])

print(f['gt'][:].shape)
print(f['ms'][:].shape)
print(f['lms'][:].shape)
print(f['pan'][:].shape)


readgt = np.zeros((20,4,128,128))
readms = np.zeros((20,4,32,32))
readlms = np.zeros((20,4,128,128))
readpan = np.zeros((20,1,128,128))
for i in range(128):
    for j in range(128):
        readgt[:, :, i, j] = (f['gt'][:, :, 2 * i, 2 * j] + f['gt'][:, :, 2 * i + 1, 2 * j] + f['gt'][:, :, 2 * i, 2 * j + 1] + f['gt'][:, :, 2 * i + 1, 2 * j + 1]) / 4
        readlms[:, :, i, j] = (f['lms'][:, :, 2 * i, 2 * j] + f['lms'][:, :, 2 * i + 1, 2 * j] + f['lms'][:, :, 2 * i, 2 * j + 1] + f['lms'][:, :, 2 * i + 1, 2 * j + 1]) / 4
        readpan[:, :, i, j] = (f['pan'][:, :, 2 * i, 2 * j] + f['pan'][:, :, 2 * i + 1, 2 * j] + f['pan'][:, :, 2 * i, 2 * j + 1] + f['pan'][:, :, 2 * i + 1, 2 * j + 1]) / 4
        print(i,j)
for i in range(32):
    for j in range(32):
        readms[:, :, i, j] = (f['ms'][:, :, 2 * i, 2 * j] + f['ms'][:, :, 2 * i + 1, 2 * j] + f['ms'][:, :, 2 * i, 2 * j + 1] + f['ms'][:, :, 2 * i + 1, 2 * j + 1]) / 4

fre.create_dataset('gt', data=readgt)
fre.create_dataset('lms', data=readlms)
fre.create_dataset('ms', data=readms)
fre.create_dataset('pan', data=readpan)
print(fre['gt'][:].shape)
print(fre['ms'][:].shape)
print(fre['lms'][:].shape)
print(fre['pan'][:].shape)


print('values of gt:', f['gt'][:])
print('values of gt1:', fre['gt'][:])



readgt = np.zeros((20,4,128,128))
readgt = f['gt'][:,:,128:,:128]
readgt1 = np.zeros((20,4,128,128))
readgt1 = f['gt'][:,:,128:,:128]
readgt2 = np.zeros((20,4,128,128))
readgt2 = f['gt'][:,:,:128,128:]
readgt3 = np.zeros((20,4,128,128))
readgt3 = f['gt'][:,:,128:,128:]

# print('values of gt:', f['gt'][:])
# print('values of readgt:', readgt[:])
# print(readgt[:].shape)

readms1 = np.zeros((20,4,32,32))
readms1 = f['ms'][:,:,32:,:32]
readlms1 = np.zeros((20,4,128,128))
readlms1 = f['lms'][:,:,128:,:128]
readpan1 = np.zeros((20,1,128,128))
readpan1 = f['pan'][:,:,128:,:128]

readms2 = np.zeros((20,4,32,32))
readms2 = f['ms'][:,:,:32,32:]
readlms2 = np.zeros((20,4,128,128))
readlms2 = f['lms'][:,:,:128,128:]
readpan2 = np.zeros((20,1,128,128))
readpan2 = f['pan'][:,:,:128,128:]

readms3 = np.zeros((20,4,32,32))
readms3 = f['ms'][:,:,32:,32:]
readlms3 = np.zeros((20,4,128,128))
readlms3 = f['lms'][:,:,128:,128:]
readpan3 = np.zeros((20,1,128,128))
readpan3 = f['pan'][:,:,128:,128:]



readgt = np.zeros((20,4,128,128))
readgt1 = np.zeros((20,4,128,128))
readgt2 = np.zeros((20,4,128,128))
readgt3 = np.zeros((20,4,128,128))
readms = np.zeros((20,4,32,32))
readms1 = np.zeros((20,4,32,32))
readms2 = np.zeros((20,4,32,32))
readms3 = np.zeros((20,4,32,32))
readlms = np.zeros((20,4,128,128))
readlms1 = np.zeros((20,4,128,128))
readlms2 = np.zeros((20,4,128,128))
readlms3 = np.zeros((20,4,128,128))
readpan = np.zeros((20,1,128,128))
readpan1 = np.zeros((20,1,128,128))
readpan2 = np.zeros((20,1,128,128))
readpan3 = np.zeros((20,1,128,128))

for i in range(128):
    for j in range(128):
        readgt[:, :, i, j] = f['gt'][:, :, 2 * i - 1, 2 * j - 1]
        readgt1[:, :, i, j] = f['gt'][:, :, 2 * i - 1, 2 * j]
        readgt2[:, :, i, j] = f['gt'][:, :, 2 * i, 2 * j - 1]
        readgt3[:, :, i, j] = f['gt'][:, :, 2 * i, 2 * j]
        readlms[:, :, i, j] = f['lms'][:, :, 2 * i - 1, 2 * j - 1]
        readlms1[:, :, i, j] = f['lms'][:, :, 2 * i - 1, 2 * j]
        readlms2[:, :, i, j] = f['lms'][:, :, 2 * i, 2 * j - 1]
        readlms3[:, :, i, j] = f['lms'][:, :, 2 * i, 2 * j]
        readpan[:, :, i, j] = f['pan'][:, :, 2 * i - 1, 2 * j - 1]
        readpan1[:, :, i, j] = f['pan'][:, :, 2 * i - 1, 2 * j]
        readpan2[:, :, i, j] = f['pan'][:, :, 2 * i, 2 * j - 1]
        readpan3[:, :, i, j] = f['pan'][:, :, 2 * i, 2 * j]
        print(i,j)
for i in range(32):
    for j in range(32):
        readms[:, :, i, j] = f['ms'][:, :, 2 * i - 1, 2 * j - 1]
        readms1[:, :, i, j] = f['ms'][:, :, 2 * i - 1, 2 * j]
        readms2[:, :, i, j] = f['ms'][:, :, 2 * i, 2 * j - 1]
        readms3[:, :, i, j] = f['ms'][:, :, 2 * i, 2 * j]


fre.create_dataset('gt', data=readgt)
fre.create_dataset('lms', data=readlms)
fre.create_dataset('ms', data=readms)
fre.create_dataset('pan', data=readpan)
print(fre['gt'][:].shape)
print(fre['ms'][:].shape)
print(fre['lms'][:].shape)
print(fre['pan'][:].shape)

fre1.create_dataset('gt', data=readgt1)
fre1.create_dataset('lms', data=readlms1)
fre1.create_dataset('ms', data=readms1)
fre1.create_dataset('pan', data=readpan1)
print(fre1['gt'][:].shape)
print(fre1['ms'][:].shape)
print(fre1['lms'][:].shape)
print(fre1['pan'][:].shape)

fre2.create_dataset('gt', data=readgt2)
fre2.create_dataset('lms', data=readlms2)
fre2.create_dataset('ms', data=readms2)
fre2.create_dataset('pan', data=readpan2)
print(fre2['gt'][:].shape)
print(fre2['ms'][:].shape)
print(fre2['lms'][:].shape)
print(fre2['pan'][:].shape)

fre3.create_dataset('gt', data=readgt3)
fre3.create_dataset('lms', data=readlms3)
fre3.create_dataset('ms', data=readms3)
fre3.create_dataset('pan', data=readpan3)
print(fre3['gt'][:].shape)
print(fre3['ms'][:].shape)
print(fre3['lms'][:].shape)
print(fre3['pan'][:].shape)



print('values of gt:', f['gt'][:])
print('values of gt1:', fre['gt'][:])
print('values of gt2:', fre1['gt'][:])
print('values of gt3:', fre2['gt'][:])
print('values of gt4:', fre3['gt'][:])
'''


f.close()
# fre.close()
# fre1.close()
# fre2.close()
# fre3.close()