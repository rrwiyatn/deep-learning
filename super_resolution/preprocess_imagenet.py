import numpy as np
from PIL import Image
import time
import glob
import cv2
import bcolz

MAX_SIZE = 288
lim = 9999
start_time = time.time()

path2im = 'datasets/imagenet/train/'

count_image = 0
for file in glob.glob(path2im+'**/*.JPEG'):
    count_image += 1
print('Count training data: %d' % count_image)

x_train_good = np.zeros((lim,MAX_SIZE,MAX_SIZE,3))

i = 0
for file in glob.glob(path2im+'**/*.JPEG'):
    if(i%1000==0):
        print('%s, Image #%d' % (file,i))
    try:
        im = Image.open(file)
    except IOError:
        print("cannot open", file)
        break
    mat = np.array(im)
    if(len(mat.shape)==2):      #if image is in greyscale, make it 3 channel
        matR=matG=matB=mat
        mat = np.stack((matR,matG,matB),axis=2)
    height = mat.shape[0]
    width = mat.shape[1]
    while(height > MAX_SIZE or width > MAX_SIZE):
        if(height > width):
            factor = float(MAX_SIZE)/float(height)
        elif(width > height):
            factor = float(MAX_SIZE)/float(width)
        mat = cv2.resize(mat, (0,0), fx=factor, fy=factor)
        height = mat.shape[0]
        width = mat.shape[1]
    diffHeight = MAX_SIZE - height
    diffWidth = MAX_SIZE - width
    padded0 = np.pad(mat[:,:,0],((diffHeight/2,diffHeight/2),(diffWidth/2,diffWidth/2)),'constant',constant_values=(0,0))
    padded1 = np.pad(mat[:,:,1],((diffHeight/2,diffHeight/2),(diffWidth/2,diffWidth/2)), 'constant',constant_values=(0,0))
    padded2 = np.pad(mat[:,:,2],((diffHeight/2,diffHeight/2),(diffWidth/2,diffWidth/2)), 'constant',constant_values=(0,0))
    if(diffHeight%2!=0):
        padded0 = np.pad(padded0,((1,0),(0,0)), 'constant',constant_values=(0,0))
        padded1 = np.pad(padded1,((1,0),(0,0)), 'constant',constant_values=(0,0))
        padded2 = np.pad(padded2,((1,0),(0,0)), 'constant',constant_values=(0,0))
    if(diffWidth%2!=0):
        padded0 = np.pad(padded0,((0,0),(1,0)), 'constant',constant_values=(0,0))
        padded1 = np.pad(padded1,((0,0),(1,0)), 'constant',constant_values=(0,0))
        padded2 = np.pad(padded2,((0,0),(1,0)), 'constant',constant_values=(0,0))
    padded = np.stack((padded0,padded1,padded2), axis=2)
    x_train_good[i,:] = padded
    i += 1
    if i==lim:
        break

print('Saving high res images...')
x_train_good = x_train_good.astype('uint8') 
dat=bcolz.carray(x_train_good, rootdir='datasets/imagenet/y_train.bc', mode='w')
dat.flush()