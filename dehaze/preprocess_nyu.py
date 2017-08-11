import numpy as np
from PIL import Image
import time
import glob
import os
import bcolz

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
IMAGE_CHANNEL = 3

start_time = time.time()

path_xtrain ='data/x_jpg/'
path_ytrain ='data/y_jpg/'

#We want to save the data as ndarray (num_images,480,720,3)
count_train = 0


for file in glob.glob(path_xtrain+'*.bmp'):
    #a = file.split('/')
    #a = a[2].split('_')
    #dst = 'data/NYU_Hazy_renamed/' + a[0] + '.bmp'
    #os.rename(file, dst)
    a = file.split('/')
    a = a[2].split('_')
    a = a[0].split('.')
    title = a[0] + '.jpg'
    im = Image.open(file)
    im.save('data/x_jpg/' + title)
    count_train += 1
print('Count x training data: %d' % count_train)


count_train = 0
for file in glob.glob(path_ytrain+'*.jpg'):
    #a = file.split('/')
    #a = a[2].split('_')
    #dst = 'data/NYU_GT_renamed/' + a[0] + '.bmp'
    #os.rename(file, dst)
    a = file.split('/')
    a = a[2].split('_')
    a = a[0].split('.')
    title = a[0] + '.jpg'
    im = Image.open(file)
    im.save('data/y_jpg/' + title)
    count_train += 1
print('Count y training data: %d' % count_train)


x_train = np.zeros((count_train,IMAGE_HEIGHT,IMAGE_WIDTH,IMAGE_CHANNEL))
y_train = np.zeros((count_train,IMAGE_HEIGHT,IMAGE_WIDTH,IMAGE_CHANNEL))


def save_array(fname, arr):
    c=bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()


i = 0
for file in glob.glob(path_xtrain+'*.jpg'):
    print('%s, Image #%d' % (file,i))
    im = Image.open(file)
    mat = np.asarray(im)
    x_train[i,:] = mat
    i += 1
print('X training ready')

i = 0
for file in glob.glob(path_ytrain+'*.jpg'):
    print('%s, Image #%d' % (file,i))
    im = Image.open(file)
    mat = np.asarray(im)
    y_train[i,:] = mat
    i += 1
print('Y training ready')

print('Saving...')
save_array('data/x_train.bc', x_train)
save_array('data/y_train.bc', y_train)

end_time = time.time()
print('Time needed: %ds' % (end_time - start_time))
