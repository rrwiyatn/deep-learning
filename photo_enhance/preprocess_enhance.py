import numpy as np
from PIL import Image
import time
import glob

start_time = time.time()

path2im = 'images/'
path_xtrain ='images/x_train/'
path_ytrain ='images/y_train/'
path_xvalidate ='images/x_validate/'
path_yvalidate ='images/y_validate/'

#We want to save the data as ndarray (num_images,480,720,3)
count_train = 0
for file in glob.glob(path_xtrain+'*.jpeg'):
    count_train += 1
print('Count training data: %d' % count_train)

count_validate = 0
for file in glob.glob(path_xvalidate+'*.jpeg'):
    count_validate += 1
print('Count validation data: %d' % count_validate)

x_train = np.zeros((count_train,480,720,3))
y_train = np.zeros((count_train,480,720,3))
x_validate = np.zeros((count_validate,480,720,3))
y_validate = np.zeros((count_validate,480,720,3))

i = 0
#for each image in the folder, resize to 480 x 720
for file in glob.glob(path_xtrain+'*.jpeg'):
    print('%s, Image #%d' % (file,i))
    im = Image.open(file)
    im = im.resize((720,480))
    mat = np.array(im) #(480,720,3)
    x_train[i,:] = mat
    i += 1
print('X training ready')

i = 0
for file in glob.glob(path_ytrain+'*.jpeg'):
    print('%s, Image #%d' % (file,i))
    im = Image.open(file)
    im = im.resize((720,480))
    mat = np.array(im) #(480,720,3)
    y_train[i,:] = mat
    i += 1
print('Y training ready')

i = 0
for file in glob.glob(path_xvalidate+'*.jpeg'):
    print('%s, Image #%d' % (file,i))
    im = Image.open(file)
    im = im.resize((720,480))
    mat = np.array(im) #(480,720,3)
    x_validate[i,:] = mat
    i += 1
print('X validation ready')

i = 0
for file in glob.glob(path_yvalidate+'*.jpeg'):
    print('%s, Image #%d' % (file,i))
    im = Image.open(file)
    im = im.resize((720,480))
    mat = np.array(im) #(480,720,3)
    y_validate[i,:] = mat
    i += 1
print('Y validation ready')

np.save("data/x_train.npy",x_train)
np.save("data/y_train.npy",y_train)
#np.save("x_validate.npy",x_validate)
#np.save("y_validate.npy",y_validate)
print ('Saved items:')
print ('1. x_train.npy')
print ('2. y_train.npy')
print ('3. x_validate.npy"')
print ('4. y_validate.npy"')

end_time = time.time()
print('Time needed: %ds' % (end_time - start_time))