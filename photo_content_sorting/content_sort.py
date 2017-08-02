from __future__ import print_function
import time
from PIL import Image
import numpy as np
import glob
from keras import backend
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

path2im = '/home/rey/Github/content_sort/'
album = ['art/','mountain/','person/','beach/','friends_family/']
feature_avg = [np.zeros((1,7,7,512)) for i in xrange(0,len(album))]
model = VGG16(weights='imagenet',include_top=False)

'''Train based on current albums'''
for j in xrange(len(album)):
    c = 0.0
    for file in glob.glob(path2im+album[j]+'*.jpg'):
        im = Image.open(file)
        im = im.resize((224,224))
        mat = np.asarray(im, dtype='float32')
        mat = np.expand_dims(im, axis=0)
        mat.setflags(write=1)
        mat = np.float64(mat)
        mat = preprocess_input(mat)
        feature = model.predict(mat)
        feature_avg[j] += feature
        c += 1.0
    feature_avg[j] = feature_avg[j]/c

'''Predict one image that is not in the album'''
im = path2im + 'test_images/dali.jpg'
im = full_res = Image.open(im)
im = im.resize((224, 224))
feature = np.asarray(im, dtype='float32')
feature = np.expand_dims(feature, axis=0)
feature = preprocess_input(feature)
preds = model.predict(feature)
loss = [np.mean(pow((preds-feature_avg[i]),2)) for i in xrange(0,len(album))]
prediction = album[np.argmin(loss)]
print('Prediction: %s' % prediction)

'''Predict all images that are not in the album'''
for file in glob.glob(path2im+'test_images/'+'*.jpg'):
    im = Image.open(file)
    im = im.resize((224, 224))
    feature = np.asarray(im, dtype='float32')
    feature = np.expand_dims(feature, axis=0)
    feature = preprocess_input(feature)
    preds = model.predict(feature)
    loss = [np.mean(pow((preds-feature_avg[i]),2)) for i in xrange(0,(album))]
    prediction = album[np.argmin(loss)]
    print('%s --Prediction:%s' % (file[42:],prediction))
