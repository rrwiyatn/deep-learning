import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np
import keras
from keras import backend as K
from keras.layers import Input, Dense, Conv2D, MaxPooling2D,\
        UpSampling2D, Lambda, Activation, merge
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.applications.vgg16 import VGG16
from PIL import Image
import scipy.misc
import matplotlib.pyplot as plt
from scipy.misc import imsave
import bcolz

#Load dataset
x_train = bcolz.open('data/x_train.bc')[:]
y_train = bcolz.open('data/y_train.bc')[:]
output_shape = y_train[0].shape
num_images = y_train.shape[0]

def convolution_block(x, filters, size, strides=(1,1), padding='same', act=True):
    x = Conv2D(filters, (size,size), strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    if act == True:
        x = Activation('relu')(x)
    return x

def residual_block(blockInput, num_filters=64):
    x = convolution_block(blockInput, num_filters, 3)
    x = convolution_block(x, num_filters, 3, act=False)
    x = merge([x, blockInput], mode='sum')
    return x

def upsampling_block(x, filters, size):
    x = UpSampling2D()(x)
    x = Conv2D(filters, (size,size), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

'''NETWORK #1'''
inp=Input(x_train.shape[1:])
x=convolution_block(inp, 64, 9)
x=residual_block(x)
x=residual_block(x)
x=residual_block(x)
x=residual_block(x)
x=upsampling_block(x, 64, 3)
x=upsampling_block(x, 64, 3)
#n1_out=convolution_block(x, 3, 9)
x=Conv2D(3, (9,9), activation='tanh', padding='same')(x)
n1_out=Lambda(lambda x: (x+1)*127.5)(x) #scale output so we get 0 to 255

'''NETWORK #2 - VGG'''
#We want to use VGG so that we know the difference in activation between
#high-res image and output of the low-res image
#High-res -> VGG -> high-res activation
#Low-res -> trainableCNN -> VGG -> generated image activation

#Note that there are 2 inputs for VGG network:
#   1. Output of the low-res image from trainable network
#   2. High-res image

'''VGG input preprocessing as stated in the paper'''
vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)
preprocess_vgg = lambda x: (x - vgg_mean)[:, :, :, ::-1]

vgg_inp=Input(output_shape)
vgg= VGG16(include_top=False, input_tensor=Lambda(preprocess_vgg)(vgg_inp))

for layer in vgg.layers: 
    layer.trainable=False

#Note that we want the activation from early layers
#Recall from style transfer paper:
#   Early conv layers gives clearer contents

def get_output(m, ln):
    name = 'block' + str(ln) + '_conv1'
    return m.get_layer(name).output

#Define model that will grab the activations from first 3 conv layers
vgg_content = Model(vgg_inp, [get_output(vgg, o) for o in [1,2,3]])

#vgg1 = for the high res image
vgg1 = vgg_content(vgg_inp)

#vgg2 = for the generated image
vgg2 = vgg_content(n1_out)

def mean_squared_error(diff): 
    dims = list(range(1,K.ndim(diff)))
    return K.expand_dims(K.sqrt(K.mean(diff**2, dims)), 0)

layer_weights=[0.3, 0.65, 0.05]
def content_fn(x):
    res = 0
    n=len(layer_weights)
    for i in range(n):
        res += mean_squared_error(x[i]-x[i+n]) * layer_weights[i]
    return res

#Define the model that actually minimizes the loss
model = Model([inp, vgg_inp], Lambda(content_fn)(vgg1+vgg2))
#We want the output of our model (loss) to be zeros
target = np.zeros((num_images, 1)) 

model.compile(optimizer='adam', loss='mse')
model.fit([x_train, y_train], target, batch_size=25, epochs=100)

#Define the trained model
trained_model = Model(inp, n1_out)

#Let's predict the first 30 images in our dataset
predictions = trained_model.predict(x_train[:30])

i = 28
single_prediction = predictions[i].astype('uint8')
imsave('before.jpg',x_train[i])
imsave('after.jpg',single_prediction)

#Save weights
trained_model.save_weights('model.h5')
