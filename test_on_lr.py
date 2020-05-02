import numpy as np
import pandas as pd
import shutil, os
import skimage.io
from keras.models import Model
from keras.layers import Input
from keras.layers import Lambda
import tensorflow as tf
from skimage import data, io, filters
from numpy import array
from numpy.random import randint
from skimage.transform import resize
import sys
import matplotlib.pyplot as plt

from keras.applications.vgg19 import VGG19
import keras.backend as K
from keras.models import Model
from keras.optimizers import Adam

from keras.layers import Dense
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.core import Flatten
from keras.layers import Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers import add
import imageio
import cv2

import skimage.io
img_lr = skimage.io.imread('/kaggle/input/test-image-lr/test_lion.jpg')
img_lr = cv2.resize(img_lr,(96,96))
output_dir = '/kaggle/working/output/'
os.mkdir('/kaggle/working/output')

os.mkdir('/kaggle/working/lr_image')
input_low_res = '/kaggle/working/lr_image'
skimage.io.imsave('/kaggle/working/lr_image/lr96_img.jpg',img_lr) #(96,96) image
number_of_images = 1
model_dir = '/kaggle/input/model-gan/gen_model3000.h5'



class VGG_LOSS(object):

    def __init__(self, image_shape):
        
        self.image_shape = image_shape

    # computes VGG loss or content loss
    def vgg_loss(self, y_true, y_pred):
    
        vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=self.image_shape)
        vgg19.trainable = False
        # Make trainable as False
        for l in vgg19.layers:
            l.trainable = False
        model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
        model.trainable = False
    
        return K.mean(K.square(model(y_true) - model(y_pred))) # content loss
    
    
image_shape = (96,96,3)
loss = VGG_LOSS(image_shape)
from keras.models import load_model
model = load_model(model_dir , custom_objects={'vgg_loss': loss.vgg_loss})

def load_path(path):
    directories = []
    if os.path.isdir(path):
        directories.append(path)
    for elem in os.listdir(path):
        if os.path.isdir(os.path.join(path,elem)):
            directories = directories + load_path(os.path.join(path,elem))
            directories.append(os.path.join(path,elem))
    return directories

def load_data_from_dirs(dirs, ext):
    files = []
    file_names = []
    count = 0
    for d in dirs:
        for f in os.listdir(d): 
            if f.endswith(ext):
                image = skimage.io.imread(os.path.join(d,f))
                if len(image.shape) > 2:
                    print (image.shape)
                    files.append(image)
                    file_names.append(os.path.join(d,f))
                count = count + 1
    print ('load data from dirs')
    return files     

def normalize(input_data):

    return (input_data.astype(np.float32) - 127.5)/127.5 


# Save only one image as sample  
def denormalize(input_data):
    input_data = (input_data + 1) * 127.5
    return input_data.astype(np.uint8)
   

def load_test_data(directory, ext, number_of_images = 100):

    files = load_data_from_dirs(load_path(directory), ext)
    
    if len(files) < number_of_images:
        print("Number of image files are less then you specified")
        print("Please reduce number of images to %d" % len(files))
        sys.exit()
        
    x_test_lr = np.array(files)
    x_test_lr = normalize(x_test_lr)

    return x_test_lr

        

x_test_lr = load_test_data(input_low_res, 'jpg', number_of_images)
image_batch_lr = denormalize(x_test_lr)
gen_img = model.predict(x_test_lr)
generated_image = denormalize(gen_img)


fig, axes = plt.subplots(nrows=1, ncols=2)

ax = axes.ravel()
ax[0].imshow(generated_image[0])
ax[0].set_title("low reso img")

ax[1].imshow(image_batch_lr[0])
ax[1].set_title("low reso img")


