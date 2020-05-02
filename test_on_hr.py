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


dst_dir = '/working/input_img/'
os.mkdir('/working/input_img')
os.mkdir('/working/output')

#copy and paste selected images from dataset for checking the result

files = ['/input/coco-image/val2017/000000122166.jpg',
'/input/coco-image/val2017/000000406997.jpg']
for f in files:
    shutil.copy(f, dst_dir)

os.listdir(dst_dir)

model_dir = '/input/model-gan/gen_model3000.h5'
input_hig_res = dst_dir #This directory is used to read input images 
number_of_images = 2
output_dir = '/output/' #This is used to save the generated images

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

#load hr images
def load_data_from_dirs(dirs, ext):
    files = []
    file_names = []
    count = 0
    for d in dirs:
        for f in os.listdir(d): 
            if f.endswith(ext):
                image = skimage.io.imread(os.path.join(d,f))
                if len(image.shape) > 2:
                    image = cv2.resize(image, (384, 384))
                    files.append(image)
                    file_names.append(os.path.join(d,f))
                count = count + 1
    print ('load data from dirs')
    return files     

def load_path(path):
    directories = []
    if os.path.isdir(path):
        directories.append(path)
    for elem in os.listdir(path):
        if os.path.isdir(os.path.join(path,elem)):
            directories = directories + load_path(os.path.join(path,elem))
            directories.append(os.path.join(path,elem))
    return directories

# Takes list of images and provide HR images in form of numpy array
def hr_images(images):
    images_hr = array(images)
    return images_hr

# Takes list of images and provide LR images in form of numpy array
def lr_images(images_real , downscale):
    
    images = []
    print (len(images_real))
    for img in  range(len(images_real)):
        images.append(resize(images_real[img], [images_real[img].shape[0]//downscale,images_real[img].shape[1]//downscale],anti_aliasing=True))    
    images_lr = array(images)
    return images_lr
    
def normalize(input_data):

    return (input_data.astype(np.float32) - 127.5)/127.5 


def load_test_data_for_model(directory, ext, number_of_images = 100):

    files = load_data_from_dirs(load_path(directory), ext)
    
    if len(files) < number_of_images:
        print("Number of image files are less then you specified")
        print("Please reduce number of images to %d" % len(files))
        sys.exit()
        
    x_test_hr = hr_images(files)
    x_test_hr = normalize(x_test_hr)
    x_test_lr = lr_images(files, 4)
    return x_test_lr, x_test_hr

x_test_lr, x_test_hr = load_test_data_for_model(input_hig_res, 'jpg', number_of_images)

# Save only one image as sample  
def denormalize(input_data):
    input_data = (input_data + 1) * 127.5
    return input_data.astype(np.uint8)
   
    
def plot_test_generated_images_for_model(output_dir, generator, x_test_hr, x_test_lr , dim=(1, 3), figsize=(15, 5)):
    
    examples = x_test_hr.shape[0]
    image_batch_hr = denormalize(x_test_hr)
    image_batch_lr = x_test_lr
    gen_img = generator.predict(image_batch_lr)
    generated_image = gen_img
    print (image_batch_lr.shape)
    print (generated_image.shape)
    print (image_batch_hr.shape)
    
    for index in range(examples):
    
        plt.figure(figsize=figsize)
    
        plt.subplot(dim[0], dim[1], 1)
        plt.imshow(image_batch_lr[index], interpolation='nearest')
        plt.axis('off')
        
        plt.subplot(dim[0], dim[1], 2)
        plt.imshow(generated_image[index], interpolation='nearest')
        plt.axis('off')
    
        plt.subplot(dim[0], dim[1], 3)
        plt.imshow(image_batch_hr[index], interpolation='nearest')
        plt.axis('off')
    
        plt.tight_layout()
        plt.savefig(output_dir + 'test_generated_image_%d.png' % index)
    
        plt.show()
        
plot_test_generated_images_for_model(output_dir, model, x_test_hr, x_test_lr)