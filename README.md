# Image-high-resolution-SRgan
To provide high resolution image based on lower scaled image
input  image : 96x96 pixels
output image : 384x384 pixels
upscale image : 4x

Keras-SRGAN
Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network implemented in Keras

For more about topic check Single Image Super Resolution Using GANs — Keras

Problem Statement:
Enhancing low resolution images by applying deep network with adversarial network (Generative Adversarial Networks) 
to produce high resolutions images.
Architecture:
Basic Architecture

Generator and Discriminator Network:
Network

Network Details:
* 16 Residual blocks used.
* PixelShuffler x2: This is feature map upscaling. 2 sub-pixel CNN are used in Generator.
* PRelu(Parameterized Relu): We are using PRelu in place of Relu or LeakyRelu. It introduces learn-able parameter 
  that makes it possible to adaptively learn the negative part coefficient.
* k3n64s1 this means kernel 3, channels 64 and strides 1.
* Loss Function: We are using Perceptual loss. It comprises of Content(Reconstruction) loss and Adversarial loss.
How it Works:
* We process the HR(High Resolution) images to get down-sampled LR(Low Resolution) images. Now we have both HR 
  and LR images for training data set.
* We pass LR images through Generator which up-samples and gives SR(Super Resolution) images.
* We use a discriminator to distinguish the HR images and back-propagate the GAN loss to train the discriminator
  and the generator.
* As a result of this, the generator learns to produce more and more realistic images(High Resolution images) as 
  it trains.
Documentation:
You can find more about this implementation in my post : Single Image Super Resolution Using GANs — Keras

Requirements:
You will need the following to run the above:
Python 3.5.4
tensorflow 1.11.0
keras 2.2.4
numpy 1.10.4
matplotlib, skimage, scipy

For training: Good GPU, I trained my model on NVIDIA Tesla P100 (This model is trained by real owner,i had run for only 5 epochs as my GPU compatibility is low
Data set:
* Used COCO data set 2017. It is around 18GB having images of different dimensions.
* Used 800 images for training(Very less, You can take more (approx. 350 according to original paper) thousand is you can
  collect and have very very good GPU). Preprocessing includes cropping images so that we can have same dimension images. 
  Images with same width and height are preferred. I used images of size 384 for high resolution.
* After above step you have High Resolution images. Now you have to get Low Resolution images which you can get by down 
  scaling HR images. I used down scale = 4. So Low resolution image of size 96 we will get. Sample code for this.
File Structure:
*test_on_hr.py : This will take high resolution image and compare the output with input and real image
*test_on_lr.py : This will take low resolution image and convert it into high resolution image
*train.py : This will train the GAN model and the model is saved in model directory
*output : All the tested images are in this folder


*for full description of code and model visit : https://github.com/deepak112/Keras-SRGAN
