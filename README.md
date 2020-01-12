# Super Resolution Keras

This repository use a Super Resolution Deep Learning approach using SRCNN and FSRCNN implementation.

SRCNN use bicubic interpolation before feeding in the CNN the Low Resolution Image.
FSRCNN use deconvolution layer in order to upsample the Low Resolution Image.

![](https://github.com/Shiro-LK/Super-Resolution/blob/master/image/fsrcnn.png)


# Evaluation on Set5 dataset
PSNR metric

SRCNN : 33.62

SRCNNex : 33.12

FSRCNN (ELu): 33.5

FSRCNN (PReLU) : 33.96


# To do
implementation of Super Resolution GAN algorithm




