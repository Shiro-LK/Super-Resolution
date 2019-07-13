# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 13:20:53 2019

@author: Shiro
"""

import cv2
import numpy as np
import keras
import os
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Convolution2D, Conv2DTranspose, BatchNormalization, Conv2D, Activation, add
from keras.layers import GlobalMaxPooling2D, Flatten, PReLU, Lambda
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau

from keras.optimizers import Nadam, Adam, SGD
def PSNR(y_true, y_pred):
    y_pred = K.round(y_pred)
    MSE = K.mean( K.square( y_true - y_pred ) )
    
    d = 255.0**2
    
    metric = 10.0 * K.log(d/MSE)/K.log(10.0)
    return metric

def SRCNN(input_shape= (None, None, 3), depth_multiplier=1): #33.62
    """
        first conv = extract features
        2nd conv = mapping
        3rd conv = prediction
        @ multi_output : set to True 
    """
    inputs = Input(input_shape, name="inputs")
    
    # normalize value between 0 and 1
    inputs_norm = Lambda(lambda x: x / 255.0)(inputs)
    
    
    conv1 = Convolution2D(filters=64*depth_multiplier, kernel_size=9, padding="same", name="conv1", activation="relu")(inputs_norm)
    #conv1 = BatchNormalization(name='bn_conv1')(conv1)
    
    mapping = Convolution2D(filters=32*depth_multiplier, kernel_size=1, padding="same", name="mapping", activation="relu")(conv1)
    #mapping = BatchNormalization(name='bn_mapping')(mapping)
    
    out = Convolution2D(filters=3, kernel_size=5, padding="same", name="output", activation="sigmoid")(mapping)
    output = Lambda(lambda x: x * 255.0)(out)
    return Model(inputs, output)

def SRCNNex(input_shape= (None, None, 3), depth_multiplier=1, multi_output=False): #33.12
    """
        Implementation of SRCNNex. The kernel size of the mapping layer is increased from 1 to 5.
        @ multi_output : set to True 
    """
    
    inputs = Input(input_shape, name="inputs")
    # normalize value between 0 and 1
    inputs_norm = Lambda(lambda x: x / 255.0)(inputs)
    
    conv1 = Convolution2D(filters=64, kernel_size=9, padding="same", name="conv1", activation="relu")(inputs_norm)
    mapping = Convolution2D(filters=32, kernel_size=5, padding="same", name="mapping", activation="relu")(conv1)
    
    out = Convolution2D(filters=3, kernel_size=5, padding="same", name="output", activation="sigmoid")(mapping)
    output = Lambda(lambda x: x * 255.0)(out)
    return Model(inputs, output)


def FSRCNN(input_shape=(None, None, 3), depth_multi=1, scale=2): # 33.5
    s = 12
    d=56
    inputs = Input(input_shape, name="inputs")
    #normalize input
    inputs_norm = Lambda(lambda x: x / 255.0)(inputs)
    
    conv1 = Convolution2D(filters=d, kernel_size=5, padding="same", name="conv1", activation="elu")(inputs_norm)

    conv2 = Convolution2D(filters=s, kernel_size=1, padding="same", name="conv2", activation="elu")(conv1)

    conv3 = conv2
    for i in range(4):
        conv3 = Convolution2D(filters=s, kernel_size=3, padding="same", name="conv3_"+str(i), activation="elu")(conv3)


    conv4 = Convolution2D(filters=d, kernel_size=1, padding="same", name="conv4" , activation="elu")(conv3)

    deconv = Conv2DTranspose(filters=3, kernel_size=9, strides=scale, padding="same", name="deconv", activation="sigmoid")(conv4)

    
    output = Lambda(lambda x: x * 255.0)(deconv)
    return Model(inputs, output)    

def FSRCNN2(input_shape=(None, None, 3), depth_multi=1, scale=2): # 33.96
    s = 12
    d=56
    inputs = Input(input_shape, name="inputs")
    #normalize input
    inputs_norm = Lambda(lambda x: x / 255.0)(inputs)
    
    conv1 = Convolution2D(filters=d, kernel_size=5, padding="same", name="conv1")(inputs_norm)
    conv1 = PReLU(shared_axes=[1,2])(conv1)
    conv2 = Convolution2D(filters=s, kernel_size=1, padding="same", name="conv2")(conv1)
    conv2 = PReLU(shared_axes=[1,2])(conv2)
    conv3 = conv2
    for i in range(4):
        conv3 = Convolution2D(filters=s, kernel_size=3, padding="same", name="conv3_"+str(i))(conv3)
        conv3 = PReLU(shared_axes=[1,2])(conv3)

    conv4 = Convolution2D(filters=d, kernel_size=1, padding="same", name="conv4" )(conv3)
    conv4 = PReLU(shared_axes=[1,2])(conv4)
    deconv = Conv2DTranspose(filters=3, kernel_size=9, strides=scale, padding="same", name="deconv", activation="sigmoid")(conv4)
    #out = Convolution2D(filters=3, kernel_size=5, padding="same", name="output", activation="sigmoid")(deconv)

    
    output = Lambda(lambda x: x * 255.0)(deconv)
    return Model(inputs, output)    
class generator_SR():
    def __init__(self, filenames, scale, mod="SRCNN"):
        self.filenames = filenames
        self.scale = scale
        self.mod = mod
        self.steps = len(filenames)
        
        
    def generator(self, augmentation=False, augmentation_scale=False):

        filenames = self.filenames
        while True:
            
            filenames.sort()
            for name in filenames:
                    y = cv2.imread(name)
                    
                    if augmentation_scale:
                        if np.random.randint(0,2) == 1:
                            h, w, c = y.shape
                            x1 = np.random.randint(0, w//3)
                            y1 = np.random.randint(0, h//3)
                            x2 = np.random.randint(w//2, w)
                            y2 = np.random.randint(h//2, h)
                        
                            y = y[x1:x2, y1:y2 ]
                    h, w, c = y.shape
                    h_left = h%self.scale
                    w_left = w%self.scale
                    if h_left != 0 or w_left !=0:
                        y = cv2.resize(y, (w-w_left, h-h_left), interpolation=cv2.INTER_AREA)
                    
                    x = cv2.resize(y, (0, 0), fx=1/self.scale, fy = 1/self.scale, interpolation=cv2.INTER_AREA)
                    
                    if self.mod == "SRCNN":
                        x = cv2.resize(x, (y.shape[1], y.shape[0]), interpolation = cv2.INTER_CUBIC)
                    
                    
                        
                    
                    
                    if augmentation:
                        rot = np.random.choice([0,1,2,3])
                        if rot == 1:
                            x = cv2.rotate(x, cv2.ROTATE_90_CLOCKWISE)
                            y = cv2.rotate(y, cv2.ROTATE_90_CLOCKWISE)
                        elif rot == 2: 
                            x = cv2.rotate(x, cv2.ROTATE_180)
                            y = cv2.rotate(y, cv2.ROTATE_180)
                        elif rot == 3:
                            x = cv2.rotate(x, cv2.ROTATE_90_COUNTERCLOCKWISE)
                            y = cv2.rotate(y, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        fl = np.random.choice([0,1,2])
                        
                        if fl == 1:
                            x = cv2.flip(x, 0)
                            y = cv2.flip(y, 0)
                            
                        elif fl == 2:
                            x = cv2.flip(x, 1)
                            y = cv2.flip(y, 1)
                            
                    yield np.expand_dims(x, axis=0), np.expand_dims(y, axis=0)
            
    def getSteps(self):
        return self.steps
    
