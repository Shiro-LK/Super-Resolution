# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 13:20:53 2019

@author: Shiro
"""

import cv2
import numpy as np
import keras
import os

def get_train_test(path="data/"):
    ## train
    
    path_train = path + "train/"
    
    train_file = open("train.txt", "w") 
    for folder in os.listdir(path_train):
        p = path_train + folder + "/"
        for file in os.listdir(p):
            train_file.write(p + file + "\n")
            
    train_file.close()
    
    
    ## test
    path_test = path + "test/"
    
    for folder in os.listdir(path_test):
        
        p = path_test + folder + "/"
        test_file = open(folder+".txt", "w")
        for file in os.listdir(p):
            test_file.write(p + file + "\n")
            
        test_file.close()
get_train_test()     
def PSNR(y_true, y_pred):
    d = 255
    
    m,n,c = y_true.shape
    EQM = np.sum(np.square(y_true-y_pred)) / (m*n)
    
    PSNR = 10 * np.log10((d**2)/EQM)
    return PSNR

def baseline_bicubic(filename, scale=2):
    with open(filename, "r") as f:
        list_path = [line.rstrip() for line in f]
        
    psnr = 0
    for path in list_path:
        y = cv2.imread(path)
        temp = y
        x = cv2.resize(temp, (0,0), fx=1/scale, fy=1/scale)/255
        
        preds = np.round(cv2.resize(x, (y.shape[1], y.shape[0]), interpolation = cv2.INTER_CUBIC)*255).astype(np.uint8)      
        psnr += PSNR(y, preds)
        
    return psnr/len(list_path)
        
    
psnr = baseline_bicubic("Set5.txt") # 30.08
print(psnr)