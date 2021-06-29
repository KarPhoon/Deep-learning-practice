# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 23:02:01 2018

@author: 詹凱丰
"""
import numpy as np

def step_fun(x):    #階梯函數
    y = x > 0
    return y.astype(np.int) 

def sigmoid(x):     #sigmiod函數
    return 1/(1 + np.exp(-x))

def relu(x):        #Relu函數
    return np.maximun(0,x)

def softmax(x):     #softmax函數
    c = np.max(x)
    exp_x = np.exp(x-c)
    sum_exp_x = np.sum(exp_x)
    y = exp_x/sum_exp_x
    
    return y

def cross_entropy_error(y, t):      #交叉商誤差
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = t.reshape(1, y.size)
        
    batch_size = y.shape[0]
    return -np.sum(np.log(y + 1e-7)) / batch_size

    
   