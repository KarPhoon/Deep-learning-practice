# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 11:41:47 2019

@author: 詹凱丰
"""

import sys, os
sys.path.append(os.pardir)
import numpy as np
from ch5_124 import *
from ch4_103 import numerical_gradient
from collections import OrderedDict

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        
        #權重初始化
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        
        #產生各層
        self.layers = OrderedDict()     #OrddeedDict:有序的字典型態
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self['W2'], self.params['b2'])
        
        self.lastlayer = SoftmaxWithLoss()
        
    def predict(self, x):       #正向傳播
        for layer in self.layers.values():      #依序執行layers內神經網路層，x是剛執行時的輸入，也是執行完的輸出
            x = layer.forward(x)
        
        return x
    
    def loss(self, x ,t):       #執行輸出層與運算損失函數
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
    
    def accuracy(self, x , t):      #計算誤差
        y = self.predict(x)
        y = np.argmax(y ,axis=1)
        if t.ndim != 1: t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) /float(x.shape[0])
        return accuracy
    
    #x:輸入資料，t訓練資料
    def numerical_gradient(self, x, t):     #計算梯度，使用數值方法
        loss_W = lambda W: self.loss(x,t)
        
        grads = {}
        grads['W1'] =numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] =numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] =numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] =numerical_gradient(loss_W, self.params['b2'])
        
        return grads
    
    def gradient(self,  x , t):     #計算梯度，使用誤差反向傳播法
        #forward
        self.loss(x, t)
        
        #backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        
        layers = list(self.layer.values())      #將layer中的元素按順序轉成list型太
        layers.reverse()    #將layer中元素順序顛倒
        for layer in layers:    #進行反向傳播
            dout = layer.backward(dout)
        
        #setup
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        
        return grads
    