# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 08:30:10 2018

@author: 詹凱丰
"""
import numpy as np 

def function_2(x):  #範例函數(f(x) = x0**2 + x1**2)
    return np.sum(x**2)

def numerical_gradient(f, x):   #利用數值方法進行偏微分(數值微分)
    h = 1e-4 #0.001
    grad = np.zeros_like(x)
    
    for idx in range(x.size):   #更改單一變數的輸入數值，進行偏微分
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h     #計算f(x+h)
        fxh1 = f(x)
        
        x[idx] = tmp_val - h    #計算f(x-h)
        fxh2 = f(x)  
        
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val
        
return grad
    

