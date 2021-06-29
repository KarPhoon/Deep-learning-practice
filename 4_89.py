# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 21:08:45 2018

@author: 詹凱丰
"""

import numpy as np
import matplotlib.pylab as plt 

def numerical_diff(f, x):
    h = 1e-4    #0.0001
    return (f(x+h) - f(x-h))/ (2*h)

def function_1(x):
    return 0.01*x**2 + 0.1*x


def tangent_line(f, x):
    d = numerical_diff(f, x)
    print(d)
    y = f(x) - d*x
    return lambda t:d*t+ y

def function_2(x):
    return x[0]**2 + x[1]*2

def _numerical_gradient_no_batch(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)
    
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        print(grad)
        x[idx] = tmp_val  # 値を元に戻す
        
    return grad

x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")

tf = tangent_line(function_1,5)
y2 = tf(x)

plt.plot(x, y)
plt.plot(x, y2)

plt.show


