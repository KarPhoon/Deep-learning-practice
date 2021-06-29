# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 14:36:55 2018

@author: 詹凱丰
"""

import numpy as np 
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D

def function_2(x):  #範例函數(f(x) = x0**2 + x1**2)
    if x.ndim == 1:
        return np.sum(x**2)
    else:
        return np.sum(x**2, axis=1)

def _numerical_gradient_no_batch(f, x):   #利用數值方法進行偏微分(數值微分)，無批次
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

def numerical_gradient(f,X):
    if X.ndim == 1:
        return _numerical_gradient_no_batch(f, X)    #若X維度維1，執行無批次篇維分
    else:
        grad = np.zeros_like(X)
        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_no_batch(f, x)
            
        return grad 
    
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr*grad
        
    return x


x0 = np.arange(-2, 2.25, 0.25)   #產生-2 ~ 2.0v 間隔0.25的np陣列
x1 = np.arange(-2, 2.25, 0.25)   #產生-2 ~ 2.0v 間隔0.25的np陣列
X, Y = np.meshgrid(x0, x1)

X = X.flatten()
Y = Y.flatten()
g = np.array([X, Y])
grad = numerical_gradient(function_2, np.array([X, Y]))

plt.figure()
plt.quiver(X, Y, -grad[0], -grad[1],  angles="xy",color="#666666")  #畫出quiver方向圖
plt.xlim([-2, 2])   #設定顯示大小
plt.ylim([-2, 2])
plt.xlabel('x0')    #設定標籤
plt.ylabel('x1')
plt.grid()
#plt.legend()
#plt.draw()
plt.show()


