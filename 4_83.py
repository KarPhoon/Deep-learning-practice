# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 19:51:22 2018

@author: 詹凱丰
"""

import sys, os 
sys.path.append(os.pardir)
import numpy as np
import pickle
from yyy.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)

print(x_train.shape)
print(t_train.shape)

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

print (x_batch.shape)
print (t_batch.shape)

def cross_entorpy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    
    batch_size = y.shape[0]
    return -np.sum(t* np.log(y+ 1e-7)/batch_size )
