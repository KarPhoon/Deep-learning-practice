# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 19:26:59 2019

@author: 詹凱丰
"""

import sys, os
sys.path.append(os.pardir)
import numpy as np
from mnist import load_mnist
from ch4_103 import TwoLayerNet

#載入資料
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784,hidden_size=50, output_size=10)

x_batch = x_train[:3]
t_batch = t_train[:3]
grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)

#計算個權重的絕對誤差平均值
for key in grad_numerical.keys():
    diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
    print(key + ":" + str(diff))
    